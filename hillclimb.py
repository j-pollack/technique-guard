"""One hill-climbing round over the question library, on dev only.

    .venv/bin/python hillclimb.py [n_per_group]   # default 150

1. Answers every question in questions.LIBRARY for a fixed dev sample (n per (source, family)),
   querying Jev only for (question, text) pairs not already in data/answers.jsonl.
2. Fits logistic regression on the answers with 5-fold CV split by near-dup group.
3. Prints catch rate at fixed benign-FPR budgets (overall and per family) against the v0 baselines,
   per-question separability, and the largest weights; appends the round to data/hillclimb.jsonl.
Test is never touched here.
"""

import hashlib
import json
import os
import sys
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import technique_guard
from questions import HARM, INTERACT, LIBRARY, NO_MAIN, v0_risk

DATA = Path(__file__).with_name("data")
ANSWERS, LOG = DATA / "answers.jsonl", DATA / "hillclimb.jsonl"
BUDGETS = (0.02, 0.05, 0.10)
BATCH = 40  # questions per Jev call
USD_PER_M = 0.042


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def qkey(q: dict) -> str:
    """Cache key: changes whenever the question's content or the pinned model changes."""
    return sha(json.dumps({"model": technique_guard.MODEL, "q": q}, sort_keys=True))


def dev_sample(n: int) -> list[dict]:
    from corpus import POOL, sample
    pool = [json.loads(l) for l in POOL.read_text().splitlines()]
    return sample(pool, "dev", n)


def ask_with_retry(text: str, questions: dict) -> dict:
    for attempt in range(4):
        try:
            return technique_guard.ask(text, questions, timeout=60)
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == 3:
                return {"error": repr(e)}
            time.sleep(2 ** attempt)


def fill(items: list[dict], questions: dict = LIBRARY) -> dict:
    cache = {}
    if ANSWERS.exists():
        for r in map(json.loads, ANSWERS.read_text().splitlines()):
            cache[(r["t"], r["q"])] = r["a"]
    keys = {name: qkey(q) for name, q in questions.items()}
    jobs = []
    for item in items:
        t = sha(item["text"])
        missing = [n for n in questions if (t, keys[n]) not in cache]
        jobs += [(item["text"], t, missing[i:i + BATCH]) for i in range(0, len(missing), BATCH)]
    print(f"{len(items)} items x {len(questions)} questions; {len(jobs)} Jev calls needed")

    def run(job):
        text, t, names = job
        return t, names, ask_with_retry(text, {n: questions[n] for n in names})

    tokens, errors, models = 0, 0, set()
    # Line-buffered and in completion order, so a crash loses at most the calls still in flight.
    with ThreadPoolExecutor(8) as pool, ANSWERS.open("a", buffering=1) as out:
        for f in as_completed([pool.submit(run, j) for j in jobs]):
            t, names, data = f.result()
            if "error" in data:
                errors += 1
                continue
            models.add(data["model"])
            tokens += data["usage"]["input_tokens"]
            for n in names:
                cache[(t, keys[n])] = data["answers"][n]
                out.write(json.dumps({"t": t, "q": keys[n], "a": data["answers"][n]}) + "\n")
    if jobs:
        print(f"  {errors} failed calls, served by {sorted(models)}, ${tokens * USD_PER_M / 1e6:.3f} this round")
    assert models <= {technique_guard.MODEL}, f"unexpected model {models}"
    return {(i["text"], n): cache.get((sha(i["text"]), keys[n])) for i in items for n in questions}


def row(answers: dict, text: str) -> tuple[list[str], list[float]] | None:
    by_name = {n: answers[(text, n)] for n in LIBRARY}
    if any(a is None for a in by_name.values()):
        return None
    return technique_guard.featurize(by_name, LIBRARY, HARM, INTERACT, NO_MAIN)


def fit(X: np.ndarray, y: np.ndarray, steps: int = 3000, lr: float = 1.0, l2: float = 1e-3) -> np.ndarray:
    Xb = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(Xb.shape[1])
    cw = np.where(y == 1, 0.5 / y.mean(), 0.5 / (1 - y.mean()))  # balance attacks vs negatives
    for _ in range(steps):
        p = 1 / (1 + np.exp(-np.clip(Xb @ w, -30, 30)))
        g = Xb.T @ ((p - y) * cw) / len(y)
        g[1:] += l2 * w[1:]
        w -= lr * g
    return w


def predict(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(w[0] + X @ w[1:], -30, 30)))


def cv_scores(X: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    out = np.zeros(len(y))
    for k in range(5):
        tr = folds != k
        out[~tr] = predict(fit(X[tr], y[tr]), X[~tr])
    return out


def threshold_at(scores: np.ndarray, y: np.ndarray, budget: float) -> float:
    neg = np.sort(scores[y == 0])[::-1]
    return neg[min(int(budget * len(neg)), len(neg) - 1)]


def auc(s: np.ndarray, y: np.ndarray) -> float:
    ranks = s.argsort().argsort() + 1
    pos = y == 1
    return (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())


def load(n: int, traffic_dev: bool = False):
    """The dev sample as (items, X, y, y_orig, names, folds); y is the content-agnostic label (audit.technique_label),
    and items without one (Sonnet refused) are dropped.

    traffic_dev adds Sonnet-labelled real chat prompts (traffic.dev_items, sources "traffic_*") for training.
    """
    items = dev_sample(n)
    if traffic_dev:
        import traffic
        items += traffic.dev_items()
    answers = fill(items)
    import audit
    labels = audit.technique_labels()
    rows = [(i, row(answers, i["text"])) for i in items]
    rows = [(i, r) for i, r in rows if r is not None and audit.technique_label(i, labels) is not None]
    items = [i for i, _ in rows]
    names = rows[0][1][0]
    X = np.array([r[1] for _, r in rows])
    y_orig = np.array([i["label"] for i in items])
    y = np.array([audit.technique_label(i, labels) for i in items])
    folds = np.array([int(hashlib.sha256(i["group"].encode()).hexdigest(), 16) % 5 for i in items])
    return items, X, y, y_orig, names, folds, answers


def main(n: int):
    items, X, y, y_orig, names, folds, answers = load(n)
    print(f"{len(items)} usable ({int(y.sum())} attacks, {int((y == 0).sum())} negatives), {len(names)} features, "
          f"{int((y != y_orig).sum())} labels changed by audit\n")
    v0 = [j for j, nm in enumerate(names) if nm.startswith("v0_") and "*" not in nm]

    v0_answers = lambda i: {n[3:]: answers[(i["text"], n)] for n in LIBRARY if n.startswith("v0_")}
    systems = {
        "v0 max": np.array([v0_risk(v0_answers(i)) for i in items]),
        "v0 logit": cv_scores(X[:, v0], y, folds),
        "library logit": cv_scores(X, y, folds),
        # Variant: jailbreak techniques only count when the underlying request is restricted.
        "lib, jb×harm only": cv_scores(X[:, [j for j, nm in enumerate(names) if nm not in INTERACT]], y, folds),
    }
    thresholds = {s: {b: threshold_at(sc, y, b) for b in BUDGETS} for s, sc in systems.items()}

    print(f"{'catch at FPR':14}" + "".join(f"{s:>15}" for s in systems))
    summary = {}
    for b in BUDGETS:
        catches = {s: float((sc[y == 1] > thresholds[s][b]).mean()) for s, sc in systems.items()}
        summary[f"catch@{b:.0%}"] = catches
        print(f"{b:<14.0%}" + "".join(f"{c:15.0%}" for c in catches.values()))
    print(f"{'AUC':14}" + "".join(f"{auc(sc, y):15.3f}" for sc in systems.values()))
    orig = cv_scores(X, y_orig, folds)
    summary["library logit, original labels"] = {
        f"{b:.0%}": float((orig[y_orig == 1] > threshold_at(orig, y_orig, b)).mean()) for b in BUDGETS}
    print("library logit on original (unaudited) labels: " +
          ", ".join(f"{k} FPR → {v:.0%}" for k, v in summary["library logit, original labels"].items()))

    print(f"\nper family at 5% FPR (attacks: catch; negatives: FPR)")
    print(f"{'source':14} {'family':18} {'n':>4}" + "".join(f"{s:>15}" for s in systems))
    per_family = {}
    for key in sorted({(i["source"], i["family"]) for i in items}):
        m = np.array([(i["source"], i["family"]) == key for i in items])
        rates = {s: float((sc[m] > thresholds[s][0.05]).mean()) for s, sc in systems.items()}
        per_family["/".join(key)] = rates
        print(f"{key[0]:14} {key[1]:18} {m.sum():4}" + "".join(f"{r:15.0%}" for r in rates.values()))

    print("\nper-question AUC (attack vs negative), new questions:")
    q_auc = {nm: auc(X[:, j], y) for j, nm in enumerate(names)}
    for nm, a in sorted(q_auc.items(), key=lambda p: -p[1]):
        if not nm.startswith("v0_"):
            print(f"  {a:.3f}  {nm}")

    w = fit(X, y)
    top = sorted(zip(names, w[1:]), key=lambda p: -abs(p[1]))[:12]
    print("\nlargest weights (all dev):", ", ".join(f"{nm}={v:+.2f}" for nm, v in top))

    with LOG.open("a") as f:
        f.write(json.dumps({
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": technique_guard.MODEL, "n_items": len(items),
            "questions": {nm: qkey(q) for nm, q in LIBRARY.items()}, "summary": summary,
            "per_family_5pct": per_family, "question_auc": q_auc,
        }) + "\n")


if __name__ == "__main__":
    env = Path(__file__).with_name(".env.local")
    for line in env.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("\"'"))
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 150)
