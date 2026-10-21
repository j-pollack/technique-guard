"""Refit how Jev's answers combine, using stored dev results only (no API calls).

    python3 calibrate.py

Features: every noul probability + the intent choice distribution. Model: logistic regression
(stdlib gradient descent). Evaluated with 5-fold cross-validation split by near-dup group, so a
template's copies never sit on both sides. Compares against the current max-of-signals risk.
"""

import hashlib
import json
import math
from pathlib import Path

from anti_venom import QUESTIONS

DATA = Path(__file__).with_name("data")
FPR_BUDGETS = (0.02, 0.05, 0.10)


def features(answers: dict) -> list[float]:
    x = [answers[k]["noul"] for k, q in QUESTIONS.items() if q["type"] == "noul"]
    p = answers["intent"]["probabilities"]
    return x + [p["prompt_injection"], p["jailbreak"], answers["intent"]["confidence"]]


def fit(X: list[list[float]], y: list[int], steps: int = 2000, lr: float = 0.5, l2: float = 1e-3) -> list[float]:
    # ponytail: full-batch gradient descent on ~14 features; swap for sklearn if features grow
    w = [0.0] * (len(X[0]) + 1)
    pos = sum(y) / len(y)
    cw = {1: 0.5 / pos, 0: 0.5 / (1 - pos)}  # balance classes so the big attack pool doesn't dominate
    for _ in range(steps):
        g = [0.0] * len(w)
        for xi, yi in zip(X, y):
            err = (predict(w, xi) - yi) * cw[yi]
            g[0] += err
            for j, v in enumerate(xi):
                g[j + 1] += err * v
        w = [wj - lr * (gj / len(X) + l2 * wj * (j > 0)) for j, (wj, gj) in enumerate(zip(w, g))]
    return w


def predict(w: list[float], x: list[float]) -> float:
    z = w[0] + sum(wj * v for wj, v in zip(w[1:], x))
    return 1 / (1 + math.exp(-max(min(z, 30), -30)))


def catch_at_fpr(scores: list[tuple[float, int]], budget: float) -> tuple[float, float]:
    """Highest catch rate whose benign FPR stays within budget; returns (threshold, catch)."""
    benign = sorted((s for s, y in scores if y == 0), reverse=True)
    t = benign[int(budget * len(benign))] if benign else 1.0
    attacks = [s for s, y in scores if y == 1]
    return t, sum(s > t for s in attacks) / len(attacks)


def main():
    pool = {i["text"]: i for i in map(json.loads, (DATA / "pool.jsonl").read_text().splitlines())}
    rows = []
    for r in map(json.loads, (DATA / "results.jsonl").read_text().splitlines()):
        item = pool.get(r["text"])
        if item and item["split"] == "dev" and item["label"] is not None and "answers" in r:
            rows.append({**item, **r, "x": features(r["answers"])})
    rows = list({r["text"]: r for r in rows}.values())
    fold = lambda r: int(hashlib.sha256(r["group"].encode()).hexdigest(), 16) % 5
    print(f"{len(rows)} dev items ({sum(r['label'] for r in rows)} attacks)\n")

    for r in rows:
        r["lr"] = None
    for k in range(5):
        train = [r for r in rows if fold(r) != k]
        w = fit([r["x"] for r in train], [r["label"] for r in train])
        for r in rows:
            if fold(r) == k:
                r["lr"] = predict(w, r["x"])

    print(f"{'benign FPR budget':>18} {'max-of-signals':>15} {'logistic (CV)':>14}")
    for b in FPR_BUDGETS:
        _, base = catch_at_fpr([(r["risk"], r["label"]) for r in rows], b)
        _, lr = catch_at_fpr([(r["lr"], r["label"]) for r in rows], b)
        print(f"{b:18.0%} {base:15.0%} {lr:14.0%}")

    t, _ = catch_at_fpr([(r["lr"], r["label"]) for r in rows], 0.05)
    tb, _ = catch_at_fpr([(r["risk"], r["label"]) for r in rows], 0.05)
    print(f"\nper family at 5% FPR (catch for attacks, FPR for benign):")
    print(f"{'source':14} {'family':18} {'n':>4} {'max':>6} {'logit':>6}")
    for key in sorted({(r["source"], r["family"]) for r in rows}):
        rs = [r for r in rows if (r["source"], r["family"]) == key]
        print(f"{key[0]:14} {key[1]:18} {len(rs):4} {sum(r['risk'] > tb for r in rs) / len(rs):6.0%} "
              f"{sum(r['lr'] > t for r in rs) / len(rs):6.0%}")

    w = fit([r["x"] for r in rows], [r["label"] for r in rows])
    names = [k for k, q in QUESTIONS.items() if q["type"] == "noul"] + ["intent_injection", "intent_jailbreak", "intent_confidence"]
    print("\nweights (all dev):", ", ".join(f"{n}={v:+.2f}" for n, v in sorted(zip(names, w[1:]), key=lambda p: -abs(p[1]))))


if __name__ == "__main__":
    main()
