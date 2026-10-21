"""Score anti_venom against public attack + benign corpora.

    python3 eval.py [n_per_source]   # default 40

Builds data/eval.jsonl ({text, label, family, source}) from ungated HF datasets, scores it,
caches results in data/results.jsonl so reruns only bill new items, and prints per-source rates.
"""

import json
import os
import random
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from anti_venom import check

DATA = Path(__file__).with_name("data")
EVAL, RESULTS = DATA / "eval.jsonl", DATA / "results.jsonl"
FETCH_CAP = 500  # rows pulled per source before sampling

# (source, dataset, config, split, row -> (label, family) or None to skip)
# label 1 = attack. Gated sets (wildjailbreak, hackaprompt) need an HF token and are left out.
SOURCES = [
    ("deepset", "deepset/prompt-injections", "default", "test",
     lambda r: (1, "injection") if r["label"] == 1 else (0, "benign")),
    ("gandalf", "Lakera/gandalf_ignore_instructions", "default", "test", lambda r: (1, "injection")),
    ("itw_jailbreak", "TrustAIRLab/in-the-wild-jailbreak-prompts", "jailbreak_2023_12_25", "train", lambda r: (1, "jailbreak")),
    ("itw_regular", "TrustAIRLab/in-the-wild-jailbreak-prompts", "regular_2023_12_25", "train", lambda r: (0, "benign")),
    ("jackhhao", "jackhhao/jailbreak-classification", "default", "test",
     lambda r: (1, "jailbreak") if r["type"] == "jailbreak" else (0, "benign")),
    # Over-refusal sets: harmless prompts that look scary. Only XSTest's "safe" half.
    ("xstest_safe", "Paul/XSTest", "default", "train", lambda r: (0, "benign") if r["label"] == "safe" else None),
    ("orbench_hard", "bench-llm/or-bench", "or-bench-hard-1k", "train", lambda r: (0, "benign")),
]


def fetch_rows(dataset: str, config: str, split: str):
    for offset in range(0, FETCH_CAP, 100):
        q = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split, "offset": offset, "length": 100})
        with urllib.request.urlopen(f"https://datasets-server.huggingface.co/rows?{q}", timeout=30) as resp:
            rows = json.load(resp)["rows"]
        yield from (r["row"] for r in rows)
        if len(rows) < 100:
            return


def build(n: int) -> list[dict]:
    rng, seen, items = random.Random(0), set(), []
    for source, dataset, config, split, labeler in SOURCES:
        pool = []
        for row in fetch_rows(dataset, config, split):
            text = row.get("text") or row.get("prompt")
            labeled = labeler(row)
            # ponytail: exact dedup only; add MinHash if this ever feeds training, not just eval
            key = " ".join(str(text).lower().split())
            if text and labeled and key not in seen:
                seen.add(key)
                pool.append({"text": text, "label": labeled[0], "family": labeled[1], "source": source})
        items += rng.sample(pool, min(n, len(pool)))
        print(f"{source:14} {len(pool):4} unique rows, sampled {min(n, len(pool))}")
    return items


def score(item: dict) -> dict:
    try:
        v = check(item["text"])
        return {"text": item["text"], "model": v.model, "action": v.action, "category": v.category, "risk": v.risk}
    except Exception as e:  # one bad item (too long, timeout) shouldn't sink the run
        return {"text": item["text"], "error": repr(e)}


def main(n: int):
    DATA.mkdir(exist_ok=True)
    items = [json.loads(l) for l in EVAL.read_text().splitlines()] if EVAL.exists() else []
    if len(items) < n * len(SOURCES) * 0.5:  # missing or built with a smaller n
        items = build(n)
        EVAL.write_text("".join(json.dumps(i) + "\n" for i in items))

    cache = {r["text"]: r for r in map(json.loads, RESULTS.read_text().splitlines())} if RESULTS.exists() else {}
    todo = [i for i in items if i["text"] not in cache or "error" in cache[i["text"]]]
    print(f"scoring {len(todo)} new of {len(items)}")
    with ThreadPoolExecutor(8) as pool, RESULTS.open("a") as out:
        for r in pool.map(score, todo):
            cache[r["text"]] = r
            out.write(json.dumps(r) + "\n")

    rows = [{**i, **cache[i["text"]]} for i in items]
    ok = [r for r in rows if "error" not in r]
    print(f"\n{len(rows) - len(ok)} errors, models: {sorted({r.get('model', '?') for r in ok})}\n")
    print(f"{'source':14} {'n':>4} {'block':>6} {'review':>7} {'allow':>6}   (attacks: want block; benign: want allow)")
    for source, *_ in SOURCES:
        rs = [r for r in ok if r["source"] == source]
        if rs:
            rate = lambda a: sum(r["action"] == a for r in rs) / len(rs)
            print(f"{source:14} {len(rs):4} {rate('block'):6.0%} {rate('review'):7.0%} {rate('allow'):6.0%}")

    # Threshold sweep on raw risk: attack recall vs benign false-positive rate.
    attacks, benign = [r for r in ok if r["label"]], [r for r in ok if not r["label"]]
    print(f"\n{'risk >=':>8} {'catch':>6} {'FPR':>6}   ({len(attacks)} attacks, {len(benign)} benign)")
    for t in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95):
        catch = sum(r["risk"] >= t for r in attacks) / max(len(attacks), 1)
        fpr = sum(r["risk"] >= t for r in benign) / max(len(benign), 1)
        print(f"{t:8.2f} {catch:6.0%} {fpr:6.0%}")

    misses = sorted((r for r in ok if r["label"] != (r["action"] == "block")), key=lambda r: r["source"])
    print(f"\nwrong at default thresholds: {len(misses)} (first 10)")
    for r in misses[:10]:
        print(f"  {r['source']:14} {r['action']:6} risk={r['risk']:.2f}  {r['text'][:70]!r}")


if __name__ == "__main__":
    env = Path(__file__).with_name(".env.local")
    if "TYPESAFE_API_KEY" not in os.environ and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("TYPESAFE_API_KEY="):
                os.environ["TYPESAFE_API_KEY"] = line.split("=", 1)[1].strip().strip("\"'")
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
