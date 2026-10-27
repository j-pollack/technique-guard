"""The frozen guard on realistic chat traffic: false-positive rate, attack prevalence, and precision.

    .venv/bin/python traffic.py [n] [source]   # default 3000 first user turns from wildchat (or lmsys)

Real traffic contains real attacks, so the guard's blocks can't all be counted as false positives. Sonnet
labels (audit.label, our attack definition) every prompt the guard blocks or reviews, plus a random AUDIT_FRAC of
the ones it allows, to estimate how many attacks slip through. Prompts are split in two halves by hash: pick
thresholds on "a", report on "b". Answers and labels are cached (data/answers.jsonl, data/traffic_labels.jsonl).
"""

import hashlib
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import anti_venom
import hillclimb as h
from corpus import hf, raw_rows
from final_eval import wilson

LABELS = Path(__file__).with_name("data") / "traffic_labels.jsonl"
AUDIT_FRAC = 0.1
# Same row shape: a "conversation" list of {role, content} turns and a "language". LMSYS needs its terms accepted on HF.
SOURCES = {"wildchat": "allenai/WildChat-1M", "lmsys": "lmsys/lmsys-chat-1m"}
EVAL_N = 3000  # prompts(EVAL_N) is the evaluation sample; the rest of each cached log is traffic-dev, for training


def prompts(n: int, source: str = "wildchat") -> list[dict]:
    """n distinct first user turns from a chat log (random pages, cached in datasets/<source>.jsonl)."""
    rows = raw_rows(source, lambda: hf(SOURCES[source], "default", "train", cap=6000))
    seen, out = set(), []
    for r in rows:
        text = next((t["content"] for t in r["conversation"] if t["role"] == "user"), "").strip()
        if text and text not in seen and len(text) < 20000:
            seen.add(text)
            out.append({"text": text, "language": r.get("language"),
                        "half": "a" if int(hashlib.sha256(text.encode()).hexdigest(), 16) % 2 else "b"})
    random.Random(0).shuffle(out)
    return out[:n]


def dev_items() -> list[dict]:
    """Traffic-dev: Sonnet-labelled prompts outside the evaluation sample, in hillclimb's item shape.

    Picked by guard v1 (every review-or-block, plus a random AUDIT_FRAC of the rest), so hard negatives are
    over-represented. Only labelled prompts are returned: unlabelled ones would be ~3% unknown attacks on LMSYS.
    """
    g = json.loads(Path(__file__).with_name("guard_v1.json").read_text())
    rng, pick = random.Random(2), []
    for s in SOURCES:
        items = prompts(10**6, s)[EVAL_N:]
        answers = h.fill(items, g["questions"])
        for i in items:
            a = {q: answers[(i["text"], q)] for q in g["questions"]}
            if all(v is not None for v in a.values()) and (anti_venom.judge(a, g).action != "allow" or rng.random() < AUDIT_FRAC):
                pick.append(i | {"source": f"traffic_{s}"})
    labels = label_all(pick)
    return [i | {"family": "traffic", "split": "dev", "group": "traffic:" + hashlib.sha256(i["text"].encode()).hexdigest()[:16],
                 "label": int(labels[i["text"]]["sonnet"] != "not_attack")}
            for i in pick if "sonnet" in labels[i["text"]]]


def label_all(items: list[dict]) -> dict:
    """Sonnet labels under the content-agnostic definition (audit.SYSTEM_TECHNIQUE), cached in audit.TECHNIQUE.

    LABELS (traffic_labels.jsonl) holds the earlier policy-definition labels and is no longer written."""
    import anthropic
    import audit
    audit.client = anthropic.Anthropic(max_retries=8)  # SDK backs off on 429/529; more retries = slower, not failed
    done = ({r["text"]: r for r in map(json.loads, audit.TECHNIQUE.read_text().splitlines()) if r.get("def") == audit.DEF}
            if audit.TECHNIQUE.exists() else {})
    todo = [i for i in items if i["text"] not in done]
    print(f"Sonnet labels needed: {len(todo)}")
    with ThreadPoolExecutor(16) as pool, audit.TECHNIQUE.open("a", buffering=1) as out:
        futures = {pool.submit(audit.label, i["text"], audit.MODEL, audit.SYSTEM_TECHNIQUE): i for i in todo}
        for f in as_completed(futures):
            row = {"text": futures[f]["text"], "def": audit.DEF, **f.result()}
            done[row["text"]] = row
            out.write(json.dumps(row) + "\n")
    return done


def main(n: int, source: str = "wildchat"):
    g = anti_venom.guard()
    items = prompts(n, source)
    answers = h.fill(items, g["questions"])
    rows = []
    for i in items:
        a = {q: answers[(i["text"], q)] for q in g["questions"]}
        if all(v is not None for v in a.values()):
            v = anti_venom.judge(a, g)
            rows.append({**i, "score": v.score, "action": v.action})
    rng = random.Random(1)
    for r in rows:  # flagged: always labelled; allowed: a random slice, weighted up when estimating
        r["audit"] = r["action"] != "allow" or rng.random() < AUDIT_FRAC
    labels = label_all([r for r in rows if r["audit"]])
    for r in rows:
        lab = labels.get(r["text"]) if r["audit"] else None
        r["attack"] = None if lab is None or "sonnet" not in lab else lab["sonnet"] != "not_attack"
        r["w"] = 1 / AUDIT_FRAC if r["action"] == "allow" else 1.0

    print(f"\n{len(rows)} {source} prompts scored with guard (block >= {g['block_at']:.3f}); "
          f"{sum(r['audit'] and r['attack'] is None for r in rows)} Sonnet refusals left unlabelled")
    for half in ("a", "b", "all"):
        rs = [r for r in rows if half == "all" or r["half"] == half]
        lab = [r for r in rs if r["attack"] is not None]
        w = np.array([r["w"] for r in lab])
        att = np.array([r["attack"] for r in lab])
        blk = np.array([r["action"] == "block" for r in lab])
        neg_n, att_n = w[~att].sum(), w[att].sum()
        fp = int(blk[~att].sum())  # every flagged prompt is labelled, so false positives are counted, not estimated
        lo, hi = wilson(fp, round(neg_n))
        prev = att_n / w.sum()
        rec = (w * att * blk).sum() / att_n if att_n else np.nan
        print(f"half {half:3}: {len(rs):5} prompts | attack prevalence ~{prev:.1%} | FPR (block) {fp / neg_n:.2%} [{lo:.2%}, {hi:.2%}]"
              f" | review-or-block {(w * ~att * (np.array([r['action'] != 'allow' for r in lab]))).sum() / neg_n:.2%}"
              f" | attacks blocked ~{rec:.0%} | precision {blk[att].sum() / max(blk.sum(), 1):.0%}")
    print("\nfalse positives (block) by language:",
          dict(sorted(((l, sum(1 for r in rows if r["attack"] is False and r["action"] == "block" and r["language"] == l))
                       for l in {r["language"] for r in rows}), key=lambda p: -p[1])[:6]))


if __name__ == "__main__":
    env = Path(__file__).with_name(".env.local")
    for line in env.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("\"'"))
    main(int(sys.argv[1]) if len(sys.argv) > 1 else EVAL_N, *sys.argv[2:])
