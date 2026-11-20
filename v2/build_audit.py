"""Draw the 500-row audit from rows not used for tuning, with the frozen rule. Verdicts stay out of the blind file.

    .venv/bin/python v2/build_audit.py   # writes v2/data/audit_blind.csv and v2/data/audit_key.jsonl

Each population (labeled datasets, traffic) gets 250 rows across four cells: Jev and Claude's verdicts crossed. Rare cells
would vanish from a random sample, and only the two disagreement cells decide the accuracy gap. Each row keeps a weight,
cell size over rows drawn, so estimates describe the population.
"""

import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import freeze
import kimi
import labeler

POPULATIONS = {"heldout": "datasets", "heldout_v2": "datasets", "traffic": "traffic"}
PER_POPULATION, SEED = 250, 0


def allocate(sizes: dict, total: int) -> dict:
    """Equal shares, capped by cell size, with leftovers passed to cells that still have rows."""
    take = dict.fromkeys(sizes, 0)
    while sum(take.values()) < total and any(take[c] < sizes[c] for c in sizes):
        for c in sorted(sizes, key=lambda c: take[c]):
            if take[c] < sizes[c] and sum(take.values()) < total:
                take[c] += 1
    return take


def verdicts() -> list[dict]:
    rows = [r for r in labeler.read_rows("v2/data/rows.jsonl") if r["split"] in POPULATIONS]
    labels = {a["key"]: a for a in labeler.read_jsonl(Path("v2/data/labels.jsonl"))}
    scores = {x["key"]: x["noul"] for x in labeler.read_jsonl(Path("v2/data/jev_scores.jsonl"))}
    second = kimi.answers()
    out = []
    for r in rows:
        k = labeler.key(r)
        a = labels[k] if labels[k].get("label") else second[k]
        out.append({"key": k, "population": POPULATIONS[r["split"]], "split": r["split"], "source": r["source"], "channel": r["channel"], "text": r["text"],
                    "jev": freeze.blocked(r, scores[k]), "claude": a["label"] != "not_attack", "label": a["label"], "by": a["by"]})
    return out


def draw(rows: list[dict], rng: random.Random) -> list[dict]:
    out = []
    for population in ("datasets", "traffic"):
        cells = defaultdict(list)
        for r in rows:
            if r["population"] == population:
                cells[(r["jev"], r["claude"])].append(r)
        take = allocate({c: len(v) for c, v in cells.items()}, PER_POPULATION)
        for c, members in cells.items():
            out += [r | {"weight": len(members) / take[c]} for r in rng.sample(sorted(members, key=lambda r: r["key"]), take[c])]
    return out


def main():
    assert allocate({"a": 3, "b": 100, "c": 100}, 30) == {"a": 3, "b": 14, "c": 13}
    rows = verdicts()
    rng = random.Random(SEED)
    sample = draw(rows, rng)
    rng.shuffle(sample)
    assert len(sample) == 500 == len({r["key"] for r in sample})
    assert sum(r["weight"] for r in sample) == sum(1 for _ in rows) or abs(sum(r["weight"] for r in sample) - len(rows)) < 1e-6
    fields = ("key", "population", "split", "source", "jev", "claude", "label", "by", "weight")
    with open("v2/data/audit_key.jsonl", "w") as f:
        for i, r in enumerate(sample, 1):
            f.write(json.dumps({"id": f"A{i:03}"} | {k: r[k] for k in fields}) + "\n")
    with open("v2/data/audit_blind.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "channel", "text", "is_attack"])
        w.writerows([f"A{i:03}", r["channel"], r["text"], ""] for i, r in enumerate(sample, 1))
    Path("v2/data/audit_keys.txt").write_text("\n".join(r["key"] for r in sample))
    print(Counter((r["population"], r["jev"], r["claude"]) for r in rows))
    print(Counter((r["population"], r["jev"], r["claude"]) for r in sample))


if __name__ == "__main__":
    main()
