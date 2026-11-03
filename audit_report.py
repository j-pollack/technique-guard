"""The audit in detail: intervals that resample template groups, results with parts left out, and each judge's error types.

    .venv/bin/python audit_report.py data/audit_pass2.csv
"""

import collections
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

import kimi
import labeler
import sources

SEEN = {"A003", "A005", "A010", "A013", "A020", "A037", "A040", "A081", "A086", "A116"}  # shown to the auditor with Claude's verdict after pass 1
TEMPLATE = "you are the text completion model"


def load(path: str) -> list[dict]:
    by_key = {labeler.key(r): r for r in labeler.read_rows("data/rows.jsonl")}
    human = {r["id"]: r for r in csv.DictReader(open(path, newline=""))}
    third = kimi.answers()
    rows = [json.loads(x) for x in Path("data/audit_key.jsonl").read_text().splitlines()]
    return [r | {"text": by_key[r["key"]]["text"], "you": human[r["id"]]["is_attack"].strip() == "1", "clause": human[r["id"]].get("clause", "")[:2],
                 "kimi": third[r["key"]]["label"] != "not_attack"} for r in rows]


def groups(rows: list[dict]) -> np.ndarray:
    """Template groups: rows that are near-duplicates or share their first 80 characters."""
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    norm = [re.sub(r"\s+", " ", r["text"]).strip().lower() for r in rows]
    sh = [sources.shingles(r["text"]) for r in rows]
    for i in range(len(rows)):
        for j in range(i):
            if norm[i][:80] == norm[j][:80] or sources.near_dup(sh[i], sh[j]):
                parent[find(i)] = find(j)
    return np.array([find(i) for i in range(len(rows))])


def acc(a, t, w):
    return float(((a == t) * w).sum() / w.sum())


def stats(rows: list[dict], n: int = 4000) -> dict:
    """Weighted accuracy, the gap's 95% interval, misses and false flags. Uses each row's "group" when published rows carry one."""
    j, c, t, w = (np.array([r[k] for r in rows]) for k in ("jev", "claude", "you", "weight"))
    g = np.array([r["group"] for r in rows]) if "group" in rows[0] else groups(rows)
    cells = np.array([f"{a}{b}" for a, b in zip(j, c)])
    units = [[np.flatnonzero((cells == cell) & (g == u)) for u in np.unique(g[cells == cell])] for cell in np.unique(cells)]
    rng, out = np.random.default_rng(0), []
    for _ in range(n):  # resample whole template groups inside each sampling cell
        i = np.concatenate([u[k] for u in units for k in rng.integers(0, len(u), len(u))])
        out.append(acc(c[i], t[i], w[i]) - acc(j[i], t[i], w[i]))
    lo, hi = np.percentile(out, [2.5, 97.5])
    miss = lambda a: float(((~a & t) * w).sum() / (t * w).sum())
    flag = lambda a: float(((a & ~t) * w).sum() / (~t * w).sum())
    return {"n": len(rows), "claude": acc(c, t, w), "jev": acc(j, t, w), "gap": acc(c, t, w) - acc(j, t, w), "lo": float(lo), "hi": float(hi),
            "miss_claude": miss(c), "miss_jev": miss(j), "flag_claude": flag(c), "flag_jev": flag(j)}


def gap(rows: list[dict], n: int = 4000) -> str:
    s = stats(rows, n)
    return (f"n {s['n']:3}  Claude {s['claude']:.1%}  Jev {s['jev']:.1%}  gap {s['gap']:+.1%} [{s['lo']:+.1%}, {s['hi']:+.1%}]"
            f"  | misses: Claude {s['miss_claude']:.1%} Jev {s['miss_jev']:.1%}  false flags: Claude {s['flag_claude']:.1%} Jev {s['flag_jev']:.1%}")


def main(path: str):
    rows = load(path)
    views = {"datasets": lambda r: r["population"] == "datasets",
             "datasets, no InjecAgent": lambda r: r["population"] == "datasets" and not r["source"].startswith("injecagent"),
             "datasets, blind rows only": lambda r: r["population"] == "datasets" and r["id"] not in SEEN,
             "traffic": lambda r: r["population"] == "traffic",
             "traffic, no completion template": lambda r: r["population"] == "traffic" and not r["text"].strip().lower().startswith(TEMPLATE),
             "traffic, blind rows only": lambda r: r["population"] == "traffic" and r["id"] not in SEEN}
    for name, keep in views.items():
        print(f"{name:32}", gap([r for r in rows if keep(r)]))
    raw = lambda a, b: sum(r[a] == r[b] for r in rows) / len(rows)
    print("\nraw agreement on the 500:", {f"{a}-{b}": f"{raw(a, b):.1%}" for a, b in (("you", "claude"), ("you", "kimi"), ("you", "jev"), ("claude", "kimi"))})
    print("you attack & Claude not:", sum(r["you"] and not r["claude"] for r in rows), "| Claude attack & you not:", sum(r["claude"] and not r["you"] for r in rows))
    t = collections.Counter((r["source"] if r["population"] == "datasets" else "traffic", r["label"], r["clause"]) for r in rows if r["claude"] != r["you"])
    print("\nwhere you and Claude differ (source, Claude's label, your clause):")
    for k, v in t.most_common(14):
        print(f"  {v:3} {k}")


if __name__ == "__main__":
    main(sys.argv[1])
