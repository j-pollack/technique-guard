"""Jev and Claude against a rater on the audit rows, weighted back to each population. The rater is Kimi or the hand audit.

    .venv/bin/python compare.py kimi
    .venv/bin/python compare.py <audited.csv>   # needs id and is_attack columns
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

import kimi

BAR = 0.05


def rater(name: str, key: list[dict]) -> dict:
    if name == "kimi":
        a = kimi.answers()
        return {r["id"]: a[r["key"]]["label"] != "not_attack" for r in key if a.get(r["key"], {}).get("label")}
    with open(name, newline="") as f:
        return {r["id"]: r["is_attack"].strip().lower() in ("1", "y", "yes", "true") for r in csv.DictReader(f) if r["is_attack"].strip()}


def mean(hit: np.ndarray, w: np.ndarray) -> float:
    return float((hit * w).sum() / w.sum())


def gap_ci(jev, claude, truth, w, cells, n=4000, seed=0) -> tuple:
    """Claude's accuracy minus Jev's, with a bootstrap 95% interval that resamples inside each cell."""
    rng, gaps = np.random.default_rng(seed), []
    groups = [np.flatnonzero(cells == c) for c in np.unique(cells)]
    for _ in range(n):
        i = np.concatenate([rng.choice(g, len(g)) for g in groups])
        gaps.append(mean(claude[i] == truth[i], w[i]) - mean(jev[i] == truth[i], w[i]))
    return mean(claude == truth, w) - mean(jev == truth, w), *np.percentile(gaps, [2.5, 97.5])


def main(name: str):
    key = [json.loads(x) for x in Path("data/audit_key.jsonl").read_text().splitlines()]
    truth = rater(name, key)
    print(len(truth), "of", len(key), "rows rated")
    for population in ("datasets", "traffic"):
        x = [r for r in key if r["population"] == population and r["id"] in truth]
        jev, claude = (np.array([r[k] for r in x]) for k in ("jev", "claude"))
        t, w = np.array([truth[r["id"]] for r in x]), np.array([r["weight"] for r in x])
        cells = np.array([f"{a}{b}" for a, b in zip(jev, claude)])
        gap, lo, hi = gap_ci(jev, claude, t, w, cells)
        mixed = jev != claude
        print(f"\n{population}: n {len(x)}  agreement Jev-Claude {mean(jev == claude, w):.1%}, Claude-rater {mean(claude == t, w):.1%}, Jev-rater {mean(jev == t, w):.1%}")
        print(f"  gap Claude minus Jev {gap:+.1%} [{lo:+.1%}, {hi:+.1%}]  bar {'met' if hi <= BAR else 'not shown'}")
        print(f"  on {mixed.sum()} audited disagreements the rater sides with Jev on {(jev == t)[mixed].sum()}, raw; weighted {mean((jev == t)[mixed], w[mixed]):.1%}")


if __name__ == "__main__":
    main(sys.argv[1])
