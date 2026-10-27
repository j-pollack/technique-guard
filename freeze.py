"""Freeze the guard from the current question library and write guard.json.

    .venv/bin/python freeze.py

Weights: logistic regression on the hill-climbing dev sample (audited labels) plus traffic-dev (Sonnet-labelled
real chat prompts outside the traffic evaluation sample; see traffic.dev_items).
Thresholds: read off out-of-fold scores (5-fold by near-dup group) of the benchmark dev items only, so they aren't
flattered by the in-sample fit and keep the same meaning as guard v1's. Block from the BLOCK_FPR point, review from the REVIEW_FPR point.
"""

import json
import os
import time
from pathlib import Path

import numpy as np

import technique_guard
import hillclimb as h
from questions import HARM, INTERACT, LIBRARY, NO_MAIN

BLOCK_FPR, REVIEW_FPR = 0.05, 0.10


def main():
    items, X, y, _, names, folds, _ = h.load(150, traffic_dev=True)
    oof = h.cv_scores(X, y, folds)
    dev = np.array([not i["source"].startswith("traffic_") for i in items])
    traffic_n = int((~dev).sum())
    block_at = float(h.threshold_at(oof[dev], y[dev], BLOCK_FPR))
    review_at = float(h.threshold_at(oof[dev], y[dev], REVIEW_FPR))
    w = h.fit(X, y)
    g = {
        "model": technique_guard.MODEL,
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        # Only questions the features use: other harm rewordings are in the library for comparison, not asked.
        "questions": {k: q for k, q in LIBRARY.items() if k not in NO_MAIN or k == HARM},
        "harm": HARM,
        "interact": INTERACT,
        "no_main": sorted(NO_MAIN),
        "features": names,
        "weights": [float(v) for v in w],
        "block_at": block_at,
        "review_at": review_at,
        "dev": {
            "n": int(dev.sum()), "attacks": int(y[dev].sum()), "traffic_dev": traffic_n,
            "block_fpr_target": BLOCK_FPR, "review_fpr_target": REVIEW_FPR,
            "oof_catch_at_block": float((oof[dev & (y == 1)] > block_at).mean()),
            "oof_catch_at_review": float((oof[dev & (y == 1)] > review_at).mean()),
        },
    }
    technique_guard.GUARD_PATH.write_text(json.dumps(g, indent=1) + "\n")
    print(f"wrote {technique_guard.GUARD_PATH.name}: {len(names)} features from {len(g['questions'])} questions, "
          f"block >= {block_at:.3f}, review >= {review_at:.3f}")
    print(f"dev out-of-fold: catch {g['dev']['oof_catch_at_block']:.0%} blocked, "
          f"{g['dev']['oof_catch_at_review']:.0%} blocked or reviewed")


if __name__ == "__main__":
    env = Path(__file__).with_name(".env.local")
    for line in env.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("\"'"))
    main()
