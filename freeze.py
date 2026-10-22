"""Freeze the guard from the current question library and write guard.json.

    .venv/bin/python freeze.py

Weights: logistic regression on the hill-climbing dev sample (audited labels).
Thresholds: read off out-of-fold dev scores (5-fold by near-dup group), so they aren't flattered by
the in-sample fit. Block from the BLOCK_FPR point, review from the REVIEW_FPR point.
"""

import json
import os
import time
from pathlib import Path

import anti_venom
import hillclimb as h
from questions import HARM, INTERACT, LIBRARY, NO_MAIN

BLOCK_FPR, REVIEW_FPR = 0.05, 0.10


def main():
    items, X, y, _, names, folds, _ = h.load(150)
    oof = h.cv_scores(X, y, folds)
    block_at = float(h.threshold_at(oof, y, BLOCK_FPR))
    review_at = float(h.threshold_at(oof, y, REVIEW_FPR))
    w = h.fit(X, y)
    g = {
        "model": anti_venom.MODEL,
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
            "n": len(items), "attacks": int(y.sum()),
            "block_fpr_target": BLOCK_FPR, "review_fpr_target": REVIEW_FPR,
            "oof_catch_at_block": float((oof[y == 1] > block_at).mean()),
            "oof_catch_at_review": float((oof[y == 1] > review_at).mean()),
        },
    }
    anti_venom.GUARD_PATH.write_text(json.dumps(g, indent=1) + "\n")
    print(f"wrote {anti_venom.GUARD_PATH.name}: {len(names)} features from {len(g['questions'])} questions, "
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
