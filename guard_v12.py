"""The comparator, guard v1.2 (paper §4.3): 36 narrow Jev questions combined by a logistic regression.

guard_v12.json holds the questions, the feature recipe, the 56 weights plus intercept, and the thresholds
(block_at 0.489 is its 5%-dev-FPR point). It was fit on dev and 778 Sonnet-labelled traffic prompts from half a of
each chat log; the fit itself is not part of this repository, only its frozen output.
"""

import json
import math
from pathlib import Path

GUARD = json.loads(Path(__file__).with_name("guard_v12.json").read_text())


def features(answers: dict, g: dict = GUARD) -> tuple[list[str], list[float]]:
    """Jev answers -> (feature names, values). Each noul is a feature, each choice option's probability is a feature,
    and each `interact` technique is also multiplied by the `harm` noul. `no_main` questions only appear in products."""
    names, x = [], []
    for n, q in g["questions"].items():
        if n in g["no_main"]:
            continue
        if q["type"] == "noul":
            names.append(n)
            x.append(answers[n]["noul"])
        else:
            for o in q["criteria"]:
                names.append(f"{n}={o}")
                x.append(answers[n]["probabilities"][o])
    h = answers[g["harm"]]["noul"]
    for n in g["interact"]:
        names.append(f"{n}*{g['harm']}")
        x.append(answers[n]["noul"] * h)
    return names, x


def score(answers: dict, g: dict = GUARD) -> float:
    names, x = features(answers, g)
    assert names == g["features"], "feature order differs from the frozen guard"
    w = g["weights"]
    return 1 / (1 + math.exp(-max(min(w[0] + sum(wi * xi for wi, xi in zip(w[1:], x)), 30), -30)))


def action(s: float, g: dict = GUARD) -> str:
    return "block" if s >= g["block_at"] else "review" if s >= g["review_at"] else "allow"
