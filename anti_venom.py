"""Flag user input that tries to prompt-inject or jailbreak a downstream LLM.

    from anti_venom import check
    v = check(user_text)
    if v.action == "block": ...
    elif v.action == "review": ...

Uses TypeSafe's Jev model (https://docs.typesafe.ai) for the judgments; needs TYPESAFE_API_KEY.
"""

import json
import math
import os
import urllib.request
from dataclasses import dataclass
from functools import cache
from pathlib import Path

URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.14.0"  # pinned: guard.json's weights are only valid for this model; re-freeze to bump

GUARD_PATH = Path(__file__).with_name("guard.json")  # written by freeze.py


@cache
def guard() -> dict:
    """The frozen guard: questions, feature recipe, logistic weights, thresholds."""
    return json.loads(GUARD_PATH.read_text())


def featurize(answers: dict, questions: dict, harm: str, interact: list[str], no_main: set[str]) -> tuple[list[str], list[float]]:
    """Jev answers -> (feature names, values). Shared by the shipped guard and hill-climbing.

    Each noul is a feature, each choice option's probability is a feature, and each `interact`
    technique is also multiplied by the `harm` noul (technique AND a restricted request). Questions
    in `no_main` (the harm judgments) only appear inside those products.
    """
    names, x = [], []
    for n, q in questions.items():
        if n in no_main:
            continue
        if q["type"] == "noul":
            names.append(n)
            x.append(answers[n]["noul"])
        else:
            for o in q["criteria"]:
                names.append(f"{n}={o}")
                x.append(answers[n]["probabilities"][o])
    h = answers[harm]["noul"]
    for n in interact:
        names.append(f"{n}*{harm}")
        x.append(answers[n]["noul"] * h)
    return names, x


@dataclass
class Verdict:
    action: str  # "allow" | "review" | "block"
    score: float  # P(attack), 0..1
    reasons: list[tuple[str, float]]  # features pushing the score up most, with their contribution
    model: str = ""  # version that actually answered, as reported by the API
    answers: dict | None = None  # raw Jev answers
    input_tokens: int = 0


def judge(answers: dict, g: dict | None = None) -> Verdict:
    """Combine raw Jev answers into a Verdict with the frozen weights. Pure, so it's testable offline."""
    g = g or guard()
    names, x = featurize(answers, g["questions"], g["harm"], g["interact"], set(g["no_main"]))
    w = g["weights"]
    contrib = [wi * xi for wi, xi in zip(w[1:], x)]
    score = 1 / (1 + math.exp(-max(min(w[0] + sum(contrib), 30), -30)))
    action = "block" if score >= g["block_at"] else "review" if score >= g["review_at"] else "allow"
    reasons = sorted(((n, c) for n, c in zip(names, contrib) if c > 0), key=lambda p: -p[1])[:3]
    return Verdict(action, score, reasons)


def ask(text: str, questions: dict, api_key: str | None = None, timeout: float = 10) -> dict:
    """One Jev call: {"model", "answers", "usage"} for `questions` about `text`."""
    body = {
        "model": MODEL,
        # Wrapped in an object so the input reads as data under evaluation, not as instructions.
        "state": {"untrusted_user_input": text},
        "questions": questions,
    }
    req = urllib.request.Request(
        URL,
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {api_key or os.environ['TYPESAFE_API_KEY']}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def check(text: str, api_key: str | None = None, timeout: float = 10) -> Verdict:
    """Evaluate `text`. Raises on API/network errors: callers decide whether to fail open or closed."""
    g = guard()
    data = ask(text, g["questions"], api_key, timeout)
    if data["model"] != g["model"]:
        raise RuntimeError(f"guard was frozen on {g['model']} but {data['model']} answered")
    verdict = judge(data["answers"], g)
    verdict.model, verdict.answers, verdict.input_tokens = data["model"], data["answers"], data["usage"]["input_tokens"]
    return verdict
