"""One Noul per attack class, in place of one Noul for the whole definition: Jev's docs recommend one condition per Noul.

    .venv/bin/python v2/jev_four.py   # scores dev only and prints it beside the single Noul

Each question keeps the shared instructions and the exclusions. Only the class under test sits in criteria.true. Policy
lives in code: a row's score is the highest of its class scores, and the content-injection score counts only on CONTENT rows,
since the definition limits that class to that channel.
"""

import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).parent.parent)]
import jev_question as jq
import data
import evaluate as ev
import jev
import labeler

INJECTION = "Injection through content"


def questions(q: dict = jq.QUESTION) -> dict:
    attack = q["criteria"]["true"][jq.ATTACK_INTRO.rstrip(".")]
    return {name: {"type": "noul",
                   "instructions": {**q["instructions"], "question": f"Does this text perform {name}, as defined here?"},
                   "criteria": {"true": {name: text}, "false": {"no": f"The text does not perform {name}.", **q["criteria"]["false"]}}}
            for name, text in attack.items()}


QUESTIONS = questions()


def score(r: dict, by_class: dict) -> float:
    return max(v for name, v in by_class.items() if name != INJECTION or r["channel"] == "CONTENT")


def report(tag, s, y):
    thr = float(ev.threshold_at(s, y))
    b = s > thr
    print(f"{tag:22} thr {thr:.3f} AUC {ev.auc(s, y):.3f} agree {(b == (y == 1)).mean():.1%} caught {(b & (y == 1)).sum()}/{y.sum()} = {b[y == 1].mean():.1%} flags {b[y == 0].sum()}/{(y == 0).sum()}")
    return b


def main():
    data.load_env()
    missing = jq.check({"parts": list(QUESTIONS.values())})
    assert missing == [jq.ATTACK_INTRO, "Decide the label."], missing
    rows = [r for r in labeler.read_rows("v2/data/rows.jsonl") if r["split"] == "dev"]
    items = [{"text": jq.state_text(r)} for r in rows]
    answers = jev.fill(items, QUESTIONS)
    labels = {a["key"]: a for a in labeler.read_jsonl(Path("v2/data/labels.jsonl"))}
    one = {x["key"]: x["noul"] for x in labeler.read_jsonl(Path("v2/data/jev_scores.jsonl"))}
    kept = [(r, i) for r, i in zip(rows, items) if labels[labeler.key(r)].get("label") and all(answers[(i["text"], n)] for n in QUESTIONS)]
    print(len(kept), "of", len(rows), "dev rows scored and labeled")
    by = [{n: answers[(i["text"], n)]["noul"] for n in QUESTIONS} for _, i in kept]
    lab = [labels[labeler.key(r)]["label"] for r, _ in kept]
    y = np.array([int(a != "not_attack") for a in lab])
    report("one Noul", np.array([one[labeler.key(r)] for r, _ in kept]), y)
    report("four, any channel", np.array([max(c.values()) for c in by]), y)
    b = report("four, injection gated", np.array([score(r, c) for (r, _), c in zip(kept, by)]), y)
    for name, want in zip(QUESTIONS, ("instruction_override", "prompt_extraction", "guardrail_evasion", "content_injection")):
        m = np.array([a == want for a in lab])
        print(f"  {want:22} n {m.sum():4} caught {b[m].sum():4}  own-Noul AUC vs all others {ev.auc(np.array([c[name] for c in by]), m.astype(int)):.3f}")


if __name__ == "__main__":
    main()
