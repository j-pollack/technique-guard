"""Record the frozen Jev rule before any held-out score is read.

    .venv/bin/python v2/freeze.py   # writes v2/frozen.json, and refuses to overwrite a different freeze
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import jev_question as jq
import jev
import labeler

PATH = Path(__file__).with_name("frozen.json")
RULE = {"model": jev.MODEL, "question": jev.qkey(jq.QUESTION), "definition": labeler.DEF, "thresholds": {"USER MESSAGE": 0.45, "CONTENT": 0.03}}


def blocked(r: dict, score: float, rule: dict = RULE) -> bool:
    return score > rule["thresholds"][r["channel"]]


if __name__ == "__main__":
    assert not PATH.exists() or json.loads(PATH.read_text()) == RULE, "a different freeze is on disk"
    PATH.write_text(json.dumps(RULE, indent=1) + "\n")
    print(RULE)
