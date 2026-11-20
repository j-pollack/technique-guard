"""Score every row in v2/data/rows.jsonl with the untuned Jev question. Answers are cached, so a rerun fills only what failed.

    .venv/bin/python v2/run_jev.py   # writes v2/data/jev_scores.jsonl
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import jev_question as jq
import data
import jev
import labeler

NAME = "v2_untuned"


def main():
    data.load_env()
    rows = labeler.read_rows("v2/data/rows.jsonl")
    items = [{"text": jq.state_text(r)} for r in rows]
    answers = jev.fill(items, {NAME: jq.QUESTION})
    out = [{"key": labeler.key(r), "noul": a["noul"], "model": jev.MODEL} for r, i in zip(rows, items) if (a := answers[(i["text"], NAME)])]
    Path("v2/data/jev_scores.jsonl").write_text("".join(json.dumps(x) + "\n" for x in out))
    print(len(out), "scored of", len(rows))


if __name__ == "__main__":
    main()
