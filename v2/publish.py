"""Write the study's public results to v2/results/: every label, score and audit answer, keyed by text hash, with no text.

    .venv/bin/python v2/publish.py   # reads v2/data/, which holds the texts and stays out of git

The texts come from datasets whose licenses do not all allow redistribution. build_rows.py rebuilds them from the original
sources, and labeler.key(row) joins a rebuilt row to these files. Model reasoning is left out because it quotes the texts.
Each audit row carries its template group, which audit_report.py otherwise computes from the texts.
"""

import csv
import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).parent.parent)]
import audit_report
import kimi
import labeler

DATA, OUT = Path("v2/data"), Path("v2/results")


def write(name: str, rows: list[dict]):
    (OUT / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{name:18} {len(rows):6} rows")


def final_labels(rows: list[dict]) -> list[dict]:
    """The label each row ended with: Sonnet's, Grok's where Sonnet refused, Kimi's where both refused."""
    first = {a["key"]: a for a in labeler.read_jsonl(DATA / "labels.jsonl")}
    third = kimi.answers()
    out = []
    for r in rows:
        k = labeler.key(r)
        a = first[k] if first[k].get("label") else third[k]
        out.append({"key": k, "label": a["label"], "by": a["by"]})
    return out


def audit() -> list[dict]:
    rows = audit_report.load(str(DATA / "audit_pass2.csv"))
    for population in ("datasets", "traffic"):
        members = [r for r in rows if r["population"] == population]
        for r, g in zip(members, audit_report.groups(members)):
            r["group"] = f"{population}-{g}"
    keep = ("id", "key", "population", "split", "source", "jev", "claude", "label", "by", "weight", "you", "clause", "kimi", "group")
    return [{k: r[k] for k in keep} for r in rows]


def main():
    OUT.mkdir(exist_ok=True)
    rows = labeler.read_rows(str(DATA / "rows.jsonl"))
    write("rows.jsonl", [{"key": labeler.key(r), "split": r["split"], "source": r["source"], "channel": r["channel"]} for r in rows])
    write("labels.jsonl", final_labels(rows))
    write("kimi_labels.jsonl", [{k: a[k] for k in ("key", "by", "label") if k in a} for a in kimi.answers().values()])
    write("jev_scores.jsonl", labeler.read_jsonl(DATA / "jev_scores.jsonl"))
    write("jev_types.jsonl", labeler.read_jsonl(DATA / "jev_types.jsonl"))
    write("audit.jsonl", audit())
    notes = [r for r in csv.DictReader(open(DATA / "audit_pass2.csv", newline="")) if r["note"].strip()]
    assert not notes, "audit notes may quote texts; review them before publishing"


if __name__ == "__main__":
    main()
