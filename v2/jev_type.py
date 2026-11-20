"""Which kind of attack Jev sees: a Choice over the four classes, compared with Claude's label and the audit's clause.

    .venv/bin/python v2/jev_type.py   # writes v2/data/jev_types.jsonl and prints the confusion tables

Secondary analysis. The frozen Noul and thresholds still decide attack or not; a row's type is read only where both sides
being compared call it an attack. Audit tables come raw first, then weighted by the audit's sampling weights, by population.
"""

import collections
import csv
import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).parent.parent)]
import jev_question as jq
import data
import jev
import labeler

NAME = "v2_type"
TYPES = labeler.LABELS[:4]
CLAUSE = {"A1": "instruction_override", "A2": "prompt_extraction", "A9": "content_injection"} | {f"A{i}": "guardrail_evasion" for i in range(3, 9)}
SHORT = {"instruction_override": "override", "prompt_extraction": "extract", "guardrail_evasion": "evasion", "content_injection": "inject"}


def table(title: str, pairs: list[tuple], rows: list[str], denom: float | None = None):
    """Rows are the reference's answer, columns the judge's type. Pairs are (reference, type) or (reference, type, weight).
    With denom, cells are weighted percentages of denom, and each row ends with its total."""
    w = [p[2] if len(p) > 2 else 1 for p in pairs]
    c = collections.Counter()
    for (a, b, *_), x in zip(pairs, w):
        c[(a, b)] += x
    same = f", same type {sum(x for (a, b, *_), x in zip(pairs, w) if a == b) / sum(w):.1%}" if rows == TYPES and pairs else ""
    print(f"\n{title}  (n {len(pairs)}{same})")
    print(f"  {'':14}" + "".join(f"{SHORT[t]:>9}" for t in TYPES) + ("      all" if denom else ""))
    cell = (lambda x: f"{x / denom:9.1%}") if denom else (lambda x: f"{x:9.0f}")
    for r in rows:
        print(f"  {SHORT.get(r, r):14}" + "".join(cell(c[(r, t)]) for t in TYPES) + (cell(sum(c[(r, t)] for t in TYPES)) if denom else ""))
    if denom:
        print(f"  {'all':14}" + "".join(cell(sum(c[(r, t)] for r in rows)) for t in TYPES) + cell(sum(c.values())))


def main():
    data.load_env()
    rows = labeler.read_rows("v2/data/rows.jsonl")
    items = [{"text": jq.state_text(r)} for r in rows]
    answers = jev.fill(items, {NAME: jq.TYPE_QUESTION})
    jt = {labeler.key(r): a for r, i in zip(rows, items) if (a := answers[(i["text"], NAME)])}
    Path("v2/data/jev_types.jsonl").write_text("".join(json.dumps({"key": k, "type": a["choice"], "p": a["probabilities"], "model": jev.MODEL}) + "\n" for k, a in jt.items()))
    print(len(jt), "typed of", len(rows))

    thr = json.loads(Path("v2/frozen.json").read_text())["thresholds"]
    noul = {x["key"]: x["noul"] for x in labeler.read_jsonl(Path("v2/data/jev_scores.jsonl"))}
    flags = {labeler.key(r) for r in rows if noul.get(labeler.key(r), 0) > thr[r["channel"]]}
    claude = {a["key"]: a["label"] for a in labeler.read_jsonl(Path("v2/data/labels.jsonl")) if a.get("label")}
    both = [(claude[k], jt[k]["choice"]) for k in flags if k in jt and claude.get(k, "not_attack") != "not_attack"]
    table("All rows, Jev flags and Claude calls an attack: Claude's label x Jev's type", both, TYPES)

    audit = [json.loads(x) for x in Path("v2/data/audit_key.jsonl").read_text().splitlines()]
    you = {r["id"]: r for r in csv.DictReader(open("v2/data/audit_pass2.csv", newline=""))}
    att = [(r, CLAUSE[you[r["id"]]["clause"][:2]]) for r in audit if you[r["id"]]["is_attack"].strip() == "1"]
    all3 = [(r, c) for r, c in att if r["jev"] and r["claude"]]
    table("Audit, all three call an attack: your clause x Jev's type", [(c, jt[r["key"]]["choice"]) for r, c in all3], TYPES)
    table("Audit, all three call an attack: your clause x Claude's label", [(c, r["label"]) for r, c in all3], TYPES)
    for judge, typ in (("Jev", lambda r: jt[r["key"]]["choice"]), ("Claude", lambda r: r["label"])):
        false = [(you[r["id"]]["clause"][:2], typ(r)) for r in audit if r[judge.lower()] and you[r["id"]]["is_attack"].strip() == "0"]
        table(f"Audit, {judge}'s false flags: your exclusion clause x {judge}'s type", false, sorted({c for c, _ in false}))
    print("\nWeighted by the audit's sampling weights, so each row counts for the real rows it stands for.")
    for pop in ("datasets", "traffic"):
        mine = [r for r in audit if r["population"] == pop]
        three = [(c, r) for r, c in all3 if r["population"] == pop]
        table(f"{pop}, all three call an attack: your clause x Jev's type", [(c, jt[r["key"]]["choice"], r["weight"]) for c, r in three], TYPES)
        table(f"{pop}, all three call an attack: your clause x Claude's label", [(c, r["label"], r["weight"]) for c, r in three], TYPES)
        non = [r for r in mine if you[r["id"]]["is_attack"].strip() == "0"]
        for judge, typ in (("Jev", lambda r: jt[r["key"]]["choice"]), ("Claude", lambda r: r["label"])):
            false = [(you[r["id"]]["clause"][:2], typ(r), r["weight"]) for r in non if r[judge.lower()]]
            table(f"{pop}, {judge}'s false-flag rate: % of your non-attacks, by exclusion clause and {judge}'s type", false,
                  sorted({c for c, *_ in false}), sum(r["weight"] for r in non))


if __name__ == "__main__":
    main()
