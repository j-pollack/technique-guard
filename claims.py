"""Every number in the blog post, recomputed from the published files alone and checked against the post.

    .venv/bin/python claims.py   # reads results/ and frozen.json; needs no texts and no API keys

Post: https://www.jonathanpollack.net/writing/decision-models-as-prompt-screens/
A failed check means the post and the data disagree. Costs come from billing pages, so they are not checked here.
"""

import collections
import json
import sys
from pathlib import Path

import audit_report

R = Path("results")
TYPES = ["guardrail_evasion", "content_injection", "instruction_override", "prompt_extraction"]
CLAUSE = {"A1": "instruction_override", "A2": "prompt_extraction", "A9": "content_injection"} | {f"A{i}": "guardrail_evasion" for i in range(3, 9)}


def read(name: str) -> list[dict]:
    return [json.loads(x) for x in (R / name).read_text().split("\n") if x]


def check(claim: str, got, want):
    ok = got == want
    print(f"{'ok ' if ok else 'BAD'} {claim}: {got}" + ("" if ok else f"  (post says {want})"))
    return ok


def corpus(rows, labels) -> list[bool]:
    by = collections.Counter(a["by"] for a in labels.values())
    src = collections.Counter(r["source"] for r in rows)
    return [check("texts in the corpus", len(rows), 16326),
            check("traffic conversations", sum(r["split"] == "traffic" for r in rows), 5984),
            check("harmless content: Enron, synthetic, LLMail", (src["enron"], src["benign_resource"], src["llmail_phase1"] + src["llmail_phase2"]), (300, 170, 300)),
            check("labeled by Sonnet", by["claude-sonnet-5-5"], 15278),
            check("Sonnet declined, then Grok, then Kimi", (len(rows) - by["claude-sonnet-5-5"], by["x-ai/grok-4.7"], by["moonshotai/kimi-k3.1-pro"]), (1048, 1000, 48))]


def sample(audit) -> list[bool]:
    data = [r for r in audit if r["population"] == "datasets"]
    cells = collections.defaultdict(list)
    for r in data:
        cells[(r["jev"], r["claude"])].append(r)
    table = {c: (round(sum(r["weight"] for r in v)), len(v), round(v[0]["weight"], 1)) for c, v in cells.items()}
    size = sum(t[0] for t in table.values())
    disagree = table[(True, False)][0] + table[(False, True)][0]
    traffic = [r for r in audit if r["population"] == "traffic" and not r["jev"] and not r["claude"]]
    return [check("audit texts, datasets and traffic", (len(data), len(audit) - len(data)), (250, 250)),
            check("dataset texts in the audit population", size, 2580),
            check("weights table: (texts, read, each counts for) for both attack, both not, only Jev, only Sonnet",
                  [table[c] for c in ((True, True), (False, False), (True, False), (False, True))],
                  [(916, 63, 14.5), (1452, 63, 23.0), (150, 62, 2.4), (62, 62, 1.0)]),
            check("disagreements in a random 250 of the datasets, about 20", round(250 * disagree / size, 1), 20.5),
            check("traffic, both say not attack: (texts, read, each counts for)",
                  (round(sum(r["weight"] for r in traffic)), len(traffic), round(traffic[0]["weight"])), (5838, 104, 56))]


def accuracy(audit) -> list[bool]:
    out = []
    pct = lambda x: round(100 * x, 1)
    for population, want in (("datasets", (90.2, 86.3, 4.0, 2.7, 5.2, 5.5, 6.4, 11.8, 17.2)), ("traffic", (99.3, 98.9, 0.4, -0.1, 1.0, 2.6, 3.9, 0.7, 1.0))):
        s = audit_report.stats([r for r in audit if r["population"] == population])
        got = tuple(pct(s[k]) for k in ("claude", "jev", "gap", "lo", "hi", "miss_claude", "miss_jev", "flag_claude", "flag_jev"))
        out.append(check(f"{population}: Sonnet, Jev, gap, interval, misses, false positives", got, want))
    return out


def false_positives(audit, jev_type) -> list[bool]:
    harmless = [r for r in audit if r["population"] == "datasets" and not r["you"]]
    total = sum(r["weight"] for r in harmless)
    per_1000 = lambda judge, typ: {t: 1000 * sum(r["weight"] for r in harmless if r[judge] and typ(r) == t) / total for t in TYPES}
    sonnet, jev = per_1000("claude", lambda r: r["label"]), per_1000("jev", lambda r: jev_type[r["key"]])
    row = lambda d: [round(d[t], 1) for t in TYPES] + [round(sum(d.values()), 1)]
    return [check("false positives per 1,000 harmless dataset texts, Sonnet, by type and all", row(sonnet), [45.2, 17.6, 41.5, 13.3, 117.6]),
            check("false positives per 1,000 harmless dataset texts, Jev, by type and all", row(jev), [79.6, 46.7, 45.3, 0.0, 171.5]),
            check("difference, by type and all", row({t: jev[t] - sonnet[t] for t in TYPES}), [34.4, 29.0, 3.9, -13.3, 54.0])]


def type_match(audit, jev_type) -> list[bool]:
    out = []
    for population, want in (("datasets", {"content_injection": (34, 34, 15), "guardrail_evasion": (11, 10, 10), "instruction_override": (6, 4, 2)}),
                             ("traffic", {"guardrail_evasion": (47, 43, 47), "instruction_override": (23, 16, 12), "prompt_extraction": (2, 2, 2)})):
        three = [(r, CLAUSE[r["clause"]]) for r in audit if r["population"] == population and r["you"] and r["jev"] and r["claude"]]
        got = {t: (sum(c == t for _, c in three), sum(c == t == jev_type[r["key"]] for r, c in three), sum(c == t == r["label"] for r, c in three))
               for t in TYPES if any(c == t for _, c in three)}
        out.append(check(f"{population}: my type, texts, Jev named it, Sonnet named it", got, want))
        if population == "datasets":
            out.append(check("all dataset texts here share one weight, so counts need no weighting", len({r["weight"] for r, _ in three}), 1))
            mistyped = collections.Counter(r["label"] for r, c in three if c == "content_injection" and r["label"] != c)
            out.append(check("Sonnet's mistyped injections, by the label it gave", dict(mistyped), {"instruction_override": 19}))
    return out


def main():
    rows, audit = read("rows.jsonl"), read("audit.jsonl")
    labels = {a["key"]: a for a in read("labels.jsonl")}
    jev_type = {a["key"]: a["type"] for a in read("jev_types.jsonl")}
    frozen = json.loads(Path("frozen.json").read_text())
    results = [check("frozen model and thresholds", (frozen["model"], frozen["thresholds"]), ("jev-1.14.0", {"USER MESSAGE": 0.45, "CONTENT": 0.03})),
               *corpus(rows, labels), *sample(audit), *accuracy(audit), *false_positives(audit, jev_type), *type_match(audit, jev_type)]
    print(f"\n{sum(results)} of {len(results)} claims match the post")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
