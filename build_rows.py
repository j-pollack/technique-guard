"""Build data/rows.jsonl: the rows data.py rebuilds, the traffic sample, the synthetic rows, and the new Enron and LLMail-Inject samples.

    .venv/bin/python build_rows.py

Reads data/ and synthetic/, and refetches traffic and the new samples; writes only under data/. New samples go to split heldout_v2.
"""

import json
import random
import sys
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).parent
import data  # noqa: E402
import sources  # noqa: E402
from jev import sha  # noqa: E402

OUT = ROOT / "data/rows.jsonl"
V1_SPLITS = ("dev", "spent_test", "heldout", "gcg_proxy", "fresh")
CONTENT_PREFIXES = ("injecagent", "benign_resource", "enron", "llmail")
SAMPLE, PAGE, SEED = 300, 10, 0
MIN_CHARS, MAX_CHARS = 50, 6000  # drop empty fragments and very long threads from the new samples


def channel(source: str) -> str:
    return "CONTENT" if source.startswith(CONTENT_PREFIXES) else "USER MESSAGE"


def row(text: str, source: str, family: str, split: str, **extra) -> dict:
    return {"text": text, "source": source, "family": family, "split": split, "channel": channel(source), **extra}


def v1_rows() -> list[dict]:
    keep = ("group", "dataset_label", "label_from")
    return [row(r["text"], r["source"], r["family"], name, v1_label=r.get("label"), **{k: r.get(k) for k in keep})
            for name in V1_SPLITS for r in data.rows(name, labelled=False)]


def traffic_rows() -> list[dict]:
    out = []
    for source, (dataset, column, total) in sources.TRAFFIC.items():
        m = data.manifest(f"traffic_{source}")
        by_id = {r[column]: sources.first_user_turn(r) for r in sources.hf_pages(dataset, total)}
        found = [row(by_id[e["id"]], f"traffic_{source}", "traffic", "traffic", half=e["half"], language=e["language"])
                 for e in m if sha(by_id.get(e["id"], "")) == e["sha"]]
        print(f"traffic_{source}: {len(found)} of {len(m)}", flush=True)
        out += found
    return out


def synthetic_rows() -> list[dict]:
    d = ROOT / "synthetic"
    out = [row(r["text"], "benign_resource", r["kind"], "synthetic", template=r["template"], construction=0)
           for r in json.loads((d / "benign_resource.json").read_text())]
    out += [row(r["text"], "encoded_" + r["kind"], r["encoding"], "synthetic", construction=int(r["kind"] == "harmful"))
            for r in json.loads((d / "encoded_rows.json").read_text())]
    prefix = "CHANNEL: USER MESSAGE\nTEXT: "
    for r in json.loads((d / "encoded_matched.json").read_text()):
        assert r["text"].startswith(prefix)
        out.append(row(r["text"][len(prefix):], "encoded_" + r["kind"], r["encoding"], "synthetic", construction=int(r["kind"] == "harmful")))
    return out


def page(dataset: str, split: str, offset: int) -> list[dict]:
    q = urllib.parse.urlencode({"dataset": dataset, "config": "default", "split": split, "offset": offset, "length": PAGE})
    with sources.get(f"{sources.ROWS}/rows?{q}") as resp:
        return [x["row"] for x in json.load(resp)["rows"]]


def sample(dataset: str, split: str, total: int, n: int, source: str, render) -> list[dict]:
    """n texts from random pages, skipping lengths outside bounds and near-duplicates of a text already kept."""
    rng, kept, shingled = random.Random(f"{SEED}{source}"), [], []
    while len(kept) < n:
        offset = rng.randrange(total - PAGE)
        for r in page(dataset, split, offset):
            text = render(r)
            s = sources.shingles(text)
            if MIN_CHARS <= len(text) <= MAX_CHARS and not any(sources.near_dup(s, x) for x in shingled):
                kept.append(row(text, source, "indirect_injection" if source.startswith("llmail") else "benign_document",
                                "heldout_v2", hf_split=split, hf_offset=offset))
                shingled.append(s)
    return kept[:n]


def email(r: dict) -> str:
    return f"Subject: {r['subject']}\n\n{r['body']}"


def new_rows() -> list[dict]:
    half = SAMPLE // 2
    return (sample("LLM-PBE/enron-email", "train", 490_408, SAMPLE, "enron", lambda r: r["text"])
            + sample("microsoft/llmail-inject-challenge", "Phase1", 370_724, half, "llmail_phase1", email)
            + sample("microsoft/llmail-inject-challenge", "Phase2", 90_916, half, "llmail_phase2", email))


def main():
    data.load_env()
    groups = [v1_rows(), synthetic_rows(), new_rows(), traffic_rows()]
    seen, rows = set(), []
    for g in groups:
        for r in g:
            k = (r["channel"], r["text"])
            if k not in seen:
                seen.add(k)
                rows.append(r)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(len(rows), "rows from", sum(map(len, groups)), "; chars", sum(len(r["text"]) for r in rows))


if __name__ == "__main__":
    main()
