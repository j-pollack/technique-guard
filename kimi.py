"""A third opinion from Kimi, through the labeler's prompt and schema. Answers go to their own file, so the reference labels stay as they are.

    .venv/bin/python kimi.py <keys.txt>   # appends to data/kimi_labels.jsonl and skips keys already answered
"""

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import labeler

KIMI = "moonshotai/kimi-k3.1-pro"
OUT = Path("data/kimi_labels.jsonl")


def answers() -> dict:
    return {a["key"]: a for a in labeler.read_jsonl(OUT)} if OUT.exists() else {}


def main(keys: set):
    labeler.data.load_env()
    done = {k for k, a in answers().items() if a.get("label") or a.get("error") == "refusal"}
    todo = [r for r in labeler.read_rows("data/rows.jsonl") if labeler.key(r) in keys - done]
    lock = threading.Lock()

    def one(r):
        ans = labeler.grok(r, KIMI)
        with lock, OUT.open("a") as f:
            f.write(json.dumps({"key": labeler.key(r), "by": KIMI} | ans, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(one, todo))
    print(len(todo), "asked of", len(keys))


if __name__ == "__main__":
    main(set(Path(sys.argv[1]).read_text().split()))
