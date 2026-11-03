"""The decision model: one call asks Jev a set of questions about one text.

JEV_URL / JEV_MODEL point at another server speaking the same API; open-weight runs have used oMLX 0.7.2 at
http://127.0.0.1:8085/v1/systemone with JEV_MODEL=openjev-MLX-4bit or clef-oQ4e, and JEV_WORKERS=1.
Answers are cached in data/answers.jsonl, keyed by (sha(text), sha(model + question)), so a rerun only pays for
what is missing and a different model or wording never reuses another's answers. Needs TYPESAFE_API_KEY for Jev.
"""

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

URL = os.environ.get("JEV_URL", "https://api.typesafe.ai/v1/systemone")
MODEL = os.environ.get("JEV_MODEL", "jev-1.14.0")
ANSWERS = Path(__file__).with_name("data") / "answers.jsonl"
BATCH = 40  # questions per call
USD_PER_M = 0.042  # Jev's list price per million input tokens


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def qkey(q: dict) -> str:
    return sha(json.dumps({"model": MODEL, "q": q}, sort_keys=True))


def ask(text: str, questions: dict, timeout: float = 60) -> dict:
    """One call: {"model", "answers", "usage"}. The text goes in as data, not as instructions."""
    body = {"model": MODEL, "state": {"untrusted_user_input": text}, "questions": questions}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={
        "Authorization": f"Bearer {os.environ.get('TYPESAFE_API_KEY', '')}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def ask_with_retry(text: str, questions: dict) -> dict:
    for attempt in range(4):
        try:
            return ask(text, questions)
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == 3:
                return {"error": repr(e)}
            time.sleep(2 ** attempt)


def fill(items: list[dict], questions: dict) -> dict:
    """{(text, question name): answer or None} for every item and question, asking only for what the cache lacks."""
    cache = {}
    if ANSWERS.exists():
        for r in map(json.loads, ANSWERS.read_text().splitlines()):
            cache[(r["t"], r["q"])] = r["a"]
    keys = {name: qkey(q) for name, q in questions.items()}
    jobs = []
    for item in items:
        t = sha(item["text"])
        missing = [n for n in questions if (t, keys[n]) not in cache]
        jobs += [(item["text"], t, missing[i:i + BATCH]) for i in range(0, len(missing), BATCH)]
    if jobs:
        print(f"{len(items)} items x {len(questions)} questions: {len(jobs)} calls to {MODEL}")
        tokens, errors = 0, 0
        ANSWERS.parent.mkdir(exist_ok=True)
        with ThreadPoolExecutor(int(os.environ.get("JEV_WORKERS", 8))) as pool, ANSWERS.open("a", buffering=1) as out:
            futures = [pool.submit(lambda j: (j[1], j[2], ask_with_retry(j[0], {n: questions[n] for n in j[2]})), j) for j in jobs]
            for f in as_completed(futures):
                t, names, data = f.result()
                if "error" in data:
                    errors += 1
                    continue
                assert data["model"] == MODEL, f"asked for {MODEL}, {data['model']} answered"
                tokens += data["usage"]["input_tokens"]
                for n in names:
                    cache[(t, keys[n])] = data["answers"][n]
                    out.write(json.dumps({"t": t, "q": keys[n], "a": data["answers"][n]}) + "\n")
        cost = f", ${tokens * USD_PER_M / 1e6:.2f}" if "typesafe.ai" in URL else ""
        print(f"  {errors} failed calls{cost}")
    return {(i["text"], n): cache.get((sha(i["text"]), keys[n])) for i in items for n in questions}
