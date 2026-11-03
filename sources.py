"""The public datasets (paper §3.1, Table 2): where each comes from and how a row becomes (text, label, family).

Every text in the paper is produced by SOURCES below, applied to rows exactly as the dataset publishes them. Gated
sets (wildjailbreak, hackaprompt, wildguardmix, mhj, walledai/*, sorry-bench, lmsys) need HF_TOKEN and their terms
accepted on Hugging Face. label 1 = attack, 0 = not; this is the dataset's label, before relabelling (labeler.py).

The paper's own pulls were random 100-row pages, so they cannot be repeated row for row. data.py downloads each
source in full instead and keeps the rows the manifests in results/manifests/ list.
"""

import csv
import hashlib
import io
import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request

ROWS = "https://datasets-server.huggingface.co"
# Sources that only ever went to the test side of the split (paper §3.3): one per major family, plus NIST's
# optimisation-based and automated attacks.
HELD_OUT = {"hackaprompt", "injecagent_ds", "wildguard", "xstest_safe", "jbb_gcg", "jbb_random_search", "jbb_pair"}
NEAR_DUP = 0.6  # word-trigram Jaccard at or above which two texts share a near-duplicate group


def get(url: str, tries: int = 10):
    hf_host = urllib.parse.urlparse(url).hostname.endswith("huggingface.co")
    headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") and hf_host else {}
    for attempt in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=180)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:  # HF's datasets server throws the odd 5xx/429
            if attempt == tries - 1 or (isinstance(e, urllib.error.HTTPError) and e.code < 500 and e.code != 429):
                raise
            time.sleep(min(5 * 2 ** attempt, 300))  # up to ~20 minutes in all: outages outlast a short backoff


def hf(dataset: str, config: str, split: str):
    """Every row of a Hugging Face split, through the datasets server, 100 at a time."""
    offset, total = 0, None
    while total is None or offset < total:
        q = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split, "offset": offset, "length": 100})
        with get(f"{ROWS}/rows?{q}") as resp:
            d = json.load(resp)
        total = d["num_rows_total"]
        yield from (r["row"] for r in d["rows"])
        offset += 100
        if offset % 50000 == 0:
            print(f"    {dataset}: {offset:,} of {total:,} rows", flush=True)


def hf_pages(dataset: str, total: int, pages: int = 60):
    """The traffic pull (paper §3.3): page 0 and 59 random 100-row pages of the log, drawn with seed 0 from the row count
    the log had then. The datasets server's /filter index covers only the front of these logs, so fetch by offset."""
    for offset in [0, *sorted(random.Random(0).sample(range(100, total, 100), pages - 1))]:
        q = urllib.parse.urlencode({"dataset": dataset, "config": "default", "split": "train", "offset": offset, "length": 100})
        with get(f"{ROWS}/rows?{q}") as resp:
            yield from (r["row"] for r in json.load(resp)["rows"])


def jsonl(url: str):
    with get(url) as resp:
        yield from (json.loads(line) for line in resp)


def json_list(url: str):
    with get(url) as resp:
        yield from json.load(resp)


def csv_rows(url: str):
    with get(url) as resp:
        yield from csv.DictReader(io.StringIO(resp.read().decode()))


def mhj_transcript(r: dict) -> str:
    """User turns of a multi-turn jailbreak, joined: what a guard sees if it screens the whole conversation."""
    turns = [json.loads(v) for k, v in r.items() if k.startswith("message_") and v]
    return "\n\n".join(f"user: {t['body']}" for t in turns if t["role"] == "user")


def wildguard(r: dict):
    harmful = r["prompt_harm_label"] == "harmful"
    if r["adversarial"]:
        return (r["prompt"], 1, "jailbreak") if harmful else (r["prompt"], 0, "benign")
    return (r["prompt"], 0, "harmful_plain") if harmful else (r["prompt"], 0, "benign")


# SORRY-Bench rewrites each harmful request in 20 styles. Encodings, low-resource translation, role-play and persuasion
# get their own family; the rest (slang, typos, dialects, ...) are the plain request.
SORRY_FAMILY = {
    **dict.fromkeys(["ascii", "atbash", "caesar", "morse"], "obfuscation"),
    **dict.fromkeys(["translate-ml", "translate-mr", "translate-ta"], "low_resource_language"),
    **dict.fromkeys(["authority_endorsement", "evidence-based_persuasion", "expert_endorsement", "logical_appeal", "misrepresentation"], "persuasion"),
    "role_play": "roleplay",
}


def sorry(r: dict):
    family = SORRY_FAMILY.get(r["prompt_style"])
    return (r["turns"][0], 1, family) if family else (r["turns"][0], 0, "harmful_plain")


TT = "https://huggingface.co/datasets/qxcv/tensor-trust/resolve/main/benchmarks"
INJECAGENT = "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/main/data"
WJ = ("allenai/wildjailbreak", "eval", "train")
JBB = "https://raw.githubusercontent.com/JailbreakBench/artifacts/main/attack-artifacts"
JBB_TARGETS = ("vicuna-13b-v1.5", "llama-2-7b-chat-hf", "gpt-3.5-turbo-1106", "gpt-4-0125-preview")


def jbb(method: str, *runs: str):
    """JailbreakBench attack artifacts: one prompt per behavior per target model."""
    for run in runs:
        with get(f"{JBB}/{method}/{run}.json") as resp:
            yield from json.load(resp)["jailbreaks"]


# (source, () -> rows, row -> (text, dataset label, family) or None to skip)
SOURCES = [
    ("deepset", lambda: hf("deepset/prompt-injections", "default", "test"),
     lambda r: (r["text"], 1, "injection") if r["label"] == 1 else (r["text"], 0, "benign")),
    ("gandalf", lambda: hf("Lakera/gandalf_ignore_instructions", "default", "test"), lambda r: (r["text"], 1, "injection")),
    ("gandalf_summ", lambda: hf("Lakera/gandalf_summarization", "default", "train"), lambda r: (r["text"], 1, "injection")),
    # Only attacks that beat the target model in the competition.
    ("hackaprompt", lambda: hf("hackaprompt/hackaprompt-dataset", "default", "train"),
     lambda r: (r["user_input"], 1, "injection") if r["correct"] else None),
    ("tt_hijack", lambda: jsonl(f"{TT}/hijacking-robustness/v1/hijacking_robustness_dataset.jsonl"), lambda r: (r["attack"], 1, "injection")),
    ("tt_extract", lambda: jsonl(f"{TT}/extraction-robustness/v1/extraction_robustness_dataset.jsonl"), lambda r: (r["attack"], 1, "injection")),
    # Indirect injection: tool output (e.g. a product review) with an attacker instruction planted inside.
    ("injecagent_dh", lambda: json_list(f"{INJECAGENT}/test_cases_dh_base.json"), lambda r: (r["Tool Response"], 1, "indirect_injection")),
    ("injecagent_ds", lambda: json_list(f"{INJECAGENT}/test_cases_ds_base.json"), lambda r: (r["Tool Response"], 1, "indirect_injection")),
    ("safeguard", lambda: hf("xTRam1/safe-guard-prompt-injection", "default", "test"),
     lambda r: (r["text"], 1, "injection") if r["label"] == 1 else (r["text"], 0, "benign")),
    ("itw_jailbreak", lambda: hf("TrustAIRLab/in-the-wild-jailbreak-prompts", "jailbreak_2023_12_25", "train"), lambda r: (r["prompt"], 1, "jailbreak")),
    ("itw_regular", lambda: hf("TrustAIRLab/in-the-wild-jailbreak-prompts", "regular_2023_12_25", "train"), lambda r: (r["prompt"], 0, "benign")),
    ("jackhhao", lambda: hf("jackhhao/jailbreak-classification", "default", "test"),
     lambda r: (r["prompt"], 1, "jailbreak") if r["type"] == "jailbreak" else (r["prompt"], 0, "benign")),
    ("wj_adv_harmful", lambda: hf(*WJ), lambda r: (r["adversarial"], 1, "jailbreak") if r["data_type"] == "adversarial_harmful" else None),
    ("wj_adv_benign", lambda: hf(*WJ), lambda r: (r["adversarial"], 0, "benign") if r["data_type"] == "adversarial_benign" else None),
    ("wildguard", lambda: hf("allenai/wildguardmix", "wildguardtest", "test"), wildguard),
    ("sorry", lambda: hf("sorry-bench/sorry-bench-202406", "default", "train"), sorry),
    ("mhj", lambda: csv_rows("https://huggingface.co/datasets/ScaleAI/mhj/resolve/main/harmbench_behaviors.csv"),
     lambda r: (mhj_transcript(r), 1, "multi_turn")),
    ("harmbench", lambda: hf("walledai/HarmBench", "standard", "train"), lambda r: (r["prompt"], 0, "harmful_plain")),
    ("advbench", lambda: hf("walledai/AdvBench", "default", "train"), lambda r: (r["prompt"], 0, "harmful_plain")),
    ("strongreject", lambda: hf("walledai/StrongREJECT", "default", "train"), lambda r: (r["prompt"], 0, "harmful_plain")),
    ("xstest_safe", lambda: hf("Paul/XSTest", "default", "train"), lambda r: (r["prompt"], 0, "benign") if r["label"] == "safe" else None),
    ("orbench_hard", lambda: hf("bench-llm/or-bench", "or-bench-hard-1k", "train"), lambda r: (r["prompt"], 0, "benign")),
    ("jbb_gcg", lambda: jbb("GCG", "white_box/vicuna-13b-v1.5", "white_box/llama-2-7b-chat-hf",
                             "transfer/gpt-3.5-turbo-1106", "transfer/gpt-4-0125-preview"), lambda r: (r["prompt"], 1, "optimization")),
    ("jbb_random_search", lambda: jbb("prompt_with_random_search", *(f"black_box/{m}" for m in JBB_TARGETS)), lambda r: (r["prompt"], 1, "optimization")),
    ("jbb_pair", lambda: jbb("PAIR", *(f"black_box/{m}" for m in JBB_TARGETS)), lambda r: (r["prompt"], 1, "jailbreak")),
]

# Traffic (paper §3.3): the first user turn of a conversation, from logs whose rows share one shape.
# Each log: dataset, id column, and its row count at the paper's pull, which fixes the pages hf_pages draws.
TRAFFIC = {"wildchat": ("allenai/WildChat-1M", "conversation_hash", 837_989), "lmsys": ("lmsys/lmsys-chat-1m", "conversation_id", 1_000_000)}


def first_user_turn(r: dict) -> str:
    return next((t["content"] for t in r["conversation"] if t["role"] == "user"), "").strip()


def split_of(source: str, group: str) -> str:
    """Paper §3.3: a near-duplicate group's side comes from a hash of its id; held-out sources are test only."""
    return "test" if source in HELD_OUT or int(hashlib.sha256(group.encode()).hexdigest(), 16) % 2 else "dev"


def shingles(text: str) -> set[int]:
    w = text.lower().split()
    return {hash(tuple(w[i:i + 3])) for i in range(max(len(w) - 2, 1))}


def near_dup(a: set[int], b: set[int]) -> bool:
    return min(len(a), len(b)) >= NEAR_DUP * max(len(a), len(b)) and len(a & b) >= NEAR_DUP * len(a | b)
