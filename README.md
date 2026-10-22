# anti-venom

Flags user input that tries to prompt-inject or jailbreak a downstream LLM, before it reaches the model.

It asks TypeSafe's [Jev](https://docs.typesafe.ai) model (pinned to `jev-1.14.0`) a set of yes/no and multiple-choice questions about the text, then combines the answers with a frozen logistic regression into `allow`, `review` or `block`.

An **attack** here means an evasion technique wrapped around a request the model would normally decline, or an attempt to hijack the model (override its instructions, leak its prompt, make it act for someone else). A plain harmful request with no technique is *not* an attack: refusing it is the downstream model's job, so the guard allows it.

## Usage

```python
from anti_venom import check

v = check(user_text)          # needs TYPESAFE_API_KEY
if v.action == "block": ...
elif v.action == "review": ...

v.score    # P(attack), 0..1
v.reasons  # top features pushing the score up, with their contribution
```

`check()` raises on API or network errors, so the caller decides whether to fail open or fail closed. It also raises if the API answers with a different Jev version than the guard was frozen on.

`anti_venom.py` only uses the standard library, and the frozen guard lives in `guard.json`.

## Setup

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt   # numpy + anthropic, for the research scripts only
```

Put keys in `.env.local` (git-ignored):

| Key | Used by |
|---|---|
| `TYPESAFE_API_KEY` | `anti_venom.py`, `hillclimb.py`, `freeze.py`, `final_eval.py` |
| `HF_TOKEN` | `corpus.py`, to pull the gated Hugging Face datasets (accept their terms first) |
| `ANTHROPIC_API_KEY` | `audit.py`, `judge.py`, `replay.py` |
| `OPENROUTER_API_KEY` | `replay.py` |

## Tests

```sh
.venv/bin/python test_anti_venom.py
```

This runs offline checks on `judge()` with the frozen guard. If `TYPESAFE_API_KEY` is set, it also runs a few live examples.

## Rebuilding the guard

The research pipeline, in order:

1. **`corpus.py`** builds `data/pool.jsonl` from 22 public attack and benign datasets (cached in `datasets/`). Near-duplicates are grouped so they can't straddle the dev/test split, and four sources are held out for test only.
2. **`questions.py`** is the candidate question library. Questions describe attack *techniques*, never how harmful the payload is.
3. **`hillclimb.py`** answers every question on a dev sample (cached in `data/answers.jsonl`), fits the combiner with group-wise cross-validation, and reports catch rate at fixed false-positive budgets.
4. **`audit.py`** relabels disputed dev items with Claude Sonnet to correct label noise.
5. **`freeze.py`** writes `guard.json` with the questions, weights, and block/review thresholds (5% / 10% benign FPR on out-of-fold dev scores).
6. **`final_eval.py`** scores the frozen guard on the held-out test split. Run it once per frozen guard, because it warns on reruns.

7. **`replay.py`** (Phase 2) sends a sample of test attacks, half blocked and half missed by the frozen guard, to 9 OpenRouter models and checks whether each worked: built-in checks for HackAPrompt, Tensor Trust and InjecAgent, `judge.py` (a StrongREJECT-style Sonnet grader) for the jailbreak families. It joins the results with the guard's verdicts to give recall on attacks that work, per family and per target. `replay.py 40` runs the full ~4,300 jobs; `--dry-run` only plans and reports from the cache; `--max-openrouter` / `--max-judge` cap spend for the whole replay. Results are cached per (attack, target) in `data/replay.jsonl` and summarized in `data/replay_results.json`.

`data/` and `datasets/` are git-ignored, because several sources forbid redistribution.

## Current numbers

On the held-out test split (guard `6fdeebfecb47872f`, 3,816 items, scored once), the frozen guard **blocks 82.4% of attacks [95% CI 80.8–83.9] while blocking 7.0% of non-attacks [5.9–8.4]**. That rises to 86.0% and 10.3% if you count review. The first max-of-signals guard blocked 61.4% of attacks at 5.8%. On the four sources never seen during tuning, it blocks 86% of HackAPrompt, 99% of InjecAgent data-stealing and 93% of WildGuardMix jailbreaks, and wrongly blocks 1% of XSTest.

Phase 2 replayed 480 test attacks against 9 OpenRouter models. SORRY-Bench attacks were graded against their original plain request. Missed attacks worked 34% of the time and blocked ones 23%. So the guard blocks **62–72% of the attacks that actually work** (the range depends on how pairs the judge refused to grade are counted), against 80% of these attacks overall. With the guard in front, 4–8% of these attacks succeed on each model, against 7–24% without it. Role-play is the weakest family: the guard blocks only 36% of the role-play attacks that work.

The guard only targets attacks, not requests that are legitimate but may break a provider's terms (adult content, advocacy, professional advice), however they're wrapped. With SORRY-Bench attacks in those categories relabelled under that rule by Sonnet, test numbers are **84.6% of attacks blocked [83.0–86.1] at 7.6% FPR [6.4–8.9]**, and the guard blocks **70–77% of the attacks that work**.

On realistic chat traffic, with attacks labelled by Sonnet, the guard wrongly blocks **1.5% [1.1–2.0]** of WildChat first turns and **2.1% [1.7–2.8]** of LMSYS-Chat first turns. That's against 7.6% on the deliberately hard test negatives. LMSYS rises to 4.7% if a templated, ambiguous "say something toxic" probe is counted. Attacks made up about 0.3% (WildChat) and 3.3% (LMSYS) of these prompts. `traffic.py [n] wildchat|lmsys` reproduces this.

Known weak spots:
- **Benign prompts written in jailbreak style.** About 35% of WildGuardMix's adversarial-benign prompts are blocked. This is where nearly all false positives come from; ordinary and over-refusal prompts sit at 0–6%.
- **Role-play (61%) and persuasion (74%) jailbreaks.**
- **Low-resource-language translation (7%).** Nothing in the text marks it as an attack except the request itself.

See `NOTEBOOK.md` for the full method, per-family results, and decisions.
