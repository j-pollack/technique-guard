# technique-guard

Flags user input that tries to prompt-inject or jailbreak a downstream LLM, before it reaches the model.

It asks TypeSafe's [Jev](https://docs.typesafe.ai) model (pinned to `jev-1.14.0`) a set of yes/no and multiple-choice questions about the text, then combines the answers with a frozen logistic regression into `allow`, `review` or `block`.

An **attack** here means an attempt to hijack the model (override its instructions, leak its prompt, make it act for someone else) or to switch off its own judgment ("no rules" personas or modes, bans on refusing, demands for an unfiltered answer, encodings that hide the request). The guard judges the technique only, never what is being asked for: a direct request, role-play, fiction or persuasion is *not* an attack however harmful its subject, because whether to answer is the downstream model's and its provider's decision.

## Usage

```python
from technique_guard import check

v = check(user_text)          # needs TYPESAFE_API_KEY
if v.action == "block": ...
elif v.action == "review": ...

v.score    # P(attack), 0..1
v.reasons  # top features pushing the score up, with their contribution
```

`check()` raises on API or network errors, so the caller decides whether to fail open or fail closed. It also raises if the API answers with a different Jev version than the guard was frozen on.

`technique_guard.py` only uses the standard library, and the frozen guard lives in `guard.json`.

## Setup

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt   # numpy + anthropic, for the research scripts only
```

Put keys in `.env.local` (git-ignored):

| Key | Used by |
|---|---|
| `TYPESAFE_API_KEY` | `technique_guard.py`, `hillclimb.py`, `freeze.py`, `final_eval.py` |
| `HF_TOKEN` | `corpus.py`, to pull the gated Hugging Face datasets (accept their terms first) |
| `ANTHROPIC_API_KEY` | `audit.py`, `judge.py`, `replay.py` |
| `OPENROUTER_API_KEY` | `replay.py` |

## Tests

```sh
.venv/bin/python test_technique_guard.py
```

This runs offline checks on `judge()` with the frozen guard. If `TYPESAFE_API_KEY` is set, it also runs a few live examples.

## Rebuilding the guard

The research pipeline, in order:

1. **`corpus.py`** builds `data/pool.jsonl` from 22 public attack and benign datasets (cached in `datasets/`). Near-duplicates are grouped so they can't straddle the dev/test split, and four sources are held out for test only.
2. **`questions.py`** is the candidate question library. Questions describe attack *techniques*, never how harmful the payload is.
3. **`hillclimb.py`** answers every question on a dev sample (cached in `data/answers.jsonl`), fits the combiner with group-wise cross-validation, and reports catch rate at fixed false-positive budgets.
4. **`audit.py technique`** labels dev, test and traffic items with Claude Sonnet under the content-agnostic definition (`SYSTEM_TECHNIQUE`, via the Batches API). Every script reads labels through `audit.technique_label`.
5. **`freeze.py`** fits on dev plus traffic-dev (Sonnet-labelled real chat prompts outside the traffic evaluation sample) and writes `guard.json` with the questions, weights, and block/review thresholds (5% / 10% FPR on out-of-fold scores of the dev negatives).
6. **`final_eval.py`** scores the frozen guard on the held-out test split. Run it once per frozen guard, because it warns on reruns.

7. **`replay.py`** (Phase 2) sends a sample of test attacks, half blocked and half missed by the frozen guard, to 9 OpenRouter models and checks whether each worked: built-in checks for HackAPrompt, Tensor Trust and InjecAgent, `judge.py` (a StrongREJECT-style Sonnet grader) for the jailbreak families. It joins the results with the guard's verdicts to give recall on attacks that work, per family and per target. `replay.py 40` runs the full ~4,300 jobs; `--dry-run` only plans and reports from the cache; `--max-openrouter` / `--max-judge` cap spend for the whole replay. Results are cached per (attack, target) in `data/replay.jsonl` and summarized in `data/replay_results.json`.

`data/` and `datasets/` are git-ignored, because several sources forbid redistribution.

## Current numbers

Guard v1.1, under the content-agnostic definition. On the held-out test split (3,717 labelled items, scored once), the guard **blocks 93.7% of attacks [95% CI 92.3–94.9] while blocking 5.9% of non-attacks [5.0–6.9]** (AUC 0.987). On the four sources never seen during tuning it blocks 85% of HackAPrompt, 99% of InjecAgent data-stealing and 82% of WildGuardMix jailbreaks, and wrongly blocks none of XSTest.

Phase 2 replayed test attacks against 9 OpenRouter models. The guard blocks **90% [84–96] of the attacks that actually work**.

On realistic chat traffic, with attacks labelled by Sonnet, the guard wrongly blocks **0.29% [0.15–0.55]** of WildChat first turns and **3.2% [2.6–3.9]** of LMSYS-Chat first turns, or **0.93%** once two automated LMSYS templates are excluded. It blocks all 12 labelled WildChat attacks and 85 of 88 LMSYS ones. Attacks made up about 0.4% (WildChat) and 3% (LMSYS) of these prompts. `traffic.py [n] wildchat|lmsys` reproduces this.

Known weak spots:
- **Held-out jailbreak and injection styles**: WildGuardMix jailbreaks (82% blocked) and HackAPrompt (85%).
- **Multi-turn escalation** that also switches off the model's rules (8 of 14).
- **Developer-written system prompts** pasted as user input ("You are the text completion model…") trip the injection questions.

See `NOTEBOOK.md` for the full method, per-family results, and decisions.
