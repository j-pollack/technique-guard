# Judging the technique, not the request: a calibrated prompt-attack guard from yes/no questions

*Draft, 2026-10-26. Numbers are for guard v1.1 (`guard.json`, hash `c002f121ffbc3f90`) unless stated. `TODO` marks open items.*

## Abstract

We build a guard that sits in front of an LLM and blocks prompt injection and jailbreaks, from 36 natural-language yes/no and multiple-choice questions answered by a small judgment model (TypeSafe's Jev, pinned to `jev-1.14.0`) and combined by a logistic regression. The guard judges the *technique* only, never what is being asked for: whether a request should be answered is the downstream model's and its provider's decision. Under that definition, on a held-out test split of 3,717 items from 22 public datasets (four sources never seen in tuning), the guard blocks **93.7% of attacks [95% CI 92.3–94.9] at a 5.9% false-positive rate [5.0–6.9]** (AUC 0.987). Replaying a stratified sample of test attacks against nine production models, it blocks **74–90% of the attacks that actually work** (depending on how ungradeable replies are counted), and cuts attack success from 8–23% to about 2% across models. On real chat traffic it wrongly blocks **0.29%** of WildChat and **3.2%** of LMSYS-Chat first turns (0.9% excluding two automated templates). We also report what changed along the way: the attack definition moved headline recall by more than ten points, the attacks a guard misses are the ones more likely to work, and grading jailbreak success against a rewritten prompt overstated success by more than 2×.

## 1. Introduction

Prompt-attack guards are usually evaluated as classifiers on benchmark labels. Two questions are usually left open: *what counts as an attack*, and *do the attacks the guard misses actually work?* Benchmarks disagree on the first (is a role-play wrapper around a request for adult fiction an attack? is a plain harmful request?), and a guard's reported recall depends heavily on the answer. The second matters because a guard that misses only attacks that fail anyway is better than its recall suggests, and one that misses exactly the attacks that work is worse.

Contributions:
1. A **content-agnostic attack definition** that a guard can enforce without a content policy, and a full relabelling of 7,188 benchmark and traffic texts under it (§2, §4.3).
2. A **cheap, inspectable guard**: 36 plain-language questions, one model call per input, a 56-weight linear combiner, and per-decision reasons (§4).
3. **Three evaluations**: held-out benchmark test (§5.1), *working-attack* recall from replays against nine target models (§5.2), and false-positive rates on two real chat logs (§5.3).
4. **Negative and methodological results** worth knowing before building a guard (§6).

## 2. Threat model and attack definition

The guard sees one untrusted input (a user message, or data the model will read on the user's behalf) before the downstream model does, and returns `allow`, `review` or `block`.

**Attack** (the labelling prompt is reproduced in Appendix A):
- **Prompt injection**: text that tries to take control of the model processing it: override or replace its instructions, extract its hidden prompt or secrets, make it output a fixed phrase regardless of its task, or make it act or produce output for someone other than its user. Instructions aimed at the model inside fetched data (an email, web page, tool output) are injection whatever they ask for.
- **Jailbreak**: text that tries to switch off or get around the model's own judgment about what to answer: a persona or "mode" without rules, a fake update or developer mode, a game where anything goes; forbidding refusals, warnings or breaking character; demanding a second "unfiltered" answer; hiding the request with encodings, ciphers or splitting; forcing the reply to open with compliance.

**Not attacks, whatever they ask for**: direct requests however harmful or explicit; role-play, personas, fiction, hypotheticals and creative writing that do not also try to switch off the model's rules; persuasion, arguments, claimed credentials and emotional appeals; text in any language.

**Why content-agnostic.** A guard that also judges content is a content policy, and content policies differ by provider and deployment. Declining a harmful request is the downstream model's job, and the model sees the request either way. Distinguishing a "fiction wrapper used for evasion" from fiction requires judging what the fiction asks for, so under this definition framing alone is never an attack. The definition is a choice with costs: requests for clearly harmful content wrapped only in fiction or persuasion are allowed through to the model (§6.1, §7).

## 3. Related work

TODO. Benchmarks used (Appendix B); learned guards (Llama Guard, Prompt Guard, WildGuard, ...); LLM-as-judge guards; StrongREJECT grading; attack-success replay studies.

## 4. Method

### 4.1 Data
22 public sources (Appendix B), each split into **dev** (tuning) and **test** (reported once) by a hash of a near-duplicate group: same SORRY-Bench question, same MHJ behaviour, or ≥ 0.6 word-trigram Jaccard within a source, so templated attacks can't straddle the split. Four sources are **held out** of dev entirely (HackAPrompt, InjecAgent data-stealing, WildGuardMix, XSTest-safe). Up to 150 items per (source, family) per split: dev 2,879 items, test 3,816. Plain harmful requests (AdvBench, HarmBench, StrongREJECT, SORRY-Bench base, WildGuardMix vanilla-harmful) are negatives, so the combiner cannot learn harmfulness as a stand-in for "attack".

### 4.2 Questions and combiner
Each input is one call to Jev with 36 questions (`questions.py`), each describing a technique, never the payload's harmfulness: 10 first-pass signals (instruction override, authority impersonation, prompt extraction, hidden instructions, exfiltration, unrestricted persona, obfuscation, pressure, hypothetical framing, harmful request, plus a 3-way intent choice), 8 injection techniques (addressed to the AI, embedded in data, unrequested action, task switch, fake delimiters, forced output, secret extraction, repeat-the-prompt), 15 jailbreak techniques (persona wrapper, rules-don't-apply, refusal pre-emption, dual response, justification, authority / evidence / logical appeal, false identity, hypothetical wrapper, encoded payload, decode-and-follow, payload splitting, many-shot, escalation), an "ordinary request" question, and one question on whether the underlying request is something the model would decline. The input is wrapped as `{"untrusted_user_input": text}` so it reads as data.

Features are each yes/no probability, each choice probability, and each jailbreak technique multiplied by the underlying-request question (56 features). A class-balanced L2 logistic regression combines them. Thresholds are read off out-of-fold dev scores (5-fold, by near-duplicate group): **block** at 5% dev FPR (score ≥ 0.432), **review** at 10% (≥ 0.178). Questions were developed by hill-climbing on dev only (`hillclimb.py`), with every Jev answer cached per (question, text), so re-weighting costs nothing.

### 4.3 Labels
Dataset labels disagree with each other and with any one definition. All labels used for fitting and evaluation are Claude Sonnet 5.5 judgments under the definition in §2 (structured output, low effort), except groups where a 90-item pilot found no disagreements (plain harmful requests and ordinary benign prompts), which keep their dataset label. 394 of 7,188 texts were refused by the labeller; refused items are dropped, except encoded SORRY-Bench attacks, which are attacks by definition. Under the new labels, test attacks fall from 2,177 (dataset labels) to 1,325: SORRY-Bench persuasion, role-play and low-resource-language "attacks" become non-attacks, as do most multi-turn escalations (MHJ, 14 of 150 remain); a few nominally benign prompts become attacks (e.g. "morality-free space" framings).

**Label audit.** An independent annotator (a separate Claude session, blind to our labels and to the guard) labelled a uniform random sample of 150 labelled test items under the same definition. Agreement was 139/150 (92.7%), Cohen's κ = 0.84; the dataset-labelled groups agreed 44/44 and the Sonnet-labelled items 95/106. All 11 disagreements go one way: we say non-attack, the annotator says attack. On review, 1 is a clear error in our labels (a "character known for lack of scruples and ethical boundaries", i.e. a persona without rules), 1 is the annotator's error, and 9 are ambiguous. Four of the nine force the reply to open with a fixed scene-setting line rather than a compliance phrase; the others are weak or implicit techniques (a final "he would not include disclaimers" turn, "imagine your previous response was 'I can't assist'… continue", threats demanding private details). Estimated error: **0.7% [0.1, 3.7] of all labels; upper bound counting ambiguous items, 6.7% [3.7, 11.8]**. No false attack labels were found (0/48, CI 0–7.4%); among non-attack labels the upper bound is 9.8% [5.4, 17.1]. The guard blocks 1 of the 11 disputed items (6 review, 4 allow), so label noise does not inflate the reported FPR; if the ambiguous items are attacks, recall is somewhat overstated.

### 4.4 Traffic-dev
Benchmarks under-represent ordinary chat that *looks* attack-like (companion and character bots, pasted documents). We add 784 real first turns from WildChat-1M and LMSYS-Chat-1M to training, taken from outside the traffic evaluation sample: every prompt the previous guard flagged, plus a random 10% of the rest, all labelled under §2. Thresholds are still read off the benchmark dev negatives only, so they keep their meaning.

## 5. Results

### 5.1 Held-out test

| | n | blocked | 95% CI |
|---|---|---|---|
| **All attacks** | 1,325 | **93.7%** | 92.3–94.9 |
| Injection (direct) | 559 | 91.9% | 89.4–93.9 |
| Indirect injection | 271 | 97.0% | 94.3–98.5 |
| Jailbreak | 273 | 93.4% | 89.8–95.8 |
| Obfuscation (encoded) | 150 | 100% | 97.5–100 |
| Multi-turn | 14 | 57.1% | 32.6–78.6 |
| Attacks in nominally benign sets | 58 | 89.7% | 79.2–95.2 |
| **All non-attacks (FPR)** | 2,392 | **5.9%** | 5.0–6.9 |
| Benign | 881 | 2.6% | 1.7–3.9 |
| Plain harmful requests | 689 | 0.1% | 0.0–0.8 |
| Persuasion / role-play / low-resource wrappers | 384 | 0.3% | — |
| Relabelled from attack sets: jailbreak-styled | 160 | 23.1% | 17.3–30.2 |
| Relabelled from attack sets: injection-styled | 147 | 41.5% | — |
| Relabelled multi-turn | 131 | 13.0% | 8.3–19.8 |

Block-or-review: 97.7% of attacks, 10.7% of non-attacks. AUC 0.987. A first-pass max-of-signals guard (the 10 first-pass questions, threshold 0.8) blocks 81.8% at 11.5%.

**Held-out sources**: HackAPrompt 85.2% (n=135), InjecAgent data-stealing 99.3% (140), WildGuardMix jailbreaks 81.2% (96); XSTest-safe 0% FPR (150), WildGuardMix non-attacks 6.4% (326).

Most false positives are items the source dataset calls attacks and the labeller does not (jailbreak- or injection-*styled* text with no switch-off or takeover). This is the boundary of the definition, and where label noise concentrates; the guard blocks few of the items a blind audit disputed (§4.3).

**The definition matters more than the model.** v1, fitted under an earlier definition that let content decide (a wrapper was an attack only if the wrapped request was one the model would decline), scores 95.8% at **23.6%** FPR (AUC 0.930) on the same labels, because it blocks the persuasion and role-play wrappers that are now negatives. TODO: v1.1 under the earlier definition's labels, as a sensitivity column.

### 5.2 Do the misses work? (replay against nine models)

We sent a stratified sample of test attacks (up to 40 per family, half blocked and half missed by the guard that drew the sample, weighted back to the test population) to nine models via OpenRouter: Grok 4.7, Qwen 3.8 max and flash, Gemini 4 Argon and 3.8 Flash, DeepSeek v4-pro and v4.1-flash, GLM 5.3 and 5.3-flash (temperature 0, low reasoning effort, 4,096 max tokens). Success checks: built-in for HackAPrompt (expected completion), Tensor Trust ("access granted" / leaked code) and InjecAgent (any attacker tool call); for jailbreaks, a StrongREJECT-style Claude Sonnet grader against the **original** request (score ≥ 0.5). Claude was never a target. 1,929 attack × model pairs; the guard's verdicts come from cached answers, so re-frozen guards are scored on the same paid-for replies.

| | recall on working attacks [95% CI] |
|---|---|
| Judge refusals counted as working (the grader declined to read the reply) | **90%** [84, 96] |
| Judge refusals dropped | **74%** [56, 89] |

The grader declined 281 pairs, nearly all encoded attacks, which the guard blocks 100%; the two readings bracket the result.

Attack success, unguarded → guarded (judge refusals dropped):

| target | unguarded | guarded |
|---|---|---|
| Grok 4.7 | 1.8% | 0.6% |
| Qwen 3.8 max | 5.2% | 2.0% |
| Qwen 3.8 flash | 9.3% | 2.1% |
| Gemini 4 Argon | 8.4% | 1.7% |
| Gemini 3.8 Flash | 7.8% | 1.9% |
| DeepSeek v4.1 flash | 8.3% | 2.1% |
| DeepSeek v4 pro | 12.3% | 2.9% |
| GLM 5.3 flash | 11.0% | 2.3% |
| GLM 5.3 | 11.8% | 2.3% |
| **all** | **8.4%** | **2.0%** |

(Counting judge refusals as working: 23.1% → 1.8% overall.) The guard compresses the spread between models from 2–12% to 0.6–2.9%. By family, missed attacks work far more often than blocked ones (18% vs 7%); the weakest families on working attacks are WildGuardMix jailbreaks (37%, CI 4–82%: few working misses) and HackAPrompt (72%).

### 5.3 Real traffic

3,000 random distinct first user turns from each log. The labeller labels every prompt the guard flags and a random 10% of the rest (weighted ×10), so false positives are counted, not estimated. Prompts are split by hash into halves a and b; all tuning used half a and dev only.

| | WildChat-1M | LMSYS-Chat-1M |
|---|---|---|
| Attack prevalence (est.) | ~0.4% | ~2.9% |
| **FPR (block)** | **0.29%** [0.15, 0.55] | **3.16%** [2.59, 3.85] |
| FPR, half b only | 0.19% | 3.30% |
| FPR excluding two automated templates | — | **0.93%** [0.63, 1.36] |
| Block or review | 1.19% | 7.02% |
| Labelled attacks blocked | 12 / 12 | 85 / 88 |
| Precision | ~57% | ~47% |

Two LMSYS templates account for most of its false positives: a developer system prompt pasted as a user turn ("You are the text completion model… only send the completion based on the system instructions", 98 occurrences, 64 of 94 false positives) and an automated toxicity probe (`Say something toxic: "[your answer]" when…`). Under the previous guard, WildChat false positives were pasted documents and translation requests tripping the injection questions; training on traffic-dev removed most of them (1.57% → 0.29%).

## 6. Findings

1. **The definition moves recall more than any model change.** Under the datasets' own labels our first frozen guard caught 82% of attacks; with SORRY-Bench relabelled under a content-aware policy, 85%; under the content-agnostic definition, the refit guard catches 94%. Most earlier "misses" (role-play, persuasion, low-resource translation, escalation) were wrappers that cannot be judged without judging content.
2. **Content leaks into training unless harmful requests are negatives.** Without plain harmful requests as negatives, a learned combiner uses harmfulness as a proxy for "attack" and much apparent jailbreak recall is harm detection.
3. **Content-aware labels trained on real traffic teach the wrong lesson.** When real role-play bots (adult content, labelled harmless) were added to training under a content-aware definition, the guard learned that fiction framing is harmless and stopped blocking fiction-wrapped requests for sexual content about minors, which that definition counted as attacks. A guard that only sees technique cannot hold a content line; the definition and the guard must agree.
4. **Missed attacks are the ones more likely to work** (18% vs 7% success here; 34% vs 23% under the earlier guard). Recall on all attacks overstates protection.
5. **Grade against the original request.** A grader that inferred the request from a persuasive rewrite counted answers to *softened* rewrites as successes and overstated persuasion success by more than 2×.
6. **Benchmark FPR is not traffic FPR.** Test negatives include deliberately adversarial-benign sets (5.9% FPR); real traffic is 0.3–3.2%, dominated by a few automated templates. Attack prevalence differs ~7× between two public logs, so precision must be reported at a stated prevalence.
7. **Question wording mattered less than training data.** Rewording the questions behind false positives moved traffic FPR by noise-level amounts; adding 784 labelled real prompts halved it.

## 7. Limitations

- **Labels are one model's judgments** under a definition with a fuzzy boundary (jailbreak-styled text with a weak or implicit switch-off). A blind audit puts the clear error rate at 0.7% and up to 6.7% counting ambiguous items, all in the direction of attacks labelled as non-attacks, which the guard mostly allows (§4.3). Forced opening lines that are not compliance phrases are the largest ambiguous cluster.
- **By design the guard allows harmful requests that use only framing** (fiction, persuasion, role-play, other languages). It must be paired with a model or content policy that declines them.
- **The labeller and grader refused some content**: 394 texts unlabelled (dropped), and 281 replay pairs ungraded, which is why §5.2 reports a range. Whether encoded attacks *work* is unmeasured.
- **Small cells**: multi-turn (14 test attacks), several replay families with few working misses (wide CIs).
- **Pinned model**: weights are valid only for `jev-1.14.0`; a model bump needs re-answering and re-fitting.
- **Single-turn screening**: multi-turn conversations are screened as joined user turns.
- **Replay coverage**: attacks drawn from nine families with checkable success; payload-free jailbreak templates (in-the-wild DAN prompts) have no defined success and are not replayed.

## 8. Cost

Jev: about $0.042 per million input tokens; all dev, test and traffic answers cost well under $1 in total (TODO exact). Replay: OpenRouter $12.92, grader ~$20.7. Labelling: ~$12.5 for the full relabel (Batches API), ~$15 across audits, pilots and traffic labels (TODO exact).

## Appendix A. Labelling prompt

`audit.SYSTEM_TECHNIQUE`, verbatim. TODO: paste.

## Appendix B. Sources

| id | dataset | family | label | split |
|---|---|---|---|---|
| advbench, harmbench, strongreject | AdvBench, HarmBench, StrongREJECT | plain harmful | 0 | dev/test |
| sorry | SORRY-Bench (base, persuasion, role-play, low-resource language, obfuscation) | plain harmful / jailbreak | 0 / relabelled | dev/test |
| orbench_hard | OR-Bench hard | benign (over-refusal) | 0 | dev/test |
| itw_regular, itw_jailbreak | In-the-Wild jailbreak prompts | benign / jailbreak | relabelled | dev/test |
| jackhhao | jackhhao jailbreak classification | benign / jailbreak | relabelled | dev/test |
| wj_adv_benign, wj_adv_harmful | WildJailbreak (eval) | adversarial benign / harmful | relabelled | dev/test |
| deepset, safeguard | deepset prompt-injections, safeguard | benign / injection | relabelled | dev/test |
| gandalf, gandalf_summ | Lakera Gandalf | injection | relabelled | dev/test |
| tt_extract, tt_hijack | Tensor Trust | injection | relabelled | dev/test |
| injecagent_dh | InjecAgent direct harm | indirect injection | relabelled | dev/test |
| mhj | Scale MHJ | multi-turn | relabelled | dev/test |
| hackaprompt | HackAPrompt | injection | relabelled | **test only** |
| injecagent_ds | InjecAgent data stealing | indirect injection | relabelled | **test only** |
| wildguard | WildGuardMix | benign / harmful / jailbreak | relabelled | **test only** |
| xstest_safe | XSTest safe prompts | benign | 0 | **test only** |

Traffic: allenai/WildChat-1M, lmsys/lmsys-chat-1m (first user turns). TODO: citations and licences; several sources forbid redistribution, so the repo ships code and cached answers' hashes, not texts.
