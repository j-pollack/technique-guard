# Is one question enough? A decision model as a prompt-attack guard

Code and results for [PAPER.md](PAPER.md). Everything a reader needs to check the paper's numbers: the definition and labeler, the data sources and splits, every wording of the question, the comparator's frozen weights, and the scoring behind each table.

## What is here

| file | paper | what it holds |
|---|---|---|
| `labeler.py` | §2, §3.2, Appendix B | The labeling prompt (definition hash `dd585be4cec7`), the rules for which rows keep their dataset label, and a command to re-ask the labeler. |
| `sources.py` | §3.1, Table 2 | Where each of the 25 source cells comes from and how a dataset row becomes a text and a label. Also the split rule and the near-duplicate test. |
| `questions.py` | §4, Appendix A | Every wording of the question, stages 1 to 7 and the final one, and the clause cuts of the ablation. |
| `jev.py` | §4.1, §4.4 | The decision-model client and its answer cache. |
| `guard_v12.py`, `guard_v12.json` | §4.3 | The comparator: its 36 questions, feature recipe, 56 weights plus intercept, and thresholds. |
| `data.py` | §3.3, §3.4, Appendices C, D | Rebuilds every scored row from your own downloads, generates the fresh rows, and checks the splits. |
| `evaluate.py` | §4.2, §5 | One command per table, each compared against `results/`. |
| `results/` | | The files the paper's numbers were read from (below). |

### results/

The `*.json` files at the top level are the summaries the paper's tables were read from. They contain thresholds, counts, intervals, AUCs and per-family rates, and no dataset text.

| file | paper |
|---|---|
| `stage1_bare.json`, `stage2_definition.json` | Tables 4, 5 |
| `stage3_definition_v2.json`, `stage4_definition_v3.json`, `stage5_definition_v5.json` | §4.2, the struck figures |
| `heldout.json` | Tables 6–9 |
| `fresh.json` | Table 10 |
| `ablation.json` | Table 12 |
| `traffic.json` | Table 13 |
| `heldout_<model>.json`, `fresh_<model>.json` | Tables 15, 16 |

`results/manifests/` lists every row the paper scored, one file per split: `dev`, `spent_test`, `heldout`, `gcg_proxy`, `fresh`, `traffic_wildchat` and `traffic_lmsys`, plus `heldout_disagreements.jsonl`, the rows where the question and the label disagree. Each row is identified by `sha`, the first 16 hex characters of SHA-256 over its text. Each row also carries its source, family, near-duplicate group and split, the dataset's label, the paper's label, and `label_from`: `sonnet` if the labeler gave the label, `dataset` if a rule kept the dataset's, or null if the labeler refused and the row was dropped. Traffic rows carry the log's conversation id, the half they fall in, and the labeler's answer where there is one.

The texts are not in this repository. Several sources forbid redistribution (SORRY-Bench, LMSYS-Chat-1M), are non-commercial and gated (MHJ), carry no license (Tensor Trust, safeguard), or sit behind a use agreement (WildGuardMix, WildJailbreak, WildChat). `data.py rebuild` recovers them from your own downloads.

## Checking the numbers

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Put keys in `.env.local`:

| key | needed for |
|---|---|
| `HF_TOKEN` | `data.py rebuild`. Accept each gated dataset's terms on Hugging Face first. |
| `TYPESAFE_API_KEY` | `evaluate.py`, for Jev's answers |
| `ANTHROPIC_API_KEY` | `labeler.py` only |

**1. Rebuild the rows.**

```sh
.venv/bin/python data.py rebuild    # downloads into raw/, writes data/<split>.json
.venv/bin/python data.py check
```

The paper's own pulls were random 100-row pages of each dataset, which cannot be repeated row for row. `rebuild` therefore downloads each source in full and keeps the rows whose hashes the manifests list. It fetches the traffic rows by conversation id, regenerates the 280 fresh rows with seed 0, and checks each fresh row against its hash. If a dataset has since changed or dropped rows, it reports how many it could not find, by source. The first run is slow, because HackAPrompt is about 600k rows. `check` confirms these claims from §3.3:

- every row sits on the side its group hash gives;
- no held-out row shares a near-duplicate group with a dev or spent-test row;
- no held-out text is a near-duplicate of a dev or spent-test text;
- no held-out row is a GCG proxy row;
- no example string in the final question occurs in any spent-test, held-out or fresh row.

**2. Score.** Each command prints its table and compares its output with the matching file in `results/`, reporting every value that differs.

| command | paper |
|---|---|
| `evaluate.py stage bare`, `evaluate.py stage definition` | Tables 4, 5 |
| `evaluate.py heldout` | Tables 3, 6, 7, 8, 9, and the 15 misses |
| `evaluate.py fresh` | Table 10 |
| `evaluate.py cells fresh bare definition final no_adversarial no_transform` | Table 11 |
| `evaluate.py ablate` | Table 12 |
| `evaluate.py traffic` | Tables 13, 14 |
| `JEV_URL=… JEV_MODEL=openjev-MLX-4bit JEV_WORKERS=1 evaluate.py open heldout` (and `fresh`, and `clef-oQ4e`) | Tables 15, 16 |

Thresholds are set as the paper sets them: the score at which 5% of dev non-attacks are blocked. Answers are cached in `data/answers.jsonl`, so a rerun costs nothing. The paper's runs cost about $4.50 of Jev in total (§8). The open-weight runs need a local server speaking Jev's API; the paper used oMLX 0.7.2.

**3. Re-ask the labeler** (optional). `labeler.py heldout 100` relabels 100 random held-out rows that the labeler labelled, and reports how many agree with the paper.

## Not included

- The fit of guard v1.2. Only its frozen output, `guard_v12.json`, is here.
- The code that drew the held-out set. Its composition is in the manifest, and `data.py check` verifies the properties §3.3 claims for it.
- The two labeler agreement checks of §3.2, which were run before this work.

## License

MIT for the code. Each dataset keeps its own license.
