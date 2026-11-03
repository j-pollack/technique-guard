# Jev and Claude Sonnet 5.5 as prompt-attack screens

Code and results for the post [Can a decision model screen prompt attacks as well as Claude?](https://www.jonathanpollack.net/writing/decision-models-as-prompt-screens/)

## Check the post's numbers

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python claims.py
```

`claims.py` recomputes every figure in the post from `results/` and `frozen.json`, and compares each with the number the post prints. It needs no API keys and no dataset texts. The costs are the exception: they come from the Anthropic and TypeSafe billing pages.

## What is here

| File | Post section | What it holds |
|---|---|---|
| `prompt_attack_definition.txt` | What counts as a prompt attack, Appendix A | The definition both judges received. |
| `labeler.py` | Methodology, Appendix B | Sonnet's system prompt, user turn and output schema, the batch run, and Grok on the texts Sonnet refused. |
| `kimi.py` | Methodology | Kimi on the texts both Sonnet and Grok refused, and a third opinion on the audit. |
| `jev.py` | Methodology | The Jev client and its answer cache. |
| `jev_question.py` | Methodology, Appendix C | Jev's verdict question, a Noul, and its type question, a Choice, both built from the definition file. |
| `run_jev.py` | Methodology | Scores every text with the verdict question. |
| `freeze.py`, `frozen.json` | Methodology | The thresholds, question hash, definition hash and model version, fixed before the audit was drawn. |
| `sources.py`, `data.py` | The corpus | Where each public dataset comes from, and the rebuild of the dataset and traffic rows from your own downloads. |
| `build_rows.py` | The corpus | Builds the 16,326 texts from those rows, `synthetic/`, and the Enron and LLMail-Inject samples. |
| `build_audit.py` | The audit | Draws the 500 audited texts across the four groups and records each one's weight. |
| `audit_page.py`, `audit_page.html` | The audit | The page used to label the 500 texts blind. |
| `audit_report.py` | Results | Weighted accuracy, the gap's interval, misses and false positives, and the same with parts left out. |
| `jev_type.py` | Where Jev's false positives come from, Which type of attack each judge named | Asks the type question and compares Jev's type with Sonnet's label and the audit's clause. |
| `compare.py` | | Each judge against Kimi or the hand audit, weighted to each population. |
| `publish.py` | | Writes `results/` from the local data, without texts. |
| `claims.py` | All | Checks every number in the post against `results/`. |

## results/

Every row is identified by `key`, the first 16 hex characters of SHA-256 over `"{channel}\n{text}"` (`labeler.key`).

| File | Rows | Fields |
|---|---|---|
| `rows.jsonl` | 16,326 | `key`, `split`, `source`, `channel` |
| `labels.jsonl` | 16,326 | `key`, `label`, `by`: the reference label from Sonnet, or from Grok or Kimi where Sonnet refused |
| `kimi_labels.jsonl` | 547 | `key`, `by`, `label`: Kimi's answers, on the audited texts and on the texts that Sonnet and Grok both refused |
| `jev_scores.jsonl` | 16,326 | `key`, `noul`, `model`: Jev's probability that the text is an attack |
| `jev_types.jsonl` | 16,326 | `key`, `type`, `p`, `model`: Jev's answer to the type question, with each type's probability |
| `audit.jsonl` | 500 | `id`, `key`, `population`, `split`, `source`, `jev`, `claude`, `label`, `by`, `weight`, `you`, `clause`, `kimi`, `group` |

In `audit.jsonl`:

- `jev` and `claude` are each judge's verdict.
- `weight` is the number of texts the audited text stands for.
- `you` and `clause` are the hand label and the clause of the definition it rests on. `A1` to `A9` are attack clauses and `N1` to `N9` are exclusions.
- `group` is the template group the confidence intervals resample as one unit.

Model reasoning is left out, because it quotes the texts.

`results/manifests/` lists the dataset and traffic rows that `data.py rebuild` recovers, by a hash of each text.

## The texts

The texts are not in this repository. Several sources forbid redistribution (SORRY-Bench, LMSYS-Chat-1M), are non-commercial and gated (MHJ), carry no license (Tensor Trust, safeguard), or sit behind a use agreement (WildGuardMix, WildJailbreak, WildChat). The synthetic texts in `synthetic/` are the exception, because they were generated for this study.

## Rebuild from scratch

Put keys in `.env.local`:

| Key | Needed for |
|---|---|
| `HF_TOKEN` | `data.py rebuild` and `build_rows.py`. Accept each gated dataset's terms on Hugging Face first. |
| `ANTHROPIC_API_KEY` | `labeler.py` |
| `OPENROUTER_API_KEY` | Grok in `labeler.py`, and `kimi.py` |
| `TYPESAFE_API_KEY` | `run_jev.py`, `jev_type.py` |

The labeling and scoring steps cost money.

```sh
.venv/bin/python data.py rebuild                       # downloads into raw/, writes data/<split>.json
.venv/bin/python build_rows.py                         # data/rows.jsonl
.venv/bin/python labeler.py data/rows.jsonl data/labels.jsonl batch
.venv/bin/python kimi.py data/unlabeled_keys.txt
.venv/bin/python run_jev.py
.venv/bin/python freeze.py
.venv/bin/python build_audit.py                        # the blind sheet and the audit key
.venv/bin/python audit_report.py data/audit_pass2.csv  # after labeling the sheet by hand
.venv/bin/python jev_type.py
.venv/bin/python publish.py && .venv/bin/python claims.py
```

`data.py rebuild` downloads each source in full and keeps the rows whose hashes the manifests list. The first run is slow, because HackAPrompt is about 600k rows. To check a rebuilt corpus against this one, compare each row's `labeler.key` with `results/rows.jsonl`. A source that has changed since November 2026 will drop or add rows.

## License

MIT for the code. Each dataset keeps its own license.
