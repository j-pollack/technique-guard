# v2: Jev and Claude Sonnet 5.5 as prompt-attack screens

Code and results for the post [Can a decision model screen prompt attacks as well as Claude?](https://www.jonathanpollack.net/writing/decision-models-as-prompt-screens/)

## Check the post's numbers

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python v2/claims.py
```

`claims.py` recomputes every figure in the post from `v2/results/` and `v2/frozen.json`, and compares each with the number the post prints. It needs no API keys and no dataset texts. The costs are the exception: they come from the Anthropic and TypeSafe billing pages.

## What is here

| File | Post section | What it holds |
|---|---|---|
| `prompt_attack_definition.txt` | What counts as a prompt attack, Appendix A | The definition both judges received. |
| `labeler.py` | Methodology, Appendix B | Sonnet's system prompt, user turn and output schema, the batch run, and Grok on the texts Sonnet refused. |
| `kimi.py` | Methodology | Kimi on the 48 texts both Sonnet and Grok refused, and a third opinion on the audit. |
| `jev_question.py` | Methodology, Appendix C | Jev's verdict question, a Noul, and its type question, a Choice, both built from the definition file. |
| `run_jev.py` | Methodology | Scores every text with the verdict question. |
| `freeze.py`, `frozen.json` | Methodology | The thresholds, question hash, definition hash and model version, fixed before the audit was drawn. |
| `build_rows.py` | The corpus | Builds the 16,326 texts from the v1 rows, the traffic sample, `synthetic/`, and the new Enron and LLMail-Inject samples. |
| `build_audit.py` | The audit | Draws the 500 audited texts across the four groups and records each one's weight. |
| `audit_page.py` | The audit | The page used to label the 500 texts blind. |
| `audit_report.py` | Results | Weighted accuracy, the gap's interval, misses and false positives, and the same with parts left out. |
| `jev_type.py` | Where Jev's false positives come from, Which type of attack each judge named | Asks the type question and compares Jev's type with Sonnet's label and the audit's clause. |
| `publish.py` | | Writes `results/` from the local data, without texts. |
| `claims.py` | All | Checks every number in the post against `results/`. |
| `compare.py`, `jev_four.py`, `pilot/` | | Earlier comparisons and pilots that shaped the study. The post does not cite them. |

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

The texts are not in this repository. Several sources forbid redistribution or sit behind use agreements. The top-level [README](../README.md) lists them. The synthetic texts in `synthetic/` are the exception, because they were generated for this study.

## Rebuild from scratch

This needs the keys in the top-level README, plus `OPENROUTER_API_KEY` for Grok and Kimi. The labeling and scoring steps cost money.

```sh
.venv/bin/python data.py rebuild                            # v1 rows, from your own downloads
.venv/bin/python v2/build_rows.py                           # v2/data/rows.jsonl
.venv/bin/python v2/labeler.py v2/data/rows.jsonl v2/data/labels.jsonl batch
.venv/bin/python v2/kimi.py v2/data/unlabeled_keys.txt
.venv/bin/python v2/run_jev.py
.venv/bin/python v2/freeze.py
.venv/bin/python v2/build_audit.py                          # the blind sheet and the audit key
.venv/bin/python v2/audit_report.py v2/data/audit_pass2.csv # after labeling the sheet by hand
.venv/bin/python v2/jev_type.py
.venv/bin/python v2/publish.py && .venv/bin/python v2/claims.py
```

To check a rebuilt corpus against this one, compare each row's `labeler.key` with `results/rows.jsonl`. A source that has changed since November 2026 will drop or add rows.
