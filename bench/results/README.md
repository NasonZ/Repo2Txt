# Results files

Each file holds one row per task, method and budget. The files at this level come from the
first harness (14-15 November) and are kept as recorded; `python -m bench.rescore` scores their
saved selections through the current one. `test.jsonl` is the verified subset's held-out split
(219 tasks, 19 repositories); the others are from its tuning split (281 tasks, 40
repositories). The rebuilt harness's files are in `v2/`, `v3/` and `arb/`, described below.

| File | What it holds |
|---|---|
| `pilot.jsonl` | 20 tasks from the first sample, before the split existed. Kept for the record. |
| `dev-free.jsonl` | The model-free methods on all 281 tuning tasks. |
| `dev-llm.jsonl` | The first model selector, alone and topped up, on 100 tasks at three budgets. |
| `dev-tune.jsonl`, `dev-tune2.jsonl`, `dev-tune-x3.jsonl` | Selector changes tried at 32k on the same 100 tasks. |
| `dev-v2-budgets.jsonl` | The selector that was kept, at 8k and 128k. |
| `dev-packing.jsonl` | Whole files, focused views and a hybrid, scored by needed lines (`bench/packing.py`). |
| `dev-ranges.jsonl` | The model choosing line ranges from outlines, without the whole-file fill (`bench/ranges.py`). |
| `test.jsonl` | Every kept method on the held-out split, with `llm-lines` and `oracle-lines`. |

Method names in these files, some of which no longer exist in the code:

| Name | Meaning | Outcome |
|---|---|---|
| `smallest` | As many files as fit, smallest first | Floor |
| `bm25-plain` | BM25 over paths and text | Reference |
| `bm25` | BM25 plus path mentions, definitions and demoted tests | The default for `select` |
| `bm25-cut` | `bm25`, dropping files under a quarter of the top score | Lost recall; removed |
| `oracle` | The needed files themselves | Ceiling |
| `llm` | The first model selector: tree only, short prompt | Replaced |
| `llm-v1` | `llm` plus the top 30 `bm25` files beside the request | No gain alone |
| `llm-v2` | `llm-v1` plus a prompt that favours completeness and reports budget use | Kept: this is `--method llm` now |
| `llm-v2n` | `llm-v2` without the shortlist | 4 points worse; shortlist kept |
| `llm-v3` | `llm-v2` plus search and outline tools | Worse: a search bug and rounds running out |
| `llm-v3b` | `llm-v3` with those faults fixed | Tied `llm-v2` at 1.5 times the cost; removed |
| `llm-v2x3` | Three `llm-v2` runs merged by votes | No gain at three times the cost; removed |
| `llm-lines` | `llm` at four times the budget, then line ranges from outlines, then whole files | Kept: `--lines` |
| `oracle-lines` | The needed lines themselves | Line ceiling |
| `*-fill` | A model selection topped up from `bm25-cut` | +2 points at 128k for half the precision; removed |

The model was `claude-haiku-5-5` throughout.

## The rebuilt harness (15-16 November)

Rows from here on carry the delivery itself, scores against each definition of what a task
needs (`scores.annotated`, `scores.edited`, `scores.explored`) and the code, data and model
they came from (`run`). See bench/README.md.

### `v3/`: the pre-registered run (bench/preregistration-2026-11-15.md)

| File | What it holds |
|---|---|
| `fresh-reference.jsonl.xz` | Model-free methods and oracles on the 255 fresh held-out tasks, 8k/32k/128k. |
| `fresh-llm-haiku.jsonl`, `-sonnet`, `-deepseek` | The model selector with each model, 8k and 32k. |
| `fresh-lines-haiku.jsonl` | Line ranges with Haiku at 8k. |
| `arb-llm-haiku.jsonl.xz`, `-sonnet`, `-deepseek` | Agent Retrieval Bench details, one per model; `arb-report/` is its evaluator's output. |

### `v2/`: reference runs, repeats and experiments

| File | What it holds |
|---|---|
| `dev-reference.jsonl.xz`, `test-reference.jsonl.xz` | Model-free methods and oracles on the verified subset's two splits. |
| `test-llm-saved.jsonl`, `dev-llm-saved.jsonl` | The earlier harness's saved model selections, rescored (`bench.rescore`). |
| `test-llm-run1..3.jsonl` | Three repeat held-out runs at 32k (compatible route, the selector before the prefix change), for `bench.variance`. |
| `test-repomap.jsonl` | Aider's repo map on the held-out split. |
| `pilot-*.jsonl` | Each model, natively, on 10 tuning tasks. |
| `ab-*.jsonl` | Tuning-task experiments at 32k: `current` against `v3` (prompt v3), `state` (selected paths in tool replies), on Haiku, DeepSeek and the compatible route. |

### `arb/`: Agent Retrieval Bench, first runs

`repo2txt-bm25`, `-bm25-plain` and `-llm` (the earlier selector, compatible route), with
`report/` from the benchmark's evaluator.
