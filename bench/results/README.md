# Results files

Each file holds one row per task, method and budget. `python -m bench.report FILE` prints a
table. `test.jsonl` is the held-out split (219 tasks, 19 repositories), run once after the methods
were fixed. The others are from the tuning split (281 tasks, 40 repositories).

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
