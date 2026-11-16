# The benchmark

How Repo2Txt's selection is measured, and how to reproduce every number in the README.

## What it measures

A method is given an issue's text and the repository at the issue's commit, and delivers
files or line ranges within a token budget. The delivery is scored against what the issue
needs. "Needed" is not one thing, so each task carries three definitions where they exist:

| Definition | Source | Notes |
|---|---|---|
| `annotated` | ContextBench's gold context: code people traced out from the fix, checked as enough for a model to write one | The primary definition. Sufficient, not minimal. |
| `edited` | The lines the fix itself changed, parsed from its patch (existing files only) | Narrower. On 21% of ContextBench tasks the annotation omits a file the fix edited. |
| `explored` | [SWE-Explore](https://arxiv.org/abs/2606.07297): lines that at least two successful agent runs read | Covers 155 of the verified tasks and fewer of the rest; broader than `annotated`. |

Scores per row and definition: file recall and precision, line recall and precision (overlapping
spans merged; whole files count all their lines), whether every needed file arrived, and
recall without test files. An empty delivery has no precision rather than zero.

## Data

ContextBench at revision `c2855792` (`bench/tasks.py`, checked by sha256). Tasks are split by
repository: `dev` for tuning, `test` held out. `bench/manifest.json` lists the verified
subset's 500 tasks; `bench/manifest-fresh.json` the full release's 636 others (`--subset
fresh`), whose held-out side carried the pre-registered run. `python -m bench.tasks` checks the
data still matches.

## Methods

`bm25`, `llm` and `llm-lines` call `repo_select`, the function behind `repo2txt select`, so
what is measured is what the command delivers. References: `smallest`, `bm25-plain`, and
oracles for each definition (`oracle:<gold>` whole files, `oracle-lines:<gold>` lines).
Every delivery is costed by the library's own packing cost.

Model methods use `BENCH_MODEL` (see `bench/models.py`: model, effort and caching per entry,
set explicitly so runs compare). `compat` calls Anthropic's OpenAI-compatible endpoint with
`LLM_*` settings. The other entries call each provider's native API through a client supplied
locally: `BENCH_CLIENT` names a file whose `make_complete(model)` returns a
`repo2txt.llm.Complete` that passes reasoning state back and reports cache use.

## Checking before trusting

- `python -m bench.check --tasks 12` runs the reference methods twice on real tasks and asserts
  what must hold: oracles reach their ceilings, nothing exceeds its budget, scores lie in
  [0, 1], an empty delivery scores zero, runs reproduce.
- `python -m bench.check --results FILE` audits a results file: errors, overspends, unequal
  coverage, mixed code or model states. `bench.report` runs it first.
- Every row records the code (a hash covering uncommitted changes), the dataset revision, and
  the model, effort and prompt.

## Statistics

`bench/stats.py`: means with 95% intervals from a bootstrap over repositories (tasks from one
repository are not independent), paired differences with better and worse counts,
repository-weighted means, and the range with any one repository left out. Intervals from
fewer than ten repositories are marked. `python -m bench.variance` splits repeated runs'
variance into task and run.

## Reproducing

```bash
uv sync --group bench
python -m bench.check --tasks 12
# the fresh held-out set, model-free methods
python -m bench.run --split test --unseal --subset fresh --tasks 300 \
  --methods smallest,bm25-plain,bm25,oracle:annotated,oracle:edited,oracle-lines:annotated \
  --budgets 8k,32k,128k --out fresh-reference.jsonl
# the model selector (a native client file is needed for anything but compat)
BENCH_MODEL=haiku BENCH_CLIENT=native_client.py python -m bench.run --split test --unseal \
  --subset fresh --tasks 300 --methods llm --budgets 8k,32k --out fresh-llm-haiku.jsonl
python -m bench.report fresh-reference.jsonl --compare bm25:bm25-plain
```

Agent Retrieval Bench: `python -m bench.arb --data DATA --method llm --out FILE` ranks its
samples; `bench/arb_score.py`, run with the benchmark's own package installed, scores them
with its evaluator. Aider's repo map: `bench/aider_rank.py` (export, then rank in aider's own
environment, then `bench.rescore`).

## Results

`bench/results/README.md` indexes the files. In short: `v3/` is the pre-registered run
(`bench/preregistration-2026-11-15.md`), `v2/` the rebuilt harness's reference runs, repeat
runs and prompt experiments, `arb/` Agent Retrieval Bench, and the top level the earlier
tuning, kept as recorded. Large files are `.jsonl.xz`; every reader accepts them.
