# Pre-registration: the cross-model run, 15 November 2026

Written and committed before any of the runs below. The configuration is frozen at the commit
that adds this file; anything changed afterwards is reported as a deviation.

## What is being tested

Repo2Txt's `select --method llm`, the model selector, as it now stands: a fixed system prompt
and tools on every round (so a prompt cache can reuse each request's opening), each tool
reply ending with the selected paths, reasoning passed back between rounds, and each model
called through its provider's own API. Against it: `select` (BM25 with three signals, the
default) and plain BM25, both without a model, and the needed files themselves as a ceiling.

Models, effort set explicitly on each (bench/models.py): Claude Haiku 5.5 (medium), Claude
Sonnet 5.5 (medium), DeepSeek flash (high, its lowest honoured level). GPT-6.1 Terra joins if
Bedrock access is granted before the run; if so it is run identically and reported
separately as a late addition.

## Data

1. **Fresh held-out set**: the 255 tasks of ContextBench's full release (revision
   c2855792, sha256 in bench/tasks.py) that are outside the verified subset and fall on the
   held-out side of the repository split; 21 repositories. Task ids: bench/manifest-fresh.json.
   No method has been run on these tasks, and nothing has been tuned on them.
2. **Agent Retrieval Bench**: all 345 positive samples, scored by the benchmark's own
   evaluator. No method here was tuned on it; the model selector ran on it once before, with
   the earlier selector, and that result stays as recorded.

## Runs

- Fresh set: `smallest`, `bm25-plain`, `bm25`, oracles, at 8k, 32k and 128k; the model
  selector with each model at 8k and 32k; `llm-lines` (line ranges) with Haiku at 8k.
  One run each.
- Agent Retrieval Bench: the model selector with each model at 8k, one run each.

## Hypotheses (primary)

Each tested on the fresh set at 32k against the annotated gold, as a paired difference in
file recall with a 95% interval from resampling whole repositories (bench/stats.py: 4,000
resamples, seed 0).

- **H1**, for each model separately: the model selector's recall exceeds `select`'s.
  Supported if the interval lies above zero.
- **H2**: `select` (with its three signals) exceeds plain BM25. Supported if the interval
  lies above zero.

## Reported, not tested (secondary)

- The same comparisons against the edited gold and, where it exists, SWE-Explore's gold.
- The models against each other: recall, precision, tokens delivered, model calls, cost per
  selection from measured tokens and published prices, and seconds per selection.
- 8k results, and line recall for `llm-lines` against the line ceiling.
- Agent Retrieval Bench: Recall@5/20, MRR and BCY at 4k-32k, by task type and by its dev and
  held-out halves.

With 21 repositories the intervals are wide; differences between models smaller than about
five points of recall are not expected to be resolvable, and will be described as such.
Earlier work measured a run-to-run spread of 0.003 in mean recall (four runs, 219 tasks), so
one run per configuration is taken as adequate for means; single-task results are not.

## Known limits, stated now

- The selector's structure changed on the tuning tasks today (stable prefix, paths in tool
  replies); on those tasks the change cost an estimated 0.016 to 0.020 recall against the old
  structure, an interval that includes zero.
- Token budgets are counted in o200k_base; each provider's own tokeniser differs.
- Whether models have seen these repositories or fixes in training is not measured.

## Deviations, recorded as they happen

- **One fresh task has no issue text.** `SWE-Bench-Pro__python__maintenance__bugfix__788af515`
  (ansible/ansible) has an empty `problem_statement` in the dataset, so no method can be
  given a question for it. It is skipped with that reason; the fresh set is 254 tasks. None
  of the verified subset's 500 tasks has this defect.
- **Download failures were being recorded as skips.** Two repositories (huggingface/transformers,
  cli/cli) failed to download once during the reference run; the runner marked those tasks
  skipped, which a resumed run does not retry. They are now errors, which it does; the
  affected rows are removed and the run resumed. No result changes from this, only coverage.
