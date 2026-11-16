"""How much a model method's result moves between identical runs.

    uv run python -m bench.variance --method llm --budget 32000 RUN.jsonl RUN.jsonl ...

Given several runs of one method on the same tasks, reports each run's mean with its
interval, the spread of those means, and how the variance splits between tasks (how hard
each issue is) and runs (the model choosing differently each time). If most of it lies
between runs, more runs sharpen an estimate; if between tasks, only more tasks do. It also
reports how often a task's score changes from run to run, the noise a single-run comparison
of two methods carries.
"""

import argparse
from collections import defaultdict
from statistics import mean, pstdev, variance

from .io import read_rows
from .stats import estimate, field


def load(path: str, method: str, budget: int) -> list[dict]:
    rows = []
    for row in read_rows(path):
        if row["method"].split("@")[0] == method and row["budget"] == budget:
            if row["status"] == "ok":
                rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--method", default="llm")
    parser.add_argument("--budget", type=int, default=32000)
    parser.add_argument("--gold", default="annotated")
    parser.add_argument("--score", default="recall")
    args = parser.parse_args()
    value = field(args.score, args.gold)
    runs = [load(p, args.method, args.budget) for p in args.runs]
    common = set.intersection(*({r["task"] for r in run} for run in runs))
    runs = [[r for r in run if r["task"] in common] for run in runs]
    print(
        f"{len(runs)} runs of {args.method} at {args.budget // 1000}k on {len(common)} tasks, "
        f"{args.score} against the {args.gold} gold\n"
    )
    means = []
    for path, run in zip(args.runs, runs, strict=True):
        e = estimate(run, value)
        means.append(e.mean)
        print(f"  {path}: {e.mean:.3f} ({e.low:.3f}-{e.high:.3f})")
    print(f"\nspread of the run means: sd {pstdev(means):.4f}, range {max(means) - min(means):.4f}")

    scores: dict[str, list[float]] = defaultdict(list)
    for run in runs:
        for row in run:
            v = value(row)
            if v is not None:
                scores[row["task"]].append(v)
    complete = {t: v for t, v in scores.items() if len(v) == len(runs)}
    within = mean(variance(v) for v in complete.values())
    between = variance([mean(v) for v in complete.values()])
    changed = sum(1 for v in complete.values() if max(v) != min(v))
    total = between + within
    print(
        f"variance between tasks {between:.4f}, within a task across runs {within:.4f}: "
        f"{between / total:.0%} of it is the task, {within / total:.0%} the run"
    )
    print(
        f"tasks whose score changed between runs: {changed} of {len(complete)} "
        f"({changed / len(complete):.0%})"
    )
    every = sum(1 for v in complete.values() if min(v) == 1.0)
    some = sum(1 for v in complete.values() if max(v) == 1.0)
    print(f"all needed files delivered in every run: {every}; in at least one run: {some}")


if __name__ == "__main__":
    main()
