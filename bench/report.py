"""Turn a results file into a table.

uv run python -m bench.report bench/results/run.jsonl
"""

import json
import random
import sys
from collections import defaultdict
from statistics import mean

RESAMPLES = 2000


def interval(rows: list[dict], key: str) -> tuple[float, float]:
    """A 95% interval for the mean, resampling whole repositories, since tasks share them."""
    by_repo: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_repo[row["repo"]].append(row[key])
    groups, rng, means = list(by_repo.values()), random.Random(0), []
    for _ in range(RESAMPLES):
        drawn = [value for group in rng.choices(groups, k=len(groups)) for value in group]
        means.append(mean(drawn))
    means.sort()
    return means[int(0.025 * RESAMPLES)], means[int(0.975 * RESAMPLES) - 1]


def table(rows: list[dict]) -> str:
    cells: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        cells[(row["method"], row["budget"])].append(row)
    lines = [
        "| Method | Budget | Tasks | Recall (95% CI) | Line recall | Precision | All gold "
        "| Tokens used |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for (method, budget), group in sorted(cells.items(), key=lambda item: (item[0][1], item[0][0])):
        low, high = interval(group, "recall")
        lines.append(
            f"| {method} | {budget // 1000}k | {len(group)} | "
            f"{mean(r['recall'] for r in group):.2f} ({low:.2f}-{high:.2f}) | "
            f"{mean(r['line_recall'] for r in group):.2f} | "
            f"{mean(r['precision'] for r in group):.2f} | "
            f"{mean(r['all_gold'] for r in group):.0%} | "
            f"{mean(r['tokens'] for r in group):,.0f} |"
        )
    return "\n".join(lines)


def main() -> None:
    rows = [json.loads(line) for line in open(sys.argv[1])]
    print(table(rows))
    tasks = {r["task"]: r for r in rows}.values()
    print(
        f"\n{len(tasks)} tasks, {len({r['repo'] for r in rows})} repositories. "
        f"Gold files absent at the commit and dropped: {sum(r['gold_unresolved'] for r in tasks)}. "
        f"Gold files excluded by repo2txt's filters, counted as misses: "
        f"{sum(r['gold_filtered'] for r in tasks)}."
    )
    # llm-fill reuses llm's call, so count each call once.
    spent = [
        (r["input_tokens"], r["output_tokens"])
        for r in rows
        if r["input_tokens"] and r["method"] != "llm-fill"
    ]
    if spent:
        print(
            f"Model usage: {sum(i for i, _ in spent):,} input and "
            f"{sum(o for _, o in spent):,} output tokens over {len(spent)} selections."
        )
    covered = {
        frozenset(r["task"] for r in rows if (r["method"], r["budget"]) == cell)
        for cell in {(r["method"], r["budget"]) for r in rows}
    }
    if len(covered) > 1:
        print("Warning: the rows do not all cover the same tasks, so they are not comparable.")
    versions = {(r["method"], r["commit"], r["model"]) for r in rows}
    if len(versions) > len({r["method"] for r in rows}):
        print(
            "Note: a method was run at more than one commit or model: "
            + ", ".join(sorted(f"{m}@{c}" for m, c, _ in versions))
        )
    errors = [r for r in rows if r["error"]]
    if errors:
        print(f"{len(errors)} selections failed and scored zero, e.g. {errors[0]['error'][:120]}")


if __name__ == "__main__":
    main()
