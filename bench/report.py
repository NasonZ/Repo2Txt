"""Turn a results file into tables, after checking it.

    uv run python -m bench.report bench/results/FILE.jsonl [--gold annotated|edited]
        [--compare llm:bm25]

Files and lines are reported in separate tables. A method that shows line ranges is scored
in the line table only: counting a file as found when one of its lines was shown would put
it beside whole-file methods on a different footing.
"""

import argparse
import json
from collections import defaultdict
from statistics import mean

from .check import check_rows
from .stats import Estimate, estimate, field, leave_one_repo_out, paired

LINE_METHODS = ("llm-lines", "oracle-lines")


def _cell(e: Estimate) -> str:
    return f"{e.mean:.2f} ({e.low:.2f}-{e.high:.2f})"


def _defined(rows: list[dict], name: str, gold: str) -> float | None:
    values = [r["scores"][gold][name] for r in rows if r["scores"][gold][name] is not None]
    return mean(values) if values else None


def tables(rows: list[dict], gold: str) -> str:
    cells: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        if row["status"] == "ok":
            cells[(row["method"], row["budget"])].append(row)
    order = sorted(cells, key=lambda c: (c[1], c[0]))
    out = [
        f"Gold: {gold}. Means over tasks, 95% intervals resampling repositories; "
        "'by repo' weights each repository equally.\n",
        "| Method | Budget | Tasks | File recall | by repo | Precision | Empty | Tokens |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for method, budget in order:
        if method.startswith(LINE_METHODS):
            continue
        group = cells[(method, budget)]
        recall = estimate(group, field("recall", gold))
        precision = _defined(group, "precision", gold)
        empty = sum(1 for r in group if not r["delivered"]) / len(group)
        out.append(
            f"| {method} | {budget // 1000}k | {recall.tasks} | {_cell(recall)} | "
            f"{recall.repo_mean:.2f} | {precision if precision is None else f'{precision:.2f}'} | "
            f"{empty:.0%} | {mean(r['tokens'] for r in group):,.0f} |"
        )
    out += [
        "",
        "| Method | Budget | Tasks | Line recall | by repo | Line precision | Tokens |",
        "|---|---|---|---|---|---|---|",
    ]
    for method, budget in order:
        group = cells[(method, budget)]
        if not any(r["scores"][gold]["line_recall"] is not None for r in group):
            continue
        lines = estimate(group, field("line_recall", gold))
        precision = _defined(group, "line_precision", gold)
        out.append(
            f"| {method} | {budget // 1000}k | {lines.tasks} | {_cell(lines)} | "
            f"{lines.repo_mean:.2f} | {precision if precision is None else f'{precision:.2f}'} | "
            f"{mean(r['tokens'] for r in group):,.0f} |"
        )
    return "\n".join(out)


def comparison(rows: list[dict], a: str, b: str, gold: str) -> str:
    out = [
        f"\nPaired, {a} minus {b} ({gold} gold): difference, interval, better/worse tasks, "
        "and the range of the difference with any one repository left out."
    ]
    budgets = sorted({r["budget"] for r in rows if r["method"] in (a, b)})
    for budget in budgets:
        ra = [r for r in rows if r["method"] == a and r["budget"] == budget and r["status"] == "ok"]
        rb = [r for r in rows if r["method"] == b and r["budget"] == budget and r["status"] == "ok"]
        if not ra or not rb:
            continue
        lines = a.startswith(LINE_METHODS) or b.startswith(LINE_METHODS)
        name = "line_recall" if lines else "recall"
        d = paired(ra, rb, field(name, gold))
        low, high = leave_one_repo_out(ra, rb, field(name, gold))
        out.append(
            f"- {budget // 1000}k {name}: {d.mean:+.3f} ({d.low:+.3f} to {d.high:+.3f}), "
            f"better {d.better}, worse {d.worse}, {d.repos} repos; "
            f"leave one out {low:+.3f} to {high:+.3f}"
        )
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results")
    parser.add_argument("--gold", default="annotated", choices=["annotated", "edited", "explored"])
    parser.add_argument("--compare", action="append", default=[], help="A:B, e.g. llm:bm25")
    args = parser.parse_args()
    rows = [json.loads(line) for line in open(args.results)]
    problems = check_rows(rows)
    for message in problems:
        print("WARNING", message)
    print(tables(rows, args.gold))
    for pair in args.compare:
        print(comparison(rows, *pair.split(":"), args.gold))
    ok = [r for r in rows if r["status"] == "ok"]
    tasks = {r["task"]: r for r in ok}.values()
    gold = [r["gold"][args.gold] for r in tasks]
    skipped = {r["task"] for r in rows if r["status"] == "skipped"}
    print(
        f"\n{len(tasks)} tasks from {len({r['repo'] for r in tasks})} repositories; "
        f"{len(skipped)} skipped. Gold files absent at the commit, dropped: "
        f"{sum(g['unresolved'] for g in gold)}; excluded by repo2txt's filters, always missed: "
        f"{sum(g['filtered'] for g in gold)}."
    )
    spent = [r for r in ok if r["model_calls"]]
    if spent:
        print(
            f"Model usage: {sum(r['input_tokens'] for r in spent):,} input and "
            f"{sum(r['output_tokens'] for r in spent):,} output tokens in "
            f"{sum(r['model_calls'] for r in spent):,} calls."
        )


if __name__ == "__main__":
    main()
