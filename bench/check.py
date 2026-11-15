"""Check the harness before trusting a number from it, and check a results file before reporting it.

    uv run python -m bench.check --tasks 12            # run the reference methods and test them
    uv run python -m bench.check --results FILE        # audit a results file

The first runs the methods that need no model on a few tasks and asserts what must hold if
the harness is right: oracles reach their ceilings, nothing exceeds its budget, scores lie in
[0, 1], an empty delivery scores zero, and the methods without a model give the same
delivery twice. The second asserts that a results file is complete and consistent.
"""

import argparse
import json
import sys
from collections import defaultdict

from repo2txt.api import HEAD_TOKENS

from .metrics import score
from .run import MODEL_METHODS, RUNNERS, run_task
from .tasks import sample

BUDGETS = [8000, 32000]
CHECKED = [m for m in RUNNERS if m not in MODEL_METHODS]


def _scores_in_range(row: dict) -> list[str]:
    bad = []
    for kind, scores in row["scores"].items():
        for name, value in scores.items():
            if value is not None and not 0 <= value <= 1:
                bad.append(f"{kind}.{name}={value}")
    return bad


def check_rows(rows: list[dict]) -> list[str]:
    """Problems with a set of rows, as messages. An empty list means none were found."""
    problems = []
    cells: dict[tuple[str, int], set[str]] = defaultdict(set)
    runs: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        where = f"{row['task']} {row['method']}@{row['budget']}"
        if row["status"] == "error":
            problems.append(f"{where}: failed: {row['error']}")
            continue
        cells[(row["method"], row["budget"])].add(row["task"])
        if row["status"] == "skipped":
            continue
        runs[row["method"]].add(json.dumps(row["run"], sort_keys=True))
        if row["tokens"] > row["budget"] - HEAD_TOKENS:
            problems.append(f"{where}: {row['tokens']} tokens exceed the room of {row['room']}")
        if bad := _scores_in_range(row):
            problems.append(f"{where}: scores out of range: {bad}")
        if row["selected"] != len(row["delivered"]):
            problems.append(f"{where}: selected count does not match the delivery")
    if len({frozenset(tasks) for tasks in cells.values()}) > 1:
        sizes = {f"{m}@{b}": len(t) for (m, b), t in sorted(cells.items())}
        problems.append(f"methods do not cover the same tasks: {sizes}")
    for method, seen in runs.items():
        if len(seen) > 1:
            problems.append(f"{method} rows come from {len(seen)} different code or model states")
    return problems


def check_harness(count: int) -> list[str]:
    problems = []
    tasks = sample(count, "dev")
    for task in tasks:
        todo = [(m, b) for m in CHECKED for b in BUDGETS]
        meta = {"check": True}
        first, second = run_task(task, todo, meta), run_task(task, todo, meta)
        problems += check_rows(first)
        for a, b in zip(first, second, strict=True):
            if a["status"] == "ok" and a["delivered"] != b["delivered"]:
                problems.append(f"{task.id} {a['method']}@{a['budget']}: not reproducible")
        for row in first:
            if row["status"] != "ok" or not row["method"].startswith("oracle"):
                continue
            kind = row["method"].split(":")[1]
            gold, scores = row["gold"][kind], row["scores"][kind]
            whole_files = not row["method"].startswith("oracle-lines")
            readable = gold["files"] - gold["filtered"]
            if not gold["files"] or not readable:
                continue
            ceiling = readable / gold["files"]
            if whole_files and row["selected"] == readable and scores["recall"] != ceiling:
                problems.append(f"{task.id} {row['method']}: recall below {ceiling}")
            if whole_files and scores["precision"] not in (None, 1.0):
                problems.append(f"{task.id} {row['method']}: precision {scores['precision']} < 1")
            if not whole_files and scores["line_precision"] not in (None, 1.0):
                problems.append(f"{task.id} {row['method']}: shows lines that are not needed")
    empty = score({}, ["a.py"], [["a.py", 1, 3]], {"a.py": 10})
    if empty["recall"] != 0 or empty["line_recall"] != 0 or empty["precision"] is not None:
        problems.append(f"an empty delivery scores {empty}")
    print(f"checked {len(tasks)} tasks x {len(CHECKED)} methods x {len(BUDGETS)} budgets, twice")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tasks", type=int, default=12)
    parser.add_argument("--results", help="Audit this results file instead.")
    args = parser.parse_args()
    if args.results:
        rows = [json.loads(line) for line in open(args.results)]
        problems = check_rows(rows)
    else:
        problems = check_harness(args.tasks)
    for message in problems:
        print("FAIL", message)
    print("ok" if not problems else f"{len(problems)} problems")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
