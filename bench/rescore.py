"""Score selections saved by an earlier harness, through the current one, without new model calls.

    uv run python -m bench.rescore bench/results/test.jsonl --methods llm --out FILE

Each saved selection's files, in their saved order, are fitted to the current budget room and
scored against every definition of what the task needs. The method is renamed
`<method>@saved`, since it is the earlier selection, not a new run of the method: the
earlier harness gave the model 60 more tokens of budget, so a selection may lose a file at
the boundary.
"""

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import run
from .tasks import all_tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path)
    parser.add_argument("--methods", default="llm")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    methods = args.methods.split(",")
    saved: dict[str, dict[tuple[str, int], list[str]]] = defaultdict(dict)
    for line in args.results.read_text().splitlines():
        row = json.loads(line)
        if row["method"] in methods and not row.get("error"):
            saved[row["task"]][(row["method"], row["budget"])] = row["paths"]
    for method in methods:
        name = f"{method}@saved"

        def runner(repo, task, gold, room, usage, method=method):
            budget = room + run.HEAD_TOKENS
            paths = [p for p in saved[task.id][(method, budget)] if p in repo.files]
            return run._fitted(repo, paths, room)

        run.RUNNERS[name] = runner
    meta = run.provenance([]) | {"source": str(args.results)}
    tasks = {t.id: t for t in all_tasks()}
    jobs = [
        (tasks[task_id], [(f"{m}@saved", b) for m, b in sorted(cells)], meta)
        for task_id, cells in saved.items()
    ]
    with ThreadPoolExecutor(4) as pool:
        rows = [r for done in pool.map(lambda job: run.run_task(*job), jobs) for r in done]
    args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{args.out}: {len(rows)} rows from {len(jobs)} tasks")


if __name__ == "__main__":
    main()
