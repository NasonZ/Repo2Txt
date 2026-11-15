"""Run selection methods over tasks at several token budgets.

    uv run python -m bench.run --tasks 20 --methods smallest,bm25,oracle --budgets 8k,32k,128k

Rows are appended to a JSONL file as they finish, and rows already there are skipped, so a
run can be stopped and resumed.
"""

import argparse
import json
import os
import subprocess
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from repo2txt import rank
from repo2txt.api import select_lines
from repo2txt.llm import LLMSettings, openai_complete
from repo2txt.models import Repository
from repo2txt.selection import fit
from repo2txt.selector import Selector
from repo2txt.tokens import parse_budget

from . import packing
from .metrics import Gold, resolve, score
from .tasks import Task, repository, sample

# A method returns ranked paths and what it cost: (paths, input tokens, output tokens).
Method = Callable[[Repository, Task, Gold, int], tuple[list[str], int, int]]


def bm25_plain(repo: Repository, task: Task, gold: Gold, budget: int) -> tuple[list[str], int, int]:
    """BM25 alone, as a reference point for the signals added to it."""
    return [path for path, _ in rank.bm25(repo, task.query, boosts=False)], 0, 0


def bm25(repo: Repository, task: Task, gold: Gold, budget: int) -> tuple[list[str], int, int]:
    return [path for path, _ in rank.bm25(repo, task.query)], 0, 0


def smallest(repo: Repository, task: Task, gold: Gold, budget: int) -> tuple[list[str], int, int]:
    """The floor: as many files as fit, ignoring the question."""
    repo.count()
    return sorted(repo.files, key=lambda p: (repo.files[p].tokens, p)), 0, 0


def oracle(repo: Repository, task: Task, gold: Gold, budget: int) -> tuple[list[str], int, int]:
    """The ceiling: the gold files themselves, smallest first, as far as the budget allows."""
    readable = [p for p in gold.files if p in repo.files]
    repo.count(readable)
    return sorted(readable, key=lambda p: repo.files[p].tokens), 0, 0


_chosen: dict[tuple[str, int], tuple[list[str], int, int]] = {}


def llm(repo: Repository, task: Task, gold: Gold, budget: int) -> tuple[list[str], int, int]:
    key = (task.id, budget)
    if key not in _chosen:
        selector = Selector(repo, budget, openai_complete(LLMSettings.from_env()))
        selector.ask(task.query)
        _chosen[key] = (selector.selection.paths, selector.input_tokens, selector.output_tokens)
    return _chosen[key]


def llm_lines(repo: Repository, task: Task, gold: Gold, budget: int):
    """The model's choice at four times the budget, narrowed to line ranges that fit.

    Only the second stage's tokens are counted here; the first stage is the llm row at 4x.
    """
    chosen, _, _ = llm(repo, task, gold, budget * 4)
    usage = [0, 0]

    def complete(messages, tools):
        reply = openai_complete(LLMSettings.from_env())(messages, tools)
        usage[0] += reply.input_tokens
        usage[1] += reply.output_tokens
        return reply

    items = select_lines(repo, task.query, budget, complete, chosen=chosen)
    return _shown(items), sum(cost for _, cost in items), usage[0], usage[1]


def oracle_lines(repo: Repository, task: Task, gold: Gold, budget: int):
    """The needed lines themselves, smallest span first: the ceiling for any packing."""
    shown, used = packing.oracle_lines(repo, task.spans, gold.files, budget)
    return shown, used, 0, 0


def _shown(items: list[tuple[str, int]]) -> dict[str, set[int] | None]:
    shown: dict[str, set[int] | None] = {}
    for item, _ in items:
        path, _, span = item.rpartition(":")
        if path and "-" in span and span.replace("-", "").isdigit():
            first, last = (int(x) for x in span.split("-"))
            if shown.get(path, set()) is not None:
                shown.setdefault(path, set()).update(range(first, last + 1))
        else:
            shown[item] = None
    return shown


# Methods scored by the lines they show, not by whole files kept in rank order.
LINE_RUNNERS = {"llm-lines": llm_lines, "oracle-lines": oracle_lines}


RUNNERS: dict[str, Method] = {
    "smallest": smallest,
    "bm25-plain": bm25_plain,
    "bm25": bm25,
    "oracle": oracle,
    "llm": llm,
}


def run_task(number: int, task: Task, todo: list[tuple[str, int]], meta: dict) -> list[dict]:
    try:
        repo, names = repository(task)
    except Exception as exc:
        print(f"[{number}] {task.id}: could not read {task.repo}: {exc}", flush=True)
        return []
    gold = resolve(task.gold, names, set(repo.files))
    if not gold.files:
        print(f"[{number}] {task.id}: no gold file exists at the commit; skipped", flush=True)
        return []
    rows = []
    for method, budget in todo:
        started = time.time()
        shown = None
        try:
            if method in LINE_RUNNERS:
                shown, used, tokens_in, tokens_out = LINE_RUNNERS[method](repo, task, gold, budget)
                ranked = list(shown)
            else:
                ranked, tokens_in, tokens_out = RUNNERS[method](repo, task, gold, budget)
            error = None
        except Exception as exc:  # A failed call is recorded, reported, and run again next time.
            ranked, tokens_in, tokens_out, error = [], 0, 0, f"{type(exc).__name__}: {exc}"
        selected = ranked if shown is not None else fit(ranked, repo, budget)[0]
        if shown is None:
            shown = dict.fromkeys(selected)
            used = repo.cost(selected)
        lines = packing.score_lines(shown, task.spans, gold.files)
        rows.append(
            {
                "task": task.id,
                "repo": task.repo,
                "language": task.language,
                "source": task.source,
                "split": task.split,
                "method": method,
                "budget": budget,
                "files": len(repo.files),
                "repo_tokens": repo.tokens(),
                "gold": len(gold.files),
                "gold_unresolved": len(gold.unresolved),
                "gold_filtered": len(gold.filtered),
                "selected": len(selected),
                "tokens": used,
                **score(selected, gold.files, task.spans),
                "line_recall": lines["line_recall"],
                "paths": selected,
                "input_tokens": tokens_in,
                "output_tokens": tokens_out,
                "seconds": round(time.time() - started, 2),
                "error": error,
                "model": meta["model"] if method.startswith("llm") else None,
                "commit": meta["commit"],
            }
        )
    print(f"[{number}] {task.id} {task.repo} {len(repo.files)} files", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tasks", type=int, default=20)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--unseal", action="store_true", help="Required to run the test split.")
    parser.add_argument("--methods", default="smallest,bm25,oracle")
    parser.add_argument("--budgets", default="8k,32k,128k")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="bench/results/run.jsonl")
    args = parser.parse_args()
    if args.split == "test" and not args.unseal:
        parser.error("The test split is held out. Tune on dev; pass --unseal for the final run.")
    methods = args.methods.split(",")
    budgets = [parse_budget(b) for b in args.budgets.split(",")]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Keep finished rows; drop failed ones and a last line cut short, so they are run again.
    rows = []
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not row["error"]:
                rows.append(row)
        out.write_text("".join(json.dumps(row) + "\n" for row in rows))
    done = {(row["task"], row["method"], row["budget"]) for row in rows}
    # A model selection already made is reused, so a later method that builds on it is cheap.
    for row in rows:
        if row["method"] == "llm":
            _chosen[(row["task"], row["budget"])] = (row["paths"], 0, 0)
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    meta = {"commit": commit, "model": os.getenv("LLM_MODEL")}

    jobs = []
    for number, task in enumerate(sample(args.tasks, args.split), 1):
        todo = [(m, b) for m in methods for b in budgets if (task.id, m, b) not in done]
        if todo:
            jobs.append((number, task, todo, meta))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for finished in pool.map(lambda job: run_task(*job), jobs):
            with out.open("a") as handle:
                handle.writelines(json.dumps(row) + "\n" for row in finished)


if __name__ == "__main__":
    main()
