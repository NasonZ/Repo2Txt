"""Run selection methods over tasks at several token budgets.

    uv run python -m bench.run --tasks 20 --methods smallest,bm25,oracle --budgets 8k,32k,128k

Rows are appended to a JSONL file as they finish, and rows already there are skipped, so a
run can be stopped and resumed.
"""

import argparse
import io
import json
import os
import subprocess
import tarfile
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from repo2txt import rank
from repo2txt.llm import LLMSettings, openai_complete
from repo2txt.models import RepoFile, Repository
from repo2txt.selection import fit
from repo2txt.selector import Selector
from repo2txt.sources import decode, download_tarball, is_candidate, tar_files
from repo2txt.tokens import parse_budget

from .metrics import Gold, resolve, score
from .tasks import CACHE, Task, sample

# A method returns ranked paths and what it cost: (paths, input tokens, output tokens).
Method = Callable[[Repository, Task, Gold, int], tuple[list[str], int, int]]


# Files scoring below this share of the best score are left out rather than used as padding.
CUT = 0.25
_llm_paths: dict[tuple, tuple[list[str], int, int]] = {}


def bm25_plain(repo: Repository, task: Task, gold: Gold, budget: int) -> tuple[list[str], int, int]:
    """BM25 alone, as a reference point for the signals added to it."""
    return [path for path, _ in rank.bm25(repo, task.query, boosts=False)], 0, 0


def bm25(repo: Repository, task: Task, gold: Gold, budget: int) -> tuple[list[str], int, int]:
    return [path for path, _ in rank.bm25(repo, task.query)], 0, 0


def bm25_cut(repo: Repository, task: Task, gold: Gold, budget: int) -> tuple[list[str], int, int]:
    ranked = rank.bm25(repo, task.query)
    floor = CUT * ranked[0][1] if ranked else 0
    return [path for path, score in ranked if score >= floor], 0, 0


def smallest(repo: Repository, task: Task, gold: Gold, budget: int) -> tuple[list[str], int, int]:
    """The floor: as many files as fit, ignoring the question."""
    repo.count()
    return sorted(repo.files, key=lambda p: (repo.files[p].tokens, p)), 0, 0


def oracle(repo: Repository, task: Task, gold: Gold, budget: int) -> tuple[list[str], int, int]:
    """The ceiling: the gold files themselves, smallest first, as far as the budget allows."""
    readable = [p for p in gold.files if p in repo.files]
    repo.count(readable)
    return sorted(readable, key=lambda p: repo.files[p].tokens), 0, 0


# Settings of the model selector under test. "llm" is the selector as first shipped.
VARIANTS: dict[str, dict] = {
    "llm": {},
    "llm-v1": {"shortlist": 30},
    "llm-v2": {"shortlist": 30, "recall": True},
    "llm-v3": {"shortlist": 30, "recall": True, "explore": True},
}
RUNS = 3  # for the "x3" methods, which merge several independent selections


def _select(variant: str, repo: Repository, task: Task, budget: int, run: int = 0):
    key = (variant, task.id, budget, run)
    if key not in _llm_paths:
        selector = Selector(
            repo, budget, openai_complete(LLMSettings.from_env()), **VARIANTS[variant]
        )
        selector.ask(task.query)
        _llm_paths[key] = (selector.selection.paths, selector.input_tokens, selector.output_tokens)
    return _llm_paths[key]


def model(variant: str) -> Method:
    return lambda repo, task, gold, budget: _select(variant, repo, task, budget)


def filled(method: Method) -> Method:
    """The model's choice first, then ranked files in whatever budget it left unused."""

    def run(repo: Repository, task: Task, gold: Gold, budget: int):
        chosen, tokens_in, tokens_out = method(repo, task, gold, budget)
        rest, _, _ = bm25_cut(repo, task, gold, budget)
        return list(dict.fromkeys([*chosen, *rest])), tokens_in, tokens_out

    return run


def merged(variant: str) -> Method:
    """Several independent selections, merged: files chosen by more runs come first."""

    def run(repo: Repository, task: Task, gold: Gold, budget: int):
        runs = [_select(variant, repo, task, budget, n) for n in range(RUNS)]
        votes: dict[str, list[int]] = {}
        for paths, _, _ in runs:
            for position, path in enumerate(paths):
                votes.setdefault(path, []).append(position)
        order = sorted(votes, key=lambda p: (-len(votes[p]), sum(votes[p]) / len(votes[p]), p))
        return order, sum(r[1] for r in runs), sum(r[2] for r in runs)

    return run


RUNNERS: dict[str, Method] = {
    "smallest": smallest,
    "bm25-plain": bm25_plain,
    "bm25": bm25,
    "bm25-cut": bm25_cut,
    "oracle": oracle,
}
for _name in VARIANTS:
    RUNNERS[_name] = model(_name)
    RUNNERS[f"{_name}-fill"] = filled(model(_name))
    RUNNERS[f"{_name}x3"] = merged(_name)
    RUNNERS[f"{_name}x3-fill"] = filled(merged(_name))


def repository(task: Task) -> tuple[Repository, set[str]]:
    """The repository at the task's commit, and every path in its archive."""
    owner, name = task.repo.split("/")
    path = CACHE / "archives" / f"{owner}__{name}" / f"{task.sha}.tar.gz"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        archive, _ = download_tarball(owner, name, task.sha)
        partial = path.with_suffix(".part")
        partial.write_bytes(archive)
        partial.rename(path)  # An interrupted download never leaves a file that looks whole.
    archive = path.read_bytes()
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        names = {m.name.partition("/")[2] for m in tar if m.isfile()}
    files = {}
    for file_path, data in tar_files(archive, lambda p, size: is_candidate(p, size, 1_000_000)):
        if (text := decode(data)) is not None:
            files[file_path] = RepoFile(file_path, text)
    where = ["-r", task.repo, "--ref", task.sha]
    return Repository(name, where, task.sha, dict(sorted(files.items()))), names


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
        try:
            ranked, tokens_in, tokens_out = RUNNERS[method](repo, task, gold, budget)
            error = None
        except Exception as exc:  # A failed call is recorded, reported, and run again next time.
            ranked, tokens_in, tokens_out, error = [], 0, 0, f"{type(exc).__name__}: {exc}"
        selected, _ = fit(ranked, repo, budget)
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
                "tokens": repo.cost(selected),
                **score(selected, gold.files, task.spans),
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
