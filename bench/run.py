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
from pathlib import Path

from repo2txt.api import METHODS
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


def bm25(repo: Repository, task: Task, gold: Gold, budget: int) -> tuple[list[str], int, int]:
    return [path for path, _ in METHODS["bm25"](repo, task.query, budget, None)], 0, 0


def smallest(repo: Repository, task: Task, gold: Gold, budget: int) -> tuple[list[str], int, int]:
    """The floor: as many files as fit, ignoring the question."""
    repo.count()
    return sorted(repo.files, key=lambda p: (repo.files[p].tokens, p)), 0, 0


def oracle(repo: Repository, task: Task, gold: Gold, budget: int) -> tuple[list[str], int, int]:
    """The ceiling: the gold files themselves, smallest first, as far as the budget allows."""
    readable = [p for p in gold.files if p in repo.files]
    repo.count(readable)
    return sorted(readable, key=lambda p: repo.files[p].tokens), 0, 0


def llm(repo: Repository, task: Task, gold: Gold, budget: int) -> tuple[list[str], int, int]:
    selector = Selector(repo, budget, openai_complete(LLMSettings.from_env()))
    selector.ask(task.query)
    return selector.selection.paths, selector.input_tokens, selector.output_tokens


RUNNERS: dict[str, Method] = {"smallest": smallest, "bm25": bm25, "oracle": oracle, "llm": llm}


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tasks", type=int, default=20)
    parser.add_argument("--methods", default="smallest,bm25,oracle")
    parser.add_argument("--budgets", default="8k,32k,128k")
    parser.add_argument("--out", default="bench/results/run.jsonl")
    args = parser.parse_args()
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
    model = {"llm": os.getenv("LLM_MODEL")}

    for number, task in enumerate(sample(args.tasks), 1):
        todo = [(m, b) for m in methods for b in budgets if (task.id, m, b) not in done]
        if not todo:
            continue
        try:
            repo, names = repository(task)
        except Exception as exc:
            print(f"[{number}] {task.id}: could not read {task.repo}: {exc}", flush=True)
            continue
        gold = resolve(task.gold, names, set(repo.files))
        if not gold.files:
            print(f"[{number}] {task.id}: no gold file exists at the commit; skipped", flush=True)
            continue
        for method, budget in todo:
            started = time.time()
            try:
                ranked, tokens_in, tokens_out = RUNNERS[method](repo, task, gold, budget)
                error = None
            except Exception as exc:  # A failed call is recorded as an empty selection.
                ranked, tokens_in, tokens_out, error = [], 0, 0, f"{type(exc).__name__}: {exc}"
            selected, _ = fit(ranked, repo, budget)
            row = {
                "task": task.id,
                "repo": task.repo,
                "language": task.language,
                "source": task.source,
                "method": method,
                "budget": budget,
                "files": len(repo.files),
                "repo_tokens": repo.tokens(),
                "gold": len(gold.files),
                "gold_unresolved": len(gold.unresolved),
                "gold_filtered": len(gold.filtered),
                "selected": len(selected),
                "tokens": repo.tokens(selected),
                **score(selected, gold.files),
                "input_tokens": tokens_in,
                "output_tokens": tokens_out,
                "seconds": round(time.time() - started, 2),
                "error": error,
                "model": model.get(method),
                "commit": commit,
            }
            with out.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
        print(f"[{number}] {task.id} {task.repo} {len(repo.files)} files", flush=True)


if __name__ == "__main__":
    main()
