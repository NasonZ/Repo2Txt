"""Run selection methods over tasks at several token budgets.

    uv run python -m bench.run --tasks 20 --methods smallest,bm25,oracle --budgets 8k,32k,128k

`bm25`, `llm` and `llm-lines` call `repo_select`, the function behind `repo2txt select`, so
what is measured is what the command delivers. The other methods are references. Every
delivery is costed by the library's own packing cost and scored against each definition of
what the task needs (see `Task.golds`).

Rows are appended to a JSONL file as they finish, and rows already there are skipped, so a
run can be stopped and resumed. Each row records the code, data and model it came from.
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

from repo2txt import rank
from repo2txt.api import HEAD_TOKENS, delivery_cost, line_items, repo_select
from repo2txt.llm import ConfigError, LLMSettings
from repo2txt.models import Problem, Repository
from repo2txt.selection import fit
from repo2txt.tokens import ENCODING, parse_budget

from . import models
from .metrics import Delivery, Gold, needed_lines, resolve, score
from .tasks import REVISION, Task, repository, sample

ROOT = Path(__file__).resolve().parent.parent


class Usage:
    """Counts the model tokens a method spends, across every call it makes."""

    def __init__(self) -> None:
        self.input = self.output = self.calls = self.cached = self.cache_written = 0

    def complete(self):
        inner = models.complete_for(models.selected()[1])

        def complete(messages, tools):
            reply = inner(messages, tools)
            self.input += reply.input_tokens
            self.output += reply.output_tokens
            self.cached += reply.cached_tokens
            self.cache_written += reply.cache_write_tokens
            self.calls += 1
            return reply

        return complete


# A method returns what it delivers within `room` tokens.
Method = Callable[[Repository, Task, dict[str, Gold], int, Usage], Delivery]


def _fitted(repo: Repository, ranked: list[str], room: int) -> Delivery:
    return dict.fromkeys(fit(ranked, repo, room)[0])


def _parse(items: list[str]) -> Delivery:
    """`repo_select` items, `path` or `path:first-last`, as a delivery."""
    delivered: Delivery = {}
    for item in items:
        path, _, span = item.rpartition(":")
        first, _, last = span.partition("-")
        if path and first.isdigit() and last.isdigit():
            lines = delivered.setdefault(path, set())
            if lines is not None:
                lines.update(range(int(first), int(last) + 1))
        else:
            delivered[item] = None
    return delivered


def _selected(repo, task, room, usage, **options) -> Delivery:
    try:
        result = repo_select(repo, task.query, room + HEAD_TOKENS, **options)
    except ConfigError:
        raise
    except Problem:  # nothing matched, or nothing fitted: an empty delivery
        return {}
    return _parse([f.path for f in result.files])


def bm25(repo, task, gold, room, usage):
    return _selected(repo, task, room, usage, method="bm25")


def llm(repo, task, gold, room, usage):
    return _selected(repo, task, room, usage, method="llm", complete=usage.complete())


def llm_lines(repo, task, gold, room, usage):
    return _selected(repo, task, room, usage, method="llm", complete=usage.complete(), lines=True)


def bm25_plain(repo, task, gold, room, usage):
    """BM25 alone, as a reference point for the signals added to it."""
    return _fitted(repo, [p for p, _ in rank.bm25(repo, task.query, boosts=False)], room)


def smallest(repo, task, gold, room, usage):
    """The floor: as many files as fit, ignoring the question."""
    repo.count()
    return _fitted(repo, sorted(repo.files, key=lambda p: (repo.files[p].tokens, p)), room)


def oracle_files(kind: str) -> Method:
    def run(repo, task, gold, room, usage):
        """The gold files themselves, smallest first: the ceiling for whole files."""
        readable = [p for p in gold[kind].files if p in repo.files]
        repo.count(readable)
        return _fitted(repo, sorted(readable, key=lambda p: (repo.files[p].tokens, p)), room)

    return run


def oracle_lines(kind: str) -> Method:
    def run(repo, task, gold, room, usage):
        """The needed lines themselves, cheapest run of lines first: the ceiling for ranges.

        Greedy, so close to the best possible rather than provably the best.
        """
        readable = [p for p in gold[kind].files if p in repo.files]
        runs = []
        for path, lines in needed_lines(task.golds()[kind], readable).items():
            size = len(repo.files[path].text.split("\n"))
            lines = {n for n in lines if 1 <= n <= size}
            for item, cost in line_items(repo, path, lines) if lines else []:
                first, last = map(int, item.rpartition(":")[2].split("-"))
                runs.append((cost, path, first, last))
        delivered: Delivery = {}
        used = 0
        for _, path, first, last in sorted(runs):
            new = (delivered.get(path) or set()) | set(range(first, last + 1))
            before = delivery_cost(repo, path, delivered[path]) if path in delivered else 0
            change = delivery_cost(repo, path, new) - before
            if used + change <= room:
                delivered[path], used = new, used + change
        return delivered

    return run


GOLDS = ("annotated", "edited", "explored")
RUNNERS: dict[str, Method] = {
    "smallest": smallest,
    "bm25-plain": bm25_plain,
    "bm25": bm25,
    "llm": llm,
    "llm-lines": llm_lines,
    **{f"oracle:{k}": oracle_files(k) for k in GOLDS},
    **{f"oracle-lines:{k}": oracle_lines(k) for k in GOLDS},
}
MODEL_METHODS = {"llm", "llm-lines"}


def _ranges(lines: set[int]) -> list[list[int]]:
    out: list[list[int]] = []
    for n in sorted(lines):
        if out and n == out[-1][1] + 1:
            out[-1][1] = n
        else:
            out.append([n, n])
    return out


def run_task(task: Task, todo: list[tuple[str, int]], meta: dict) -> list[dict]:
    base = {
        "task": task.id,
        "repo": task.repo,
        "language": task.language,
        "source": task.source,
        "split": task.split,
        "run": meta,
    }
    if not (task.query or "").strip():
        reason = "the dataset gives this task no issue text"
        return [
            base | {"method": m, "budget": b, "status": "skipped", "error": reason} for m, b in todo
        ]
    try:
        repo, names = repository(task)
    except Exception as exc:  # usually a network failure: an error, so a resumed run retries it
        reason = f"could not read {task.repo}: {type(exc).__name__}: {exc}"
        return [
            base | {"method": m, "budget": b, "status": "error", "error": reason} for m, b in todo
        ]
    readable = set(repo.files)
    gold = {
        kind: resolve(sorted({f for f, _, _ in spans}), names, readable)
        for kind, spans in task.golds().items()
    }
    if not gold["annotated"].files:  # the primary definition; the others may be empty
        reason = "no annotated gold file exists at the commit"
        return [
            base | {"method": m, "budget": b, "status": "skipped", "error": reason} for m, b in todo
        ]
    sizes = {p: len(f.text.split("\n")) for p, f in repo.files.items()}
    rows = []
    for method, budget in todo:
        kind = method.partition(":")[2]
        if kind and not gold[kind].files:  # an oracle for a definition this task lacks
            reason = f"no {kind} gold for this task"
            rows.append(
                base | {"method": method, "budget": budget, "status": "skipped", "error": reason}
            )
            continue
        started, usage, room = time.time(), Usage(), max(budget - HEAD_TOKENS, 0)
        try:
            delivered, error = RUNNERS[method](repo, task, gold, room, usage), None
        except Exception as exc:  # recorded, and run again next time
            delivered, error = {}, f"{type(exc).__name__}: {exc}"
        tokens = sum(delivery_cost(repo, p, lines) for p, lines in delivered.items())
        rows.append(
            base
            | {
                "method": method,
                "budget": budget,
                "room": room,
                "status": "error" if error else "ok",
                "error": error,
                "files": len(repo.files),
                "repo_tokens": repo.tokens(),
                "gold": {
                    kind: {
                        "files": len(g.files),
                        "unresolved": len(g.unresolved),
                        "filtered": len(g.filtered),
                    }
                    for kind, g in gold.items()
                },
                "selected": len(delivered),
                "tokens": tokens,
                "scores": {
                    kind: score(delivered, g.files, task.golds()[kind], sizes)
                    for kind, g in gold.items()
                },
                "delivered": {
                    p: None if lines is None else _ranges(lines) for p, lines in delivered.items()
                },
                "input_tokens": usage.input,
                "output_tokens": usage.output,
                "cached_tokens": usage.cached,
                "cache_write_tokens": usage.cache_written,
                "model_calls": usage.calls,
                "seconds": round(time.time() - started, 2),
            }
        )
    return rows


def load_prompt() -> str | None:
    """Apply an experimental prompt (BENCH_PROMPT, a file with `apply()`), returning its hash."""
    path = os.getenv("BENCH_PROMPT")
    if not path:
        return None
    import importlib.util

    spec = importlib.util.spec_from_file_location("bench_prompt", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.apply()
    return f"{Path(path).name}:{hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]}"


def provenance(methods: list[str]) -> dict:
    """What a row came from: the exact code, the data revision and the model settings."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, cwd=ROOT
        ).stdout.strip()

    digest = hashlib.sha256()
    for name in sorted(git("ls-files", "src", "bench", "pyproject.toml").splitlines()):
        path = ROOT / name
        if path.suffix in {".py", ".toml"} and path.exists():
            digest.update(name.encode() + b"\0" + path.read_bytes())
    meta = {
        "commit": git("describe", "--always", "--dirty"),
        "code": digest.hexdigest()[:12],  # covers uncommitted changes, unlike the commit
        "dataset": REVISION[:12],
        "encoding": ENCODING,
    }
    if MODEL_METHODS & set(methods):
        name, model = models.selected()
        meta |= {
            "model_spec": name,
            "provider": model.provider,
            "model": model.model,
            "effort": model.effort,
            "options": model.options,
        }
        if model.provider == "openai-compat":
            settings = LLMSettings.from_env()
            meta |= {
                "model": settings.model,
                "endpoint": urlparse(settings.base_url).netloc if settings.base_url else "openai",
                "temperature": settings.temperature,
            }
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tasks", type=int, default=20)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--unseal", action="store_true", help="Required to run the test split.")
    parser.add_argument(
        "--subset",
        default="verified",
        choices=["verified", "fresh"],
        help="fresh: the full release's tasks outside the verified subset.",
    )
    parser.add_argument("--methods", default="smallest,bm25,oracle:annotated")
    parser.add_argument("--budgets", default="8k,32k,128k")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="bench/results/run.jsonl")
    args = parser.parse_args()
    if args.split == "test" and not args.unseal:
        parser.error("The test split is held out. Tune on dev; pass --unseal for the final run.")
    methods = args.methods.split(",")
    unknown = [m for m in methods if m not in RUNNERS]
    if unknown:
        parser.error(f"unknown methods {unknown}; choose from {sorted(RUNNERS)}")
    budgets = [parse_budget(b) for b in args.budgets.split(",")]
    prompt = load_prompt()
    meta = provenance(methods) | {"prompt": prompt or "built-in"}
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
            if row["status"] != "error":
                rows.append(row)
        out.write_text("".join(json.dumps(row) + "\n" for row in rows))
    done = {(row["task"], row["method"], row["budget"]) for row in rows}
    jobs = []
    for task in sample(args.tasks, args.split, subset=args.subset):
        todo = [(m, b) for m in methods for b in budgets if (task.id, m, b) not in done]
        if todo:
            jobs.append((task, todo, meta))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for finished in pool.map(lambda job: run_task(*job), jobs):
            with out.open("a") as handle:
                handle.writelines(json.dumps(row) + "\n" for row in finished)
            if finished:
                print(f"{finished[0]['task']} {finished[0]['repo']}", flush=True)


if __name__ == "__main__":
    main()
