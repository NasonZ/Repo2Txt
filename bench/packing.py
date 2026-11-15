"""Packing a ranked list of files into a budget, whole or focused, and scoring it by lines.

    uv run python -m bench.packing --split dev --tasks 300

A packing turns ranked paths into what a reader receives: for each file, the lines shown.
Line recall is the share of the needed lines that were shown.
"""

import argparse
import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from repo2txt import focus, rank
from repo2txt.models import FRAME_TOKENS, Repository
from repo2txt.tokens import count_tokens

from .metrics import resolve
from .tasks import Task, repository, sample

Shown = dict[str, set[int] | None]  # path -> lines shown, or None for the whole file


def _view(repo: Repository, path: str, query: str) -> tuple[set[int], int]:
    key = ("view", query, path)
    if key not in repo.cache:
        v = focus.view(repo.files[path].text, focus.weights(query, rank.idf(repo, query)))
        cost = count_tokens(v.text) + count_tokens(path) + FRAME_TOKENS if v.ranges else 0
        lines = {n for a, b in v.ranges for n in range(a, b + 1)}
        repo.cache[key] = (lines, cost)
    return repo.cache[key]


def whole(repo: Repository, ranked: list[str], query: str, budget: int) -> tuple[Shown, int]:
    shown, used = {}, 0
    for path in ranked:
        cost = repo.files[path].cost
        if used + cost <= budget:
            shown[path], used = None, used + cost
    return shown, used


def focused(repo: Repository, ranked: list[str], query: str, budget: int) -> tuple[Shown, int]:
    shown, used = {}, 0
    for path in ranked:
        lines, cost = _view(repo, path, query)
        if lines and used + cost <= budget:
            shown[path], used = lines, used + cost
    return shown, used


def hybrid(repo: Repository, ranked: list[str], query: str, budget: int) -> tuple[Shown, int]:
    """Each file whole if it fits, otherwise focused if that fits."""
    shown, used = {}, 0
    for path in ranked:
        if used + repo.files[path].cost <= budget:
            shown[path], used = None, used + repo.files[path].cost
            continue
        lines, cost = _view(repo, path, query)
        if lines and used + cost <= budget:
            shown[path], used = lines, used + cost
    return shown, used


PACKINGS: dict[str, Callable] = {"whole": whole, "focused": focused, "hybrid": hybrid}


def score_lines(shown: Shown, spans: list[list], gold: list[str]) -> dict[str, float]:
    needed = [(f, a, b) for f, a, b in spans if f in gold]
    total = sum(b - a + 1 for _, a, b in needed) or 1
    got = 0
    for f, a, b in needed:
        if f in shown:
            lines = shown[f]
            got += (b - a + 1) if lines is None else sum(1 for n in range(a, b + 1) if n in lines)
    return {
        "line_recall": got / total,
        "file_recall": sum(1 for g in gold if g in shown) / len(gold),
    }


def oracle_lines(repo: Repository, spans: list[list], gold: list[str], budget: int):
    """The needed lines themselves, smallest span first: the ceiling for any packing."""
    shown: Shown = {}
    used = 0
    # A needed file that repo2txt's filters leave out cannot be shown, so it stays a miss.
    readable = [(f, a, b) for f, a, b in spans if f in gold and f in repo.files]
    for f, a, b in sorted(readable, key=lambda s: s[2] - s[1]):
        text = "\n".join(repo.files[f].text.split("\n")[a - 1 : b])
        cost = count_tokens(text) + (0 if f in shown else count_tokens(f) + FRAME_TOKENS)
        if used + cost <= budget:
            shown.setdefault(f, set()).update(range(a, b + 1))
            used += cost
    return shown, used


def stored_selections(*files: str) -> dict[tuple[str, int], list[str]]:
    """The kept model selector's choices, from earlier runs."""
    out = {}
    for name in files:
        for line in Path(f"bench/results/{name}.jsonl").open():
            row = json.loads(line)
            if row["method"] == "llm-v2":
                out[(row["task"], row["budget"])] = row["paths"]
    return out


def evaluate(task: Task, budgets: list[int], selections: dict) -> list[dict]:
    try:
        repo, names = repository(task)
    except Exception:
        return []
    gold = [p for p in resolve(task.gold, names, set(repo.files)).files if p in repo.files]
    if not gold:
        return []
    ranked = {"bm25": [p for p, _ in rank.bm25(repo, task.query)]}
    rows = []
    for budget in budgets:
        lists = dict(ranked)
        if (task.id, budget) in selections:
            lists["llm"] = selections[(task.id, budget)]
        if (task.id, 32000) in selections:
            lists["llm@32k-list"] = selections[(task.id, 32000)]
        for source, paths in lists.items():
            paths = [p for p in paths if p in repo.files]
            for name, pack in PACKINGS.items():
                shown, used = pack(repo, paths, task.query, budget)
                rows.append(
                    {
                        "task": task.id,
                        "repo": task.repo,
                        "budget": budget,
                        "method": f"{source}/{name}",
                        "tokens": used,
                        **score_lines(shown, task.spans, gold),
                    }
                )
        shown, used = oracle_lines(repo, task.spans, gold, budget)
        rows.append(
            {
                "task": task.id,
                "repo": task.repo,
                "budget": budget,
                "method": "oracle/lines",
                "tokens": used,
                **score_lines(shown, task.spans, gold),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev")
    parser.add_argument("--tasks", type=int, default=300)
    parser.add_argument("--out", default="bench/results/dev-packing.jsonl")
    args = parser.parse_args()
    selections = stored_selections("dev-tune", "dev-v2-budgets")
    tasks = sample(args.tasks, args.split)
    with ThreadPoolExecutor(3) as pool:
        results = pool.map(lambda t: evaluate(t, [8000, 32000], selections), tasks)
        with open(args.out, "w") as out:
            for rows in results:
                out.writelines(json.dumps(r) + "\n" for r in rows)


if __name__ == "__main__":
    main()
