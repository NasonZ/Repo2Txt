"""Gather the numbers, examples and transcripts the web piece shows, as one JSON file.

    uv run python -m bench.export --results bench/results/test.jsonl --out docs/web/data.json

Every figure on the page is drawn from this file, and this file only from results and saved
transcripts, so the page changes only when the measurements do.
"""

import argparse
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

from .charts import paired, rows
from .metrics import resolve
from .tasks import all_tasks, repository

METHODS = ["smallest", "bm25-plain", "bm25", "llm", "llm-lines", "oracle", "oracle-lines"]
SHOWN = 8  # chosen files listed per method in the task browser


def interval(values_by_repo: dict[str, list[float]]) -> tuple[float, float]:
    pools, rng = list(values_by_repo.values()), random.Random(0)
    means = sorted(
        mean(v for pool in rng.choices(pools, k=len(pools)) for v in pool) for _ in range(2000)
    )
    return means[50], means[1949]


def curve(results: list[dict]) -> dict:
    out: dict = defaultdict(dict)
    cells: dict = defaultdict(list)
    for r in results:
        if not r["error"]:
            cells[(r["method"], r["budget"])].append(r)
    for (method, budget), group in cells.items():
        by_repo: dict[str, list[float]] = defaultdict(list)
        for r in group:
            by_repo[r["repo"]].append(r["recall"])
        lo, hi = interval(by_repo)
        out[method][budget] = {
            "tasks": len(group),
            "recall": round(mean(r["recall"] for r in group), 3),
            "lo": round(lo, 3),
            "hi": round(hi, 3),
            "precision": round(mean(r["precision"] for r in group), 3),
            "line": round(mean(r["line_recall"] for r in group), 3),
            "tokens": round(mean(r["tokens"] for r in group)),
            "files": round(mean(r["selected"] for r in group), 1),
        }
    return out


def changes() -> list[dict]:
    """Each change tried on the tuning split, and what it did to recall at 32k."""
    free = rows("dev-free")
    tuned = rows("dev-llm", "dev-tune", "dev-tune2", "dev-tune-x3")
    spec = [
        ("lexical", "Three ranking signals", "plain BM25", free, "bm25", "bm25-plain", True),
        ("lexical", "Score cut-off", "the three signals", free, "bm25-cut", "bm25", False),
        ("model", "Shortlist, first prompt", "first selector", tuned, "llm-v1", "llm", False),
        ("model", "Shortlist and recall prompt", "first selector", tuned, "llm-v2", "llm", True),
        ("model", "Shortlist, recall prompt", "the prompt alone", tuned, "llm-v2", "llm-v2n", True),
        (
            "model",
            "Search and outline tools, as built",
            "kept selector",
            tuned,
            "llm-v3",
            "llm-v2",
            False,
        ),
        (
            "model",
            "Search and outline tools, fixed",
            "kept selector",
            tuned,
            "llm-v3b",
            "llm-v2",
            False,
        ),
        ("model", "Three runs merged", "kept selector", tuned, "llm-v2x3", "llm-v2", False),
        ("model", "Top-up from the ranker", "kept selector", tuned, "llm-v2-fill", "llm-v2", False),
    ]
    out = []
    for group, label, versus, data, a, b, kept in spec:
        d, lo, hi, n = paired(data, a, b, 32000)
        out.append(
            {
                "group": group,
                "label": label,
                "versus": versus,
                "d": round(d, 3),
                "lo": round(lo, 3),
                "hi": round(hi, 3),
                "tasks": n,
                "kept": kept,
            }
        )
    return out


def title(query: str) -> str:
    first = next((line.strip() for line in query.splitlines() if line.strip()), "")
    return first.lstrip("#").strip()[:140]


def tasks_view(results: list[dict], strips: set[str]) -> list[dict]:
    """One entry per task: what was needed and what each method delivered."""
    by_task: dict[str, dict] = defaultdict(dict)
    for r in results:
        by_task[r["task"]][(r["method"], r["budget"])] = r
    known = {t.id: t for t in all_tasks()}

    def one(task_id: str) -> dict | None:
        task, got = known[task_id], by_task[task_id]
        try:
            repo, names = repository(task)
        except Exception:
            return None
        gold = [p for p in resolve(task.gold, names, set(repo.files)).files if p in repo.files]
        entry = {
            "id": task_id,
            "repo": task.repo,
            "title": title(task.query),
            "language": task.language,
            "files": len(repo.files),
            "tokens": repo.tokens(),
            "needed": [{"path": p, "tokens": repo.files[p].cost} for p in gold],
            "picks": {},
        }
        for (method, budget), r in got.items():
            if method not in ("bm25", "llm", "llm-lines"):
                continue
            paths = r["paths"]
            item = {
                "n": len(paths),
                "tokens": r["tokens"],
                "recall": r["recall"],
                "line": r["line_recall"],
                "hits": [p for p in gold if p in paths],
            }
            if task_id in strips:
                item["all"] = [[p, repo.files[p].cost if p in repo.files else 0] for p in paths]
            else:
                item["first"] = paths[:SHOWN]
            entry["picks"][f"{method}@{budget}"] = item
        return entry

    with ThreadPoolExecutor(3) as pool:
        return [e for e in pool.map(one, sorted(by_task)) if e]


def transcript(path: str) -> dict | None:
    if not Path(path).exists():
        return None
    record = json.loads(Path(path).read_text())
    steps = []
    for message in record["messages"][1:]:
        if message["role"] == "user":
            continue
        if message["role"] == "assistant":
            calls = [
                {"tool": c["function"]["name"], "args": json.loads(c["function"]["arguments"])}
                for c in message.get("tool_calls", [])
            ]
            steps.append({"say": message.get("content") or "", "calls": calls, "results": []})
        elif steps:
            steps[-1]["results"].append(message["content"][:1200])
    return {key: record[key] for key in ("task", "repo", "budget", "gold", "selected")} | {
        "steps": steps
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="bench/results/test.jsonl")
    parser.add_argument("--strips", default="", help="Task id suffixes to export in full.")
    parser.add_argument("--out", default="docs/web/data.json")
    args = parser.parse_args()
    results = [json.loads(line) for line in Path(args.results).open()]
    strip_ids = {
        r["task"] for r in results for s in args.strips.split(",") if s and r["task"].endswith(s)
    }
    data = {
        "split": results[0].get("split", "dev"),
        "tasks": len({r["task"] for r in results}),
        "repos": len({r["repo"] for r in results}),
        "model": next((r["model"] for r in results if r.get("model")), None),
        "cost_tokens": {
            "input": sum(r["input_tokens"] for r in results),
            "output": sum(r["output_tokens"] for r in results),
        },
        "curve": curve(results),
        "changes": changes(),
        "examples": tasks_view(results, strip_ids),
        "transcripts": {
            "failing": transcript("bench/transcripts/tools-as-built-jackson.json"),
            "kept": transcript("bench/transcripts/kept-langchain.json"),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(data, separators=(",", ":")))
    print(f"{args.out}: {Path(args.out).stat().st_size // 1024} KB, {data['tasks']} tasks")


if __name__ == "__main__":
    main()
