"""Gather the numbers, examples and transcripts the web piece shows, as one JSON file.

    uv run python -m bench.export --strips TASK_SUFFIX --out docs/web/data.json

Every figure on the page is drawn from this file, and this file only from results files and
saved transcripts, so the page changes only when the measurements do. The headline is the
pre-registered run on the fresh held-out set (bench/results/v3); the tuning history comes from
the earlier files, which are kept as recorded.
"""

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

from . import models as bench_models
from .charts import paired, rows
from .io import read_rows
from .metrics import resolve
from .stats import estimate, field
from .stats import paired as paired_rows
from .tasks import pool as task_pool
from .tasks import repository

V3 = Path("bench/results/v3")
V2 = Path("bench/results/v2")
# The fresh set's files, and the name each method takes on the page.
FRESH = {
    "fresh-reference.jsonl.xz": {
        "smallest": "smallest",
        "bm25-plain": "bm25-plain",
        "bm25": "bm25",
        "oracle:annotated": "oracle",
        "oracle-lines:annotated": "oracle-lines",
    },
    "fresh-llm-haiku.jsonl": {"llm": "llm"},
    "fresh-llm-haiku-128k.jsonl": {"llm": "llm"},
    "fresh-llm-sonnet.jsonl": {"llm": "llm-sonnet"},
    "fresh-llm-deepseek.jsonl": {"llm": "llm-deepseek"},
    "fresh-lines-haiku.jsonl": {"llm-lines": "llm-lines"},
}
SHOWN = 8  # chosen files listed per method in the task browser


def fresh_rows() -> list[dict]:
    """The pre-registered run's rows that completed, renamed for the page."""
    out = []
    for name, rename in FRESH.items():
        path = V3 / name
        if not path.exists():
            continue
        for row in read_rows(path):
            if row["status"] == "ok" and row["method"] in rename:
                out.append(row | {"method": rename[row["method"]]})
    return out


def _defined(rows: list[dict], name: str) -> float | None:
    values = [r["scores"]["annotated"][name] for r in rows]
    values = [v for v in values if v is not None]
    return round(mean(values), 3) if values else None


def curve(results: list[dict]) -> dict:
    out: dict = defaultdict(dict)
    cells: dict = defaultdict(list)
    for r in results:
        cells[(r["method"], r["budget"])].append(r)
    # A cell from a run still in progress would show an average over the tasks done so far.
    most = defaultdict(int)
    for (method, _), group in cells.items():
        most[method] = max(most[method], len(group))
    for (method, budget), group in cells.items():
        if len(group) < 0.95 * most[method]:
            print(f"left out {method} at {budget}: {len(group)} of {most[method]} tasks so far")
            continue
        e = estimate(group, field("recall"))
        out[method][budget] = {
            "tasks": e.tasks,
            "repos": e.repos,
            "recall": round(e.mean, 3),
            "lo": round(e.low, 3),
            "hi": round(e.high, 3),
            "precision": _defined(group, "precision"),
            "line": _defined(group, "line_recall"),
            "tokens": round(mean(r["tokens"] for r in group)),
            "files": round(mean(r["selected"] for r in group), 1),
        }
    return out


def _pair(results, a, b, budget, key="recall", gold="annotated") -> dict:
    ra = [r for r in results if r["method"] == a and r["budget"] == budget]
    rb = [r for r in results if r["method"] == b and r["budget"] == budget]
    full = max(len({r["task"] for r in results if r["method"] == m}) for m in (a, b))
    if min(len(ra), len(rb)) < 0.95 * full:  # a run still in progress
        raise ValueError(f"{a} or {b} incomplete at {budget}")
    d = paired_rows(ra, rb, field(key, gold))
    return {
        "d": round(d.mean, 3),
        "lo": round(d.low, 3),
        "hi": round(d.high, 3),
        "n": d.tasks,
        "better": d.better,
        "worse": d.worse,
    }


def pairs(results: list[dict]) -> dict:
    """Paired differences on the same fresh held-out tasks."""
    out = {}
    for budget in (8000, 32000, 128000):
        for name, a, b in (
            ("llm-bm25", "llm", "bm25"),
            ("bm25-plain", "bm25", "bm25-plain"),
            ("sonnet-bm25", "llm-sonnet", "bm25"),
            ("deepseek-bm25", "llm-deepseek", "bm25"),
            ("sonnet-llm", "llm-sonnet", "llm"),
            ("deepseek-llm", "llm-deepseek", "llm"),
        ):
            try:
                out[f"{name}@{budget}"] = _pair(results, a, b, budget)
            except ValueError:  # a method not run at this budget
                pass
    out["lines-llm@8000"] = _pair(results, "llm-lines", "llm", 8000, "line_recall")
    return out


def model_costs(results: list[dict]) -> dict:
    """What each model cost and how it behaved, per budget, from measured tokens."""
    spec = {"llm": "haiku", "llm-sonnet": "sonnet", "llm-deepseek": "deepseek"}
    out: dict = defaultdict(dict)
    for method, name in spec.items():
        for budget in (8000, 32000):
            group = [r for r in results if r["method"] == method and r["budget"] == budget]
            if not group:
                continue
            seconds = sorted(r["seconds"] for r in group)
            out[method][budget] = {
                "cost": round(bench_models.cost(bench_models.MODELS[name], group), 2),
                "calls": round(mean(r["model_calls"] for r in group), 2),
                "seconds": seconds[len(seconds) // 2],
                "cached": round(
                    sum(r["cached_tokens"] for r in group)
                    / max(1, sum(r["input_tokens"] for r in group)),
                    2,
                ),
                "output": round(mean(r["output_tokens"] for r in group)),
            }
    return out


def agreement() -> dict:
    """How far the definitions of needed agree: a perfect choice under one, scored by another."""
    rows = [
        r
        for r in read_rows(V3 / "fresh-reference.jsonl.xz")
        if r["status"] == "ok" and r["budget"] == 128000 and r["method"].startswith("oracle:")
    ]
    out: dict = {}
    for source in ("annotated", "edited", "explored"):
        group = [r for r in rows if r["method"] == f"oracle:{source}"]
        for gold in ("annotated", "edited", "explored"):
            values = [r["scores"][gold]["recall"] for r in group]
            values = [v for v in values if v is not None]
            if values:
                out[f"{source}>{gold}"] = {"recall": round(mean(values), 2), "n": len(values)}
    return out


def _recall_at(gold: list[str], ranked: list[str], k: int) -> float:
    return sum(g in ranked[:k] for g in gold) / len(gold)


def _reciprocal_rank(gold: list[str], ranked: list[str]) -> float:
    return next((1 / i for i, p in enumerate(ranked, 1) if p in gold), 0.0)


def arb() -> dict:
    """Agent Retrieval Bench, from the benchmark's own evaluator's report and the details."""
    report = json.loads((V3 / "arb-report" / "bcy_budget_curve.json").read_text())
    bcy = {run["label"]: run["overall"] for run in report["runs"]}
    out = {}
    for label, path in (
        ("llm", V3 / "arb-llm-haiku.jsonl.xz"),
        ("llm-sonnet", V3 / "arb-llm-sonnet.jsonl.xz"),
        ("llm-deepseek", V3 / "arb-llm-deepseek.jsonl.xz"),
        ("bm25", Path("bench/results/arb/repo2txt-bm25.jsonl.xz")),
        ("bm25-plain", Path("bench/results/arb/repo2txt-bm25-plain.jsonl.xz")),
    ):
        details = read_rows(path)
        key = path.name.removesuffix(".xz").removesuffix(".jsonl")
        recall20 = mean(
            sum(g in d["ranked_files"][:20] for g in d["gold_files"]) / len(d["gold_files"])
            for d in details
        )
        mrr = mean(
            next((1 / i for i, p in enumerate(d["ranked_files"], 1) if p in d["gold_files"]), 0.0)
            for d in details
        )
        entry = {"samples": len(details), "recall20": round(recall20, 3), "mrr": round(mrr, 3)}
        if key in bcy:
            entry["bcy8k"] = round(bcy[key]["BCY@8000"], 3)
        out[label] = entry
    # Published comparison points, from the benchmark's paper and its community results.
    out["published"] = {
        "RepoMap": 0.379,
        "Qwen3-Embedding-8B": 0.373,
        "Oko 0.9.1": 0.533,
        "BM25 (benchmark's)": 0.205,
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
        ("model", "Recall prompt with shortlist", "first selector", tuned, "llm-v2", "llm", True),
        ("model", "Adding the shortlist", "recall prompt alone", tuned, "llm-v2", "llm-v2n", True),
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

    # Calling models natively (15 November): same 100 tasks, 32k, rows from the rebuilt harness.
    def v2(name: str) -> list[dict]:
        return [
            r
            for r in read_rows(V2 / f"{name}.jsonl")
            if r["status"] == "ok" and r["budget"] == 32000
        ]

    native = [
        (
            "Request opening kept stable",
            "selection restated",
            "ab-compat-current",
            "dev-llm-saved",
            True,
        ),
        (
            "Selected paths in tool replies",
            "stable opening alone",
            "ab-compat-state",
            "ab-compat-current",
            True,
        ),
        (
            "Cross-provider prompt, Haiku",
            "current prompt",
            "ab-haiku-v3",
            "ab-haiku-current",
            False,
        ),
        (
            "Cross-provider prompt, DeepSeek",
            "current prompt",
            "ab-deepseek-v3",
            "ab-deepseek-current",
            False,
        ),
    ]
    for label, versus, a, b, kept in native:
        d = paired_rows(v2(a), v2(b), field("recall"))
        out.append(
            {
                "group": "native",
                "label": label,
                "versus": versus,
                "d": round(d.mean, 3),
                "lo": round(d.low, 3),
                "hi": round(d.high, 3),
                "tasks": d.tasks,
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
    known = {t.id: t for t in task_pool("fresh")}

    def one(task_id: str) -> dict | None:
        task, got = known[task_id], by_task[task_id]
        if not (task.query or "").strip():  # no issue text: nothing for a method to answer
            return None
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
            paths = list(r["delivered"])
            scores = r["scores"]["annotated"]
            item = {
                "n": len(paths),
                "tokens": r["tokens"],
                "recall": scores["recall"],
                "line": scores["line_recall"],
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
    parser.add_argument("--strips", default="", help="Task id suffixes to export in full.")
    parser.add_argument("--out", default="docs/web/data.json")
    args = parser.parse_args()
    results = fresh_rows()
    model_rows = [r for r in results if r["method"].startswith("llm")]
    strip_ids = {
        r["task"] for r in results for s in args.strips.split(",") if s and r["task"].endswith(s)
    }
    scored = [r for r in results if r["method"] == "llm"]
    data = {
        "split": "fresh held-out",
        "tasks": len({r["task"] for r in scored}),
        "repos": len({r["repo"] for r in scored}),
        "model": "claude-haiku-5-5",
        "cost_tokens": {
            "input": sum(r["input_tokens"] for r in model_rows),
            "output": sum(r["output_tokens"] for r in model_rows),
            "dollars": round(
                sum(m["cost"] for per in model_costs(results).values() for m in per.values()), 2
            ),
        },
        "curve": curve(results),
        "pairs": pairs(results),
        "models": model_costs(results),
        "agreement": agreement(),
        "arb": arb(),
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
