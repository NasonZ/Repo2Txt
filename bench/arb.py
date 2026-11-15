"""Rank Agent Retrieval Bench samples with Repo2Txt, in the layout its evaluator rescores.

    uv run python -m bench.arb --data /path/to/arb/data --method bm25 --out arb-bm25.jsonl

Agent Retrieval Bench (Qin and Xie, arXiv 2607.24882) asks for the files a coding agent needs
next, from a workflow signal: a failing trace, a review comment, a pull request's intent or
one edited file. Each details row holds the ranked files and the sample's gold files, in the
layout of the benchmark's own details files, so its unmodified evaluator, or the rescoring
script published with the Oko community result, computes Recall@k, MRR and Budgeted Context
Yield from them. Recall@k and MRR are also computed here, by the same definitions, as a check.

The repository at each sample's base commit is rebuilt from the `kind=file` rows of the
released corpus, then passed through Repo2Txt's own filters, so a gold file the tool would
never read counts as a miss, as it would in use.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from repo2txt import rank
from repo2txt.models import RepoFile, Repository
from repo2txt.sources import is_candidate

SUBSETS = ("v2_code2test", "v2_comment2context", "v2_trace2code", "v2_edit2ripple")
# sha256 of each samples.jsonl, as published with the Oko community result.
SAMPLES_SHA256 = {
    "v2_code2test": "712b2699d3f963c2f940688ae5248304263c4ecda1b3831b2801c95712a50b85",
    "v2_comment2context": "543267024f7f06127c50664a3a1b8825d3c213e4b12dbbfabf9a3346902ad796",
    "v2_trace2code": "9d0ff50155fa4f65c1bbb632abc4f1f11393c50fa7b0a9249e06128b368b6266",
    "v2_edit2ripple": "31d97ffe815dda8730149880c0159239a986eceb306e5a5df816dfc420717cef",
}


def _dedupe(values) -> list[str]:
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


def _paths(values) -> list[str]:
    return _dedupe(
        v if isinstance(v, str) else str(v["path"])
        for v in values
        if isinstance(v, str) or (isinstance(v, dict) and v.get("path"))
    )


def gold_files(sample: dict) -> list[str]:
    """The benchmark's `target_gold_files`, at commit 9e4d689, line for line."""
    gold = sample.get("gold") or {}
    if gold.get("no_gold") is True:
        return []
    explicit = _paths(gold.get("files") or [])
    if explicit:
        return explicit
    if sample.get("task_type") == "code2test":
        return _dedupe(gold.get("related_tests") or [])
    if sample.get("task_type") == "comment2context":
        context = _paths(gold.get("must_context_files") or gold.get("context_files") or [])
        if context:
            return context
    roots = _dedupe(gold.get("root_cause_files") or [])
    return roots or _dedupe(gold.get("related_tests") or [])


def query_text(sample: dict) -> str:
    """Every field of the workflow signal, by name, as one request."""
    query = sample.get("query") or {}
    parts = []
    for key in sorted(query):
        value = query[key]
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        if text.strip():
            parts.append(f"{key}:\n{text}")
    return "\n\n".join(parts)


def split(sample_id: str) -> str:
    """The halves used by the Oko community result: even first byte of the id's hash is dev."""
    return "dev" if hashlib.sha256(sample_id.encode()).digest()[0] % 2 == 0 else "heldout"


class Corpora:
    """Repositories rebuilt from released corpus files, one commit at a time."""

    def __init__(self, data: Path):
        self.paths = {}
        for chunks in sorted(data.glob("corpus/v2_*/*/*.chunks.jsonl")):
            repo = chunks.parent.name.replace("__", "/", 1)
            self.paths[(repo, chunks.name.removesuffix(".chunks.jsonl"))] = chunks
        self.cached: tuple | None = None

    def load(self, repo: str, commit: str) -> tuple[Repository, set[str]]:
        if self.cached and self.cached[0] == (repo, commit):
            return self.cached[1]
        files, every = {}, set()
        with self.paths[(repo, commit)].open() as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("kind") != "file":
                    continue
                path, text = row["path"], row.get("text") or ""
                every.add(path)
                if is_candidate(path, len(text.encode()), 1_000_000):
                    files[path] = RepoFile(path, text)
        loaded = (Repository(repo.split("/")[1], ["-r", repo], commit, files), every)
        self.cached = ((repo, commit), loaded)
        return loaded


# The model selector returns a selection within a budget, not a ranking. Its files come first,
# in its order, then the lexical ranking, so the measures that look past the selection still
# have a list to read. This budget is the benchmark's headline one.
SELECT_BUDGET = 8000


def ranking(method: str, repo: Repository, query: str) -> tuple[list[str], dict]:
    """Ranked paths, and the model usage spent on them."""
    if method == "bm25":
        return [p for p, _ in rank.bm25(repo, query)], {}
    if method == "bm25-plain":
        return [p for p, _ in rank.bm25(repo, query, boosts=False)], {}
    if method == "llm":
        from repo2txt.api import repo_select
        from repo2txt.models import Problem

        from .run import Usage

        usage = Usage()
        try:
            result = repo_select(
                repo, query, SELECT_BUDGET, method="llm", complete=usage.complete()
            )
            chosen = [f.path for f in result.files]
        except Problem:
            chosen = []
        rest = [p for p, _ in rank.bm25(repo, query) if p not in chosen]
        spent = {"input_tokens": usage.input, "output_tokens": usage.output, "calls": usage.calls}
        return chosen + rest, spent | {"selected": len(chosen)}
    raise ValueError(f"unknown method {method}")


def rank_metrics(gold: list[str], ranked: list[str]) -> dict[str, float]:
    out = {f"Recall@{k}": sum(p in ranked[:k] for p in gold) / len(gold) for k in (5, 10, 20)}
    out["MRR"] = next((1 / r for r, p in enumerate(ranked, 1) if p in gold), 0.0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="The benchmark's data/ folder.")
    parser.add_argument("--method", default="bm25", choices=["bm25", "bm25-plain", "llm"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="A fixed sample of this many, for a pilot.")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    samples = []
    for subset in SUBSETS:
        path = args.data / "benchmark" / subset / "samples.jsonl"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != SAMPLES_SHA256[subset]:
            raise SystemExit(f"{path} has sha256 {digest}, not the published one")
        samples += [json.loads(line) for line in path.read_text().splitlines()]
    samples = [s for s in samples if gold_files(s)]
    if args.limit:
        samples = sorted(samples, key=lambda s: hashlib.sha256(s["id"].encode()).hexdigest())
        samples = samples[: args.limit]
    done = set()
    if args.out.exists():  # resume: keep the samples already ranked
        done = {json.loads(line)["sample_id"] for line in args.out.read_text().splitlines()}
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for sample in samples:
        if sample["id"] not in done:
            groups[(sample["repo"], sample["base_commit"])].append(sample)

    def run_group(group: list[dict]) -> list[dict]:
        corpora = Corpora(args.data)  # one per group: each loads its commit once
        rows = []
        for sample in group:
            repo, every = corpora.load(sample["repo"], sample["base_commit"])
            rows.append(detail(sample, repo, every))
        return rows

    def detail(sample: dict, repo: Repository, every: set[str]) -> dict:
        gold = gold_files(sample)
        ranked, spent = ranking(args.method, repo, query_text(sample))
        return {
            "sample_id": sample["id"],
            "task_type": sample["task_type"],
            "repo": sample["repo"],
            "base_commit": sample["base_commit"],
            "split": split(sample["id"]),
            "ranker": f"repo2txt-{args.method}",
            "candidate_filter": "repo2txt",
            "gold_files": gold,
            "top_files": ranked[:20],
            "ranked_files": ranked,
            "metrics": rank_metrics(gold, ranked),
            "filtered_gold": sum(1 for g in gold if g in every and g not in repo.files),
            "usage": spent,
        }

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(args.workers) as pool, args.out.open("a") as handle:
        for rows in pool.map(run_group, groups.values()):
            handle.writelines(json.dumps(r) + "\n" for r in rows)
            handle.flush()
    details = [json.loads(line) for line in args.out.read_text().splitlines()]
    filtered = sum(d.get("filtered_gold", 0) for d in details)
    spent_in = sum(d.get("usage", {}).get("input_tokens", 0) for d in details)
    spent_out = sum(d.get("usage", {}).get("output_tokens", 0) for d in details)
    if spent_in:
        print(f"model usage: {spent_in:,} input and {spent_out:,} output tokens")
    by_task = defaultdict(list)
    for d in details:
        by_task[d["task_type"]].append(d["metrics"])
    print(f"{len(details)} samples; gold files left out by repo2txt's filters: {filtered}")
    for task, rows in sorted(by_task.items()) + [("all", [d["metrics"] for d in details])]:
        cells = " ".join(
            f"{k} {mean(r[k] for r in rows):.4f}" for k in ("Recall@5", "Recall@20", "MRR")
        )
        print(f"{task:16} n={len(rows):3} {cells}")


if __name__ == "__main__":
    main()
