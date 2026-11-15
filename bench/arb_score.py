"""Score saved Agent Retrieval Bench rankings with the benchmark's own, unmodified evaluator.

Run with a Python that has `agent_retrieval_bench` installed (the benchmark's repository at
commit 9e4d689), after `bench.arb` has written details files:

    python bench/arb_score.py --corpus-root /path/to/arb/data/corpus bench/results/arb/*.jsonl

Adapted from the rescoring script published with the Oko community result: Recall@k and MRR
from each row's ranking, Budgeted Context Yield from `report_bcy_budget_curve`.
"""

import argparse
import json
import tempfile
from pathlib import Path

from agent_retrieval_bench.bcy_curve import report_bcy_budget_curve


def rank_metrics(gold, ranked):
    out = {f"Recall@{k}": sum(p in ranked[:k] for p in gold) / len(gold) for k in (5, 10, 20)}
    out["MRR"] = next((1 / r for r, p in enumerate(ranked, 1) if p in gold), 0.0)
    return out


def corpus_manifest(root: Path, path: Path) -> None:
    rows = []
    for chunks in sorted(root.glob("v2_*/*/*.chunks.jsonl")):
        rows.append(
            {
                "repo": chunks.parent.name.replace("__", "/", 1),
                "base_commit": chunks.name.removesuffix(".chunks.jsonl"),
                "chunks_path": str(chunks.resolve()),
                "status": "ok",
            }
        )
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("bench/results/arb/report"))
    parser.add_argument("details", type=Path, nargs="+")
    args = parser.parse_args()
    print("| Ranking | Half | Samples | Recall@5 | Recall@20 | MRR |")
    print("| --- | --- | ---: | ---: | ---: | ---: |")
    for path in args.details:
        details = [json.loads(line) for line in path.read_text().splitlines()]
        for half in ("all", "dev", "heldout"):
            rows = [d for d in details if half == "all" or d["split"] == half]
            scored = [rank_metrics(d["gold_files"], d["ranked_files"]) for d in rows]
            cells = " | ".join(
                f"{sum(s[k] for s in scored) / len(scored):.4f}"
                for k in ("Recall@5", "Recall@20", "MRR")
            )
            print(f"| {path.stem} | {half} | {len(rows)} | {cells} |")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        manifest = Path(temp) / "corpus_manifest.jsonl"
        corpus_manifest(args.corpus_root, manifest)
        report = report_bcy_budget_curve(
            corpus_manifest_path=manifest,
            out_path=args.out_dir / "bcy_budget_curve.json",
            markdown_out_path=args.out_dir / "bcy_budget_curve.md",
            runs=[(p.stem, "repo2txt", p) for p in args.details],
        )
    print("\n| Ranking | BCY@4k | BCY@8k | BCY@16k | BCY@32k |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for run in report["runs"]:
        overall = run["overall"]
        cells = " | ".join(f"{overall[f'BCY@{b}']:.4f}" for b in (4000, 8000, 16000, 32000))
        print(f"| {run['label']} | {cells} |")


if __name__ == "__main__":
    main()
