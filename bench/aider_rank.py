"""Rank ContextBench tasks with aider's repo map, as a baseline from outside this project.

Two steps, in two environments, so aider never shares one with Repo2Txt:

    uv run python bench/aider_rank.py export --split test --out /srv/dev/aider/test-inputs.jsonl
    /path/to/aider-venv/bin/python bench/aider_rank.py rank \
        --inputs /srv/dev/aider/test-inputs.jsonl --out /srv/dev/aider/test-repomap.jsonl
    uv run python -m bench.rescore /srv/dev/aider/test-repomap.jsonl --methods repomap \
        --out bench/results/v2/test-repomap.jsonl

`export` lists, for each task, the files Repo2Txt can read, so both rank the same files.
`rank` writes them out from the cached archive and ranks them as aider would with the issue
typed into a chat. The files the issue names (aider's `get_file_mentions`) come first, as
aider offers them for adding to the chat; the rest are ordered by the repo map with those
files in the chat, its PageRank over definitions and references personalised by the
identifiers and file names the issue mentions (`get_ident_mentions`,
`get_ident_filename_matches`, `RepoMap.get_ranked_tags`). An earlier version left the named
files out of the chat, which ranks a named file below its neighbours; that is not how aider
is used, and it scored 0.11 at 32k.
"""

import argparse
import io
import json
import re
import sys
import tarfile
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

BUDGETS = (8000, 32000, 128000)


def export(split: str, out: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from bench.tasks import CACHE, all_tasks, repository

    rows = []
    for task in all_tasks():
        if task.split != split:
            continue
        repo, _ = repository(task)
        owner, name = task.repo.split("/")
        archive = CACHE / "archives" / f"{owner}__{name}" / f"{task.sha}.tar.gz"
        rows.append(
            {
                "task": task.id,
                "repo": task.repo,
                "query": task.query,
                "archive": str(archive),
                "files": sorted(repo.files),
            }
        )
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{out}: {len(rows)} tasks")


class _Quiet:
    """The parts of aider's InputOutput that the repo map calls."""

    def tool_output(self, *args, **kwargs):
        pass

    tool_warning = tool_error = tool_output

    def read_text(self, path):
        try:
            return Path(path).read_text(errors="replace")
        except OSError:
            return None


def _file_mentions(text: str, files: list[str]) -> set[str]:
    """aider's Coder.get_file_mentions, with no files in the chat."""
    words = {w.rstrip(",.!;:?").strip("\"'`*_") for w in text.split()}
    mentioned = {f for f in files if f in words}
    by_name: dict[str, list[str]] = defaultdict(list)
    for f in files:
        base = Path(f).name
        if any(c in base for c in "/\\._-"):
            by_name[base].append(f)
    mentioned |= {paths[0] for base, paths in by_name.items() if len(paths) == 1 and base in words}
    return mentioned


def _ident_file_matches(idents: set[str], files: list[str]) -> set[str]:
    """aider's Coder.get_ident_filename_matches."""
    by_stem: dict[str, set[str]] = defaultdict(set)
    for f in files:
        stem = Path(f).stem.lower()
        if len(stem) >= 5:
            by_stem[stem].add(f)
    return {f for i in idents if len(i) >= 5 for f in by_stem[i.lower()]}


def rank_one(row: dict) -> dict:
    from aider.repomap import RepoMap

    files = row["files"]
    with tempfile.TemporaryDirectory(dir="/srv/dev") as root:
        wanted = set(files)
        data = Path(row["archive"]).read_bytes()
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            for member in tar:
                rel = member.name.partition("/")[2]
                if member.isfile() and rel in wanted:
                    target = Path(root) / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(tar.extractfile(member).read())
        repomap = RepoMap(root=root, io=_Quiet(), map_tokens=1024)
        idents = set(re.split(r"\W+", row["query"]))
        # aider offers the files a message names for adding to the chat; the map then ranks
        # the rest of the repository around the files in the chat.
        chat = sorted(_file_mentions(row["query"], files))
        mentioned = set(chat) | _ident_file_matches(idents, files)
        others = [str(Path(root) / f) for f in files if f not in chat]
        try:
            tags = repomap.get_ranked_tags(
                [str(Path(root) / f) for f in chat], others, mentioned, idents
            )
            error = None
        except Exception as exc:  # recorded as an error row, scored as a failure
            tags, error = [], f"{type(exc).__name__}: {exc}"
    ranked: list[str] = list(chat)
    for tag in tags:
        if tag and tag[0] not in ranked:
            ranked.append(tag[0])
    return {"task": row["task"], "repo": row["repo"], "paths": ranked, "error": error}


def rank(inputs: Path, out: Path, workers: int) -> None:
    rows = [json.loads(line) for line in inputs.read_text().splitlines()]
    with ProcessPoolExecutor(workers) as pool, out.open("w") as handle:
        for result in pool.map(rank_one, rows):
            for budget in BUDGETS:
                handle.write(json.dumps(result | {"method": "repomap", "budget": budget}) + "\n")
            print(result["task"], len(result["paths"]), result["error"] or "", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="step", required=True)
    e = sub.add_parser("export")
    e.add_argument("--split", default="test")
    e.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("rank")
    r.add_argument("--inputs", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.step == "export":
        export(args.split, args.out)
    else:
        rank(args.inputs, args.out, args.workers)


if __name__ == "__main__":
    main()
