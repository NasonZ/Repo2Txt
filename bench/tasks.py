"""ContextBench tasks, reduced to what a selection method may see and what scores it.

The dataset is read from one fixed revision and checked against a recorded hash, so a task
list, its split and its gold cannot change under a result. `bench/manifest.json` lists the
task ids of each split; `python -m bench.tasks --check` confirms the data still matches it.
"""

import argparse
import hashlib
import io
import json
import os
import re
import tarfile
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

from repo2txt.models import RepoFile, Repository
from repo2txt.sources import decode, download_tarball, is_candidate, tar_files

DATASET = "Contextbench/ContextBench"
REVISION = "c2855792b006af41c67202d33883fb9d46362853"  # 23 January 2026
SUBSETS = {
    # name: (file in the dataset, sha256 of that file)
    "verified": (
        "data/contextbench_verified.parquet",
        "e9dcfd504cbfb849ac815a79040c793d0d92f94eecc9b5a4ee3e1445a2f8a791",
    ),
    "full": (
        "data/full.parquet",
        "2f56535bdc73eb8a68bf4ebb49789d8e9cd4f219ea60df6290b85278aee61ca8",
    ),
}
# SWE-Explore (arXiv 2606.07297): the lines that at least two successful agent runs read,
# refined and audited. It shares 155 issues with the verified subset. Its licence
# (CC BY-NC-ND 4.0) allows evaluating against it, so it is fetched, never copied in here.
EXPLORE = (
    "https://huggingface.co/datasets/SWE-Explore-Bench/SWE-Explore-Bench/resolve/"
    "bdb0ae45d7c337d9e1dc3ebfe2a0af6bc7c1fbd9/bench.final.public.jsonl",
    "dc4f114ececd0bfb987361c26ae5e2440456e2cccb36adfccb09ea5385aec202",
)
MANIFEST = Path(__file__).with_name("manifest.json")
# Some gold paths are absolute paths inside the container the task was annotated in.
CONTAINER = re.compile(r"^/(?:workspace/[^/]+|testbed|app)/")
CACHE = Path(os.getenv("REPO2TXT_BENCH_CACHE", Path.home() / ".cache" / "repo2txt-bench"))
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@")

Span = list  # [file, first line, last line], 1-based and inclusive


@dataclass
class Task:
    """One issue. A method is given `query` and the repository; the rest is for scoring.

    Three definitions of what the issue needs are carried. `spans` is ContextBench's: the
    code people traced out from the fix and checked was enough for a model to write one.
    `edits` is the lines the fix itself changed or inserted next to, in files that existed
    before it. `explored` is SWE-Explore's, where it covers the issue: the lines successful
    agents read. It is empty for most tasks.
    """

    id: str
    repo: str  # owner/name
    sha: str
    language: str
    source: str  # which benchmark the task came from
    query: str
    spans: list[Span]
    edits: list[Span] = field(default_factory=list)
    original_id: str = ""
    explored: list[Span] = field(default_factory=list)

    @property
    def gold(self) -> list[str]:
        return sorted({file for file, _, _ in self.spans})

    def golds(self) -> dict[str, list[Span]]:
        return {"annotated": self.spans, "edited": self.edits, "explored": self.explored}

    @property
    def split(self) -> str:
        """Which set the task belongs to. Whole repositories are held out, not single tasks."""
        digest = hashlib.sha256(f"split:{self.repo}".encode()).digest()
        return "test" if digest[0] % 10 < 3 else "dev"


def edited_spans(patch: str) -> list[Span]:
    """The lines a patch removes or replaces, by their numbers before it, per existing file.

    A hunk that only adds lines marks the line it was inserted after, since that is where a
    reader has to look. New and deleted files are left out: no method could select a file
    that does not exist yet, and a deleted file is not needed to write the fix.
    """
    spans: list[Span] = []
    for block in re.split(r"^diff --git ", patch, flags=re.M)[1:]:
        head, _, body = block.partition("\n@@")
        named = re.match(r"a/(\S+) b/", head)
        if not named or "new file mode" in head or "deleted file mode" in head:
            continue
        path, line, start, removed, lines = named.group(1), 0, 1, False, set()
        for text in ("@@" + body).split("\n"):
            hunk = _HUNK.match(text)
            if hunk:
                line, start = int(hunk.group(1)), int(hunk.group(1))
                removed = False
                continue
            if text.startswith("-"):
                lines.add(line)
                line, removed = line + 1, True
            elif text.startswith("+"):
                if not removed:
                    lines.add(max(line - 1, start))
            elif text.startswith(" "):
                line, removed = line + 1, False
        for first, last in _runs(lines):
            spans.append([path, first, last])
    return sorted(spans)


def _runs(lines: set[int]) -> list[tuple[int, int]]:
    runs: list[list[int]] = []
    for n in sorted(lines):
        if runs and n == runs[-1][1] + 1:
            runs[-1][1] = n
        else:
            runs.append([n, n])
    return [(a, b) for a, b in runs]


def _fetch(url: str, digest: str, path: Path) -> Path:
    """A file downloaded once and checked against its recorded sha256 on every use."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        response = httpx.get(url, follow_redirects=True, timeout=300)
        response.raise_for_status()
        partial = path.with_suffix(".part")
        partial.write_bytes(response.content)
        partial.rename(path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != digest:
        raise RuntimeError(f"{path} has sha256 {actual}, expected {digest}; delete it to refetch")
    return path


def _parquet(subset: str) -> Path:
    name, digest = SUBSETS[subset]
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{name}"
    return _fetch(url, digest, CACHE / "dataset" / REVISION / name)


def explored_spans() -> dict[str, list[Span]]:
    """SWE-Explore's core regions, by the original benchmark's instance id."""
    url, digest = EXPLORE
    path = _fetch(url, digest, CACHE / "swe-explore" / digest[:12] / "bench.jsonl")
    out = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        regions = row["ground_truth"]["read_core_regions"]
        out[row["instance_id"]] = sorted([r["path"], r["start"], r["end"]] for r in regions)
    return out


def _task(row: dict) -> Task:
    spans = json.loads(row["gold_context"] or "[]")
    return Task(
        row["instance_id"],
        # The `repo` field is not always the name GitHub knows.
        row["repo_url"].removesuffix(".git").split("github.com/")[-1],
        row["base_commit"],
        row["language"],
        row["source"],
        row["problem_statement"],
        sorted([CONTAINER.sub("", s["file"]), s["start_line"], s["end_line"]] for s in spans),
        edited_spans(row["patch"] or ""),
        row["original_inst_id"],
    )


_loaded: dict[str, list[Task]] = {}


def all_tasks(subset: str = "verified") -> list[Task]:
    """Every task of a subset, read from the pinned revision. The patch is not kept."""
    if subset not in _loaded:
        cache = CACHE / "dataset" / REVISION / f"tasks-{subset}.jsonl"
        if not cache.exists():
            import pyarrow.parquet as pq

            rows = pq.read_table(_parquet(subset)).to_pylist()
            cache.write_text("".join(json.dumps(asdict(_task(r))) + "\n" for r in rows))
        tasks = [Task(**json.loads(line)) for line in cache.read_text().splitlines()]
        explored = explored_spans()
        for task in tasks:
            task.explored = explored.get(task.original_id, [])
        _loaded[subset] = tasks
    return _loaded[subset]


def manifest(subset: str = "verified") -> dict:
    tasks = pool(subset)
    source = "full" if subset == FRESH else subset
    return {
        "dataset": DATASET,
        "revision": REVISION,
        "subset": subset,
        "file": SUBSETS[source][0],
        "sha256": SUBSETS[source][1],
        "splits": {
            split: sorted(t.id for t in tasks if t.split == split) for split in ("dev", "test")
        },
    }


FRESH = "fresh"  # the full release's tasks that are not in the verified subset


def pool(subset: str = "verified") -> list[Task]:
    """The tasks of a subset; `fresh` is the full release minus the verified subset."""
    if subset == FRESH:
        seen = {t.id for t in all_tasks("verified")}
        return [t for t in all_tasks("full") if t.id not in seen]
    return all_tasks(subset)


def sample(
    count: int, split: str = "dev", seed: str = "repo2txt", subset: str = "verified"
) -> list[Task]:
    """A fixed sample of one split that takes one task from each repository in turn."""

    def key(text: str) -> str:
        return hashlib.sha256(f"{seed}:{text}".encode()).hexdigest()

    by_repo: dict[str, list[Task]] = defaultdict(list)
    for task in pool(subset):
        if task.split == split:
            by_repo[task.repo].append(task)
    queues = [
        sorted(tasks, key=lambda t: key(t.id))
        for _, tasks in sorted(by_repo.items(), key=lambda item: key(item[0]))
    ]
    chosen: list[Task] = []
    while len(chosen) < count and any(queues):
        for queue in queues:
            if queue and len(chosen) < count:
                chosen.append(queue.pop(0))
    return chosen


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
    parser = argparse.ArgumentParser(description="Write or check the task manifest.")
    parser.add_argument("--write", action="store_true", help="Rewrite the manifest.")
    parser.add_argument("--subset", default="verified", choices=["verified", FRESH])
    args = parser.parse_args()
    current = manifest(args.subset)
    path = (
        MANIFEST
        if args.subset == "verified"
        else MANIFEST.with_name(f"manifest-{args.subset}.json")
    )
    if args.write:
        path.write_text(json.dumps(current, indent=1) + "\n")
        print(f"wrote {path}")
        return
    recorded = json.loads(path.read_text())
    if recorded != current:
        raise SystemExit(f"The tasks differ from {path}.")
    counts = {s: len(ids) for s, ids in current["splits"].items()}
    print(f"tasks match the manifest: {counts}")


if __name__ == "__main__":
    main()
