"""ContextBench tasks, reduced to what a selection method may see and what scores it."""

import hashlib
import io
import json
import os
import re
import tarfile
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

from repo2txt.models import RepoFile, Repository
from repo2txt.sources import decode, download_tarball, is_candidate, tar_files

DATASET, CONFIG = "Contextbench/ContextBench", "contextbench_verified"
ROWS = "https://datasets-server.huggingface.co/rows"
# Some gold paths are absolute paths inside the container the task was annotated in.
CONTAINER = re.compile(r"^/(?:workspace/[^/]+|testbed|app)/")
CACHE = Path(os.getenv("REPO2TXT_BENCH_CACHE", Path.home() / ".cache" / "repo2txt-bench"))


@dataclass
class Task:
    """One issue. The fix and its tests are never copied out of the dataset."""

    id: str
    repo: str  # owner/name
    sha: str
    language: str
    source: str  # which benchmark the task came from
    query: str  # the issue text, and the only thing a method is given
    spans: list[list]  # [file, first line, last line] a person judged necessary

    @property
    def gold(self) -> list[str]:
        return sorted({file for file, _, _ in self.spans})

    @property
    def split(self) -> str:
        """Which set the task belongs to. Whole repositories are held out, not single tasks."""
        digest = hashlib.sha256(f"split:{self.repo}".encode()).digest()
        return "test" if digest[0] % 10 < 3 else "dev"


def all_tasks() -> list[Task]:
    """Every task of the 500-task subset, downloaded once."""
    path = CACHE / "tasks-v2.jsonl"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        tasks, offset = [], 0
        with httpx.Client(timeout=120) as client:
            while True:
                params = {
                    "dataset": DATASET,
                    "config": CONFIG,
                    "split": "train",
                    "offset": offset,
                    "length": 100,
                }
                for attempt in range(5):  # The dataset server fails now and then.
                    response = client.get(ROWS, params=params)
                    if response.status_code < 500:
                        break
                    time.sleep(2 * (attempt + 1))
                response.raise_for_status()
                rows = response.json()["rows"]
                if not rows:
                    break
                for item in rows:
                    row = item["row"]
                    spans = json.loads(row["gold_context"] or "[]")
                    tasks.append(
                        Task(
                            row["instance_id"],
                            # The `repo` field is not always the name GitHub knows.
                            row["repo_url"].removesuffix(".git").split("github.com/")[-1],
                            row["base_commit"],
                            row["language"],
                            row["source"],
                            row["problem_statement"],
                            sorted(
                                [CONTAINER.sub("", s["file"]), s["start_line"], s["end_line"]]
                                for s in spans
                            ),
                        )
                    )
                offset += len(rows)
        path.write_text("".join(json.dumps(asdict(t)) + "\n" for t in tasks))
    return [Task(**json.loads(line)) for line in path.read_text().splitlines()]


def sample(count: int, split: str = "dev", seed: str = "repo2txt") -> list[Task]:
    """A fixed sample of one split that takes one task from each repository in turn."""

    def key(text: str) -> str:
        return hashlib.sha256(f"{seed}:{text}".encode()).hexdigest()

    by_repo: dict[str, list[Task]] = defaultdict(list)
    for task in all_tasks():
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
