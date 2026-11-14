"""ContextBench tasks, reduced to what a selection method may see and what scores it."""

import hashlib
import json
import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

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
    gold: list[str]  # files a person judged necessary to resolve the issue


def all_tasks() -> list[Task]:
    """Every task of the 500-task subset, downloaded once."""
    path = CACHE / "tasks.jsonl"
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
                response = client.get(ROWS, params=params)
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
                            sorted({CONTAINER.sub("", span["file"]) for span in spans}),
                        )
                    )
                offset += len(rows)
        path.write_text("".join(json.dumps(asdict(t)) + "\n" for t in tasks))
    return [Task(**json.loads(line)) for line in path.read_text().splitlines()]


def sample(count: int, seed: str = "repo2txt") -> list[Task]:
    """A fixed sample that takes one task from each repository in turn."""

    def key(text: str) -> str:
        return hashlib.sha256(f"{seed}:{text}".encode()).hexdigest()

    by_repo: dict[str, list[Task]] = defaultdict(list)
    for task in all_tasks():
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
