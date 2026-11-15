"""Save complete selector conversations for chosen tasks, for reading and for the web piece.

    uv run python -m bench.transcripts TASK_SUFFIX BUDGET OUT.json

TASK_SUFFIX is the end of a task id, as printed by the runner.
"""

import json
import sys

from repo2txt.llm import LLMSettings, openai_complete
from repo2txt.selector import Selector

from .metrics import resolve
from .tasks import all_tasks, repository


def main() -> None:
    suffix, budget, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    task = next(t for t in all_tasks() if t.id.endswith(suffix))
    repo, names = repository(task)
    gold = [p for p in resolve(task.gold, names, set(repo.files)).files if p in repo.files]
    selector = Selector(repo, budget, openai_complete(LLMSettings.from_env()))
    selector.ask(task.query)
    record = {
        "task": task.id,
        "repo": task.repo,
        "budget": budget,
        "gold": gold,
        "selected": selector.selection.paths,
        "messages": selector._conversation(),
        "input_tokens": selector.input_tokens,
        "output_tokens": selector.output_tokens,
    }
    with open(out, "w") as handle:
        json.dump(record, handle, indent=1)
    found = len(set(gold) & set(selector.selection.paths))
    print(
        f"{task.repo}: {found} of {len(gold)} needed files, {len(selector.selection.paths)} chosen"
    )


if __name__ == "__main__":
    main()
