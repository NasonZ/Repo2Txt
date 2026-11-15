"""Experiment: let the model choose line ranges from outlines of the files it selected.

    uv run python -m bench.ranges --budget 8000

Stage one is the kept selector's choice at 32k, read from earlier results. Stage two shows the
model the outline of each chosen file (the lines that open blocks, with their line ranges and
sizes) and asks for the ranges to read within the smaller budget. Scored by line recall.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from repo2txt.focus import _blocks
from repo2txt.llm import LLMSettings, openai_complete
from repo2txt.models import FRAME_TOKENS
from repo2txt.tokens import count_tokens

from .metrics import resolve
from .packing import score_lines, stored_selections
from .tasks import repository, sample

MAX_HEADERS = 120
PROMPT = """\
<outlines>
{outlines}
</outlines>

Above are outlines of the files chosen as relevant to the request below. Each line gives a \
block's line range, its size in tokens, and the line that opens it. Choose the line ranges \
someone needs to read to deal with the request, within {budget:,} tokens in total. Whoever \
reads them works from those lines alone: a part they need and were not given stops them, and \
an extra part costs only its tokens, so lean towards including. Include the definitions the \
relevant code relies on and the code that calls it, not only the place of the change. A range \
can cover a whole small file. Give the most important ranges first.

<request>
{request}
</request>"""
TOOL = {
    "type": "function",
    "function": {
        "name": "read_ranges",
        "description": "The line ranges to read, most important first, as path:first-last.",
        "parameters": {
            "type": "object",
            "properties": {"ranges": {"type": "array", "items": {"type": "string"}}},
            "required": ["ranges"],
        },
    },
}


def outline(path: str, text: str) -> str:
    lines = text.split("\n")
    blocks = _blocks(lines)
    rows = [f"{path} ({len(lines)} lines, {count_tokens(text):,} tokens)"]
    for start in sorted(blocks)[:MAX_HEADERS]:
        end = blocks[start]
        size = count_tokens("\n".join(lines[start : end + 1]))
        rows.append(f"  {start + 1}-{end + 1} ({size:,}) {lines[start].strip()[:120]}")
    if len(blocks) > MAX_HEADERS:
        rows.append(f"  ... {len(blocks) - MAX_HEADERS} more blocks")
    return "\n".join(rows)


def run(task, budget, selections, complete):
    chosen = selections.get((task.id, 32000))
    if not chosen:
        return None
    repo, names = repository(task)
    gold = [p for p in resolve(task.gold, names, set(repo.files)).files if p in repo.files]
    chosen = [p for p in chosen if p in repo.files]
    text = "\n\n".join(outline(p, repo.files[p].text) for p in chosen)
    messages = [
        {"role": "user", "content": PROMPT.format(outlines=text, budget=budget, request=task.query)}
    ]
    reply = complete(messages, [TOOL])
    asked = []
    for call in reply.calls:
        if call.name == "read_ranges" and call.arguments:
            asked += [r for r in call.arguments.get("ranges", []) if isinstance(r, str)]
    shown, used = {}, 0
    for item in asked:
        path, _, span = item.strip().rpartition(":")
        if path not in repo.files or "-" not in span:
            continue
        try:
            first, last = (int(x) for x in span.split("-", 1))
        except ValueError:
            continue
        body = "\n".join(repo.files[path].text.split("\n")[first - 1 : last])
        cost = count_tokens(body) + (0 if path in shown else count_tokens(path) + FRAME_TOKENS)
        if first < 1 or last < first or used + cost > budget:
            continue
        shown.setdefault(path, set()).update(range(first, last + 1))
        used += cost
    return {
        "task": task.id,
        "repo": task.repo,
        "budget": budget,
        "method": "llm/ranges",
        "tokens": used,
        "asked": asked,
        "input_tokens": reply.input_tokens,
        "output_tokens": reply.output_tokens,
        **score_lines(shown, task.spans, gold),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=8000)
    parser.add_argument("--out", default="bench/results/dev-ranges.jsonl")
    args = parser.parse_args()
    selections = stored_selections("dev-tune")
    complete = openai_complete(LLMSettings.from_env())
    with ThreadPoolExecutor(4) as pool:
        rows = pool.map(lambda t: run(t, args.budget, selections, complete), sample(100, "dev"))
        with open(args.out, "a") as out:
            out.writelines(json.dumps(r) + "\n" for r in rows if r)


if __name__ == "__main__":
    main()
