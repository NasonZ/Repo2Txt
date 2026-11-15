"""Choosing files with a model: a prompt, three tools, a short loop."""

import json
from collections.abc import Callable
from typing import Any

from . import focus, rank, tree
from .api import repo_map
from .llm import Call, Complete
from .models import Repository
from .render import map_text
from .selection import Change, Selection

# Rounds of tool calls the model gets per request, so it can correct a refused selection.
MAX_ROUNDS = 5
# Tokens of the system prompt given to the repository tree. Deeper directories are collapsed
# and reached with list_dir.
TREE_TOKENS = 6000
# How many lexically ranked files are shown beside each request, as leads.
SHORTLIST = 30
LIST_LIMIT = 200

SYSTEM = """\
<repository name="{name}">
{tree}
</repository>

You choose which files of the repository above someone should read to deal with a request. \
Whoever reads your selection works from those files alone. A file they need and were not \
given stops them; a file they did not need costs only its tokens. So lean towards including.

How to read the repository: indentation shows nesting, so a file's path is its directory \
names and its own name joined with `/`, as in `src/app/main.py`. A directory line reads \
`name/ tokens (files)`; a directory shown without its contents is collapsed, and list_dir \
opens it. Sizes are in tokens.

Start with the files where the behaviour in the request is implemented. Then add what a \
reader needs to follow them: the definitions they rely on, the code that calls them, and \
any configuration or schema they read. Add the closest tests last, if there is room. A \
selection that uses a small part of the budget has usually left something out, so aim to \
use most of it, and keep going after the first plausible files.

Budget: {budget:,} tokens. A change that would exceed it is refused and leaves the selection \
as it was. Selected so far: {selected}

You have {rounds} rounds of tool calls. Calls that do not depend on each other can go in \
the same round. Give paths exactly as they appear, most important first. When the selection \
covers the request, reply in one or two sentences saying what you chose and why.
"""

CANDIDATES = """\
<candidates>
The files a word search ranks highest for this request, best first, with sizes in tokens. \
It finds files that mention the request's terms and misses ones that do not, so treat these \
as leads to weigh with the tree, not as the answer.
{rows}
</candidates>

<request>
{request}
</request>"""

_PATHS = {"type": "array", "items": {"type": "string"}}
SCHEMAS = [
    {
        "name": "select_files",
        "description": "Replace the selection with these paths, most important first.",
        "parameters": {"type": "object", "properties": {"paths": _PATHS}, "required": ["paths"]},
    },
    {
        "name": "adjust_selection",
        "description": "Add paths to the selection and remove paths from it.",
        "parameters": {"type": "object", "properties": {"add": _PATHS, "remove": _PATHS}},
    },
    {
        "name": "list_dir",
        "description": "List the files and directories directly inside a directory.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
]
TOOLS = [{"type": "function", "function": schema} for schema in SCHEMAS]


class Selector:
    """A conversation with a model that ends in a `Selection` within budget."""

    def __init__(self, repo: Repository, budget: int, complete: Complete):
        self.repo = repo
        self.complete = complete
        self.selection = Selection(repo, budget)
        self.messages: list[dict[str, Any]] = []
        self.input_tokens = 0
        self.output_tokens = 0
        self._tree = map_text(repo_map(repo, max_tokens=TREE_TOKENS), TREE_TOKENS)
        self._handlers: dict[str, Callable[..., str]] = {
            "select_files": lambda paths: self._report(self.selection.replace(_paths(paths))),
            "adjust_selection": lambda add=(), remove=(): self._report(
                self.selection.adjust(_paths(add), _paths(remove))
            ),
            "list_dir": self._list_dir,
        }

    def ask(self, request: str) -> str:
        """Add a request to the conversation, let the model work, and return its reply."""
        leads = [path for path, _ in rank.bm25(self.repo, request)[:SHORTLIST]]
        if leads:
            self.repo.count(leads)
            rows = "\n".join(f"{p} {self.repo.files[p].tokens:,}" for p in leads)
            request = CANDIDATES.format(rows=rows, request=request)
        self.messages.append({"role": "user", "content": request})
        for round_number in range(MAX_ROUNDS + 1):
            # After the last round the model must answer in words.
            tools = TOOLS if round_number < MAX_ROUNDS else []
            reply = self.complete(self._conversation(), tools)
            self.input_tokens += reply.input_tokens
            self.output_tokens += reply.output_tokens
            self.messages.append(_assistant(reply.text, reply.calls))
            if not reply.calls:
                return reply.text
            for call in reply.calls:
                self.messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": self._run(call)}
                )
        return reply.text

    def _conversation(self) -> list[dict[str, Any]]:
        selection = self.selection
        selected = "nothing"
        if selection.paths:
            selected = f"{len(selection.paths)} files, {selection.total:,} tokens: " + ", ".join(
                selection.paths
            )
        system = SYSTEM.format(
            name=self.repo.name,
            budget=selection.budget,
            selected=selected,
            tree=self._tree,
            rounds=MAX_ROUNDS,
        )
        return [{"role": "system", "content": system}, *self.messages]

    def _run(self, call: Call) -> str:
        """Run one tool call. Whatever goes wrong, the model gets a reply it can act on."""
        handler = self._handlers.get(call.name)
        if handler is None:
            return f"There is no tool named {call.name}."
        if call.arguments is None:
            return f"The arguments for {call.name} were not valid JSON. Nothing was changed."
        try:
            result = handler(**call.arguments)
        except (TypeError, ValueError) as exc:
            return f"Bad arguments for {call.name}: {exc}. Nothing was changed."
        # Every reply says where the budget stands, so the model can see what room is left.
        selection = self.selection
        left = selection.budget - selection.total
        used = f"{len(selection.paths)} files, {selection.total:,} of {selection.budget:,} tokens"
        return f"{result}\n[Selected: {used}; {left:,} left]"

    def _report(self, change: Change) -> str:
        """Tell the model what its change did, in terms it can act on."""
        parts = []
        if not change.applied:
            parts.append(
                f"Refused: that would be {change.over_by:,} tokens over the budget. The "
                "selection is unchanged. Choose fewer or smaller files."
            )
        else:
            parts.append("Done.")
        if change.unknown:
            parts.append(
                "These paths are not in the repository or the selection and were ignored: "
                + ", ".join(change.unknown)
                + ". Check them against the tree, or use list_dir."
            )
        return " ".join(parts)

    def _list_dir(self, path: str = "") -> str:
        node = tree.find(tree.of(self.repo), path)
        if node is None or not node.is_dir:
            return f"There is no directory {path!r}. Use a directory path from the tree."
        items = tree.entries(node, 1)
        lines = [
            f"{e.path} {e.tokens:,}" + (f" ({e.files} files)" if e.files else "")
            for e in items[:LIST_LIMIT]
        ]
        if len(items) > LIST_LIMIT:
            lines.append(f"... {len(items) - LIST_LIMIT} more; list a subdirectory instead.")
        return "\n".join(lines)


def _paths(value: Any) -> list[str]:
    if isinstance(value, list | tuple) and all(isinstance(item, str) for item in value):
        return list(value)
    raise ValueError("paths must be a list of strings")


def _assistant(text: str, calls: list[Call]) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if calls:
        message["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
            }
            for c in calls
        ]
    return message


RANGES_PROMPT = """\
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
RANGES_TOOL = {
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


def choose_ranges(
    repo: Repository, paths: list[str], request: str, budget: int, complete: Complete
) -> tuple[list[tuple[str, int, int]], int, int]:
    """Ask the model which line ranges of `paths` to read within `budget`.

    Returns the ranges in the model's order, and the input and output tokens it used.
    """
    outlines = "\n\n".join(focus.outline(p, repo.files[p].text) for p in paths)
    prompt = RANGES_PROMPT.format(outlines=outlines, budget=budget, request=request)
    reply = complete([{"role": "user", "content": prompt}], [RANGES_TOOL])
    ranges = []
    for call in reply.calls:
        if call.name != "read_ranges" or not call.arguments:
            continue
        for item in call.arguments.get("ranges", []):
            path, _, span = str(item).strip().rpartition(":")
            first, _, last = span.partition("-")
            if path in repo.files and first.isdigit() and last.isdigit():
                lines = len(repo.files[path].text.split("\n"))
                first, last = max(1, int(first)), min(lines, int(last))
                if first <= last:
                    ranges.append((path, first, last))
    return ranges, reply.input_tokens, reply.output_tokens
