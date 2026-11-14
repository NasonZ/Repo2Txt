"""Choosing files with a model: one prompt, three tools, a short loop."""

import json
from collections.abc import Callable
from typing import Any

from . import tree
from .api import repo_map
from .llm import Call, Complete
from .models import Repository
from .selection import Change, Selection

# Rounds of tool calls the model gets per request, so it can correct a refused selection.
MAX_ROUNDS = 5
# Tokens of the system prompt given to the repository tree. Deeper directories are collapsed
# and reached with list_dir.
TREE_TOKENS = 6000
LIST_LIMIT = 200

SYSTEM = """\
You choose which files of the repository `{name}` someone should read to deal with a request.
You cannot read the files; you see their paths and sizes in tokens.

Budget: {budget:,} tokens. A selection that would exceed it is refused and left unchanged.
Selected so far: {selected}

Prefer the files where the behaviour in question is implemented, then what they depend on to \
be understood. Leave out tests, fixtures and documentation unless the request is about them. \
Give paths exactly as listed, most important first.

Tools:
- select_files(paths): replace the whole selection.
- adjust_selection(add, remove): change part of it.
- list_dir(path): list a directory that is shown collapsed below.

When the selection is right, reply in one or two sentences saying what you chose and why.

Repository ({files} files, {tokens:,} tokens; a directory line reads `name/ tokens (files)`):
{tree}
"""

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
        self._map = repo_map(repo, max_tokens=TREE_TOKENS)
        self._tree = tree.render(self._map.entries)
        self._handlers: dict[str, Callable[..., str]] = {
            "select_files": lambda paths=(): self._report(self.selection.replace(paths)),
            "adjust_selection": lambda add=(), remove=(): self._report(
                self.selection.adjust(add, remove)
            ),
            "list_dir": self._list_dir,
        }

    def ask(self, request: str) -> str:
        """Add a request to the conversation, let the model work, and return its reply."""
        self.messages.append({"role": "user", "content": request})
        for round_number in range(MAX_ROUNDS + 1):
            # After the last round the model must answer in words.
            reply = self.complete(self._conversation(), TOOLS if round_number < MAX_ROUNDS else [])
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
            files=self._map.files,
            tokens=self._map.tokens,
            tree=self._tree,
        )
        return [{"role": "system", "content": system}, *self.messages]

    def _run(self, call: Call) -> str:
        handler = self._handlers.get(call.name)
        if handler is None:
            return f"There is no tool named {call.name}."
        try:
            return handler(**call.arguments)
        except TypeError as exc:
            return f"Bad arguments for {call.name}: {exc}"

    def _report(self, change: Change) -> str:
        """Tell the model what its change did, in terms it can act on."""
        selection = self.selection
        state = f"{len(selection.paths)} files, {selection.total:,} of {selection.budget:,} tokens"
        parts = []
        if not change.applied:
            parts.append(
                f"Refused: that would be {change.over_by:,} tokens over the budget. The "
                f"selection is unchanged ({state}). Choose fewer or smaller files."
            )
        else:
            parts.append(f"Done. Selection: {state}.")
        if change.unknown:
            parts.append(
                "These paths are not in the repository or the selection and were ignored: "
                + ", ".join(change.unknown)
                + ". Check them against the tree, or use list_dir."
            )
        return " ".join(parts)

    def _list_dir(self, path: str = "") -> str:
        root = tree.build_tree({p: f.tokens for p, f in self.repo.files.items()})
        node = tree.find(root, path)
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
