"""Choosing files with a model: a prompt, a few tools, a short loop."""

import json
import re
from collections.abc import Callable
from typing import Any

from . import rank, tree
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
LIST_LIMIT = 200
SEARCH_FILES = 15
OUTLINE_LINES = 60
_DECLARATION = re.compile(
    r"^\s*(?:(?:export|default|public|private|protected|static|async|abstract|final|pub)\s+)*"
    r"(?:def|class|function|fn|func|struct|interface|trait|impl|type|enum|module|namespace)\b"
)

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

The repository follows. Indentation shows nesting, so a file's path is its directory names
and its own name joined with `/`, as in `src/app/main.py`. A directory line reads
`name/ tokens (files)`; directories shown without their contents are collapsed.

{tree}
"""

SYSTEM_RECALL = """\
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
{explore}
Budget: {budget:,} tokens. A change that would exceed it is refused and leaves the selection \
as it was. Selected so far: {selected}

You have {rounds} rounds of tool calls. Calls that do not depend on each other can go in \
the same round. Give paths exactly as they appear, most important first. When the selection \
covers the request, reply in one or two sentences saying what you chose and why.
"""

EXPLORE = """
File names mislead, so check before you choose. Use search to find where a function, class, \
setting or error message from the request actually lives, and outline to see what a file \
defines before spending budget on it.
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
EXPLORE_SCHEMAS = [
    {
        "name": "search",
        "description": (
            "Search file contents with a regular expression. Returns the files with the most "
            "matching lines, each with its first matches and their line numbers."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "A regular expression."},
                "path": {"type": "string", "description": "Only search under this directory."},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "outline",
        "description": "Show what a file defines: its classes and functions with line numbers.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
]


def _tools(schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"type": "function", "function": schema} for schema in schemas]


class Selector:
    """A conversation with a model that ends in a `Selection` within budget."""

    def __init__(
        self,
        repo: Repository,
        budget: int,
        complete: Complete,
        shortlist: int = 0,
        recall: bool = False,
        explore: bool = False,
    ):
        """`shortlist` puts that many lexically ranked files beside each request. `recall`
        uses the prompt that favours completeness and reports budget use after every call.
        `explore` adds tools to search contents and outline a file.
        """
        self.repo = repo
        self.complete = complete
        self.shortlist, self.recall, self.explore = shortlist, recall, explore
        self.tools = _tools(SCHEMAS + (EXPLORE_SCHEMAS if explore else []))
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
            "search": self._search,
            "outline": self._outline,
        }

    def ask(self, request: str) -> str:
        """Add a request to the conversation, let the model work, and return its reply."""
        if self.shortlist:
            self.repo.count(
                leads := [p for p, _ in rank.bm25(self.repo, request)[: self.shortlist]]
            )
            rows = "\n".join(f"{p} {self.repo.files[p].tokens:,}" for p in leads)
            request = CANDIDATES.format(rows=rows, request=request) if leads else request
        self.messages.append({"role": "user", "content": request})
        for round_number in range(MAX_ROUNDS + 1):
            # After the last round the model must answer in words.
            tools = self.tools if round_number < MAX_ROUNDS else []
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
        system = (SYSTEM_RECALL if self.recall else SYSTEM).format(
            name=self.repo.name,
            budget=selection.budget,
            selected=selected,
            tree=self._tree,
            explore=EXPLORE if self.explore else "",
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
        if not self.recall:
            return result
        selection = self.selection
        left = selection.budget - selection.total
        used = f"{len(selection.paths)} files, {selection.total:,} of {selection.budget:,} tokens"
        return f"{result}\n[Selected: {used}; {left:,} left]"

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

    def _search(self, pattern: str, path: str = "") -> str:
        try:
            regex = re.compile(pattern)
        except re.error:
            regex = re.compile(re.escape(pattern))
        prefix = path.strip("/") + "/" if path.strip("/") else ""
        found = []
        for file_path, file in self.repo.files.items():
            if not file_path.startswith(prefix):
                continue
            lines = [
                (n, line) for n, line in enumerate(file.text.split("\n"), 1) if regex.search(line)
            ]
            if lines:
                found.append((len(lines), file_path, lines[:2]))
        if not found:
            return f"No file under {prefix or 'the repository'} matches {pattern!r}."
        found.sort(key=lambda item: (-item[0], item[1]))
        out = []
        for count, file_path, lines in found[:SEARCH_FILES]:
            tokens = self.repo.files[file_path].tokens
            out.append(f"{file_path} ({count} matching lines, {tokens:,} tokens)")
            out += [f"  {n}: {line.strip()[:160]}" for n, line in lines]
        if len(found) > SEARCH_FILES:
            out.append(
                f"... and {len(found) - SEARCH_FILES} more files. Narrow the pattern or path."
            )
        return "\n".join(out)

    def _outline(self, path: str) -> str:
        file = self.repo.files.get(path.strip().removeprefix("./"))
        if file is None:
            return f"There is no file {path!r}. Use a path from the tree, a search or list_dir."
        lines = file.text.split("\n")
        declared = [(n, line) for n, line in enumerate(lines, 1) if _DECLARATION.match(line)]
        shown = declared[:OUTLINE_LINES] or list(enumerate(lines[:15], 1))
        out = [f"{file.path}: {len(lines)} lines, {file.tokens:,} tokens"]
        out += [f"  {n}: {line.rstrip()[:160]}" for n, line in shown]
        if len(declared) > OUTLINE_LINES:
            out.append(f"... and {len(declared) - OUTLINE_LINES} more definitions.")
        return "\n".join(out)


def _paths(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
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
