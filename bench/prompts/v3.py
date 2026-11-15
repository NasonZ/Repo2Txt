"""Prompt v3 for the model selector: the cross-provider middle ground, as an experiment.

Loaded with BENCH_PROMPT=bench/prompts/v3.py; it replaces the selector's prompt, tool
descriptions and change reports for that run only. What changes from the current prompt, and
why (sources in the 15 November research notes):

- The reason for including stays ("a file they need and were not given stops them; one they
  did not need costs only its tokens"); the unconditional push to use most of the budget goes,
  and a stop condition comes in. Anthropic's and OpenAI's guides agree on stating a goal, a
  reason and a stop rule, and on avoiding unconditional rules, which are followed literally.
- Each tool says when to use it and when not to (Anthropic's tool-description guidance).
- A path that is not in the repository is answered with the nearest paths that are, so the
  correction is in the reply the model reads next (Anthropic's error-result guidance; the
  user's own lesson that guidance works best at the moment of decision).
"""

import difflib

from repo2txt import selector

SYSTEM = """\
<repository name="{name}">
{tree}
</repository>

You choose which files of the repository above someone should read to deal with a request. \
Whoever reads your selection works from those files alone: a file they need and were not \
given stops them, while a file they did not need costs only its tokens.

How to read the repository: indentation shows nesting, so a file's path is its directory \
names and its own name joined with `/`, as in `src/app/main.py`. A directory line reads \
`name/ tokens (files)`; a directory shown without its contents is collapsed. Sizes are in \
tokens.

Include the files where the behaviour in the request is implemented, and what a reader needs \
to follow and change them: the definitions they rely on, the code that calls them, any \
configuration or schema they read, and the closest tests. When you are unsure whether a file \
is needed and it fits, include it. Stop when the selection covers what the request touches \
and what remains would only be nearby.

Budget: {budget:,} tokens. A change that would exceed it is refused and leaves the selection \
as it was. Each tool reply ends with what is selected and how much of the budget is left.

You have {rounds} rounds of tool calls. Calls that do not depend on each other can go in \
the same round. Give paths exactly as they appear, most important first. When you stop, \
reply in one or two sentences saying what you chose and why.
"""

_PATHS = {"type": "array", "items": {"type": "string"}}
SCHEMAS = [
    {
        "name": "select_files",
        "description": "Replace the whole selection with these paths, most important first. "
        "Use it for the first selection or to start over; to add or drop a few files, use "
        "adjust_selection.",
        "parameters": {"type": "object", "properties": {"paths": _PATHS}, "required": ["paths"]},
    },
    {
        "name": "adjust_selection",
        "description": "Add paths to the selection and remove paths from it, keeping the rest "
        "in order. Use it to grow or trim a selection you already have.",
        "parameters": {"type": "object", "properties": {"add": _PATHS, "remove": _PATHS}},
    },
    {
        "name": "list_dir",
        "description": "List the files and directories directly inside a directory. Use it "
        "for a directory the tree shows collapsed; files the tree already shows need no "
        "listing.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
]


def _report(self, change):
    parts = []
    if not change.applied:
        parts.append(
            f"Refused: that would be {change.over_by:,} tokens over the budget. The "
            "selection is unchanged. Choose fewer or smaller files."
        )
    else:
        parts.append("Done.")
    for path in change.unknown:
        near = difflib.get_close_matches(path, list(self.repo.files), n=3, cutoff=0.6)
        hint = f" Did you mean: {', '.join(near)}?" if near else " Check the tree, or use list_dir."
        parts.append(f"{path} is not in the repository or the selection and was ignored.{hint}")
    return " ".join(parts)


def apply() -> None:
    selector.SYSTEM = SYSTEM
    selector.SCHEMAS = SCHEMAS
    selector.TOOLS = [{"type": "function", "function": s} for s in SCHEMAS]
    selector.Selector._report = _report
