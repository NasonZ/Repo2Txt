"""Turning results into text or JSON that fits a token cap."""

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import Any
from xml.sax.saxutils import quoteattr

from . import tree
from .api import MapResult, PackResult, SelectResult
from .tokens import count_tokens

XML_LINES = ' lines="{}-{}"'


def capped(
    items: Sequence[Any], text_of: Callable[[Sequence[Any], bool], str], max_tokens: int
) -> str:
    """Render as many leading items as fit in `max_tokens`.

    `text_of(items, hints)` renders a subset and states what it left out. Hints are dropped
    before items are.
    """
    for hints in (True, False):
        if count_tokens(text := text_of(items, hints)) <= max_tokens:
            return text
        low, high = 0, len(items) - 1
        while low < high:
            middle = (low + high + 1) // 2
            if count_tokens(text_of(items[:middle], hints)) <= max_tokens:
                low = middle
            else:
                high = middle - 1
        if low > 0:
            return text_of(items[:low], hints)
    return text_of(items[:0], False)


def _notes(shown: int, total: int, what: str, hints: list[str]) -> list[str]:
    notes = [f"# showing {shown} of {total} {what}"] if shown < total else []
    return notes + [f"# next: {hint}" for hint in hints]


def map_text(result: MapResult, max_tokens: int) -> str:
    where = f"{result.repo}/{result.prefix}".rstrip("/")
    ref = f" ref={result.ref}" if result.ref else ""
    head = f"# {where}{ref}: {result.files} files, {result.tokens:,} tokens ({result.encoding})"
    depth = []
    if result.depth < result.max_depth:
        depth = [f"# depth {result.depth} of {result.max_depth}: deeper directories are collapsed"]

    def text(entries: Sequence[tree.Entry], hints: bool) -> str:
        more = result.next if hints else []
        notes = depth + _notes(len(entries), len(result.entries), "entries", more)
        return "\n".join([head, tree.render(list(entries)), *notes])

    return capped(result.entries, text, max_tokens)


def select_text(result: SelectResult, max_tokens: int) -> str:
    head = (
        f"# {result.repo}: {len(result.files)} files, {result.tokens:,} of {result.budget:,} "
        f"tokens, from {result.candidates} candidates ({result.method})"
    )
    skipped = [f"# did not fit: {s.path} {s.tokens:,}" for s in result.skipped]

    def text(files: Sequence[Any], hints: bool) -> str:
        rows = [f"{f.path}\t{f.tokens}" for f in files]
        more = (skipped, result.next) if hints else ([], [])
        notes = more[0] + _notes(len(files), len(result.files), "files", more[1])
        return "\n".join([head, *rows, *notes])

    return capped(result.files, text, max_tokens)


def as_json(result: MapResult | SelectResult, max_tokens: int) -> str:
    """JSON for a map or a selection, with its main list cut to fit and the cut stated."""
    key = "entries" if isinstance(result, MapResult) else "files"
    data = _clean(asdict(result))
    rows = data[key]
    if key == "entries":
        for row in rows:
            del row["depth"]

    def text(subset: Sequence[dict[str, Any]], hints: bool) -> str:
        shown = {**data, key: list(subset)}
        if not hints:
            shown.pop("next", None)
            shown.pop("skipped", None)
        if len(subset) < len(rows):
            shown["showing"] = f"{len(subset)} of {len(rows)} {key}"
        return json.dumps(shown, separators=(",", ":"))

    return capped(rows, text, max_tokens)


def pack_json(result: PackResult) -> str:
    return json.dumps(_clean(asdict(result)), separators=(",", ":"))


def pack_text(result: PackResult, fmt: str = "markdown") -> str:
    """The packed files as one document. Content is never altered or cut."""
    if fmt == "xml":
        head = (
            f"<repository name={quoteattr(result.repo)} ref={quoteattr(result.ref or '')} "
            f'files="{len(result.files)}" tokens="{result.tokens}">'
        )
        notes = [
            f'<omitted path={quoteattr(o.path)} tokens="{o.tokens}" reason="{o.reason}"/>'
            for o in result.omitted
        ] + [f"<next>{hint}</next>" for hint in result.next]
        blocks = [
            f"<file path={quoteattr(f.path)}{_lines(f.lines, XML_LINES)}>\n{f.content}\n</file>"
            for f in result.files
        ]
        return "\n".join([head, *notes, *blocks, "</repository>"]) + "\n"

    ref = f" ({result.ref})" if result.ref else ""
    head = f"# {result.repo}{ref}: {len(result.files)} files, {result.tokens:,} tokens"
    notes = [f"- omitted: {o.path}, {o.tokens:,} tokens, {o.reason}" for o in result.omitted]
    notes += [f"- next: `{hint}`" for hint in result.next]
    blocks = []
    for f in result.files:
        fence = "`" * max(3, 1 + max((len(run) for run in re.findall(r"`+", f.content)), default=0))
        blocks.append(
            f"### {f.path}{_lines(f.lines, ' (lines {}-{})')}\n\n{fence}\n{f.content}\n{fence}"
        )
    return "\n\n".join(["\n".join([head, *notes]), *blocks]) + "\n"


def pack_summary(result: PackResult) -> str:
    """What was packed, without the content: for when the document goes to a file."""
    lines = [f"# {result.repo}: {len(result.files)} files, {result.tokens:,} tokens"]
    lines += [f"{f.path}{_lines(f.lines, ':{}-{}')}\t{f.tokens}" for f in result.files]
    lines += [f"# omitted: {o.path} {o.tokens:,} {o.reason}" for o in result.omitted]
    return "\n".join(lines + [f"# next: {hint}" for hint in result.next])


def _lines(lines: tuple[int, int] | None, template: str) -> str:
    return template.format(*lines) if lines else ""


def _clean(value: Any) -> Any:
    """Drop empty optional fields so JSON carries only what is there."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items() if v is not None and v != []}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value
