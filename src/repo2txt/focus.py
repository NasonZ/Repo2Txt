"""Showing the parts of a file that matter to a question, and eliding the rest.

A file is cut into blocks by indentation: a line followed by more deeply indented lines opens
a block, which runs until the indentation returns. Blocks that mention the question's rarer
words are kept whole, with the outline of the file around them: the lines that open its
blocks, down to a shallow depth. Everything else is replaced by a marker that says which
lines were left out, so a reader can ask for them with a line range.
"""

import re
from dataclasses import dataclass

from .rank import STOP, terms

# A matching block longer than this is not kept whole; the lines around each match are.
MAX_BLOCK = 80
CONTEXT = 3
# Lines that open blocks are kept as the file's outline down to this depth of nesting.
OUTLINE_DEPTH = 2
# A line is relevant when its question words weigh at least this much (inverse document
# frequency, so a word found in most files counts for little).
MIN_WEIGHT = 2.0
_CLOSER = re.compile(r"^\s*([}\])]|end\b|fi\b|done\b)")


@dataclass
class View:
    """The lines of a file to show, as 1-based inclusive ranges, and the text to show them."""

    ranges: list[tuple[int, int]]
    text: str
    lines: int

    @property
    def whole(self) -> bool:
        return self.ranges == [(1, self.lines)]


def _indent(line: str) -> int | None:
    stripped = line.lstrip(" \t")
    if not stripped:
        return None
    prefix = line[: len(line) - len(stripped)]
    return prefix.count(" ") + 4 * prefix.count("\t")


def _blocks(lines: list[str]) -> dict[int, int]:
    """Map each line that opens a block (0-based) to the last line of the block."""
    indents = [_indent(line) for line in lines]
    following = [None] * len(lines)  # the next non-blank line after each line
    upcoming = None
    for i in range(len(lines) - 1, -1, -1):
        following[i] = upcoming
        if indents[i] is not None:
            upcoming = i
    blocks: dict[int, int] = {}
    open_stack: list[int] = []
    for i, level in enumerate(indents):
        if level is None:
            continue
        while open_stack and level <= indents[open_stack[-1]]:
            start = open_stack.pop()
            closes = level == indents[start] and _CLOSER.match(lines[i])
            blocks[start] = i if closes else _last_content(indents, i - 1)
        nxt = following[i]
        if nxt is not None and indents[nxt] > level:
            open_stack.append(i)
    for start in open_stack:
        blocks[start] = _last_content(indents, len(lines) - 1)
    return blocks


def _last_content(indents: list[int | None], i: int) -> int:
    while i > 0 and indents[i] is None:
        i -= 1
    return i


def view(text: str, weights: dict[str, float]) -> View:
    """The parts of `text` relevant to a question whose words carry `weights`."""
    lines = text.split("\n")
    total = len(lines)
    blocks = _blocks(lines)

    # For each line, the blocks that contain it, outermost first.
    starts = sorted(blocks)
    chains: list[list[int]] = []
    stack: list[int] = []
    pending = 0
    for i in range(total):
        while stack and blocks[stack[-1]] < i:
            stack.pop()
        while pending < len(starts) and starts[pending] == i:
            stack.append(i)
            pending += 1
        chains.append(list(stack))

    keep: set[int] = set()
    for i, line in enumerate(lines):
        found = {t for t in terms(line) if t in weights}
        if sum(weights[t] for t in found) < MIN_WEIGHT:
            continue
        small = [s for s in chains[i] if blocks[s] - s < MAX_BLOCK]
        if small:
            start = small[-1]
            keep.update(range(start, blocks[start] + 1))
        else:
            keep.update(range(max(0, i - CONTEXT), min(total, i + CONTEXT + 1)))

    # The outline: the lines that open shallow blocks, so the kept parts can be placed.
    for start in starts:
        if len(chains[start]) <= OUTLINE_DEPTH:
            keep.add(start)

    if not keep:
        return View([], "", total)
    ranges: list[tuple[int, int]] = []
    for i in sorted(keep):
        if ranges and i == ranges[-1][1]:
            ranges[-1] = (ranges[-1][0], i + 1)
        else:
            ranges.append((i + 1, i + 1))
    shown, previous = [], 0
    for first, last in [*ranges, (total + 1, total)]:
        gap = lines[previous : first - 1]
        if any(line.strip() for line in gap):  # a gap of blank lines needs no marker
            shown.append(f"… lines {previous + 1}-{first - 1} not shown")
        shown.extend(lines[first - 1 : last])
        previous = last
    return View(ranges, "\n".join(shown), total)


def weights(query: str, idf: dict[str, float]) -> dict[str, float]:
    """The question's words with their weights, leaving out the commonest."""
    return {t: idf[t] for t in set(terms(query)) - STOP if t in idf}


MAX_HEADERS = 120


def outline(path: str, text: str) -> str:
    """Every block of a file with its line range and size: what a model reads to pick ranges."""
    from .tokens import count_tokens

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
