"""Scoring what a method delivered against what a task needs.

A delivery maps each path shown to the lines shown from it, or to None for the whole file.
Gold is a list of spans, [file, first, last]; overlapping spans are merged before scoring,
so a line counts once however many spans name it.
"""

import re
from dataclasses import dataclass

TEST_PATH = re.compile(
    r"(^|/)(tests?|testing|spec|__tests__)/|(^|/)test_|_test\.|\.test\.|\.spec\."
)

Delivery = dict[str, set[int] | None]


@dataclass
class Gold:
    """The gold files of a task, checked against the repository at its commit."""

    files: list[str]  # present at the commit: what recall is measured against
    unresolved: list[str]  # named by the dataset but absent at the commit; dropped
    filtered: list[str]  # present, but left out by repo2txt's own filters; always a miss


def resolve(gold: list[str], in_archive: set[str], readable: set[str]) -> Gold:
    present = [p for p in gold if p in in_archive]
    return Gold(
        present,
        [p for p in gold if p not in in_archive],
        [p for p in present if p not in readable],
    )


def needed_lines(spans: list[list], files: list[str]) -> dict[str, set[int]]:
    """The needed lines of each gold file, with overlapping spans merged."""
    keep, out = set(files), {}
    for file, first, last in spans:
        if file in keep:
            out.setdefault(file, set()).update(range(first, last + 1))
    return out


def score(
    delivered: Delivery, gold: list[str], spans: list[list], sizes: dict[str, int]
) -> dict[str, float | None]:
    """File and line recall and precision of a delivery.

    - `recall`: share of gold files of which anything was shown.
    - `all_gold`: 1 if every gold file was shown, else 0.
    - `line_recall`: share of the needed lines shown, a line counted once.
    - `precision`: share of the files shown that are gold; None when nothing was shown,
      since an empty answer has no precision and averaging it as 0 would mislead.
    - `line_precision`: share of the lines shown that are needed lines; None when empty.
    - `recall_code`: `recall` with test files left out of the gold; None if gold is tests only.

    `sizes` gives each file's line count, so a whole file counts as all its lines.
    """
    wanted, shown = set(gold), set(delivered)
    hit = len(shown & wanted)
    needed = needed_lines(spans, gold)
    total_needed = sum(len(v) for v in needed.values())
    got = shown_lines = 0
    for path, lines in delivered.items():
        size = sizes.get(path, 0)
        shown_lines += size if lines is None else len(lines)
        if path in needed:
            got += len(needed[path]) if lines is None else len(needed[path] & lines)
    code = {p for p in wanted if not TEST_PATH.search(p)}
    return {
        "recall": hit / len(wanted) if wanted else None,
        "all_gold": float(wanted <= shown) if wanted else None,
        "precision": hit / len(shown) if shown else None,
        "line_recall": got / total_needed if total_needed else None,
        "line_precision": got / shown_lines if shown_lines else None,
        "recall_code": len(shown & code) / len(code) if code else None,
    }
