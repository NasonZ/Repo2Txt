"""Scoring a selection against the files a task needs."""

import re
from dataclasses import dataclass

TEST_PATH = re.compile(
    r"(^|/)(tests?|testing|spec|__tests__)/|(^|/)test_|_test\.|\.test\.|\.spec\."
)


def _lines(spans: list[list], files: set[str]) -> int:
    return sum(last - first + 1 for file, first, last in spans if file in files)


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


def score(selected: list[str], gold: list[str], spans: list[list]) -> dict[str, float]:
    """File-level recall, precision and F1 of `selected` against `gold`.

    `line_recall` is the share of the needed lines that were delivered: a file counts for as
    many lines as were marked in it, so missing the file that holds most of the answer costs
    more than missing one that holds a line.

    `recall_code` leaves test files out of the gold, since a method told to skip tests should
    not be marked down for it. It is None when the gold is tests only.
    """
    chosen, wanted = set(selected), set(gold)
    hit = len(chosen & wanted)
    recall = hit / len(wanted)
    precision = hit / len(chosen) if chosen else 0.0
    code = {p for p in wanted if not TEST_PATH.search(p)}
    return {
        "recall": recall,
        "precision": precision,
        "f1": 2 * recall * precision / (recall + precision) if hit else 0.0,
        "all_gold": float(wanted <= chosen),
        "recall_code": len(chosen & code) / len(code) if code else None,
        "line_recall": _lines(spans, chosen & wanted) / (_lines(spans, wanted) or 1),
    }
