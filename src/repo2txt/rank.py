"""Ranking files against a query without a model: BM25 over paths and contents."""

import math
import re
from collections import Counter

from .models import Repository

K1, B = 1.2, 0.75
# A query word found in a file's path counts as this many occurrences in its text.
PATH_WEIGHT = 5

_WORD = re.compile(r"[A-Za-z0-9]+")
_CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")
_STOP = frozenset(
    "a an and are as at be but by do does for from how in is it its of on or so that the this "
    "to was we what when where which why with".split()
)


def terms(text: str) -> list[str]:
    """Lower-case words, with identifiers split: `parseGitHubRef_v2` gives parse, git, hub..."""
    out = []
    for word in _WORD.findall(text):
        parts = _CAMEL.findall(word)
        out += [p.lower() for p in parts]
        if len(parts) > 1:
            out.append(word.lower())
    return [t for t in out if len(t) > 1]


def bm25(repo: Repository, query: str) -> list[tuple[str, float]]:
    """Return `(path, score)` for every file sharing a term with the query, best first."""
    wanted = {t for t in terms(query) if t not in _STOP}
    if not wanted or not repo.files:
        return []

    lengths: dict[str, int] = {}
    counts: dict[str, Counter[str]] = {}
    for path, file in repo.files.items():
        if not file.text.strip():
            continue  # An empty file can match by its path alone, and tells the reader nothing.
        tokens = terms(file.text)
        lengths[path] = len(tokens)
        found = Counter(t for t in tokens if t in wanted)
        for term in terms(path):
            if term in wanted:
                found[term] += PATH_WEIGHT
        if found:
            counts[path] = found

    total = len(lengths)
    average = sum(lengths.values()) / (total or 1) or 1
    containing = Counter(term for found in counts.values() for term in found)
    idf = {t: math.log(1 + (total - n + 0.5) / (n + 0.5)) for t, n in containing.items()}

    scores = {}
    for path, found in counts.items():
        norm = K1 * (1 - B + B * lengths[path] / average)
        scores[path] = sum(idf[t] * tf * (K1 + 1) / (tf + norm) for t, tf in found.items())
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
