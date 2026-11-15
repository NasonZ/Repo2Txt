"""Ranking files against a query without a model: BM25 over paths and contents."""

import math
import re
from collections import Counter

from .models import Repository

K1, B = 1.2, 0.75
# A query word found in a file's path counts as this many occurrences in its text.
PATH_WEIGHT = 5
# A file that defines an identifier the query names scores this much more per identifier.
DEFINITION_BOOST = 1.5
# Tests and prose describe behaviour; the code that has it ranks above them.
SECONDARY_WEIGHT = 0.5

_WORD = re.compile(r"[A-Za-z0-9]+")
_CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")
_IDENTIFIER = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b")
_DEFINES = r"\b(?:def|class|function|fn|func|struct|interface|trait|type|enum|module)\s+(?:{})\b"
_SECONDARY = re.compile(
    r"(^|/)(tests?|testing|spec|specs|__tests__|docs?|examples?|fixtures)/"
    r"|(^|/)test_|_test\.|\.test\.|\.spec\.|\.(md|rst|txt|adoc)$"
)
STOP = frozenset(
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


def bm25(repo: Repository, query: str, boosts: bool = True) -> list[tuple[str, float]]:
    """Return `(path, score)` for every file sharing a term with the query, best first.

    With `boosts`, three things a lexical score misses are added: a file the query names
    outright comes first, a file that defines an identifier the query names is lifted, and
    tests and prose rank below code.
    """
    key = ("bm25", query, boosts)
    if key not in repo.cache:
        result = _bm25(repo, query, boosts)
        # Keep the rankings for one query at a time.
        for old in [k for k in repo.cache if k[0] in ("bm25", "idf") and k[1] != query]:
            del repo.cache[old]
        repo.cache[key] = result
    return repo.cache[key]


def idf(repo: Repository, query: str) -> dict[str, float]:
    """How rare each of the query's words is across the repository's files."""
    bm25(repo, query)
    return repo.cache.get(("idf", query), {})


def _bm25(repo: Repository, query: str, boosts: bool) -> list[tuple[str, float]]:
    wanted = {t for t in terms(query) if t not in STOP}
    if not wanted or not repo.files:
        return []

    # Identifiers worth looking for as definitions: written like code, not like prose.
    named = {
        word
        for word in _IDENTIFIER.findall(query)
        if "_" in word or re.search(r"[a-z][A-Z]", word) or f"`{word}`" in query
    }
    defines = re.compile(_DEFINES.format("|".join(map(re.escape, named)))) if named else None

    lengths: dict[str, int] = {}
    counts: dict[str, Counter[str]] = {}
    defined: dict[str, int] = {}
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
            if boosts and defines:
                defined[path] = len(set(defines.findall(file.text)))

    total = len(lengths)
    average = sum(lengths.values()) / (total or 1) or 1
    containing = Counter(term for found in counts.values() for term in found)
    idf = {t: math.log(1 + (total - n + 0.5) / (n + 0.5)) for t, n in containing.items()}

    repo.cache[("idf", query)] = idf
    scores = {}
    for path, found in counts.items():
        norm = K1 * (1 - B + B * lengths[path] / average)
        scores[path] = sum(idf[t] * tf * (K1 + 1) / (tf + norm) for t, tf in found.items())

    if boosts and scores:
        top = max(scores.values())
        for path in scores:
            scores[path] *= DEFINITION_BOOST ** defined.get(path, 0)
            if _SECONDARY.search(path):
                scores[path] *= SECONDARY_WEIGHT
        for path in repo.files:
            name = path.rsplit("/", 1)[-1]
            if (
                path in query
                or (len(name) > 4 and re.search(rf"(?<![\w/.-]){re.escape(name)}\b", query))
            ) and repo.files[path].text.strip():
                scores[path] = scores.get(path, 0.0) + 2 * top
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
