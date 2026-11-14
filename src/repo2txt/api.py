"""The three things repo2txt does. The CLI, MCP server, chat and benchmark call only these."""

import re
import shlex
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from fnmatch import fnmatchcase

from . import tree
from .llm import Complete
from .models import Problem, Repository
from .rank import bm25
from .selection import fit
from .tokens import ENCODING, count_tokens

# Tokens a packed file costs beyond its content and its path: heading, fence and blank lines.
PACK_OVERHEAD = 8
_RANGE = re.compile(r"^(.+):(\d+)-(\d+)$")


@dataclass
class MapResult:
    repo: str
    ref: str | None
    encoding: str
    prefix: str
    files: int
    tokens: int
    depth: int
    max_depth: int
    entries: list[tree.Entry]
    next: list[str] = field(default_factory=list)


@dataclass
class Ranked:
    path: str
    tokens: int
    score: float


@dataclass
class SelectResult:
    repo: str
    ref: str | None
    query: str
    method: str
    budget: int
    tokens: int
    candidates: int
    files: list[Ranked]
    skipped: list[Ranked]
    next: list[str] = field(default_factory=list)


@dataclass
class Packed:
    path: str
    tokens: int
    content: str
    lines: tuple[int, int] | None = None


@dataclass
class Omitted:
    path: str
    tokens: int
    reason: str  # over_budget | not_found


@dataclass
class PackResult:
    repo: str
    ref: str | None
    budget: int | None
    tokens: int
    files: list[Packed]
    omitted: list[Omitted]
    next: list[str] = field(default_factory=list)


def repo_map(
    repo: Repository, prefix: str = "", max_tokens: int = 4000, depth: int | None = None
) -> MapResult:
    """The directory tree under `prefix` with token counts.

    Without `depth`, the tree is shown as deep as fits in `max_tokens`; directories below that
    are collapsed to one line with their totals.
    """
    repo.count()
    root = tree.build_tree({p: f.tokens for p, f in repo.files.items()})
    node = tree.find(root, prefix)
    if node is None or not node.is_dir:
        raise Problem(f"No directory {prefix!r} in {repo.name}.", [_cmd("map", repo)])

    deepest = tree.max_depth(node)
    if depth is None:
        depth = 1
        for candidate in range(deepest, 1, -1):
            if count_tokens(tree.render(tree.entries(node, candidate))) <= max_tokens:
                depth = candidate
                break
    depth = max(1, min(depth, deepest))
    items = tree.entries(node, depth)

    collapsed = [e for e in items if e.files is not None and e.depth == depth and depth < deepest]
    collapsed.sort(key=lambda e: -e.tokens)
    hints = [_cmd("map", repo, e.path) for e in collapsed[:3]]
    return MapResult(
        repo.name,
        repo.ref,
        ENCODING,
        prefix.strip("/"),
        node.files,
        node.tokens,
        depth,
        deepest,
        items,
        hints,
    )


def repo_select(
    repo: Repository,
    query: str,
    budget: int = 32_000,
    method: str = "bm25",
    complete: Complete | None = None,
) -> SelectResult:
    """Rank files against `query` and keep, in rank order, those that fit `budget`."""
    if method not in METHODS:
        raise Problem(f"Unknown method {method!r}. Use one of: {', '.join(METHODS)}.")
    ranked = METHODS[method](repo, query, budget, complete)
    if not ranked:
        raise Problem(f"Nothing in {repo.name} matches {query!r}.", [_cmd("map", repo)])

    scores = dict(ranked)
    kept, skipped = fit(scores, repo, budget)
    repo.count(skipped[:10])

    def row(path: str) -> Ranked:
        return Ranked(path, repo.files[path].tokens, round(scores[path], 3))

    return SelectResult(
        repo.name,
        repo.ref,
        query,
        method,
        budget,
        repo.tokens(kept),
        len(ranked),
        [row(p) for p in kept],
        [row(p) for p in skipped[:10]],
        [_cmd("pack", repo, *kept[:3], "-b", str(budget))] if kept else [],
    )


def repo_pack(repo: Repository, items: Sequence[str] = (), budget: int | None = None) -> PackResult:
    """The contents of the named files, fitted whole to `budget` in the order given.

    An item is a path, a directory, a glob, or `path:10-80` for a line range. With no items the
    whole repository is packed, smallest files first. Nothing is cut: what does not fit is
    listed in `omitted` with its cost.
    """
    if items:
        wanted, missing = _resolve(repo, items)
    else:
        repo.count()
        wanted = [(p, None) for p in sorted(repo.files, key=lambda p: (repo.files[p].tokens, p))]
        missing = []

    packed, omitted, total = [], [Omitted(item, 0, "not_found") for item in missing], 0
    for path, lines in wanted:
        text = repo.files[path].text
        if lines:
            content = "\n".join(text.splitlines()[lines[0] - 1 : lines[1]])
            tokens = count_tokens(content)
        else:
            content, tokens = text, repo.files[path].tokens
        cost = tokens + count_tokens(path) + PACK_OVERHEAD
        label = f"{path}:{lines[0]}-{lines[1]}" if lines else path
        if budget is not None and total + cost > budget:
            omitted.append(Omitted(label, cost, "over_budget"))
            continue
        packed.append(Packed(path, cost, content, lines))
        total += cost

    if not packed:
        hint = [_cmd("map", repo)]
        if over := [o for o in omitted if o.reason == "over_budget"]:
            first = over[0].path.split(":")[0]
            hint = [_cmd("pack", repo, f"{first}:1-200"), _cmd("pack", repo, first, "-b", "32k")]
        raise Problem("Nothing was packed: " + _describe(omitted), hint)

    hints = []
    if over := [o for o in omitted if o.reason == "over_budget"]:
        need = str(sum(o.tokens for o in over[:3]))
        hints = [_cmd("pack", repo, *(o.path for o in over[:3]), "-b", need)]
    return PackResult(repo.name, repo.ref, budget, total, packed, omitted, hints)


def _resolve(
    repo: Repository, items: Sequence[str]
) -> tuple[list[tuple[str, tuple[int, int] | None]], list[str]]:
    """Turn items into `(path, line range)` pairs, and list the items that match nothing."""
    resolved: dict[tuple[str, tuple[int, int] | None], None] = {}
    missing = []
    for item in items:
        item = item.strip()
        ranged = _RANGE.match(item)
        if item in repo.files:
            resolved[(item, None)] = None
        elif ranged and ranged[1] in repo.files and 1 <= int(ranged[2]) <= int(ranged[3]):
            resolved[(ranged[1], (int(ranged[2]), int(ranged[3])))] = None
        elif matches := [p for p in repo.files if p.startswith(item.rstrip("/") + "/")]:
            resolved.update(dict.fromkeys((p, None) for p in matches))
        elif matches := [p for p in repo.files if fnmatchcase(p, item)]:
            resolved.update(dict.fromkeys((p, None) for p in matches))
        else:
            missing.append(item)
    return list(resolved), missing


def _describe(omitted: list[Omitted]) -> str:
    return "; ".join(
        f"{o.path} not found" if o.reason == "not_found" else f"{o.path} needs {o.tokens:,} tokens"
        for o in omitted[:5]
    )


def _cmd(command: str, repo: Repository, *args: str) -> str:
    """A runnable command for a hint."""
    words = [a if a.startswith("-") else shlex.quote(a) for a in args]
    where = [] if repo.source == "." else ["-r", shlex.quote(repo.source)]
    return " ".join(["repo2txt", command, *words, *where])


def _smallest(repo: Repository, *_: object) -> list[tuple[str, float]]:
    repo.count()
    return [(p, 0.0) for p in sorted(repo.files, key=lambda p: (repo.files[p].tokens, p))]


def _llm(
    repo: Repository, query: str, budget: int, complete: Complete | None
) -> list[tuple[str, float]]:
    from .llm import LLMSettings, openai_complete
    from .selector import Selector

    selector = Selector(repo, budget, complete or openai_complete(LLMSettings.from_env()))
    selector.ask(query)
    # The model gives an order, not scores: rank 1 scores highest.
    count = len(selector.selection.paths)
    return [(p, float(count - i)) for i, p in enumerate(selector.selection.paths)]


Method = Callable[[Repository, str, int, Complete | None], list[tuple[str, float]]]
METHODS: dict[str, Method] = {
    "bm25": lambda repo, query, budget, complete: bm25(repo, query),
    "llm": _llm,
    "smallest": _smallest,
}
