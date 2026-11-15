"""The three things repo2txt does. The CLI, MCP server, chat and benchmark call only these."""

import re
import shlex
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from fnmatch import fnmatchcase

from . import tree
from .llm import Complete
from .models import FRAME_TOKENS, Problem, Repository
from .rank import bm25
from .selection import fit
from .tokens import ENCODING, count_tokens

# Room a packed document needs for its own heading. `select` leaves this much of its budget
# unused, so that what it selects can be packed within the same budget.
HEAD_TOKENS = 60
_RANGE = re.compile(r"^(.+):(-?\d+)-(-?\d+)$")


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
    tokens: int  # what the file takes when packed
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
    reason: str  # over_budget | not_found | bad_range


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
    _require_files(repo)
    node = tree.find(tree.of(repo), prefix)
    if node is None or not node.is_dir:
        raise Problem(f"No directory {prefix!r} in {repo.name}.", [_cmd("map", repo)])

    deepest = tree.max_depth(node)
    if depth is None:
        depth = 1
        room = max_tokens - HEAD_TOKENS
        for candidate in range(deepest, 1, -1):
            if count_tokens(tree.render(tree.entries(node, candidate))) <= room:
                depth = candidate
                break
    depth = max(1, min(depth, deepest))
    items = tree.entries(node, depth)

    collapsed = [e for e in items if e.files is not None and e.depth == depth and depth < deepest]
    collapsed.sort(key=lambda e: -e.tokens)
    hints = [_cmd("map", repo, args=[e.path]) for e in collapsed[:3]]
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
    lines: bool = False,
) -> SelectResult:
    """Rank files against `query` and keep, in rank order, those that fit `budget` when packed.

    With `lines` (model method only), the model then chooses line ranges from outlines of the
    files it picked, and items may be `path:first-last`.
    """
    _require_files(repo)
    if method not in METHODS:
        raise Problem(f"Unknown method {method!r}. Use one of: {', '.join(METHODS)}.")
    room = max(budget - HEAD_TOKENS, 0)
    if lines:
        if method != "llm":
            raise Problem("Line ranges are chosen by the model: use --method llm with --lines.")
        items = select_lines(repo, query, room, complete or _default_complete())
        if not items:
            raise Problem(f"The model chose nothing in {repo.name} for {query!r}.")
        rows = [Ranked(item, cost, float(len(items) - i)) for i, (item, cost) in enumerate(items)]
        total = sum(cost for _, cost in items)
        hint = [_cmd("pack", repo, ["-b", str(budget)], [item for item, _ in items[:3]])]
        return SelectResult(
            repo.name, repo.ref, query, method, budget, total, len(items), rows, [], hint
        )
    ranked = METHODS[method](repo, query, room, complete)
    if not ranked:
        raise Problem(f"Nothing in {repo.name} matches {query!r}.", [_cmd("map", repo)])

    scores = dict(ranked)
    kept, skipped = fit(scores, repo, room)
    repo.count(skipped[:10])
    if not kept:
        need = min(repo.files[p].cost for p in skipped) + HEAD_TOKENS
        raise Problem(
            f"No matching file fits in {budget:,} tokens; the smallest needs {need:,}.",
            [_cmd("select", repo, ["-b", str(need)], [query])],
        )

    def row(path: str) -> Ranked:
        return Ranked(path, repo.files[path].cost, round(scores[path], 3))

    return SelectResult(
        repo.name,
        repo.ref,
        query,
        method,
        budget,
        repo.cost(kept),
        len(ranked),
        [row(p) for p in kept],
        [row(p) for p in skipped[:10]],
        [_cmd("pack", repo, ["-b", str(budget)], kept[:3])] if kept else [],
    )


def repo_pack(
    repo: Repository,
    items: Sequence[str] = (),
    budget: int | None = None,
    render: Callable[[PackResult], str] | None = None,
) -> PackResult:
    """The contents of the named files, fitted whole to `budget` in the order given.

    An item is a path, a directory, a glob, or `path:10-80` for a line range. With no items the
    whole repository is packed, smallest files first. Nothing is cut: what does not fit is
    listed in `omitted` with its cost.

    `render` is how the result will be printed. When it is given, the budget covers the whole
    printed document, heading and notes included, and files are dropped from the end until
    the document fits.
    """
    _require_files(repo)
    if items:
        wanted, omitted = _resolve(repo, items)
    else:
        repo.count()
        wanted = [(p, None) for p in sorted(repo.files, key=lambda p: (repo.files[p].tokens, p))]
        omitted = []

    room = None if budget is None else max(budget - HEAD_TOKENS, 0)
    packed, total = [], 0
    for path, lines in wanted:
        file = repo.files[path]
        if lines:
            content = "\n".join(file.text.split("\n")[lines[0] - 1 : lines[1]])
            label = f"{path}:{lines[0]}-{lines[1]}"
            cost = count_tokens(content) + count_tokens(label) + FRAME_TOKENS
        else:
            content, label, cost = file.text, path, file.cost
        if room is not None and total + cost > room:
            omitted.append(Omitted(label, cost, "over_budget"))
            continue
        packed.append(Packed(path, cost, content, lines))
        total += cost

    result = PackResult(repo.name, repo.ref, budget, total, packed, omitted)

    def hints() -> list[str]:
        over = [o for o in result.omitted if o.reason == "over_budget"][:3]
        if not over:
            return []
        need = str(sum(o.tokens for o in over) + HEAD_TOKENS)
        return [_cmd("pack", repo, ["-b", need], [o.path for o in over])]

    result.next = hints()
    if budget is not None and render is not None:
        # The budget covers the printed document: drop the hint first, then files from the end.
        while result.files and count_tokens(render(result)) > budget:
            if result.next:
                result.next = []
                continue
            last = result.files.pop()
            label = f"{last.path}:{last.lines[0]}-{last.lines[1]}" if last.lines else last.path
            result.omitted.append(Omitted(label, last.tokens, "over_budget"))
            result.tokens -= last.tokens

    if not result.files:
        hints = [_cmd("map", repo)]
        if over := [o for o in result.omitted if o.reason == "over_budget"]:
            first = over[0].path.split(":")[0]
            hints = [
                _cmd("pack", repo, args=[f"{first}:1-200"]),
                _cmd("pack", repo, ["-b", str(over[0].tokens + HEAD_TOKENS)], [first]),
            ]
        raise Problem("Nothing was packed: " + _describe(repo, result.omitted), hints)
    return result


def _require_files(repo: Repository) -> None:
    if repo.files:
        return
    message = f"No readable text files in {repo.name}."
    if "-i" in repo.where or "-e" in repo.where:
        message += " Patterns are globs matched against the whole path, such as -i 'src/*'."
    raise Problem(message)


def _resolve(
    repo: Repository, items: Sequence[str]
) -> tuple[list[tuple[str, tuple[int, int] | None]], list[Omitted]]:
    """Turn items into `(path, line range)` pairs, and list the items that cannot be read."""
    resolved: dict[tuple[str, tuple[int, int] | None], None] = {}
    problems = []
    for item in items:
        item = item.strip().removeprefix("./")
        ranged = _RANGE.match(item)
        if item in repo.files:
            resolved[(item, None)] = None
        elif ranged and ranged[1] in repo.files:
            path, first, last = ranged[1], int(ranged[2]), int(ranged[3])
            if 1 <= first <= last and first <= len(repo.files[path].text.split("\n")):
                resolved[(path, (first, last))] = None
            else:
                problems.append(Omitted(item, 0, "bad_range"))
        elif matches := [p for p in repo.files if p.startswith(item.rstrip("/") + "/")]:
            resolved.update(dict.fromkeys((p, None) for p in matches))
        elif matches := [p for p in repo.files if fnmatchcase(p, item)]:
            resolved.update(dict.fromkeys((p, None) for p in matches))
        else:
            problems.append(Omitted(item, 0, "not_found"))
    return list(resolved), problems


def _describe(repo: Repository, omitted: list[Omitted]) -> str:
    parts = []
    for o in omitted[:5]:
        if o.reason == "not_found":
            parts.append(f"{o.path} not found")
        elif o.reason == "bad_range":
            path = _RANGE.match(o.path)[1]
            lines = len(repo.files[path].text.split("\n"))
            parts.append(f"{o.path} is not a line range within {path}, which has {lines} lines")
        else:
            parts.append(f"{o.path} needs {o.tokens:,} tokens")
    return "; ".join(parts)


def _cmd(
    command: str, repo: Repository, options: Sequence[str] = (), args: Sequence[str] = ()
) -> str:
    """A command for a hint. It names the same repository, ref and filters as this run did."""
    words = ["repo2txt", command, *repo.where, *options]
    if any(arg.startswith("-") for arg in args):
        words.append("--")
    return " ".join(shlex.quote(word) for word in [*words, *args])


def select_lines(
    repo: Repository,
    query: str,
    budget: int,
    complete: Complete,
    chosen: list[str] | None = None,
) -> list[tuple[str, int]]:
    """Files chosen at four times the budget, narrowed by the model to line ranges that fit.

    The ranges come first, in the model's order; the rest of the budget goes to the chosen
    files whole, in the order they were chosen. Returns `(item, cost)` pairs, where an item is
    a path or `path:first-last`. `chosen` skips the first stage when it has already been run.
    """
    from .selector import choose_ranges

    if chosen is None:
        chosen = [path for path, _ in _llm(repo, query, budget * 4, complete)]
    ranges, _, _ = choose_ranges(repo, chosen, query, budget, complete)

    shown: dict[str, set[int] | None] = {}

    def cost(path: str, lines: set[int] | None) -> int:
        if lines is None:
            return repo.files[path].cost
        return sum(c for _, c in _intervals(repo, path, lines))

    used = 0
    for path, first, last in ranges:
        if shown.get(path, set()) is None:
            continue
        new = (shown.get(path) or set()) | set(range(first, last + 1))
        change = cost(path, new) - (cost(path, shown[path]) if path in shown else 0)
        if used + change <= budget:
            shown[path], used = new, used + change
    for path in chosen:
        change = repo.files[path].cost - (cost(path, shown[path]) if path in shown else 0)
        if shown.get(path, set()) is not None and used + change <= budget:
            shown[path], used = None, used + change

    items: list[tuple[str, int]] = []
    for path, lines in shown.items():
        items += [(path, repo.files[path].cost)] if lines is None else _intervals(repo, path, lines)
    return items


def _intervals(repo: Repository, path: str, lines: set[int]) -> list[tuple[str, int]]:
    """Contiguous runs of `lines` as `path:first-last` items, with what each costs to pack."""
    text = repo.files[path].text.split("\n")
    runs: list[list[int]] = []
    for n in sorted(lines):
        if runs and n == runs[-1][1] + 1:
            runs[-1][1] = n
        else:
            runs.append([n, n])
    out = []
    for first, last in runs:
        label = f"{path}:{first}-{last}"
        body = "\n".join(text[first - 1 : last])
        out.append((label, count_tokens(body) + count_tokens(label) + FRAME_TOKENS))
    return out


def _default_complete() -> Complete:
    from .llm import LLMSettings, openai_complete

    return openai_complete(LLMSettings.from_env())


def _llm(
    repo: Repository, query: str, budget: int, complete: Complete | None
) -> list[tuple[str, float]]:
    from .selector import Selector

    selector = Selector(repo, budget, complete or _default_complete())
    selector.ask(query)
    # The model gives an order, not scores: rank 1 scores highest.
    count = len(selector.selection.paths)
    return [(p, float(count - i)) for i, p in enumerate(selector.selection.paths)]


Method = Callable[[Repository, str, int, Complete | None], list[tuple[str, float]]]
METHODS: dict[str, Method] = {
    "bm25": lambda repo, query, budget, complete: bm25(repo, query),
    "llm": _llm,
}
