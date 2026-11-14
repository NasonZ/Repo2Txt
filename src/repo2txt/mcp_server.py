"""map, select and pack as MCP tools. Each returns the same text as the command line."""

import functools
from pathlib import Path

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from . import render
from .api import repo_map, repo_pack, repo_select
from .models import Problem, Repository
from .sources import load

# Below the size at which Claude Code warns (10,000 tokens) and pi truncates (20 KB), with room
# for a model whose tokeniser counts higher than ours.
DEFAULT_CAP = 4000

server = MCPServer(
    "repo2txt",
    instructions=(
        "Choose what to read from a repository within a token budget. Start with repo_map, "
        "narrow with repo_select, then read with repo_pack. `repo` is a local directory, a "
        "GitHub URL or owner/repo."
    ),
)
read_only = ToolAnnotations(readOnlyHint=True, idempotentHint=True)


@functools.lru_cache(maxsize=4)
def _remote(repo: str, ref: str | None) -> Repository:
    return load(repo, ref=ref)


def _load(repo: str, ref: str | None) -> Repository:
    """Local directories are read afresh; a GitHub download is reused within the session.

    A client can ask for any path, so only the directory the server was started in, and what
    lies under it, may be read.
    """
    if not Path(repo).is_dir():
        return _remote(repo, ref)
    if not Path(repo).resolve().is_relative_to(Path.cwd().resolve()):
        raise Problem(f"{repo} is outside the directory this server was started in.")
    return load(repo)


def _tool(function):
    """Register a tool whose failures reach the client as an error with a next step."""

    @functools.wraps(function)
    def run(*args, **kwargs) -> str:
        try:
            return function(*args, **kwargs)
        except Problem as exc:
            raise ToolError("\n".join([str(exc), *(f"next: {h}" for h in exc.next)])) from exc

    return server.tool(name=function.__name__.removeprefix("_"), annotations=read_only)(run)


@_tool
def _repo_map(
    repo: str = ".", directory: str = "", max_tokens: int = DEFAULT_CAP, ref: str | None = None
) -> str:
    """Show the tree of a repository, or of one directory in it, with token counts.

    Directories too deep to fit in max_tokens are collapsed to one line with their totals;
    call again with `directory` set to look inside one.
    """
    return render.map_text(repo_map(_load(repo, ref), directory, max_tokens), max_tokens)


@_tool
def _repo_select(
    query: str,
    repo: str = ".",
    budget: int = 32_000,
    max_tokens: int = DEFAULT_CAP,
    ref: str | None = None,
) -> str:
    """Rank files against a question and list those that fit a token budget, best first.

    Returns paths and token counts, not contents. Ranking is lexical (BM25 over paths,
    identifiers and text), so name the functions, classes or terms you expect in the code.
    """
    return render.select_text(repo_select(_load(repo, ref), query, budget), max_tokens)


@_tool
def _repo_pack(
    items: list[str], repo: str = ".", budget: int = DEFAULT_CAP, ref: str | None = None
) -> str:
    """Return the contents of files, fitted whole to a token budget in the order given.

    An item is a path, a directory, a glob, or `path:10-80` for a line range. Files that do
    not fit are listed with their token cost, never cut: ask for a line range or a larger budget.
    """
    return render.pack_text(repo_pack(_load(repo, ref), items, budget, render.pack_text))
