"""The repository as every other module sees it."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import cached_property

from .tokens import count_many, count_tokens

# What a packed file costs beyond its content and its path: heading, fence and blank lines.
FRAME_TOKENS = 16


class Problem(Exception):
    """Something the caller can act on. `next` holds commands worth trying."""

    def __init__(self, message: str, next: list[str] | None = None):
        super().__init__(message)
        self.next = next or []


@dataclass
class RepoFile:
    path: str
    text: str

    @cached_property
    def tokens(self) -> int:
        return count_tokens(self.text)

    @cached_property
    def cost(self) -> int:
        """Tokens this file takes in a packed document: its content and its frame."""
        return self.tokens + count_tokens(self.path) + FRAME_TOKENS


@dataclass
class Repository:
    """The text files of a repository, after filtering.

    `where` holds the command-line options that name this repository again, filters included,
    so that hints can be commands which read the same files.
    """

    name: str
    where: list[str]
    ref: str | None
    files: dict[str, RepoFile]
    errors: list[str] = field(default_factory=list)
    # Work derived from the files and worth doing once: the tree, the last ranking.
    cache: dict = field(default_factory=dict, repr=False, compare=False)

    def tokens(self, paths: Iterable[str] | None = None) -> int:
        """Total content tokens of `paths`, or of every file."""
        selected = list(self.files if paths is None else paths)
        self.count(selected)
        return sum(self.files[p].tokens for p in selected)

    def cost(self, paths: Iterable[str]) -> int:
        """What `paths` take in a packed document."""
        selected = list(paths)
        self.count(selected)
        return sum(self.files[p].cost for p in selected)

    def count(self, paths: Iterable[str] | None = None) -> None:
        """Count tokens for many files in one parallel batch, where not already counted."""
        pending = [
            f
            for f in (self.files[p] for p in (self.files if paths is None else paths))
            if "tokens" not in f.__dict__
        ]
        for file, tokens in zip(pending, count_many([f.text for f in pending]), strict=True):
            file.__dict__["tokens"] = tokens
