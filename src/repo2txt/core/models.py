"""Data structures shared across the pipeline."""

import os
from dataclasses import dataclass, field

DEFAULT_EXCLUDED_DIRS = frozenset(
    {
        "__pycache__",
        "node_modules",
        "venv",
        "env",
        "virtualenv",
        "vendor",
        "bower_components",
        "dist",
        "build",
        "out",
        "coverage",
        "datasets",
    }
)


@dataclass
class Config:
    """Limits and filters applied when reading a repository."""

    excluded_dirs: set[str] = field(default_factory=lambda: set(DEFAULT_EXCLUDED_DIRS))
    max_file_size: int = 1024 * 1024
    max_files: int = 5000
    github_token: str | None = field(default_factory=lambda: os.getenv("GITHUB_TOKEN") or None)


@dataclass
class RepoFile:
    path: str
    content: str
    tokens: int


@dataclass
class Repository:
    """A repository after filtering, with the text and token count of every candidate file."""

    name: str
    branch: str | None
    files: dict[str, RepoFile]
    errors: list[str] = field(default_factory=list)

    def tokens(self, paths: list[str] | None = None) -> int:
        selected = self.files if paths is None else paths
        return sum(self.files[p].tokens for p in selected)

    def token_map(self) -> dict[str, int]:
        return {p: f.tokens for p, f in self.files.items()}
