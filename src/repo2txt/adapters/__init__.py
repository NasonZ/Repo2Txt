"""Repository sources."""

from pathlib import Path

from ..core.models import Config
from .base import Source, SourceError
from .github import GitHubSource, parse_github_ref
from .local import LocalSource


def open_source(ref: str, config: Config, branch: str | None = None) -> Source:
    """Open a local directory, or a GitHub repository given as a URL or `owner/repo`."""
    if Path(ref).is_dir():
        return LocalSource(ref, config)
    parsed = parse_github_ref(ref)
    if parsed is None:
        raise SourceError(f"{ref!r} is not a directory, a GitHub URL or an owner/repo name.")
    return GitHubSource(*parsed, config, branch=branch)
