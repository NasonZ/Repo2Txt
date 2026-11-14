"""Local directory source."""

import os
import subprocess
from pathlib import Path

from ..core.files import is_candidate, is_skipped_dir
from ..core.models import Config
from .base import Source, SourceError


class LocalSource(Source):
    workers = 8

    def __init__(self, root: str | Path, config: Config):
        super().__init__(config)
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise SourceError(f"Not a directory: {root}")
        self.name = self.root.name or str(self.root)

    def list_files(self) -> dict[str, int]:
        paths = self._git_paths()
        if paths is None:
            paths = self._walk_paths()
        listing: dict[str, int] = {}
        for rel in paths:
            full = self.root / rel
            try:
                if full.is_symlink() or not full.is_file():
                    continue
                size = full.stat().st_size
            except OSError:
                continue
            if is_candidate(rel, size, self.config):
                listing[rel] = size
        return listing

    def _git_paths(self) -> list[str] | None:
        """Tracked and untracked-but-not-ignored files, or None outside a git work tree."""
        try:
            result = subprocess.run(
                [
                    "git",
                    "-C",
                    str(self.root),
                    "ls-files",
                    "-z",
                    "--cached",
                    "--others",
                    "--exclude-standard",
                ],
                capture_output=True,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return [p for p in result.stdout.decode("utf-8", "surrogateescape").split("\0") if p]

    def _walk_paths(self) -> list[str]:
        paths: list[str] = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            current = Path(dirpath)
            dirnames[:] = [
                d
                for d in dirnames
                if not is_skipped_dir(d, self.config) and not (current / d).is_symlink()
            ]
            paths += [(current / f).relative_to(self.root).as_posix() for f in filenames]
        return paths

    def fetch(self, path: str) -> bytes:
        return (self.root / path).read_bytes()
