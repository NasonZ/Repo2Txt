"""Local directory source."""

import os
from pathlib import Path

from ..core.files import is_candidate
from ..core.models import Config
from .base import Source


class LocalSource(Source):
    workers = 8

    def __init__(self, root: str | Path, config: Config):
        super().__init__(config)
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ValueError(f"Not a directory: {root}")
        self.name = self.root.name or str(self.root)

    def list_files(self) -> dict[str, int]:
        listing: dict[str, int] = {}
        for dirpath, dirnames, filenames in os.walk(self.root):
            current = Path(dirpath)
            dirnames[:] = [d for d in dirnames if not (current / d).is_symlink()]
            for filename in filenames:
                full = current / filename
                if full.is_symlink():
                    continue
                rel = full.relative_to(self.root).as_posix()
                try:
                    size = full.stat().st_size
                except OSError:
                    continue
                if is_candidate(rel, size, self.config):
                    listing[rel] = size
        return listing

    def fetch(self, path: str) -> bytes:
        return (self.root / path).read_bytes()
