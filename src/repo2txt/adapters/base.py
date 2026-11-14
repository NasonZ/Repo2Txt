"""The interface every repository source implements."""

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

from ..core.files import decode
from ..core.models import Config, RepoFile, Repository
from ..core.tokens import count_tokens


class Source(ABC):
    """A place files can be listed from and read."""

    name: str
    branch: str | None = None
    workers: int = 1

    def __init__(self, config: Config):
        self.config = config

    @abstractmethod
    def list_files(self) -> dict[str, int]:
        """Return `{posix_path: size}` for every file that passes the filters."""

    @abstractmethod
    def fetch(self, path: str) -> bytes:
        """Return the raw bytes of one file. Raise on failure."""

    def load(self) -> Repository:
        """Read every candidate file and count its tokens."""
        listing = self.list_files()
        if len(listing) > self.config.max_files:
            raise ValueError(
                f"{self.name} has {len(listing):,} candidate files; the limit is "
                f"{self.config.max_files:,}. Exclude directories with --exclude-dirs."
            )

        errors: list[str] = []

        def read(path: str) -> RepoFile | None:
            try:
                text = decode(self.fetch(path))
            except Exception as exc:
                errors.append(f"{path}: {exc}")
                return None
            if text is None:
                return None
            return RepoFile(path, text, count_tokens(text))

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            results = pool.map(read, sorted(listing))
        files = {f.path: f for f in results if f is not None}
        return Repository(self.name, self.branch, files, errors)
