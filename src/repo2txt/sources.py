"""Reading a repository from a local directory or from GitHub."""

import io
import os
import re
import subprocess
import tarfile
from collections.abc import Callable, Iterator
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath

import httpx

from .models import Problem, RepoFile, Repository

API = "https://api.github.com"
MAX_ARCHIVE_BYTES = 500 * 1024 * 1024

SKIP_DIRS = frozenset({"node_modules", "__pycache__"})
# Only used when walking a directory that git does not describe.
WALK_SKIP_DIRS = frozenset({"venv", "env", "virtualenv", "dist", "build", "out", "coverage"})
VISIBLE_DOT_DIRS = frozenset({".github", ".gitlab"})
SKIP_NAMES = frozenset(
    {
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "poetry.lock",
        "uv.lock",
        "Pipfile.lock",
        "composer.lock",
        "Cargo.lock",
    }
)
SKIP_SUFFIXES = (".min.js", ".min.css", ".map")
# Formats that are never text. Anything else is read and checked for null bytes.
BINARY_EXTENSIONS = frozenset(
    ".png .jpg .jpeg .gif .bmp .ico .webp .pdf .zip .gz .tar .7z .jar .whl .so .dll .dylib .exe "
    ".pyc .class .o .a .woff .woff2 .ttf .mp3 .mp4 .mov .wav .sqlite .db".split()
)

Keep = Callable[[str, int], bool]

_URL_PREFIX = re.compile(r"^(?:https?://)?(?:www\.)?github\.com/")
_OWNER = re.compile(r"^[A-Za-z0-9][\w-]*$")
_REPO = re.compile(r"^[\w.-]+$")


class SourceError(Problem):
    """The repository cannot be read: not found, refused, unreachable."""


def _skip_dir(name: str) -> bool:
    hidden = name.startswith(".") and name not in VISIBLE_DOT_DIRS
    return hidden or name in SKIP_DIRS or name.endswith(".egg-info")


def is_candidate(path: str, size: int, max_file_size: int) -> bool:
    """Decide from the path and size alone whether a file is worth reading."""
    *dirs, name = PurePosixPath(path).parts
    if any(_skip_dir(part) for part in dirs):
        return False
    if name.startswith(".") or name in SKIP_NAMES or name.endswith(SKIP_SUFFIXES):
        return False
    if PurePosixPath(name).suffix.lower() in BINARY_EXTENSIONS:
        return False
    return size <= max_file_size


def decode(data: bytes) -> str | None:
    """Decode file bytes as text, or return None if they look binary."""
    if b"\x00" in data[:8192]:
        return None
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def parse_github_ref(ref: str) -> tuple[str, str] | None:
    """Return (owner, repo) for `owner/repo` or a github.com URL, else None."""
    rest, is_url = _URL_PREFIX.subn("", ref.strip().rstrip("/"))
    parts = rest.split("/")
    if len(parts) < 2 or (len(parts) > 2 and not is_url):
        return None
    owner, repo = parts[0], parts[1].removesuffix(".git")
    return (owner, repo) if _OWNER.match(owner) and _REPO.match(repo) else None


def load(
    source: str,
    *,
    ref: str | None = None,
    include: tuple[str, ...] = (),
    exclude: tuple[str, ...] = (),
    max_file_size: int = 1_000_000,
    transport: httpx.BaseTransport | None = None,
) -> Repository:
    """Read a local directory, or a GitHub repository given as a URL or `owner/repo`.

    `ref` is a branch, tag or commit and applies to GitHub only. `include` and `exclude` are
    glob patterns matched against the whole path, where `*` also crosses directories.
    """

    def keep(path: str, size: int) -> bool:
        if include and not any(fnmatchcase(path, pattern) for pattern in include):
            return False
        if any(fnmatchcase(path, pattern) for pattern in exclude):
            return False
        return is_candidate(path, size, max_file_size)

    errors: list[str] = []
    if Path(source).is_dir():
        root = Path(source).resolve()
        name, pairs = root.name or str(root), local_files(root, keep, errors)
    elif parsed := parse_github_ref(source):
        archive, ref = download_tarball(*parsed, ref, transport)
        name, pairs = parsed[1], tar_files(archive, keep)
    else:
        raise SourceError(f"{source!r} is not a directory, a GitHub URL or an owner/repo name.")

    files = {}
    for path, data in pairs:
        if (text := decode(data)) is not None:
            files[path] = RepoFile(path, text)
    return Repository(name, source, ref, dict(sorted(files.items())), errors)


def local_files(root: Path, keep: Keep, errors: list[str]) -> Iterator[tuple[str, bytes]]:
    """Yield (path, bytes) for the files of a directory, leaving out what git ignores."""
    paths = _git_paths(root)
    if paths is None:
        paths = _walk_paths(root)
    for rel in paths:
        full = root / rel
        try:
            if full.is_symlink() or not full.is_file() or not keep(rel, full.stat().st_size):
                continue
            yield rel, full.read_bytes()
        except OSError as exc:
            errors.append(f"{rel}: {exc}")


def _git_paths(root: Path) -> list[str] | None:
    """Tracked files and untracked files that are not ignored, or None outside a work tree."""
    command = ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others"]
    try:
        result = subprocess.run([*command, "--exclude-standard"], capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return [p for p in result.stdout.decode("utf-8", "surrogateescape").split("\0") if p]


def _walk_paths(root: Path) -> list[str]:
    paths: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        dirnames[:] = [
            d
            for d in dirnames
            if d not in WALK_SKIP_DIRS and not _skip_dir(d) and not (current / d).is_symlink()
        ]
        paths += [(current / f).relative_to(root).as_posix() for f in filenames]
    return paths


def download_tarball(
    owner: str, repo: str, ref: str | None, transport: httpx.BaseTransport | None = None
) -> tuple[bytes, str]:
    """Download a repository's tarball. Returns the archive and the ref it was read at."""
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "repo2txt"}
    if token := os.getenv("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"

    def check(response: httpx.Response) -> None:
        if response.status_code == 404:
            where = f"{owner}/{repo}" + (f" at {ref}" if ref else "")
            raise SourceError(f"{where} not found. For a private repository set GITHUB_TOKEN.")
        if response.status_code in (403, 429):
            raise SourceError(
                f"GitHub refused the request ({response.status_code}): rate limit reached or "
                "access denied. Set GITHUB_TOKEN."
            )
        if response.is_error:
            raise SourceError(f"GitHub returned {response.status_code} for {response.url}.")

    try:
        with httpx.Client(
            base_url=API, headers=headers, timeout=60, follow_redirects=True, transport=transport
        ) as client:
            if ref is None:
                response = client.get(f"/repos/{owner}/{repo}")
                check(response)
                ref = response.json()["default_branch"]
            chunks, size = [], 0
            with client.stream("GET", f"/repos/{owner}/{repo}/tarball/{ref}") as response:
                check(response)
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > MAX_ARCHIVE_BYTES:
                        raise SourceError(
                            f"The archive of {owner}/{repo} is larger than "
                            f"{MAX_ARCHIVE_BYTES // 2**20} MB. Clone it and use the directory."
                        )
                    chunks.append(chunk)
    except httpx.HTTPError as exc:
        raise SourceError(f"Could not reach GitHub: {exc}") from exc
    return b"".join(chunks), ref


def tar_files(archive: bytes, keep: Keep) -> Iterator[tuple[str, bytes]]:
    """Yield (path, bytes) for the regular files of a GitHub tarball."""
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        for member in tar:
            # Every path starts with one generated directory, "owner-repo-sha/".
            _, _, path = member.name.partition("/")
            if member.isfile() and keep(path, member.size):
                yield path, tar.extractfile(member).read()
