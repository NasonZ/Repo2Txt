"""Rules for which files are worth reading, and how to decode them."""

from pathlib import PurePosixPath

from .models import Config

BINARY_EXTENSIONS = frozenset(
    {
        ".exe",
        ".dll",
        ".so",
        ".a",
        ".lib",
        ".dylib",
        ".o",
        ".obj",
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".7z",
        ".rar",
        ".jar",
        ".war",
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".ico",
        ".svg",
        ".webp",
        ".mp3",
        ".mp4",
        ".avi",
        ".mov",
        ".wmv",
        ".flv",
        ".webm",
        ".wav",
        ".flac",
        ".ogg",
        ".m4a",
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        ".db",
        ".sqlite",
        ".mdb",
        ".accdb",
        ".pyc",
        ".pyo",
        ".pyd",
        ".whl",
        ".class",
        ".dex",
        ".apk",
        ".ipa",
        ".ds_store",
    }
)

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
VISIBLE_DOT_DIRS = frozenset({".github", ".gitlab"})


def is_skipped_dir(name: str, config: Config) -> bool:
    """Whether a directory, and everything under it, is left out."""
    hidden = name.startswith(".") and name not in VISIBLE_DOT_DIRS
    return name in config.excluded_dirs or hidden or name.endswith(".egg-info")


def is_candidate(path: str, size: int, config: Config) -> bool:
    """Decide from the path and size alone whether a file should be read."""
    pure = PurePosixPath(path)
    *dirs, name = pure.parts
    if any(is_skipped_dir(part, config) for part in dirs):
        return False
    if name.startswith("."):
        return False
    if name in SKIP_NAMES or name.endswith(SKIP_SUFFIXES):
        return False
    if pure.suffix.lower() in BINARY_EXTENSIONS:
        return False
    return size <= config.max_file_size


def decode(data: bytes) -> str | None:
    """Decode file bytes as text, or return None if they look binary."""
    if b"\x00" in data[:8192]:
        return None
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("latin-1")
