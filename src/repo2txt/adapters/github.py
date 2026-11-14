"""GitHub source, read from the repository's tarball: two requests whatever its size."""

import io
import re
import tarfile

import httpx

from ..core.files import is_candidate
from ..core.models import Config
from .base import Source, SourceError

API = "https://api.github.com"
MAX_ARCHIVE_BYTES = 500 * 1024 * 1024
_URL_PREFIX = re.compile(r"^(?:https?://)?(?:www\.)?github\.com/")
_NAME = re.compile(r"^[A-Za-z0-9][\w-]*$")
_REPO = re.compile(r"^[\w.-]+$")


def parse_github_ref(ref: str) -> tuple[str, str] | None:
    """Return (owner, repo) for `owner/repo` or a github.com URL, else None."""
    ref = ref.strip().rstrip("/")
    rest, is_url = _URL_PREFIX.subn("", ref)
    parts = rest.split("/")
    if len(parts) < 2 or (len(parts) > 2 and not is_url):
        return None
    owner, repo = parts[0], parts[1].removesuffix(".git")
    if _NAME.match(owner) and _REPO.match(repo):
        return owner, repo
    return None


class GitHubSource(Source):
    def __init__(
        self,
        owner: str,
        repo: str,
        config: Config,
        branch: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ):
        super().__init__(config)
        self.owner, self.repo = owner, repo
        self.name = repo
        self.branch = branch
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "repo2txt"}
        if config.github_token:
            headers["Authorization"] = f"Bearer {config.github_token}"
        self.client = httpx.Client(
            base_url=API, headers=headers, timeout=60, follow_redirects=True, transport=transport
        )
        self._contents: dict[str, bytes] = {}

    def _check(self, response: httpx.Response) -> None:
        if response.status_code == 404:
            where = f"{self.owner}/{self.repo}" + (f" at {self.branch}" if self.branch else "")
            raise SourceError(f"{where} not found. For a private repository set GITHUB_TOKEN.")
        if response.status_code in (403, 429):
            raise SourceError(
                f"GitHub refused the request ({response.status_code}): rate limit reached or "
                "access denied. Set GITHUB_TOKEN."
            )
        if response.is_error:
            raise SourceError(f"GitHub returned {response.status_code} for {response.url}.")

    def _download_archive(self) -> bytes:
        url = f"/repos/{self.owner}/{self.repo}/tarball/{self.branch}"
        chunks, size = [], 0
        with self.client.stream("GET", url) as response:
            self._check(response)
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > MAX_ARCHIVE_BYTES:
                    raise SourceError(
                        f"The archive of {self.owner}/{self.repo} is larger than "
                        f"{MAX_ARCHIVE_BYTES // 2**20} MB. Clone it and run repo2txt on the "
                        "directory."
                    )
                chunks.append(chunk)
        return b"".join(chunks)

    def list_files(self) -> dict[str, int]:
        try:
            if self.branch is None:
                response = self.client.get(f"/repos/{self.owner}/{self.repo}")
                self._check(response)
                self.branch = response.json()["default_branch"]
            archive = self._download_archive()
        except httpx.HTTPError as exc:
            raise SourceError(f"Could not reach GitHub: {exc}") from exc

        listing: dict[str, int] = {}
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
            for member in tar:
                # Every path starts with one generated directory, "owner-repo-sha/".
                _, _, path = member.name.partition("/")
                if not member.isfile() or not is_candidate(path, member.size, self.config):
                    continue
                listing[path] = member.size
                self._contents[path] = tar.extractfile(member).read()
        return listing

    def fetch(self, path: str) -> bytes:
        return self._contents.pop(path)
