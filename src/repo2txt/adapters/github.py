"""GitHub source, read through the REST API's git tree and blob endpoints."""

import base64
import re

import httpx

from ..core.files import is_candidate
from ..core.models import Config
from .base import Source

API = "https://api.github.com"
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
    workers = 16

    def __init__(self, owner: str, repo: str, config: Config, branch: str | None = None):
        super().__init__(config)
        self.owner, self.repo = owner, repo
        self.name = repo
        self.branch = branch
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "repo2txt"}
        if config.github_token:
            headers["Authorization"] = f"Bearer {config.github_token}"
        self.client = httpx.Client(base_url=API, headers=headers, timeout=30)
        self._shas: dict[str, str] = {}

    def _get(self, url: str, **params: str) -> dict:
        response = self.client.get(url, params=params)
        if response.status_code == 404:
            raise ValueError(
                f"{self.owner}/{self.repo} not found. For a private repository set GITHUB_TOKEN."
            )
        if response.status_code == 403 and response.headers.get("x-ratelimit-remaining") == "0":
            raise ValueError("GitHub rate limit reached. Set GITHUB_TOKEN to raise the limit.")
        response.raise_for_status()
        return response.json()

    def list_files(self) -> dict[str, int]:
        base = f"/repos/{self.owner}/{self.repo}"
        if self.branch is None:
            self.branch = self._get(base)["default_branch"]
        tree = self._get(f"{base}/git/trees/{self.branch}", recursive="1")
        if tree.get("truncated"):
            raise ValueError("The repository tree is too large for the GitHub API to list in full.")
        listing: dict[str, int] = {}
        for entry in tree["tree"]:
            if entry["type"] != "blob" or entry["mode"] == "120000":
                continue
            if is_candidate(entry["path"], entry["size"], self.config):
                listing[entry["path"]] = entry["size"]
                self._shas[entry["path"]] = entry["sha"]
        return listing

    def fetch(self, path: str) -> bytes:
        blob = self._get(f"/repos/{self.owner}/{self.repo}/git/blobs/{self._shas[path]}")
        return base64.b64decode(blob["content"])
