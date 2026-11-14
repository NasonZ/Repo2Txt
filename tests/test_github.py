import io
import os
import tarfile

import httpx
import pytest

from repo2txt.adapters import SourceError, open_source
from repo2txt.adapters.github import GitHubSource, parse_github_ref
from repo2txt.core.models import Config


@pytest.mark.parametrize(
    "ref",
    [
        "owner/repo",
        "github.com/owner/repo",
        "https://github.com/owner/repo",
        "https://github.com/owner/repo.git",
        "https://github.com/owner/repo/tree/main/src",
    ],
)
def test_parses_repository_references(ref):
    assert parse_github_ref(ref) == ("owner", "repo")


@pytest.mark.parametrize("ref", ["./relative/path", "/abs/path", "a/b/c", "justaname", ""])
def test_rejects_things_that_are_not_repositories(ref):
    assert parse_github_ref(ref) is None


def test_unknown_reference_is_an_error(tmp_path):
    with pytest.raises(SourceError, match="not a directory"):
        open_source(str(tmp_path / "missing" / "dir"), Config())


def archive(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for path, data in files.items():
            info = tarfile.TarInfo(f"owner-repo-abc123/{path}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def github(files: dict[str, bytes], requests: list[str]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        if request.url.path == "/repos/owner/repo":
            return httpx.Response(200, json={"default_branch": "trunk"})
        if request.url.path == "/repos/owner/repo/tarball/trunk":
            return httpx.Response(200, content=archive(files))
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def test_reads_a_repository_from_its_archive_in_two_requests():
    files = {f"src/module_{i}.py": f"VALUE = {i}\n".encode() for i in range(150)}
    files |= {"node_modules/dep/index.js": b"x", "logo.png": b"\x89PNG\x00"}
    requests: list[str] = []

    source = GitHubSource(
        "owner", "repo", Config(github_token=None), transport=github(files, requests)
    )
    repository = source.load()

    assert len(requests) == 2
    assert repository.branch == "trunk"
    assert len(repository.files) == 150
    assert repository.files["src/module_149.py"].content == "VALUE = 149\n"


def test_a_refused_request_stops_the_run():
    transport = httpx.MockTransport(lambda request: httpx.Response(403))

    with pytest.raises(SourceError, match="GITHUB_TOKEN"):
        GitHubSource("owner", "repo", Config(github_token=None), transport=transport).load()


@pytest.mark.skipif(not os.getenv("GITHUB_TOKEN"), reason="needs GITHUB_TOKEN and network access")
def test_loads_a_public_repository():
    repository = open_source("Doriandarko/RepoToTextForLLMs", Config()).load()

    assert repository.name == "RepoToTextForLLMs"
    assert isinstance(repository.branch, str)
    assert "repototxt.py" in repository.files
    assert repository.tokens() > 0
