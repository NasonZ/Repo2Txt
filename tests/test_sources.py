import io
import os
import tarfile

import httpx
import pytest

from repo2txt.sources import SourceError, load, parse_github_ref


@pytest.mark.parametrize(
    "ref",
    [
        "owner/repo",
        "github.com/owner/repo",
        "https://github.com/owner/repo.git",
        "https://github.com/owner/repo/tree/main/src",
    ],
)
def test_parses_repository_references(ref):
    assert parse_github_ref(ref) == ("owner", "repo")


@pytest.mark.parametrize("ref", ["repo", "a/b/c", "https://example.com/owner/repo", "-x/repo"])
def test_rejects_things_that_are_not_repositories(ref):
    assert parse_github_ref(ref) is None


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
        if request.url.path in (
            "/repos/owner/repo/tarball/trunk",
            "/repos/owner/repo/tarball/abc123",
        ):
            return httpx.Response(200, content=archive(files))
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def test_reads_a_repository_from_its_archive_in_two_requests():
    files = {f"src/module_{i}.py": f"VALUE = {i}\n".encode() for i in range(150)}
    files |= {"node_modules/dep/index.js": b"x", "logo.png": b"\x89PNG\x00"}
    requests: list[str] = []

    repo = load("owner/repo", transport=github(files, requests))

    assert len(requests) == 2
    assert repo.ref == "trunk"
    assert len(repo.files) == 150
    assert repo.files["src/module_149.py"].text == "VALUE = 149\n"


def test_a_named_commit_is_read_without_asking_for_the_default_branch():
    requests: list[str] = []

    repo = load("owner/repo", ref="abc123", transport=github({"a.py": b"A = 1\n"}, requests))

    assert requests == ["/repos/owner/repo/tarball/abc123"]
    assert repo.ref == "abc123" and list(repo.files) == ["a.py"]


def test_a_refused_request_stops_with_advice():
    transport = httpx.MockTransport(lambda request: httpx.Response(403))

    with pytest.raises(SourceError, match="GITHUB_TOKEN"):
        load("owner/repo", transport=transport)


def test_something_that_is_neither_a_directory_nor_a_repository_is_an_error(tmp_path):
    with pytest.raises(SourceError, match="not a directory"):
        load(str(tmp_path / "missing"))


@pytest.mark.skipif(not os.getenv("GITHUB_TOKEN"), reason="needs GITHUB_TOKEN and network access")
def test_loads_a_public_repository():
    repo = load("Doriandarko/RepoToTextForLLMs")

    assert repo.files and repo.tokens() > 0
