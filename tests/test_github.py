import os

import pytest

from repo2txt.adapters import open_source
from repo2txt.adapters.github import parse_github_ref
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
    with pytest.raises(ValueError, match="not a directory"):
        open_source(str(tmp_path / "missing" / "dir"), Config())


@pytest.mark.skipif(not os.getenv("GITHUB_TOKEN"), reason="needs GITHUB_TOKEN and network access")
def test_loads_a_public_repository():
    repository = open_source("Doriandarko/RepoToTextForLLMs", Config()).load()

    assert repository.name == "RepoToTextForLLMs"
    assert isinstance(repository.branch, str)
    assert "repototxt.py" in repository.files
    assert repository.tokens() > 0
