import pytest

from repo2txt.core.files import decode, is_candidate
from repo2txt.core.models import Config


@pytest.mark.parametrize(
    "path, size, expected",
    [
        ("src/app.py", 100, True),
        (".github/workflows/ci.yml", 100, True),
        (".git/config", 10, False),
        (".env", 10, False),
        ("node_modules/x/index.js", 10, False),
        ("pkg.egg-info/PKG-INFO", 10, False),
        ("logo.png", 10, False),
        ("static/app.min.js", 10, False),
        ("uv.lock", 10, False),
        ("big.py", 2 * 1024 * 1024, False),
    ],
)
def test_is_candidate(path, size, expected):
    assert is_candidate(path, size, Config()) is expected


def test_decode_rejects_binary_and_falls_back_for_legacy_text():
    assert decode(b"\x89PNG\x00\x00") is None
    assert decode("café".encode()) == "café"
    assert decode("café".encode("latin-1")) == "café"
