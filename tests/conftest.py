import pytest
from click.testing import CliRunner

from repo2txt.cli import cli
from repo2txt.sources import load

FILES = {
    "README.md": "# Demo\n\nA small repository.\n",
    "src/small.py": "def small():\n    return 1\n",
    "src/large.py": "\n".join(f"def function_{i}():\n    return {i}" for i in range(200)),
    "src/auth/login.py": "def check_password(user, password):\n    return verifyHash(password)\n",
    "src/auth/session.py": "class SessionStore:\n    pass\n",
    "tests/test_small.py": "def test_small():\n    assert True\n",
    "node_modules/dep/index.js": "module.exports = 1;\n",
    ".git/config": "[core]\n",
    ".github/workflows/ci.yml": "name: ci\n",
    "uv.lock": "version = 1\n",
    "app.min.js": "var a=1;",
}
TEXT_FILES = {
    "README.md",
    "src/small.py",
    "src/large.py",
    "src/auth/login.py",
    "src/auth/session.py",
    "tests/test_small.py",
    ".github/workflows/ci.yml",
}


@pytest.fixture
def root(tmp_path):
    """A small repository with text, binary, generated and excluded files."""
    root = tmp_path / "demo"
    for rel, text in FILES.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (root / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00")
    (root / "blob.dat").write_bytes(b"\x00\x01\x02binary")
    return root


@pytest.fixture
def repo(root):
    return load(str(root))


@pytest.fixture
def run(root):
    """Run the command on the fixture repository and return the result."""

    def run(*args, input=None):
        return CliRunner().invoke(cli, [*args, "-r", str(root)], input=input)

    return run
