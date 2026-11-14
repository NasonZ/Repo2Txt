import pytest


@pytest.fixture
def fixture_repo(tmp_path):
    """A tiny repository with text, binary, generated and excluded files."""
    root = tmp_path / "demo"
    files = {
        "README.md": "# Demo\n\nA small repository.\n",
        "src/small.py": "def small():\n    return 1\n",
        "src/large.py": "\n".join(f"def function_{i}():\n    return {i}" for i in range(200)),
        "tests/test_small.py": "def test_small():\n    assert True\n",
        "node_modules/dep/index.js": "module.exports = 1;\n",
        ".git/config": "[core]\n",
        ".github/workflows/ci.yml": "name: ci\n",
        "uv.lock": "version = 1\n",
        "app.min.js": "var a=1;",
    }
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (root / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00")
    (root / "blob.dat").write_bytes(b"\x00\x01\x02binary")
    return root
