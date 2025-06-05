import pytest
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def temp_workspace():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_repo(temp_workspace):
    """Create a sample repository structure for testing."""
    repo_root = temp_workspace / "sample_repo"
    repo_root.mkdir()
    
    # Create directory structure
    (repo_root / "src").mkdir()
    (repo_root / "src" / "utils").mkdir()
    (repo_root / "tests").mkdir()
    (repo_root / "docs").mkdir()
    (repo_root / ".git").mkdir()
    (repo_root / "node_modules").mkdir()
    
    # Create files
    (repo_root / "README.md").write_text("# Sample Repository\n\nTest repository for repo2txt")
    (repo_root / "setup.py").write_text("from setuptools import setup\n\nsetup(name='sample')")
    (repo_root / "src" / "__init__.py").write_text("")
    (repo_root / "src" / "main.py").write_text("def main():\n    print('Hello, World!')")
    (repo_root / "src" / "utils" / "helpers.py").write_text("def helper():\n    return 42")
    (repo_root / "tests" / "test_main.py").write_text("def test_main():\n    assert True")
    (repo_root / ".git" / "config").write_text("[core]\n    repositoryformatversion = 0")
    (repo_root / "node_modules" / "package.json").write_text('{"name": "test"}')
    
    # Create binary file
    (repo_root / "image.png").write_bytes(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR')
    
    return repo_root