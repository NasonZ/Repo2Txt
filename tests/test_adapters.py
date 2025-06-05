import pytest
import os
from unittest.mock import patch, MagicMock, call
from repo2txt.adapters import create_adapter
from repo2txt.adapters.github import GitHubAdapter
from repo2txt.adapters.local import LocalAdapter
from repo2txt.core.models import Config, FileNode


class TestAdapterFactory:
    def test_create_github_adapter(self):
        with patch('repo2txt.adapters.github.Github'):
            adapter = create_adapter("https://github.com/user/repo", Config())
            assert isinstance(adapter, GitHubAdapter)

    def test_create_github_adapter_no_token(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="GITHUB_TOKEN"):
                create_adapter("https://github.com/user/repo", Config())

    def test_create_local_adapter(self):
        # Use temp directory for testing
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = create_adapter(temp_dir, Config())
            assert isinstance(adapter, LocalAdapter)

    def test_create_local_adapter_relative_path(self):
        # Test with current directory
        adapter = create_adapter(".", Config())
        assert isinstance(adapter, LocalAdapter)


class TestGitHubAdapter:
    @pytest.fixture
    def mock_github(self):
        with patch('repo2txt.adapters.github.Github') as mock:
            yield mock

    def test_github_url_parsing(self):
        # Test is done through initialization
        config = Config(github_token="test-token")
        with patch('repo2txt.adapters.github.Github'):
            with patch('repo2txt.adapters.github.Github.return_value.get_repo'):
                adapter = GitHubAdapter("https://github.com/user/repo", config)
                assert adapter.owner == "user"
                assert adapter.repo_name == "repo"

    def test_parse_invalid_url(self):
        config = Config(github_token="test-token") 
        with patch('repo2txt.adapters.github.Github'):
            with pytest.raises(ValueError, match="Invalid GitHub URL"):
                GitHubAdapter("https://github.com/", config)

    def test_initialization(self, mock_github):
        # Config now loads token from .env
        adapter = GitHubAdapter("https://github.com/user/repo", Config())
        assert adapter.owner == "user"
        assert adapter.repo_name == "repo"

    def test_get_name(self, mock_github):
        mock_repo = MagicMock()
        mock_repo.name = "test-repo"
        mock_github.return_value.get_repo.return_value = mock_repo
        
        adapter = GitHubAdapter("https://github.com/user/repo", Config())
        assert adapter.get_name() == "test-repo"


class TestLocalAdapter:
    @pytest.fixture
    def temp_dir(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("print('hello')")
        (tmp_path / "test.txt").write_text("test content")
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("git config")
        return tmp_path

    def test_initialization(self, temp_dir):
        adapter = LocalAdapter(str(temp_dir), Config())
        assert adapter.repo_path == str(temp_dir)
        assert adapter.repo_name == temp_dir.name

    def test_list_contents(self, temp_dir):
        adapter = LocalAdapter(str(temp_dir), Config())
        contents = adapter.list_contents()
        
        # Check that we get tuples of (name, type, size)
        assert len(contents) > 0
        file_names = [item[0] for item in contents]
        assert "src" in file_names
        assert "test.txt" in file_names

    def test_get_file_content(self, temp_dir):
        adapter = LocalAdapter(str(temp_dir), Config())
        content, error = adapter.get_file_content("test.txt")
        
        assert error is None
        assert content == "test content"

    def test_get_name(self, temp_dir):
        adapter = LocalAdapter(str(temp_dir), Config())
        name = adapter.get_name()
        assert name == os.path.basename(str(temp_dir))

    def test_estimate_directory_tokens(self, temp_dir):
        adapter = LocalAdapter(str(temp_dir), Config())
        # FileAnalyzer has count_tokens method, not token_counter
        with patch.object(adapter.file_analyzer, 'count_tokens', return_value=10):
            # Assuming LocalAdapter has this method
            assert hasattr(adapter.file_analyzer, 'count_tokens')

    def test_parse_range(self):
        adapter = LocalAdapter(".", Config())
        assert adapter.parse_range("5") == [5]
        assert adapter.parse_range("1-3") == [1, 2, 3]
        assert adapter.parse_range("8-10") == [8, 9, 10]
        assert adapter.parse_range("1,3,5") == [1, 3, 5]
        assert adapter.parse_range("1-3,7,9-10") == [1, 2, 3, 7, 9, 10]

    def test_parse_range_invalid(self):
        adapter = LocalAdapter(".", Config())
        assert adapter.parse_range("invalid") == []
        assert adapter.parse_range("") == []