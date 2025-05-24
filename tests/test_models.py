import pytest
from repo2txt.core.models import Config, FileNode, AnalysisResult, TokenBudget


class TestConfig:
    def test_default_config(self):
        config = Config()
        assert config.max_file_size == 1024 * 1024
        assert config.enable_token_counting is True
        assert config.token_encoder == "cl100k_base"
        assert config.output_format == "markdown"
        assert ".git" in config.excluded_dirs
        assert ".py" not in config.binary_extensions
        assert ".exe" in config.binary_extensions

    def test_custom_config(self):
        config = Config(
            max_file_size=2048,
            enable_token_counting=False,
            output_format="xml"
        )
        assert config.max_file_size == 2048
        assert config.enable_token_counting is False
        assert config.output_format == "xml"

    def test_encoding_fallbacks(self):
        config = Config()
        assert "utf-8" in config.encoding_fallbacks
        assert "latin-1" in config.encoding_fallbacks
        assert len(config.encoding_fallbacks) > 2


class TestFileNode:
    def test_file_node_creation(self):
        node = FileNode(
            path="src/main.py",
            name="main.py",
            type="file",
            size=100,
            token_count=3
        )
        assert node.path == "src/main.py"
        assert node.name == "main.py"
        assert node.type == "file"
        assert node.size == 100
        assert node.token_count == 3
        assert node.is_file() is True
        assert node.is_directory() is False

    def test_directory_node(self):
        node = FileNode(
            path="src",
            name="src",
            type="dir"
        )
        assert node.path == "src"
        assert node.name == "src"
        assert node.type == "dir"
        assert node.is_file() is False
        assert node.is_directory() is True
        assert node.token_count is None


class TestAnalysisResult:
    def test_empty_result(self):
        result = AnalysisResult(
            repo_name="test-repo",
            branch=None,
            readme_content="",
            structure="",
            file_contents="",
            token_data={},
            total_tokens=0,
            total_files=0,
            errors=[]
        )
        assert result.repo_name == "test-repo"
        assert result.branch is None
        assert result.errors == []
        assert result.total_tokens == 0
        assert result.total_files == 0
        assert result.has_errors() is False

    def test_result_with_data(self):
        token_data = {
            "a.py": 10,
            "b.py": 15,
            "c.txt": 5
        }
        result = AnalysisResult(
            repo_name="test-repo",
            branch="main",
            readme_content="# Test Repo",
            structure="- a.py\n- b.py\n- c.txt",
            file_contents="file contents here",
            token_data=token_data,
            total_tokens=30,
            total_files=3,
            errors=[]
        )
        assert result.repo_name == "test-repo"
        assert result.branch == "main"
        assert result.total_tokens == 30
        assert result.total_files == 3
        assert len(result.token_data) == 3

    def test_result_with_errors(self):
        errors = ["Error 1", "Error 2"]
        result = AnalysisResult(
            repo_name="test-repo",
            branch=None,
            readme_content="",
            structure="",
            file_contents="",
            token_data={},
            total_tokens=0,
            total_files=0,
            errors=errors
        )
        assert len(result.errors) == 2
        assert "Error 1" in result.errors
        assert result.has_errors() is True
        assert "2 errors encountered:" in result.get_error_summary()


class TestTokenBudget:
    def test_token_budget_creation(self):
        budget = TokenBudget(
            max_tokens=4000,
            used_tokens=1500,
            reserved_tokens=500
        )
        assert budget.max_tokens == 4000
        assert budget.used_tokens == 1500
        assert budget.reserved_tokens == 500

    def test_available_tokens(self):
        budget = TokenBudget(
            max_tokens=4000,
            used_tokens=1500,
            reserved_tokens=500
        )
        assert budget.available_tokens == 2000

    def test_can_fit(self):
        budget = TokenBudget(
            max_tokens=4000,
            used_tokens=1500,
            reserved_tokens=500
        )
        assert budget.can_fit(1000) is True
        assert budget.can_fit(2500) is False

    def test_use_tokens(self):
        budget = TokenBudget(
            max_tokens=4000,
            used_tokens=1500,
            reserved_tokens=500
        )
        assert budget.use(1000) is True
        assert budget.used_tokens == 2500
        assert budget.available_tokens == 1000

    def test_cannot_use_oversized_tokens(self):
        budget = TokenBudget(
            max_tokens=4000,
            used_tokens=3500,
            reserved_tokens=500
        )
        assert budget.use(1000) is False
        assert budget.used_tokens == 3500  # Should not change

    def test_usage_percentage(self):
        budget = TokenBudget(
            max_tokens=4000,
            used_tokens=1000
        )
        assert budget.usage_percentage == 25.0

    def test_reset(self):
        budget = TokenBudget(
            max_tokens=4000,
            used_tokens=2000,
            reserved_tokens=500
        )
        budget.reset()
        assert budget.used_tokens == 0
        assert budget.available_tokens == 3500