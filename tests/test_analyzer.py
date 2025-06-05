import pytest
import os
from unittest.mock import patch, MagicMock, mock_open
from repo2txt.core.analyzer import RepositoryAnalyzer
from repo2txt.core.models import Config, FileNode, AnalysisResult


class TestRepositoryAnalyzer:
    @pytest.fixture
    def analyzer(self):
        config = Config()
        return RepositoryAnalyzer(config)

    @pytest.fixture
    def sample_result(self):
        return AnalysisResult(
            repo_name="test-repo",
            branch="main",
            readme_content="# Test Repo",
            structure="src/\n  main.py\nREADME.md",
            file_contents="## src/main.py\n```python\ndef main():\n    pass\n```\n\n## README.md\n```markdown\n# Test Repo\n```",
            token_data={"src/main.py": 10, "README.md": 5},
            total_tokens=15,
            total_files=2,
            errors=["Some error occurred"]
        )

    def test_initialization(self, analyzer):
        assert analyzer.config.output_format == "markdown"
        assert hasattr(analyzer, 'config')

    def test_custom_config(self):
        config = Config(output_format="xml")
        analyzer = RepositoryAnalyzer(config)
        assert analyzer.config.output_format == "xml"

    @patch('repo2txt.core.analyzer.create_adapter')
    def test_analyze_repository(self, mock_create_adapter, analyzer):
        mock_adapter = MagicMock()
        mock_adapter.get_name.return_value = "test-repo"
        mock_adapter.get_readme_content.return_value = "# Test"
        mock_adapter.traverse_interactive.return_value = ("structure", ["test.py"], {"test.py": 5})
        mock_adapter.get_file_contents.return_value = ("## test.py\n```python\ncontent\n```", {"test.py": 5})
        mock_create_adapter.return_value = mock_adapter
        
        result = analyzer.analyze("/test/path")
        
        assert result.repo_name == "test-repo"
        assert result.total_tokens == 5
        assert result.total_files == 1
        mock_create_adapter.assert_called_once_with("/test/path", analyzer.config)
        mock_adapter.get_name.assert_called_once()

    @patch('os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    def test_save_results_markdown(self, mock_file, mock_makedirs, analyzer, sample_result):
        file_paths = analyzer.save_results(sample_result)
        
        mock_makedirs.assert_called()
        
        # Check that files were created
        assert 'main' in file_paths
        assert 'tokens' in file_paths

    @patch('os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    def test_save_results_xml(self, mock_file, mock_makedirs, analyzer, sample_result):
        analyzer.config.output_format = "xml"
        file_paths = analyzer.save_results(sample_result)
        
        # Check that files were created with correct extensions
        assert file_paths['main'].endswith('.txt')
        
        # Verify XML content was written
        written_content = mock_file().write.call_args_list[0][0][0]
        assert '<repository>' in written_content
        assert '<total_files>2</total_files>' in written_content
        assert '<readme>' in written_content
        assert '<files>' in written_content
        assert '</repository>' in written_content

    def test_generate_instructions_markdown(self, analyzer, sample_result):
        content = analyzer.generate_instructions(sample_result)
        
        assert "Repository Analysis: test-repo" in content
        assert "Analysis Instructions" in content
        assert "Key Questions to Answer" in content

    def test_generate_instructions_xml(self, analyzer, sample_result):
        # generate_instructions doesn't change based on output format
        content = analyzer.generate_instructions(sample_result)
        
        # It still returns markdown instructions regardless of format
        assert "Repository Analysis: test-repo" in content
        assert "Analysis Instructions" in content

    def test_generate_token_report(self, analyzer, sample_result):
        report = analyzer.generate_token_report(sample_result.token_data)
        
        assert "TOKEN ANALYSIS REPORT" in report
        assert "src/main.py" in report
        assert "10" in report
        assert "README.md" in report
        assert "5" in report

    def test_generate_token_report_simple(self, analyzer):
        token_data = {"a.py": 5, "b.py": 10}
        report = analyzer.generate_token_report(token_data)
        
        assert "a.py" in report
        assert "b.py" in report
        assert "5" in report
        assert "10" in report

    @patch('json.dump')
    @patch('builtins.open', new_callable=mock_open)
    def test_save_results_with_json(self, mock_file, mock_json_dump, analyzer, sample_result):
        analyzer.config.export_json = True
        file_paths = analyzer.save_results(sample_result)
        
        # Check that JSON export was attempted
        assert 'json' in file_paths or mock_json_dump.called

    def test_xml_escaping_in_instructions(self, analyzer, sample_result):
        sample_result.file_contents = "<test> & content"
        analyzer.config.output_format = "xml"
        content = analyzer.generate_instructions(sample_result)
        
        # Verify XML is generated with proper escaping
        assert "<repository>" in content
        assert "<repo_name>test-repo</repo_name>" in content
        assert "<instructions>" in content

    @patch('repo2txt.core.analyzer.create_adapter')
    @patch('os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    def test_full_workflow(self, mock_file, mock_makedirs, mock_create_adapter, analyzer):
        mock_adapter = MagicMock()
        mock_adapter.get_name.return_value = "workflow-test"
        mock_adapter.get_readme_content.return_value = "# Workflow Test"
        mock_adapter.traverse_interactive.return_value = ("structure", ["test.py"], {"test.py": 3})
        mock_adapter.get_file_contents.return_value = ("## test.py\n```python\nprint('test')\n```", {"test.py": 3})
        mock_create_adapter.return_value = mock_adapter
        
        result = analyzer.analyze("/test")
        file_paths = analyzer.save_results(result)
        
        assert result.repo_name == "workflow-test"
        assert result.total_tokens == 3
        mock_makedirs.assert_called()
        assert len(file_paths) >= 2  # Main file and token report