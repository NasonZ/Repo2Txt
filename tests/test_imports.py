"""Test that all modules can be imported successfully."""

import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_core_imports():
    """Test core module imports."""
    from repo2txt.core import Config, FileNode, AnalysisResult, TokenBudget, FileAnalyzer, TokenCounter
    
    # Basic instantiation tests
    config = Config()
    assert config.max_file_size == 1024 * 1024
    
    node = FileNode(path="/test", name="test", type="file")
    assert node.is_file()
    
    token_counter = TokenCounter()
    assert hasattr(token_counter, 'count')


def test_utils_imports():
    """Test utils module imports."""
    from repo2txt.utils import FileFilter, EncodingDetector
    from repo2txt.core import Config
    
    config = Config()
    file_filter = FileFilter(config)
    assert hasattr(file_filter, 'should_exclude_directory')
    
    encoder = EncodingDetector()
    assert hasattr(encoder, 'decode_bytes')


def test_adapter_imports():
    """Test adapter module imports."""
    from repo2txt.adapters import RepositoryAdapter
    
    # Should be able to import but not instantiate (abstract class)
    assert hasattr(RepositoryAdapter, 'get_name')


if __name__ == "__main__":
    test_core_imports()
    print("* Core modules OK")
    
    test_utils_imports()
    print("* Utils modules OK")
    
    test_adapter_imports()
    print("* Adapter interfaces OK")
    
    print("\nAll imports verified.")