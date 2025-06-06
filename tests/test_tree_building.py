"""Tests for tree building and path handling utilities."""

import pytest
from repo2txt.utils import PathUtils, FileTreeBuilder
from repo2txt.core.models import FileNode


class TestPathUtils:
    """Test path normalization and manipulation utilities."""
    
    def test_normalize_path_forward_slashes(self):
        """Unix-style paths should remain unchanged."""
        path = "src/main.py"
        result = PathUtils.normalize_path(path)
        assert result == "src/main.py"
    
    def test_normalize_path_backslashes(self):
        """Windows-style paths should be normalized to forward slashes."""
        path = "src\\utils\\helper.py"
        result = PathUtils.normalize_path(path)
        assert result == "src/utils/helper.py"
    
    def test_normalize_path_mixed_separators(self):
        """Mixed separators should be normalized to forward slashes."""
        path = "src/utils\\subfolder/file.py"
        result = PathUtils.normalize_path(path)
        assert result == "src/utils/subfolder/file.py"
    
    def test_normalize_path_empty_string(self):
        """Empty string should remain empty."""
        result = PathUtils.normalize_path("")
        assert result == ""
    
    def test_normalize_and_split_simple(self):
        """Simple path should split into components."""
        path = "src/main.py"
        result = PathUtils.normalize_and_split(path)
        assert result == ["src", "main.py"]
    
    def test_normalize_and_split_nested(self):
        """Nested path should split into all components."""
        path = "src/utils/subfolder/helper.py"
        result = PathUtils.normalize_and_split(path)
        assert result == ["src", "utils", "subfolder", "helper.py"]
    
    def test_normalize_and_split_backslashes(self):
        """Windows paths should be normalized then split."""
        path = "src\\utils\\helper.py"
        result = PathUtils.normalize_and_split(path)
        assert result == ["src", "utils", "helper.py"]
    
    def test_normalize_and_split_single_file(self):
        """Single filename should return single-item list."""
        path = "README.md"
        result = PathUtils.normalize_and_split(path)
        assert result == ["README.md"]
    
    def test_join_path_components_simple(self):
        """Simple components should join with forward slashes."""
        components = ["src", "main.py"]
        result = PathUtils.join_path_components(components)
        assert result == "src/main.py"
    
    def test_join_path_components_nested(self):
        """Nested components should join with forward slashes."""
        components = ["src", "utils", "subfolder", "helper.py"]
        result = PathUtils.join_path_components(components)
        assert result == "src/utils/subfolder/helper.py"
    
    def test_join_path_components_empty_list(self):
        """Empty list should return empty string."""
        result = PathUtils.join_path_components([])
        assert result == ""
    
    def test_round_trip_path_operations(self):
        """Path splitting and joining should be reversible."""
        original_paths = [
            "src/main.py",
            "src/utils/helper.py", 
            "tests/test_main.py",
            "docs/api/reference.md"
        ]
        
        for path in original_paths:
            components = PathUtils.normalize_and_split(path)
            reconstructed = PathUtils.join_path_components(components)
            assert reconstructed == path


class TestFileTreeBuilder:
    """Test file tree construction and manipulation utilities."""
    
    def test_from_paths_simple_files(self):
        """Should build tree with files at root level."""
        paths = ["README.md", "setup.py"]
        tree = FileTreeBuilder.from_paths("test-repo", paths)
        
        assert tree.name == "test-repo"
        assert tree.type == "dir"
        assert len(tree.children) == 2
        
        child_names = [child.name for child in tree.children]
        assert "README.md" in child_names
        assert "setup.py" in child_names
        
        for child in tree.children:
            assert child.type == "file"
    
    def test_from_paths_nested_structure(self):
        """Should build proper hierarchical directory structure."""
        paths = [
            "src/main.py",
            "src/utils/helper.py",
            "tests/test_main.py",
            "README.md"
        ]
        tree = FileTreeBuilder.from_paths("test-repo", paths)
        
        assert tree.name == "test-repo"
        assert len(tree.children) == 3  # src, tests, README.md
        
        # Find src directory
        src_dir = next(child for child in tree.children if child.name == "src")
        assert src_dir.type == "dir"
        assert len(src_dir.children) == 2  # main.py, utils
        
        # Find utils subdirectory
        utils_dir = next(child for child in src_dir.children if child.name == "utils")
        assert utils_dir.type == "dir"
        assert len(utils_dir.children) == 1  # helper.py
        
        # Check file in nested directory
        helper_file = utils_dir.children[0]
        assert helper_file.name == "helper.py"
        assert helper_file.type == "file"
        assert helper_file.path == "src/utils/helper.py"
    
    def test_from_paths_with_token_data(self):
        """Should incorporate token counts into file nodes."""
        paths = ["src/main.py", "src/utils/helper.py"]
        token_data = {
            "src/main.py": 150,
            "src/utils/helper.py": 75
        }
        
        tree = FileTreeBuilder.from_paths("test-repo", paths, token_data)
        
        # Find main.py
        src_dir = next(child for child in tree.children if child.name == "src")
        main_file = next(child for child in src_dir.children if child.name == "main.py")
        assert main_file.token_count == 150
        
        # Find helper.py
        utils_dir = next(child for child in src_dir.children if child.name == "utils")
        helper_file = utils_dir.children[0]
        assert helper_file.token_count == 75
    
    def test_from_paths_without_token_data(self):
        """Should work without token data (default to 0)."""
        paths = ["src/main.py"]
        tree = FileTreeBuilder.from_paths("test-repo", paths)
        
        src_dir = tree.children[0]
        main_file = src_dir.children[0]
        assert main_file.token_count == 0
    
    def test_from_paths_empty_list(self):
        """Should create empty root directory for empty file list."""
        tree = FileTreeBuilder.from_paths("empty-repo", [])
        
        assert tree.name == "empty-repo"
        assert tree.type == "dir"
        assert len(tree.children) == 0
    
    def test_from_paths_duplicate_directories(self):
        """Should not create duplicate directory nodes."""
        paths = [
            "src/file1.py",
            "src/file2.py",
            "src/utils/helper1.py",
            "src/utils/helper2.py"
        ]
        tree = FileTreeBuilder.from_paths("test-repo", paths)
        
        # Should have only one src directory
        src_dirs = [child for child in tree.children if child.name == "src"]
        assert len(src_dirs) == 1
        
        src_dir = src_dirs[0]
        assert len(src_dir.children) == 3  # file1.py, file2.py, utils
        
        # Should have only one utils directory
        utils_dirs = [child for child in src_dir.children if child.name == "utils"]
        assert len(utils_dirs) == 1
        
        utils_dir = utils_dirs[0]
        assert len(utils_dir.children) == 2  # helper1.py, helper2.py
    
    def test_paths_from_tree(self):
        """Should extract all file paths from a tree structure."""
        # Create a test tree
        paths = [
            "src/main.py",
            "src/utils/helper.py", 
            "tests/test_main.py",
            "README.md"
        ]
        tree = FileTreeBuilder.from_paths("test-repo", paths)
        
        # Extract paths back
        extracted_paths = FileTreeBuilder.paths_from_tree(tree)
        
        assert len(extracted_paths) == 4
        assert "src/main.py" in extracted_paths
        assert "src/utils/helper.py" in extracted_paths
        assert "tests/test_main.py" in extracted_paths
        assert "README.md" in extracted_paths
    
    def test_paths_from_tree_empty(self):
        """Should return empty set for tree with no files."""
        tree = FileTreeBuilder.from_paths("empty-repo", [])
        extracted_paths = FileTreeBuilder.paths_from_tree(tree)
        assert len(extracted_paths) == 0
    
    def test_calculate_directory_tokens(self):
        """Should aggregate token counts bottom-up in directory tree."""
        paths = [
            "src/main.py",
            "src/utils/helper.py",
            "tests/test_main.py"
        ]
        token_data = {
            "src/main.py": 100,
            "src/utils/helper.py": 50,
            "tests/test_main.py": 25
        }
        
        tree = FileTreeBuilder.from_paths("test-repo", paths, token_data)
        FileTreeBuilder.calculate_directory_tokens(tree)
        
        # Root should have total of all tokens
        assert tree.total_tokens == 175
        
        # src directory should have 150 tokens (100 + 50)
        src_dir = next(child for child in tree.children if child.name == "src")
        assert src_dir.total_tokens == 150
        
        # utils directory should have 50 tokens
        utils_dir = next(child for child in src_dir.children if child.name == "utils") 
        assert utils_dir.total_tokens == 50
        
        # tests directory should have 25 tokens
        tests_dir = next(child for child in tree.children if child.name == "tests")
        assert tests_dir.total_tokens == 25
    
    def test_round_trip_tree_operations(self):
        """Should maintain consistency in paths → tree → paths operations."""
        original_paths = [
            "src/core/analyzer.py",
            "src/utils/path_utils.py",
            "src/utils/tree_builder.py", 
            "tests/test_analyzer.py",
            "README.md"
        ]
        
        # Build tree from paths
        tree = FileTreeBuilder.from_paths("test-repo", original_paths)
        
        # Extract paths back from tree
        extracted_paths = FileTreeBuilder.paths_from_tree(tree)
        
        # Should be identical
        assert extracted_paths == set(original_paths)
    
    def test_cross_platform_path_handling(self):
        """Should handle mixed Windows/Unix paths correctly."""
        mixed_paths = [
            "src/main.py",           # Unix style
            "src\\utils\\helper.py", # Windows style  
            "tests/test_main.py"     # Unix style
        ]
        
        tree = FileTreeBuilder.from_paths("test-repo", mixed_paths)
        
        # Extract normalized paths
        extracted_paths = FileTreeBuilder.paths_from_tree(tree)
        
        # All paths should be normalized to forward slashes
        expected_paths = {
            "src/main.py",
            "src/utils/helper.py",  # Normalized from Windows style
            "tests/test_main.py"
        }
        assert extracted_paths == expected_paths


class TestTreeBuildingIntegration:
    """Integration tests for tree building with FileNode models."""
    
    def test_filenode_compatibility(self):
        """Should create FileNode objects compatible with existing models."""
        paths = ["src/main.py"]
        tree = FileTreeBuilder.from_paths("test-repo", paths)
        
        # Verify FileNode attributes
        assert hasattr(tree, 'path')
        assert hasattr(tree, 'name') 
        assert hasattr(tree, 'type')
        assert hasattr(tree, 'children')
        assert hasattr(tree, 'token_count')
        assert hasattr(tree, 'total_tokens')
        
        # Verify types
        assert isinstance(tree.children, list)
        assert tree.type in ["file", "dir"]
    
    def test_addresses_original_bug(self):
        """Should fix the original flat structure bug from analyzer.py."""
        # This replicates the scenario that was failing before our fix
        selected_paths = [
            "src/repo2txt/core/analyzer.py",
            "src/repo2txt/ai/command_handler.py", 
            "tests/test_analyzer.py"
        ]
        
        tree = FileTreeBuilder.from_paths("RepoToTextForLLMs", selected_paths)
        
        # Verify hierarchical structure (not flat)
        assert tree.name == "RepoToTextForLLMs"
        assert len(tree.children) == 2  # src, tests (not 3 flat files)
        
        # Verify src directory structure
        src_dir = next(child for child in tree.children if child.name == "src")
        assert src_dir.type == "dir"
        
        repo2txt_dir = src_dir.children[0]
        assert repo2txt_dir.name == "repo2txt"
        assert repo2txt_dir.type == "dir"
        
        # Files should be in their proper directories, not at root
        root_files = [child for child in tree.children if child.type == "file"]
        assert len(root_files) == 0  # No files should be at root level