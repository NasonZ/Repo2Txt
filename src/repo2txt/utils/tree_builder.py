"""FileNode tree building utilities."""

from typing import Dict, List, Optional, Set
from ..core.models import FileNode
from .path_utils import PathUtils


class FileTreeBuilder:
    """Utilities for building FileNode trees from various inputs."""
    
    @staticmethod
    def from_paths(
        repo_name: str,
        file_paths: List[str], 
        token_data: Optional[Dict[str, int]] = None
    ) -> FileNode:
        """
        Build hierarchical FileNode tree from flat list of file paths.
        
        This consolidates the tree building logic that was duplicated across
        analyzer.py and command_handler.py.
        
        Args:
            repo_name: Name of the repository (root node name)
            file_paths: List of file paths to include in tree
            token_data: Optional mapping of file paths to token counts
            
        Returns:
            Root FileNode with hierarchical structure
        """
        if token_data is None:
            token_data = {}
            
        # Create root node
        root_node = FileNode(
            path=repo_name,
            name=repo_name,
            type="dir"
        )
        
        # Build directory structure
        for file_path in file_paths:
            # Split path into components (handle both / and \ separators)
            parts = PathUtils.normalize_and_split(file_path)
            current_node = root_node
            
            # Create directory nodes as needed
            for i, part in enumerate(parts[:-1]):
                # Check if this directory already exists
                found = False
                for child in current_node.children:
                    if child.name == part and child.type == "dir":
                        current_node = child
                        found = True
                        break
                
                # Create directory node if it doesn't exist
                if not found:
                    dir_path = PathUtils.join_path_components(parts[:i+1])
                    dir_node = FileNode(
                        path=dir_path,
                        name=part,
                        type="dir"
                    )
                    current_node.children.append(dir_node)
                    current_node = dir_node
            
            # Add the file node
            file_node = FileNode(
                path=file_path,
                name=parts[-1],  # Just the filename
                type="file",
                token_count=token_data.get(file_path, 0)
            )
            current_node.children.append(file_node)
        
        return root_node
    
    @staticmethod
    def paths_from_tree(tree: FileNode) -> Set[str]:
        """
        Extract all file paths from a FileNode tree.
        
        Args:
            tree: Root FileNode to traverse
            
        Returns:
            Set of all file paths in the tree
        """
        paths = set()
        
        def _collect_paths(node: FileNode) -> None:
            if node.type == "file":
                paths.add(node.path)
            for child in node.children:
                _collect_paths(child)
        
        _collect_paths(tree)
        return paths
    
    @staticmethod
    def calculate_directory_tokens(tree: FileNode) -> None:
        """
        Calculate total token counts for directory nodes bottom-up.
        
        Modifies the tree in-place by setting total_tokens on directory nodes.
        
        Args:
            tree: Root FileNode to process
        """
        def _calculate_tokens(node: FileNode) -> int:
            if node.type == "file":
                return node.token_count or 0
            
            total = 0
            for child in node.children:
                total += _calculate_tokens(child)
            
            node.total_tokens = total
            return total
        
        _calculate_tokens(tree)