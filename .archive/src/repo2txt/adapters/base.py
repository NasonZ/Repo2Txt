"""
Base repository adapter interface.

This module defines the abstract interface that all repository adapters
must implement, ensuring consistent behavior across different repository types.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Set, Dict, Any

from ..core.models import Config, FileNode
from ..core.file_analyzer import FileAnalyzer


class RepositoryAdapter(ABC):
    """
    Abstract base class for repository adapters.
    
    This interface defines the contract that all repository adapters
    (GitHub, GitLab, local filesystem, etc.) must implement.
    """
    
    def __init__(self, config: Config):
        """Initialize adapter with configuration."""
        self.config = config
        self.file_analyzer = FileAnalyzer(config)
        self.errors = []
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the repository name."""
        pass
    
    @abstractmethod
    def get_readme_content(self) -> str:
        """Get the README content."""
        pass
    
    @abstractmethod
    def list_contents(self, path: str = "") -> List[Tuple[str, str, int]]:
        """
        List contents of a directory.
        
        Args:
            path: The directory path to list (empty string for root).
            
        Returns:
            List of tuples (name, type, size).
        """
        pass
    
    @abstractmethod
    def traverse_interactive(self) -> Tuple[str, Set[str], Dict[str, int]]:
        """
        Interactively traverse the repository.
        
        Returns:
            Tuple of (structure, selected_paths, token_data).
        """
        pass
    
    @abstractmethod
    def get_file_content(self, file_path: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Get the content of a file.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            Tuple of (content, error_message).
            If successful, content is the file text and error_message is None.
            If failed, content is None and error_message describes the issue.
        """
        pass
    
    @abstractmethod
    def get_file_contents(self, selected_files: List[str]) -> Tuple[str, Dict[str, int]]:
        """
        Get contents of multiple files.
        
        Args:
            selected_files: List of file paths to read.
            
        Returns:
            Tuple of (formatted_contents, token_data).
        """
        pass
    
    @abstractmethod
    def build_file_tree(self) -> str:
        """
        Build a text representation of the repository file tree.
        
        Returns:
            String representation of the file tree structure.
        """
        pass
    
    @abstractmethod 
    def get_file_list(self) -> List[str]:
        """
        Get a list of all files in the repository.
        
        Returns:
            List of file paths relative to repository root.
        """
        pass
    
    def parse_range(self, range_str: str) -> List[int]:
        """Parse a range string like '1-3,5,7-9' into a list of integers."""
        if not range_str.strip():
            return []
        
        ranges = []
        try:
            for part in range_str.split(','):
                part = part.strip()
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    ranges.extend(range(start, end + 1))
                else:
                    ranges.append(int(part))
            return sorted(set(ranges))  # Remove duplicates and sort
        except ValueError:
            return []