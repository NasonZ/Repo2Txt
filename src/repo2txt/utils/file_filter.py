"""
File filtering utilities for repo2txt.

This module provides functionality for filtering files based on various criteria
such as binary detection, file patterns, and directory exclusions.
"""

import os
import mimetypes
from pathlib import Path
from typing import Set, List, Optional

from ..core.models import Config


class FileFilter:
    """Handles file filtering logic."""
    
    def __init__(self, config: Config):
        self.config = config
        # Initialize mimetypes
        mimetypes.init()
    
    def should_exclude_directory(self, dir_name: str) -> bool:
        """
        Check if a directory should be excluded.
        
        Args:
            dir_name: Name of the directory (not full path).
            
        Returns:
            True if directory should be excluded, False otherwise.
        """
        return dir_name in self.config.excluded_dirs
    
    def should_skip_file(self, file_path: str) -> bool:
        """
        Check if a file should be skipped based on patterns.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            True if file should be skipped, False otherwise.
        """
        path = Path(file_path)
        file_name = path.name
        
        # Check skip patterns
        for pattern in self.config.skip_patterns:
            if pattern.startswith('*'):
                # Wildcard pattern (e.g., *.min.js)
                if file_name.endswith(pattern[1:]):
                    return True
            elif file_name == pattern:
                # Exact match
                return True
        
        return False
    
    def is_binary_extension(self, file_path: str) -> bool:
        """
        Check if file has a known binary extension.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            True if file has binary extension, False otherwise.
        """
        path = Path(file_path)
        return path.suffix.lower() in self.config.binary_extensions
    
    def guess_is_binary(self, file_path: str) -> bool:
        """
        Guess if a file is binary based on its MIME type.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            True if file is likely binary, False otherwise.
        """
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type:
            return not mime_type.startswith('text/')
        return False
    
    def is_hidden_file(self, file_path: str) -> bool:
        """
        Check if a file is hidden (starts with dot).
        
        Args:
            file_path: Path to the file.
            
        Returns:
            True if file is hidden, False otherwise.
        """
        path = Path(file_path)
        # Check if any part of the path starts with a dot
        for part in path.parts:
            if part.startswith('.') and part not in {'.github', '.gitlab'}:
                return True
        return False
    
    def filter_files(self, file_paths: List[str]) -> List[str]:
        """
        Filter a list of file paths based on all criteria.
        
        Args:
            file_paths: List of file paths to filter.
            
        Returns:
            Filtered list of file paths.
        """
        filtered = []
        for file_path in file_paths:
            if (not self.should_skip_file(file_path) and
                not self.is_binary_extension(file_path) and
                not self.is_hidden_file(file_path)):
                filtered.append(file_path)
        return filtered
    
    def get_excluded_reason(self, file_path: str) -> Optional[str]:
        """
        Get the reason why a file would be excluded.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            Reason string if file would be excluded, None otherwise.
        """
        if self.is_hidden_file(file_path):
            return "Hidden file"
        if self.is_binary_extension(file_path):
            return "Binary file extension"
        if self.should_skip_file(file_path):
            return "Matches skip pattern"
        if self.guess_is_binary(file_path):
            return "Likely binary (MIME type)"
        return None