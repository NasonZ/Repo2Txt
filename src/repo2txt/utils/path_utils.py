"""Path normalization utilities for cross-platform compatibility."""

from typing import List


class PathUtils:
    """Utilities for consistent path handling across platforms."""
    
    @staticmethod
    def normalize_path(path: str) -> str:
        """
        Normalize path separators to forward slashes.
        
        Args:
            path: File path with potentially mixed separators
            
        Returns:
            Path with forward slashes only
        """
        return path.replace('\\\\', '/').replace('\\', '/')
    
    @staticmethod
    def normalize_and_split(path: str) -> List[str]:
        """
        Normalize path and split into components.
        
        Args:
            path: File path to split
            
        Returns:
            List of path components
        """
        return PathUtils.normalize_path(path).split('/')
    
    @staticmethod
    def join_path_components(components: List[str]) -> str:
        """
        Join path components with forward slashes.
        
        Args:
            components: List of path components
            
        Returns:
            Joined path with forward slashes
        """
        return '/'.join(components)