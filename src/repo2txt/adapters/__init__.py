"""Repository adapters for different source types."""
import os
import sys
from pathlib import Path
from typing import Union

from ..core.models import Config
from .base import RepositoryAdapter
from .github import GitHubAdapter
from .local import LocalAdapter


def _normalize_path(path_str: str) -> str:
    """
    Normalize a path string for cross-platform compatibility.
    
    Args:
        path_str: Input path string
        
    Returns:
        Normalized path string
    """
    # Convert to Path object for normalization
    path = Path(path_str)
    
    # On WSL, try to convert Windows paths to WSL mount paths
    if sys.platform.startswith('linux') and path_str.startswith(('C:', 'D:', 'E:', 'F:', 'G:')):
        # Convert Windows drive letters to WSL mount points
        drive_letter = path_str[0].lower()
        remaining_path = path_str[2:].replace('\\', '/')  # Remove 'C:' and normalize slashes
        wsl_path = f"/mnt/{drive_letter}{remaining_path}"
        return wsl_path
    
    # Return normalized path
    return str(path.resolve())


def _is_directory_path(path_str: str) -> bool:
    """
    Check if a path string represents a directory, with cross-platform handling.
    
    Args:
        path_str: Path string to check
        
    Returns:
        True if the path is a valid directory
    """
    # First try the original path
    if os.path.isdir(path_str):
        return True
    
    # Try normalized path
    normalized = _normalize_path(path_str)
    if os.path.isdir(normalized):
        return True
    
    # For potential paths that look like directories but don't exist yet,
    # check if it has directory-like characteristics
    # (This is more permissive to handle edge cases)
    path = Path(path_str)
    
    # If it has no suffix and looks like a path, assume it could be a directory
    if not path.suffix and ('/' in path_str or '\\' in path_str or os.sep in path_str):
        # Check if parent directory exists
        try:
            parent = path.parent
            if parent.exists():
                return True
        except (OSError, ValueError):
            pass
    
    return False


def _path_exists(path_str: str) -> bool:
    """
    Check if a path exists, with cross-platform handling.
    
    Args:
        path_str: Path string to check
        
    Returns:
        True if the path exists
    """
    # First try the original path
    if os.path.exists(path_str):
        return True
    
    # Try normalized path
    normalized = _normalize_path(path_str)
    if os.path.exists(normalized):
        return True
    
    return False


def create_adapter(repo_url_or_path: str, config: Config) -> RepositoryAdapter:
    """
    Create appropriate repository adapter based on input.
    
    Args:
        repo_url_or_path: GitHub URL or local directory path
        config: Configuration object
        
    Returns:
        Appropriate RepositoryAdapter instance
        
    Raises:
        ValueError: If input format is invalid
    """
    # Check if it's a GitHub URL first (most specific)
    if repo_url_or_path.startswith('https://github.com/') or repo_url_or_path.startswith('github.com/'):
        return GitHubAdapter(repo_url_or_path, config)
    
    # Check if it looks like a local path (contains path separators or drive letters)
    is_path_like = ('/' in repo_url_or_path or 
                   '\\' in repo_url_or_path or 
                   os.sep in repo_url_or_path or
                   (len(repo_url_or_path) > 2 and repo_url_or_path[1] == ':'))  # Windows drive letter
    
    if is_path_like:
        # Try to handle as local path
        if _is_directory_path(repo_url_or_path):
            # Use normalized path for the adapter
            normalized_path = _normalize_path(repo_url_or_path)
            if os.path.isdir(normalized_path):
                return LocalAdapter(normalized_path, config)
            else:
                return LocalAdapter(repo_url_or_path, config)  # Let LocalAdapter handle the error
        
        # Check if path exists but is not a directory
        if _path_exists(repo_url_or_path):
            raise ValueError(f"Path exists but is not a directory: {repo_url_or_path}")
        
        # Path doesn't exist - still try LocalAdapter (it will provide a better error)
        return LocalAdapter(repo_url_or_path, config)
    
    # Check if it might be a GitHub shorthand (owner/repo)
    if '/' in repo_url_or_path and len(repo_url_or_path.split('/')) == 2:
        # Make sure it doesn't look like a file path
        parts = repo_url_or_path.split('/')
        if not any('.' in part for part in parts):  # Avoid treating file paths as GitHub repos
            github_url = f"https://github.com/{repo_url_or_path}"
            return GitHubAdapter(github_url, config)
    
    raise ValueError(
        f"Invalid input: {repo_url_or_path}\n"
        "Expected: GitHub URL (https://github.com/owner/repo) or local directory path"
    )


__all__ = ['RepositoryAdapter', 'GitHubAdapter', 'LocalAdapter', 'create_adapter']