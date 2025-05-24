"""Repository adapters for different source types."""
import os
from typing import Union

from ..core.models import Config
from .base import RepositoryAdapter
from .github import GitHubAdapter
from .local import LocalAdapter


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
    # Check if it's a local directory
    if os.path.isdir(repo_url_or_path):
        return LocalAdapter(repo_url_or_path, config)
    
    # Check if it's a GitHub URL
    if repo_url_or_path.startswith('https://github.com/') or repo_url_or_path.startswith('github.com/'):
        return GitHubAdapter(repo_url_or_path, config)
    
    # Check if path exists (might be a file instead of directory)
    if os.path.exists(repo_url_or_path):
        raise ValueError(f"Path exists but is not a directory: {repo_url_or_path}")
    
    # Assume it might be a GitHub shorthand (owner/repo)
    if '/' in repo_url_or_path and not os.sep in repo_url_or_path:
        github_url = f"https://github.com/{repo_url_or_path}"
        return GitHubAdapter(github_url, config)
    
    raise ValueError(
        f"Invalid input: {repo_url_or_path}\n"
        "Expected: GitHub URL (https://github.com/owner/repo) or local directory path"
    )


__all__ = ['RepositoryAdapter', 'GitHubAdapter', 'LocalAdapter', 'create_adapter']