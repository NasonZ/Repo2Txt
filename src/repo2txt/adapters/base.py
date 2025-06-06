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
        
        # Resource protection limits
        self.max_files = config.max_files
        self.max_total_size = config.max_total_size
        self.total_size_processed = 0
        self.files_processed = 0
    
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
    def build_file_tree(self) -> FileNode:
        """
        Build a hierarchical tree structure of the repository.
        
        Returns:
            Root FileNode with children representing the file tree structure.
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
        
        # resource protection limits
        if len(range_str) > 1000:
            self.errors.append("Range string too long (max 1000 chars)")
            return []
        
        ranges = []
        try:
            for part in range_str.split(',')[:100]: 
                part = part.strip()
                if '-' in part:
                    parts = part.split('-', 1)  # Only split once
                    if len(parts) == 2:
                        start, end = map(int, parts)
                        # check ranges
                        if 0 <= start <= 10000 and 0 <= end <= 10000 and start <= end:
                            #  prevent memory exhaustion
                            ranges.extend(range(start, min(end + 1, start + 1000)))
                else:
                    num = int(part)
                    if 0 <= num <= 10000:  # Reasonable bounds
                        ranges.append(num)
            return sorted(set(ranges))[:1000]  # restrict final result size
        except (ValueError, OverflowError):
            return []
    
    def _format_file_content(self, file_path: str, content: Optional[str], error: Optional[str]) -> str:
        """Format file content based on output format setting. Shared by all adapters."""
        if self.config.output_format == 'xml':
            if content:
                return f'<file path="{file_path}">\n{content}\n</file>\n'
            else:
                return f'<file path="{file_path}" error="{error}" />\n'
        else:  # markdown
            if content:
                return f'```{file_path}\n{content}\n```\n'
            else:
                return f'```{file_path}\n# Error: {error}\n```\n'
    
    def _sanitize_error(self, error: str, sensitive_data: Optional[List[str]] = None) -> str:
        """Remove sensitive data from error messages. Used by all adapters."""
        if not sensitive_data:
            return error
        
        sanitized = error
        for sensitive in sensitive_data:
            if sensitive:
                sanitized = sanitized.replace(str(sensitive), "[REDACTED]")
        return sanitized
    
    def _count_tokens_safe(self, file_path: str, content: str = None) -> int:
        """Safe token counting with resource tracking. Used by all adapters."""
        if not self.config.enable_token_counting:
            return 0
        
        # Check resource limits
        if self.files_processed >= self.max_files:
            return 0
        
        if content is None:
            # Try to get content
            content_result, error = self.get_file_content(file_path) 
            if not content_result:
                return 0
            content = content_result
        
        # Track resources
        self.files_processed += 1
        try:
            size = len(content.encode('utf-8'))
            self.total_size_processed += size
            if self.total_size_processed > self.max_total_size:
                return 0
        except Exception as e:
            # Size calculation failed, continue with token counting
            # This is not critical - just resource tracking
            pass
        
        return self.file_analyzer.count_tokens(content)