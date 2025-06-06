"""
Core data models for repo2txt.

This module contains the fundamental data structures used throughout
the application for configuration, file representation, and analysis results.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Literal, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class Config:
    """Configuration settings for repo2txt."""
    
    github_token: str = field(default_factory=lambda: os.getenv('GITHUB_TOKEN', ''))
    
    # Directories to exclude
    excluded_dirs: Set[str] = field(default_factory=lambda: {
        '__pycache__', '.git', '.hg', '.svn', '.idea', '.vscode', 
        'node_modules', '.pytest_cache', '.mypy_cache', '.tox',
        'venv', 'env', '.env', 'virtualenv', '.virtualenv',
        'bower_components', 'vendor', 'coverage', '.coverage',
        '.sass-cache', '.cache', 'dist', 'build', '.next',
        '.nuxt', '.output', '.parcel-cache', 'out', 'datasets'
    })
    
    # Common binary/non-text extensions
    binary_extensions: Set[str] = field(default_factory=lambda: {
        # Executables & Libraries
        '.exe', '.dll', '.so', '.a', '.lib', '.dylib', '.o', '.obj',
        # Archives
        '.zip', '.tar', '.gz', '.bz2', '.7z', '.rar', '.jar', '.war',
        # Media
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.ico', '.svg', '.webp',
        '.mp3', '.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm',
        '.wav', '.flac', '.ogg', '.m4a', '.aac',
        # Documents
        '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
        # Data
        '.db', '.sqlite', '.mdb', '.accdb',
        # Other
        '.pyc', '.pyo', '.pyd', '.whl', '.egg-info', '.dist-info',
        '.class', '.jar', '.dex', '.apk', '.ipa',
        '.DS_Store', '.localized', '.Spotlight-V100', '.Trashes'
    })
    
    # Large files that might be text but should be skipped
    skip_patterns: Set[str] = field(default_factory=lambda: {
        'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml',
        'poetry.lock', 'Pipfile.lock', 'composer.lock',
        '*.min.js', '*.min.css', '*.map'
    })
    
    # Encoding fallbacks
    encoding_fallbacks: List[str] = field(default_factory=lambda: [
        'utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1'
    ])
    
    max_file_size: int = 1024 * 1024  # 1MB default
    enable_token_counting: bool = True
    token_encoder: str = "cl100k_base"
    output_format: Literal['xml', 'markdown'] = 'markdown'  # Output format for file contents
    
    # Security limits
    max_files: int = 5000  # Maximum files per repository
    max_total_size: int = 1024 * 1024 * 1024  # 1GB total size limit
    
    # AI Selection settings
    ai_select: bool = False  # Enable AI-assisted file selection
    ai_query: Optional[str] = None  # Query for AI file selection
    token_budget: int = 100000  # Token budget for AI selection
    export_json: bool = False  # Export results as JSON
    debug: bool = False  # Enable debug mode for AI selection (shows tool panels, system prompts)
    prompt_style: str = 'standard'  # Prompt style for AI selection


@dataclass
class FileNode:
    """Represents a file or directory in the repository."""
    
    path: str
    name: str
    type: str  # 'file' or 'dir'
    size: Optional[int] = None
    token_count: Optional[int] = None
    error: Optional[str] = None
    children: List['FileNode'] = field(default_factory=list)
    total_tokens: int = 0  # Total tokens including children
    
    def is_file(self) -> bool:
        """Check if this node represents a file."""
        return self.type == 'file'
    
    def is_directory(self) -> bool:
        """Check if this node represents a directory."""
        return self.type == 'dir'
    
    @property
    def is_dir(self) -> bool:
        """Alias for is_directory for compatibility."""
        return self.is_directory()


@dataclass
class AnalysisResult:
    """Result of repository analysis."""
    
    # Required fields first
    repo_path: str
    repo_name: str
    file_tree: FileNode               # Hierarchical structure
    file_paths: List[str]             # Flat list for iteration
    total_files: int                  # Count
    
    # Optional fields with defaults
    branch: Optional[str] = None
    readme_content: Optional[str] = None
    total_tokens: int = 0
    file_contents: str = ""           # Generated file contents
    token_data: Dict[str, int] = field(default_factory=dict)  # File path -> token count
    errors: List[str] = field(default_factory=list)
    file_list: List[Dict[str, Any]] = field(default_factory=list)  # [{'path': str, 'tokens': int}, ...]
    
    # Computed properties
    @property
    def file_tree_string(self) -> str:
        """Generate string representation from FileNode tree."""
        if not self.file_tree:
            return ""
        
        # ensure file_tree is a FileNode and not a list
        if not isinstance(self.file_tree, FileNode):
            return f"Error: file_tree is not a FileNode (got {type(self.file_tree)})"
        
        lines = []
        
        def format_recursive(node: FileNode, prefix: str = "", is_last: bool = True):
            # ensure node is a FileNode
            if not isinstance(node, FileNode):
                lines.append(f"{prefix}[ERROR: Expected FileNode, got {type(node)}]")
                return
                
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{node.name}")
            
            if node.is_dir:
                extension = "    " if is_last else "│   "
                for i, child in enumerate(sorted(node.children, key=lambda x: (x.is_file(), x.name))):
                    is_child_last = (i == len(node.children) - 1)
                    format_recursive(child, prefix + extension, is_child_last)
        
        format_recursive(self.file_tree)
        return "\n".join(lines)
    
    @property
    def structure(self) -> str:
        """Alias for file_tree_string for legacy compatibility."""
        return self.file_tree_string
    
    def extract_token_data_from_tree(self) -> Dict[str, int]:
        """Extract token data from the file tree."""
        token_data = {}
        
        def extract_recursive(node: FileNode):
            # ensure node is a FileNode
            if not isinstance(node, FileNode):
                return
                
            if node.is_file() and node.token_count and node.token_count > 0:
                token_data[node.path] = node.token_count
            else:
                for child in node.children:
                    extract_recursive(child)
        
        if self.file_tree and isinstance(self.file_tree, FileNode):
            extract_recursive(self.file_tree)
        return token_data
    
    def update_token_data_from_tree(self):
        """Update the token_data field from the file tree."""
        self.token_data = self.extract_token_data_from_tree()
    
    def has_errors(self) -> bool:
        """Check if any errors occurred during analysis."""
        return len(self.errors) > 0
    
    def get_error_summary(self) -> str:
        """Get a summary of all errors."""
        if not self.errors:
            return "No errors encountered."
        return f"{len(self.errors)} errors encountered:\n" + "\n".join(f"- {e}" for e in self.errors)


@dataclass
class TokenBudget:
    """
    Manages token budget for LLM interactions.
    
    This class helps track and manage token usage to ensure
    we stay within LLM context limits.
    """
    
    max_tokens: int
    used_tokens: int = 0
    reserved_tokens: int = 0  # For response, system prompts, etc.
    
    @property
    def available_tokens(self) -> int:
        """Calculate available tokens."""
        return self.max_tokens - self.used_tokens - self.reserved_tokens
    
    @property
    def usage_percentage(self) -> float:
        """Calculate token usage percentage."""
        return (self.used_tokens / self.max_tokens) * 100
    
    def can_fit(self, tokens: int) -> bool:
        """Check if given number of tokens can fit in budget."""
        return tokens <= self.available_tokens
    
    def use(self, tokens: int) -> bool:
        """
        Use tokens from budget.
        
        Returns True if successful, False if would exceed budget.
        """
        if not self.can_fit(tokens):
            return False
        self.used_tokens += tokens
        return True
    
    def reset(self) -> None:
        """Reset the token budget."""
        self.used_tokens = 0