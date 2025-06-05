"""
Core data models for repo2txt.

This module contains the fundamental data structures used throughout
the application for configuration, file representation, and analysis results.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Literal
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
    
    def is_file(self) -> bool:
        """Check if this node represents a file."""
        return self.type == 'file'
    
    def is_directory(self) -> bool:
        """Check if this node represents a directory."""
        return self.type == 'dir'


@dataclass
class AnalysisResult:
    """Results from analyzing a repository."""
    
    repo_name: str
    branch: Optional[str]
    readme_content: str
    structure: str
    file_contents: str
    token_data: Dict[str, int]
    total_tokens: int
    total_files: int
    errors: List[str]
    
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