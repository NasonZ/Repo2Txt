#!/usr/bin/env python3
"""
Repo2Txt - Enhanced Version
A tool to interactively traverse and analyze GitHub repositories or local folders,
extracting structure and contents into a text file with optional token counting.
"""

import os
import sys
import argparse
import mimetypes
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict
import json
import logging
from datetime import datetime

try:
    from github import Github
    from github.Repository import Repository as GithubRepo
except ImportError:
    print("Error: PyGithub is required. Install with: pip install PyGithub")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("Error: tqdm is required. Install with: pip install tqdm")
    sys.exit(1)

try:
    import tiktoken
except ImportError:
    tiktoken = None
    print("Warning: tiktoken not installed. Token counting disabled. Install with: pip install tiktoken")

# Configuration
@dataclass
class Config:
    """Configuration settings for repo2txt"""
    github_token: str = field(default_factory=lambda: os.getenv('GITHUB_TOKEN', ''))
    
    # Directories to exclude
    excluded_dirs: Set[str] = field(default_factory=lambda: {
        '__pycache__', '.git', '.hg', '.svn', '.idea', '.vscode', 
        'node_modules', '.pytest_cache', '.mypy_cache', '.tox',
        'venv', 'env', '.env', 'virtualenv', '.virtualenv',
        'bower_components', 'vendor', 'coverage', '.coverage',
        '.sass-cache', '.cache', 'dist', 'build', '.next',
        '.nuxt', '.output', '.parcel-cache', 'out'
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


# Data structures
@dataclass
class FileNode:
    """Represents a file or directory in the repository"""
    path: str
    name: str
    type: str  # 'file' or 'dir'
    size: Optional[int] = None
    token_count: Optional[int] = None
    error: Optional[str] = None


@dataclass
class AnalysisResult:
    """Results from analyzing a repository"""
    repo_name: str
    branch: Optional[str]
    readme_content: str
    structure: str
    file_contents: str
    token_data: Dict[str, int]
    total_tokens: int
    total_files: int
    errors: List[str]


class FileAnalyzer:
    """Handles file analysis and content extraction"""
    
    def __init__(self, config: Config):
        self.config = config
        self.token_encoder = None
        if tiktoken and config.enable_token_counting:
            try:
                self.token_encoder = tiktoken.get_encoding(config.token_encoder)
            except Exception as e:
                logging.warning(f"Failed to initialize token encoder: {e}")
    
    def is_binary_file(self, file_path: str, content: bytes = None) -> bool:
        """Improved binary file detection"""
        # Check extension first
        path = Path(file_path)
        if path.suffix.lower() in self.config.binary_extensions:
            return True
        
        # Check skip patterns
        for pattern in self.config.skip_patterns:
            if pattern.startswith('*'):
                if path.name.endswith(pattern[1:]):
                    return True
            elif path.name == pattern:
                return True
        
        # Use mimetypes as fallback
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type:
            return not mime_type.startswith('text/')
        
        # If we have content, check for binary data
        if content:
            # Check for null bytes in first 8192 bytes
            sample = content[:8192]
            if b'\x00' in sample:
                return True
            
            # Try to decode as text
            try:
                sample.decode('utf-8')
                return False
            except UnicodeDecodeError:
                return True
        
        return False
    
    def read_file_content(self, file_path: str) -> Tuple[Optional[str], Optional[str]]:
        """Read file content with multiple encoding fallbacks"""
        try:
            # Check file size first
            file_size = os.path.getsize(file_path)
            if file_size > self.config.max_file_size:
                return None, f"File too large ({file_size:,} bytes)"
            
            # Read raw content
            with open(file_path, 'rb') as f:
                raw_content = f.read()
            
            # Check if binary
            if self.is_binary_file(file_path, raw_content):
                return None, "Binary file"
            
            # Try different encodings
            for encoding in self.config.encoding_fallbacks:
                try:
                    return raw_content.decode(encoding), None
                except UnicodeDecodeError:
                    continue
            
            return None, "Unable to decode file with available encodings"
            
        except PermissionError:
            return None, "Permission denied"
        except Exception as e:
            return None, f"Error reading file: {str(e)}"
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if not self.token_encoder:
            return 0
        try:
            return len(self.token_encoder.encode(text))
        except Exception:
            return 0


class RepositoryAnalyzer:
    """Main analyzer for repository traversal and analysis"""
    
    def __init__(self, config: Config):
        self.config = config
        self.file_analyzer = FileAnalyzer(config)
        self.errors = []
    
    def parse_range(self, range_str: str) -> List[int]:
        """Parse a range string like '1-3,5,7-9' into a list of integers"""
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
    
    def get_readme_content(self, repo: Union[str, GithubRepo], branch: Optional[str] = None) -> str:
        """Get README content from repository"""
        if isinstance(repo, str):  # Local path
            readme_files = ['README.md', 'readme.md', 'README.MD', 'README', 
                           'readme', 'README.txt', 'readme.txt']
            for readme_name in readme_files:
                readme_path = os.path.join(repo, readme_name)
                if os.path.exists(readme_path):
                    content, error = self.file_analyzer.read_file_content(readme_path)
                    if content:
                        return content
                    elif error:
                        self.errors.append(f"README {readme_name}: {error}")
            return "README not found."
        else:  # GitHub repo
            try:
                # Try multiple README variations
                for readme_name in ['README.md', 'readme.md', 'README', 'readme']:
                    try:
                        readme = repo.get_contents(readme_name, ref=branch)
                        return readme.decoded_content.decode('utf-8')
                    except:
                        continue
                return "README not found."
            except Exception as e:
                self.errors.append(f"Error fetching README: {str(e)}")
                return "README not found."
    
    def traverse_local_interactive(self, repo_path: str, current_path: str = "") -> Tuple[str, Set[str], Dict[str, int]]:
        """Interactive traversal of local repository with back navigation"""
        # Stack to track navigation history for back functionality
        navigation_stack = []
        
        # Main selection state
        structure = ""
        selected_paths = set()
        token_data = {}
        
        # Start with root directory
        current_state = {
            'path': current_path,
            'structure': "",
            'selected_paths': set(),
            'token_data': {}
        }
        
        while True:
            full_path = os.path.join(repo_path, current_state['path'])
            
            try:
                # Get items in current directory
                items = []
                for item in sorted(os.listdir(full_path)):
                    if item.startswith('.') and item not in {'.github', '.gitlab'}:
                        continue  # Skip hidden files except certain dirs
                        
                    item_path = os.path.join(full_path, item)
                    if os.path.isdir(item_path):
                        if item not in self.config.excluded_dirs:
                            items.append((item, 'dir'))
                    else:
                        items.append((item, 'file'))
                
                if not items:
                    if navigation_stack:
                        # Go back automatically if directory is empty
                        print("Empty directory, going back...")
                        current_state = navigation_stack.pop()
                        continue
                    else:
                        return structure, selected_paths, token_data
                
                # Display items
                print(f"\nContents of {current_state['path'] or 'root'}:")
                for i, (item, item_type) in enumerate(items, start=1):
                    print(f"{i:3d}. {item} ({item_type})")
                
                # Show navigation options
                print("\nOptions: Enter numbers (e.g., 1-5,7), 'a' for all, 's' to skip", end="")
                if navigation_stack:
                    print(", 'b' to go back", end="")
                print(", 'q' to quit")
                
                # Get user selection
                while True:
                    selection = input("Your choice: ").strip().lower()
                    
                    # Handle navigation commands
                    if selection == 'q':
                        confirm = input("Quit selection? You'll lose current selections. (y/n): ").lower()
                        if confirm == 'y':
                            return "", set(), {}
                    
                    if selection == 'b' and navigation_stack:
                        # Go back to previous state
                        current_state = navigation_stack.pop()
                        break
                    
                    if selection == 's':
                        # Skip this directory
                        if navigation_stack:
                            # Restore previous state and continue
                            prev_state = navigation_stack.pop()
                            return (prev_state['structure'] + structure, 
                                   prev_state['selected_paths'] | selected_paths,
                                   {**prev_state['token_data'], **token_data})
                        else:
                            return structure, selected_paths, token_data
                    
                    if selection == 'a':
                        selected_indices = list(range(1, len(items) + 1))
                        break
                    else:
                        selected_indices = self.parse_range(selection)
                        if selected_indices:
                            # Validate indices
                            if all(1 <= idx <= len(items) for idx in selected_indices):
                                break
                            else:
                                print("Invalid indices. Please try again.")
                        else:
                            print("Invalid input. Please use format like: 1-3,5,7")
                
                if selection in ['b', 'q']:
                    continue  # Restart the loop with new/previous state
                
                # Process selected items
                temp_structure = ""
                temp_selected = set()
                temp_tokens = {}
                subdirs_to_explore = []
                
                for i, (item, item_type) in enumerate(items, start=1):
                    item_path = os.path.join(full_path, item)
                    rel_item_path = os.path.relpath(item_path, repo_path)
                    
                    if i in selected_indices:
                        if item_type == 'dir':
                            temp_structure += f"{rel_item_path}/\n"
                            subdirs_to_explore.append((item, rel_item_path))
                        else:  # file
                            temp_structure += f"{rel_item_path}\n"
                            temp_selected.add(rel_item_path)
                            
                            # Count tokens if enabled
                            if self.config.enable_token_counting:
                                content, error = self.file_analyzer.read_file_content(item_path)
                                if content:
                                    tokens = self.file_analyzer.count_tokens(content)
                                    if tokens > 0:
                                        temp_tokens[rel_item_path] = tokens
                    else:
                        # Not selected - mark as omitted
                        if item_type == 'dir':
                            temp_structure += f"{rel_item_path}/ (not selected)\n"
                        else:
                            temp_structure += f"{rel_item_path} (not selected)\n"
                
                # Update current state with selections
                structure += temp_structure
                selected_paths.update(temp_selected)
                token_data.update(temp_tokens)
                
                # Now handle subdirectories
                for subdir_name, subdir_path in subdirs_to_explore:
                    while True:
                        print(f"\nSelect items in '{subdir_name}'?")
                        sub_choice = input("Options: (y)es, (n)o, (a)ll, (b)ack: ").lower()
                        
                        if sub_choice == 'b':
                            # Don't process this subdirectory, but keep current selections
                            break
                        
                        if sub_choice in ['y', 'n', 'a']:
                            if sub_choice == 'y':
                                # Save current state to stack before diving deeper
                                navigation_stack.append({
                                    'path': current_state['path'],
                                    'structure': structure,
                                    'selected_paths': selected_paths.copy(),
                                    'token_data': token_data.copy()
                                })
                                
                                # Explore subdirectory interactively
                                sub_structure, sub_selected, sub_tokens = self.traverse_local_interactive(
                                    repo_path, 
                                    os.path.join(current_state['path'], subdir_name)
                                )
                                structure += sub_structure
                                selected_paths.update(sub_selected)
                                token_data.update(sub_tokens)
                                
                                # Restore context after subdirectory exploration
                                if navigation_stack:
                                    navigation_stack.pop()
                                    
                            elif sub_choice == 'a':
                                # Select all files recursively
                                for root, dirs, files in os.walk(os.path.join(full_path, subdir_name)):
                                    # Filter excluded directories
                                    dirs[:] = [d for d in dirs if d not in self.config.excluded_dirs]
                                    rel_root = os.path.relpath(root, repo_path)
                                    
                                    for file in files:
                                        file_path = os.path.join(rel_root, file)
                                        structure += f"{file_path}\n"
                                        selected_paths.add(file_path)
                                        
                                        # Count tokens if enabled
                                        if self.config.enable_token_counting:
                                            full_file_path = os.path.join(root, file)
                                            content, error = self.file_analyzer.read_file_content(full_file_path)
                                            if content:
                                                tokens = self.file_analyzer.count_tokens(content)
                                                if tokens > 0:
                                                    token_data[file_path] = tokens
                            break
                        else:
                            print("Invalid choice. Please enter 'y', 'n', 'a', or 'b'.")
                
                # If we're at root level, we're done
                if not current_state['path']:
                    return structure, selected_paths, token_data
                else:
                    # Return to parent context
                    if navigation_stack:
                        parent_state = navigation_stack.pop()
                        return (parent_state['structure'] + structure,
                               parent_state['selected_paths'] | selected_paths,
                               {**parent_state['token_data'], **token_data})
                    else:
                        return structure, selected_paths, token_data
            
            except PermissionError:
                self.errors.append(f"Permission denied: {full_path}")
                if navigation_stack:
                    current_state = navigation_stack.pop()
                else:
                    return structure, selected_paths, token_data
            except Exception as e:
                self.errors.append(f"Error in {full_path}: {str(e)}")
                if navigation_stack:
                    current_state = navigation_stack.pop()
                else:
                    return structure, selected_paths, token_data
    
    def traverse_github_interactive(self, repo: GithubRepo, path: str = "", branch: Optional[str] = None) -> Tuple[str, Set[str], Dict[str, int]]:
        """Interactive traversal of GitHub repository with back navigation"""
        # Stack to track navigation history
        navigation_stack = []
        
        # Main selection state
        structure = ""
        selected_paths = set()
        token_data = {}
        
        # Start with current directory
        current_state = {
            'path': path,
            'structure': "",
            'selected_paths': set(),
            'token_data': {}
        }
        
        while True:
            try:
                contents = repo.get_contents(current_state['path'], ref=branch)
                
                # Sort and filter contents
                items = []
                for content in sorted(contents, key=lambda x: (x.type != 'dir', x.name)):
                    if content.type == 'dir' and content.name not in self.config.excluded_dirs:
                        items.append((content, 'dir'))
                    elif content.type == 'file':
                        items.append((content, 'file'))
                
                if not items:
                    if navigation_stack:
                        # Go back automatically if directory is empty
                        print("Empty directory, going back...")
                        current_state = navigation_stack.pop()
                        continue
                    else:
                        return structure, selected_paths, token_data
                
                # Display items
                print(f"\nContents of {current_state['path'] or 'root'}:")
                for i, (content, item_type) in enumerate(items, start=1):
                    size_str = f" ({content.size:,} bytes)" if item_type == 'file' and content.size else ""
                    print(f"{i:3d}. {content.name} ({item_type}){size_str}")
                
                # Show navigation options
                print("\nOptions: Enter numbers (e.g., 1-5,7), 'a' for all, 's' to skip", end="")
                if navigation_stack:
                    print(", 'b' to go back", end="")
                print(", 'q' to quit")
                
                # Get user selection
                while True:
                    selection = input("Your choice: ").strip().lower()
                    
                    # Handle navigation commands
                    if selection == 'q':
                        confirm = input("Quit selection? You'll lose current selections. (y/n): ").lower()
                        if confirm == 'y':
                            return "", set(), {}
                    
                    if selection == 'b' and navigation_stack:
                        # Go back to previous state
                        current_state = navigation_stack.pop()
                        break
                    
                    if selection == 's':
                        # Skip this directory
                        if navigation_stack:
                            # Restore previous state and continue
                            prev_state = navigation_stack.pop()
                            return (prev_state['structure'] + structure, 
                                   prev_state['selected_paths'] | selected_paths,
                                   {**prev_state['token_data'], **token_data})
                        else:
                            return structure, selected_paths, token_data
                    
                    if selection == 'a':
                        selected_indices = list(range(1, len(items) + 1))
                        break
                    else:
                        selected_indices = self.parse_range(selection)
                        if selected_indices and all(1 <= idx <= len(items) for idx in selected_indices):
                            break
                        print("Invalid input. Please try again.")
                
                if selection in ['b', 'q']:
                    continue  # Restart the loop with new/previous state
                
                # Process selected items
                temp_structure = ""
                temp_selected = set()
                temp_tokens = {}
                subdirs_to_explore = []
                
                for i, (content, item_type) in enumerate(items, start=1):
                    if i in selected_indices:
                        if item_type == 'dir':
                            temp_structure += f"{content.path}/\n"
                            subdirs_to_explore.append((content.name, content.path))
                        else:  # file
                            temp_structure += f"{content.path}\n"
                            temp_selected.add(content.path)
                            
                            # Count tokens if enabled
                            if self.config.enable_token_counting and content.encoding != 'none':
                                try:
                                    file_content = content.decoded_content.decode('utf-8')
                                    tokens = self.file_analyzer.count_tokens(file_content)
                                    if tokens > 0:
                                        temp_tokens[content.path] = tokens
                                except Exception:
                                    pass
                    else:
                        # Not selected
                        if item_type == 'dir':
                            temp_structure += f"{content.path}/ (not selected)\n"
                        else:
                            temp_structure += f"{content.path} (not selected)\n"
                
                # Update current state with selections
                structure += temp_structure
                selected_paths.update(temp_selected)
                token_data.update(temp_tokens)
                
                # Now handle subdirectories
                for subdir_name, subdir_path in subdirs_to_explore:
                    while True:
                        print(f"\nSelect items in '{subdir_name}'?")
                        sub_choice = input("Options: (y)es, (n)o, (a)ll, (b)ack: ").lower()
                        
                        if sub_choice == 'b':
                            # Don't process this subdirectory, but keep current selections
                            break
                        
                        if sub_choice in ['y', 'n', 'a']:
                            if sub_choice == 'y':
                                # Save current state to stack before diving deeper
                                navigation_stack.append({
                                    'path': current_state['path'],
                                    'structure': structure,
                                    'selected_paths': selected_paths.copy(),
                                    'token_data': token_data.copy()
                                })
                                
                                # Explore subdirectory interactively
                                sub_structure, sub_selected, sub_tokens = self.traverse_github_interactive(
                                    repo, subdir_path, branch
                                )
                                structure += sub_structure
                                selected_paths.update(sub_selected)
                                token_data.update(sub_tokens)
                                
                                # Restore context after subdirectory exploration
                                if navigation_stack:
                                    navigation_stack.pop()
                                    
                            elif sub_choice == 'a':
                                # Recursively get all files in subdirectory
                                sub_structure, sub_selected, sub_tokens = self._get_all_github_files(
                                    repo, subdir_path, branch
                                )
                                structure += sub_structure
                                selected_paths.update(sub_selected)
                                token_data.update(sub_tokens)
                            break
                        else:
                            print("Invalid choice. Please enter 'y', 'n', 'a', or 'b'.")
                
                # If we're at root level, we're done
                if not current_state['path']:
                    return structure, selected_paths, token_data
                else:
                    # Return to parent context
                    if navigation_stack:
                        parent_state = navigation_stack.pop()
                        return (parent_state['structure'] + structure,
                               parent_state['selected_paths'] | selected_paths,
                               {**parent_state['token_data'], **token_data})
                    else:
                        return structure, selected_paths, token_data
            
            except Exception as e:
                self.errors.append(f"Error accessing {current_state['path']}: {str(e)}")
                if navigation_stack:
                    current_state = navigation_stack.pop()
                else:
                    return structure, selected_paths, token_data
    
    def _get_all_github_files(self, repo: GithubRepo, path: str, branch: Optional[str] = None) -> Tuple[str, Set[str], Dict[str, int]]:
        """Recursively get all files from a GitHub directory"""
        structure = ""
        selected_paths = set()
        token_data = {}
        
        try:
            contents = repo.get_contents(path, ref=branch)
            for content in contents:
                if content.type == "dir":
                    if content.name not in self.config.excluded_dirs:
                        structure += f"{content.path}/\n"
                        sub_structure, sub_selected, sub_tokens = self._get_all_github_files(
                            repo, content.path, branch
                        )
                        structure += sub_structure
                        selected_paths.update(sub_selected)
                        token_data.update(sub_tokens)
                else:
                    structure += f"{content.path}\n"
                    selected_paths.add(content.path)
                    
                    if self.config.enable_token_counting and content.encoding != 'none':
                        try:
                            decoded_content = content.decoded_content.decode('utf-8')
                            token_count = self.file_analyzer.count_tokens(decoded_content)
                            if token_count > 0:
                                token_data[content.path] = token_count
                        except Exception:
                            pass
        except Exception as e:
            self.errors.append(f"Error in {path}: {str(e)}")
        
        return structure, selected_paths, token_data
    
    def get_file_contents(self, repo: Union[str, GithubRepo], selected_files: List[str], 
                         branch: Optional[str] = None) -> Tuple[str, Dict[str, int]]:
        """Get contents of selected files"""
        file_contents = ""
        token_data = {}
        is_local = isinstance(repo, str)
        
        print(f"\nProcessing {len(selected_files)} selected files...")
        
        for file_path in tqdm(selected_files, desc="Reading files"):
            try:
                if is_local:
                    full_path = os.path.join(repo, file_path)
                    content, error = self.file_analyzer.read_file_content(full_path)
                    
                    if content:
                        file_contents += f"{'='*80}\n"
                        file_contents += f"File: {file_path}\n"
                        file_contents += f"{'='*80}\n"
                        file_contents += content
                        if not content.endswith('\n'):
                            file_contents += '\n'
                        file_contents += "\n"
                        
                        if self.config.enable_token_counting:
                            tokens = self.file_analyzer.count_tokens(content)
                            if tokens > 0:
                                token_data[file_path] = tokens
                    else:
                        file_contents += f"{'='*80}\n"
                        file_contents += f"File: {file_path}\n"
                        file_contents += f"Error: {error}\n"
                        file_contents += f"{'='*80}\n\n"
                else:  # GitHub
                    try:
                        content_obj = repo.get_contents(file_path, ref=branch)
                        
                        if content_obj.type == 'file':
                            if self.file_analyzer.is_binary_file(file_path):
                                file_contents += f"{'='*80}\n"
                                file_contents += f"File: {file_path}\n"
                                file_contents += f"Error: Binary file\n"
                                file_contents += f"{'='*80}\n\n"
                            elif content_obj.size > self.config.max_file_size:
                                file_contents += f"{'='*80}\n"
                                file_contents += f"File: {file_path}\n"
                                file_contents += f"Error: File too large ({content_obj.size:,} bytes)\n"
                                file_contents += f"{'='*80}\n\n"
                            elif content_obj.encoding == 'none':
                                file_contents += f"{'='*80}\n"
                                file_contents += f"File: {file_path}\n"
                                file_contents += f"Error: No encoding available\n"
                                file_contents += f"{'='*80}\n\n"
                            else:
                                # Try to decode content
                                decoded = False
                                for encoding in self.config.encoding_fallbacks:
                                    try:
                                        content = content_obj.decoded_content.decode(encoding)
                                        decoded = True
                                        break
                                    except UnicodeDecodeError:
                                        continue
                                
                                if decoded:
                                    file_contents += f"{'='*80}\n"
                                    file_contents += f"File: {file_path}\n"
                                    file_contents += f"{'='*80}\n"
                                    file_contents += content
                                    if not content.endswith('\n'):
                                        file_contents += '\n'
                                    file_contents += "\n"
                                    
                                    if self.config.enable_token_counting:
                                        tokens = self.file_analyzer.count_tokens(content)
                                        if tokens > 0:
                                            token_data[file_path] = tokens
                                else:
                                    file_contents += f"{'='*80}\n"
                                    file_contents += f"File: {file_path}\n"
                                    file_contents += f"Error: Unable to decode file\n"
                                    file_contents += f"{'='*80}\n\n"
                    except Exception as e:
                        file_contents += f"{'='*80}\n"
                        file_contents += f"File: {file_path}\n"
                        file_contents += f"Error: {str(e)}\n"
                        file_contents += f"{'='*80}\n\n"
            except Exception as e:
                self.errors.append(f"Error processing {file_path}: {str(e)}")
        
        return file_contents, token_data
    
    def generate_token_report(self, token_data: Dict[str, int]) -> str:
        """
        Generate a focused token report for iterative file selection.
        
        Two main views:
        1. Tree representation - visual structure understanding
        2. Full table - precise token counts for budgeting
        
        Optimized for workflow: analyze → select → re-run → repeat
        """
        if not token_data:
            return "No token data available.\n"
        
        report = "TOKEN ANALYSIS REPORT\n"
        report += "=" * 80 + "\n\n"
        
        # Calculate totals
        total_tokens = sum(token_data.values())
        total_files = len(token_data)
        
        # Build directory tree with aggregated token counts
        tree = defaultdict(lambda: {"tokens": 0, "files": 0, "subdirs": {}})
        dir_totals = {}  # For tracking all directories
        
        for file_path, tokens in token_data.items():
            parts = file_path.split(os.sep)
            
            # Track all directory paths and their totals
            for i in range(len(parts)):
                dir_path = os.sep.join(parts[:i+1])
                if i < len(parts) - 1:  # It's a directory
                    if dir_path not in dir_totals:
                        dir_totals[dir_path] = {"tokens": 0, "files": 0}
                    dir_totals[dir_path]["tokens"] += tokens
                    dir_totals[dir_path]["files"] += 1
            
            # Build tree structure
            current = tree
            for i, part in enumerate(parts[:-1]):
                if part not in current:
                    current[part] = {"tokens": 0, "files": 0, "subdirs": {}}
                current[part]["tokens"] += tokens
                current = current[part]["subdirs"]
            
            # Add the file
            file_name = parts[-1]
            current[file_name] = {"tokens": tokens, "files": 1, "subdirs": {}}
            
            # Update parent directory counts
            current = tree
            for part in parts[:-1]:
                current[part]["files"] += 1
                current = current[part]["subdirs"]
        
        # 1. TREE VIEW
        report += "📂 Directory Tree:\n"
        report += "-" * 80 + "\n"
        
        def print_tree(node: dict, name: str = "root", prefix: str = "", is_last: bool = True) -> str:
            output = ""
            if name != "root":
                output += prefix + ("└── " if is_last else "├── ")
                if node.get("subdirs"):  # Directory
                    output += f"{name}/ (Tokens: {node['tokens']:,}, Files: {node['files']})\n"
                else:  # File
                    output += f"{name} (Tokens: {node['tokens']:,})\n"
            
            if node.get("subdirs"):
                items = sorted(node["subdirs"].items())
                for i, (child_name, child_node) in enumerate(items):
                    is_last_child = (i == len(items) - 1)
                    child_prefix = prefix + ("    " if is_last else "│   ") if name != "root" else ""
                    output += print_tree(child_node, child_name, child_prefix, is_last_child)
            
            return output
        
        # Print tree
        for name, node in sorted(tree.items()):
            is_last = (name == sorted(tree.keys())[-1]) if tree else True
            report += print_tree(node, name, "", is_last)
        
        report += f"\nTOTAL: {total_tokens:,} tokens, {total_files} files\n"
        
        # 2. TABLE VIEW
        report += "\n" + "=" * 80 + "\n"
        report += "📊 Token Count Summary:\n"
        report += "-" * 100 + "\n"
        report += f"{'File/Directory':<70} | {'Token Count':>12} | {'File Count':>10}\n"
        report += "-" * 100 + "\n"
        
        # Collect all paths (both directories and files)
        all_paths = []
        
        # Add all directories
        for dir_path, info in sorted(dir_totals.items()):
            all_paths.append({
                'path': dir_path,
                'tokens': info['tokens'],
                'files': info['files'],
                'is_dir': True,
                'depth': dir_path.count(os.sep)
            })
        
        # Add all files
        for file_path, tokens in token_data.items():
            all_paths.append({
                'path': file_path,
                'tokens': tokens,
                'files': 1,
                'is_dir': False,
                'depth': file_path.count(os.sep)
            })
        
        # Sort by path to maintain hierarchy
        all_paths.sort(key=lambda x: x['path'])
        
        # Group by top-level directory and add separators
        current_top_dir = None
        for item in all_paths:
            # Get top-level directory
            parts = item['path'].split(os.sep)
            top_dir = parts[0] if parts else ""
            
            # Add separator between top-level directories
            if top_dir != current_top_dir and current_top_dir is not None:
                report += "-" * 100 + "\n"
            current_top_dir = top_dir
            
            # Format the line
            path = item['path']
            tokens = item['tokens']
            files = item['files'] if item['is_dir'] else ''
            
            # Add indentation based on depth
            indent = "  " * item['depth']
            display_name = indent + os.path.basename(path)
            if item['is_dir']:
                display_name = indent + os.path.basename(path)
            else:
                display_name = indent + "  " + os.path.basename(path)
            
            # Handle long paths
            if len(display_name) > 69:
                # Show full path on separate line
                report += f"{path}\n"
                report += f"{' ' * 70} | {tokens:>12,} | {files:>10}\n"
            else:
                report += f"{display_name:<70} | {tokens:>12,} | {files:>10}\n"
        
        report += "-" * 100 + "\n"
        report += f"{'TOTAL':<70} | {total_tokens:>12,} | {total_files:>10}\n"
        report += "=" * 100 + "\n"
        
        # 3. QUICK REFERENCE (minimal, just the essentials)
        report += "\n📈 Quick Stats:\n"
        report += "-" * 80 + "\n"
        
        # Summary statistics
        report += f"Total: {total_tokens:,} tokens across {total_files} files\n"
        report += f"Average: {total_tokens // total_files:,} tokens/file\n"
        
        # Calculate distribution statistics
        token_values = list(token_data.values())
        token_values.sort()
        
        # Min/Max
        min_tokens = min(token_values) if token_values else 0
        max_tokens = max(token_values) if token_values else 0
        
        # Median
        median_tokens = token_values[len(token_values) // 2] if token_values else 0
        
        # Standard deviation (simple calculation)
        mean = total_tokens / total_files if total_files > 0 else 0
        variance = sum((x - mean) ** 2 for x in token_values) / total_files if total_files > 0 else 0
        std_dev = int(variance ** 0.5)
        
        report += f"\nDistribution:\n"
        report += f"  Min: {min_tokens:,} | Median: {median_tokens:,} | Max: {max_tokens:,} | Std Dev: {std_dev:,}\n"
        
        # File size buckets
        tiny = sum(1 for t in token_values if t <= 100)
        small = sum(1 for t in token_values if 100 < t <= 500)
        medium = sum(1 for t in token_values if 500 < t <= 1000)
        large = sum(1 for t in token_values if 1000 < t <= 5000)
        huge = sum(1 for t in token_values if t > 5000)
        
        report += f"\nFile size distribution:\n"
        if tiny > 0: 
            report += f"  ≤100 tokens:    {tiny:>4} files ({tiny*100//total_files:>3}%)\n"
        if small > 0: 
            report += f"  101-500:        {small:>4} files ({small*100//total_files:>3}%)\n"
        if medium > 0: 
            report += f"  501-1000:       {medium:>4} files ({medium*100//total_files:>3}%)\n"
        if large > 0: 
            report += f"  1001-5000:      {large:>4} files ({large*100//total_files:>3}%)\n"
        if huge > 0: 
            report += f"  >5000:          {huge:>4} files ({huge*100//total_files:>3}%)\n"
        
        # Top 5 largest directories
        report += "\nTop 5 largest directories:\n"
        sorted_dirs = sorted(dir_totals.items(), key=lambda x: x[1]['tokens'], reverse=True)[:5]
        for dir_path, info in sorted_dirs:
            avg_per_file = info['tokens'] // info['files'] if info['files'] > 0 else 0
            report += f"  {info['tokens']:>8,} tokens: {dir_path}/ ({info['files']} files, avg {avg_per_file:,}/file)\n"
        
        # Top 10 largest files
        report += "\nTop 10 largest files:\n"
        sorted_files = sorted(token_data.items(), key=lambda x: x[1], reverse=True)[:10]
        for file_path, tokens in sorted_files:
            report += f"  {tokens:>8,} tokens: {file_path}\n"
        
        report += "\n" + "=" * 80 + "\n"
        
        return report
    
    def analyze(self, repo_url_or_path: str) -> AnalysisResult:
        """Main analysis function"""
        is_local = os.path.isdir(repo_url_or_path)
        branch = None
        
        if is_local:
            repo = repo_url_or_path
            repo_name = os.path.basename(os.path.abspath(repo_url_or_path))
            print(f"\nAnalyzing local repository: {repo_name}")
        else:
            if not self.config.github_token:
                raise ValueError("GitHub token not found. Set GITHUB_TOKEN environment variable.")
            
            # Parse GitHub URL
            parts = repo_url_or_path.replace('https://github.com/', '').strip('/').split('/')
            if len(parts) < 2:
                raise ValueError("Invalid GitHub URL format")
            
            owner, name = parts[0], parts[1]
            
            print(f"\nConnecting to GitHub repository: {owner}/{name}")
            g = Github(self.config.github_token)
            repo = g.get_repo(f"{owner}/{name}")
            repo_name = repo.name
            
            # Select branch
            branches = [b.name for b in repo.get_branches()]
            print(f"\nAvailable branches: {', '.join(branches[:10])}")
            if len(branches) > 10:
                print(f"... and {len(branches) - 10} more")
            
            branch_input = input("\nEnter branch name (or press Enter for default): ").strip()
            if branch_input and branch_input in branches:
                branch = branch_input
            elif branch_input:
                print(f"Branch '{branch_input}' not found. Using default branch.")
        
        # Get README
        print("\nFetching README...")
        readme_content = self.get_readme_content(repo, branch)
        
        # Interactive traversal
        print("\nStarting interactive file selection...")
        if is_local:
            structure, selected_paths, token_data = self.traverse_local_interactive(repo)
        else:
            structure, selected_paths, token_data = self.traverse_github_interactive(repo, "", branch)
        
        # Get file contents
        if selected_paths:
            file_contents, file_token_data = self.get_file_contents(repo, list(selected_paths), branch)
            token_data.update(file_token_data)
        else:
            file_contents = "No files selected.\n"
        
        # Calculate totals
        total_tokens = sum(token_data.values())
        total_files = len(selected_paths)
        
        return AnalysisResult(
            repo_name=repo_name,
            branch=branch,
            readme_content=readme_content,
            structure=structure,
            file_contents=file_contents,
            token_data=token_data,
            total_tokens=total_tokens,
            total_files=total_files,
            errors=self.errors
        )


def generate_instructions(repo_name: str, branch: Optional[str] = None) -> str:
    """Generate analysis instructions"""
    branch_info = f" (branch: {branch})" if branch else ""
    
    instructions = f"""# Repository Analysis: {repo_name}{branch_info}

## Analysis Instructions

Please analyze this repository to understand its structure, purpose, and functionality. Follow these steps:

1. **README Review**: Start by reading the README to understand the project's purpose, setup, and usage.

2. **Structure Analysis**: Examine the repository structure to understand the organization and architecture.

3. **Entry Points**: Identify the main entry point(s) of the application and trace the execution flow.

4. **Dependencies**: Note the key dependencies and libraries used in the project.

5. **Core Components**: Analyze the main modules, classes, and functions that form the core functionality.

6. **Configuration**: Look for configuration files and environment variables to understand deployment options.

7. **Testing**: Review any test files to understand the testing approach and coverage.

8. **Code Quality**: Assess code organization, documentation, patterns used, and potential improvements.

9. **Security**: Note any security considerations or potential vulnerabilities.

10. **Summary**: Provide a comprehensive summary of the project's purpose, architecture, strengths, and areas for improvement.

---

"""
    return instructions


def save_results(result: AnalysisResult, output_dir: str = ".", save_json: bool = False) -> Tuple[str, Optional[str], Optional[str]]:
    """Save analysis results to files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_branch = result.branch.replace('/', '_') if result.branch else "main"
    base_name = f"{result.repo_name}_{safe_branch}_{timestamp}"
    
    # Save main analysis file
    main_file = os.path.join(output_dir, f"{base_name}_analysis.txt")
    with open(main_file, 'w', encoding='utf-8') as f:
        # Write header
        f.write(generate_instructions(result.repo_name, result.branch))
        
        # Write README
        f.write("## README Content\n\n")
        f.write(result.readme_content)
        f.write("\n\n")
        
        # Write structure
        f.write("## Repository Structure\n\n")
        f.write("```\n")
        f.write(result.structure)
        f.write("```\n\n")
        
        # Write summary statistics
        f.write("## Summary Statistics\n\n")
        f.write(f"- Total files selected: {result.total_files}\n")
        if result.total_tokens > 0:
            f.write(f"- Total tokens: {result.total_tokens:,}\n")
        f.write(f"- Errors encountered: {len(result.errors)}\n\n")
        
        # Write file contents
        f.write("## File Contents\n\n")
        f.write(result.file_contents)
        
        # Write errors if any
        if result.errors:
            f.write("\n## Errors Encountered\n\n")
            for error in result.errors:
                f.write(f"- {error}\n")
    
    # Save token report if available
    token_file = None
    if result.token_data:
        token_file = os.path.join(output_dir, f"{base_name}_tokens.txt")
        analyzer = RepositoryAnalyzer(Config())
        token_report = analyzer.generate_token_report(result.token_data)
        with open(token_file, 'w', encoding='utf-8') as f:
            f.write(token_report)
    
    # Save JSON only if requested
    json_file = None
    if save_json and result.token_data:
        json_file = os.path.join(output_dir, f"{base_name}_token_data.json")
        
        # Create structured data for analysis
        analysis_data = {
            "metadata": {
                "repo_name": result.repo_name,
                "branch": result.branch,
                "total_tokens": result.total_tokens,
                "total_files": result.total_files,
                "timestamp": timestamp
            },
            "files": [
                {
                    "path": path,
                    "tokens": tokens,
                    "directory": os.path.dirname(path),
                    "filename": os.path.basename(path),
                    "extension": os.path.splitext(path)[1],
                    "depth": path.count(os.sep)
                }
                for path, tokens in result.token_data.items()
            ],
            "directories": {}
        }
        
        # Aggregate directory data
        for file_data in analysis_data["files"]:
            dir_path = file_data["directory"]
            while dir_path:
                if dir_path not in analysis_data["directories"]:
                    analysis_data["directories"][dir_path] = {
                        "path": dir_path,
                        "total_tokens": 0,
                        "file_count": 0,
                        "depth": dir_path.count(os.sep),
                        "parent": os.path.dirname(dir_path) if os.sep in dir_path else None
                    }
                analysis_data["directories"][dir_path]["total_tokens"] += file_data["tokens"]
                analysis_data["directories"][dir_path]["file_count"] += 1
                dir_path = os.path.dirname(dir_path)
                if not dir_path or dir_path == ".":
                    break
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2)
    
    return main_file, token_file, json_file


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Repo2Txt - Analyze GitHub repositories and local folders',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s https://github.com/owner/repo
  %(prog)s /path/to/local/repo
  %(prog)s . --no-tokens
  %(prog)s https://github.com/owner/repo --json
  %(prog)s . --output-dir ./reports --json
        """
    )
    
    parser.add_argument('repo', help='GitHub repository URL or local directory path')
    parser.add_argument('--no-tokens', action='store_true', help='Disable token counting')
    parser.add_argument('--json', action='store_true', help='Export token data as JSON for analysis')
    parser.add_argument('--output-dir', default=None, help='Output directory for results (default: repository name)')
    parser.add_argument('--max-file-size', type=int, default=1024*1024, help='Maximum file size in bytes (default: 1MB)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    # Create configuration
    config = Config(
        enable_token_counting=not args.no_tokens and tiktoken is not None,
        max_file_size=args.max_file_size
    )
    
    # Determine repository name for default output directory
    if os.path.isdir(args.repo):
        # Local repository
        repo_name = os.path.basename(os.path.abspath(args.repo))
        if repo_name == '.' or repo_name == '':
            # If analyzing current directory, use the actual directory name
            repo_name = os.path.basename(os.getcwd())
    else:
        # GitHub repository
        parts = args.repo.replace('https://github.com/', '').strip('/').split('/')
        if len(parts) >= 2:
            repo_name = parts[1]  # Use repository name from URL
        else:
            repo_name = 'repo_analysis'  # Fallback
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else repo_name
    
    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    try:
        # Run analysis
        analyzer = RepositoryAnalyzer(config)
        result = analyzer.analyze(args.repo)
        
        # Save results
        main_file, token_file, json_file = save_results(result, output_dir, save_json=args.json)
        
        print(f"\n{'='*80}")
        print("Analysis Complete!")
        print(f"{'='*80}")
        print(f"Output directory: {output_dir}/")
        print(f"  Main analysis: {os.path.basename(main_file)}")
        if token_file:
            print(f"  Token report: {os.path.basename(token_file)}")
        if json_file:
            print(f"  Token data (JSON): {os.path.basename(json_file)}")
        print(f"\nTotal files analyzed: {result.total_files}")
        if result.total_tokens > 0:
            print(f"Total tokens: {result.total_tokens:,}")
        if result.errors:
            print(f"Errors encountered: {len(result.errors)}")
        
        if json_file:
            print(f"\n💡 Data Analysis Tip:")
            print(f"   Load JSON in Python: ")
            print(f"   with open('{os.path.join(output_dir, os.path.basename(json_file))}') as f:")
            print(f"       data = json.load(f)")
        
    except KeyboardInterrupt:
        print("\n\nAnalysis cancelled by user.")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Analysis failed: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()