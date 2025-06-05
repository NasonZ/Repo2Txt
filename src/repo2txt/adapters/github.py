"""GitHub repository adapter implementation."""
import os
from typing import Optional, List, Set, Dict, Tuple
from github import Github, GithubException
from github.Repository import Repository as GithubRepo
from github.ContentFile import ContentFile
from tqdm import tqdm

from ..core.models import Config, FileNode
from ..core.file_analyzer import FileAnalyzer
from .base import RepositoryAdapter


class GitHubAdapter(RepositoryAdapter):
    """Adapter for analyzing GitHub repositories."""
    
    def __init__(self, repo_url: str, config: Config):
        """Initialize GitHub adapter with repository URL."""
        super().__init__(config)
        
        if not config.github_token:
            raise ValueError("GitHub token not found. Set GITHUB_TOKEN environment variable.")
        
        # Parse GitHub URL
        parts = repo_url.replace('https://github.com/', '').strip('/').split('/')
        if len(parts) < 2:
            raise ValueError("Invalid GitHub URL format")
        
        self.owner = parts[0]
        self.repo_name = parts[1]
        
        # Initialize GitHub client and repository
        self.github = Github(config.github_token)
        self.repo = self.github.get_repo(f"{self.owner}/{self.repo_name}")
        self.branch = None
        
        print(f"|>| Connected to GitHub repository: {self.owner}/{self.repo_name}")
    
    def get_name(self) -> str:
        """Get repository name."""
        return self.repo.name
    
    def select_branch(self) -> Optional[str]:
        """Allow user to select a branch."""
        try:
            branches = [b.name for b in self.repo.get_branches()]
            print(f"\n|>| Available branches: {', '.join(branches[:10])}")
            if len(branches) > 10:
                print(f"    ... and {len(branches) - 10} more")
            
            branch_input = input("\n[?] Enter branch name (or press Enter for default): ").strip()
            if branch_input and branch_input in branches:
                self.branch = branch_input
                return self.branch
            elif branch_input:
                print(f"[!] Branch '{branch_input}' not found. Using default branch.")
            
            # Use default branch
            self.branch = self.repo.default_branch
            return self.branch
            
        except Exception as e:
            self.errors.append(f"Error fetching branches: {str(e)}")
            return None
    
    def get_readme_content(self) -> str:
        """Get README content from repository."""
        readme_names = ['README.md', 'readme.md', 'README.MD', 'README', 
                       'readme', 'README.txt', 'readme.txt']
        
        for readme_name in readme_names:
            try:
                readme = self.repo.get_contents(readme_name, ref=self.branch)
                return readme.decoded_content.decode('utf-8')
            except:
                continue
        
        return "README not found."
    
    def list_contents(self, path: str = "") -> List[Tuple[str, str, int]]:
        """List contents of a directory."""
        try:
            contents = self.repo.get_contents(path, ref=self.branch)
            items = []
            
            for content in sorted(contents, key=lambda x: (x.type != 'dir', x.name)):
                # Apply same filtering as local adapter
                if content.name.startswith('.') and content.name not in {'.github', '.gitlab'}:
                    continue  # Skip hidden files except certain dirs
                    
                if content.type == 'dir' and content.name not in self.config.excluded_dirs:
                    items.append((content.name, 'dir', 0))
                elif content.type == 'file':
                    items.append((content.name, 'file', content.size))
            
            return items
            
        except Exception as e:
            self.errors.append(f"Error listing contents of {path}: {str(e)}")
            return []
    
    def traverse_interactive(self) -> Tuple[str, Set[str], Dict[str, int]]:
        """Interactive traversal of repository."""
        return self._traverse_interactive_impl("", self.branch)
    
    def _traverse_interactive_impl(self, path: str, branch: Optional[str]) -> Tuple[str, Set[str], Dict[str, int]]:
        """Implementation of interactive traversal with navigation stack."""
        navigation_stack = []
        structure = ""
        selected_paths = set()
        token_data = {}
        
        current_state = {
            'path': path,
            'structure': "",
            'selected_paths': set(),
            'token_data': {}
        }
        
        while True:
            try:
                contents = self.repo.get_contents(current_state['path'], ref=branch)
                
                # Sort and filter contents
                items = []
                for content in sorted(contents, key=lambda x: (x.type != 'dir', x.name)):
                    if content.type == 'dir' and content.name not in self.config.excluded_dirs:
                        items.append((content, 'dir'))
                    elif content.type == 'file':
                        items.append((content, 'file'))
                
                if not items:
                    if navigation_stack:
                        print("|<| Empty directory, going back...")
                        current_state = navigation_stack.pop()
                        continue
                    else:
                        return structure, selected_paths, token_data
                
                # Display items with better spacing
                print(f"\n|>| Contents of {current_state['path'] or 'root'}:\n")
                for i, (content, item_type) in enumerate(items, start=1):
                    if item_type == 'file' and self.config.enable_token_counting:
                        # Show token estimate for files
                        if content.encoding != 'none' and not self.file_analyzer.is_binary_file(content.path):
                            try:
                                file_content = content.decoded_content.decode('utf-8')
                                tokens = self.file_analyzer.count_tokens(file_content)
                                print(f"    {i:2d}.   {content.name:<28} ~{tokens:,} tokens")
                            except:
                                print(f"    {i:2d}.   {content.name:<28} decode error")
                        else:
                            print(f"    {i:2d}.   {content.name:<28} binary")
                    elif item_type == 'dir' and self.config.enable_token_counting:
                        # Show aggregate token estimates for directories
                        dir_tokens = self._estimate_directory_tokens(content.path, branch)
                        if dir_tokens > 0:
                            print(f"    {i:2d}. ▸ {content.name:<28} ~{dir_tokens:,} tokens")
                        else:
                            print(f"    {i:2d}. ▸ {content.name:<28} empty")
                    else:
                        # Token counting disabled or just show item
                        if item_type == 'dir':
                            print(f"    {i:2d}. ▸ {content.name}")
                        else:
                            print(f"    {i:2d}.   {content.name}")
                
                # Show current selection status if tokens are being counted
                if self.config.enable_token_counting and (selected_paths or token_data):
                    total_tokens = sum(token_data.values())
                    print(f"\n[info]Selected: {len(selected_paths)} files | {total_tokens:,} tokens[/info]")
                
                # Show navigation options
                print("\n|>| Options: Enter numbers (e.g., 1-5,7), 'a' for all, 's' to skip", end="")
                if navigation_stack:
                    print(", 'b' to go back", end="")
                print(", 'q' to quit")
                
                # Get user selection
                while True:
                    selection = input("[?] Your choice: ").strip().lower()
                    
                    # Handle navigation commands
                    if selection == 'q':
                        confirm = input("[?] Quit selection? You'll lose current selections. (y/n): ").lower()
                        if confirm == 'y':
                            return "", set(), {}
                    
                    if selection == 'b' and navigation_stack:
                        current_state = navigation_stack.pop()
                        break
                    
                    if selection == 's':
                        if navigation_stack:
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
                        print("[!] Invalid input. Please try again.")
                
                if selection in ['b', 'q']:
                    continue
                
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
                
                # Handle subdirectories
                for subdir_name, subdir_path in subdirs_to_explore:
                    while True:
                        print(f"\n[?] Select items in '{subdir_name}'?")
                        sub_choice = input("    Options: (y)es, (n)o, (a)ll, (b)ack: ").lower()
                        
                        if sub_choice == 'b':
                            break
                        
                        if sub_choice in ['y', 'n', 'a']:
                            if sub_choice == 'y':
                                navigation_stack.append({
                                    'path': current_state['path'],
                                    'structure': structure,
                                    'selected_paths': selected_paths.copy(),
                                    'token_data': token_data.copy()
                                })
                                
                                sub_structure, sub_selected, sub_tokens = self._traverse_interactive_impl(
                                    subdir_path, branch
                                )
                                structure += sub_structure
                                selected_paths.update(sub_selected)
                                token_data.update(sub_tokens)
                                
                                if navigation_stack:
                                    navigation_stack.pop()
                                    
                            elif sub_choice == 'a':
                                sub_structure, sub_selected, sub_tokens = self._get_all_files(
                                    subdir_path, branch
                                )
                                structure += sub_structure
                                selected_paths.update(sub_selected)
                                token_data.update(sub_tokens)
                            break
                        else:
                            print("[!] Invalid choice. Please enter 'y', 'n', 'a', or 'b'.")
                
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
    
    def _get_all_files(self, path: str, branch: Optional[str]) -> Tuple[str, Set[str], Dict[str, int]]:
        """Recursively get all files from a directory."""
        structure = ""
        selected_paths = set()
        token_data = {}
        
        try:
            contents = self.repo.get_contents(path, ref=branch)
            for content in contents:
                if content.type == "dir":
                    if content.name not in self.config.excluded_dirs:
                        structure += f"{content.path}/\n"
                        sub_structure, sub_selected, sub_tokens = self._get_all_files(
                            content.path, branch
                        )
                        structure += sub_structure
                        selected_paths.update(sub_selected)
                        token_data.update(sub_tokens)
                else:  # file
                    if not self.file_analyzer.is_binary_file(content.path):
                        if content.size <= self.config.max_file_size:
                            structure += f"{content.path}\n"
                            selected_paths.add(content.path)
                            
                            # Count tokens if enabled
                            if self.config.enable_token_counting and content.encoding != 'none':
                                try:
                                    file_content = content.decoded_content.decode('utf-8')
                                    tokens = self.file_analyzer.count_tokens(file_content)
                                    if tokens > 0:
                                        token_data[content.path] = tokens
                                except Exception:
                                    pass
        except Exception as e:
            self.errors.append(f"Error accessing {path}: {str(e)}")
        
        return structure, selected_paths, token_data
    
    def _estimate_directory_tokens(self, path: str, branch: Optional[str]) -> int:
        """Quickly estimate total tokens in a GitHub directory."""
        total_tokens = 0
        
        try:
            contents = self.repo.get_contents(path, ref=branch)
            
            for content in contents:
                if content.type == "dir":
                    if content.name not in self.config.excluded_dirs:
                        # Recursively estimate subdirectories
                        total_tokens += self._estimate_directory_tokens(content.path, branch)
                else:  # file
                    # Skip binary files
                    if self.file_analyzer.is_binary_file(content.path):
                        continue
                    
                    # Skip large files
                    if content.size > self.config.max_file_size:
                        continue
                    
                    # Skip files with no encoding
                    if content.encoding == 'none':
                        continue
                    
                    # Try to decode and count tokens
                    try:
                        file_content = content.decoded_content.decode('utf-8')
                        total_tokens += self.file_analyzer.count_tokens(file_content)
                    except:
                        # Skip files that can't be decoded
                        pass
                        
        except Exception:
            # Silently handle errors during estimation
            pass
            
        return total_tokens
    
    def get_file_content(self, file_path: str) -> Tuple[Optional[str], Optional[str]]:
        """Get content of a specific file."""
        try:
            content_obj = self.repo.get_contents(file_path, ref=self.branch)
            
            if content_obj.type != 'file':
                return None, f"Not a file: {file_path}"
            
            if self.file_analyzer.is_binary_file(file_path):
                return None, "Binary file"
            
            if content_obj.size > self.config.max_file_size:
                return None, f"File too large ({content_obj.size:,} bytes)"
            
            if content_obj.encoding == 'none':
                return None, "No encoding available"
            
            # Try to decode content
            for encoding in self.config.encoding_fallbacks:
                try:
                    content = content_obj.decoded_content.decode(encoding)
                    return content, None
                except UnicodeDecodeError:
                    continue
            
            return None, "Unable to decode file"
            
        except Exception as e:
            return None, str(e)
    
    def _format_file_content(self, file_path: str, content: Optional[str], error: Optional[str]) -> str:
        """Format file content based on output format setting."""
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
    
    def get_file_contents(self, selected_files: List[str]) -> Tuple[str, Dict[str, int]]:
        """Get contents of selected files."""
        file_contents = ""
        token_data = {}
        
        print(f"\n|>| Processing {len(selected_files)} selected files...")
        
        for file_path in tqdm(selected_files, desc="Reading files"):
            content, error = self.get_file_content(file_path)
            
            if content:
                file_contents += self._format_file_content(file_path, content, None)
                
                if self.config.enable_token_counting:
                    tokens = self.file_analyzer.count_tokens(content)
                    if tokens > 0:
                        token_data[file_path] = tokens
            else:
                file_contents += self._format_file_content(file_path, None, error)
        
        return file_contents, token_data
    
    def build_file_tree(self) -> str:
        """
        Build a text representation of the repository file tree.
        
        Returns:
            String representation of the file tree structure.
        """
        tree_lines = []
        
        def _build_tree_recursive(current_path: str, prefix: str = "", is_last: bool = True):
            """Recursively build tree structure."""
            try:
                items = self.list_contents(current_path)
                # Sort directories first, then files, alphabetically
                dirs = [(name, type_, size) for name, type_, size in items if type_ == 'dir']
                files = [(name, type_, size) for name, type_, size in items if type_ == 'file']
                
                all_items = sorted(dirs) + sorted(files)
                
                for i, (name, type_, size) in enumerate(all_items):
                    is_item_last = (i == len(all_items) - 1)
                    connector = "└── " if is_item_last else "├── "
                    tree_lines.append(f"{prefix}{connector}{name}")
                    
                    if type_ == 'dir':
                        # Add to tree recursively
                        extension = "    " if is_item_last else "│   "
                        new_path = os.path.join(current_path, name) if current_path else name
                        _build_tree_recursive(new_path, prefix + extension, is_item_last)
                        
            except Exception as e:
                self.errors.append(f"Error building tree for {current_path}: {str(e)}")
        
        # Start from root
        _build_tree_recursive("")
        return "\n".join(tree_lines)
    
    def get_file_list(self) -> List[str]:
        """
        Get a list of all files in the repository.
        
        Returns:
            List of file paths relative to repository root.
        """
        file_list = []
        
        def _collect_files(current_path: str):
            """Recursively collect all files."""
            try:
                items = self.list_contents(current_path)
                
                for name, type_, size in items:
                    item_path = os.path.join(current_path, name) if current_path else name
                    
                    if type_ == 'file':
                        file_list.append(item_path)
                    elif type_ == 'dir':
                        _collect_files(item_path)
                        
            except Exception as e:
                self.errors.append(f"Error collecting files from {current_path}: {str(e)}")
        
        # Start from root
        _collect_files("")
        return sorted(file_list)