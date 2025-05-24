"""Local filesystem repository adapter implementation."""
import os
from typing import Optional, List, Set, Dict, Tuple
from tqdm import tqdm

from ..core.models import Config, FileNode
from ..core.file_analyzer import FileAnalyzer
from .base import RepositoryAdapter


class LocalAdapter(RepositoryAdapter):
    """Adapter for analyzing local repositories."""
    
    def __init__(self, repo_path: str, config: Config):
        """Initialize local adapter with repository path."""
        super().__init__(config)
        
        if not os.path.isdir(repo_path):
            raise ValueError(f"Path is not a directory: {repo_path}")
        
        self.repo_path = os.path.abspath(repo_path)
        self.repo_name = os.path.basename(self.repo_path)
        
        print(f"[>] Analyzing local repository: {self.repo_name}")
    
    def get_name(self) -> str:
        """Get repository name."""
        return self.repo_name
    
    def get_readme_content(self) -> str:
        """Get README content from repository."""
        readme_names = ['README.md', 'readme.md', 'README.MD', 'README', 
                       'readme', 'README.txt', 'readme.txt']
        
        for readme_name in readme_names:
            readme_path = os.path.join(self.repo_path, readme_name)
            if os.path.exists(readme_path):
                content, error = self.file_analyzer.read_file_content(readme_path)
                if content:
                    return content
                elif error:
                    self.errors.append(f"README {readme_name}: {error}")
        
        return "README not found."
    
    def list_contents(self, path: str = "") -> List[Tuple[str, str, int]]:
        """List contents of a directory."""
        full_path = os.path.join(self.repo_path, path)
        
        try:
            items = []
            for item in sorted(os.listdir(full_path)):
                if item.startswith('.') and item not in {'.github', '.gitlab'}:
                    continue  # Skip hidden files except certain dirs
                    
                item_path = os.path.join(full_path, item)
                if os.path.isdir(item_path):
                    if item not in self.config.excluded_dirs:
                        items.append((item, 'dir', 0))
                else:
                    size = os.path.getsize(item_path)
                    items.append((item, 'file', size))
            
            return items
            
        except Exception as e:
            self.errors.append(f"Error listing contents of {path}: {str(e)}")
            return []
    
    def traverse_interactive(self) -> Tuple[str, Set[str], Dict[str, int]]:
        """Interactive traversal of repository."""
        return self._traverse_interactive_impl("")
    
    def _traverse_interactive_impl(self, current_path: str) -> Tuple[str, Set[str], Dict[str, int]]:
        """Implementation of interactive traversal with navigation stack."""
        navigation_stack = []
        structure = ""
        selected_paths = set()
        token_data = {}
        
        current_state = {
            'path': current_path,
            'structure': "",
            'selected_paths': set(),
            'token_data': {}
        }
        
        while True:
            full_path = os.path.join(self.repo_path, current_state['path'])
            
            try:
                # Get items in current directory
                items = []
                for item in sorted(os.listdir(full_path)):
                    if item.startswith('.') and item not in {'.github', '.gitlab'}:
                        continue
                        
                    item_path = os.path.join(full_path, item)
                    if os.path.isdir(item_path):
                        if item not in self.config.excluded_dirs:
                            items.append((item, 'dir'))
                    else:
                        items.append((item, 'file'))
                
                if not items:
                    if navigation_stack:
                        print("[<] Empty directory, going back...")
                        current_state = navigation_stack.pop()
                        continue
                    else:
                        return structure, selected_paths, token_data
                
                # Display items with better spacing
                print(f"\n[>] Contents of {current_state['path'] or 'root'}:\n")
                for i, (item, item_type) in enumerate(items, start=1):
                    if item_type == 'file' and self.config.enable_token_counting:
                        # Show token estimate for files
                        item_path = os.path.join(full_path, item)
                        content, _ = self.file_analyzer.read_file_content(item_path)
                        if content:
                            tokens = self.file_analyzer.count_tokens(content)
                            print(f"    {i:2d}.   {item:<28} ~{tokens:,} tokens")
                        else:
                            print(f"    {i:2d}.   {item:<28} binary")
                    elif item_type == 'dir' and self.config.enable_token_counting:
                        # Show aggregate token estimates for directories
                        dir_path = os.path.join(full_path, item)
                        dir_tokens = self._estimate_directory_tokens(dir_path)
                        if dir_tokens > 0:
                            print(f"    {i:2d}. ▸ {item:<28} ~{dir_tokens:,} tokens")
                        else:
                            print(f"    {i:2d}. ▸ {item:<28} empty")
                    else:
                        # Token counting disabled or just show item
                        if item_type == 'dir':
                            print(f"    {i:2d}. ▸ {item}")
                        else:
                            print(f"    {i:2d}.   {item}")
                
                # Show current selection status if tokens are being counted
                if self.config.enable_token_counting and (selected_paths or token_data):
                    total_tokens = sum(token_data.values())
                    print(f"\n[info]Selected: {len(selected_paths)} files | {total_tokens:,} tokens[/info]")
                
                # Show navigation options
                print("\n[>] Options: Enter numbers (e.g., 1-5,7), 'a' for all, 's' to skip", end="")
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
                
                for i, (item, item_type) in enumerate(items, start=1):
                    item_path = os.path.join(full_path, item)
                    rel_item_path = os.path.relpath(item_path, self.repo_path)
                    
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
                        # Not selected
                        if item_type == 'dir':
                            temp_structure += f"{rel_item_path}/ (not selected)\n"
                        else:
                            temp_structure += f"{rel_item_path} (not selected)\n"
                
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
                                    subdir_path
                                )
                                structure += sub_structure
                                selected_paths.update(sub_selected)
                                token_data.update(sub_tokens)
                                
                                if navigation_stack:
                                    navigation_stack.pop()
                                    
                            elif sub_choice == 'a':
                                sub_structure, sub_selected, sub_tokens = self._get_all_files(
                                    subdir_path
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
    
    def _get_all_files(self, path: str) -> Tuple[str, Set[str], Dict[str, int]]:
        """Recursively get all files from a directory."""
        structure = ""
        selected_paths = set()
        token_data = {}
        
        full_path = os.path.join(self.repo_path, path)
        
        try:
            for item in sorted(os.listdir(full_path)):
                if item.startswith('.') and item not in {'.github', '.gitlab'}:
                    continue
                    
                item_path = os.path.join(full_path, item)
                rel_item_path = os.path.relpath(item_path, self.repo_path)
                
                if os.path.isdir(item_path):
                    if item not in self.config.excluded_dirs:
                        structure += f"{rel_item_path}/\n"
                        sub_structure, sub_selected, sub_tokens = self._get_all_files(
                            rel_item_path
                        )
                        structure += sub_structure
                        selected_paths.update(sub_selected)
                        token_data.update(sub_tokens)
                else:  # file
                    if not self.file_analyzer.is_binary_file(item_path):
                        if os.path.getsize(item_path) <= self.config.max_file_size:
                            structure += f"{rel_item_path}\n"
                            selected_paths.add(rel_item_path)
                            
                            # Count tokens if enabled
                            if self.config.enable_token_counting:
                                content, error = self.file_analyzer.read_file_content(item_path)
                                if content:
                                    tokens = self.file_analyzer.count_tokens(content)
                                    if tokens > 0:
                                        token_data[rel_item_path] = tokens
        except Exception as e:
            self.errors.append(f"Error accessing {path}: {str(e)}")
        
        return structure, selected_paths, token_data
    
    def _estimate_directory_tokens(self, dir_path: str) -> int:
        """Quickly estimate total tokens in a directory."""
        total_tokens = 0
        
        try:
            for root, dirs, files in os.walk(dir_path):
                # Skip excluded directories
                dirs[:] = [d for d in dirs if d not in self.config.excluded_dirs]
                
                for file in files:
                    if file.startswith('.'):
                        continue
                        
                    file_path = os.path.join(root, file)
                    
                    # Skip binary files
                    if self.file_analyzer.is_binary_file(file_path):
                        continue
                    
                    # Skip large files
                    if os.path.getsize(file_path) > self.config.max_file_size:
                        continue
                    
                    # Read and count tokens
                    content, _ = self.file_analyzer.read_file_content(file_path)
                    if content:
                        total_tokens += self.file_analyzer.count_tokens(content)
                        
        except Exception:
            # Silently handle errors during estimation
            pass
            
        return total_tokens
    
    def get_file_content(self, file_path: str) -> Tuple[Optional[str], Optional[str]]:
        """Get content of a specific file."""
        full_path = os.path.join(self.repo_path, file_path)
        
        if not os.path.exists(full_path):
            return None, f"File not found: {file_path}"
        
        if os.path.isdir(full_path):
            return None, f"Not a file: {file_path}"
        
        return self.file_analyzer.read_file_content(full_path)
    
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
        
        print(f"\n[>] Processing {len(selected_files)} selected files...")
        
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