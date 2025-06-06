"""GitHub repository adapter implementation."""
import os
import asyncio
import aiohttp
import base64
from typing import Optional, List, Set, Dict, Tuple
from github import Github, GithubException
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass

try:
    from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from ..core.models import Config, FileNode
from ..core.file_analyzer import FileAnalyzer
from .base import RepositoryAdapter


@dataclass
class FileResult:
    """Clean result container for file processing."""
    path: str
    content: Optional[str] = None
    tokens: int = 0
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        return self.error is None


class AsyncGitHubClient:
    """Clean async GitHub API client with proper resource management."""
    
    def __init__(self, token: str, owner: str, repo: str, branch: str, config: Config):
        self.token = token
        self.owner = owner
        self.repo = repo
        self.branch = branch
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.semaphore = asyncio.Semaphore(20)  # Rate limiting
        self.file_analyzer = FileAnalyzer(config)
    
    async def __aenter__(self):
        """Async context manager entry with session setup."""
        headers = {
            'Authorization': f'token {self.token}',
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'repo2txt'
        }
        connector = aiohttp.TCPConnector(
            limit=config.connection_limit, 
            limit_per_host=config.connection_limit_per_host
        )
        self.session = aiohttp.ClientSession(headers=headers, connector=connector)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with guaranteed cleanup."""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def fetch_file(self, file_path: str) -> FileResult:
        """Fetch a single file with proper error handling."""
        async with self.semaphore:
            try:
                # Normalize path for GitHub API
                api_path = file_path.replace('\\', '/')
                url = f"https://api.github.com/repos/{self.owner}/{self.repo}/contents/{api_path}"
                
                params = {}
                if self.branch:
                    params['ref'] = self.branch
                
                async with self.session.get(url, params=params) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        return FileResult(file_path, error=f"HTTP {response.status}: {error_text}")
                    
                    data = await response.json()
                    
                    # Validate file type
                    if data.get('type') != 'file':
                        return FileResult(file_path, error=f"Not a file: {file_path}")
                    
                    # Check file size
                    file_size = data.get('size', 0)
                    if file_size > self.config.max_file_size:
                        return FileResult(file_path, error=f"File too large ({file_size:,} bytes)")
                    
                    # Decode content
                    encoding = data.get('encoding')
                    if encoding != 'base64':
                        return FileResult(file_path, error=f"Unsupported encoding: {encoding}")
                    
                    content_b64 = data.get('content', '').replace('\n', '')
                    content_bytes = base64.b64decode(content_b64)
                    
                    # Check if binary
                    if self.file_analyzer.is_binary_file(file_path, content_bytes):
                        return FileResult(file_path, error="Binary file")
                    
                    # Decode as text
                    for encoding_name in self.config.encoding_fallbacks:
                        try:
                            content = content_bytes.decode(encoding_name)
                            return FileResult(file_path, content=content)
                        except UnicodeDecodeError:
                            continue
                    
                    return FileResult(file_path, error="Unable to decode file")
                    
            except Exception as e:
                # prevent token exposure
                safe_error = str(e)
                if self.token and self.token in safe_error:
                    safe_error = safe_error.replace(self.token, "[REDACTED]")
                return FileResult(file_path, error=f"Network error: {safe_error}")


async def process_files_batch(
    github_client: AsyncGitHubClient,
    file_paths: List[str],
    enable_token_counting: bool = True,
    progress_callback=None
) -> Dict[str, FileResult]:
    """Process files in batches with clean separation of concerns."""
    results = {}
    
    # Step 1: Fetch all files concurrently
    tasks = [github_client.fetch_file(path) for path in file_paths]
    file_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Step 2: Process results and count tokens
    for i, result in enumerate(file_results):
        if isinstance(result, Exception):
            results[file_paths[i]] = FileResult(file_paths[i], error=str(result))
        else:
            # Count tokens if enabled and content exists
            if enable_token_counting and result.content:
                result.tokens = github_client.file_analyzer.count_tokens(result.content)
            
            results[result.path] = result
            
            # Progress callback
            if progress_callback:
                progress_callback(i + 1, len(file_paths))
    
    return results


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
            
            # If there's only one branch, use it automatically
            if len(branches) == 1:
                self.branch = branches[0]
                print(f"|>| Using branch: {self.branch}")
                return self.branch
            
            # Multiple branches - show options and ask user to choose
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
            safe_error = self._sanitize_error(str(e), [self.config.github_token])
            self.errors.append(f"Error fetching branches: {safe_error}")
            return None
    
    def get_readme_content(self) -> str:
        """Get README content from repository."""
        readme_names = ['README.md', 'readme.md', 'README.MD', 'README', 
                       'readme', 'README.txt', 'readme.txt']
        
        for readme_name in readme_names:
            try:
                readme = self.repo.get_contents(readme_name, ref=self.branch)
                return readme.decoded_content.decode('utf-8')
            except Exception:
                # README file not found or inaccessible, try next one
                continue
        
        return "README not found."
    
    def list_contents(self, path: str = "") -> List[Tuple[str, str, int]]:
        """List contents of a directory."""
        try:
            contents = self.repo.get_contents(path, ref=self.branch)
            items = []
            
            # Calculate expected path depth for direct children
            if path:
                expected_prefix = path + "/"
                expected_depth = path.count('/') + 2  # path depth + 1 for the child
            else:
                expected_prefix = ""
                expected_depth = 1  # root level children
            
            for content in sorted(contents, key=lambda x: (x.type != 'dir', x.name)):
                # Only include direct children by checking path depth
                content_depth = content.path.count('/') + 1
                if content_depth != expected_depth:
                    continue
                
                # Verify this is actually a direct child of the requested path
                if path and not content.path.startswith(expected_prefix):
                    continue
                    
                # Apply same filtering as local adapter
                if content.name.startswith('.') and content.name not in {'.gitlab'}:
                    continue  # Skip hidden files except certain dirs
                    
                if content.type == 'dir' and content.name not in self.config.excluded_dirs:
                    items.append((content.name, 'dir', 0))
                elif content.type == 'file':
                    items.append((content.name, 'file', content.size))
            return items
            
        except Exception as e:
            safe_error = self._sanitize_error(str(e), [self.config.github_token])
            self.errors.append(f"Error listing contents of {path}: {safe_error}")
            return []
    
    def traverse_interactive(self) -> Tuple[str, Set[str], Dict[str, int]]:
        """Interactive traversal of repository."""
        # Pre-scan the repository to build file tree with token counts
        print("|>| Scanning repository structure...")
        self._cached_file_tree = self._build_file_tree_with_tokens()
        
        # Build a lookup table for quick access to nodes by path
        self._path_to_node = {}
        self._build_path_lookup(self._cached_file_tree)
        
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
                # Get items from cached file tree
                items = []
                
                # Get the current node from the cached tree
                current_node = self._path_to_node.get(current_state['path'], None)
                if current_node is None and current_state['path'] == "":
                    # Root node
                    current_node = self._cached_file_tree
                
                if current_node and current_node.type == 'dir':
                    # Create content objects from cached node children
                    class CachedContent:
                        def __init__(self, node: FileNode):
                            self.name = node.name
                            self.path = node.path
                            self.type = 'dir' if node.type == 'dir' else 'file'
                            self.size = node.size if node.type == 'file' else 0
                            self.encoding = 'base64' if node.type == 'file' else None
                            self.total_tokens = node.total_tokens
                    
                    for child in sorted(current_node.children, key=lambda x: (x.type != 'dir', x.name)):
                        content_obj = CachedContent(child)
                        items.append((content_obj, child.type))
                
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
                        # Show token estimate for files (already cached)
                        if hasattr(content, 'total_tokens') and content.total_tokens > 0:
                            print(f"    {i:2d}.   {content.name:<28} ~{content.total_tokens:,} tokens")
                        elif not self.file_analyzer.is_binary_file(content.path):
                            print(f"    {i:2d}.   {content.name:<28} ~0 tokens")
                        else:
                            print(f"    {i:2d}.   {content.name:<28} binary")
                    elif item_type == 'dir' and self.config.enable_token_counting:
                        # Show aggregate token estimates for directories (from cache)
                        if hasattr(content, 'total_tokens') and content.total_tokens > 0:
                            print(f"    {i:2d}. ▸ {content.name:<28} ~{content.total_tokens:,} tokens")
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
                            
                            # Use cached token count
                            if self.config.enable_token_counting and hasattr(content, 'total_tokens'):
                                if content.total_tokens > 0:
                                    temp_tokens[content.path] = content.total_tokens
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
                # Apply same hidden file filtering as local adapter
                if content.name.startswith('.') and content.name not in {'.gitlab'}:
                    continue  # Skip hidden files except certain dirs
                    
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
            file_paths = []
            
            # Collect all file paths first
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
                    
                    file_paths.append(content.path)
            
            # Use async processing for file content if we have many files
            if len(file_paths) > 5:
                # For estimation, just use synchronous processing to keep it simple
                for file_path in file_paths:
                    try:
                        content_obj = self.repo.get_contents(file_path, ref=branch)
                        file_content = content_obj.decoded_content.decode('utf-8')
                        total_tokens += self.file_analyzer.count_tokens(file_content)
                    except Exception:
                        # File inaccessible or binary, skip for estimation
                        continue
        except Exception:
            # Silently handle errors during estimation
            pass
            
        return total_tokens
    
    def get_file_content(self, file_path: str) -> Tuple[Optional[str], Optional[str]]:
        """Get content of a specific file."""
        try:
            # Normalize path separators for GitHub API (always use forward slashes)
            api_path = file_path.replace('\\', '/')
            content_obj = self.repo.get_contents(api_path, ref=self.branch)
            
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
    
    # Note: _format_file_content is now inherited from base class
    
    def get_file_contents(self, selected_files: List[str]) -> Tuple[str, Dict[str, int]]:
        """Get contents of selected files."""
        # Try async processing first
        try:
            # Simply call async processing directly
            file_contents_dict, token_data = self._run_async_processing(selected_files)
            
            # Format the content
            file_contents = ""
            for file_path in selected_files:
                content = file_contents_dict.get(file_path, f"# Error: File not found")
                if content.startswith("# Error:"):
                    file_contents += self._format_file_content(file_path, None, content[8:])
                else:
                    file_contents += self._format_file_content(file_path, content, None)
            
            return file_contents, token_data
            
        except Exception as e:
            print(f"[!] Async processing failed: {e}, falling back to synchronous processing...")
            # Fall back to synchronous processing
            return self._get_file_contents_sync(selected_files)

    def _run_async_processing(self, selected_files: List[str]) -> Tuple[Dict[str, str], Dict[str, int]]:
        """Simplified async entry point."""
        return asyncio.run(self._process_files_clean(selected_files))

    def _get_file_contents_sync(self, selected_files: List[str]) -> Tuple[str, Dict[str, int]]:
        """Synchronous fallback for get_file_contents."""
        file_contents = ""
        token_data = {}
        
        print(f"\n|>| Processing {len(selected_files)} selected files...")
        
        for file_path in tqdm(selected_files, desc="Reading files"):
            content, error = self.get_file_content(file_path)
            
            if content:
                file_contents += self._format_file_content(file_path, content, None)
                
                if self.config.enable_token_counting:
                    tokens = self.file_analyzer.count_tokens(content)
                    # Include all token counts, even 0, so tree display is accurate
                    token_data[file_path] = tokens
            else:
                file_contents += self._format_file_content(file_path, None, error)
        
        return file_contents, token_data
    
    def build_file_tree(self) -> FileNode:
        """
        Build a hierarchical FileNode tree structure of the repository.
        
        Returns:
            Root FileNode with children representing the file tree structure.
        """
        root = FileNode(
            path="",
            name=self.repo_name,
            type='dir'
        )
        self._build_tree_recursive("", root, self.branch)
        
        # Aggregate total tokens for the root directory
        root.total_tokens = sum(c.total_tokens for c in root.children)
        
        return root
    
    def _build_tree_recursive(self, path: str, node: FileNode, branch: Optional[str] = None) -> None:
        """Recursively build file tree from GitHub API."""
        try:
            # Use the instance branch if no branch specified
            if branch is None:
                branch = self.branch
                
            contents = self.repo.get_contents(path, ref=branch)
            if not isinstance(contents, list):
                contents = [contents]
            
            for content in contents:
                # Skip excluded directories
                if content.type == "dir" and content.name in self.config.excluded_dirs:
                    continue
                
                if content.type == "dir":
                    child = FileNode(
                        path=content.path,
                        name=content.name,
                        type='dir'
                    )
                    node.children.append(child)
                    self._build_tree_recursive(content.path, child, branch)
                    
                    # Calculate total tokens for directory
                    child.total_tokens = sum(c.total_tokens for c in child.children)
                else:  # file
                    # Skip binary files
                    ext = os.path.splitext(content.name)[1].lower()
                    if ext in self.config.binary_extensions:
                        continue
                    
                    # Check file size
                    if content.size and content.size > self.config.max_file_size:
                        continue
                    
                    # Skip token counting during tree building to avoid blocking HTTP requests
                    # Tree building should be fast - token counting will be done later when files are selected
                    tokens = 0
                    
                    child = FileNode(
                        path=content.path,
                        name=content.name,
                        type='file',
                        size=content.size,
                        token_count=tokens,
                        total_tokens=tokens
                    )
                    node.children.append(child)
        except Exception as e:
            self.errors.append(f"Error building tree for {path}: {e}")
    
    def _build_file_tree_with_tokens(self) -> FileNode:
        """Build file tree with actual token counts (for interactive mode)."""
        print("|>| Building file tree...")
        root = FileNode(
            path="",
            name=self.repo_name,
            type='dir'
        )
        
        # First build the tree structure
        self._build_tree_recursive("", root, self.branch)
        
        # Then count tokens for all files
        if self.config.enable_token_counting:
            print("|>| Counting tokens...")
            # First count total files
            total_files = self._count_files(root)
            # Then count tokens with progress
            self._files_processed = 0
            self._count_tokens_recursive_with_progress(root, total_files)
        
        # Aggregate total tokens for directories
        self._aggregate_directory_tokens(root)
        
        # Clear the progress line
        if self.config.enable_token_counting:
            print()  # New line after progress
        
        return root
    
    def _count_files(self, node: FileNode) -> int:
        """Count total number of files in the tree."""
        if node.type == 'file':
            return 1
        total = 0
        for child in node.children:
            total += self._count_files(child)
        return total
    
    def _count_tokens_recursive_with_progress(self, node: FileNode, total_files: int) -> None:
        """Recursively count tokens with progress tracking."""
        if node.type == 'file':
            try:
                # Get file content and count tokens
                content_obj = self.repo.get_contents(node.path, ref=self.branch)
                if content_obj.encoding != 'none' and not self.file_analyzer.is_binary_file(node.path):
                    file_content = content_obj.decoded_content.decode('utf-8')
                    node.token_count = self.file_analyzer.count_tokens(file_content)
                    node.total_tokens = node.token_count
            except Exception:
                node.token_count = 0
                node.total_tokens = 0
            
            # Update progress
            self._files_processed += 1
            if self._files_processed % 10 == 0 or self._files_processed == total_files:
                print(f"\r|>| Processed {self._files_processed}/{total_files} files...", end='', flush=True)
        else:
            # Process children first
            for child in node.children:
                self._count_tokens_recursive_with_progress(child, total_files)
    
    def _count_tokens_recursive(self, node: FileNode) -> None:
        """Recursively count tokens for all files in the tree."""
        if node.type == 'file':
            try:
                # Get file content and count tokens
                content_obj = self.repo.get_contents(node.path, ref=self.branch)
                if content_obj.encoding != 'none' and not self.file_analyzer.is_binary_file(node.path):
                    file_content = content_obj.decoded_content.decode('utf-8')
                    node.token_count = self.file_analyzer.count_tokens(file_content)
                    node.total_tokens = node.token_count
            except Exception:
                node.token_count = 0
                node.total_tokens = 0
        else:
            # Process children first
            for child in node.children:
                self._count_tokens_recursive(child)
    
    def _aggregate_directory_tokens(self, node: FileNode) -> int:
        """Aggregate token counts for directories (bottom-up)."""
        if node.type == 'file':
            return node.token_count
        
        total = 0
        for child in node.children:
            total += self._aggregate_directory_tokens(child)
        
        node.total_tokens = total
        return total
    
    def _build_path_lookup(self, node: FileNode, parent_path: str = "") -> None:
        """Build a path-to-node lookup table for quick access."""
        # Store this node
        self._path_to_node[node.path] = node
        
        # Recursively process children
        if node.type == 'dir':
            for child in node.children:
                self._build_path_lookup(child, node.path)
    
    def build_file_tree_string(self) -> str:
        """
        Build a text representation of the repository file tree.
        
        Returns:
            String representation of the file tree structure.
        """
        tree_lines = []
        
        def _build_tree_recursive(current_path: str = "", prefix: str = "", is_last: bool = True):
            """Recursively build tree structure."""
            try:
                contents = self.repo.get_contents(current_path)
                if not isinstance(contents, list):
                    contents = [contents]
                
                # Sort directories first, then files, alphabetically
                dirs = [c for c in contents if c.type == "dir" and c.name not in self.config.excluded_dirs]
                files = [c for c in contents if c.type == "file"]
                
                all_items = sorted(dirs, key=lambda x: x.name) + sorted(files, key=lambda x: x.name)
                
                for i, content in enumerate(all_items):
                    is_item_last = (i == len(all_items) - 1)
                    connector = "└── " if is_item_last else "├── "
                    tree_lines.append(f"{prefix}{connector}{content.name}")
                    
                    if content.type == "dir":
                        # Add to tree recursively
                        extension = "    " if is_item_last else "│   "
                        _build_tree_recursive(content.path, prefix + extension, is_item_last)
                        
            except Exception as e:
                self.errors.append(f"Error building tree for {current_path}: {str(e)}")
        
        # Start from root
        _build_tree_recursive()
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
                    if current_path:
                        item_path = f"{current_path}/{name}"  # Always use forward slashes for GitHub
                    else:
                        item_path = name
                    
                    if type_ == 'file':
                        file_list.append(item_path)
                    elif type_ == 'dir':
                        _collect_files(item_path)
                        
            except Exception as e:
                self.errors.append(f"Error collecting files from {current_path}: {str(e)}")
        
        # Start from root
        _collect_files("")
        return sorted(file_list)
    
    def build_file_tree_and_list(self) -> Tuple[str, List[str]]:
        """
        Efficiently build both file tree and file list in a single traversal.
        This avoids the duplicate repository scanning.
        
        Returns:
            Tuple of (file_tree_string, file_paths_list)
        """
        tree_lines = []
        file_paths = []
        
        def _traverse_recursive(current_path: str, prefix: str = "", is_last: bool = True, depth: int = 0):
            """Recursively traverse and build both tree and file list."""
            try:
                items = self.list_contents(current_path)  # This already has filtering
                if not items:
                    return
                
                # Sort directories first, then files, alphabetically
                dirs = [(name, type_, size) for name, type_, size in items if type_ == 'dir']
                files = [(name, type_, size) for name, type_, size in items if type_ == 'file']
                all_items = sorted(dirs) + sorted(files)
                
                for i, (name, type_, size) in enumerate(all_items):
                    is_item_last = (i == len(all_items) - 1)
                    connector = "└── " if is_item_last else "├── "
                    tree_lines.append(f"{prefix}{connector}{name}")
                    
                    if current_path:
                        item_path = f"{current_path}/{name}"  # Always use forward slashes for GitHub
                    else:
                        item_path = name
                    
                    if type_ == 'dir':
                        # Only recurse into directories that passed the filtering
                        # (list_contents already filtered out excluded dirs)
                        extension = "    " if is_item_last else "│   "
                        _traverse_recursive(item_path, prefix + extension, is_item_last, depth + 1)
                    else:  # file
                        # Add file to list
                        file_paths.append(item_path)
                        
            except Exception as e:
                self.errors.append(f"Error traversing {current_path}: {str(e)}")
        
        # Start from root
        _traverse_recursive("")
        
        tree_string = "\n".join(tree_lines)
        return tree_string, file_paths

    async def _process_files_clean(self, file_paths: List[str]) -> Tuple[Dict[str, str], Dict[str, int]]:
        """Clean async file processing with proper resource management."""
        file_contents = {}
        token_data = {}
        
        # Progress tracking setup
        progress_counter = 0
        total_files = len(file_paths)
        
        def update_progress(completed: int, total: int):
            nonlocal progress_counter
            progress_counter = completed
        
        async with AsyncGitHubClient(
            self.config.github_token, 
            self.owner, 
            self.repo_name, 
            self.branch,
            self.config
        ) as github_client:
            
            # Process with progress tracking
            if RICH_AVAILABLE:
                with Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TextColumn("{task.completed}/{task.total}"),
                    TimeElapsedColumn(),
                ) as progress:
                    task = progress.add_task("Processing files", total=total_files)
                    
                    def rich_progress(completed, total):
                        progress.update(task, completed=completed)
                    
                    results = await process_files_batch(
                        github_client, 
                        file_paths, 
                        self.config.enable_token_counting,
                        rich_progress
                    )
            else:
                # Fallback to simple counter
                results = await process_files_batch(
                    github_client, 
                    file_paths, 
                    self.config.enable_token_counting,
                    update_progress
                )
                print(f"Processed {total_files} files")
        
        # Convert results to expected format
        for path, result in results.items():
            if result.success:
                file_contents[path] = result.content
                token_data[path] = result.tokens
            else:
                file_contents[path] = f"# Error: {result.error}"
                token_data[path] = 0
        
        return file_contents, token_data
    
    def _sanitize_error(self, error_msg: str, secrets: List[str]) -> str:
        """Remove sensitive information from error messages."""
        sanitized = error_msg
        for secret in secrets:
            if secret and secret in sanitized:
                sanitized = sanitized.replace(secret, "[REDACTED]")
        return sanitized