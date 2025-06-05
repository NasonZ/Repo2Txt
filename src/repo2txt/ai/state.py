"""State management for AI file selection.

This module provides classes for managing the state of file selections,
including tracking selected files, token counts, and budget usage.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional, Tuple

from ..core.models import AnalysisResult


@dataclass
class FileSelectionState:
    """Holds the current file selection state."""
    selected_files: List[str] = field(default_factory=list)
    token_budget: int = 50000
    total_tokens_selected: int = 0
    # Add tracking for previous state to show diffs
    previous_files: List[str] = field(default_factory=list)
    previous_tokens: int = 0
    
    def get_budget_usage_percent(self) -> float:
        """Calculate budget usage percentage to 3 significant figures."""
        if self.token_budget > 0:
            return round((self.total_tokens_selected / self.token_budget) * 100, 3)
        else:
            return 0
    
    def get_budget_remaining(self) -> int:
        """Calculate remaining budget."""
        return self.token_budget - self.total_tokens_selected
    
    def is_over_budget(self) -> bool:
        """Check if current selection exceeds budget."""
        return self.total_tokens_selected > self.token_budget


class StateManager:
    """Manages file selection state with validation and token calculation."""
    
    def __init__(self, analysis_result: AnalysisResult, token_budget: int = 50000):
        self.analysis_result = analysis_result
        self.state = FileSelectionState(token_budget=token_budget)
        
        # Create lookup maps for fast validation
        self.available_files: Set[str] = {f['path'] for f in analysis_result.file_list}
        self.file_tokens: Dict[str, int] = {f['path']: f.get('tokens', 0) for f in analysis_result.file_list}
        
    def _save_previous_state(self):
        """Save current state as previous for diff tracking."""
        self.state.previous_files = self.state.selected_files.copy()
        self.state.previous_tokens = self.state.total_tokens_selected
        
    def replace_selection(self, paths: List[str], reasoning: str) -> Dict[str, Any]:
        """Replace entire selection with new paths."""
        # Save previous state for diff
        self._save_previous_state()
        
        # Validate paths
        valid_paths = []
        invalid_paths = []
        
        for path in paths:
            path = path.strip()
            if path in self.available_files:
                if path not in valid_paths:  # Avoid duplicates
                    valid_paths.append(path)
            else:
                invalid_paths.append(path)
        
        # Update state
        self.state.selected_files = sorted(valid_paths)
        self.state.total_tokens_selected = sum(self.file_tokens.get(path, 0) for path in valid_paths)
        
        # Build result with helpful feedback
        if invalid_paths and not valid_paths:
            # All paths were invalid
            feedback = (
                f"ERROR: None of the {len(invalid_paths)} paths were found in the repository. "
                f"Invalid paths: {', '.join(invalid_paths)}. "
                f"Please check the repository structure in the system prompt and correct these paths. "
                f"Common issues: 1) Missing 'src/' prefix, 2) Wrong directory nesting, 3) Typos in filenames. "
                f"Try again with corrected paths that exist in the shown file tree."
            )
        elif invalid_paths:
            # Some paths were invalid
            feedback = (
                f"Selected {len(valid_paths)} files ({self.state.total_tokens_selected:,} tokens). "
                f"WARNING: {len(invalid_paths)} paths were not found: {', '.join(invalid_paths)}. "
                f"Please verify these paths against the repository structure and either: "
                f"1) Correct them if they were typos/mistakes, or 2) Remove them if they don't exist. "
                f"Use adjust_selection to add the corrected paths."
            )
        else:
            # All paths were valid
            feedback = (
                f"Successfully selected {len(valid_paths)} files ({self.state.total_tokens_selected:,} tokens). "
                f"Budget usage: {self.state.get_budget_usage_percent():.1f}% "
                f"({self.state.total_tokens_selected:,}/{self.state.token_budget:,} tokens)"
            )
        
        return {
            "selected_paths": self.state.selected_files,
            "total_files_selected": len(self.state.selected_files),
            "total_tokens_selected": self.state.total_tokens_selected,
            "budget_remaining": self.state.get_budget_remaining(),
            "budget_usage_percent": self.state.get_budget_usage_percent(),
            "feedback": feedback,
            "invalid_paths": invalid_paths,
            "valid_paths": valid_paths
        }
    
    def modify_selection(self, add_files: Optional[List[str]] = None, 
                        remove_files: Optional[List[str]] = None, 
                        reasoning: str = "") -> Dict[str, Any]:
        """Modify current selection by adding/removing files."""
        # Save previous state for diff
        self._save_previous_state()
        
        add_files = add_files or []
        remove_files = remove_files or []
        
        current_selection = set(self.state.selected_files)
        added_actual, removed_actual, not_found = [], [], []
        
        # Remove files
        for path in remove_files:
            path = path.strip()
            if path in current_selection:
                current_selection.remove(path)
                removed_actual.append(path)
            else:
                not_found.append(f"{path} (not currently selected)")
        
        # Add files
        for path in add_files:
            path = path.strip()
            if path in self.available_files:
                if path not in current_selection:
                    current_selection.add(path)
                    added_actual.append(path)
            else:
                not_found.append(f"{path} (not found in repository)")
        
        # Update state
        self.state.selected_files = sorted(list(current_selection))
        self.state.total_tokens_selected = sum(self.file_tokens.get(path, 0) for path in self.state.selected_files)
        
        # Build helpful feedback
        if not_found and not added_actual and not removed_actual:
            # Only invalid operations
            feedback = (
                f"ERROR: No valid operations performed. Issues encountered: {', '.join(not_found)}. "
                f"Please check paths against the repository structure in the system prompt. "
                f"For additions, ensure paths exist in the file tree. "
                f"For removals, ensure files are currently selected."
            )
        elif not_found:
            # Some operations succeeded, some failed
            feedback_parts = []
            if added_actual or removed_actual:
                feedback_parts.append(f"Partially adjusted selection")
                if added_actual:
                    feedback_parts.append(f"Added {len(added_actual)} files")
                if removed_actual:
                    feedback_parts.append(f"Removed {len(removed_actual)} files")
            feedback_parts.append(f"ERRORS: {', '.join(not_found)}")
            feedback_parts.append("Please verify and retry with corrected paths")
            feedback = ". ".join(feedback_parts)
        else:
            # All operations succeeded
            feedback_parts = [f"Successfully adjusted selection"]
            if added_actual:
                tokens_added = sum(self.file_tokens.get(path, 0) for path in added_actual)
                feedback_parts.append(f"Added {len(added_actual)} files (+{tokens_added:,} tokens)")
            if removed_actual:
                tokens_removed = sum(self.file_tokens.get(path, 0) for path in removed_actual)
                feedback_parts.append(f"Removed {len(removed_actual)} files (-{tokens_removed:,} tokens)")
            feedback_parts.append(f"New total: {self.state.total_tokens_selected:,}/{self.state.token_budget:,} tokens ({self.state.get_budget_usage_percent():.1f}%)")
            feedback = ". ".join(feedback_parts)
        
        return {
            "selected_paths": self.state.selected_files,
            "total_files_selected": len(self.state.selected_files),
            "total_tokens_selected": self.state.total_tokens_selected,
            "budget_remaining": self.state.get_budget_remaining(),
            "budget_usage_percent": self.state.get_budget_usage_percent(),
            "feedback": feedback,
            "added_files": added_actual,
            "removed_files": removed_actual,
            "not_found": not_found
        }
    
    def get_current_selection_summary(self) -> str:
        """Get summary of current selection for system prompt."""
        if not self.state.selected_files:
            return "No files currently selected."
        
        selected_files_details = []
        for path_str in self.state.selected_files:
            tokens = self.file_tokens.get(path_str, 0)
            selected_files_details.append(f"  • {path_str} (~{tokens:,} tokens)")
        
        budget_percentage = self.state.get_budget_usage_percent()
        
        summary = [
            f"**Currently selected files ({len(self.state.selected_files)} files, ~{self.state.total_tokens_selected:,} tokens):**",
            *selected_files_details,
            f"**Budget Usage:** ~{self.state.total_tokens_selected:,}/{self.state.token_budget:,} tokens ({budget_percentage:.1f}%)"
        ]
        return "\n".join(summary)
    
    def clear_selection(self):
        """Clear all selected files."""
        self._save_previous_state()
        self.state.selected_files = []
        self.state.total_tokens_selected = 0


class TokenCache:
    """Cache token counts with file modification time checking.
    
    This cache helps avoid recounting tokens for files that haven't
    changed since the last count. It uses file modification times
    to automatically invalidate stale entries.
    """
    
    def __init__(self, encoding_name: str = "cl100k_base"):
        """Initialize the token cache.
        
        Args:
            encoding_name: The tiktoken encoding to use
        """
        self._cache: Dict[str, Tuple[float, int]] = {}  # path -> (mtime, token_count)
        self._counter = None  # Lazy initialization
        self._encoding_name = encoding_name
        self._hits = 0
        self._misses = 0
        
    def _ensure_counter(self):
        """Ensure token counter is initialized."""
        if self._counter is None:
            from ..core.tokenizer import TokenCounter
            self._counter = TokenCounter(self._encoding_name)
        
    def get_token_count(self, file_path: str, content: Optional[str] = None) -> int:
        """Get token count for a file, using cache when possible.
        
        Args:
            file_path: Path to the file
            content: Optional file content (if not provided, file will be read)
            
        Returns:
            Token count for the file
        """
        import os
        import logging
        from pathlib import Path
        
        try:
            # Get file stats
            path = Path(file_path)
            if not path.exists():
                return 0
                
            stat = path.stat()
            mtime = stat.st_mtime
            
            # Check cache
            if file_path in self._cache:
                cached_mtime, cached_count = self._cache[file_path]
                if cached_mtime == mtime:
                    self._hits += 1
                    return cached_count
            
            # Cache miss - need to count
            self._misses += 1
            
            # Get content if not provided
            if content is None:
                try:
                    content = path.read_text(encoding='utf-8')
                except Exception as e:
                    logging.debug(f"Error reading file {file_path}: {e}")
                    return 0
            
            # Count tokens
            self._ensure_counter()
            token_count = self._counter.count(content)
            
            # Update cache
            self._cache[file_path] = (mtime, token_count)
            
            return token_count
            
        except Exception as e:
            logging.debug(f"Error in token cache for {file_path}: {e}")
            return 0
    
    def invalidate(self, file_path: str) -> None:
        """Invalidate cache entry for a specific file.
        
        Args:
            file_path: Path to invalidate
        """
        self._cache.pop(file_path, None)
    
    def clear(self) -> None:
        """Clear the entire cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        
        return {
            'hits': self._hits,
            'misses': self._misses,
            'total': total,
            'hit_rate': round(hit_rate, 1),
            'cache_size': len(self._cache)
        }