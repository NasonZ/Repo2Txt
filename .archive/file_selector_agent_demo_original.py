#!/usr/bin/env python3
"""
File Selector Agent Demo

A single-script demonstration of a file selection agent using OpenAI,
inspired by the tool_calling_demo.py structure and using existing repo2txt components.
Allows users to interactively select files from a local repository.

Usage:
python file_selector_agent_demo.py --repo-path /path/to/your/repo --api-key YOUR_OPENAI_KEY
"""

import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field

import click
from openai import OpenAI, APIError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.prompt import Prompt, Confirm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the src directory to the path so we can import repo2txt modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from repo2txt.core.tokenizer import TokenCounter
from repo2txt.core.models import Config, FileNode, AnalysisResult
from repo2txt.core.file_analyzer import FileAnalyzer
from repo2txt.core.analyzer import RepositoryAnalyzer
from repo2txt.adapters import create_adapter

# Configure logging
logging.basicConfig(level=logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# --- State Management Classes ---

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
    
    def modify_selection(self, add_files: Optional[List[str]] = None, remove_files: Optional[List[str]] = None, reasoning: str = "") -> Dict[str, Any]:
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

# --- Simplified Tool Classes ---

class Tool:
    def __init__(self, name: str, description: str, parameters: Dict[str, Any]):
        self.name = name
        self.description = description
        self.parameters = parameters

class ToolCall:
    def __init__(self, id: str, tool_name: str, input: Dict[str, Any]):
        self.id = id
        self.tool_name = tool_name
        self.input = input

class ToolResult:
    def __init__(self, tool_call_id: str, output: Any, error: Optional[str] = None, is_json_output: bool = False):
        self.tool_call_id = tool_call_id
        self.output = output
        self.error = error
        self.is_json_output = is_json_output

class ToolExecutor:
    def __init__(self, state_manager: StateManager):
        self._registered_tools: Dict[str, Tuple[Tool, callable]] = {}
        self.state_manager = state_manager

    def register_tool(self, tool: Tool, func: callable):
        self._registered_tools[tool.name] = (tool, func)
        logging.debug(f"Tool registered: {tool.name}")

    async def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        logging.debug(f"Executing tool: {tool_call.tool_name} with input: {tool_call.input}")
        if tool_call.tool_name not in self._registered_tools:
            logging.error(f"Tool '{tool_call.tool_name}' not found.")
            return ToolResult(tool_call_id=tool_call.id, output=None, error=f"Tool '{tool_call.tool_name}' not found.")
        
        _tool_obj, func = self._registered_tools[tool_call.tool_name]
        
        try:
            if asyncio.iscoroutinefunction(func):
                output = await func(**tool_call.input)
            else:
                output = func(**tool_call.input)
            logging.debug(f"Tool {tool_call.tool_name} executed successfully.")
            return ToolResult(tool_call_id=tool_call.id, output=output, is_json_output=isinstance(output, (dict, list)))
        except Exception as e:
            logging.error(f"Error executing tool {tool_call.tool_name}: {e}")
            return ToolResult(tool_call_id=tool_call.id, output=None, error=str(e))

# --- Retro UI (Adapted from tool_calling_demo.py) ---
class RetroUI:
    def __init__(self, theme: str = "green", repo_root_for_sanitize: Optional[Path] = None, debug_mode: bool = False):
        self.console = Console()
        self.theme = theme
        self.debug_mode = debug_mode
        self.colors = self._get_theme_colors(theme)
        self.repo_root_for_sanitize = repo_root_for_sanitize
        self._shown_initial_system_prompt = False  # Track if we've shown the initial prompt
        
    def _get_theme_colors(self, theme: str) -> Dict[str, str]:
        themes = {
            "green": {"primary": "bright_green", "secondary": "green", "accent": "bright_cyan", "text": "white", "dim": "bright_black", "error": "bright_red", "warning": "bright_yellow", "success": "bright_green", "debug": "bright_magenta"},
            "amber": {"primary": "bright_yellow", "secondary": "yellow", "accent": "bright_white", "text": "white", "dim": "bright_black", "error": "bright_red", "warning": "bright_yellow", "success": "bright_green", "debug": "bright_magenta"},
            "matrix": {"primary": "bright_green", "secondary": "green", "accent": "bright_white", "text": "green", "dim": "dim green", "error": "bright_red", "warning": "bright_yellow", "success": "bright_green", "debug": "bright_magenta"}
        }
        return themes.get(theme, themes["green"])

    def print_banner(self, title="FILE SELECTOR AGENT", subtitle="AI-Assisted File Selection"):
        debug_indicator = " [DEBUG MODE]" if self.debug_mode else ""
        banner_text = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                            {title.center(62)}                                ║
║                         {subtitle.center(62)}                                ║
║                         {"DEBUG MODE ENABLED".center(62) if self.debug_mode else ""}                                ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        self.console.print(Text(banner_text, style=self.colors["primary"]))

    def print_section(self, title: str, content: Any = ""):
        text_content = content if isinstance(content, (str, Text)) else str(content)
        if len(text_content) > 2000:
            text_content = text_content[:2000] + "\n... [Section Content Truncated]"
        panel = Panel(
            text_content,
            title=f"[{self.colors['accent']}]{title}[/{self.colors['accent']}]",
            border_style=self.colors["secondary"],
            padding=(1, 2)
        )
        self.console.print(panel)

    def print_debug_system_prompt(self, system_prompt: str):
        """Print system prompt in debug mode with retro styling - only once at start."""
        if not self.debug_mode or self._shown_initial_system_prompt:
            return
            
        # More robust tree truncation that works with different prompt formats
        display_prompt = system_prompt
        
        # Try to find and truncate the file tree section using multiple possible patterns
        tree_section_patterns = [
            "Repository Structure (with token counts):",
            "## Repository Structure",
            "### Repository Structure",
            "Repository Structure:",
            "File Tree:",
            "## File Tree"
        ]
        
        tree_end_patterns = [
            "Available Tools:",
            "## Available Tools",
            "### Available Tools", 
            "Guidelines:",
            "## Guidelines",
            "### Guidelines",
            "## Meta-Reasoning Framework",
            "### Meta-Reasoning Framework",
            "Meta-Reasoning Framework",
            "## Selection Strategy",
            "### Selection Strategy",
            "Selection Strategy:",
            "## Critical Constraint",
            "### Critical Constraint",
            "Critical Constraint:",
            "## Tools Available",
            "### Tools Available",
            "Tools Available:"
        ]
        
        # Find tree section start
        tree_start_pos = -1
        tree_start_pattern = None
        for pattern in tree_section_patterns:
            pos = system_prompt.find(pattern)
            if pos != -1:
                tree_start_pos = pos
                tree_start_pattern = pattern
                break
        
        if tree_start_pos != -1:
            # Find tree section end
            tree_end_pos = len(system_prompt)  # Default to end of prompt
            
            # Look for end patterns after the tree start
            search_start = tree_start_pos + len(tree_start_pattern)
            for pattern in tree_end_patterns:
                pos = system_prompt.find(pattern, search_start)
                if pos != -1:
                    tree_end_pos = pos
                    break
            
            # Extract the tree section
            pre_tree = system_prompt[:tree_start_pos + len(tree_start_pattern)]
            tree_content = system_prompt[tree_start_pos + len(tree_start_pattern):tree_end_pos]
            post_tree = system_prompt[tree_end_pos:]
            
            # Truncate only the tree content if it's too long for display
            max_tree_display_length = 1500
            if len(tree_content) > max_tree_display_length:
                tree_content = tree_content[:max_tree_display_length] + "\n... [Tree Truncated for Display - Full tree available to AI]"
            
            display_prompt = pre_tree + tree_content + post_tree
        else:
            # Fallback: If no tree section found, just truncate the whole prompt if too long
            max_total_length = 3000
            if len(display_prompt) > max_total_length:
                display_prompt = display_prompt[:max_total_length] + "\n... [System Prompt Truncated for Display]"
            
        panel = Panel(
            self._sanitize_paths(display_prompt),
            title=f"[{self.colors['debug']}]🔍 SYSTEM PROMPT (DEBUG)[/{self.colors['debug']}]",
            border_style=self.colors["debug"],
            padding=(1, 2)
        )
        self.console.print(panel)
        self._shown_initial_system_prompt = True

    def print_state_diff(self, state_manager: StateManager):
        """Show what changed in the file selection state - works in both debug and clean modes."""
        if not state_manager.state.previous_files:
            return
        
        prev_set = set(state_manager.state.previous_files)
        curr_set = set(state_manager.state.selected_files)
        
        added = curr_set - prev_set
        removed = prev_set - curr_set
        
        if added or removed:
            # Show changes with appropriate styling for mode
            if self.debug_mode:
                self.console.print(f"\n[{self.colors['debug']}]🔍 DEBUG: Selection Changes[/{self.colors['debug']}]")
            else:
                self.console.print(f"\n[{self.colors['accent']}]📝 Selection Changes:[/{self.colors['accent']}]")
                
            for path in sorted(removed):
                self.console.print(f"  [{self.colors['error']}]-[/{self.colors['error']}] {path}")
            for path in sorted(added):
                self.console.print(f"  [{self.colors['success']}]+[/{self.colors['success']}] {path}")
            
            # Show token impact
            token_diff = state_manager.state.total_tokens_selected - state_manager.state.previous_tokens
            if self.debug_mode:
                self.console.print(f"\n[{self.colors['debug']}]🔍 DEBUG: Token Impact[/{self.colors['debug']}]")
            else:
                self.console.print(f"\n[{self.colors['accent']}]📊 Token Impact:[/{self.colors['accent']}]")
                
            self.console.print(f"New total: {state_manager.state.total_tokens_selected:,} tokens (was {state_manager.state.previous_tokens:,})")
            if token_diff > 0:
                self.console.print(f"[{self.colors['warning']}]+{token_diff:,} tokens[/{self.colors['warning']}]")
            elif token_diff < 0:
                self.console.print(f"[{self.colors['success']}]{token_diff:,} tokens[/{self.colors['success']}]")
            else:
                self.console.print(f"[{self.colors['dim']}]No token change[/{self.colors['dim']}]")

    def print_tool_call(self, tool_call: ToolCall):
        if not self.debug_mode:
            return
            
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Key", style=self.colors["accent"])
        table.add_column("Value", style=self.colors["text"])
        table.add_row("Tool:", tool_call.tool_name)
        table.add_row("ID:", tool_call.id)
        for key, value in tool_call.input.items():
            val_str = json.dumps(value, indent=2) if isinstance(value, (list, dict)) else str(value)
            table.add_row(f"{key}:", self._sanitize_paths(val_str))
        panel = Panel(table, title=f"[{self.colors['warning']}]🔧 TOOL CALL (DEBUG)[/{self.colors['warning']}]", border_style=self.colors["warning"], padding=(0, 1))
        self.console.print(panel)

    def print_tool_result(self, result: ToolResult):
        if not self.debug_mode:
            return
            
        if result.error:
            style = self.colors["error"]
            title = "❌ TOOL ERROR (DEBUG)"
            content_text = f"Error: {result.error}"
        else:
            style = self.colors["success"]
            title = "✅ TOOL RESULT (DEBUG)"
            content_text = json.dumps(result.output, indent=2) if result.is_json_output else str(result.output)
            content_text = self._sanitize_paths(content_text)

        if len(content_text) > 2500:
            content_text = content_text[:2500] + "\n... [Result Truncated]"
        panel = Panel(content_text, title=f"[{style}]{title}[/{style}]", border_style=style, padding=(0, 1))
        self.console.print(panel)
    
    def print_debug_info(self, message: str):
        """Print debug information with special styling."""
        if not self.debug_mode:
            return
        self.console.print(f"[{self.colors['debug']}]🔍 DEBUG: {self._sanitize_paths(message)}[/{self.colors['debug']}]")

    def _sanitize_paths(self, content: str) -> str:
        if self.repo_root_for_sanitize:
            content = content.replace(str(self.repo_root_for_sanitize), "REPO_ROOT")
        home_path = str(Path.home())
        content = content.replace(home_path, "USER_HOME")
        return content

    def print_streaming_delta(self, delta: str, is_tool_call: bool = False):
        style = self.colors["warning"] if is_tool_call else self.colors["text"]
        self.console.print(self._sanitize_paths(delta), style=style, end="")

    def print_error(self, message: str):
        self.console.print(f"[{self.colors['error']}]❌ {self._sanitize_paths(message)}[/{self.colors['error']}]")

    def print_success(self, message: str):
        self.console.print(f"[{self.colors['success']}]✅ {self._sanitize_paths(message)}[/{self.colors['success']}]")

    def print_info(self, message: str):
        self.console.print(f"[{self.colors['accent']}]ℹ️  {self._sanitize_paths(message)}[/{self.colors['accent']}]")

    def print_warning(self, message: str):
        self.console.print(f"[{self.colors['warning']}]⚠️  {self._sanitize_paths(message)}[/{self.colors['warning']}]")

# --- File Selector Agent (Now Stateless) ---
class FileSelectorAgent:
    def __init__(self, repo_path: str, openai_api_key: str, model: str, base_url: str = None, theme: str = "green", token_budget: int = 50000, debug_mode: bool = False, prompt_style: str = "standard"):
        self.repo_path = Path(repo_path).resolve()
        if not self.repo_path.is_dir():
            raise ValueError(f"Repository path '{repo_path}' not found or not a directory.")
        
        self.openai_api_key = openai_api_key
        self.model = model
        self.base_url = base_url
        self.debug_mode = debug_mode
        self.prompt_style = prompt_style
        
        # Runtime toggles
        self.use_streaming = True
        self.enable_thinking = True  # Start with thinking ON (model default)
        
        # Initialize OpenAI client
        client_kwargs = {"api_key": openai_api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)
        
        # Initialize UI with debug mode
        self.ui = RetroUI(theme, repo_root_for_sanitize=self.repo_path, debug_mode=debug_mode)
        
        # Initialize repo2txt components
        self.config = Config()
        self.config.enable_token_counting = True
        
        # Analyze repository
        self._analyze_repository()
        
        # Initialize state manager (single source of truth)
        self.state_manager = StateManager(self.analysis_result, token_budget)
        
        # Set up tool executor with state manager
        self.tool_executor = ToolExecutor(self.state_manager)
        self._register_tools()
        
        # Message history for conversation
        self.messages = []
        
        # State snapshots for undo functionality
        self.state_snapshots = []  # List of (messages, selected_files, total_tokens) tuples

    def _analyze_repository(self):
        """Analyze repository using repo2txt components."""
        self.ui.print_info(f"Analyzing repository: {self.repo_path}")
        
        try:
            # Create adapter for the repository
            adapter = create_adapter(str(self.repo_path), self.config)
            repo_name = adapter.get_name()
            
            # Build file tree and get file list
            file_tree = adapter.build_file_tree()
            file_list = adapter.get_file_list()
            
            # Get README content
            readme_content = adapter.get_readme_content()
            
            # Create analysis result
            self.analysis_result = AnalysisResult(
                repo_name=repo_name,
                branch=None,
                readme_content=readme_content,
                structure="",
                file_contents="",
                token_data={},
                total_tokens=sum(f.get('tokens', 0) for f in file_list),
                total_files=len(file_list),
                errors=adapter.errors,
                file_tree=file_tree,
                file_list=file_list
            )
            
            self.ui.print_info(f"Found {self.analysis_result.total_files} files with {self.analysis_result.total_tokens:,} tokens")
            
        except Exception as e:
            self.ui.print_error(f"Failed to analyze repository: {e}")
            raise

    def _format_file_tree(self, node: FileNode, prefix: str = "", is_last: bool = True) -> str:
        """Format file tree for display."""
        if not node:
            return ""
        
        lines = []
        connector = "└── " if is_last else "├── "
        
        if node.is_dir:
            lines.append(f"{prefix}{connector}📁 {node.name}/ (~{node.total_tokens:,} tokens)")
            extension = "    " if is_last else "│   "
            for i, child in enumerate(node.children):
                is_last_child = i == len(node.children) - 1
                lines.append(self._format_file_tree(child, prefix + extension, is_last_child))
        else:
            token_str = f"~{node.token_count:,} tokens" if node.token_count else "0 tokens"
            lines.append(f"{prefix}{connector}📄 {node.name} ({token_str})")
        
        return "\n".join(lines)

    def _create_system_prompt_standard(self) -> str:
        """Create standard system prompt for the agent."""
        # Get current selection from state manager
        selection_summary = self.state_manager.get_current_selection_summary()
        
        # Format file tree (always truncate for system prompt to manage token usage)
        tree_str = self._format_file_tree(self.analysis_result.file_tree)
        # max_tree_len = 15000  # Smaller limit for system prompt
        # if len(tree_str) > max_tree_len:
        #     tree_str = tree_str[:max_tree_len] + "\n... [tree truncated for token management]"

        return f"""You are an AI assistant helping a user select files from the repository '{self.analysis_result.repo_name}'.
The repository contains {self.analysis_result.total_files} files with {self.analysis_result.total_tokens:,} total tokens.
Your goal is to help the user pick a set of files relevant to their task, within a budget of {self.state_manager.state.token_budget:,} tokens.

{selection_summary}

Repository Structure (with token counts):
{tree_str}

Available Tools:
- `select_files`: Selects an initial set of files (replaces current selection)
- `adjust_selection`: Modifies the current selection by adding/removing files

Guidelines:
- Explain your reasoning for selections
- Be mindful of the token budget - ask about tradeoffs if nearing budget
- Paths are relative to the repository root (e.g., 'src/main.py')
- Use the repository structure provided to ensure file paths exist
- Prioritize files that are most relevant to the user's request
"""

    # Produces richer, more thoughtful reasoning but still suffers from hallucinations when dealing with large trees (>150 files) with models smaller than gpt-4.1-mini.
    def _create_system_prompt_meta_reasoning(self) -> str:
        """Create system prompt with meta-reasoning framework for intelligent file selection."""
        # Get current selection from state manager
        selection_summary = self.state_manager.get_current_selection_summary()
        
        # Format file tree (truncate if too long)
        tree_str = self._format_file_tree(self.analysis_result.file_tree)
        # max_tree_len = 20000
        # if len(tree_str) > max_tree_len:
        #     tree_str = tree_str[:max_tree_len] + "\n... [tree truncated]"

        return f"""You are an expert code curator. Your job: select files from this repository that will enable someone to definitively answer the user's question within a {self.state_manager.state.token_budget:,} token budget.

Repository: '{self.analysis_result.repo_name}' ({self.analysis_result.total_files} files, {self.analysis_result.total_tokens:,} tokens)

{selection_summary}

## Repository Structure
{tree_str}

## Meta-Reasoning Framework

Think: "What would I need to read to confidently answer this user's question?"

### 1. Question Decomposition
Break the user's query into specific sub-questions that need answers. For example:
- "How does authentication work?" → What's the auth model? Where are credentials checked? How are sessions managed?
- "How are boundary conditions configured?" → What's the data structure? Where in the training loop? How are they applied?

### 2. Information Mapping
For each sub-question, identify what file types typically contain answers:
- **Data structures**: Models, schemas, type definitions
- **Entry points**: Where the feature gets triggered (routes, main functions, event handlers)
- **Core logic**: Where decisions happen (services, algorithms, business logic)
- **Integration points**: How components connect (middleware, adapters, interfaces)
- **Usage examples**: How it's actually used (examples/, demos/, tests/)

### 3. Coverage Optimization
Prioritize files that answer multiple sub-questions efficiently. A well-designed middleware file might reveal both "where auth happens" AND "how it integrates with the API."

### 4. Learning Dependency
Consider what information builds on other information:
- Understand interfaces before implementations
- Understand data models before the logic that uses them
- Understand core concepts before edge cases

### 5. Completeness Check
Ask: "If someone read only these files, could they provide a complete answer? What would be missing?"

## Deductive Reasoning from Available Information

**Architectural Signal Reading:**

File organization reflects design decisions. When components are separated into different files/directories, it reveals the architect's mental model of how the system should be decomposed. Ask yourself: "Why did someone choose to separate these concerns?" 

For example, if auth logic is in middleware rather than embedded in each endpoint, that's an architectural choice that prioritizes centralized control over distributed responsibility. This choice affects where permission logic lives, how it can be modified, and how it integrates with other systems.

**Information Density Analysis:**

File size and naming patterns indicate information value. Small files with generic names (`base.py`, `interface.py`) often contain high-density architectural contracts. Large files with specific names (`user_management_handlers.py`) often contain detailed implementations.

The token count gives you efficiency signals. A 500-token interface file might teach you more about system design than a 3000-token implementation file. Consider what ratio of "understanding per token" you're getting.

**System Flow Inference:**

Directory structure reveals information flow and dependency patterns. Deep nesting suggests layered architecture where data flows through multiple abstraction levels. Flat structure suggests direct, simple interactions.

When you see separation patterns like `models/` vs `services/` vs `controllers/`, you're seeing the architect's theory about where different types of logic should live. This separation tells you where to look for different aspects of functionality.

**Inductive Pattern Recognition:**

Look for recurring patterns across the codebase. If you see multiple `*_adapter.py` files, you can induce there's a pluggable architecture even without reading the implementations. If you see `test_*` files alongside implementation files, you can induce this system values testability.

Use your knowledge of common software patterns to fill in gaps. If you see factory patterns, adapter patterns, or observer patterns in the file structure, you can induce how those components likely interact even before reading the code.

**Feedback Loop Reasoning:**

User questions reveal the gaps in their mental model. When they ask "but how does X work?", they're telling you that your previous selection didn't adequately explain X. Treat follow-up questions as requirements refinement - they're showing you what cognitive bridges are missing.

User expertise level affects information needs. A beginner asking "how does auth work?" needs the conceptual foundation. An expert asking "how does token refresh handle race conditions?" needs specific implementation details. Calibrate your selection to match their existing knowledge level.

## Collaborative Reasoning

**Voice your reasoning process:**

"Based on your question about [topic], I'm looking for files that show [what you expect to find]. I see several candidates: [list options]. I'm prioritizing [chosen files] because [reasoning], but I'm uncertain about [specific concern] - what's your take?"

"I'm seeing an interesting architecture here - instead of the typical [expected pattern], this codebase uses [observed pattern]. This suggests [architectural implication]. Should I focus on understanding how this non-standard approach works, or are you more interested in [alternative focus]?"

"Your question about [topic] could go two directions: understanding [approach A] or diving into [approach B]. Given our token budget, I can do a deep dive on one or a broader overview of both. Which would be more valuable for what you're trying to accomplish?"

"Building on our previous selection of [files], I'm now looking at [new area]. I notice [observation about relationship/dependencies]. This makes me think we should also include [files] to complete the picture, but that would put us at [token count]. Is that trade-off worth it, or should I find a more focused approach?"

"I'm not finding the typical [expected files/patterns] that I'd expect for [topic]. Instead, I'm seeing [what actually exists]. This might mean [possible explanation]. Before I select what's available, can you help me understand if you're looking for [clarification of actual need]?"

**Express uncertainties explicitly:**
- "I'm not sure if you need implementation details or just the interfaces"
- "There are two possible approaches here - I could focus on [option A] or [option B]"
- "I don't see obvious [expected pattern] files - the architecture might be different than I expected"

**Use conversation to refine understanding:**
- Ask clarifying questions when the query could go multiple directions
- Reference previous exchanges: "You mentioned X earlier, so I'm also including Y"
- Acknowledge feedback: "Since you said the last selection was too implementation-heavy, I'm focusing more on interfaces this time"

**Adapt based on user's style:**
- If they want details → include more implementation files
- If they want overview → focus on interfaces and examples
- If they're debugging → include error handling and edge cases

## Selection Strategy

**High-value targets:**
- Files whose names semantically match the user's query (use your knowledge of common patterns)
- Entry points that trigger the relevant functionality
- Core implementation files that contain the main logic
- Representative examples that show real usage
- Tests that reveal expected behavior and edge cases

**Efficiency principles:**
- Choose files that teach concepts, not just show code
- Prefer files that reveal design decisions
- Include enough context to understand the full picture
- Balance breadth (understanding the system) with depth (understanding specifics)

## Critical Constraint

You MUST only select files that exist exactly as shown in the repository structure above. Reference the specific location where you found each file.

## Tools Available
- `select_files`: Choose initial file set based on your analysis
- `adjust_selection`: Refine selection based on user feedback

**Remember: 
    1. You're collaborating, not just executing.** Share your reasoning, voice uncertainties, and use the conversation to refine your understanding. The goal is working together to create the optimal reading list for answering their question.
    2. Always double check your selections are valid file paths against the repository structure above. 
"""

    # There seems to be no be no significant benefit to using XML formatting
    def _create_system_prompt_meta_reasoning_xml(self) -> str:
        """Create system prompt with meta-reasoning framework using XML formatting."""
        # Get current selection from state manager
        selection_summary = self.state_manager.get_current_selection_summary()
        
        tree_str = self._format_file_tree(self.analysis_result.file_tree)

        return f"""<system_prompt>
<role>expert code curator</role>
<objective>Select files from this repository that will enable someone to definitively answer the user's question within a {self.state_manager.state.token_budget:,} token budget</objective>

<repository>
    <name>{self.analysis_result.repo_name}</name>
    <stats>
        <files>{self.analysis_result.total_files}</files>
        <tokens>{self.analysis_result.total_tokens:,}</tokens>
    </stats>
</repository>

<current_selection>
{selection_summary}
</current_selection>

<repository_structure>
{tree_str}
</repository_structure>

<meta_reasoning_framework>
    <core_principle>Think: "What would I need to read to confidently answer this user's question?"</core_principle>
    
    <step id="1">
        <name>Question Decomposition</name>
        <description>Break the user's query into specific sub-questions that need answers</description>
        <examples>
            <example>
                <query>How does authentication work?</query>
                <decomposition>
                    <sub_question>What's the auth model?</sub_question>
                    <sub_question>Where are credentials checked?</sub_question>
                    <sub_question>How are sessions managed?</sub_question>
                </decomposition>
            </example>
            <example>
                <query>How are boundary conditions configured?</query>
                <decomposition>
                    <sub_question>What's the data structure?</sub_question>
                    <sub_question>Where in the training loop?</sub_question>
                    <sub_question>How are they applied?</sub_question>
                </decomposition>
            </example>
        </examples>
    </step>
    
    <step id="2">
        <name>Information Mapping</name>
        <description>For each sub-question, identify what file types typically contain answers</description>
        <mappings>
            <mapping type="data_structures">Models, schemas, type definitions</mapping>
            <mapping type="entry_points">Where the feature gets triggered (routes, main functions, event handlers)</mapping>
            <mapping type="core_logic">Where decisions happen (services, algorithms, business logic)</mapping>
            <mapping type="integration_points">How components connect (middleware, adapters, interfaces)</mapping>
            <mapping type="usage_examples">How it's actually used (examples/, demos/, tests/)</mapping>
        </mappings>
    </step>
    
    <step id="3">
        <name>Coverage Optimization</name>
        <description>Prioritize files that answer multiple sub-questions efficiently. A well-designed middleware file might reveal both "where auth happens" AND "how it integrates with the API."</description>
    </step>
    
    <step id="4">
        <name>Learning Dependency</name>
        <description>Consider what information builds on other information</description>
        <principles>
            <principle>Understand interfaces before implementations</principle>
            <principle>Understand data models before the logic that uses them</principle>
            <principle>Understand core concepts before edge cases</principle>
        </principles>
    </step>
    
    <step id="5">
        <name>Completeness Check</name>
        <description>Ask: "If someone read only these files, could they provide a complete answer? What would be missing?"</description>
    </step>
</meta_reasoning_framework>

<deductive_reasoning>
    <architectural_signal_reading>
        <description>File organization reflects design decisions. When components are separated into different files/directories, it reveals the architect's mental model of how the system should be decomposed. Ask yourself: "Why did someone choose to separate these concerns?"</description>
        <example>If auth logic is in middleware rather than embedded in each endpoint, that's an architectural choice that prioritizes centralized control over distributed responsibility. This choice affects where permission logic lives, how it can be modified, and how it integrates with other systems.</example>
    </architectural_signal_reading>
    
    <information_density_analysis>
        <description>File size and naming patterns indicate information value. Small files with generic names (`base.py`, `interface.py`) often contain high-density architectural contracts. Large files with specific names (`user_management_handlers.py`) often contain detailed implementations.</description>
        <insight>The token count gives you efficiency signals. A 500-token interface file might teach you more about system design than a 3000-token implementation file. Consider what ratio of "understanding per token" you're getting.</insight>
    </information_density_analysis>
    
    <system_flow_inference>
        <description>Directory structure reveals information flow and dependency patterns. Deep nesting suggests layered architecture where data flows through multiple abstraction levels. Flat structure suggests direct, simple interactions.</description>
        <pattern>When you see separation patterns like `models/` vs `services/` vs `controllers/`, you're seeing the architect's theory about where different types of logic should live. This separation tells you where to look for different aspects of functionality.</pattern>
    </system_flow_inference>
    
    <inductive_pattern_recognition>
        <description>Look for recurring patterns across the codebase. If you see multiple `*_adapter.py` files, you can induce there's a pluggable architecture even without reading the implementations. If you see `test_*` files alongside implementation files, you can induce this system values testability.</description>
        <guidance>Use your knowledge of common software patterns to fill in gaps. If you see factory patterns, adapter patterns, or observer patterns in the file structure, you can induce how those components likely interact even before reading the code.</guidance>
    </inductive_pattern_recognition>
    
    <feedback_loop_reasoning>
        <description>User questions reveal the gaps in their mental model. When they ask "but how does X work?", they're telling you that your previous selection didn't adequately explain X. Treat follow-up questions as requirements refinement - they're showing you what cognitive bridges are missing.</description>
        <expertise_calibration>User expertise level affects information needs. A beginner asking "how does auth work?" needs the conceptual foundation. An expert asking "how does token refresh handle race conditions?" needs specific implementation details. Calibrate your selection to match their existing knowledge level.</expertise_calibration>
    </feedback_loop_reasoning>
</deductive_reasoning>

<collaborative_reasoning>
    <voice_reasoning>
        <template id="1">Based on your question about [topic], I'm looking for files that show [what you expect to find]. I see several candidates: [list options]. I'm prioritizing [chosen files] because [reasoning], but I'm uncertain about [specific concern] - what's your take?</template>
        <template id="2">I'm seeing an interesting architecture here - instead of the typical [expected pattern], this codebase uses [observed pattern]. This suggests [architectural implication]. Should I focus on understanding how this non-standard approach works, or are you more interested in [alternative focus]?</template>
        <template id="3">Your question about [topic] could go two directions: understanding [approach A] or diving into [approach B]. Given our token budget, I can do a deep dive on one or a broader overview of both. Which would be more valuable for what you're trying to accomplish?</template>
        <template id="4">Building on our previous selection of [files], I'm now looking at [new area]. I notice [observation about relationship/dependencies]. This makes me think we should also include [files] to complete the picture, but that would put us at [token count]. Is that trade-off worth it, or should I find a more focused approach?</template>
        <template id="5">I'm not finding the typical [expected files/patterns] that I'd expect for [topic]. Instead, I'm seeing [what actually exists]. This might mean [possible explanation]. Before I select what's available, can you help me understand if you're looking for [clarification of actual need]?</template>
    </voice_reasoning>
    
    <express_uncertainties>
        <uncertainty>I'm not sure if you need implementation details or just the interfaces</uncertainty>
        <uncertainty>There are two possible approaches here - I could focus on [option A] or [option B]</uncertainty>
        <uncertainty>I don't see obvious [expected pattern] files - the architecture might be different than I expected</uncertainty>
    </express_uncertainties>
    
    <conversation_refinement>
        <technique>Ask clarifying questions when the query could go multiple directions</technique>
        <technique>Reference previous exchanges: "You mentioned X earlier, so I'm also including Y"</technique>
        <technique>Acknowledge feedback: "Since you said the last selection was too implementation-heavy, I'm focusing more on interfaces this time"</technique>
    </conversation_refinement>
    
    <adaptation>
        <if_detail_oriented>Include more implementation files</if_detail_oriented>
        <if_overview_oriented>Focus on interfaces and examples</if_overview_oriented>
        <if_debugging>Include error handling and edge cases</if_debugging>
    </adaptation>
</collaborative_reasoning>

<selection_strategy>
    <high_value_targets>
        <target>Files whose names semantically match the user's query (use your knowledge of common patterns)</target>
        <target>Entry points that trigger the relevant functionality</target>
        <target>Core implementation files that contain the main logic</target>
        <target>Representative examples that show real usage</target>
        <target>Tests that reveal expected behavior and edge cases</target>
    </high_value_targets>
    
    <efficiency_principles>
        <principle>Choose files that teach concepts, not just show code</principle>
        <principle>Prefer files that reveal design decisions</principle>
        <principle>Include enough context to understand the full picture</principle>
        <principle>Balance breadth (understanding the system) with depth (understanding specifics)</principle>
    </efficiency_principles>
</selection_strategy>

<critical_constraint>
    You MUST only select files that exist exactly as shown in the repository structure above. Reference the specific location where you found each file.
</critical_constraint>

<available_tools>
    <tool name="select_files">Choose initial file set based on your analysis</tool>
    <tool name="adjust_selection">Refine selection based on user feedback</tool>
</available_tools>

<remember>
    <point>You're collaborating, not just executing. Share your reasoning, voice uncertainties, and use the conversation to refine your understanding. The goal is working together to create the optimal reading list for answering their question.</point>
    <point>Always double check your selections are valid file paths against the repository structure above.</point>
</remember>
</system_prompt>"""

    def _create_system_prompt(self) -> str:
        """Create system prompt using selected style."""
        if self.prompt_style == "meta-reasoning":
            system_prompt = self._create_system_prompt_meta_reasoning()
        elif self.prompt_style == "xml":
            system_prompt = self._create_system_prompt_meta_reasoning_xml()
        else:
            system_prompt = self._create_system_prompt_standard()
        
        # Show system prompt in debug mode
        self.ui.print_debug_system_prompt(system_prompt)
        return system_prompt

    # Tool implementations (now delegate to state manager)
    def select_files_impl(self, paths: List[str], reasoning: str) -> Dict[str, Any]:
        """Select initial set of files via state manager."""
        return self.state_manager.replace_selection(paths, reasoning)

    def adjust_selection_impl(self, add_files: Optional[List[str]] = None, remove_files: Optional[List[str]] = None, reasoning: str = "") -> Dict[str, Any]:
        """Adjust current selection via state manager."""
        return self.state_manager.modify_selection(add_files, remove_files, reasoning)

    def _register_tools(self):
        """Register tools with state manager integration."""
        tool_defs = [
            (Tool(
                "select_files", 
                "Selects an initial set of files, replacing any current selection", 
                {
                    "type": "object", 
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": "Detailed rationale for this selection that: 1) Follows the meta-reasoning framework from the system prompt diligently, 2) Explains how each file helps answer the user's question, 3) Verifies each path exists in the repository structure shown in the system prompt Be thorough and explicit about your reasoning process."
                            # Does lead to more thorough reasoning but hallucinations still happen. 
                        },
                        "paths": {
                            "type": "array", 
                            "items": {"type": "string"},
                            "description": "List of file paths to select"
                        }
                    }, 
                    "required": ["reasoning", "paths"]
                }
            ), self.select_files_impl),
            (Tool(
                "adjust_selection", 
                "Modifies the current selection by adding or removing files", 
                {
                    "type": "object", 
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": "Detailed rationale for this adjustment that: 1) Follows the meta-reasoning framework from the system prompt, 2) Explains why files are being added/removed based on user feedback or new understanding, 3) Verifies each path against the repository structure, 4) Considers how changes affect token budget and information completeness, 5) References previous conversation context. Be thorough about why this adjustment improves the selection."
                        }, 
                        "add_files": {
                            "type": "array", 
                            "items": {"type": "string"},
                            "description": "Files to add to current selection"
                        }, 
                        "remove_files": {
                            "type": "array", 
                            "items": {"type": "string"},
                            "description": "Files to remove from current selection"
                        }
                    }, 
                    "required": ["reasoning"]
                }
            ), self.adjust_selection_impl)
        ]
        
        for tool_obj, func in tool_defs:
            self.tool_executor.register_tool(tool_obj, func)

    def _convert_to_openai_tools(self) -> List[Dict[str, Any]]:
        """Convert tools to OpenAI format."""
        return [{"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
                for t, _ in self.tool_executor._registered_tools.values()]

    def _parse_openai_tool_calls(self, openai_tool_calls_list: Optional[List[Any]]) -> List[ToolCall]:
        """Parse OpenAI tool calls."""
        if not openai_tool_calls_list:
            return []
        
        parsed = []
        for call in openai_tool_calls_list:
            try:
                args = json.loads(call.function.arguments)
            except json.JSONDecodeError:
                args = {}
                self.ui.print_warning(f"Invalid JSON args for {call.function.name}: {call.function.arguments}")
            parsed.append(ToolCall(id=call.id, tool_name=call.function.name, input=args))
        return parsed
    
    def _save_state_snapshot(self):
        """Save current state for undo functionality."""
        # Deep copy messages
        messages_copy = [msg.copy() for msg in self.messages]
        # Copy file selection state
        selected_files_copy = self.state_manager.state.selected_files.copy()
        total_tokens = self.state_manager.state.total_tokens_selected
        
        self.state_snapshots.append((messages_copy, selected_files_copy, total_tokens))
    
    def _clean_thinking_tags(self, content: str) -> str:
        """Remove <think>...</think> blocks from content."""
        # Remove think blocks and clean up extra whitespace
        cleaned = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL)
        return cleaned.strip()
    
    def _handle_command(self, command: str) -> bool:
        """Handle slash commands. Returns True if command was processed."""
        parts = command.split()
        cmd = parts[0].lower()
        
        if cmd == '/clear':
            new_system_prompt = self._create_system_prompt()
            self.messages = [{"role": "system", "content": new_system_prompt}]
            self.state_manager.clear_selection()
            # Reset thinking to default (ON)
            self.enable_thinking = True
            self.ui.print_success("Conversation history and selection cleared.")
            return True
            
        elif cmd == '/toggle':
            if len(parts) < 2:
                self.ui.print_warning("Usage: /toggle [streaming|thinking|reasoning]")
                return True
                
            toggle_type = parts[1].lower()
            
            if toggle_type == 'streaming':
                self.use_streaming = not self.use_streaming
                status = "enabled" if self.use_streaming else "disabled"
                self.ui.print_success(f"Streaming {status}")
                return True
                
            elif toggle_type == 'thinking':
                if "qwen" not in self.model.lower():
                    self.ui.print_warning("Thinking mode is only available for Qwen models")
                    return True
                self.enable_thinking = not self.enable_thinking
                status = "enabled" if self.enable_thinking else "disabled"
                self.ui.print_success(f"Thinking mode {status}")
                return True
                
            elif toggle_type == 'prompt':
                # Cycle through available styles
                if self.prompt_style == "standard":
                    new_style = "meta-reasoning"
                elif self.prompt_style == "meta-reasoning":
                    new_style = "xml"
                else:  # xml
                    new_style = "standard"
                
                self.prompt_style = new_style
                # Update system prompt immediately
                updated_system_prompt = self._create_system_prompt()
                self.messages[0]["content"] = updated_system_prompt
                self.ui.print_success(f"Prompt style changed to: {new_style}")
                return True
                
            elif toggle_type == 'budget':
                if len(parts) < 3:
                    self.ui.print_warning("Usage: /toggle budget <number>")
                    return True
                try:
                    new_budget = int(parts[2])
                    if new_budget <= 0:
                        self.ui.print_warning("Budget must be a positive number")
                        return True
                    old_budget = self.state_manager.state.token_budget
                    self.state_manager.state.token_budget = new_budget
                    # Update system prompt with new budget
                    updated_system_prompt = self._create_system_prompt()
                    self.messages[0]["content"] = updated_system_prompt
                    self.ui.print_success(f"Token budget changed from {old_budget:,} to {new_budget:,}")
                    return True
                except ValueError:
                    self.ui.print_warning("Budget must be a valid number")
                    return True
                
            else:
                self.ui.print_warning("Available toggles: streaming, thinking, prompt, budget")
                return True
                
        elif cmd == '/debug':
            if len(parts) < 2:
                self.debug_mode = not self.debug_mode
                status = "enabled" if self.debug_mode else "disabled"
                self.ui.print_success(f"Debug mode {status}")
                return True
            else:
                debug_type = parts[1].lower()
                if debug_type == 'state':
                    self._show_debug_state()
                    return True
                else:
                    self.ui.print_warning("Available debug options: /debug or /debug state")
                    return True
                    
        elif cmd == '/undo':
            if not self.state_snapshots:
                self.ui.print_warning("No actions to undo")
                return True
            
            # Restore previous state
            prev_messages, prev_files, prev_tokens = self.state_snapshots.pop()
            
            # Restore messages
            self.messages = prev_messages
            
            # Restore file selection state
            self.state_manager.state.selected_files = prev_files
            self.state_manager.state.total_tokens_selected = prev_tokens
            
            # Update system prompt to reflect restored state
            updated_system_prompt = self._create_system_prompt()
            self.messages[0]["content"] = updated_system_prompt
            
            self.ui.print_success("Undone last action - restored previous state")
            return True
            
        elif cmd == '/redo':
            if len(self.messages) <= 1:  # Only system message
                self.ui.print_warning("No message to regenerate")
                return True
            
            # Find the last user message
            last_user_idx = -1
            for i in range(len(self.messages) - 1, 0, -1):
                if self.messages[i]["role"] == "user":
                    last_user_idx = i
                    break
            
            if last_user_idx == -1:
                self.ui.print_warning("No user message found to regenerate response for")
                return True
            
            # Remove all messages after the last user message
            self.messages = self.messages[:last_user_idx + 1]
            
            # Update system prompt
            updated_system_prompt = self._create_system_prompt()
            self.messages[0]["content"] = updated_system_prompt
            
            self.ui.print_info("Regenerating AI response...")
            # Return special marker to trigger regeneration in chat loop
            return "REDO"
            
        elif cmd in ['/quit', '/exit']:
            return False  # Signal to exit chat loop
            
        elif cmd == '/help':
            self._show_help()
            return True
            
        else:
            self.ui.print_warning(f"Unknown command: {cmd}")
            self._show_help()
            return True
    
    def _show_debug_state(self):
        """Show current debug state information."""
        self.ui.print_section("DEBUG STATE", f"""
Streaming: {self.use_streaming}
Thinking: {self.enable_thinking} {'(Qwen only)' if 'qwen' not in self.model.lower() else ''}
Prompt style: {self.prompt_style}
Debug mode: {self.debug_mode}
Model: {self.model}
Base URL: {self.base_url or 'Default OpenAI'}
Selected files: {len(self.state_manager.state.selected_files)}
Token budget: {self.state_manager.state.token_budget:,}
Tokens used: {self.state_manager.state.total_tokens_selected:,}
        """.strip())
        
    def _show_help(self):
        """Show available commands."""
        self.ui.print_section("COMMANDS", """
/clear                  - Clear conversation and selection
/undo                   - Undo last action (restore previous state)
/redo                   - Regenerate last AI response
/toggle streaming       - Toggle streaming on/off
/toggle thinking        - Toggle Qwen thinking mode (Qwen models only)
/toggle prompt          - Cycle through prompt styles (standard/meta-reasoning/xml)
/toggle budget <n>      - Set token budget to <n> tokens
/debug                  - Toggle debug mode
/debug state            - Show current configuration
/help                   - Show this help
/quit or /exit         - Exit the application
        """.strip())
    
    async def _execute_tool_calls_and_update_history(self, tool_calls: List[ToolCall]):
        """Execute tool calls and update conversation history."""
        tool_results_for_history = []
        for tool_call in tool_calls:
            self.ui.print_tool_call(tool_call)
            result = await self.tool_executor.execute_tool(tool_call)
            self.ui.print_tool_result(result)
            
            # Show state diff after tool execution in BOTH debug and clean modes
            self.ui.print_state_diff(self.state_manager)
            
            # Always provide user feedback about tool results, even in non-debug mode
            if not self.debug_mode:
                if result.error:
                    self.ui.print_error(f"Tool {tool_call.tool_name} failed: {result.error}")
                else:
                    feedback = result.output.get('feedback', 'Tool executed successfully') if isinstance(result.output, dict) else 'Tool executed successfully'
                    self.ui.print_success(feedback)
            
            output_content = json.dumps(result.output) if result.is_json_output else str(result.output)
            if result.error:
                output_content = json.dumps({"error": result.error, "details": str(result.output)})

            tool_results_for_history.append({
                "role": "tool", 
                "tool_call_id": result.tool_call_id, 
                "name": tool_call.tool_name, 
                "content": output_content
            })
        
        self.messages.extend(tool_results_for_history)

    async def _handle_streaming_response(self, stream_response):
        """Handle streaming response from OpenAI."""
        content_buffer = ""
        tool_calls_data_buffer: Dict[int, Dict[str, Any]] = {}
        
        self.ui.console.print(f"\n[{self.ui.colors['accent']}]🤖 Assistant:[/{self.ui.colors['accent']}] ", end="")
        
        for chunk in stream_response:
            delta = chunk.choices[0].delta
            
            if delta.content:
                content_buffer += delta.content
                self.ui.print_streaming_delta(delta.content)
            
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_data_buffer:
                        tool_calls_data_buffer[idx] = {"id": None, "function": {"name": None, "arguments": ""}}
                    
                    if tc_delta.id:
                        tool_calls_data_buffer[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tool_calls_data_buffer[idx]["function"]["name"] = tc_delta.function.name
                            self.ui.console.print(f"\n[{self.ui.colors['warning']}]🔧 Calling tool {idx + 1}: {tc_delta.function.name}[/{self.ui.colors['warning']}]")
                            self.ui.console.print(f" [{self.ui.colors['dim']}]Args:[/{self.ui.colors['dim']}] ", end="")
                        if tc_delta.function.arguments:
                            tool_calls_data_buffer[idx]["function"]["arguments"] += tc_delta.function.arguments
                            self.ui.print_streaming_delta(tc_delta.function.arguments, is_tool_call=True)
        
        self.ui.console.print()

        # Convert buffer to tool calls
        openai_tool_calls_list_from_stream = []
        for _idx, call_data in sorted(tool_calls_data_buffer.items()):
            if call_data["id"] and call_data["function"]["name"]:
                mock_function = type('Function', (), call_data["function"])()
                mock_tool_call = type('ToolCall', (), {'id': call_data["id"], 'function': mock_function, 'type': 'function'})()
                openai_tool_calls_list_from_stream.append(mock_tool_call)
            else:
                self.ui.print_error(f"Incomplete tool call from stream at index {_idx}: {call_data}")
        
        return content_buffer, self._parse_openai_tool_calls(openai_tool_calls_list_from_stream)

    async def chat_loop(self):
        """Main chat loop."""
        mode_info = "DEBUG MODE - Full system internals visible" if self.debug_mode else "CLEAN MODE - User-friendly experience"
        self.ui.print_section("CHAT MODE", f"{mode_info}\nType '/help' for commands, '/quit' to exit.")
        
        system_prompt = self._create_system_prompt()
        self.messages = [{"role": "system", "content": system_prompt}]

        while True:
            try:
                user_input = Prompt.ask(f"\n[{self.ui.colors['primary']}]👤 You[/{self.ui.colors['primary']}]").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    break
                    
                # Handle commands
                if user_input.startswith('/'):
                    command_result = self._handle_command(user_input.strip())
                    if command_result is False:  # /quit or /exit
                        break
                    elif command_result == "REDO":  # Special case for redo
                        # Don't continue - let it fall through to regenerate
                        pass
                    else:
                        continue
                        
                if not user_input:
                    continue

                # Save state snapshot before processing (unless it's a redo)
                if user_input.startswith('/') and user_input.strip().split()[0].lower() == '/redo':
                    # For redo, we already have the user message in history
                    pass
                else:
                    self._save_state_snapshot()
                    # Always append thinking tag based on current state
                    if "qwen3" in self.model.lower():
                        thinking_tag = " /think" if self.enable_thinking else " /no_think"
                        user_input += thinking_tag
                    self.messages.append({"role": "user", "content": user_input})
                
                # Update system prompt with current state from state manager
                updated_system_prompt = self._create_system_prompt()
                self.messages[0]["content"] = updated_system_prompt

                # Use configured streaming setting
                use_streaming = self.use_streaming
                
                openai_tools_fmt = self._convert_to_openai_tools()
                
                common_params = {
                    "model": self.model, 
                    "messages": self.messages, 
                    "tools": openai_tools_fmt, 
                    "tool_choice": "auto"
                }
                
                # Add Qwen-specific parameters (no longer control thinking via API)
                if "qwen" in self.model.lower():
                    # Always use same parameters, thinking controlled via /think /no_think tags
                    common_params.update({
                        "temperature": 0.7,
                        "top_p": 0.8,
                        "extra_body": {
                            "top_k": 20,
                            "presence_penalty": 1.5
                        }
                    })

                response_content = ""
                invoked_tool_calls = []

                if use_streaming:
                    stream = self.client.chat.completions.create(**common_params, stream=True)
                    response_content, invoked_tool_calls = await self._handle_streaming_response(stream)
                else:
                    completion = self.client.chat.completions.create(**common_params)
                    msg = completion.choices[0].message
                    response_content = msg.content or ""
                    self.ui.console.print(f"\n[{self.ui.colors['accent']}]🤖 Assistant:[/{self.ui.colors['accent']}] {self.ui._sanitize_paths(response_content)}")
                    invoked_tool_calls = self._parse_openai_tool_calls(msg.tool_calls)
                
                # Add assistant message to history
                assistant_msg: Dict[str, Any] = {"role": "assistant"}
                if response_content:
                    # Clean thinking tags before storing
                    cleaned_content = self._clean_thinking_tags(response_content)
                    assistant_msg["content"] = cleaned_content
                if invoked_tool_calls:
                    assistant_msg["tool_calls"] = [
                        {"id": tc.id, "type": "function", "function": {"name": tc.tool_name, "arguments": json.dumps(tc.input)}} 
                        for tc in invoked_tool_calls
                    ]
                
                if response_content or invoked_tool_calls:
                    self.messages.append(assistant_msg)

                # Execute tools if any
                if invoked_tool_calls:
                    if self.debug_mode:
                        self.ui.print_debug_info(f"Executing {len(invoked_tool_calls)} tool call(s)")
                    
                    await self._execute_tool_calls_and_update_history(invoked_tool_calls)
                    
                    # Update system prompt again after tool execution (state may have changed)
                    final_system_prompt = self._create_system_prompt()
                    self.messages[0]["content"] = final_system_prompt
                    
                    # Ensure last user message has thinking tag for Qwen models
                    if "qwen" in self.model.lower():
                        thinking_tag = " /think" if self.enable_thinking else " /no_think"
                        # Find last user message and add tag if not already present
                        for i in range(len(self.messages) - 1, -1, -1):
                            if self.messages[i]["role"] == "user":
                                content = self.messages[i]["content"]
                                if not content.endswith("/think") and not content.endswith("/no_think"):
                                    self.messages[i]["content"] = content + thinking_tag
                                break
                    
                    if self.debug_mode:
                        self.ui.console.print(f"\n[{self.ui.colors['accent']}]🔄 Getting final response after tools...[/{self.ui.colors['accent']}]")
                    
                    final_common_params = {"model": self.model, "messages": self.messages}
                    
                    # Add Qwen-specific parameters (no longer control thinking via API)
                    if "qwen" in self.model.lower():
                        # Always use same parameters, thinking controlled via /think /no_think tags
                        final_common_params.update({
                            "temperature": 0.7,
                            "top_p": 0.8,
                            "extra_body": {
                                "top_k": 20,
                                "presence_penalty": 1.5
                            }
                        })

                    if use_streaming:
                        final_stream = self.client.chat.completions.create(**final_common_params, stream=True)
                        final_response_content, _ = await self._handle_streaming_response(final_stream)
                    else:
                        final_completion = self.client.chat.completions.create(**final_common_params)
                        final_response_content = final_completion.choices[0].message.content or ""
                        self.ui.console.print(f"\n[{self.ui.colors['accent']}]🤖 Assistant:[/{self.ui.colors['accent']}] {self.ui._sanitize_paths(final_response_content)}")
                    
                    if final_response_content:
                        # Clean thinking tags before storing
                        cleaned_final_content = self._clean_thinking_tags(final_response_content)
                        self.messages.append({"role": "assistant", "content": cleaned_final_content})
            
            except APIError as e:
                self.ui.print_error(f"OpenAI API Error: {e.code} - {e.message}")
            except KeyboardInterrupt:
                self.ui.print_warning("\nInterrupted by user.")
                break
            except Exception as e:
                self.ui.print_error(f"Unexpected error: {e}")
                logging.error("Chat loop error", exc_info=True)
                break

    def run(self):
        """Run the agent."""
        self.ui.print_banner()
        prompt_style_display = f" [{self.prompt_style.upper()}]" if self.prompt_style != "standard" else ""
        self.ui.print_info(f"Model: {self.model}")
        self.ui.print_info(f"Repository: {self.analysis_result.repo_name}")
        self.ui.print_info(f"Files: {self.analysis_result.total_files}")
        self.ui.print_info(f"Total tokens: {self.analysis_result.total_tokens:,}")
        self.ui.print_info(f"Token budget: {self.state_manager.state.token_budget:,}")
        self.ui.print_info(f"Tools: {', '.join(self.tool_executor._registered_tools.keys())}")
        self.ui.print_info(f"Prompt style: {self.prompt_style}{prompt_style_display}")
        if self.debug_mode:
            self.ui.print_debug_info("Debug mode enabled - system internals will be shown")
        
        try:
            asyncio.run(self.chat_loop())
        except KeyboardInterrupt:
            pass
        finally:
            self.ui.print_success("File Selector Agent shutting down. Goodbye! 👋")

def get_llm_config_from_env():
    """Get LLM configuration from environment variables, following the same pattern as domain_cli_demo.py"""
    llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
    llm_model = os.getenv("LLM_MODEL")
    api_key = None
    base_url = None
    
    if llm_provider == "openai":
        # Check both OPENAI_API_KEY and LLM_API_KEY
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        
        # For local endpoints (ollama), we don't need a real API key
        base_url = os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        if base_url and "openai.com" not in base_url:
            # Local endpoint - use dummy key if none provided
            if not api_key:
                api_key = "dummy-key"
        elif not api_key:
            # Real OpenAI - need actual key
            raise ValueError("OPENAI_API_KEY not found in environment")
        
        if not llm_model:
            llm_model = "gpt-4.1-mini"  # Default fallback
    else:
        # For other providers, use similar logic
        base_url = os.getenv("LLM_BASE_URL", "http://localhost:8000")
        api_key = os.getenv("LLM_API_KEY", "dummy-key")
        if not llm_model:
            llm_model = "gpt-4.1-mini"  # Default fallback
    
    return {
        "model": llm_model,
        "api_key": api_key,
        "base_url": base_url,
        "provider": llm_provider
    }

@click.command()
@click.option('--repo-path', default=".", help="Path to the repository to analyze.")
@click.option('--model', default=None, help="OpenAI model name. If not provided, uses LLM_MODEL from .env file.")
@click.option('--api-key', default=None, help="OpenAI API key. If not provided, uses LLM_API_KEY or OPENAI_API_KEY from .env file.")
@click.option('--base-url', default=None, help="Base URL for API. If not provided, uses LLM_BASE_URL from .env file.")
@click.option('--theme', type=click.Choice(['green', 'amber', 'matrix']), default='green', help='UI theme.')
@click.option('--budget', default=50000, type=int, help="Token budget for selection.")
@click.option('--debug', is_flag=True, help="Enable debug mode to show system prompt and tool details.")
@click.option('--prompt-style', type=click.Choice(['standard', 'meta-reasoning', 'xml']), default='standard', help="System prompt style for A/B testing.")
def main_cli(repo_path: str, model: str, api_key: str, base_url: str, theme: str, budget: int, debug: bool, prompt_style: str):
    """Interactive File Selection Agent Demo
    
    This demo creates an AI assistant that helps you select files from a repository
    using OpenAI's tool calling API with smart token management.
    
    Configuration is loaded from .env file by default. You can override specific
    values using command line options.
    
    Use --debug to see system prompt and tool calling details.
    Use --prompt-style to A/B test different prompting approaches:
    - standard: Simple, direct instructions
    - meta-reasoning: Detailed framework with reasoning guidelines
    - xml: Same as meta-reasoning but with XML formatting
    """
    
    # Get configuration from environment
    try:
        llm_config = get_llm_config_from_env()
    except ValueError as e:
        click.echo(f"❌ Configuration error: {e}", err=True)
        click.echo("💡 Make sure to set your API key in the .env file or via command line options", err=True)
        sys.exit(1)
    
    # Override with command line options if provided
    final_model = model or llm_config["model"]
    final_api_key = api_key or llm_config["api_key"]
    final_base_url = base_url or llm_config["base_url"]
    
    # Validate final configuration
    if not final_api_key:
        click.echo("❌ No API key provided. Set OPENAI_API_KEY or LLM_API_KEY in .env file or use --api-key option.", err=True)
        sys.exit(1)
    
    # Show configuration
    mode_indicator = " [DEBUG]" if debug else ""
    prompt_indicator = f" [{prompt_style.upper()}]" if prompt_style != "standard" else ""
    config_display = f"Model: {final_model}{mode_indicator}{prompt_indicator}"
    if final_base_url:
        config_display += f" | Base URL: {final_base_url}"
    config_display += f" | Budget: {budget:,} tokens"
    click.echo(f"🔧 {config_display}")
    
    try:
        # Create and run agent
        agent = FileSelectorAgent(
            repo_path=repo_path,
            openai_api_key=final_api_key,
            model=final_model,
            base_url=final_base_url,
            theme=theme,
            token_budget=budget,
            debug_mode=debug,
            prompt_style=prompt_style
        )
        agent.run()
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)

if __name__ == '__main__':
    main_cli()
