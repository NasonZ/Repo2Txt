"""Chat-specific console implementation for AI interactions.

This module provides specialized console output for interactive chat sessions
with AI, including streaming support, debug panels, and tool visualization.
"""

from typing import Optional, Dict, Any, TYPE_CHECKING
from pathlib import Path
import json

from ..utils.console_base import ConsoleBase, StatusType

try:
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

if TYPE_CHECKING:
    from .file_selector_agent import StateManager, ToolCall, ToolResult


class ChatConsole(ConsoleBase):
    """Console implementation specialized for AI chat interactions.
    
    Extends ConsoleBase with chat-specific features like streaming output,
    debug panels, tool call visualization, and state diff display.
    """
    
    def __init__(self, theme: str = "green", repo_root: Optional[Path] = None, 
                 debug_mode: bool = False, file: Optional[Any] = None,
                 force_plain: bool = False):
        """Initialize chat console.
        
        Args:
            theme: Theme name from THEMES
            repo_root: Repository root for path sanitization
            debug_mode: Enable debug output (tool calls, system prompts, etc.)
            file: Output file (defaults to sys.stdout)
            force_plain: Force plain output even if Rich is available
        """
        # Always enable path sanitization for chat
        super().__init__(theme=theme, file=file, force_plain=force_plain,
                        sanitize_paths=True, repo_root=repo_root)
        
        self.debug_mode = debug_mode
        self._shown_initial_system_prompt = False
        
        # Quick access to colors dictionary for compatibility
        self.colors = {
            'primary': self.theme_colors.primary,
            'secondary': self.theme_colors.secondary,
            'accent': self.theme_colors.accent,
            'text': self.theme_colors.text,
            'dim': self.theme_colors.dim,
            'error': self.theme_colors.error,
            'warning': self.theme_colors.warning,
            'success': self.theme_colors.success,
            'debug': self.theme_colors.debug
        }
    
    def print_banner(self, title: str = "FILE SELECTOR AGENT", 
                     subtitle: str = "AI-Assisted File Selection"):
        """Print a stylized banner for the chat interface."""
        # Create retro-style banner with simple text
        debug_line = "░░ DEBUG MODE ENABLED ░░" if self.debug_mode else ""
        
        banner_text = f"""
▓▒░                                                                ░▒▓ 
▓▒░                Intelligent File Scouting                       ░▒▓ 
{f'▓▒░                 {debug_line}                       ░▒▓' if debug_line else '▓▒░                                                                ░▒▓'}
▓▒░                                                                ░▒▓ 
▓▒░                                                                ░▒▓
"""
        if self.use_rich:
            self.console.print(Text(banner_text, style=self.theme_colors.primary))
        else:
            print(banner_text, file=self.file)
    
    def print_section(self, title: str, content: Any = ""):
        """Print a content section with a titled panel."""
        text_content = content if isinstance(content, (str, Text)) else str(content)
        text_content = self._sanitize_content(text_content)
        
        if len(text_content) > 2000:
            text_content = text_content[:2000] + "\n... [Section Content Truncated]"
        
        if self.use_rich:
            panel = Panel(
                text_content,
                title=f"[{self.theme_colors.accent}]{title}[/{self.theme_colors.accent}]",
                border_style=self.theme_colors.secondary,
                padding=(1, 2)
            )
            self.console.print(panel)
        else:
            print(f"\n=== {title} ===", file=self.file)
            print(text_content, file=self.file)
            print("=" * (len(title) + 8), file=self.file)
    
    def print_debug_system_prompt(self, system_prompt: str):
        """Print system prompt in debug mode - only once at start."""
        if not self.debug_mode or self._shown_initial_system_prompt:
            return
        
        display_prompt = self._truncate_file_tree(system_prompt)
        
        if self.use_rich:
            panel = Panel(
                self._sanitize_content(display_prompt),
                title=f"[{self.theme_colors.debug}]🔍 SYSTEM PROMPT (DEBUG)[/{self.theme_colors.debug}]",
                border_style=self.theme_colors.debug,
                padding=(1, 2)
            )
            self.console.print(panel)
        else:
            print("\n🔍 SYSTEM PROMPT (DEBUG)", file=self.file)
            print("-" * 50, file=self.file)
            print(display_prompt, file=self.file)
            print("-" * 50, file=self.file)
        
        self._shown_initial_system_prompt = True
    
    def _truncate_file_tree(self, system_prompt: str) -> str:
        """Truncate file tree section in system prompt for display."""
        # Try to find and truncate the file tree section
        tree_section_patterns = [
            "Repository Structure (with token counts):",
            "## Repository Structure",
            "Repository Structure:",
            "File Tree:",
        ]
        
        tree_end_patterns = [
            "Available Tools:",
            "Guidelines:",
            "Meta-Reasoning Framework",
            "Selection Strategy:",
            "Critical Constraint:",
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
            tree_end_pos = len(system_prompt)
            search_start = tree_start_pos + len(tree_start_pattern)
            
            for pattern in tree_end_patterns:
                pos = system_prompt.find(pattern, search_start)
                if pos != -1:
                    tree_end_pos = pos
                    break
            
            # Extract and truncate
            pre_tree = system_prompt[:tree_start_pos + len(tree_start_pattern)]
            tree_content = system_prompt[tree_start_pos + len(tree_start_pattern):tree_end_pos]
            post_tree = system_prompt[tree_end_pos:]
            
            max_tree_display_length = 1500
            if len(tree_content) > max_tree_display_length:
                tree_content = tree_content[:max_tree_display_length] + "\n... [Tree Truncated for Display - Full tree available to AI]"
            
            return pre_tree + tree_content + post_tree
        else:
            # Fallback: truncate whole prompt if too long
            max_total_length = 3000
            if len(system_prompt) > max_total_length:
                return system_prompt[:max_total_length] + "\n... [System Prompt Truncated for Display]"
            return system_prompt
    
    def print_state_diff(self, state_manager: 'StateManager'):
        """Show what changed in the file selection state."""
        # For first selection, treat all files as added
        prev_set = set(state_manager.state.previous_files) if state_manager.state.previous_files else set()
        curr_set = set(state_manager.state.selected_files)
        
        added = curr_set - prev_set
        removed = prev_set - curr_set
        
        if not added and not removed:
            return
        
        # Show changes with appropriate styling
        if self.debug_mode:
            self.print(f"\n[{self.theme_colors.debug}]🔍 DEBUG: Selection Changes[/{self.theme_colors.debug}]")
        else:
            self.print(f"\n[{self.theme_colors.accent}]📝 Selection Changes:[/{self.theme_colors.accent}]")
        
        for path in sorted(removed):
            self.print(f"  [{self.theme_colors.error}]-[/{self.theme_colors.error}] {path}")
        for path in sorted(added):
            self.print(f"  [{self.theme_colors.success}]+[/{self.theme_colors.success}] {path}")
        
        # Show token impact with budget percentage
        token_diff = state_manager.state.total_tokens_selected - state_manager.state.previous_tokens
        budget_percentage = state_manager.state.get_budget_usage_percent()
        
        if self.debug_mode:
            self.print(f"\n[{self.theme_colors.debug}]🔍 DEBUG: Token Impact[/{self.theme_colors.debug}]")
        else:
            self.print(f"\n[{self.theme_colors.accent}]📊 Token Impact:[/{self.theme_colors.accent}]")
        
        self.print(f"New total: [{self.theme_colors.token_count}]{state_manager.state.total_tokens_selected:,}[/{self.theme_colors.token_count}] tokens (was [{self.theme_colors.token_count}]{state_manager.state.previous_tokens:,}[/{self.theme_colors.token_count}])")
        self.print(f"Budget usage: [{self.theme_colors.token_count}]{budget_percentage:.1f}%[/{self.theme_colors.token_count}] ([{self.theme_colors.token_count}]{state_manager.state.total_tokens_selected:,}[/{self.theme_colors.token_count}]/[{self.theme_colors.token_count}]{state_manager.state.token_budget:,}[/{self.theme_colors.token_count}])")
        
        if token_diff > 0:
            self.print(f"[{self.theme_colors.warning}]+{token_diff:,} tokens[/{self.theme_colors.warning}]")
        elif token_diff < 0:
            self.print(f"[{self.theme_colors.success}]{token_diff:,} tokens[/{self.theme_colors.success}]")
        else:
            self.print(f"[{self.theme_colors.dim}]No token change[/{self.theme_colors.dim}]")
    
    def print_tool_call(self, tool_call: 'ToolCall'):
        """Display tool call details in debug mode."""
        if not self.debug_mode:
            return
        
        if self.use_rich and RICH_AVAILABLE:
            table = Table(show_header=False, box=None, padding=(0, 1))
            table.add_column("Key", style=self.theme_colors.accent)
            table.add_column("Value", style=self.theme_colors.text)
            table.add_row("Tool:", tool_call.tool_name)
            table.add_row("ID:", tool_call.id)
            
            for key, value in tool_call.input.items():
                val_str = json.dumps(value, indent=2) if isinstance(value, (list, dict)) else str(value)
                table.add_row(f"{key}:", self._sanitize_content(val_str))
            
            panel = Panel(
                table, 
                title=f"[{self.theme_colors.warning}]🔧 TOOL CALL (DEBUG)[/{self.theme_colors.warning}]",
                border_style=self.theme_colors.warning,
                padding=(0, 1)
            )
            self.console.print(panel)
        else:
            print("\n🔧 TOOL CALL (DEBUG)", file=self.file)
            print(f"Tool: {tool_call.tool_name}", file=self.file)
            print(f"ID: {tool_call.id}", file=self.file)
            for key, value in tool_call.input.items():
                print(f"{key}: {value}", file=self.file)
    
    def print_tool_result(self, result: 'ToolResult'):
        """Display tool result in debug mode."""
        if not self.debug_mode:
            return
        
        if result.error:
            style = self.theme_colors.error
            title = "❌ TOOL ERROR (DEBUG)"
            content_text = f"Error: {result.error}"
        else:
            style = self.theme_colors.success
            title = "✅ TOOL RESULT (DEBUG)"
            content_text = json.dumps(result.output, indent=2) if result.is_json_output else str(result.output)
            content_text = self._sanitize_content(content_text)
        
        if len(content_text) > 2500:
            content_text = content_text[:2500] + "\n... [Result Truncated]"
        
        if self.use_rich:
            panel = Panel(
                content_text,
                title=f"[{style}]{title}[/{style}]",
                border_style=style,
                padding=(0, 1)
            )
            self.console.print(panel)
        else:
            print(f"\n{title}", file=self.file)
            print(content_text, file=self.file)
    
    def print_debug_info(self, message: str):
        """Print debug information with special styling."""
        if not self.debug_mode:
            return
        
        message = self._sanitize_content(message)
        if self.use_rich:
            self.console.print(f"[{self.theme_colors.debug}]🔍 DEBUG: {message}[/{self.theme_colors.debug}]")
        else:
            print(f"🔍 DEBUG: {message}", file=self.file)
    
    def print_streaming_delta(self, delta: str, is_tool_call: bool = False):
        """Print streaming content from AI responses."""
        delta = self._sanitize_content(delta)
        
        if self.use_rich:
            style = self.theme_colors.warning if is_tool_call else self.theme_colors.text
            self.console.print(delta, style=style, end="")
        else:
            print(delta, end="", file=self.file)