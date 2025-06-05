"""Shared console utilities for consistent UI across repo2txt tools.

This module provides a unified interface for console output that works with
both Rich (for interactive terminals) and plain text (for logs/CI).
"""

from typing import Optional, List, Tuple, Dict, Any, Union
import sys

from .console_base import ConsoleBase, StatusType, ThemeColors, THEMES

try:
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.panel import Panel
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


# StatusType, ThemeColors, and THEMES are now imported from console_base


class ConsoleManager(ConsoleBase):
    """Unified console management supporting Rich and plain output.
    
    Extends ConsoleBase with additional features like tables, progress bars,
    file lists, and other batch processing UI elements.
    """
    
    def __init__(self, use_rich: bool = True, theme: str = "manhattan", 
                 file: Optional[Any] = None, force_plain: bool = False,
                 sanitize_paths: bool = False, repo_root: Optional[Any] = None):
        """Initialize console manager.
        
        Args:
            use_rich: Whether to use Rich formatting if available
            theme: Theme name from THEMES
            file: Output file (defaults to sys.stdout)
            force_plain: Force plain output even if Rich is available
            sanitize_paths: Enable path sanitization (default: False)
            repo_root: Repository root for path sanitization
        """
        # Initialize base class
        super().__init__(theme=theme, file=file, force_plain=force_plain,
                        sanitize_paths=sanitize_paths, repo_root=repo_root)
    
    # Methods from ConsoleBase are inherited:
    # - _should_use_rich_terminal()
    # - _create_rich_theme()
    # - _sanitize_content()
    # - print()
    # - print_status()
    # - print_error(), print_success(), print_info(), print_warning()
    # - print_separator()
    # - print_exception()
    
    def print_header(self, title: str, subtitle: Optional[str] = None, 
                     width: int = 80, style: str = "header"):
        """Print a section header with consistent styling."""
        if self.use_rich:
            # Rich formatted header
            header_text = Text(title.center(width - 4), style=style)
            if subtitle:
                header_text.append("\n" + subtitle.center(width - 4), style="dim")
            self.console.print(Panel(header_text, width=width, style="panel.border"))
        else:
            # Plain text header
            border = "═" * width
            print(f"╔{border}╗", file=self.file)
            print(f"║ {title.center(width - 2)} ║", file=self.file)
            if subtitle:
                print(f"║ {subtitle.center(width - 2)} ║", file=self.file)
            print(f"╚{border}╝", file=self.file)
    
    def print_status(self, status: StatusType, message: str, 
                     details: Optional[str] = None, prefix: str = ""):
        """Print a status line with icon and optional details.
        
        Extends base implementation to support details parameter.
        """
        message = self._sanitize_content(message)
        icon, _, color = status.value
        
        if self.use_rich:
            status_text = Text()
            if prefix:
                status_text.append(prefix + " ")
            status_text.append(f"{icon} ", style=color)
            status_text.append(message)
            if details:
                status_text.append(f" - {self._sanitize_content(details)}", style="dim")
            self.console.print(status_text)
        else:
            line = f"{prefix}{icon} {message}"
            if details:
                line += f" - {details}"
            print(line, file=self.file)
    
    def print_progress_bar(self, current: float, total: float, 
                          label: str = "", width: int = 30, 
                          show_percentage: bool = True):
        """Print a progress bar using Unicode characters."""
        if total == 0:
            percentage = 0
        else:
            percentage = min(100, (current / total) * 100)
        
        filled_width = int(width * percentage / 100)
        empty_width = width - filled_width
        
        bar = "█" * filled_width + "░" * empty_width
        
        if self.use_rich:
            progress_text = Text()
            if label:
                progress_text.append(f"{label}: ")
            progress_text.append(bar, style="highlight")
            if show_percentage:
                progress_text.append(f" {percentage:>3.0f}%", style="number")
            self.console.print(progress_text)
        else:
            line = f"{label}: " if label else ""
            line += bar
            if show_percentage:
                line += f" {percentage:>3.0f}%"
            print(line, file=self.file)
    
    def print_table(self, headers: List[str], rows: List[List[str]], 
                    title: Optional[str] = None, caption: Optional[str] = None):
        """Print a formatted table (Rich table or ASCII)."""
        if self.use_rich:
            table = Table(title=title, caption=caption, show_header=True, 
                         header_style="highlight")
            
            for header in headers:
                table.add_column(header)
            
            for row in rows:
                table.add_row(*row)
            
            self.console.print(table)
        else:
            # ASCII table
            # Calculate column widths
            col_widths = [len(h) for h in headers]
            for row in rows:
                for i, cell in enumerate(row):
                    col_widths[i] = max(col_widths[i], len(str(cell)))
            
            # Print title
            if title:
                print(f"\n{title}", file=self.file)
                print("-" * sum(col_widths), file=self.file)
            
            # Print headers
            header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
            print(header_line, file=self.file)
            print("-" * len(header_line), file=self.file)
            
            # Print rows
            for row in rows:
                row_line = " | ".join(str(c).ljust(w) for c, w in zip(row, col_widths))
                print(row_line, file=self.file)
            
            # Print caption
            if caption:
                print(f"\n{caption}", file=self.file)
    
    def print_trajectory(self, attempts: List[Dict[str, Any]], 
                        test_name: str, show_details: bool = True):
        """Print a trajectory visualization for benchmark attempts."""
        if not attempts:
            return
        
        self.print_header(f"🔬 Trajectory Analysis: {test_name}", width=60)
        
        for i, attempt in enumerate(attempts):
            # Status based on validation error
            if "SUCCESS" in attempt.get("validation_error", ""):
                status = StatusType.SUCCESS
            else:
                status = StatusType.ERROR
            
            # Stage icon
            stage_icons = {
                "PATH_VALIDATION": "📁",
                "PATH_HEALING": "🔧",
                "BUDGET_VALIDATION": "💰"
            }
            stage = attempt.get("stage", "UNKNOWN")
            stage_icon = stage_icons.get(stage, "❓")
            
            # Print attempt header
            self.print_status(
                status,
                f"Attempt {attempt.get('attempt_number', i+1)}: {stage_icon} {stage}",
                details=f"{attempt.get('duration_ms', 0):.0f}ms"
            )
            
            if show_details:
                # Show file changes
                files_added = attempt.get("files_added", [])
                files_removed = attempt.get("files_removed", [])
                
                if files_added:
                    self.print(f"  [dim]Added:[/dim] {', '.join(files_added[:3])}", 
                              style="success" if self.use_rich else None)
                if files_removed:
                    self.print(f"  [dim]Removed:[/dim] {', '.join(files_removed[:3])}", 
                              style="error" if self.use_rich else None)
                
                # Show token change
                token_delta = attempt.get("token_delta", 0)
                if token_delta != 0:
                    delta_str = f"+{token_delta}" if token_delta > 0 else str(token_delta)
                    self.print(f"  [dim]Token Δ:[/dim] {delta_str}", 
                              style="number" if self.use_rich else None)
    
    # print_separator() inherited from ConsoleBase
    
    def print_file_list(self, files: List[Tuple[str, int]], title: str = "Files", 
                       show_index: bool = True, highlight_index: Optional[int] = None):
        """Print a styled file list with token counts and visual hierarchy."""
        if not files:
            return
            
        if self.use_rich:
            # Rich formatted list with enhanced styling
            self.console.print(f"\n[accent]> {title}:[/accent]")
            
            for i, (filename, tokens) in enumerate(files, 1):
                index_style = 'selection_active' if highlight_index == i else 'muted'
                
                # Create formatted line
                line_parts = []
                if show_index:
                    line_parts.append(f"[{index_style}]{i:>3}.[/{index_style}]")
                
                # Simple file icon
                line_parts.extend([
                    f" 📄",
                    f" {filename}",
                    f" [token_count]~{tokens:,} tokens[/token_count]"
                ])
                
                self.console.print("".join(line_parts))
        else:
            # Plain text version
            print(f"\n> {title}:", file=self.file)
            for i, (filename, tokens) in enumerate(files, 1):
                prefix = f"{i:>3}. " if show_index else "  "
                marker = ">> " if highlight_index == i else "   "
                print(f"{marker}{prefix}{filename} ~{tokens:,} tokens", file=self.file)
    
    def print_interactive_prompt(self, message: str, options: Optional[str] = None,
                               show_cursor: bool = True):
        """Print an interactive prompt with enhanced styling."""
        cursor = " ▶" if show_cursor else ""
        
        if self.use_rich:
            prompt_text = f"[interactive]?[/interactive] [prompt]{message}[/prompt]{cursor}"
            if options:
                prompt_text += f"\n  [muted]{options}[/muted]"
            self.console.print(prompt_text)
        else:
            line = f"? {message}{cursor}"
            if options:
                line += f"\n  {options}"
            print(line, file=self.file)
    
    def print_selection_feedback(self, selection: str, is_valid: bool = True):
        """Print feedback for user selections with appropriate styling."""
        if self.use_rich:
            if is_valid:
                self.console.print(f"[success]✓[/success] [accent]{selection}[/accent]")
            else:
                self.console.print(f"[error]✗[/error] [dim]{selection}[/dim]")
        else:
            symbol = "✓" if is_valid else "✗"
            print(f"{symbol} {selection}", file=self.file)
    
    def print_section_header(self, title: str, icon: str = "", level: int = 1):
        """Print a section header with visual hierarchy."""
        if self.use_rich:
            if level == 1:
                # Main section
                self.console.print(f"\n[accent]{icon} {title}[/accent]")
                self.console.print("[accent]" + "─" * (len(title) + len(icon) + 1) + "[/accent]")
            elif level == 2:
                # Subsection
                self.console.print(f"\n[highlight]{icon} {title}[/highlight]")
            else:
                # Minor section
                self.console.print(f"[muted]{icon} {title}[/muted]")
        else:
            prefix = "==" if level == 1 else "--" if level == 2 else "  "
            print(f"\n{prefix} {icon} {title}", file=self.file)
    
    def print_token_summary(self, total_tokens: int, file_count: int, 
                          context: str = "Selected"):
        """Print a token summary with enhanced formatting."""
        if self.use_rich:
            self.console.print(
                f"[muted]{context}:[/muted] "
                f"[number]{file_count}[/number] [muted]files,[/muted] "
                f"[token_count]{total_tokens:,}[/token_count] [muted]tokens[/muted]"
            )
        else:
            print(f"{context}: {file_count} files, {total_tokens:,} tokens", file=self.file)
    
    def create_progress_context(self):
        """Create a progress context for long-running operations."""
        if self.use_rich:
            return Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=self.console
            )
        else:
            # Return a dummy context manager for plain output
            class DummyProgress:
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
                def add_task(self, description, total):
                    print(f"Starting: {description}", file=self.file)
                    return None
                def update(self, task_id, advance=1):
                    pass
            
            return DummyProgress()
    
    def print_breadcrumbs(self, path_components: List[str], separator: str = " > "):
        """Print navigation breadcrumbs."""
        if self.use_rich:
            breadcrumb = Text()
            for i, component in enumerate(path_components):
                if i > 0:
                    breadcrumb.append(separator, style="dim")
                breadcrumb.append(component, style="path")
            self.console.print(breadcrumb)
        else:
            breadcrumb = separator.join(path_components)
            print(breadcrumb, file=self.file)
    
    def print_status_bar(self, selected_count: int, total_tokens: int = 0, show_tokens: bool = True):
        """Print bottom status bar with selection information."""
        if show_tokens and total_tokens > 0:
            status_text = f"Selected: {selected_count} files, {total_tokens:,} tokens"
        else:
            status_text = f"Selected: {selected_count} files"
        
        if self.use_rich:
            # Create a simple bottom status line
            self.console.print(f"\n[dim]─[/dim] [accent]{status_text}[/accent] [dim]─[/dim]")
        else:
            print(f"\n--- {status_text} ---")
    
    # print_exception() inherited from ConsoleBase


# Convenience functions for quick access
_default_console = None

def get_console(theme: str = "manhattan", force_new: bool = False) -> ConsoleManager:
    """Get or create a default console manager."""
    global _default_console
    if _default_console is None or force_new:
        _default_console = ConsoleManager(theme=theme)
    return _default_console


def print_status(status: StatusType, message: str, details: Optional[str] = None):
    """Quick status print using default console."""
    console = get_console()
    console.print_status(status, message, details)


def print_header(title: str, subtitle: Optional[str] = None):
    """Quick header print using default console."""
    console = get_console()
    console.print_header(title, subtitle)


def print_progress(current: float, total: float, label: str = ""):
    """Quick progress bar using default console."""
    console = get_console()
    console.print_progress_bar(current, total, label)