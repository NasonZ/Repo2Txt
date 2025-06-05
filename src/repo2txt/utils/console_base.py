"""Base console functionality shared across different console implementations.

This module provides the foundation for console output with theme support,
Rich/plain text fallback, and common printing methods.

Note: This base class is intentionally minimal, containing only functionality
that is actually shared between ConsoleManager and ChatConsole.
"""

from typing import Optional, Dict, Any
from enum import Enum
from pathlib import Path
import sys
import hashlib
import time
from dataclasses import dataclass
import random
from datetime import datetime

try:
    from rich.console import Console
    from rich.theme import Theme
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class StatusType(Enum):
    """Standard status types with associated symbols."""
    SUCCESS = ("[✓]", "success", "green")
    ERROR = ("[x]", "error", "red")
    WARNING = ("[!]", "warning", "yellow")
    INFO = ("[!]", "info", "cyan")
    RUNNING = ("[~]", "running", "blue")
    PENDING = ("[.]", "pending", "dim")
    DEBUG = ("[D]", "debug", "magenta")


@dataclass
class ThemeColors:
    """Color definitions for a theme."""
    # Core status colors
    info: str
    warning: str
    error: str
    success: str
    highlight: str
    panel_border: str
    header: str
    prompt: str
    path: str
    number: str
    dim: str
    # Semantic colors for the interface
    interactive: str        # For interactive prompts and selections
    file_config: str       # For folders (repurposed from config files)
    selection_active: str  # For currently selected items
    selection_inactive: str # For unselected items
    token_count: str      # For token count displays
    accent: str           # For accent elements and highlights
    muted: str            # For secondary information
    # Additional colors for specific use cases
    primary: str = None    # Primary theme color
    secondary: str = None  # Secondary theme color
    text: str = None      # Default text color
    debug: str = None     # Debug information color
    heading: str = None   # Heading color

    def __post_init__(self):
        """Set default values for optional colors."""
        if self.primary is None:
            self.primary = self.accent
        if self.secondary is None:
            self.secondary = self.highlight
        if self.text is None:
            self.text = "white"
        if self.debug is None:
            self.debug = "magenta"
        if self.heading is None:
            self.heading = "bright_yellow"


# Define retro terminal themes
THEMES = {
    'manhattan': ThemeColors(
        info='cyan',
        warning='yellow',
        error='red',
        success='green',
        highlight='bright_cyan',
        panel_border='bright_cyan',
        header='bold bright_cyan on black',
        prompt='bright_cyan',
        path='white',
        number='bright_blue',
        dim='bright_black',
        interactive='bold bright_cyan',
        file_config='gold1',
        selection_active='bold bright_white on blue',
        selection_inactive='dim white',
        token_count='bright_blue',
        accent='cyan',
        muted='bright_black',
        primary='bright_cyan',
        secondary='cyan',
        text='white',
        debug='bright_magenta',
        heading='bright_yellow'
    ),
    'green': ThemeColors(
        info='green',
        warning='yellow',
        error='red',
        success='bright_green',
        highlight='bold green',
        panel_border='green',
        header='bold green on black',
        prompt='bright_green',
        path='bright_green',
        number='green',
        dim='green',
        interactive='bold bright_green',
        file_config='bright_yellow',
        selection_active='bold black on green',
        selection_inactive='dim green',
        token_count='bright_white',
        accent='bright_green',
        muted='dim green',
        primary='bright_green',
        secondary='green',
        text='white',
        debug='bright_magenta',
        heading='bright_cyan'
    ),
    'matrix': ThemeColors(
        info='bright_green',
        warning='yellow',
        error='red',
        success='green',
        highlight='bold bright_green',
        panel_border='green',
        header='bold bright_green on black',
        prompt='bright_green',
        path='green',
        number='bright_green',
        dim='green',
        interactive='bold bright_green',
        file_config='bright_cyan',
        selection_active='bold black on bright_green',
        selection_inactive='dim green',
        token_count='bright_white',
        accent='bright_green',
        muted='dim green',
        primary='bright_green',
        secondary='green',
        text='green',
        debug='bright_magenta',
        heading='bright_yellow'
    ),
    'sunset': ThemeColors(
        info='orange3',
        warning='yellow',
        error='red3',
        success='green',
        highlight='bold orange1',
        panel_border='orange3',
        header='bold orange1 on black',
        prompt='orange1',
        path='wheat1',
        number='orange1',
        dim='grey50',
        interactive='bold bright_cyan',
        file_config='gold1',
        selection_active='bold black on orange1',
        selection_inactive='dim orange3',
        token_count='orange1',
        accent='dark_orange3',
        muted='grey50',
        primary='dark_orange3',  
        secondary='yellow',
        text='white',
        debug='bright_magenta',
        heading='dark_orange3'
    )
}


class ConsoleBase:
    """Base class for console management with theme support and Rich/plain fallback."""
    
    def __init__(self, theme: str = "manhattan", file: Optional[Any] = None, 
                 force_plain: bool = False, sanitize_paths: bool = False,
                 repo_root: Optional[Path] = None):
        """Initialize console base.
        
        Args:
            theme: Theme name from THEMES
            file: Output file (defaults to sys.stdout)
            force_plain: Force plain output even if Rich is available
            sanitize_paths: Enable path sanitization for privacy
            repo_root: Repository root for path sanitization
        """
        self.theme_name = theme
        self.theme_colors = THEMES.get(theme, THEMES['manhattan'])
        self.file = file or sys.stdout
        self.sanitize_paths = sanitize_paths
        self.repo_root = repo_root
        
        # Determine if we can/should use Rich
        should_use_rich = self._should_use_rich_terminal()
        
        self.use_rich = (
            RICH_AVAILABLE and 
            not force_plain and
            should_use_rich
        )
        
        if self.use_rich:
            # Create Rich theme from our theme colors
            rich_theme = self._create_rich_theme()
            
            # Get terminal width
            import os
            try:
                terminal_width = os.get_terminal_size().columns
                console_width = max(80, terminal_width)
            except (OSError, AttributeError):
                console_width = 80
                
            self.console = Console(
                theme=rich_theme, 
                file=self.file, 
                force_terminal=should_use_rich,
                width=console_width,
                highlight=False
            )
        else:
            self.console = None
    
    def _should_use_rich_terminal(self) -> bool:
        """Enhanced terminal detection for Rich compatibility."""
        import os
        
        # Standard TTY detection
        if hasattr(self.file, 'isatty') and self.file.isatty():
            return True
        
        # Check for NO_COLOR environment variable (universal disable)
        if os.environ.get('NO_COLOR'):
            return False
        
        # Check for color terminal indicators
        term = os.environ.get('TERM', '').lower()
        colorterm = os.environ.get('COLORTERM', '').lower()
        force_color = os.environ.get('FORCE_COLOR')
        
        # Explicit force color
        if force_color:
            return True
        
        # Common color terminal indicators
        color_terms = {
            'xterm-256color', 'xterm-color', 'screen-256color', 'tmux-256color',
            'rxvt-unicode-256color', 'alacritty', 'kitty'
        }
        
        if term in color_terms or 'color' in term:
            return True
        
        # COLORTERM indicates true color support
        if colorterm in {'truecolor', '24bit'}:
            return True
        
        # For benchmarking tools and scripts, assume Rich is desired unless explicitly disabled
        if term and not os.environ.get('CI'):
            return True
        
        return False
    
    def _create_rich_theme(self) -> 'Theme':
        """Create Rich theme from our theme colors."""
        theme_dict = {
            'info': self.theme_colors.info,
            'warning': self.theme_colors.warning,
            'error': self.theme_colors.error,
            'success': self.theme_colors.success,
            'highlight': self.theme_colors.highlight,
            'panel.border': self.theme_colors.panel_border,
            'header': self.theme_colors.header,
            'prompt': self.theme_colors.prompt,
            'path': self.theme_colors.path,
            'number': self.theme_colors.number,
            'dim': self.theme_colors.dim,
            'interactive': self.theme_colors.interactive,
            'file_config': self.theme_colors.file_config,
            'selection_active': self.theme_colors.selection_active,
            'selection_inactive': self.theme_colors.selection_inactive,
            'token_count': self.theme_colors.token_count,
            'accent': self.theme_colors.accent,
            'muted': self.theme_colors.muted,
            'primary': self.theme_colors.primary,
            'secondary': self.theme_colors.secondary,
            'text': self.theme_colors.text,
            'debug': self.theme_colors.debug,
            'heading': self.theme_colors.heading,
            # Add basic color names for direct usage
            'cyan': 'cyan',
            'red': 'red',
            'green': 'green',
            'blue': 'blue',
            'yellow': 'yellow',
            'magenta': 'magenta',
            'white': 'white',
            'black': 'black',
            'bright_cyan': 'bright_cyan',
            'bright_red': 'bright_red',
            'bright_green': 'bright_green',
            'bright_blue': 'bright_blue',
            'bright_yellow': 'bright_yellow',
            'bright_magenta': 'bright_magenta',
            'bright_white': 'bright_white',
            'bright_black': 'bright_black'
        }
        return Theme(theme_dict)
    
    def _sanitize_content(self, content: str) -> str:
        """Sanitize paths in content if enabled."""
        if not self.sanitize_paths:
            return content
            
        if self.repo_root:
            content = content.replace(str(self.repo_root), "REPO_ROOT")
        
        home_path = str(Path.home())
        content = content.replace(home_path, "USER_HOME")
        
        return content
    
    def print(self, *args, **kwargs):
        """Print with optional Rich formatting."""
        # Apply path sanitization if enabled
        sanitized_args = []
        for arg in args:
            if isinstance(arg, str):
                sanitized_args.append(self._sanitize_content(arg))
            else:
                sanitized_args.append(arg)
        
        if self.use_rich:
            self.console.print(*sanitized_args, **kwargs)
        else:
            # Filter out Rich-specific kwargs for plain print
            plain_kwargs = {k: v for k, v in kwargs.items() 
                           if k not in ['style', 'justify', 'overflow', 'crop', 
                                       'soft_wrap', 'markup', 'highlight']}
            plain_kwargs['file'] = self.file
            print(*sanitized_args, **plain_kwargs)
    
    def print_status(self, status: StatusType, message: str, prefix: str = ""):
        """Print a status line with icon."""
        icon, _, color = status.value
        message = self._sanitize_content(message)
        
        if self.use_rich:
            status_text = Text()
            if prefix:
                status_text.append(prefix + " ")
            status_text.append(f"{icon} ", style=color)
            status_text.append(message)
            self.console.print(status_text)
        else:
            line = f"{prefix}{icon} {message}"
            print(line, file=self.file)
    
    def print_error(self, message: str):
        """Print an error message."""
        self.print_status(StatusType.ERROR, message)
    
    def print_success(self, message: str):
        """Print a success message."""
        self.print_status(StatusType.SUCCESS, message)
    
    def print_info(self, message: str):
        """Print an info message."""
        self.print_status(StatusType.INFO, message)
    
    def print_info_with_heading(self, heading: str, value: str):
        """Print an info message with a colored heading and regular value."""
        message = self._sanitize_content(f"{heading} {value}")
        
        if self.use_rich:
            from rich.text import Text
            text = Text()
            text.append("i ", style=self.theme_colors.info)
            text.append(heading, style=self.theme_colors.heading)
            text.append(f" {value}", style=self.theme_colors.info)
            self.console.print(text)
        else:
            # Fallback for plain text
            icon, _, _ = StatusType.INFO.value
            print(f"{icon} {heading} {value}", file=self.file)
    
    def print_warning(self, message: str):
        """Print a warning message."""
        self.print_status(StatusType.WARNING, message)
    
    def print_separator(self, char: str = "─", width: int = 80):
        """Print a separator line."""
        if self.use_rich:
            self.console.print(char * width, style="dim")
        else:
            print(char * width, file=self.file)
    
    def print_exception(self):
        """Print exception traceback with Rich formatting if available."""
        if self.use_rich:
            self.console.print_exception()
        else:
            import traceback
            traceback.print_exc(file=self.file)

def get_alternating_theme() -> str:
    """Manhattan during day, sunset in evening, with 20% chance for any theme."""
    import time
    import random
    from datetime import datetime
    
    # Get current local hour (0-23)
    local_hour = datetime.now().hour
    
    # Day theme: 6 AM to 6 PM = manhattan, otherwise sunset
    is_day = 6 <= local_hour < 18
    base_theme = 'manhattan' if is_day else 'sunset'
    
    # 20% chance to pick any theme
    if random.random() < 0.2:
        return random.choice(['manhattan', 'green', 'matrix', 'sunset'])
    else:
        return base_theme