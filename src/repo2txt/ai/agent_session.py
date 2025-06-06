"""
Agent Session Management

Centralizes all state and context for a file selection session,
providing a single source of truth for the agent's runtime state.
"""

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

from ..core.models import Config, AnalysisResult
from .state import StateManager, TokenCache
from .prompts import PromptGenerator
from .tools import ToolExecutor
from .llm import LLMClient, MessageManager
from .console_chat import ChatConsole


@dataclass
class SessionConfig:
    """Configuration for an agent session."""
    repo_path: Path
    model: str
    base_url: Optional[str] = None
    theme: str = "green"
    token_budget: int = 50000
    debug_mode: bool = False
    prompt_style: str = "standard"
    use_streaming: bool = True
    enable_thinking: bool = True
    

class AgentSession:
    """Manages all state and context for a file selection session.
    
    This class serves as the single source of truth for all session state,
    making it easier to test, serialize, and reason about the agent's behavior.
    """
    
    def __init__(self, config: SessionConfig, analysis_result: AnalysisResult, 
                 openai_api_key: str):
        """Initialize session with configuration and analysis results.
        
        Args:
            config: Session configuration
            analysis_result: Pre-analyzed repository data
            openai_api_key: API key for LLM
        """
        # Core configuration
        self.config = config
        self.analysis_result = analysis_result
        
        # Runtime settings (toggleable)
        self.use_streaming = config.use_streaming
        self.enable_thinking = config.enable_thinking
        self.prompt_style = config.prompt_style
        self.debug_mode = config.debug_mode
        
        # Initialize components
        self.llm_client = LLMClient(openai_api_key, config.model, config.base_url)
        self.message_manager = MessageManager()
        self.state_manager = StateManager(analysis_result, config.token_budget)
        self.prompt_generator = PromptGenerator(self.state_manager, analysis_result)
        self.tool_executor = ToolExecutor(self.state_manager)
        
        # UI console
        self.ui = ChatConsole(
            theme=config.theme, 
            debug_mode=config.debug_mode
        )
        
        # State snapshots for undo functionality
        self.state_snapshots: List[Tuple[List[Dict], List[str], int]] = []
        
        # Repository configuration
        self.repo_config = Config()
        self.repo_config.enable_token_counting = True
        
        # Token cache (optional optimization)
        self.token_cache = TokenCache()
        
    def save_snapshot(self) -> None:
        """Save current state for undo functionality."""
        self.state_snapshots.append((
            self.message_manager.get_messages().copy(),
            self.state_manager.state.selected_files.copy(),
            self.state_manager.state.total_tokens_selected
        ))
        
    def restore_snapshot(self) -> bool:
        """Restore previous state snapshot.
        
        Returns:
            True if snapshot was restored, False if no snapshots available
        """
        if not self.state_snapshots:
            return False
            
        messages, files, tokens = self.state_snapshots.pop()
        self.message_manager.messages = messages
        self.state_manager.state.selected_files = files
        self.state_manager.state.total_tokens_selected = tokens
        return True
        
    def toggle_streaming(self) -> bool:
        """Toggle streaming mode.
        
        Returns:
            New streaming state
        """
        self.use_streaming = not self.use_streaming
        return self.use_streaming
        
    def toggle_thinking(self) -> bool:
        """Toggle thinking mode (Qwen models only).
        
        Returns:
            New thinking state
        """
        self.enable_thinking = not self.enable_thinking
        return self.enable_thinking
        
    def set_prompt_style(self, style: str) -> None:
        """Set the prompt style.
        
        Args:
            style: One of "standard", "meta-reasoning", or "xml"
        """
        if style not in ["standard", "meta-reasoning", "xml"]:
            raise ValueError(f"Invalid prompt style: {style}")
        self.prompt_style = style
        
    def cycle_prompt_style(self) -> str:
        """Cycle through available prompt styles.
        
        Returns:
            New prompt style
        """
        styles = ["standard", "meta-reasoning", "xml"]
        current_idx = styles.index(self.prompt_style)
        new_idx = (current_idx + 1) % len(styles)
        self.prompt_style = styles[new_idx]
        return self.prompt_style
        
    def set_token_budget(self, budget: int) -> None:
        """Update the token budget.
        
        Args:
            budget: New token budget (must be positive)
        """
        if budget <= 0:
            raise ValueError("Token budget must be positive")
        old_budget = self.state_manager.state.token_budget
        self.state_manager.state.token_budget = budget
        return old_budget
        
    def toggle_debug(self) -> bool:
        """Toggle debug mode.
        
        Returns:
            New debug state
        """
        self.debug_mode = not self.debug_mode
        self.ui.debug_mode = self.debug_mode
        return self.debug_mode
        
    def get_debug_info(self) -> Dict[str, Any]:
        """Get current debug information.
        
        Returns:
            Dictionary of debug information
        """
        return {
            "streaming": self.use_streaming,
            "thinking": self.enable_thinking,
            "thinking_available": "qwen" in self.config.model.lower(),
            "prompt_style": self.prompt_style,
            "debug_mode": self.debug_mode,
            "model": self.config.model,
            "base_url": self.config.base_url or "Default OpenAI",
            "selected_files": len(self.state_manager.state.selected_files),
            "token_budget": self.state_manager.state.token_budget,
            "tokens_used": self.state_manager.state.total_tokens_selected,
        }
        
    def clear_conversation(self) -> None:
        """Clear conversation history and file selection."""
        self.message_manager.clear()
        self.state_manager.clear_selection()
        # Reset thinking to default (ON)
        self.enable_thinking = True
        
    def get_selected_files(self) -> List[str]:
        """Get currently selected files.
        
        Returns:
            List of selected file paths
        """
        return self.state_manager.state.selected_files.copy()