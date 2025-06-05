"""AI-powered file selection for repo2txt.

This package provides AI-assisted file selection capabilities using
LLMs to help users intelligently select relevant files from repositories
within token budgets.
"""

from .state import FileSelectionState, StateManager
from .tools import Tool, ToolCall, ToolResult, ToolExecutor, create_file_selection_tools
from .prompts import PromptGenerator
from .llm import LLMClient, MessageManager, get_llm_config_from_env
from .file_selector_agent import FileSelectorAgent
from .console_chat import ChatConsole

__all__ = [
    # State management
    'FileSelectionState',
    'StateManager',
    
    # Tool system
    'Tool',
    'ToolCall', 
    'ToolResult',
    'ToolExecutor',
    'create_file_selection_tools',
    
    # Prompts
    'PromptGenerator',
    
    # LLM interaction
    'LLMClient',
    'MessageManager',
    'get_llm_config_from_env',
    
    # Main agent
    'FileSelectorAgent',
    
    # Console
    'ChatConsole',
]