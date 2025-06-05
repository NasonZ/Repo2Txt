"""LLM interaction and streaming management.

This module handles all interactions with the LLM API, including
streaming responses, message management, and response parsing.
"""

import json
from typing import Dict, List, Any, Optional, Tuple, AsyncGenerator

from openai import OpenAI, APIError

from .tools import ToolCall, parse_openai_tool_calls
from ..ai.console_chat import ChatConsole
from .qwen_utils import clean_thinking_tags as qwen_clean_thinking_tags


class LLMClient:
    """Manages LLM interactions and streaming."""
    
    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None):
        """Initialize LLM client.
        
        Args:
            api_key: API key for the LLM service
            model: Model name to use
            base_url: Optional base URL for the API
        """
        self.model = model
        
        # Initialize OpenAI client
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)
        
        # Default parameters for requests
        self.default_params = {}
        
        # Add Qwen-specific parameters if using Qwen model
        if "qwen" in model.lower():
            self.default_params.update({
                "temperature": 0.7,
                "top_p": 0.8,
                "extra_body": {
                    "top_k": 20,
                    "presence_penalty": 1.5
                }
            })
    
    def create_completion(self, messages: List[Dict[str, Any]], 
                         tools: Optional[List[Dict[str, Any]]] = None,
                         stream: bool = False) -> Any:
        """Create a chat completion.
        
        Args:
            messages: List of message dictionaries
            tools: Optional list of tool definitions
            stream: Whether to stream the response
            
        Returns:
            Completion object or stream
        """
        params = {
            "model": self.model,
            "messages": messages,
            **self.default_params
        }
        
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"
        
        if stream:
            params["stream"] = True
        
        return self.client.chat.completions.create(**params)
    
    def clean_thinking_tags(self, content: str) -> str:
        """Remove <think>...</think> blocks from content.
        
        Args:
            content: Content potentially containing thinking tags
            
        Returns:
            Cleaned content
        """
        return qwen_clean_thinking_tags(content)
    
    async def handle_streaming_response(self, stream_response: Any, 
                                      ui: ChatConsole) -> Tuple[str, List[ToolCall]]:
        """Handle streaming response from the LLM.
        
        Args:
            stream_response: Streaming response object
            ui: Console UI for output
            
        Returns:
            Tuple of (content, tool_calls)
        """
        content_buffer = ""
        tool_calls_data_buffer: Dict[int, Dict[str, Any]] = {}
        
        ui.print(f"\n[{ui.colors['accent']}][<] [/{ui.colors['accent']}] ", end="")
        
        for chunk in stream_response:
            delta = chunk.choices[0].delta
            
            if delta.content:
                content_buffer += delta.content
                ui.print_streaming_delta(delta.content)
            
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_data_buffer:
                        tool_calls_data_buffer[idx] = {
                            "id": None, 
                            "function": {"name": None, "arguments": ""}
                        }
                    
                    if tc_delta.id:
                        tool_calls_data_buffer[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tool_calls_data_buffer[idx]["function"]["name"] = tc_delta.function.name
                            ui.print(f"\n[{ui.colors['warning']}]🔧 Calling tool {idx + 1}: "
                                   f"{tc_delta.function.name}[/{ui.colors['warning']}]")
                            if ui.debug_mode:
                                ui.print(f" [{ui.colors['dim']}]Args:[/{ui.colors['dim']}] ", end="")
                        if tc_delta.function.arguments:
                            tool_calls_data_buffer[idx]["function"]["arguments"] += tc_delta.function.arguments
                            if ui.debug_mode:
                                ui.print_streaming_delta(tc_delta.function.arguments, is_tool_call=True)
        
        ui.print()
        
        # Convert buffer to tool calls
        openai_tool_calls_list_from_stream = []
        for _idx, call_data in sorted(tool_calls_data_buffer.items()):
            if call_data["id"] and call_data["function"]["name"]:
                # Create mock objects that match OpenAI's structure
                mock_function = type('Function', (), call_data["function"])()
                mock_tool_call = type('ToolCall', (), {
                    'id': call_data["id"], 
                    'function': mock_function, 
                    'type': 'function'
                })()
                openai_tool_calls_list_from_stream.append(mock_tool_call)
            else:
                ui.print_error(f"Incomplete tool call from stream at index {_idx}: {call_data}")
        
        return content_buffer, parse_openai_tool_calls(openai_tool_calls_list_from_stream)


class MessageManager:
    """Manages conversation message history."""
    
    def __init__(self):
        self.messages: List[Dict[str, Any]] = []
        
    def add_system_message(self, content: str):
        """Add or update the system message."""
        if self.messages and self.messages[0]["role"] == "system":
            self.messages[0]["content"] = content
        else:
            self.messages.insert(0, {"role": "system", "content": content})
    
    def add_user_message(self, content: str):
        """Add a user message."""
        self.messages.append({"role": "user", "content": content})
    
    def add_assistant_message(self, content: Optional[str] = None, 
                            tool_calls: Optional[List[ToolCall]] = None):
        """Add an assistant message with optional tool calls."""
        message: Dict[str, Any] = {"role": "assistant"}
        
        if content:
            message["content"] = content
            
        if tool_calls:
            message["tool_calls"] = [
                {
                    "id": tc.id, 
                    "type": "function", 
                    "function": {
                        "name": tc.tool_name, 
                        "arguments": json.dumps(tc.input)
                    }
                } 
                for tc in tool_calls
            ]
        
        if content or tool_calls:
            self.messages.append(message)
    
    def add_tool_results(self, tool_results: List[Dict[str, Any]]):
        """Add tool result messages."""
        self.messages.extend(tool_results)
    
    def update_system_message(self, content: str):
        """Update the system message (assumes it's the first message)."""
        if self.messages and self.messages[0]["role"] == "system":
            self.messages[0]["content"] = content
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all messages."""
        return self.messages
    
    def clear(self):
        """Clear all messages."""
        self.messages = []
    


def get_llm_config_from_env() -> Dict[str, Any]:
    """Get LLM configuration from environment variables.
    
    Returns:
        Dictionary with model, api_key, base_url, and provider
    """
    import os
    
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
            llm_model = "gpt-4-mini"  # Default fallback
    else:
        # For other providers, use similar logic
        base_url = os.getenv("LLM_BASE_URL", "http://localhost:8000")
        api_key = os.getenv("LLM_API_KEY", "dummy-key")
        if not llm_model:
            llm_model = "gpt-4-mini"  # Default fallback
    
    return {
        "model": llm_model,
        "api_key": api_key,
        "base_url": base_url,
        "provider": llm_provider
    }