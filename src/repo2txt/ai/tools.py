"""Tool system for AI file selection.

This module provides the tool infrastructure for AI agents to interact
with the file selection system, including tool definitions, execution,
and result handling.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple, Callable

from .state import StateManager


class Tool:
    """Represents a tool that can be called by the AI."""
    
    def __init__(self, name: str, description: str, parameters: Dict[str, Any]):
        self.name = name
        self.description = description
        self.parameters = parameters


class ToolCall:
    """Represents a request to call a tool."""
    
    def __init__(self, id: str, tool_name: str, input: Dict[str, Any]):
        self.id = id
        self.tool_name = tool_name
        self.input = input


class ToolResult:
    """Represents the result of a tool call."""
    
    def __init__(self, tool_call_id: str, output: Any, error: Optional[str] = None, 
                 is_json_output: bool = False):
        self.tool_call_id = tool_call_id
        self.output = output
        self.error = error
        self.is_json_output = is_json_output


class ToolExecutor:
    """Manages tool registration and execution."""
    
    def __init__(self, state_manager: StateManager):
        self._registered_tools: Dict[str, Tuple[Tool, Callable]] = {}
        self.state_manager = state_manager

    def register_tool(self, tool: Tool, func: Callable):
        """Register a tool with its implementation function."""
        self._registered_tools[tool.name] = (tool, func)
        logging.debug(f"Tool registered: {tool.name}")

    async def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call and return the result."""
        logging.debug(f"Executing tool: {tool_call.tool_name} with input: {tool_call.input}")
        
        if tool_call.tool_name not in self._registered_tools:
            logging.error(f"Tool '{tool_call.tool_name}' not found.")
            return ToolResult(
                tool_call_id=tool_call.id, 
                output=None, 
                error=f"Tool '{tool_call.tool_name}' not found."
            )
        
        _tool_obj, func = self._registered_tools[tool_call.tool_name]
        
        try:
            if asyncio.iscoroutinefunction(func):
                output = await func(**tool_call.input)
            else:
                output = func(**tool_call.input)
            logging.debug(f"Tool {tool_call.tool_name} executed successfully.")
            return ToolResult(
                tool_call_id=tool_call.id, 
                output=output, 
                is_json_output=isinstance(output, (dict, list))
            )
        except Exception as e:
            logging.error(f"Error executing tool {tool_call.tool_name}: {e}")
            return ToolResult(tool_call_id=tool_call.id, output=None, error=str(e))

    def get_tools_for_openai(self) -> List[Dict[str, Any]]:
        """Convert registered tools to OpenAI function format."""
        return [
            {
                "type": "function", 
                "function": {
                    "name": tool.name, 
                    "description": tool.description, 
                    "parameters": tool.parameters
                }
            }
            for tool, _ in self._registered_tools.values()
        ]


def create_file_selection_tools() -> List[Tuple[Tool, str]]:
    """Create the standard file selection tools.
    
    Returns:
        List of (Tool, method_name) tuples for registration
    """
    select_files_tool = Tool(
        "select_files", 
        "Selects an initial set of files, replacing any current selection", 
        {
            "type": "object", 
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": (
                        "Detailed rationale for this selection that: "
                        "1) Follows the meta-reasoning framework from the system prompt diligently, "
                        "2) Explains how each file helps answer the user's question, "
                        "3) Verifies each path exists in the repository structure shown in the system prompt. "
                        "Be thorough and explicit about your reasoning process."
                    )
                },
                "paths": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "List of file paths to select"
                }
            }, 
            "required": ["reasoning", "paths"]
        }
    )
    
    adjust_selection_tool = Tool(
        "adjust_selection", 
        "Modifies the current selection by adding or removing files", 
        {
            "type": "object", 
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": (
                        "Detailed rationale for this adjustment that: "
                        "1) Follows the meta-reasoning framework from the system prompt, "
                        "2) Explains why files are being added/removed based on user feedback or new understanding, "
                        "3) Verifies each path against the repository structure, "
                        "4) Considers how changes affect token budget and information completeness, "
                        "5) References previous conversation context. "
                        "Be thorough about why this adjustment improves the selection."
                    )
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
    )
    
    return [
        (select_files_tool, "select_files_impl"),
        (adjust_selection_tool, "adjust_selection_impl")
    ]


def parse_openai_tool_calls(openai_tool_calls_list: Optional[List[Any]]) -> List[ToolCall]:
    """Parse OpenAI tool calls into our ToolCall format.
    
    Args:
        openai_tool_calls_list: List of OpenAI tool call objects
        
    Returns:
        List of ToolCall objects
    """
    if not openai_tool_calls_list:
        return []
    
    parsed = []
    for call in openai_tool_calls_list:
        try:
            args = json.loads(call.function.arguments)
        except json.JSONDecodeError:
            args = {}
            logging.warning(f"Invalid JSON args for {call.function.name}: {call.function.arguments}")
        parsed.append(ToolCall(id=call.id, tool_name=call.function.name, input=args))
    return parsed