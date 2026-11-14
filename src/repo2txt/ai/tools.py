"""Tools the model can call to change the selection."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ToolCall:
    id: str
    tool_name: str
    input: dict[str, Any]


@dataclass
class ToolResult:
    tool_call_id: str
    output: Any
    error: str | None = None


class ToolExecutor:
    def __init__(self) -> None:
        self._tools: dict[str, tuple[Tool, Callable[..., Any]]] = {}

    def register(self, tool: Tool, func: Callable[..., Any]) -> None:
        self._tools[tool.name] = (tool, func)

    def execute_tool(self, call: ToolCall) -> ToolResult:
        if call.tool_name not in self._tools:
            return ToolResult(call.id, None, f"Tool '{call.tool_name}' not found.")
        try:
            return ToolResult(call.id, self._tools[call.tool_name][1](**call.input))
        except Exception as exc:
            return ToolResult(call.id, None, str(exc))

    def get_tools_for_openai(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool, _ in self._tools.values()
        ]


_PATHS = {"type": "array", "items": {"type": "string"}}


def create_file_selection_tools() -> tuple[Tool, Tool]:
    """Return the `select_files` and `adjust_selection` tool definitions."""
    select = Tool(
        "select_files",
        "Selects an initial set of files, replacing any current selection",
        {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Why these files answer the user's request.",
                },
                "paths": {**_PATHS, "description": "Repository paths to select; each must exist."},
            },
            "required": ["reasoning", "paths"],
        },
    )
    adjust = Tool(
        "adjust_selection",
        "Modifies the current selection by adding or removing files",
        {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Why files are being added or removed, given the token budget.",
                },
                "add_files": {**_PATHS, "description": "Files to add to the selection"},
                "remove_files": {**_PATHS, "description": "Files to remove from the selection"},
            },
            "required": ["reasoning"],
        },
    )
    return select, adjust


def parse_openai_tool_calls(calls: list[Any] | None) -> list[ToolCall]:
    """Convert tool calls from a non-streamed completion; bad JSON arguments become empty."""
    parsed = []
    for call in calls or []:
        try:
            arguments = json.loads(call.function.arguments)
        except json.JSONDecodeError:
            arguments = {}
        parsed.append(ToolCall(call.id, call.function.name, arguments))
    return parsed
