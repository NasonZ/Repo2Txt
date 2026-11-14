"""Access to any OpenAI-compatible chat endpoint, and the conversation history."""

import json
import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from ..ui import UI
from .tools import ToolCall


class ConfigError(Exception):
    """The environment does not describe a usable LLM endpoint."""


@dataclass
class LLMSettings:
    model: str
    api_key: str
    base_url: str | None = None

    @classmethod
    def from_env(cls) -> "LLMSettings":
        """Read LLM_MODEL, LLM_BASE_URL and LLM_API_KEY.

        A model is always required. A key is required unless a base URL is set, since local
        servers usually ignore it.
        """
        model = os.getenv("LLM_MODEL")
        base_url = os.getenv("LLM_BASE_URL") or None
        api_key = os.getenv("LLM_API_KEY")
        if not model:
            raise ConfigError("LLM_MODEL is not set. Name the model your endpoint serves.")
        if not api_key:
            if base_url is None:
                raise ConfigError(
                    "LLM_API_KEY is not set. Set it, or set LLM_BASE_URL for a local server."
                )
            api_key = "unused"
        return cls(model, api_key, base_url)


class LLMClient:
    def __init__(self, settings: LLMSettings):
        self.model = settings.model
        self.client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)

    def create_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
    ) -> Any:
        params: dict[str, Any] = {"model": self.model, "messages": messages, "stream": stream}
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"
        return self.client.chat.completions.create(**params)

    def read_stream(self, stream: Any, ui: UI) -> tuple[str, list[ToolCall]]:
        """Print a streamed reply as it arrives and return its text and tool calls."""
        content = ""
        pending: dict[int, dict[str, str]] = {}
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                content += delta.content
                ui.stream(delta.content)
            for part in delta.tool_calls or []:
                call = pending.setdefault(part.index, {"id": "", "name": "", "arguments": ""})
                call["id"] = part.id or call["id"]
                if part.function:
                    call["name"] = part.function.name or call["name"]
                    call["arguments"] += part.function.arguments or ""
        ui.print()
        calls = [
            ToolCall(c["id"], c["name"], _loads(c["arguments"]))
            for _, c in sorted(pending.items())
            if c["id"] and c["name"]
        ]
        return content, calls


def _loads(arguments: str) -> dict[str, Any]:
    try:
        return json.loads(arguments) if arguments else {}
    except json.JSONDecodeError:
        return {}


class MessageManager:
    """Conversation history, with the system prompt always first."""

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    def set_system(self, content: str) -> None:
        if self.messages and self.messages[0]["role"] == "system":
            self.messages[0]["content"] = content
        else:
            self.messages.insert(0, {"role": "system", "content": content})

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str | None = None, tool_calls: list[ToolCall] | None = None):
        message: dict[str, Any] = {"role": "assistant"}
        if content:
            message["content"] = content
        if tool_calls:
            message["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.tool_name, "arguments": json.dumps(tc.input)},
                }
                for tc in tool_calls
            ]
        if content or tool_calls:
            self.messages.append(message)

    def add_tool_results(self, results: list[dict[str, Any]]) -> None:
        self.messages.extend(results)

    def clear(self) -> None:
        self.messages = []
