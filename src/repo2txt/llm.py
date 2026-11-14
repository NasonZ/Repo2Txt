"""The one place a model is called.

`Complete` is the seam: anything that takes a conversation and tool schemas and returns a
`Reply` can drive the selector. The default talks to an OpenAI-compatible endpoint.
"""

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .models import Problem


@dataclass
class Call:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Reply:
    text: str
    calls: list[Call] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


Complete = Callable[[list[dict[str, Any]], list[dict[str, Any]]], Reply]


class ConfigError(Problem):
    """The environment does not describe a usable model endpoint."""


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


def openai_complete(settings: LLMSettings) -> Complete:
    """A `Complete` for any endpoint that speaks the OpenAI chat completions API."""
    from openai import OpenAI

    client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)

    def complete(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Reply:
        extra = {"tools": tools, "tool_choice": "auto"} if tools else {}
        response = client.chat.completions.create(model=settings.model, messages=messages, **extra)
        message = response.choices[0].message
        calls = []
        for call in message.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            calls.append(Call(call.id, call.function.name, arguments))
        usage = response.usage
        return Reply(
            message.content or "",
            calls,
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
        )

    return complete
