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
    arguments: dict[str, Any] | None  # None when the model sent something that is not JSON


@dataclass
class Reply:
    """A model's turn.

    `input_tokens` counts all input, cached or not; `cached_tokens` and `cache_write_tokens`
    are the parts of it read from and written to a prompt cache, where the endpoint says.
    `message`, when set, is the turn as the endpoint wants it back in the history, carrying
    whatever reasoning state it must see again; otherwise the history is rebuilt from `text`
    and `calls`.
    """

    text: str
    calls: list[Call] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    message: Any = None


Complete = Callable[[list[dict[str, Any]], list[dict[str, Any]]], Reply]


class ConfigError(Problem):
    """The environment does not describe a usable model endpoint."""


@dataclass
class LLMSettings:
    model: str
    api_key: str
    base_url: str | None = None
    temperature: float | None = None  # None leaves the endpoint's default

    @classmethod
    def from_env(cls) -> "LLMSettings":
        """Read LLM_MODEL, LLM_BASE_URL, LLM_API_KEY and, optionally, LLM_TEMPERATURE.

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
        temperature = os.getenv("LLM_TEMPERATURE")
        try:
            value = float(temperature) if temperature else None
        except ValueError as exc:
            raise ConfigError(f"LLM_TEMPERATURE must be a number, not {temperature!r}.") from exc
        return cls(model, api_key, base_url, value)


def openai_complete(settings: LLMSettings) -> Complete:
    """A `Complete` for any endpoint that speaks the OpenAI chat completions API."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ConfigError(
            "Choosing with a model needs the openai package.", ["pip install 'repo2txt[llm]'"]
        ) from exc

    client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)

    def complete(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Reply:
        extra = {"tools": tools, "tool_choice": "auto"} if tools else {}
        if settings.temperature is not None:
            extra["temperature"] = settings.temperature
        response = client.chat.completions.create(model=settings.model, messages=messages, **extra)
        message = response.choices[0].message
        calls = []
        for call in message.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = None
            if not isinstance(arguments, dict):
                arguments = None
            calls.append(Call(call.id, call.function.name, arguments))
        usage = response.usage
        return Reply(
            message.content or "",
            calls,
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
        )

    return complete
