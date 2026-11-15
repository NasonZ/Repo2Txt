"""The models the benchmark compares, each called through its provider's own API.

    BENCH_MODEL=sonnet BENCH_CLIENT=/path/to/native_client.py python -m bench.run ...

Each entry fixes everything that decides how a model is called, so that runs are comparable
and every row can say what it used: the provider, the model, the effort and the prompt
caching asked for. Effort is set on every model rather than left to each provider's default,
which differs (Haiku 5.5 medium, Sonnet 5.5 high, GPT-6.1 Terra medium, DeepSeek high); each
is set to its provider's suggestion for short agentic tool loops. DeepSeek honours only high
and max.

The client that speaks each provider's native API is supplied locally: BENCH_CLIENT names a
Python file whose `make_complete(model)` returns a `repo2txt.llm.Complete` for a `Model`.
It must pass each turn's reasoning state back (as `Reply.message`) and report cache reads and
writes. `compat` needs no such file: it is the route the earlier results used, Anthropic's
OpenAI-compatible endpoint, which supports no prompt caching and ignores effort.

Prices are dollars per million tokens, from the providers' pricing pages on 15 November 2026,
for computing what a run cost. Haiku 5.5's are for prompts up to 100k tokens, which every
selector request stays under. DeepSeek's are its peak-hour rates.
"""

import functools
import importlib.util
import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Model:
    provider: str
    model: str
    effort: str | None
    price_in: float
    price_out: float
    price_cached: float
    price_cache_write: float = 0.0  # Anthropic charges for writing the cache; others do not
    options: dict = field(default_factory=dict)  # passed to every request
    region: str | None = None


ANTHROPIC_CACHE = {"cache_control": {"type": "ephemeral"}}

MODELS = {
    "haiku": Model(
        "anthropic", "claude-haiku-5-5", "medium", 0.10, 0.50, 0.01, 0.125, ANTHROPIC_CACHE
    ),
    "sonnet": Model(
        "anthropic", "claude-sonnet-5-5", "medium", 2.0, 10.0, 0.10, 2.50, ANTHROPIC_CACHE
    ),
    "terra": Model(
        "bedrock-openai",
        "global.openai.gpt-6.1-terra",
        "medium",
        1.0,
        5.0,
        0.05,
        region="eu-west-2",
    ),
    "deepseek": Model("deepseek", "deepseek-flash", "high", 0.30, 1.20, 0.006),
    "compat": Model("openai-compat", "claude-haiku-5-5", None, 0.10, 0.50, 0.01),
}


def selected() -> tuple[str, Model]:
    name = os.getenv("BENCH_MODEL", "compat")
    if name not in MODELS:
        raise SystemExit(f"BENCH_MODEL={name!r}; choose from {sorted(MODELS)}")
    return name, MODELS[name]


def cost(model: Model, rows: list[dict]) -> float:
    """Dollars spent on these rows: uncached input, cached input, cache writes and output."""
    total = 0.0
    for row in rows:
        cached = row.get("cached_tokens", 0)
        written = row.get("cache_write_tokens", 0)
        fresh = row.get("input_tokens", 0) - cached - written
        total += (
            fresh * model.price_in
            + cached * model.price_cached
            + written * (model.price_cache_write or model.price_in)
            + row.get("output_tokens", 0) * model.price_out
        ) / 1e6
    return total


def complete_for(model: Model):
    """A `Complete` for the model: the compatible endpoint, or the local native client."""
    if model.provider == "openai-compat":
        from repo2txt.llm import LLMSettings, openai_complete

        return openai_complete(LLMSettings.from_env())
    path = os.getenv("BENCH_CLIENT")
    if not path:
        raise SystemExit(f"{model.model} needs a native client: set BENCH_CLIENT to its file")
    return _client_module(path).make_complete(model)


@functools.cache
def _client_module(path: str):
    """The local client, loaded once per process so its connections and loops are reused."""
    spec = importlib.util.spec_from_file_location("bench_native_client", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
