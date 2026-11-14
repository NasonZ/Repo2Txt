"""Token counting and budgets. The encoding is chosen here and nowhere else."""

import re
from functools import lru_cache

import tiktoken

ENCODING = "o200k_base"

_BUDGET = re.compile(r"^(\d+(?:\.\d+)?)([km]?)$")
_SCALE = {"": 1, "k": 1_000, "m": 1_000_000}


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding(ENCODING)


def count_tokens(text: str) -> int:
    """Count tokens. Special-token strings in source files count as ordinary text."""
    return len(_encoding().encode_ordinary(text))


def count_many(texts: list[str]) -> list[int]:
    """Count tokens for many texts at once, in parallel."""
    return [len(tokens) for tokens in _encoding().encode_ordinary_batch(texts)]


def parse_budget(value: str | int) -> int:
    """Read a token budget such as `8000`, `8k` or `1.5m`."""
    match = _BUDGET.match(str(value).strip().lower())
    if not match or (budget := int(float(match[1]) * _SCALE[match[2]])) < 1:
        raise ValueError(f"{value!r} is not a token budget. Use a number such as 8000, 8k or 1.5m.")
    return budget
