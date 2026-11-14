"""Token counting. The encoding is chosen here and nowhere else."""

from functools import lru_cache

import tiktoken

ENCODING_NAME = "cl100k_base"


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding(ENCODING_NAME)


def count_tokens(text: str) -> int:
    """Count tokens in text. Special-token strings in source files count as ordinary text."""
    return len(_encoding().encode(text, disallowed_special=()))
