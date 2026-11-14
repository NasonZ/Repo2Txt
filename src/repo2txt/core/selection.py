"""Choosing files under a token budget."""

from .models import Repository


def select_within_budget(repo: Repository, budget: int | None) -> tuple[list[str], list[str]]:
    """Return `(selected, dropped)` paths.

    With no budget, or when everything fits, all files are selected. Otherwise the largest files
    are dropped one at a time until the total fits, so the most files survive.
    """
    selected = sorted(repo.files)
    if budget is None:
        return selected, []
    dropped: list[str] = []
    total = repo.tokens(selected)
    for path in sorted(selected, key=lambda p: (-repo.files[p].tokens, p)):
        if total <= budget:
            break
        dropped.append(path)
        total -= repo.files[path].tokens
    kept = set(selected) - set(dropped)
    return sorted(kept), dropped
