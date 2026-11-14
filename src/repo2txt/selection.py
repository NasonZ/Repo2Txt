"""A set of files held within a token budget."""

from collections.abc import Iterable
from dataclasses import dataclass, field

from .models import Repository


@dataclass
class Change:
    """What an attempted change to a selection did.

    When `applied` is false the selection is as it was, and `over_by` says how far over the
    budget the proposed selection would have been.
    """

    applied: bool
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    over_by: int = 0


class Selection:
    """Paths chosen from a repository. A change that would exceed the budget is refused."""

    def __init__(self, repo: Repository, budget: int):
        self.repo = repo
        self.budget = budget
        self.paths: list[str] = []

    @property
    def total(self) -> int:
        """What the selection takes when packed, which is what the budget limits."""
        return self.repo.cost(self.paths)

    def replace(self, paths: Iterable[str]) -> Change:
        """Make `paths` the whole selection, in the order given."""
        return self._apply(paths, [], start=[])

    def adjust(self, add: Iterable[str] = (), remove: Iterable[str] = ()) -> Change:
        """Add and remove paths, keeping the rest in order."""
        return self._apply(add, remove, start=self.paths)

    def _apply(self, add: Iterable[str], remove: Iterable[str], start: list[str]) -> Change:
        remove = {p.strip() for p in remove}
        add = list(dict.fromkeys(p.strip() for p in add))
        unknown = [p for p in add if p not in self.repo.files]
        unknown += sorted(remove - set(start))
        kept = [p for p in start if p not in remove]
        proposed = kept + [p for p in add if p in self.repo.files and p not in kept]

        total = self.repo.cost(proposed)
        # Shrinking a selection that is already over budget (the budget was lowered) is allowed.
        if total > self.budget and total >= self.total:
            return Change(False, unknown=unknown, over_by=total - self.budget)
        before = set(self.paths)
        self.paths = proposed
        return Change(
            True,
            added=[p for p in proposed if p not in before],
            removed=sorted(before - set(proposed)),
            unknown=unknown,
        )


def fit(ranked: Iterable[str], repo: Repository, budget: int | None) -> tuple[list[str], list[str]]:
    """Take paths in rank order while their packed cost fits the budget.

    Returns `(kept, skipped)`.

    A file that does not fit is skipped and later, smaller ones are still tried.
    """
    ranked = list(ranked)
    if budget is None:
        return ranked, []
    repo.count(ranked)
    kept, skipped, total = [], [], 0
    for path in ranked:
        cost = repo.files[path].cost
        if total + cost <= budget:
            kept.append(path)
            total += cost
        else:
            skipped.append(path)
    return kept, skipped
