"""File selection state, with validation of every path the model proposes."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FileSelectionState:
    token_budget: int
    selected_files: list[str] = field(default_factory=list)
    total_tokens_selected: int = 0
    # The selection before the last change, used to show what changed.
    previous_files: list[str] = field(default_factory=list)
    previous_tokens: int = 0

    def get_budget_usage_percent(self) -> float:
        if self.token_budget <= 0:
            return 0
        return round(self.total_tokens_selected / self.token_budget * 100, 3)

    def get_budget_remaining(self) -> int:
        return self.token_budget - self.total_tokens_selected


class StateManager:
    """Applies the model's selection changes.

    Paths that do not exist are rejected, and so is any change that would leave the selection
    over the token budget, so a selection made here always fits.
    """

    def __init__(self, file_tokens: dict[str, int], token_budget: int):
        self.state = FileSelectionState(token_budget=token_budget)
        self.file_tokens = file_tokens

    def _set_selection(self, paths: set[str] | list[str]) -> None:
        state = self.state
        state.previous_files = list(state.selected_files)
        state.previous_tokens = state.total_tokens_selected
        state.selected_files = sorted(paths)
        state.total_tokens_selected = sum(self.file_tokens[p] for p in state.selected_files)

    def _over_budget(self, paths: set[str]) -> str | None:
        """Explain why `paths` cannot become the selection, or return None if it fits.

        A change that shrinks a selection already over budget (the budget was lowered) is allowed.
        """
        state = self.state
        total = self._tokens(paths)
        if total <= state.token_budget or total < state.total_tokens_selected:
            return None
        largest = sorted(paths, key=lambda p: -self.file_tokens[p])[:5]
        listed = ", ".join(f"{p} ({self.file_tokens[p]:,})" for p in largest)
        return (
            f"ERROR: selection not changed. {len(paths)} files total {total:,} tokens, "
            f"{total - state.token_budget:,} over the budget of {state.token_budget:,}. "
            f"Largest: {listed}. Choose fewer or smaller files and try again."
        )

    def _result(self, feedback: str, **extra: Any) -> dict[str, Any]:
        state = self.state
        return {
            "selected_paths": state.selected_files,
            "total_files_selected": len(state.selected_files),
            "total_tokens_selected": state.total_tokens_selected,
            "budget_remaining": state.get_budget_remaining(),
            "budget_usage_percent": state.get_budget_usage_percent(),
            "feedback": feedback,
            **extra,
        }

    def _budget_line(self) -> str:
        state = self.state
        return (
            f"{state.total_tokens_selected:,}/{state.token_budget:,} tokens "
            f"({state.get_budget_usage_percent():.1f}%)"
        )

    def replace_selection(self, paths: list[str], reasoning: str = "") -> dict[str, Any]:
        """Replace the whole selection."""
        valid = {p.strip() for p in paths if p.strip() in self.file_tokens}
        invalid = [p for p in paths if p.strip() not in self.file_tokens]
        if refusal := self._over_budget(valid):
            return self._result(refusal, invalid_paths=invalid, valid_paths=[])
        self._set_selection(valid)
        if invalid and not valid:
            feedback = (
                f"ERROR: none of the {len(invalid)} paths exist in the repository: "
                f"{', '.join(invalid)}. Check them against the structure in the system prompt "
                "(missing 'src/' prefix, wrong nesting, typos) and try again."
            )
        elif invalid:
            feedback = (
                f"Selected {len(valid)} files, {self._budget_line()}. WARNING: these paths do not "
                f"exist: {', '.join(invalid)}. Correct them and add them with adjust_selection."
            )
        else:
            feedback = f"Selected {len(valid)} files, {self._budget_line()}."
        return self._result(feedback, invalid_paths=invalid, valid_paths=sorted(valid))

    def modify_selection(
        self,
        add_files: list[str] | None = None,
        remove_files: list[str] | None = None,
        reasoning: str = "",
    ) -> dict[str, Any]:
        """Add files to, and remove files from, the current selection."""
        current = set(self.state.selected_files)
        added, removed, problems = [], [], []
        for path in (p.strip() for p in remove_files or []):
            if path in current:
                current.remove(path)
                removed.append(path)
            else:
                problems.append(f"{path} (not currently selected)")
        for path in (p.strip() for p in add_files or []):
            if path not in self.file_tokens:
                problems.append(f"{path} (not found in repository)")
            elif path not in current:
                current.add(path)
                added.append(path)
        if refusal := self._over_budget(current):
            return self._result(refusal, added_files=[], removed_files=[], not_found=problems)
        self._set_selection(current)

        parts = []
        if added:
            parts.append(f"Added {len(added)} files (+{self._tokens(added):,} tokens)")
        if removed:
            parts.append(f"Removed {len(removed)} files (-{self._tokens(removed):,} tokens)")
        if not (added or removed):
            parts.append("ERROR: no valid changes made")
        if problems:
            parts.append(f"Problems: {', '.join(problems)}. Check the paths and retry")
        parts.append(f"Now {self._budget_line()}")
        return self._result(
            ". ".join(parts), added_files=added, removed_files=removed, not_found=problems
        )

    def _tokens(self, paths: list[str] | set[str]) -> int:
        return sum(self.file_tokens[p] for p in paths)

    def get_current_selection_summary(self) -> str:
        """Describe the current selection for the system prompt."""
        state = self.state
        if not state.selected_files:
            return "No files currently selected."
        lines = [
            f"**Currently selected files ({len(state.selected_files)} files, "
            f"~{state.total_tokens_selected:,} tokens):**"
        ]
        lines += [f"  - {p} (~{self.file_tokens[p]:,} tokens)" for p in state.selected_files]
        lines.append(f"**Budget Usage:** ~{self._budget_line()}")
        return "\n".join(lines)

    def clear_selection(self) -> None:
        self._set_selection(set())
