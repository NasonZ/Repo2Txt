"""Terminal output."""

import json
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text


class UI:
    """Thin wrapper over a Rich console with the few message styles repo2txt uses."""

    def __init__(self, debug: bool = False):
        self.console = Console(highlight=False, soft_wrap=True)
        self.debug = debug

    def print(self, *args: Any, **kwargs: Any) -> None:
        """Print with Rich markup. Escape any path or model text interpolated into it."""
        self.console.print(*args, **kwargs)

    def plain(self, text: str) -> None:
        """Print text exactly as given, with no markup."""
        self.console.print(Text(text))

    def info(self, message: str) -> None:
        self.console.print(Text(message, style="cyan"))

    def success(self, message: str) -> None:
        self.console.print(Text(message, style="green"))

    def warning(self, message: str) -> None:
        self.console.print(Text(message, style="yellow"))

    def error(self, message: str) -> None:
        self.console.print(Text(message, style="red"))

    def spinner(self, text: str):
        return self.console.status(text)

    def section(self, title: str, body: str) -> None:
        self.console.print(Panel(Text(body), title=title, border_style="blue", padding=(1, 2)))

    def stream(self, text: str, tool_call: bool = False) -> None:
        self.console.print(Text(text, style="yellow" if tool_call else ""), end="")

    def debug_panel(self, title: str, body: str, style: str = "magenta", limit: int = 2500) -> None:
        if not self.debug:
            return
        if len(body) > limit:
            body = body[:limit] + "\n... [truncated]"
        self.console.print(Panel(Text(body), title=title, border_style=style, padding=(0, 1)))

    def tool_call(self, name: str, arguments: dict[str, Any]) -> None:
        self.debug_panel(f"TOOL CALL {name}", json.dumps(arguments, indent=2), style="yellow")

    def tool_result(self, output: Any, error: str | None) -> None:
        if error:
            self.debug_panel("TOOL ERROR", error, style="red")
        else:
            body = json.dumps(output, indent=2) if isinstance(output, (dict, list)) else str(output)
            self.debug_panel("TOOL RESULT", body, style="green")

    def selection_diff(self, state: Any) -> None:
        """Show files added or removed by the last action and the resulting budget use."""
        before, after = set(state.previous_files), set(state.selected_files)
        if before == after:
            return
        self.print("\nSelection changes:")
        for path in sorted(before - after):
            self.print(f"  [red]-[/red] {escape(path)}")
        for path in sorted(after - before):
            self.print(f"  [green]+[/green] {escape(path)}")
        delta = state.total_tokens_selected - state.previous_tokens
        self.print(
            f"Total {state.total_tokens_selected:,} tokens ({delta:+,}), "
            f"{state.get_budget_usage_percent():.1f}% of {state.token_budget:,}"
        )
