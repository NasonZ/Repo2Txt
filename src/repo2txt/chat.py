"""A terminal conversation over the selector."""

from pathlib import Path

from rich.console import Console
from rich.text import Text

from .api import repo_pack
from .render import pack_text
from .selector import Selector
from .tokens import parse_budget

HELP = """\
/generate [FILE]   write the selected files to FILE (default <name>-context.md)
/undo              go back to before your last message
/clear             start again
/budget N          change the token budget, e.g. /budget 50k
/quit              leave"""


def run_chat(selector: Selector) -> None:
    console = Console(highlight=False)
    say = lambda text, style="": console.print(Text(text, style=style))  # noqa: E731
    selection, repo = selector.selection, selector.repo
    history: list[tuple[int, list[str]]] = []  # (messages kept, paths) before each request

    say(f"{repo.name}: {len(repo.files)} files, {repo.tokens():,} tokens.")
    say(f"Budget {selection.budget:,}. Say what you need the files for. /help lists commands.")
    while True:
        try:
            text = console.input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        command, _, argument = text.partition(" ")
        if not text:
            continue
        if command in ("/quit", "/exit"):
            return
        if command == "/help":
            say(HELP)
        elif command == "/clear":
            selector.messages.clear()
            selection.paths = []
            history.clear()
            say("Cleared.", "dim")
        elif command == "/undo":
            if not history:
                say("Nothing to undo.", "dim")
                continue
            kept, selection.paths = history.pop()
            del selector.messages[kept:]
            say(f"Back to {len(selection.paths)} files.", "dim")
        elif command == "/budget":
            try:
                selection.budget = parse_budget(argument)
                say(f"Budget {selection.budget:,}.", "dim")
            except ValueError as exc:
                say(str(exc), "red")
        elif command == "/generate":
            if not selection.paths:
                say("Nothing is selected yet.", "dim")
                continue
            target = Path(argument.strip() or f"{repo.name}-context.md")
            result = repo_pack(repo, selection.paths)
            target.write_text(pack_text(result), encoding="utf-8")
            say(f"Wrote {len(result.files)} files, {result.tokens:,} tokens, to {target}.", "green")
        elif command.startswith("/"):
            say(f"Unknown command {command}. /help lists commands.", "red")
        else:
            history.append((len(selector.messages), list(selection.paths)))
            before = set(selection.paths)
            try:
                reply = selector.ask(text)
            except Exception as exc:  # A failed request should not end the conversation.
                say(f"The model could not be reached: {exc}", "red")
                continue
            for path in sorted(before - set(selection.paths)):
                say(f"  - {path}", "red")
            for path in [p for p in selection.paths if p not in before]:
                say(f"  + {path}", "green")
            say(reply)
            used = f"{selection.total:,} of {selection.budget:,} tokens"
            say(f"{len(selection.paths)} files, {used}.", "dim")
