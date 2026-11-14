"""Slash commands available during the chat."""

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from ..core.output import write_outputs
from .prompts import PROMPT_STYLES

if TYPE_CHECKING:
    from .agent import FileSelectorAgent

HELP = """\
/generate [markdown|xml] [json]  write the selected files to the output directory
/undo                           restore the state before the last message
/redo                           regenerate the last reply
/clear                          reset the conversation and the selection
/save [name]                    save the conversation as JSON
/toggle streaming               switch streamed replies on or off
/toggle prompt                  cycle the prompt style
/toggle budget <n>              set the token budget
/debug                          show or hide prompts and tool calls
/quit, /exit                    leave the chat"""


def handle_command(agent: "FileSelectorAgent", text: str) -> bool:
    """Run one slash command. Returns False when the chat should end."""
    command, *args = text.split()
    command = command.lower()
    ui = agent.ui

    if command in ("/quit", "/exit"):
        return False
    if command == "/help":
        ui.section("COMMANDS", HELP)
    elif command == "/clear":
        agent.messages.clear()
        agent.state.clear_selection()
        agent.refresh_system_prompt()
        ui.success("Conversation and selection cleared.")
    elif command == "/undo":
        if agent.restore_snapshot():
            agent.refresh_system_prompt()
            ui.success("Restored the state before the last message.")
        else:
            ui.warning("Nothing to undo.")
    elif command == "/redo":
        _redo(agent)
    elif command == "/generate":
        _generate(agent, [a.lower() for a in args])
    elif command == "/save":
        _save(agent, args[0] if args else None)
    elif command == "/toggle":
        _toggle(agent, args)
    elif command == "/debug":
        ui.debug = not ui.debug
        ui.success(f"Debug {'on' if ui.debug else 'off'}.")
    else:
        ui.warning(f"Unknown command {command}. Try /help.")
    return True


def _redo(agent: "FileSelectorAgent") -> None:
    messages = agent.messages.messages
    last_user = next(
        (i for i in range(len(messages) - 1, 0, -1) if messages[i]["role"] == "user"), None
    )
    if last_user is None:
        agent.ui.warning("There is no message to regenerate a reply for.")
        return
    del messages[last_user + 1 :]
    agent.respond()


def _generate(agent: "FileSelectorAgent", args: list[str]) -> None:
    paths = agent.selected_files
    if not paths:
        agent.ui.warning("No files selected yet.")
        return
    fmt = "xml" if "xml" in args else "markdown"
    written = write_outputs(agent.repo, paths, agent.output_dir, fmt, export_json="json" in args)
    agent.ui.success(f"Wrote {len(paths)} files ({agent.repo.tokens(paths):,} tokens):")
    for path in written.values():
        agent.ui.plain(f"  {path}")


def _save(agent: "FileSelectorAgent", name: str | None) -> None:
    name = name or f"chat_{datetime.now():%Y%m%d_%H%M%S}"
    path = Path(agent.output_dir) / "chat_history" / (name.removesuffix(".json") + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    state = agent.state.state
    data = {
        "model": agent.model,
        "token_budget": state.token_budget,
        "selected_files": state.selected_files,
        "messages": agent.messages.messages[1:],
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    agent.ui.success(f"Saved to {path}")


def _toggle(agent: "FileSelectorAgent", args: list[str]) -> None:
    ui = agent.ui
    what = args[0].lower() if args else ""
    if what == "streaming":
        agent.use_streaming = not agent.use_streaming
        ui.success(f"Streaming {'on' if agent.use_streaming else 'off'}.")
    elif what == "prompt":
        styles = PROMPT_STYLES
        agent.prompt_style = styles[(styles.index(agent.prompt_style) + 1) % len(styles)]
        agent.refresh_system_prompt()
        ui.success(f"Prompt style: {agent.prompt_style}")
    elif what == "budget" and len(args) == 2 and args[1].isdigit() and int(args[1]) > 0:
        agent.state.state.token_budget = int(args[1])
        agent.refresh_system_prompt()
        ui.success(f"Token budget: {int(args[1]):,}")
    else:
        ui.warning("Usage: /toggle streaming | prompt | budget <n>")
