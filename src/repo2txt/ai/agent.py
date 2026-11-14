"""The chat session in which an LLM selects files for the user."""

import json
from typing import Any

from openai import APIError
from rich.prompt import Prompt

from ..core.models import Repository
from ..ui import UI
from .commands import handle_command
from .llm import LLMClient, LLMSettings, MessageManager
from .prompts import PromptGenerator
from .state import StateManager
from .tools import ToolCall, ToolExecutor, create_file_selection_tools, parse_openai_tool_calls

# Rounds of tool calls the model gets per message, so it can correct a rejected selection.
MAX_TOOL_ROUNDS = 5


class FileSelectorAgent:
    def __init__(
        self,
        repo: Repository,
        settings: LLMSettings,
        token_budget: int,
        ui: UI,
        prompt_style: str = "standard",
        output_dir: str = "output",
    ):
        self.repo = repo
        self.ui = ui
        self.output_dir = output_dir
        self.prompt_style = prompt_style
        self.use_streaming = True
        self.model = settings.model

        self.llm = LLMClient(settings)
        self.messages = MessageManager()
        self.state = StateManager(repo.token_map(), token_budget)
        self.prompts = PromptGenerator(self.state, repo.name)
        self.tools = ToolExecutor()
        select_tool, adjust_tool = create_file_selection_tools()
        self.tools.register(select_tool, self.state.replace_selection)
        self.tools.register(adjust_tool, self.state.modify_selection)
        # Undo history: (messages, selected files, token total) before each user turn.
        self.snapshots: list[tuple[list[dict[str, Any]], list[str], int]] = []

    @property
    def selected_files(self) -> list[str]:
        return list(self.state.state.selected_files)

    def refresh_system_prompt(self) -> None:
        self.messages.set_system(self.prompts.generate(self.prompt_style))

    def save_snapshot(self) -> None:
        state = self.state.state
        self.snapshots.append(
            (list(self.messages.messages), list(state.selected_files), state.total_tokens_selected)
        )

    def restore_snapshot(self) -> bool:
        if not self.snapshots:
            return False
        self.messages.messages, files, tokens = self.snapshots.pop()
        self.state.state.selected_files = files
        self.state.state.total_tokens_selected = tokens
        return True

    # Conversation

    def run(self) -> list[str]:
        """Chat with the user until they quit, and return the selected files."""
        state = self.state.state
        self.ui.section(
            "FILE SELECTOR",
            f"Model: {self.model}\nRepository: {self.repo.name}\n"
            f"Files: {len(self.repo.files)} ({self.repo.tokens():,} tokens)\n"
            f"Token budget: {state.token_budget:,}\n\n"
            "Describe what you need. /help lists commands, /generate writes the output, "
            "/quit exits.",
        )
        self.refresh_system_prompt()
        if self.ui.debug:
            self.ui.debug_panel("SYSTEM PROMPT", self.messages.messages[0]["content"], limit=15000)
        while True:
            try:
                text = Prompt.ask("\n[bold][>][/bold]").strip()
                if text.lower() in ("quit", "exit"):
                    break
                if not text:
                    continue
                if text.startswith("/"):
                    if handle_command(self, text) is False:
                        break
                    continue
                self.save_snapshot()
                self.messages.add_user(text)
                self.respond()
            except APIError as exc:
                self.ui.error(f"API error: {exc.message}")
            except KeyboardInterrupt:
                self.ui.warning("\nInterrupted.")
                break
        return self.selected_files

    def run_once(self, query: str) -> list[str]:
        """Answer a single query without a chat session and return the selected files."""
        self.refresh_system_prompt()
        self.messages.add_user(query)
        self.respond()
        return self.selected_files

    def respond(self) -> None:
        """Get the model's reply, running its tool calls until it answers without one."""
        for _ in range(MAX_TOOL_ROUNDS):
            self.refresh_system_prompt()
            content, calls = self._complete(with_tools=True)
            self.messages.add_assistant(content, calls)
            if not calls:
                return
            self.messages.add_tool_results([self._run_tool(call) for call in calls])
        self.refresh_system_prompt()
        content, _ = self._complete(with_tools=False)
        self.messages.add_assistant(content)

    def _complete(self, with_tools: bool) -> tuple[str, list[ToolCall]]:
        tools = self.tools.get_tools_for_openai() if with_tools else None
        history = self.messages.messages
        if self.use_streaming:
            self.ui.print("\n[cyan][<][/cyan] ", end="")
            stream = self.llm.create_completion(history, tools, stream=True)
            content, calls = self.llm.read_stream(stream, self.ui)
        else:
            message = self.llm.create_completion(history, tools).choices[0].message
            content = message.content or ""
            calls = parse_openai_tool_calls(message.tool_calls)
            self.ui.print("\n[cyan][<][/cyan] ", end="")
            self.ui.plain(content)
        for call in calls:
            self.ui.warning(f"Calling tool: {call.tool_name}")
        return content, calls

    def _run_tool(self, call: ToolCall) -> dict[str, Any]:
        self.ui.tool_call(call.tool_name, call.input)
        result = self.tools.execute_tool(call)
        self.ui.tool_result(result.output, result.error)
        self.ui.selection_diff(self.state.state)
        if result.error:
            self.ui.error(f"Tool {call.tool_name} failed: {result.error}")
            content = json.dumps({"error": result.error})
        else:
            content = json.dumps(result.output)
        return {
            "role": "tool",
            "tool_call_id": result.tool_call_id,
            "name": call.tool_name,
            "content": content,
        }
