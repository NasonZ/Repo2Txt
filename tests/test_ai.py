import json
from types import SimpleNamespace

import pytest

from repo2txt.ai import FileSelectorAgent
from repo2txt.ai.llm import ConfigError, LLMSettings
from repo2txt.ai.state import StateManager
from repo2txt.core.models import RepoFile, Repository
from repo2txt.ui import UI

TOKENS = {"a.py": 100, "b.py": 250, "docs/c.md": 40}


def test_replacing_a_selection_rejects_unknown_paths_and_counts_the_rest():
    state = StateManager(TOKENS, token_budget=500)

    result = state.replace_selection(["a.py", "nope.py", " b.py "], "reasoning")

    assert result["valid_paths"] == ["a.py", "b.py"]
    assert result["invalid_paths"] == ["nope.py"]
    assert result["total_tokens_selected"] == 350
    assert result["budget_remaining"] == 150


def test_adjusting_a_selection_adds_removes_and_reports_problems():
    state = StateManager(TOKENS, token_budget=500)
    state.replace_selection(["a.py", "b.py"], "")

    result = state.modify_selection(add_files=["docs/c.md", "x.py"], remove_files=["a.py", "c.py"])

    assert result["selected_paths"] == ["b.py", "docs/c.md"]
    assert result["total_tokens_selected"] == 290
    assert len(result["not_found"]) == 2
    assert state.state.previous_tokens == 350


def test_llm_settings_require_a_model(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    with pytest.raises(ConfigError, match="LLM_MODEL"):
        LLMSettings.from_env()


def test_llm_settings_need_a_key_only_for_the_default_endpoint(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "some-model")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    with pytest.raises(ConfigError, match="LLM_API_KEY"):
        LLMSettings.from_env()

    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:8080/v1")
    settings = LLMSettings.from_env()
    assert (settings.model, settings.base_url) == ("some-model", "http://localhost:8080/v1")


class ScriptedLLM:
    """Stands in for the model endpoint: replies with the next scripted message."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.seen = []

    def create_completion(self, messages, tools=None, stream=False):
        self.seen.append([dict(m) for m in messages])
        content, paths = self.replies.pop(0)
        calls = None
        if paths is not None:
            arguments = json.dumps({"reasoning": "r", "paths": paths})
            function = SimpleNamespace(name="select_files", arguments=arguments)
            calls = [SimpleNamespace(id=f"call_{len(self.seen)}", function=function)]
        message = SimpleNamespace(content=content, tool_calls=calls)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def agent_for(llm, budget):
    files = {"a.py": RepoFile("a.py", "a", 60), "b.py": RepoFile("b.py", "b", 60)}
    agent = FileSelectorAgent(
        Repository("demo", None, files), LLMSettings("model", "key"), budget, UI()
    )
    agent.llm, agent.use_streaming = llm, False
    return agent


def test_the_model_can_correct_a_selection_that_was_refused():
    llm = ScriptedLLM(
        ("", ["a.py", "b.py"]),  # 120 tokens: over the budget of 100
        ("", ["src/a.py"]),  # does not exist
        ("", ["a.py"]),
        ("Selected a.py.", None),
    )

    selected = agent_for(llm, budget=100).run_once("where is a?")

    assert selected == ["a.py"]
    refusals = [m["content"] for m in llm.seen[-1] if m["role"] == "tool"]
    assert "over the budget" in refusals[0] and "do not exist" not in refusals[0]
    assert "src/a.py" in refusals[1]


def test_a_selection_never_exceeds_the_budget_even_if_the_model_insists():
    llm = ScriptedLLM(*[("", ["a.py", "b.py"])] * 5, ("I give up.", None))

    assert agent_for(llm, budget=100).run_once("everything") == []
