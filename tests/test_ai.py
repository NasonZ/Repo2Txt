import pytest

from repo2txt.ai.llm import ConfigError, LLMSettings
from repo2txt.ai.state import StateManager

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
