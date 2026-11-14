"""The non-AI path end to end: a directory in, a context file out."""

import json

from click.testing import CliRunner

from repo2txt.cli import main

TEXT_FILES = {
    "README.md",
    "src/small.py",
    "src/large.py",
    "tests/test_small.py",
    ".github/workflows/ci.yml",
}


def run(repo, out, *args):
    result = CliRunner().invoke(main, [str(repo), "-o", str(out), "--json", *args])
    assert result.exit_code == 0, result.output
    (directory,) = out.iterdir()
    return directory, json.loads(next(directory.glob("*_tokens.json")).read_text())


def test_selects_only_text_files_worth_reading(fixture_repo, tmp_path):
    directory, data = run(fixture_repo, tmp_path / "out")

    assert set(data["files"]) == TEXT_FILES
    assert data["total_tokens"] == sum(data["files"].values()) > 0
    context = next(directory.glob("*_analysis.md")).read_text()
    assert "def function_199():" in context
    assert "module.exports" not in context


def test_token_total_respects_the_budget(fixture_repo, tmp_path):
    _, everything = run(fixture_repo, tmp_path / "full")
    budget = everything["total_tokens"] - 1

    directory, data = run(fixture_repo, tmp_path / "limited", "--token-budget", str(budget))

    assert data["total_tokens"] <= budget
    assert "src/large.py" not in data["files"]
    assert set(data["files"]) == TEXT_FILES - {"src/large.py"}
    context = next(directory.glob("*_analysis.md")).read_text()
    assert "def function_199():" not in context


def test_xml_output_wraps_each_file(fixture_repo, tmp_path):
    directory, _ = run(fixture_repo, tmp_path / "out", "--format", "xml")

    context = next(directory.glob("*_analysis.xml")).read_text()
    assert '<file path="src/small.py">' in context
    assert "def small():" in context


def test_missing_model_stops_ai_mode_before_reading_anything(fixture_repo, tmp_path, monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)

    result = CliRunner().invoke(main, [str(fixture_repo), "-o", str(tmp_path), "--ai-select"])

    assert result.exit_code == 1
    assert "LLM_MODEL" in result.output
    assert not list(tmp_path.glob("*_*"))
