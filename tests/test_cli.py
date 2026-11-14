"""The commands end to end: a directory in, text or JSON out."""

import json
import shutil
import subprocess

import pytest
from click.testing import CliRunner
from conftest import TEXT_FILES

from repo2txt.cli import cli
from repo2txt.tokens import count_tokens


def test_map_lists_every_text_file_with_totals_that_add_up(run):
    data = json.loads(run("map", "--json").output)

    files = {e["path"]: e["tokens"] for e in data["entries"] if "files" not in e}
    assert set(files) == TEXT_FILES
    assert data["files"] == len(TEXT_FILES)
    assert data["tokens"] == sum(files.values())
    assert data["encoding"] == "o200k_base"


def test_map_collapses_below_the_depth_shown_and_says_how_to_see_more(run):
    output = run("map", "--depth", "1").output

    assert "src/ " in output and "(4 files)" in output
    assert "login.py" not in output
    assert any(
        line.startswith("# next: repo2txt map ") and line.endswith(" src/")
        for line in output.splitlines()
    )
    assert "login.py" in run("map", "src/auth").output


def test_map_output_never_exceeds_its_cap(run):
    for cap in (40, 80, 400):
        result = run("map", "--max-tokens", str(cap))

        assert result.exit_code == 0
        assert count_tokens(result.output) <= cap
    assert "showing" in run("map", "--max-tokens", "40").output
    assert "login.py" in run("map", "--max-tokens", "400").output


def test_pack_keeps_files_whole_and_lists_what_did_not_fit(run):
    result = run("pack", "src/large.py", "src/small.py", "-b", "200", "--json")

    data = json.loads(result.output)
    assert [f["path"] for f in data["files"]] == ["src/small.py"]
    assert data["files"][0]["content"] == "def small():\n    return 1\n"
    assert data["tokens"] <= 200
    (omitted,) = data["omitted"]
    assert omitted["path"] == "src/large.py" and omitted["reason"] == "over_budget"
    assert omitted["tokens"] > 200


def test_pack_takes_directories_globs_and_line_ranges(run):
    output = run("pack", "src/auth/", "*.md", "src/large.py:3-4").output

    assert "### src/auth/login.py" in output and "### src/auth/session.py" in output
    assert "### README.md" in output
    assert "### src/large.py (lines 3-4)\n\n```\ndef function_1():\n    return 1\n```" in output
    assert "function_2" not in output


def test_pack_with_nothing_to_show_fails_and_suggests_a_line_range(run):
    result = run("pack", "src/large.py", "-b", "50")

    assert result.exit_code == 1
    assert "src/large.py needs" in result.output
    assert "src/large.py:1-200\n" in result.output and "next: repo2txt pack " in result.output


def test_select_ranks_the_file_that_defines_the_named_identifier_first(run):
    result = run("select", "where is verifyHash used to check a password")

    first = result.output.splitlines()[1]
    assert first.split("\t")[0] == "src/auth/login.py"


def test_a_selection_can_be_piped_into_pack(run):
    selected = run("select", "session store", "-b", "200").output

    packed = run("pack", "--stdin", input=selected).output

    assert "### src/auth/session.py" in packed
    assert "class SessionStore" in packed


def test_select_stays_within_its_budget(run):
    data = json.loads(run("select", "function small", "-b", "150", "--json").output)

    assert 0 < data["tokens"] <= 150
    assert "src/large.py" not in [f["path"] for f in data["files"]]
    assert "src/large.py" in [f["path"] for f in data["skipped"]]


def test_misuse_exits_2_and_failure_exits_1_each_with_a_next_step(run, tmp_path):
    misuse = run("map", "--bogus")
    assert misuse.exit_code == 2
    assert "error: No such option" in misuse.output and "next: " in misuse.output

    nothing = run("select", "zzzqqq", "--json")
    assert nothing.exit_code == 1
    assert json.loads(nothing.output)["error"]["next"]


def test_xml_keeps_content_verbatim(run):
    output = run("pack", "src/small.py", "-f", "xml").output

    assert '<file path="src/small.py">\ndef small():\n    return 1\n\n</file>' in output


def test_markdown_fences_outgrow_backticks_inside_files(root, run):
    (root / "notes.md").write_text("a ```` b\n")

    assert "`````\na ```` b\n\n`````" in run("pack", "notes.md").output


@pytest.mark.skipif(not shutil.which("git"), reason="needs git")
def test_files_ignored_by_git_are_left_out(root, run):
    (root / ".gitignore").write_text("*.log\nscratch/\n")
    (root / "debug.log").write_text("noise\n")
    (root / "scratch").mkdir()
    (root / "scratch" / "notes.md").write_text("draft\n")
    (root / "src" / "untracked.py").write_text("NEW = 1\n")
    shutil.rmtree(root / ".git")
    subprocess.run(["git", "init", "-q", str(root)], check=True)

    data = json.loads(run("map", "--json").output)

    files = {e["path"] for e in data["entries"] if "files" not in e}
    assert files == TEXT_FILES | {"src/untracked.py"}


def test_include_and_exclude_filter_by_glob(run):
    data = json.loads(run("map", "-i", "src/*", "-e", "*large*", "--json").output)

    files = {e["path"] for e in data["entries"] if "files" not in e}
    assert files == {"src/small.py", "src/auth/login.py", "src/auth/session.py"}


def test_what_select_chooses_packs_within_the_same_budget(run):
    for budget in ("150", "400", "3k"):
        selected = run("select", "function small password session", "-b", budget)
        lines = selected.output.splitlines()
        chosen = [line.split("\t")[0] for line in lines if not line.startswith("#")]

        packed = run("pack", "--stdin", "-b", budget, input=selected.output)

        assert chosen and "omitted" not in packed.output
        assert all(f"### {path}" in packed.output for path in chosen)
        assert count_tokens(packed.output) <= int(budget.replace("k", "000"))


def test_pack_output_never_exceeds_its_budget(run):
    for budget in (120, 300, 1000, 2500):
        result = run("pack", "-b", str(budget))

        assert result.exit_code == 0 and count_tokens(result.output) <= budget


def test_empty_standard_input_is_an_error_not_the_whole_repository(run):
    result = run("pack", "--stdin", input="# nothing matched\n")

    assert result.exit_code == 1
    assert "def small" not in result.output


def test_line_ranges_count_lines_as_an_editor_does(root, run):
    (root / "page.py").write_text("a = 1\n\x0c\nb = 2\r\nc = 3\n")

    # The test runner normalises line endings in captured output.
    assert "```\nb = 2\nc = 3\n```" in run("pack", "page.py:3-4").output.replace("\r", "")

    past = run("pack", "page.py:9-12")
    assert (
        past.exit_code == 1 and "not a line range within page.py, which has 5 lines" in past.output
    )
    assert run("pack", "page.py:3-2").exit_code == 1


def test_hints_name_the_same_files_and_survive_awkward_names(root, run):
    (root / "-dash.py").write_text("x = 1\n" * 400)

    args = ["pack", "-r", str(root), "-i", "*.py", "-b", "100", "--", "-dash.py"]
    hint = CliRunner().invoke(cli, args).output.splitlines()[1]

    assert hint.startswith("next: repo2txt pack -r ") and " -i '*.py' " in hint
    assert hint.endswith(" -- -dash.py:1-200")


def test_settings_are_read_from_a_dotenv_file_in_the_working_directory(run, tmp_path, monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("LLM_MODEL=from-dotenv\nLLM_BASE_URL=http://127.0.0.1:9/v1\n")

    result = run("select", "anything", "--method", "llm")

    assert result.exit_code == 1
    assert "LLM_MODEL is not set" not in result.output
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
