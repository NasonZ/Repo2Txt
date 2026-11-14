"""The commands end to end: a directory in, text or JSON out."""

import json
import shutil
import subprocess

import pytest
from conftest import TEXT_FILES

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
    assert "next: repo2txt map src/" in output
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
    assert "next: repo2txt pack src/large.py:1-200" in result.output


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
    data = json.loads(run("select", "function small", "-b", "60", "--json").output)

    assert 0 < data["tokens"] <= 60
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
