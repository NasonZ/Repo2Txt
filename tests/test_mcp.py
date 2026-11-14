import asyncio

import pytest

pytest.importorskip("mcp")

from mcp.client import Client  # noqa: E402

from repo2txt.mcp_server import server  # noqa: E402
from repo2txt.tokens import count_tokens  # noqa: E402


def call(name: str, **arguments):
    async def run():
        async with Client(server) as client:
            return await client.call_tool(name, arguments)

    return asyncio.run(run())


def test_the_three_tools_are_read_only_and_answer_within_their_cap(root, monkeypatch):
    monkeypatch.chdir(root.parent)

    async def tools():
        async with Client(server) as client:
            return (await client.list_tools()).tools

    listed = {tool.name: tool for tool in asyncio.run(tools())}
    assert set(listed) == {"repo_map", "repo_select", "repo_pack"}
    assert all(tool.annotations.read_only_hint for tool in listed.values())

    tree = call("repo_map", repo=str(root), max_tokens=60)
    assert not tree.is_error and count_tokens(tree.content[0].text) <= 60

    selected = call("repo_select", repo=str(root), query="session store")
    assert "src/auth/session.py" in selected.content[0].text

    packed = call("repo_pack", repo=str(root), items=["src/large.py", "src/small.py"], budget=200)
    text = packed.content[0].text
    assert "def small():" in text and "omitted: src/large.py" in text


def test_a_failure_reaches_the_client_as_an_error_with_a_next_step(root, monkeypatch):
    monkeypatch.chdir(root.parent)
    result = call("repo_pack", repo=str(root), items=["nope.py"])

    assert result.is_error
    assert "nope.py not found" in result.content[0].text and "next: " in result.content[0].text


def test_a_directory_outside_where_the_server_started_is_refused(root, tmp_path, monkeypatch):
    monkeypatch.chdir(root)

    result = call("repo_map", repo=str(tmp_path))

    assert result.is_error and "outside the directory" in result.content[0].text
