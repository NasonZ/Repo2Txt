"""The selector's loop, driven by scripted model replies."""

from repo2txt.llm import Call, Reply
from repo2txt.models import RepoFile, Repository
from repo2txt.selection import Selection
from repo2txt.selector import Selector


def repository(sizes: dict[str, int]) -> Repository:
    files = {}
    for path, tokens in sizes.items():
        files[path] = RepoFile(path, "")
        files[path].__dict__["tokens"] = tokens
    return Repository("demo", [], None, files)


class Script:
    """Stands in for the model: replies with the next scripted message."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.seen = []

    def __call__(self, messages, tools):
        self.seen.append(list(messages))
        reply = self.replies.pop(0)
        if isinstance(reply, str):
            return Reply(reply)
        name, arguments = reply
        return Reply("", [Call(f"call_{len(self.seen)}", name, arguments)])


def results(script: Script) -> list[str]:
    return [m["content"] for m in script.seen[-1] if m["role"] == "tool"]


def test_the_model_can_correct_a_selection_that_was_refused():
    repo = repository({"a.py": 60, "b.py": 60})
    script = Script(
        ("select_files", {"paths": ["a.py", "b.py"]}),  # 120 tokens: over the budget of 100
        ("select_files", {"paths": ["src/a.py"]}),  # does not exist
        ("select_files", {"paths": ["a.py"]}),
        "Selected a.py.",
    )
    selector = Selector(repo, 100, script)

    assert selector.ask("where is a?") == "Selected a.py."
    assert selector.selection.paths == ["a.py"]
    refused, unknown, done = results(script)
    over = repo.cost(["a.py", "b.py"]) - 100
    assert f"{over} tokens over the budget" in refused and "unchanged" in refused
    assert "src/a.py" in unknown
    assert f"1 files, {repo.cost(['a.py'])} of 100 tokens" in done


def test_a_selection_never_exceeds_the_budget_even_if_the_model_insists():
    repo = repository({"a.py": 60, "b.py": 60})
    script = Script(*[("select_files", {"paths": ["a.py", "b.py"]})] * 5, "I give up.")

    selector = Selector(repo, 100, script)
    selector.ask("everything")

    assert selector.selection.paths == []
    assert script.replies == []  # the sixth request offered no tools, so the model had to answer


def test_a_collapsed_directory_can_be_listed_and_selected_from():
    sizes = {f"pkg/mod_{i:04d}/core/impl.py": 10 for i in range(1500)}
    repo = repository(sizes)
    target = "pkg/mod_0042/core/impl.py"
    script = Script(
        ("list_dir", {"path": "pkg/mod_0042/core"}),
        ("select_files", {"paths": [target]}),
        "Done.",
    )
    selector = Selector(repo, 100, script)

    selector.ask("mod 42")

    system = script.seen[0][0]["content"]
    assert target not in system and "pkg/" in system
    assert target in results(script)[0]
    assert selector.selection.paths == [target]


def test_adjusting_keeps_order_and_reports_unknown_paths():
    selection = Selection(repository({"a.py": 10, "b.py": 10, "c.py": 10}), 100)
    selection.replace(["c.py", "a.py"])

    change = selection.adjust(add=["b.py", "nope.py"], remove=["c.py", "z.py"])

    assert selection.paths == ["a.py", "b.py"]
    assert (change.added, change.removed) == (["b.py"], ["c.py"])
    assert change.unknown == ["nope.py", "z.py"]


def test_a_selection_over_a_lowered_budget_may_shrink_but_not_grow():
    selection = Selection(repository({"a.py": 60, "b.py": 60, "c.py": 5}), 200)
    selection.replace(["a.py", "b.py"])
    selection.budget = 100

    assert not selection.adjust(add=["c.py"]).applied
    assert selection.adjust(remove=["b.py"]).applied
    assert selection.paths == ["a.py"]


def test_bad_tool_arguments_change_nothing_and_always_get_a_reply():
    repo = repository({"a.py": 10})
    script = Script(
        ("select_files", {"paths": ["a.py"]}),
        ("select_files", None),  # not JSON
        ("select_files", {"paths": [1]}),
        ("select_files", {"paths": "a.py"}),
        ("nonsense", {}),
        "Done.",
    )
    selector = Selector(repo, 100, script)

    selector.ask("a")

    assert selector.selection.paths == ["a.py"]
    calls = [m for m in selector.messages if m["role"] == "assistant" and m.get("tool_calls")]
    assert len(results(script)) == len(calls) == 5
    assert all("Nothing was changed" in text or "no tool" in text for text in results(script)[1:])


def test_line_ranges_come_first_and_the_rest_of_the_budget_takes_whole_files(repo):
    from repo2txt.api import repo_pack, select_lines

    def complete(messages, tools):
        names = [tool["function"]["name"] for tool in tools]
        if "read_ranges" in names:
            ranges = ["src/large.py:3-6", "src/large.py:5-8", "nope.py:1-2", "src/small.py:9-1"]
            return Reply("", [Call("r", "read_ranges", {"ranges": ranges})])
        if names and not any(m["role"] == "tool" for m in messages):
            paths = ["src/large.py", "src/small.py"]
            return Reply("", [Call("s", "select_files", {"paths": paths})])
        return Reply("Done.")

    items = select_lines(repo, "function 2", 1500, complete)

    assert items[0][0] == "src/large.py:3-8"  # overlapping ranges merged, bad ones ignored
    assert ("src/small.py", repo.files["src/small.py"].cost) in items  # filled whole
    assert sum(cost for _, cost in items) <= 1500
    packed = repo_pack(repo, [item for item, _ in items], 1500 + 60)
    assert not packed.omitted
