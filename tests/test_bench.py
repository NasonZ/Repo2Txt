import random

import pytest

from bench import run as bench_run
from bench.check import check_rows
from bench.metrics import needed_lines, resolve, score
from bench.stats import bootstrap, estimate, leave_one_repo_out, paired
from bench.tasks import CONTAINER, Task, edited_spans
from repo2txt.api import HEAD_TOKENS, delivery_cost
from repo2txt.models import RepoFile, Repository


def test_scoring_counts_files_and_merged_lines():
    gold = ["src/a.py", "src/b.py", "tests/test_a.py"]
    # The two spans in a.py overlap on lines 5-10: those lines count once.
    spans = [
        ["src/a.py", 1, 10],
        ["src/a.py", 5, 14],
        ["src/b.py", 1, 80],
        ["tests/test_a.py", 5, 9],
    ]
    sizes = {"src/a.py": 20, "src/b.py": 100, "tests/test_a.py": 30, "docs/x.md": 10}

    result = score({"src/a.py": None, "docs/x.md": None}, gold, spans, sizes)

    assert result["recall"] == 1 / 3
    assert result["precision"] == 1 / 2
    assert result["recall_code"] == 1 / 2
    assert result["all_gold"] == 0.0
    assert result["line_recall"] == 14 / (14 + 80 + 5)
    assert result["line_precision"] == 14 / (20 + 10)


def test_a_line_range_counts_only_the_needed_lines_it_shows():
    spans = [["a.py", 10, 19], ["b.py", 1, 10]]

    result = score({"b.py": set(range(1, 6))}, ["a.py", "b.py"], spans, {"b.py": 50})

    assert result["line_recall"] == 5 / 20
    assert result["line_precision"] == 1.0
    assert result["recall"] == 0.5


def test_an_empty_delivery_scores_zero_and_has_no_precision():
    result = score({}, ["a.py"], [["a.py", 1, 3]], {"a.py": 10})

    assert result["recall"] == 0 and result["line_recall"] == 0
    assert result["precision"] is None and result["line_precision"] is None


def test_needed_lines_merges_overlapping_spans():
    assert needed_lines([["a", 1, 3], ["a", 2, 5], ["b", 1, 1]], ["a"]) == {"a": {1, 2, 3, 4, 5}}


def test_gold_is_checked_against_the_commit_before_scoring():
    gold = resolve(
        ["src/a.py", "gone.py", "assets/logo.png"],
        in_archive={"src/a.py", "assets/logo.png"},
        readable={"src/a.py"},
    )

    assert gold.files == ["src/a.py", "assets/logo.png"]
    assert gold.unresolved == ["gone.py"]
    assert gold.filtered == ["assets/logo.png"]


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/workspace/facebook__zstd__0.1/lib/zstd.h", "lib/zstd.h"),
        ("/testbed/astropy/io/fits.py", "astropy/io/fits.py"),
        ("/app/src/index.ts", "src/index.ts"),
        ("src/plain.py", "src/plain.py"),
    ],
)
def test_container_prefixes_are_stripped_from_gold_paths(path, expected):
    assert CONTAINER.sub("", path) == expected


PATCH = """\
diff --git a/pkg/mod.py b/pkg/mod.py
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -10,4 +10,4 @@ def f():
     a = 1
-    b = 2
+    b = 3
     c = 4
     d = 5
@@ -40,3 +40,4 @@ def g():
     x = 1
     y = 2
+    z = 3
     w = 4
diff --git a/pkg/new.py b/pkg/new.py
new file mode 100644
--- /dev/null
+++ b/pkg/new.py
@@ -0,0 +1,2 @@
+print(1)
+print(2)
diff --git a/pkg/old.py b/pkg/old.py
deleted file mode 100644
--- a/pkg/old.py
+++ /dev/null
@@ -1 +0,0 @@
-gone = True
"""


def test_edited_spans_mark_changed_lines_and_insertion_points_in_existing_files():
    assert edited_spans(PATCH) == [["pkg/mod.py", 11, 11], ["pkg/mod.py", 41, 41]]


def _repo() -> Repository:
    files = {
        "a.py": RepoFile("a.py", "\n".join(f"line_a_{n} = {n}" for n in range(1, 201))),
        "b.py": RepoFile("b.py", "\n".join(f"value_b_{n} = {n}" for n in range(1, 41))),
        "c.py": RepoFile("c.py", "\n".join(f"other_c_{n} = {n}" for n in range(1, 31))),
    }
    return Repository("demo", ["-r", "demo"], "abc", files)


def _task() -> Task:
    return Task(
        "t1",
        "o/demo",
        "abc",
        "python",
        "test",
        "value_b_3 is wrong",
        [["a.py", 50, 60], ["b.py", 1, 5]],
        [["b.py", 3, 3]],
    )


def test_every_reference_method_stays_within_the_budget(monkeypatch):
    repo = _repo()
    monkeypatch.setattr(bench_run, "repository", lambda task: (repo, set(repo.files)))
    methods = [m for m in bench_run.RUNNERS if m not in bench_run.MODEL_METHODS]

    rows = bench_run.run_task(_task(), [(m, b) for m in methods for b in (200, 2000)], {})

    assert all(row["status"] == "ok" for row in rows)
    for row in rows:
        assert row["tokens"] <= row["budget"] - HEAD_TOKENS
    assert check_rows(rows) == []


def test_line_oracle_shows_only_needed_lines_and_reaches_its_ceiling(monkeypatch):
    repo = _repo()
    monkeypatch.setattr(bench_run, "repository", lambda task: (repo, set(repo.files)))

    rows = bench_run.run_task(_task(), [("oracle-lines:annotated", 4000)], {})

    scores = rows[0]["scores"]["annotated"]
    assert scores["line_recall"] == 1.0 and scores["line_precision"] == 1.0
    assert rows[0]["delivered"] == {"a.py": [[50, 60]], "b.py": [[1, 5]]}
    expected = delivery_cost(repo, "a.py", set(range(50, 61)))
    expected += delivery_cost(repo, "b.py", set(range(1, 6)))
    assert rows[0]["tokens"] == expected


def test_parsing_select_items_keeps_paths_with_colons():
    delivered = bench_run._parse(["src/a.py:3-5", "src/a.py:9-9", "docs/c:d.md", "x.py"])

    assert delivered == {"src/a.py": {3, 4, 5, 9}, "docs/c:d.md": None, "x.py": None}


def test_results_check_finds_overspend_and_unequal_coverage():
    def row(task, method, tokens):
        return {
            "task": task,
            "method": method,
            "budget": 1000,
            "room": 1000 - HEAD_TOKENS,
            "status": "ok",
            "tokens": tokens,
            "selected": 0,
            "delivered": {},
            "run": {},
            "scores": {"annotated": {"recall": 0.5}},
        }

    problems = check_rows([row("t1", "a", 990), row("t1", "b", 10), row("t2", "b", 10)])

    assert any("exceed" in p for p in problems)
    assert any("same tasks" in p for p in problems)


def _rows(values: dict[str, list[float]], offset: float = 0.0) -> list[dict]:
    rows, n = [], 0
    for repo, group in values.items():
        for v in group:
            rows.append({"task": f"t{n}", "repo": repo, "v": v + offset})
            n += 1
    return rows


def test_bootstrap_matches_scipy_when_every_task_is_its_own_cluster():
    stats = pytest.importorskip("scipy.stats")
    rng = random.Random(1)
    values = [rng.random() for _ in range(200)]

    low, high = bootstrap([[v] for v in values])
    reference = stats.bootstrap(
        (values,), lambda x: sum(x) / len(x), method="percentile", random_state=1, n_resamples=4000
    ).confidence_interval

    assert low == pytest.approx(reference.low, abs=0.01)
    assert high == pytest.approx(reference.high, abs=0.01)


def test_clustered_interval_is_wider_than_the_naive_one_when_repositories_differ():
    values = {"r1": [0.9] * 20, "r2": [0.1] * 20, "r3": [0.5] * 20}
    clustered = estimate(_rows(values), lambda r: r["v"])
    naive = bootstrap([[v] for group in values.values() for v in group])

    assert clustered.high - clustered.low > naive[1] - naive[0]
    assert clustered.repos == 3 and clustered.tasks == 60


def test_paired_difference_of_a_constant_shift_is_exact():
    values = {"r1": [0.2, 0.4], "r2": [0.6], "r3": [0.1, 0.9, 0.5]}
    a, b = _rows(values, 0.1), _rows(values)

    result = paired(a, b, lambda r: r["v"])

    assert result.mean == pytest.approx(0.1)
    assert result.low == pytest.approx(0.1) and result.high == pytest.approx(0.1)
    assert result.better == 6 and result.worse == 0
    assert leave_one_repo_out(a, b, lambda r: r["v"]) == pytest.approx((0.1, 0.1))
