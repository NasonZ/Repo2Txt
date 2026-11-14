from bench.metrics import resolve, score


def test_scoring_counts_files_found_and_sets_aside_tests():
    gold = ["src/a.py", "src/b.py", "tests/test_a.py"]

    result = score(["src/a.py", "docs/x.md"], gold)

    assert result["recall"] == 1 / 3
    assert result["precision"] == 1 / 2
    assert result["recall_code"] == 1 / 2
    assert result["all_gold"] == 0.0
    assert score(gold, gold)["f1"] == 1.0
    assert score([], gold)["precision"] == 0.0


def test_gold_is_checked_against_the_commit_before_scoring():
    gold = resolve(
        ["src/a.py", "gone.py", "assets/logo.png"],
        in_archive={"src/a.py", "assets/logo.png"},
        readable={"src/a.py"},
    )

    assert gold.files == ["src/a.py", "assets/logo.png"]
    assert gold.unresolved == ["gone.py"]
    assert gold.filtered == ["assets/logo.png"]
