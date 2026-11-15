from bench.metrics import resolve, score


def test_scoring_counts_files_found_and_sets_aside_tests():
    gold = ["src/a.py", "src/b.py", "tests/test_a.py"]

    spans = [["src/a.py", 1, 10], ["src/b.py", 1, 80], ["tests/test_a.py", 5, 14]]
    result = score(["src/a.py", "docs/x.md"], gold, spans)

    assert result["recall"] == 1 / 3
    assert result["precision"] == 1 / 2
    assert result["recall_code"] == 1 / 2
    assert result["all_gold"] == 0.0
    assert result["line_recall"] == 10 / 100
    assert score(gold, gold, spans)["f1"] == 1.0
    assert score([], gold, spans)["precision"] == 0.0


def test_gold_is_checked_against_the_commit_before_scoring():
    gold = resolve(
        ["src/a.py", "gone.py", "assets/logo.png"],
        in_archive={"src/a.py", "assets/logo.png"},
        readable={"src/a.py"},
    )

    assert gold.files == ["src/a.py", "assets/logo.png"]
    assert gold.unresolved == ["gone.py"]
    assert gold.filtered == ["assets/logo.png"]


def test_line_recall_counts_only_the_needed_lines_that_were_shown():
    from bench.packing import score_lines

    spans = [["a.py", 10, 19], ["b.py", 1, 10]]
    gold = ["a.py", "b.py"]

    whole_a = score_lines({"a.py": None}, spans, gold)
    part_b = score_lines({"b.py": set(range(1, 6)), "c.py": None}, spans, gold)

    assert whole_a == {"line_recall": 0.5, "file_recall": 0.5}
    assert part_b == {"line_recall": 0.25, "file_recall": 0.5}
