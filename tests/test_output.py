from repo2txt.core.models import RepoFile, Repository
from repo2txt.core.output import render


def test_markdown_fences_outgrow_backticks_inside_files():
    content = "Example:\n```python\nprint(1)\n```\n"
    repo = Repository("r", None, {"doc.md": RepoFile("doc.md", content, 10)})

    rendered = render(repo, ["doc.md"], "markdown")

    opening = next(
        line for line in rendered.splitlines() if line.startswith("`") and "doc.md" in line
    )
    assert opening.startswith("````")
    assert rendered.rstrip().endswith("````")
