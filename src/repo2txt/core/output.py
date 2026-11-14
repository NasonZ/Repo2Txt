"""Rendering a selection as Markdown or XML, plus the token report and JSON export."""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Literal
from xml.sax.saxutils import escape, quoteattr

from .models import Repository
from .tree import build_tree, render_tree

Format = Literal["markdown", "xml"]


def _fence(content: str) -> str:
    longest = max((len(run) for run in re.findall(r"`+", content)), default=0)
    return "`" * max(3, longest + 1)


def render(repo: Repository, paths: list[str], fmt: Format) -> str:
    structure = render_tree(build_tree(repo.name, {p: repo.files[p].tokens for p in paths}))
    total = repo.tokens(paths)
    if fmt == "xml":
        files = "".join(
            f"<file path={quoteattr(p)}>\n{repo.files[p].content}\n</file>\n" for p in paths
        )
        return (
            f"<repository name={quoteattr(repo.name)} branch={quoteattr(repo.branch or '')} "
            f'files="{len(paths)}" tokens="{total}">\n'
            f"<structure>\n{escape(structure)}\n</structure>\n<files>\n{files}</files>\n"
            "</repository>\n"
        )
    branch = f" (branch: {repo.branch})" if repo.branch else ""
    blocks = []
    for p in paths:
        content = repo.files[p].content
        fence = _fence(content)
        blocks.append(f"{fence}{p}\n{content}\n{fence}\n")
    return (
        f"# {repo.name}{branch}\n\n- Files: {len(paths)}\n- Tokens: {total:,}\n\n"
        f"## Structure\n\n```text\n{structure}\n```\n\n## Files\n\n" + "\n".join(blocks)
    )


def token_report(repo: Repository, paths: list[str]) -> str:
    total = repo.tokens(paths)
    rows = sorted(paths, key=lambda p: -repo.files[p].tokens)
    lines = [f"{repo.name}: {len(paths)} files, {total:,} tokens", ""]
    for p in rows:
        tokens = repo.files[p].tokens
        share = 100 * tokens / total if total else 0
        lines.append(f"{tokens:>10,}  {share:5.1f}%  {p}")
    return "\n".join(lines) + "\n"


def write_outputs(
    repo: Repository,
    paths: list[str],
    output_dir: str | Path,
    fmt: Format = "markdown",
    export_json: bool = False,
) -> dict[str, Path]:
    """Write the context file and token report (and optionally JSON) into a new directory."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = re.sub(r"[^\w.-]", "_", repo.name)
    directory = Path(output_dir) / f"{safe}_{stamp}"
    directory.mkdir(parents=True, exist_ok=True)

    written = {
        "main": directory / f"{safe}_analysis.{'md' if fmt == 'markdown' else 'xml'}",
        "tokens": directory / f"{safe}_token_report.txt",
    }
    written["main"].write_text(render(repo, paths, fmt), encoding="utf-8")
    written["tokens"].write_text(token_report(repo, paths), encoding="utf-8")
    if export_json:
        written["json"] = directory / f"{safe}_tokens.json"
        data = {
            "repo_name": repo.name,
            "branch": repo.branch,
            "total_files": len(paths),
            "total_tokens": repo.tokens(paths),
            "files": {p: repo.files[p].tokens for p in paths},
        }
        written["json"].write_text(json.dumps(data, indent=2), encoding="utf-8")
    return written
