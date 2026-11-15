"""The command line: map, select, pack, chat and mcp."""

import functools
import json
import sys
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv

from . import render
from .api import METHODS, repo_map, repo_pack, repo_select
from .models import Problem
from .sources import load
from .tokens import parse_budget


class Budget(click.ParamType):
    """A token count such as 8000, 8k or 1.5m."""

    name = "tokens"

    def convert(self, value, param, ctx):
        try:
            return parse_budget(value)
        except ValueError as exc:
            self.fail(str(exc), param, ctx)


class Group(click.Group):
    """Reports every failure the same way: what went wrong, then a command to try.

    Exit status is 0 on success, 1 when the work failed and 2 when the command was misused.
    """

    def main(self, args=None, **extra):
        args = sys.argv[1:] if args is None else list(args)
        try:
            super().main(args=args, standalone_mode=False, **extra)
        except click.UsageError as exc:
            command = exc.ctx.command_path if exc.ctx else "repo2txt"
            _fail(exc.format_message(), [f"{command} --help"], "--json" in args, status=2)
        except click.Abort:
            sys.exit(130)
        except Problem as exc:
            _fail(str(exc), exc.next, "--json" in args, status=1)
        except Exception as exc:  # Anything else is still reported, not dumped as a traceback.
            _fail(f"{type(exc).__name__}: {exc}", [], "--json" in args, status=1)


def _fail(message: str, hints: list[str], as_json: bool, status: int) -> None:
    if as_json:
        click.echo(json.dumps({"error": {"message": message, "next": hints}}))
    else:
        click.echo("\n".join([f"error: {message}", *(f"next: {hint}" for hint in hints)]))
    sys.exit(status)


def repository(command):
    """Options every command shares for naming and filtering the repository."""

    @click.option(
        "-r", "--repo", default=".", help="Directory, GitHub URL or owner/repo.  [default: .]"
    )
    @click.option("--ref", help="GitHub branch, tag or commit. Defaults to the default branch.")
    @click.option("-i", "--include", multiple=True, help="Only files matching this glob.")
    @click.option("-e", "--exclude", multiple=True, help="Leave out files matching this glob.")
    @functools.wraps(command)
    def wrapper(repo, ref, include, exclude, **kwargs):
        return command(load(repo, ref=ref, include=include, exclude=exclude), **kwargs)

    return wrapper


@click.group(cls=Group)
@click.version_option(package_name="repo2txt")
def cli():
    """Choose what an LLM or a coding agent reads from a repository, within a token budget.

    \b
    repo2txt map                        the tree, with token counts
    repo2txt select "rate limiting"     files ranked for a question, fitted to a budget
    repo2txt pack src/api.py docs/      the contents of named files, fitted to a budget
    repo2txt select "auth" | repo2txt pack --stdin -b 16k
    """
    load_dotenv(find_dotenv(usecwd=True))


@cli.command()
@click.argument("directory", default="")
@repository
@click.option("--max-tokens", type=Budget(), default="4k", show_default=True, help="Output cap.")
@click.option("--depth", type=click.IntRange(min=1), help="Levels to show. Default: what fits.")
@click.option("--json", "as_json", is_flag=True, help="Print JSON.")
def map(repo, directory, max_tokens, depth, as_json):
    """Show the tree under DIRECTORY with token counts.

    Directories deeper than fits in --max-tokens are collapsed to one line with their totals.
    """
    result = repo_map(repo, directory, max_tokens, depth)
    click.echo(
        render.as_json(result, max_tokens) if as_json else render.map_text(result, max_tokens)
    )


@cli.command()
@click.argument("query")
@repository
@click.option("-b", "--budget", type=Budget(), default="32k", show_default=True)
@click.option("--method", type=click.Choice(list(METHODS)), default="bm25", show_default=True)
@click.option(
    "--lines",
    is_flag=True,
    help="With --method llm, choose line ranges, not whole files. Helps most at small budgets.",
)
@click.option("--max-tokens", type=Budget(), default="4k", show_default=True, help="Output cap.")
@click.option("--json", "as_json", is_flag=True, help="Print JSON.")
def select(repo, query, budget, method, lines, max_tokens, as_json):
    """Rank files for QUERY and list those that fit the budget, best first.

    Prints paths and token counts, not contents: pipe the result to `pack --stdin`.
    bm25 needs no model. llm asks the model named by LLM_MODEL to choose from the tree.
    """
    result = repo_select(repo, query, budget, method, lines=lines)
    click.echo(
        render.as_json(result, max_tokens) if as_json else render.select_text(result, max_tokens)
    )


@cli.command()
@click.argument("items", nargs=-1)
@repository
@click.option("--stdin", "from_stdin", is_flag=True, help="Also read items from standard input.")
@click.option("-b", "--budget", type=Budget(), help="Token budget. Default: no limit.")
@click.option("-f", "--format", "fmt", type=click.Choice(["markdown", "xml"]), default="markdown")
@click.option("-o", "--output", type=click.Path(dir_okay=False), help="Write the document here.")
@click.option("--json", "as_json", is_flag=True, help="Print JSON.")
def pack(repo, items, from_stdin, budget, fmt, output, as_json):
    """Print the contents of ITEMS, fitted whole to the budget in the order given.

    An item is a path, a directory, a glob, or path:10-80 for a line range. With no items the
    whole repository is packed, smallest files first. Files that do not fit are listed, not cut.
    """
    if from_stdin:
        lines = (line.split("\t")[0].strip() for line in sys.stdin.read().splitlines())
        items += tuple(line for line in lines if line and not line.startswith("#"))
        if not items:
            raise Problem("Nothing to pack: standard input named no files.")

    def document(result):
        return render.pack_json(result) if as_json else render.pack_text(result, fmt)

    result = repo_pack(repo, items, budget, document)
    if output:
        Path(output).write_text(document(result), encoding="utf-8")
        click.echo(render.pack_summary(result))
    else:
        click.echo(document(result), nl=False)


@cli.command()
@repository
@click.option("-b", "--budget", type=Budget(), default="100k", show_default=True)
def chat(repo, budget):
    """Choose files by talking to the model named by LLM_MODEL."""
    try:
        from .chat import run_chat
    except ImportError as exc:
        raise Problem("The chat needs the rich package.", ["pip install 'repo2txt[llm]'"]) from exc
    from .llm import LLMSettings, openai_complete
    from .selector import Selector

    run_chat(Selector(repo, budget, openai_complete(LLMSettings.from_env())))


@cli.command()
def mcp():
    """Serve map, select and pack to an MCP client over standard input and output."""
    try:
        from .mcp_server import server
    except ImportError as exc:
        raise Problem(
            "The MCP server needs the mcp package.", ["pip install 'repo2txt[mcp]'"]
        ) from exc
    server.run()
