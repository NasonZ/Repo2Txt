"""Command-line interface."""

import sys

import click
from dotenv import load_dotenv

from .adapters import SourceError, open_source
from .ai import PROMPT_STYLES, ConfigError, FileSelectorAgent, LLMSettings
from .core.models import Config
from .core.output import write_outputs
from .core.selection import select_within_budget
from .ui import UI

DEFAULT_AI_BUDGET = 100_000


@click.command()
@click.argument("repo")
@click.option(
    "--output-dir", "-o", default="output", show_default=True, help="Where to write results."
)
@click.option(
    "--format",
    "-f",
    "fmt",
    type=click.Choice(["markdown", "xml"]),
    default="markdown",
    show_default=True,
    help="Format of the context file.",
)
@click.option(
    "--token-budget",
    type=click.IntRange(min=1),
    help="Token budget. Without --ai-select, the largest files are dropped to fit. "
    f"With it, defaults to {DEFAULT_AI_BUDGET:,}.",
)
@click.option(
    "--max-file-size",
    "-m",
    type=int,
    default=1024 * 1024,
    show_default=True,
    help="Skip files larger than this many bytes.",
)
@click.option("--exclude-dirs", help="Comma-separated extra directory names to skip.")
@click.option(
    "--branch", help="GitHub branch to read. Defaults to the repository's default branch."
)
@click.option("--json", "export_json", is_flag=True, help="Also write token counts as JSON.")
@click.option(
    "--ai-select",
    is_flag=True,
    help="Have an LLM select the files. Needs LLM_MODEL; see the README.",
)
@click.option(
    "--ai-query", help="With --ai-select, select for this request and exit, without chat."
)
@click.option(
    "--prompt-style",
    type=click.Choice(PROMPT_STYLES),
    default="standard",
    show_default=True,
    help="System prompt used by --ai-select.",
)
@click.option("--debug", is_flag=True, help="Show prompts and tool calls during --ai-select.")
@click.version_option()
def main(
    repo,
    output_dir,
    fmt,
    token_budget,
    max_file_size,
    exclude_dirs,
    branch,
    export_json,
    ai_select,
    ai_query,
    prompt_style,
    debug,
):
    """Write the files of a repository as one context file for an LLM.

    REPO is a local directory, a GitHub URL, or owner/repo. Every text file is included unless
    --token-budget is exceeded, or --ai-select lets an LLM choose.

    \b
    repo2txt .
    repo2txt owner/repo --token-budget 50000
    repo2txt . --ai-select --ai-query "how does authentication work"
    """
    load_dotenv()
    ui = UI(debug=debug)
    if ai_query and not ai_select:
        raise click.UsageError("--ai-query needs --ai-select.")

    config = Config(max_file_size=max_file_size)
    if exclude_dirs:
        config.excluded_dirs.update(d.strip() for d in exclude_dirs.split(",") if d.strip())

    try:
        settings = LLMSettings.from_env() if ai_select else None
        with ui.spinner("Reading repository..."):
            repository = open_source(repo, config, branch).load()
    except (ConfigError, SourceError) as exc:
        ui.error(str(exc))
        sys.exit(1)

    ui.info(f"{repository.name}: {len(repository.files)} files, {repository.tokens():,} tokens")
    for error in repository.errors:
        ui.warning(f"Skipped {error}")

    if ai_select:
        budget = token_budget or DEFAULT_AI_BUDGET
        agent = FileSelectorAgent(repository, settings, budget, ui, prompt_style, output_dir)
        try:
            selected = agent.run_once(ai_query) if ai_query else agent.run()
        except KeyboardInterrupt:
            sys.exit(130)
        except Exception as exc:
            ui.error(f"AI selection failed: {exc}")
            sys.exit(1)
        if not ai_query:
            return  # Output was written from the chat with /generate.
        if not selected:
            ui.error("The model did not select any files.")
            sys.exit(1)
    else:
        selected, dropped = select_within_budget(repository, token_budget)
        if not selected:
            ui.error(
                "No files to write."
                if token_budget is None
                else f"No file fits within {token_budget:,} tokens."
            )
            sys.exit(1)
        if dropped:
            ui.warning(f"Dropped {len(dropped)} largest files to fit the budget:")
            for path in dropped:
                ui.plain(f"  {path} ({repository.files[path].tokens:,} tokens)")

    written = write_outputs(repository, selected, output_dir, fmt, export_json)
    ui.success(f"Selected {len(selected)} files, {repository.tokens(selected):,} tokens.")
    for path in written.values():
        ui.plain(f"  {path}")
