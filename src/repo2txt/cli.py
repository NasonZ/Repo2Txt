"""Command-line interface for repo2txt."""
import os
import sys
import logging
from pathlib import Path

import click
from .core.models import Config
from .core.analyzer import RepositoryAnalyzer
from .utils.console import ConsoleManager, StatusType, get_console
from .utils.console_base import get_alternating_theme


def setup_logging(debug: bool) -> None:
    """Configure logging based on debug flag."""
    # Always use standard minimal logging
    # The debug flag is for UI debug mode, not logging debug
    level = logging.WARNING
    format_string = '[%(levelname)s] %(message)s'
    
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=[logging.StreamHandler(sys.stderr)]
    )
    
    # Reduce noise from external libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def print_banner(console: ConsoleManager) -> None:
    """Print retro terminal banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║  ██████╗ ███████╗██████╗  ██████╗ ██████╗ ████████╗██╗  ██╗████████╗ ║
║  ██╔══██╗██╔════╝██╔══██╗██╔═══██╗╚════██╗╚══██╔══╝╚██╗██╔╝╚══██╔══╝ ║
║  ██████╔╝█████╗  ██████╔╝██║   ██║ █████╔╝   ██║    ╚███╔╝    ██║    ║
║  ██╔══██╗██╔══╝  ██╔═══╝ ██║   ██║██╔═══╝    ██║    ██╔██╗    ██║    ║
║  ██║  ██║███████╗██║     ╚██████╔╝███████╗   ██║   ██╔╝ ██╗   ██║    ║
║  ╚═╝  ╚═╝╚══════╝╚═╝      ╚═════╝ ╚══════╝   ╚═╝   ╚═╝  ╚═╝   ╚═╝    ║
║                                                                      ║
║                  REPOSITORY ANALYSIS TERMINAL v2.0                   ║
╚══════════════════════════════════════════════════════════════════════╝
"""

    console.print(banner, style="highlight")


@click.command()
@click.argument('repo', required=True)
@click.option('--output-dir', '-o', default='output', help='Output directory for results')
@click.option('--max-file-size', '-m', type=int, help='Maximum file size in bytes (default: 1MB)')
@click.option('--no-tokens', is_flag=True, help='Disable token counting')
@click.option('--format', '-f', type=click.Choice(['xml', 'markdown']), default='markdown', 
              help='Output format for file contents')
@click.option('--json', 'export_json', is_flag=True, help='Export token data as JSON')
@click.option('--theme', '-t', type=click.Choice(['manhattan', 'green', 'matrix', 'sunset']), 
              default=get_alternating_theme(), help='Terminal color theme')
@click.option('--ai-select', is_flag=True, help='Use AI to help select files via conversation')
@click.option('--ai-query', help='Query for AI file selection (use with --ai-select)')
@click.option('--token-budget', type=int, default=100000, help='Token budget for AI selection')
@click.option('--debug', is_flag=True, help='Enable debug mode for AI selection (shows system prompts, tool calls)')
@click.option('--prompt-style', type=click.Choice(['standard', 'meta-reasoning', 'xml']), default='standard',
              help='System prompt style for AI selection (standard, meta-reasoning, or xml)')
@click.version_option()
def main(repo: str, output_dir: str, max_file_size: int, no_tokens: bool, 
         format: str, export_json: bool, theme: str, ai_select: bool,
         ai_query: str, token_budget: int, debug: bool, prompt_style: str) -> None:
    """
    Analyze a GitHub repository or local codebase.
    
    REPO can be:
    - GitHub URL: https://github.com/owner/repo
    - GitHub shorthand: owner/repo  
    - Local directory path: /path/to/repo or .
    
    Examples:
    
        repo2txt https://github.com/django/django
        
        repo2txt astropy/astropy
        
        repo2txt . --no-tokens
        
        repo2txt /path/to/project --format xml --json

        repo2txt https://github.com/openai/openai-python --token-budget 20000 --ai-select 
        

    """
    # Create console manager with selected theme
    console = ConsoleManager(theme=theme)
    
    setup_logging(debug)
    
    # Print banner
    print_banner(console)
    
    try:
        # Initialize
        console.print("\n[dim]INITIALIZING SYSTEM...[/dim]")
        console.print("[dim]LOADING CONFIGURATION...[/dim]")
        
        # Create configuration
        config = Config(
            github_token=os.environ.get('GITHUB_TOKEN', ''),
            enable_token_counting=not no_tokens,
            output_format=format,
            ai_select=ai_select,
            ai_query=ai_query,
            token_budget=token_budget,
            export_json=export_json,
            debug=debug,
            prompt_style=prompt_style
        )
        
        if max_file_size:
            config.max_file_size = max_file_size
        
        # Create analyzer
        console.print("[dim]STARTING ANALYZER...[/dim]")
        analyzer = RepositoryAnalyzer(config, theme)
        
        # Analyze repository
        console.print(f"\n[highlight]> ACQUIRED REPO:[/highlight] [path]{repo}[/path]")
        console.print("[info]> COMMENCING ANALYSIS...[/info]\n")
        
        result = analyzer.analyze(repo)
        
        # Only auto-save for non-AI mode
        if not config.ai_select:
            # Save results
            console.print("\n[info]> WRITING OUTPUT FILES...[/info]")
            output_files = analyzer.save_results(result, output_dir)
        else:
            output_files = {}
        
        # Display results
        console.print("\n" + "═" * 60)
        console.print("[success]ANALYSIS COMPLETE[/success]")
        console.print("═" * 60)
        
        console.print(f"[info]REPOSITORY:[/info] [path]{result.repo_name}[/path]")
        if result.branch:
            console.print(f"[info]BRANCH:[/info] [path]{result.branch}[/path]")
        console.print(f"[info]FILES ANALYZED:[/info] [number]{result.total_files}[/number]")
        
        if config.enable_token_counting:
            console.print(f"[info]TOTAL TOKENS:[/info] [number]{result.total_tokens:,}[/number]")
        
        # Output files
        if output_files:
            console.print("\n[info]OUTPUT FILES:[/info]")
            for file_type, file_path in output_files.items():
                rel_path = os.path.relpath(file_path)
                console.print(f"  [dim]>[/dim] {file_type.upper()}: [path]{rel_path}[/path]")
        elif config.ai_select:
            console.print("\n[info]Use /generate command in AI chat to create output files[/info]")
        
        # Warnings
        if result.has_errors():
            console.print(f"\n[warning]WARNINGS DETECTED:[/warning] [number]{len(result.errors)}[/number]")
            if debug:
                for error in result.errors[:5]:
                    console.print(f"  [dim]>[/dim] {error}")
                if len(result.errors) > 5:
                    console.print(f"  [dim]... +{len(result.errors) - 5} more[/dim]")
        
        console.print("\n[dim]SYSTEM SHUTDOWN[/dim]")
        console.print("═" * 60)
    
    except KeyboardInterrupt:
        console.print("\n[error]> PROCESS TERMINATED BY USER[/error]")
        console.print("[dim]SYSTEM SHUTDOWN[/dim]")
        sys.exit(1)
    
    except Exception as e:
        console.print(f"\n[error]> CRITICAL ERROR:[/error] {str(e)}")
        if debug:
            console.print_exception()
        console.print("[dim]SYSTEM SHUTDOWN[/dim]")
        sys.exit(1)


if __name__ == '__main__':
    main()