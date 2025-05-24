"""Command-line interface for repo2txt."""
import os
import sys
import logging
from pathlib import Path

import click
from rich.console import Console
from rich.theme import Theme

from .core.models import Config
from .core.analyzer import RepositoryAnalyzer

# Define retro terminal themes
THEMES = {
    'manhattan': Theme({
        'info': 'cyan',
        'warning': 'yellow', 
        'error': 'red',
        'success': 'green',
        'highlight': 'bright_cyan',
        'panel.border': 'bright_cyan',
        'header': 'bold bright_cyan on black',
        'prompt': 'cyan',
        'path': 'white',
        'number': 'bright_blue',
        'dim': 'bright_black'
    }),
    'amber': Theme({
        'info': 'yellow',
        'warning': 'bright_yellow',
        'error': 'red',
        'success': 'green',
        'highlight': 'bold yellow',
        'panel.border': 'yellow',
        'header': 'bold yellow on black',
        'prompt': 'yellow',
        'path': 'bright_yellow',
        'number': 'yellow',
        'dim': 'yellow'
    }),
    'green': Theme({
        'info': 'green',
        'warning': 'yellow',
        'error': 'red', 
        'success': 'bright_green',
        'highlight': 'bold green',
        'panel.border': 'green',
        'header': 'bold green on black',
        'prompt': 'green',
        'path': 'bright_green',
        'number': 'green',
        'dim': 'green'
    }),
    'matrix': Theme({
        'info': 'bright_green',
        'warning': 'yellow',
        'error': 'red',
        'success': 'green',
        'highlight': 'bold bright_green',
        'panel.border': 'green',
        'header': 'bold bright_green on black',
        'prompt': 'bright_green',
        'path': 'green',
        'number': 'bright_green',
        'dim': 'green'
    })
}

# Initialize console with Dr. Manhattan theme by default
console = Console(theme=THEMES['manhattan'])


def setup_logging(debug: bool) -> None:
    """Configure logging based on debug flag."""
    level = logging.DEBUG if debug else logging.INFO
    format_string = '[%(levelname)s] %(message)s'
    
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=[logging.StreamHandler(sys.stderr)]
    )


def print_banner() -> None:
    """Print retro terminal banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║  ██████╗ ███████╗██████╗  ██████╗ ██████╗ ████████╗██╗  ██╗ ║
║  ██╔══██╗██╔════╝██╔══██╗██╔═══██╗╚════██╗╚══██╔══╝╚██╗██╔╝ ║
║  ██████╔╝█████╗  ██████╔╝██║   ██║ █████╔╝   ██║    ╚███╔╝  ║
║  ██╔══██╗██╔══╝  ██╔═══╝ ██║   ██║██╔═══╝    ██║    ██╔██╗  ║
║  ██║  ██║███████╗██║     ╚██████╔╝███████╗   ██║   ██╔╝ ██╗ ║
║  ╚═╝  ╚═╝╚══════╝╚═╝      ╚═════╝ ╚══════╝   ╚═╝   ╚═╝  ╚═╝ ║
║                                                               ║
║              REPOSITORY ANALYSIS TERMINAL v1.0                ║
╚═══════════════════════════════════════════════════════════════╝
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
@click.option('--theme', '-t', type=click.Choice(['manhattan', 'amber', 'green', 'matrix']), 
              default='manhattan', help='Terminal color theme')
@click.option('--debug', is_flag=True, help='Enable debug logging')
@click.version_option()
def main(repo: str, output_dir: str, max_file_size: int, no_tokens: bool, 
         format: str, export_json: bool, theme: str, debug: bool) -> None:
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
    """
    # Apply selected theme
    global console
    console = Console(theme=THEMES[theme])
    
    setup_logging(debug)
    
    # Print banner
    print_banner()
    
    try:
        # Initialize
        console.print("\n[dim]INITIALIZING SYSTEM...[/dim]")
        console.print("[dim]LOADING CONFIGURATION...[/dim]")
        
        # Create configuration
        config = Config(
            github_token=os.environ.get('GITHUB_TOKEN', ''),
            enable_token_counting=not no_tokens,
            output_format=format
        )
        
        if max_file_size:
            config.max_file_size = max_file_size
        
        # Add export_json to config for analyzer
        config.export_json = export_json
        
        # Create analyzer
        console.print("[dim]STARTING ANALYZER...[/dim]")
        analyzer = RepositoryAnalyzer(config)
        
        # Analyze repository
        console.print(f"\n[highlight]> TARGET ACQUIRED:[/highlight] [path]{repo}[/path]")
        console.print("[info]> COMMENCING ANALYSIS...[/info]\n")
        
        result = analyzer.analyze(repo)
        
        # Save results
        console.print("\n[info]> WRITING OUTPUT FILES...[/info]")
        output_files = analyzer.save_results(result, output_dir)
        
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
        console.print("\n[info]OUTPUT FILES:[/info]")
        for file_type, file_path in output_files.items():
            rel_path = os.path.relpath(file_path)
            console.print(f"  [dim]>[/dim] {file_type.upper()}: [path]{rel_path}[/path]")
        
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