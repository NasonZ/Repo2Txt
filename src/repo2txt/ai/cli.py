"""CLI interface for AI file selection functionality.

This module provides the command-line interface specifically for
the AI file selection agent.
"""

import sys
import logging
from pathlib import Path

import click
from dotenv import load_dotenv

from .file_selector_agent import FileSelectorAgent
from .llm import get_llm_config_from_env

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


@click.command()
@click.option('--repo-path', default=".", help="Path to the repository to analyze.")
@click.option('--model', default=None, help="OpenAI model name. If not provided, uses LLM_MODEL from .env file.")
@click.option('--api-key', default=None, help="OpenAI API key. If not provided, uses LLM_API_KEY or OPENAI_API_KEY from .env file.")
@click.option('--base-url', default=None, help="Base URL for API. If not provided, uses LLM_BASE_URL from .env file.")
@click.option('--theme', type=click.Choice(['green', 'amber', 'matrix']), default='green', help='UI theme.')
@click.option('--budget', default=50000, type=int, help="Token budget for selection.")
@click.option('--debug', is_flag=True, help="Enable debug mode to show system prompts and tool details.")
@click.option('--prompt-style', type=click.Choice(['standard', 'meta-reasoning', 'xml']), default='standard', help="System prompt style for A/B testing.")
def main_cli(repo_path: str, model: str, api_key: str, base_url: str, theme: str, 
             budget: int, debug: bool, prompt_style: str):
    """Interactive File Selection Agent Demo
    
    This demo creates an AI assistant that helps you select files from a repository
    using OpenAI's tool calling API with smart token management.
    
    Configuration is loaded from .env file by default. You can override specific
    values using command line options.
    
    Use --debug to see system prompt and tool calling details.
    Use --prompt-style to A/B test different prompting approaches:
    - standard: Simple, direct instructions
    - meta-reasoning: Detailed framework with reasoning guidelines
    - xml: Same as meta-reasoning but with XML formatting
    """
    
    # Get configuration from environment
    try:
        llm_config = get_llm_config_from_env()
    except ValueError as e:
        click.echo(f"❌ Configuration error: {e}", err=True)
        click.echo("💡 Make sure to set your API key in the .env file or via command line options", err=True)
        sys.exit(1)
    
    # Override with command line options if provided
    final_model = model or llm_config["model"]
    final_api_key = api_key or llm_config["api_key"]
    final_base_url = base_url or llm_config["base_url"]
    
    # Validate final configuration
    if not final_api_key:
        click.echo("❌ No API key provided. Set OPENAI_API_KEY or LLM_API_KEY in .env file or use --api-key option.", err=True)
        sys.exit(1)
    
    # Show configuration
    mode_indicator = " [DEBUG]" if debug else ""
    prompt_indicator = f" [{prompt_style.upper()}]" if prompt_style != "standard" else ""
    config_display = f"Model: {final_model}{mode_indicator}{prompt_indicator}"
    if final_base_url:
        config_display += f" | Base URL: {final_base_url}"
    config_display += f" | Budget: {budget:,} tokens"
    click.echo(f"🔧 {config_display}")
    
    try:
        # Create and run agent
        agent = FileSelectorAgent(
            repo_path=repo_path,
            openai_api_key=final_api_key,
            model=final_model,
            base_url=final_base_url,
            theme=theme,
            token_budget=budget,
            debug_mode=debug,
            prompt_style=prompt_style
        )
        agent.run()
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main_cli()