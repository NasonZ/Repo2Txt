# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Repo2Txt is a Python tool that converts GitHub repositories or local codebases into LLM-friendly text formats with intelligent token management. It provides interactive file selection and comprehensive token analysis to help users stay within LLM token limits while extracting the most relevant code.

## Current State

The project is transitioning from a monolithic script (`repo2txt.py`) to a modular architecture under `src/repo2txt/`. Both implementations currently exist:
- **Legacy**: `repo2txt.py` - Working monolithic implementation (59KB)
- **New**: `src/repo2txt/` - Modular architecture in development

## Architecture Overview

### Core Components

1. **RepositoryAnalyzer** (`src/repo2txt/core/analyzer.py`)
   - Main orchestrator for analysis workflow
   - Manages output generation with format-aware file extensions (.md for markdown, .txt for xml)
   - Handles token report generation and JSON exports

2. **Repository Adapters** (`src/repo2txt/adapters/`)
   - `base.py`: Abstract base class with parse_range() utility
   - `github.py`: GitHub API integration via PyGithub
   - `local.py`: Local filesystem handling
   - Factory function in `__init__.py` creates appropriate adapter

3. **FileAnalyzer** (`src/repo2txt/core/file_analyzer.py`)
   - Binary file detection (extensions, MIME types, null bytes)
   - Multi-encoding support with fallback chain
   - File content reading with error handling

4. **TokenCounter** (`src/repo2txt/core/tokenizer.py`)
   - Graceful tiktoken handling (works without it)
   - Token estimation fallback using character count

5. **CLI** (`src/repo2txt/cli.py`)
   - Retro terminal interface with ASCII art banner
   - Theme support: manhattan (default), amber, green, matrix
   - Rich console output with Dr. Manhattan color scheme

### Key UX Features

- **Token-First Display**: Shows `~X,XXX tokens` instead of bytes
- **Visual Hierarchy**: Directories use `▸` prefix, files are indented
- **Running Totals**: Shows selected files and total tokens during selection
- **Clean Spacing**: Columnar layout prevents information density
- **Output Formats**: XML for machines, Markdown for humans

## Common Commands

### Running the Tool
```bash
# From project root with venv activated
source .venv/bin/activate

# Current method (until package install works)
python -m src.repo2txt <repo> [options]

# Examples
python -m src.repo2txt .                          # Current directory
python -m src.repo2txt https://github.com/org/repo
python -m src.repo2txt . --format xml --output-dir results
python -m src.repo2txt . --theme amber --no-tokens

# Legacy method still works
python repo2txt.py <repo> [options]
```

### Development Commands
```bash
# Code formatting
black src/ tests/ --line-length 100

# Linting  
ruff src/ tests/

# Type checking
mypy src/

# Run tests
pytest
pytest --cov=repo2txt --cov-report=html
```

## CLI Options

- `--output-dir, -o`: Output directory (default: "output")
- `--max-file-size, -m`: Max file size in bytes (default: 1MB)
- `--no-tokens`: Disable token counting for faster processing
- `--format, -f`: Choose 'xml' or 'markdown' (default: markdown)
- `--json`: Export token data as JSON
- `--theme, -t`: Terminal theme: manhattan, amber, green, matrix
- `--debug`: Enable debug logging

## Project Structure

```
repo2txt/
├── repo2txt.py          # Original implementation (to be refactored)
├── src/repo2txt/
│   ├── __main__.py      # Package entry point
│   ├── cli.py           # CLI with retro terminal UI
│   ├── core/
│   │   ├── analyzer.py  # Main orchestration
│   │   ├── file_analyzer.py
│   │   ├── models.py    # Config, FileNode, AnalysisResult
│   │   └── tokenizer.py
│   ├── adapters/
│   │   ├── base.py      # Abstract adapter
│   │   ├── github.py    # GitHub implementation
│   │   └── local.py     # Local filesystem
│   └── utils/
│       ├── encodings.py # Encoding detection
│       └── file_filter.py
├── tests/               # Test suite (needs implementation)
├── docs/
│   ├── development/vision.md  # 4-phase roadmap
│   └── technical/*.md   # Design documents
└── .cursor/rules/       # Development guidelines
```

## Configuration

### Environment Variables
- `GITHUB_TOKEN`: Required for GitHub repository analysis
- `UV_LINK_MODE`: Set to 'copy' if hardlinks fail across filesystems

### Config Model (`src/repo2txt/core/models.py`)
- `excluded_dirs`: Directories to skip (node_modules, .git, etc.)
- `binary_extensions`: File types to exclude
- `encoding_fallbacks`: Try UTF-8, Latin-1, CP1252, etc.
- `max_file_size`: Default 1MB
- `output_format`: 'xml' or 'markdown'

## Development Guidelines

### Code Style
- Line length: 100 characters (black/ruff configured)
- No comments unless essential - code should be self-documenting
- Use descriptive variable names over comments
- Type hints required (mypy strict mode)

### Error Handling
- Binary files show as `[binary]` in file list
- Encoding errors show as `[decode error]`  
- Empty directories show as `[empty]`
- All errors append to `adapter.errors` list

### Testing Approach
- Unit tests for each module
- Integration tests for adapters
- Mock external dependencies (GitHub API)
- Use pytest fixtures for test data

## Current Focus

1. **Immediate**: The modular architecture is functional but needs:
   - Comprehensive test coverage
   - Refactor repo2txt.py to use new modules
   - Package installation fixes (pip install -e .)

2. **Next Phase**: Per `docs/development/vision.md`:
   - LLM integration for intelligent file selection
   - Matplotlib visualizations
   - REST API development

## Cursor Rules Integration

This project follows guidelines in `.cursor/rules/`:
- **ideate.mdc**: Evidence-based reasoning, systems thinking
- **logging.mdc**: Structured logging with appropriate levels
- **task-list.mdc**: TodoWrite/TodoRead for task tracking
- **installing-packages.mdc**: Use uv for package management