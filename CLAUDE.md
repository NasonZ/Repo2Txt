# Claude Development Guide - Repo2Txt

## Overview

Repo2Txt is an AI-powered repository analysis tool that converts GitHub repositories and local codebases into LLM-friendly text formats. It features a sophisticated AI-powered file selection system with conversational interface, comprehensive token analysis, and multiple output formats.

**Key Capabilities:**
- AI-assisted file selection via conversational chat interface
- Manual interactive file selection with directory navigation
- Multi-source support (GitHub, local directories)
- Token counting and budget management
- Multiple output formats (Markdown, XML, JSON)
- Rich terminal UI with themes

## Architecture Overview

### High-Level Components

```
repo2txt/
├── cli.py                 # Main CLI entry point
├── adapters/             # Repository source adapters
│   ├── base.py           # Abstract adapter interface
│   ├── github.py         # GitHub API integration
│   └── local.py          # Local filesystem adapter
├── ai/                   # AI-powered file selection system
│   ├── file_selector_agent.py  # Main AI agent orchestrator
│   ├── agent_session.py        # Session state management
│   ├── chat_orchestrator.py    # Chat flow coordination
│   ├── command_handler.py      # Slash command processing
│   ├── llm.py                  # LLM client & streaming
│   ├── prompts.py              # System prompt generation
│   ├── state.py                # File selection state & token cache
│   ├── tools.py                # AI function calling tools
│   └── console_chat.py         # Terminal chat UI
├── core/                 # Analysis engine
│   ├── analyzer.py       # Main analysis orchestrator
│   ├── file_analyzer.py  # Individual file processing
│   ├── models.py         # Core data structures
│   └── tokenizer.py      # Token counting utilities
└── utils/               # Shared utilities
    ├── console.py        # Terminal UI management
    ├── encodings.py      # File encoding detection
    ├── file_filter.py    # File filtering logic
    └── logging_config.py # Logging configuration
```

### Core Design Principles

1. **Modular Architecture**: Each component has a single, clear responsibility
2. **Survival-Oriented Error Handling**: Graceful degradation when components fail
3. **Performance by Default**: Token caching, efficient processing, minimal I/O
4. **Developer Experience**: Rich debugging, comprehensive testing, clear APIs

## Development Workflow

### Environment Setup

The project uses **uv** for dependency management (preferred) with pip as fallback:

```bash
# Recommended: uv
uv sync --dev
source .venv/bin/activate

# Alternative: pip
pip install -e ".[dev]"
```

### Development Commands

Based on `pyproject.toml` configuration:

```bash
# Testing
pytest                              # Run all tests
pytest --cov=repo2txt --cov-report=html  # With coverage
pytest tests/test_ai_components.py  # Specific test category

# Code Quality
black src/ tests/           # Code formatting (line-length: 100)
ruff check src/ tests/      # Linting
mypy src/                   # Type checking

# Build
python -m build            # Build distribution packages
```

### Project Structure Standards

- **Source code**: `src/repo2txt/` (using src layout)
- **Tests**: `tests/` (unit tests) and `tests_integration/` (integration tests)
- **Line length**: 100 characters (Black + Ruff configured)
- **Python versions**: 3.9+ (primary target: 3.11+)

### Adapter Implementation Notes

**Critical Architecture Guidance**: The LocalAdapter and GitHubAdapter implementations appear similar but have fundamentally different internal architectures:

- **LocalAdapter**: Real-time I/O strategy (filesystem access, on-demand token counting)
- **GitHubAdapter**: Pre-cached API strategy (memory-based, pre-computed tokens)

**⚠️ Do not attempt to consolidate `traverse_interactive()` implementations** despite UI similarity. These differences exist due to:
- API constraints vs filesystem access patterns
- Network latency vs direct I/O characteristics  
- Rate limiting vs unlimited filesystem operations

For detailed technical comparison, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md#repository-adapter-implementations).

## AI System Architecture

### Core AI Components

The AI system is built around a conversational agent that helps users select relevant files:

#### 1. FileSelectorAgent (`ai/file_selector_agent.py`)
- **Role**: Main orchestrator for AI-powered file selection
- **Key Features**:
  - Conversational interface for natural file selection
  - Integration with repository analysis
  - Token budget management
  - Multiple prompt styles (standard, meta-reasoning, xml)
  - Debug mode with system prompt visibility

#### 2. State Management (`ai/state.py`)
- **FileSelectionState**: Tracks selected files, token counts, budget usage
- **StateManager**: Validates file selections, manages token calculations
- **Features**:
  - Real-time budget tracking
  - File path validation
  - State snapshots for undo functionality

#### 3. Tool System (`ai/tools.py`)
- **Purpose**: Enables AI to interact with file selection via function calling
- **Tools Available**:
  - `select_files`: Replace entire selection with new files
  - `adjust_selection`: Add/remove files from current selection
- **Features**: Async execution, error handling, OpenAI format conversion

#### 4. LLM Integration (`ai/llm.py`)
- **LLMClient**: Manages API interactions with OpenAI/compatible endpoints
- **Features**:
  - Streaming response handling
  - Qwen model optimizations
  - Tool calling support
  - Response parsing and cleaning

#### 5. Chat Interface (`ai/console_chat.py`, `ai/command_handler.py`)
- **ChatConsole**: Rich terminal interface with themes
- **CommandHandler**: Processes slash commands (`/help`, `/generate`, `/debug`, etc.)
- **Features**:
  - Real-time state visualization
  - Command system for advanced operations
  - Debug output with tool call visibility

### AI Integration Flow

1. **Repository Analysis**: Core analyzer processes repository, builds file tree
2. **AI Agent Initialization**: Agent receives pre-analyzed data (file list, token counts)
3. **Conversational Selection**: User chats with AI to select relevant files
4. **Tool Execution**: AI uses function calling to modify file selection
5. **Output Generation**: Selected files are processed into requested formats

## Core Analysis Engine

### Repository Adapters

The adapter pattern enables support for multiple repository sources:

#### Base Adapter (`adapters/base.py`)
- **Abstract interface** for all repository types
- **Key methods**:
  - `get_name()`: Repository name extraction
  - `get_readme_content()`: README content retrieval
  - `traverse_interactive()`: Manual file selection
  - `get_file_content()`: Single file content retrieval
  - `build_file_tree()`: Generate text tree representation

#### Implementations
- **LocalAdapter** (`adapters/local.py`): Filesystem access with encoding detection
- **GitHubAdapter** (`adapters/github.py`): GitHub API integration with rate limiting

### Analysis Pipeline

#### 1. Repository Analyzer (`core/analyzer.py`)
- **Main orchestrator** for analysis workflow
- **Dual modes**:
  - Traditional interactive selection
  - AI-assisted selection (new)
- **Output generation**: Multiple formats with token reports

#### 2. File Processing (`core/file_analyzer.py`)
- **Token counting**: Using tiktoken (cl100k_base encoder)
- **Content analysis**: File type detection, encoding handling
- **Filtering**: Binary detection, size limits, exclusion patterns

#### 3. Data Models (`core/models.py`)
- **Config**: Application configuration with environment variable loading
- **AnalysisResult**: Complete analysis output structure
- **FileNode**: Individual file/directory representation
- **TokenBudget**: Budget tracking for LLM interactions

## Key Features & Implementation

### 1. Token Management
- **Caching**: Calculated tokens are cached to avoid recomputation
- **Budget tracking**: Real-time monitoring of token usage vs. budget
- **Optimization**: AI selection respects token limits automatically

### 2. Error Handling
- **Graceful degradation**: AI selection falls back to manual on failure
- **Comprehensive logging**: Detailed error reporting with debug mode
- **User feedback**: Clear error messages with actionable suggestions

### 3. Performance Optimizations
- **Lazy loading**: File contents loaded only when needed
- **Progress tracking**: Rich progress bars for long operations
- **Efficient processing**: Minimal file system operations

### 4. Extensibility
- **Plugin architecture**: New adapters can be added easily
- **Tool system**: New AI tools can be registered
- **Theme support**: Multiple terminal color schemes

## Environment Configuration

### Required Environment Variables

Create `.env` file in project root:

```bash
# LLM Configuration (required for AI mode)
LLM_PROVIDER=openai          # openai, ollama, llamacpp
LLM_MODEL=gpt-4-turbo        # Model name
LLM_API_KEY=your_api_key     # API key
LLM_BASE_URL=                # Custom endpoint (optional)

# GitHub Access (optional)
GITHUB_TOKEN=your_token      # For private repositories
```

### Development Environment

The project supports multiple LLM providers:
- **OpenAI**: GPT-4.1, GPT-3.5-turbo
- **Ollama**: Local models (qwen3:32b, llama3.2, etc.)
- **llama.cpp**: Local server endpoints

## Usage Patterns

### 1. AI-Assisted Analysis (Recommended)
```bash
# Interactive AI selection
repo2txt . --ai-select

# Query-driven selection
repo2txt . --ai-select --ai-query "Show me the main API endpoints and models"

# Advanced options
repo2txt . --ai-select --prompt-style meta-reasoning --token-budget 50000 --debug
```

### 2. Manual Analysis
```bash
# Interactive directory navigation
repo2txt /path/to/repo

# GitHub repository
repo2txt https://github.com/owner/repo

# Multiple output formats
repo2txt . --format xml --json
```

### 3. Programmatic Usage
```python
from repo2txt.core.analyzer import RepositoryAnalyzer
from repo2txt.core.models import Config

# Traditional analysis
config = Config(enable_token_counting=True)
analyzer = RepositoryAnalyzer(config, theme="manhattan")
result = analyzer.analyze("/path/to/repo")

# AI-assisted analysis
config.ai_select = True
config.ai_query = "Select all Python files related to data processing"
result = analyzer.analyze("/path/to/repo")
```

## Testing Strategy

### Test Organization
- **Unit tests**: `tests/` - Individual component testing
- **Integration tests**: `tests_integration/` - End-to-end workflows
- **Coverage**: Configured for comprehensive reporting

### Key Test Areas
- **Adapter functionality**: Repository access and parsing
- **AI components**: Tool execution, state management
- **Token counting**: Accuracy and performance
- **Error handling**: Graceful failure modes

### Running Tests
```bash
# All tests
pytest

# Specific categories
pytest tests/test_analyzer.py      # Core analysis
pytest tests/test_ai_components.py # AI system

# With coverage
pytest --cov=repo2txt --cov-report=html
```

## Debugging and Development

### Debug Mode
Enable comprehensive debugging with `--debug` flag:
- System prompts shown before AI interaction
- Tool calls and responses displayed
- Token usage tracking
- File selection reasoning

### Logging Configuration
Centralized logging in `utils/logging_config.py`:
- Module-specific log levels
- Rich formatting for development
- External library noise reduction

### Common Development Tasks

#### Adding New AI Tools
1. Define tool in `ai/tools.py`
2. Implement tool function
3. Register in `FileSelectorAgent._register_tools()`
4. Test with debug mode

#### Adding New Repository Sources
1. Extend `adapters/base.py`
2. Implement required methods
3. Register in adapter factory
4. Add integration tests

#### Extending Output Formats
1. Modify `core/analyzer.py` output generation
2. Update `core/models.py` configuration
3. Add format-specific tests

## Performance Considerations

### Token Counting
- Uses tiktoken library for accuracy
- Results cached to avoid recomputation
- Progress bars for large repositories

### Memory Usage
- Lazy loading of file contents
- Streaming for large files
- Efficient data structures

### API Rate Limiting
- GitHub API respects rate limits
- Configurable retry logic
- Token-based authentication preferred

## Future Development Areas

### Planned Enhancements
- Additional LLM providers (Anthropic Claude, local models)
- Enhanced file filtering options
- Better diff visualization for file selection
- Export to additional formats

### Architecture Improvements
- Plugin system for custom analyzers
- Distributed processing for large repositories
- Enhanced caching mechanisms
- Better error recovery strategies

---

This guide provides the foundational knowledge needed to understand, modify, and extend the Repo2Txt codebase. The modular architecture and comprehensive documentation make it accessible for both maintenance and feature development.