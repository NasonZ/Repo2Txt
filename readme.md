# Repo2Txt

**AI-Powered Repository Analysis & LLM-Ready Text Conversion**

Transform GitHub repositories and local codebases into LLM-friendly formats with intelligent file selection, comprehensive token analysis, and conversational AI assistance.

## 🌟 What's New

### 🤖 AI-Powered File Selection
- **Conversational Interface**: Chat with an AI assistant to select files naturally
- **Smart Recommendations**: AI understands your intent and suggests relevant files
- **Multiple Prompt Styles**: Choose between standard or meta-reasoning prompts
- **Token Budget Management**: AI respects your token limits and optimises selections
- **Interactive Commands**: Full command system for fine-tuning your selection

### 🏗️ Modern Architecture
- **Modular Design**: Clean, testable components built for extensibility
- **Robust Error Handling**: Graceful fallbacks and survival-oriented design
- **Performance Optimized**: Token caching and efficient processing
- **Developer Friendly**: Rich debugging tools and comprehensive logging

## 🚀 Key Features

### Dual Analysis Modes

**🎯 AI-Assisted Selection** (*Recommended*)
- Natural language queries for file selection
- Intelligent recommendations based on your goals
- Interactive chat interface with command system
- Real-time token budget tracking
- Multiple output formats (Markdown, XML, JSON)

**📂 Manual Selection** 
- Interactive directory navigation
- Granular file/folder selection
- Back navigation and flexible options
- Visual progress tracking

### Advanced Capabilities
- **Multi-Source Support**: GitHub repositories, local directories, and archives
- **Intelligent Processing**: Binary detection, encoding fallbacks, size limits
- **Rich Output Formats**: Markdown, XML, or JSON with detailed token reports
- **Token Analysis**: Comprehensive counting, distribution stats, and budget recommendations
- **Theme Support**: Multiple terminal color themes (Manhattan, Matrix, Green, Sunset)

## 📋 Prerequisites

- **Python 3.8+** (recommended: 3.11+)
- **LLM API Access**: OpenAI, Ollama, llama.cpp, or compatible endpoints
- **GitHub Token**: For private repositories (optional)

## 🔧 Installation

### Quick Start with uv (Recommended)
```bash
git clone https://github.com/your-username/repo2txt.git
cd repo2txt
uv sync
source .venv/bin/activate
```

### Traditional Installation
```bash
git clone https://github.com/your-username/repo2txt.git
cd repo2txt
pip install -e .
```

### Development Setup
```bash
uv sync --dev
source .venv/bin/activate
pytest  # Run tests
```

## 🚀 Quick Start

### 1. AI-Assisted Analysis (Recommended)

```bash
# Analyze current directory with AI assistance
repo2txt . --ai-select

# Use specific query for targeted selection  
repo2txt . --ai-select --ai-query "Show me the main API endpoints and database models"

# Advanced options
repo2txt . --ai-select --prompt-style meta-reasoning --token-budget 50000 --theme matrix
```

### 2. Traditional Manual Selection

```bash
# Analyze with interactive selection
repo2txt /path/to/repo

# GitHub repository
repo2txt https://github.com/owner/repo

# Export multiple formats
repo2txt . --format xml --json
```

## 🤖 AI Chat Commands

During AI-assisted selection, use these commands:

### Core Commands
- `/help` - Show all available commands
- `/generate [format]` - Create output files (markdown, xml, json, all)
- `/save [filename]` - Save chat history
- `/clear` - Reset conversation and selection
- `/quit` - Exit the application

### Selection Control
- `/undo` - Undo last action
- `/redo` - Regenerate last AI response
- `/toggle streaming` - Enable/disable streaming responses
- `/toggle thinking` - Enable/disable thinking mode (Qwen models)
- `/toggle prompt` - Cycle through prompt styles
- `/toggle budget <N>` - Set token budget

### Debug & Development
- `/debug` - Toggle debug mode
- `/debug state` - Show current configuration

## 🎨 Configuration

### Environment Variables

Create a `.env` file:
```bash
# LLM Configuration
LLM_PROVIDER=openai          # openai, ollama, llamacpp
LLM_MODEL=gpt-4-turbo        # Model name
LLM_API_KEY=your_api_key     # API key
LLM_BASE_URL=                # Custom endpoint (optional)

# GitHub Access
GITHUB_TOKEN=your_token      # For private repos

# UI Preferences  
DEFAULT_THEME=manhattan      # manhattan, matrix, green, sunset
```

### Command Line Options

```bash
repo2txt [REPO] [OPTIONS]

Arguments:
  REPO                    Repository path, GitHub URL, or GitHub shorthand (owner/repo)

Core Options:
  --output-dir, -o DIR    Output directory (default: output)
  --format, -f FORMAT     Output format: xml, markdown (default: markdown)
  --theme, -t THEME       Terminal theme: manhattan, green, matrix, sunset
  --max-file-size SIZE    Maximum file size in bytes (default: 1MB)
  --no-tokens            Disable token counting
  --json                 Export token data as JSON

AI Selection Options:
  --ai-select            Enable AI-assisted file selection
  --ai-query QUERY       Specific query for AI selection
  --token-budget N       Token budget for AI selection (default: 100000)
  --prompt-style STYLE   Prompt style: standard, meta-reasoning, xml
  --debug               Enable debug mode (shows system prompts, tool calls)
```

## 📊 Output Structure

### Default Output (2-3 files)
```
output/
└── RepoName_20240315_143022/
    ├── RepoName_analysis.md      # Main content in Markdown
    ├── RepoName_tokens.txt       # Detailed token report  
    └── RepoName_tokens.json      # Token data (if --json)
```

### AI Chat History (optional)
```
output/
└── chat_history/
    └── chat_history_20240315_143022.json
```

## 🎯 Example Workflows

### 1. Quick API Documentation
```bash
repo2txt . --ai-select --ai-query "Select all API route files and documentation"
```

### 2. Architecture Overview
```bash
repo2txt . --ai-select --ai-query "Show me the main architecture components and configuration files"
```
### 3. Interactive Exploration
```bash
repo2txt . --ai-select --debug --theme matrix
# Then chat: "What are the main components of this project?"
```

## 🏗️ Architecture

### Core Components

```
repo2txt/
├── adapters/           # Repository source adapters (GitHub, local, etc.)
├── ai/                # AI-powered file selection system
│   ├── agent_session.py    # Session state management
│   ├── chat_orchestrator.py # Chat flow coordination  
│   ├── command_handler.py   # Command processing
│   ├── file_selector_agent.py # Main AI agent
│   ├── llm.py              # LLM client & streaming
│   ├── prompts.py          # System prompt generation
│   ├── qwen_utils.py       # Qwen model utilities
│   ├── state.py            # File selection state & token cache
│   └── tools.py            # AI function calling tools
├── core/              # Analysis engine
│   ├── analyzer.py         # Main analysis orchestrator
│   ├── file_analyzer.py    # Individual file processing
│   ├── models.py           # Core data structures
│   └── tokenizer.py        # Token counting utilities
└── utils/             # Shared utilities
    ├── console.py          # Terminal UI management
    ├── console_base.py     # Base console functionality
    ├── encodings.py        # File encoding detection
    ├── file_filter.py      # File filtering logic
    └── logging_config.py   # Logging configuration
```

### Design Principles

- **Modular Architecture**: Each component has a single, clear responsibility
- **Survival-Oriented Error Handling**: Graceful degradation when components fail
- **Performance by Default**: Token caching, efficient processing, minimal I/O
- **Developer Experience**: Rich debugging, comprehensive testing, clear APIs

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/test_ai_components.py      # AI system tests
pytest tests/test_analyzer.py           # Core analysis tests
pytest tests/test_tokenizer.py          # Token counting tests

# Run with coverage
pytest --cov=repo2txt --cov-report=html
```

## 🔧 Advanced Usage

### Custom LLM Endpoints

```bash
# Ollama
export LLM_PROVIDER=ollama
export LLM_BASE_URL=http://localhost:11434
export LLM_MODEL=qwen3:32b

# llama.cpp server
export LLM_PROVIDER=llamacpp  
export LLM_BASE_URL=http://localhost:8080
export LLM_MODEL=llama3.2:3b
```

### Programmatic Usage

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

## 🐛 Troubleshooting

### Common Issues

**AI Selection Not Working**
- Verify LLM_API_KEY is set correctly
- Check LLM_BASE_URL for custom endpoints
- Use `--debug` to see system prompts and API calls

**Token Counting Errors**
- Use `--no-tokens` to disable if needed

**GitHub Rate Limiting**
- Set GITHUB_TOKEN environment variable
- Use personal access token with appropriate scopes

**Large Repository Performance**
- Use `--max-file-size` to limit individual files
- Set appropriate `--token-budget` for AI selection
- Consider analyzing specific subdirectories
- Models less performant than gpt-4.1-mini start to hallcinate when give large file tree's (> 250 files)

### Debug Mode

Enable comprehensive debugging:
```bash
repo2txt . --ai-select --debug
```

This shows:
- Message history sent to the LLM
- Tool calls and responses
- Token usage and budget tracking
- File selection reasoning

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Fork & Branch**: Create feature branches from `main`
2. **Test**: Ensure all tests pass with `pytest`
3. **Document**: Update README for user-facing changes
4. **Pull Request**: Submit with clear description of changes

### Development Setup
```bash
git clone https://github.com/your-username/repo2txt.git
cd repo2txt
uv sync --dev
source .venv/bin/activate
pytest  # Verify setup
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Original Concept**: [Doriandarko/RepoToTextForLLMs](https://github.com/Doriandarko/RepoToTextForLLMs)
- **Enhanced Architecture**:  Redesigned with imporve UX and added AI assistance.
