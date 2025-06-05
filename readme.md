# Repo2Txt

**Transform any codebase into LLM-ready text with AI-powered intelligence**

> From GitHub repos to local projects — skip the copy-paste cycle. Get perfectly formatted LLM inputs with intelligent file selection, token budget management, and multi-format output.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Why Repo2Txt?

**The Problem**: Loading relevant context into LLMs is cumbersome and time-consuming:

- 🤔 **Decision Paralysis**: Which files are actually relevant? How many tokens will this selection use?
- ⏰ **Manual Tedium**: Copy-pasting files one by one, then formatting for LLM consumption
- 💸 **Token Waste**: Include too much → expensive. Include too little → incomplete analysis
- 🔄 **Iteration Hell**: Realise you need different files, start the process again

**The Solution**: Automated file selection and formatting with optional AI intelligence:
- 🚀 **Core Automation**: Skip the copy-paste-format cycle entirely - go straight from repo to LLM-ready text
- 🎯 **Manual Control**: Interactive directory navigation with full control over selection
- 🧠 **AI Enhancement**: Optional intelligent selection - "Show me the authentication system" → AI finds routes, models, middleware, tests
- 🎛️ **Token Aware**: Real-time token counting and budget management (both modes)
- 📄 **Multiple Formats**: Markdown, XML, JSON output for any LLM workflow
- ⚡ **Instant Output**: From repo to formatted text in seconds, not minutes

## 📋 Prerequisites

- **Python 3.8+** *(3.11+ recommended)*
- **LLM API Access**: OpenAI, Ollama, llama.cpp, or compatible endpoints
- **GitHub Token**: For private repositories *(optional)*

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

### Basic Usage
```bash
# Interactive manual selection
repo2txt /path/to/repo

# GitHub repository
repo2txt https://github.com/owner/repo

# Export multiple formats
repo2txt <repo> --format xml --json
```

### AI-Assisted Selection
```bash
# Let AI select files intelligently
repo2txt <repo> --ai-select

# Targeted selection with query
repo2txt <repo> --ai-select --ai-query "Show me the main API endpoints and database models"

# Advanced options
repo2txt <repo> --ai-select --prompt-style meta-reasoning --token-budget 50000 --theme matrix

# Skip large directories
repo2txt <repo> --exclude-dirs "datasets,logs,cache" --ai-select
```

## 🎬 See It in Action

```bash
$ python -m repo2txt ./my-ecommerce-api --ai-select --prompt-style meta-reasoning --exclude-dirs "datasets,logs" --token-budget 25000
```

```
[>]: How does the payment processing system work? Show me the main components and error handling.

[<] 
🔧 Calling tool 1: select_files

📝 Selection Changes:
  + src/payments/processor.py
  + src/payments/models.py  
  + src/payments/exceptions.py
  + src/api/payment_routes.py
  + tests/test_payments.py
  + config/payment_config.py

📊 Token Impact:
New total: 18,420 tokens (was 0)
Budget usage: 73.7% (18,420/25,000)
+18,420 tokens

[<] Here's my reasoning and the curated set of files to answer your question:

You want to understand the payment processing system's architecture and error handling. This requires:
- Core payment logic and business rules
- Data models and validation
- API endpoints and request/response handling  
- Error handling patterns and custom exceptions
- Configuration and external service integration
- Test coverage showing expected behaviours

To confidently answer this, I selected files that together reveal:

1. **Core Logic**: How payments are processed, validated, and completed (src/payments/processor.py)
2. **Data Models**: Payment entities, states, and relationships (src/payments/models.py)
3. **Error Handling**: Custom exceptions and failure scenarios (src/payments/exceptions.py)
4. **API Layer**: HTTP endpoints, request validation, response formatting (src/api/payment_routes.py)
5. **Configuration**: Payment provider settings, API keys, timeouts (config/payment_config.py)
6. **Testing**: Expected behaviours, edge cases, error scenarios (tests/test_payments.py)

With these files, you'll see the complete payment flow from API request through to external provider integration, including all error handling patterns.

— Want to dive deeper into specific providers or add webhook handling? Let me know!
```

## 🚀 Key Features

### Dual Selection Modes
- **🤖 AI-Assisted**: Conversational interface with intelligent recommendations and meta-reasoning prompts
- **📂 Manual Control**: Interactive directory navigation with granular file/folder selection

### Multi-Source Support
- **GitHub repositories**: Public/private repos with token authentication
- **Local directories**: Any folder structure with encoding detection
- **Archive support**: Handle compressed codebases

### Token Management
- **Real-time counting**: See token usage as you select files
- **Budget tracking**: Set limits and get warnings before exceeding
- **Distribution analysis**: Understand where your tokens are going
- **Caching**: Avoid recalculating tokens for unchanged files

### Output Formats
- **Markdown**: Clean, readable format for most LLMs
- **XML**: Structured format with clear file boundaries  
- **JSON**: Programmatic access with metadata
- **Multiple exports**: Generate all formats simultaneously

### Developer Experience
- **Terminal themes**: Manhattan, Matrix, Green, Sunset
- **Debug mode**: See system prompts, tool calls, and reasoning
- **Command system**: Interactive controls during AI sessions
- **Progress tracking**: Visual feedback for long operations

## 🛣️ Roadmap

### 🤝 Code Co-Pilot Mode
**Closing the loop**: Move beyond static output generation to dynamic code interaction.

- **Direct File Access**: AI reads selected files on-demand during conversation (no pre-generation needed)
- **Real-time Analysis**: Ask questions, get grounded answers based on live code examination
- **Adaptive Workflow**: Advanced models use read tools directly; simpler models generate output then switch to analysis mode
- **Continuous Interaction**: Analyse → Discuss → Refine → Repeat without export/import cycles

### 📝 Custom Output Templates
Take full control of your LLM-ready output with customisable templates:

```markdown
## Custom Template Example

<custom_prompt>
You are analyzing a ${PROJECT_TYPE} project. Focus on ${ANALYSIS_FOCUS}.
{ANALYSIS_INSTRUCTIONS}
</custom_prompt>

<readme>
${README_CONTENT}
</readme>

<project_files>
${SELECTED_FILES}
</project_files>

<reminder>
Key areas to examine: ${KEY_AREAS}
Remember to follow the instructions given.
</reminder>
```

**Template Features:**
- **Variable Substitution**: Dynamic content based on your project
- **Section Control**: Choose which sections to include/exclude
- **Format Flexibility**: Create templates for specific LLM workflows
- **Reusable Configs**: Save and share templates across projects

## 🤖 Commands & Configuration

### AI Chat Commands
During AI-assisted selection:
- `/help` - Show available commands
- `/generate [format]` - Create output (markdown, xml, json, all)
- `/save [filename]` - Save chat history
- `/clear` - Reset conversation and selection
- `/undo` - Undo last action
- `/toggle streaming` - Enable/disable streaming responses
- `/toggle thinking` - Enable/disable thinking mode (Qwen models)
- `/toggle prompt` - Cycle through prompt styles
- `/toggle budget <N>` - Set token budget

### Debug & Development
- `/debug` - Toggle debug mode
- `/debug state` - Show current configuration

### Environment Setup
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

Core Options:
  --output-dir, -o DIR    Output directory (default: output)
  --format, -f FORMAT     Output format: xml, markdown (default: markdown)
  --theme, -t THEME       Terminal theme: manhattan, green, matrix, sunset
  --max-file-size SIZE    Maximum file size in bytes (default: 1MB)
  --exclude-dirs DIRS     Comma-separated list of additional directories to exclude
  --no-tokens            Disable token counting
  --json                 Export token data as JSON

AI Options:
  --ai-select            Enable AI-assisted selection
  --ai-query QUERY       Specific query for AI selection
  --token-budget N       Token budget for AI selection (default: 100000)
  --prompt-style STYLE   Prompt style: standard, meta-reasoning, xml
  --debug               Enable debug mode (shows system prompts, tool calls)
```

## 📊 Output Structure

### Default Output
```
output/
└── RepoName_20240315_143022/
    ├── RepoName_analysis.md      # Main content
    ├── RepoName_tokens.txt       # Token report  
    └── RepoName_tokens.json      # Token data (if --json)
```

### Example Workflows
```bash
# Quick API documentation
repo2txt <repo> --ai-select --ai-query "Select all API route files and documentation"

# Architecture overview
repo2txt <repo> --ai-select --ai-query "Show me the main architecture components"

# Focus on core code (skip datasets)
repo2txt <repo> --exclude-dirs "datasets,logs,cache" --ai-select
```

## 🏗️ Architecture

### Core Components
```
repo2txt/
├── adapters/           # Repository adapters (GitHub, local, etc.)
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
- **Modular Architecture**: Clean, testable components
- **Survival-Oriented**: Graceful degradation when components fail
- **Performance First**: Token caching, efficient processing
- **Developer Experience**: Rich debugging, comprehensive testing

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
result = analyzer.analyse("/path/to/repo")

# AI-assisted analysis
config.ai_select = True
config.ai_query = "Select all Python files related to data processing"
result = analyzer.analyse("/path/to/repo")
```

## 🐛 Troubleshooting

### Common Issues

**AI Selection Not Working**
- Verify LLM_API_KEY is set correctly
- Check LLM_BASE_URL for custom endpoints
- Use `--debug` to see system prompts and API calls

**Performance Issues**
- Use `--exclude-dirs` to skip large directories
- Set `--max-file-size` to limit individual files
- Consider `--token-budget` for AI selection

**GitHub Issues**
- Set `GITHUB_TOKEN` for private repos and rate limiting
- Use personal access token with appropriate scopes

**Large Repository Performance**
- Use `--max-file-size` to limit individual files
- Set appropriate `--token-budget` for AI selection
- Consider analysing specific subdirectories
- Use `--exclude-dirs` to skip large dataset/cache directories
- Models less performant than gpt-4.1-mini start to hallucinate when given large file trees (> 250 files)

**Directory Exclusions**
- By default, excludes: `__pycache__`, `.git`, `node_modules`, `venv`, `datasets`, etc.
- Add custom exclusions: `--exclude-dirs "logs,temp,cache,data"`
- Exclusions apply to both local and GitHub repository analysis
- **Missing directories?** Check if they're excluded by default (see `excluded_dirs` in `src/repo2txt/core/models.py`)

## 🧪 Testing
```bash
pytest                              # Run all tests
pytest tests/test_ai_components.py  # AI system tests
pytest --cov=repo2txt               # With coverage
```

## 🤝 Contributing

1. **Fork & Branch**: Create feature branches from `main`
2. **Test**: Ensure all tests pass with `pytest`
3. **Document**: Update README for user-facing changes
4. **Pull Request**: Submit with clear description

## 📄 Licence

MIT Licence - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- **Original Concept**: [Doriandarko/RepoToTextForLLMs](https://github.com/Doriandarko/RepoToTextForLLMs)
- **Enhanced Architecture**: Redesigned with improved UX and AI assistance
