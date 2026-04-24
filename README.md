# Repo2Txt

Repo2Txt turns a repository into bounded, token-aware context for LLMs.

It reads a local directory or GitHub repository, lets you choose the files that matter, counts the tokens, and writes a single context file in Markdown or XML. File selection can be manual or assisted by an LLM.

The useful abstraction is not "repo to text". It is context selection under a budget.

## Why this exists

LLM work on codebases is usually limited by context, not by access to files. A repository can have hundreds or thousands of files. The useful question is rarely "can I dump the repo?" It is:

- Which files should be in context for this task?
- How many tokens will they consume?
- What can be removed without losing the evidence needed for the next step?
- How do I keep the selection explicit enough to inspect and revise?

Repo2Txt makes that selection visible. It builds a file tree, tracks token counts, validates selected paths, and produces an output file that can be given to another model.

This also makes it a natural base for agent context management. Current agent workflows increasingly need tools that can load, retain, and discard context rather than blindly append more text. Repo2Txt exposes those primitives today through a CLI; an MCP server is the obvious next interface.

## Current status

Implemented:

- Local repository input
- GitHub repository input
- Manual interactive file selection
- AI-assisted file selection with tool calls
- Token counting and token budgets
- Markdown and XML output
- JSON token export
- Rich terminal UI themes
- Undo during AI-assisted selection

Not implemented yet:

- MCP server interface
- Read/write code-editing tools
- Custom output templates

## Installation

Repo2Txt requires Python 3.9 or newer. Python 3.11+ is recommended.

With `uv`:

```bash
git clone https://github.com/NasonZ/repo2txt.git
cd repo2txt
uv sync
source .venv/bin/activate
```

With `pip`:

```bash
git clone https://github.com/NasonZ/repo2txt.git
cd repo2txt
pip install -e .
```

For development:

```bash
uv sync --dev
source .venv/bin/activate
pytest
```

## Quick start

Select files manually from a local repository:

```bash
repo2txt /path/to/repo
```

Select files manually from GitHub:

```bash
repo2txt https://github.com/owner/repo
```

Ask an LLM to select files for a task:

```bash
repo2txt /path/to/repo \
  --ai-select \
  --ai-query "Show me the authentication flow, including routes, models, middleware, and tests" \
  --token-budget 50000
```

Write XML instead of Markdown:

```bash
repo2txt /path/to/repo --format xml
```

Export token data as JSON as well:

```bash
repo2txt /path/to/repo --json
```

Exclude large or irrelevant directories:

```bash
repo2txt /path/to/repo --exclude-dirs "datasets,logs,cache"
```

## Selection modes

### Manual selection

Manual mode lets you walk the repository tree and choose files or directories yourself.

Commands during traversal:

```text
1-5,7,9-12  select item ranges
a            select all items in the current directory
s            skip the current directory
b            go back
q            quit
```

Use this mode when you know the codebase or want full control over what enters context.

### AI-assisted selection

AI-assisted mode gives the model the repository tree, token counts, README content, and a small set of file-selection tools. The model does not edit files. It only proposes and adjusts a context set.

The two core tools are:

```text
select_files       replace the current selection with a new set of files
adjust_selection  add or remove files from the current selection
```

Every selected path is validated against the real repository. If the model invents a path, uses the wrong prefix, or selects a file that does not exist, Repo2Txt returns structured feedback and keeps the session alive.

Example:

```bash
repo2txt ./service \
  --ai-select \
  --prompt-style meta-reasoning \
  --token-budget 25000
```

Then ask:

```text
How does payment processing work? Include the main components and error handling.
```

The agent can select files such as:

```text
src/payments/processor.py
src/payments/models.py
src/payments/exceptions.py
src/api/payment_routes.py
tests/test_payments.py
config/payment_config.py
```

Repo2Txt reports the token impact immediately, so the selection can be revised before generating output.

## AI chat commands

During AI-assisted selection:

```text
/help                 show available commands
/generate [format]    create output: markdown, xml, json, or all
/save [filename]      save chat history
/clear                reset conversation and selection
/undo                 undo last selection action
/toggle streaming     enable or disable streaming responses
/toggle thinking      enable or disable thinking mode for Qwen models
/toggle prompt        cycle prompt style
/toggle budget <N>    set token budget
/debug                toggle debug mode
/debug state          show current state
```

In AI mode, output is generated from inside the chat with `/generate`. In manual mode, output is written after selection completes.

## Command-line options

```text
repo2txt [REPO] [OPTIONS]

Core options:
  --output-dir, -o DIR     Output directory. Default: output
  --format, -f FORMAT      Output format: markdown or xml. Default: markdown
  --theme, -t THEME        Terminal theme: manhattan, green, matrix, sunset
  --max-file-size SIZE     Maximum file size in bytes. Default: 1MB
  --exclude-dirs DIRS      Comma-separated additional directories to exclude
  --no-tokens              Disable token counting
  --json                   Export token data as JSON

AI options:
  --ai-select              Enable AI-assisted selection
  --ai-query QUERY         Initial query for AI selection
  --token-budget N         Token budget for AI selection. Default: 100000
  --prompt-style STYLE     standard, meta-reasoning, or xml
  --debug                  Show prompts, tool calls, and debug panels
```

## Configuration

Create a `.env` file when using GitHub private repositories or AI-assisted selection:

```bash
# LLM configuration
LLM_PROVIDER=openai        # openai, ollama, llamacpp
LLM_MODEL=gpt-4-turbo
LLM_API_KEY=...
LLM_BASE_URL=              # optional custom endpoint

# GitHub access
GITHUB_TOKEN=...           # required for private repos; useful for rate limits

# UI
DEFAULT_THEME=manhattan    # manhattan, matrix, green, sunset
```

For local models:

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

## Output

Repo2Txt creates a timestamped output directory:

```text
output/
└── RepoName_20240315_143022/
    ├── RepoName_analysis.md       # selected files plus instructions
    ├── RepoName_token_report.txt  # token tree, totals, largest files
    └── RepoName_tokens.json       # optional, with --json
```

The main analysis file includes:

- repository name and branch
- analysis instructions
- selected repository structure
- total selected files
- total selected tokens
- README content
- selected file contents
- errors encountered during processing, if any

Markdown output uses fenced blocks with file paths:

````markdown
```src/example.py
print("hello")
```
````

XML output uses explicit file elements:

```xml
<file path="src/example.py">
print("hello")
</file>
```

## Architecture

Repo2Txt has three layers.

### Repository adapters

Adapters normalise different repository sources behind one interface.

- `LocalAdapter` reads directly from the filesystem and counts tokens on demand.
- `GitHubAdapter` works through the GitHub API and pre-caches repository structure to avoid repeated network calls.

The two adapters expose the same behaviour to the rest of the system, but they should not be collapsed into one implementation. Filesystem traversal and GitHub API traversal have different performance constraints.

### Selection state

Selection state records:

- available files
- selected files
- token count per file
- total selected tokens
- token budget
- previous state for undo

This is the centre of the AI-assisted workflow. The LLM can request changes, but the state manager validates them.

### Output pipeline

Both manual and AI-assisted selection converge on the same pipeline:

```text
repository source
  -> file tree
  -> selected paths
  -> file contents
  -> token report
  -> markdown/xml/json output
```

That convergence is deliberate. Selection is separate from formatting.

## Design notes

### Context is a managed resource

Long context windows do not remove the need to choose. More context can add cost, latency, and distractors. Repo2Txt keeps token pressure visible while selection is happening, not after the output is already built.

### Search and generation are separate jobs

The AI-assisted selector does not answer the user's code question. It selects the evidence needed to answer it. That boundary matters: one component curates context, another reasons over it.

### Invalid AI actions are data

LLMs make path mistakes. Repo2Txt treats those mistakes as recoverable tool errors. Invalid paths are returned with explicit feedback, allowing the model to correct its next action instead of silently producing a bad context set.

## Development

Run tests:

```bash
pytest
```

Run a focused test file:

```bash
pytest tests/test_ai_components.py
```

Run quality checks:

```bash
black src/ tests/
ruff check src/ tests/
mypy src/
```

Build the package:

```bash
python -m build
```

Useful implementation docs:

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- [`docs/SYSTEM_DESIGN.md`](docs/SYSTEM_DESIGN.md)
- [`CLAUDE.md`](CLAUDE.md)

## Acknowledgements

Repo2Txt builds on the original idea from [Doriandarko/RepoToTextForLLMs](https://github.com/Doriandarko/RepoToTextForLLMs).

## Licence

MIT Licence. See [`LICENSE`](LICENSE).
