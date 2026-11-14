# Repo2Txt

Token-aware repository context selection for LLMs and coding agents.

Repo2Txt reads a local directory or a GitHub repository, counts the tokens in every text file, and writes the files you want as one Markdown or XML document. Files can be chosen by a token budget alone, or by an LLM that you talk to.

## Install

Python 3.11 or later.

```bash
git clone https://github.com/NasonZ/Repo2Txt.git
cd Repo2Txt
uv sync            # or: pip install -e .
```

## Use

```bash
repo2txt .                                   # every text file in the current directory
repo2txt owner/repo                          # a GitHub repository
repo2txt https://github.com/owner/repo --branch dev
repo2txt . --token-budget 50000              # drop the largest files until the total fits
repo2txt . --format xml --json               # XML output, plus token counts as JSON
repo2txt . --exclude-dirs "fixtures,logs"    # skip more directories
```

Results go to `output/<name>_<timestamp>/`:

```text
<name>_analysis.md          the repository structure and the selected files
<name>_token_report.txt     tokens per file, largest first
<name>_tokens.json          only with --json
```

In the Markdown output each file is a fenced block whose info string is its path. With `--format xml` each file is a `<file path="...">` element and the file is named `<name>_analysis.xml`.

### What is included

Hidden files and directories (except `.github` and `.gitlab`), directories such as `node_modules`, `venv`, `dist` and `build`, lockfiles, minified files, source maps, files with binary extensions, files that contain null bytes, and files larger than `--max-file-size` (1 MiB by default) are skipped. Symbolic links are not followed.

### Token budget

Without `--ai-select`, `--token-budget N` removes the largest files, one at a time, until the total is at most N, and prints what it removed. Tokens are counted with tiktoken's `cl100k_base` encoding, so counts are an estimate for models that use a different tokeniser. tiktoken downloads that encoding the first time it is used.

### GitHub

Public repositories work without credentials, within GitHub's unauthenticated rate limit. Set `GITHUB_TOKEN` for private repositories and a higher limit. A repository whose tree is too large for the GitHub API to list in one response is rejected.

## LLM-assisted selection

```bash
export LLM_MODEL=<model name>
export LLM_API_KEY=...                       # not needed if LLM_BASE_URL points at a local server
export LLM_BASE_URL=http://localhost:8080/v1 # optional; any OpenAI-compatible endpoint

repo2txt . --ai-select                       # chat, then /generate
repo2txt . --ai-select --ai-query "how does authentication work" --token-budget 30000
```

These variables, and `GITHUB_TOKEN`, can also go in a `.env` file; see `.env.example`. There is no default model, so `LLM_MODEL` must be set.

The model is given the repository tree with token counts and two tools, `select_files` and `adjust_selection`. It cannot read or edit files. Every path it proposes is checked against the repository; unknown paths are reported back to the model and nothing invalid enters the selection. The budget defaults to 100,000 tokens (`--token-budget`) and is shown, not enforced: the model is told the budget and the running total.

With `--ai-query` the model makes one selection and the output is written straight away. Without it you chat, and write the output with `/generate`.

| Command | Effect |
| --- | --- |
| `/generate [markdown\|xml] [json]` | Write the selected files to the output directory |
| `/undo` | Restore the state before your last message |
| `/redo` | Regenerate the last reply |
| `/clear` | Reset the conversation and the selection |
| `/save [name]` | Save the conversation as JSON |
| `/toggle streaming` | Switch streamed replies on or off |
| `/toggle prompt` | Cycle the prompt style |
| `/toggle budget <n>` | Set the token budget |
| `/debug` | Show or hide prompts and tool calls |
| `/help`, `/quit` | Help; leave the chat |

`--prompt-style` chooses between three system prompts: `standard`, `meta-reasoning` and `xml`. `--debug` shows the system prompt and every tool call.

## Options

```text
repo2txt REPO [OPTIONS]

  -o, --output-dir DIR      Where to write results (default: output)
  -f, --format FORMAT       markdown or xml (default: markdown)
  --token-budget N          Token budget
  -m, --max-file-size N     Skip files larger than N bytes (default: 1048576)
  --exclude-dirs LIST       Comma-separated extra directory names to skip
  --branch NAME             GitHub branch (default: the repository's default)
  --json                    Also write token counts as JSON
  --ai-select               Select files with an LLM
  --ai-query TEXT           With --ai-select, one-shot selection without chat
  --prompt-style STYLE      standard, meta-reasoning or xml
  --debug                   Show prompts and tool calls
  --version, --help
```

## How it is put together

```text
adapters/   LocalSource and GitHubSource list candidate files and fetch their bytes
core/       filtering, token counting, budget selection, Markdown/XML/JSON output
ai/         the chat session, its commands, prompts, tools and selection state
ui.py       terminal output
cli.py      the command
```

A source lists candidate files, reads them concurrently, and returns a `Repository`: the text and token count of every file. Selection is a list of paths into that repository, chosen either by `select_within_budget` or by the AI session. Output is a function of the repository and the selected paths, so both routes share it.

GitHub is read through the git tree endpoint, which lists every file with its size in one request, and the blob endpoint for contents. Nothing is cloned.

## Development

```bash
uv run pytest
uv run ruff check
uv run ruff format
```

The end-to-end tests build a small repository in a temporary directory and run the command on it. The GitHub test needs network access and `GITHUB_TOKEN` and is skipped otherwise. The AI session needs a model and is not covered by the tests.

## Licence

MIT. See [LICENSE](LICENSE). The project began as a fork of [RepoToTextForLLMs](https://github.com/Doriandarko/RepoToTextForLLMs) by Pietro Schirano, which is also MIT licensed.
