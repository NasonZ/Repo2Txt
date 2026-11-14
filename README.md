# Repo2Txt

Choose what an LLM or a coding agent reads from a repository, within a token budget.

A model reading code has a fixed amount of room, and most of a repository is not about the
question in hand. Repo2Txt answers three questions, each with a cap on how much it prints:
what is in this repository and how big is each part, which files matter for this question,
and what is in these files.

```console
$ repo2txt map -r pallets/click --max-tokens 300
# click ref=main: 169 files, 327,719 tokens (o200k_base)
.github/ 3,438 (10 files)
docs/ 61,020 (42 files)
examples/ 8,716 (36 files)
src/ 102,393 (18 files)
tests/ 130,218 (59 files)
CHANGES.md 19,683
LICENSE.txt 296
README.md 403
pyproject.toml 1,552
# depth 1 of 5: deeper directories are collapsed
# next: repo2txt map tests/ -r pallets/click
# next: repo2txt map src/ -r pallets/click
# next: repo2txt map docs/ -r pallets/click

$ repo2txt select "ParamType convert fail" -r pallets/click -b 30k
# click: 18 files, 29,987 of 30,000 tokens, from 97 candidates (bm25)
src/click/types.py	10953
docs/parameter-types.md	2269
examples/validation/validation.py	326
...

$ repo2txt select "ParamType convert fail" -r pallets/click -b 12k | repo2txt pack --stdin -r pallets/click -b 12k
```

Output from 14 November 2026; the last command prints the files themselves.

## Install

```bash
uv tool install git+https://github.com/NasonZ/Repo2Txt     # or: pip install git+https://...
```

Python 3.11 or later.

## Commands

Every command takes `-r/--repo` (a directory, a GitHub URL or `owner/repo`; default `.`),
`--ref` (a GitHub branch, tag or commit), `-i/--include` and `-e/--exclude` globs, and
`--json`. Budgets are written `8000`, `8k` or `1.5m`. Output goes to stdout.

**`repo2txt map [DIRECTORY]`** prints the tree with token counts. If the whole tree does not
fit `--max-tokens` (default 4k) it is shown to the deepest level that does, and directories
below that are one line each with their totals. `--depth N` fixes the level.

**`repo2txt select QUERY`** ranks files for a question and lists those that fit `-b/--budget`
(default 32k), best first, as `path<TAB>tokens`. It prints no contents. `--method bm25` (the
default) scores paths, identifiers and text and needs no model. `--method llm` gives a model
the tree and lets it choose; see below.

**`repo2txt pack [ITEMS...]`** prints the contents of files as Markdown (or `-f xml`). An item
is a path, a directory, a glob, or `path:10-80` for a line range. Files are included whole in
the order given until `-b/--budget` is full. What does not fit is listed with its token cost
and is never cut. With no items the whole repository is packed, smallest files first.
`--stdin` reads items from a pipe; `-o FILE` writes the document and prints a summary.

**`repo2txt chat`** is a terminal conversation with a model that builds a selection:
`/generate [FILE]`, `/undo`, `/clear`, `/budget N`, `/quit`.

**`repo2txt mcp`** serves `repo_map`, `repo_select` and `repo_pack` over MCP (below).

Failures print `error: <what went wrong>` and, where there is one, `next: <a command to try>`.
Exit status is 0 on success, 1 when the work failed and 2 when the command was misused.

### What is read

In a git work tree the file list comes from `git ls-files`: tracked files, and untracked files
that git does not ignore. A GitHub repository is read from its tarball in two requests, so
public repositories need no token; set `GITHUB_TOKEN` for private ones. Hidden files and
directories (except `.github` and `.gitlab`), `node_modules`, lockfiles, minified files,
source maps, binary formats, files containing null bytes and files over 1 MB are left out.
Symbolic links are not followed.

Tokens are counted with tiktoken's `o200k_base`. Other tokenisers differ, usually by under a
fifth, so treat counts as estimates and leave headroom.

## For coding agents

**Skill.** `skills/repo2txt/SKILL.md` follows the [Agent Skills](https://agentskills.io)
format, which Claude Code, Codex and pi all read. Copy the directory to `~/.claude/skills/`
for Claude Code or `~/.agents/skills/` for Codex and pi.

**MCP.** Install with the extra and register the server:

```bash
uv tool install "repo2txt[mcp] @ git+https://github.com/NasonZ/Repo2Txt"
claude mcp add repo2txt -- repo2txt mcp
```

The three tools are read-only and return the same text as the commands. Their default cap is
4,000 tokens, under the size at which Claude Code warns (10,000) and pi truncates (20 KB).

The command line is the main interface: a skill costs an agent a few hundred tokens of
context, and tool definitions cost more.

## Choosing with a model

```bash
export LLM_MODEL=<model name>
export LLM_API_KEY=...                        # not needed if LLM_BASE_URL is a local server
export LLM_BASE_URL=http://localhost:8080/v1  # optional; any OpenAI-compatible endpoint

repo2txt select "how are sessions invalidated on logout" --method llm -b 30k
```

The model sees the tree with token counts, collapsed to about 6,000 tokens, and has three
tools: replace the selection, adjust it, and list a collapsed directory. It cannot read file
contents. Every path is checked against the repository, and a change that would exceed the
budget is refused and explained, so the result always fits. It gets five rounds of tool calls
per request. These variables can also go in a `.env` file.

## How it is put together

```text
sources.py     a directory or a GitHub tarball, as (path, bytes) pairs, filtered
models.py      Repository and RepoFile; tokens are counted per file on first use
selection.py   Selection, a set of paths held within a budget; fit() for ranked lists
tree.py        the tree with totals, and its entries down to a depth
rank.py        BM25 over paths, identifiers and text
api.py         repo_map, repo_select, repo_pack: everything else calls only these
render.py      text and JSON that fit a cap; hints are dropped before items are
llm.py         Complete, the one function type through which a model is called
selector.py    the model's prompt, tools and loop over a Selection
cli.py  chat.py  mcp_server.py   three ways in
```

`llm.Complete` takes a conversation and tool schemas and returns text and tool calls. The
default calls an OpenAI-compatible endpoint; any other client can be passed in its place.

## Development

```bash
uv run pytest
uv run ruff check && uv run ruff format --check
```

The tests run the commands on a small repository built in a temporary directory, read GitHub
through a stand-in transport that serves a generated tarball, drive the selector with scripted
model replies, and call the MCP server through the SDK's in-memory client. One further test
reads a real repository and needs network access and `GITHUB_TOKEN`. No test calls a model.

## Licence

MIT. See [LICENSE](LICENSE). The project began as a fork of
[RepoToTextForLLMs](https://github.com/Doriandarko/RepoToTextForLLMs) by Pietro Schirano.
