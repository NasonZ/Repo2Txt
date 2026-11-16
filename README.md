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
uv tool install git+https://github.com/NasonZ/Repo2Txt
```

Python 3.11 or later. `map`, `select` and `pack` need nothing more. Choosing with a model
needs the `llm` extra and the MCP server the `mcp` extra:
`uv tool install "repo2txt[llm,mcp] @ git+https://github.com/NasonZ/Repo2Txt"`.

## Commands

`map`, `select` and `pack` take `-r/--repo` (a directory, a GitHub URL or `owner/repo`; default `.`),
`--ref` (a GitHub branch, tag or commit), `-i/--include` and `-e/--exclude` globs, and
`--json`. Budgets are written `8000`, `8k` or `1.5m`. Output goes to stdout.

**`repo2txt map [DIRECTORY]`** prints the tree with token counts. If the whole tree does not
fit `--max-tokens` (default 4k) it is shown to the deepest level that does, and directories
below that are one line each with their totals. `--depth N` fixes the level.

**`repo2txt select QUERY`** ranks files for a question and lists those that fit `-b/--budget`
(default 32k), best first, as `path<TAB>tokens`. It prints no contents. The count is what the
file takes when packed, its heading included, and 60 tokens of the budget are left for the
packed document's own heading, so a selection always packs within the same budget. `--method bm25` (the
default) scores paths, identifiers and text and needs no model. `--method llm` gives a model
the tree and lets it choose; see below. With `--method llm`, `--lines` lists line ranges
(`path:first-last`) as well as whole files.

**`repo2txt pack [ITEMS...]`** prints the contents of files as Markdown (or `-f xml`). An item
is a path, a directory, a glob, or `path:10-80` for a line range. Files are included whole in
the order given until `-b/--budget` is full. The budget covers everything printed, headings
and notes included. What does not fit is listed with its token cost and is never cut. With
`-f xml` the files sit inside XML-style tags and their contents are left exactly as they
are, so the result is for a model to read, not for an XML parser. With no items the whole repository is packed, smallest files first.
`--stdin` reads items from a pipe, and fails if the pipe names none; `-o FILE` writes the
document and prints a summary.

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
source maps, binary formats, files with a null byte in their first 8 KB and files over 1 MB
are left out.
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

The three tools are read-only and return the same text as the commands. The server reads
local directories only inside the directory it was started in, and any GitHub repository. Their default cap is
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

The model sees the tree with token counts, collapsed to about 6,000 tokens, and the 30 files
the lexical ranking puts first for the request. It has three tools: replace the selection,
adjust it, and list a collapsed directory. It cannot read file
contents. Every path is checked against the repository, and a change that would exceed the
budget is refused and explained, so the result always fits. It gets five rounds of tool calls
per request. These variables can also go in a `.env` file.

At small budgets the needed files often do not fit whole. `--lines` adds a second step: the
model first chooses files at four times the budget, then reads an outline of each (every
block with its line range and size) and picks the ranges to include. What budget is left
goes to its chosen files whole. The output is `path:first-last` items, so
`select --method llm --lines ... | pack --stdin` packs them. It costs one more model call.

## How well it selects

Measured on [ContextBench](https://huggingface.co/datasets/Contextbench/ContextBench): real
issues from open-source repositories, each with the code a person traced out from the fix
and checked was enough to write one. A method is given the issue text and the repository at
that commit. Recall is the share of the needed files it delivered within the budget;
precision, in brackets, the share of what it delivered that was needed.

The methods were tuned on other tasks. These 254 issues from 21 repositories were run once,
after the configuration was frozen and the tests written down
([pre-registration](bench/preregistration-2026-11-15.md)).

| Method | 8k tokens | 32k tokens | 128k tokens |
|---|---|---|---|
| Smallest files first | 0.01 | 0.04 | 0.12 |
| BM25 alone | 0.09 (0.08) | 0.24 (0.10) | 0.44 (0.07) |
| `select` (BM25 with three added signals) | 0.19 (0.12) | 0.40 (0.14) | 0.56 (0.09) |
| `select --method llm`, Claude Haiku 5.5 | 0.34 (0.42) | **0.59** (0.41) | |
| `select --method llm`, Claude Sonnet 5.5 | 0.33 (0.56) | 0.55 (0.58) | |
| `select --method llm`, DeepSeek flash | 0.35 (0.35) | 0.60 (0.27) | |
| The needed files themselves | 0.61 | 0.89 | 0.99 |

Compared issue by issue at 32k, with 95% intervals from resampling whole repositories, the
model selector beats `select` by 0.19 with Haiku (0.14 to 0.24), 0.15 with Sonnet (0.11 to
0.20) and 0.20 with DeepSeek (0.15 to 0.24); `select` beats BM25 alone by 0.16 (0.10 to
0.22). All four were the tests written down beforehand, and all four held. The order of
methods is the same when "needed" means the files the fix edited, and where SWE-Explore says
which lines successful agents read.

**Which model.** Haiku matches DeepSeek and beats Sonnet on recall, at a sixth and a
fourteenth of their cost: 255 selections at 32k cost $0.47, $3.00 and $6.72. Sonnet picks
fewer files, more of them needed, and misses more; DeepSeek takes about four rounds of tool
calls where Haiku takes under three, and six times as long. For this step a larger model did
not help.

**Line ranges.** At 8k the needed files often do not fit whole. `select --method llm --lines`
has the model pick line ranges from outlines of the files it chose; it raises the share of
needed lines delivered from 0.38 to 0.48 (0.04 to 0.17 better), against a ceiling of 0.92 for
the lines alone.

**On another benchmark.** [Agent Retrieval Bench](https://arxiv.org/abs/2607.24882) asks for
the files a coding agent needs next, from a failing trace, a review comment or a pull
request. Scored by its own evaluator, the model selector's 8k selection reaches a Budgeted
Context Yield of 0.63 with Haiku, against 0.38 for the benchmark's repo map, 0.37 for its best
embedding model and 0.53 for the best community result. The lexical default does not transfer
as well (0.27): one of its signals ranks tests last, which is wrong when the task is to find
tests.

**How sure.** Four repeat runs of the model selector moved its mean recall by 0.003, so a
single run is enough for the averages; a quarter of single issues change between runs, so
one issue's result is not. The needed files are one sufficient set, not the only one, so
precision is probably understated. Whether the models had seen these repositories in
training was not measured.

[docs/report-2026-11-16.md](docs/report-2026-11-16.md) is the full account: what "needed"
means, what the harness checks, the outside benchmarks, the models and the limits.
[bench/README.md](bench/README.md) explains how to reproduce every number, and
[docs/report-2026-11-14.md](docs/report-2026-11-14.md) is the earlier tuning.

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

git is run with `core.fsmonitor` and hooks disabled, so a repository's own configuration
cannot start a program when it is read.

The tests run the commands on a small repository built in a temporary directory, read GitHub
through a stand-in transport that serves a generated tarball, drive the selector with scripted
model replies, and call the MCP server through the SDK's in-memory client. One further test
reads a real repository and needs network access and `GITHUB_TOKEN`. No test calls a model.

## Licence

MIT. See [LICENSE](LICENSE). The project began as a fork of
[RepoToTextForLLMs](https://github.com/Doriandarko/RepoToTextForLLMs) by Pietro Schirano.
