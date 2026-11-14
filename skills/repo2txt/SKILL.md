---
name: repo2txt
description: Find and read the right files of a repository within a token budget. Use when exploring an unfamiliar or large codebase, when a question needs files you have not located yet, when you must hand a subagent or another model a bounded bundle of code, or when a repository lives on GitHub and is not checked out locally.
license: MIT
compatibility: Needs the repo2txt command (uv tool install git+https://github.com/NasonZ/Repo2Txt). Reading from GitHub needs network access.
---

# repo2txt

Three commands, each with a hard cap on what it prints. Use them in this order, and stop as
soon as you have what you need.

```bash
repo2txt map                          # the tree, with token counts per file and directory
repo2txt select "QUERY" -b 16k        # files ranked for a question, fitted to a budget
repo2txt pack PATH... -b 16k          # the contents of named files, fitted to a budget
```

Add `-r DIR`, `-r owner/repo` or `-r https://github.com/owner/repo` to work on a repository
other than the current directory; `--ref` picks a branch, tag or commit. Add `--json` for JSON.

## map

Shows sizes before you read anything. Large trees are collapsed to fit `--max-tokens`
(default 4k): a line such as `src/ 120,331 (88 files)` is a directory you can open with
`repo2txt map src/`. Use it to decide what is affordable, then use your own search tools.

## select

Ranking is lexical (BM25 over paths, identifiers and text). Put the function names, class
names, error strings or domain terms you expect in the code into the query; a description
of behaviour alone ranks poorly. It prints `path<TAB>tokens`, best first, and never contents.
Files that ranked well but did not fit are listed as `# did not fit`.

`--method llm` asks the model named by `LLM_MODEL` to choose from the tree instead. It is
slower and costs a model call; use it when the query cannot name anything in the code.

## pack

Items are paths, directories (`src/auth/`), globs (`*.sql`) or line ranges
(`src/app.py:120-260`). Files are included whole, in the order given, until the budget is
full. Anything left out is listed with its token cost and is never cut, so if a file you
need is omitted, ask for a line range or a larger budget. Pipe a selection straight in:

```bash
repo2txt select "retry backoff HttpClient" -b 12k | repo2txt pack --stdin -b 12k
```

`-o FILE` writes the document to a file and prints only the summary, which is the way to
hand a bundle to a subagent without spending your own context on it.

## Reading the output

Lines starting `#` are notes, not data. `# next:` lines are commands worth running.
`error:` on the first line means the command failed; the `next:` line after it says what to
try. Exit status is 0 on success, 1 on failure and 2 when the command was misused.
