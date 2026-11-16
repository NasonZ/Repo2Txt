"""Reading results files, plain or compressed.

Runs append plain JSONL as they go, so a stopped run can resume. Large finished files are kept
in the repository xz-compressed (`.jsonl.xz`); everything that reads results accepts either.

    xz -9 -k bench/results/v3/fresh-reference.jsonl    # compress a finished file
"""

import json
import lzma
from pathlib import Path


def open_text(path: str | Path):
    path = Path(path)
    if path.suffix == ".xz":
        return lzma.open(path, "rt")
    return path.open()


def read_rows(path: str | Path) -> list[dict]:
    with open_text(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]
