"""Build the web piece from docs/web/page.html and a data file.

    uv run python -m bench.site --data docs/web/data.json

Writes docs/index.html, a standalone page for GitHub Pages, and docs/web/artifact.html, the
same page without its document wrapper.
"""

import argparse
from pathlib import Path

WRAPPER = """<!doctype html>
<html lang="en-GB">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
{head}
</head>
<body>
{body}
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="docs/web/data.json")
    parser.add_argument("--page", default="docs/web/page.html")
    parser.add_argument("--out", default="docs")
    args = parser.parse_args()
    page = Path(args.page).read_text(encoding="utf-8")
    data = Path(args.data).read_text(encoding="utf-8").replace("</", "<\\/")
    filled = page.replace("__DATA__", data)
    out = Path(args.out)
    (out / "web").mkdir(parents=True, exist_ok=True)
    (out / "web" / "artifact.html").write_text(filled, encoding="utf-8")
    head, _, body = filled.partition('<div class="page">')
    (out / "index.html").write_text(
        WRAPPER.format(head=head.strip(), body='<div class="page">' + body), encoding="utf-8"
    )
    print(out / "index.html", out / "web" / "artifact.html")


if __name__ == "__main__":
    main()
