"""Draw the report's figures from the results files, as SVG in a light and a dark version.

    uv run python -m bench.charts

Writes to docs/img/. Each figure is computed from bench/results/*.jsonl, so it changes only
when the results do.
"""

import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from xml.sax.saxutils import escape

RESULTS = Path("bench/results")
OUT = Path("docs/img")

# Validated with the dataviz skill's palette checker: slots 1-3 of its reference palette.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e4e3df",
        "blue": "#2a78d6",
        "orange": "#eb6834",
        "aqua": "#1baf7a",
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#f0efec",
        "muted": "#c3c2b7",
        "grid": "#33332f",
        "blue": "#3987e5",
        "orange": "#d95926",
        "aqua": "#199e70",
    },
}
FONT = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
BUDGETS = [8000, 32000, 128000]


def rows(*names: str) -> list[dict]:
    return [json.loads(line) for n in names for line in (RESULTS / f"{n}.jsonl").open()]


def paired(data: list[dict], a: str, b: str, budget: int) -> tuple[float, float, float, int]:
    """Mean of (a - b) recall over shared tasks, with a 95% interval resampling repositories."""
    by = {(r["method"], r["task"]): r for r in data if r["budget"] == budget}
    tasks = sorted({t for m, t in by if m == a} & {t for m, t in by if m == b})
    groups: dict[str, list[float]] = defaultdict(list)
    for t in tasks:
        groups[by[(a, t)]["repo"]].append(by[(a, t)]["recall"] - by[(b, t)]["recall"])
    pools, rng = list(groups.values()), random.Random(0)
    means = sorted(
        mean(v for pool in rng.choices(pools, k=len(pools)) for v in pool) for _ in range(4000)
    )
    return mean(v for pool in pools for v in pool), means[100], means[3899], len(tasks)


def text(x, y, s, fill, size=13, anchor="start", weight=400) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}">{escape(s)}</text>'
    )


def svg(width: int, height: int, body: list[str], c: dict, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="{FONT}" role="img" '
        f'aria-label="{escape(title)}">'
        f'<rect width="{width}" height="{height}" rx="10" fill="{c["surface"]}"/>'
        + "".join(body)
        + "</svg>\n"
    )


def recall_by_budget(c: dict) -> str:
    """Recall at each budget for the methods that matter, on the same 100 tasks."""
    data = rows("dev-free", "dev-llm", "dev-tune", "dev-v2-budgets")
    tasks = {r["task"] for r in data if r["method"] == "llm-v2" and r["budget"] == 32000}
    series = [
        ("oracle", "Ceiling: the needed files", c["muted"], True),
        ("llm-v2", "Model selector, kept", c["blue"], False),
        ("llm", "Model selector, first version", c["aqua"], False),
        ("bm25", "Lexical default", c["orange"], False),
    ]
    points = {
        m: [
            mean(
                r["recall"]
                for r in data
                if r["method"] == m and r["budget"] == b and r["task"] in tasks
            )
            for b in BUDGETS
        ]
        for m, *_ in series
    }

    width, height = 760, 430
    left, right, top, bottom = 64, 250, 92, 58
    plot_w, plot_h = width - left - right, height - top - bottom

    def x(i: int) -> float:
        return left + plot_w * i / (len(BUDGETS) - 1)

    def y(v: float) -> float:
        return top + plot_h * (1 - v)

    body = [
        text(
            24,
            36,
            "The kept model selector finds 0.68 of the needed files at 32k tokens",
            c["text"],
            17,
            weight=600,
        ),
        text(
            24,
            60,
            "Share of the needed files that fit in the budget. Same 100 tasks for every line.",
            c["muted"],
            13,
        ),
    ]
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        body.append(
            f'<line x1="{left}" x2="{left + plot_w}" y1="{y(v):.1f}" y2="{y(v):.1f}" '
            f'stroke="{c["grid"]}" stroke-width="1"/>'
        )
        body.append(
            text(
                left - 10,
                y(v) + 4,
                f"{v:.2f}".rstrip("0").rstrip(".") or "0",
                c["muted"],
                12,
                "end",
            )
        )
    for i, b in enumerate(BUDGETS):
        body.append(text(x(i), top + plot_h + 24, f"{b // 1000}k tokens", c["muted"], 12, "middle"))

    labels = []
    for m, name, colour, dashed in series:
        coords = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(points[m]))
        dash = ' stroke-dasharray="6 5"' if dashed else ""
        body.append(
            f'<polyline points="{coords}" fill="none" stroke="{colour}" '
            f'stroke-width="2"{dash} stroke-linejoin="round"/>'
        )
        for i, v in enumerate(points[m]):
            body.append(
                f'<circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="4.5" fill="{colour}" '
                f'stroke="{c["surface"]}" stroke-width="2"/>'
            )
        labels.append([y(points[m][-1]), name, points[m][-1], colour])

    # Direct labels at the right end, pushed apart where lines finish close together.
    labels.sort()
    for i in range(1, len(labels)):
        labels[i][0] = max(labels[i][0], labels[i - 1][0] + 34)
    for ly, name, v, colour in labels:
        lx = x(len(BUDGETS) - 1) + 14
        body.append(
            f'<rect x="{lx:.1f}" y="{ly - 7:.1f}" width="10" height="3" rx="1.5" fill="{colour}"/>'
        )
        body.append(text(lx + 16, ly - 1, name, c["text"], 13))
        body.append(text(lx + 16, ly + 15, f"{v:.2f} at 128k", c["muted"], 12))
    return svg(width, height, body, c, "Recall against token budget")


def what_each_change_did(c: dict) -> str:
    """Paired difference in recall at 32k for each change tried, with 95% intervals."""
    free = rows("dev-free")
    tuned = rows("dev-llm", "dev-tune", "dev-tune2", "dev-tune-x3")
    groups = [
        (
            "Lexical ranking, 281 tasks",
            [
                (
                    "Three ranking signals",
                    "vs plain BM25",
                    paired(free, "bm25", "bm25-plain", 32000),
                    True,
                ),
                (
                    "Score cut-off",
                    "vs the three signals",
                    paired(free, "bm25-cut", "bm25", 32000),
                    False,
                ),
            ],
        ),
        (
            "Model selector, 100 tasks",
            [
                (
                    "Shortlist, with the first prompt",
                    "vs first selector",
                    paired(tuned, "llm-v1", "llm", 32000),
                    False,
                ),
                (
                    "Shortlist and recall prompt",
                    "vs first selector",
                    paired(tuned, "llm-v2", "llm", 32000),
                    True,
                ),
                (
                    "Shortlist, with the recall prompt",
                    "vs the prompt alone",
                    paired(tuned, "llm-v2", "llm-v2n", 32000),
                    True,
                ),
                (
                    "Search and outline tools, as built",
                    "vs kept selector",
                    paired(tuned, "llm-v3", "llm-v2", 32000),
                    False,
                ),
                (
                    "Search and outline tools, fixed",
                    "vs kept selector",
                    paired(tuned, "llm-v3b", "llm-v2", 32000),
                    False,
                ),
                (
                    "Three runs merged",
                    "vs kept selector",
                    paired(tuned, "llm-v2x3", "llm-v2", 32000),
                    False,
                ),
                (
                    "Top-up from the ranker",
                    "vs kept selector",
                    paired(tuned, "llm-v2-fill", "llm-v2", 32000),
                    False,
                ),
            ],
        ),
    ]

    width, row_h, label_w = 760, 40, 300
    lo, hi = -0.15, 0.25
    plot_l, plot_r = label_w + 24, width - 90
    n_rows = sum(len(items) for _, items in groups)
    top = 104
    height = top + n_rows * row_h + len(groups) * 30 + 40

    def x(v: float) -> float:
        return plot_l + (plot_r - plot_l) * (v - lo) / (hi - lo)

    body = [
        text(
            24,
            36,
            "The gains came from the ranking signals and the recall prompt with its shortlist",
            c["text"],
            17,
            weight=600,
        ),
        text(
            24,
            60,
            "Change in recall at 32k tokens, paired on the same tasks. Lines are 95% intervals.",
            c["muted"],
            13,
        ),
    ]
    legend_y = 84
    body.append(f'<circle cx="30" cy="{legend_y - 4}" r="5" fill="{c["blue"]}"/>')
    body.append(text(42, legend_y, "kept", c["muted"], 12))
    body.append(
        f'<circle cx="90" cy="{legend_y - 4}" r="5" fill="{c["surface"]}" '
        f'stroke="{c["muted"]}" stroke-width="2"/>'
    )
    body.append(text(102, legend_y, "not kept", c["muted"], 12))

    y = top
    grid_top = y
    for name, items in groups:
        y += 22
        body.append(text(24, y, name, c["text"], 13, weight=600))
        y += 8
        for label, versus, (d, a, b, _), kept in items:
            y += row_h
            cy = y - row_h / 2
            body.append(text(24, cy - 1, label, c["text"], 13))
            body.append(text(24, cy + 15, versus, c["muted"], 12))
            body.append(
                f'<line x1="{x(a):.1f}" x2="{x(b):.1f}" y1="{cy:.1f}" y2="{cy:.1f}" '
                f'stroke="{c["blue"] if kept else c["muted"]}" stroke-width="2" '
                f'stroke-linecap="round"/>'
            )
            fill = c["blue"] if kept else c["surface"]
            stroke = c["surface"] if kept else c["muted"]
            body.append(
                f'<circle cx="{x(d):.1f}" cy="{cy:.1f}" r="6" fill="{fill}" '
                f'stroke="{stroke}" stroke-width="2"/>'
            )
            body.append(text(plot_r + 16, cy + 4, f"{d:+.2f}", c["text"], 13))
    grid_bottom = y + 6
    for v in (-0.1, 0, 0.1, 0.2):
        strong = v == 0
        body.insert(
            3,
            f'<line x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{grid_top:.1f}" '
            f'y2="{grid_bottom:.1f}" stroke="{c["muted"] if strong else c["grid"]}" '
            f'stroke-width="{1.5 if strong else 1}"/>',
        )
        body.append(
            text(x(v), grid_bottom + 18, f"{v:+.1f}" if v else "0", c["muted"], 12, "middle")
        )
    return svg(width, height, body, c, "Change in recall for each change tried")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, draw in (
        ("recall-by-budget", recall_by_budget),
        ("what-each-change-did", what_each_change_did),
    ):
        for mode, colours in THEMES.items():
            (OUT / f"{name}-{mode}.svg").write_text(draw(colours), encoding="utf-8")
            print(OUT / f"{name}-{mode}.svg")


if __name__ == "__main__":
    main()
