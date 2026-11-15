"""Means, intervals and paired differences, resampling whole repositories.

Tasks from one repository are not independent, so every interval here comes from a
bootstrap over repositories (a cluster bootstrap), with a fixed seed and resample count so
that every script reports the same interval for the same rows.
"""

import random
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from statistics import mean

RESAMPLES = 4000
SEED = 0

Value = Callable[[dict], float | None]


@dataclass
class Estimate:
    mean: float
    low: float
    high: float
    tasks: int
    repos: int
    repo_mean: float  # each repository weighted equally, not each task
    better: int = 0  # paired only: tasks where the first method scored higher
    worse: int = 0


def _clusters(rows: list[dict], value: Value) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        v = value(row)
        if v is not None:
            out[row["repo"]].append(v)
    return out


def bootstrap(groups: list[list[float]], resamples: int = RESAMPLES, seed: int = SEED):
    """A 95% percentile interval for the mean of all values, resampling whole groups."""
    rng, means = random.Random(seed), []
    for _ in range(resamples):
        drawn = [v for g in rng.choices(groups, k=len(groups)) for v in g]
        means.append(mean(drawn))
    means.sort()
    return means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]


def _estimate(groups: dict[str, list[float]]) -> Estimate:
    values = [v for g in groups.values() for v in g]
    if not values:
        raise ValueError("no values to estimate from")
    low, high = bootstrap(list(groups.values()))
    return Estimate(
        mean(values),
        low,
        high,
        len(values),
        len(groups),
        mean(mean(g) for g in groups.values()),
    )


def estimate(rows: list[dict], value: Value) -> Estimate:
    """The mean of `value` over rows, with its interval. Rows where it is None are left out."""
    return _estimate(_clusters(rows, value))


def _differences(a: list[dict], b: list[dict], value: Value) -> dict[str, list[float]]:
    by_task = {row["task"]: row for row in b}
    diffs: dict[str, list[float]] = defaultdict(list)
    for row in a:
        other = by_task.get(row["task"])
        if other is None:
            continue
        x, y = value(row), value(other)
        if x is not None and y is not None:
            diffs[row["repo"]].append(x - y)
    return diffs


def paired(a: list[dict], b: list[dict], value: Value) -> Estimate:
    """The mean difference a - b over the tasks both have, with its interval."""
    diffs = _differences(a, b, value)
    result = _estimate(diffs)
    flat = [d for g in diffs.values() for d in g]
    result.better = sum(d > 0 for d in flat)
    result.worse = sum(d < 0 for d in flat)
    return result


def leave_one_repo_out(a: list[dict], b: list[dict], value: Value) -> tuple[float, float]:
    """The smallest and largest paired mean difference with any one repository removed."""
    diffs = _differences(a, b, value)
    means = [
        mean(v for other, g in diffs.items() if other != repo for v in g)
        for repo in diffs
        if len(diffs) > 1
    ]
    return min(means), max(means)


def field(name: str, gold: str = "annotated") -> Value:
    """Read one score of one gold definition from a row."""
    return lambda row: row["scores"][gold][name]
