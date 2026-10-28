"""benchkit: small, dependency-light statistics used across the benchmark analyses.

Everything here is plain numpy/pandas so each repository can vendor this file and run its analysis offline.
Functions are covered by tests/test_benchkit.py in every repository.
"""
from __future__ import annotations

import math
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ intervals and tests
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (k successes out of n)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def rule_of_three(n: int) -> float:
    """Upper 95% bound on an event rate after n trials with zero events."""
    return 3.0 / n


def bootstrap_ci(values: Sequence[float], stat: Callable = np.median, B: int = 2000, alpha: float = 0.05, seed: int = 7) -> tuple[float, float, float]:
    """Percentile bootstrap interval for stat(values). Returns (low, point, high)."""
    v = np.asarray(list(values), dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    boots = np.array([stat(rng.choice(v, size=len(v), replace=True)) for _ in range(B)])
    return (float(np.percentile(boots, 100 * alpha / 2)), float(stat(v)), float(np.percentile(boots, 100 * (1 - alpha / 2))))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation without scipy (average ranks for ties)."""
    rx = pd.Series(list(x)).rank().to_numpy()
    ry = pd.Series(list(y)).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


# ------------------------------------------------------------------ cost arithmetic
def usd_per_million_tokens(dph_usd: float, tokens_per_second: float) -> float:
    """Hourly price divided by tokens produced per hour, per million tokens."""
    return dph_usd / (tokens_per_second * 3600.0) * 1e6


def usd_per_thousand(dph_usd: float, seconds_per_unit: float) -> float:
    """Cost of 1,000 units that each take seconds_per_unit of a rented hour."""
    return dph_usd / 3600.0 * seconds_per_unit * 1000.0


def implied_bandwidth_gb_s(tokens_per_second: float, weights_gib: float) -> float:
    """Decode is bandwidth bound: speed x weight bytes read per token ~ memory bandwidth (GB/s)."""
    return tokens_per_second * weights_gib * 1.073741824


# ------------------------------------------------------------------ queueing (Erlang C)
def erlang_c(c: int, a: float) -> float:
    """Probability that an arriving request must wait: c servers, offered load a erlangs (a < c)."""
    s = sum(a ** k / math.factorial(k) for k in range(c))
    t = a ** c / math.factorial(c) * c / (c - a)
    return t / (s + t)


def p95_wait(c: int, rho: float, mu: float) -> float:
    """95th percentile queueing delay (seconds) at utilization rho with service rate mu per server."""
    a = rho * c
    if a >= c:
        return float("inf")
    pc = erlang_c(c, a)
    return 0.0 if pc <= 0.05 else math.log(pc / 0.05) / (c * mu - a * mu)


def max_utilization(c: int, service_s: float, wait_target_s: float = 1.0, step: float = 0.005) -> float:
    """Highest utilization at which the 95th percentile wait stays within wait_target_s."""
    mu = 1.0 / service_s
    best = 0.0
    for r in np.arange(step, 0.995, step):
        if p95_wait(c, float(r), mu) <= wait_target_s:
            best = float(r)
    return best


def unit_price(cost_full: float, utilization: float, failure_share: float = 0.05, overhead: float = 0.30, margin: float = 0.40) -> float:
    """Price per unit from the cost at full load: divide by utilization and (1 - failures), add overhead, gross up for margin."""
    return cost_full / (utilization * (1 - failure_share)) * (1 + overhead) / (1 - margin)


# ------------------------------------------------------------------ additive (rank-one) model in logs
def additive_log_fit(table: pd.DataFrame, iters: int = 300) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Fit log(latency[row, col]) = a[row] + b[col] on a table with missing cells.

    Returns (row_factor, col_factor, residuals_in_log). Factors are returned in the original (exp) scale.
    """
    L = np.log(table.astype(float))
    a = pd.Series(0.0, index=L.index)
    b = pd.Series(0.0, index=L.columns)
    for _ in range(iters):
        a = (L - b).mean(axis=1)
        b = L.sub(a, axis=0).mean(axis=0)
    pred = pd.DataFrame(np.add.outer(a.values, b.values), index=L.index, columns=L.columns)
    return np.exp(a), np.exp(b), L - pred
