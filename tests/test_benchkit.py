import math

import numpy as np
import pandas as pd

from analysis import benchkit as bk


def test_wilson_known_value():
    lo, hi = bk.wilson(2, 10)
    assert math.isclose(lo, 0.0567, abs_tol=2e-3) and math.isclose(hi, 0.5098, abs_tol=2e-3)


def test_rule_of_three():
    assert math.isclose(bk.rule_of_three(50), 0.06)


def test_mcnemar_symmetry_and_extremes():
    assert bk.mcnemar_exact(3, 3) == 1.0
    assert bk.mcnemar_exact(0, 0) == 1.0
    assert bk.mcnemar_exact(10, 0) < 0.01


def test_bootstrap_contains_point():
    lo, pt, hi = bk.bootstrap_ci(np.arange(1, 101), B=500)
    assert lo <= pt <= hi


def test_cost_arithmetic():
    assert math.isclose(bk.usd_per_million_tokens(0.72, 100), 2.0, rel_tol=1e-9)
    assert math.isclose(bk.usd_per_thousand(3.6, 1.0), 1.0, rel_tol=1e-9)


def test_bandwidth_identity():
    assert math.isclose(bk.implied_bandwidth_gb_s(10, 10), 107.37, abs_tol=0.01)


def test_erlang_c_limits():
    assert bk.erlang_c(1, 0.5) == 0.5  # M/M/1: P(wait) = rho
    assert bk.erlang_c(8, 0.1) < 1e-6


def test_pooling_raises_the_utilization_ceiling():
    u1 = bk.max_utilization(1, 6.19)
    u32 = bk.max_utilization(32, 6.19)
    assert u1 < 0.1 < 0.7 < u32 < 1.0


def test_unit_price_monotone_in_utilization():
    assert bk.unit_price(0.22, 0.4) > bk.unit_price(0.22, 0.8)


def test_additive_fit_recovers_exact_structure():
    a = np.array([1.0, 2.0, 4.0]); b = np.array([10.0, 20.0, 5.0])
    t = pd.DataFrame(np.outer(a, b), index=list("xyz"), columns=list("pqr"))
    t.iloc[0, 2] = np.nan
    fa, fb, res = bk.additive_log_fit(t)
    assert np.nanmax(np.abs(res.values)) < 1e-6
