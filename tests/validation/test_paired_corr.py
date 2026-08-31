"""Paired correlation-difference estimator tests (NOX-016 AT3).

The whole point of this estimator is that comparing two correlations by checking whether their
marginal CIs overlap is under-powered: both correlations share the same ``x``, so the difference is
far better determined than the margins suggest. AT3 asserts exactly that, so the wrong method
cannot pass silently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from noxus.validation.paired_corr import paired_corr_difference
from noxus.validation.robust import block_bootstrap_ci


def _ar1(rng, n, phi, sd=1.0):
    x = np.zeros(n)
    e = rng.normal(0.0, sd, n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def _fixture(n=300, gain_a=0.25, gain_b=0.85, seed=11):
    """Serially dependent x; y_a weakly linked, y_b strongly linked, both with independent noise."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-05", periods=n, freq="W")
    x = _ar1(rng, n, 0.8)
    y_a = gain_a * x + _ar1(rng, n, 0.5)
    y_b = gain_b * x + _ar1(rng, n, 0.5)
    return (pd.Series(x, idx), pd.Series(y_a, idx), pd.Series(y_b, idx))


def test_at3_recovers_a_planted_difference_and_excludes_zero():
    x, y_a, y_b = _fixture()
    res = paired_corr_difference(x, y_a, y_b, n_boot=800, seed=3)
    assert res.r_b > res.r_a
    assert res.difference == pytest.approx(res.r_b - res.r_a, abs=1e-12)
    assert res.excludes_zero, f"a planted difference must be detected: {res}"
    assert res.lo < res.difference < res.hi


def test_at3_paired_interval_is_narrower_than_the_naive_marginal_comparison():
    # The failure mode this estimator exists to prevent: judging a contrast by whether two marginal
    # CIs overlap. The paired interval must be materially narrower than the marginal-difference one.
    x, y_a, y_b = _fixture()
    res = paired_corr_difference(x, y_a, y_b, n_boot=800, seed=3)
    paired_width = res.hi - res.lo

    a_lo, a_hi = block_bootstrap_ci(x, y_a, block=res.block, n_draws=800, seed=3)
    b_lo, b_hi = block_bootstrap_ci(x, y_b, block=res.block, n_draws=800, seed=3)
    marginal_width = (b_hi - b_lo) + (a_hi - a_lo)  # width of r_b - r_a treating them independently
    assert paired_width < marginal_width, (
        f"paired {paired_width:.4f} should beat marginal {marginal_width:.4f}"
    )


def test_at3_equal_correlations_give_a_ci_containing_zero():
    # The estimator must not manufacture a contrast where none exists.
    x, y_a, y_b = _fixture(gain_a=0.5, gain_b=0.5, seed=29)
    res = paired_corr_difference(x, y_a, y_b, n_boot=800, seed=5)
    assert not res.excludes_zero, f"no planted difference, yet CI excludes zero: {res}"


def test_measured_on_the_common_sample_only():
    # Both correlations must be measured over the SAME periods, or the difference confounds a
    # change in coupling with a change in sample.
    x, y_a, y_b = _fixture(n=200)
    y_b_short = y_b.iloc[:120]
    res = paired_corr_difference(x, y_a, y_b_short, n_boot=200, seed=7)
    assert res.n == 120


def test_short_or_degenerate_input_returns_nan_rather_than_raising():
    idx = pd.date_range("2020-01-05", periods=2, freq="W")
    res = paired_corr_difference(
        pd.Series([1.0, 2.0], idx), pd.Series([1.0, 2.0], idx), pd.Series([2.0, 1.0], idx)
    )
    assert res.n == 2 and np.isnan(res.difference) and not res.excludes_zero
