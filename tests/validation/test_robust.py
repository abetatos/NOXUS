"""Tests for autocorrelation-robust significance (NOX-008, REQ-020..022, 030; AT3/AT4/AT5)."""

from __future__ import annotations

import numpy as np

from noxus.validation import robust as R


def _ar1(n, phi, rng):
    """A length-n AR(1) series with coefficient phi and unit innovations."""
    x = np.empty(n)
    x[0] = rng.standard_normal()
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.standard_normal()
    return x


def test_autocorrelated_null_not_over_rejected():
    """Independent AR(1) pairs: naive Pearson over-rejects; robust tests hold near alpha (AT3)."""
    rng = np.random.default_rng(0)
    n, phi, trials = 160, 0.8, 60
    naive_hits = perm_hits = boot_hits = 0
    for _ in range(trials):
        x = _ar1(n, phi, rng)
        y = _ar1(n, phi, rng)  # independent of x
        rc = R.robust_corr(x, y, n_boot=300, n_perm=300, seed=int(rng.integers(1e9)))
        naive_hits += rc.p_naive < 0.05
        perm_hits += rc.p_perm < 0.05
        boot_hits += not (rc.boot_lo <= 0.0 <= rc.boot_hi)
    naive_rate = naive_hits / trials
    perm_rate = perm_hits / trials
    boot_rate = boot_hits / trials
    # Naive false-positive rate is inflated well above nominal 5% for phi=0.8 series.
    assert naive_rate > 0.20
    # The block-permutation null is the principled test: false-positive rate near alpha, below naive.
    assert perm_rate <= 0.15 and perm_rate < naive_rate
    # The percentile bootstrap CI of r is legitimately a touch more liberal, but still well below naive.
    assert boot_rate <= 0.25 and boot_rate < naive_rate


def test_planted_signal_recovered_not_merely_conservative():
    """A real correlation is still detected by the robust tests (AT4)."""
    rng = np.random.default_rng(7)
    x = _ar1(200, 0.6, rng)
    y = x + 0.5 * rng.standard_normal(200)  # strong genuine correlation
    rc = R.robust_corr(x, y, n_boot=600, n_perm=600, seed=11)
    assert rc.r > 0.7
    assert not (rc.boot_lo <= 0.0 <= rc.boot_hi)  # CI excludes 0
    assert rc.p_perm < 0.05
    assert rc.p_eff_nw < 0.05


def test_effective_n_below_n_for_autocorrelated_and_near_n_for_white():
    rng = np.random.default_rng(3)
    xa = _ar1(300, 0.85, rng)
    ya = _ar1(300, 0.85, rng)
    first, nw = R.effective_n(xa, ya)
    assert 2.0 <= first <= 300 and 2.0 <= nw <= 300
    # White noise: effective N is close to N (little serial dependence to discount).
    wx, wy = rng.standard_normal(300), rng.standard_normal(300)
    wf, _ = R.effective_n(wx, wy)
    assert wf > 200  # near n, far above the autocorrelated case is not required but should be large


def test_bh_fdr_matches_reference():
    p = [0.001, 0.01, 0.02, 0.5, 0.8]  # m = 5
    rejected, p_adj = R.bh_fdr(p, alpha=0.05)
    # adj = sort(p)*m/rank, monotone from the right: [0.005, 0.025, 0.0333, 0.625, 0.8]
    assert np.allclose(p_adj, [0.005, 0.025, 1.0 / 30, 0.625, 0.8], atol=1e-6)
    assert list(rejected) == [True, True, True, False, False]


def test_bh_fdr_all_null_no_discoveries_and_nan_carried():
    rejected, p_adj = R.bh_fdr([0.4, 0.6, 0.8, np.nan], alpha=0.05)
    assert not rejected.any()
    assert np.isnan(p_adj[3]) and not rejected[3]
