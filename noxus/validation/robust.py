"""Autocorrelation-robust significance for index<->benchmark correlations (NOX-008, REQ-020..023, 030).

``noxus/validation/leadlag.py`` reports the Pearson p-value from ``scipy.stats.pearsonr``, which assumes
i.i.d. observations. NO2 and the steel benchmark are smooth, strongly autocorrelated series, so that
p-value (and the +-1.96/sqrt(n) lead-lag band) overstate significance: the effective number of
independent observations is far below ``n``. This module promotes the 2026-06-14 exploratory probe
(``analysis/autocorr_significance.py``) into tested, reusable code that judges a correlation honestly:

  - ``effective_n`` — effective sample size under serial dependence: a first-order (Bayley-Hammersley)
    estimate and a full-order (Bartlett / Newey-West) estimate; ``p_from_r`` turns either into a p-value.
  - ``block_bootstrap_ci`` — moving-block bootstrap CI of ``r`` (preserves within-block memory).
  - ``block_permutation_p`` — block-permutation null p (reorder blocks of x against y: keeps each
    series' autocorrelation, destroys the cross-correlation).
  - ``bh_fdr`` — Benjamini-Hochberg multiple-testing correction across a family of p-values.
  - ``robust_corr`` — one call returning the naive Pearson result plus all of the above.

These are additive: ``leadlag.py``'s ``correlate``/``lead_lag``/``verify_sign`` are unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class RobustCorr:
    """Pearson ``r`` with naive and autocorrelation-robust significance (REQ-020..022)."""

    r: float
    n: int
    p_naive: float
    n_eff_first: float
    p_eff_first: float
    n_eff_nw: float
    p_eff_nw: float
    boot_lo: float
    boot_hi: float
    p_perm: float
    block: int
    n_draws: int


def _aligned(x, y) -> tuple[np.ndarray, np.ndarray]:
    """Jointly non-NaN values of two (optionally pandas) series as float arrays."""
    if isinstance(x, pd.Series) or isinstance(y, pd.Series):
        df = pd.concat({"x": pd.Series(x), "y": pd.Series(y)}, axis=1).dropna()
        return df["x"].to_numpy(float), df["y"].to_numpy(float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def _autocorr(x: np.ndarray, k: int) -> float:
    """Lag-``k`` autocorrelation of a 1-D array (0 if degenerate)."""
    n = len(x)
    if k <= 0 or k >= n or np.std(x) == 0:
        return 0.0
    xc = x - x.mean()
    denom = np.sum(xc * xc)
    if denom == 0:
        return 0.0
    return float(np.sum(xc[:-k] * xc[k:]) / denom)


def block_length(n: int, *, exponent: float = 1.0 / 3.0, floor: int = 2) -> int:
    """Moving-block length ``round(n**exponent)`` floored at ``floor`` and capped below ``n`` (Q2)."""
    if n < 2 * floor:
        return max(1, n // 2)
    return int(min(max(round(n**exponent), floor), max(floor, n // 2)))


def effective_n(x, y, *, order: str = "auto") -> tuple[float, float]:
    """Effective sample size under serial dependence (REQ-020).

    Returns ``(n_eff_first, n_eff_nw)``: the first-order Bayley-Hammersley estimate
    ``n*(1-a*b)/(1+a*b)`` (``a``,``b`` = lag-1 autocorrelations) and a full-order Bartlett/Newey-West
    estimate ``n / (1 + 2*sum_k (1-k/(L+1)) * rho_x(k)*rho_y(k))`` with bandwidth ``L`` from the
    Newey-West rule. Both are clamped to ``[2, n]``. ``order`` selects which to actually compute
    (``"first"``/``"newey-west"``/``"auto"`` = both); the unused slot mirrors the computed one.
    """
    xa, ya = _aligned(x, y)
    n = int(len(xa))
    if n < 3 or np.std(xa) == 0 or np.std(ya) == 0:
        return float(max(n, 2)), float(max(n, 2))

    a, b = _autocorr(xa, 1), _autocorr(ya, 1)
    denom = 1.0 + a * b
    n_first = float(np.clip(n * (1.0 - a * b) / denom, 2.0, n)) if denom > 0 else 2.0

    n_nw = n_first
    if order in ("auto", "newey-west"):
        bandwidth = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
        bandwidth = max(1, min(bandwidth, n - 2))
        s = 0.0
        for k in range(1, bandwidth + 1):
            w = 1.0 - k / (bandwidth + 1.0)
            s += w * _autocorr(xa, k) * _autocorr(ya, k)
        inflation = 1.0 + 2.0 * s
        n_nw = float(np.clip(n / inflation, 2.0, n)) if inflation > 0 else 2.0

    if order == "first":
        return n_first, n_first
    if order == "newey-west":
        return n_nw, n_nw
    return n_first, n_nw


def p_from_r(r: float, n_eff: float) -> float:
    """Two-sided p-value for Pearson ``r`` on ``n_eff-2`` degrees of freedom (REQ-020)."""
    df = n_eff - 2.0
    if df <= 0 or abs(r) >= 1.0:
        return float("nan")
    t = r * np.sqrt(df / (1.0 - r * r))
    return float(2.0 * stats.t.sf(abs(t), df))


def block_bootstrap_ci(
    x, y, *, block: int, n_draws: int = 5000, seed: int = 20260614, alpha: float = 0.05
) -> tuple[float, float]:
    """Moving-block bootstrap (1-alpha) CI for Pearson ``r`` of the paired series (REQ-021)."""
    xa, ya = _aligned(x, y)
    n = len(xa)
    if n < 2 * block or np.std(xa) == 0 or np.std(ya) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts_pool = np.arange(0, n - block + 1)
    rs = np.empty(n_draws)
    for i in range(n_draws):
        starts = rng.choice(starts_pool, size=n_blocks, replace=True)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        xb, yb = xa[idx], ya[idx]
        rs[i] = np.corrcoef(xb, yb)[0, 1] if np.std(xb) > 0 and np.std(yb) > 0 else 0.0
    lo = float(np.nanpercentile(rs, 100 * alpha / 2))
    hi = float(np.nanpercentile(rs, 100 * (1 - alpha / 2)))
    return lo, hi


def block_permutation_p(
    x, y, *, block: int, n_draws: int = 5000, seed: int = 20260614, r_obs: float | None = None
) -> float:
    """Two-sided block-permutation null p-value: reorder contiguous blocks of x against y (REQ-022).

    Cutting x into contiguous blocks and permuting the block order destroys the x<->y cross-correlation
    while approximately preserving x's own autocorrelation, giving a null calibrated for serial
    dependence. ``(count+1)/(n_draws+1)`` so the p-value is never zero (EDGE-003).
    """
    xa, ya = _aligned(x, y)
    n = len(xa)
    if n < 2 * block or np.std(xa) == 0 or np.std(ya) == 0:
        return float("nan")
    if r_obs is None:
        r_obs = float(np.corrcoef(xa, ya)[0, 1])
    rng = np.random.default_rng(seed)
    edges = list(range(0, n, block)) + [n]
    blocks = [xa[edges[i] : edges[i + 1]] for i in range(len(edges) - 1)]
    count = 0
    for _ in range(n_draws):
        order = rng.permutation(len(blocks))
        xp = np.concatenate([blocks[k] for k in order])
        r = np.corrcoef(xp, ya)[0, 1] if np.std(xp) > 0 else 0.0
        if abs(r) >= abs(r_obs):
            count += 1
    return float((count + 1) / (n_draws + 1))


def bh_fdr(pvalues: Sequence[float], alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR over a family of p-values (REQ-030).

    Returns ``(rejected, p_adj)`` aligned to the input order. NaN p-values are carried through as
    not-rejected with NaN adjusted value and excluded from the family size ``m``.
    """
    p = np.asarray(pvalues, dtype=float)
    rejected = np.zeros(p.shape, dtype=bool)
    p_adj = np.full(p.shape, np.nan)
    finite = np.where(np.isfinite(p))[0]
    m = finite.size
    if m == 0:
        return rejected, p_adj
    order = finite[np.argsort(p[finite])]
    ranked = p[order]
    adj = ranked * m / (np.arange(1, m + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]  # enforce monotonicity
    adj = np.clip(adj, 0.0, 1.0)
    p_adj[order] = adj
    rejected[order] = adj <= alpha
    return rejected, p_adj


def robust_corr(
    x,
    y,
    *,
    n_boot: int = 5000,
    n_perm: int = 5000,
    seed: int = 20260614,
    order: str = "auto",
    block: int | None = None,
) -> RobustCorr:
    """Pearson ``r`` with naive + autocorrelation-robust significance in one call (REQ-020..022)."""
    xa, ya = _aligned(x, y)
    n = int(len(xa))
    if n < 3 or np.std(xa) == 0 or np.std(ya) == 0:
        return RobustCorr(
            r=0.0,
            n=n,
            p_naive=1.0,
            n_eff_first=float(max(n, 2)),
            p_eff_first=1.0,
            n_eff_nw=float(max(n, 2)),
            p_eff_nw=1.0,
            boot_lo=float("nan"),
            boot_hi=float("nan"),
            p_perm=1.0,
            block=0,
            n_draws=0,
        )
    res = stats.pearsonr(xa, ya)
    r, p_naive = float(res.statistic), float(res.pvalue)
    blk = block if block is not None else block_length(n)
    n_first, n_nw = effective_n(xa, ya, order=order)
    lo, hi = block_bootstrap_ci(xa, ya, block=blk, n_draws=n_boot, seed=seed)
    p_perm = block_permutation_p(xa, ya, block=blk, n_draws=n_perm, seed=seed, r_obs=r)
    return RobustCorr(
        r=r,
        n=n,
        p_naive=p_naive,
        n_eff_first=n_first,
        p_eff_first=p_from_r(r, n_first),
        n_eff_nw=n_nw,
        p_eff_nw=p_from_r(r, n_nw),
        boot_lo=lo,
        boot_hi=hi,
        p_perm=p_perm,
        block=blk,
        n_draws=n_boot,
    )
