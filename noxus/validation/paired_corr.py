"""Paired correlation-difference estimator (NOX-016 REQ-012).

The scale contrast asks whether one NO2 feature tracks a NATIONAL steel target better than it
tracks the Tangshan cluster target. Answering that by checking whether the two marginal confidence
intervals overlap is a well-known error: the two correlations share the same ``x`` over an
overlapping window, so their difference has far lower variance than the margins suggest, and an
overlap test is badly under-powered.

This module estimates ``r(x, y_b) - r(x, y_a)`` directly on the JOINTLY observed sample with a
moving-block bootstrap, reusing the NOX-008 block machinery so the block-length rule and the seed
discipline are identical to every other interval in the project.

Takes series, not targets: it has no notion of which side is the benchmark, so nothing here can be
tuned toward an outcome.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from noxus.validation.robust import block_bootstrap_statistic, block_length


@dataclass(frozen=True)
class PairedCorrDifference:
    """``r_b - r_a`` measured on the common sample, with a block-bootstrap CI."""

    r_a: float
    r_b: float
    difference: float
    lo: float
    hi: float
    n: int
    block: int

    @property
    def excludes_zero(self) -> bool:
        return bool(
            np.isfinite(self.lo) and np.isfinite(self.hi) and (self.lo > 0.0 or self.hi < 0.0)
        )


def _corr(u: np.ndarray, v: np.ndarray) -> float:
    if len(u) < 3 or np.std(u) == 0 or np.std(v) == 0:
        return float("nan")
    return float(np.corrcoef(u, v)[0, 1])


def paired_corr_difference(
    x,
    y_a,
    y_b,
    *,
    n_boot: int = 5000,
    seed: int = 20260828,
    alpha: float = 0.05,
    block: int | None = None,
) -> PairedCorrDifference:
    """Block-bootstrap CI for ``corr(x, y_b) - corr(x, y_a)`` on the jointly observed sample.

    All three series are inner-joined on their index first, so both correlations are measured over
    the SAME periods -- otherwise the difference would confound a change in coupling with a change
    in sample. Returns NaNs rather than raising when the common sample is too short or degenerate,
    so a caller can record a refusal instead of losing the whole run.
    """
    df = pd.concat(
        {
            "x": pd.Series(x).astype(float),
            "a": pd.Series(y_a).astype(float),
            "b": pd.Series(y_b).astype(float),
        },
        axis=1,
    ).dropna()
    n = int(len(df))
    if n < 3:
        return PairedCorrDifference(
            float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), n, 0
        )

    xa = df["x"].to_numpy()
    ya = df["a"].to_numpy()
    yb = df["b"].to_numpy()
    blk = block_length(n) if block is None else max(1, int(block))

    r_a, r_b = _corr(xa, ya), _corr(xa, yb)

    def stat(x_s, a_s, b_s) -> float:
        return _corr(x_s, b_s) - _corr(x_s, a_s)

    lo, hi = block_bootstrap_statistic(
        stat, xa, ya, yb, block=blk, n_draws=n_boot, seed=seed, alpha=alpha
    )
    return PairedCorrDifference(
        r_a=r_a, r_b=r_b, difference=r_b - r_a, lo=lo, hi=hi, n=n, block=blk
    )
