"""Planted-signal / planted-null fixtures for the validation stage (NOX-003 T2/T9/T10).

The headline determinism + honesty guarantee (AT6): a planted-signal pair must recover the sign, the
lag, and an r in band; an independent (planted-null) pair must report r near zero with a
non-significant p. Deterministic via fixed RNG seeds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

LAG_K = 3  # the index leads the benchmark by this many weeks in the planted-signal fixture


def _index_frame(values, idx) -> pd.DataFrame:
    return pd.DataFrame(
        {"date": idx, "index_value": values, "valid_coverage": np.full(len(values), 0.9)}
    )


def _benchmark_frame(values, idx) -> pd.DataFrame:
    return pd.DataFrame({"date": idx, "value": values})


@pytest.fixture
def planted_signal_pair():
    """benchmark(t) = a·index(t-LAG_K) + noise, positive sign. Returns (index_df, benchmark_df, k)."""
    rng = np.random.default_rng(7)
    n = 120
    idx = pd.date_range("2021-01-03", periods=n, freq="W")
    x = np.cumsum(rng.normal(0, 1, n))  # a persistent index series
    bench = np.empty(n)
    bench[:] = np.nan
    bench[LAG_K:] = 2.0 * x[:-LAG_K] + rng.normal(0, 0.4, n - LAG_K)
    return _index_frame(x, idx), _benchmark_frame(bench, idx), LAG_K


@pytest.fixture
def planted_null_pair():
    """Two independent random-walk series — no real relationship. Returns (index_df, benchmark_df)."""
    rng = np.random.default_rng(99)
    n = 120
    idx = pd.date_range("2021-01-03", periods=n, freq="W")
    x = rng.normal(0, 1, n)
    y = rng.normal(0, 1, n)  # independent of x
    return _index_frame(x, idx), _benchmark_frame(y, idx)
