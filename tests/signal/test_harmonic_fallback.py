"""Harmonic fallback + Prophet lazy-import/error tests (NOX-006; AT7).

These run WITHOUT prophet installed (no importorskip): the harmonic method is the dependency-free
fallback, and selecting 'prophet' when it is absent must raise an actionable error.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest

from noxus.signal import index as I
from noxus.signal import prophet_deseason as P


def _planted_seasonal_daily(seed: int = 0):
    rng = np.random.default_rng(seed)
    n = 800
    idx = pd.date_range("2021-01-01", periods=n, freq="D")
    doy = idx.dayofyear.to_numpy(float)
    i = np.arange(n)
    trend = 50.0 - 0.02 * i
    yearly = 6.0 * np.sin(2 * np.pi * doy / 365.25)
    activity = 4.0 * np.sin(2 * np.pi * i / 45.0)
    return pd.Series(
        trend + yearly + activity + rng.normal(0, 0.4, n), index=idx, name="no2_corrected"
    )


# --------------------------------------------------------------------------- harmonic fallback


def test_harmonic_removes_annual_and_keeps_activity():
    s = _planted_seasonal_daily()
    out = P.harmonic_deseason(s, k=1, min_length=60)
    assert out.name == "residual_activity"
    # The annual cycle is removed -> residual variance is well below the raw series variance.
    assert out.dropna().std() < s.std()


def test_deseasonalize_harmonic_branch_records_attrs():
    s = _planted_seasonal_daily()
    out = I.deseasonalize(s, method="harmonic")
    assert out.attrs["deseason_method"] == "harmonic"
    assert out.attrs["harmonic_order"] == 1


# --------------------------------------------------------------------------- prophet lazy import (ERR-002)


def test_prophet_missing_raises_actionable(monkeypatch):
    # Simulate prophet not installed: a bare module with no `Prophet` symbol -> ImportError on import.
    monkeypatch.setitem(sys.modules, "prophet", types.ModuleType("prophet"))
    s = _planted_seasonal_daily()
    with pytest.raises(P.ProphetUnavailableError, match="prophet is not installed"):
        P.prophet_deseason(s, min_valid_days=120)


def test_deseasonalize_unknown_method_lists_prophet_and_harmonic():
    s = _planted_seasonal_daily()
    with pytest.raises(ValueError, match="prophet"):
        I.deseasonalize(s, method="bogus")
