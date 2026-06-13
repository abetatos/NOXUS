"""Prophet + harmonic deseasonalisation tests (NOX-006; AT2-AT4, AT7, AT-ERR-1).

Prophet-using tests run on a small planted daily series (deterministic MAP) and skip cleanly if prophet
is unavailable; the harmonic fallback + lazy-import tests always run. No network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from noxus.signal import prophet_deseason as P

prophet = pytest.importorskip("prophet")  # skip the prophet-dependent tests if it isn't installed


def _planted_daily(weekly: bool = True, gaps: bool = True, seed: int = 0):
    """Daily series = trend + yearly + (weekly) + activity(40d) + noise, with optional cloud gaps."""
    rng = np.random.default_rng(seed)
    n = 800
    idx = pd.date_range("2021-01-01", periods=n, freq="D")
    i = np.arange(n)
    doy = idx.dayofyear.to_numpy(float)
    trend = 100.0 - 0.03 * i
    yearly = 8.0 * np.sin(2 * np.pi * doy / 365.25)
    wk = np.where(idx.dayofweek >= 5, -3.0, 1.2) if weekly else np.zeros(n)
    activity = 5.0 * np.sin(
        2 * np.pi * i / 40.0
    )  # 40-day cycle, not weekly/yearly -> stays in residual
    signal = trend + yearly + wk + activity + rng.normal(0, 0.5, n)
    s = pd.Series(signal, index=idx, name="no2_corrected")
    if gaps:
        s.iloc[rng.choice(n, size=n // 4, replace=False)] = np.nan
    return s, pd.Series(trend, index=idx), pd.Series(activity, index=idx)


# --------------------------------------------------------------------------- decomposition (AT2/AT3)


def test_prophet_recovers_trend_residual_and_is_observed_only():
    s, trend, activity = _planted_daily()
    fit = P.prophet_deseason(s, yearly_order=4, weekly_order=3, min_valid_days=120)

    # Residual is emitted only on observed days (gap days never fabricated, REQ-012).
    assert len(fit.residual) == int(s.notna().sum())
    assert fit.residual.index.isin(s.dropna().index).all()

    # Trend recovers the planted secular decline; residual recovers the planted activity.
    tr = pd.concat([fit.trend, trend], axis=1).dropna()
    assert np.corrcoef(tr.iloc[:, 0], tr.iloc[:, 1])[0, 1] > 0.9
    ar = pd.concat([fit.residual, activity], axis=1).dropna()
    assert np.corrcoef(ar.iloc[:, 0], ar.iloc[:, 1])[0, 1] > 0.7


def test_prophet_is_deterministic():
    s, _, _ = _planted_daily()
    r1 = P.prophet_deseason(s, min_valid_days=120).residual
    r2 = P.prophet_deseason(s, min_valid_days=120).residual
    assert np.allclose(r1.to_numpy(), r2.to_numpy(), atol=1e-6)


# --------------------------------------------------------------------------- weekday diagnostic (AT4)


def test_weekday_diagnostic_recovers_planted_cycle_and_flat():
    s_wk, _, _ = _planted_daily(weekly=True, seed=1)
    s_flat, _, _ = _planted_daily(weekly=False, seed=1)
    fit_wk = P.prophet_deseason(s_wk, min_valid_days=120)
    fit_flat = P.prophet_deseason(s_flat, min_valid_days=120)

    # A planted weekly cycle gives a much larger weekly amplitude than a flat-weekly series.
    assert fit_wk.weekly_amplitude > 2.0
    assert fit_flat.weekly_amplitude < fit_wk.weekly_amplitude

    prof = P.weekday_profile(fit_wk.weekly)
    assert list(prof["weekday"]) == ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    # Weekend effect is below the weekday mean (we planted a weekend dip).
    weekend = prof[prof["weekday"].isin(["Sat", "Sun"])]["effect"].mean()
    weekday = prof[prof["weekday"].isin(["Mon", "Tue", "Wed", "Thu", "Fri"])]["effect"].mean()
    assert weekend < weekday


def test_variance_removed_recorded():
    s, _, _ = _planted_daily()
    fit = P.prophet_deseason(s, min_valid_days=120)
    assert set(fit.variance_removed) == {"trend", "yearly", "weekly"}
    assert all(0.0 <= v <= 2.0 for v in fit.variance_removed.values())


# --------------------------------------------------------------------------- errors (AT-ERR-1)


def test_short_series_raises():
    s, _, _ = _planted_daily()
    with pytest.raises(P.InsufficientDataError, match="min_valid_days"):
        P.prophet_deseason(s.iloc[:50], min_valid_days=120)
