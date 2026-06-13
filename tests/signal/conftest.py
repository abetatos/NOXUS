"""Planted synthetic fixtures for the signal stage (NOX-003 T2/T6/T7/T8).

Deterministic (fixed RNG seed). The planted series let the tests assert that meteo regress-out
recovers a known activity component, deseasonalisation removes a known seasonal term, and the index
is a strictly relative transform — never asserting on live data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def planted_meteo_signal():
    """signal = activity + beta·wind + small noise, with the true activity returned for comparison.

    Returns (signal, meteo, true_activity), all weekly-indexed over ~2 years so the regression has
    ample degrees of freedom. The activity is a smooth ramp; wind/blh are independent random drivers.
    """
    rng = np.random.default_rng(42)
    n = 104
    idx = pd.date_range("2021-01-03", periods=n, freq="W")
    activity = np.linspace(0.0, 5.0, n) + 2.0 * np.sin(np.arange(n) / 8.0)
    u10 = rng.normal(0, 1, n)
    v10 = rng.normal(0, 1, n)
    blh = rng.normal(500, 50, n)
    wind_speed = np.hypot(u10, v10)
    noise = rng.normal(0, 0.05, n)
    signal = activity + 1.5 * wind_speed - 0.01 * blh + noise

    signal_s = pd.Series(signal, index=idx, name="no2_corrected")
    meteo = pd.DataFrame({"u10": u10, "v10": v10, "blh": blh, "wind_speed": wind_speed}, index=idx)
    true_activity = pd.Series(activity, index=idx, name="activity")
    return signal_s, meteo, true_activity


@pytest.fixture
def planted_seasonal_series():
    """A series with a strong annual cycle on top of a trend, weekly over 3 years.

    Returns (series, trend) so the test can check that yoy-double-diff removes the annual cycle.
    """
    n = 156  # 3 years weekly
    idx = pd.date_range("2020-01-05", periods=n, freq="W")
    trend = np.linspace(10.0, 20.0, n)
    season = 5.0 * np.sin(2 * np.pi * np.arange(n) / 52.0)
    series = pd.Series(trend + season, index=idx, name="no2_corrected")
    return series, pd.Series(trend, index=idx, name="trend")
