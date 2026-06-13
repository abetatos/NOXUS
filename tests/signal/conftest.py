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


@pytest.fixture
def planted_decomposition():
    """Planted intensity decomposition (NOX-003.1 AT101/AT102).

    Mirrors the real decoupling mechanism: a dominant, smooth, *declining* emission-intensity trend
    ``s_true`` (retrofit) while *activity* trends mildly upward with a faster oscillation. The signal
    keeps the declining trend plus the activity oscillation, so:

      - in **levels** the signal falls while activity rises -> corr(signal, activity) < 0;
      - after removing the smooth trend the **residual** retains the oscillation -> corr(residual,
        activity) > 0.

    Returns (signal, benchmark, true_trend, true_activity_osc). ``benchmark`` is the activity (the
    stand-in for the CREA BF rate). The true trend is linear, so CV should select a low df (~2).
    """
    n = 120
    idx = pd.date_range("2020-01-05", periods=n, freq="W")
    t = np.linspace(0.0, 1.0, n)
    rng = np.random.default_rng(7)
    s_true = 100.0 - 55.0 * t  # dominant, smooth, declining intensity (retrofit)
    a_trend = 20.0 * t  # mild upward activity trend
    a_osc = 6.0 * np.sin(2 * np.pi * t * 12.0)  # faster activity oscillation (recoverable)
    activity = a_trend + a_osc
    signal = s_true + a_osc + rng.normal(0, 0.5, n)

    signal_s = pd.Series(signal, index=idx, name="no2_corrected")
    benchmark = pd.Series(activity, index=idx, name="value")
    true_trend = pd.Series(s_true, index=idx, name="trend")
    true_osc = pd.Series(a_osc, index=idx, name="activity_osc")
    return signal_s, benchmark, true_trend, true_osc


@pytest.fixture
def planted_null_decomposition():
    """A trended signal and an INDEPENDENT benchmark (NOX-003.1 AT103).

    No smoothing choice should manufacture a residual correlation, because the benchmark is unrelated
    to the signal's activity term. Returns (signal, benchmark).
    """
    n = 120
    idx = pd.date_range("2020-01-05", periods=n, freq="W")
    t = np.linspace(0.0, 1.0, n)
    rng = np.random.default_rng(11)
    signal = 100.0 - 40.0 * t + 5.0 * np.sin(2 * np.pi * t * 9.0) + rng.normal(0, 0.5, n)
    benchmark = rng.normal(50.0, 5.0, n)  # independent of the signal
    return (
        pd.Series(signal, index=idx, name="no2_corrected"),
        pd.Series(benchmark, index=idx, name="value"),
    )
