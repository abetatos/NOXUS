"""Planted fixtures for the catalyst tests (NOX-004). Deterministic (fixed seeds)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

N = 120
SURGE_IDX = (40, 80)
DROP_IDX = (60,)
STAGNATION_IDX = 100  # a weather-driven surge that must be screened out


@pytest.fixture
def planted_event_residual():
    """Residual ~ N(0,1) with planted step events; returns (residual, coverage, meteo, truth).

    truth is a DataFrame of the real events (date, direction). A high-NO2 SURGE is also planted at
    STAGNATION_IDX coinciding with a ventilation-index collapse (stagnant air) — it should be rejected
    by the meteo screen, so it is NOT in truth.
    """
    rng = np.random.default_rng(3)
    idx = pd.date_range("2021-01-03", periods=N, freq="W")
    x = rng.normal(0, 1.0, N)
    for i in SURGE_IDX:
        x[i] += 6.0
    for i in DROP_IDX:
        x[i] -= 6.0
    x[STAGNATION_IDX] += 6.0  # weather-driven (see meteo below)

    residual = pd.Series(x, index=idx, name="residual_activity")
    coverage = pd.Series(0.9, index=idx)

    # Meteo: ventilation varies week to week (so the robust baseline has non-zero scale), collapsing
    # sharply only at STAGNATION_IDX (low wind + shallow PBL -> stagnant air -> spurious NO2 surge).
    wind = np.abs(rng.normal(4.0, 0.8, N))
    blh = rng.normal(800.0, 100.0, N)
    wind[STAGNATION_IDX] = 0.3
    blh[STAGNATION_IDX] = 120.0
    meteo = pd.DataFrame({"u10": wind, "v10": np.zeros(N), "blh": blh}, index=idx)

    truth = (
        pd.DataFrame(
            {
                "date": [idx[i] for i in (*SURGE_IDX, *DROP_IDX)],
                "direction": ["surge"] * len(SURGE_IDX) + ["drop"] * len(DROP_IDX),
            }
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    return residual, coverage, meteo, truth


@pytest.fixture
def planted_null_residual():
    """Pure-noise residual (no events) for the false-positive test."""
    rng = np.random.default_rng(99)
    idx = pd.date_range("2021-01-03", periods=N, freq="W")
    return pd.Series(rng.normal(0, 1.0, N), index=idx, name="residual_activity")


@pytest.fixture
def planted_production_events(planted_event_residual):
    """Production events lagging the NO2 events by +1 week (so NO2 leads by 7 days)."""
    _, _, _, truth = planted_event_residual
    prod = truth.copy()
    prod["date"] = pd.to_datetime(prod["date"]) + pd.Timedelta(days=7)
    prod["direction"] = prod["direction"].map({"surge": "up", "drop": "down"})
    prod["z"] = 3.0
    prod["cause"] = "bf_rate"
    return prod.reset_index(drop=True)


@pytest.fixture
def planted_market_prices(planted_event_residual):
    """Daily abnormal returns per instrument with a planted +ve bump after surge events.

    Returns {instrument: DataFrame(date, abnormal_return)} over a daily grid covering the events.
    """
    _, _, _, truth = planted_event_residual
    days = pd.date_range("2021-01-01", periods=N * 7 + 30, freq="D")
    rng = np.random.default_rng(7)
    out = {}
    for inst in ("BHP", "RIO"):
        ar = rng.normal(0, 0.002, len(days))
        s = pd.Series(ar, index=days)
        # Plant a positive abnormal return in the few sessions after each surge (latency 2d).
        for _, e in truth[truth["direction"] == "surge"].iterrows():
            start = pd.Timestamp(e["date"]) + pd.Timedelta(days=2)
            s.loc[(s.index >= start) & (s.index < start + pd.Timedelta(days=6))] += 0.01
        out[inst] = pd.DataFrame({"date": s.index, "abnormal_return": s.to_numpy()})
    return out
