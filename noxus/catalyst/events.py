"""NO2 production-event detection (NOX-004, REQ-002..005).

Detects discrete production events — sharp surges/drops in the intensity-detrended, meteo-normalised
footprint residual (NOX-003.1) — as the basis for the catalyst. Two disciplines are baked in:

- **Causal / no look-ahead (REQ-041):** the robust anomaly baseline at period *t* uses only data up to
  *t* (an expanding median/MAD), so the event decision at *t* can never change when future data is
  appended. This is what makes the marker tradeable rather than a hindsight artefact.
- **Coverage + strengthened meteo screen (REQ-003/004/005):** a period below the inherited valid-
  coverage floor is discarded (a cloud gap is not an event), and an event whose direction is explained
  by a same-sign ventilation anomaly (a stagnant-air week, not a production change) is rejected. The
  ventilation control is the cheap, in-framework lever for "better weather correlation" — not the
  deferred flux-divergence method (NOX-005).

The detector emits a typed event table: ``date, direction (surge|drop), z, magnitude, detector``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_MAD_TO_SIGMA = 1.4826  # MAD -> std for a normal distribution


def ventilation_index(meteo: pd.DataFrame) -> pd.Series:
    """Ventilation index ≈ 10 m wind speed × boundary-layer height (REQ-005).

    High = well-ventilated (NO2 disperses); low = stagnant (NO2 accumulates → a weather-driven surge).
    Accepts a frame carrying ``wind_speed`` (or ``u10``/``v10`` to derive it) and ``blh``.
    """
    m = meteo
    if "wind_speed" in m.columns:
        wind = m["wind_speed"].astype(float)
    elif {"u10", "v10"}.issubset(m.columns):
        wind = np.hypot(m["u10"].astype(float), m["v10"].astype(float))
    else:
        raise ValueError("ventilation_index needs 'wind_speed' or both 'u10' and 'v10'.")
    blh = m["blh"].astype(float) if "blh" in m.columns else pd.Series(1.0, index=m.index)
    return pd.Series(wind.to_numpy() * blh.to_numpy(), index=m.index, name="ventilation_index")


def causal_robust_z(s: pd.Series, *, min_periods: int) -> pd.Series:
    """Causal robust z-score: at each t, standardise by the expanding median/MAD of data up to t−1.

    Using only past observations (shifted expanding window) guarantees the score at *t* is invariant to
    future data (no look-ahead, REQ-041). Returns NaN until ``min_periods`` past points exist or where
    the MAD is zero/undefined.
    """
    x = s.astype(float)
    past = x.shift(1)  # exclude the current point from its own baseline
    med = past.expanding(min_periods=min_periods).median()
    mad = (past - med).abs().expanding(min_periods=min_periods).median()
    scale = _MAD_TO_SIGMA * mad
    z = (x - med) / scale.where(scale > 0)
    return z.rename("z")


def detect_events(
    residual: pd.Series,
    coverage: pd.Series | None = None,
    meteo: pd.DataFrame | None = None,
    *,
    z_thresh: float = 2.0,
    method: str = "zscore",
    min_periods: int = 12,
    min_coverage: float = 0.25,
    meteo_screen: bool = True,
    ventilation_z: float = 1.5,
) -> pd.DataFrame:
    """Detect coverage- and meteo-screened NO2 production events (REQ-002..005).

    ``residual`` is the NOX-003.1 activity residual (date-indexed). An event is a period whose causal
    robust z exceeds ``z_thresh`` in magnitude (``method='zscore'``), or a CUSUM level-shift
    (``'cusum'``), or either (``'both'``). Periods with ``coverage`` < ``min_coverage`` are dropped
    (REQ-003). When ``meteo_screen`` and ``meteo`` are given, an event explained by a same-sign
    ventilation anomaly is rejected (REQ-005): a *surge* coinciding with a stagnation (ventilation z ≤
    −``ventilation_z``) or a *drop* coinciding with strong dispersion (ventilation z ≥ +``ventilation_z``).

    Returns columns ``date, direction, z, magnitude, detector``; empty (typed) frame if none survive.
    """
    s = residual.astype(float)
    z = causal_robust_z(s, min_periods=min_periods)

    if method == "zscore":
        fired = z.abs() >= z_thresh
        detector = pd.Series("zscore", index=s.index)
    elif method == "cusum":
        fired = _cusum_flags(s, min_periods=min_periods, k=0.5, h=z_thresh)
        detector = pd.Series("cusum", index=s.index)
    elif method == "both":
        zf = z.abs() >= z_thresh
        cf = _cusum_flags(s, min_periods=min_periods, k=0.5, h=z_thresh)
        fired = zf | cf
        detector = pd.Series(
            np.where(zf & cf, "both", np.where(zf, "zscore", "cusum")), index=s.index
        )
    else:
        raise ValueError(f"Unknown detector method {method!r} (use 'zscore'/'cusum'/'both').")

    fired = fired.fillna(False) & s.notna()

    # Coverage screen (REQ-003): a low-coverage period is not an event (cloud gap, not production).
    if coverage is not None:
        cov = coverage.reindex(s.index)
        fired = fired & ~(cov.notna() & (cov < min_coverage))

    direction = pd.Series(np.where(z >= 0, "surge", "drop"), index=s.index)

    # Strengthened meteo screen (REQ-005): reject weather-explained events.
    if meteo_screen and meteo is not None:
        vent = ventilation_index(meteo).reindex(s.index)
        vent_z = causal_robust_z(vent, min_periods=min_periods)
        stagnation = vent_z <= -ventilation_z  # poor ventilation -> spurious surge
        dispersion = vent_z >= ventilation_z  # strong ventilation -> spurious drop
        weather = ((direction == "surge") & stagnation.fillna(False)) | (
            (direction == "drop") & dispersion.fillna(False)
        )
        fired = fired & ~weather

    idx = s.index[fired.to_numpy()]
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(idx),
            "direction": direction.reindex(idx).to_numpy(),
            "z": z.reindex(idx).to_numpy(),
            "magnitude": s.reindex(idx).to_numpy(),
            "detector": detector.reindex(idx).to_numpy(),
        }
    ).reset_index(drop=True)
    return out


def _cusum_flags(s: pd.Series, *, min_periods: int, k: float, h: float) -> pd.Series:
    """Causal two-sided CUSUM level-shift flags on the causal-z of ``s`` (REQ-002).

    Accumulates positive/negative excursions of the causal robust z beyond a slack ``k``; flags a period
    when the running sum exceeds ``h`` and resets. Causal by construction (built on ``causal_robust_z``).
    """
    z = causal_robust_z(s, min_periods=min_periods).fillna(0.0).to_numpy()
    flags = np.zeros(len(z), dtype=bool)
    sp = sn = 0.0
    for i, zi in enumerate(z):
        sp = max(0.0, sp + zi - k)
        sn = min(0.0, sn + zi + k)
        if sp > h:
            flags[i] = True
            sp = 0.0
        elif sn < -h:
            flags[i] = True
            sn = 0.0
    return pd.Series(flags, index=s.index)
