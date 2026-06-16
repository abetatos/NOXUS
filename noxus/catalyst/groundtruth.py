"""Ground-truth production events (NOX-004, REQ-010..012).

A production event is labelled from two public sources, combined (developer decision 2026-06-13):

- **CREA BF operating-rate jumps** (REQ-010): a large week-over-week change in the benchmark, typed
  ``up``/``down`` — objective and reproducible from the NOX-001 parquet.
- **Public curtailment calendar** (REQ-011): documented MEE/CREA episodes (heating-season cuts,
  blue-sky/summit curtailments, sudden shutdowns) as discrete intervals — interpretable but best-effort
  (NOX-003 Q6a); absent by default.

The two are combined and cause-tagged (REQ-012). Direction convention matches NO2 events: a production
``down`` (curtailment) is expected to coincide with an NO2 ``drop``, a production ``up`` with a ``surge``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from noxus.catalyst.events import causal_robust_z

_BENCH_COLUMNS = ("value", "operating_rate")


def _benchmark_series(bench: pd.DataFrame) -> pd.Series:
    col = next((c for c in _BENCH_COLUMNS if c in bench.columns), None)
    if col is None:
        raise ValueError(
            f"benchmark frame lacks a value column (expected one of {_BENCH_COLUMNS})."
        )
    idx = pd.to_datetime(bench["date"]) if "date" in bench.columns else pd.to_datetime(bench.index)
    s = pd.Series(bench[col].to_numpy(dtype=float), index=pd.DatetimeIndex(idx))
    return s[~s.index.duplicated(keep="last")].sort_index()


def bf_rate_events(
    bench: pd.DataFrame, *, z_thresh: float = 1.5, min_periods: int = 12
) -> pd.DataFrame:
    """Production events from large week-over-week BF operating-rate changes (REQ-010).

    The change series is standardised with the same causal robust z used for NO2 events, so the
    threshold is comparable. Returns ``date, direction (up|down), z, cause='bf_rate'``.
    """
    s = _benchmark_series(bench)
    change = s.diff()
    z = causal_robust_z(change, min_periods=min_periods)
    fired = (z.abs() >= z_thresh).fillna(False)
    idx = s.index[fired.to_numpy()]
    return pd.DataFrame(
        {
            "date": pd.to_datetime(idx),
            "direction": np.where(z.reindex(idx) >= 0, "up", "down"),
            "z": z.reindex(idx).to_numpy(),
            "cause": "bf_rate",
        }
    ).reset_index(drop=True)


def load_curtailment_calendar(path: Path | None) -> pd.DataFrame:
    """Load a best-effort public curtailment calendar (REQ-011); empty frame if absent (Q6a/Q4).

    Expected CSV columns: ``start, end, direction (up|down), cause`` (dates ISO). A curtailment is a
    production ``down``; a restart an ``up``. Returns a typed empty frame when ``path`` is None/missing
    so the pipeline degrades to BF-rate jumps only, documented rather than fabricated.
    """
    cols = ["start", "end", "direction", "cause"]
    if path is None or not Path(path).exists():
        return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})
    cal = pd.read_csv(path)
    missing = {"start", "end", "direction"} - set(cal.columns)
    if missing:
        raise ValueError(f"curtailment calendar missing columns {sorted(missing)}.")
    cal["start"] = pd.to_datetime(cal["start"])
    cal["end"] = pd.to_datetime(cal["end"])
    if "cause" not in cal.columns:
        cal["cause"] = "curtailment"
    return cal[cols]


def _calendar_edge_events(calendar: pd.DataFrame) -> pd.DataFrame:
    """Convert calendar intervals to edge events (start + end), not every period inside (EDGE-002)."""
    rows = []
    for _, r in calendar.iterrows():
        rows.append(
            {"date": r["start"], "direction": r["direction"], "z": np.nan, "cause": r["cause"]}
        )
        # The interval end is the opposite-direction edge (a curtailment ends -> production up).
        opp = "up" if r["direction"] == "down" else "down"
        rows.append({"date": r["end"], "direction": opp, "z": np.nan, "cause": f"{r['cause']}_end"})
    return pd.DataFrame(rows, columns=["date", "direction", "z", "cause"])


def production_events(
    bf_events: pd.DataFrame, calendar: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Combine BF-rate jump events and curtailment-calendar edges into one cause-tagged set (REQ-012)."""
    parts = [bf_events]
    if calendar is not None and len(calendar):
        parts.append(_calendar_edge_events(calendar))
    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return (
        out.sort_values("date").drop_duplicates(subset=["date", "direction"]).reset_index(drop=True)
    )
