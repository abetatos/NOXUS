"""VIIRS facility-radius join + weekly/daily thermal-activity series (NOX-011, REQ-010..012).

Turns AOI-clipped VIIRS detections into a facility-anchored thermal-activity series for the Tangshan
steel cluster. Two steps, both **benchmark-blind** (NFR-006, A9): they take only detections +
facilities + fixed pre-registered parameters and **never** the CREA benchmark or the NO2 result, so
the thermal index cannot be tuned to the correlation.

  * :func:`join_to_facilities` — assign each detection to the *nearest* catalogued facility centroid
    within a fixed radius (numpy haversine, no geopandas — NFR-004); drop detections outside every
    facility radius (REQ-010, EDGE-002).
  * :func:`build_thermal_series` — aggregate the joined detections to a weekly (primary) / daily
    (secondary) series: **summed radiant heat (MW)** is the headline index, with **detection count**
    and **mean detection temperature** retained as alternates (REQ-011, A3). Zero-detection periods are
    zero-activity, not interpolated (EDGE-001); sparse periods are flagged (EDGE-005).

The facility radius (A2), aggregation frequency (A5) and primary aggregate (A3) are fixed in
``ThermalTriangulationConfig`` and recorded in ``assumptions.html`` *before* the triangulation is
viewed (anti-p-hacking, NFR-006).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from noxus.data.viirs import (
    LATITUDE,
    LONGITUDE,
    RADIANT_HEAT_MW,
    TEMPERATURE_K,
    TIMESTAMP,
)

_EARTH_RADIUS_KM = 6371.0088
_REQUIRED_FACILITY_COLUMNS = ("name", "latitude", "longitude")


class FacilitiesError(ValueError):
    """The facilities catalogue is missing or malformed (ERR-005)."""


def load_facilities(csv_path: Path | str) -> pd.DataFrame:
    """Load + validate the catalogued steel facilities (join target + overlay), raising on error.

    Raises :class:`FacilitiesError` rather than silently yielding an empty series when the CSV is
    missing or malformed (ERR-005).
    """
    path = Path(csv_path)
    if not path.exists():
        raise FacilitiesError(f"Facilities CSV not found: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001 - surface a malformed CSV descriptively
        raise FacilitiesError(f"Facilities CSV could not be read ({path}): {exc}") from exc
    missing = [c for c in _REQUIRED_FACILITY_COLUMNS if c not in df.columns]
    if missing:
        raise FacilitiesError(
            f"Facilities CSV {path} is missing required column(s) {missing}; "
            f"present columns were {list(df.columns)}."
        )
    if df.empty:
        raise FacilitiesError(f"Facilities CSV {path} has no rows.")
    return df


def _haversine_km(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Great-circle distance (km) between two sets of lat/lon points (broadcast, numpy only)."""
    lat1r, lon1r, lat2r, lon2r = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def join_to_facilities(
    detections: pd.DataFrame, facilities: pd.DataFrame, radius_km: float
) -> pd.DataFrame:
    """Assign each detection to the nearest facility within ``radius_km``; drop the rest (REQ-010).

    Benchmark-blind. Adds ``facility_name``, ``facility_index`` and ``distance_km``. A detection within
    range of more than one facility goes to the nearest (EDGE-002); one outside every radius is dropped.
    """
    if detections.empty:
        return detections.assign(facility_name=[], facility_index=[], distance_km=[])
    det_lat = pd.to_numeric(detections[LATITUDE], errors="coerce").to_numpy(float)
    det_lon = pd.to_numeric(detections[LONGITUDE], errors="coerce").to_numpy(float)
    fac_lat = facilities["latitude"].to_numpy(float)
    fac_lon = facilities["longitude"].to_numpy(float)
    fac_name = facilities["name"].to_numpy()

    # Distance matrix: detections (rows) x facilities (cols).
    dist = _haversine_km(det_lat[:, None], det_lon[:, None], fac_lat[None, :], fac_lon[None, :])
    nearest = np.argmin(dist, axis=1)
    nearest_dist = dist[np.arange(len(detections)), nearest]
    keep = nearest_dist <= radius_km

    out = detections.loc[keep].reset_index(drop=True).copy()
    out["facility_index"] = nearest[keep]
    out["facility_name"] = fac_name[nearest[keep]]
    out["distance_km"] = nearest_dist[keep]
    return out


def build_thermal_series(
    joined: pd.DataFrame,
    freq: str = "W",
    aggregate: str = "radiant_heat",
    sparse_min_nights: int = 2,
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Aggregate joined detections to a weekly/daily thermal-activity series (REQ-011, benchmark-blind).

    Columns: ``radiant_heat_mw`` (summed — the primary index), ``detection_count``,
    ``mean_temperature_k`` (alternates, A3), ``n_detection_nights`` and ``sparse`` (fewer than
    ``sparse_min_nights`` detection-nights in the period — EDGE-005). When ``start``/``end`` are given
    the series is reindexed to the full period range so zero-detection periods appear as zero-activity,
    not gaps and never interpolated (EDGE-001). ``aggregate`` selects which column is copied to
    ``activity`` as the headline index (default summed radiant heat).
    """
    idx_name = "period"
    if joined.empty:
        empty = pd.DataFrame(
            columns=[
                "radiant_heat_mw",
                "detection_count",
                "mean_temperature_k",
                "n_detection_nights",
                "sparse",
                "activity",
            ]
        )
        empty.index.name = idx_name
        return _reindex_full(empty, freq, start, end, sparse_min_nights)

    df = joined.copy()
    df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP])
    df["_night"] = df[TIMESTAMP].dt.normalize()
    df["_rh"] = pd.to_numeric(df[RADIANT_HEAT_MW], errors="coerce")
    df["_temp"] = pd.to_numeric(df.get(TEMPERATURE_K), errors="coerce")

    grouper = pd.Grouper(key=TIMESTAMP, freq=freq)
    agg = df.groupby(grouper).agg(
        radiant_heat_mw=("_rh", "sum"),
        detection_count=("_rh", "size"),
        mean_temperature_k=("_temp", "mean"),
        n_detection_nights=("_night", "nunique"),
    )
    agg.index.name = idx_name
    agg["sparse"] = agg["n_detection_nights"] < sparse_min_nights
    agg = _reindex_full(agg, freq, start, end, sparse_min_nights)
    agg["activity"] = agg[_aggregate_column(aggregate)]
    return agg


def _aggregate_column(aggregate: str) -> str:
    return {
        "radiant_heat": "radiant_heat_mw",
        "count": "detection_count",
        "temperature": "mean_temperature_k",
    }.get(aggregate, "radiant_heat_mw")


def _reindex_full(
    agg: pd.DataFrame,
    freq: str,
    start: pd.Timestamp | str | None,
    end: pd.Timestamp | str | None,
    sparse_min_nights: int,
) -> pd.DataFrame:
    """Reindex to the full period range, filling zero-activity periods (EDGE-001) — never interpolate."""
    if start is None and end is None:
        return agg
    lo = pd.Timestamp(start) if start is not None else agg.index.min()
    hi = pd.Timestamp(end) if end is not None else agg.index.max()
    full = pd.date_range(start=lo, end=hi, freq=freq)
    agg = agg.reindex(full)
    agg.index.name = "period"
    for col in ("radiant_heat_mw", "detection_count", "n_detection_nights"):
        if col in agg.columns:
            agg[col] = agg[col].fillna(0.0)
    agg["sparse"] = agg["n_detection_nights"] < sparse_min_nights
    return agg


def thermal_activity(series: pd.DataFrame) -> pd.Series:
    """Return the headline activity series (name ``activity``) for the triangulation."""
    return series["activity"] if "activity" in series.columns else series["radiant_heat_mw"]
