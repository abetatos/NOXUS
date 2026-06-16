"""Footprint sampling + regional background correction (NOX-003, REQ-001..004).

The relative-index / cluster paradigm (Kim 2023): rather than a flat AOI mean, sample the NO2 cube on
a footprint around the *operating* steel facilities, estimate a regional background from an annular
ring outside that footprint, and emit a per-period background-corrected footprint signal with the
NOX-002b valid-coverage propagated and never interpolated (REQ-004).

The cube is the NOX-002b weekly product (``noxus/data/gridding.py``): dims ``time``/``y``/``x`` with
``x`` holding longitude and ``y`` holding latitude (the openEO grid), variables ``no2`` and
``coverage``. Distances are computed with the haversine great-circle formula, not raw degrees, so the
configured ``radius_km`` is a true ground distance.

Flux-divergence point-source attribution (Beirle) is explicitly out of scope here — deferred to
NOX-005.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from noxus.config.region import TANGSHAN, Region
from noxus.config.run import ACTIVE_FACILITY_STATUSES, SignalConfig
from noxus.data.gridding import COVERAGE
from noxus.data.tropomi import NO2

EARTH_RADIUS_KM = 6371.0088  # mean Earth radius (IUGG)


class GeometryError(RuntimeError):
    """Degenerate attribution geometry (ERR-003): no facility in extent, or an empty background ring."""


def attribute(no2_field):  # pragma: no cover - retained scaffold, superseded by footprint_signal
    """Deprecated scaffold kept for backward compatibility.

    The real attribution path is ``footprint_mask`` → ``background_ring`` → ``footprint_signal``.
    """
    raise NotImplementedError(
        "attribute() is superseded by footprint_mask/background_ring/footprint_signal"
    )


def _cube_lonlat(cube: xr.Dataset) -> tuple[str, str]:
    """Return the (lon_name, lat_name) coordinate names of the cube.

    The NOX-002b openEO cube carries longitude under ``x`` and latitude under ``y``. Common
    geographic aliases are accepted too, so the functions also work on a CF-style cube.
    """
    lon = next((c for c in ("x", "lon", "longitude") if c in cube.coords), None)
    lat = next((c for c in ("y", "lat", "latitude") if c in cube.coords), None)
    if lon is None or lat is None:
        raise GeometryError(
            "Cube has no recognisable lon/lat coordinates (expected x/y, lon/lat, or "
            f"longitude/latitude); found {list(cube.coords)}."
        )
    return lon, lat


def _haversine_km(lon1, lat1, lon2, lat2) -> np.ndarray:
    """Great-circle distance in km between scalars/arrays of (lon, lat) in degrees."""
    lon1, lat1, lon2, lat2 = (
        np.radians(np.asarray(v, dtype=float)) for v in (lon1, lat1, lon2, lat2)
    )
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def load_facilities(csv: Path | str, active_status: set[str] | None = None) -> pd.DataFrame:
    """Load the facility CSV, filtering to active facilities (REQ-001).

    Keeps rows whose ``status`` is in ``active_status`` (default: the active set from config —
    ``operating`` and ``operating pre-retirement``). Co-located duplicates are *not* dropped here;
    de-duplication happens at the cell level in ``footprint_mask`` (EDGE-001), because two facilities
    at the same point still map to the same footprint cells.
    """
    csv = Path(csv)
    if not csv.exists():
        raise FileNotFoundError(
            f"Facilities CSV not found: {csv}. Expected the committed "
            "data/derived/tangshan_steel_facilities.csv."
        )
    statuses = set(active_status) if active_status is not None else set(ACTIVE_FACILITY_STATUSES)
    df = pd.read_csv(csv)
    missing = {"latitude", "longitude", "status"} - set(df.columns)
    if missing:
        raise ValueError(f"Facilities CSV missing columns: {sorted(missing)}")
    active = df[df["status"].isin(statuses)].reset_index(drop=True)
    return active


def _cell_centres(cube: xr.Dataset) -> tuple[np.ndarray, np.ndarray, str, str]:
    """Return broadcast (lon, lat) grids of every cell centre, plus the coord names."""
    lon_name, lat_name = _cube_lonlat(cube)
    lon = cube[lon_name].values
    lat = cube[lat_name].values
    # The grid dims may be ordered (y, x); build a 2-D mesh matching (lat, lon).
    lon2d, lat2d = np.meshgrid(lon, lat)
    return lon2d, lat2d, lon_name, lat_name


def _mask_within_radius(
    cube: xr.Dataset, facilities: pd.DataFrame, radius_km: float
) -> xr.DataArray:
    """Boolean (y, x) mask of cells within ``radius_km`` of any facility centre."""
    lon2d, lat2d, lon_name, lat_name = _cell_centres(cube)
    mask = np.zeros(lon2d.shape, dtype=bool)
    for _, fac in facilities.iterrows():
        d = _haversine_km(lon2d, lat2d, float(fac["longitude"]), float(fac["latitude"]))
        mask |= d <= radius_km
    # Dedup is automatic: a cell selected by several facilities is one True entry (EDGE-001).
    return xr.DataArray(
        mask,
        dims=(lat_name, lon_name),
        coords={lat_name: cube[lat_name], lon_name: cube[lon_name]},
        name="footprint",
    )


def footprint_mask(cube: xr.Dataset, facilities: pd.DataFrame, radius_km: float) -> xr.DataArray:
    """Boolean cell mask of the footprint around the operating facilities (REQ-002).

    Selects every cube cell whose centre lies within ``radius_km`` (haversine) of any facility in
    ``facilities``. Co-located facilities collapse to the same cells, so cells are counted once
    (EDGE-001). Raises ``GeometryError`` if no facility falls inside the cube extent (ERR-003).
    """
    if facilities.empty:
        raise GeometryError("No active facilities supplied; cannot build a footprint (ERR-003).")
    mask = _mask_within_radius(cube, facilities, radius_km)
    if not bool(mask.any()):
        raise GeometryError(
            "No facility falls within the cube extent at the configured footprint radius "
            f"({radius_km} km); footprint is empty (ERR-003)."
        )
    return mask


def background_ring(
    cube: xr.Dataset,
    footprint: xr.DataArray,
    inner_km: float,
    outer_km: float,
    facilities: pd.DataFrame | None = None,
    aoi: Region | None = None,
) -> xr.DataArray:
    """Annular-ring background mask around the cluster, clipped to the AOI (REQ-003, EDGE-002).

    Cells are in the ring when their distance to the cluster centroid is in ``[inner_km, outer_km]``;
    the ring then excludes any footprint cell and is clipped to the AOI bounding box. The cluster
    centroid is the mean of the facility coordinates when supplied, else the footprint centroid.
    Raises ``GeometryError`` if the resulting ring contains no cells (ERR-003).
    """
    lon2d, lat2d, lon_name, lat_name = _cell_centres(cube)

    if facilities is not None and not facilities.empty:
        clon = float(facilities["longitude"].mean())
        clat = float(facilities["latitude"].mean())
    else:
        fp = footprint.values
        clon = float(lon2d[fp].mean())
        clat = float(lat2d[fp].mean())

    dist = _haversine_km(lon2d, lat2d, clon, clat)
    ring = (dist >= inner_km) & (dist <= outer_km)
    # Exclude footprint cells so the background never overlaps the source (EDGE-002).
    ring &= ~footprint.values
    # Clip to the AOI bounding box (EDGE-002).
    aoi = aoi or TANGSHAN
    in_aoi = (
        (lon2d >= aoi.min_lon)
        & (lon2d <= aoi.max_lon)
        & (lat2d >= aoi.min_lat)
        & (lat2d <= aoi.max_lat)
    )
    ring &= in_aoi

    ring_da = xr.DataArray(
        ring,
        dims=(lat_name, lon_name),
        coords={lat_name: cube[lat_name], lon_name: cube[lon_name]},
        name="background",
    )
    if not bool(ring_da.any()):
        raise GeometryError(
            "Background ring contains no cells after excluding the footprint and clipping to the "
            f"AOI (inner={inner_km} km, outer={outer_km} km); cannot estimate a background (ERR-003)."
        )
    return ring_da


def _masked_period_mean(cube: xr.Dataset, mask: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    """Per-period (time) mean of ``no2`` over the masked cells, plus mean valid-coverage.

    Returns (no2_mean, coverage_mean) arrays aligned to the cube ``time`` axis. Cells already masked
    by NOX-002b coverage screening are NaN and skipped (skipna); periods with no valid masked cell
    yield NaN — never interpolated (REQ-004).
    """
    no2 = cube[NO2].where(mask)
    spatial_dims = [d for d in no2.dims if d != "time"]
    no2_mean = no2.mean(dim=spatial_dims, skipna=True)
    if COVERAGE in cube:
        cov = cube[COVERAGE].where(mask).mean(dim=spatial_dims, skipna=True)
        cov_vals = cov.values
    else:
        cov_vals = np.full(no2_mean.sizes.get("time", len(no2_mean)), np.nan)
    return no2_mean.values, cov_vals


def footprint_signal(
    cube: xr.Dataset,
    footprint: xr.DataArray,
    background: xr.DataArray,
    mode: str = "subtract",
) -> pd.DataFrame:
    """Per-period background-corrected footprint signal (REQ-003/004).

    Columns: ``date``, ``no2_footprint``, ``no2_bg``, ``no2_corrected``, ``valid_coverage``.

    ``mode="subtract"`` (default) → corrected = footprint − background. ``mode="normalise"`` →
    corrected = footprint / background (ratio). Below-coverage / cloud-gapped periods are NaN and are
    *not* interpolated; the row is retained so downstream alignment sees the gap honestly (REQ-004).
    """
    fp_no2, fp_cov = _masked_period_mean(cube, footprint)
    bg_no2, _ = _masked_period_mean(cube, background)

    if mode == "subtract":
        corrected = fp_no2 - bg_no2
    elif mode == "normalise":
        with np.errstate(divide="ignore", invalid="ignore"):
            corrected = np.where(bg_no2 != 0, fp_no2 / bg_no2, np.nan)
    else:
        raise ValueError(
            f"Unknown background correction mode: {mode!r} (use 'subtract'/'normalise')."
        )

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(cube["time"].values),
            "no2_footprint": fp_no2,
            "no2_bg": bg_no2,
            "no2_corrected": corrected,
            "valid_coverage": fp_cov,
        }
    )
    return df.reset_index(drop=True)


def build_footprint_signal(cfg: SignalConfig | None = None) -> Path:
    """End-to-end footprint signal: load cube + facilities → mask → background → write parquet.

    Emits ``steel_footprint_signal.parquet`` under ``data/derived/no2/`` (gitignored — NO2-derived).
    Returns the output path. Raises ``FileNotFoundError`` (ERR-001) if the cube is missing.
    """
    cfg = cfg or SignalConfig()
    cube_path = Path(cfg.cube_path)
    if not cube_path.exists():
        raise FileNotFoundError(
            f"NO2 cube not found: {cube_path}. Run 'noxus grid' first to build the weekly cube "
            "(ERR-001)."
        )
    cube = xr.open_dataset(cube_path).load()
    cube.close()

    facilities = load_facilities(cfg.facilities_csv, set(cfg.active_statuses))
    fp = footprint_mask(cube, facilities, cfg.footprint_radius_km)
    bg = background_ring(
        cube, fp, cfg.background_inner_km, cfg.background_outer_km, facilities=facilities
    )
    df = footprint_signal(cube, fp, bg, mode=cfg.background_mode)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / cfg.footprint_signal_name
    df.to_parquet(out_path, index=False)
    return out_path
