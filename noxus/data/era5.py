"""ERA5 meteorological ingest for the meteo regress-out stage (NOX-003, REQ-010).

Meteorology (10 m wind components + boundary-layer height) is the dominant week-to-week confounder of
the NO2 column: ventilation, not activity, drives much of the variance (Mao 2025 / Li 2024). To
remove it we ingest ERA5 over the AOI and aggregate it to the steel footprint at the TROPOMI overpass
window, then ``noxus.signal.index.regress_out_meteo`` residualises it out of the footprint signal.

Ingest source — **Copernicus CDS** (decision 2026-06-13 / Q3), public/free (NFR-002). The
``cds.climate.copernicus.eu`` API via ``cdsapi`` (dataset ``reanalysis-era5-single-levels``) subsets
the AOI / era / overpass-hour **server-side**, so only a few MB cross the wire for the whole
2019→present era. Needs a free CDS account credential read by reference only — ``CDSAPI_KEY`` (+
optional ``CDSAPI_URL``) from ``.env`` / the environment, falling back to ``~/.cdsapirc`` — never read
into logs, printed, or committed (protected-area policy). A one-time licence acceptance is required on
the dataset page. (The ARCO-ERA5 Zarr alternative was dropped: its chunk-1 layout downloads a full
global field per timestep, ~30–760 GB for a small AOI over a long era — wrong access pattern here.)

The ingest writes a dated local ``.nc`` snapshot; analysis then reads only the snapshot (NFR-001: no
live fetch at analysis time). Tests inject the retrieval via ``_cds_fetch``: no live calls and no
credentials are required to run the suite.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# Map our config short names to the CDS request variable (long) names. CDS returns the GRIB short
# names (u10/v10/blh) in the NetCDF, which already match ``cfg.era5_vars``.
ERA5_VAR_NAMES = {
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "blh": "boundary_layer_height",
}

# CDS (Copernicus Climate Data Store) — the ERA5 source, server-side AOI subsetting.
CDS_DATASET = "reanalysis-era5-single-levels"
CDS_DEFAULT_URL = "https://cds.climate.copernicus.eu/api"
# Env var names (loaded from .env via python-dotenv); these are cdsapi's own conventional names. The
# values are read by reference only — never read into logs, printed, or committed (protected area).
CDS_URL_ENV = "CDSAPI_URL"
CDS_KEY_ENV = "CDSAPI_KEY"

# TROPOMI Sentinel-5P crosses ~13:30 local solar time. Tangshan (~118.5 E) is ~UTC+8, so the overpass
# falls around 05:30 UTC. We aggregate ERA5 within a window centred there before compositing.
OVERPASS_UTC_HOUR = 5
OVERPASS_WINDOW_HOURS = 1


class ERA5SnapshotError(RuntimeError):
    """The ERA5 snapshot is missing or unreadable while meteo normalisation is enabled (ERR-002)."""


class ERA5SourceError(RuntimeError):
    """The ERA5 ingest could not be completed (e.g. missing credential / malformed snapshot)."""


def snapshot_path(snapshot_dir: Path, today: date) -> Path:
    """Return the dated snapshot path ``<snapshot_dir>/era5_<YYYY-MM-DD>.nc``."""
    return Path(snapshot_dir) / f"era5_{today.isoformat()}.nc"


def _overpass_hours() -> list[str]:
    """The overpass-window UTC hours as CDS ``HH:00`` strings (e.g. 04:00, 05:00, 06:00)."""
    lo = OVERPASS_UTC_HOUR - OVERPASS_WINDOW_HOURS
    hi = OVERPASS_UTC_HOUR + OVERPASS_WINDOW_HOURS
    return [f"{h:02d}:00" for h in range(lo, hi + 1)]


def ingest_era5(
    cfg,
    snapshot_dir: Path | None = None,
    *,
    today: date | None = None,
    _cds_fetch=None,
) -> Path:
    """Fetch an AOI/era/overpass-hour ERA5 subset from the CDS and write a dated snapshot (REQ-010).

    ``cfg`` is a :class:`~noxus.config.run.SignalConfig` (read for ``era5_vars``, ``era5_start`` and
    ``era5_snapshot_dir``). The CDS retrieves the AOI **server-side**, so only a few MB cross the wire
    for the whole era. The request is the cross-product of (years x months x days x overpass-hours);
    CDS silently ignores invalid/unavailable dates, and we trim to ``[era5_start, today]`` after
    download. The snapshot carries only the configured variables over the AOI, with the time
    coordinate normalised to ``time`` and short variable names, so it aligns with the NOX-002b grid
    (EDGE-006) and analysis is reproducible from the snapshot alone.

    Tests pass ``_cds_fetch`` to replace the network retrieval — the suite needs no network and no
    credentials.
    """
    from noxus.config.region import TANGSHAN

    snapshot_dir = Path(snapshot_dir) if snapshot_dir is not None else Path(cfg.era5_snapshot_dir)
    today = today or date.today()
    aoi = TANGSHAN

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    out = snapshot_path(snapshot_dir, today)

    variables_long = [ERA5_VAR_NAMES.get(s, s) for s in cfg.era5_vars]
    # CDS area order is [North, West, South, East].
    area = [aoi.max_lat, aoi.min_lon, aoi.min_lat, aoi.max_lon]
    start = pd.Timestamp(cfg.era5_start)
    end = pd.Timestamp(today)
    months = [f"{m:02d}" for m in range(1, 13)]
    days = [f"{d:02d}" for d in range(1, 32)]
    hours = _overpass_hours()

    fetch = _cds_fetch if _cds_fetch is not None else _fetch_cds_netcdf
    # Fetch one year per request: a single multi-year request exceeds the CDS per-request cost limit.
    parts: list[xr.Dataset] = []
    raws: list[Path] = []
    for year in range(start.year, end.year + 1):
        raw = snapshot_dir / f".era5_cds_raw_{today.isoformat()}_{year}.nc"
        raws.append(raw)
        fetch(
            target=raw,
            variables=variables_long,
            area=area,
            years=[str(year)],
            months=months,
            days=days,
            hours=hours,
        )
        parts.append(_normalise_cds(xr.open_dataset(raw).load(), tuple(cfg.era5_vars)))

    ds = xr.concat(parts, dim="time") if len(parts) > 1 else parts[0]
    if "time" in ds.coords:
        ds = ds.sortby("time")
        # Drop any duplicate timestamps from overlapping requests, then trim to the era of record.
        _, keep = np.unique(ds["time"].values, return_index=True)
        ds = ds.isel(time=keep).sel(time=slice(str(start.date()), str(end.date())))
    ds.to_netcdf(out, engine="h5netcdf")
    for raw in raws:
        try:
            raw.unlink()
        except OSError:
            pass
    return out


def _normalise_cds(ds: xr.Dataset, short_vars: tuple[str, ...]) -> xr.Dataset:
    """Normalise a CDS ERA5 NetCDF to the snapshot convention (short var names, ``time`` coord).

    The new CDS names the time coordinate ``valid_time`` and may add singleton ``number``/``expver``
    dims; CDS already returns the GRIB short names (u10/v10/blh) matching ``cfg.era5_vars``.
    """
    if "valid_time" in ds.variables:
        ds = ds.rename({"valid_time": "time"})
    for extra in ("number", "expver"):
        if extra in ds.dims:
            ds = ds.isel({extra: 0}, drop=True)
        elif extra in ds.coords:
            ds = ds.reset_coords(extra, drop=True)
    keep = [v for v in short_vars if v in ds.data_vars]
    return ds[keep] if keep else ds


def _fetch_cds_netcdf(
    *,
    target: Path,
    variables: list[str],
    area: list[float],
    years: list[str],
    months: list[str],
    days: list[str],
    hours: list[str],
) -> Path:  # pragma: no cover - network path
    """Retrieve an ERA5 single-levels AOI subset via the CDS API to ``target`` (REQ-010, NFR-002).

    Credentials are read by reference only (protected-area policy): ``CDSAPI_KEY`` (+ optional
    ``CDSAPI_URL``) from the environment / ``.env``, falling back to ``~/.cdsapirc`` when unset. The
    key value is never read into logs, printed, or committed. Isolated so the ``_cds_fetch`` injection
    in :func:`ingest_era5` lets tests avoid the network entirely.
    """
    import os

    import cdsapi
    from dotenv import load_dotenv

    load_dotenv()  # surface CDSAPI_KEY/CDSAPI_URL from .env into the environment (names only)
    key = os.environ.get(CDS_KEY_ENV)
    url = os.environ.get(CDS_URL_ENV, CDS_DEFAULT_URL)
    # Explicit url+key when CDSAPI_KEY is set (the .env path); otherwise let cdsapi read ~/.cdsapirc.
    client = cdsapi.Client(url=url, key=key) if key else cdsapi.Client()
    client.retrieve(
        CDS_DATASET,
        {
            "product_type": "reanalysis",
            "variable": list(variables),
            "year": years,
            "month": months,
            "day": days,
            "time": hours,
            "area": area,
            "data_format": "netcdf",
            "download_format": "unarchived",
        },
        str(target),
    )
    return Path(target)


def _overpass_window(ds: xr.Dataset) -> xr.Dataset:
    """Restrict an hourly ERA5 dataset to the TROPOMI overpass window (~05:30 UTC over Tangshan).

    If the dataset has no recognisable hourly time axis (e.g. already daily), it is returned as-is.
    """
    if "time" not in ds.coords:
        return ds
    hours = pd.DatetimeIndex(ds["time"].values).hour
    lo = OVERPASS_UTC_HOUR - OVERPASS_WINDOW_HOURS
    hi = OVERPASS_UTC_HOUR + OVERPASS_WINDOW_HOURS
    in_window = (hours >= lo) & (hours <= hi)
    if not in_window.any():
        return ds  # not hourly / window absent — fall back to all timesteps
    return ds.isel(time=np.flatnonzero(in_window))


def era5_footprint_series(
    snapshot: Path,
    footprint: xr.DataArray,
    freq: str = "W",
    *,
    cube: xr.Dataset | None = None,
) -> pd.DataFrame:
    """Aggregate ERA5 to the footprint at the overpass window, composited to ``freq`` (REQ-010/011).

    Reads the dated ERA5 snapshot, regrids it onto the footprint mask's grid (nearest-neighbour, so
    the coarser ERA5 grid is aligned to the NO2 grid, EDGE-006), restricts to the overpass window,
    averages spatially over the footprint cells, and composites to the requested frequency.

    Returns a DataFrame indexed implicitly by a ``date`` column with one column per ERA5 variable
    (e.g. ``u10``, ``v10``, ``blh``) plus derived ``wind_speed``. The ``date`` is the period-end of
    the composite, matching the NOX-002b weekly cube convention.
    """
    snapshot = Path(snapshot)
    if not snapshot.exists():
        raise ERA5SnapshotError(
            f"ERA5 snapshot not found: {snapshot}. Run 'noxus ingest-era5' to fetch a snapshot "
            "from the Copernicus CDS, or disable meteo normalisation (ERR-002)."
        )
    ds = xr.open_dataset(snapshot).load()
    ds.close()

    ds = _overpass_window(ds)

    lon_name = next((c for c in ("longitude", "lon", "x") if c in ds.coords), "x")
    lat_name = next((c for c in ("latitude", "lat", "y") if c in ds.coords), "y")

    fp_lon = next((c for c in ("x", "lon", "longitude") if c in footprint.coords), "x")
    fp_lat = next((c for c in ("y", "lat", "latitude") if c in footprint.coords), "y")

    # Regrid ERA5 onto the footprint grid (nearest) so masking is on a common grid (EDGE-006).
    target = {lon_name: footprint[fp_lon].values, lat_name: footprint[fp_lat].values}
    ds = ds.reindex(target, method="nearest")
    ds = ds.rename({lon_name: fp_lon, lat_name: fp_lat})

    mask = footprint.astype(bool)
    masked = ds.where(mask)
    spatial_dims = [fp_lat, fp_lon]
    fp_mean = masked.mean(dim=spatial_dims, skipna=True)

    if "time" in fp_mean.coords and fp_mean["time"].size > 0:
        df = fp_mean.to_dataframe().reset_index()
        df = df[[c for c in df.columns if c in {"time", *ds.data_vars}]]
        df = df.set_index("time").resample(freq).mean()
        df.index.name = "date"
        df = df.reset_index()
    else:
        # No time axis — single record (rare; defensive).
        df = fp_mean.to_dataframe().reset_index()
        df["date"] = pd.NaT

    if {"u10", "v10"}.issubset(df.columns):
        df["wind_speed"] = np.hypot(df["u10"], df["v10"])
    return df.reset_index(drop=True)
