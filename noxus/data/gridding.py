"""Composite per-overpass NO2 into an analysis-ready cube (NOX-002b).

The NOX-002a store is already on a regular ~0.05x0.035 deg grid (openEO), so this is **temporal
compositing**, not swath oversampling: stack the per-overpass AOI grids into a time cube, resample to a
target frequency (weekly default) by the mean over valid cells, track valid-coverage, and apply a
minimum-coverage threshold — masking gaps, never interpolating (cloud decision 2026-06-13). Fine ~1 km
oversampling would need raw L2 and is out of scope (deferred to NOX-003).

Also emits an interim AOI-mean series (naive spatial mean, explicitly pre-attribution) matching the
validation predictor contract, to allow a first end-to-end smoke of the validation engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from noxus.config.run import GriddingConfig
from noxus.data.tropomi import NO2, load_manifest

CLOUD = "cloud"
COVERAGE = "coverage"


class GriddingError(RuntimeError):
    """Gridding could not complete (no input, or irreconcilable grids)."""


class IncompleteCubeError(RuntimeError):
    """The cube is too sparse/short for the full-series re-run (NOX-003.1 ERR-103)."""


def cube_completeness(cube: xr.Dataset, *, expected_start, expected_end, freq: str = "W") -> float:
    """Fraction of expected periods that carry any valid coverage (NOX-003.1 REQ-120/121).

    Compares the count of cube periods with finite ``coverage`` against the number of ``freq`` periods
    spanning ``[expected_start, expected_end]``. 1.0 means a period exists for every expected slot;
    a partial fetch (sparse weeks) scores lower. Used to gate the full-series re-run.
    """
    expected = pd.period_range(pd.Timestamp(expected_start), pd.Timestamp(expected_end), freq=freq)
    if len(expected) == 0:
        return 0.0
    if COVERAGE in cube.data_vars:
        cov = cube[COVERAGE]
        dims = [d for d in cov.dims if d != "time"]
        per_period = cov.notnull().any(dim=dims) if dims else cov.notnull()
        n_valid = int(per_period.sum())
    else:
        n_valid = int(cube.sizes.get("time", 0))
    return min(n_valid / len(expected), 1.0)


def assert_cube_complete(
    cube: xr.Dataset,
    *,
    expected_start,
    expected_end,
    freq: str = "W",
    min_fraction: float = 0.9,
    require: bool = True,
) -> float:
    """Refuse a partial cube for the full-series re-run when ``require`` is set (ERR-103).

    Returns the completeness fraction. When ``require`` and the fraction is below ``min_fraction``,
    raises :class:`IncompleteCubeError` naming the command that completes the fetch — so a partial run
    is never silently reported as a full-series result (NOX-003.1 AC-106). With ``require=False`` it is
    a pure measurement (the partial run proceeds, labelled partial).
    """
    fraction = cube_completeness(
        cube, expected_start=expected_start, expected_end=expected_end, freq=freq
    )
    if require and fraction < min_fraction:
        raise IncompleteCubeError(
            f"Cube covers only {fraction:.0%} of expected {freq} periods "
            f"(< {min_fraction:.0%}); the TROPOMI fetch is incomplete. Run 'noxus fetch' to finish "
            "acquisition then 'noxus grid' to rebuild before the full-series re-run (ERR-103). "
            "Use require=False to run on the partial series (labelled partial)."
        )
    return fraction


def load_overpass_cube(raw_dir: Path | str) -> xr.Dataset:
    """Stack the per-overpass NetCDFs (from the manifest) into a time cube on a common grid (REQ-001)."""
    raw_dir = Path(raw_dir)
    manifest = load_manifest(raw_dir)
    entries = manifest.get("overpasses", {})
    if not entries:
        raise GriddingError(
            f"No overpasses found in {raw_dir} (manifest empty). Run 'noxus fetch' first."
        )

    datasets = []
    shapes = set()
    for opid, entry in entries.items():
        path = Path(entry.get("path", raw_dir / f"{opid[:4]}" / f"{opid}.nc"))
        if not path.exists():
            continue
        ds = xr.open_dataset(path).load()
        ds.close()
        if "time" not in ds.dims:
            ds = ds.expand_dims("time")
        keep = [v for v in (NO2, CLOUD) if v in ds.data_vars]
        ds = ds[keep]
        shapes.add((ds.sizes.get("y"), ds.sizes.get("x")))
        datasets.append(ds)

    if not datasets:
        raise GriddingError(f"Manifest lists overpasses but no files exist under {raw_dir}.")
    if len(shapes) > 1:
        raise GriddingError(
            f"Overpass grids are not uniform ({sorted(shapes)}); resampling to a common target "
            "grid is not implemented (acquire with one AOI/buffer, or extend gridding)."
        )
    return xr.concat(datasets, dim="time").sortby("time")


def composite(cube: xr.Dataset, cfg: GriddingConfig) -> xr.Dataset:
    """Resample to cfg.freq by mean over valid cells; add coverage; mask below thresholds (REQ-002..004)."""
    grouped = cube[NO2].resample(time=cfg.freq)
    no2_mean = grouped.mean()  # skipna -> mean over valid observations
    valid = cube[NO2].notnull().resample(time=cfg.freq).sum()  # valid overpasses per cell-period
    total = xr.ones_like(cube[NO2]).resample(time=cfg.freq).sum()  # overpasses per period
    coverage_cell = (valid / total).where(total > 0, 0.0)

    # Cell-level mask: require a minimum number of valid overpasses in the cell-period.
    no2_mean = no2_mean.where(valid >= cfg.min_cell_obs)

    # Period-level mask: require a minimum fraction of AOI cells valid, else drop the whole period.
    period_coverage = (valid >= cfg.min_cell_obs).mean(dim=["y", "x"])
    no2_mean = no2_mean.where(period_coverage >= cfg.min_period_coverage)

    out = xr.Dataset({NO2: no2_mean, COVERAGE: coverage_cell})
    out["period_coverage"] = period_coverage
    out.attrs.update(
        {
            "freq": cfg.freq,
            "min_cell_obs": cfg.min_cell_obs,
            "min_period_coverage": cfg.min_period_coverage,
            "note": "temporal composite of openEO-gridded TROPOMI NO2; no interpolation across gaps",
        }
    )
    return out


def aoi_mean_series(composited: xr.Dataset) -> pd.DataFrame:
    """Naive AOI spatial-mean NO2 per period — pre-attribution stand-in for the predictor (REQ-010/011)."""
    no2_aoi = composited[NO2].mean(dim=["y", "x"], skipna=True)
    cov = composited[COVERAGE].mean(dim=["y", "x"], skipna=True)
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(composited["time"].values),
            "no2_attributed": no2_aoi.values,
            "valid_coverage": cov.values,
        }
    )
    df = df.dropna(subset=["no2_attributed"]).reset_index(drop=True)
    # Unmistakable label: this is a spatial mean, NOT source attribution (that is NOX-003).
    df["kind"] = "naive_aoi_mean_pre_attribution"
    return df


def _cube_lonlat(cube: xr.Dataset) -> tuple[str, str]:
    """Return the (lon_name, lat_name) coordinate names of the cube (x/y, or geographic aliases)."""
    lon = next((c for c in ("x", "lon", "longitude") if c in cube.coords), None)
    lat = next((c for c in ("y", "lat", "latitude") if c in cube.coords), None)
    if lon is None or lat is None:
        raise GriddingError(
            f"Cube has no recognisable lon/lat coordinates (expected x/y); found {list(cube.coords)}."
        )
    return lon, lat


def _native_spacing(cube: xr.Dataset, name: str) -> float:
    """Median absolute step of a 1-D coordinate, in degrees (the native cell spacing)."""
    vals = np.asarray(cube[name].values, dtype=float)
    if vals.size < 2:
        return float("nan")
    return float(np.median(np.abs(np.diff(vals))))


def clip_cube_to_region(cube: xr.Dataset, region) -> xr.Dataset:
    """Clip the cube to the cells whose centres fall inside ``region`` — a strict subset (NOX-008 REQ-001).

    No interpolation or resampling: cells are selected by their existing centre coordinates, so the
    0.25 deg -> 0.10 deg tight-AOI comparison is a pure subset of the same grid. Raises
    :class:`GriddingError` (ERR-003) if no cell overlaps the region.
    """
    lon_name, lat_name = _cube_lonlat(cube)
    lon = np.asarray(cube[lon_name].values, dtype=float)
    lat = np.asarray(cube[lat_name].values, dtype=float)
    xm = np.where((lon >= region.min_lon) & (lon <= region.max_lon))[0]
    ym = np.where((lat >= region.min_lat) & (lat <= region.max_lat))[0]
    if xm.size == 0 or ym.size == 0:
        raise GriddingError(
            f"AOI clip to region '{getattr(region, 'name', '?')}' "
            f"({region.as_bbox()}) yields no overlapping cells; cube spans "
            f"lon [{lon.min():.3f}, {lon.max():.3f}], lat [{lat.min():.3f}, {lat.max():.3f}] (ERR-003)."
        )
    out = cube.isel({lon_name: xm, lat_name: ym})
    out.attrs = {
        **cube.attrs,
        "aoi_clip": getattr(region, "name", "region"),
        "aoi_bbox": list(region.as_bbox()),
    }
    return out


def coarsen_cube(cube: xr.Dataset, target_deg: float) -> xr.Dataset:
    """Block-average the cube to ~``target_deg`` cell spacing — aggregation, never interpolation (NOX-008 REQ-002/003).

    Coarsening by a block-mean over non-overlapping native-cell blocks is the *information-preserving*
    direction (the 2026-06-13 native-resolution decision rejected interpolating to a FINER grid). NaN
    (cloud-gapped) cells are skipped; an all-NaN block is NaN; ``coverage`` is propagated as the block
    mean. The realised cell spacing (which may differ from the nominal target when blocks don't divide
    evenly, EDGE-001) and the block factors are recorded in ``.attrs``. Raises ``ValueError`` (ERR-002)
    when ``target_deg`` is finer than native (no upsampling).
    """
    lon_name, lat_name = _cube_lonlat(cube)
    dx = _native_spacing(cube, lon_name)
    dy = _native_spacing(cube, lat_name)
    fx = int(round(target_deg / dx)) if dx > 0 else 1
    fy = int(round(target_deg / dy)) if dy > 0 else 1
    if fx < 1 or fy < 1:
        raise ValueError(
            f"Target resolution {target_deg} deg is finer than native (~{max(dx, dy):.4f} deg); "
            "upsampling/interpolation is refused (native-resolution decision, 2026-06-13) (ERR-002)."
        )
    if fx == 1 and fy == 1:
        out = cube.copy()
    else:
        out = cube.coarsen({lon_name: fx, lat_name: fy}, boundary="trim").mean(skipna=True)
    out.attrs = {
        **cube.attrs,
        "coarsen_target_deg": float(target_deg),
        "coarsen_factor": [fx, fy],
        "coarsen_native_deg": [round(dx, 5), round(dy, 5)],
        "coarsen_realised_deg": [
            round(_native_spacing(out, lon_name), 5),
            round(_native_spacing(out, lat_name), 5),
        ],
    }
    return out


@dataclass(frozen=True)
class GridReport:
    """Summary of a gridding run."""

    n_periods: int
    n_series_rows: int
    cube_path: str
    series_path: str | None


def _write_cube(cube: xr.Dataset, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cube = cube.copy()
    for var in list(cube.variables):
        cube[var].encoding = {}
    cube.to_netcdf(path, engine="h5netcdf")
    return path


def build_cube(cfg: GriddingConfig | None = None) -> GridReport:
    """End-to-end: load overpasses -> composite -> write the cube (+ interim AOI series) (REQ-005)."""
    cfg = cfg or GriddingConfig()
    cube = load_overpass_cube(cfg.raw_dir)
    composited = composite(cube, cfg)

    out_dir = Path(cfg.out_dir)
    cube_path = _write_cube(composited, out_dir / f"no2_cube_{cfg.freq.lower()}.nc")

    series_path = None
    n_rows = 0
    if cfg.emit_aoi_series:
        series = aoi_mean_series(composited)
        n_rows = len(series)
        series_path = out_dir / "no2_aoi_mean.parquet"
        series.to_parquet(series_path, index=False)

    return GridReport(
        n_periods=int(composited.sizes.get("time", 0)),
        n_series_rows=n_rows,
        cube_path=str(cube_path),
        series_path=str(series_path) if series_path else None,
    )
