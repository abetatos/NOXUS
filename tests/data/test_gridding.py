"""Tests for NO2 gridding / temporal compositing (REQ-001..005, 010, 011, ERR-001/002)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from noxus.config.run import GriddingConfig
from noxus.data import gridding as G
from noxus.data.tropomi import NO2, _save_manifest


def _write_overpass(raw, date_str, value, ny=2, nx=2, cloud=0.1):
    opid = pd.Timestamp(date_str).strftime("%Y-%m-%dT%H%M%S")
    path = raw / opid[:4] / f"{opid}.nc"
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((ny, nx), float(value)) if np.isscalar(value) else np.asarray(value, float)
    ds = xr.Dataset(
        {NO2: (("y", "x"), arr), G.CLOUD: (("y", "x"), np.full((ny, nx), cloud))},
        coords={"time": pd.Timestamp(date_str), "y": np.arange(ny), "x": np.arange(nx)},
    )
    ds.to_netcdf(path, engine="h5netcdf")
    return opid, str(path)


def _store(raw, specs):
    raw.mkdir(parents=True, exist_ok=True)
    overpasses = {}
    for date_str, value in specs:
        opid, path = _write_overpass(raw, date_str, value)
        overpasses[opid] = {"path": path, "processor_version": "v2.x"}
    _save_manifest(raw, {"overpasses": overpasses, "batches_done": [], "batch_errors": {}})


def test_weekly_cube_equals_manual_means(tmp_path):
    raw = tmp_path / "tropomi"
    # Week of 06-05..06-11: values 1 and 3 -> mean 2; week 06-12..06-18: 10 and 20 -> mean 15.
    _store(
        raw, [("2023-06-05", 1.0), ("2023-06-07", 3.0), ("2023-06-12", 10.0), ("2023-06-14", 20.0)]
    )
    cube = G.composite(G.load_overpass_cube(raw), GriddingConfig())
    assert cube.sizes["time"] == 2
    vals = sorted(round(float(v), 3) for v in cube[NO2].mean(dim=["y", "x"]).values)
    assert vals == [2.0, 15.0]


def test_coverage_and_threshold_masking_no_interpolation(tmp_path):
    raw = tmp_path / "tropomi"
    # One dense week, and a sparse week where only 1 of 4 cells is valid (coverage 0.25).
    sparse = np.array([[5.0, np.nan], [np.nan, np.nan]])
    _store(raw, [("2023-06-05", 1.0), ("2023-06-07", 3.0), ("2023-06-12", sparse)])
    cfg = GriddingConfig(min_period_coverage=0.5)  # sparse week (0.25) must be masked
    cube = G.composite(G.load_overpass_cube(raw), cfg)
    # Dense week kept (coverage 1.0); sparse week fully masked.
    by_week = {
        pd.Timestamp(t).isocalendar().week: cube[NO2].sel(time=t) for t in cube["time"].values
    }
    dense = [w for w in by_week if by_week[w].notnull().any()]
    masked = [w for w in by_week if not by_week[w].notnull().any()]
    assert len(dense) == 1 and len(masked) == 1
    # No interpolation: the masked week has no values invented anywhere.
    assert float(cube[G.COVERAGE].max()) <= 1.0


def test_aoi_mean_series_is_spatial_mean_and_labelled(tmp_path):
    raw = tmp_path / "tropomi"
    _store(raw, [("2023-06-05", 1.0), ("2023-06-07", 3.0)])
    cube = G.composite(G.load_overpass_cube(raw), GriddingConfig())
    df = G.aoi_mean_series(cube)
    assert list(df.columns) == ["date", "no2_attributed", "valid_coverage", "kind"]
    assert df["no2_attributed"].iloc[0] == pytest.approx(2.0)
    assert (df["kind"] == "naive_aoi_mean_pre_attribution").all()


def test_build_cube_writes_outputs(tmp_path):
    raw = tmp_path / "tropomi"
    out = tmp_path / "no2"
    _store(raw, [("2023-06-05", 1.0), ("2023-06-07", 3.0)])
    rep = G.build_cube(GriddingConfig(raw_dir=raw, out_dir=out))
    assert rep.n_periods == 1
    assert rep.n_series_rows == 1
    assert (out / "no2_cube_w.nc").exists()
    assert (out / "no2_aoi_mean.parquet").exists()


def test_empty_store_raises(tmp_path):
    raw = tmp_path / "tropomi"
    raw.mkdir(parents=True)
    _save_manifest(raw, {"overpasses": {}, "batches_done": [], "batch_errors": {}})
    with pytest.raises(G.GriddingError):
        G.load_overpass_cube(raw)


def test_grid_mismatch_raises(tmp_path):
    raw = tmp_path / "tropomi"
    _store(raw, [("2023-06-05", 1.0)])
    # Add an overpass with a different grid shape.
    opid, path = _write_overpass(raw, "2023-06-07", 2.0, ny=3, nx=3)
    from noxus.data.tropomi import load_manifest

    m = load_manifest(raw)
    m["overpasses"][opid] = {"path": path, "processor_version": "v2.x"}
    _save_manifest(raw, m)
    with pytest.raises(G.GriddingError):
        G.load_overpass_cube(raw)


# --------------------------------------------------------------------------- cube completeness gate (NOX-003.1 ERR-103)


def _coverage_cube(dates, frac_valid=1.0):
    """A weekly cube with `coverage` present on the first `frac_valid` share of the given dates."""
    n = len(dates)
    cov = np.full((n, 2, 2), 0.8)
    n_nan = int(round((1 - frac_valid) * n))
    if n_nan:
        cov[-n_nan:] = np.nan  # trailing periods unobserved (partial fetch)
    return xr.Dataset(
        {
            G.NO2: (("time", "y", "x"), np.full((n, 2, 2), 1e-4)),
            G.COVERAGE: (("time", "y", "x"), cov),
        },
        coords={"time": pd.DatetimeIndex(dates), "y": np.arange(2), "x": np.arange(2)},
    )


def test_cube_completeness_full_is_one():
    dates = pd.date_range("2019-01-06", "2019-12-29", freq="W")
    cube = _coverage_cube(dates, frac_valid=1.0)
    frac = G.cube_completeness(
        cube, expected_start="2019-01-06", expected_end="2019-12-29", freq="W"
    )
    assert frac > 0.95


def test_assert_cube_complete_raises_on_partial():
    dates = pd.date_range("2019-01-06", "2019-12-29", freq="W")
    cube = _coverage_cube(dates, frac_valid=0.5)  # only half the weeks observed
    with pytest.raises(G.IncompleteCubeError, match="ERR-103"):
        G.assert_cube_complete(
            cube, expected_start="2019-01-06", expected_end="2019-12-29", freq="W"
        )


def test_assert_cube_complete_partial_allowed_when_not_required():
    dates = pd.date_range("2019-01-06", "2019-12-29", freq="W")
    cube = _coverage_cube(dates, frac_valid=0.5)
    frac = G.assert_cube_complete(
        cube,
        expected_start="2019-01-06",
        expected_end="2019-12-29",
        freq="W",
        require=False,
    )
    assert frac < 0.9  # measured, but no error raised (partial run proceeds, labelled partial)
