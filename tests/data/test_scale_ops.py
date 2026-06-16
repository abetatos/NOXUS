"""Tests for the NOX-008 cube scale operations: clip (AOI extent) and coarsen (resolution).

Both are pure re-aggregations of the committed cube — clip is a strict cell subset, coarsen is a
block-mean — never interpolation (REQ-001/002/003, EDGE-001, ERR-002/003).
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from noxus.config.region import Region
from noxus.data import gridding as G
from noxus.data.tropomi import NO2


def _cube(ny=4, nx=4, lon0=117.0, lat0=38.0, step=0.05, nan_block=None):
    """A 1-time synthetic cube on a regular grid; no2[y,x] = y*10 + x; coverage all 1.0."""
    lon = lon0 + step * np.arange(nx)
    lat = lat0 + step * np.arange(ny)
    field = (np.arange(ny)[:, None] * 10 + np.arange(nx)[None, :]).astype(float)
    if nan_block is not None:
        ys, xs = nan_block
        field[ys, xs] = np.nan
    cov = np.ones((ny, nx))
    return xr.Dataset(
        {NO2: (("time", "y", "x"), field[None]), G.COVERAGE: (("time", "y", "x"), cov[None])},
        coords={"time": np.array(["2023-06-04"], dtype="datetime64[ns]"), "y": lat, "x": lon},
    )


def test_clip_is_strict_subset_no_interpolation():
    cube = _cube()  # lon/lat 117.00..117.15 / 38.00..38.15
    region = Region("tight", min_lon=117.04, min_lat=38.04, max_lon=117.11, max_lat=38.11)
    clipped = G.clip_cube_to_region(cube, region)
    # Inner 2x2 cells (indices 1,2 on each axis) selected; fewer cells, exact original values kept.
    assert clipped.sizes["x"] == 2 and clipped.sizes["y"] == 2
    assert np.allclose(clipped["x"].values, [117.05, 117.10])
    assert np.allclose(clipped["y"].values, [38.05, 38.10])
    # Values unchanged (no averaging): cell (y=1,x=2) == 1*10 + 2 == 12.
    assert float(clipped[NO2].isel(time=0).sel(x=117.10, y=38.05)) == 12.0
    # Every retained value equals the original at that coordinate.
    for xv in clipped["x"].values:
        for yv in clipped["y"].values:
            got = float(clipped[NO2].isel(time=0).sel(x=xv, y=yv))
            assert got == float(cube[NO2].isel(time=0).sel(x=xv, y=yv))


def test_clip_outside_extent_raises():
    cube = _cube()
    far = Region("far", min_lon=120.0, min_lat=41.0, max_lon=121.0, max_lat=42.0)
    with pytest.raises(G.GriddingError):
        G.clip_cube_to_region(cube, far)


def test_coarsen_block_mean_and_realised_spacing():
    cube = _cube()  # native step 0.05
    coarse = G.coarsen_cube(cube, 0.10)  # factor 2 on each axis -> 2x2 coarse cells
    assert coarse.sizes["x"] == 2 and coarse.sizes["y"] == 2
    # Top-left coarse cell = mean of native block {0,1,10,11} = 5.5.
    assert float(coarse[NO2].isel(time=0, y=0, x=0)) == pytest.approx(5.5)
    # Realised spacing recorded (~0.10) and may be reported separately from the nominal target.
    assert coarse.attrs["coarsen_target_deg"] == 0.10
    assert coarse.attrs["coarsen_factor"] == [2, 2]
    assert coarse.attrs["coarsen_realised_deg"][0] == pytest.approx(0.10, abs=1e-6)


def test_coarsen_all_nan_block_is_nan():
    # Make the bottom-right 2x2 native block all-NaN -> its coarse cell is NaN.
    cube = _cube(nan_block=(slice(2, 4), slice(2, 4)))
    coarse = G.coarsen_cube(cube, 0.10)
    assert bool(np.isnan(float(coarse[NO2].isel(time=0, y=1, x=1))))
    # A fully-valid block stays finite.
    assert np.isfinite(float(coarse[NO2].isel(time=0, y=0, x=0)))


def test_coarsen_finer_than_native_refused():
    cube = _cube(step=0.05)
    with pytest.raises(ValueError, match="finer than native"):
        G.coarsen_cube(cube, 0.02)
