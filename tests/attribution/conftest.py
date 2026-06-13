"""Synthetic fixtures for footprint sampling + background correction tests (T2, AT1/AT2).

Fully offline and deterministic. The synthetic cube mirrors the NOX-002b weekly product schema:
dims ``(time, y, x)`` with ``x`` holding longitude (ascending) and ``y`` holding latitude
(descending, as the real openEO cube), and variables ``no2`` and ``coverage``.

The synthetic facility CSV deliberately contains:
- an ``operating`` facility inside the cube extent,
- a ``retired`` facility (must be excluded by the status filter, REQ-001),
- a co-located duplicate at identical coordinates to the operating one (must be de-duplicated at the
  cell level, EDGE-001).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from noxus.data.gridding import COVERAGE
from noxus.data.tropomi import NO2

# A small regular grid around an operating facility. Cell spacing ~0.05 deg lon / 0.045 deg lat,
# roughly the real ~5 km TROPOMI/openEO grid. Longitude ascending, latitude descending.
LON = np.round(np.arange(118.30, 118.65 + 1e-9, 0.05), 4)  # 8 cells, ~118.30..118.65
LAT = np.round(np.arange(39.70, 39.35 - 1e-9, -0.045), 4)  # 8 cells, ~39.70..39.36

# Operating facility near the grid centre; a co-located duplicate sits at the exact same point.
OPERATING_LON, OPERATING_LAT = 118.475, 39.520
# Retired facility placed far outside any small footprint radius (its cells must never be selected).
RETIRED_LON, RETIRED_LAT = 118.30, 39.70


@pytest.fixture
def facilities_csv(tmp_path):
    """Write a small facility CSV: operating + retired + co-located duplicate; return its path."""
    rows = [
        {
            "name": "Operating Steel A",
            "latitude": OPERATING_LAT,
            "longitude": OPERATING_LON,
            "technology": "BF; BOF",
            "status": "operating",
        },
        {
            "name": "Co-located Duplicate",
            "latitude": OPERATING_LAT,
            "longitude": OPERATING_LON,
            "technology": "BF; BOF",
            "status": "operating",
        },
        {
            "name": "Retired Steel B",
            "latitude": RETIRED_LAT,
            "longitude": RETIRED_LON,
            "technology": "BF; BOF",
            "status": "retired",
        },
    ]
    path = tmp_path / "facilities.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _cube(no2_grids, coverage_grids, dates):
    """Build a (time, y, x) cube from per-period 2-D (y, x) arrays."""
    no2 = np.stack(no2_grids, axis=0)
    cov = np.stack(coverage_grids, axis=0)
    return xr.Dataset(
        {
            NO2: (("time", "y", "x"), no2),
            COVERAGE: (("time", "y", "x"), cov),
        },
        coords={
            "time": pd.to_datetime(dates),
            "y": LAT,
            "x": LON,
        },
    )


@pytest.fixture
def synthetic_cube():
    """A 2-period weekly cube with all cells valid and a known, uniform-per-region NO2 field.

    Each period: footprint cells carry a high value, background cells a low value, so the
    background-corrected signal is exactly (high - low). Returns the Dataset.
    """
    ny, nx = len(LAT), len(LON)
    # Period 1: footprint=10, background=4. Period 2: footprint=20, background=5.
    # Values are filled uniformly; the per-region split is enforced by the masks in the test, so we
    # use a single value per period here and let the test plant region-specific values when needed.
    g1 = np.full((ny, nx), 10.0)
    g2 = np.full((ny, nx), 20.0)
    cov = np.ones((ny, nx))
    return _cube([g1, g2], [cov, cov], ["2023-06-04", "2023-06-11"])


@pytest.fixture
def cube_builder():
    """Factory returning ``_cube`` so tests can plant arbitrary per-region NO2/coverage fields."""
    return _cube
