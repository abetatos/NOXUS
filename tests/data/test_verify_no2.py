"""Tests for the NO2 verification (REQ-010, REQ-011)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from noxus.config.region import tangshan_aoi
from noxus.data import tropomi as T
from noxus.data.verify_no2 import render_day, select_clear_high_days


def test_select_clear_high_days_filters_and_ranks(tmp_path):
    raw = tmp_path / "tropomi"
    raw.mkdir(parents=True)
    T._save_manifest(
        raw,
        {
            "overpasses": {
                "A": {"cloud_mean": 0.1, "no2_mean": 5.0, "valid_coverage": 0.9, "path": "a.nc"},
                "B": {"cloud_mean": 0.1, "no2_mean": 2.0, "valid_coverage": 0.9, "path": "b.nc"},
                "C": {"cloud_mean": 0.5, "no2_mean": 9.0, "valid_coverage": 0.9, "path": "c.nc"},
                "D": {"cloud_mean": 0.1, "no2_mean": 8.0, "valid_coverage": 0.2, "path": "d.nc"},
            },
            "batches_done": [],
            "batch_errors": {},
        },
    )
    picks = select_clear_high_days(raw, n=5, max_cloud=0.2, min_coverage=0.5)
    ids = [p["id"] for p in picks]
    assert ids == ["A", "B"]  # C cloudy, D low coverage; A>B by NO2


def test_render_day_writes_png_with_facilities(tmp_path):
    # A small NO2 field.
    ny, nx = 6, 6
    ds = xr.Dataset(
        {T.NO2: (("y", "x"), np.random.default_rng(0).random((ny, nx)))},
        coords={"y": np.arange(ny), "x": np.arange(nx)},
    )
    nc = tmp_path / "overpass.nc"
    ds.to_netcdf(nc, engine="h5netcdf")

    fac = tmp_path / "fac.csv"
    pd.DataFrame(
        {"name": ["p1", "p2"], "latitude": [39.6, 39.9], "longitude": [118.2, 118.6]}
    ).to_csv(fac, index=False)

    out = render_day(nc, fac, tmp_path / "out.png", aoi=tangshan_aoi())
    assert out.exists()
    assert out.stat().st_size > 0
