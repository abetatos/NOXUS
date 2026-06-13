"""ERA5 ingest + footprint aggregation tests (NOX-003 T5; AT3 ingest half, AT-ERR-1 ERA5).

Fully offline: the CDS retrieval is mocked via the ``_cds_fetch`` injection — no network, no
credentials. The real fetch happens out of band in T15.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from noxus.config.run import SignalConfig
from noxus.data import era5 as E


def _fake_cds_fetch(*, target, variables, area, years, months, days, hours, **_):
    """Stand-in for the CDS retrieval: write a CDS-style NetCDF (valid_time coord, short var names).

    Year-aware so per-year requests (the ingest chunks by year) produce distinct timestamps.
    """
    yr = int(years[0])
    times = pd.date_range(f"{yr}-06-04T05:00", periods=6, freq="D")
    lat = np.array([40.5, 40.0, 39.5, 39.0])  # descending
    lon = np.array([117.5, 118.0, 118.5, 119.0])
    shape = (len(times), len(lat), len(lon))
    rng = np.random.default_rng(1)
    ds = xr.Dataset(
        {
            "u10": (("valid_time", "latitude", "longitude"), rng.random(shape)),
            "v10": (("valid_time", "latitude", "longitude"), rng.random(shape)),
            "blh": (("valid_time", "latitude", "longitude"), 500 + rng.random(shape)),
        },
        coords={"valid_time": times, "latitude": lat, "longitude": lon},
    )
    ds.to_netcdf(target)
    return target


def test_ingest_era5_cds_server_side_subset(tmp_path):
    # ERA5 is sourced from the CDS; the injected fetch stands in for the server-side retrieval.
    cfg = SignalConfig(era5_snapshot_dir=tmp_path)
    out = E.ingest_era5(cfg, today=date(2023, 6, 12), _cds_fetch=_fake_cds_fetch)
    assert out.name == "era5_2023-06-12.nc"
    assert out.exists()
    snap = xr.open_dataset(out)
    # Short var names kept; valid_time normalised to a 'time' coord.
    assert set(cfg.era5_vars).issubset(set(snap.data_vars))
    assert "time" in snap.coords
    snap.close()
    # The raw intermediate is cleaned up.
    assert not (tmp_path / ".era5_cds_raw_2023-06-12.nc").exists()


def test_ingest_era5_cds_request_covers_aoi_and_overpass(tmp_path):
    # The CDS request carries the AOI bbox [N, W, S, E], the era years, and the overpass-window hours.
    captured = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return _fake_cds_fetch(**kwargs)

    cfg = SignalConfig(era5_snapshot_dir=tmp_path, era5_start="2023-01-01")
    E.ingest_era5(cfg, today=date(2023, 6, 12), _cds_fetch=_capture)
    assert captured["area"][0] > captured["area"][2]  # North > South
    assert captured["area"][3] > captured["area"][1]  # East > West
    assert captured["years"] == ["2023"]
    assert captured["hours"] == ["04:00", "05:00", "06:00"]  # overpass window ±1h around 05 UTC


def test_era5_footprint_series_aggregates_to_footprint(tmp_path):
    cfg = SignalConfig(era5_snapshot_dir=tmp_path)
    snapshot = E.ingest_era5(cfg, today=date(2023, 6, 12), _cds_fetch=_fake_cds_fetch)

    # A footprint mask on a grid that overlaps the AOI; mark the centre cells True.
    lon = np.array([118.0, 118.5, 119.0])
    lat = np.array([40.0, 39.5, 39.0])
    mask = np.zeros((len(lat), len(lon)), dtype=bool)
    mask[1, 1] = True
    footprint = xr.DataArray(mask, dims=("y", "x"), coords={"y": lat, "x": lon})

    df = E.era5_footprint_series(snapshot, footprint, freq="W")
    assert "date" in df.columns
    assert {"u10", "v10", "blh"}.issubset(df.columns)
    assert "wind_speed" in df.columns
    assert df["wind_speed"].notna().any()


def test_era5_footprint_series_missing_snapshot_raises(tmp_path):
    footprint = xr.DataArray(
        np.array([[True]]), dims=("y", "x"), coords={"y": [39.5], "x": [118.5]}
    )
    with pytest.raises(E.ERA5SnapshotError, match="ingest-era5"):
        E.era5_footprint_series(tmp_path / "nope.nc", footprint)
