"""Tests for the NOX-008 spatial-scale sweep driver (REQ-010/011/040/041; AT6)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from noxus.config.run import ScaleSweepConfig
from noxus.data.tropomi import NO2
from noxus.data import gridding as G
from noxus.validation import scale as S


@pytest.fixture
def synthetic_cube():
    """A weekly cube over the real Tangshan AOI extent so the footprint has cells."""
    rng = np.random.default_rng(0)
    lon = np.round(np.arange(117.70, 119.35, 0.055), 4)
    lat = np.round(np.arange(38.70, 40.45, 0.035), 4)
    dates = pd.date_range("2019-01-06", periods=90, freq="W")
    nt, ny, nx = len(dates), len(lat), len(lon)
    # Mildly autocorrelated field so the robust tests have something to chew on.
    base = np.cumsum(rng.standard_normal((nt, 1, 1)), axis=0) * 1e-4
    no2 = base + rng.standard_normal((nt, ny, nx)) * 1e-4 + 5e-4
    cov = np.ones((nt, ny, nx))
    return xr.Dataset(
        {NO2: (("time", "y", "x"), no2), G.COVERAGE: (("time", "y", "x"), cov)},
        coords={"time": dates, "y": lat, "x": lon},
    )


@pytest.fixture
def benchmark(synthetic_cube):
    rng = np.random.default_rng(1)
    dates = pd.to_datetime(synthetic_cube["time"].values)
    return pd.Series(np.cumsum(rng.standard_normal(len(dates))) + 80.0, index=dates)


def _fast_cfg():
    return ScaleSweepConfig(
        buffers=(0.25, 0.10),
        resolutions=("native", 0.10),
        variants=("level", "yoy"),
        freqs=("W",),
        n_boot=60,
        n_perm=60,
        min_overlap=20,
    )


def test_sweep_table_has_columns_and_is_deterministic(synthetic_cube, benchmark):
    cfg = _fast_cfg()
    df1 = S.scale_sweep(synthetic_cube, benchmark, cfg)
    assert not df1.empty
    expected = {
        "buffer",
        "resolution",
        "realised_deg",
        "signal",
        "n_cells",
        "freq",
        "variant",
        "lag",
        "lag_kind",
        "n",
        "r",
        "p_naive",
        "n_eff_first",
        "p_eff_first",
        "n_eff_nw",
        "p_eff_nw",
        "boot_lo",
        "boot_hi",
        "p_perm",
        "p_fdr",
        "fdr_reject",
        "verdict",
    }
    assert expected.issubset(df1.columns)
    # Both extents and both resolutions appear.
    assert set(df1["buffer"]) == {0.25, 0.10}
    assert set(df1["resolution"]) == {"native", "0.1"}
    # lag0 family is always present; verdicts are from the allowed vocabulary.
    assert "lag0" in set(df1["lag_kind"])
    assert set(df1["verdict"]).issubset({"robust", "fragile (naive-only)", "ns"})
    # Determinism: identical seed/config reproduces the table exactly.
    df2 = S.scale_sweep(synthetic_cube, benchmark, cfg)
    pd.testing.assert_frame_equal(df1, df2)


def test_coarser_resolution_records_realised_spacing(synthetic_cube, benchmark):
    cfg = _fast_cfg()
    df = S.scale_sweep(synthetic_cube, benchmark, cfg)
    coarse = df[df["resolution"] == "0.1"]
    # Native ~0.055/0.035; coarsening to 0.1 deg yields a realised spacing coarser than native.
    assert (coarse["realised_deg"] > 0.05).all()


def test_aoi_mean_fallback_labelled_when_too_few_cells(synthetic_cube):
    from noxus.attribution.source import load_facilities

    facilities = load_facilities(ScaleSweepConfig().facilities_csv)
    cfg = replace(ScaleSweepConfig(), min_footprint_cells=100000)  # force the fallback
    sig = S.derive_scale_signal(synthetic_cube, 0.10, "native", facilities, cfg)
    assert sig.label == "aoi-mean-fallback"
    assert sig.series.notna().any()
