"""Meteo regress-out, deseasonalise, relative index tests (NOX-003 T6/T7/T8; AT3/AT4/AT5).

Fully offline, deterministic. Statistical correctness is asserted only on planted synthetic series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from noxus.config.run import SignalConfig
from noxus.signal import index as I


# --------------------------------------------------------------------------- regress-out (AT3, REQ-011)


def test_regress_out_meteo_recovers_activity(planted_meteo_signal):
    signal, meteo, true_activity = planted_meteo_signal
    resid = I.regress_out_meteo(signal, meteo, form="linear")
    # The residual should track the planted activity (up to a constant the OLS intercept absorbs).
    resid_c = resid - resid.mean()
    act_c = true_activity - true_activity.mean()
    r = np.corrcoef(resid_c.to_numpy(), act_c.to_numpy())[0, 1]
    assert r > 0.95
    assert resid.attrs["meteo_form"] == "linear"
    assert "u10" in resid.attrs["meteo_covariates"]


def test_regress_out_meteo_records_loess_form(planted_meteo_signal):
    signal, meteo, _ = planted_meteo_signal
    resid = I.regress_out_meteo(signal, meteo, form="loess")
    assert resid.attrs["meteo_form"] == "loess"


def test_regress_out_meteo_preserves_nan_gaps(planted_meteo_signal):
    signal, meteo, _ = planted_meteo_signal
    signal = signal.copy()
    signal.iloc[5] = np.nan  # a cloud gap
    resid = I.regress_out_meteo(signal, meteo, form="linear")
    assert np.isnan(resid.iloc[5])  # never interpolated (REQ-004)


def test_regress_out_meteo_no_covariates_raises():
    s = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2021-01-03", periods=3, freq="W"))
    with pytest.raises(ValueError, match="covariates"):
        I.regress_out_meteo(s, pd.DataFrame(index=s.index))


# --------------------------------------------------------------------------- deseasonalise (AT4)


def test_deseasonalize_removes_seasonal_term(planted_seasonal_series):
    series, _ = planted_seasonal_series
    deseasoned = I.deseasonalize(series, method="yoy-double-diff")
    valid = deseasoned.dropna()
    # After yoy + double diff, the strong annual cycle should be gone -> small residual variance.
    assert valid.std() < series.std()
    assert deseasoned.attrs["deseason_method"] == "yoy-double-diff"


def test_deseasonalize_records_structural_terms(planted_seasonal_series):
    series, _ = planted_seasonal_series
    heating = I.heating_season_indicator(series.index, (11, 12, 1, 2, 3))
    curtailment = pd.Series(0.0, index=series.index)
    curtailment.iloc[50:55] = 1.0  # a planted curtailment window
    deseasoned = I.deseasonalize(
        series, method="yoy-double-diff", heating_season=heating, curtailment=curtailment
    )
    assert "heating_season" in deseasoned.attrs["structural_terms"]
    assert "curtailment" in deseasoned.attrs["structural_terms"]


def test_heating_season_indicator_marks_heating_months():
    idx = pd.to_datetime(["2021-01-03", "2021-07-04", "2021-12-05"])
    h = I.heating_season_indicator(idx, (11, 12, 1, 2, 3))
    assert list(h.to_numpy()) == [1.0, 0.0, 1.0]


def test_deseasonalize_yoy_removes_annual_cycle(planted_seasonal_series):
    # yoy (the default, gentler than double-diff): subtracting the year-prior value kills the cycle.
    series, _ = planted_seasonal_series
    deseasoned = I.deseasonalize(series, method="yoy")
    valid = deseasoned.dropna()
    assert valid.std() < series.std()
    assert deseasoned.attrs["deseason_method"] == "yoy"


def test_deseasonalize_stl_removes_seasonal_and_trend(planted_seasonal_series):
    # STL residual has the seasonal + trend removed -> smaller variance than the raw series.
    series, _ = planted_seasonal_series
    deseasoned = I.deseasonalize(series, method="stl")
    valid = deseasoned.dropna()
    assert len(valid) > 0
    assert valid.std() < series.std()
    assert deseasoned.attrs["deseason_method"] == "stl"


def test_deseasonalize_unknown_method_raises(planted_seasonal_series):
    series, _ = planted_seasonal_series
    with pytest.raises(ValueError, match="deseasonalisation method"):
        I.deseasonalize(series, method="bogus")


# --------------------------------------------------------------------------- relative index (AT5)


def test_build_index_is_unitless_zscore(planted_seasonal_series):
    series, _ = planted_seasonal_series
    df = I.build_index(series, anchor="zscore", attributable_cap=(0.30, 0.43))
    vals = df["index_value"].to_numpy()
    assert abs(np.nanmean(vals)) < 1e-9  # z-score is centred
    assert abs(np.nanstd(vals) - 1.0) < 1e-6  # and unit-variance
    assert df.attrs["attributable_cap"] == [0.30, 0.43]
    assert "index_value" in df.columns
    # No absolute-tonnage column anywhere (REQ-030).
    assert not any("ton" in c.lower() for c in df.columns)


def test_build_index_carries_provenance(planted_meteo_signal):
    signal, meteo, _ = planted_meteo_signal
    resid = I.regress_out_meteo(signal, meteo, form="linear")
    deseasoned = I.deseasonalize(resid, method="yoy-double-diff")
    df = I.build_index(deseasoned, anchor="zscore")
    assert df.attrs["meteo_form"] == "linear"
    assert df.attrs["deseason_method"] == "yoy-double-diff"


def test_write_and_read_index_provenance_roundtrip(tmp_path, planted_seasonal_series):
    series, _ = planted_seasonal_series
    df = I.build_index(series, anchor="zscore")
    out = I.write_index(df, tmp_path / "steel_activity_index.parquet", extra_provenance={"k": 1})
    prov = I.read_index_provenance(out)
    assert prov["anchor"] == "zscore"
    assert prov["attributable_cap"] == [0.30, 0.43]
    assert prov["k"] == 1


def test_build_index_anchor_period_is_relative(planted_seasonal_series):
    series, _ = planted_seasonal_series
    df = I.build_index(series, anchor="2020")  # index to the 2020 baseline
    # Indexed-to-baseline series is centred near 100 in the base year, strictly relative.
    base = df[df["date"].dt.year == 2020]["index_value"]
    assert abs(base.mean() - 100.0) < 5.0


# --------------------------------------------------------------------------- end-to-end orchestration


def test_build_activity_index_missing_signal_raises(tmp_path):
    cfg = SignalConfig(out_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="attribute"):
        I.build_activity_index(cfg, use_meteo=False)


def test_build_activity_index_no_meteo_end_to_end(tmp_path, planted_seasonal_series):
    series, _ = planted_seasonal_series
    cfg = SignalConfig(out_dir=tmp_path)
    # Plant a footprint signal parquet matching the attribution-stage contract.
    fp = pd.DataFrame(
        {
            "date": series.index,
            "no2_footprint": series.to_numpy() + 4.0,
            "no2_bg": np.full(len(series), 4.0),
            "no2_corrected": series.to_numpy(),
            "valid_coverage": np.full(len(series), 0.9),
        }
    )
    fp.to_parquet(tmp_path / cfg.footprint_signal_name, index=False)

    out = I.build_activity_index(cfg, use_meteo=False)
    assert out.exists()
    idx_df = pd.read_parquet(out)
    assert "index_value" in idx_df.columns
    prov = I.read_index_provenance(out)
    assert prov["deseason_method"] == "yoy"  # the new default (double-diff erased the signal)
    assert "heating_season" in prov["structural_terms"]
