"""Explicit emission-intensity (decoupling) model tests (NOX-003.1 T3/T4; AT101-AT103, AT-ERR-101).

Fully offline, deterministic. Statistical behaviour is asserted only on planted synthetic series — the
benchmark is never used to select the trend smoothness (the central anti-p-hacking guarantee).
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from noxus.signal import index as I
from noxus.signal import intensity as IT


# --------------------------------------------------------------------------- trend + residual (AT101)


def test_fit_recovers_trend_and_residual_and_sign_inversion(planted_decomposition):
    signal, benchmark, true_trend, true_osc = planted_decomposition
    fit = IT.fit_intensity_trend(signal, df_grid=(2, 3, 4, 6, 8, 10), cv_folds=5)

    # The fitted secular trend recovers the planted declining intensity.
    r_trend = np.corrcoef(fit.trend.to_numpy(), true_trend.to_numpy())[0, 1]
    assert r_trend > 0.95

    # The residual recovers the planted activity oscillation.
    r_resid = np.corrcoef(fit.residual.to_numpy(), true_osc.to_numpy())[0, 1]
    assert r_resid > 0.8

    # Sign inversion: levels fall as activity rises (corr < 0); residual tracks activity (corr > 0).
    levels_r = np.corrcoef(signal.to_numpy(), benchmark.to_numpy())[0, 1]
    resid_r = np.corrcoef(fit.residual.to_numpy(), benchmark.to_numpy())[0, 1]
    assert levels_r < 0 < resid_r


def test_residual_preserves_nan_gaps(planted_decomposition):
    signal, *_ = planted_decomposition
    signal = signal.copy()
    signal.iloc[10] = np.nan  # a cloud gap
    fit = IT.fit_intensity_trend(signal, df_grid=(2, 3, 4), cv_folds=4)
    assert np.isnan(fit.trend.iloc[10])  # never interpolated (REQ-004)
    assert np.isnan(fit.residual.iloc[10])


# --------------------------------------------------------------------------- CV selection (AT102)


def test_cv_selects_low_df_for_linear_trend(planted_decomposition):
    # The planted trend is linear -> CV should pick a small effective df (not chase the oscillation).
    signal, *_ = planted_decomposition
    fit = IT.fit_intensity_trend(signal, df_grid=(2, 3, 4, 6, 8, 12), cv_folds=5)
    assert fit.df <= 4
    assert fit.criterion == "blocked-cv"
    assert fit.estimator == "spline"
    assert np.isfinite(fit.cv_score)


def test_selection_never_sees_the_benchmark():
    # The anti-p-hacking guarantee (NFR-102): the fit signature has no benchmark/target parameter.
    params = set(inspect.signature(IT.fit_intensity_trend).parameters)
    for forbidden in ("benchmark", "target", "y_true", "crea"):
        assert forbidden not in params


def test_gcv_criterion_spline(planted_decomposition):
    signal, *_ = planted_decomposition
    fit = IT.fit_intensity_trend(signal, df_grid=(2, 3, 4, 6), criterion="gcv")
    assert fit.criterion == "gcv"
    assert fit.df <= 6


# --------------------------------------------------------------------------- sensitivity sweep (AT103)


def test_sweep_returns_row_per_df_with_levels(planted_decomposition):
    signal, benchmark, *_ = planted_decomposition
    grid = (2, 3, 4, 6, 8)
    sweep = IT.smoothness_sweep(signal, benchmark, df_grid=grid, max_lag=6)
    assert list(sweep["df"]) == [float(d) for d in grid]
    assert {"df", "residual_r", "residual_p", "peak_lag", "peak_r", "levels_r"} <= set(
        sweep.columns
    )
    # levels_r is the single raw-levels correlation, identical across rows.
    assert sweep["levels_r"].nunique() == 1
    assert sweep["levels_r"].iloc[0] < 0  # decoupling sign in levels


def test_sweep_stays_flat_on_planted_null(planted_null_decomposition):
    signal, benchmark = planted_null_decomposition
    sweep = IT.smoothness_sweep(signal, benchmark, df_grid=(2, 3, 4, 6, 8, 10), max_lag=6)
    # No smoothing choice manufactures a residual correlation on an independent benchmark.
    assert (sweep["residual_r"].abs() < 0.3).all()


# --------------------------------------------------------------------------- errors (AT-ERR-101)


def test_short_series_raises(planted_decomposition):
    signal, *_ = planted_decomposition
    short = signal.iloc[:10]
    with pytest.raises(IT.IntensityModelError, match="min_length"):
        IT.fit_intensity_trend(short, min_length=24)


def test_empty_grid_raises(planted_decomposition):
    signal, *_ = planted_decomposition
    with pytest.raises(IT.IntensityModelError, match="df_grid"):
        IT.fit_intensity_trend(signal, df_grid=[])


def test_one_fold_raises(planted_decomposition):
    signal, *_ = planted_decomposition
    with pytest.raises(IT.IntensityModelError, match="cv_folds"):
        IT.fit_intensity_trend(signal, cv_folds=1)


# --------------------------------------------------------------------------- wiring into deseasonalize


def test_deseasonalize_intensity_model_records_attrs(planted_decomposition):
    signal, *_ = planted_decomposition
    out = I.deseasonalize(signal, method="intensity-model")
    assert out.attrs["deseason_method"] == "intensity-model"
    assert "intensity_df" in out.attrs
    assert out.attrs["intensity_estimator"] == "spline"
    # The residual is the activity proxy on the same index, gaps preserved.
    assert len(out) == len(signal)


def test_deseasonalize_unknown_method_message_lists_intensity():
    s = pd.Series(np.arange(30.0), index=pd.date_range("2021-01-03", periods=30, freq="W"))
    with pytest.raises(ValueError, match="intensity-model"):
        I.deseasonalize(s, method="bogus")
