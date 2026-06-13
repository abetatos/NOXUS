"""Validation tests: align, sign, r/p, lead-lag, report, null (NOX-003 T9/T10/T11; AT6/AT7/AT-ERR-1).

The planted-signal / planted-null pair is the headline guarantee: the engine must recover a real
signal AND honestly report a null. Fully offline, deterministic.
"""

from __future__ import annotations

import pandas as pd
import pytest

from noxus.validation import leadlag as L
from noxus.validation import report as R
from noxus.validation.preprocess import AlignmentError, align_series


# --------------------------------------------------------------------------- align (REQ-040)


def test_align_series_accepts_index_value_column(planted_signal_pair):
    index_df, bench_df, _ = planted_signal_pair
    aligned = align_series(index_df, bench_df, freq="W", min_coverage=0.25)
    assert list(aligned.columns) == ["index", "benchmark"]
    assert len(aligned) > 0
    assert aligned.notna().all().all()  # inner join drops the leading NaN benchmark weeks


def test_align_series_coverage_screen_drops_low_cov(planted_signal_pair):
    index_df, bench_df, _ = planted_signal_pair
    index_df = index_df.copy()
    index_df.loc[index_df.index[:10], "valid_coverage"] = 0.05  # below floor
    aligned = align_series(index_df, bench_df, freq="W", min_coverage=0.25)
    full = align_series(planted_signal_pair[0], bench_df, freq="W", min_coverage=0.25)
    assert len(aligned) < len(full)


def test_align_series_missing_predictor_column_raises():
    bad = pd.DataFrame({"date": pd.date_range("2021-01-03", periods=3, freq="W"), "foo": [1, 2, 3]})
    bench = pd.DataFrame(
        {"date": pd.date_range("2021-01-03", periods=3, freq="W"), "value": [1, 2, 3]}
    )
    with pytest.raises(AlignmentError, match="predictor"):
        align_series(bad, bench)


# --------------------------------------------------------------------------- planted signal (AT6a)


def test_planted_signal_recovers_sign_lag_and_band(planted_signal_pair):
    index_df, bench_df, k = planted_signal_pair
    aligned = align_series(index_df, bench_df, freq="W")
    idx, bench = aligned["index"], aligned["benchmark"]

    sign = L.verify_sign(idx, bench)
    assert sign.sign == "positive"
    assert sign.significant

    ccf = L.lead_lag(idx, bench, max_lag=8)
    assert ccf.peak_lag == k  # the index leads the benchmark by k weeks

    corr = L.correlate(idx.shift(k), bench)
    assert corr.pearson_r > 0.5  # in / above the literature band on the lead-aligned series
    assert corr.p_value < 0.05
    assert corr.ci_low == corr.ci_low  # a CI is reported (not NaN)


# --------------------------------------------------------------------------- planted null (AT6b)


def test_planted_null_reports_null(planted_null_pair):
    index_df, bench_df = planted_null_pair
    aligned = align_series(index_df, bench_df, freq="W")
    idx, bench = aligned["index"], aligned["benchmark"]

    corr = L.correlate(idx, bench)
    assert abs(corr.pearson_r) < 0.3  # near zero
    assert corr.p_value > 0.05  # not significant

    sign = L.verify_sign(idx, bench)
    assert sign.sign == "indeterminate"  # never silently flipped to positive (EDGE-007)


# --------------------------------------------------------------------------- report (AT7)


def test_report_planted_signal_concludes_lead(planted_signal_pair, tmp_path):
    index_df, bench_df, k = planted_signal_pair
    art = R.report(
        index_df,
        bench_df,
        max_lag=8,
        min_overlap=10,
        config_echo={"deseason_method": "yoy-double-diff", "meteo_form": "linear"},
        out_dir=tmp_path,
    )
    assert art.results["conclusion"] == "lead"
    assert art.results["peak_lag"] == k
    assert art.results["sign"] == "positive"
    assert art.results["bar_class"] in {"in-band", "above-bar"}
    assert art.results_path.exists()
    assert art.summary_path.exists()
    summary = art.summary_path.read_text(encoding="utf-8")
    assert "LEAD" in summary
    assert "deseason_method" in summary
    assert "attributable_cap" in summary


def test_report_planted_null_states_null_explicitly(planted_null_pair, tmp_path):
    index_df, bench_df = planted_null_pair
    art = R.report(index_df, bench_df, max_lag=8, min_overlap=10, out_dir=tmp_path)
    assert art.results["conclusion"] == "null"
    summary = art.summary_path.read_text(encoding="utf-8")
    assert "NULL" in summary
    assert "Morris" in summary  # cites the honest-null literature


def test_report_echoes_config_and_bar(planted_signal_pair, tmp_path):
    index_df, bench_df, _ = planted_signal_pair
    echo = {
        "deseason_method": "yoy-double-diff",
        "meteo_covariates": ["u10", "v10", "blh"],
        "footprint_geometry": 15.0,
        "attributable_cap": [0.30, 0.43],
    }
    art = R.report(
        index_df, bench_df, max_lag=8, min_overlap=10, config_echo=echo, out_dir=tmp_path
    )
    assert art.results["success_bar"] == [0.50, 0.75]
    summary = art.summary_path.read_text(encoding="utf-8")
    assert "0.30, 0.43" in summary or "[0.3, 0.43]" in summary


def test_report_adds_decoupling_levels_relationship(planted_signal_pair, tmp_path):
    # NOX-003.1 REQ-110: with a levels_frame, the report states both the levels (decoupling, negative)
    # and the residual (activity, positive) relationship.
    index_df, bench_df, _ = planted_signal_pair
    lv = bench_df.dropna().copy()
    levels_frame = pd.DataFrame(
        {
            "date": lv["date"],
            "index_value": -lv["value"].to_numpy(),
        }  # negatively related to benchmark
    )
    art = R.report(
        index_df,
        bench_df,
        max_lag=8,
        min_overlap=10,
        config_echo={
            "deseason_method": "intensity-model",
            "intensity_df": 3.0,
            "intensity_estimator": "spline",
            "intensity_criterion": "blocked-cv",
        },
        levels_frame=levels_frame,
        out_dir=tmp_path,
    )
    assert "levels_r" in art.results
    assert art.results["levels_r"] < 0  # decoupling sign in levels
    assert art.results["decoupling"] is True  # levels < 0 < residual
    summary = art.summary_path.read_text(encoding="utf-8")
    assert "Decoupling" in summary
    assert "intensity_trend" in summary  # the intensity-model echo line


def test_classify_bar_bands():
    assert R.classify_bar(0.80, (0.50, 0.75)) == "above-bar"
    assert R.classify_bar(0.60, (0.50, 0.75)) == "in-band"
    assert R.classify_bar(0.10, (0.50, 0.75)) == "below-bar"
    assert R.classify_bar(-0.80, (0.50, 0.75)) == "above-bar"  # uses |r|


# --------------------------------------------------------------------------- errors (AT-ERR-1)


def test_short_overlap_refuses(planted_signal_pair):
    index_df, bench_df, _ = planted_signal_pair
    short_idx = index_df.head(8)
    short_bench = bench_df.head(8)
    with pytest.raises(R.InsufficientOverlapError, match="ERR-004"):
        R.report(short_idx, short_bench, min_overlap=26)


def test_run_validation_missing_index_raises(tmp_path):
    from dataclasses import replace

    from noxus.config.run import SignalConfig, ValidationConfig

    scfg = SignalConfig(out_dir=tmp_path)
    vcfg = replace(ValidationConfig(), benchmark_path=tmp_path / "bench.parquet")
    with pytest.raises(FileNotFoundError, match="noxus index"):
        R.run_validation(scfg, vcfg)


def test_run_validation_missing_benchmark_raises(tmp_path, planted_signal_pair):
    from dataclasses import replace

    from noxus.config.run import SignalConfig, ValidationConfig
    from noxus.signal.index import write_index

    index_df, _, _ = planted_signal_pair
    scfg = SignalConfig(out_dir=tmp_path)
    idx = index_df.set_index("date")["index_value"]
    write_index(
        pd.DataFrame({"date": idx.index, "index_value": idx.to_numpy()}),
        tmp_path / scfg.index_name,
    )
    vcfg = replace(ValidationConfig(), benchmark_path=tmp_path / "no_bench.parquet")
    with pytest.raises(FileNotFoundError, match="ingest-benchmark"):
        R.run_validation(scfg, vcfg)
