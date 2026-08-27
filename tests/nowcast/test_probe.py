"""Unit tests for the T1 feasibility spike (NOX-013, REQ-015, AT8).

Covers the pure verdict thresholds, the no-VIIRS-fabrication guarantee, feature-availability
detection, and the incremental-information quick pass on planted-lead / planted-null synthetics.
The real end-to-end run against on-disk data is exercised separately (test_probe_realdata).
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from noxus.config.run import NowcastConfig
from noxus.nowcast.probe import (
    ProbeVerdict,
    _assess_thermal,
    _events_available,
    _incremental_probe,
    decide_verdict,
    probe,
)

# --------------------------------------------------------------------------------------------------
# Pure verdict logic (thresholds / three-way gate)
# --------------------------------------------------------------------------------------------------


def test_decide_verdict_go_primary_requires_valid_thermal():
    # A valid physical proxy short-circuits to GO-primary regardless of NO2 incremental info.
    assert decide_verdict(True, True, False) == ("GO-primary", False, False)
    assert decide_verdict(True, True, True) == ("GO-primary", False, False)


def test_decide_verdict_go_degraded_when_no2_carries_info():
    # Thermal present but invalid, yet a NO2 feature has incremental info → reduced-power build.
    assert decide_verdict(True, False, True) == ("GO-degraded", True, False)
    # Thermal entirely absent but NO2 info present → still GO-degraded.
    assert decide_verdict(False, False, True) == ("GO-degraded", True, False)


def test_decide_verdict_no_go_is_honest_null_and_escalates():
    # No valid proxy and no incremental info → NO-GO, reduced_power flagged, ERR-002 escalate.
    assert decide_verdict(True, False, False) == ("NO-GO", True, True)
    assert decide_verdict(False, False, False) == ("NO-GO", True, True)


# --------------------------------------------------------------------------------------------------
# Feature-availability detection (incl. no-VIIRS-fabrication)
# --------------------------------------------------------------------------------------------------


def _tri_csv(path, *, significant):
    pd.DataFrame(
        {
            "link": ["no2_viirs"],
            "r": [-0.05],
            "significant": [significant],
        }
    ).to_csv(path, index=False)


def test_thermal_unavailable_when_series_absent_never_fabricates(tmp_path):
    cfg = dataclasses.replace(
        NowcastConfig(),
        thermal_series_path=tmp_path / "does_not_exist.parquet",
        thermal_triangulation_path=tmp_path / "no_tri.csv",
    )
    available, valid, reason = _assess_thermal(cfg)
    assert available is False
    assert valid is False
    assert "unavailable" in reason.lower()


def test_thermal_present_but_nonsignificant_is_invalid(tmp_path):
    series = tmp_path / "viirs.parquet"
    pd.DataFrame({"activity": [1.0, 2.0]}).to_parquet(series)
    tri = tmp_path / "tri.csv"
    _tri_csv(tri, significant=False)
    cfg = dataclasses.replace(
        NowcastConfig(), thermal_series_path=series, thermal_triangulation_path=tri
    )
    available, valid, reason = _assess_thermal(cfg)
    assert available is True
    assert valid is False
    assert "invalid" in reason.lower()


def test_thermal_present_and_significant_is_valid(tmp_path):
    series = tmp_path / "viirs.parquet"
    pd.DataFrame({"activity": [1.0, 2.0]}).to_parquet(series)
    tri = tmp_path / "tri.csv"
    _tri_csv(tri, significant=True)
    cfg = dataclasses.replace(
        NowcastConfig(), thermal_series_path=series, thermal_triangulation_path=tri
    )
    available, valid, _ = _assess_thermal(cfg)
    assert available is True
    assert valid is True


def test_events_availability_reads_catalyst_conclusion(tmp_path):
    results = tmp_path / "catalyst.json"
    results.write_text(json.dumps({"n_no2_events": 27, "conclusion": "null"}))
    cfg = dataclasses.replace(NowcastConfig(), events_results_path=results)
    available, note = _events_available(cfg)
    assert available is True
    assert "null" in note


def test_events_unavailable_when_missing(tmp_path):
    cfg = dataclasses.replace(NowcastConfig(), events_results_path=tmp_path / "missing.json")
    available, _ = _events_available(cfg)
    assert available is False


# --------------------------------------------------------------------------------------------------
# Incremental-information quick pass (planted lead / planted null)
# --------------------------------------------------------------------------------------------------


def _weekly_frame(values, *, col="index_value", start="2019-01-06"):
    idx = pd.date_range(start=start, periods=len(values), freq="W")
    return pd.DataFrame({"date": idx, col: np.asarray(values, dtype=float)})


def test_incremental_probe_detects_planted_lead():
    # Feature leads CREA by 2 weeks: crea[t] = feature[t-2]. lead_lag should peak at a positive lag.
    rng = np.random.default_rng(0)
    n = 120
    base = np.cumsum(rng.normal(size=n))  # persistent signal
    feature = base
    crea = np.roll(base, 2)  # CREA is the feature delayed 2 weeks
    crea[:2] = np.nan
    feat_df = _weekly_frame(feature)
    crea_df = _weekly_frame(crea, col="value")
    ev = _incremental_probe("planted_lead", feat_df, crea_df, NowcastConfig())
    assert ev.peak_lag > 0
    assert ev.leads_crea is True
    assert ev.incremental_info is True


def test_incremental_probe_planted_null_finds_no_info():
    # Two independent white-noise series → no lead, not robustly significant.
    rng = np.random.default_rng(1)
    n = 200
    feat_df = _weekly_frame(rng.normal(size=n))
    crea_df = _weekly_frame(rng.normal(size=n), col="value")
    ev = _incremental_probe(
        "planted_null",
        feat_df,
        crea_df,
        dataclasses.replace(NowcastConfig(), n_boot=1000, n_perm=1000),
    )
    assert ev.incremental_info is False
    assert ev.robust_significant is False


def test_incremental_probe_too_few_overlap_is_not_info():
    feat_df = _weekly_frame(np.arange(10.0))
    crea_df = _weekly_frame(np.arange(10.0), col="value")
    ev = _incremental_probe("short", feat_df, crea_df, NowcastConfig())
    assert ev.incremental_info is False
    assert "too few" in ev.note


# --------------------------------------------------------------------------------------------------
# Full probe wiring on synthetic files (verdict + report written)
# --------------------------------------------------------------------------------------------------


def _synthetic_cfg(tmp_path, *, crea_from_feature_lag=None, thermal_significant=False):
    # Null branch uses the verified-null construction (independent noise, seed=1, n=200) that the
    # unit-level planted-null test pins down, so the full-probe NO-GO case is deterministic rather
    # than relying on a lucky single random draw (single nulls are ~5-10% flaky at alpha=0.05).
    if crea_from_feature_lag is None:
        rng = np.random.default_rng(1)
        n = 200
        residual = rng.normal(size=n)
        crea_vals = rng.normal(size=n)
    else:
        rng = np.random.default_rng(7)
        n = 160
        base = np.cumsum(rng.normal(size=n))
        residual = base
        crea_vals = np.roll(base, crea_from_feature_lag).astype(float)
        crea_vals[:crea_from_feature_lag] = np.nan

    idx = pd.date_range(start="2019-01-06", periods=n, freq="W")
    footprint = tmp_path / "footprint.parquet"
    pd.DataFrame(
        {"date": idx, "no2_corrected": residual, "valid_coverage": np.full(n, 0.9)}
    ).to_parquet(footprint)
    decomp = tmp_path / "decomp.parquet"
    pd.DataFrame({"date": idx, "residual_activity": residual}).to_parquet(decomp)
    bench = tmp_path / "crea.parquet"
    pd.DataFrame({"date": idx, "value": crea_vals}).to_parquet(bench)
    catalyst = tmp_path / "catalyst.json"
    catalyst.write_text(json.dumps({"n_no2_events": 27, "conclusion": "null"}))
    thermal = tmp_path / "viirs.parquet"
    pd.DataFrame({"activity": [1.0, 2.0, 3.0]}).to_parquet(thermal)
    tri = tmp_path / "tri.csv"
    _tri_csv(tri, significant=thermal_significant)

    return dataclasses.replace(
        NowcastConfig(),
        benchmark_path=bench,
        no2_residual_path=footprint,
        intensity_decomp_path=decomp,
        events_results_path=catalyst,
        thermal_series_path=thermal,
        thermal_triangulation_path=tri,
        out_dir=tmp_path,
        n_boot=800,
        n_perm=800,
    )


def test_probe_no_go_writes_report_and_escalates(tmp_path):
    cfg = _synthetic_cfg(tmp_path, crea_from_feature_lag=None, thermal_significant=False)
    v = probe(cfg)
    assert isinstance(v, ProbeVerdict)
    assert v.verdict == "NO-GO"
    assert v.escalate is True
    assert v.reduced_power is True
    assert v.thermal_valid is False
    report = cfg.probe_report_path()
    assert report.exists()
    text = report.read_text()
    assert "VERDICT: NO-GO" in text
    assert "HONEST NULL" in text
    assert "Tangshan" in text  # scale caveat present


def test_probe_go_degraded_when_feature_leads(tmp_path):
    cfg = _synthetic_cfg(tmp_path, crea_from_feature_lag=2, thermal_significant=False)
    v = probe(cfg)
    assert v.verdict == "GO-degraded"
    assert v.reduced_power is True
    assert v.escalate is False
    assert v.any_incremental_info is True


def test_probe_go_primary_when_valid_thermal_no_fabrication(tmp_path):
    # Valid thermal → GO-primary; the NO2 quick pass is skipped (no evidence), nothing fabricated.
    cfg = _synthetic_cfg(tmp_path, crea_from_feature_lag=None, thermal_significant=True)
    v = probe(cfg)
    assert v.verdict == "GO-primary"
    assert v.reduced_power is False
    assert v.thermal_valid is True
    assert v.evidence == ()  # dormant branch consumed no fabricated NO2 series


@pytest.mark.parametrize("missing_thermal", [True, False])
def test_probe_never_fabricates_thermal_series(tmp_path, missing_thermal):
    cfg = _synthetic_cfg(tmp_path, crea_from_feature_lag=None, thermal_significant=False)
    if missing_thermal:
        cfg = dataclasses.replace(cfg, thermal_series_path=tmp_path / "gone.parquet")
    v = probe(cfg)
    # With no valid proxy and null NO2 features, the verdict is NO-GO in both cases; the probe
    # never invents a thermal activity series to force a GO.
    assert v.verdict == "NO-GO"
    assert v.thermal_valid is False
