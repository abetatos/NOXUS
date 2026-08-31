"""Scale-contrast tests (NOX-016; AT1, AT2, AT4-AT8, AT-ERR-1).

This is the project's first task that reads several targets on purpose, so the NOX-009 structural
anti-p-hacking guarantee (an estimator with no benchmark parameter) is unavailable. AT2 and AT4
enforce the guarantees that replace it: features are never refit against a target, and the BH family
covers every computed pairing including refused ones.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from noxus.config.run import ScaleContrastConfig
from noxus.validation import scale_contrast as SC


def _ar1(rng, n, phi, sd=1.0):
    x = np.zeros(n)
    e = rng.normal(0.0, sd, n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


@pytest.fixture
def derived(tmp_path):
    """A miniature derived/ tree: two features and the two benchmark parquets."""
    rng = np.random.default_rng(4)
    n = 320
    idx = pd.date_range("2019-01-06", periods=n, freq="W")
    driver = _ar1(rng, n, 0.85)

    d = tmp_path / "derived"
    (d / "no2").mkdir(parents=True)
    pd.DataFrame(
        {
            "date": idx,
            "no2_corrected": driver + rng.normal(0, 0.4, n),
            "valid_coverage": np.where(np.arange(n) % 40 == 0, 0.1, 0.9),  # some weeks screened out
        }
    ).to_parquet(d / "no2" / "steel_footprint_signal.parquet")
    pd.DataFrame(
        {"date": idx, "residual_activity": driver + rng.normal(0, 0.4, n)}
    ).to_parquet(d / "no2" / "steel_intensity_decomposition.parquet")

    pd.DataFrame({"date": idx, "value": 0.2 * driver + _ar1(rng, n, 0.6)}).to_parquet(
        d / "benchmark_tangshan_bf_operating_rate.parquet"
    )
    # National: weekly BF rate (strong link) + a 10-day output series (coarse cadence).
    coarse_idx = pd.date_range("2019-01-10", periods=n // 3, freq="10D")
    aux = pd.concat(
        [
            pd.DataFrame(
                {
                    "date": idx,
                    "series": "China: Blast Furnace Starting Rate (247)",
                    "value": 0.9 * driver + _ar1(rng, n, 0.4, 0.3),
                }
            ),
            pd.DataFrame(
                {
                    "date": coarse_idx,
                    "series": "China: Estimated Daily Average Output: Crude Steel",
                    "value": rng.normal(0, 1, len(coarse_idx)),
                }
            ),
        ]
    )
    aux.to_parquet(d / "benchmark_auxiliary.parquet")
    return d


def _cfg(derived, tmp_path, **over):
    base = dict(
        derived_dir=derived,
        out_dir=tmp_path / "out",
        n_boot=200,
        n_perm=200,
        targets=(
            ("tangshan_bf_rate", "cluster", "benchmark_tangshan_bf_operating_rate.parquet", ""),
            (
                "national_bf_rate",
                "national",
                "benchmark_auxiliary.parquet",
                "China: Blast Furnace Starting Rate (247)",
            ),
            (
                "national_crude_steel",
                "national",
                "benchmark_auxiliary.parquet",
                "China: Estimated Daily Average Output: Crude Steel",
            ),
        ),
    )
    base.update(over)
    return dataclasses.replace(ScaleContrastConfig(), **base)


# ------------------------------------------------------------------------------- AT1 loading/cadence


def test_at1_coarse_target_aggregates_both_sides_down_never_interpolates(derived, tmp_path):
    cfg = _cfg(derived, tmp_path)
    df = SC.scale_contrast(cfg)
    weekly = df[(df["target"] == "national_bf_rate") & (df["subperiod"] == "full")]
    coarse = df[(df["target"] == "national_crude_steel") & (df["subperiod"] == "full")]
    assert set(weekly["cadence"]) == {"W"}
    assert set(coarse["cadence"]) == {"ME"}
    # Aggregating down must REDUCE n; an interpolated target would keep the weekly count.
    assert coarse["n"].max() < weekly["n"].max()


def test_at1_coverage_screen_is_applied_to_the_feature_that_carries_it(derived, tmp_path):
    cfg = _cfg(derived, tmp_path)
    resid = SC.load_feature(cfg, "no2_residual")
    inten = SC.load_feature(cfg, "no2_intensity")
    assert resid.isna().sum() > 0  # screened weeks are dropped, not imputed
    assert inten.isna().sum() == 0  # this parquet carries no valid_coverage column


# ------------------------------------------------------------------------------- AT2 no refit


def test_at2_no_fitting_entry_point_in_the_module():
    tree = ast.parse(Path(SC.__file__).read_text(encoding="utf-8"))
    names = {
        n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert not [n for n in names if any(k in n.lower() for k in ("fit", "tune", "select", "train"))]


def test_at2_run_completes_with_every_signal_fitter_disabled(derived, tmp_path, monkeypatch):
    # If any target could reach feature construction, disabling the fitters would break the run.
    import noxus.signal.index as index_mod

    def _boom(*_a, **_k):
        raise SC.FeatureRefitAttempted("a feature was refit against a target (ERR-004)")

    for name in ("deseasonalize", "regress_out_meteo"):
        if hasattr(index_mod, name):
            monkeypatch.setattr(index_mod, name, _boom)
    df = SC.scale_contrast(_cfg(derived, tmp_path))
    assert not df.empty


# ------------------------------------------------------------------------------- AT4 BH family


def test_at4_family_covers_every_pairing_including_refused(derived, tmp_path):
    cfg = _cfg(derived, tmp_path, min_overlap=200)  # forces the coarse pairings to refuse
    df = SC.scale_contrast(cfg)
    assert df["refused"].any(), "fixture should produce at least one refusal"
    assert int(df["family_size"].iloc[0]) == len(df)


def test_at4_dropping_a_pairing_changes_the_adjusted_pvalues(derived, tmp_path):
    from noxus.validation.robust import bh_fdr

    df = SC.scale_contrast(_cfg(derived, tmp_path))
    p = df["p_perm"].to_numpy(float)
    _, full_adj = bh_fdr(p, alpha=0.05)
    _, short_adj = bh_fdr(p[:-1], alpha=0.05)
    assert not np.allclose(full_adj[:-1], short_adj, equal_nan=True), (
        "if the correction were not family-wide, dropping a pairing would not move it"
    )


# ------------------------------------------------------------------------------- AT5 lag discipline


def test_at5_boundary_and_insignificant_peaks_are_flagged(derived, tmp_path):
    df = SC.scale_contrast(_cfg(derived, tmp_path))
    full = df[(df["subperiod"] == "full") & (~df["refused"])]
    assert {"peak_is_boundary", "peak_significant"}.issubset(full.columns)
    # A peak at +/- max_lag must be flagged as a boundary selection wherever it occurs.
    at_edge = full[full["peak_lag"].abs() == _cfg(derived, tmp_path).max_lag]
    assert bool(at_edge["peak_is_boundary"].all())
    # And "significant" must mean strictly outside the per-lag white-noise band.
    for _, r in full.iterrows():
        assert bool(r["peak_significant"]) == bool(abs(r["peak_r"]) > r["sig_bound"])


# ------------------------------------------------------------------------------- AT6 verdict branches


def _vframe(r_nat, contrast, lo, hi, *, fdr_sig, base_sig=False):
    rows = []
    for target, r, sig in (
        ("tangshan_bf_rate", 0.05, base_sig),
        ("national_bf_rate", r_nat, fdr_sig),
    ):
        rows.append(
            {
                "feature": "no2_intensity", "target": target, "subperiod": "full",
                "refused": False, "r": r, "p_fdr": 0.01 if sig else 0.6,
                "fdr_significant": sig,
                "contrast_vs_baseline": np.nan if target.startswith("tangshan") else contrast,
                "contrast_lo": np.nan if target.startswith("tangshan") else lo,
                "contrast_hi": np.nan if target.startswith("tangshan") else hi,
                "contrast_excludes_zero": False if target.startswith("tangshan") else (lo > 0 or hi < 0),
            }
        )
    return pd.DataFrame(rows)


def test_at6_scale_confirmed_branch():
    v, _ = SC.classify_scale_verdict(_vframe(0.62, 0.57, 0.30, 0.80, fdr_sig=True))
    assert v == SC.VERDICT_SCALE_CONFIRMED


def test_at6_inverted_branch():
    v, _ = SC.classify_scale_verdict(
        _vframe(0.05, -0.40, -0.60, -0.20, fdr_sig=False, base_sig=True)
    )
    assert v == SC.VERDICT_INVERTED


def test_at6_no_scale_effect_branch():
    v, _ = SC.classify_scale_verdict(_vframe(0.35, 0.10, -0.20, 0.40, fdr_sig=False))
    assert v == SC.VERDICT_NO_SCALE_EFFECT


def test_at6_pipeline_suspect_branch():
    v, why = SC.classify_scale_verdict(_vframe(0.05, 0.02, -0.20, 0.24, fdr_sig=False))
    assert v == SC.VERDICT_PIPELINE_SUSPECT and "0.5-0.75" in why


def test_at6_threshold_is_read_from_config_not_inlined():
    frame = _vframe(0.35, 0.10, -0.20, 0.40, fdr_sig=False)
    strict = dataclasses.replace(ScaleContrastConfig(), pipeline_suspect_abs_r=0.5)
    assert SC.classify_scale_verdict(frame, strict)[0] == SC.VERDICT_PIPELINE_SUSPECT
    assert SC.classify_scale_verdict(frame)[0] == SC.VERDICT_NO_SCALE_EFFECT


# ------------------------------------------------------------------------------- AT7 Arm-B gate


def test_at7_probe_no_go_reports_arm_a_only_and_records_the_gap(derived, tmp_path):
    from noxus.data.nbs import NbsProbeResult

    df = SC.scale_contrast(_cfg(derived, tmp_path))
    probe = NbsProbeResult(go=False, host_reachable=True, routes=(), note="NO-GO. test")
    text = SC.render_findings(df, _cfg(derived, tmp_path), probe)
    assert "NO-GO" in text
    assert "no paywalled substitute" in text.lower() or "no INDEPENDENT" in text
    assert "SHARED PUBLISHER" in text


def test_at7_probe_go_is_reported_as_go(derived, tmp_path):
    from noxus.data.nbs import NbsProbeResult

    df = SC.scale_contrast(_cfg(derived, tmp_path))
    probe = NbsProbeResult(go=True, host_reachable=True, routes=(), note="GO")
    assert "GO - an NBS route responded" in SC.render_findings(df, _cfg(derived, tmp_path), probe)


def test_arm_b_fetcher_refuses_rather_than_returning_empty():
    from noxus.data.nbs import fetch_nbs_steel

    with pytest.raises(NotImplementedError, match="NO-GO"):
        fetch_nbs_steel()


# ------------------------------------------------------------------------------- AT8 subperiods/caveats


def test_at8_subperiods_and_mandatory_caveats_present(derived, tmp_path):
    cfg = _cfg(derived, tmp_path)
    df = SC.scale_contrast(cfg)
    assert set(df["subperiod"]) == {"full", "2019-2021", "2022-2026"}
    text = SC.render_findings(df, cfg, None)
    assert "PROCESSOR VERSIONS" in text  # EDGE-005
    assert "13%" in text  # EDGE-006: the correct reading of a national correlation
    assert "SHARED PUBLISHER" in text  # REQ-022


# ------------------------------------------------------------------------------- AT-ERR-1


def test_at_err_1_missing_benchmark_names_the_ingest_command(derived, tmp_path):
    (derived / "benchmark_auxiliary.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="noxus ingest-benchmark"):
        SC.scale_contrast(_cfg(derived, tmp_path))


def test_at_err_1_missing_feature_names_the_index_command(derived, tmp_path):
    (derived / "no2" / "steel_intensity_decomposition.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="noxus index"):
        SC.scale_contrast(_cfg(derived, tmp_path))


def test_at_err_1_short_overlap_refuses_without_an_estimate(derived, tmp_path):
    df = SC.scale_contrast(_cfg(derived, tmp_path, min_overlap=200))
    refused = df[df["refused"]]
    assert len(refused) and refused["r"].isna().all()
    assert refused["refused_reason"].str.contains("min_overlap").all()
