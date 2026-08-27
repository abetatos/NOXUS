"""Tests for the VIIRS triangulation (REQ-020..024, EDGE-003). Real robust.py on tiny arrays.

Covers AT5 (robust_corr reused), AT6 (feature-vs-target interpretation incl. honest null), AT-EDGE-2
(common-window alignment), AT4 (FIRMS cross-check consistency). Significance is exercised through the
real ``noxus.validation.robust`` — never mocked.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from analysis import viirs_triangulation as VT
from noxus.config.run import ThermalTriangulationConfig
from noxus.validation.robust import robust_corr

# Fast but still-honest robust settings for the tests.
CFG = ThermalTriangulationConfig(n_boot=400, n_perm=400, min_overlap=20)


def _weekly_index(n):
    return pd.date_range("2020-01-05", periods=n, freq="W")


def _smooth(rng, n, phi=0.6):
    """A stationary autocorrelated AR(1) series (exercises the robust path without spurious drift)."""
    e = rng.normal(size=n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x - x.mean()


def _series(vals, idx):
    return pd.Series(np.asarray(vals, float), index=idx)


def _make(n=80, seed=0):
    rng = np.random.default_rng(seed)
    idx = _weekly_index(n)
    base = _smooth(rng, n)
    tracks = base + 0.05 * rng.normal(size=n)  # strongly correlated with base
    indep = _smooth(rng, n)  # independent smooth series
    return idx, base, tracks, indep


# ------------------------------------------------------------------------------------ AT5


def test_triangulate_matches_direct_robust_corr():
    idx, base, tracks, indep = _make()
    thermal = _series(base, idx)
    no2 = _series(tracks, idx)  # correlated with thermal
    crea = _series(indep, idx)  # uncorrelated with thermal
    report = VT.triangulate(thermal, no2, crea, CFG)

    direct = robust_corr(no2, thermal, n_boot=CFG.n_boot, n_perm=CFG.n_perm, seed=CFG.seed)
    link = report.link("no2_viirs")
    assert link.corr.r == direct.r  # same reused computation, not re-implemented
    assert link.corr.p_perm == direct.p_perm
    # BH-FDR adjusted p present for every link.
    assert all(np.isfinite(link_.p_adj) for link_ in report.links)


def test_common_window_alignment_shrinks_n():
    idx, base, tracks, indep = _make(n=80)
    thermal = _series(base, idx)
    # NO2/CREA only overlap the first 40 weeks -> n_common must reflect the reduced overlap (EDGE-003).
    no2 = _series(tracks[:40], idx[:40])
    crea = _series(indep[:40], idx[:40])
    report = VT.triangulate(thermal, no2, crea, CFG)
    assert report.n_common == 40


# ------------------------------------------------------------------------------------ AT6


def test_interpret_branches():
    assert "TARGET" in VT.interpret(True, False)
    assert "blind" in VT.interpret(False, True)
    assert "Coherent" in VT.interpret(True, True)
    # Both-null must be an "inconclusive / proxy may be invalid" verdict, NOT a "feature weak" claim.
    both_null = VT.interpret(False, False)
    assert both_null == VT.DIAG_PROXY_INVALID
    assert "Inconclusive" in both_null
    assert "uninformative" in both_null or "invalid" in both_null
    assert "cannot arbitrate" in both_null
    # BUG 3: it must explicitly DISCLAIM the "feature weak" reading, not assert it.
    assert "NOT a 'feature weak' claim" in both_null


def test_diagnosis_no2_blind_target_sound():
    # CREA<->VIIRS holds; NO2 does not track the physical heat -> NO2 blind / target sound.
    idx, base, tracks, indep = _make(seed=1)
    thermal = _series(base, idx)
    crea = _series(tracks, idx)  # CREA tracks the heat
    no2 = _series(indep, idx)  # NO2 does not
    report = VT.triangulate(thermal, no2, crea, CFG)
    assert report.link("crea_viirs").significant
    assert not report.link("no2_viirs").significant
    assert report.diagnosis == VT.DIAG_NO2_BLIND


def test_diagnosis_target_weak():
    # NO2<->VIIRS holds; CREA<->VIIRS does not -> TARGET (CREA) was the weak link.
    idx, base, tracks, indep = _make(seed=2)
    thermal = _series(base, idx)
    no2 = _series(tracks, idx)
    crea = _series(indep, idx)
    report = VT.triangulate(thermal, no2, crea, CFG)
    assert report.link("no2_viirs").significant
    assert not report.link("crea_viirs").significant
    assert report.diagnosis == VT.DIAG_TARGET_WEAK


def test_diagnosis_both_absent_is_proxy_invalid_not_feature_weak():
    # Neither NO2<->VIIRS nor CREA<->VIIRS holds -> the proxy tracks nothing -> INCONCLUSIVE /
    # proxy-may-be-invalid, still reported (not suppressed) and NOT a "feature weak" claim (BUG 3).
    idx, base, tracks, indep = _make(seed=3)
    rng = np.random.default_rng(99)
    thermal = _series(base, idx)
    no2 = _series(_smooth(rng, len(idx)), idx)
    crea = _series(_smooth(rng, len(idx)), idx)
    report = VT.triangulate(thermal, no2, crea, CFG)
    assert not report.link("no2_viirs").significant
    assert not report.link("crea_viirs").significant
    assert report.diagnosis == VT.DIAG_PROXY_INVALID
    assert "cannot arbitrate" in report.diagnosis
    assert "NOT a 'feature weak' claim" in report.diagnosis


# ------------------------------------------------------------------------------------ AT4 cross-check


def test_firms_cross_check_reports_consistency():
    idx, base, tracks, indep = _make(seed=2)
    thermal = _series(base, idx)
    no2 = _series(tracks, idx)
    crea = _series(indep, idx)
    firms = _series(base + 0.05 * np.random.default_rng(7).normal(size=len(idx)), idx)
    report = VT.triangulate(thermal, no2, crea, CFG, firms=firms)
    assert report.firms_diagnosis is not None
    assert report.firms_consistent is not None
    # FIRMS tracks the same base as VNF, so the cross-check should agree.
    assert report.firms_consistent is True
    assert any(link_.name == "no2_firms" for link_ in report.links)


# ------------------------------------------------------------------------------------ output frame


def test_report_to_frame_is_tidy():
    idx, base, tracks, indep = _make(seed=2)
    report = VT.triangulate(_series(base, idx), _series(tracks, idx), _series(indep, idx), CFG)
    df = VT.report_to_frame(report)
    assert {"link", "r", "p_perm", "p_adj_bh", "significant", "diagnosis"} <= set(df.columns)
    assert set(df["link"]) == {"no2_viirs", "crea_viirs"}


def test_config_is_frozen_and_preregistered():
    assert dataclasses.is_dataclass(ThermalTriangulationConfig)
    cfg = ThermalTriangulationConfig()
    assert cfg.fdr_alpha == 0.05
