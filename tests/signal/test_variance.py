"""Tests for the NOX-009 change-variance decomposition (AT1..AT8, AT-EDGE-1..3, AT-ERR-1).

Deterministic synthetic fixtures only — no network, no cube on disk, no credentials, no benchmark
(NFR-001/004). Bootstrap draws are reduced for speed while still asserting the width behaviour.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from noxus.config.run import VarianceDecompConfig
from noxus.signal import variance as V


# --------------------------------------------------------------------------- fixtures


def _orthonormal(mat: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(mat)
    return q


@pytest.fixture
def planted_budget():
    """footprint_change = regional + β·meteo + local + noise with orthogonal, known variance shares.

    Returns (footprint, background, meteo, true_shares). The components are made mutually orthogonal so
    the nested partition recovers the planted shares crisply; ``local`` is a smooth (persistent)
    sinusoid and ``noise`` is white, so the structured/white split is well posed (AT3).
    """
    rng = np.random.default_rng(0)
    n = 300
    idx = pd.date_range("2020-01-05", periods=n, freq="W")
    t = np.arange(n)
    raw = np.column_stack(
        [
            rng.normal(0, 1, n),
            rng.normal(0, 1, n),
            np.sin(2 * np.pi * t / 40.0),
            rng.normal(0, 1, n),
        ]
    )
    q = _orthonormal(raw)
    shares = np.array([0.40, 0.30, 0.20, 0.10])
    comp = q * np.sqrt(shares * n)
    regional, meteo_drv, local, noise = comp[:, 0], comp[:, 1], comp[:, 2], comp[:, 3]
    footprint = pd.Series(regional + meteo_drv + local + noise, index=idx, name="footprint")
    background = pd.Series(regional, index=idx, name="background")
    meteo = pd.DataFrame(
        {"u10": meteo_drv, "v10": rng.normal(0, 1, n), "blh": rng.normal(0, 1, n)}, index=idx
    )
    return (
        footprint,
        background,
        meteo,
        dict(regional=0.40, meteo=0.30, steel_local=0.20, noise=0.10),
    )


def _ar1(n: int, phi: float, seed: int) -> np.ndarray:
    r = np.random.default_rng(seed)
    e = r.normal(0, 1, n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def _levels(resolution: str, n: int, seed: int):
    """A pair of level footprint/background series + meteo for the driver (AT4/AT8)."""
    freq = "D" if resolution == "D" else "W"
    idx = pd.date_range("2019-01-06", periods=n, freq=freq)
    t = np.arange(n)
    rng = np.random.default_rng(seed)
    regional = np.cumsum(rng.normal(0, 0.3, n))
    season = 4.0 * np.sin(2 * np.pi * t / (365 if resolution == "D" else 52))
    local = 1.2 * np.sin(2 * np.pi * t / 23.0)
    footprint = pd.Series(
        regional + season + local + rng.normal(0, 0.4, n), index=idx, name="footprint"
    )
    background = pd.Series(regional + season + rng.normal(0, 0.3, n), index=idx, name="background")
    meteo = pd.DataFrame(
        {
            "u10": rng.normal(0, 1, n),
            "v10": rng.normal(0, 1, n),
            "blh": rng.normal(500, 40, n),
            "wind_speed": np.abs(rng.normal(3, 1, n)),
        },
        index=idx,
    )
    return footprint, background, meteo


def _fast_cfg(**kw) -> VarianceDecompConfig:
    from dataclasses import replace

    return replace(VarianceDecompConfig(), n_boot=120, **kw)


# --------------------------------------------------------------------------- AT2 / AT-ERR-1 anti-p-hacking


def test_estimator_has_no_benchmark_argument():
    # AT2 / AC-002: the core estimator exposes no benchmark/output/crea parameter.
    params = set(inspect.signature(V.partition_change_variance).parameters)
    assert not (params & {"benchmark", "crea", "output", "target", "y_benchmark"})


def test_module_never_reads_the_benchmark():
    # AT2: the module does not import or open the CREA benchmark (structural exclusion, NFR-001).
    # (The word "benchmark" appears only in docstrings explaining the exclusion; assert on imports
    # and the CREA artefact paths, which are what would actually leak the benchmark into the estimator.)
    src = Path(V.__file__).read_text(encoding="utf-8")
    assert "noxus.data.benchmark" not in src
    assert "benchmark_tangshan" not in src  # the committed CREA parquet name
    assert "BenchmarkConfig" not in src
    import_lines = [ln for ln in src.splitlines() if ln.strip().startswith(("import ", "from "))]
    assert not any("benchmark" in ln.lower() for ln in import_lines)


def test_passing_a_benchmark_is_refused(planted_budget):
    # AT-ERR-1 / ERR-004: no hidden keyword accepts a benchmark -> TypeError.
    footprint, background, meteo, _ = planted_budget
    with pytest.raises(TypeError):
        V.partition_change_variance(
            footprint, background, meteo, basis="first-diff", benchmark=footprint
        )


# --------------------------------------------------------------------------- AT3 planted recovery


def test_planted_shares_recovered_and_sum_to_one(planted_budget):
    footprint, background, meteo, truth = planted_budget
    b = V.partition_change_variance(footprint, background, meteo, basis="first-diff")
    assert b.share_regional == pytest.approx(truth["regional"], abs=0.04)
    assert b.share_meteo == pytest.approx(truth["meteo"], abs=0.04)
    assert b.share_steel_local == pytest.approx(truth["steel_local"], abs=0.04)
    assert b.share_noise == pytest.approx(truth["noise"], abs=0.04)
    total = b.share_regional + b.share_meteo + b.share_steel_local + b.share_noise
    assert total == pytest.approx(1.0, abs=0.01)
    # REQ-021: entry order + per-step raw ΔR² are recorded and auditable.
    assert b.order == ("regional", "meteo", "local")
    assert b.dr2_regional > 0 and b.dr2_meteo > 0 and b.dr2_local > 0


# --------------------------------------------------------------------------- AT4 bases × resolutions


def test_both_bases_both_resolutions_one_row_each():
    cfg = _fast_cfg()
    levels = {"W": _levels("W", 220, 1), "D": _levels("D", 320, 2)}
    df = V.decompose_from_levels(levels, cfg)
    assert len(df) == 4  # 2 bases × 2 resolutions
    assert set(zip(df["basis"], df["resolution"], strict=False)) == {
        ("first-diff", "W"),
        ("deseason", "W"),
        ("first-diff", "D"),
        ("deseason", "D"),
    }
    for col in (
        "n",
        "share_regional",
        "share_meteo",
        "share_steel_local",
        "share_noise",
        "dr2_regional",
        "dr2_meteo",
        "dr2_local",
        "change_ceiling",
        "verdict",
    ):
        assert col in df.columns
    # No basis/resolution dropped; every share in [0, 1].
    for c in ("share_regional", "share_meteo", "share_steel_local", "share_noise"):
        assert (df[c] >= 0).all() and (df[c] <= 1.0001).all()


# --------------------------------------------------------------------------- AT5 ceiling contrast


def test_implied_and_level_ceilings_and_verdict():
    assert V.implied_change_ceiling(0.25) == pytest.approx(0.5)
    assert V.implied_change_ceiling(-0.1) == 0.0  # collapses, never NaN (EDGE-006)
    lo, hi = V.level_ceiling((0.30, 0.43))
    assert (lo, hi) == pytest.approx((np.sqrt(0.30), np.sqrt(0.43)))
    # headroom vs ceiling-bound vs SNR-bound verdicts are functions of the measured share only.
    assert "headroom" in V.classify_verdict(0.70, 0.60, 0.80).lower()
    assert "physical" in V.classify_verdict(0.35, 0.25, 0.45).lower()
    assert "snr-bound" in V.classify_verdict(0.004, 0.0, 0.015).lower()


# --------------------------------------------------------------------------- AT6 block-bootstrap CI


def test_block_bootstrap_ci_wider_than_iid_on_ar1():
    n = 250
    idx = pd.date_range("2019-01-06", periods=n, freq="W")
    phi = 0.85
    regional = pd.Series(_ar1(n, phi, 1), index=idx)
    local = np.sin(2 * np.pi * np.arange(n) / 35.0)
    footprint = pd.Series(
        0.6 * regional.to_numpy() + 0.8 * local + np.random.default_rng(2).normal(0, 0.5, n),
        index=idx,
    )
    meteo = pd.DataFrame({"u10": _ar1(n, phi, 3), "v10": _ar1(n, phi, 4)}, index=idx)
    from noxus.validation.robust import block_length

    blk = block_length(n)
    ci = V._share_ci(footprint, regional, meteo, block=blk, n_boot=400, seed=20260706)
    block_width = ci["share_hi"] - ci["share_lo"]
    iid_width = ci["share_hi_iid"] - ci["share_lo_iid"]
    assert ci["block"] == blk
    assert np.isfinite(block_width) and np.isfinite(iid_width)
    assert block_width > iid_width  # autocorrelation inflates the honest CI (NFR-002)


# --------------------------------------------------------------------------- AT-EDGE-1 negative clamp


def test_negative_incremental_dr2_is_clamped_and_flagged():
    n = 200
    idx = pd.date_range("2020-01-05", periods=n, freq="W")
    t = np.arange(n)
    rng = np.random.default_rng(3)
    regional = rng.normal(0, 1, n)
    meteo_drv = rng.normal(0, 1, n)
    alternating = 1.5 * ((-1.0) ** t)  # anti-persistent -> var(Δ) ≫ var, structured estimate < 0
    footprint = pd.Series(regional + meteo_drv + alternating, index=idx)
    background = pd.Series(regional, index=idx)
    meteo = pd.DataFrame({"u10": meteo_drv}, index=idx)
    b = V.partition_change_variance(footprint, background, meteo, basis="first-diff")
    assert b.dr2_local < 0  # raw incremental recorded as negative
    assert b.share_steel_local == 0.0  # clamped, no negative share emitted
    assert "local_clamped" in b.flags


# --------------------------------------------------------------------------- AT-EDGE-2 gap-aware n


def test_gap_aware_first_difference_records_effective_n():
    idx = pd.date_range("2021-01-01", periods=6, freq="D")
    s = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0, 6.0], index=idx, name="footprint")
    d = V.first_difference_gap_aware(s)
    # No difference is computed across the gap (position after the NaN is masked); no interpolation.
    assert d.notna().sum() == 3  # (2-1), (5-4), (6-5) — never (4-NaN)
    assert np.isnan(d.iloc[3])  # the diff spanning the gap is NaN
    assert d.dropna().to_list() == [1.0, 1.0, 1.0]


def test_too_sparse_basis_is_labelled_low_power_not_crashed():
    idx = pd.date_range("2021-01-01", periods=5, freq="D")
    footprint = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0], index=idx)
    background = pd.Series([0.5, np.nan, 1.0, np.nan, 1.5], index=idx)
    b = V.partition_change_variance(
        V.first_difference_gap_aware(footprint),
        V.first_difference_gap_aware(background),
        None,
        basis="first-diff",
    )
    assert "low_power" in b.flags  # labelled, not crashed


# --------------------------------------------------------------------------- AT-EDGE-3 degeneracies


def test_degenerate_background_drops_regional_split():
    # EDGE-004: no usable background -> regional term skipped and flagged, budget still returned.
    n = 120
    idx = pd.date_range("2020-01-05", periods=n, freq="W")
    rng = np.random.default_rng(5)
    footprint = pd.Series(np.sin(np.arange(n) / 5.0) + rng.normal(0, 0.3, n), index=idx)
    background = pd.Series(np.full(n, np.nan), index=idx)  # degenerate ring
    b = V.partition_change_variance(footprint, background, None, basis="first-diff")
    assert b.share_regional == 0.0
    assert "no_regional" in b.flags


def test_missing_meteo_rows_records_covered_fraction():
    # EDGE-005: meteo fit on jointly-available rows; covered fraction recorded, no extrapolation.
    n = 120
    idx = pd.date_range("2020-01-05", periods=n, freq="W")
    rng = np.random.default_rng(6)
    regional = rng.normal(0, 1, n)
    footprint = pd.Series(regional + rng.normal(0, 0.5, n), index=idx)
    background = pd.Series(regional, index=idx)
    u = rng.normal(0, 1, n)
    u[:30] = np.nan  # a quarter of the meteo covariate missing
    meteo = pd.DataFrame({"u10": u}, index=idx)
    b = V.partition_change_variance(footprint, background, meteo, basis="first-diff")
    assert b.meteo_covered_fraction < 1.0
    assert b.meteo_covered_fraction == pytest.approx(0.75, abs=0.05)


def test_near_zero_local_share_gives_snr_bound_verdict():
    # EDGE-006: a footprint fully explained by regional+meteo+white noise -> steel-local ≈ 0.
    n = 200
    idx = pd.date_range("2020-01-05", periods=n, freq="W")
    rng = np.random.default_rng(7)
    regional = rng.normal(0, 1, n)
    meteo_drv = rng.normal(0, 1, n)
    footprint = pd.Series(regional + meteo_drv + rng.normal(0, 0.5, n), index=idx)
    background = pd.Series(regional, index=idx)
    meteo = pd.DataFrame({"u10": meteo_drv}, index=idx)
    b = V.partition_change_variance(footprint, background, meteo, basis="first-diff")
    assert b.share_steel_local < 0.05
    ceiling = V.implied_change_ceiling(b.share_steel_local)
    assert ceiling < 0.25
    verdict = V.classify_verdict(b.share_steel_local, 0.0, b.share_steel_local + 0.02)
    assert "snr-bound" in verdict.lower()


# --------------------------------------------------------------------------- AT8 determinism


def test_decompose_is_deterministic_under_seed():
    cfg = _fast_cfg()
    levels = {"W": _levels("W", 220, 1), "D": _levels("D", 320, 2)}
    df1 = V.decompose_from_levels(levels, cfg)
    df2 = V.decompose_from_levels(levels, cfg)
    pd.testing.assert_frame_equal(df1, df2)


# --------------------------------------------------------------------------- AT7 figures + findings


def test_report_artefacts_are_produced_and_honest(tmp_path):
    from analysis.variance_decomposition import build_report

    cfg = _fast_cfg()
    levels = {"W": _levels("W", 220, 1), "D": _levels("D", 320, 2)}
    df = V.decompose_from_levels(levels, cfg)
    artefacts = build_report(df, tmp_path)
    assert artefacts["stacked"].exists()
    assert artefacts["ceiling"].exists()
    findings = artefacts["findings"].read_text(encoding="utf-8")
    # All four cells appear; the note frames the null honestly and names the downstream tasks.
    for basis in ("first-diff", "deseason"):
        assert basis in findings
    assert "NOX-010" in findings and "NOX-011" in findings
    assert "failure" not in findings.lower()
