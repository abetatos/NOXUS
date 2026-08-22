"""Regime-study tests: rolling regime, jackknife, match, scan-max null, FDR, no-look-ahead, null.

Fully offline + deterministic (fixed seeds); mirrors the NOX-004 / NOX-008 style. Statistics are
asserted only on planted synthetic fixtures; no network. Maps to AT3–AT9, AT-EDGE-1/2, AT-ERR-2.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from noxus.catalyst import regime as R

WEEKS = 320
WINDOW = 78  # ~18 months at weekly cadence
COUPLED = (120, 198)  # a contiguous ~78-week coupled block (the planted regime)


def _weekly_index(n=WEEKS):
    return pd.date_range("2019-01-06", periods=n, freq="W")


@pytest.fixture
def planted_regime():
    """resid & bench uncorrelated except inside one 78-week window where bench ~ a*resid (AT3/AT6)."""
    rng = np.random.default_rng(7)
    idx = _weekly_index()
    resid = rng.normal(0, 1.0, WEEKS)
    bench = rng.normal(0, 1.0, WEEKS)
    lo, hi = COUPLED
    bench[lo:hi] = 0.9 * resid[lo:hi] + rng.normal(0, 0.3, hi - lo)
    return (
        pd.Series(resid, index=idx, name="residual_activity"),
        pd.Series(bench, index=idx, name="value"),
    )


def _ar1_series(seed, phi=0.8, n=WEEKS):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal(0, 1.0)
    return pd.Series(x, index=_weekly_index(n))


@pytest.fixture
def ar1_null():
    """Two INDEPENDENT strongly-autocorrelated AR(1) series — a representative clean null (AT9).

    Seeds 100/1100 are a null realization whose scan-max sits inside the band (verified across seeds
    in ``test_scan_max_null_fp_rate_near_alpha``); used for the deterministic honest-null verdict.
    """
    return (
        _ar1_series(100).rename("residual_activity"),
        _ar1_series(1100).rename("value"),
    )


@pytest.fixture
def event_bound():
    """A peak-regime r created by a FEW large aligned step events (AT4)."""
    rng = np.random.default_rng(11)
    n = 90
    idx = pd.date_range("2023-01-01", periods=n, freq="W")
    resid = rng.normal(0, 1.0, n)
    bench = rng.normal(0, 1.0, n)
    ev_pos = [20, 45, 70]
    for i in ev_pos:
        resid[i] += 8.0
        bench[i] += 8.0  # aligned big steps drive the whole correlation
    events = pd.DataFrame({"date": [idx[i] for i in ev_pos], "direction": "surge"})
    return pd.Series(resid, index=idx), pd.Series(bench, index=idx), events


@pytest.fixture
def continuous_coupling():
    """A broadly-distributed continuous coupling: bench ~ resid everywhere (AT4)."""
    rng = np.random.default_rng(13)
    n = 90
    idx = pd.date_range("2023-01-01", periods=n, freq="W")
    resid = rng.normal(0, 1.0, n)
    bench = 0.8 * resid + rng.normal(0, 0.4, n)
    events = pd.DataFrame({"date": [idx[i] for i in (15, 40, 65)], "direction": "surge"})
    return pd.Series(resid, index=idx), pd.Series(bench, index=idx), events


# --------------------------------------------------------------------------- AT3 (rolling regime)


def test_rolling_regime_recovers_peak(planted_regime):
    resid, bench = planted_regime
    res = R.rolling_regime(resid, bench, window=WINDOW, n_boot=200, n_perm=200)
    lo, hi = COUPLED
    idx = _weekly_index()
    # The rolling-r peak (trailing window) lands inside/just after the coupled block.
    peak_pos = idx.get_indexer([res.peak_date])[0]
    assert lo <= peak_pos <= hi + WINDOW // 3
    # The peak regime is strongly coupled; off-peak is not.
    assert res.segments["peak"].corr.r > 0.5
    assert abs(res.segments["off_peak"].corr.r) < res.segments["peak"].corr.r
    assert res.boundary_sweep.shape[0] >= 1  # robustness sweep reported


def test_rolling_regime_seed_and_order_stable(planted_regime):
    resid, bench = planted_regime
    a = R.rolling_regime(resid, bench, window=WINDOW, n_boot=200, n_perm=200)
    b = R.rolling_regime(resid, bench, window=WINDOW, n_boot=200, n_perm=200)
    assert a.peak_date == b.peak_date
    assert a.peak_r == pytest.approx(b.peak_r)


def test_rolling_regime_exposes_no_crea_tuning_param():
    """The segmentation function takes NO CREA-agreement/optimisation argument (AT3, NFR-005)."""
    params = set(inspect.signature(R.rolling_regime).parameters)
    forbidden = {"target", "optimize", "maximize", "benchmark_objective", "crea", "tune"}
    assert not (params & forbidden)


def test_rolling_regime_labelled_descriptive(planted_regime):
    resid, bench = planted_regime
    res = R.rolling_regime(resid, bench, window=WINDOW, n_boot=100, n_perm=100)
    assert "descriptive" in res.descriptive_note.lower()


# --------------------------------------------------------------------------- AT4 (event-bound jackknife)


def test_jackknife_attributes_event_bound(event_bound):
    resid, bench, events = event_bound
    res = R.event_bound_jackknife(resid, bench, events, window_periods=2, min_events=3)
    assert res.refused is False
    assert res.verdict == R.VERDICT_EVENT_BOUND
    # Removing the few big events collapses the peak r.
    assert res.r_without_top < 0.5 * res.full_r
    assert len(res.contributions) == 3


def test_jackknife_attributes_continuous(continuous_coupling):
    resid, bench, events = continuous_coupling
    res = R.event_bound_jackknife(resid, bench, events, window_periods=2, min_events=3)
    assert res.verdict == R.VERDICT_CONTINUOUS
    # No single event dominates: removing them barely moves r.
    assert res.r_without_top > 0.5 * res.full_r


def test_jackknife_refuses_below_min_events(continuous_coupling):
    """Below min_events the event-level jackknife is refused -> coarse fallback (ERR-003, AT-ERR-2)."""
    resid, bench, events = continuous_coupling
    res = R.event_bound_jackknife(resid, bench, events.head(2), min_events=3)
    assert res.refused is True
    assert res.verdict == "insufficient-events"


def test_jackknife_exposes_no_crea_tuning_param():
    params = set(inspect.signature(R.event_bound_jackknife).parameters)
    forbidden = {"target", "optimize", "maximize", "crea", "tune"}
    assert not (params & forbidden)


# --------------------------------------------------------------------------- AT5 (regime match reuses NOX-004)


def test_regime_event_study_reports_precision_recall_lead(event_bound):
    """Per-regime precision/recall/FDR/lead via reused NOX-004 primitives, not a Pearson r (AT5)."""
    resid, bench_s, _ = event_bound
    bench_df = pd.DataFrame({"date": bench_s.index, "value": bench_s.to_numpy()})
    regimes = {"peak": (resid.index.min(), resid.index.max())}
    study = R.regime_event_study(
        resid,
        None,
        None,
        bench_df,
        None,
        regimes=regimes,
        z_thresh=2.0,
        bf_event_z=1.5,
        meteo_screen=False,
    )
    mr = study.by_regime["peak"]
    # It is a MatchResult (event-study metric), never a correlation.
    assert hasattr(mr, "precision") and hasattr(mr, "recall") and hasattr(mr, "median_lead_days")
    assert not hasattr(mr, "pearson_r")
    assert "precision/recall" in study.note


def test_regime_event_study_calendar_edge_matched():
    """Heating-season blanket intervals are edge-matched via the reused NOX-004 edge-matcher (EDGE-002)."""
    idx = pd.date_range("2023-01-01", periods=40, freq="W")
    resid = pd.Series(np.zeros(40), index=idx)
    bench_df = pd.DataFrame({"date": idx, "value": np.zeros(40)})
    calendar = pd.DataFrame(
        {
            "start": [pd.Timestamp("2023-02-01")],
            "end": [pd.Timestamp("2023-08-01")],  # a long blanket
            "direction": ["down"],
            "cause": ["heating_season_restriction"],
        }
    )
    regimes = {"peak": (idx.min(), idx.max())}
    study = R.regime_event_study(
        resid, None, None, bench_df, calendar, regimes=regimes, meteo_screen=False
    )
    # The blanket contributes only its two edges (start down + end up), not every week inside.
    assert study.calendar_used is True
    assert study.by_regime["peak"].n_prod <= 4  # BF jumps (none here) + 2 calendar edges


# --------------------------------------------------------------------------- AT6 (scan-maximum null)


def test_scan_max_null_fp_rate_near_alpha():
    """Across independent AR(1) seeds the scan-max FP rate stays near alpha (AT6): a genuinely-null
    series does not *routinely* yield a significant peak. (Any single seed may exceed ~5% of the time —
    that is precisely the multiple-comparisons phenomenon the null is designed to expose.)"""
    beyond = 0
    n_seeds = 12
    for s in range(200, 200 + n_seeds):
        a, b = _ar1_series(s), _ar1_series(s + 5000)
        res = R.scan_max_null(a, b, window=WINDOW, n_draws=200, seed=20260720)
        beyond += int(res.beyond_band)
    assert beyond / n_seeds <= 0.25  # near alpha (with block-permutation approximation slack)


def test_scan_max_null_clean_null_inside_band(ar1_null):
    """The representative clean-null fixture sits inside the band (block preserves autocorrelation)."""
    resid, bench = ar1_null
    res = R.scan_max_null(resid, bench, window=WINDOW, n_draws=300, seed=20260720)
    assert res.n_draws > 0
    assert res.beyond_band is False
    assert res.p_value > 0.05
    assert res.block >= 2 and res.n_windows > 0


def test_scan_max_null_flags_planted_peak(planted_regime):
    """The planted-regime peak sits beyond the scan-max null band (AT6)."""
    resid, bench = planted_regime
    res = R.scan_max_null(resid, bench, window=WINDOW, n_draws=300, seed=20260720)
    assert res.beyond_band is True
    assert res.p_value < 0.05


# --------------------------------------------------------------------------- AT7 (multiplicity / grid)


def test_build_grid_cells_enumerates_full_product():
    # The correlation FDR family is variants x windows x regimes. Thresholds are NOT a correlation
    # dimension (they do not change the correlation statistic) so they are excluded rather than padded
    # into BH-FDR with duplicate p-values (F2); they parametrise the event-study table instead.
    cells = R.build_grid_cells(("harmonic", "intensity-model"), (78, 52), ("peak", "collapse"))
    assert len(cells) == 2 * 2 * 2


def test_bh_fdr_wipes_lone_favourable_cell():
    """A lone nominally-significant cell that should not survive BH-FDR is flagged non-significant (AT7)."""
    from noxus.validation.robust import bh_fdr

    pvals = [0.04, 0.6, 0.7, 0.8, 0.9, 0.95]  # one 'favourable' 0.04 among nulls
    rejected, p_adj = bh_fdr(pvals, alpha=0.05)
    assert not rejected[0]  # 0.04 * 6 / 1 = 0.24 > 0.05 -> does not survive
    assert p_adj[0] == pytest.approx(0.24)


# --------------------------------------------------------------------------- AT8 (no-look-ahead guard)


def test_detection_causal_unchanged_by_future_data():
    """Detection at t is unchanged when future data is appended (reused causal detector; AT8)."""
    from noxus.catalyst.events import detect_events

    rng = np.random.default_rng(5)
    idx = pd.date_range("2021-01-03", periods=120, freq="W")
    x = rng.normal(0, 1.0, 120)
    x[60] += 6.0
    s = pd.Series(x, index=idx)
    ev_now = detect_events(s, None, None, z_thresh=2.0, meteo_screen=False)
    future = pd.Series(
        rng.normal(0, 1.0, 40),
        index=pd.date_range(idx[-1] + pd.Timedelta(weeks=1), periods=40, freq="W"),
    )
    ev_future = detect_events(pd.concat([s, future]), None, None, z_thresh=2.0, meteo_screen=False)
    up_to = ev_future[ev_future["date"] <= idx[-1]].reset_index(drop=True)
    assert set(ev_now["date"]) == set(up_to["date"])


def test_regime_study_labels_identification_descriptive(planted_regime):
    resid, bench = planted_regime
    art = R.regime_study(_fast_cfg(), resid=resid, bench=bench, calendar=None, write=False)
    assert "descriptive" in art.results["descriptive_note"].lower()


# --------------------------------------------------------------------------- AT9 (honest-null path)


def test_honest_null_verdict_is_valid_pass(ar1_null):
    """On an AR(1) null the driver concludes 'artefact' as a completed, valid outcome (AT9, AC-009)."""
    resid, bench = ar1_null
    art = R.regime_study(_fast_cfg(), resid=resid, bench=bench, calendar=None, write=False)
    assert art.verdict == R.VERDICT_ARTEFACT  # not an error / failure
    assert art.n_tests > 0


def test_planted_regime_not_artefact(planted_regime):
    """A genuine planted regime is NOT dismissed as an artefact (the null is not trigger-happy)."""
    resid, bench = planted_regime
    art = R.regime_study(_fast_cfg(), resid=resid, bench=bench, calendar=None, write=False)
    assert art.scanmax.beyond_band is True
    assert art.verdict in (R.VERDICT_EVENT_BOUND, R.VERDICT_CONTINUOUS)


# --------------------------------------------------------------------------- AT-EDGE-2 (2024 collapse dual)


def test_collapse_attribution_not_identifiable_when_both_shift():
    """Both a calendar-cadence change and an intensity shift -> not-identifiable (AT-EDGE-2, EDGE-006)."""
    idx = pd.date_range("2022-01-02", periods=160, freq="W")
    resid = pd.Series(np.zeros(160), index=idx)
    bench = pd.Series(np.zeros(160), index=idx)
    # Secular trend jumps in 2024 (intensity shift).
    trend = pd.Series(1.0, index=idx)
    trend[idx >= pd.Timestamp("2024-01-01")] = 3.0
    # Calendar cadence: 2 events/yr in 2022-23, 8 in 2024 (regime change).
    starts = ["2022-03-01", "2022-09-01", "2023-03-01", "2023-09-01"] + [
        f"2024-0{m}-01" for m in range(1, 9)
    ]
    calendar = pd.DataFrame(
        {
            "start": pd.to_datetime(starts),
            "end": pd.to_datetime(starts) + pd.Timedelta(days=5),
            "direction": ["down"] * len(starts),
            "cause": ["campaign"] * len(starts),
        }
    )
    diag = R.collapse_attribution(
        resid,
        bench,
        calendar,
        trend,
        pre_period=("2022-01-01", "2023-12-31"),
        collapse_window=("2024-01-01", "2024-12-31"),
    )
    assert diag.calendar_shift and diag.intensity_shift
    assert diag.verdict == R.COLLAPSE_NOT_IDENTIFIABLE


def test_collapse_attribution_single_cause():
    """Only an intensity shift (stable cadence) -> intensity-decoupling, not forced regime-change."""
    idx = pd.date_range("2022-01-02", periods=160, freq="W")
    resid = pd.Series(np.zeros(160), index=idx)
    bench = pd.Series(np.zeros(160), index=idx)
    trend = pd.Series(1.0, index=idx)
    trend[idx >= pd.Timestamp("2024-01-01")] = 4.0
    starts = ["2022-03-01", "2022-09-01", "2023-03-01", "2023-09-01", "2024-03-01", "2024-09-01"]
    calendar = pd.DataFrame(
        {
            "start": pd.to_datetime(starts),
            "end": pd.to_datetime(starts) + pd.Timedelta(days=5),
            "direction": ["down"] * len(starts),
            "cause": ["campaign"] * len(starts),
        }
    )
    diag = R.collapse_attribution(
        resid,
        bench,
        calendar,
        trend,
        pre_period=("2022-01-01", "2023-12-31"),
        collapse_window=("2024-01-01", "2024-12-31"),
    )
    assert diag.intensity_shift and not diag.calendar_shift
    assert diag.verdict == R.COLLAPSE_INTENSITY


# --------------------------------------------------------------------------- AT-EDGE-1 (low-power label)


def test_regime_event_study_labels_low_power():
    """A sparse-event regime is labelled low-power, not omitted (EDGE-003, AT-EDGE-1)."""
    idx = pd.date_range("2023-01-01", periods=30, freq="W")
    resid = pd.Series(np.zeros(30), index=idx)
    bench_df = pd.DataFrame({"date": idx, "value": np.zeros(30)})
    regimes = {"peak": (idx.min(), idx.max())}
    study = R.regime_event_study(
        resid,
        None,
        None,
        bench_df,
        None,
        regimes=regimes,
        meteo_screen=False,
        min_events_per_regime=3,
    )
    assert study.low_power["peak"] is True
    assert "peak" in study.by_regime  # reported, not omitted


# --------------------------------------------------------------------------- determinism (AT10)


def test_regime_study_deterministic(planted_regime):
    resid, bench = planted_regime
    a = R.regime_study(_fast_cfg(), resid=resid, bench=bench, calendar=None, write=False)
    b = R.regime_study(_fast_cfg(), resid=resid, bench=bench, calendar=None, write=False)
    pd.testing.assert_frame_equal(a.table, b.table)
    pd.testing.assert_frame_equal(a.event_study_table, b.event_study_table)
    assert a.verdict == b.verdict
    assert a.n_tests == b.n_tests


# --------------------------------------------------------------------------- F1 (event-study table wired)


def test_driver_emits_event_study_table_and_figure(planted_regime, tmp_path):
    """The driver produces the REQ-020 event-study table (not just the unit function), and the figure
    driver renders the precision/recall PNG (F1)."""
    from dataclasses import replace

    resid, bench = planted_regime
    cfg = replace(_fast_cfg(), out_dir=tmp_path, figures_dir=tmp_path / "figs")
    art = R.regime_study(cfg, resid=resid, bench=bench, calendar=None, write=True)

    # The event-study table is carried through RegimeArtifacts and written to disk.
    assert isinstance(art.event_study_table, pd.DataFrame) and len(art.event_study_table) > 0
    for col in ("threshold", "regime", "precision", "recall", "median_lead_days"):
        assert col in art.event_study_table.columns
    assert art.event_study_path is not None and art.event_study_path.exists()
    # One row per (threshold × regime) — the threshold dimension is genuinely exercised here.
    assert set(art.event_study_table["threshold"]) == set(cfg.thresholds)

    # The figure driver renders the 4th figure its docstring promises.
    import analysis.regime_2023_event_study as A

    png = A._fig_precision_recall(art, tmp_path / "pr.png")
    assert png.exists() and png.stat().st_size > 0


# --------------------------------------------------------------------------- F2 (no fabricated multiplicity)


def test_fdr_grid_has_no_duplicate_pvalues(planted_regime):
    """Every correlation-grid cell is a genuinely-distinct test; n_tests is the true distinct count
    with NO duplicate p-values padded into BH-FDR (F2)."""
    resid, bench = planted_regime
    art = R.regime_study(_fast_cfg(), resid=resid, bench=bench, calendar=None, write=False)
    # n_tests equals the number of rows actually computed (no padding).
    assert art.n_tests == len(art.table)
    # No two rows share the same (variant, window, regime) identity.
    ids = list(zip(art.table["variant"], art.table["window"], art.table["regime"]))
    assert len(ids) == len(set(ids))


def test_collapsed_variants_do_not_inflate_n_tests(planted_regime):
    """Nominal variants that collapse to the same residual offline are deduped — n_tests is NOT
    doubled by an inert 'variant' dimension (F2)."""
    from dataclasses import replace

    resid, bench = planted_regime
    one = R.regime_study(
        replace(_fast_cfg(), variants=("harmonic",)),
        resid=resid,
        bench=bench,
        calendar=None,
        write=False,
    )
    two = R.regime_study(
        replace(_fast_cfg(), variants=("harmonic", "intensity-model")),
        resid=resid,
        bench=bench,
        calendar=None,
        write=False,
    )
    assert two.results["variants_distinct"] == ["harmonic"]  # collapsed offline
    assert two.n_tests == one.n_tests  # not inflated


# --------------------------------------------------------------------------- F3 (52-week secondary window)


def test_secondary_window_is_exercised(planted_regime):
    """REQ-010's 52-week secondary rolling window is run alongside the 18-month primary (F3)."""
    resid, bench = planted_regime
    art = R.regime_study(_fast_cfg(), resid=resid, bench=bench, calendar=None, write=False)
    assert art.rolling_secondary is not None
    assert art.results["secondary_window"] == 52
    # A distinct 52-week peak cell appears in the correlation grid.
    assert 52 in set(art.table["window"])
    assert 78 in set(art.table["window"])


# --------------------------------------------------------------------------- F4 (cloud-gap honesty, EDGE-004)


def test_cloud_gapped_curtailment_is_honest_missed_recall():
    """A curtailment hidden by a cloud gap (low coverage) is NOT detected as an NO2 event, so it
    surfaces as an honest missed recall rather than a silently absorbed false negative (EDGE-004)."""
    rng = np.random.default_rng(21)
    n, drop_at = 80, 50
    idx = pd.date_range("2023-01-01", periods=n, freq="W")
    x = rng.normal(0, 1.0, n)  # noise so the causal robust baseline has non-zero scale
    x[drop_at] = -8.0  # a real production drop, past the causal-baseline warm-up
    resid = pd.Series(x, index=idx)
    # A curtailment calendar 'down' at the same week (ground-truth production down).
    calendar = pd.DataFrame(
        {
            "start": [idx[drop_at]],
            "end": [idx[drop_at]],
            "direction": ["down"],
            "cause": ["cloud_gapped_curtailment"],
        }
    )
    bench_df = pd.DataFrame({"date": idx, "value": np.zeros(n)})
    regimes = {"peak": (idx.min(), idx.max())}

    # Clear sky: the drop IS detected as an NO2 'drop' event.
    clear_cov = pd.Series(0.9, index=idx)
    clear = R.regime_event_study(
        resid, clear_cov, None, bench_df, calendar, regimes=regimes, meteo_screen=False
    )
    clear_mr = clear.by_regime["peak"]
    assert clear_mr.n_matched >= 1
    assert idx[drop_at] in set(pd.to_datetime(clear_mr.matched["no2_date"]))

    # Cloud gap at the drop week: coverage below the floor -> the drop is screened out, NOT detected.
    cloud_cov = clear_cov.copy()
    cloud_cov.iloc[drop_at] = 0.05
    cloudy = R.regime_event_study(
        resid,
        cloud_cov,
        None,
        bench_df,
        calendar,
        regimes=regimes,
        meteo_screen=False,
        min_coverage=0.25,
    )
    mr = cloudy.by_regime["peak"]
    detected_no2 = set(
        pd.to_datetime(mr.matched.get("no2_date", pd.Series([], dtype="datetime64[ns]")))
    )
    assert idx[drop_at] not in detected_no2  # cloud-gapped drop not detected
    # The curtailment is still counted as ground truth -> the miss is an honest lost recall, not
    # a silently absorbed false negative.
    assert mr.n_prod >= 1
    assert mr.recall < clear_mr.recall


# --------------------------------------------------------------------------- B3 (weekday-anchor alignment)


def _misaligned_pair(n_weeks=260):
    """A Sunday-anchored weekly residual + a Friday-dated near-daily benchmark (the real-run shape).

    Mirrors the on-disk artifacts: the NOX-003.1 residual is Sunday-anchored weekly while the CREA
    benchmark is mostly Friday-dated (near-daily). With an exact-date inner join only a handful of
    dates coincide, so any rolling correlation dies with "series too short" (B3). The two series
    carry a genuine weekly-scale coupling so that, once weekly-aligned, a correlation exists.
    """
    rng = np.random.default_rng(42)
    sun_idx = pd.date_range("2019-01-06", periods=n_weeks, freq="W-SUN")  # Sunday-anchored weekly
    base = rng.normal(0, 1.0, n_weeks)
    resid = pd.Series(base, index=sun_idx, name="residual_activity")
    # A Friday-anchored near-daily benchmark spanning the same era; its weekly mean tracks `base`
    # inside a coupled block so the aligned correlation is well-defined.
    fri_idx = pd.date_range("2019-01-04", periods=n_weeks, freq="W-FRI")
    bench_vals = base + rng.normal(0, 0.2, n_weeks)
    bench = pd.Series(bench_vals, index=fri_idx, name="value")
    # Add extra mid-week Wednesday points so the benchmark is not itself on a single weekday.
    wed_idx = pd.date_range("2019-01-02", periods=n_weeks, freq="W-WED")
    bench_extra = pd.Series(base + rng.normal(0, 0.2, n_weeks), index=wed_idx, name="value")
    bench = pd.concat([bench, bench_extra]).sort_index()
    return resid, bench


def test_rolling_regime_aligns_mismatched_weekday_anchors():
    """B3 regression: a Sunday residual + Friday/Wednesday benchmark must weekly-align and correlate,
    NOT raise 'series too short'. Exact-date inner join yields ~0 common dates; this fails without the
    resample-to-freq fix in _join and passes with it."""
    resid, bench = _misaligned_pair()
    # Sanity: the two series share almost no exact dates (the bug's precondition).
    common = set(pd.to_datetime(resid.index)) & set(pd.to_datetime(bench.index))
    assert len(common) < 5
    res = R.rolling_regime(resid, bench, window=WINDOW, n_boot=100, n_perm=100, freq="W")
    assert res.peak_r > 0.5  # a real coupling recovered after weekly alignment
    assert res.segments["peak"].n > WINDOW // 2


def test_join_without_resample_would_be_too_short():
    """The exact-date join (freq=None) on mismatched anchors is what raised 'series too short'; the
    weekly-aligned join (freq='W') recovers a usable overlap (B3)."""
    resid, bench = _misaligned_pair()
    exact = R._join(resid, bench, freq=None)
    weekly = R._join(resid, bench, freq="W")
    assert len(exact) < WINDOW  # too short for the rolling window -> the original crash
    assert len(weekly) >= WINDOW  # weekly alignment restores the overlap


def test_regime_study_end_to_end_on_mismatched_anchors():
    """The full driver runs (no 'series too short') on real-shaped mismatched-anchor inputs (B3)."""
    resid, bench = _misaligned_pair()
    art = R.regime_study(_fast_cfg(), resid=resid, bench=bench, calendar=None, write=False)
    assert art.n_tests > 0
    assert art.verdict in (
        R.VERDICT_EVENT_BOUND,
        R.VERDICT_CONTINUOUS,
        R.VERDICT_ARTEFACT,
        R.VERDICT_INCONCLUSIVE,
    )


# --------------------------------------------------------------------------- B4 (secular-trend column contract)


def test_load_series_reads_secular_trend_from_writer_column(tmp_path):
    """B4 regression: the NOX-003.1 writer emits the secular trend as 'trend_s_t' (the real column
    names are date/signal/trend_s_t/residual_activity). The loader must read it — not the stale
    's_trend' name — so secular_trend is non-None and the collapse intensity arm is exercised, not
    forced-inert. Fails before the fix (loader read 's_trend' -> None), passes after."""
    from dataclasses import replace

    from noxus.config.run import RegimeStudyConfig

    n = 340  # span 2019-01-06 into 2025 so the pre-period (2022-23) and 2024 collapse both have data
    idx = _weekly_index(n)
    rng = np.random.default_rng(9)
    t = np.arange(n)
    # A decomp frame with the EXACT writer column names.
    decomp = pd.DataFrame(
        {
            "date": idx,
            "signal": 5.0 + rng.normal(0, 0.3, n),
            "trend_s_t": 5.0 - 0.01 * t,  # a real secular intensity decline
            "residual_activity": rng.normal(0, 1.0, n),
        }
    )
    decomp_path = tmp_path / "steel_intensity_decomposition.parquet"
    decomp.to_parquet(decomp_path)
    bench = pd.DataFrame(
        {"date": pd.date_range("2019-01-04", periods=n, freq="W-FRI"), "value": rng.normal(0, 1, n)}
    )
    bench_path = tmp_path / "benchmark.parquet"
    bench.to_parquet(bench_path)

    cfg = replace(RegimeStudyConfig(), decomposition_path=decomp_path, benchmark_path=bench_path)
    _resid, _bench, secular, _decomp = R._load_series_from_cfg(cfg)
    assert secular is not None  # the whole point of B4
    assert len(secular) == len(decomp)
    assert np.isfinite(secular.to_numpy()).all()

    # The collapse diagnostic's intensity arm is genuinely exercised (finite pre/collapse means),
    # not forced-inert by a missing secular trend.
    diag = R.collapse_attribution(
        _resid,
        _bench.resample("W").mean(),
        None,
        secular,
        pre_period=cfg.pre_period,
        collapse_window=cfg.collapse_window,
    )
    assert np.isfinite(diag.intensity_pre)
    assert np.isfinite(diag.intensity_collapse)


def test_load_series_secular_trend_legacy_fallback(tmp_path):
    """A legacy decomp using the old 's_trend' name still loads (fallback), and one lacking any
    trend column yields None (genuinely absent) — never a crash (B4)."""
    from dataclasses import replace

    from noxus.config.run import RegimeStudyConfig

    idx = _weekly_index(120)
    rng = np.random.default_rng(4)
    bench = pd.DataFrame({"date": idx, "value": rng.normal(0, 1, 120)})
    bench_path = tmp_path / "b.parquet"
    bench.to_parquet(bench_path)

    legacy = pd.DataFrame(
        {"date": idx, "s_trend": np.linspace(5, 4, 120), "residual_activity": rng.normal(0, 1, 120)}
    )
    lp = tmp_path / "legacy.parquet"
    legacy.to_parquet(lp)
    cfg = replace(RegimeStudyConfig(), decomposition_path=lp, benchmark_path=bench_path)
    _, _, sec_legacy, _ = R._load_series_from_cfg(cfg)
    assert sec_legacy is not None  # legacy name accepted as fallback

    absent = pd.DataFrame({"date": idx, "residual_activity": rng.normal(0, 1, 120)})
    ap = tmp_path / "absent.parquet"
    absent.to_parquet(ap)
    cfg2 = replace(RegimeStudyConfig(), decomposition_path=ap, benchmark_path=bench_path)
    _, _, sec_absent, _ = R._load_series_from_cfg(cfg2)
    assert sec_absent is None  # genuinely absent -> None, no crash


# --------------------------------------------------------------------------- B1 (harmonic variant on file path)


def test_harmonic_variant_on_decomp_frame_does_not_raise():
    """B1 regression: the 'harmonic' variant through the real file path (a decomp frame carrying a
    'signal' column) re-deseasonalises without AttributeError and yields a series distinct from the
    passed intensity residual. This crashed before harmonic_order was added to RegimeStudyConfig."""
    from noxus.config.run import RegimeStudyConfig

    idx = _weekly_index(200)
    rng = np.random.default_rng(3)
    # A raw footprint 'signal' with an annual cycle (what the file path deseasonalises).
    t = np.arange(200)
    signal = 5.0 + 2.0 * np.sin(2 * np.pi * t / 52.0) + rng.normal(0, 0.3, 200)
    resid = pd.Series(rng.normal(0, 1.0, 200), index=idx)  # the intensity residual as passed
    decomp = pd.DataFrame({"date": idx, "signal": signal})
    cfg = RegimeStudyConfig()
    out = R._deseason_variant(resid, "harmonic", decomp, cfg)
    assert isinstance(out, pd.Series)
    assert len(out) == len(resid)
    # Genuinely distinct from the intensity residual (not a silent fallback).
    assert not np.allclose(np.nan_to_num(out.to_numpy()), np.nan_to_num(resid.to_numpy()))


def _fast_cfg():
    """A RegimeStudyConfig with small draw counts for fast, deterministic tests."""
    from dataclasses import replace

    from noxus.config.run import RegimeStudyConfig

    return replace(
        RegimeStudyConfig(),
        n_boot=200,
        n_perm=200,
        scanmax_draws=200,
        variants=("harmonic",),
    )
