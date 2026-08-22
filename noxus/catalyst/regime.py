"""2023-regime event study: rolling regime, jackknife, dual collapse diagnostic, scan-max null (NOX-012).

Built **on top of** the NOX-004 catalyst and NOX-008 robust-significance machinery, never parallel to
it. Everything statistical is reused read-only:

- detection / ground-truth / matching — ``noxus.catalyst.events.detect_events``,
  ``groundtruth.{bf_rate_events,load_curtailment_calendar,production_events}``, ``study.match_events``;
- autocorrelation-robust significance — ``noxus.validation.robust.{robust_corr, block_permutation_p,
  bh_fdr, block_length}``; lead-lag — ``noxus.validation.leadlag.lead_lag``.

The genuinely new pieces are (a) rolling-regime identification with a **fixed pre-registered rule**
(peak-centred window + fixed calendar-year collapse split, Q4/A5 — never tuned to CREA), (b) a
**leave-one-large-event-out jackknife** that tests whether the 2023 peak is *event-bound* or a
*continuous* coupling, (c) a **dual 2024-collapse diagnostic** (calendar-cadence change vs NOX-003.1
secular-trend behaviour) that refuses to force a single verdict when both fit (EDGE-006), and (d) a
**rolling-window scan-maximum null**: under a serial-dependence-preserving block-permutation null, the
distribution of the *maximum* rolling correlation attainable across the many overlapping windows, so
the r≈0.76 peak is judged against the multiplicity it arose from (Morris & Zhang 2019). The honest-null
"multiple-comparisons artefact" verdict is a **first-class, valid outcome** (AC-009).

No look-ahead: detection uses the reused causal detector; curtailment dates are exogenous/public; the
grid and windows are fixed before any outcome is viewed. Regime *identification* is descriptive, not
predictive, and is labelled so in the output.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from noxus.catalyst.events import detect_events
from noxus.catalyst.groundtruth import (
    bf_rate_events,
    load_curtailment_calendar,
    production_events,
)
from noxus.catalyst.study import MatchResult, match_events
from noxus.validation.robust import (
    RobustCorr,
    bh_fdr,
    block_length,
    robust_corr,
)

# Verdict labels (stable strings tests key on).
VERDICT_EVENT_BOUND = "event-bound"
VERDICT_CONTINUOUS = "continuous"
VERDICT_ARTEFACT = "multiple-comparisons-artefact"
VERDICT_INCONCLUSIVE = "inconclusive"

COLLAPSE_REGIME_CHANGE = "regime-change"
COLLAPSE_INTENSITY = "intensity-decoupling"
COLLAPSE_NOT_IDENTIFIABLE = "not-identifiable"
COLLAPSE_INCONCLUSIVE = "inconclusive"


def _join(resid: pd.Series, bench: pd.Series, freq: str | None = "W") -> pd.DataFrame:
    """Weekly-align two date-indexed series to ``freq`` then inner-join on jointly-observed periods.

    B3 fix: the NOX-003.1 residual is Sunday-anchored weekly while the CREA benchmark is mostly
    Friday-dated, so an exact-date inner join keeps only the handful of coincident dates (47 on the
    real series) and every downstream rolling correlation then dies with "series too short". Both
    series are first composited to the configured cadence with ``.resample(freq).mean()`` — the
    canonical convention already used by ``noxus.validation.preprocess.align_series`` and
    ``noxus.validation.scale`` — and then inner-joined on the periods observed in **both** (no
    interpolation, no look-ahead, no fabricated coverage). ``freq=None`` disables resampling for
    callers that pass series already on a shared index (offline test fixtures).
    """
    x = pd.Series(resid).astype(float)
    y = pd.Series(bench).astype(float)
    x.index = pd.to_datetime(x.index)
    y.index = pd.to_datetime(y.index)
    if freq is not None:
        x = x.resample(freq).mean()
        y = y.resample(freq).mean()
    df = pd.concat({"x": x, "y": y}, axis=1)
    return df.dropna().sort_index()


# --------------------------------------------------------------------------- rolling regime (REQ-010/011)


@dataclass(frozen=True)
class RegimeSegment:
    """One regime segment scored with autocorrelation-robust significance (REQ-011)."""

    label: str
    start: pd.Timestamp
    end: pd.Timestamp
    n: int
    corr: RobustCorr
    low_power: bool  # too few observations for a stable estimate (EDGE-003)


@dataclass(frozen=True)
class RegimeResult:
    """Rolling-correlation regime identification (descriptive, not predictive; REQ-010/011)."""

    rolling_r: pd.Series
    window: int
    peak_date: pd.Timestamp
    peak_r: float
    collapse_date: pd.Timestamp
    collapse_r: float
    peak_window: tuple[pd.Timestamp, pd.Timestamp]
    segments: dict[str, RegimeSegment]
    boundary_sweep: pd.DataFrame  # shift, start, end, r (REQ-011 robustness sweep)
    descriptive_note: str = "Regime identification is descriptive/in-sample, not a forward-predictive claim (no-look-ahead)."


def rolling_regime(
    resid: pd.Series,
    bench: pd.Series,
    *,
    window: int,
    min_obs_per_segment: int = 12,
    n_boot: int = 5000,
    n_perm: int = 5000,
    seed: int = 20260720,
    boundary_shifts: tuple[int, ...] = (-4, 0, 4),
    freq: str | None = "W",
) -> RegimeResult:
    """Recompute the rolling correlation, locate the peak/collapse, segment by a fixed rule (REQ-010/011).

    The rolling ``window``-period Pearson correlation of ``resid`` against ``bench`` is computed on the
    jointly-observed dates; the **peak** is the label of maximum rolling r and the **collapse** is the
    minimum rolling r *after* the peak. The "peak" regime is the ``window`` observations ending at the
    peak label (the window that produced the peak — a fixed rule, i.e. peak-centred on that window's
    midpoint); "off_peak" is the remainder. Each segment is scored with ``robust_corr`` (unchanged).
    A boundary-robustness sweep recomputes the peak-window r at shifted boundaries (A5).

    The function exposes **no** CREA-agreement / optimisation argument: the boundary is never selected
    to maximise the within-regime correlation (NFR-005, ERR-004).

    ``freq`` weekly-aligns the residual/benchmark pair before correlating (B3); ``None`` for
    already-aligned series.
    """
    df = _join(resid, bench, freq=freq)
    min_periods = max(3, window // 2)
    rolling = df["x"].rolling(window, min_periods=min_periods).corr(df["y"])
    valid = rolling.dropna()
    if valid.empty:
        raise ValueError(
            "rolling_regime: no valid rolling-correlation windows (series too short for the window)."
        )

    peak_date = valid.idxmax()
    peak_r = float(valid.loc[peak_date])
    after = valid[valid.index > peak_date]
    if len(after):
        collapse_date = after.idxmin()
        collapse_r = float(after.loc[collapse_date])
    else:
        collapse_date, collapse_r = peak_date, peak_r

    idx = df.index
    peak_pos = idx.get_indexer([peak_date])[0]
    lo_pos = max(0, peak_pos - window + 1)
    peak_start, peak_end = idx[lo_pos], idx[peak_pos]

    peak_mask = (idx >= peak_start) & (idx <= peak_end)
    off_mask = ~peak_mask

    segments: dict[str, RegimeSegment] = {}
    for label, mask in (("peak", peak_mask), ("off_peak", off_mask)):
        sub = df[mask]
        n = int(len(sub))
        corr = robust_corr(sub["x"], sub["y"], n_boot=n_boot, n_perm=n_perm, seed=seed)
        segments[label] = RegimeSegment(
            label=label,
            start=sub.index.min() if n else peak_start,
            end=sub.index.max() if n else peak_end,
            n=n,
            corr=corr,
            low_power=n < min_obs_per_segment,
        )

    sweep_rows = []
    for shift in boundary_shifts:
        s_lo = max(0, min(len(idx) - 1, lo_pos + shift))
        s_hi = max(0, min(len(idx) - 1, peak_pos + shift))
        if s_hi <= s_lo:
            continue
        sub = df.iloc[s_lo : s_hi + 1]
        r = float(np.corrcoef(sub["x"], sub["y"])[0, 1]) if len(sub) > 2 else float("nan")
        sweep_rows.append({"shift": shift, "start": idx[s_lo], "end": idx[s_hi], "r": r})
    boundary_sweep = pd.DataFrame(sweep_rows, columns=["shift", "start", "end", "r"])

    return RegimeResult(
        rolling_r=rolling,
        window=window,
        peak_date=peak_date,
        peak_r=peak_r,
        collapse_date=collapse_date,
        collapse_r=collapse_r,
        peak_window=(peak_start, peak_end),
        segments=segments,
        boundary_sweep=boundary_sweep,
    )


# --------------------------------------------------------------------------- event-bound jackknife (REQ-021)


@dataclass(frozen=True)
class JackknifeResult:
    """Leave-one-large-event-out attribution of the peak r (REQ-021, AT4)."""

    verdict: str  # event-bound | continuous | insufficient-events
    refused: bool  # True when < min_events (ERR-003) -> fall back to coarse
    full_r: float
    r_without_top: float
    n_events: int
    n_top_removed: int
    contributions: pd.DataFrame  # event_date, loo_r, contribution
    note: str = ""


def event_bound_jackknife(
    resid: pd.Series,
    bench: pd.Series,
    events: pd.DataFrame,
    *,
    window_periods: int = 4,
    min_events: int = 3,
    drop_frac: float = 0.5,
    freq: str | None = "W",
) -> JackknifeResult:
    """Leave-one-large-event-out on the peak regime: is the r event-bound or continuous? (REQ-021).

    ``resid``/``bench`` are already restricted to the peak regime. For each event in ``events`` (dated),
    the observations within ``window_periods`` of the event are removed and the correlation recomputed
    (``loo_r``); an event's ``contribution`` = ``full_r - loo_r`` (how much of the peak leans on it).
    Removing the top ``min(3, n)`` highest-contribution events and re-measuring: if that collapses the
    peak by >= ``drop_frac`` the coupling is **event-bound**; otherwise **continuous** (no single event
    dominates). Below ``min_events`` securely-dated events the event-level test is **refused** and the
    study falls back to the coarse calendar (ERR-003) — it does not emit an unstable rate.

    Exposes **no** CREA-agreement objective (NFR-005, ERR-004). ``freq`` weekly-aligns the pair
    before correlating (B3); ``None`` for already-aligned series.
    """
    df = _join(resid, bench, freq=freq)
    full_r = float(np.corrcoef(df["x"], df["y"])[0, 1]) if len(df) > 2 else float("nan")
    ev_dates = (
        pd.to_datetime(pd.Series(events["date"]).reset_index(drop=True))
        if len(events)
        else pd.Series([], dtype="datetime64[ns]")
    )
    n_events = int(len(ev_dates))

    if n_events < min_events:
        return JackknifeResult(
            verdict="insufficient-events",
            refused=True,
            full_r=full_r,
            r_without_top=full_r,
            n_events=n_events,
            n_top_removed=0,
            contributions=pd.DataFrame(columns=["event_date", "loo_r", "contribution"]),
            note=(
                f"Refused event-level jackknife: {n_events} < min_events={min_events}; fall back to "
                "the coarse-calendar study (ERR-003)."
            ),
        )

    win = pd.Timedelta(days=window_periods * 7)
    rows = []
    for d in ev_dates:
        keep = (df.index < d - win) | (df.index > d + win)
        sub = df[keep]
        loo_r = float(np.corrcoef(sub["x"], sub["y"])[0, 1]) if len(sub) > 2 else float("nan")
        rows.append({"event_date": d, "loo_r": loo_r, "contribution": full_r - loo_r})
    contributions = (
        pd.DataFrame(rows).sort_values("contribution", ascending=False).reset_index(drop=True)
    )

    n_top = min(3, n_events)
    top_dates = contributions["event_date"].head(n_top).tolist()
    keep_top = pd.Series(True, index=df.index)
    for d in top_dates:
        keep_top &= (df.index < d - win) | (df.index > d + win)
    sub_top = df[keep_top.to_numpy()]
    r_without_top = (
        float(np.corrcoef(sub_top["x"], sub_top["y"])[0, 1]) if len(sub_top) > 2 else float("nan")
    )

    collapsed = (
        np.isfinite(full_r)
        and np.isfinite(r_without_top)
        and abs(full_r) > 1e-9
        and (full_r - r_without_top) / abs(full_r) >= drop_frac
    )
    verdict = VERDICT_EVENT_BOUND if collapsed else VERDICT_CONTINUOUS
    note = (
        f"Removing the top {n_top} events moved r {full_r:.3f} -> {r_without_top:.3f}; "
        f"{'event-bound' if collapsed else 'continuous'} coupling."
    )
    return JackknifeResult(
        verdict=verdict,
        refused=False,
        full_r=full_r,
        r_without_top=r_without_top,
        n_events=n_events,
        n_top_removed=n_top,
        contributions=contributions,
        note=note,
    )


# --------------------------------------------------------------------------- regime event study (REQ-020)


@dataclass(frozen=True)
class RegimeEventStudy:
    """Per-regime precision/recall/lead reusing the NOX-004 primitives (REQ-020, AT5)."""

    by_regime: dict[str, MatchResult]
    low_power: dict[str, bool]
    calendar_used: bool
    note: str = ""


def regime_event_study(
    resid: pd.Series,
    coverage: pd.Series | None,
    meteo: pd.DataFrame | None,
    bench: pd.DataFrame,
    calendar: pd.DataFrame | None,
    *,
    regimes: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
    z_thresh: float = 2.0,
    bf_event_z: float = 1.5,
    match_window: int = 2,
    min_periods: int = 12,
    min_coverage: float = 0.25,
    meteo_screen: bool = True,
    ventilation_z: float = 1.5,
    min_events_per_regime: int = 3,
) -> RegimeEventStudy:
    """Regime-conditioned event study: precision/recall/FDR/lead per regime, **not** a Pearson r.

    Reuses ``detect_events`` (causal + coverage + ventilation screen, unchanged), ``bf_rate_events`` +
    ``production_events`` (BF jumps combined with the calendar; heating-season blankets are edge-matched
    via the NOX-004 ``_calendar_edge_events``, EDGE-002), and ``match_events`` (unchanged). Detection is
    run once on the full series (causal, no look-ahead) and events are then partitioned by regime, so a
    regime boundary never changes a detection baseline. A regime with too few events is labelled
    low-power (EDGE-003), never silently omitted; a cloud-gapped curtailment simply fails to detect and
    is an honest missed recall (EDGE-004, inherited from the coverage screen).
    """
    no2_all = detect_events(
        resid,
        coverage,
        meteo,
        z_thresh=z_thresh,
        min_periods=min_periods,
        min_coverage=min_coverage,
        meteo_screen=meteo_screen,
        ventilation_z=ventilation_z,
    )
    cal = calendar if calendar is not None else load_curtailment_calendar(None)
    prod_all = production_events(bf_rate_events(bench, z_thresh=bf_event_z), cal)
    calendar_used = cal is not None and len(cal) > 0

    no2_all = no2_all.copy()
    no2_all["date"] = pd.to_datetime(no2_all["date"])
    prod_all = prod_all.copy()
    prod_all["date"] = pd.to_datetime(prod_all["date"])

    window_days = float(match_window) * 7.0
    by_regime: dict[str, MatchResult] = {}
    low_power: dict[str, bool] = {}
    for label, (lo, hi) in regimes.items():
        n2 = no2_all[(no2_all["date"] >= lo) & (no2_all["date"] <= hi)]
        pr = prod_all[(prod_all["date"] >= lo) & (prod_all["date"] <= hi)]
        by_regime[label] = match_events(n2, pr, window_days=window_days)
        low_power[label] = (len(n2) < min_events_per_regime) or (len(pr) < min_events_per_regime)

    note = (
        "Headline is an event-study metric (precision/recall/FDR/lead), never a Pearson r; "
        "heating-season blankets edge-matched (EDGE-002); low-power regimes labelled (EDGE-003)."
    )
    if not calendar_used:
        note += " Calendar absent -> BF-rate jumps only (ERR-002)."
    return RegimeEventStudy(
        by_regime=by_regime, low_power=low_power, calendar_used=calendar_used, note=note
    )


# --------------------------------------------------------------------------- 2024 collapse (REQ-022)


@dataclass(frozen=True)
class CollapseDiagnosis:
    """Dual 2024-collapse diagnostic: no forced verdict when both fit (REQ-022, EDGE-006)."""

    verdict: str  # regime-change | intensity-decoupling | not-identifiable | inconclusive
    calendar_shift: bool
    intensity_shift: bool
    pre_events_per_year: float
    collapse_events: int
    intensity_pre: float
    intensity_collapse: float
    note: str = ""


def collapse_attribution(
    resid: pd.Series,
    bench: pd.Series,
    calendar: pd.DataFrame | None,
    secular_trend: pd.Series | None,
    *,
    pre_period: tuple[str, str],
    collapse_window: tuple[str, str],
    cadence_tol: float = 0.5,
    intensity_tol: float = 0.5,
) -> CollapseDiagnosis:
    """Report the two competing 2024-collapse causes side by side; force no single verdict (REQ-022).

    Arm (a) **curtailment-regime change**: the 2024 calendar event count vs the mean annual count over
    the pre-period (2022–2023). Arm (b) **accelerating intensity-decoupling**: the mean of the reused
    NOX-003.1 secular-trend / intensity residual in the collapse window vs the pre-period (A8, read
    only). If **both** shifts are present the verdict is ``not-identifiable`` (EDGE-006); if only one,
    that cause; if neither, ``inconclusive``. The estimator never forces one cause when both are
    consistent with the data.
    """
    pre_lo, pre_hi = pd.Timestamp(pre_period[0]), pd.Timestamp(pre_period[1])
    coll_lo, coll_hi = pd.Timestamp(collapse_window[0]), pd.Timestamp(collapse_window[1])

    # Arm (a): calendar cadence.
    if calendar is not None and len(calendar):
        starts = pd.to_datetime(calendar["start"])
        pre_events = int(((starts >= pre_lo) & (starts <= pre_hi)).sum())
        coll_events = int(((starts >= coll_lo) & (starts <= coll_hi)).sum())
        pre_years = max(1.0, (pre_hi.year - pre_lo.year) + 1)
        pre_per_year = pre_events / pre_years
    else:
        pre_per_year, coll_events = float("nan"), 0
    calendar_shift = (
        np.isfinite(pre_per_year)
        and pre_per_year > 0
        and abs(coll_events - pre_per_year) / pre_per_year >= cadence_tol
    )

    # Arm (b): NOX-003.1 secular-trend behaviour (read-only).
    if secular_trend is not None and len(secular_trend):
        st = pd.Series(secular_trend).astype(float)
        st.index = pd.to_datetime(st.index)
        pre_st = float(st[(st.index >= pre_lo) & (st.index <= pre_hi)].mean())
        coll_st = float(st[(st.index >= coll_lo) & (st.index <= coll_hi)].mean())
    else:
        pre_st, coll_st = float("nan"), float("nan")
    intensity_shift = (
        np.isfinite(pre_st)
        and np.isfinite(coll_st)
        and abs(pre_st) > 1e-12
        and abs(coll_st - pre_st) / abs(pre_st) >= intensity_tol
    )

    if calendar_shift and intensity_shift:
        verdict = COLLAPSE_NOT_IDENTIFIABLE
    elif calendar_shift:
        verdict = COLLAPSE_REGIME_CHANGE
    elif intensity_shift:
        verdict = COLLAPSE_INTENSITY
    else:
        verdict = COLLAPSE_INCONCLUSIVE
    note = (
        "Both diagnostics reported side by side; attribution not forced when both fit (EDGE-006)."
        if verdict == COLLAPSE_NOT_IDENTIFIABLE
        else "Single arm shifted."
        if verdict in (COLLAPSE_REGIME_CHANGE, COLLAPSE_INTENSITY)
        else "Neither arm shifted decisively."
    )
    return CollapseDiagnosis(
        verdict=verdict,
        calendar_shift=bool(calendar_shift),
        intensity_shift=bool(intensity_shift),
        pre_events_per_year=pre_per_year,
        collapse_events=coll_events,
        intensity_pre=pre_st,
        intensity_collapse=coll_st,
        note=note,
    )


# --------------------------------------------------------------------------- scan-maximum null (REQ-030)


@dataclass(frozen=True)
class ScanMaxNull:
    """Rolling-window scan-maximum null: is the peak surprising given the windows scanned? (REQ-030)."""

    observed_scanmax: float
    p_value: float  # P(null scan-max >= observed); high => unsurprising (artefact-consistent)
    percentile: float  # observed percentile within the null distribution
    null_hi: float  # upper (1-alpha) quantile of the null scan-max
    block: int
    n_windows: int
    n_draws: int
    beyond_band: bool  # observed beyond the null (1-alpha) band


def scan_max_null(
    resid: pd.Series,
    bench: pd.Series,
    *,
    window: int,
    n_draws: int = 2000,
    block: int | None = None,
    seed: int = 20260720,
    alpha: float = 0.05,
    freq: str | None = "W",
) -> ScanMaxNull:
    """Block-permutation distribution of the MAX rolling-``window`` correlation across windows (REQ-030).

    The overlapping rolling-window scan yields many highly-autocorrelated correlations; the naive "one
    window hit 0.76" is never reported without this null (EDGE-005). Under a serial-dependence-preserving
    null (contiguous moving-block permutation of ``resid`` against ``bench`` — the NOX-008 block
    convention, reusing ``robust.block_length``) the maximum rolling correlation over the whole scan is
    recorded per draw. The observed scan-max's percentile / p-value against that distribution is the
    reported significance: a high p-value means the peak is **unsurprising** given how many windows were
    scanned (an honest-null, artefact-consistent outcome). ``freq`` weekly-aligns the pair before
    scanning (B3); ``None`` for already-aligned series.
    """
    df = _join(resid, bench, freq=freq)
    n = len(df)
    min_periods = max(3, window // 2)
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()

    def _scanmax(xv: np.ndarray) -> float:
        r = pd.Series(xv, index=df.index).rolling(window, min_periods=min_periods).corr(df["y"])
        r = r.dropna()
        return float(r.max()) if len(r) else float("nan")

    observed = _scanmax(x)
    n_windows = int(
        pd.Series(x, index=df.index)
        .rolling(window, min_periods=min_periods)
        .corr(df["y"])
        .notna()
        .sum()
    )

    blk = block if block is not None else block_length(n)
    if n < 2 * blk or np.std(x) == 0 or np.std(y) == 0:
        return ScanMaxNull(
            observed_scanmax=observed,
            p_value=float("nan"),
            percentile=float("nan"),
            null_hi=float("nan"),
            block=blk,
            n_windows=n_windows,
            n_draws=0,
            beyond_band=False,
        )

    rng = np.random.default_rng(seed)
    edges = list(range(0, n, blk)) + [n]
    blocks = [x[edges[i] : edges[i + 1]] for i in range(len(edges) - 1)]
    null_max = np.empty(n_draws)
    for i in range(n_draws):
        order = rng.permutation(len(blocks))
        xp = np.concatenate([blocks[k] for k in order])
        null_max[i] = _scanmax(xp)

    null_max = null_max[np.isfinite(null_max)]
    if not len(null_max):
        p_value = float("nan")
        percentile = float("nan")
        null_hi = float("nan")
    else:
        p_value = float((np.sum(null_max >= observed) + 1) / (len(null_max) + 1))
        percentile = float(np.mean(null_max < observed))
        null_hi = float(np.quantile(null_max, 1 - alpha))
    beyond_band = bool(np.isfinite(observed) and np.isfinite(null_hi) and observed > null_hi)
    return ScanMaxNull(
        observed_scanmax=observed,
        p_value=p_value,
        percentile=percentile,
        null_hi=null_hi,
        block=blk,
        n_windows=n_windows,
        n_draws=int(len(null_max)),
        beyond_band=beyond_band,
    )


# --------------------------------------------------------------------------- fixed grid + BH-FDR (REQ-031)


def build_grid_cells(
    variants: tuple[str, ...],
    windows: tuple[int, ...],
    regimes: tuple[str, ...],
) -> list[dict]:
    """Enumerate the fixed **correlation** test grid (variants × windows × regimes) BEFORE outcomes.

    Each cell is a genuinely-distinct correlation test: the deseason ``variant`` changes the residual,
    the rolling ``window`` changes the peak-regime span, and the ``regime`` changes the subset. The
    event-magnitude *threshold* is **not** a correlation dimension (it does not change the correlation
    statistic), so it is excluded here to avoid fabricating multiplicity with duplicate p-values (F2);
    the threshold parametrises the separate event-study table instead (:func:`regime_event_study`,
    REQ-020). ``n_tests`` for BH-FDR is the number of **distinct** cells the driver actually computes.
    Enumerating up front — never expanding after viewing outcomes — is the anti-p-hacking discipline
    (REQ-031/032, NFR-005).
    """
    cells = []
    for variant, window, regime in itertools.product(variants, windows, regimes):
        cells.append({"variant": variant, "window": window, "regime": regime})
    return cells


# --------------------------------------------------------------------------- driver (REQ-050)


@dataclass(frozen=True)
class RegimeArtifacts:
    """Driver outputs: the fixed-grid table, event-study table, verdict and paths (REQ-050, AT10)."""

    table: pd.DataFrame
    event_study_table: pd.DataFrame
    n_tests: int
    verdict: str
    scanmax: ScanMaxNull
    jackknife: JackknifeResult
    collapse: CollapseDiagnosis
    regime: RegimeResult
    event_study: dict[str, RegimeEventStudy]
    rolling_secondary: pd.Series | None
    results: dict
    results_path: Path | None = None
    summary_path: Path | None = None
    event_study_path: Path | None = None


def _variant_residuals(cfg, resid: pd.Series, decomp: pd.DataFrame | None) -> dict[str, pd.Series]:
    """Genuinely-distinct deseason-variant residuals, deduped by content (F2 honesty).

    Each configured variant maps to a residual series; when a variant cannot be produced distinctly in
    the current path (e.g. offline with only the intensity residual available) it collapses onto an
    existing series and is **deduped**, so ``n_tests`` reflects the true number of distinct tests, never
    a padded count. In the file path a richer decomposition yields genuinely-different residuals per
    variant. Deseasonalisation is on the NO2 series **alone** — never against CREA (NFR-005).
    """
    per_variant: dict[str, pd.Series] = {}
    for v in cfg.variants:
        per_variant[v] = _deseason_variant(resid, v, decomp, cfg)
    distinct: dict[str, pd.Series] = {}
    kept: list[np.ndarray] = []
    for v, s in per_variant.items():
        arr = np.nan_to_num(s.to_numpy(dtype=float), nan=0.0)
        if any(a.shape == arr.shape and np.allclose(a, arr) for a in kept):
            continue
        distinct[v] = s
        kept.append(arr)
    return distinct


def _deseason_variant(
    resid: pd.Series, variant: str, decomp: pd.DataFrame | None, cfg
) -> pd.Series:
    """Return the residual for a deseason ``variant`` (read-only; no CREA involvement).

    ``intensity-model`` is the NOX-003.1 residual as passed. ``harmonic`` is derived by re-deseasonalising
    a raw footprint-signal column of the decomposition when present (reusing
    ``noxus.signal.index.deseasonalize`` unchanged); when that column is absent (offline/injected path)
    the variant collapses onto the base residual and is deduped by :func:`_variant_residuals`.
    """
    if variant == "harmonic" and decomp is not None:
        for col in ("signal", "footprint_signal", "steel_footprint_signal", "value"):
            if col in decomp.columns:
                from noxus.signal.index import deseasonalize

                base = pd.Series(
                    decomp[col].to_numpy(dtype=float), index=pd.DatetimeIndex(resid.index)
                )
                return deseasonalize(base, method="harmonic", cfg=cfg).reindex(resid.index)
    return resid


def _load_series_from_cfg(
    cfg,
) -> tuple[pd.Series, pd.Series, pd.Series | None, pd.DataFrame]:
    """Load the NOX-003.1 residual + CREA benchmark (+ the decomp frame) from cfg paths (ERR-001)."""
    decomp_path = Path(cfg.decomposition_path)
    if not decomp_path.exists():
        raise FileNotFoundError(
            f"Intensity decomposition not found: {decomp_path}. Run 'noxus index --deseason "
            "intensity-model' first (ERR-001)."
        )
    bench_path = Path(cfg.benchmark_path)
    if not bench_path.exists():
        raise FileNotFoundError(
            f"Benchmark not found: {bench_path}. Run 'noxus ingest-benchmark' first (ERR-001)."
        )
    decomp = pd.read_parquet(decomp_path)
    idx = pd.DatetimeIndex(pd.to_datetime(decomp["date"]))
    resid = pd.Series(decomp["residual_activity"].to_numpy(dtype=float), index=idx)
    # Secular intensity trend s(t) for the collapse diagnostic's intensity arm (B4). The NOX-003.1
    # writer (_write_intensity_decomposition, signal/index.py) emits it as the canonical column
    # ``trend_s_t``; ``s_trend`` is accepted as a legacy fallback. Only None when genuinely absent —
    # never silently None on a real decomposition (which would leave intensity_shift inert).
    trend_col = next((c for c in ("trend_s_t", "s_trend") if c in decomp.columns), None)
    secular = (
        pd.Series(decomp[trend_col].to_numpy(dtype=float), index=idx)
        if trend_col is not None
        else None
    )
    bench_df = pd.read_parquet(bench_path)
    bcol = "value" if "value" in bench_df.columns else "operating_rate"
    bidx = (
        pd.DatetimeIndex(pd.to_datetime(bench_df["date"]))
        if "date" in bench_df.columns
        else pd.DatetimeIndex(pd.to_datetime(bench_df.index))
    )
    bench = pd.Series(bench_df[bcol].to_numpy(dtype=float), index=bidx)
    return resid, bench, secular, decomp


def regime_study(
    cfg=None,
    *,
    resid: pd.Series | None = None,
    bench: pd.Series | None = None,
    calendar: pd.DataFrame | None = None,
    coverage: pd.Series | None = None,
    meteo: pd.DataFrame | None = None,
    secular_trend: pd.Series | None = None,
    events: pd.DataFrame | None = None,
    write: bool = True,
) -> RegimeArtifacts:
    """Driver: fixed grid -> regime table + BH-FDR + scan-max null + jackknife + verdict (REQ-050).

    Series may be **injected** (offline/test path) or loaded from ``cfg`` paths (ERR-001 if missing).
    The verdict is one of *event-bound* / *continuous* / *multiple-comparisons artefact* /
    *inconclusive*; the artefact conclusion is a first-class, valid outcome (AC-009). Re-running with
    the same seed/config reproduces the table identically (AT10).
    """
    from noxus.config.run import RegimeStudyConfig

    cfg = cfg or RegimeStudyConfig()
    decomp: pd.DataFrame | None = None
    if resid is None or bench is None:
        loaded_resid, loaded_bench, loaded_secular, decomp = _load_series_from_cfg(cfg)
        resid = resid if resid is not None else loaded_resid
        bench = bench if bench is not None else loaded_bench
        secular_trend = secular_trend if secular_trend is not None else loaded_secular
    if calendar is None:
        calendar = load_curtailment_calendar(cfg.calendar_path)

    # Genuinely-distinct deseason-variant residuals (deduped; F2) and rolling windows (primary +
    # the 52-week secondary, REQ-010/F3). Each is a real, distinct correlation test — never padded.
    variant_series = _variant_residuals(cfg, resid, decomp)
    headline = (
        cfg.headline_variant
        if cfg.headline_variant in variant_series
        else next(iter(variant_series))
    )
    primary_window = cfg.window_periods()
    windows = [primary_window]
    if cfg.freq.upper().startswith("W") and cfg.window_weeks_secondary not in windows:
        windows.append(cfg.window_weeks_secondary)

    coll_lo = pd.Timestamp(f"{cfg.collapse_year}-01-01")
    coll_hi = pd.Timestamp(f"{cfg.collapse_year}-12-31")

    # Rolling-regime identification per (variant, window); the headline+primary drives the report.
    rolling_by_vw: dict[tuple[str, int], RegimeResult] = {}
    for vlabel, vseries in variant_series.items():
        for w in windows:
            rolling_by_vw[(vlabel, w)] = rolling_regime(
                vseries,
                bench,
                window=w,
                n_boot=cfg.n_boot,
                n_perm=cfg.n_perm,
                seed=cfg.seed,
                boundary_shifts=cfg.boundary_robustness_periods,
                freq=cfg.freq,
            )
    regime = rolling_by_vw[(headline, primary_window)]
    peak_lo, peak_hi = regime.peak_window
    rolling_secondary = (
        rolling_by_vw[(headline, cfg.window_weeks_secondary)].rolling_r
        if (headline, cfg.window_weeks_secondary) in rolling_by_vw
        else None
    )

    # --- fixed correlation grid + BH-FDR (REQ-031/032): only genuinely-distinct cells (F2) ---
    rows = []
    seen: set[tuple] = set()
    for vlabel, vseries in variant_series.items():
        df_v = _join(vseries, bench, freq=cfg.freq)
        # peak regime: span depends on the window -> distinct per (variant, window).
        for w in windows:
            plo, phi = rolling_by_vw[(vlabel, w)].peak_window
            key = (vlabel, "peak", plo, phi)
            if key in seen:
                continue
            seen.add(key)
            rows.append(_corr_row(df_v, vlabel, w, "peak", plo, phi, cfg))
        # collapse regime: fixed 2024 calendar year -> window-independent, one cell per variant.
        key = (vlabel, "collapse", coll_lo, coll_hi)
        if key not in seen:
            seen.add(key)
            rows.append(_corr_row(df_v, vlabel, windows[0], "collapse", coll_lo, coll_hi, cfg))
    table = pd.DataFrame(rows)
    n_tests = int(len(table))  # true number of distinct tests (no duplicate p-values, F2)
    rejected, p_adj = bh_fdr(table["p_perm"].to_numpy(), alpha=cfg.fdr_alpha)
    table["p_fdr"] = p_adj
    table["fdr_significant"] = rejected
    table = table.sort_values(["regime", "variant", "window"]).reset_index(drop=True)

    # --- scan-max null on the headline variant + primary window (REQ-030) ---
    scanmax = scan_max_null(
        variant_series[headline],
        bench,
        window=primary_window,
        n_draws=cfg.scanmax_draws,
        seed=cfg.seed,
        alpha=cfg.fdr_alpha,
        freq=cfg.freq,
    )

    # --- regime-conditioned event study: precision/recall/lead per (threshold × regime) (REQ-020, F1) ---
    bench_df = pd.DataFrame({"date": pd.DatetimeIndex(bench.index), "value": bench.to_numpy()})
    regime_spans = {"peak": (peak_lo, peak_hi), "collapse": (coll_lo, coll_hi)}
    match_window = cfg.match_windows[0] if cfg.match_windows else 2
    event_study: dict[str, RegimeEventStudy] = {}
    es_rows = []
    for threshold in cfg.thresholds:
        study = regime_event_study(
            variant_series[headline],
            coverage,
            meteo,
            bench_df,
            calendar,
            regimes=regime_spans,
            z_thresh=threshold,
            bf_event_z=cfg.bf_event_z,
            match_window=match_window,
            min_periods=cfg.detect_min_periods,
            min_coverage=cfg.min_coverage,
            meteo_screen=cfg.meteo_screen,
            ventilation_z=cfg.ventilation_z,
            min_events_per_regime=cfg.min_events,
        )
        event_study[str(threshold)] = study
        for regime_label, mr in study.by_regime.items():
            es_rows.append(
                {
                    "threshold": threshold,
                    "regime": regime_label,
                    "n_no2": mr.n_no2,
                    "n_prod": mr.n_prod,
                    "n_matched": mr.n_matched,
                    "precision": mr.precision,
                    "recall": mr.recall,
                    "false_discovery_rate": mr.false_discovery_rate,
                    "median_lead_days": mr.median_lead_days,
                    "lead_ci_lo": mr.lead_ci[0],
                    "lead_ci_hi": mr.lead_ci[1],
                    "low_power": study.low_power[regime_label],
                    "calendar_used": study.calendar_used,
                }
            )
    event_study_table = (
        pd.DataFrame(es_rows).sort_values(["regime", "threshold"]).reset_index(drop=True)
    )

    # --- event-bound jackknife on the peak regime (REQ-021) ---
    hv = variant_series[headline]
    peak_resid = hv[(pd.to_datetime(hv.index) >= peak_lo) & (pd.to_datetime(hv.index) <= peak_hi)]
    peak_bench = bench[
        (pd.to_datetime(bench.index) >= peak_lo) & (pd.to_datetime(bench.index) <= peak_hi)
    ]
    if events is None:
        detected = detect_events(
            hv,
            coverage,
            meteo,
            z_thresh=cfg.z_thresh,
            min_periods=cfg.detect_min_periods,
            min_coverage=cfg.min_coverage,
            meteo_screen=cfg.meteo_screen and meteo is not None,
            ventilation_z=cfg.ventilation_z,
        )
    else:
        detected = events
    det = detected.copy()
    det["date"] = pd.to_datetime(det["date"])
    peak_events = det[(det["date"] >= peak_lo) & (det["date"] <= peak_hi)]
    jackknife = event_bound_jackknife(
        peak_resid,
        peak_bench,
        peak_events,
        window_periods=cfg.jackknife_event_window,
        min_events=cfg.min_events,
        drop_frac=cfg.event_bound_drop_frac,
        freq=cfg.freq,
    )

    # --- 2024 collapse dual diagnostic (REQ-022) ---
    collapse = collapse_attribution(
        variant_series[headline],
        bench,
        calendar,
        secular_trend,
        pre_period=cfg.pre_period,
        collapse_window=cfg.collapse_window,
    )

    # --- verdict (honest null is first-class, AC-009) ---
    peak_cell = table[table["regime"] == "peak"]
    peak_survives_fdr = bool(peak_cell["fdr_significant"].any()) if len(peak_cell) else False
    scan_unsurprising = (not scanmax.beyond_band) or (
        np.isfinite(scanmax.p_value) and scanmax.p_value > cfg.fdr_alpha
    )
    if scan_unsurprising or not peak_survives_fdr:
        verdict = VERDICT_ARTEFACT
    elif jackknife.verdict == VERDICT_EVENT_BOUND:
        verdict = VERDICT_EVENT_BOUND
    elif jackknife.verdict == VERDICT_CONTINUOUS:
        verdict = VERDICT_CONTINUOUS
    else:
        verdict = VERDICT_INCONCLUSIVE

    results = {
        "verdict": verdict,
        "n_tests": n_tests,
        "variants_distinct": list(variant_series.keys()),
        "windows_tested": windows,
        "secondary_window": cfg.window_weeks_secondary if rolling_secondary is not None else None,
        "peak_date": str(regime.peak_date.date()),
        "peak_r": regime.peak_r,
        "collapse_date": str(regime.collapse_date.date()),
        "collapse_r": regime.collapse_r,
        "scanmax_observed": scanmax.observed_scanmax,
        "scanmax_p": scanmax.p_value,
        "scanmax_percentile": scanmax.percentile,
        "scanmax_beyond_band": scanmax.beyond_band,
        "jackknife_verdict": jackknife.verdict,
        "jackknife_full_r": jackknife.full_r,
        "jackknife_r_without_top": jackknife.r_without_top,
        "collapse_verdict": collapse.verdict,
        "peak_survives_fdr": peak_survives_fdr,
        "descriptive_note": regime.descriptive_note,
    }

    results_path = summary_path = event_study_path = None
    if write:
        out_dir = Path(cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        results_path = out_dir / cfg.results_name
        table.to_csv(results_path, index=False)
        event_study_path = out_dir / cfg.event_study_name
        event_study_table.to_csv(event_study_path, index=False)
        summary_path = out_dir / cfg.summary_name
        summary_path.write_text(
            _summary_text(results, scanmax, jackknife, collapse, event_study_table),
            encoding="utf-8",
        )

    return RegimeArtifacts(
        table=table,
        event_study_table=event_study_table,
        n_tests=n_tests,
        verdict=verdict,
        scanmax=scanmax,
        jackknife=jackknife,
        collapse=collapse,
        regime=regime,
        event_study=event_study,
        rolling_secondary=rolling_secondary,
        results=results,
        results_path=results_path,
        summary_path=summary_path,
        event_study_path=event_study_path,
    )


def _corr_row(
    df_v: pd.DataFrame,
    variant: str,
    window: int,
    regime_label: str,
    lo: pd.Timestamp,
    hi: pd.Timestamp,
    cfg,
) -> dict:
    """One genuinely-distinct correlation-grid row (robust_corr over the variant×window×regime span)."""
    sub = df_v[(df_v.index >= lo) & (df_v.index <= hi)]
    rc = robust_corr(sub["x"], sub["y"], n_boot=cfg.n_boot, n_perm=cfg.n_perm, seed=cfg.seed)
    return {
        "variant": variant,
        "window": window,
        "regime": regime_label,
        "n": rc.n,
        "r": rc.r,
        "p_naive": rc.p_naive,
        "p_eff_nw": rc.p_eff_nw,
        "boot_lo": rc.boot_lo,
        "boot_hi": rc.boot_hi,
        "p_perm": rc.p_perm,
    }


def _summary_text(
    results: dict,
    scanmax: ScanMaxNull,
    jackknife: JackknifeResult,
    collapse: CollapseDiagnosis,
    event_study_table: pd.DataFrame,
) -> str:
    """Plain-language verdict incl. the scan-max-null artefact conclusion + n_tests (REQ-050)."""
    verdict_line = {
        VERDICT_EVENT_BOUND: "The 2023 peak is EVENT-BOUND: a few large curtailment events drive it.",
        VERDICT_CONTINUOUS: "The 2023 peak reflects a CONTINUOUS coupling (no single event dominates).",
        VERDICT_ARTEFACT: (
            "The 2023 peak is consistent with a MULTIPLE-COMPARISONS ARTEFACT of the rolling window "
            "(scan-maximum null): the r is unsurprising given how many overlapping windows were "
            "effectively scanned. This is a valid, first-class finding (Morris & Zhang 2019)."
        ),
        VERDICT_INCONCLUSIVE: "The 2023 peak is INCONCLUSIVE on this data.",
    }[results["verdict"]]
    lines = [
        "NOX-012 — 2023 regime event study (regime identification is descriptive, not predictive).",
        "",
        f"VERDICT: {verdict_line}",
        "",
        f"n_tests (fixed correlation grid, BH-FDR — genuinely distinct cells): {results['n_tests']}",
        f"  distinct deseason variants: {results['variants_distinct']}; "
        f"rolling windows tested (periods): {results['windows_tested']} "
        f"(18-month primary + 52-week secondary, REQ-010).",
        "  (Event-magnitude thresholds parametrise the event-study table below, not the correlation "
        "grid — a threshold does not change the correlation statistic, so it is not padded into the "
        "FDR family.)",
        f"peak r={results['peak_r']:.3f} at {results['peak_date']} (rolling-max 'collapse' locator "
        f"r={results['collapse_r']:.3f} at {results['collapse_date']}; note this rolling-min locator "
        "differs from the fixed 2024 calendar-year 'collapse' regime used in the grid/diagnostic).",
        f"scan-max null: observed={scanmax.observed_scanmax:.3f} p={scanmax.p_value} "
        f"percentile={scanmax.percentile} beyond_band={scanmax.beyond_band} "
        f"(block={scanmax.block}, windows={scanmax.n_windows}, draws={scanmax.n_draws}).",
        f"peak survives BH-FDR: {results['peak_survives_fdr']}.",
        f"event-bound jackknife: {jackknife.verdict} "
        f"(full r={jackknife.full_r:.3f} -> without top {jackknife.n_top_removed} events "
        f"{jackknife.r_without_top:.3f}). {jackknife.note}",
        f"2024 collapse: {collapse.verdict} "
        f"(calendar_shift={collapse.calendar_shift}, intensity_shift={collapse.intensity_shift}). "
        f"{collapse.note}",
        "",
        "Event-study table (precision/recall/lead per threshold × regime — the REQ-020 headline "
        "metric, never a Pearson r):",
    ]
    for _, r in event_study_table.iterrows():
        lines.append(
            f"  thr={r['threshold']} regime={r['regime']}: "
            f"precision={r['precision']:.3f} recall={r['recall']:.3f} "
            f"fdr={r['false_discovery_rate']:.3f} median_lead_days={r['median_lead_days']} "
            f"(n_no2={r['n_no2']}, n_prod={r['n_prod']}, matched={r['n_matched']}, "
            f"low_power={r['low_power']})."
        )
    return "\n".join(lines) + "\n"
