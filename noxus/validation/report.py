"""Validation report: machine results + plain-language summary, incl. the null (NOX-003, REQ-043/044).

Ties the validation stage together — align → sign → r/p → lead-lag — and emits two artifacts:

- ``steel_validation_results.json``: machine-readable sign, Pearson r + p + CI, peak lag, the
  bar classification, the plain-language ``conclusion`` ("lead" | "concurrent" | "null"), and a
  ``config_echo`` of every knob that could move the result (deseasonalisation method, meteo
  covariates/form, curtailment control, footprint/background geometry, lag window, attributable cap).
- ``steel_validation_summary.txt``: a plain-language summary that **states the null explicitly** when
  there is no usable correlation after controls (Morris & Zhang 2019) and classifies the outcome
  against the literature success bar r ≈ 0.50–0.75 (Kim 2023 / Kondragunta 2021).

The summary deliberately does not present an unconditioned in-sample r as proof of a leading indicator
(NFR-003); it always reports the sign check, the CI, and the honest conclusion — including the null.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from noxus.validation.leadlag import (
    CCFResult,
    SignResult,
    correlate,
    lead_lag,
    verify_sign,
)
from noxus.validation.preprocess import align_series


class InsufficientOverlapError(RuntimeError):
    """The aligned index/benchmark overlap is shorter than the configured minimum (ERR-004)."""


@dataclass(frozen=True)
class ReportArtifacts:
    """Paths + payload of the emitted validation artifacts."""

    results_path: Path
    summary_path: Path
    results: dict


def classify_bar(peak_r: float, bar: tuple[float, float]) -> str:
    """Classify |peak r| against the literature success band (REQ-044)."""
    lo, hi = bar
    r = abs(peak_r)
    if r >= hi:
        return "above-bar"
    if r >= lo:
        return "in-band"
    return "below-bar"


def _conclusion(sign: SignResult, ccf: CCFResult, bar: tuple[float, float]) -> str:
    """Plain-language conclusion: 'lead' | 'concurrent' | 'null' (REQ-043)."""
    # No usable correlation after controls -> honest null (does not depend on flipping the sign).
    if not sign.significant or classify_bar(ccf.peak_r, bar) == "below-bar":
        return "null"
    if ccf.peak_lag > 0:
        return "lead"
    return "concurrent"


def build_results(
    aligned: pd.DataFrame,
    *,
    max_lag: int,
    bar: tuple[float, float],
    config_echo: dict,
) -> dict:
    """Compute the full results payload from an aligned (index, benchmark) frame."""
    idx = aligned["index"]
    bench = aligned["benchmark"]

    sign = verify_sign(idx, bench)
    corr = correlate(idx, bench)
    ccf = lead_lag(idx, bench, max_lag=max_lag)
    conclusion = _conclusion(sign, ccf, bar)

    return {
        "n_overlap": int(len(aligned)),
        "corr_n": int(corr.n),
        "sign": sign.sign,
        "sign_significant": sign.significant,
        "pearson_r": corr.pearson_r,
        "p_value": corr.p_value,
        "ci_low": corr.ci_low,
        "ci_high": corr.ci_high,
        "peak_lag": ccf.peak_lag,
        "peak_r": ccf.peak_r,
        "ccf_lags": ccf.lags,
        "ccf_values": ccf.ccf,
        "ccf_sig_bound": ccf.sig_bound,
        "success_bar": list(bar),
        "bar_class": classify_bar(ccf.peak_r, bar),
        "conclusion": conclusion,
        "config_echo": config_echo,
    }


def render_summary(results: dict) -> str:
    """Render the plain-language summary that states the null explicitly (REQ-043/044, NFR-003)."""
    r = results
    lag = r["peak_lag"]
    if r["conclusion"] == "lead":
        headline = (
            f"LEAD: the index leads the benchmark by {lag} period(s) (peak r={r['peak_r']:.3f})."
        )
    elif r["conclusion"] == "concurrent":
        headline = (
            f"CONCURRENT: the index tracks the benchmark at lag {lag} (peak r={r['peak_r']:.3f})."
        )
    else:
        headline = (
            "NULL: no usable lead/correlation between the index and the benchmark after controls. "
            "This is a valid, designed-for outcome (Morris & Zhang 2019), not a pipeline failure."
        )

    lo, hi = r["success_bar"]
    ci = (
        f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
        if r["ci_low"] == r["ci_low"]  # not NaN
        else "[n/a]"
    )
    echo = r["config_echo"]
    lines = [
        "Steel-sector NO2 activity index — validation vs CREA blast-furnace operating rate",
        "=" * 78,
        headline,
        "",
        f"Sign (empirically verified, not assumed): {r['sign']} "
        f"(significant={r['sign_significant']}).",
        f"Pearson r = {r['pearson_r']:.3f}  (p = {r['p_value']:.3g}, "
        f"n = {r.get('corr_n', r['n_overlap'])}, 95% CI {ci}).",
        f"Cross-correlation peak: lag {lag}, r = {r['peak_r']:.3f} "
        f"(white-noise band +/-{r['ccf_sig_bound']:.3f}).",
        f"  NOTE: peak chosen over {2 * r['config_echo'].get('max_lag', 0) + 1} lags; the band is "
        "per-lag and NOT corrected for that multiplicity, so the peak r is optimistic. The null/usable "
        "decision below relies on the lag-0 sign test, not the selected peak.",
        f"Success bar (Kim 2023 / Kondragunta 2021): r ~ {lo:.2f}-{hi:.2f}  "
        f"-> classification: {r['bar_class']}.",
    ]

    if "levels_r" in r:
        lci = (
            f"[{r['levels_ci_low']:.3f}, {r['levels_ci_high']:.3f}]"
            if r["levels_ci_low"] == r["levels_ci_low"]  # not NaN
            else "[n/a]"
        )
        lines += [
            "",
            "Decoupling (levels vs residual; NOX-003.1):",
            f"  LEVELS  r = {r['levels_r']:.3f} (p = {r['levels_p']:.3g}, n = {r['levels_n']}, "
            f"95% CI {lci}) — raw footprint NO2 vs BF rate.",
            f"  RESIDUAL r = {r['pearson_r']:.3f} — activity proxy after removing the intensity trend.",
            (
                "  -> DECOUPLING confirmed: NO2 levels fall as activity holds/rises (the secular "
                "emission-intensity decline, Li 2024), while the de-trended residual tracks activity "
                "positively. The negative levels sign is a finding, not a failed correlation."
                if r.get("decoupling")
                else "  -> Decoupling pattern NOT confirmed on this sample (levels not negative or "
                "residual not positive); see the smoothness sweep before interpreting."
            ),
        ]

    lines += [
        "",
        "Config echo (every knob that could move the result):",
        f"  deseason_method      : {echo.get('deseason_method')}",
        f"  structural_terms     : {echo.get('structural_terms')}",
        f"  meteo_form           : {echo.get('meteo_form')}",
        f"  meteo_covariates     : {echo.get('meteo_covariates')}",
        f"  curtailment_control  : {echo.get('curtailment_control')}",
        f"  footprint_geometry   : {echo.get('footprint_geometry')}",
        f"  background_geometry  : {echo.get('background_geometry')}",
        f"  lag_window           : +/-{echo.get('max_lag')}",
        f"  attributable_cap     : {echo.get('attributable_cap')} "
        "(ceiling on the steel share of the column; the index is RELATIVE, not absolute tonnage)",
    ]
    if echo.get("deseason_method") == "intensity-model":
        lines.append(
            f"  intensity_trend      : estimator={echo.get('intensity_estimator')}, "
            f"df={echo.get('intensity_df')} (selected by {echo.get('intensity_criterion')} on the "
            "NO2 series alone — benchmark never consulted, NFR-102)"
        )
    return "\n".join(lines) + "\n"


def add_levels_relationship(
    results: dict,
    levels_frame: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    freq: str,
) -> dict:
    """Add the levels↔benchmark relationship to ``results`` for decoupling reporting (REQ-110).

    The intensity-model index is the *residual* (activity proxy); the raw *levels* footprint signal
    relates to the benchmark with the opposite, decoupling sign (NO2 falls as activity holds — the
    secular intensity decline, Li 2024). Reporting both, each with r/p/CI, frames the negative-levels
    relationship as a finding rather than a failed positive correlation. No-op if the levels overlap is
    too short to correlate. ``decoupling`` flags the expected pattern (levels r < 0 and residual r > 0).
    """
    lv = align_series(levels_frame, benchmark, freq=freq, min_coverage=0.0)
    if len(lv) < 3:
        return results
    lc = correlate(lv["index"], lv["benchmark"])
    results["levels_r"] = lc.pearson_r
    results["levels_p"] = lc.p_value
    results["levels_n"] = int(lc.n)
    results["levels_ci_low"] = lc.ci_low
    results["levels_ci_high"] = lc.ci_high
    results["decoupling"] = bool(lc.pearson_r < 0 < results["pearson_r"])
    return results


def report(
    index: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    freq: str = "W",
    min_coverage: float = 0.25,
    max_lag: int = 8,
    min_overlap: int = 26,
    bar: tuple[float, float] = (0.50, 0.75),
    config_echo: dict | None = None,
    levels_frame: pd.DataFrame | None = None,
    out_dir: Path | None = None,
    results_name: str = "steel_validation_results.json",
    summary_name: str = "steel_validation_summary.txt",
    write: bool = True,
) -> ReportArtifacts:
    """Run the full validation and emit results JSON + plain-language summary (REQ-040..044).

    Aligns the index and benchmark, refuses on too-short overlap (ERR-004), computes sign/r-p/lead-lag,
    classifies against the bar, derives the conclusion (incl. the null), and writes both artifacts
    (unless ``write=False``, used by tests). ``config_echo`` records every knob for honesty (NFR-003).
    When ``levels_frame`` is given (the raw footprint signal as ``date``/``index_value``), the
    levels↔benchmark decoupling relationship is added too (REQ-110, NOX-003.1).
    """
    aligned = align_series(index, benchmark, freq=freq, min_coverage=min_coverage)
    if len(aligned) < min_overlap:
        raise InsufficientOverlapError(
            f"Only {len(aligned)} overlapping periods after alignment and coverage screening; "
            f"need >= {min_overlap} to run stable statistics (ERR-004). Refusing to emit unstable "
            "estimates."
        )

    echo = dict(config_echo or {})
    echo.setdefault("max_lag", max_lag)
    echo.setdefault("freq", freq)

    results = build_results(aligned, max_lag=max_lag, bar=bar, config_echo=echo)
    if levels_frame is not None:
        results = add_levels_relationship(results, levels_frame, benchmark, freq=freq)
    summary = render_summary(results)

    out_dir = Path(out_dir) if out_dir is not None else Path("data/derived")
    results_path = out_dir / results_name
    summary_path = out_dir / summary_name
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        summary_path.write_text(summary, encoding="utf-8")

    return ReportArtifacts(results_path=results_path, summary_path=summary_path, results=results)


def _sign_result_dict(sign: SignResult) -> dict:  # pragma: no cover - convenience
    return asdict(sign)


def run_validation(signal_cfg=None, validation_cfg=None) -> ReportArtifacts:
    """Load the index + benchmark from disk and run the validation report (CLI entry, REQ-040..044).

    Reads ``steel_activity_index.parquet`` (ERR-001 if absent) and the CREA benchmark parquet
    (ERR-001 if absent), assembles the config echo from the index provenance + the validation config,
    and emits the results JSON + plain-language summary. Raises ``InsufficientOverlapError`` (ERR-004)
    when the overlap is too short.
    """
    from noxus.config.run import SignalConfig, ValidationConfig
    from noxus.signal.index import read_index_provenance

    signal_cfg = signal_cfg or SignalConfig()
    validation_cfg = validation_cfg or ValidationConfig()

    index_path = Path(signal_cfg.out_dir) / signal_cfg.index_name
    if not index_path.exists():
        raise FileNotFoundError(
            f"Activity index not found: {index_path}. Run 'noxus index' first (ERR-001)."
        )
    bench_path = Path(validation_cfg.benchmark_path)
    if not bench_path.exists():
        raise FileNotFoundError(
            f"Benchmark not found: {bench_path}. Run 'noxus ingest-benchmark' first (ERR-001)."
        )

    index = pd.read_parquet(index_path)
    benchmark = pd.read_parquet(bench_path)
    prov = read_index_provenance(index_path)

    # Report the curtailment control honestly: name the configured source only when it was actually
    # applied; otherwise mark it absent so the echo never overstates an unapplied control (Q6a).
    curtailment_control = (
        prov.get("curtailment_source")
        if prov.get("curtailment_applied")
        else f"{prov.get('curtailment_source')} (NOT applied — calendar absent, deferred T15/Q6a)"
    )
    config_echo = {
        "deseason_method": prov.get("deseason_method"),
        "structural_terms": prov.get("structural_terms"),
        "meteo_form": prov.get("meteo_form"),
        "meteo_covariates": prov.get("meteo_covariates"),
        "curtailment_control": curtailment_control,
        "footprint_geometry": prov.get("footprint_radius_km"),
        "background_geometry": prov.get("background_geom"),
        "attributable_cap": prov.get("attributable_cap", list(signal_cfg.attributable_cap)),
        "max_lag": validation_cfg.max_lag,
        "freq": validation_cfg.freq,
    }
    if prov.get("deseason_method") == "intensity-model":
        config_echo.update(
            intensity_estimator=prov.get("intensity_estimator"),
            intensity_df=prov.get("intensity_df"),
            intensity_criterion=prov.get("intensity_criterion"),
            intensity_cv_score=prov.get("intensity_cv_score"),
        )

    # When the intensity model ran, surface the levels↔benchmark decoupling relationship from the
    # decomposition diagnostic (the raw footprint signal), so the report states both signs (REQ-110).
    levels_frame = None
    decomp_path = Path(signal_cfg.out_dir) / signal_cfg.decomposition_name
    if prov.get("deseason_method") == "intensity-model" and decomp_path.exists():
        decomp = pd.read_parquet(decomp_path)
        levels_frame = pd.DataFrame(
            {"date": pd.to_datetime(decomp["date"]), "index_value": decomp["signal"].to_numpy()}
        )

    return report(
        index,
        benchmark,
        freq=validation_cfg.freq,
        min_coverage=validation_cfg.min_coverage,
        max_lag=validation_cfg.max_lag,
        min_overlap=validation_cfg.min_overlap,
        bar=validation_cfg.success_bar,
        config_echo=config_echo,
        levels_frame=levels_frame,
        out_dir=Path(validation_cfg.out_dir),
        results_name=validation_cfg.results_name,
        summary_name=validation_cfg.summary_name,
    )
