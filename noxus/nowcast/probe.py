"""T1 feasibility spike — the go/no-go gate for the activity nowcast (NOX-013, REQ-015, AT8).

The spike answers two questions against the **real on-disk artefacts**, then emits a verdict that
gates the rest of the build:

(a) FEATURE AVAILABILITY. The intended primary physical proxy — VIIRS thermal activity (NOX-011)
    — is UNAVAILABLE: VNF is paywalled and FIRMS active-fire is empirically invalid for this
    cluster (crop-residue-burning-dominated, r=-0.05 ns vs both NO2 and CREA; finding 2026-08-27,
    see ``decisions/architecture-decisions.md`` and NOX-011). The spike therefore confirms that no
    *valid* thermal series is present (a thermal parquet may exist but its triangulation is
    non-significant → still invalid), and confirms that the NO2 event markers (NOX-004/012) and the
    NO2 intensity residual (NOX-003.1) ARE available on disk.

(b) INCREMENTAL INFORMATION. Whether the available NO2 features show *any* incremental information
    over the CREA incumbent at the fixed defaults — a quick pass that REUSES
    ``noxus.validation.robust.robust_corr`` and ``noxus.validation.leadlag.lead_lag`` byte-for-byte
    on the NOX-003/008 weekly-resample-then-align convention (``align_series``), avoiding the
    NOX-012 B3 exact-date-join bug. No tuning, no new estimators, benchmark-blind (CREA is read only
    to MEASURE incremental info, never to fit anything).

Verdict (three-way aware; realistically GO-degraded vs NO-GO):

* ``GO-primary``  — a *valid* VIIRS/thermal series is supplied (dormant branch; never fabricated).
* ``GO-degraded`` — VIIRS unavailable but a NO2 feature carries some incremental info → build the
  reduced-power nowcast (reduced_power flag set).
* ``NO-GO``       — VIIRS unavailable and no feature carries incremental info → honest null; stop,
  escalate to the developer (ERR-002).

The verdict and its evidence are written to ``data/derived/nowcast_probe.txt``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from noxus.config.run import NowcastConfig
from noxus.validation.leadlag import lead_lag
from noxus.validation.preprocess import AlignmentError, align_series
from noxus.validation.robust import robust_corr

_SCALE_CAVEAT = (
    "Scale caveat: Tangshan is ~13% of national steel; this is a cluster signal, "
    "Hebei-provincial statistics preferred where available."
)


@dataclass(frozen=True)
class FeatureEvidence:
    """Incremental-information evidence for one candidate NO2 feature vs the CREA incumbent."""

    name: str
    available: bool
    n_common: int
    r: float
    p_eff_nw: float
    p_perm: float
    boot_lo: float
    boot_hi: float
    peak_lag: int
    peak_r: float
    sig_bound: float
    leads_crea: bool
    robust_significant: bool
    incremental_info: bool
    note: str = ""


@dataclass(frozen=True)
class ProbeVerdict:
    """The T1 go/no-go verdict plus the evidence that produced it."""

    verdict: str  # "GO-primary" | "GO-degraded" | "NO-GO"
    reduced_power: bool
    escalate: bool  # ERR-002: NO-GO stops the build and escalates to the developer
    thermal_available: bool
    thermal_valid: bool
    thermal_reason: str
    events_available: bool
    residual_available: bool
    any_incremental_info: bool
    evidence: tuple[FeatureEvidence, ...] = field(default_factory=tuple)
    report_path: Path | None = None


# --------------------------------------------------------------------------------------------------
# Feature-availability assessment
# --------------------------------------------------------------------------------------------------


def _assess_thermal(cfg: NowcastConfig) -> tuple[bool, bool, str]:
    """Return ``(available, valid, reason)`` for the VIIRS thermal proxy.

    Availability requires a thermal series file on disk; validity additionally requires the
    triangulation to have found a *significant* NO2/CREA link. The FIRMS-derived series present in
    this project is empirically invalid (non-significant, crop-burning-dominated), so the dormant
    GO-primary branch does not fire. This function NEVER fabricates a thermal series.
    """
    series_present = cfg.thermal_series_path.exists()
    if not series_present:
        return (
            False,
            False,
            "No thermal series on disk (VNF paywalled since 2025; FIRMS MAP_KEY path not run). "
            "VIIRS proxy unavailable — dormant GO-primary branch not eligible.",
        )

    # A series file exists (the FIRMS active-fire run). Check the triangulation verdict for validity.
    tri_path = cfg.thermal_triangulation_path
    if not tri_path.exists():
        return (
            True,
            False,
            "Thermal series present but no triangulation verdict on disk; cannot certify validity. "
            "Treated as INVALID (not selectable) pending a significant NO2/CREA link.",
        )

    tri = pd.read_csv(tri_path)
    significant = False
    if "significant" in tri.columns:
        # Robust to bool/str encodings written by the CSV round-trip.
        sig_col = tri["significant"].astype(str).str.strip().str.lower()
        significant = bool((sig_col == "true").any())

    if significant:
        return (
            True,
            True,
            "Thermal series present AND triangulation significant — a valid physical proxy is "
            "available; GO-primary branch is eligible.",
        )
    return (
        True,
        False,
        "Thermal series present but triangulation NON-significant (FIRMS active-fire empirically "
        "invalid for this cluster: crop-residue-burning-dominated, r=-0.05 ns vs NO2 and CREA; "
        "misses year-round furnaces). VIIRS proxy INVALID — not selectable (finding 2026-08-27).",
    )


def _events_available(cfg: NowcastConfig) -> tuple[bool, str]:
    """Whether the NO2 event/catalyst markers (NOX-004/012) are present, with the catalyst verdict."""
    if not cfg.events_results_path.exists():
        return False, "No catalyst results on disk."
    try:
        payload = json.loads(cfg.events_results_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - defensive
        return False, f"Catalyst results unreadable: {exc}."
    n_events = payload.get("n_no2_events")
    conclusion = payload.get("conclusion", "unknown")
    return True, f"catalyst markers present (n_no2_events={n_events}, conclusion={conclusion!r})."


# --------------------------------------------------------------------------------------------------
# Incremental-information quick pass (reused robust_corr / lead_lag, byte-for-byte)
# --------------------------------------------------------------------------------------------------


def _load_benchmark(cfg: NowcastConfig) -> pd.DataFrame:
    """Load the CREA incumbent parquet (read-only; measured against, never fit to)."""
    return pd.read_parquet(cfg.benchmark_path)


def _feature_frame(path: Path, column: str) -> pd.DataFrame:
    """Load a NO2 feature parquet and expose ``column`` under a name ``align_series`` recognises.

    Keeps ``date`` and ``valid_coverage`` (if present) so the reused weekly-resample-then-align
    coverage screen behaves exactly as in NOX-003/008. ``no2_corrected`` is already a recognised
    predictor name and is passed through unchanged.
    """
    df = pd.read_parquet(path)
    if column in ("index_value", "no2_attributed", "no2_corrected"):
        return df
    keep = ["date", column] + (["valid_coverage"] if "valid_coverage" in df.columns else [])
    return df[keep].rename(columns={column: "index_value"})


def _incremental_probe(
    name: str,
    feature: pd.DataFrame,
    benchmark: pd.DataFrame,
    cfg: NowcastConfig,
) -> FeatureEvidence:
    """Quick incremental-info pass for one continuous NO2 feature vs CREA.

    Reuses ``align_series`` (weekly resample + coverage screen + inner join — avoids the NOX-012 B3
    exact-date-join bug), then ``robust_corr`` and ``lead_lag`` byte-for-byte. Incremental info is
    declared when the feature either (i) is robustly correlated with CREA contemporaneously (both
    the Newey-West effective-N p-value AND the block-permutation p-value below ``sig_alpha``) or
    (ii) *leads* CREA — the lead_lag peak sits at a positive lag AND that lead correlation itself
    survives the same autocorrelation-robust inference (guards against a spurious peak picked out
    of the ±max_lag scan on autocorrelated noise). This is a coarse T1 gate, not the full REQ-011
    test (that is T6, with BH-FDR + orthogonality partialling).
    """
    try:
        joined = align_series(feature, benchmark, freq=cfg.freq, min_coverage=cfg.min_coverage)
    except AlignmentError as exc:
        return FeatureEvidence(
            name=name,
            available=True,
            n_common=0,
            r=float("nan"),
            p_eff_nw=1.0,
            p_perm=1.0,
            boot_lo=float("nan"),
            boot_hi=float("nan"),
            peak_lag=0,
            peak_r=float("nan"),
            sig_bound=float("inf"),
            leads_crea=False,
            robust_significant=False,
            incremental_info=False,
            note=f"alignment failed: {exc}",
        )

    n_common = int(len(joined))
    if n_common < cfg.min_overlap:
        return FeatureEvidence(
            name=name,
            available=True,
            n_common=n_common,
            r=float("nan"),
            p_eff_nw=1.0,
            p_perm=1.0,
            boot_lo=float("nan"),
            boot_hi=float("nan"),
            peak_lag=0,
            peak_r=float("nan"),
            sig_bound=float("inf"),
            leads_crea=False,
            robust_significant=False,
            incremental_info=False,
            note=f"too few common weeks ({n_common} < min_overlap {cfg.min_overlap})",
        )

    rc = robust_corr(
        joined["index"],
        joined["benchmark"],
        n_boot=cfg.n_boot,
        n_perm=cfg.n_perm,
        seed=cfg.seed,
    )
    ll = lead_lag(joined["index"], joined["benchmark"], max_lag=cfg.max_lag)

    robust_significant = (rc.p_eff_nw < cfg.sig_alpha) and (rc.p_perm < cfg.sig_alpha)

    # A positive peak lag alone is not enough: scanning ±max_lag lags on autocorrelated noise picks
    # spurious peaks. Require the peak *lead* correlation to survive the same robust inference.
    leads_crea = False
    if ll.peak_lag > 0:
        shifted = joined["index"].shift(ll.peak_lag)
        rc_lead = robust_corr(
            shifted,
            joined["benchmark"],
            n_boot=cfg.n_boot,
            n_perm=cfg.n_perm,
            seed=cfg.seed,
        )
        leads_crea = (rc_lead.p_eff_nw < cfg.sig_alpha) and (rc_lead.p_perm < cfg.sig_alpha)

    incremental_info = robust_significant or leads_crea

    return FeatureEvidence(
        name=name,
        available=True,
        n_common=n_common,
        r=rc.r,
        p_eff_nw=rc.p_eff_nw,
        p_perm=rc.p_perm,
        boot_lo=rc.boot_lo,
        boot_hi=rc.boot_hi,
        peak_lag=ll.peak_lag,
        peak_r=ll.peak_r,
        sig_bound=ll.sig_bound,
        leads_crea=leads_crea,
        robust_significant=robust_significant,
        incremental_info=incremental_info,
    )


# --------------------------------------------------------------------------------------------------
# Verdict + report
# --------------------------------------------------------------------------------------------------


def decide_verdict(
    thermal_available: bool,
    thermal_valid: bool,
    any_incremental_info: bool,
) -> tuple[str, bool, bool]:
    """Pure verdict logic → ``(verdict, reduced_power, escalate)``.

    GO-primary requires a *valid* thermal series (dormant). Otherwise GO-degraded if any NO2 feature
    carries incremental info, else NO-GO (honest null → escalate, ERR-002).
    """
    if thermal_available and thermal_valid:
        return "GO-primary", False, False
    if any_incremental_info:
        return "GO-degraded", True, False
    return "NO-GO", True, True


def _render_report(v: ProbeVerdict, cfg: NowcastConfig) -> str:
    lines: list[str] = []
    lines.append("NOX-013 T1 feasibility spike — activity-nowcast go/no-go gate")
    lines.append("=" * 78)
    lines.append(f"Generated: {datetime.now(UTC).isoformat(timespec='seconds')}")
    lines.append("")
    lines.append(f"VERDICT: {v.verdict}")
    lines.append(f"  reduced_power: {v.reduced_power}")
    lines.append(f"  escalate (ERR-002 honest-null stop): {v.escalate}")
    lines.append("")
    lines.append("Feature availability")
    lines.append("-" * 78)
    lines.append(
        f"  VIIRS thermal proxy: available={v.thermal_available} valid={v.thermal_valid}"
    )
    lines.append(f"    {v.thermal_reason}")
    lines.append(f"  NO2 event markers (NOX-004/012): available={v.events_available}")
    lines.append(f"  NO2 intensity residual (NOX-003.1): available={v.residual_available}")
    lines.append("")
    lines.append("Incremental information over CREA (quick pass; robust_corr + lead_lag, no tuning)")
    lines.append("-" * 78)
    lines.append(
        "  Convention: weekly-resample-then-align (align_series), benchmark-blind — CREA is read "
        "only to MEASURE incremental info, never to fit."
    )
    lines.append(
        f"  Fixed defaults: freq={cfg.freq} seed={cfg.seed} n_boot={cfg.n_boot} "
        f"n_perm={cfg.n_perm} max_lag={cfg.max_lag} min_overlap={cfg.min_overlap} "
        f"sig_alpha={cfg.sig_alpha}"
    )
    lines.append("")
    for e in v.evidence:
        lines.append(f"  [{e.name}]")
        if e.note:
            lines.append(f"    NOTE: {e.note}")
        lines.append(
            f"    n_common={e.n_common} r={e.r:+.4f} "
            f"p_eff_nw={e.p_eff_nw:.4f} p_perm={e.p_perm:.4f} "
            f"boot95=[{e.boot_lo:+.4f}, {e.boot_hi:+.4f}]"
        )
        lines.append(
            f"    lead_lag: peak_lag={e.peak_lag} peak_r={e.peak_r:+.4f} "
            f"sig_bound=±{e.sig_bound:.4f}  (peak_lag>0 ⇒ feature leads CREA)"
        )
        lines.append(
            f"    robust_significant={e.robust_significant} leads_crea={e.leads_crea} "
            f"→ incremental_info={e.incremental_info}"
        )
        lines.append("")
    lines.append(f"  any_incremental_info: {v.any_incremental_info}")
    lines.append("")
    lines.append("Interpretation")
    lines.append("-" * 78)
    if v.verdict == "NO-GO":
        lines.append(
            "  HONEST NULL. No available NO2 feature carries incremental information over the CREA "
            "incumbent at the fixed defaults, and no valid physical proxy exists. Per ERR-002 the "
            "build STOPS here and escalates to the developer. This is a valid, designed-for outcome "
            "(consistent with NOX-009 ~0 steel-local NO2 change-variance and NOX-012 weak/artefactual "
            "continuous coupling), not a pipeline failure."
        )
    elif v.verdict == "GO-degraded":
        lines.append(
            "  GO-degraded. A NO2 feature shows some incremental information over CREA. Proceed to "
            "build the REDUCED-POWER nowcast (no physical proxy; NO2-only feature layer). Every "
            "downstream output must carry the reduced-power caveat."
        )
    else:
        lines.append(
            "  GO-primary. A valid VIIRS/thermal physical proxy is available; build with the "
            "physical-proxy feature. (Dormant branch — only reachable if a significant thermal "
            "series is ever supplied.)"
        )
    lines.append("")
    lines.append(_SCALE_CAVEAT)
    lines.append("")
    return "\n".join(lines)


def probe(cfg: NowcastConfig | None = None) -> ProbeVerdict:
    """Run the T1 feasibility spike against the real on-disk data and write the verdict report.

    Returns a :class:`ProbeVerdict`. Writes ``data/derived/nowcast_probe.txt``. Reads CREA only to
    measure incremental info; never fits/tunes anything against it; never fabricates a VIIRS series.
    """
    cfg = cfg or NowcastConfig()

    thermal_available, thermal_valid, thermal_reason = _assess_thermal(cfg)
    events_available, _events_note = _events_available(cfg)
    residual_available = cfg.no2_residual_path.exists()

    # Incremental-information quick pass over the available continuous NO2 features.
    evidence: list[FeatureEvidence] = []
    if thermal_available and thermal_valid:
        # Dormant GO-primary: a valid physical proxy short-circuits the NO2 quick pass. We do not
        # fabricate or re-run anything here; verdict logic below handles it.
        pass
    else:
        benchmark = _load_benchmark(cfg)
        if residual_available:
            evidence.append(
                _incremental_probe(
                    "no2_residual (no2_corrected, NOX-003.1 footprint)",
                    _feature_frame(cfg.no2_residual_path, cfg.no2_residual_column),
                    benchmark,
                    cfg,
                )
            )
        if cfg.intensity_decomp_path.exists():
            evidence.append(
                _incremental_probe(
                    "no2_intensity_residual (residual_activity, NOX-003.1 decomposition covariate)",
                    _feature_frame(cfg.intensity_decomp_path, cfg.intensity_residual_column),
                    benchmark,
                    cfg,
                )
            )

    any_incremental_info = any(e.incremental_info for e in evidence)
    verdict, reduced_power, escalate = decide_verdict(
        thermal_available, thermal_valid, any_incremental_info
    )

    result = ProbeVerdict(
        verdict=verdict,
        reduced_power=reduced_power,
        escalate=escalate,
        thermal_available=thermal_available,
        thermal_valid=thermal_valid,
        thermal_reason=thermal_reason,
        events_available=events_available,
        residual_available=residual_available,
        any_incremental_info=any_incremental_info,
        evidence=tuple(evidence),
    )

    report_path = cfg.probe_report_path()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(result, cfg))

    # Re-emit with the resolved report path (frozen dataclass → rebuild).
    return ProbeVerdict(
        verdict=result.verdict,
        reduced_power=result.reduced_power,
        escalate=result.escalate,
        thermal_available=result.thermal_available,
        thermal_valid=result.thermal_valid,
        thermal_reason=result.thermal_reason,
        events_available=result.events_available,
        residual_available=result.residual_available,
        any_incremental_info=result.any_incremental_info,
        evidence=result.evidence,
        report_path=report_path,
    )
