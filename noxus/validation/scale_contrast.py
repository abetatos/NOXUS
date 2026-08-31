"""Provincial/national benchmark arm: the scale contrast (NOX-016, paper critique C2).

The project's near-null has two live explanations and has never run the test that separates them:

  (A) the binding constraint is SPATIAL -- the Tangshan cluster term is not separable from the
      regional common mode, which is what the paper argues; or
  (B) the pipeline is broken, or the CREA trader survey is a poor measure of physical output.

Both produce the same near-null against a Tangshan-specific target, so scoring against that target
alone cannot tell them apart. Scoring the SAME benchmark-blind features against WIDER-SCALE steel
activity can: if a Tangshan-AOI NO2 feature tracks national steel activity better than it tracks
Tangshan's own blast-furnace rate, the AOI signal is carrying something regionally coherent -- which
is (A) -- and the feature is measuring something real about steel, which weakens (B).

Read the result carefully. A positive contrast does NOT mean TROPOMI over Tangshan measures Chinese
steel output: Tangshan is ~13% of national output and the AOI spans under 2 degrees. It means the
AOI column is dominated by a component that moves with wide-area industrial activity.

**This module reads targets on purpose**, so it cannot inherit the NOX-009 structural guarantee of
having no benchmark parameter. Three testable guarantees replace it:
  1. features are loaded read-only and never refit against any target (NFR-001);
  2. every computed pairing enters the BH family, including refused ones (NFR-002); and
  3. the verdict thresholds live in ``ScaleContrastConfig``, fixed before the run.

Arm B (NBS Hebei/national, an INDEPENDENT publisher) is gated on :func:`noxus.data.nbs.probe_nbs`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from noxus.validation.leadlag import lead_lag
from noxus.validation.paired_corr import paired_corr_difference
from noxus.validation.robust import bh_fdr, robust_corr

VERDICT_SCALE_CONFIRMED = "scale-confirmed"
VERDICT_INVERTED = "inverted"
VERDICT_NO_SCALE_EFFECT = "no-scale-effect"
VERDICT_PIPELINE_SUSPECT = "pipeline-suspect"


# --------------------------------------------------------------------------- read-only loading (REQ-003)


class FeatureRefitAttempted(RuntimeError):
    """Raised if anything tries to fit a feature against a target (ERR-004)."""


def load_feature(cfg, label: str) -> pd.Series:
    """Load one committed NO2 feature, read-only. There is deliberately no fitting entry point."""
    spec = next((f for f in cfg.features if f[0] == label), None)
    if spec is None:
        raise KeyError(f"Unknown feature {label!r}; known: {[f[0] for f in cfg.features]}")
    _, rel, column = spec
    path = Path(cfg.derived_dir) / rel
    if not path.exists():
        raise FileNotFoundError(
            f"NO2 feature not found: {path}. Run 'noxus attribute' / 'noxus index' first (ERR-001)."
        )
    df = pd.read_parquet(path)
    s = pd.Series(df[column].to_numpy(float), index=pd.DatetimeIndex(pd.to_datetime(df["date"])))
    if "valid_coverage" in df.columns:
        cov = pd.Series(df["valid_coverage"].to_numpy(float), index=s.index)
        s = s.mask(cov.notna() & (cov < cfg.min_coverage))
    return s.rename(label)


def load_target(cfg, label: str) -> pd.Series:
    """Load one scoring target, read-only (REQ-001)."""
    spec = next((t for t in cfg.targets if t[0] == label), None)
    if spec is None:
        raise KeyError(f"Unknown target {label!r}; known: {[t[0] for t in cfg.targets]}")
    _, _scale, rel, series = spec
    path = Path(cfg.derived_dir) / rel
    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark not found: {path}. Run 'noxus ingest-benchmark' first (ERR-001)."
        )
    df = pd.read_parquet(path)
    if series:
        df = df[df["series"] == series]
    s = pd.Series(df["value"].to_numpy(float), index=pd.DatetimeIndex(pd.to_datetime(df["date"])))
    return s.dropna().rename(label)


def _cadence_for(cfg, target_label: str) -> str:
    """Coarsest cadence both sides support. A 10-day target aggregates DOWN, never up (REQ-002)."""
    return cfg.freq_coarse if target_label in cfg.coarse_targets else cfg.freq_weekly


def _resample(s: pd.Series, freq: str) -> pd.Series:
    return s.resample(freq).mean()


def _window(s: pd.Series, lo: str, hi: str) -> pd.Series:
    return s[(s.index >= pd.Timestamp(lo)) & (s.index <= pd.Timestamp(hi))]


# --------------------------------------------------------------------------- scoring (REQ-010..013)


def _score_pairing(cfg, feature: pd.Series, target: pd.Series, freq: str) -> dict:
    f, t = _resample(feature, freq), _resample(target, freq)
    joined = pd.concat({"f": f, "t": t}, axis=1).dropna()
    n = int(len(joined))
    if n < cfg.min_overlap:
        # ERR-002: refuse the estimate, but the pairing still counts in the BH family (NFR-002).
        return {
            "n": n,
            "refused": True,
            "refused_reason": f"overlap {n} < min_overlap {cfg.min_overlap}",
            "r": float("nan"),
            "p_perm": float("nan"),
        }
    rc = robust_corr(joined["f"], joined["t"], n_boot=cfg.n_boot, n_perm=cfg.n_perm, seed=cfg.seed)
    ccf = lead_lag(joined["f"], joined["t"], max_lag=cfg.max_lag)
    return {
        "n": rc.n,
        "refused": False,
        "refused_reason": "",
        "r": rc.r,
        "p_naive": rc.p_naive,
        "n_eff_nw": rc.n_eff_nw,
        "p_eff_nw": rc.p_eff_nw,
        "boot_lo": rc.boot_lo,
        "boot_hi": rc.boot_hi,
        "p_perm": rc.p_perm,
        "block": rc.block,
        "peak_lag": ccf.peak_lag,
        "peak_r": ccf.peak_r,
        "sig_bound": ccf.sig_bound,
        # REQ-013: a peak at the edge of the scanned window is a boundary selection, not an
        # estimated lag, and a peak inside its own white-noise band is not a lag at all.
        "peak_is_boundary": bool(abs(int(ccf.peak_lag)) == int(cfg.max_lag)),
        "peak_significant": bool(abs(float(ccf.peak_r)) > float(ccf.sig_bound)),
    }


def scale_contrast(cfg=None) -> pd.DataFrame:
    """Score every feature x target x sub-period pairing and contrast each against the baseline.

    Returns one tidy row per pairing. Every computed pairing -- including refused ones -- carries the
    same ``family_size`` and enters the BH correction (NFR-002, REQ-011).
    """
    from noxus.config.run import ScaleContrastConfig

    cfg = cfg or ScaleContrastConfig()
    scales = {t[0]: t[1] for t in cfg.targets}

    features = {f[0]: load_feature(cfg, f[0]) for f in cfg.features}
    targets = {t[0]: load_target(cfg, t[0]) for t in cfg.targets}

    rows: list[dict] = []
    for feat_label, feat in features.items():
        for sub_label, lo, hi in cfg.subperiods:
            f_win = _window(feat, lo, hi)
            for tgt_label, tgt in targets.items():
                freq = _cadence_for(cfg, tgt_label)
                t_win = _window(tgt, lo, hi)
                scored = _score_pairing(cfg, f_win, t_win, freq)
                row = {
                    "feature": feat_label,
                    "target": tgt_label,
                    "target_scale": scales[tgt_label],
                    "cadence": freq,
                    "subperiod": sub_label,
                    **scored,
                }
                # REQ-012: the paired contrast against the Tangshan baseline, measured on the
                # COMMON sample -- never by eyeballing two marginal CIs for overlap.
                if tgt_label != cfg.baseline_target:
                    base = _window(targets[cfg.baseline_target], lo, hi)
                    pc = paired_corr_difference(
                        _resample(f_win, freq),
                        _resample(base, freq),
                        _resample(t_win, freq),
                        n_boot=cfg.n_boot,
                        seed=cfg.seed,
                    )
                    row.update(
                        contrast_vs_baseline=pc.difference,
                        contrast_lo=pc.lo,
                        contrast_hi=pc.hi,
                        contrast_n=pc.n,
                        contrast_excludes_zero=pc.excludes_zero,
                    )
                else:
                    row.update(
                        contrast_vs_baseline=float("nan"),
                        contrast_lo=float("nan"),
                        contrast_hi=float("nan"),
                        contrast_n=scored["n"],
                        contrast_excludes_zero=False,
                    )
                rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    rejected, p_adj = bh_fdr(df["p_perm"].to_numpy(float), alpha=cfg.fdr_alpha)
    df["p_fdr"] = p_adj
    df["fdr_significant"] = rejected
    df["family_size"] = int(len(df))  # every pairing, refused included (NFR-002)
    verdict, reason = classify_scale_verdict(df, cfg)
    df["verdict"] = verdict
    df["verdict_reason"] = reason
    return df.sort_values(["feature", "subperiod", "target"]).reset_index(drop=True)


# --------------------------------------------------------------------------- verdict (REQ-020/021)


def classify_scale_verdict(df: pd.DataFrame, cfg=None) -> tuple[str, str]:
    """Apply the **pre-registered** REQ-021 rule. A pure function of ``df`` and ``cfg``.

    Decided on the full record, on ``cfg.primary_national`` (same variable/publisher/cadence as the
    baseline, so only spatial scale differs), over either feature.
    """
    from noxus.config.run import ScaleContrastConfig

    cfg = cfg or ScaleContrastConfig()
    full = df[(df["subperiod"] == "full") & (~df["refused"])]
    prim = full[full["target"] == cfg.primary_national]
    if prim.empty:
        return VERDICT_NO_SCALE_EFFECT, "the primary national pairing did not complete"

    confirmed = prim[
        prim["fdr_significant"]
        & (prim["contrast_vs_baseline"] > 0)
        & prim["contrast_excludes_zero"]
    ]
    if len(confirmed):
        best = confirmed.loc[confirmed["contrast_vs_baseline"].idxmax()]
        return (
            VERDICT_SCALE_CONFIRMED,
            f"{best['feature']} tracks {cfg.primary_national} at r={best['r']:+.3f} "
            f"(BH p={best['p_fdr']:.3f}), a paired gain of {best['contrast_vs_baseline']:+.3f} over "
            f"the Tangshan target with CI [{best['contrast_lo']:+.3f}, {best['contrast_hi']:+.3f}]",
        )

    inverted = prim[prim["contrast_excludes_zero"] & (prim["contrast_vs_baseline"] < 0)]
    base = full[full["target"] == cfg.baseline_target]
    if len(inverted) and bool(base["fdr_significant"].any()):
        return (
            VERDICT_INVERTED,
            "the Tangshan target is tracked significantly BETTER than the national target",
        )

    max_abs_r = float(prim["r"].abs().max())
    if max_abs_r < cfg.pipeline_suspect_abs_r:
        return (
            VERDICT_PIPELINE_SUSPECT,
            f"no national pairing exceeds |r|={cfg.pipeline_suspect_abs_r} (best "
            f"|r|={max_abs_r:.3f}), far below the 0.5-0.75 range reported at national scale, so "
            "these data do not support the spatial-scale explanation",
        )
    return (
        VERDICT_NO_SCALE_EFFECT,
        f"the national pairing reaches |r|={max_abs_r:.3f} but is not robust-significant and the "
        "paired contrast CI contains zero",
    )


# --------------------------------------------------------------------------- reporting (REQ-040/022)


def render_findings(df: pd.DataFrame, cfg=None, probe=None) -> str:
    """Findings note: the verdict, the numbers, and -- mandatory -- what it does NOT establish."""
    from noxus.config.run import ScaleContrastConfig

    cfg = cfg or ScaleContrastConfig()
    verdict, reason = classify_scale_verdict(df, cfg)
    full = df[df["subperiod"] == "full"]
    lines = [
        "NOX-016 - provincial/national benchmark arm (paper critique C2)",
        "=" * 78,
        f"VERDICT: {verdict}",
        f"  reason: {reason}",
        f"  BH family size: {int(df['family_size'].iloc[0])} pairings "
        f"(every computed pairing, refused included); alpha={cfg.fdr_alpha}",
        f"  primary national target (decides the verdict): {cfg.primary_national}",
        "",
        "Arm A - full record, by target scale",
        "-" * 78,
    ]
    cols = [
        "feature",
        "target",
        "target_scale",
        "cadence",
        "n",
        "r",
        "p_perm",
        "p_fdr",
        "fdr_significant",
        "contrast_vs_baseline",
        "contrast_lo",
        "contrast_hi",
    ]
    lines += ["  " + ln for ln in full[cols].round(4).to_string(index=False).splitlines()]

    lines += [
        "",
        "Sub-period split (EDGE-003: a shared secular trend can manufacture a contrast)",
        "-" * 78,
    ]
    sub = df[(df["subperiod"] != "full") & (~df["refused"])]
    if not sub.empty:
        piv = sub.pivot_table(index=["feature", "target"], columns="subperiod", values="r")
        lines += ["  " + ln for ln in piv.round(4).to_string().splitlines()]

    lines += ["", "Lead/lag reporting discipline (REQ-013)", "-" * 78]
    for _, r in full.iterrows():
        if r["refused"]:
            continue
        marks = []
        if r["peak_is_boundary"]:
            marks.append("AT SCAN BOUNDARY -> not an estimated lag")
        if not r["peak_significant"]:
            marks.append("inside its own white-noise band -> not significant")
        lines.append(
            f"  {r['feature']:>14s} x {r['target']:<22s} peak_lag={int(r['peak_lag']):+d} "
            f"peak_r={r['peak_r']:+.4f} bound=±{r['sig_bound']:.4f}"
            + ("  [" + "; ".join(marks) + "]" if marks else "")
        )

    lines += ["", "Arm B - independent publisher (NBS)", "-" * 78]
    if probe is None:
        lines.append("  not probed in this run.")
    elif probe.go:
        lines.append("  GO - an NBS route responded; Arm B pairings are folded into the family.")
    else:
        lines += [
            "  NO-GO. No free programmatic NBS route from this environment, so no INDEPENDENT",
            "  publisher was added and, per REQ-031, no paywalled substitute was used.",
        ]

    lines += [
        "",
        "What this does and does NOT establish (REQ-022, EDGE-005/006)",
        "-" * 78,
        "  Limits that apply whatever the verdict:",
    ]
    lines += [
        "  * SHARED PUBLISHER. Every Arm-A target is CREA/WIND, as is the Tangshan benchmark. This",
        "    arm therefore tests spatial SCALE with everything else held fixed, but it does NOT",
        "    break the feature-versus-target ambiguity: a survey-methodology artefact would move",
        "    all four targets together. Arm B existed to fix that and returned NO-GO.",
        "  * PROCESSOR VERSIONS. The 2019-2026 TROPOMI record is not version-homogenised, so any",
        "    secular component of a national correlation is confounded with retrieval versioning.",
        "    Read the sub-period split above, not the full record alone.",
        "  * SCALE OF THE FEATURE. The NO2 feature is a Tangshan-AOI window (~13% of national",
        "    steel, under 2 degrees across), NOT national-coverage NO2. A high national correlation",
        "    would mean the AOI column is regionally dominated -- not that TROPOMI over Tangshan",
        "    measures Chinese steel output. A LOW one likewise does not prove the pipeline is",
        "    broken: the decisive version of that test needs a province- or nation-scale cube,",
        "    which is out of scope here.",
    ]

    lines += ["", "Interpretation", "-" * 78]
    if verdict == VERDICT_SCALE_CONFIRMED:
        lines += [
            "  The same feature tracks national steel activity BETTER than it tracks Tangshan's own",
            "  blast-furnace rate. That is positive evidence for the paper's spatial reading: the",
            "  AOI column is dominated by a regionally coherent component, and the cluster term is",
            "  what cannot be separated from it.",
        ]
    elif verdict == VERDICT_INVERTED:
        lines += [
            "  The Tangshan target is tracked BETTER than the national one, which contradicts the",
            "  regional-common-mode reading. The paper's synthesis requires rewriting.",
        ]
    elif verdict == VERDICT_PIPELINE_SUSPECT:
        lines += [
            "  The feature tracks national steel activity NO BETTER than it tracks Tangshan's own",
            "  rate -- both are near-null. This does NOT confirm the paper's spatial-scale",
            "  explanation; the discriminating test came back on the side that does not support it.",
            "",
            "  Be precise about what follows. It is not established that the pipeline is broken:",
            "  the feature is a cluster-scale window, so it was never guaranteed to proxy national",
            "  output, and the national targets are survey aggregates from the same publisher as",
            "  the Tangshan one. What IS established is that the paper can no longer present the",
            "  spatial explanation as the one its own evidence selects -- the cheap test that would",
            "  have supported it was run, and it did not.",
            "",
            "  Consequence for the paper: Section 7 must state that the wider-scale control was run",
            "  and returned no wider-scale signal either, and the 'binding constraint is spatial'",
            "  claim must be demoted from a conclusion to a hypothesis that this arm failed to",
            "  corroborate. The honest remaining route is a province/nation-scale cube (spec Q3).",
        ]
    else:
        lines += [
            "  Neither scale is tracked robustly and the paired contrast CI contains zero, so this",
            "  arm does not discriminate between the two explanations. Report as inconclusive.",
        ]
    return "\n".join(lines) + "\n"


def run_scale_contrast(cfg=None, probe_armb: bool = True) -> Path:
    """Run the contrast, write CSV + findings (+ the Arm-B probe note), return the CSV path."""
    from noxus.config.run import ScaleContrastConfig
    from noxus.data.nbs import probe_nbs

    cfg = cfg or ScaleContrastConfig()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    probe = probe_nbs(out_dir / cfg.probe_name) if probe_armb else None
    df = scale_contrast(cfg)
    out_path = out_dir / cfg.results_name
    df.to_csv(out_path, index=False)
    (out_dir / cfg.findings_name).write_text(render_findings(df, cfg, probe), encoding="utf-8")
    return out_path
