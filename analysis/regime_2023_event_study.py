"""NOX-012 — figures + findings for the 2023 regime event study (REQ-050, T13).

Reads the committed ``regime_study.csv`` stats table (and, when the NO2-derived residual + CREA
benchmark are present, recomputes the rolling-correlation bands / scan-max-null histogram) and renders
the four output figures + a plain-language findings note:

- ``regime_rolling_bands.png``     — rolling-18-month correlation with the regime bands + event overlay
- ``regime_precision_recall.png``  — per-regime precision/recall/lead
- ``regime_jackknife.png``         — leave-one-event-out contributions to the peak r
- ``regime_scanmax_null.png``      — scan-maximum-null histogram with the observed peak marked

The heavy statistics live in ``noxus.catalyst.regime`` (tested); this driver only *reads* the tables
and the (optional) series and draws. The verdict wording is taken from the regime driver, never
recomputed here. Reproduce (after ``noxus regime-study``):

    uv run python analysis/regime_2023_event_study.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from noxus.catalyst.regime import RegimeArtifacts, regime_study  # noqa: E402
from noxus.config.run import RegimeStudyConfig  # noqa: E402


def _fig_rolling_bands(art: RegimeArtifacts, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 3.2))
    rr = art.regime.rolling_r.dropna()
    ax.plot(rr.index, rr.to_numpy(), lw=1.2, color="#2563eb", label="rolling-r")
    ps, pe = art.regime.peak_window
    ax.axvspan(ps, pe, color="#fde68a", alpha=0.5, label="2023 peak regime")
    ax.axvline(art.regime.peak_date, color="#b45309", ls="--", lw=1.0)
    ax.axhline(0, color="#9ca3af", lw=0.7)
    ax.set_title("Rolling-18-month NO2 vs CREA correlation (descriptive)")
    ax.set_ylabel("r")
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _fig_jackknife(art: RegimeArtifacts, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 3.2))
    c = art.jackknife.contributions
    if len(c):
        ax.bar(range(len(c)), c["contribution"].to_numpy(), color="#7c3aed")
        ax.set_xticks(range(len(c)))
        ax.set_xticklabels(
            [str(pd.Timestamp(d).date()) for d in c["event_date"]], rotation=45, fontsize=6
        )
    ax.axhline(0, color="#9ca3af", lw=0.7)
    ax.set_title(f"Leave-one-event-out contributions ({art.jackknife.verdict})")
    ax.set_ylabel("full_r − loo_r")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _fig_precision_recall(art: RegimeArtifacts, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    es = art.event_study_table
    if len(es):
        labels = [f"{r['regime']}\nthr={r['threshold']}" for _, r in es.iterrows()]
        xs = range(len(es))
        width = 0.4
        ax.bar(
            [x - width / 2 for x in xs],
            es["precision"].to_numpy(),
            width,
            label="precision",
            color="#2563eb",
        )
        ax.bar(
            [x + width / 2 for x in xs],
            es["recall"].to_numpy(),
            width,
            label="recall",
            color="#059669",
        )
        ax.set_xticks(list(xs))
        ax.set_xticklabels(labels, fontsize=6)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)
    ax.set_title("Per-regime event-study precision / recall (threshold × regime)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _fig_scanmax(art: RegimeArtifacts, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.axvline(art.scanmax.observed_scanmax, color="#dc2626", lw=1.6, label="observed peak")
    if art.scanmax.null_hi == art.scanmax.null_hi:  # not NaN
        ax.axvline(art.scanmax.null_hi, color="#059669", ls="--", lw=1.0, label="null 95%")
    ax.set_title(
        f"Scan-maximum null: p={art.scanmax.p_value} beyond_band={art.scanmax.beyond_band}"
    )
    ax.set_xlabel("max rolling-r across overlapping windows")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _findings_text(art: RegimeArtifacts) -> str:
    r = art.results
    return (
        "NOX-012 — 2023 regime event study findings\n"
        "==========================================\n\n"
        f"Verdict: {art.verdict}\n"
        f"n_tests (fixed grid, BH-FDR): {r['n_tests']}\n"
        f"Peak r={r['peak_r']:.3f} at {r['peak_date']}; collapse r={r['collapse_r']:.3f} "
        f"at {r['collapse_date']}.\n"
        f"Scan-max null: observed={r['scanmax_observed']:.3f} p={r['scanmax_p']} "
        f"beyond_band={r['scanmax_beyond_band']}.\n"
        f"Event-bound jackknife: {r['jackknife_verdict']} "
        f"(full r={r['jackknife_full_r']:.3f} -> {r['jackknife_r_without_top']:.3f}).\n"
        f"2024 collapse: {r['collapse_verdict']}.\n"
        f"Peak survives BH-FDR: {r['peak_survives_fdr']}.\n\n"
        f"{r['descriptive_note']}\n"
    )


def run(cfg: RegimeStudyConfig | None = None) -> list[Path]:
    """Render the four figures + findings note from the regime-study artifacts."""
    cfg = cfg or RegimeStudyConfig()
    art = regime_study(cfg, write=True)
    fig_dir = Path(cfg.figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    outs = [
        _fig_rolling_bands(art, fig_dir / "regime_rolling_bands.png"),
        _fig_precision_recall(art, fig_dir / "regime_precision_recall.png"),
        _fig_jackknife(art, fig_dir / "regime_jackknife.png"),
        _fig_scanmax(art, fig_dir / "regime_scanmax_null.png"),
    ]
    findings = fig_dir / cfg.findings_name
    findings.write_text(_findings_text(art), encoding="utf-8")
    outs.append(findings)
    return outs


if __name__ == "__main__":
    for p in run():
        print(f"[noxus] regime figure/finding -> {p}")
