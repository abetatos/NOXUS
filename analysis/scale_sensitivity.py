"""Spatial-scale sensitivity figures + findings (NOX-008 T7).

Reads (or runs) the scale-sensitivity sweep table and renders the extent x resolution scale grid of
the NO2<->BF correlation + its robust verdict, then writes a findings file that states the two
verdicts the task is designed to answer: does the Parubets & Naito (2025) pattern hold (significant at
coarse resolution, lost/flipped at fine) on our Tangshan steel data, and does a tighter AOI rescue the
signal (source-isolation, Wen 2024)? Honest null is a valid outcome (Morris & Zhang 2019).

Reproduce (after the cube + benchmark exist):
    uv run python analysis/scale_sensitivity.py
Writes docs/figures/exploration/scale_grid_*.png + scale_sensitivity_findings.txt.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from noxus.config.run import ScaleSweepConfig
from noxus.validation.scale import run_scale_sweep

_RES_ORDER = ["native", "0.1", "0.15", "0.25"]


def _load_table(cfg: ScaleSweepConfig) -> pd.DataFrame:
    path = Path(cfg.out_dir) / cfg.results_name
    if not path.exists():
        print(f"[scale] {path} missing — running the sweep first…")
        path = run_scale_sweep(cfg)
    return pd.read_csv(path)


def _grid(
    df: pd.DataFrame, value: str, freq: str, variant: str, lag_kind: str = "lag0"
) -> pd.DataFrame:
    sub = df[(df["freq"] == freq) & (df["variant"] == variant) & (df["lag_kind"] == lag_kind)]
    if sub.empty:
        return pd.DataFrame()
    piv = sub.pivot_table(index="resolution", columns="buffer", values=value, aggfunc="first")
    rows = [r for r in _RES_ORDER if r in piv.index]
    return piv.reindex(rows)


def fig_scale_grid_r(df: pd.DataFrame, figdir: Path, freq: str, variant: str) -> None:
    piv = _grid(df, "r", freq, variant)
    if piv.empty:
        return
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    im = ax.imshow(piv.to_numpy(float), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{c:.2f}°" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    ax.set_xlabel("AOI buffer (extent)")
    ax.set_ylabel("grid resolution")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iloc[i, j]
            if pd.notna(v):
                ax.text(
                    j,
                    i,
                    f"{v:+.2f}",
                    ha="center",
                    va="center",
                    color="white" if abs(v) > 0.5 else "black",
                    fontsize=9,
                )
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r (NO₂ vs BF, lag 0)")
    ax.set_title(f"Scale grid of r — {variant}, {freq}")
    fig.tight_layout()
    fig.savefig(figdir / f"scale_grid_r_{variant}_{freq}.png", dpi=130)
    plt.close(fig)


def fig_scale_grid_sig(df: pd.DataFrame, figdir: Path, freq: str, variant: str) -> None:
    piv = _grid(df, "verdict", freq, variant)
    if piv.empty:
        return
    code = {"robust": 2, "fragile (naive-only)": 1, "ns": 0}
    num = piv.apply(lambda col: col.map(lambda v: code.get(v, np.nan)))
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.imshow(num.to_numpy(float), cmap="RdYlGn", vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{c:.2f}°" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    ax.set_xlabel("AOI buffer (extent)")
    ax.set_ylabel("grid resolution")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iloc[i, j]
            if pd.notna(v):
                ax.text(
                    j, i, str(v).replace(" (naive-only)", ""), ha="center", va="center", fontsize=7
                )
    ax.set_title(f"Robust verdict (FDR) — {variant}, {freq}")
    fig.tight_layout()
    fig.savefig(figdir / f"scale_grid_sig_{variant}_{freq}.png", dpi=130)
    plt.close(fig)


def hunt_findings(df: pd.DataFrame) -> str:
    lines = ["SPATIAL-SCALE SENSITIVITY — findings (NOX-008)", "=" * 64]
    n_robust = int((df["verdict"] == "robust").sum())
    lines.append(
        f"\nrows: {len(df)} | robust (FDR + CI excl. 0): {n_robust} | "
        f"fragile (naive-only): {int((df['verdict'] == 'fragile (naive-only)').sum())}"
    )

    lines.append("\n[1] Robust findings (verdict == robust):")
    rob = df[df["verdict"] == "robust"]
    if rob.empty:
        lines.append(
            "  NONE — no scale x variant x lag survives autocorrelation-robust + FDR testing."
        )
    else:
        for _, r in rob.iterrows():
            lines.append(
                f"  buffer {r['buffer']:.2f} res {r['resolution']:>6} {r['variant']:14s} {r['freq']:2s} "
                f"lag{int(r['lag']):+d} r={r['r']:+.2f} p_fdr={r['p_fdr']:.3f}"
            )

    # Parubets verdict: for each (buffer, variant, freq, lag0) does |r| change / sign flip coarse->fine?
    lines.append("\n[2] Parubets test (does coarsening change r? lag0):")
    for freq in sorted(df["freq"].unique()):
        for variant in sorted(df["variant"].unique()):
            g = _grid(df, "r", freq, variant)
            if g.empty or "native" not in g.index:
                continue
            for buf in g.columns:
                col = g[buf].dropna()
                if len(col) < 2:
                    continue
                fine = col.get("native", np.nan)
                coarse = col.iloc[-1]
                flip = (
                    np.isfinite(fine) and np.isfinite(coarse) and (np.sign(fine) != np.sign(coarse))
                )
                lines.append(
                    f"  {freq:2s} {variant:14s} buf {buf:.2f}: native r={fine:+.2f} -> "
                    f"coarsest r={coarse:+.2f}{'  [SIGN FLIP]' if flip else ''}"
                )

    # Source-isolation verdict: tight vs wide AOI at native resolution.
    lines.append("\n[3] Source-isolation test (tight 0.10 vs wide 0.25 AOI, native res, lag0):")
    for freq in sorted(df["freq"].unique()):
        for variant in sorted(df["variant"].unique()):
            g = _grid(df, "r", freq, variant)
            if g.empty or "native" not in g.index:
                continue
            row = g.loc["native"]
            wide = row.get(0.25, np.nan)
            tight = row.get(0.10, np.nan)
            if np.isfinite(wide) and np.isfinite(tight):
                better = "tighter helps" if abs(tight) > abs(wide) else "tighter does not help"
                lines.append(
                    f"  {freq:2s} {variant:14s}: wide r={wide:+.2f} tight r={tight:+.2f} ({better})"
                )

    lines.append(
        "\nVerdict framing: an honest NULL across scales is a valid finding (Morris & Zhang 2019); "
        "coarsening adds no information (it only changes background dilution). Every scale is reported "
        "with the FDR/multiplicity caveat — no single scale is the headline."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    cfg = ScaleSweepConfig()
    df = _load_table(cfg)
    figdir = Path(cfg.figures_dir)
    figdir.mkdir(parents=True, exist_ok=True)
    for freq in df["freq"].unique():
        for variant in df["variant"].unique():
            fig_scale_grid_r(df, figdir, freq, variant)
            fig_scale_grid_sig(df, figdir, freq, variant)
    findings = hunt_findings(df)
    (figdir / cfg.findings_name).write_text(findings, encoding="utf-8")
    print(findings)
    print(f"figures + findings -> {figdir}")


if __name__ == "__main__":
    main()
