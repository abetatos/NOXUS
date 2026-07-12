"""NOX-009 — change-variance decomposition figures + findings.

Turns the ``variance_decomposition.csv`` table (emitted by ``noxus variance`` /
``noxus.signal.variance.decompose``) into two committable figures and a findings note:

  * ``variance_budget_stacked.png``  — the four-way variance budget (regional / meteo / steel-local /
    noise) as a stacked bar per (basis × resolution) cell; ALL cells shown (no cherry-picking, NFR-003);
  * ``variance_ceiling_compare.png`` — the implied *change*-detector ceiling (√ steel-local share) with
    its moving-block-bootstrap CI, contrasted with the imported mean-*level* ceiling (√ Wen cap);
  * ``variance_decomposition_findings.txt`` — the physical-vs-method-bound verdict and its NOX-010/011
    implication, worded so a confirmed ceiling reads as a valid finding, not a failure (REQ-050/051).

Reproduce (after ``noxus variance`` has written the table):
    uv run python analysis/variance_decomposition.py

Reads the committable stats CSV (``data/derived/variance_decomposition.csv``); never reads the CREA
benchmark (NFR-001). Deterministic — the table is computed under a fixed seed upstream (NFR-004).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_SHARE_COLS = ("share_regional", "share_meteo", "share_steel_local", "share_noise")
_SHARE_LABELS = ("regional", "meteo", "steel-local", "noise")
_SHARE_COLOURS = ("#93c5fd", "#c4b5fd", "#f59e0b", "#d1d5db")


def _cell_labels(df: pd.DataFrame) -> list[str]:
    return [f"{b}\n{r}" for b, r in zip(df["basis"], df["resolution"], strict=False)]


def plot_stacked_budget(df: pd.DataFrame, out_path: Path) -> Path:
    """Stacked variance-partition bar per (basis × resolution) cell (REQ-050)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = _cell_labels(df)
    x = np.arange(len(df))
    bottom = np.zeros(len(df))
    fig, ax = plt.subplots(figsize=(1.6 * len(df) + 3, 5))
    for col, lab, colour in zip(_SHARE_COLS, _SHARE_LABELS, _SHARE_COLOURS, strict=False):
        vals = df[col].to_numpy(float)
        ax.bar(x, vals, bottom=bottom, label=lab, color=colour, edgecolor="white")
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("share of change-variance")
    ax.set_ylim(0, 1.05)
    ax.set_title("NO2 change-variance budget (regional / meteo / steel-local / noise)")
    ax.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_ceiling_compare(df: pd.DataFrame, out_path: Path) -> Path:
    """Change ceiling (√ steel-local share) with CI vs the level ceiling (√ Wen cap) (REQ-030/050)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = _cell_labels(df)
    x = np.arange(len(df))
    change = df["change_ceiling"].to_numpy(float)
    lo = df.get("change_ceiling_lo", pd.Series(np.nan, index=df.index)).to_numpy(float)
    hi = df.get("change_ceiling_hi", pd.Series(np.nan, index=df.index)).to_numpy(float)
    yerr = np.vstack(
        [
            np.clip(change - np.where(np.isfinite(lo), lo, change), 0, None),
            np.clip(np.where(np.isfinite(hi), hi, change) - change, 0, None),
        ]
    )
    lev_lo = float(df["level_ceiling_lo"].iloc[0]) if len(df) else 0.0
    lev_hi = float(df["level_ceiling_hi"].iloc[0]) if len(df) else 0.0

    fig, ax = plt.subplots(figsize=(1.6 * len(df) + 3, 5))
    ax.axhspan(
        lev_lo,
        lev_hi,
        color="#fca5a5",
        alpha=0.35,
        label=f"level ceiling √cap [{lev_lo:.2f},{lev_hi:.2f}]",
    )
    ax.errorbar(
        x,
        change,
        yerr=yerr,
        fmt="o",
        color="#b45309",
        capsize=4,
        label="change ceiling √f (block-bootstrap CI)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("implied SNR ceiling (|r|)")
    ax.set_ylim(0, max(1.0, float(np.nanmax([*change, lev_hi])) * 1.15))
    ax.set_title("Change-detector ceiling vs imported mean-level ceiling")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_findings(df: pd.DataFrame, out_path: Path, *, headline=("deseason", "W")) -> Path:
    """Findings note: all cells + the headline physical-vs-method verdict (REQ-051, NFR-003)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("NOX-009 — NO2 change-variance decomposition — findings")
    lines.append("=" * 64)
    lines.append("")
    lines.append(
        "Benchmark-blind (NFR-001): the partition is computed on NO2 + meteo + background only; "
        "the CREA benchmark is never an input. Both change bases (first-diff, deseason) at both "
        "resolutions (W, D) are reported below — none is cherry-picked (NFR-003)."
    )
    lines.append("")
    for _, r in df.iterrows():
        lines.append(
            f"[{r['basis']} × {r['resolution']}]  n={int(r['n'])}  block={int(r['block'])}"
        )
        lines.append(
            "  shares: regional={:.3f}  meteo={:.3f}  steel-local={:.3f}  noise={:.3f}".format(
                r["share_regional"], r["share_meteo"], r["share_steel_local"], r["share_noise"]
            )
        )
        lines.append(
            "  raw ΔR²: regional={:.3f}  meteo={:.3f}  local={:.3f}  (entry order {})".format(
                r["dr2_regional"], r["dr2_meteo"], r["dr2_local"], r["order"]
            )
        )
        lines.append(
            "  change ceiling √f={:.3f}  block CI=[{:.3f},{:.3f}]  i.i.d. share CI=[{:.3f},{:.3f}]".format(
                r["change_ceiling"],
                r.get("change_ceiling_lo", float("nan")),
                r.get("change_ceiling_hi", float("nan")),
                r.get("share_lo_iid", float("nan")),
                r.get("share_hi_iid", float("nan")),
            )
        )
        lines.append(
            "  level ceiling √cap=[{:.3f},{:.3f}]".format(
                r["level_ceiling_lo"], r["level_ceiling_hi"]
            )
        )
        if str(r.get("flags", "")):
            lines.append(f"  flags: {r['flags']}")
        lines.append(f"  verdict: {r['verdict']}")
        lines.append("")

    head = df[(df["basis"] == headline[0]) & (df["resolution"] == headline[1])]
    if len(head):
        r = head.iloc[0]
        lines.append(
            "HEADLINE (deseasonalised residual, weekly — the basis behind the reported r≈0.07):"
        )
        lines.append(f"  {r['verdict']}")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def build_report(df: pd.DataFrame, figures_dir: Path) -> dict[str, Path]:
    """Emit both figures + the findings note for a completed budget; return their paths (AT7)."""
    figures_dir = Path(figures_dir)
    return {
        "stacked": plot_stacked_budget(df, figures_dir / "variance_budget_stacked.png"),
        "ceiling": plot_ceiling_compare(df, figures_dir / "variance_ceiling_compare.png"),
        "findings": write_findings(df, figures_dir / "variance_decomposition_findings.txt"),
    }


def main() -> int:
    from noxus.config.run import VarianceDecompConfig

    cfg = VarianceDecompConfig()
    table = Path(cfg.out_dir) / cfg.results_name
    if not table.exists():
        print(f"[noxus] {table} missing — run 'noxus variance' first.")
        return 1
    df = pd.read_csv(table)
    artefacts = build_report(df, cfg.figures_dir)
    for name, path in artefacts.items():
        print(f"[noxus] {name} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
