"""NOX-016 - scale-contrast figure (paper critique C2).

Turns ``scale_contrast.csv`` (emitted by ``noxus scale-contrast``) into one committable figure:
the correlation of each NO2 feature against each steel target, ordered by the target's SPATIAL
SCALE, with bootstrap intervals. The question it has to answer at a glance is whether the coupling
strengthens as the target widens from the Tangshan cluster to the nation -- which is what the
paper's spatial explanation predicts.

Reproduce (after ``noxus scale-contrast``):
    uv run python analysis/scale_contrast.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TABLE = Path("data/derived/scale_contrast.csv")
FIG_DIR = Path("docs/figures/exploration")
INK, EMBER, GREY = "#0F6B73", "#A8492C", "#6B6B6B"
ORDER = ["tangshan_bf_rate", "national_bf_rate", "national_crude_steel", "national_pig_iron"]


def _load(path: Path = TABLE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Scale-contrast table not found: {path}. Run 'noxus scale-contrast'."
        )
    return pd.read_csv(path)


def plot_scale_contrast(df: pd.DataFrame, out: Path) -> Path:
    full = df[(df["subperiod"] == "full") & (~df["refused"])]
    targets = [t for t in ORDER if t in set(full["target"])]
    features = sorted(full["feature"].unique())
    colors = {features[0]: INK, features[-1]: EMBER}

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    width = 0.36
    for i, feat in enumerate(features):
        g = full[full["feature"] == feat].set_index("target").reindex(targets)
        xs = np.arange(len(targets)) + (i - (len(features) - 1) / 2) * width
        lo = (g["r"] - g["boot_lo"]).to_numpy()
        hi = (g["boot_hi"] - g["r"]).to_numpy()
        ax.errorbar(
            xs,
            g["r"],
            yerr=np.vstack([lo, hi]),
            fmt="o",
            ms=6,
            capsize=4,
            color=colors.get(feat, GREY),
            label=feat,
            lw=1.6,
        )
        for x, n in zip(xs, g["n"]):
            if np.isfinite(n):
                ax.annotate(
                    f"n={int(n)}",
                    (x, ax.get_ylim()[0]),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                    color=GREY,
                )
    ax.axhline(0, color="black", lw=0.9)
    ax.axhspan(0.5, 0.75, color=GREY, alpha=0.12)
    ax.annotate(
        "national-scale literature range 0.5-0.75",
        (len(targets) - 0.5, 0.625),
        ha="right",
        va="center",
        fontsize=8,
        color=GREY,
    )
    ax.set_xticks(np.arange(len(targets)))
    ax.set_xticklabels([t.replace("_", "\n") for t in targets], fontsize=8)
    ax.set_xlabel("scoring target, ordered by spatial scale  (cluster $\\rightarrow$ national)")
    ax.set_ylabel("Pearson r (lag 0), block-bootstrap 95% CI")
    ax.set_title(
        "NOX-016 - widening the target does not strengthen the coupling:\n"
        "the national pairings are near-null too",
        fontsize=10,
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    out = plot_scale_contrast(_load(), FIG_DIR / "scale_contrast.png")
    print(f"[analysis] scale-contrast figure -> {out}")


if __name__ == "__main__":
    main()
