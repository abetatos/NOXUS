"""NOX-015 - background-ring radius sensitivity figure (paper critique C1).

Turns ``ring_sensitivity.csv`` (emitted by ``noxus ring-sweep``) into one committable figure:
the cluster-local and regional change-variance shares as a function of the ring's INNER radius,
with the NOX-005 advection scale marked. The question the figure has to answer at a glance is
whether the ~0 cluster-local share of NOX-009 is a property of the cluster or a property of where
we put the background ring.

All (basis x resolution) cells are drawn -- none is dropped for disagreeing (NFR-003). The two
change bases disagree sharply on this data, and that disagreement is the point, so the figure must
not average it away.

Reproduce (after ``noxus ring-sweep`` has written the table):
    uv run python analysis/ring_sensitivity.py

Reads only the committable stats CSV; never reads the CREA benchmark (NFR-001).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

TABLE = Path("data/derived/ring_sensitivity.csv")
FIG_DIR = Path("docs/figures/exploration")
INK, EMBER, GREY = "#0F6B73", "#A8492C", "#6B6B6B"


def _load(path: Path = TABLE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Ring-sensitivity table not found: {path}. Run 'noxus ring-sweep'."
        )
    df = pd.read_csv(path)
    return df[~df["refused"]] if "refused" in df.columns else df


def plot_ring_sensitivity(df: pd.DataFrame, out: Path) -> Path:
    plain = df[df["weighting"] == "plain"]
    octant = df[df["weighting"] == "octant"]
    advection = float(df["advection_km"].iloc[0])

    # Categorical ring axis, not the inner radius: the default grid holds BOTH (60,100) and
    # (60,120), which collide on a numeric inner-radius axis and read as a vertical jump.
    rings = (
        plain[["ring_inner_km", "ring_outer_km"]]
        .drop_duplicates()
        .sort_values(["ring_inner_km", "ring_outer_km"])
        .itertuples(index=False)
    )
    rings = [(float(a), float(b)) for a, b in rings]
    pos = {r: i for i, r in enumerate(rings)}
    labels = [f"{int(a)}-{int(b)}" for a, b in rings]
    straddles = [i for r, i in pos.items() if r[0] < advection]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharex=True)
    styles = {
        ("deseason", "W"): ("-", "o", INK),
        ("deseason", "D"): ("--", "s", INK),
        ("first-diff", "W"): ("-", "^", EMBER),
        ("first-diff", "D"): ("--", "v", EMBER),
    }

    def _xy(grp, col):
        g = grp.copy()
        g["pos"] = [
            pos[(float(a), float(b))] for a, b in zip(g["ring_inner_km"], g["ring_outer_km"])
        ]
        g = g.sort_values("pos")
        return g["pos"], g[col]

    for ax, col, title in (
        (axes[0], "share_steel_local", "cluster-local share"),
        (axes[1], "share_regional", "regional common-mode share"),
    ):
        for i in straddles:
            ax.axvspan(i - 0.5, i + 0.5, color=GREY, alpha=0.10)
        for (basis, res), grp in plain.groupby(["basis", "resolution"]):
            ls, mk, c = styles[(basis, res)]
            x, y = _xy(grp, col)
            ax.plot(x, y, ls, marker=mk, color=c, lw=1.8, ms=5, label=f"{basis} - {res}")
        for (basis, res), grp in octant.groupby(["basis", "resolution"]):
            _, _, c = styles[(basis, res)]
            x, y = _xy(grp, col)
            ax.plot(x, y, ":", color=c, lw=1.0, alpha=0.55)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlabel("background ring, inner-outer (km)")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.25)
    axes[0].annotate(
        f"shaded: ring straddles the\nadvection scale |U|·τ ≈ {advection:.0f} km",
        (0.02, 0.97),
        xycoords="axes fraction",
        fontsize=7.5,
        color=GREY,
        va="top",
    )
    axes[0].set_ylabel("share of change-variance")
    axes[0].legend(
        fontsize=8,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.84),
        title="solid/dashed: plain\ndotted: octant-weighted",
        title_fontsize=7,
    )
    fig.suptitle(
        "NOX-015 - the cluster-local share is ring-DEPENDENT on the deseason basis and flat at zero "
        "on first-difference:\nthe two bases disagree, so neither self-subtraction nor "
        "ring-robustness is established",
        fontsize=9.5,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    df = _load()
    out = plot_ring_sensitivity(df, FIG_DIR / "ring_sensitivity.png")
    print(f"[analysis] ring-sensitivity figure -> {out}")


if __name__ == "__main__":
    main()
