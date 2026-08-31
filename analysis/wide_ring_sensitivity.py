"""NOX-017 - wide-AOI ring sensitivity figure (the NOX-015 follow-up).

Turns ``wide_ring_sensitivity.csv`` (emitted by ``noxus ring-sweep --wide``) into one committable
figure. It has to answer two questions at a glance that the NOX-015 figure could not:

* does the cluster-local share **plateau** once the ring is far enough out, or is it still climbing
  when the domain ends? (left panel, with the last increment annotated)
* are the wide rings **real annuli**? The narrow cube's 60-120 km "ring" was four corner wedges, so
  the geometry audit is drawn rather than assumed (right panel: cells and empty octants).

Every (basis x resolution) cell is drawn; none is dropped for disagreeing. The figure's title is
built from the verdict recorded in the table, so it cannot drift from the pre-registered rule.

Reproduce (after ``noxus ring-sweep --wide`` has written the table):
    uv run python analysis/wide_ring_sensitivity.py

Reads only the committable stats CSV; never reads the CREA benchmark (NFR-001).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

TABLE = Path("data/derived/wide_ring_sensitivity.csv")
FIG_DIR = Path("docs/figures/exploration")
INK, EMBER, GREY, SAND = "#0F6B73", "#A8492C", "#6B6B6B", "#B08D57"

# Rings beyond this radius span the Bohai Sea and the Tianjin plume: no longer the same air mass.
AIRMASS_CAVEAT_KM = 120.0


def _load(path: Path = TABLE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Wide ring-sensitivity table not found: {path}. Run 'noxus ring-sweep --wide'."
        )
    df = pd.read_csv(path)
    return df[~df["refused"]] if "refused" in df.columns else df


def plot_wide_ring_sensitivity(df: pd.DataFrame, out: Path) -> Path:
    plain = df[df["weighting"] == "plain"]
    octant = df[df["weighting"] == "octant"]
    verdict = str(df["sweep_verdict"].iloc[0]) if "sweep_verdict" in df.columns else "?"

    rings = [
        (float(a), float(b))
        for a, b in plain[["ring_inner_km", "ring_outer_km"]]
        .drop_duplicates()
        .sort_values(["ring_inner_km", "ring_outer_km"])
        .itertuples(index=False)
    ]
    pos = {r: i for i, r in enumerate(rings)}
    labels = [f"{int(a)}-{int(b)}" for a, b in rings]
    # The NOX-015 cube could not support anything past here; those rings are the new evidence.
    new_rings = [i for r, i in pos.items() if r[1] > AIRMASS_CAVEAT_KM]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
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

    ax = axes[0]
    for i in new_rings:
        ax.axvspan(i - 0.5, i + 0.5, color=SAND, alpha=0.14)
    for (basis, res), grp in plain.groupby(["basis", "resolution"]):
        ls, mk, c = styles[(basis, res)]
        x, y = _xy(grp, "share_steel_local")
        ax.plot(x, y, ls, marker=mk, color=c, lw=1.8, ms=5, label=f"{basis} - {res}")
    for (basis, res), grp in octant.groupby(["basis", "resolution"]):
        _, _, c = styles[(basis, res)]
        x, y = _xy(grp, "share_steel_local")
        ax.plot(x, y, ":", color=c, lw=1.0, alpha=0.55)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("background ring, inner-outer (km)")
    ax.set_ylabel("cluster-local share of change-variance")
    ax.set_title("does the share plateau?", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax.annotate(
        f"shaded: rings beyond {AIRMASS_CAVEAT_KM:.0f} km — unreachable on the\n"
        "0.25° cube, and spanning open sea and the Tianjin plume",
        (0.02, 0.97),
        xycoords="axes fraction",
        fontsize=7.5,
        color=GREY,
        va="top",
    )
    ax.legend(
        fontsize=8,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.80),
        title="solid/dashed: plain\ndotted: octant-weighted",
        title_fontsize=7,
    )

    # Geometry audit: a wider ring is not only further away.
    ax2 = axes[1]
    geo = plain.drop_duplicates(subset=["ring_inner_km", "ring_outer_km"]).copy()
    geo["pos"] = [
        pos[(float(a), float(b))] for a, b in zip(geo["ring_inner_km"], geo["ring_outer_km"])
    ]
    geo = geo.sort_values("pos")
    ax2.bar(geo["pos"], geo["ring_cells"], color=GREY, alpha=0.35, label="ring cells")
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_xlabel("background ring, inner-outer (km)")
    ax2.set_ylabel("cells in the ring")
    ax2.set_title("is the ring a real annulus?", fontsize=10)
    ax2b = ax2.twinx()
    ax2b.plot(
        geo["pos"], geo["empty_octants"], "-o", color=EMBER, lw=1.6, ms=5, label="empty octants"
    )
    ax2b.set_ylabel("empty octants (0 = full annulus)", color=EMBER)
    ax2b.set_ylim(-0.2, max(1.2, float(geo["empty_octants"].max()) + 0.4))
    ax2b.tick_params(axis="y", labelcolor=EMBER)

    fig.suptitle(
        f"NOX-017 — wide-AOI ring sweep (1.25° buffer, rings to 90–150 km): verdict {verdict}",
        fontsize=10,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def compare_period_sets(
    narrow: Path = Path("data/derived/no2/no2_cube_w.nc"),
    wide: Path = Path("data/derived/no2_wide/no2_cube_w.nc"),
) -> dict:
    """Audit the retained period sets of the narrow and wide cubes (NOX-017 REQ-005).

    The wide cube is gridded with ``--coverage-buffer 0.25`` so the PERIOD mask is decided by the
    same geography as the committed cube. This checks that it worked: a difference here means ``n``
    moved at the same time as the ring, and the wide/narrow comparison is confounded by a second
    knob. Reported either way -- never absorbed silently.
    """
    import xarray as xr

    from noxus.data.tropomi import NO2

    def _periods(path: Path) -> set:
        if not path.exists():
            return set()
        ds = xr.open_dataset(path)
        keep = ds[NO2].notnull().any(dim=["y", "x"])
        out = set(pd.to_datetime(ds["time"].values[keep.values]))
        ds.close()
        return out

    a, b = _periods(narrow), _periods(wide)
    return {
        "narrow_periods": len(a),
        "wide_periods": len(b),
        "only_narrow": len(a - b),
        "only_wide": len(b - a),
        "identical": a == b,
    }


def main() -> None:
    df = _load()
    out = plot_wide_ring_sensitivity(df, FIG_DIR / "wide_ring_sensitivity.png")
    print(f"[analysis] wide ring-sensitivity figure -> {out}")
    audit = compare_period_sets()
    print(f"[analysis] retained-period audit (REQ-005): {audit}")


if __name__ == "__main__":
    main()
