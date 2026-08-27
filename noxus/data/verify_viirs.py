"""Verify the acquired VIIRS detections against the steel facilities (NOX-011, REQ-030/031).

The co-location sanity check: render the AOI-clipped VIIRS thermal detections with the catalogued
steel-facility locations overlaid, so a reviewer can confirm the thermal detections land on the steel
facilities (a space-based thermometer over the furnaces). Mirrors ``verify_no2.render_day``: a small
committable PNG under ``data/derived/verification/viirs/``. Raw detections stay gitignored (NFR-003);
only this figure + the derived series/table are committable.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from noxus.config.region import Region
from noxus.data.viirs import LATITUDE, LONGITUDE, RADIANT_HEAT_MW, load_detections


def render_detections(
    detections: pd.DataFrame,
    facilities_csv: Path | str,
    out_png: Path | str,
    aoi: Region | None = None,
    title: str | None = None,
) -> Path:
    """Scatter VIIRS detections (sized by radiant heat) with facility points overlaid (REQ-030)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 7))
    if not detections.empty:
        rh = pd.to_numeric(detections.get(RADIANT_HEAT_MW), errors="coerce").fillna(0.0)
        sizes = 12.0 + 40.0 * (rh / rh.max()) if rh.max() > 0 else 18.0
        sc = ax.scatter(
            pd.to_numeric(detections[LONGITUDE], errors="coerce"),
            pd.to_numeric(detections[LATITUDE], errors="coerce"),
            s=sizes,
            c=rh,
            cmap="inferno",
            alpha=0.8,
            label="VIIRS detections",
        )
        fig.colorbar(sc, ax=ax, label="radiant heat (MW)")

    fac = pd.read_csv(facilities_csv)
    ax.scatter(
        fac["longitude"],
        fac["latitude"],
        s=42,
        marker="^",
        c="#2563eb",
        edgecolors="white",
        linewidths=0.6,
        label="steel plants",
    )
    if aoi is not None:
        ax.set_xlim(aoi.min_lon, aoi.max_lon)
        ax.set_ylim(aoi.min_lat, aoi.max_lat)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(title or "VIIRS thermal detections vs steel facilities")
    ax.legend(loc="upper right", fontsize=8)

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_png


def render_from_store(
    raw_dir: Path | str,
    facilities_csv: Path | str,
    out_png: Path | str,
    source: str = "firms",
    aoi: Region | None = None,
) -> Path:
    """Load all persisted detections for a source and render the co-location figure (REQ-030)."""
    detections = load_detections(Path(raw_dir), source)
    return render_detections(detections, facilities_csv, out_png, aoi=aoi)
