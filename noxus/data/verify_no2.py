"""Verify the acquired NO2 against the steel facilities.

The location/sanity check the developer asked for: on clear-sky, high-NO2 days, render the TROPOMI NO2
field over the AOI with the steel-facility locations overlaid, so one can confirm the NO2 enhancement
co-locates with the steel sub-clusters. Optionally add a Sentinel-2 optical thumbnail (same openEO
connection) as a final check that a plant is physically present where the NO2 appears (REQ-010..012).

Remember the resolution reality: at ~5.5x3.5 km a plant is sub-pixel and many Tangshan plants share a
pixel — this confirms the signal at cluster/sub-cluster level, not per-factory.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr

from noxus.config.region import Region
from noxus.data.tropomi import NO2, load_manifest


def select_clear_high_days(
    raw_dir: Path | str, n: int = 5, max_cloud: float = 0.2, min_coverage: float = 0.5
) -> list[dict]:
    """Pick the clearest, highest-NO2 overpasses from the acquired manifest (REQ-010).

    Returns up to ``n`` entries (each with ``id``, ``path``, ``cloud_mean``, ``no2_mean``), filtered to
    cloud_mean ≤ ``max_cloud`` and coverage ≥ ``min_coverage``, sorted by NO2 mean descending.
    """
    manifest = load_manifest(Path(raw_dir))
    rows = []
    for opid, e in manifest["overpasses"].items():
        cloud = e.get("cloud_mean")
        no2 = e.get("no2_mean")
        cov = e.get("valid_coverage", 0.0)
        if no2 is None:
            continue
        if cloud is not None and cloud > max_cloud:
            continue
        if cov < min_coverage:
            continue
        rows.append({"id": opid, "path": e.get("path"), "cloud_mean": cloud, "no2_mean": no2})
    rows.sort(key=lambda r: r["no2_mean"], reverse=True)
    return rows[:n]


def render_day(
    overpass_path: Path | str,
    facilities_csv: Path | str,
    out_png: Path | str,
    aoi: Region | None = None,
    title: str | None = None,
) -> Path:
    """Render the NO2 field of one overpass with steel-facility points overlaid (REQ-011)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = xr.open_dataset(overpass_path)
    field = ds[NO2]
    if "time" in field.dims:
        field = field.isel(time=0)
    data = field.values

    extent = list(aoi.as_bbox()) if aoi is not None else None  # [min_lon,min_lat,max_lon,max_lat]
    fig, ax = plt.subplots(figsize=(6, 7))
    if extent is not None:
        img_extent = [extent[0], extent[2], extent[1], extent[3]]
        im = ax.imshow(data, origin="lower", extent=img_extent, aspect="auto", cmap="viridis")
    else:
        im = ax.imshow(data, origin="lower", aspect="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, label="tropospheric NO₂")

    fac = pd.read_csv(facilities_csv)
    ax.scatter(
        fac["longitude"],
        fac["latitude"],
        s=18,
        c="red",
        edgecolors="white",
        linewidths=0.5,
        label="steel plants",
    )
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(title or f"NO₂ vs steel facilities — {Path(overpass_path).stem}")
    ax.legend(loc="upper right", fontsize=8)

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_png


def fetch_optical_thumbnail(aoi: Region, day: str, out_png: Path | str, cfg=None) -> Path:
    """Fetch a Sentinel-2 true-colour thumbnail of the AOI via openEO (optional, REQ-012).

    Requires CDSE credentials / network; not exercised in unit tests. Confirms a plant is physically
    present where the NO2 enhancement appears.
    """
    from noxus.config.run import AcquisitionConfig
    from noxus.data.tropomi import connect_openeo

    cfg = cfg or AcquisitionConfig()
    conn = connect_openeo(cfg)
    west, south, east, north = aoi.as_bbox()
    cube = conn.load_collection(
        cfg.optical_collection_id,
        spatial_extent={"west": west, "south": south, "east": east, "north": north},
        temporal_extent=[day, day],
        bands=list(cfg.optical_bands),
    )
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cube.download(str(out_png))
    return out_png
