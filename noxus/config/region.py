"""Study-region definitions.

A `Region` is the geographic envelope over which the NO2 column is ingested and attributed. The case
study is the Tangshan (Hebei) steel cluster. The area of interest (AOI) is **derived from the located
steel facilities**, not guessed: it is the bounding envelope of all TROPOMI-era integrated
blast-furnace (BF/BOF) plants in Tangshan prefecture (from the Global Iron & Steel Tracker), expanded
by a configurable buffer so the attribution (flux-divergence) has plume context around edge sources.

The facility envelope below is committed as constants for a stable, import-cheap AOI; it is derived
from `data/derived/tangshan_steel_facilities.csv` and can be recomputed with
`facilities_envelope_from_csv` (see the test in `tests/config/test_region.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Region:
    """A named geographic bounding box, in WGS84 decimal degrees."""

    name: str
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def as_bbox(self) -> tuple[float, float, float, float]:
        """Return (min_lon, min_lat, max_lon, max_lat), the order most APIs expect."""
        return (self.min_lon, self.min_lat, self.max_lon, self.max_lat)

    def buffered(self, buffer_deg: float, name: str | None = None) -> Region:
        """Return a copy expanded by `buffer_deg` degrees on every side."""
        return Region(
            name or self.name,
            self.min_lon - buffer_deg,
            self.min_lat - buffer_deg,
            self.max_lon + buffer_deg,
            self.max_lat + buffer_deg,
        )

    def contains(self, lon: float, lat: float) -> bool:
        """True if (lon, lat) falls within the box (inclusive)."""
        return self.min_lon <= lon <= self.max_lon and self.min_lat <= lat <= self.max_lat


# Envelope of all TROPOMI-era integrated BF/BOF steel plants in Tangshan prefecture, derived from
# data/derived/tangshan_steel_facilities.csv (GEM Global Iron & Steel Tracker, March 2026 V1):
# the min/max of the facility coordinates. Spans ~85 km (E-W) x ~140 km (N-S).
TANGSHAN_FACILITIES_ENVELOPE = Region(
    name="tangshan-facilities",
    min_lon=117.964,
    min_lat=38.954,
    max_lon=119.047,
    max_lat=40.210,
)

# Buffer added around the facility envelope to give flux-divergence attribution plume context without
# pulling in too much external NO2 (the western/SW edge approaches the Beijing-Tianjin-Hebei urban
# plume). 0.25 deg (~28 km) is the default; 0.10 deg (~11 km) is the tighter test alternative. The
# acquisition is done once at the larger buffer and the tighter AOI is derived by clipping.
DEFAULT_AOI_BUFFER_DEG = 0.25
TIGHT_AOI_BUFFER_DEG = 0.10


def tangshan_aoi(buffer_deg: float = DEFAULT_AOI_BUFFER_DEG) -> Region:
    """Return the Tangshan AOI: the facility envelope expanded by `buffer_deg`."""
    return TANGSHAN_FACILITIES_ENVELOPE.buffered(buffer_deg, name="tangshan")


def facilities_envelope_from_csv(
    csv_path: str | Path = "data/derived/tangshan_steel_facilities.csv",
    name: str = "tangshan-facilities",
) -> Region:
    """Recompute the facility envelope from the committed facilities CSV (reproducibility helper)."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    return Region(
        name,
        float(df["longitude"].min()),
        float(df["latitude"].min()),
        float(df["longitude"].max()),
        float(df["latitude"].max()),
    )


# Default study region: the facility-derived AOI at the default buffer.
TANGSHAN = tangshan_aoi()
