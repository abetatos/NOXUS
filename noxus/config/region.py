"""Study-region definitions.

A `Region` is the geographic envelope over which the NO2 column is aggregated and attributed.
The case study is the Tangshan (Hebei) steel cluster; its bounding box is a placeholder to be
refined against the actual spatial extent of the integrated steel plants.
"""

from __future__ import annotations

from dataclasses import dataclass


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


# Placeholder envelope around the Tangshan steel cluster — refine before use.
TANGSHAN = Region(
    name="tangshan",
    min_lon=117.8,
    min_lat=39.4,
    max_lon=118.8,
    max_lat=39.9,
)
