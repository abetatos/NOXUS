"""Ingest Sentinel-5P/TROPOMI tropospheric NO2 column data.

Source: Copernicus Data Space Ecosystem (free account), or the Google Earth Engine mirror for
research use (requires the optional `geo` extra). Roughly one overpass per day at local early
afternoon; cloud-flagged observations must be dropped rather than interpolated — see the Known
Limitations in the README.
"""

from __future__ import annotations

from datetime import date

from noxus.config.region import Region


def fetch_no2(region: Region, start: date, end: date):
    """Fetch daily tropospheric NO2 columns over `region` for [start, end].

    Returns a gridded, cloud-screened time series (intended as an `xarray.Dataset`).

    Not yet implemented — this is a scaffold. The implementation will authenticate against the
    Copernicus Data Space Ecosystem, request the L2 NO2 product clipped to the region bbox, apply
    the recommended quality filter (qa_value > 0.75), and return per-overpass columns with cloud
    gaps left as missing.
    """
    raise NotImplementedError("TROPOMI NO2 ingestion is not yet implemented")
