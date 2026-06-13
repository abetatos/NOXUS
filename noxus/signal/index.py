"""Build the monthly activity index from the per-overpass attributed NO2 proxy.

Aggregates irregular, cloud-gapped daily proxies into a monthly index aligned to the benchmark,
handling missing observations explicitly (gaps correlate with season and weather, so they are not
missing-at-random and must not be naively interpolated).
"""

from __future__ import annotations


def build_index(attributed_proxy):
    """Aggregate the per-overpass proxy into a monthly activity index.

    Not yet implemented — this is a scaffold.
    """
    raise NotImplementedError("Activity-index construction is not yet implemented")
