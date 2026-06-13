"""Tests for the Tangshan AOI derived from the steel facilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from noxus.config.region import (
    DEFAULT_AOI_BUFFER_DEG,
    TANGSHAN,
    TANGSHAN_FACILITIES_ENVELOPE,
    facilities_envelope_from_csv,
    tangshan_aoi,
)

CSV = Path(__file__).parent.parent.parent / "data" / "derived" / "tangshan_steel_facilities.csv"


def test_buffered_aoi_contains_all_facilities():
    # The operationally meaningful invariant: the AOI we actually ingest (envelope + buffer) must
    # contain every facility, with margin. (The tight envelope constants are rounded to 3 dp, so a
    # boundary plant can sit a fraction of a metre outside the *tight* box — the buffer covers it.)
    df = pd.read_csv(CSV)
    aoi = TANGSHAN
    for _, r in df.iterrows():
        assert aoi.contains(r["longitude"], r["latitude"]), r["name"]


def test_hardcoded_envelope_in_sync_with_csv():
    # Guards against the committed constants drifting from the facilities CSV.
    recomputed = facilities_envelope_from_csv(CSV)
    assert recomputed.min_lon == pytest.approx(TANGSHAN_FACILITIES_ENVELOPE.min_lon, abs=1e-3)
    assert recomputed.min_lat == pytest.approx(TANGSHAN_FACILITIES_ENVELOPE.min_lat, abs=1e-3)
    assert recomputed.max_lon == pytest.approx(TANGSHAN_FACILITIES_ENVELOPE.max_lon, abs=1e-3)
    assert recomputed.max_lat == pytest.approx(TANGSHAN_FACILITIES_ENVELOPE.max_lat, abs=1e-3)


def test_buffer_expands_envelope():
    aoi = tangshan_aoi(DEFAULT_AOI_BUFFER_DEG)
    env = TANGSHAN_FACILITIES_ENVELOPE
    assert aoi.min_lon == pytest.approx(env.min_lon - DEFAULT_AOI_BUFFER_DEG)
    assert aoi.max_lat == pytest.approx(env.max_lat + DEFAULT_AOI_BUFFER_DEG)
    # A tighter buffer yields a strictly smaller box on every side.
    tight = tangshan_aoi(0.10)
    assert tight.min_lon > aoi.min_lon
    assert tight.max_lat < aoi.max_lat


def test_default_tangshan_is_buffered_aoi():
    assert TANGSHAN.name == "tangshan"
    assert TANGSHAN.min_lon < TANGSHAN.max_lon
    assert TANGSHAN.min_lat < TANGSHAN.max_lat
    # Default AOI must contain a known major plant (Shougang Jingtang, Caofeidian).
    assert TANGSHAN.contains(118.50339, 38.953664)
