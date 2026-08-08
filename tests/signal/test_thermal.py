"""Tests for the VIIRS facility join + thermal series (REQ-010..012, EDGE-001/002/005, ERR-005).

Covers AT3a (radius join), AT3b (weekly/daily aggregation), AT3c (benchmark-blind), AT-EDGE-1
(empty/sparse), and the bad-facilities-CSV error (AT-ERR-1). Synthetic detections only; no network.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from noxus.data import viirs as V
from noxus.signal import thermal as TH

# Two facilities ~40 km apart so a 3 km radius cleanly separates them.
FAC = pd.DataFrame(
    {
        "name": ["A", "B"],
        "latitude": [39.60, 39.95],
        "longitude": [118.20, 118.50],
    }
)


def _det(lat, lon, day="2023-06-05T18:00", rh=5.0, temp=1500.0, did="d"):
    return {
        V.TEMPERATURE_K: temp,
        V.SOURCE_AREA: 0.01,
        V.RADIANT_HEAT_MW: rh,
        V.LATITUDE: lat,
        V.LONGITUDE: lon,
        V.TIMESTAMP: pd.Timestamp(day),
        V.DETECTION_ID: did,
        V.QUALITY_FLAG: 1,
    }


# ------------------------------------------------------------------------------------ AT3a


def test_join_assigns_nearest_within_radius_drops_outside():
    dets = pd.DataFrame(
        [
            _det(39.601, 118.201, did="near_A"),  # ~0.15 km from A
            _det(39.951, 118.501, did="near_B"),  # near B
            _det(39.00, 117.00, did="far"),  # outside every radius
        ]
    )
    joined = TH.join_to_facilities(dets, FAC, radius_km=3.0)
    assert len(joined) == 2  # 'far' dropped
    names = set(joined["facility_name"])
    assert names == {"A", "B"}
    # Each detection assigned exactly once (no double counting).
    assert list(joined[V.DETECTION_ID]) == ["near_A", "near_B"]


def test_join_ties_to_nearest_facility():
    # A detection near A but within radius of both if radius were large -> must pick the nearest (A).
    dets = pd.DataFrame([_det(39.605, 118.205, did="x")])
    joined = TH.join_to_facilities(dets, FAC, radius_km=100.0)
    assert len(joined) == 1
    assert joined["facility_name"].iloc[0] == "A"


def test_join_uses_no_geopandas():
    src = inspect.getsource(TH)
    assert "import geopandas" not in src and "import shapely" not in src


# ------------------------------------------------------------------------------------ AT3b


def test_weekly_aggregation_sum_count_meantemp():
    dets = pd.DataFrame(
        [
            _det(39.601, 118.201, day="2023-06-05T18:00", rh=4.0, temp=1400.0),
            _det(39.601, 118.201, day="2023-06-06T18:00", rh=6.0, temp=1600.0),
        ]
    )
    joined = TH.join_to_facilities(dets, FAC, radius_km=3.0)
    weekly = TH.build_thermal_series(joined, freq="W")
    assert len(weekly) == 1
    row = weekly.iloc[0]
    assert row["radiant_heat_mw"] == pytest.approx(10.0)  # summed radiant heat = primary
    assert row["detection_count"] == 2
    assert row["mean_temperature_k"] == pytest.approx(1500.0)
    assert row["activity"] == pytest.approx(10.0)  # headline = summed radiant heat


def test_daily_and_weekly_both_produced():
    dets = pd.DataFrame([_det(39.601, 118.201, day="2023-06-05T18:00", rh=3.0)])
    joined = TH.join_to_facilities(dets, FAC, radius_km=3.0)
    daily = TH.build_thermal_series(joined, freq="D")
    weekly = TH.build_thermal_series(joined, freq="W")
    assert daily["radiant_heat_mw"].sum() == pytest.approx(3.0)
    assert weekly["radiant_heat_mw"].sum() == pytest.approx(3.0)


def test_alternate_aggregate_count():
    dets = pd.DataFrame(
        [
            _det(39.601, 118.201, day="2023-06-05T18:00", rh=4.0),
            _det(39.601, 118.201, day="2023-06-06T18:00", rh=6.0),
        ]
    )
    joined = TH.join_to_facilities(dets, FAC, radius_km=3.0)
    weekly = TH.build_thermal_series(joined, freq="W", aggregate="count")
    assert weekly["activity"].iloc[0] == 2  # headline switched to detection count


# ------------------------------------------------------------------------------------ AT3c


def test_series_build_is_benchmark_blind():
    join_params = set(inspect.signature(TH.join_to_facilities).parameters)
    build_params = set(inspect.signature(TH.build_thermal_series).parameters)
    forbidden = {"crea", "benchmark", "no2", "target"}
    assert not (join_params & forbidden)
    assert not (build_params & forbidden)
    # And the module never imports/loads the CREA benchmark or NO2 series.
    src = inspect.getsource(TH)
    assert "noxus.data.benchmark" not in src
    assert "benchmark_path" not in src


# ------------------------------------------------------------------------------------ AT-EDGE-1


def test_sparse_week_flagged_and_zero_periods_filled():
    # One detection-night in an otherwise empty multi-week range -> sparse flag + zero-filled weeks.
    dets = pd.DataFrame([_det(39.601, 118.201, day="2023-06-05T18:00", rh=5.0)])
    joined = TH.join_to_facilities(dets, FAC, radius_km=3.0)
    weekly = TH.build_thermal_series(
        joined, freq="W", sparse_min_nights=2, start="2023-06-01", end="2023-06-28"
    )
    assert len(weekly) >= 4  # full range reindexed, zero periods present
    assert (weekly["radiant_heat_mw"] == 0).any()  # empty weeks are zero-activity, not gaps
    assert bool(weekly["sparse"].all())  # every week has < 2 detection-nights -> flagged
    assert not weekly["radiant_heat_mw"].isna().any()  # nothing interpolated to NaN


def test_empty_join_yields_empty_series():
    empty = TH.join_to_facilities(V.standardise(V._empty_frame()), FAC, radius_km=3.0)
    series = TH.build_thermal_series(empty, freq="W")
    assert series.empty


# ------------------------------------------------------------------------------------ ERR-005


def test_load_facilities_missing_file_raises(tmp_path):
    with pytest.raises(TH.FacilitiesError):
        TH.load_facilities(tmp_path / "nope.csv")


def test_load_facilities_missing_column_raises(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"name": ["A"], "lat": [1.0]}).to_csv(bad, index=False)
    with pytest.raises(TH.FacilitiesError):
        TH.load_facilities(bad)


def test_haversine_known_distance():
    # ~111 km per degree of latitude at the equator; assert the join distance is in a sane range.
    d = TH._haversine_km(np.array([0.0]), np.array([0.0]), np.array([1.0]), np.array([0.0]))
    assert d[0] == pytest.approx(111.19, rel=0.02)
