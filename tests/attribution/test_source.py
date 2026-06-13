"""Footprint sampling + background correction tests (NOX-003 T3/T4; AT1, AT2, AT-ERR-1).

Fully offline and deterministic. Uses the synthetic cube + facility CSV fixtures in conftest.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from noxus.attribution import source as S
from noxus.config.region import Region
from noxus.config.run import SignalConfig

from .conftest import LAT, LON

OPERATING_RADIUS_KM = 5.0  # selects the 2 cells nearest the operating facility
RING_INNER_KM, RING_OUTER_KM = 8.0, 25.0


# --------------------------------------------------------------------------- load_facilities (REQ-001)


def test_load_facilities_filters_to_active(facilities_csv):
    df = S.load_facilities(facilities_csv)
    # The retired facility is excluded; the operating one and its co-located duplicate remain.
    assert set(df["status"]) == {"operating"}
    assert len(df) == 2


def test_load_facilities_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        S.load_facilities(tmp_path / "nope.csv")


def test_load_facilities_custom_status_set(facilities_csv):
    # An empty active set selects nothing.
    df = S.load_facilities(facilities_csv, active_status=set())
    assert df.empty


# ----------------------------------------------------------------------------- footprint_mask (AT1)


def test_footprint_mask_selects_in_radius_operating_cells(synthetic_cube, facilities_csv):
    fac = S.load_facilities(facilities_csv)
    mask = S.footprint_mask(synthetic_cube, fac, OPERATING_RADIUS_KM)
    # Exactly the cells within the radius of the operating facility (2 cells at 5 km).
    assert int(mask.sum()) == 2
    # The mask is shaped on the cube's (y, x) grid.
    assert mask.dims == ("y", "x")
    assert mask.sizes == {"y": len(LAT), "x": len(LON)}


def test_footprint_mask_dedups_colocated_facilities(synthetic_cube, facilities_csv):
    fac = S.load_facilities(facilities_csv)
    # Two co-located operating rows; the mask must equal the single-facility mask (EDGE-001).
    mask_both = S.footprint_mask(synthetic_cube, fac, OPERATING_RADIUS_KM)
    single = fac.iloc[[0]]
    mask_single = S.footprint_mask(synthetic_cube, single, OPERATING_RADIUS_KM)
    assert int(mask_both.sum()) == int(mask_single.sum())
    assert bool((mask_both == mask_single).all())


def test_footprint_mask_excludes_retired(synthetic_cube, facilities_csv):
    # Build a footprint from the retired facility ONLY: at a small radius it must select nothing in
    # range of the operating cells, confirming the operating footprint is not influenced by it.
    all_rows = pd.read_csv(facilities_csv)
    retired = all_rows[all_rows["status"] == "retired"].reset_index(drop=True)
    mask_retired = S.footprint_mask(synthetic_cube, retired, OPERATING_RADIUS_KM)
    operating = S.load_facilities(facilities_csv)
    mask_op = S.footprint_mask(synthetic_cube, operating, OPERATING_RADIUS_KM)
    # The retired-only footprint and the operating footprint do not overlap (25 km apart).
    assert not bool((mask_retired & mask_op).any())


def test_footprint_mask_no_facility_in_extent_raises(synthetic_cube, facilities_csv):
    fac = S.load_facilities(facilities_csv)
    # A microscopic radius around facilities that sit between cell centres selects no cell -> ERR-003.
    far = fac.copy()
    far["latitude"] = 50.0
    far["longitude"] = 130.0
    with pytest.raises(S.GeometryError):
        S.footprint_mask(synthetic_cube, far, OPERATING_RADIUS_KM)


def test_footprint_mask_empty_facilities_raises(synthetic_cube):
    empty = pd.DataFrame(columns=["latitude", "longitude", "status"])
    with pytest.raises(S.GeometryError):
        S.footprint_mask(synthetic_cube, empty, OPERATING_RADIUS_KM)


# ------------------------------------------------------------------------ footprint_signal (AT1/AT2)


def test_footprint_value_equals_manual_mean(cube_builder, facilities_csv):
    """Per-period footprint value = manual mean over the valid footprint cells (AT1)."""
    fac = S.load_facilities(facilities_csv)
    ny, nx = len(LAT), len(LON)
    # Plant region-specific values: footprint cells = 12, everything else = 3.
    fp_template = S.footprint_mask(
        cube_builder([np.zeros((ny, nx))], [np.ones((ny, nx))], ["2023-06-04"]),
        fac,
        OPERATING_RADIUS_KM,
    ).values
    g = np.where(fp_template, 12.0, 3.0)
    cube = cube_builder([g], [np.ones((ny, nx))], ["2023-06-04"])

    fp = S.footprint_mask(cube, fac, OPERATING_RADIUS_KM)
    bg = S.background_ring(cube, fp, RING_INNER_KM, RING_OUTER_KM, facilities=fac)
    df = S.footprint_signal(cube, fp, bg, mode="subtract")

    assert df["no2_footprint"].iloc[0] == pytest.approx(12.0)  # mean over footprint cells (all 12)
    assert df["no2_bg"].iloc[0] == pytest.approx(3.0)  # mean over ring cells (all 3)
    assert df["no2_corrected"].iloc[0] == pytest.approx(9.0)  # 12 - 3


def test_footprint_signal_columns_and_subtract(synthetic_cube, facilities_csv):
    fac = S.load_facilities(facilities_csv)
    fp = S.footprint_mask(synthetic_cube, fac, OPERATING_RADIUS_KM)
    bg = S.background_ring(synthetic_cube, fp, RING_INNER_KM, RING_OUTER_KM, facilities=fac)
    df = S.footprint_signal(synthetic_cube, fp, bg, mode="subtract")
    assert list(df.columns) == [
        "date",
        "no2_footprint",
        "no2_bg",
        "no2_corrected",
        "valid_coverage",
    ]
    # Uniform field per period -> footprint == background -> corrected == 0.
    assert df["no2_corrected"].abs().max() == pytest.approx(0.0)
    assert len(df) == 2  # both periods retained


def test_footprint_signal_normalise_mode(synthetic_cube, facilities_csv):
    fac = S.load_facilities(facilities_csv)
    fp = S.footprint_mask(synthetic_cube, fac, OPERATING_RADIUS_KM)
    bg = S.background_ring(synthetic_cube, fp, RING_INNER_KM, RING_OUTER_KM, facilities=fac)
    df = S.footprint_signal(synthetic_cube, fp, bg, mode="normalise")
    # Uniform field -> ratio == 1.
    assert df["no2_corrected"].iloc[0] == pytest.approx(1.0)


# --------------------------------------------------------- coverage propagation, no interpolation (AT2)


def test_below_coverage_week_is_masked_not_interpolated(cube_builder, facilities_csv):
    """A cloud-gapped week (NaN footprint) stays NaN; it is never interpolated (REQ-004)."""
    fac = S.load_facilities(facilities_csv)
    ny, nx = len(LAT), len(LON)
    fp_template = S.footprint_mask(
        cube_builder([np.zeros((ny, nx))], [np.ones((ny, nx))], ["2023-06-04"]),
        fac,
        OPERATING_RADIUS_KM,
    ).values

    good = np.where(fp_template, 12.0, 3.0)
    # Second week: footprint cells are NaN (below coverage / clouded out).
    bad = np.where(fp_template, np.nan, 3.0)
    cov_good = np.ones((ny, nx))
    cov_bad = np.where(fp_template, 0.0, 1.0)
    cube = cube_builder([good, bad], [cov_good, cov_bad], ["2023-06-04", "2023-06-11"])

    fp = S.footprint_mask(cube, fac, OPERATING_RADIUS_KM)
    bg = S.background_ring(cube, fp, RING_INNER_KM, RING_OUTER_KM, facilities=fac)
    df = S.footprint_signal(cube, fp, bg, mode="subtract")

    assert df["no2_corrected"].iloc[0] == pytest.approx(9.0)
    # The clouded week is missing, NOT filled from the neighbour.
    assert np.isnan(df["no2_corrected"].iloc[1])
    assert np.isnan(df["no2_footprint"].iloc[1])


# --------------------------------------------------------------- background ring geometry (AT2/ERR-003)


def test_background_ring_excludes_footprint(synthetic_cube, facilities_csv):
    fac = S.load_facilities(facilities_csv)
    fp = S.footprint_mask(synthetic_cube, fac, OPERATING_RADIUS_KM)
    bg = S.background_ring(synthetic_cube, fp, RING_INNER_KM, RING_OUTER_KM, facilities=fac)
    # No cell is in both the footprint and the background ring.
    assert not bool((fp & bg).any())
    assert int(bg.sum()) > 0


def test_background_ring_clipped_to_aoi(synthetic_cube, facilities_csv):
    """The ring is trimmed to the AOI bounding box (REQ-003, EDGE-002).

    With the default (wide) AOI the ring reaches the eastern columns; a tight AOI that cuts the grid
    at lon=118.50 must remove every ring cell east of that meridian while keeping the western ring.
    """
    fac = S.load_facilities(facilities_csv)
    fp = S.footprint_mask(synthetic_cube, fac, OPERATING_RADIUS_KM)

    # Control: default TANGSHAN AOI covers the whole grid -> ring reaches cells east of 118.50.
    wide = S.background_ring(synthetic_cube, fp, RING_INNER_KM, RING_OUTER_KM, facilities=fac)
    assert bool((wide & (wide.x > 118.50)).any())

    # Clip to an AOI whose eastern edge is 118.50: no ring cell may survive east of it.
    tight = Region(name="tight", min_lon=118.30, min_lat=39.30, max_lon=118.50, max_lat=39.75)
    clipped = S.background_ring(
        synthetic_cube, fp, RING_INNER_KM, RING_OUTER_KM, facilities=fac, aoi=tight
    )
    assert not bool((clipped & (clipped.x > 118.50)).any())
    # The western ring still exists, so this is a genuine clip, not an empty-ring error.
    assert int(clipped.sum()) > 0
    assert int(clipped.sum()) < int(wide.sum())


def test_empty_background_ring_raises(synthetic_cube, facilities_csv):
    fac = S.load_facilities(facilities_csv)
    fp = S.footprint_mask(synthetic_cube, fac, OPERATING_RADIUS_KM)
    # An impossible ring (inner > the grid's reach) contains no cells -> ERR-003.
    with pytest.raises(S.GeometryError):
        S.background_ring(synthetic_cube, fp, inner_km=500.0, outer_km=600.0, facilities=fac)


# ------------------------------------------------------------------------ end-to-end build (ERR-001)


def test_build_footprint_signal_writes_parquet(tmp_path, cube_builder, facilities_csv):
    ny, nx = len(LAT), len(LON)
    g = np.full((ny, nx), 7.0)
    cube = cube_builder([g], [np.ones((ny, nx))], ["2023-06-04"])
    cube_path = tmp_path / "no2_cube_w.nc"
    cube.to_netcdf(cube_path, engine="h5netcdf")

    cfg = SignalConfig(
        footprint_radius_km=OPERATING_RADIUS_KM,
        background_inner_km=RING_INNER_KM,
        background_outer_km=RING_OUTER_KM,
        cube_path=cube_path,
        facilities_csv=facilities_csv,
        out_dir=tmp_path / "out",
    )
    out = S.build_footprint_signal(cfg)
    assert out.exists()
    df = pd.read_parquet(out)
    assert list(df.columns) == [
        "date",
        "no2_footprint",
        "no2_bg",
        "no2_corrected",
        "valid_coverage",
    ]


def test_build_footprint_signal_missing_cube_raises(tmp_path, facilities_csv):
    cfg = SignalConfig(cube_path=tmp_path / "absent.nc", facilities_csv=facilities_csv)
    with pytest.raises(FileNotFoundError):
        S.build_footprint_signal(cfg)
