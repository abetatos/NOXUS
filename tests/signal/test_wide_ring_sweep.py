"""Wide-AOI ring sweep tests (NOX-017; AT1-AT7, AT-REG-1, AT-ERR-1).

Offline and deterministic. Two things make this suite different from the NOX-015 one:

* the synthetic cube spans the **wide** 1.25-deg AOI, because the point of the task is that
  ``background_ring`` clips to a box that defaults to the narrow 0.25-deg region -- a wide sweep
  that forgets to pass its AOI silently reproduces the narrow one (AT3);
* the verdict is decided by three pre-registered rules instead of one, and the rule NOX-015 got
  wrong (level, not rise) is now covered by a negative control that must NOT fire (AT7).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from noxus.config.region import tangshan_aoi
from noxus.config.run import RingSweepConfig, WideRingSweepConfig
from noxus.data.gridding import COVERAGE
from noxus.data.tropomi import NO2
from noxus.signal import ring_sweep as R

CLON, CLAT = 118.50, 39.58
# Wide grid: spans the 1.25-deg AOI, i.e. well beyond the committed 0.25-deg box.
LON = np.round(np.arange(116.80, 120.25 + 1e-9, 0.06), 4)
LAT = np.round(np.arange(41.40, 37.75 - 1e-9, -0.05), 4)
N_PERIODS = 200
WIDE_AOI = tangshan_aoi(1.25)


def _dist_km(lon2d, lat2d, clon=CLON, clat=CLAT):
    return np.hypot((lon2d - clon) * 85.0, (lat2d - clat) * 111.0)


def _ar1(rng, n, phi, sd=1.0):
    """AR(1): the partition books the *serially structured* part of the residual as 'local', so a
    white-noise source would land in 'noise' by construction and every control would pass vacuously."""
    x = np.zeros(n)
    e = rng.normal(0.0, sd, n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def _wide_cube(plume_gain: float, seed: int = 11):
    """Wide cube: regional common mode + persistent cluster activity, optional in-ring plume."""
    rng = np.random.default_rng(seed)
    lon2d, lat2d = np.meshgrid(LON, LAT)
    d = _dist_km(lon2d, lat2d)
    regional = _ar1(rng, N_PERIODS, 0.85)
    local = _ar1(rng, N_PERIODS, 0.92)
    footprint = d <= 15.0
    plume = (d >= 25.0) & (d <= 55.0)
    grids = []
    for t in range(N_PERIODS):
        g = np.full(d.shape, 50.0) + regional[t]
        g[footprint] += 3.0 * local[t]
        if plume_gain:
            g[plume] += plume_gain * local[t]
        g += rng.normal(0.0, 0.05, d.shape)
        grids.append(g)
    dates = pd.date_range("2020-01-05", periods=N_PERIODS, freq="W")
    return xr.Dataset(
        {
            NO2: (("time", "y", "x"), np.stack(grids)),
            COVERAGE: (("time", "y", "x"), np.ones((N_PERIODS, len(LAT), len(LON)))),
        },
        coords={"time": dates, "y": LAT, "x": LON},
    )


@pytest.fixture
def facilities_csv(tmp_path):
    path = tmp_path / "facilities.csv"
    pd.DataFrame(
        [
            {
                "name": "Synthetic BF",
                "latitude": CLAT,
                "longitude": CLON,
                "technology": "BF; BOF",
                "status": "operating",
            }
        ]
    ).to_csv(path, index=False)
    return path


def _cfg(facilities_csv, tmp_path, **over):
    base = dict(
        resolutions=("W",),
        weightings=("plain",),
        n_boot=150,
        facilities_csv=facilities_csv,
        out_dir=tmp_path / "out",
    )
    base.update(over)
    return dataclasses.replace(WideRingSweepConfig(), **base)


def _shares(cube, cfg, ring, aoi=WIDE_AOI):
    """Local/regional share for one ring, without ERA5 (the meteo term is not under test here)."""
    from noxus.attribution.source import (
        background_ring,
        footprint_mask,
        footprint_signal,
        load_facilities,
    )
    from noxus.config.run import SignalConfig
    from noxus.signal.variance import decompose_from_levels

    sig = SignalConfig()
    fac = load_facilities(cfg.facilities_csv, set(sig.active_statuses))
    fp = footprint_mask(cube, fac, sig.footprint_radius_km)
    bg = background_ring(cube, fp, ring[0], ring[1], facilities=fac, aoi=aoi)
    df = footprint_signal(cube, fp, bg, mode=sig.background_mode)
    idx = pd.DatetimeIndex(pd.to_datetime(df["date"]))
    levels = (
        pd.Series(df["no2_footprint"].to_numpy(float), index=idx, name="footprint"),
        pd.Series(df["no2_bg"].to_numpy(float), index=idx, name="background"),
        None,
    )
    out = decompose_from_levels({"W": levels}, cfg)
    return out.set_index("basis")[["share_steel_local", "share_regional", "share_lo"]]


def _frame(rows):
    """Minimal sweep frame the verdict rules can read."""
    return pd.DataFrame(
        [
            {
                "weighting": "plain",
                "refused": False,
                "beyond_advection": r["inner"] >= 35.71,
                "ring_inner_km": r["inner"],
                "ring_outer_km": r["outer"],
                "basis": r.get("basis", "deseason"),
                "resolution": r.get("resolution", "W"),
                "share_steel_local": r["share"],
                "share_lo": r.get("lo", 0.0),
                "share_regional": r.get("regional", 0.5),
            }
            for r in rows
        ]
    )


def _ladder(shares, lo=0.01):
    """One cell per basis over the pre-registered grid, with the given share ladder."""
    grid = WideRingSweepConfig().ring_grid
    assert len(shares) == len(grid)
    return _frame(
        [
            {"inner": g[0], "outer": g[1], "basis": b, "share": s, "lo": lo}
            for g, s in zip(grid, shares)
            for b in ("deseason", "first-diff")
        ]
    )


# ------------------------------------------------------------------- AT1  wide fetch, separate store


def test_at1_fetch_raw_dir_routes_writes_to_its_own_store(tmp_path, monkeypatch):
    # REQ-001/002: the manifest is keyed by date alone, so a wide fetch landing in the committed
    # store would either skip everything or merge two geometries into one record.
    from datetime import date

    from noxus.config.run import AcquisitionConfig
    from noxus.data import tropomi as T

    class _Fake:
        def fetch_window(self, aoi, t0, t1):
            times = pd.date_range(t0, periods=1, freq="D")
            return xr.Dataset(
                {T.NO2: (("time", "y", "x"), np.ones((1, 2, 2)))},
                coords={"time": times, "y": [39.5, 39.6], "x": [118.4, 118.5]},
            )

    store = tmp_path / "wide_store"
    cfg = dataclasses.replace(
        AcquisitionConfig(), start=date(2021, 1, 1), end=date(2021, 1, 2), raw_dir=store
    )
    report = T.acquire_no2(tangshan_aoi(1.25), cfg, fetcher=_Fake())
    assert report.fetched >= 1
    assert (store / "manifest.json").exists()
    assert Path(report.raw_dir) == store
    assert not (tmp_path / "tropomi").exists()  # nothing written outside the requested store


def test_at1_cli_passes_raw_dir_and_buffer_through_to_the_acquisition(monkeypatch, tmp_path):
    from noxus.cli.main import main
    from noxus.data import tropomi as T

    seen = {}

    def _spy(aoi, cfg, *a, **kw):
        seen["bbox"] = aoi.as_bbox()
        seen["raw_dir"] = Path(cfg.raw_dir)
        return T.AcquireReport(0, 0, 0, ("v2.x",), str(cfg.raw_dir))

    monkeypatch.setattr(T, "acquire_no2", _spy)
    rc = main(["fetch", "--buffer", "1.25", "--raw-dir", str(tmp_path / "wide")])
    assert rc == 0
    assert seen["raw_dir"] == tmp_path / "wide"
    assert seen["bbox"] == tangshan_aoi(1.25).as_bbox()


# ------------------------------------------------------------------- AT2  coverage sub-box


def _coverage_cube(valid_narrow: bool, valid_wide: bool):
    """One period. The narrow half of the box is valid iff ``valid_narrow``, the wide half iff
    ``valid_wide`` -- so the period survives or not depending on where coverage is measured."""
    lon = np.array([117.9, 118.5, 119.9])  # first two inside the 0.25-deg AOI, third outside
    lat = np.array([39.6, 39.4])
    vals = np.full((1, len(lat), len(lon)), np.nan)
    if valid_narrow:
        vals[:, :, :2] = 5.0
    if valid_wide:
        vals[:, :, 2] = 5.0
    return xr.Dataset(
        {NO2: (("time", "y", "x"), vals)},
        coords={"time": pd.to_datetime(["2021-01-03"]), "y": lat, "x": lon},
    )


def test_at2_coverage_bbox_none_reproduces_the_committed_behaviour():
    from noxus.config.run import GriddingConfig
    from noxus.data.gridding import composite

    cube = _coverage_cube(valid_narrow=True, valid_wide=False)
    cfg = dataclasses.replace(GriddingConfig(), freq="W", min_period_coverage=0.25)
    assert cfg.coverage_bbox is None  # the default must not change
    out = composite(cube, cfg)
    assert out.attrs["coverage_bbox"] == "whole-cube"
    assert float(out["period_coverage"].values[0]) == pytest.approx(4 / 6)


def test_at2_coverage_bbox_changes_only_which_periods_survive():
    # REQ-004: coverage measured over the committed sub-box, so a wider acquisition keeps the SAME
    # temporal mask. The retained VALUES must be untouched -- only the masking decision moves.
    from noxus.config.run import GriddingConfig
    from noxus.data.gridding import composite

    cube = _coverage_cube(valid_narrow=False, valid_wide=True)
    narrow_bbox = tangshan_aoi(0.25).as_bbox()
    wide_cfg = dataclasses.replace(GriddingConfig(), freq="W", min_period_coverage=0.25)
    sub_cfg = dataclasses.replace(wide_cfg, coverage_bbox=narrow_bbox)

    whole = composite(cube, wide_cfg)
    sub = composite(cube, sub_cfg)
    # Whole-cube coverage sees the far-east cells and keeps the period; the sub-box does not.
    assert float(whole["period_coverage"].values[0]) == pytest.approx(2 / 6)
    assert float(sub["period_coverage"].values[0]) == pytest.approx(0.0)
    assert np.isnan(sub[NO2].values).all()
    assert sub.attrs["coverage_bbox"] == str(narrow_bbox)


def test_at2_coverage_bbox_off_the_cube_is_refused():
    from noxus.config.run import GriddingConfig
    from noxus.data.gridding import GriddingError, composite

    cube = _coverage_cube(valid_narrow=True, valid_wide=True)
    cfg = dataclasses.replace(GriddingConfig(), coverage_bbox=(0.0, 0.0, 1.0, 1.0))
    with pytest.raises(GriddingError, match="selects no cells"):
        composite(cube, cfg)


# ------------------------------------------------------------------- AT3  the AOI must reach the ring


def test_at3_wide_ring_is_clipped_to_the_narrow_box_without_an_explicit_aoi(facilities_csv):
    # REQ-010: this is the failure mode the whole task turns on. Forgetting the AOI does not raise
    # -- it silently returns the narrow ring, so the wide sweep would look like a plateau.
    from noxus.attribution.source import background_ring, footprint_mask, load_facilities
    from noxus.config.run import SignalConfig

    cube = _wide_cube(plume_gain=0.0)
    sig = SignalConfig()
    fac = load_facilities(facilities_csv, set(sig.active_statuses))
    fp = footprint_mask(cube, fac, sig.footprint_radius_km)
    default = R.ring_geometry(cube, fac, background_ring(cube, fp, 90.0, 150.0, facilities=fac))
    wide = R.ring_geometry(
        cube, fac, background_ring(cube, fp, 90.0, 150.0, facilities=fac, aoi=WIDE_AOI)
    )
    assert wide["ring_cells"] > 3 * default["ring_cells"]
    assert wide["realised_max_km"] > default["realised_max_km"] + 20.0
    assert wide["empty_octants"] == 0  # a real annulus, not four corner wedges


def test_at3_sweep_aoi_resolves_from_the_config():
    assert R.sweep_aoi(RingSweepConfig()) is None  # NOX-015 behaviour: fall back to TANGSHAN
    aoi = R.sweep_aoi(WideRingSweepConfig())
    assert aoi is not None
    assert aoi.as_bbox() == tangshan_aoi(1.25).as_bbox()


# ------------------------------------------------------------------- AT4  grid + geometry audit


def test_at4_extended_grid_reproduces_the_nox015_grid_then_extends_it():
    narrow, wide = RingSweepConfig().ring_grid, WideRingSweepConfig().ring_grid
    assert wide[: len(narrow)] == narrow, "the first rings must match NOX-015 exactly, or the two "
    "runs are not comparable"
    assert len(wide) == len(narrow) + 2
    assert wide[-1] == (90.0, 150.0)
    inners = [r[0] for r in wide]
    outers = [r[1] for r in wide]
    assert inners == sorted(inners) and outers == sorted(outers)


def test_at4_wide_config_inherits_the_nox015_estimator_settings():
    # REQ-011: the ring is the only knob that moves.
    narrow, wide = RingSweepConfig(), WideRingSweepConfig()
    for field in ("bases", "order", "deseason_method", "meteo_covariates", "n_boot", "seed",
                  "self_subtraction_threshold", "min_cells_agreeing", "wind_ms", "tau_h"):
        assert getattr(wide, field) == getattr(narrow, field), field


# ------------------------------------------------------------------- AT5  verdict logic


def test_at5_self_subtraction_when_both_rules_fire():
    df = _ladder([0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.255], lo=0.01)
    verdict, reason, detail = R.classify_wide_verdict(df)
    assert verdict == R.VERDICT_SELF_SUBTRACTION
    assert detail["rule_a"] and detail["rule_b"]
    assert "both pre-registered rules fire" in reason


def test_at5_ring_robust_when_neither_rule_fires_and_the_share_has_plateaued():
    df = _ladder([0.010, 0.011, 0.012, 0.012, 0.013, 0.013, 0.013], lo=-0.01)
    verdict, _, detail = R.classify_wide_verdict(df)
    assert verdict == R.VERDICT_ROBUST
    assert detail["plateau"] and not detail["rule_a"] and not detail["rule_b"]


def test_at5_domain_limited_when_the_rise_is_still_climbing_at_the_edge():
    # The NOX-015 situation, now nameable: a share still moving when the cube runs out.
    df = _ladder([0.005, 0.010, 0.016, 0.022, 0.028, 0.034, 0.048], lo=0.001)
    verdict, reason, detail = R.classify_wide_verdict(df)
    assert verdict == R.VERDICT_DOMAIN_LIMITED
    assert not detail["plateau"]
    assert "NOT plateaued" in reason


def test_at5_a_cell_clamped_at_zero_cannot_carry_a_plateau_on_its_own():
    # The defect a pilot run exposed BEFORE the full-record run, and the reason the plateau rule was
    # revised from a quorum to "every measurable cell": the first-difference basis clamps at exactly
    # 0.0 at every ring, so its increment is 0 by construction. Under a quorum those two censored
    # cells declared a plateau while the deseason cells were still climbing steeply.
    grid = WideRingSweepConfig().ring_grid
    climbing = [0.00, 0.01, 0.03, 0.07, 0.10, 0.11, 0.27]  # deseason: still rising hard
    rows = []
    for g, s_ in zip(grid, climbing):
        rows.append({"inner": g[0], "outer": g[1], "basis": "deseason", "share": s_, "lo": 0.01})
        rows.append({"inner": g[0], "outer": g[1], "basis": "first-diff", "share": 0.0, "lo": 0.0})
    verdict, _, detail = R.classify_wide_verdict(_frame(rows))
    assert detail["plateau_cells"] == 1  # the clamped cell alone
    assert detail["cells_with_two_rings"] == 2
    assert detail["plateau"] is False
    assert verdict == R.VERDICT_DOMAIN_LIMITED


def test_at5_rules_disagree_is_a_first_class_outcome():
    # High everywhere and flat: Rule A (level) fires, Rule B (rise) does not. Neither is chosen.
    df = _ladder([0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30], lo=0.20)
    verdict, reason, detail = R.classify_wide_verdict(df)
    assert verdict == R.VERDICT_RULES_DISAGREE
    assert detail["rule_a"] and not detail["rule_b"]
    assert "neither is selected after the fact" in reason


def test_at5_outermost_is_keyed_on_the_ring_pair_not_the_inner_radius():
    # The NOX-015 pooling bug: (60,100) and (60,120) share an inner radius. Keying on it alone would
    # pool two rings and double the cell count the rule counts against min_cells_agreeing.
    df = _ladder([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.20], lo=0.01)
    plain = df[(df["weighting"] == "plain") & (~df["refused"])]
    outer = R._outermost(plain)
    assert set(outer["ring_outer_km"]) == {150.0}
    assert len(outer) == 2  # two bases at ONE ring, not four rows pooled from two rings
    _, _, detail = R.classify_wide_verdict(df)
    assert detail["n_outer_cells"] == 2


def test_at5_thresholds_come_from_the_config_not_from_literals():
    df = _ladder([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.20], lo=0.01)
    strict = dataclasses.replace(WideRingSweepConfig(), rise_threshold=0.5, plateau_threshold=1.0)
    verdict, _, detail = R.classify_wide_verdict(df, strict)
    assert detail["rise_threshold"] == 0.5
    assert not detail["rule_b"]  # a rise of 0.19 no longer clears the raised bar
    assert verdict in (R.VERDICT_RULES_DISAGREE, R.VERDICT_ROBUST)


def test_at5_empty_and_all_refused_frames_are_inconclusive():
    assert R.classify_wide_verdict(pd.DataFrame())[0] == R.VERDICT_INCONCLUSIVE
    df = _ladder([0.01] * 7)
    df["refused"] = True
    assert R.classify_wide_verdict(df)[0] == R.VERDICT_INCONCLUSIVE


# ------------------------------------------------------------------- AT7  controls on Rule B


@pytest.mark.parametrize(
    "plume_gain,expect_rule_b",
    [(2.0, True), (0.0, False)],
    ids=["positive-in-ring-plume", "negative-footprint-confined"],
)
def test_at7_rule_b_fires_only_for_a_genuinely_contaminated_ring(
    facilities_csv, tmp_path, plume_gain, expect_rule_b
):
    # The load-bearing pair. A plume planted INSIDE the 25-60 km ring makes that ring a partial copy
    # of the source, so the share must rise once the ring moves out (Rule B fires). A source confined
    # to the footprint is evidence AGAINST self-subtraction, and Rule B must stay silent -- the case
    # the NOX-015 LEVEL rule got wrong, which is why the rise formulation is pre-registered here.
    # One resolution in the fixture, so the quorum is set to one cell: the control tests whether
    # Rule B RESPONDS to a contaminated ring, not the (basis x resolution) quorum, which AT5 covers.
    # Note which basis moves -- the deseason basis recovers the planted source and the
    # first-difference basis does not, the same split the real data shows.
    cfg = _cfg(facilities_csv, tmp_path, min_cells_agreeing=1)
    cube = _wide_cube(plume_gain=plume_gain)
    rows = []
    for inner, outer in ((25.0, 60.0), (90.0, 150.0)):
        sh = _shares(cube, cfg, (inner, outer))
        for basis in ("deseason", "first-diff"):
            rows.append(
                {
                    "inner": inner,
                    "outer": outer,
                    "basis": basis,
                    "share": float(sh.loc[basis, "share_steel_local"]),
                    "lo": float(sh.loc[basis, "share_lo"]),
                    "regional": float(sh.loc[basis, "share_regional"]),
                }
            )
    _, _, detail = R.classify_wide_verdict(_frame(rows), cfg)
    assert detail["rule_b"] is expect_rule_b, detail["rises"]


# ------------------------------------------------------------------- AT-REG-1  no regression


def test_at_reg1_narrow_sweep_defaults_are_untouched():
    narrow = RingSweepConfig()
    assert narrow.ring_grid == ((25.0, 60.0), (40.0, 80.0), (50.0, 90.0), (60.0, 100.0), (60.0, 120.0))
    assert narrow.results_name == "ring_sensitivity.csv"
    assert not hasattr(narrow, "aoi_buffer_deg")
    assert not hasattr(narrow, "rise_threshold")  # the narrow verdict rule is unchanged


def test_at_reg1_wide_run_writes_to_its_own_artefacts():
    wide = WideRingSweepConfig()
    assert wide.results_name != RingSweepConfig().results_name
    assert wide.findings_name != RingSweepConfig().findings_name
    assert wide.cube_path_weekly != RingSweepConfig().cube_path_weekly


def test_at_reg1_level_series_default_path_still_takes_no_aoi():
    # NFR-003: the aoi hook must be opt-in, or the committed NOX-009 numbers would move.
    import inspect

    from noxus.signal.variance import _level_series_for_cube

    sig = inspect.signature(_level_series_for_cube)
    assert sig.parameters["aoi"].default is None
    assert sig.parameters["ring"].default is None


# ------------------------------------------------------------------- AT-ERR-1  missing input


def test_at_err1_missing_wide_cube_names_the_command_that_builds_it(tmp_path):
    cfg = dataclasses.replace(
        WideRingSweepConfig(), cube_path_weekly=tmp_path / "absent.nc", out_dir=tmp_path
    )
    with pytest.raises(FileNotFoundError, match="noxus grid"):
        R.sweep_rings(cfg)
