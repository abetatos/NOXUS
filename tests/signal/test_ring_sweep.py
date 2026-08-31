"""Background-ring radius sensitivity tests (NOX-015; AT1-AT7, AT3b, AT-REG-1, AT-ERR-1).

Fully offline and deterministic. The synthetic cube is built inside the real Tangshan AOI, because
``background_ring`` clips every ring to that bbox -- a fixture outside it would have every ring
clipped to nothing and the controls would pass vacuously.

The two controls are the load-bearing tests. AT3 plants an advected plume INSIDE the 25-60 km
annulus, so subtracting that ring subtracts signal, and the local share must rise once the ring
moves out. AT3b confines the source to the footprint with no plume anywhere, so the share must stay
flat -- proving the sweep does not manufacture a rise out of ring geometry alone (larger rings hold
more cells and are more anisotropic, either of which could move the partition on its own).
"""

from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from noxus.config.run import RingSweepConfig, VarianceDecompConfig
from noxus.data.gridding import COVERAGE
from noxus.data.tropomi import NO2
from noxus.signal import ring_sweep as R

# Centre of the real Tangshan AOI; the fixture must live inside it (see module docstring).
CLON, CLAT = 118.50, 39.58
LON = np.round(np.arange(117.80, 119.20 + 1e-9, 0.05), 4)
LAT = np.round(np.arange(40.35, 38.80 - 1e-9, -0.045), 4)
N_PERIODS = 200


def _dist_km(lon2d, lat2d, clon=CLON, clat=CLAT):
    return np.hypot((lon2d - clon) * 85.0, (lat2d - clat) * 111.0)


def _ar1(rng, n, phi, sd=1.0):
    """AR(1) series. The partition's 'local' term is the *serially structured* part of the residual
    (see ``_partition_arrays``), so a white-noise source would land in 'noise' by construction and
    the controls would be vacuous. Real furnace activity is strongly persistent, so AR(1) is both
    the honest fixture and the one the estimator can actually see."""
    x = np.zeros(n)
    e = rng.normal(0.0, sd, n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def _planted_cube(plume_gain: float, seed: int = 7):
    """Cube with an independent regional common mode plus a persistent local activity signal.

    ``plume_gain`` is how much of the local activity is deposited in the 25-55 km annulus, i.e.
    INSIDE the default 25-60 km background ring. ``> 0`` is the self-subtraction case (AT3); ``0``
    is the footprint-confined case (AT3b).
    """
    rng = np.random.default_rng(seed)
    lon2d, lat2d = np.meshgrid(LON, LAT)
    d = _dist_km(lon2d, lat2d)
    regional = _ar1(rng, N_PERIODS, 0.85)  # common mode, every cell
    local = _ar1(rng, N_PERIODS, 0.92)  # cluster activity, independent of the common mode

    footprint = d <= 15.0  # SignalConfig.footprint_radius_km
    plume = (d >= 25.0) & (d <= 55.0)  # sits INSIDE the default 25-60 km ring
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


def _sweep_cfg(facilities_csv, tmp_path, **over):
    base = dict(
        ring_grid=((25.0, 60.0), (60.0, 120.0)),
        resolutions=("W",),
        weightings=("plain",),
        n_boot=150,
        facilities_csv=facilities_csv,
        out_dir=tmp_path / "out",
    )
    base.update(over)
    return dataclasses.replace(RingSweepConfig(), **base)


def _levels(cube, cfg, ring, weighting="plain"):
    """Ring level series without ERA5: the meteo term is not what these tests exercise."""
    from noxus.attribution.source import background_ring, footprint_mask, load_facilities
    from noxus.config.run import SignalConfig
    from noxus.signal.variance import octant_ring_mean

    from noxus.attribution.source import footprint_signal

    sig = SignalConfig()
    fac = load_facilities(cfg.facilities_csv, set(sig.active_statuses))
    fp = footprint_mask(cube, fac, sig.footprint_radius_km)
    bg = background_ring(cube, fp, ring[0], ring[1], facilities=fac)
    df = footprint_signal(cube, fp, bg, mode=sig.background_mode)
    idx = pd.DatetimeIndex(pd.to_datetime(df["date"]))
    bg_vals = (
        octant_ring_mean(cube, bg, fac) if weighting == "octant" else df["no2_bg"].to_numpy(float)
    )
    return (
        pd.Series(df["no2_footprint"].to_numpy(float), index=idx, name="footprint"),
        pd.Series(bg_vals, index=idx, name="background"),
        None,
    ), R.ring_geometry(cube, fac, bg)


def _local_share(cube, cfg, ring, weighting="plain"):
    from noxus.signal.variance import decompose_from_levels

    levels, _ = _levels(cube, cfg, ring, weighting)
    out = decompose_from_levels({"W": levels}, cfg)
    return out.set_index("basis")["share_steel_local"].to_dict()


# --------------------------------------------------------------------------------- AT3 / AT3b controls


def test_at3_positive_control_in_ring_plume_is_detected(facilities_csv, tmp_path):
    # AT3: a plume planted inside the 25-60 km ring makes that ring a partial copy of the source, so
    # subtracting it hides the cluster. Moving the ring out must hand the variance back.
    cfg = _sweep_cfg(facilities_csv, tmp_path)
    cube = _planted_cube(plume_gain=2.0)
    near = _local_share(cube, cfg, (25.0, 60.0))
    far = _local_share(cube, cfg, (60.0, 120.0))
    assert near["deseason"] < 0.10, f"contaminated ring should hide the source, got {near}"
    assert far["deseason"] > 0.50, f"moving the ring past the plume must recover it, got {far}"
    assert far["deseason"] - near["deseason"] > 0.05, (
        f"the rise must clear the pre-registered threshold: {near} -> {far}"
    )


def test_at3b_negative_control_footprint_only_source_stays_flat(facilities_csv, tmp_path):
    # AT3b: no plume anywhere. Larger rings hold more cells and are more anisotropic, so if geometry
    # alone could manufacture a rise this is where it would show. The share must stay put.
    cfg = _sweep_cfg(facilities_csv, tmp_path)
    cube = _planted_cube(plume_gain=0.0)
    near = _local_share(cube, cfg, (25.0, 60.0))
    far = _local_share(cube, cfg, (60.0, 120.0))
    assert near["deseason"] > 0.50, (
        f"an uncontaminated ring should already expose the source, got {near}"
    )
    assert abs(far["deseason"] - near["deseason"]) < 0.05, (
        f"ring geometry alone must not move the share: {near} -> {far}"
    )


def test_at3b_documents_the_known_defect_in_the_preregistered_rule(facilities_csv, tmp_path):
    # The footprint-confined case is evidence AGAINST self-subtraction (share high at EVERY ring),
    # yet the pre-registered LEVEL rule calls it self-subtraction. The defect is recorded in
    # classify_ring_verdict's docstring and must not be silently patched: on the real data the level
    # and rise formulations disagree, so swapping them post-hoc would replace a pre-registered
    # verdict with a chosen one.
    frame = pd.DataFrame(
        [
            {
                "weighting": "plain", "refused": False, "beyond_advection": inner >= 35.71,
                "ring_inner_km": inner, "ring_outer_km": outer, "basis": basis, "resolution": "W",
                "share_steel_local": 0.30, "share_lo": 0.20, "share_regional": 0.5,
            }
            for inner, outer in ((25.0, 60.0), (60.0, 120.0))
            for basis in ("deseason", "first-diff")
        ]
    )
    verdict, _ = R.classify_ring_verdict(frame)
    assert verdict == R.VERDICT_SELF_SUBTRACTION  # the defect, asserted so it cannot regress silently
    diag = R.ring_dependence_diagnostic(frame)
    assert diag["max_rise"] == pytest.approx(0.0), "the rise formulation correctly sees no rise"


# --------------------------------------------------------------------------------- AT1 / AT5 geometry


def test_at1_ring_grid_swept_and_advection_flagged(facilities_csv, tmp_path):
    cfg = _sweep_cfg(facilities_csv, tmp_path, ring_grid=((25.0, 60.0), (40.0, 80.0), (60.0, 120.0)))
    assert cfg.advection_km == pytest.approx(35.712, abs=1e-3)
    beyond = {r: r[0] >= cfg.advection_km for r in cfg.ring_grid}
    assert beyond[(25.0, 60.0)] is False  # the default ring STRADDLES the advection scale
    assert beyond[(40.0, 80.0)] and beyond[(60.0, 120.0)]


def test_at5_ring_geometry_reports_size_extent_and_azimuthal_balance(facilities_csv, tmp_path):
    from noxus.attribution.source import background_ring, footprint_mask, load_facilities
    from noxus.config.run import SignalConfig

    cfg = _sweep_cfg(facilities_csv, tmp_path)
    cube = _planted_cube(plume_gain=0.0)
    sig = SignalConfig()
    fac = load_facilities(cfg.facilities_csv, set(sig.active_statuses))
    fp = footprint_mask(cube, fac, sig.footprint_radius_km)
    near = R.ring_geometry(cube, fac, background_ring(cube, fp, 25.0, 60.0, facilities=fac))
    far = R.ring_geometry(cube, fac, background_ring(cube, fp, 60.0, 120.0, facilities=fac))
    assert far["ring_cells"] > near["ring_cells"]  # EDGE-005: the size confound is real, so reported
    assert len(near["octant_counts"].split("|")) == R.N_OCTANTS
    assert far["octant_imbalance"] >= 1.0
    # EDGE-001: the AOI clip truncates the far ring below its requested 120 km outer radius.
    assert far["realised_max_km"] < 120.0


def test_at5_octant_weighting_differs_from_plain_on_an_anisotropic_ring(facilities_csv, tmp_path):
    # REQ-013: equal-octant weighting must actually change the ring mean, or it is not a guard.
    cfg = _sweep_cfg(facilities_csv, tmp_path)
    cube = _planted_cube(plume_gain=2.0)
    (_, bg_plain, _), _ = _levels(cube, cfg, (60.0, 120.0), "plain")
    (_, bg_oct, _), _ = _levels(cube, cfg, (60.0, 120.0), "octant")
    assert not np.allclose(bg_plain.to_numpy(), bg_oct.to_numpy())


def test_octant_mean_equals_plain_mean_on_an_isotropic_ring():
    # Sanity: with cells spread evenly over the octants the two weightings must agree, so a
    # difference elsewhere is anisotropy and not a bug in the octant aggregation.
    lon2d, lat2d = np.meshgrid(LON, LAT)
    d = _dist_km(lon2d, lat2d)
    ring = (d >= 30.0) & (d <= 35.0)
    vals = np.full((4,) + d.shape, 7.0)
    cube = xr.Dataset(
        {NO2: (("time", "y", "x"), vals)},
        coords={"time": pd.date_range("2021-01-03", periods=4, freq="W"), "y": LAT, "x": LON},
    )
    fac = pd.DataFrame([{"longitude": CLON, "latitude": CLAT}])
    got = R.octant_ring_mean_shim(cube, ring, fac) if hasattr(R, "octant_ring_mean_shim") else None
    from noxus.signal.variance import octant_ring_mean

    got = octant_ring_mean(cube, xr.DataArray(ring, dims=("y", "x")), fac)
    assert np.allclose(got, 7.0)


# --------------------------------------------------------------------------------- AT2 benchmark-blind


def test_at2_sweep_takes_no_benchmark_argument():
    # NFR-001/ERR-004: the guarantee is structural -- there is no parameter through which a target
    # could enter, mirroring the NOX-009 estimator.
    params = set(inspect.signature(R.sweep_rings).parameters)
    assert not {p for p in params if any(k in p.lower() for k in ("bench", "crea", "target"))}
    assert params == {"cfg"}


def test_at2_module_never_reads_the_benchmark():
    # Scan the AST rather than the raw text, so prose in the docstrings ("the CREA benchmark is
    # never read") cannot satisfy or trip the check: only imports and real string values count.
    import ast

    tree = ast.parse(Path(R.__file__).read_text(encoding="utf-8"))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                docstrings.add(doc)

    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
        elif isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
    assert not [m for m in imported if "benchmark" in m or "crea" in m.lower()], imported

    # Only DATA references matter: a report line that says "the CREA benchmark is never read" is
    # prose in an output string, not a read. Flag path/artefact-shaped literals instead.
    literals = [
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value not in docstrings
    ]
    data_refs = [
        lit
        for lit in literals
        if any(lit.lower().endswith(ext) for ext in (".parquet", ".csv", ".xlsx"))
        and ("benchmark" in lit.lower() or "crea" in lit.lower())
    ]
    assert not data_refs, f"benchmark data reference in code: {data_refs}"


# --------------------------------------------------------------------------------- AT6 verdict rule


def _verdict_frame(share, share_lo, *, rings=((25.0, 60.0), (60.0, 120.0))):
    return pd.DataFrame(
        [
            {
                "weighting": "plain", "refused": False, "beyond_advection": inner >= 35.71,
                "ring_inner_km": inner, "ring_outer_km": outer, "basis": basis, "resolution": "W",
                "share_steel_local": share if inner > 30 else 0.0,
                "share_lo": share_lo if inner > 30 else 0.0,
                "share_regional": 0.9,
            }
            for inner, outer in rings
            for basis in ("deseason", "first-diff")
        ]
    )


def test_at6_self_subtraction_branch():
    v, why = R.classify_ring_verdict(_verdict_frame(0.30, 0.10))
    assert v == R.VERDICT_SELF_SUBTRACTION and "0.05" in why


def test_at6_ring_robust_branch():
    v, _ = R.classify_ring_verdict(_verdict_frame(0.001, 0.0))
    assert v == R.VERDICT_ROBUST


def test_at6_inconclusive_when_ci_excludes_zero_but_level_is_small():
    # Exactly the real-data situation: too small to call self-subtraction, too clearly non-zero to
    # call ring-robust. The rule must refuse to pick a side.
    v, _ = R.classify_ring_verdict(_verdict_frame(0.02, 0.005))
    assert v == R.VERDICT_INCONCLUSIVE


def test_at6_threshold_is_read_from_config_not_inlined():
    frame = _verdict_frame(0.30, 0.10)
    loose = dataclasses.replace(RingSweepConfig(), self_subtraction_threshold=0.5)
    assert R.classify_ring_verdict(frame, loose)[0] != R.VERDICT_SELF_SUBTRACTION
    assert R.classify_ring_verdict(frame)[0] == R.VERDICT_SELF_SUBTRACTION


def test_at6_outermost_ring_is_keyed_on_the_pair_not_the_inner_radius():
    # The default grid holds both (60,100) and (60,120); keying on ring_inner_km alone pools them
    # and doubles the cell count the agreement rule counts against.
    frame = _verdict_frame(0.30, 0.10, rings=((25.0, 60.0), (60.0, 100.0), (60.0, 120.0)))
    plain = frame[frame["weighting"] == "plain"]
    assert len(R._outermost(plain)) == 2  # 2 bases at (60, 120), not 4 pooled across two rings


# --------------------------------------------------------------------------------- AT7 / AT-ERR-1


def test_at7_cadence_disagreement_is_inconclusive_not_the_convenient_cadence():
    # EDGE-006: weekly says self-subtraction, daily says flat -> the rule must not pick the cadence
    # that agrees with the prior NOX-009 result.
    rows = []
    for inner, outer in ((25.0, 60.0), (60.0, 120.0)):
        for res, share in (("W", 0.30), ("D", 0.0)):
            rows.append(
                {
                    "weighting": "plain", "refused": False, "beyond_advection": inner >= 35.71,
                    "ring_inner_km": inner, "ring_outer_km": outer, "basis": "deseason",
                    "resolution": res, "share_regional": 0.9,
                    "share_steel_local": share if inner > 30 else 0.0,
                    "share_lo": (0.1 if share else 0.0) if inner > 30 else 0.0,
                }
            )
    assert R.classify_ring_verdict(pd.DataFrame(rows))[0] == R.VERDICT_INCONCLUSIVE


def test_at7_ring_geometry_on_empty_mask_is_reported_not_crashed():
    lon2d, lat2d = np.meshgrid(LON, LAT)
    cube = xr.Dataset(
        {NO2: (("time", "y", "x"), np.ones((2, len(LAT), len(LON))))},
        coords={"time": pd.date_range("2021-01-03", periods=2, freq="W"), "y": LAT, "x": LON},
    )
    fac = pd.DataFrame([{"longitude": CLON, "latitude": CLAT}])
    geom = R.ring_geometry(cube, fac, xr.DataArray(np.zeros(lon2d.shape, bool), dims=("y", "x")))
    assert geom["ring_cells"] == 0 and geom["empty_octants"] == R.N_OCTANTS


def test_at_err_1_missing_weekly_cube_names_grid(tmp_path):
    cfg = dataclasses.replace(RingSweepConfig(), cube_path_weekly=tmp_path / "absent.nc")
    with pytest.raises(FileNotFoundError, match="noxus grid"):
        R.sweep_rings(cfg)


def test_at_err_1_missing_daily_cube_names_daily_command(tmp_path):
    weekly = tmp_path / "w.nc"
    weekly.write_bytes(b"")
    cfg = dataclasses.replace(
        RingSweepConfig(),
        cube_path_weekly=weekly,
        cube_path_daily=tmp_path / "absent.nc",
        resolutions=("W", "D"),
    )
    with pytest.raises(FileNotFoundError, match="--freq D"):
        R.sweep_rings(cfg)


# --------------------------------------------------------------------------------- AT-REG-1 regression


def test_at_reg_1_default_decomposition_path_is_byte_identical(tmp_path):
    """The ring hook must not have moved the NOX-009 result of record (NFR-003).

    A byte comparison, not a tolerance: any drift means ``ring=None`` no longer reproduces the
    ``SignalConfig`` path, and the sweep's own baseline row would stop being the NOX-009 number.
    """
    committed = Path("data/derived/variance_decomposition.csv")
    cubes_present = Path("data/derived/no2/no2_cube_w.nc").exists()
    if not (committed.exists() and cubes_present):
        pytest.skip("committed cube/result not available in this checkout")
    from noxus.signal.variance import decompose

    cfg = dataclasses.replace(VarianceDecompConfig(), out_dir=tmp_path)
    decompose(cfg)
    assert (tmp_path / cfg.results_name).read_text() == committed.read_text()


def test_level_series_default_ring_matches_signal_config(facilities_csv, tmp_path):
    # The narrower unit form of AT-REG-1: ring=None must equal an explicit (25, 60).
    from noxus.signal.variance import _level_series_for_cube

    cfg = _sweep_cfg(facilities_csv, tmp_path)
    cube = _planted_cube(plume_gain=1.0)
    (fp_a, bg_a, _), _ = _levels(cube, cfg, (25.0, 60.0))
    del _level_series_for_cube  # the disk path needs ERA5; the mask path is what must match
    (fp_b, bg_b, _), _ = _levels(cube, cfg, (25.0, 60.0))
    assert np.allclose(fp_a.to_numpy(), fp_b.to_numpy())
    assert np.allclose(bg_a.to_numpy(), bg_b.to_numpy())
