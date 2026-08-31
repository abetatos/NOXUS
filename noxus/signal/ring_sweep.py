"""Background-ring radius sensitivity of the change-variance decomposition (NOX-015).

NOX-009 reported a steel-local change-variance share of ~0, measured against a **fixed** 25-60 km
background ring. That number is load-bearing for the project's central claim, and it has an
alternative explanation the original design cannot exclude.

The NOX-005 flux probe measures a mean effective wind of ``|U| = 2.48 m/s`` over the AOI under an
assumed NO2 lifetime ``tau = 4 h``. The implied advection length is ``|U| * tau ~ 36 km`` -- which
lands in the *middle* of the 25-60 km ring. If a material part of the cluster's emitted NO2 sits in
that ring as advected plume, subtracting the ring subtracts signal along with background; and
because the partition enters terms in the fixed order ``regional -> meteo -> local``, the variance
shared between ring and cluster is booked to "regional" **by construction**. A steel-local share of
~0 is therefore exactly what ring self-subtraction would also produce.

This module moves the ring past the advection scale and re-runs the **same** partition. Nothing in
the estimator changes: ``background_ring``, ``footprint_mask``, ``partition_change_variance`` and
``decompose_from_levels`` are reused unmodified, and only the ring geometry varies. The outcome is
decided by a rule fixed in :class:`~noxus.config.run.RingSweepConfig` before the run (REQ-031):

* ``ring-robust``     -- the share stays ~0 at every ring beyond the advection scale. Physics.
* ``self-subtraction`` -- the share rises materially once the ring clears the plume. Artefact.
* ``inconclusive``    -- neither pattern holds cleanly (includes cadence disagreement).

Like the decomposition it audits, this module takes **no benchmark argument** and never reads the
CREA series (NFR-001, ERR-004). It writes its own artefacts and never overwrites
``variance_decomposition.csv`` (the NOX-009 result of record).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

VERDICT_ROBUST = "ring-robust"
VERDICT_SELF_SUBTRACTION = "self-subtraction"
VERDICT_INCONCLUSIVE = "inconclusive"
# NOX-017 wide-sweep outcomes. "domain-limited" is the verdict NOX-015 needed and did not have: a
# rise still climbing when the cube runs out cannot be read either way. "rules-disagree" is a
# first-class outcome, never resolved by choosing a rule after seeing the numbers.
VERDICT_DOMAIN_LIMITED = "domain-limited"
VERDICT_RULES_DISAGREE = "rules-disagree"

N_OCTANTS = 8


# --------------------------------------------------------------------------- ring geometry (REQ-003/013/022)


def ring_geometry(cube, facilities, ring_mask) -> dict:
    """Auditable geometry of a realised ring: extent, size and azimuthal balance.

    A larger ring is not only further away -- it also holds more cells and, because the AOI is a
    rectangle rather than a disc, it is azimuthally lop-sided. Both could move the partition for
    reasons that have nothing to do with the plume, so both are recorded per row (EDGE-002/005) and
    the octant counts feed the equal-octant weighting (REQ-013).

    ``realised_max_km`` is the furthest cell actually retained, which is < the requested outer radius
    whenever the AOI clip truncates the ring (EDGE-001).
    """
    from noxus.attribution.source import _cell_centres, _haversine_km

    lon2d, lat2d, _, _ = _cell_centres(cube)
    clon = float(facilities["longitude"].mean())
    clat = float(facilities["latitude"].mean())
    sel = np.asarray(ring_mask.values if hasattr(ring_mask, "values") else ring_mask, dtype=bool)

    n_cells = int(sel.sum())
    if n_cells == 0:
        return {
            "ring_cells": 0,
            "realised_min_km": float("nan"),
            "realised_max_km": float("nan"),
            "octant_counts": "",
            "octant_imbalance": float("nan"),
            "empty_octants": N_OCTANTS,
        }

    dist = _haversine_km(lon2d, lat2d, clon, clat)
    az = np.degrees(np.arctan2((lon2d - clon) * np.cos(np.radians(clat)), lat2d - clat)) % 360.0
    counts = [int(((az >= k * 45.0) & (az < (k + 1) * 45.0) & sel).sum()) for k in range(N_OCTANTS)]
    nonzero = [c for c in counts if c > 0]
    return {
        "ring_cells": n_cells,
        "realised_min_km": round(float(dist[sel].min()), 2),
        "realised_max_km": round(float(dist[sel].max()), 2),
        "octant_counts": "|".join(str(c) for c in counts),
        "octant_imbalance": round(max(nonzero) / min(nonzero), 3) if nonzero else float("nan"),
        "empty_octants": int(sum(1 for c in counts if c == 0)),
    }


def sweep_aoi(cfg):
    """The AOI box a sweep clips its rings to (NOX-017 REQ-010).

    ``cfg.aoi_buffer_deg`` is absent on :class:`~noxus.config.run.RingSweepConfig` and ``None`` by
    default on the wide variant, in which case ``background_ring`` falls back to the committed
    0.25-deg ``TANGSHAN`` box -- the NOX-015 behaviour, unchanged.
    """
    from noxus.config.region import tangshan_aoi

    buffer_deg = getattr(cfg, "aoi_buffer_deg", None)
    return None if buffer_deg is None else tangshan_aoi(buffer_deg)


def _levels_for_ring(cube, cfg, resolution, ring, weighting, aoi=None):
    """Footprint/background/meteo level series for one ring, plus that ring's geometry.

    Delegates entirely to the NOX-009 series builder (with its NOX-015 ring hook) so the sweep can
    never diverge from the decomposition it is auditing.

    ``aoi`` (NOX-017 REQ-010) is the box the ring is clipped to. ``None`` keeps the committed 0.25-deg
    default; a wide-cube sweep must pass its own AOI or every ring cell beyond the narrow box is
    discarded and the wide sweep silently reproduces the narrow one.
    """
    from noxus.attribution.source import background_ring, footprint_mask, load_facilities
    from noxus.config.run import SignalConfig
    from noxus.signal.variance import _level_series_for_cube

    sig = SignalConfig()
    facilities = load_facilities(cfg.facilities_csv, set(sig.active_statuses))
    fp_mask = footprint_mask(cube, facilities, sig.footprint_radius_km)
    bg_mask = background_ring(cube, fp_mask, ring[0], ring[1], facilities=facilities, aoi=aoi)
    geom = ring_geometry(cube, facilities, bg_mask)
    levels = _level_series_for_cube(
        cube, cfg, resolution, ring=ring, ring_weighting=weighting, aoi=aoi
    )
    return levels, geom


# --------------------------------------------------------------------------- verdict rule (REQ-030/031)


def _outermost(plain: pd.DataFrame) -> pd.DataFrame:
    """Rows of the single outermost ring.

    Selected on the (inner, outer) PAIR, not on ``ring_inner_km`` alone: the default grid contains
    both (60, 100) and (60, 120), so keying on the inner radius alone would silently pool two rings
    and double the cell count the verdict rule counts against ``min_cells_agreeing``.
    """
    inner = plain["ring_inner_km"].max()
    at_inner = plain[plain["ring_inner_km"] == inner]
    return at_inner[at_inner["ring_outer_km"] == at_inner["ring_outer_km"].max()]


def _innermost(plain: pd.DataFrame) -> pd.DataFrame:
    """Rows of the single innermost (baseline, 25-60 km) ring; same pairing caveat as above."""
    inner = plain["ring_inner_km"].min()
    at_inner = plain[plain["ring_inner_km"] == inner]
    return at_inner[at_inner["ring_outer_km"] == at_inner["ring_outer_km"].min()]


def _baseline_shares(plain: pd.DataFrame) -> dict:
    """Cluster-local share at the innermost (baseline, 25-60 km) ring, keyed by (basis, resolution)."""
    return _innermost(plain).set_index(["basis", "resolution"])["share_steel_local"].to_dict()


def classify_ring_verdict(df: pd.DataFrame, cfg=None) -> tuple[str, str]:
    """The **pre-registered** REQ-031 rule. This is the verdict of record.

    Fixed in ``RingSweepConfig`` before the run and applied unchanged afterwards: at the outermost
    ring, on the plain weighting, ``self-subtraction`` when the clamped cluster-local share exceeds
    ``self_subtraction_threshold`` with a bootstrap CI excluding zero in at least
    ``min_cells_agreeing`` of the (basis x resolution) cells; ``ring-robust`` when every cell beyond
    the advection scale stays at or below the threshold with a CI containing zero; otherwise
    ``inconclusive``.

    A pure function of ``df`` and ``cfg``: it reads no benchmark quantity, so the verdict cannot
    drift with how the result looks.

    KNOWN DEFECT, recorded rather than patched (see :func:`ring_dependence_diagnostic`): this rule
    thresholds the share's *level*, so a source that is genuinely footprint-confined -- high share at
    *every* ring, which is evidence AGAINST self-subtraction -- would be misclassified as
    self-subtraction. The acceptance-test negative control (AT3b) demonstrates this. C1 asks whether
    the share *rises* once the ring clears the plume, so a future pre-registration should threshold
    the rise. It is NOT retro-fitted here: on this data the two formulations disagree, so changing
    the rule after seeing the result would substitute a post-hoc verdict for a pre-registered one.
    """
    from noxus.config.run import RingSweepConfig

    cfg = cfg or RingSweepConfig()
    thr = cfg.self_subtraction_threshold
    if df.empty:
        return VERDICT_INCONCLUSIVE, "empty sweep frame"

    plain = df[(df["weighting"] == "plain") & (~df["refused"])]
    if plain.empty:
        return VERDICT_INCONCLUSIVE, "every ring refused"

    outer = _outermost(plain)
    beyond = plain[plain["beyond_advection"]]
    if outer.empty or beyond.empty:
        return VERDICT_INCONCLUSIVE, "no ring beyond the advection scale completed"

    risen = outer[(outer["share_steel_local"] > thr) & (outer["share_lo"] > 0.0)]
    n_rise, n_outer = int(len(risen)), int(len(outer))
    if n_rise >= cfg.min_cells_agreeing:
        return (
            VERDICT_SELF_SUBTRACTION,
            f"{n_rise}/{n_outer} cells at the outermost ring exceed the pre-registered threshold "
            f"{thr} with a bootstrap CI excluding zero",
        )

    flat = beyond[(beyond["share_steel_local"] <= thr) & (beyond["share_lo"] <= 0.0)]
    if len(flat) == len(beyond):
        return (
            VERDICT_ROBUST,
            f"all {len(beyond)} cells beyond the advection scale stay at or below {thr} with a "
            "bootstrap CI containing zero",
        )
    ci_excl = int((beyond["share_lo"] > 0.0).sum())
    return (
        VERDICT_INCONCLUSIVE,
        f"{n_rise}/{n_outer} outermost cells clear the {thr} level threshold "
        f"({cfg.min_cells_agreeing} needed), so self-subtraction is not established; but "
        f"{ci_excl}/{len(beyond)} beyond-advection cells carry a CI EXCLUDING zero, so the share is "
        "not indistinguishable from zero either and ring-robustness is not established",
    )


def ring_dependence_diagnostic(df: pd.DataFrame, cfg=None) -> dict:
    """POST-HOC diagnostic: how much does the local share *rise* as the ring moves out? (REQ-020/021)

    Reported alongside, never instead of, :func:`classify_ring_verdict`. This quantity is what
    critique C1 actually asks about, and the acceptance-test controls key on it -- but it was not
    the pre-registered decision rule, and on this data the two disagree. Labelling it post-hoc keeps
    the pre-registration meaningful; a future task should pre-register the rise formulation and
    re-run.

    Returns per-cell rises from the baseline ring, the largest rise, how many cells carry a CI
    excluding zero at the outermost ring, and whether the regional share falls as the local share
    rises -- the transfer signature self-subtraction predicts (REQ-021).
    """
    from noxus.config.run import RingSweepConfig

    cfg = cfg or RingSweepConfig()
    plain = df[(df["weighting"] == "plain") & (~df["refused"])] if not df.empty else df
    if plain.empty:
        return {"rises": {}, "max_rise": float("nan"), "n_ci_excludes_zero": 0, "monotone_cells": 0}

    base_local = _baseline_shares(plain)
    inner_row, outer_row = _innermost(plain), _outermost(plain)
    base_reg = inner_row.set_index(["basis", "resolution"])["share_regional"].to_dict()
    outer = outer_row

    rises, monotone, transfers = {}, 0, 0
    for (basis, res), grp in plain.groupby(["basis", "resolution"]):
        grp = grp.sort_values("ring_inner_km")
        key = f"{basis}.{res}"
        rises[key] = round(
            float(grp["share_steel_local"].iloc[-1]) - float(base_local.get((basis, res), 0.0)), 6
        )
        if grp["share_steel_local"].is_monotonic_increasing and rises[key] > 0:
            monotone += 1
        reg_fall = float(base_reg.get((basis, res), 0.0)) - float(grp["share_regional"].iloc[-1])
        if rises[key] > 0 and reg_fall > 0:
            transfers += 1
    return {
        "rises": rises,
        "max_rise": round(max(rises.values()), 6) if rises else float("nan"),
        "n_ci_excludes_zero": int((outer["share_lo"] > 0.0).sum()),
        "n_outer_cells": int(len(outer)),
        "monotone_cells": monotone,
        "transfer_cells": transfers,
        "baseline_ring": f"{inner_row['ring_inner_km'].iloc[0]:.0f}-{inner_row['ring_outer_km'].iloc[0]:.0f} km",
        "outer_ring": f"{outer_row['ring_inner_km'].iloc[0]:.0f}-{outer_row['ring_outer_km'].iloc[0]:.0f} km",
    }


def _cell_series(plain: pd.DataFrame) -> dict:
    """Local share per (basis, resolution), ordered by ring, keyed on the (inner, outer) pair."""
    out = {}
    for key, grp in plain.groupby(["basis", "resolution"]):
        grp = grp.sort_values(["ring_inner_km", "ring_outer_km"])
        out[key] = grp
    return out


def classify_wide_verdict(df: pd.DataFrame, cfg=None) -> tuple[str, str, dict]:
    """The **pre-registered** NOX-017 rule (REQ-020..023). This is the wide sweep's verdict of record.

    Three criteria, all fixed in :class:`~noxus.config.run.WideRingSweepConfig` before the wide cube
    was acquired:

    * **Rule A (level)** -- carried over from NOX-015 verbatim, so the two runs stay comparable:
      clamped local share above ``self_subtraction_threshold`` with a bootstrap CI excluding zero, in
      at least ``min_cells_agreeing`` cells at the outermost ring.
    * **Rule B (rise)** -- the corrected formulation. ``share(outermost) - share(baseline)`` above
      ``rise_threshold`` in at least ``min_cells_agreeing`` cells, with the outermost CI excluding
      zero. This thresholds what self-subtraction actually predicts; Rule A thresholds the level,
      which is the defect recorded in the NOX-015 review.
    * **Plateau** -- **every** cell whose increment between the two outermost rings is measurable
      moves by less than ``plateau_threshold``. This is the whole reason a wider cube was acquired: a
      rise that has not flattened by the edge of the domain is not evidence for either reading.

      REVISED 2026-08-28, before the full-record run and disclosed as such. As first written this
      was a quorum ("flat in at least ``min_cells_agreeing`` cells"), which a pilot on 8 of the 89
      acquired months showed to be trivially satisfiable: the first-difference basis clamps at
      exactly 0.0 at every ring, so its increment is 0 by construction and two such cells met the
      quorum while the deseason cells were still climbing by +0.16. The quorum reading also
      contradicts the plain meaning of the word -- if any cell is still moving, the share has not
      plateaued. The revision makes ``domain-limited`` MORE reachable, i.e. it moves the verdict away
      from the answer that would flatter the paper, and the full-record verdict had not been computed
      when it was made.

    The verdict is their conjunction, in a fixed order. Rules A and B disagreeing returns
    ``rules-disagree`` and both numbers -- picking one after the fact is precisely the failure this
    task exists to avoid. Returns ``(verdict, reason, detail)``; ``detail`` carries every input to the
    decision so the rule can be re-checked by hand from the CSV.
    """
    from noxus.config.run import WideRingSweepConfig

    cfg = cfg or WideRingSweepConfig()
    detail: dict = {
        "level_threshold": cfg.self_subtraction_threshold,
        "rise_threshold": cfg.rise_threshold,
        "plateau_threshold": cfg.plateau_threshold,
        "min_cells_agreeing": cfg.min_cells_agreeing,
    }
    if df.empty:
        return VERDICT_INCONCLUSIVE, "empty sweep frame", detail

    plain = df[(df["weighting"] == "plain") & (~df["refused"])]
    if plain.empty:
        return VERDICT_INCONCLUSIVE, "every ring refused", detail

    outer = _outermost(plain)
    if outer.empty:
        return VERDICT_INCONCLUSIVE, "the outermost ring completed no cell", detail

    base_local = _baseline_shares(plain)
    detail["baseline_ring"] = (
        f"{_innermost(plain)['ring_inner_km'].iloc[0]:.0f}-"
        f"{_innermost(plain)['ring_outer_km'].iloc[0]:.0f} km"
    )
    detail["outer_ring"] = (
        f"{outer['ring_inner_km'].iloc[0]:.0f}-{outer['ring_outer_km'].iloc[0]:.0f} km"
    )

    # Rule A -- level at the outermost ring (NOX-015, unchanged).
    thr = cfg.self_subtraction_threshold
    a_cells = outer[(outer["share_steel_local"] > thr) & (outer["share_lo"] > 0.0)]
    detail["rule_a_cells"] = int(len(a_cells))
    detail["n_outer_cells"] = int(len(outer))
    rule_a = len(a_cells) >= cfg.min_cells_agreeing

    # Rule B -- rise from the baseline ring, with the outermost CI excluding zero.
    rises, b_cells = {}, 0
    for _, row in outer.iterrows():
        key = (row["basis"], row["resolution"])
        rise = float(row["share_steel_local"]) - float(base_local.get(key, 0.0))
        rises[f"{key[0]}.{key[1]}"] = round(rise, 6)
        if rise > cfg.rise_threshold and float(row["share_lo"]) > 0.0:
            b_cells += 1
    detail["rises"] = rises
    detail["max_rise"] = round(max(rises.values()), 6) if rises else float("nan")
    detail["rule_b_cells"] = b_cells
    rule_b = b_cells >= cfg.min_cells_agreeing

    # Plateau -- last increment, per cell. EVERY measurable cell must be flat: a cell clamped at
    # zero is flat by construction and must not be able to carry a plateau on its own.
    increments, flat_cells, cells_with_two = {}, 0, 0
    for (basis, res), grp in _cell_series(plain).items():
        if len(grp) < 2:
            continue
        cells_with_two += 1
        inc = float(grp["share_steel_local"].iloc[-1]) - float(grp["share_steel_local"].iloc[-2])
        increments[f"{basis}.{res}"] = round(inc, 6)
        if abs(inc) < cfg.plateau_threshold:
            flat_cells += 1
    detail["last_increments"] = increments
    detail["plateau_cells"] = flat_cells
    detail["cells_with_two_rings"] = cells_with_two
    plateau = cells_with_two >= cfg.min_cells_agreeing and flat_cells == cells_with_two
    detail["plateau"] = bool(plateau)
    detail["rule_a"] = bool(rule_a)
    detail["rule_b"] = bool(rule_b)

    n_out = len(outer)
    if rule_a and rule_b:
        return (
            VERDICT_SELF_SUBTRACTION,
            f"both pre-registered rules fire at the outermost ring: {len(a_cells)}/{n_out} cells "
            f"exceed the {thr} level with a CI excluding zero (Rule A), and {b_cells}/{n_out} rise "
            f"by more than {cfg.rise_threshold} from the baseline ring (Rule B)",
            detail,
        )
    if not rule_a and not rule_b and plateau:
        return (
            VERDICT_ROBUST,
            f"neither rule fires ({len(a_cells)}/{n_out} on level, {b_cells}/{n_out} on rise, "
            f"{cfg.min_cells_agreeing} needed) and the share has plateaued (all "
            f"{cells_with_two} measurable cells move by less than {cfg.plateau_threshold} "
            "between the two outermost rings), so the ring radius does not drive the result",
            detail,
        )
    if not plateau:
        return (
            VERDICT_DOMAIN_LIMITED,
            f"the local share has NOT plateaued at the edge of the domain: only "
            f"{flat_cells}/{cells_with_two} cells (all are required) move by less than "
            f"{cfg.plateau_threshold} between "
            f"the two outermost rings (largest last increment "
            f"{max((abs(v) for v in increments.values()), default=float('nan')):+.4f}). A rise still "
            "climbing when the cube runs out is not evidence for either reading, and a wider cube "
            "would be required to settle it",
            detail,
        )
    return (
        VERDICT_RULES_DISAGREE,
        f"the two pre-registered rules disagree: Rule A (level) fires in {len(a_cells)}/{n_out} "
        f"cells, Rule B (rise) in {b_cells}/{n_out}, with {cfg.min_cells_agreeing} needed. Both are "
        "reported; neither is selected after the fact",
        detail,
    )


# --------------------------------------------------------------------------- driver (REQ-010/011/012/040)


def sweep_rings(cfg=None) -> pd.DataFrame:
    """Re-run the NOX-009 decomposition across the configured ring grid (REQ-010/011/012).

    Takes **no benchmark argument** (NFR-001, ERR-004). Returns one tidy row per
    (ring x weighting x basis x resolution); a ring that yields no cells is recorded as ``refused``
    and the sweep continues (REQ-004/ERR-001). Every cell is reported -- none is dropped for
    returning an inconvenient number (REQ-012).
    """
    from noxus.config.run import RingSweepConfig
    from noxus.signal.variance import _latest_era5, _load_cube, decompose_from_levels

    cfg = cfg or RingSweepConfig()

    weekly_path = Path(cfg.cube_path_weekly)
    if not weekly_path.exists():
        raise FileNotFoundError(
            f"Weekly NO2 cube not found: {weekly_path}. Run 'noxus grid' first to build the weekly "
            "cube (ERR-002)."
        )
    daily_path = Path(cfg.cube_path_daily)
    if "D" in cfg.resolutions and not daily_path.exists():
        raise FileNotFoundError(
            f"Daily NO2 cube not found: {daily_path}. Run 'noxus grid --freq D' to build the NOX-006 "
            "daily cube before requesting daily resolution (ERR-002)."
        )
    _latest_era5(Path(cfg.era5_snapshot_dir))  # fail early rather than silently drop the meteo term

    cubes = {r: _load_cube(weekly_path if r == "W" else daily_path) for r in cfg.resolutions}
    advection_km = cfg.advection_km
    aoi = sweep_aoi(cfg)

    rows: list[dict] = []
    for ring in cfg.ring_grid:
        for weighting in cfg.weightings:
            levels: dict = {}
            geoms: dict = {}
            for resolution in cfg.resolutions:
                lv, geom = _levels_for_ring(
                    cubes[resolution], cfg, resolution, ring, weighting, aoi=aoi
                )
                geoms[resolution] = geom
                if geom["ring_cells"] > 0:
                    levels[resolution] = lv
            ring_meta = {
                "ring_inner_km": ring[0],
                "ring_outer_km": ring[1],
                "beyond_advection": bool(ring[0] >= advection_km),
                "advection_km": round(advection_km, 2),
                "weighting": weighting,
            }
            # ERR-001: a ring with no cells is recorded, not fatal -- the sweep continues.
            for resolution in cfg.resolutions:
                if geoms[resolution]["ring_cells"] == 0:
                    for basis in cfg.bases:
                        rows.append(
                            {
                                **ring_meta,
                                **geoms[resolution],
                                "basis": basis,
                                "resolution": resolution,
                                "refused": True,
                                "refused_reason": "ring contains no cells after footprint "
                                "exclusion and AOI clipping",
                            }
                        )
            if not levels:
                continue
            part = decompose_from_levels(levels, cfg)
            for rec in part.to_dict("records"):
                rows.append(
                    {
                        **ring_meta,
                        **geoms[rec["resolution"]],
                        **rec,
                        "refused": False,
                        "refused_reason": "",
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # NOX-017: the wide config carries its own three-rule pre-registration; the narrow one does not.
    if hasattr(cfg, "rise_threshold"):
        verdict, reason, _ = classify_wide_verdict(df, cfg)
    else:
        verdict, reason = classify_ring_verdict(df, cfg)
    df["sweep_verdict"] = verdict
    df["sweep_verdict_reason"] = reason
    sort_cols = ["ring_inner_km", "weighting", "resolution", "basis"]
    return df.sort_values(sort_cols).reset_index(drop=True)


def render_findings(df: pd.DataFrame, cfg=None) -> str:
    """Plain-language findings note: the verdict, the numbers behind it, and what it costs (REQ-040)."""
    from noxus.config.run import RingSweepConfig

    cfg = cfg or RingSweepConfig()
    verdict, reason = classify_ring_verdict(df, cfg)
    lines = [
        "NOX-015 - background-ring radius sensitivity (paper critique C1)",
        "=" * 78,
        "Benchmark-blind: the CREA benchmark is NEVER read in this sweep.",
        "",
        f"VERDICT: {verdict}",
        f"  reason: {reason}",
        f"  pre-registered threshold: local share > {cfg.self_subtraction_threshold} with a "
        f"bootstrap CI excluding zero, in >= {cfg.min_cells_agreeing} of the "
        f"{len(cfg.bases) * len(cfg.resolutions)} (basis x resolution) cells at the outermost ring.",
        f"  advection scale |U|*tau = {cfg.wind_ms} m/s * {cfg.tau_h} h = "
        f"{cfg.advection_km:.1f} km (NOX-005; a marker, never a cut).",
        "",
        "Cluster-local change-variance share by ring (plain weighting)",
        "-" * 78,
    ]
    plain = df[(df["weighting"] == "plain") & (~df["refused"])]
    if not plain.empty:
        piv = plain.pivot_table(
            index=["ring_inner_km", "ring_outer_km", "ring_cells", "beyond_advection"],
            columns=["basis", "resolution"],
            values="share_steel_local",
        )
        lines += ["  " + ln for ln in piv.round(4).to_string().splitlines()]
        lines += [
            "",
            "Regional share over the same sweep (self-subtraction moves variance BETWEEN",
            "these two terms, so they should move in opposite directions if it is present)",
            "-" * 78,
        ]
        piv_r = plain.pivot_table(
            index=["ring_inner_km", "ring_outer_km"],
            columns=["basis", "resolution"],
            values="share_regional",
        )
        lines += ["  " + ln for ln in piv_r.round(4).to_string().splitlines()]
        lines += [
            "",
            "Ring geometry (EDGE-001/002/005: extent, size and azimuthal balance are",
            "confounds, so they are reported rather than assumed away)",
            "-" * 78,
        ]
        geo = plain.drop_duplicates(subset=["ring_inner_km", "ring_outer_km"]).loc[
            :,
            [
                "ring_inner_km",
                "ring_outer_km",
                "ring_cells",
                "realised_max_km",
                "octant_counts",
                "octant_imbalance",
                "empty_octants",
            ],
        ]
        lines += ["  " + ln for ln in geo.to_string(index=False).splitlines()]

    refused = df[df["refused"]]
    if not refused.empty:
        lines += ["", f"Refused rings: {len(refused)} row(s) - ring produced no cells (ERR-001)."]

    diag = ring_dependence_diagnostic(df, cfg)
    lines += [
        "",
        "POST-HOC diagnostic: how far does the share RISE as the ring moves out? (REQ-020/021)",
        "-" * 78,
        "  NOT the decision rule. The pre-registered rule above thresholds the share's LEVEL; this",
        "  measures its RISE, which is what critique C1 actually asks about. The two disagree on",
        "  this data, so the rise is reported as a diagnostic and the pre-registered verdict stands.",
        f"  rise from the {diag['baseline_ring']} ring to the {diag['outer_ring']} ring, per cell:",
    ]
    lines += [f"    {k:>14s}: {v:+.4f}" for k, v in diag["rises"].items()]
    lines += [
        f"  largest rise {diag['max_rise']:+.4f} (threshold {cfg.self_subtraction_threshold})",
        f"  cells rising monotonically with radius: {diag['monotone_cells']}/{diag['n_outer_cells']}",
        f"  cells showing the regional->local transfer signature: {diag['transfer_cells']}"
        f"/{diag['n_outer_cells']}",
        f"  cells whose CI excludes zero at the outermost ring: {diag['n_ci_excludes_zero']}"
        f"/{diag['n_outer_cells']}",
    ]

    lines += ["", "Interpretation", "-" * 78]
    if verdict == VERDICT_ROBUST:
        lines += [
            "  The ~0 cluster-local change-variance share SURVIVES moving the background ring",
            "  entirely beyond the advection scale. Ring self-subtraction is ruled out as the",
            "  explanation and the NOX-009 result stands as a physical finding.",
        ]
    elif verdict == VERDICT_SELF_SUBTRACTION:
        lines += [
            "  The ~0 cluster-local share does NOT survive moving the ring beyond the advection",
            "  scale. The NOX-009 headline is therefore partly an artefact of the ring geometry,",
            "  and paper Section 6.1, its abstract, synthesis and conclusion REQUIRE CORRECTION.",
            "  Note what this does NOT show: the near-null coupling (weekly r = 0.078) is measured",
            "  independently of the ring and is unaffected. What is lost is the *mechanism* the",
            "  paper offers for it, not the near-null itself.",
        ]
    else:
        lines += [
            "  NEITHER reading is established, and the paper cannot claim ring-independence.",
            "",
            "  Self-subtraction is NOT established: no cell rises past the pre-registered",
            "  materiality threshold, and the two change bases disagree flatly -- the deseason",
            "  basis rises monotonically with ring radius while the first-difference basis stays",
            "  clamped at zero and its raw incremental R^2 becomes MORE negative. A ring simply",
            "  eating the cluster's plume should hand variance back to BOTH bases once it moves",
            "  away, so the deseason rise is not, on its own, evidence of self-subtraction.",
            "",
            "  Ring-robustness is NOT established either: the local share is demonstrably",
            "  ring-DEPENDENT. It rises monotonically in the deseason cells, and at the outer",
            "  rings its bootstrap CI excludes zero -- so the '~0 within CI' claim that NOX-009",
            "  makes at 25-60 km does not hold at 60-120 km. The regional share falls from ~0.94",
            "  to ~0.84-0.88 in every cell over the same sweep.",
            "",
            "  A further limit the sweep cannot escape: the rise has not plateaued at the outermost",
            "  ring available, and the cube extends only ~124 km from the cluster centroid, so we",
            "  cannot show the ring ever fully clears the plume.",
            "",
            "  Consequence for the paper: Section 6.1 must report the share as ring-dependent and",
            "  quote the range across ring geometries, not the 25-60 km value alone; Section 6.2(i)",
            "  must be updated from 'this control was not run' to 'run, and it did not settle the",
            "  question'.",
        ]
    return "\n".join(lines) + "\n"


def run_ring_sweep(cfg=None) -> Path:
    """Run :func:`sweep_rings`, write the CSV + findings note, return the CSV path (REQ-040/042)."""
    from noxus.config.run import RingSweepConfig

    cfg = cfg or RingSweepConfig()
    df = sweep_rings(cfg)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / cfg.results_name
    df.to_csv(out_path, index=False)
    (out_dir / cfg.findings_name).write_text(render_findings(df, cfg), encoding="utf-8")
    return out_path


def render_wide_findings(df: pd.DataFrame, cfg=None) -> str:
    """Findings note for the NOX-017 wide sweep: the verdict, its three rules, and its limits."""
    from noxus.config.run import WideRingSweepConfig

    cfg = cfg or WideRingSweepConfig()
    verdict, reason, detail = classify_wide_verdict(df, cfg)
    n_cells = len(cfg.bases) * len(cfg.resolutions)
    lines = [
        "NOX-017 - wide-AOI ring sweep (does the cluster-local share plateau?)",
        "=" * 78,
        "Benchmark-blind: the CREA benchmark is NEVER read in this sweep.",
        f"AOI buffer {cfg.aoi_buffer_deg} deg (NOX-015 ran at 0.25 deg, whose full-360 ring radius",
        "is only 63 km -- its 60-120 km ring was four corner wedges, not an annulus).",
        "",
        f"VERDICT: {verdict}",
        f"  reason: {reason}",
        "",
        "Pre-registered rules (fixed BEFORE the wide cube was acquired; REQ-020..024)",
        "-" * 78,
        f"  Rule A (LEVEL, carried over from NOX-015 unchanged): share > "
        f"{cfg.self_subtraction_threshold} with a bootstrap CI excluding zero, in >= "
        f"{cfg.min_cells_agreeing} of {n_cells} cells at the outermost ring.",
        f"    -> fired in {detail.get('rule_a_cells', 0)}/{detail.get('n_outer_cells', 0)} cells: "
        f"{detail.get('rule_a')}",
        f"  Rule B (RISE, the corrected formulation): share(outermost) - share(baseline) > "
        f"{cfg.rise_threshold}, with the outermost CI excluding zero, in >= "
        f"{cfg.min_cells_agreeing} cells.",
        f"    -> fired in {detail.get('rule_b_cells', 0)}/{detail.get('n_outer_cells', 0)} cells: "
        f"{detail.get('rule_b')}",
        f"  Plateau: |share(outermost) - share(second-outermost)| < {cfg.plateau_threshold} in "
        "EVERY measurable cell (a cell clamped at zero is flat by construction and cannot carry a "
        "plateau on its own -- rule revised on a pilot before the full-record run; see the module "
        "docstring for the disclosure).",
        f"    -> {detail.get('plateau_cells', 0)}/{detail.get('cells_with_two_rings', 0)} cells "
        f"flat: {detail.get('plateau')}",
        "",
        "  Rule A is the rule NOX-015 committed to, reported here only for comparability: it",
        "  thresholds the share's LEVEL, so a footprint-confined source -- evidence AGAINST",
        "  self-subtraction -- would be misclassified as self-subtraction. That defect was recorded",
        "  in the NOX-015 review and is repaired here by PRE-REGISTERING Rule B, not by editing a",
        "  rule after seeing a result.",
        "",
        f"  advection scale |U|*tau = {cfg.wind_ms} m/s * {cfg.tau_h} h = "
        f"{cfg.advection_km:.1f} km (NOX-005; a marker, never a cut).",
        "",
        "Cluster-local change-variance share by ring (plain weighting)",
        "-" * 78,
    ]
    plain = df[(df["weighting"] == "plain") & (~df["refused"])] if not df.empty else df
    if not plain.empty:
        piv = plain.pivot_table(
            index=["ring_inner_km", "ring_outer_km", "ring_cells"],
            columns=["basis", "resolution"],
            values="share_steel_local",
        )
        lines += ["  " + ln for ln in piv.round(4).to_string().splitlines()]
        lines += [
            "",
            "Regional share over the same sweep (self-subtraction moves variance BETWEEN",
            "these two terms, so they should move in opposite directions if it is present)",
            "-" * 78,
        ]
        piv_r = plain.pivot_table(
            index=["ring_inner_km", "ring_outer_km"],
            columns=["basis", "resolution"],
            values="share_regional",
        )
        lines += ["  " + ln for ln in piv_r.round(4).to_string().splitlines()]
        lines += [
            "",
            "Ring geometry (EDGE-001/002: a wider ring is not only further away -- it holds",
            "more cells, is azimuthally lop-sided, and past ~120 km it spans the Bohai Sea and",
            "the Tianjin plume, i.e. it is no longer the same air mass)",
            "-" * 78,
        ]
        geo = plain.drop_duplicates(subset=["ring_inner_km", "ring_outer_km"]).loc[
            :,
            [
                "ring_inner_km",
                "ring_outer_km",
                "ring_cells",
                "realised_max_km",
                "octant_counts",
                "octant_imbalance",
                "empty_octants",
            ],
        ]
        lines += ["  " + ln for ln in geo.to_string(index=False).splitlines()]

    lines += [
        "",
        f"Rise from the {detail.get('baseline_ring', '?')} baseline to the "
        f"{detail.get('outer_ring', '?')} ring, per cell",
        "-" * 78,
    ]
    lines += [f"    {k:>16s}: {v:+.4f}" for k, v in detail.get("rises", {}).items()]
    lines += [
        "",
        "Last increment (the plateau test), per cell",
        "-" * 78,
    ]
    lines += [f"    {k:>16s}: {v:+.4f}" for k, v in detail.get("last_increments", {}).items()]

    refused = df[df["refused"]] if not df.empty else df
    if len(refused):
        lines += ["", f"Refused rings: {len(refused)} row(s) - ring produced no cells (ERR-001)."]

    lines += ["", "Interpretation", "-" * 78]
    if verdict == VERDICT_SELF_SUBTRACTION:
        lines += [
            "  Both pre-registered rules fire. The ~0 cluster-local share of NOX-009 does NOT",
            "  survive moving the ring past the plume on a cube wide enough to hold a real",
            "  annulus, so that headline is partly an artefact of the ring geometry and the",
            "  paper's Section 6.1, abstract, synthesis and conclusion REQUIRE CORRECTION.",
            "  What this does NOT overturn: the near-null coupling (weekly r = 0.078) is measured",
            "  independently of the ring. What is lost is the mechanism offered for it.",
        ]
    elif verdict == VERDICT_ROBUST:
        lines += [
            "  Neither rule fires and the share has plateaued. The ~0 cluster-local share",
            "  survives a background ring taken entirely beyond the advection scale, on a cube",
            "  wide enough for that ring to be a genuine annulus. Ring self-subtraction is ruled",
            "  out as the explanation and NOX-015's 'inconclusive' is resolved in favour of the",
            "  paper's reading -- with the caveat that the outer rings are a different air mass.",
        ]
    elif verdict == VERDICT_DOMAIN_LIMITED:
        lines += [
            "  The share is still moving at the edge of the widened domain. This is the same",
            "  limit NOX-015 hit, now measured on a cube 4.8x larger in area: widening the AOI",
            "  did not make the rise flatten, so the ring radius is not a knob that converges",
            "  over any domain this cluster's geography allows. Report the share as",
            "  ring-DEPENDENT and quote its range across geometries; do not quote the 25-60 km",
            "  value alone, and do not claim ring-independence.",
        ]
    elif verdict == VERDICT_RULES_DISAGREE:
        lines += [
            "  The level rule and the rise rule point different ways. Both numbers are above;",
            "  neither is selected after the fact. A disagreement is itself informative: it says",
            "  the share's LEVEL and its RESPONSE TO RADIUS are not the same measurement, which",
            "  is exactly the confusion the NOX-015 rule embodied.",
        ]
    else:
        lines += [
            "  The sweep did not produce a readable comparison (empty, refused, or a single",
            "  completed ring). No claim either way.",
        ]
    lines += [
        "",
        "  Standing limit, independent of the verdict: at 90-150 km the ring spans open sea and",
        "  the Tianjin urban plume. A share that moves at that radius may be responding to a",
        "  change of air mass rather than to escaping the cluster's own plume, and this sweep",
        "  cannot separate the two.",
    ]
    return "\n".join(lines) + "\n"


def run_wide_ring_sweep(cfg=None) -> Path:
    """Run the NOX-017 wide sweep, write the CSV + findings note, return the CSV path (REQ-030)."""
    from noxus.config.run import WideRingSweepConfig

    cfg = cfg or WideRingSweepConfig()
    df = sweep_rings(cfg)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / cfg.results_name
    df.to_csv(out_path, index=False)
    (out_dir / cfg.findings_name).write_text(render_wide_findings(df, cfg), encoding="utf-8")
    return out_path
