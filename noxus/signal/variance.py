"""Change-variance decomposition of the footprint NO2 signal (NOX-009, REQ-020..060).

The project's signal-to-noise *ceiling* has been justified with an **imported mean-level share**
(Wen 2024: steel = 30-43% of Tangshan NO2 *concentration*) → a level-correlation ceiling of
``sqrt(0.40) ≈ 0.63``. That is the wrong ceiling for a **change detector**: a steel-*activity*
detector lives in the **covariance of changes**, not the mean level, and the mean share was never
measured on *our* data. This module measures, benchmark-blind, what fraction of the **variance of
changes** in the meteo-normalised footprint NO2 signal is attributable to a four-term budget and
converts the steel-local share into an implied SNR ceiling for a change detector, contrasted with the
level ceiling.

The budget is a **sequential nested-R² partition** in the fixed order regional → +meteo → +local
(A1/A2), on the footprint (5 km) change signal:

    regional (a)     = R²  of  Δfootprint ~ Δbackground              (the BTH common mode)
    meteo    (b)     = ΔR² of  Δfootprint ~ Δbackground + meteo      (incremental, ERA5 linear)
    steel-local (c)  = the *structured* (persistent) part of the post-meteo footprint residual
    noise    (d)     = the white / retrieval part of that residual   (1 − everything above)

The four shares sum to 1 (within rounding). Term (c)/(d) split the residual variance ``γ0`` into a
structured component and a white component with the classic difference-based white-noise estimator
``σ² ≈ ½·mean(Δresidual²)`` (exact when the local component is smooth and the retrieval noise white):
``structured = γ0 − σ²``. A negative structured estimate (anti-persistent residual / suppression /
sampling noise) is clamped to 0, the raw value recorded and the term flagged (EDGE-002).

**Anti-p-hacking (NFR-001):** :func:`partition_change_variance` — like NOX-003.1
``fit_intensity_trend`` — takes **no benchmark argument**; the module never imports or reads the CREA
benchmark. p-hacking against the benchmark is therefore structurally impossible (ERR-004).

Every share and the implied ceiling carry a **moving-block-bootstrap CI** reusing
``noxus.validation.robust`` (NOX-008), reported alongside the naive i.i.d. interval so the
autocorrelation inflation is explicit (REQ-040). Reuses the attribution / meteo / deseason pathways
read-only (REQ-001/002); no new dependency (NFR-007).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from noxus.validation.robust import block_bootstrap_statistic, block_length

CANONICAL_ORDER: tuple[str, str, str] = ("regional", "meteo", "local")


@dataclass(frozen=True)
class VarianceBudget:
    """Four-way change-variance budget for one (basis, resolution) cell (REQ-022).

    ``share_*`` are the clamped (≥0) variance shares that sum to ≈ 1 (regional + meteo + steel_local +
    noise). ``dr2_*`` are the *raw* incremental R² terms in entry order (audit trail, REQ-021), before
    clamping — so a negative steel-local increment (EDGE-002) is visible. ``order`` records the fixed
    entry order; ``flags`` records clamping / degeneracy (e.g. ``"local_clamped"``, ``"no_regional"``,
    ``"low_power"``); ``meteo_covered_fraction`` is the fraction of response rows with complete meteo
    (EDGE-005); ``n`` is the effective sample size after gap-aware construction (EDGE-003).
    """

    basis: str
    resolution: str
    n: int
    share_regional: float
    share_meteo: float
    share_steel_local: float
    share_noise: float
    dr2_regional: float
    dr2_meteo: float
    dr2_local: float
    order: tuple[str, ...] = CANONICAL_ORDER
    meteo_covered_fraction: float = 1.0
    flags: tuple[str, ...] = field(default_factory=tuple)


# --------------------------------------------------------------------------- least squares helpers


def _ols_r2(y: np.ndarray, regressors: np.ndarray | None) -> tuple[float, np.ndarray]:
    """R² of ``y`` on ``regressors`` (intercept added) and the residual vector.

    ``regressors`` is an ``n x k`` matrix or ``None`` (intercept-only → R²=0, residual = y − mean).
    R² = 1 − RSS/TSS with TSS about the mean; a degenerate TSS returns R²=0.
    """
    n = len(y)
    tss = float(np.sum((y - y.mean()) ** 2))
    if regressors is None or regressors.size == 0:
        return 0.0, y - y.mean()
    design = np.column_stack([np.ones(n), regressors])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coef
    resid = y - fitted
    rss = float(np.sum(resid**2))
    r2 = 1.0 - rss / tss if tss > 0 else 0.0
    return r2, resid


def _white_noise_var(resid: np.ndarray) -> float:
    """Difference-based white-noise variance estimate ``σ² ≈ ½·mean(Δresidual²)`` (structured/noise split).

    Exact for a smooth structured component plus white retrieval noise: ``var(Δresidual) ≈ 2σ²`` when
    the structured part varies slowly. Consecutive positions in the (contiguous) core sample are used.
    """
    if len(resid) < 2:
        return float(np.var(resid))
    d = np.diff(resid)
    return 0.5 * float(np.mean(d**2))


def _numeric_meteo(meteo: pd.DataFrame | np.ndarray | None) -> np.ndarray | None:
    """Coerce a meteo DataFrame/array to a float ``n x k`` matrix (or None if empty)."""
    if meteo is None:
        return None
    if isinstance(meteo, pd.DataFrame):
        num = meteo.select_dtypes(include=[np.number])
        if num.shape[1] == 0:
            return None
        return num.to_numpy(dtype=float)
    arr = np.asarray(meteo, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr if arr.size else None


# --------------------------------------------------------------------------- core estimator (NFR-001)


def _partition_arrays(
    y: np.ndarray, bg: np.ndarray | None, meteo: np.ndarray | None
) -> dict[str, float]:
    """Sequential nested-R² partition on aligned numpy arrays; returns raw + clamped shares.

    Fixed order regional → +meteo → +local (A2). ``bg`` / ``meteo`` may be ``None`` (degenerate ring /
    no usable meteo) → that term contributes 0 and is skipped in the nesting. Shares are fractions of
    ``var(y)``; regional + meteo + residual_frac = 1 and residual_frac = local + noise, so the four
    sum to 1 exactly on the core sample (before clamping).
    """
    v = float(np.var(y))
    flags: list[str] = []
    if v <= 0:
        return {
            "r2_regional": 0.0,
            "dr2_meteo": 0.0,
            "dr2_local": 0.0,
            "share_regional": 0.0,
            "share_meteo": 0.0,
            "share_local": 0.0,
            "share_noise": 0.0,
            "flags": (),
        }

    # (a) regional
    if bg is not None and float(np.var(bg)) > 0:
        r2_reg, _ = _ols_r2(y, bg.reshape(-1, 1))
        reg_cols = bg.reshape(-1, 1)
    else:
        r2_reg = 0.0
        reg_cols = None
        if bg is None or float(np.var(bg)) <= 0:
            flags.append("no_regional")

    # (b) + meteo (incremental over regional)
    if meteo is not None and meteo.shape[1] > 0:
        rm_cols = meteo if reg_cols is None else np.column_stack([reg_cols, meteo])
        r2_rm, resid = _ols_r2(y, rm_cols)
    else:
        r2_rm, resid = (r2_reg, _ols_r2(y, reg_cols)[1])
        flags.append("no_meteo")
    dr2_meteo = r2_rm - r2_reg

    # (c)/(d) structured vs white split of the post-meteo residual. The difference-based estimator can
    # overshoot (noise_var > γ0) on an anti-persistent / very noisy residual -> a NEGATIVE structured
    # increment: record it raw (audit, EDGE-002) but cap the effective noise at γ0 so the local share
    # clamps to 0, the noise share never exceeds the residual, and the four shares still sum to 1.
    gamma0 = float(np.var(resid))
    noise_var = _white_noise_var(resid)
    residual_frac = 1.0 - r2_rm  # == gamma0 / v
    share_local_raw = (gamma0 - noise_var) / v  # raw incremental structured share (may be < 0)
    noise_var_eff = min(max(noise_var, 0.0), gamma0)
    share_local_eff = (gamma0 - noise_var_eff) / v  # >= 0, <= residual_frac
    share_noise_eff = noise_var_eff / v  # >= 0, <= residual_frac

    if share_local_raw < 0:
        flags.append("local_clamped")

    return {
        "r2_regional": r2_reg,
        "dr2_meteo": dr2_meteo,
        "dr2_local": share_local_raw,  # raw incremental structured share (may be < 0)
        "share_regional": max(r2_reg, 0.0),
        "share_meteo": max(dr2_meteo, 0.0),
        "share_local": share_local_eff,
        "share_noise": share_noise_eff,
        "residual_frac": residual_frac,
        "flags": tuple(flags),
    }


def _align_finite(
    footprint: pd.Series, background: pd.Series | None, meteo: pd.DataFrame | None
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, float]:
    """Core sample: rows where the footprint response is finite; align background + meteo.

    The meteo term is fit on the jointly-available rows only (EDGE-005); the covered fraction of
    response rows is returned. Background is aligned on the core sample; if it is missing/degenerate
    there it is returned as ``None`` (the regional term is then skipped, EDGE-004).
    """
    y = footprint.astype(float)
    valid = y.notna()
    idx = y.index[valid]
    y_arr = y[valid].to_numpy()

    bg_arr: np.ndarray | None = None
    if background is not None:
        bg = background.astype(float).reindex(idx)
        if len(bg) >= 2 and bg.notna().all() and float(np.var(bg.to_numpy())) > 0:
            bg_arr = bg.to_numpy()

    meteo_arr: np.ndarray | None = None
    covered = 1.0
    num = _numeric_meteo(meteo.reindex(idx) if isinstance(meteo, pd.DataFrame) else meteo)
    if num is not None:
        row_ok = np.isfinite(num).all(axis=1)
        covered = float(row_ok.mean()) if len(row_ok) else 0.0
        # Fit the meteo term on the jointly-available rows: restrict the whole core sample to them so
        # every term is on one sample and the shares still sum to 1 (no extrapolation, EDGE-005).
        if row_ok.any() and not row_ok.all():
            y_arr = y_arr[row_ok]
            if bg_arr is not None:
                bg_arr = bg_arr[row_ok]
            meteo_arr = num[row_ok]
        elif row_ok.all():
            meteo_arr = num

    return y_arr, bg_arr, meteo_arr, covered


def partition_change_variance(
    footprint: pd.Series,
    background: pd.Series | None,
    meteo: pd.DataFrame | None,
    *,
    basis: str,
    resolution: str = "",
    order: tuple[str, ...] = CANONICAL_ORDER,
) -> VarianceBudget:
    """Partition the change-variance of ``footprint`` into regional / meteo / steel-local / noise.

    ``footprint`` and ``background`` are the change-basis (already first-differenced or deseasonalised)
    footprint-5km and background-ring series; ``meteo`` is the ERA5 covariate frame aligned to them.
    ``basis`` / ``resolution`` are recorded labels. Returns a :class:`VarianceBudget` with the four
    clamped shares, the raw incremental R² terms and the entry order (REQ-020/021).

    **No benchmark argument exists (NFR-001, ERR-004):** the partition is a property of NO2 + meteo +
    background alone; a caller cannot pass a steel-output series in. The nested order is fixed
    (regional → meteo → local, A2), which credits shared variance to the confounders and makes the
    steel-local share a conservative lower bound (EDGE-001).
    """
    y_arr, bg_arr, meteo_arr, covered = _align_finite(footprint, background, meteo)
    n = int(len(y_arr))
    flags: list[str] = []
    if n < 4:
        flags.append("low_power")
    if n < 2:
        return VarianceBudget(
            basis=basis,
            resolution=resolution,
            n=n,
            share_regional=0.0,
            share_meteo=0.0,
            share_steel_local=0.0,
            share_noise=0.0,
            dr2_regional=0.0,
            dr2_meteo=0.0,
            dr2_local=0.0,
            order=tuple(order),
            meteo_covered_fraction=covered,
            flags=tuple(flags),
        )

    p = _partition_arrays(y_arr, bg_arr, meteo_arr)
    flags.extend(p["flags"])
    return VarianceBudget(
        basis=basis,
        resolution=resolution,
        n=n,
        share_regional=round(float(p["share_regional"]), 10),
        share_meteo=round(float(p["share_meteo"]), 10),
        share_steel_local=round(float(p["share_local"]), 10),
        share_noise=round(float(p["share_noise"]), 10),
        dr2_regional=round(float(p["r2_regional"]), 10),
        dr2_meteo=round(float(p["dr2_meteo"]), 10),
        dr2_local=round(float(p["dr2_local"]), 10),
        order=tuple(order),
        meteo_covered_fraction=round(covered, 6),
        flags=tuple(dict.fromkeys(flags)),
    )


def _steel_local_share_arrays(y: np.ndarray, *cols: np.ndarray) -> float:
    """Steel-local share from raw arrays — the statistic bootstrapped for the share CI.

    ``cols`` is ``(background, meteo_matrix?)`` (either may be absent). Fast, numpy-only, deterministic.
    """
    bg = cols[0] if len(cols) >= 1 and cols[0].size else None
    meteo = cols[1] if len(cols) >= 2 and cols[1].size else None
    if bg is not None:
        bg = bg.reshape(-1) if bg.ndim > 1 else bg
    p = _partition_arrays(y, bg, meteo if meteo is None else np.atleast_2d(meteo))
    return float(p["share_local"])


# --------------------------------------------------------------------------- implied ceilings (REQ-030)


def implied_change_ceiling(share_steel_local: float) -> float:
    """Implied SNR ceiling for a *change* detector: ``sqrt(steel-local change-variance share)``.

    The maximum |r| attainable if the steel-local component were a perfect proxy for the true steel
    activity change and the remaining variance were independent noise (an upper bound under that
    independence assumption, Q2 default). Collapses toward 0 as the steel-local share → 0 (EDGE-006).
    """
    return math.sqrt(max(float(share_steel_local), 0.0))


def level_ceiling(attributable_cap: tuple[float, float] = (0.30, 0.43)) -> tuple[float, float]:
    """Imported mean-*level* ceiling ``sqrt(attributable_cap)`` (Wen 2024), for the contrast (REQ-030)."""
    lo, hi = attributable_cap
    return math.sqrt(max(lo, 0.0)), math.sqrt(max(hi, 0.0))


def classify_verdict(
    share_steel_local: float,
    share_lo: float,
    share_hi: float,
    attributable_cap: tuple[float, float] = (0.30, 0.43),
) -> str:
    """Physical-vs-method-bound verdict framing the near-null against the ceiling (REQ-031).

    Never asserts an outcome in advance — it is a function of the *measured* shares:
      - steel-local share (or its CI) collapses toward ~0 → the null is **SNR-bound (physical)**;
      - steel-local change share ≈ the level share (≤ the Wen cap) → **ceiling-bound / physical**;
      - steel-local change share ≫ the level cap → **headroom / method-bound** (motivates NOX-010/011).
    """
    cap_lo, cap_hi = attributable_cap
    if share_hi <= 0.02 or share_steel_local <= 0.01:
        return (
            "SNR-bound: the steel-local change-variance share is ~0 within its CI; the near-null "
            "(weekly r≈0.07) is physical, not method-bound."
        )
    if share_steel_local <= cap_hi:
        return (
            "ceiling-bound / physical: the steel-local change-variance share "
            f"({share_steel_local:.3f}) is ≈ the imported mean-level share [{cap_lo:.2f},{cap_hi:.2f}]; "
            "the near-null (weekly r≈0.07) is SNR-bound, so a change detector cannot do much better here."
        )
    return (
        "headroom / method-bound: the steel-local change-variance share "
        f"({share_steel_local:.3f}) ≫ the imported mean-level share [{cap_lo:.2f},{cap_hi:.2f}]; "
        "the near-null (weekly r≈0.07) leaves headroom — motivating NOX-010 (deweathering) and "
        "NOX-011 (VIIRS triangulation)."
    )


# --------------------------------------------------------------------------- change-basis construction


def first_difference_gap_aware(series: pd.Series) -> pd.Series:
    """Gap-aware first difference: a difference is kept only across genuinely adjacent observations.

    No interpolation across cloud gaps (EDGE-003). A one-step ``diff`` is taken, then any difference
    whose *previous* value was NaN (i.e. spanning a gap) is set back to NaN, so the effective ``n``
    after dropping NaNs reflects only real adjacent pairs. Works on a possibly irregular index by using
    positional adjacency of the retained series rows.
    """
    s = series.astype(float)
    d = s.diff()
    # A diff is valid only where both this row and the immediately preceding row are finite.
    prev_finite = s.shift(1).notna() & s.notna()
    return d.where(prev_finite).rename(f"{series.name or 'series'}_d1")


def _deseason_residual(series: pd.Series, method: str, *, period: int, cfg=None) -> pd.Series:
    """Deseasonalised-residual basis: reuse ``noxus.signal.index.deseasonalize`` read-only (REQ-002)."""
    from noxus.signal.index import deseasonalize

    return deseasonalize(series, method=method, period=period, cfg=cfg)


def build_basis_series(
    footprint_level: pd.Series,
    background_level: pd.Series,
    meteo_level: pd.DataFrame | None,
    *,
    basis: str,
    resolution: str = "W",
    deseason_method: str = "harmonic",
    cfg=None,
) -> tuple[pd.Series, pd.Series, pd.DataFrame | None]:
    """Transform level footprint/background/meteo into a change basis (REQ-010).

    ``basis="first-diff"``: gap-aware first differences of footprint & background (EDGE-003); meteo is
    differenced onto the same index. ``basis="deseason"``: the deseasonalised residual (intensity trend
    + harmonic / intensity-model) of footprint & background; meteo covariates aligned as levels (the
    ventilation covariates of the residual). The annual ``period`` is set from ``resolution`` (52 weekly,
    365 daily). Returns ``(footprint, background, meteo)`` on the change basis. No interpolation across gaps.
    """
    if basis == "first-diff":
        fp = first_difference_gap_aware(footprint_level)
        bg = first_difference_gap_aware(background_level)
        met = meteo_level.diff() if isinstance(meteo_level, pd.DataFrame) else None
    elif basis == "deseason":
        period = 365 if resolution == "D" else 52
        fp = _deseason_residual(footprint_level, deseason_method, period=period, cfg=cfg)
        bg = _deseason_residual(background_level, deseason_method, period=period, cfg=cfg)
        met = meteo_level
    else:
        raise ValueError(f"Unknown change basis {basis!r} (use 'first-diff'/'deseason').")
    return fp, bg, met


# --------------------------------------------------------------------------- driver (REQ-011/022/040)


def _share_ci(
    footprint: pd.Series,
    background: pd.Series | None,
    meteo: pd.DataFrame | None,
    *,
    block: int,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    """Block-bootstrap CI of the steel-local share (+ naive i.i.d.), and the implied-ceiling CIs."""
    y_arr, bg_arr, meteo_arr, _ = _align_finite(footprint, background, meteo)
    n = int(len(y_arr))
    empty = {
        "share_lo": float("nan"),
        "share_hi": float("nan"),
        "share_lo_iid": float("nan"),
        "share_hi_iid": float("nan"),
        "block": block,
    }
    if n < max(4, 2 * block):
        return empty
    cols: list[np.ndarray] = [bg_arr if bg_arr is not None else np.zeros((n, 0))]
    cols.append(meteo_arr if meteo_arr is not None else np.zeros((n, 0)))
    lo, hi = block_bootstrap_statistic(
        _steel_local_share_arrays, y_arr, *cols, block=block, n_draws=n_boot, seed=seed
    )
    lo_i, hi_i = block_bootstrap_statistic(
        _steel_local_share_arrays, y_arr, *cols, block=1, n_draws=n_boot, seed=seed
    )
    return {
        "share_lo": lo,
        "share_hi": hi,
        "share_lo_iid": lo_i,
        "share_hi_iid": hi_i,
        "block": block,
    }


def decompose_from_levels(
    levels: dict[str, tuple[pd.Series, pd.Series, pd.DataFrame | None]],
    cfg=None,
) -> pd.DataFrame:
    """Run the full budget over (basis × resolution) from pre-built LEVEL series (no disk I/O).

    ``levels`` maps a resolution label ("W"/"D") to ``(footprint_level, background_level, meteo_level)``.
    For each resolution and each configured basis it builds the change-basis series, partitions the
    change-variance, attaches the block-bootstrap + naive CIs on the steel-local share, computes the
    implied change ceiling vs the level ceiling and the physical-vs-method verdict. Returns one tidy row
    per (basis, resolution). Reused by :func:`decompose`; separated so it is unit-testable and
    deterministic (NFR-004). No basis/resolution is dropped (NFR-003).
    """
    from noxus.config.run import VarianceDecompConfig

    cfg = cfg or VarianceDecompConfig()
    rows: list[dict] = []
    for resolution in cfg.resolutions:
        if resolution not in levels:
            continue
        fp_lvl, bg_lvl, met_lvl = levels[resolution]
        for basis in cfg.bases:
            fp, bg, met = build_basis_series(
                fp_lvl,
                bg_lvl,
                met_lvl,
                basis=basis,
                resolution=resolution,
                deseason_method=cfg.deseason_method,
                cfg=_signal_cfg_for(cfg),
            )
            budget = partition_change_variance(
                fp, bg, met, basis=basis, resolution=resolution, order=cfg.order
            )
            block = block_length(budget.n, exponent=cfg.block_exponent, floor=cfg.block_floor)
            ci = _share_ci(fp, bg, met, block=block, n_boot=cfg.n_boot, seed=cfg.seed)
            f = budget.share_steel_local
            change_ceiling = implied_change_ceiling(f)
            lev_lo, lev_hi = level_ceiling(cfg.attributable_cap)
            verdict = classify_verdict(
                f,
                ci["share_lo"] if np.isfinite(ci["share_lo"]) else f,
                ci["share_hi"] if np.isfinite(ci["share_hi"]) else f,
                cfg.attributable_cap,
            )
            rows.append(
                {
                    "basis": basis,
                    "resolution": resolution,
                    "n": budget.n,
                    "order": "->".join(budget.order),
                    "share_regional": budget.share_regional,
                    "share_meteo": budget.share_meteo,
                    "share_steel_local": budget.share_steel_local,
                    "share_noise": budget.share_noise,
                    "dr2_regional": budget.dr2_regional,
                    "dr2_meteo": budget.dr2_meteo,
                    "dr2_local": budget.dr2_local,
                    "meteo_covered_fraction": budget.meteo_covered_fraction,
                    "block": ci["block"],
                    "change_ceiling": round(change_ceiling, 6),
                    "change_ceiling_lo": round(implied_change_ceiling(ci["share_lo"]), 6)
                    if np.isfinite(ci["share_lo"])
                    else float("nan"),
                    "change_ceiling_hi": round(implied_change_ceiling(ci["share_hi"]), 6)
                    if np.isfinite(ci["share_hi"])
                    else float("nan"),
                    "share_lo": round(ci["share_lo"], 6)
                    if np.isfinite(ci["share_lo"])
                    else float("nan"),
                    "share_hi": round(ci["share_hi"], 6)
                    if np.isfinite(ci["share_hi"])
                    else float("nan"),
                    "share_lo_iid": round(ci["share_lo_iid"], 6)
                    if np.isfinite(ci["share_lo_iid"])
                    else float("nan"),
                    "share_hi_iid": round(ci["share_hi_iid"], 6)
                    if np.isfinite(ci["share_hi_iid"])
                    else float("nan"),
                    "level_ceiling_lo": round(lev_lo, 6),
                    "level_ceiling_hi": round(lev_hi, 6),
                    "flags": ";".join(budget.flags),
                    "verdict": verdict,
                }
            )
    return pd.DataFrame(rows)


def _signal_cfg_for(cfg):
    """A ``SignalConfig`` for the deseason reuse (harmonic order etc.), independent of the decomp cfg."""
    from noxus.config.run import SignalConfig

    return SignalConfig()


# --------------------------------------------------------------------------- disk driver + errors


def _load_cube(path: Path):
    import xarray as xr

    cube = xr.open_dataset(path).load()
    cube.close()
    return cube


def octant_ring_mean(cube, ring_mask, facilities) -> np.ndarray:
    """Equal-octant-weighted per-period mean of ``no2`` over a ring mask (NOX-015 REQ-013).

    The plain ring mean weights every cell equally, so a ring that is azimuthally lop-sided --
    which the far rings are, because the AOI is a rectangle -- is dominated by whichever lobe holds
    most cells. This aggregates within each of eight azimuthal octants about the cluster centroid
    first, then averages the *octant* means with equal weight, so ring geometry cannot masquerade as
    a change in the background. Empty octants are skipped, never imputed.

    Returns an array aligned to the cube ``time`` axis (NaN where no octant has a valid cell).
    """
    # Read-only reuse of the NOX-002b geometry helpers; source.py itself is not modified (NFR-002).
    from noxus.attribution.source import NO2, _cell_centres

    lon2d, lat2d, _, _ = _cell_centres(cube)
    clon = float(facilities["longitude"].mean())
    clat = float(facilities["latitude"].mean())
    # Azimuth about the centroid, cosine-corrected so octants are equal-angle on the ground.
    az = np.degrees(np.arctan2((lon2d - clon) * np.cos(np.radians(clat)), lat2d - clat)) % 360.0
    ring = np.asarray(ring_mask.values if hasattr(ring_mask, "values") else ring_mask, dtype=bool)

    no2 = cube[NO2]
    spatial_dims = [d for d in no2.dims if d != "time"]
    means = []
    for k in range(8):
        sel = ring & (az >= k * 45.0) & (az < (k + 1) * 45.0)
        if not sel.any():
            continue
        means.append(
            no2.where(xr.DataArray(sel, dims=ring_mask.dims))
            .mean(dim=spatial_dims, skipna=True)
            .values
        )
    if not means:
        return np.full(no2.sizes["time"], np.nan)
    with warnings.catch_warnings():
        # An all-NaN period (every octant cloud-gapped) is a legitimate gap, not an error.
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.vstack(means), axis=0)


def _level_series_for_cube(
    cube,
    cfg,
    resolution: str,
    ring: tuple[float, float] | None = None,
    ring_weighting: str = "plain",
    aoi=None,
):
    """Build the footprint (5 km) + background-ring level series + ERA5 meteo (REQ-001/002).

    Reuses ``noxus.attribution.source`` (footprint/background) and ``noxus.data.era5`` read-only.
    ``resolution`` sets the meteo composite frequency ("W"/"D").

    ``ring`` is the NOX-015 sweep hook. When ``None`` (the default, and the only path the committed
    NOX-009 run takes) the radii come from ``SignalConfig`` exactly as before -- this default must
    stay byte-identical, which ``tests/signal/test_ring_sweep.py::test_default_path_unchanged``
    enforces. When given ``(inner_km, outer_km)`` those radii are used instead.
    ``ring_weighting="octant"`` replaces the plain ring mean with the equal-octant mean (REQ-013).

    ``aoi`` is the NOX-017 wide-cube hook. ``background_ring`` clips the ring to an AOI box that
    defaults to the 0.25-deg ``TANGSHAN`` region; on a cube acquired at a wider buffer that default
    would silently discard every ring cell beyond the narrow box, so a wide sweep would reproduce the
    narrow one and look like a plateau that is really a clip (REQ-010). ``None`` keeps the committed
    default.
    """
    from noxus.attribution.source import (
        background_ring,
        footprint_mask,
        footprint_signal,
        load_facilities,
    )
    from noxus.config.run import SignalConfig
    from noxus.data.era5 import era5_footprint_series

    sig = SignalConfig()
    inner_km, outer_km = (
        ring
        if ring is not None
        else (
            sig.background_inner_km,
            sig.background_outer_km,
        )
    )
    facilities = load_facilities(cfg.facilities_csv, set(sig.active_statuses))
    fp_mask = footprint_mask(cube, facilities, sig.footprint_radius_km)
    bg_mask = background_ring(
        cube, fp_mask, inner_km, outer_km, facilities=facilities, aoi=aoi
    )
    df = footprint_signal(cube, fp_mask, bg_mask, mode=sig.background_mode)
    idx = pd.DatetimeIndex(pd.to_datetime(df["date"]))
    footprint = pd.Series(df["no2_footprint"].to_numpy(float), index=idx, name="footprint")
    if ring_weighting == "octant":
        background = pd.Series(
            octant_ring_mean(cube, bg_mask, facilities), index=idx, name="background"
        )
    elif ring_weighting == "plain":
        background = pd.Series(df["no2_bg"].to_numpy(float), index=idx, name="background")
    else:
        raise ValueError(f"Unknown ring weighting {ring_weighting!r} (use 'plain'/'octant').")

    snapshot = _latest_era5(Path(cfg.era5_snapshot_dir))
    freq = "D" if resolution == "D" else "W"
    meteo_df = era5_footprint_series(snapshot, fp_mask, freq=freq)
    meteo = meteo_df.set_index(pd.DatetimeIndex(pd.to_datetime(meteo_df["date"]))).drop(
        columns=["date"]
    )
    keep = [c for c in cfg.meteo_covariates if c in meteo.columns]
    meteo = meteo[keep].reindex(idx)
    return footprint, background, meteo


def _latest_era5(snapshot_dir: Path) -> Path:
    from noxus.data.era5 import ERA5SnapshotError

    snaps = sorted(Path(snapshot_dir).glob("era5_*.nc"))
    if not snaps:
        raise ERA5SnapshotError(
            f"No ERA5 snapshot under {snapshot_dir}. Run 'noxus ingest-era5' first to fetch a "
            "Copernicus CDS snapshot for the meteo term (ERR-003)."
        )
    return snaps[-1]


def decompose(cfg=None) -> pd.DataFrame:
    """End-to-end change-variance decomposition on the committed cubes (REQ-011/022, ERR-001/002/003).

    Loads the weekly cube (ERR-001) and, when daily resolution is requested, the NOX-006 daily cube
    (ERR-002), plus the ERA5 snapshot (ERR-003); builds the level footprint/background/meteo series per
    resolution; runs :func:`decompose_from_levels`; writes ``variance_decomposition.csv`` and returns
    the tidy frame. The CREA benchmark is never touched (NFR-001).
    """
    from noxus.config.run import VarianceDecompConfig

    cfg = cfg or VarianceDecompConfig()

    weekly_path = Path(cfg.cube_path_weekly)
    if not weekly_path.exists():
        raise FileNotFoundError(
            f"Weekly NO2 cube not found: {weekly_path}. Run 'noxus grid' first to build the weekly "
            "cube (ERR-001)."
        )
    daily_needed = "D" in cfg.resolutions
    daily_path = Path(cfg.cube_path_daily)
    if daily_needed and not daily_path.exists():
        raise FileNotFoundError(
            f"Daily NO2 cube not found: {daily_path}. Run 'noxus grid --freq D' to build the NOX-006 "
            "daily cube before requesting daily resolution (ERR-002)."
        )
    # Fail early on a missing ERA5 snapshot rather than dropping the meteo split (ERR-003).
    _latest_era5(Path(cfg.era5_snapshot_dir))

    levels: dict[str, tuple[pd.Series, pd.Series, pd.DataFrame | None]] = {}
    for resolution in cfg.resolutions:
        cube = _load_cube(weekly_path if resolution == "W" else daily_path)
        levels[resolution] = _level_series_for_cube(cube, cfg, resolution)

    df = decompose_from_levels(levels, cfg)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / cfg.results_name
    df.to_csv(out_path, index=False)
    return df


def run_decompose(cfg=None) -> Path:
    """Run :func:`decompose` and return the written CSV path (thin CLI entry, REQ-060)."""
    from noxus.config.run import VarianceDecompConfig

    cfg = cfg or VarianceDecompConfig()
    decompose(cfg)
    return Path(cfg.out_dir) / cfg.results_name
