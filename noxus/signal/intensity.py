"""Explicit emission-intensity (decoupling) model (NOX-003.1, REQ-101/102/103/104).

NOX-003 found a two-layer result: in *levels* the footprint NO2 correlates **negatively** with the
CREA blast-furnace operating rate, and only weakly positive after blind year-over-year detrending.
The negative-levels relationship is the signature of **decoupling**: emission intensity (NOx per
tonne of steel) falls secularly — driven by retrofits/policy (Li 2024) — faster than activity rises.
Blind difference filters either under-remove that trend (``yoy``) or erase the signal (``yoy-double-diff``).

This module models the intensity decline **explicitly**. On the meteorology-normalised footprint
signal it fits a smoothness-controlled secular trend ``s(t)`` and defines the **activity proxy as the
residual** ``signal - s(t)`` (the Li-2024 activity / intensity / meteorology decomposition; meteorology
is already regressed out upstream by ``regress_out_meteo``):

    signal  ≈  activity (residual)  +  s(t) (intensity trend)  +  ε

The central identifying assumption is a **timescale separation**: the retrofit decline is slow and
smooth while activity varies faster. Two disciplines guard against fooling ourselves:

1. **Smoothness is selected by cross-validation on the NO2 series alone** — the benchmark is never
   passed into the selection (:func:`fit_intensity_trend` takes no benchmark argument). Selecting the
   trend's degrees of freedom against the validation target would manufacture a correlation
   (REQ-102, NFR-102; Morris & Zhang 2019).
2. **A smoothness-sensitivity sweep** (:func:`smoothness_sweep`) reports the activity↔benchmark
   correlation as a function of trend degrees of freedom, so a signal is shown robust across smoothing
   choices rather than being an artefact of one (REQ-103).

The trend ``s(t)`` is returned as a reportable diagnostic — it *is* the decoupling — not discarded.

The default estimator is a **regression spline** whose effective degrees of freedom equal the number
of basis columns (exact, deterministic, numpy-only); a LOESS alternative is available. No new
dependency is introduced (REQ-101, NFR-106).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Default smoothness grid (effective degrees of freedom): 1=constant, 2=linear, 3=quadratic,
# >=4 = cubic regression spline with (df-4) interior knots. A short series caps the usable upper end.
DEFAULT_DF_GRID: tuple[float, ...] = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)


class IntensityModelError(RuntimeError):
    """The intensity model cannot fit a stable trend (ERR-101) or the CV config is degenerate (ERR-102)."""


@dataclass(frozen=True)
class TrendFit:
    """Result of the intensity decomposition.

    ``trend`` is ``s(t)`` (the secular intensity decline, a reportable diagnostic); ``residual`` is the
    activity proxy ``signal - s(t)``; ``df`` is the selected effective degrees of freedom; ``criterion``
    is how it was selected ("blocked-cv" | "gcv" | "fixed"); ``cv_score`` is the selection score (mean
    held-out MSE for blocked-cv, GCV value for gcv, NaN for fixed); ``estimator`` is "spline" | "loess".
    Both series are on the input index with NaN gaps preserved (no interpolation, REQ-004).
    """

    trend: pd.Series
    residual: pd.Series
    df: float
    criterion: str
    cv_score: float
    estimator: str


# --------------------------------------------------------------------------- basis / smoothers


def _normalise_t(n: int) -> np.ndarray:
    """Map ``n`` ordered observations to ``[0, 1]`` (the trend is in observation order)."""
    if n == 1:
        return np.zeros(1)
    return np.linspace(0.0, 1.0, n)


def _spline_basis(t: np.ndarray, df: int, knots: np.ndarray) -> np.ndarray:
    """Truncated-power regression-spline basis with exactly ``df`` columns.

    df<=1 -> constant; 2 -> linear; 3 -> quadratic; >=4 -> cubic with ``df-4`` interior ``knots``.
    ``t`` is normalised to [0, 1] so the basis is well-conditioned for the small df used here.
    """
    df = int(df)
    if df <= 1:
        return np.ones((len(t), 1))
    cols = [np.ones_like(t)]
    powers = min(df - 1, 3)
    for p in range(1, powers + 1):
        cols.append(t**p)
    if df >= 4:
        for k in knots:
            cols.append(np.clip(t - k, 0.0, None) ** 3)
    return np.column_stack(cols)


def _interior_knots(df: int) -> np.ndarray:
    """``df-4`` interior knots at equally spaced quantiles of the unit interval (cubic spline)."""
    n_interior = max(int(df) - 4, 0)
    if n_interior == 0:
        return np.empty(0)
    probs = np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
    return probs


def _fit_predict_spline(
    t_train: np.ndarray,
    y_train: np.ndarray,
    t_eval: np.ndarray,
    df: int,
    knots: np.ndarray,
) -> np.ndarray:
    """OLS fit of a ``df``-column spline basis on (t_train, y_train); predict at ``t_eval``."""
    b_train = _spline_basis(t_train, df, knots)
    coef, *_ = np.linalg.lstsq(b_train, y_train, rcond=None)
    return _spline_basis(t_eval, df, knots) @ coef


def _loess_frac(df: float) -> float:
    """Map an effective-df target to a LOESS span (more df -> smaller span -> wigglier)."""
    return float(np.clip(4.0 / max(df, 1.0), 0.12, 0.95))


def _fit_predict_loess(
    t_train: np.ndarray,
    y_train: np.ndarray,
    t_eval: np.ndarray,
    df: float,
) -> np.ndarray:
    """LOESS fit on (t_train, y_train); predict at ``t_eval`` by interpolating the smoothed curve."""
    from statsmodels.nonparametric.smoothers_lowess import lowess

    smoothed = lowess(y_train, t_train, frac=_loess_frac(df), return_sorted=True)
    xs, ys = smoothed[:, 0], smoothed[:, 1]
    return np.interp(t_eval, xs, ys)


def _fit_predict(estimator, t_train, y_train, t_eval, df, knots):
    if estimator == "spline":
        return _fit_predict_spline(t_train, y_train, t_eval, int(df), knots)
    if estimator == "loess":
        return _fit_predict_loess(t_train, y_train, t_eval, df)
    raise ValueError(f"Unknown intensity estimator {estimator!r} (use 'spline'/'loess').")


# --------------------------------------------------------------------------- smoothness selection


def _blocked_cv_score(
    estimator: str,
    t: np.ndarray,
    y: np.ndarray,
    df: float,
    knots: np.ndarray,
    cv_folds: int,
) -> float:
    """Mean held-out MSE under contiguous (time-respecting) blocked CV for one ``df`` (REQ-102)."""
    n = len(t)
    bounds = np.linspace(0, n, cv_folds + 1).astype(int)
    errs: list[float] = []
    for f in range(cv_folds):
        lo, hi = bounds[f], bounds[f + 1]
        if hi - lo < 1:
            continue
        test = np.zeros(n, dtype=bool)
        test[lo:hi] = True
        train = ~test
        if int(train.sum()) <= int(df) + 1:  # not enough training points to identify the basis
            continue
        pred = _fit_predict(estimator, t[train], y[train], t[test], df, knots)
        errs.append(float(np.mean((y[test] - pred) ** 2)))
    return float(np.mean(errs)) if errs else float("inf")


def _gcv_score(t: np.ndarray, y: np.ndarray, df: int, knots: np.ndarray) -> float:
    """Generalised cross-validation score for a spline of ``df`` effective parameters."""
    b = _spline_basis(t, df, knots)
    coef, *_ = np.linalg.lstsq(b, y, rcond=None)
    rss = float(np.sum((y - b @ coef) ** 2))
    n = len(y)
    denom = (1.0 - df / n) ** 2
    return (n * rss) / denom if denom > 0 else float("inf")


def fit_intensity_trend(
    signal: pd.Series,
    *,
    estimator: str = "spline",
    df_grid: Sequence[float] | None = None,
    cv_folds: int = 5,
    criterion: str = "blocked-cv",
    min_length: int = 24,
) -> TrendFit:
    """Fit the secular intensity trend ``s(t)`` and return the activity residual (REQ-101/102).

    Smoothness (effective degrees of freedom) is chosen from ``df_grid`` by ``criterion`` —
    ``"blocked-cv"`` (default; contiguous time-respecting folds) or ``"gcv"`` (spline only). The
    **benchmark is never an argument here**: selection uses only the NO2 series' own prediction error,
    so the trend cannot be tuned to manufacture a downstream correlation (NFR-102).

    Raises :class:`IntensityModelError` if the valid sample is shorter than ``min_length`` (ERR-101) or
    if the CV configuration is degenerate (empty grid, ``cv_folds < 2``, folds exceeding samples; ERR-102).
    NaN gaps in ``signal`` are preserved in both returned series (no interpolation, REQ-004).
    """
    s = signal.astype(float)
    valid = s.notna()
    n = int(valid.sum())
    grid = [float(d) for d in (df_grid if df_grid is not None else DEFAULT_DF_GRID)]

    if not grid:
        raise IntensityModelError("Intensity model: empty df_grid (degenerate CV config, ERR-102).")
    if criterion == "blocked-cv" and cv_folds < 2:
        raise IntensityModelError(
            f"Intensity model: cv_folds={cv_folds} < 2 (degenerate CV config, ERR-102)."
        )
    if n < min_length:
        raise IntensityModelError(
            f"Intensity model: only {n} valid points (< min_length={min_length}); refusing to fit "
            "an over-fit trend on too short a series (ERR-101). Use a longer series or a blind "
            "deseason method (yoy/none)."
        )
    if criterion == "blocked-cv" and cv_folds > n:
        raise IntensityModelError(
            f"Intensity model: cv_folds={cv_folds} exceeds valid samples n={n} (ERR-102)."
        )

    # Cap the grid so a candidate never requests more parameters than the (smaller) CV training set.
    max_train = n - (n // cv_folds) if criterion == "blocked-cv" else n
    usable = [d for d in grid if d <= max_train - 1] or [min(grid)]

    t_full = _normalise_t(n)
    y_full = s[valid].to_numpy()

    scored: list[tuple[float, float]] = []
    for d in usable:
        knots = _interior_knots(int(d)) if estimator == "spline" else np.empty(0)
        if criterion == "blocked-cv":
            score = _blocked_cv_score(estimator, t_full, y_full, d, knots, cv_folds)
        elif criterion == "gcv":
            if estimator != "spline":
                raise IntensityModelError(
                    "Intensity model: 'gcv' criterion requires estimator='spline'."
                )
            score = _gcv_score(t_full, y_full, int(d), knots)
        else:
            raise ValueError(f"Unknown criterion {criterion!r} (use 'blocked-cv'/'gcv').")
        scored.append((d, score))

    best_df, best_score = min(scored, key=lambda kv: kv[1])

    # Final trend fit on the full valid sample at the selected smoothness, then the residual.
    knots = _interior_knots(int(best_df)) if estimator == "spline" else np.empty(0)
    trend_valid = _fit_predict(estimator, t_full, y_full, t_full, best_df, knots)

    trend = pd.Series(np.nan, index=s.index, name="intensity_trend")
    trend.loc[valid] = trend_valid
    residual = (s - trend).rename("activity_residual")

    for series in (trend, residual):
        series.attrs.update(
            intensity_df=float(best_df),
            intensity_criterion=criterion,
            intensity_cv_score=float(best_score),
            intensity_estimator=estimator,
        )
    return TrendFit(
        trend=trend,
        residual=residual,
        df=float(best_df),
        criterion=criterion,
        cv_score=float(best_score),
        estimator=estimator,
    )


# --------------------------------------------------------------------------- sensitivity sweep


def smoothness_sweep(
    signal: pd.Series,
    benchmark: pd.Series,
    *,
    df_grid: Sequence[float] | None = None,
    max_lag: int = 8,
    estimator: str = "spline",
) -> pd.DataFrame:
    """Activity↔benchmark correlation as a function of trend smoothness (REQ-103).

    For each ``df`` in ``df_grid`` the trend is fit on the NO2 series at that *fixed* smoothness, the
    residual is correlated with ``benchmark`` (and a lead-lag peak is taken), and the **levels**
    correlation is also recorded. This is reporting only — it is **never** used to select ``df`` (that
    would be p-hacking). On a planted-null series the residual correlation stays near zero across all
    ``df``, proving no smoothing choice manufactures a signal.

    Returns one row per ``df`` with columns ``df, residual_r, residual_p, peak_lag, peak_r, levels_r``.
    """
    from noxus.validation.leadlag import correlate, lead_lag

    s = signal.astype(float)
    valid = s.notna()
    n = int(valid.sum())
    grid = [float(d) for d in (df_grid if df_grid is not None else DEFAULT_DF_GRID)]
    t_full = _normalise_t(n)
    y_full = s[valid].to_numpy()

    levels_r = float(correlate(s, benchmark).pearson_r)

    rows: list[dict] = []
    for d in grid:
        if d > n - 1:
            continue
        knots = _interior_knots(int(d)) if estimator == "spline" else np.empty(0)
        trend_valid = _fit_predict(estimator, t_full, y_full, t_full, d, knots)
        resid = s.copy()
        resid.loc[valid] = y_full - trend_valid
        cr = correlate(resid, benchmark)
        cc = lead_lag(resid, benchmark, max_lag=max_lag)
        rows.append(
            {
                "df": float(d),
                "residual_r": round(float(cr.pearson_r), 4),
                "residual_p": float(f"{cr.p_value:.3g}"),
                "peak_lag": int(cc.peak_lag),
                "peak_r": round(float(cc.peak_r), 4),
                "levels_r": round(levels_r, 4),
            }
        )
    return pd.DataFrame(rows)
