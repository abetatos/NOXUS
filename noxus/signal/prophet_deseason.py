"""Prophet (and harmonic) deseasonalisation of the daily NO2 footprint (NOX-006).

The exploration showed the weekly intensity/harmonic residual is regime-limited and the ~40% steel
*mean* share caps a level estimator. The one temporal lever short of flux divergence (NOX-005) is the
**day-of-week cycle**: steel runs 24/7 (baseload, flat across the week) while traffic/other NO2 sources
carry a weekly cycle — so removing a *weekly* seasonal term isolates the industrial component and raises
steel's *variance* share. That cycle is only visible at daily resolution (the weekly composite averages
it out), hence the daily series.

This module fits a **Prophet** decomposition on the daily footprint NO2 — piecewise-linear **trend**
(removes the secular emission-intensity decline) + **yearly** + **weekly** Fourier seasonality — and
returns the components plus the **observed-only** activity residual, deterministically (MAP estimation,
``uncertainty_samples=0``). Prophet is **lazy-imported** so the package loads without it; a dependency-
free **harmonic** fallback (annual Fourier on the intensity residual) is provided too.

Discipline: the full-series fit is **diagnostic only**; any lead/market claim must use the **causal**
rolling refit (:func:`prophet_deseason_causal`), which never uses future data (no look-ahead).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


class ProphetUnavailableError(RuntimeError):
    """Prophet is not installed (NOX-006 ERR-002); use deseason_method='harmonic'/'intensity'."""


class InsufficientDataError(RuntimeError):
    """Too few valid days for a stable Prophet fit (NOX-006 ERR-003)."""


@dataclass(frozen=True)
class ProphetFit:
    """Prophet decomposition of the daily footprint NO2 (observed days only).

    ``residual`` is the activity proxy (signal − trend − yearly − weekly); ``weekly`` is the day-of-week
    component (the source-separation handle); ``variance_removed`` is each component's share of the
    signal variance; ``weekly_amplitude`` is the peak-to-peak of the weekly effect.
    """

    residual: pd.Series
    trend: pd.Series
    yearly: pd.Series
    weekly: pd.Series
    weekly_amplitude: float
    variance_removed: dict
    params: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- Prophet decomposition


def prophet_deseason(
    daily: pd.Series,
    *,
    growth: str = "linear",
    changepoint_prior: float = 0.05,
    yearly_order: int = 4,
    weekly_order: int = 3,
    min_valid_days: int = 120,
) -> ProphetFit:
    """Fit Prophet (trend + yearly + weekly) on the gappy daily series; return observed-only residual.

    ``daily`` is the daily footprint NO2 indexed by date (NaN on cloud-gap days). Prophet fits on the
    observed days only and the residual is returned on those days — its interpolated value on a gap day
    is never emitted (REQ-012). Deterministic (MAP, ``uncertainty_samples=0``). Raises
    :class:`ProphetUnavailableError` if Prophet is absent (REQ ERR-002) and :class:`InsufficientDataError`
    below ``min_valid_days`` (ERR-003).
    """
    try:
        from prophet import Prophet
    except Exception as exc:  # noqa: BLE001 - any import failure means prophet is unusable here
        raise ProphetUnavailableError(
            "prophet is not installed. Run 'uv add prophet', or use deseason_method='harmonic' "
            "(dependency-free) / 'intensity' (ERR-002)."
        ) from exc

    s = daily.astype(float)
    obs = s.dropna()
    if len(obs) < min_valid_days:
        raise InsufficientDataError(
            f"Only {len(obs)} valid days (< min_valid_days={min_valid_days}); refusing the Prophet fit "
            "(ERR-003). Use a longer series or a lower threshold."
        )

    df = pd.DataFrame({"ds": pd.to_datetime(obs.index), "y": obs.to_numpy()})
    model = Prophet(
        growth=growth,
        yearly_seasonality=yearly_order,
        weekly_seasonality=weekly_order,
        daily_seasonality=False,
        changepoint_prior_scale=changepoint_prior,
        uncertainty_samples=0,  # MAP -> deterministic
    )
    _silence_stan()
    model.fit(df)
    fc = model.predict(df)

    idx = pd.DatetimeIndex(obs.index)
    trend = pd.Series(fc["trend"].to_numpy(), index=idx, name="trend")
    yearly = pd.Series(
        fc.get("yearly", pd.Series(0.0, index=range(len(idx)))).to_numpy(), index=idx, name="yearly"
    )
    weekly = pd.Series(
        fc.get("weekly", pd.Series(0.0, index=range(len(idx)))).to_numpy(), index=idx, name="weekly"
    )
    yhat = pd.Series(fc["yhat"].to_numpy(), index=idx)
    residual = (obs - yhat).rename("residual_activity")

    sig_var = float(np.var(obs.to_numpy())) or 1.0
    variance_removed = {
        "trend": float(np.var(trend.to_numpy())) / sig_var,
        "yearly": float(np.var(yearly.to_numpy())) / sig_var,
        "weekly": float(np.var(weekly.to_numpy())) / sig_var,
    }
    weekly_amplitude = float(np.ptp(weekly.to_numpy())) if len(weekly) else 0.0

    return ProphetFit(
        residual=residual,
        trend=trend,
        yearly=yearly,
        weekly=weekly,
        weekly_amplitude=weekly_amplitude,
        variance_removed=variance_removed,
        params={
            "growth": growth,
            "changepoint_prior": changepoint_prior,
            "yearly_order": yearly_order,
            "weekly_order": weekly_order,
            "n_obs": int(len(obs)),
        },
    )


def _silence_stan() -> None:
    """Quiet cmdstanpy/Prophet's chatty loggers so a fit doesn't flood the run output."""
    import logging

    for name in ("prophet", "cmdstanpy"):
        logging.getLogger(name).setLevel(logging.WARNING)


def weekday_profile(weekly: pd.Series) -> pd.DataFrame:
    """Average weekly seasonal effect by day-of-week (REQ-020 source-separation diagnostic).

    Returns Mon..Sun with the mean effect; a flat profile ⇒ steel-dominated (baseload), a strong
    weekday/weekend swing ⇒ a non-steel (traffic) component the residual no longer carries.
    """
    w = weekly.copy()
    dow = pd.DatetimeIndex(w.index).dayofweek
    prof = pd.Series(w.to_numpy(), index=dow).groupby(level=0).mean()
    return pd.DataFrame(
        {
            "weekday": [_WEEKDAYS[i] for i in prof.index],
            "effect": prof.to_numpy(),
        }
    )


# --------------------------------------------------------------------------- causal rolling refit


def prophet_deseason_causal(
    daily: pd.Series,
    *,
    refit_every: int = 30,
    min_train: int = 365 * 2,
    **kw,
) -> pd.Series:
    """No-look-ahead residual: refit Prophet on data up to each block and score only forward days.

    For a lead/market claim the residual at day *t* must use only data ≤ *t* (NFR-003). We refit every
    ``refit_every`` days on the history so far (≥ ``min_train`` days) and assign each new day the residual
    from the model trained strictly before it. Slower than the in-sample fit; use only for predictive
    evaluation. Returns the causal residual on observed days (NaN where no model could be trained yet).
    """
    s = daily.astype(float)
    obs = s.dropna()
    out = pd.Series(np.nan, index=s.index, name="residual_causal")
    if len(obs) < min_train + 1:
        return out

    dates = obs.index
    anchor = dates[min_train]
    while anchor <= dates[-1]:
        train = obs[obs.index < anchor]
        nxt = obs[(obs.index >= anchor) & (obs.index < anchor + pd.Timedelta(days=refit_every))]
        if len(train) >= kw.get("min_valid_days", 120) and len(nxt):
            fit = prophet_deseason(train.reindex(s.index[s.index < anchor]), **kw)
            # Score the held-out forward block with the trained model via a fresh predict.
            resid_block = _score_forward(train, nxt, fit, **kw)
            out.loc[nxt.index] = resid_block.to_numpy()
        anchor = anchor + pd.Timedelta(days=refit_every)
    return out


def _score_forward(train: pd.Series, forward: pd.Series, fit: ProphetFit, **kw) -> pd.Series:
    """Predict the forward block from a model trained on ``train`` and return observed − yhat (causal)."""
    from prophet import Prophet

    df = pd.DataFrame({"ds": pd.to_datetime(train.index), "y": train.to_numpy()})
    model = Prophet(
        growth=kw.get("growth", "linear"),
        yearly_seasonality=kw.get("yearly_order", 4),
        weekly_seasonality=kw.get("weekly_order", 3),
        daily_seasonality=False,
        changepoint_prior_scale=kw.get("changepoint_prior", 0.05),
        uncertainty_samples=0,
    )
    _silence_stan()
    model.fit(df)
    fc = model.predict(pd.DataFrame({"ds": pd.to_datetime(forward.index)}))
    yhat = pd.Series(fc["yhat"].to_numpy(), index=forward.index)
    return forward - yhat


# --------------------------------------------------------------------------- harmonic fallback


def harmonic_deseason(level: pd.Series, *, k: int = 1, min_length: int = 24) -> pd.Series:
    """Dependency-free fallback: intensity trend (NOX-003.1) + K annual Fourier harmonics removed.

    The Prophet-core seasonality without Prophet — a smooth trend (the intensity spline) plus an
    annual harmonic regression, calendar-anchored (phase from day-of-year). K=1 is the sweet spot.
    """
    from noxus.signal.intensity import fit_intensity_trend

    base = fit_intensity_trend(level, min_length=min_length).residual
    s = base.astype(float)
    valid = s.notna()
    if int(valid.sum()) < 2 * k + 4 or k == 0:
        return s.rename("residual_activity")
    doy = pd.DatetimeIndex(s.index[valid]).dayofyear.to_numpy(float)
    phase = 2.0 * np.pi * doy / 365.25
    cols = [np.ones_like(phase)]
    for j in range(1, k + 1):
        cols += [np.cos(j * phase), np.sin(j * phase)]
    x = np.column_stack(cols)
    y = s[valid].to_numpy()
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    out = s.copy()
    out.loc[valid] = y - x @ coef
    return out.rename("residual_activity")
