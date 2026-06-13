"""Sign check, Pearson r + p, and lead-lag cross-correlation (NOX-003, REQ-041/042).

Revives the lead/lag module reverted to a scaffold in NOX-001. The objective is an **honest** test,
not a positive finding: a rigorous null — no usable correlation/lead after controls — is a valid,
designed-for result (Morris & Zhang 2019) and is reported as such.

v1 leads with the task-mandated statistics: the empirically-verified **sign** (the NO2↔activity sign
is region-dependent and not fixed, Montgomery 2018), the **Pearson r with its p-value and confidence
interval**, and a **cross-correlation / lead-lag** profile over a configured lag window with a peak
and its significance bound. The heavier NOX-001 OOS + Diebold–Mariano engine is reusable but is
deferred (it tests forecasting skill, a stronger claim than "does any correlation exist"); v1 first
establishes whether a correlation exists at all.

Lag convention: a **positive peak lag k means the index leads the benchmark by k periods** — i.e. the
benchmark at time t is best explained by the index at time t−k. This is the useful direction (NO2 as
an early indicator of physical output).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SignResult:
    """Empirical sign of the index↔benchmark relationship (REQ-041)."""

    sign: str  # "positive" | "negative" | "indeterminate"
    pearson_r: float
    significant: bool


@dataclass(frozen=True)
class CorrResult:
    """Pearson correlation with p-value and confidence interval (REQ-042)."""

    pearson_r: float
    p_value: float
    n: int
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class CCFResult:
    """Cross-correlation / lead-lag profile and its peak (REQ-042)."""

    peak_lag: int
    peak_r: float
    lags: list[int] = field(default_factory=list)
    ccf: list[float] = field(default_factory=list)
    sig_bound: float = 0.0  # ~95% white-noise band, ±1.96/sqrt(n)


def _aligned_arrays(index: pd.Series, benchmark: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Return the jointly non-NaN values of two aligned series as float arrays."""
    df = pd.concat({"i": index, "b": benchmark}, axis=1).dropna()
    return df["i"].to_numpy(dtype=float), df["b"].to_numpy(dtype=float)


def correlate(index: pd.Series, benchmark: pd.Series, alpha: float = 0.05) -> CorrResult:
    """Pearson r with p-value and a (1−alpha) confidence interval (REQ-042).

    Uses ``scipy.stats.pearsonr`` (modern API: returns ``.statistic``/``.pvalue`` and a
    ``confidence_interval``). On too-few or zero-variance data the result is r=0, p=1 with a
    degenerate CI, so the caller can report a null rather than crash.
    """
    from scipy import stats

    x, y = _aligned_arrays(index, benchmark)
    n = int(len(x))
    if n < 3 or np.std(x) == 0 or np.std(y) == 0:
        return CorrResult(
            pearson_r=0.0, p_value=1.0, n=n, ci_low=float("nan"), ci_high=float("nan")
        )

    res = stats.pearsonr(x, y)
    try:
        ci = res.confidence_interval(confidence_level=1 - alpha)
        ci_low, ci_high = float(ci.low), float(ci.high)
    except Exception:  # pragma: no cover - older scipy without confidence_interval
        ci_low = ci_high = float("nan")
    return CorrResult(
        pearson_r=float(res.statistic),
        p_value=float(res.pvalue),
        n=n,
        ci_low=ci_low,
        ci_high=ci_high,
    )


def verify_sign(index: pd.Series, benchmark: pd.Series, alpha: float = 0.05) -> SignResult:
    """Empirically determine the sign of the relationship; never silently flip it (REQ-041, EDGE-007).

    A negative or insignificant sign is a reportable outcome, not a failure. The sign is
    ``indeterminate`` when the correlation is not significant at ``alpha``.
    """
    corr = correlate(index, benchmark, alpha=alpha)
    significant = corr.p_value < alpha
    if not significant:
        sign = "indeterminate"
    elif corr.pearson_r > 0:
        sign = "positive"
    else:
        sign = "negative"
    return SignResult(sign=sign, pearson_r=corr.pearson_r, significant=significant)


def lead_lag(index: pd.Series, benchmark: pd.Series, max_lag: int = 8) -> CCFResult:
    """Cross-correlation profile over ±``max_lag`` and its peak (REQ-042).

    For each lag ``k`` in ``[-max_lag, max_lag]`` we correlate the index shifted by ``k`` against the
    benchmark; ``k>0`` shifts the index forward in time so a positive peak lag means the **index leads
    the benchmark by k periods**. The peak is the lag of maximum absolute correlation. ``sig_bound``
    is the ±95% white-noise band (``1.96/sqrt(n)``) for eyeballing significance of the profile.
    """
    paired = pd.concat({"i": index, "b": benchmark}, axis=1)
    lags = list(range(-max_lag, max_lag + 1))
    ccf: list[float] = []
    for k in lags:
        shifted = paired["i"].shift(k)
        x, y = _aligned_arrays(shifted, paired["b"])
        if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
            ccf.append(0.0)
        else:
            ccf.append(float(np.corrcoef(x, y)[0, 1]))

    ccf_arr = np.asarray(ccf)
    peak_i = int(np.argmax(np.abs(ccf_arr)))
    n_overlap = int(paired.dropna().shape[0])
    sig_bound = float(1.96 / np.sqrt(n_overlap)) if n_overlap > 0 else float("inf")
    return CCFResult(
        peak_lag=int(lags[peak_i]),
        peak_r=float(ccf_arr[peak_i]),
        lags=lags,
        ccf=[float(v) for v in ccf_arr],
        sig_bound=sig_bound,
    )


def test_lead(index, benchmark):  # pragma: no cover - retained scaffold alias
    """Deprecated scaffold alias. Use ``correlate`` + ``lead_lag`` + ``verify_sign``."""
    raise NotImplementedError("test_lead() is superseded by correlate/lead_lag/verify_sign")
