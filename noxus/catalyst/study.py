"""Event study: NO2 events vs production, and market abnormal returns (NOX-004, REQ-020..033,041,042).

Two studies, both reporting catalyst metrics (precision/recall/lead/CAR with CIs) — never a Pearson r
and never a cherry-picked event:

- :func:`match_events` — match NO2 events to ground-truth production events within a window and report
  precision, recall, false-discovery rate and the signed **lead time** (NO2 date − production date; a
  positive lead means NO2 flags the event before the official print).
- :func:`market_event_study` — for each NO2 event, the first **tradeable session is the overpass date +
  processing latency** (no look-ahead, REQ-041); the forward cumulative abnormal return (CAR) over the
  study window is the catalyst's market signature. Reports CAR + directional hit rate with bootstrap CIs
  and splits production-confirmed from unconfirmed events (REQ-033).

Determinism: bootstrap CIs use a fixed seed (NFR-006). The number of tested events/instruments is
returned for multiple-testing transparency (REQ-042).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# NO2-event direction -> production-event direction it should coincide with.
_DIR_MATCH = {"surge": "up", "drop": "down"}
_BOOT_SEED = 12345


@dataclass(frozen=True)
class MatchResult:
    """Precision/recall/lead of NO2 events vs production events (REQ-020/021)."""

    n_no2: int
    n_prod: int
    n_matched: int
    precision: float
    recall: float
    false_discovery_rate: float
    median_lead_days: float
    lead_ci: tuple[float, float]
    lead_positive_frac: float  # share of matches where NO2 leads (sign test)
    matched: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass(frozen=True)
class CARResult:
    """Market cumulative-abnormal-return study around events (REQ-031/032)."""

    by_instrument: pd.DataFrame  # instrument, n_events, car_mean, car_ci_low, car_ci_high, hit_rate
    n_events_tested: int
    n_instruments: int
    per_event: pd.DataFrame = field(default_factory=pd.DataFrame)


def _bootstrap_ci(x: np.ndarray, *, n: int = 2000, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean (fixed seed -> deterministic)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(_BOOT_SEED)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def match_events(
    no2_events: pd.DataFrame,
    prod_events: pd.DataFrame,
    *,
    window_days: float = 14.0,
    total_periods: int | None = None,
) -> MatchResult:
    """Match NO2 events to production events within ``window_days`` (REQ-020/021).

    Greedy nearest-in-time matching with a direction constraint (surge↔up, drop↔down); each production
    event is consumed at most once (no double-counting, EDGE-001). Lead time = production_date −
    no2_date in days (positive = NO2 leads). False-discovery rate = unmatched NO2 / NO2; if
    ``total_periods`` is given a period-level false-positive rate could be derived, but FDR is the
    honest headline for an event marker.
    """
    no2 = no2_events.copy()
    prod = prod_events.copy()
    no2["date"] = pd.to_datetime(no2["date"])
    prod["date"] = pd.to_datetime(prod["date"])
    n_no2, n_prod = len(no2), len(prod)

    used = set()
    rows = []
    for _, e in no2.sort_values("date").iterrows():
        want = _DIR_MATCH.get(e["direction"])
        cand = prod[(prod["direction"] == want) & (~prod.index.isin(used))]
        if len(cand):
            dt_days = (cand["date"] - e["date"]).dt.total_seconds() / 86400.0
            within = cand[dt_days.abs() <= window_days]
            if len(within):
                j = (within["date"] - e["date"]).abs().idxmin()
                used.add(j)
                lead = (prod.loc[j, "date"] - e["date"]).total_seconds() / 86400.0
                rows.append(
                    {
                        "no2_date": e["date"],
                        "prod_date": prod.loc[j, "date"],
                        "direction": e["direction"],
                        "lead_days": lead,
                        "cause": prod.loc[j, "cause"],
                    }
                )

    matched = pd.DataFrame(rows)
    n_matched = len(matched)
    precision = n_matched / n_no2 if n_no2 else float("nan")
    recall = n_matched / n_prod if n_prod else float("nan")
    fdr = 1.0 - precision if n_no2 else float("nan")
    if n_matched:
        leads = matched["lead_days"].to_numpy()
        median_lead = float(np.median(leads))
        ci = _bootstrap_ci(leads)
        pos_frac = float(np.mean(leads > 0))
    else:
        median_lead, ci, pos_frac = float("nan"), (float("nan"), float("nan")), float("nan")

    return MatchResult(
        n_no2=n_no2,
        n_prod=n_prod,
        n_matched=n_matched,
        precision=precision,
        recall=recall,
        false_discovery_rate=fdr,
        median_lead_days=median_lead,
        lead_ci=ci,
        lead_positive_frac=pos_frac,
        matched=matched,
    )


def _forward_car(ar: pd.Series, trade_date: pd.Timestamp, window: int) -> float:
    """Cumulative abnormal return over the first ``window`` sessions on/after ``trade_date`` (causal)."""
    fwd = ar[ar.index >= trade_date].iloc[:window]
    return float(fwd.sum()) if len(fwd) else float("nan")


def market_event_study(
    events: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    *,
    window: int = 5,
    latency_days: int = 2,
    expected_sign: dict[str, int] | None = None,
) -> CARResult:
    """Forward cumulative-abnormal-return study around events, per instrument (REQ-031/032/041).

    ``prices[instrument]`` is a daily frame with ``date`` and ``abnormal_return``. For each event the
    first tradeable session is the overpass date + ``latency_days`` (no look-ahead); the forward CAR
    sums the next ``window`` abnormal returns. ``expected_sign`` maps event direction → +1/−1 (default
    surge→+1, drop→−1) for the directional hit rate. Aggregates mean CAR + bootstrap CI + hit rate per
    instrument.
    """
    expected_sign = expected_sign or {"surge": 1, "drop": -1}
    ev = events.copy()
    ev["date"] = pd.to_datetime(ev["date"])

    per_event = []
    for inst, px in prices.items():
        ar = pd.Series(
            px["abnormal_return"].to_numpy(dtype=float),
            index=pd.DatetimeIndex(pd.to_datetime(px["date"])),
        ).sort_index()
        for _, e in ev.iterrows():
            trade_date = e["date"] + pd.Timedelta(days=latency_days)
            car = _forward_car(ar, trade_date, window)
            exp = expected_sign.get(e["direction"], 0)
            per_event.append(
                {
                    "instrument": inst,
                    "event_date": e["date"],
                    "direction": e["direction"],
                    "confirmed": bool(e.get("confirmed", True)),
                    "car": car,
                    "hit": (np.sign(car) == exp) if (exp and car == car) else np.nan,
                }
            )

    pe = pd.DataFrame(per_event)
    rows = []
    for inst, g in pe.groupby("instrument"):
        cars = g["car"].to_numpy()
        valid = cars[~np.isnan(cars)]
        lo, hi = _bootstrap_ci(valid)
        rows.append(
            {
                "instrument": inst,
                "n_events": int(len(valid)),
                "car_mean": float(np.nanmean(cars)) if len(valid) else float("nan"),
                "car_ci_low": lo,
                "car_ci_high": hi,
                "hit_rate": float(g["hit"].dropna().mean())
                if g["hit"].notna().any()
                else float("nan"),
            }
        )
    by_instrument = pd.DataFrame(rows)
    return CARResult(
        by_instrument=by_instrument,
        n_events_tested=int(len(ev)),
        n_instruments=int(len(prices)),
        per_event=pe,
    )
