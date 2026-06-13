"""Catalyst report: detect → match → market study → plain-language report incl. null (NOX-004, REQ-040).

Ties the catalyst together and emits two artifacts:

- ``catalyst_results.json`` — events summary, precision/recall/lead (+CI), per-instrument CAR (+CI) and
  hit rate, the conclusion (``lead`` | ``coincident`` | ``null``), a separate market verdict, and the
  multiplicity count (events × instruments) for honest reporting (REQ-040/042).
- ``catalyst_summary.txt`` — plain language that **states the null explicitly** when NO2 events neither
  lead production nor move the asset beyond chance (Morris & Zhang 2019).

A catalyst is only useful if it (a) flags real production events with a positive **lead vs the official
print** and (b) precedes a market move; both are reported with CIs, never a single event.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from noxus.catalyst.events import detect_events
from noxus.catalyst.groundtruth import bf_rate_events, load_curtailment_calendar, production_events
from noxus.catalyst.market import abnormal_returns, load_latest_snapshot
from noxus.catalyst.study import market_event_study, match_events


class InsufficientEventsError(RuntimeError):
    """Fewer than the configured minimum events after screening (NOX-004 ERR-003)."""


@dataclass(frozen=True)
class CatalystArtifacts:
    results_path: Path
    summary_path: Path
    results: dict


def _confirm_events(no2: pd.DataFrame, match) -> pd.DataFrame:
    """Tag NO2 events as production-confirmed when they appear in the matched set (REQ-033)."""
    out = no2.copy()
    confirmed_dates = (
        set(pd.to_datetime(match.matched["no2_date"])) if len(match.matched) else set()
    )
    out["confirmed"] = pd.to_datetime(out["date"]).isin(confirmed_dates)
    return out


def _conclusion(match) -> str:
    """Production conclusion: 'lead' | 'coincident' | 'null' (REQ-040)."""
    if match.n_matched == 0 or not (match.recall > 0) or not (match.precision >= 0.3):
        return "null"
    if match.median_lead_days > 1.0 and match.lead_positive_frac >= 0.5:
        return "lead"
    return "coincident"


def _market_verdict(car_table: pd.DataFrame) -> str:
    """'market_signal' if any instrument's CAR CI excludes zero, else 'market_null' (REQ-032)."""
    if not len(car_table):
        return "market_null"
    excl = (car_table["car_ci_low"] > 0) | (car_table["car_ci_high"] < 0)
    return "market_signal" if bool(excl.any()) else "market_null"


def build_results(no2_events, match, car: "object | None", *, config_echo: dict) -> dict:
    by_inst = car.by_instrument.to_dict("records") if car is not None else []
    n_tests = (car.n_events_tested * car.n_instruments) if car is not None else 0
    return {
        "n_no2_events": int(len(no2_events)),
        "n_surge": int((no2_events["direction"] == "surge").sum()),
        "n_drop": int((no2_events["direction"] == "drop").sum()),
        "production": {
            "n_production_events": match.n_prod,
            "n_matched": match.n_matched,
            "precision": match.precision,
            "recall": match.recall,
            "false_discovery_rate": match.false_discovery_rate,
            "median_lead_days": match.median_lead_days,
            "lead_ci_days": list(match.lead_ci),
            "lead_positive_frac": match.lead_positive_frac,
        },
        "market": {
            "by_instrument": by_inst,
            "n_events_tested": car.n_events_tested if car is not None else 0,
            "n_instruments": car.n_instruments if car is not None else 0,
            "verdict": _market_verdict(car.by_instrument if car is not None else pd.DataFrame()),
        },
        "conclusion": _conclusion(match),
        "multiplicity_n_tests": int(n_tests),
        "config_echo": config_echo,
    }


def render_summary(r: dict) -> str:
    p = r["production"]
    concl = r["conclusion"]
    if concl == "lead":
        head = (
            f"LEAD: NO2 events flag production events ~{p['median_lead_days']:.0f} day(s) ahead "
            f"(precision={p['precision']:.2f}, recall={p['recall']:.2f})."
        )
    elif concl == "coincident":
        head = (
            f"COINCIDENT: NO2 events coincide with production events (precision={p['precision']:.2f}, "
            f"recall={p['recall']:.2f}) but with no usable lead."
        )
    else:
        head = (
            "NULL: NO2 events do not reliably correspond to production events after screening. "
            "A valid, designed-for outcome (Morris & Zhang 2019), not a pipeline failure."
        )
    lines = [
        "Steel NO2 event catalyst — production-event detection & market lead",
        "=" * 78,
        head,
        "",
        f"NO2 events: {r['n_no2_events']} ({r['n_surge']} surge / {r['n_drop']} drop).",
        f"Production link: matched {p['n_matched']}/{p['n_production_events']} production events; "
        f"precision={p['precision']:.2f}, recall={p['recall']:.2f}, FDR={p['false_discovery_rate']:.2f}.",
        f"Lead time (NO2 − production): median {p['median_lead_days']:.1f} d "
        f"(95% CI {p['lead_ci_days'][0]:.1f}..{p['lead_ci_days'][1]:.1f}; "
        f"{p['lead_positive_frac']:.0%} of matches lead).",
        "",
        f"Market event-study ({r['market']['verdict']}): forward cumulative abnormal returns",
    ]
    for rec in r["market"]["by_instrument"]:
        lines.append(
            f"  {rec['instrument']:6s} n={rec['n_events']:>3d}  CAR={rec['car_mean']:+.4f} "
            f"(95% CI {rec['car_ci_low']:+.4f}..{rec['car_ci_high']:+.4f})  hit-rate={rec['hit_rate']:.2f}"
        )
    lines += [
        "",
        f"Multiplicity: {r['multiplicity_n_tests']} event×instrument tests "
        "(defaults fixed before viewing returns; CARs are not multiplicity-adjusted — treat as exploratory).",
        "No look-ahead: detection baselines are causal; the first tradeable session is overpass + "
        f"{r['config_echo'].get('overpass_latency_days')} day(s).",
        "",
        "Config echo:",
        f"  detector={r['config_echo'].get('detector')} z_thresh={r['config_echo'].get('z_thresh')} "
        f"match_window={r['config_echo'].get('match_window')} study_window={r['config_echo'].get('study_window')}",
        f"  instruments={r['config_echo'].get('instruments')} benchmark={r['config_echo'].get('market_benchmark')}",
    ]
    return "\n".join(lines) + "\n"


def _load_meteo(cfg) -> pd.DataFrame | None:
    """Best-effort ERA5 footprint meteo for the ventilation screen; None if unavailable (recorded)."""
    try:
        from noxus.config.run import SignalConfig
        from noxus.signal.index import (
            _latest_era5_snapshot,
            _load_footprint_mask,
        )
        from noxus.data.era5 import era5_footprint_series

        scfg = SignalConfig(era5_snapshot_dir=cfg.era5_snapshot_dir)
        snap = _latest_era5_snapshot(Path(cfg.era5_snapshot_dir))
        mask = _load_footprint_mask(scfg)
        me = era5_footprint_series(snap, mask, freq=cfg.freq)
        return me.set_index(pd.DatetimeIndex(pd.to_datetime(me["date"]))).drop(columns=["date"])
    except Exception:
        return None


def run_catalyst(cfg=None, *, write: bool = True) -> CatalystArtifacts:
    """End-to-end catalyst: residual → events → match vs production → market study → report (REQ-040).

    Raises ``FileNotFoundError`` (ERR-001) if the NOX-003.1 decomposition or the benchmark is missing,
    and :class:`InsufficientEventsError` (ERR-003) if too few events survive screening.
    """
    from noxus.config.run import CatalystConfig

    cfg = cfg or CatalystConfig()
    decomp_path = Path(cfg.decomposition_path)
    if not decomp_path.exists():
        raise FileNotFoundError(
            f"Intensity decomposition not found: {decomp_path}. Run 'noxus index --deseason "
            "intensity-model' first (ERR-001)."
        )
    bench_path = Path(cfg.benchmark_path)
    if not bench_path.exists():
        raise FileNotFoundError(
            f"Benchmark not found: {bench_path}. Run 'noxus ingest-benchmark' first (ERR-001)."
        )

    decomp = pd.read_parquet(decomp_path)
    residual = pd.Series(
        decomp["residual_activity"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(pd.to_datetime(decomp["date"])),
    )
    coverage = (
        pd.Series(decomp["valid_coverage"].to_numpy(), index=residual.index)
        if "valid_coverage" in decomp.columns
        else None
    )
    meteo = _load_meteo(cfg) if cfg.meteo_screen else None

    no2_events = detect_events(
        residual,
        coverage,
        meteo,
        z_thresh=cfg.z_thresh,
        method=cfg.detector,
        min_periods=cfg.detect_min_periods,
        min_coverage=cfg.min_coverage,
        meteo_screen=cfg.meteo_screen,
        ventilation_z=cfg.ventilation_z,
    )
    if len(no2_events) < cfg.min_events:
        raise InsufficientEventsError(
            f"Only {len(no2_events)} NO2 events after screening (< min_events={cfg.min_events}); "
            "refusing an underpowered study (ERR-003). A longer series (full TROPOMI fetch, NOX-003.1 "
            "T10) or a lower z_thresh is needed."
        )

    bench = pd.read_parquet(bench_path)
    prod = production_events(
        bf_rate_events(bench, z_thresh=cfg.bf_event_z),
        load_curtailment_calendar(cfg.curtailment_calendar),
    )
    window_days = (
        float(cfg.match_window) * 7.0
        if cfg.freq.upper().startswith("W")
        else float(cfg.match_window) * 30.0
    )
    match = match_events(no2_events, prod, window_days=window_days)
    no2_events = _confirm_events(no2_events, match)

    car = None
    try:
        prices = load_latest_snapshot(cfg.market_snapshot_dir)
        ar = abnormal_returns(prices, cfg.market_benchmark, instruments=list(cfg.instruments))
        if ar:
            confirmed = no2_events[no2_events["confirmed"]]
            study_events = confirmed if len(confirmed) >= cfg.min_events else no2_events
            car = market_event_study(
                study_events, ar, window=cfg.study_window, latency_days=cfg.overpass_latency_days
            )
    except Exception:
        car = None  # market layer is optional; production study still reported (ERR-002 graceful)

    config_echo = {
        "detector": cfg.detector,
        "z_thresh": cfg.z_thresh,
        "match_window": cfg.match_window,
        "study_window": cfg.study_window,
        "overpass_latency_days": cfg.overpass_latency_days,
        "instruments": list(cfg.instruments),
        "market_benchmark": cfg.market_benchmark,
        "meteo_screen": bool(meteo is not None),
        "freq": cfg.freq,
    }
    results = build_results(no2_events, match, car, config_echo=config_echo)
    summary = render_summary(results)

    out_dir = Path(cfg.out_dir)
    results_path = out_dir / cfg.results_name
    summary_path = out_dir / cfg.summary_name
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        Path(cfg.events_out).parent.mkdir(parents=True, exist_ok=True)
        no2_events.to_parquet(cfg.events_out, index=False)
        results_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        summary_path.write_text(summary, encoding="utf-8")
    return CatalystArtifacts(results_path=results_path, summary_path=summary_path, results=results)
