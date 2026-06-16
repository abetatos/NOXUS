"""Catalyst tests: detection, screens, match/lead, market study, no-look-ahead, null (NOX-004).

Fully offline + deterministic. Statistics asserted only on planted synthetic fixtures; no network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from noxus.catalyst import events as E
from noxus.catalyst import groundtruth as G
from noxus.catalyst import market as M
from noxus.catalyst import study as S


# --------------------------------------------------------------------------- detection (AT1/AT2/AT3)


def test_detect_recovers_planted_events(planted_event_residual):
    residual, coverage, _, truth = planted_event_residual
    ev = E.detect_events(residual, coverage, None, z_thresh=2.0, meteo_screen=False)
    got = set(pd.to_datetime(ev["date"]))
    for d in truth["date"]:
        assert pd.Timestamp(d) in got  # every planted event detected
    # Directions correct at the planted dates.
    merged = ev.merge(truth, on="date", suffixes=("_got", "_true"))
    assert (merged["direction_got"] == merged["direction_true"]).all()


def test_planted_null_low_false_positive_rate(planted_null_residual):
    ev = E.detect_events(planted_null_residual, None, None, z_thresh=2.5, meteo_screen=False)
    # A 2.5-sigma threshold on ~120 noise points should flag only a handful (no manufactured events).
    assert len(ev) <= 6


def test_coverage_screen_drops_cloud_gap(planted_event_residual):
    residual, coverage, _, truth = planted_event_residual
    coverage = coverage.copy()
    first_event_pos = list(residual.index).index(pd.Timestamp(truth["date"].iloc[0]))
    coverage.iloc[first_event_pos] = 0.05
    ev = E.detect_events(
        residual, coverage, None, z_thresh=2.0, min_coverage=0.25, meteo_screen=False
    )
    assert pd.Timestamp(truth["date"].iloc[0]) not in set(pd.to_datetime(ev["date"]))


def test_meteo_screen_rejects_stagnation_spike(planted_event_residual):
    residual, coverage, meteo, _ = planted_event_residual
    stagnation_date = residual.index[100]
    no_screen = E.detect_events(residual, coverage, meteo, z_thresh=2.0, meteo_screen=False)
    screened = E.detect_events(
        residual, coverage, meteo, z_thresh=2.0, meteo_screen=True, ventilation_z=1.5
    )
    assert stagnation_date in set(pd.to_datetime(no_screen["date"]))  # detected without the screen
    assert stagnation_date not in set(pd.to_datetime(screened["date"]))  # rejected with it


# --------------------------------------------------------------------------- no look-ahead (AT6)


def test_detection_is_causal_no_lookahead(planted_event_residual):
    residual, _, _, _ = planted_event_residual
    cut = 70
    full = E.detect_events(residual, None, None, z_thresh=2.0, meteo_screen=False)
    truncated = E.detect_events(residual.iloc[:cut], None, None, z_thresh=2.0, meteo_screen=False)
    # Events before the cut are identical whether or not future data exists (causal baseline).
    before_cut = residual.index[cut - 1]
    full_before = set(pd.to_datetime(full[full["date"] <= before_cut]["date"]))
    trunc_before = set(pd.to_datetime(truncated[truncated["date"] <= before_cut]["date"]))
    assert full_before == trunc_before


# --------------------------------------------------------------------------- ground truth + match (AT4)


def test_bf_rate_events_recovers_jumps():
    idx = pd.date_range("2021-01-03", periods=60, freq="W")
    rng = np.random.default_rng(2)
    val = 70.0 + rng.normal(0, 0.5, 60)  # realistic week-to-week wobble (non-zero MAD baseline)
    val[30:] += 15.0  # a step up -> a large +ve change event at index 30
    bench = pd.DataFrame({"date": idx, "value": val})
    ev = G.bf_rate_events(bench, z_thresh=2.0)
    assert (ev["direction"] == "up").any()
    assert pd.Timestamp(idx[30]) in set(pd.to_datetime(ev["date"]))


def test_match_recovers_lead(planted_event_residual, planted_production_events):
    residual, coverage, _, _ = planted_event_residual
    no2 = E.detect_events(residual, coverage, None, z_thresh=2.0, meteo_screen=False)
    res = S.match_events(no2, planted_production_events, window_days=14)
    assert res.n_matched >= 3
    assert res.precision > 0.0 and res.recall > 0.0
    # NO2 leads production by ~7 days (production planted at +1 week).
    assert res.median_lead_days == pytest.approx(7.0, abs=1.0)
    assert res.lead_positive_frac == 1.0


def test_calendar_combine_and_edges(tmp_path):
    cal = pd.DataFrame(
        {
            "start": ["2021-06-01"],
            "end": ["2021-07-01"],
            "direction": ["down"],
            "cause": ["heating"],
        }
    )
    p = tmp_path / "cal.csv"
    cal.to_csv(p, index=False)
    loaded = G.load_curtailment_calendar(p)
    bf = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-03-07"]),
            "direction": ["up"],
            "z": [3.0],
            "cause": ["bf_rate"],
        }
    )
    prod = G.production_events(bf, loaded)
    # The interval contributes a 'down' (start) and an 'up' (end) edge event, plus the bf event.
    assert (prod["direction"] == "down").any() and (prod["direction"] == "up").any()
    assert len(prod) == 3


def test_load_calendar_absent_is_empty():
    cal = G.load_curtailment_calendar(None)
    assert len(cal) == 0
    assert list(cal.columns) == ["start", "end", "direction", "cause"]


# --------------------------------------------------------------------------- market study (AT5/AT7)


def test_market_event_study_recovers_planted_car(planted_event_residual, planted_market_prices):
    _, _, _, truth = planted_event_residual
    surges = truth[truth["direction"] == "surge"].copy()
    car = S.market_event_study(surges, planted_market_prices, window=5, latency_days=2)
    assert car.n_instruments == 2
    # A positive abnormal-return bump was planted after surge events -> positive mean CAR + high hit rate.
    for _, row in car.by_instrument.iterrows():
        assert row["car_mean"] > 0
        assert row["hit_rate"] >= 0.5


def test_market_event_study_null_when_unrelated(planted_event_residual):
    _, _, _, truth = planted_event_residual
    days = pd.date_range("2021-01-01", periods=900, freq="D")
    rng = np.random.default_rng(1)
    prices = {
        "BHP": pd.DataFrame({"date": days, "abnormal_return": rng.normal(0, 0.002, len(days))})
    }
    car = S.market_event_study(truth, prices, window=5, latency_days=2)
    row = car.by_instrument.iloc[0]
    assert row["car_ci_low"] <= 0 <= row["car_ci_high"]  # CI straddles zero -> market null


def test_abnormal_returns_market_adjusted():
    days = pd.date_range("2021-01-01", periods=10, freq="D")
    prices = pd.concat(
        [
            pd.DataFrame({"symbol": "BHP", "date": days, "close": np.linspace(10, 11, 10)}),
            pd.DataFrame({"symbol": "ACWI", "date": days, "close": np.linspace(100, 101, 10)}),
        ]
    )
    ar = M.abnormal_returns(prices, "ACWI", instruments=["BHP"])
    assert "BHP" in ar
    assert "abnormal_return" in ar["BHP"].columns


def test_abnormal_returns_missing_benchmark_raises():
    days = pd.date_range("2021-01-01", periods=5, freq="D")
    prices = pd.DataFrame({"symbol": "BHP", "date": days, "close": np.arange(5.0)})
    with pytest.raises(M.MarketDataError, match="Benchmark"):
        M.abnormal_returns(prices, "ACWI")


# --------------------------------------------------------------------------- market ingest (mocked, no network)


def test_ingest_prices_uses_injected_fetch(tmp_path):
    from noxus.config.run import CatalystConfig

    days = pd.date_range("2021-01-01", periods=6, freq="D")

    def fake_fetch(symbols, start, end):
        return pd.concat(
            [
                pd.DataFrame({"symbol": s, "date": days, "close": np.arange(6.0) + 1})
                for s in symbols
            ]
        )

    cfg = CatalystConfig()
    out = M.ingest_prices(cfg, tmp_path, today="2026-06-13", _fetch=fake_fetch)
    assert out.exists()
    snap = pd.read_parquet(out)
    assert set(cfg.instruments).issubset(set(snap["symbol"].unique()))


# --------------------------------------------------------------------------- end-to-end report (AT8 / ERR)


def test_run_catalyst_insufficient_events_raises(tmp_path):
    from dataclasses import replace

    from noxus.catalyst.report import InsufficientEventsError, run_catalyst
    from noxus.config.run import CatalystConfig

    idx = pd.date_range("2021-01-03", periods=60, freq="W")
    decomp = pd.DataFrame(
        {"date": idx, "signal": 0.0, "trend_s_t": 0.0, "residual_activity": np.zeros(60)}
    )
    dp = tmp_path / "decomp.parquet"
    decomp.to_parquet(dp, index=False)
    bench = pd.DataFrame({"date": idx, "value": np.full(60, 70.0)})
    bp = tmp_path / "bench.parquet"
    bench.to_parquet(bp, index=False)

    cfg = replace(
        CatalystConfig(),
        decomposition_path=dp,
        benchmark_path=bp,
        meteo_screen=False,
        out_dir=tmp_path,
        events_out=tmp_path / "ev.parquet",
        market_snapshot_dir=tmp_path / "nomarket",
        min_events=5,
    )
    with pytest.raises(InsufficientEventsError, match="ERR-003"):
        run_catalyst(cfg)


def test_run_catalyst_missing_inputs_raises(tmp_path):
    from dataclasses import replace

    from noxus.config.run import CatalystConfig

    from noxus.catalyst.report import run_catalyst

    cfg = replace(CatalystConfig(), decomposition_path=tmp_path / "nope.parquet")
    with pytest.raises(FileNotFoundError, match="ERR-001"):
        run_catalyst(cfg)


def test_run_catalyst_end_to_end_reports(tmp_path):
    from dataclasses import replace

    from noxus.catalyst.report import run_catalyst
    from noxus.config.run import CatalystConfig

    rng = np.random.default_rng(5)
    idx = pd.date_range("2021-01-03", periods=120, freq="W")
    x = rng.normal(0, 1.0, 120)
    for i in (40, 60, 80, 100):
        x[i] += 6.0
    decomp = pd.DataFrame(
        {"date": idx, "signal": x, "trend_s_t": 0.0, "residual_activity": x, "valid_coverage": 0.9}
    )
    dp = tmp_path / "decomp.parquet"
    decomp.to_parquet(dp, index=False)
    val = np.full(120, 70.0)
    for i in (41, 61, 81, 101):
        val[i:] += 8.0  # steps up shortly after each NO2 surge
    bench = pd.DataFrame({"date": idx, "value": val})
    bp = tmp_path / "bench.parquet"
    bench.to_parquet(bp, index=False)

    cfg = replace(
        CatalystConfig(),
        decomposition_path=dp,
        benchmark_path=bp,
        meteo_screen=False,
        out_dir=tmp_path,
        events_out=tmp_path / "ev.parquet",
        market_snapshot_dir=tmp_path / "nomarket",  # absent -> market layer gracefully skipped
        min_events=3,
        bf_event_z=1.5,
    )
    art = run_catalyst(cfg)
    assert art.results_path.exists() and art.summary_path.exists()
    assert art.results["conclusion"] in {"lead", "coincident", "null"}
    assert art.results["market"]["verdict"] == "market_null"  # no snapshot -> no market signal
    summary = art.summary_path.read_text(encoding="utf-8")
    assert "No look-ahead" in summary
    assert "Multiplicity" in summary
