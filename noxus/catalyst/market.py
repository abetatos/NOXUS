"""Market data ingest + abnormal returns (NOX-004, REQ-030/031).

Free, reproducible daily prices for the steel-exposed instruments: global miners (BHP/RIO/VALE) and a
steel ETF (SLX), plus a broad benchmark, via ``yfinance`` (lazy-imported so the module loads without
it and tests inject a fake fetch). Chinese ferrous futures (SHFE/DCE) lack a clean free API (Q1) and
are out of the default set — added behind a best-effort exchange-settlement snapshot when available.

Prices are written to a **dated snapshot** so analysis reads only the snapshot (NFR-001). Abnormal
returns are market-adjusted (instrument return − benchmark return) — a simple, robust de-meaning that
needs no estimation window and so carries no look-ahead.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class MarketDataError(RuntimeError):
    """Market data could not be fetched for an instrument (REQ-030, ERR-002)."""


def _yf_fetch(symbols: list[str], start: str, end: str | None) -> pd.DataFrame:
    """Fetch daily adjusted closes via yfinance (lazy import). Returns long: symbol, date, close."""
    import yfinance as yf  # lazy: only needed for a live fetch, never in tests

    raw = yf.download(symbols, start=start, end=end, auto_adjust=True, progress=False)
    close = raw["Close"] if "Close" in raw else raw
    close = close.to_frame() if isinstance(close, pd.Series) else close
    long = (
        close.reset_index()
        .melt(id_vars=close.index.name or "Date", var_name="symbol", value_name="close")
        .rename(columns={close.index.name or "Date": "date"})
    )
    long["date"] = pd.to_datetime(long["date"])
    return long.dropna(subset=["close"]).reset_index(drop=True)


def ingest_prices(
    cfg,
    snapshot_dir: Path | None = None,
    *,
    start: str = "2019-01-01",
    end: str | None = None,
    today: str | None = None,
    _fetch=None,
) -> Path:
    """Fetch daily prices for the configured instruments + benchmark → dated snapshot (REQ-030).

    ``_fetch`` (symbols, start, end) -> long frame is injectable for tests (no network). Instruments
    that come back empty are skipped with a recorded note rather than failing the whole run (ERR-002).
    Writes ``prices_<today>.parquet`` (instrument, date, close); returns its path.
    """
    snapshot_dir = Path(snapshot_dir or cfg.market_snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    fetch = _fetch or _yf_fetch
    symbols = list(dict.fromkeys([*cfg.instruments, cfg.market_benchmark]))

    long = fetch(symbols, start, end)
    if long is None or not len(long):
        raise MarketDataError(f"No market data returned for {symbols} (ERR-002).")
    got = set(long["symbol"].unique())
    missing = [s for s in symbols if s not in got]
    if missing:
        long.attrs["skipped"] = missing  # recorded; not fatal

    stamp = today or pd.Timestamp.today().strftime("%Y-%m-%d")
    out = snapshot_dir / f"prices_{stamp}.parquet"
    long.to_parquet(out, index=False)
    return out


def load_latest_snapshot(snapshot_dir: Path) -> pd.DataFrame:
    """Load the most recent ``prices_<date>.parquet`` snapshot (ERR-002 if none)."""
    snaps = sorted(Path(snapshot_dir).glob("prices_*.parquet"))
    if not snaps:
        raise MarketDataError(
            f"No market snapshot under {snapshot_dir}. Run 'noxus ingest-market' first (ERR-002)."
        )
    return pd.read_parquet(snaps[-1])


def abnormal_returns(
    prices: pd.DataFrame, benchmark: str, *, instruments: list[str] | None = None
) -> dict[str, pd.DataFrame]:
    """Market-adjusted abnormal returns per instrument (REQ-031).

    ``prices`` is long (instrument/symbol, date, close). Computes daily simple returns and subtracts the
    benchmark's return on the same date. Returns ``{instrument: DataFrame(date, ret, abnormal_return)}``;
    an instrument absent from ``prices`` is skipped (ERR-002) rather than fabricated.
    """
    sym_col = "symbol" if "symbol" in prices.columns else "instrument"
    wide = prices.pivot_table(
        index="date", columns=sym_col, values="close", aggfunc="last"
    ).sort_index()
    rets = wide.pct_change()
    if benchmark not in rets.columns:
        raise MarketDataError(f"Benchmark {benchmark!r} absent from market snapshot (ERR-002).")
    bench_ret = rets[benchmark]

    instruments = instruments or [c for c in rets.columns if c != benchmark]
    out: dict[str, pd.DataFrame] = {}
    for inst in instruments:
        if inst not in rets.columns:
            continue  # skipped (recorded by the caller)
        df = (
            pd.DataFrame(
                {
                    "date": rets.index,
                    "ret": rets[inst].to_numpy(),
                    "abnormal_return": (rets[inst] - bench_ret).to_numpy(),
                }
            )
            .dropna(subset=["abnormal_return"])
            .reset_index(drop=True)
        )
        out[inst] = df
    return out
