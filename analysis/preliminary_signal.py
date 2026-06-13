"""Reproducible preliminary analysis: steel NO2 activity signal vs CREA blast-furnace rate (NOX-003).

Regenerates the figures + the experiment battery behind the preliminary result. Re-runnable end to
end once the inputs exist (real, gitignored):

    uv run noxus ingest-benchmark      # -> data/derived/benchmark_tangshan_bf_operating_rate.parquet
    uv run noxus grid                  # -> data/derived/no2/no2_cube_w.nc
    uv run noxus ingest-era5           # -> data/raw/era5/era5_<date>.nc   (Copernicus CDS)
    uv run python analysis/preliminary_signal.py

It uses the production functions (footprint sampling, ERA5 footprint series, deseasonalisation, meteo
regress-out, correlation/lead-lag) so the analysis tracks the real pipeline. It writes:

    docs/figures/preliminary/*.png       # the figures embedded in docs/preliminary-results.html
    docs/figures/preliminary/battery.csv # the meteo x deseason x freq comparison table

Every execution mode is preserved and selectable so the result is replicable: deseasonalisation in
{yoy, stl, yoy-double-diff, none}, frequency in {weekly W, monthly ME}, meteo regress-out on/off, and
a footprint-radius sweep. Findings are interpreted in docs/preliminary-results.html and history.html.
"""

from __future__ import annotations

import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from noxus.attribution import source as S
from noxus.attribution.source import build_footprint_signal
from noxus.config.run import SignalConfig
from noxus.data import era5 as E
from noxus.signal.index import deseasonalize, regress_out_meteo
from noxus.validation.leadlag import correlate, lead_lag

FIGDIR = Path("docs/figures/preliminary")
RADIUS_KM = 5.0
DESEASON = ("none", "yoy", "stl", "yoy-double-diff")
FREQS = (("W", 52, "weekly"), ("ME", 12, "monthly"))


def _load_inputs():
    cfg = SignalConfig()
    cube = xr.open_dataset(cfg.cube_path)
    fac = S.load_facilities(cfg.facilities_csv)
    snap = sorted(glob.glob("data/raw/era5/era5_*.nc"))[-1]
    bench = pd.read_parquet("data/derived/benchmark_tangshan_bf_operating_rate.parquet")
    bench["date"] = pd.to_datetime(bench["date"])
    bench = bench.set_index("date")["value"].sort_index()
    return cfg, cube, fac, snap, bench


def _footprint_signal(cfg, radius_km):
    import tempfile
    from dataclasses import replace

    tmp = Path(tempfile.mkdtemp())
    rc = replace(cfg, footprint_radius_km=float(radius_km), out_dir=tmp, era5_snapshot_dir=tmp)
    build_footprint_signal(rc)
    sig = pd.read_parquet(tmp / cfg.footprint_signal_name)
    sig["date"] = pd.to_datetime(sig["date"])
    return sig.set_index("date")["no2_corrected"].sort_index()


def _meteo(snap, fp, freq):
    me = E.era5_footprint_series(snap, fp, freq=freq)
    me["date"] = pd.to_datetime(me["date"])
    return me.set_index("date")[["u10", "v10", "blh"]].sort_index()


def _aligned(x: pd.Series, b: pd.Series, freq: str) -> pd.DataFrame:
    """Resample both to a common period grid (handles the NO2 Sunday vs CREA weekday mismatch)."""
    xr_ = x.resample(freq).mean()
    br = b.resample(freq).mean()
    d = pd.concat([xr_.rename("x"), br.rename("b")], axis=1).dropna()
    return d


def run_battery(sig, snap, fp, bench):
    rows = []
    for freq, period, flab in FREQS:
        me = _meteo(snap, fp, freq)
        sg = sig.resample(freq).mean()
        sg_m = regress_out_meteo(sg, me.reindex(sg.index), form="linear")
        for meteo_on, base in ((False, sg), (True, sg_m)):
            for method in DESEASON:
                x = deseasonalize(base, method=method, period=period)
                d = _aligned(x, bench, freq)
                if len(d) < 10:
                    continue
                cr = correlate(d["x"], d["b"])
                cc = lead_lag(d["x"], d["b"], max_lag=8 if freq == "W" else 6)
                rows.append(
                    {
                        "freq": flab,
                        "meteo": meteo_on,
                        "deseason": method,
                        "n": cr.n,
                        "r": round(cr.pearson_r, 3),
                        "p": float(f"{cr.p_value:.3g}"),
                        "peak_lag": cc.peak_lag,
                        "peak_r": round(cc.peak_r, 3),
                    }
                )
    return pd.DataFrame(rows)


def radius_sweep(cfg, cube, fac, bench):
    rows = []
    for r in (3, 5, 8, 10, 12, 15):
        fp = S.footprint_mask(cube, fac, float(r))
        sig = _footprint_signal(cfg, r)
        x = deseasonalize(sig.resample("ME").mean(), method="yoy", period=12)
        d = _aligned(x, bench, "ME")
        cr = correlate(d["x"], d["b"])
        rows.append(
            {"radius_km": r, "cells": int(fp.sum()), "r": round(cr.pearson_r, 3), "p": cr.p_value}
        )
    return pd.DataFrame(rows)


def fig_decoupling(sig, bench):
    """Level NO2 vs BF rate (monthly): the opposing trends -> negative level correlation (Li 2024)."""
    s = sig.resample("ME").mean()
    b = bench.resample("ME").mean()
    d = pd.concat([s.rename("no2"), b.rename("bf")], axis=1).dropna()
    fig, ax1 = plt.subplots(figsize=(9, 4.2))
    ax1.plot(d.index, d["no2"] * 1e3, color="#b3261e", lw=1.6, label="NO2 footprint (level)")
    ax1.set_ylabel("NO2 footprint−bg (×10⁻³ mol/m²)", color="#b3261e")
    ax1.tick_params(axis="y", labelcolor="#b3261e")
    ax2 = ax1.twinx()
    ax2.plot(d.index, d["bf"], color="#1f6feb", lw=1.6, label="CREA BF operating rate")
    ax2.set_ylabel("BF operating rate (%)", color="#1f6feb")
    ax2.tick_params(axis="y", labelcolor="#1f6feb")
    r = d["no2"].corr(d["bf"])
    ax1.set_title(f"Raw levels: NO2 falls while BF rate holds — secular decoupling (r={r:.2f})")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig1_decoupling.png", dpi=130)
    plt.close(fig)


def fig_detrended(sig, bench):
    """yoy-detrended NO2 vs BF rate (monthly), standardised overlay -> weak positive co-movement."""
    x = deseasonalize(sig.resample("ME").mean(), method="yoy", period=12)
    bx = deseasonalize(bench.resample("ME").mean(), method="yoy", period=12)
    d = pd.concat([x.rename("no2"), bx.rename("bf")], axis=1).dropna()
    z = (d - d.mean()) / d.std()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(z.index, z["no2"], color="#b3261e", lw=1.5, label="NO2 footprint (yoy, z)")
    ax.plot(z.index, z["bf"], color="#1f6feb", lw=1.5, label="BF rate (yoy, z)")
    ax.axhline(0, color="#999", lw=0.6)
    r = d["no2"].corr(d["bf"])
    ax.set_title(f"Year-over-year change: a faint positive co-movement emerges (r={r:.2f})")
    ax.set_ylabel("standardised yoy change")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig2_detrended.png", dpi=130)
    plt.close(fig)


def fig_scatter(sig, bench):
    x = deseasonalize(sig.resample("ME").mean(), method="yoy", period=12)
    d = _aligned(x, bench, "ME")
    cr = correlate(d["x"], d["b"])
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.scatter(d["b"], d["x"] * 1e3, s=16, color="#1a7f37", alpha=0.7)
    m, c = np.polyfit(d["b"], d["x"] * 1e3, 1)
    xs = np.linspace(d["b"].min(), d["b"].max(), 50)
    ax.plot(xs, m * xs + c, color="#b3261e", lw=1.4)
    ax.set_xlabel("CREA BF operating rate, yoy (%)")
    ax.set_ylabel("NO2 footprint, yoy (×10⁻³ mol/m²)")
    ax.set_title(f"Monthly yoy: r={cr.pearson_r:.2f}, p={cr.p_value:.2g}, n={cr.n}")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig3_scatter.png", dpi=130)
    plt.close(fig)


def fig_leadlag(sig, bench):
    x = deseasonalize(sig.resample("ME").mean(), method="yoy", period=12)
    d = _aligned(x, bench, "ME")
    cc = lead_lag(d["x"], d["b"], max_lag=6)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.bar(cc.lags, cc.ccf, color="#1f6feb", width=0.7)
    ax.axhline(cc.sig_bound, color="#b3261e", ls="--", lw=0.9, label="white-noise band")
    ax.axhline(-cc.sig_bound, color="#b3261e", ls="--", lw=0.9)
    ax.axvline(0, color="#999", lw=0.6)
    ax.set_xlabel("lag (months; +ve = NO2 leads BF rate)")
    ax.set_ylabel("cross-correlation")
    ax.set_title(f"Lead-lag (monthly yoy): peak lag={cc.peak_lag}, r={cc.peak_r:.2f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig4_leadlag.png", dpi=130)
    plt.close(fig)


def fig_methods(battery):
    piv = battery[~battery["meteo"]].pivot(index="deseason", columns="freq", values="r")
    piv = piv.reindex(list(DESEASON))
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    piv.plot.bar(ax=ax, color={"weekly": "#9a6700", "monthly": "#1a7f37"})
    ax.axhline(0, color="#333", lw=0.8)
    ax.axhspan(0.5, 0.75, color="#1f6feb", alpha=0.12, label="literature bar 0.5–0.75")
    ax.set_ylabel("Pearson r vs BF rate")
    ax.set_xlabel("deseasonalisation method")
    ax.set_title(
        "Method × frequency comparison (no meteo): only detrending flips the sign positive"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig5_methods.png", dpi=130)
    plt.close(fig)


def fig_radius(sweep):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(sweep["radius_km"], sweep["r"], "o-", color="#1a7f37")
    ax.axhline(0, color="#999", lw=0.6)
    for _, row in sweep.iterrows():
        ax.annotate(
            f"{int(row['cells'])} cells",
            (row["radius_km"], row["r"]),
            fontsize=7,
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
        )
    ax.set_xlabel("footprint radius (km)")
    ax.set_ylabel("Pearson r (monthly yoy)")
    ax.set_title("Footprint-radius sensitivity (monthly yoy): weak and radius-insensitive")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig6_radius.png", dpi=130)
    plt.close(fig)


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    cfg, cube, fac, snap, bench = _load_inputs()
    fp = S.footprint_mask(cube, fac, RADIUS_KM)
    sig = _footprint_signal(cfg, RADIUS_KM)
    print(
        f"inputs: {int(fp.sum())} footprint cells @ {RADIUS_KM} km; signal {len(sig)} weeks; "
        f"ERA5 {Path(snap).name}; benchmark {len(bench)} obs"
    )

    battery = run_battery(sig, snap, fp, bench)
    sweep = radius_sweep(cfg, cube, fac, bench)
    battery.to_csv(FIGDIR / "battery.csv", index=False)
    sweep.to_csv(FIGDIR / "radius_sweep.csv", index=False)
    print("\n=== battery (meteo x deseason x freq) ===")
    print(battery.to_string(index=False))
    print("\n=== radius sweep (monthly yoy) ===")
    print(sweep.to_string(index=False))

    fig_decoupling(sig, bench)
    fig_detrended(sig, bench)
    fig_scatter(sig, bench)
    fig_leadlag(sig, bench)
    fig_methods(battery)
    fig_radius(sweep)
    print(f"\nfigures + tables -> {FIGDIR}")


if __name__ == "__main__":
    main()
