"""Deep exploratory analysis: NO2 variants × steel activity × economy (NOX-004 follow-up).

Goal (developer-directed 2026-06-13): stop assuming the NO2↔production lag — **infer it from the
data** — and hunt for patterns/correlations we don't see at first glance, across every representation
we've built plus the market ("economy") series.

It assembles, on a common weekly and monthly grid:
  - NO2 footprint (level), intensity residual (NOX-003.1), year-over-year, and STL residual;
  - steel activity: CREA blast-furnace operating rate (+ aux: pig-iron output, crude-steel output,
    BF starting rate 247);
  - economy: BHP / RIO / VALE / SLX returns vs the ACWI benchmark (yfinance snapshot).

and runs: lead-lag cross-correlation (every NO2 variant vs BF, and steel vs miners) to **infer the
lag**; rolling correlation to expose **regime changes**; a full correlation matrix at the inferred
lags; seasonal climatology; and an events-on-timeline overlay.

Reproduce (after the upstream artifacts exist):
    uv run python analysis/deep_exploration.py
Writes docs/figures/exploration/*.png + findings.txt. Reads gitignored NO2/market artifacts.
"""

from __future__ import annotations

import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from noxus.signal.index import deseasonalize
from noxus.validation.leadlag import correlate, lead_lag

FIGDIR = Path("docs/figures/exploration")
AUX = {
    "China: Estimated Daily Average Output: Pig Iron": "pig_iron",
    "China: Estimated Daily Average Output: Crude Steel": "crude_steel",
    "China: Blast Furnace Starting Rate (247)": "bf_start_247",
}
MINERS = ("BHP", "RIO", "VALE", "SLX")


# --------------------------------------------------------------------------- data assembly


def _series_from(df, col="value", date="date"):
    s = pd.Series(df[col].to_numpy(float), index=pd.DatetimeIndex(pd.to_datetime(df[date])))
    return s[~s.index.duplicated(keep="last")].sort_index()


def load_all():
    decomp = pd.read_parquet("data/derived/no2/steel_intensity_decomposition.parquet")
    di = pd.DatetimeIndex(pd.to_datetime(decomp["date"]))
    no2_level = pd.Series(decomp["signal"].to_numpy(float), index=di).sort_index()
    no2_resid = pd.Series(decomp["residual_activity"].to_numpy(float), index=di).sort_index()

    bench = _series_from(
        pd.read_parquet("data/derived/benchmark_tangshan_bf_operating_rate.parquet")
    )
    aux = pd.read_parquet("data/derived/benchmark_auxiliary.parquet")
    aux_series = {short: _series_from(aux[aux["series"] == long]) for long, short in AUX.items()}

    mpath = sorted(glob.glob("data/raw/market/prices_*.parquet"))[-1]
    mkt = pd.read_parquet(mpath)
    wide = mkt.pivot_table(
        index="date", columns="symbol", values="close", aggfunc="last"
    ).sort_index()
    wide.index = pd.DatetimeIndex(wide.index)
    return no2_level, no2_resid, bench, aux_series, wide


def build_master(freq, period, no2_level, no2_resid, bench, aux_series, mkt_wide):
    """A single aligned frame at `freq` with every NO2 variant + steel + market returns."""
    lvl = no2_level.resample(freq).mean()
    cols = {
        "no2_level": lvl,
        "no2_resid": no2_resid.resample(freq).mean(),
        "no2_yoy": deseasonalize(lvl, method="yoy", period=period),
        "no2_stl": deseasonalize(lvl, method="stl", period=period),
        "bf": bench.resample(freq).mean(),
    }
    for short, s in aux_series.items():
        cols[short] = s.resample(freq).mean()
    # Market: weekly/monthly simple returns of each instrument minus the ACWI benchmark.
    rets = mkt_wide.resample(freq).last().pct_change()
    if "ACWI" in rets:
        for m in MINERS:
            if m in rets:
                cols[f"ret_{m}"] = rets[m] - rets["ACWI"]
    return pd.DataFrame(cols)


# --------------------------------------------------------------------------- lead-lag (infer the lag)


def ccf_table(master, variants, target="bf", max_lag=8):
    """Best lead-lag (peak |r|) of each variant vs the target. +lag => variant LEADS target."""
    rows = []
    for v in variants:
        d = master[[v, target]].dropna()
        if len(d) < 12:
            continue
        cc = lead_lag(d[v], d[target], max_lag=max_lag)
        cr = correlate(d[v], d[target])
        rows.append(
            {
                "variant": v,
                "n": cr.n,
                "r_lag0": round(cr.pearson_r, 3),
                "best_lag": cc.peak_lag,
                "r_at_best": round(cc.peak_r, 3),
                "sig_band": round(cc.sig_bound, 3),
            }
        )
    return pd.DataFrame(rows)


def fig_ccf_curves(master, variants, freq_label, unit, target="bf", max_lag=8):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(variants)))
    band = None
    for v, c in zip(variants, colors):
        d = master[[v, target]].dropna()
        if len(d) < 12:
            continue
        cc = lead_lag(d[v], d[target], max_lag=max_lag)
        ax.plot(
            cc.lags,
            cc.ccf,
            "o-",
            ms=3,
            color=c,
            label=f"{v} (peak {cc.peak_lag:+d}, r={cc.peak_r:.2f})",
        )
        band = cc.sig_bound
    if band:
        ax.axhspan(-band, band, color="#bbb", alpha=0.3, label="white-noise band")
    ax.axvline(0, color="#333", lw=0.8)
    ax.axhline(0, color="#333", lw=0.5)
    ax.set_xlabel(f"lag ({unit}; +ve = NO2 LEADS the BF rate)")
    ax.set_ylabel("cross-correlation")
    ax.set_title(f"Lead-lag: NO2 variants vs BF operating rate ({freq_label}) — infer the delta")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGDIR / f"expl_ccf_{freq_label}.png", dpi=130)
    plt.close(fig)


def fig_best_lag_scatter(master, variant, best_lag, freq_label, target="bf"):
    d = master[[variant, target]].dropna()
    shifted = d[variant].shift(best_lag)
    dd = pd.concat([shifted.rename("x"), d[target].rename("b")], axis=1).dropna()
    cr = correlate(dd["x"], dd["b"])
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.scatter(dd["b"], dd["x"], s=16, color="#1a7f37", alpha=0.7)
    if len(dd) > 2:
        m, c = np.polyfit(dd["b"], dd["x"], 1)
        xs = np.linspace(dd["b"].min(), dd["b"].max(), 50)
        ax.plot(xs, m * xs + c, color="#b3261e", lw=1.4)
    ax.set_xlabel("BF operating rate (%)")
    ax.set_ylabel(f"{variant} shifted {best_lag:+d}")
    ax.set_title(
        f"{variant} at its best lag {best_lag:+d} ({freq_label}): r={cr.pearson_r:.2f}, n={cr.n}"
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / f"expl_bestlag_scatter_{freq_label}.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- regimes (rolling corr)


def fig_rolling_corr(master, variants, window, freq_label, target="bf"):
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for v in variants:
        d = master[[v, target]].dropna()
        if len(d) < window + 4:
            continue
        rc = d[v].rolling(window).corr(d[target])
        ax.plot(rc.index, rc, lw=1.4, label=v)
    ax.axhline(0, color="#333", lw=0.7)
    ax.axhspan(0.3, 1.0, color="#1f6feb", alpha=0.07)
    ax.axhspan(-1.0, -0.3, color="#b3261e", alpha=0.07)
    ax.set_ylim(-1, 1)
    ax.set_ylabel(f"rolling corr vs BF ({window}-period window)")
    ax.set_title(f"Time-varying correlation NO2 vs BF rate ({freq_label}) — regime changes")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(FIGDIR / f"expl_rolling_{freq_label}.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- correlation matrix


def fig_corr_heatmap(master, freq_label):
    cols = [c for c in master.columns if master[c].notna().sum() > 12]
    corr = master[cols].corr()
    fig, ax = plt.subplots(figsize=(0.7 * len(cols) + 3, 0.7 * len(cols) + 2))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=7)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols, fontsize=7)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(
                j,
                i,
                f"{corr.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                color="white" if abs(corr.iloc[i, j]) > 0.5 else "black",
            )
    fig.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title(f"Correlation matrix at lag 0 ({freq_label})")
    fig.tight_layout()
    fig.savefig(FIGDIR / f"expl_corr_matrix_{freq_label}.png", dpi=130)
    plt.close(fig)
    return corr


# --------------------------------------------------------------------------- economy linkage


def fig_economy_ccf(master, drivers, freq_label, max_lag=8):
    """Lead-lag of steel drivers (BF rate change, NO2 resid) vs each miner's abnormal return."""
    miner_cols = [c for c in master.columns if c.startswith("ret_")]
    if not miner_cols:
        return pd.DataFrame()
    rows = []
    fig, axes = plt.subplots(1, len(drivers), figsize=(5.2 * len(drivers), 4.2), squeeze=False)
    for ax, drv in zip(axes[0], drivers):
        for m in miner_cols:
            d = master[[drv, m]].dropna()
            if len(d) < 12:
                continue
            cc = lead_lag(d[drv], d[m], max_lag=max_lag)
            ax.plot(
                cc.lags, cc.ccf, "o-", ms=3, label=f"{m} (peak {cc.peak_lag:+d}, r={cc.peak_r:.2f})"
            )
            rows.append(
                {"driver": drv, "instrument": m, "best_lag": cc.peak_lag, "r": round(cc.peak_r, 3)}
            )
        ax.axvline(0, color="#333", lw=0.8)
        ax.axhline(0, color="#333", lw=0.5)
        ax.set_xlabel(f"lag (+ve = {drv} LEADS the return)")
        ax.set_title(f"{drv} vs miners ({freq_label})")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGDIR / f"expl_economy_ccf_{freq_label}.png", dpi=130)
    plt.close(fig)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- seasonal climatology


def fig_seasonal(no2_level, bench):
    lvl = no2_level.resample("ME").mean()
    bf = bench.resample("ME").mean()
    df = pd.concat([lvl.rename("no2"), bf.rename("bf")], axis=1).dropna()
    df["month"] = df.index.month
    clim = df.groupby("month").mean()
    fig, ax1 = plt.subplots(figsize=(8, 4.2))
    ax1.bar(clim.index - 0.18, clim["no2"] * 1e3, width=0.36, color="#b3261e", label="NO2 level")
    ax1.set_ylabel("NO2 footprint−bg (×10⁻³)", color="#b3261e")
    ax1.set_xlabel("month")
    ax2 = ax1.twinx()
    ax2.bar(clim.index + 0.18, clim["bf"], width=0.36, color="#1f6feb", label="BF rate")
    ax2.set_ylabel("BF operating rate (%)", color="#1f6feb")
    ax1.axvspan(10.5, 12.5, color="#999", alpha=0.12)
    ax1.axvspan(0.5, 3.5, color="#999", alpha=0.12)
    ax1.set_title("Monthly climatology: NO2 vs BF rate (grey = heating season)")
    fig.tight_layout()
    fig.savefig(FIGDIR / "expl_seasonal.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- findings hunt


def hunt_findings(corr_w, corr_m, ccf_w, ccf_m, econ_w, econ_m):
    lines = ["DEEP EXPLORATION — findings", "=" * 60]

    lines.append("\n[1] Inferred NO2->BF lag (peak |r| cross-correlation):")
    for lab, t in (("weekly", ccf_w), ("monthly", ccf_m)):
        for _, r in t.iterrows():
            lead = (
                "NO2 leads"
                if r["best_lag"] > 0
                else ("BF leads" if r["best_lag"] < 0 else "coincident")
            )
            sig = "SIG" if abs(r["r_at_best"]) > r["sig_band"] else "ns"
            lines.append(
                f"  {lab:7s} {r['variant']:10s} lag0 r={r['r_lag0']:+.2f} | "
                f"best lag {r['best_lag']:+d} r={r['r_at_best']:+.2f} ({lead}, {sig})"
            )

    lines.append("\n[2] Strongest correlations at lag 0 (monthly, |r|>=0.3, excl. self):")
    cm = corr_m.copy()
    np.fill_diagonal(cm.values, np.nan)
    pairs = cm.stack().reset_index().rename(columns={0: "r"}).assign(absr=lambda d: d["r"].abs())
    pairs = pairs[pairs["level_0"] < pairs["level_1"]].sort_values("absr", ascending=False)
    for _, r in pairs[pairs["absr"] >= 0.3].head(15).iterrows():
        lines.append(f"  {r['level_0']:12s} ~ {r['level_1']:12s}  r={r['r']:+.2f}")

    lines.append("\n[3] Economy linkage (steel driver -> miner abnormal return, peak lead-lag):")
    for lab, t in (("weekly", econ_w), ("monthly", econ_m)):
        if len(t):
            for _, r in (
                t.sort_values("r", key=lambda s: s.abs(), ascending=False).head(8).iterrows()
            ):
                who = (
                    "driver leads"
                    if r["best_lag"] > 0
                    else ("return leads" if r["best_lag"] < 0 else "coincident")
                )
                lines.append(
                    f"  {lab:7s} {r['driver']:10s} -> {r['instrument']:8s} lag {r['best_lag']:+d} r={r['r']:+.2f} ({who})"
                )
    return "\n".join(lines) + "\n"


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    no2_level, no2_resid, bench, aux_series, mkt_wide = load_all()
    bench19 = bench[bench.index >= "2019-01-01"]

    mw = build_master("W", 52, no2_level, no2_resid, bench, aux_series, mkt_wide)
    mm = build_master("ME", 12, no2_level, no2_resid, bench, aux_series, mkt_wide)
    no2_variants = ["no2_level", "no2_resid", "no2_yoy", "no2_stl"]

    ccf_w = ccf_table(mw, no2_variants, max_lag=16)
    ccf_m = ccf_table(mm, no2_variants, max_lag=8)
    print("=== weekly NO2->BF lead-lag ===\n", ccf_w.to_string(index=False))
    print("\n=== monthly NO2->BF lead-lag ===\n", ccf_m.to_string(index=False))

    fig_ccf_curves(mw, no2_variants, "weekly", "weeks", max_lag=16)
    fig_ccf_curves(mm, no2_variants, "monthly", "months", max_lag=8)
    if len(ccf_m):
        best = ccf_m.iloc[ccf_m["r_at_best"].abs().idxmax()]
        fig_best_lag_scatter(mm, best["variant"], int(best["best_lag"]), "monthly")
    fig_rolling_corr(mm, ["no2_resid", "no2_yoy"], window=18, freq_label="monthly")
    fig_rolling_corr(mw, ["no2_resid", "no2_yoy"], window=52, freq_label="weekly")
    corr_w = fig_corr_heatmap(mw, "weekly")
    corr_m = fig_corr_heatmap(mm, "monthly")

    # Economy: use BF weekly change and NO2 residual as the steel "drivers".
    mw2 = mw.assign(bf_change=mw["bf"].diff())
    mm2 = mm.assign(bf_change=mm["bf"].diff())
    econ_w = fig_economy_ccf(mw2, ["bf_change", "no2_resid"], "weekly", max_lag=12)
    econ_m = fig_economy_ccf(mm2, ["bf_change", "no2_resid"], "monthly", max_lag=6)
    fig_seasonal(no2_level, bench19)

    findings = hunt_findings(corr_w, corr_m, ccf_w, ccf_m, econ_w, econ_m)
    (FIGDIR / "findings.txt").write_text(findings, encoding="utf-8")
    print("\n" + findings)
    print(f"figures + findings -> {FIGDIR}")


if __name__ == "__main__":
    main()
