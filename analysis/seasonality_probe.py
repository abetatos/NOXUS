"""Can better seasonality removal improve the NO2<->BF coupling? (NOX-004 follow-up).

The deep exploration found the de-trended NO2 still carries a quasi-annual *residual wave* (leftover
seasonality the yoy/STL filters don't fully remove). This probe tests the developer's idea — a
Prophet-style decomposition (flexible trend + **Fourier multi-frequency seasonality**) — implemented
lightweight as **harmonic regression** (no heavy dependency): remove K annual Fourier harmonics from
the intensity residual and see whether the residual<->BF correlation (full-sample AND regime-peak via a
rolling window) improves.

It also states the honest **SNR ceiling**: if steel is only a fraction f of the column *variance*, the
attainable correlation is bounded by ~sqrt(f) — no deseasonalization beats that; only source isolation
(tighter footprint / flux divergence, NOX-005) raises f.

Reproduce: uv run python analysis/seasonality_probe.py  -> docs/figures/exploration/expl_seasonality_*.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.deep_exploration import build_master, load_all
from noxus.signal.index import deseasonalize
from noxus.signal.intensity import fit_intensity_trend
from noxus.validation.leadlag import correlate, lead_lag

FIGDIR = Path("docs/figures/exploration")


def _annual_phase(index: pd.DatetimeIndex) -> np.ndarray:
    """Annual phase in radians from the day-of-year (works for weekly or monthly indices)."""
    doy = pd.DatetimeIndex(index).dayofyear.to_numpy(float)
    return 2.0 * np.pi * doy / 365.25


def remove_harmonics(resid: pd.Series, k: int) -> pd.Series:
    """OLS-remove K annual Fourier harmonics (cos/sin at 1..K cycles/yr) — Prophet's seasonality core."""
    s = resid.astype(float)
    valid = s.notna()
    if int(valid.sum()) < 2 * k + 4 or k == 0:
        return s.copy()
    phase = _annual_phase(s.index[valid])
    cols = [np.ones_like(phase)]
    for j in range(1, k + 1):
        cols += [np.cos(j * phase), np.sin(j * phase)]
    x = np.column_stack(cols)
    y = s[valid].to_numpy()
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    out = s.copy()
    out.loc[valid] = y - x @ coef
    return out


def harmonic_deseason(level: pd.Series, *, k: int, min_length: int) -> pd.Series:
    """Prophet-style: smooth intensity trend (NOX-003.1) + Fourier annual seasonality removed."""
    base = fit_intensity_trend(level, min_length=min_length).residual
    return remove_harmonics(base, k)


def rolling_peak(a: pd.Series, b: pd.Series, window: int) -> float:
    """Max |rolling correlation| (the best regime) — what the static r hides."""
    d = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(d) < window + 4:
        return float("nan")
    rc = d["a"].rolling(window).corr(d["b"]).dropna()
    return float(rc.abs().max()) if len(rc) else float("nan")


def compare(master, level, *, period, min_length, window, label):
    bf = master["bf"]
    methods = {
        "yoy": deseasonalize(master["no2_level"], method="yoy", period=period),
        "stl": deseasonalize(master["no2_level"], method="stl", period=period),
        "intensity": master["no2_resid"],
        "harmonic K1": harmonic_deseason(
            level.resample("ME" if period == 12 else "W").mean(), k=1, min_length=min_length
        ),
        "harmonic K2": harmonic_deseason(
            level.resample("ME" if period == 12 else "W").mean(), k=2, min_length=min_length
        ),
        "harmonic K3": harmonic_deseason(
            level.resample("ME" if period == 12 else "W").mean(), k=3, min_length=min_length
        ),
        "harmonic K4": harmonic_deseason(
            level.resample("ME" if period == 12 else "W").mean(), k=4, min_length=min_length
        ),
    }
    rows = []
    for name, s in methods.items():
        s = s.reindex(bf.index)
        d = pd.concat([s.rename("x"), bf.rename("b")], axis=1).dropna()
        if len(d) < 12:
            continue
        cr = correlate(d["x"], d["b"])
        cc = lead_lag(d["x"], d["b"], max_lag=6 if period == 12 else 16)
        rows.append(
            {
                "method": name,
                "n": cr.n,
                "r_lag0": round(cr.pearson_r, 3),
                "best_lag": cc.peak_lag,
                "r_best": round(cc.peak_r, 3),
                "rolling_peak": round(rolling_peak(d["x"], d["b"], window), 3),
            }
        )
    return pd.DataFrame(rows)


def snr_ceiling(rolling_peak_best: float) -> str:
    """Honest SNR note: r <= sqrt(variance share of steel); 40% share -> ~0.63 ceiling."""
    f_levels = 0.40  # Wen 2024 mean steel share of the Tangshan column (mid of 30-43%)
    implied_f = rolling_peak_best**2 if rolling_peak_best == rolling_peak_best else float("nan")
    return (
        f"SNR ceiling: with steel ~{f_levels:.0%} of the column, attainable r ~ sqrt(variance share). "
        f"If the variance share equals the mean share (~0.40) the ceiling is ~{f_levels**0.5:.2f}. "
        f"The best regime rolling-peak r={rolling_peak_best:.2f} implies a variance share of "
        f"~{implied_f:.0%} in that window — i.e. steel drives more of the VARIANCE than of the MEAN "
        "when it is the swing factor (curtailments), which is exactly when the marker should work."
    )


def fig_compare(cmp_m, cmp_w):
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    x = np.arange(len(cmp_m))
    w = 0.38
    ax.bar(x - w / 2, cmp_m["r_best"].abs(), w, color="#1a7f37", label="monthly |best-lag r|")
    ax.bar(
        x + w / 2,
        cmp_m["rolling_peak"],
        w,
        color="#1f6feb",
        label="monthly rolling-peak |r| (best regime)",
    )
    ax.axhspan(0.5, 0.75, color="#9a6700", alpha=0.12, label="literature bar")
    ax.axhline(0.40**0.5, color="#b3261e", ls="--", lw=1.1, label="≈SNR ceiling (40% share)")
    ax.set_xticks(x)
    ax.set_xticklabels(cmp_m["method"], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("|Pearson r| vs BF rate")
    ax.set_title("Deseasonalisation method vs coupling (monthly): static best-lag vs best-regime")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "expl_seasonality_compare.png", dpi=130)
    plt.close(fig)


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    no2_level, no2_resid, bench, aux_series, mkt_wide = load_all()
    mm = build_master("ME", 12, no2_level, no2_resid, bench, aux_series, mkt_wide)
    mw = build_master("W", 52, no2_level, no2_resid, bench, aux_series, mkt_wide)

    cmp_m = compare(mm, no2_level, period=12, min_length=18, window=18, label="monthly")
    cmp_w = compare(mw, no2_level, period=52, min_length=24, window=52, label="weekly")
    print("=== monthly: deseason method vs BF ===\n", cmp_m.to_string(index=False))
    print("\n=== weekly: deseason method vs BF ===\n", cmp_w.to_string(index=False))

    fig_compare(cmp_m, cmp_w)
    best = float(cmp_m["rolling_peak"].max())
    note = snr_ceiling(best)
    print("\n" + note)
    (FIGDIR / "seasonality_findings.txt").write_text(
        cmp_m.to_string(index=False) + "\n\n" + cmp_w.to_string(index=False) + "\n\n" + note + "\n",
        encoding="utf-8",
    )
    print(f"\nfigure + findings -> {FIGDIR}")


if __name__ == "__main__":
    main()
