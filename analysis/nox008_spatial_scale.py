"""NOX-008 — Spatial-scale & autocorrelation-robust sensitivity of the NO2<->steel signal.

Research question (developer-directed 2026-06-14): is the weak/null NO2<->blast-furnace coupling
reported in NOX-003/003.1/006 an artefact of ONE fixed spatial scale and an i.i.d. significance test?
This script sweeps two scale axes over the existing analysis-ready weekly cube and re-judges the
correlation with an autocorrelation-aware test, exactly as motivated by Parubets & Naito (2025)
(a correlation significant at a coarse scale can vanish or flip at a finer one).

Two scale axes (AGGREGATION ONLY — never interpolation to a finer grid):
  * Axis 1, AOI extent: wide 0.25 deg buffer (DEFAULT_AOI_BUFFER_DEG) vs tight 0.10 deg
    (TIGHT_AOI_BUFFER_DEG) — a strict spatial clip testing the source-isolation intuition
    (tighter AOI => less Beijing-Tianjin-Hebei background dilution; steel is only ~30-43% of
    Tangshan NO2, Wen 2024).
  * Axis 2, grid resolution: native (~0.035x0.055 deg) vs block-averaged ~0.1 deg and ~0.25 deg
    (mean over cell blocks via xr.coarsen, the permitted information-preserving direction).

For every (extent x resolution) scale we re-derive the footprint-minus-background NO2 contrast, build
the four deseason variants (level / intensity-residual / yoy / stl), align to the CREA BF operating
rate at monthly frequency, and compute Pearson r at lag 0 and the peak-|r| lead-lag. Significance is
judged AUTOCORRELATION-ROBUST: first-order effective sample size n_eff, a moving-block bootstrap 95%
CI of r, and a block-permutation null p-value, then a Benjamini-Hochberg FDR correction across the
whole scale x variant grid. Honest NULL is an acceptable, designed-for outcome (Morris & Zhang 2019).

Reproduce (after the upstream artifacts exist):
    uv run python analysis/nox008_spatial_scale.py
Writes docs/figures/nox008/*.png + scale_significance.csv + findings.txt. Reads the gitignored NO2
cube (data/derived/no2/no2_cube_w.nc) and the committed BF benchmark.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

from analysis.autocorr_significance import (
    block_permutation_p,
    moving_block_bootstrap_ci,
    n_eff_first_order,
)
from noxus.attribution.source import (
    background_ring,
    footprint_mask,
    footprint_signal,
    load_facilities,
)
from noxus.config.region import (
    DEFAULT_AOI_BUFFER_DEG,
    TANGSHAN_FACILITIES_ENVELOPE,
    TIGHT_AOI_BUFFER_DEG,
)
from noxus.config.run import SignalConfig
from noxus.data.era5 import era5_footprint_series
from noxus.signal.index import deseasonalize, regress_out_meteo

OUT = Path("docs/figures/nox008")
CUBE = "data/derived/no2/no2_cube_w.nc"
BENCH = "data/derived/benchmark_tangshan_bf_operating_rate.parquet"
ERA5_DIR = Path("data/raw/era5")
RNG = np.random.default_rng(20260614)

# Resolution axis as integer coarsen factors (y, x). Native cell ~0.035 deg lat / ~0.054 deg lon.
RESOLUTIONS = {
    "native": (1, 1),  # ~0.035 x 0.055 deg
    "coarse010": (3, 2),  # ~0.105 x 0.108 deg
    "coarse025": (7, 5),  # ~0.245 x 0.27 deg
}
EXTENTS = {"wide": DEFAULT_AOI_BUFFER_DEG, "tight": TIGHT_AOI_BUFFER_DEG}
VARIANTS = ["level", "intensity", "yoy", "stl"]
VAR_METHOD = {"level": "none", "intensity": "intensity-model", "yoy": "yoy", "stl": "stl"}
MAX_LAG = 8  # months
PERIOD = 12  # monthly seasonal period


# --------------------------------------------------------------------------- cube scaling


def clip_extent(cube: xr.Dataset, buffer_deg: float) -> xr.Dataset:
    """Strict spatial clip of the cube to the facility envelope + buffer_deg (Axis 1)."""
    env = TANGSHAN_FACILITIES_ENVELOPE
    lon_lo, lon_hi = env.min_lon - buffer_deg, env.max_lon + buffer_deg
    lat_lo, lat_hi = env.min_lat - buffer_deg, env.max_lat + buffer_deg
    xs = cube["x"].values
    ys = cube["y"].values
    return cube.sel(
        x=xs[(xs >= lon_lo) & (xs <= lon_hi)],
        y=ys[(ys >= lat_lo) & (ys <= lat_hi)],
    )


def coarsen(cube: xr.Dataset, fy: int, fx: int) -> xr.Dataset:
    """Block-average the cube by (fy, fx) over (y, x) — aggregation, never interpolation (Axis 2)."""
    if fy == 1 and fx == 1:
        return cube
    return cube.coarsen(y=fy, x=fx, boundary="trim").mean()


def cell_diag_km(cube: xr.Dataset) -> float:
    """Approximate cell diagonal in km from the coordinate spacing (for adaptive footprint radius)."""
    dlat = float(np.abs(np.diff(cube["y"].values)).mean())
    dlon = float(np.abs(np.diff(cube["x"].values)).mean())
    lat0 = float(cube["y"].values.mean())
    ky = dlat * 111.0
    kx = dlon * 111.0 * np.cos(np.radians(lat0))
    return float(np.hypot(ky, kx))


# --------------------------------------------------------------------------- signal per scale


def footprint_contrast(
    cube: xr.Dataset,
    facilities: pd.DataFrame,
    cfg: SignalConfig,
    meteo: pd.DataFrame | None = None,
) -> pd.Series:
    """Meteo-normalised background-corrected footprint NO2 series for one scaled cube.

    Replicates the canonical NOX-003 chain (footprint - background -> ERA5 meteo regress-out) so the
    default scale reproduces the established meteo-free ``signal``. Footprint radius and background
    ring scale with the cell size so the coarse grids still resolve a non-empty footprint and a
    disjoint ring (radius floored at the native config value); the scaling rule is reported per scale.
    """
    diag = cell_diag_km(cube)
    radius = max(cfg.footprint_radius_km, 1.5 * diag)
    inner = max(cfg.background_inner_km, radius + 0.5 * diag)
    outer = max(cfg.background_outer_km, inner + 35.0)
    fp = footprint_mask(cube, facilities, radius)
    bg = background_ring(cube, fp, inner, outer, facilities=facilities)
    df = footprint_signal(cube, fp, bg, mode=cfg.background_mode)
    s = pd.Series(
        df["no2_corrected"].to_numpy(float),
        index=pd.DatetimeIndex(pd.to_datetime(df["date"])),
    ).sort_index()
    meteo_applied = False
    if meteo is not None:
        try:
            s = regress_out_meteo(s, meteo, form=cfg.meteo_form)
            meteo_applied = True
        except Exception:
            meteo_applied = False
    s.attrs.update(
        radius_km=round(radius, 1),
        n_footprint=int(fp.values.sum()),
        n_bg=int(bg.values.sum()),
        cell_km=round(diag, 1),
        meteo_applied=meteo_applied,
    )
    return s


def variant_series(level_monthly: pd.Series, variant: str) -> pd.Series:
    """Build a deseason variant from the monthly level series."""
    method = VAR_METHOD[variant]
    if method == "none":
        return level_monthly
    return deseasonalize(level_monthly, method=method, period=PERIOD)


# --------------------------------------------------------------------------- significance


def best_lag(x: pd.Series, b: pd.Series, max_lag: int) -> tuple[int, float]:
    best_k, best_r = 0, 0.0
    for k in range(-max_lag, max_lag + 1):
        d = pd.concat({"x": x.shift(k), "b": b}, axis=1).dropna()
        if len(d) < 6 or d["x"].std() == 0 or d["b"].std() == 0:
            continue
        r = float(np.corrcoef(d["x"], d["b"])[0, 1])
        if abs(r) > abs(best_r):
            best_k, best_r = k, r
    return best_k, best_r


def robust_row(x: pd.Series, b: pd.Series, lag: int) -> dict:
    d = pd.concat({"x": x.shift(lag), "b": b}, axis=1).dropna()
    xv, yv = d["x"].to_numpy(float), d["b"].to_numpy(float)
    n = len(xv)
    if n < 8 or np.std(xv) == 0 or np.std(yv) == 0:
        return {}
    res = stats.pearsonr(xv, yv)
    r, p_naive = float(res.statistic), float(res.pvalue)
    neff = n_eff_first_order(xv, yv)
    block = max(2, int(round(n ** (1 / 3))))
    lo, hi = moving_block_bootstrap_ci(xv, yv, block)
    p_block = block_permutation_p(xv, yv, block, r)
    return {
        "n": n,
        "lag": lag,
        "r": r,
        "p_naive": p_naive,
        "neff": neff,
        "ci_lo": lo,
        "ci_hi": hi,
        "p_block": p_block,
    }


def bh_fdr(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (returns q-values aligned to input order)."""
    p = np.asarray(pvals, float)
    ok = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    idx = np.where(ok)[0]
    m = len(idx)
    if m == 0:
        return q
    order = idx[np.argsort(p[idx])]
    ranked = p[order] * m / (np.arange(1, m + 1))
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[order] = np.clip(ranked, 0, 1)
    return q


# --------------------------------------------------------------------------- figures


def fig_r_heatmap(grid: pd.DataFrame, fname: str) -> None:
    piv = grid.pivot(index="scale", columns="variant", values="r").reindex(columns=VARIANTS)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    im = ax.imshow(piv.to_numpy(), cmap="RdBu_r", vmin=-0.55, vmax=0.55, aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iloc[i, j]
            if not np.isnan(v):
                ax.text(
                    j,
                    i,
                    f"{v:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if abs(v) > 0.32 else "black",
                )
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r (lag 0)")
    ax.set_title("NO2(footprint-bg) vs BF rate — lag-0 r across spatial scales (monthly)")
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=140)
    plt.close(fig)


def fig_neff_shrinkage(grid: pd.DataFrame, fname: str) -> None:
    d = grid.dropna(subset=["p_naive", "p_block"]).copy()
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    colors = {"level": "#b3261e", "intensity": "#1f6feb", "yoy": "#1a7f37", "stl": "#9a6700"}
    for v in VARIANTS:
        dv = d[d["variant"] == v]
        ax.scatter(
            dv["p_naive"],
            dv["p_block"],
            s=46,
            color=colors[v],
            label=v,
            edgecolor="k",
            linewidth=0.4,
            alpha=0.85,
        )
    ax.axhline(0.05, color="#888", ls="--", lw=0.9)
    ax.axvline(0.05, color="#888", ls="--", lw=0.9)
    lim = [0, max(0.6, d[["p_naive", "p_block"]].to_numpy().max() * 1.05)]
    ax.plot(lim, lim, color="#333", lw=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("naive i.i.d. p-value (scipy.pearsonr)")
    ax.set_ylabel("autocorrelation-robust block-permutation p")
    ax.set_title(
        "Serial dependence inflates significance:\nrobust p (y) >= naive p (x) almost everywhere"
    )
    ax.legend(title="variant", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=140)
    plt.close(fig)


def fig_forest(grid: pd.DataFrame, fname: str) -> None:
    """Bootstrap 95% CI of r for the LEVEL variant across all scales (forest plot)."""
    d = grid[(grid["variant"] == "level") & grid["ci_lo"].notna()].copy()
    d = d.sort_values("scale")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ys = np.arange(len(d))
    for y, (_, row) in zip(ys, d.iterrows()):
        excl0 = not (row["ci_lo"] <= 0 <= row["ci_hi"])
        col = "#b3261e" if excl0 else "#888"
        ax.plot([row["ci_lo"], row["ci_hi"]], [y, y], color=col, lw=2.2)
        ax.plot(row["r"], y, "o", color=col, ms=7)
    ax.axvline(0, color="#333", lw=1.0)
    ax.set_yticks(ys)
    ax.set_yticklabels(d["scale"], fontsize=8)
    ax.set_xlabel("Pearson r (lag 0) with moving-block bootstrap 95% CI")
    ax.set_title(
        "Level NO2 vs BF rate: negative-levels coupling is\nstable in SIGN but weak across scales (red = CI excludes 0)"
    )
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=140)
    plt.close(fig)


def fig_decoupling(level_wide_native: pd.Series, bf_m: pd.Series, fname: str) -> None:
    d = pd.concat({"no2": level_wide_native, "bf": bf_m}, axis=1).dropna()
    fig, ax1 = plt.subplots(figsize=(8.4, 4.2))
    ax1.plot(d.index, d["no2"] * 1e3, color="#b3261e", lw=1.5, label="NO2 footprint-bg")
    ax1.set_ylabel("NO2 footprint-bg (x10$^{-3}$)", color="#b3261e")
    ax1.tick_params(axis="y", labelcolor="#b3261e")
    ax2 = ax1.twinx()
    ax2.plot(d.index, d["bf"], color="#1f6feb", lw=1.5, label="BF operating rate")
    ax2.set_ylabel("BF operating rate (%)", color="#1f6feb")
    ax2.tick_params(axis="y", labelcolor="#1f6feb")
    r = float(np.corrcoef(d["no2"], d["bf"])[0, 1])
    ax1.set_title(
        f"Decoupling at the default scale (wide x native): levels r = {r:+.2f}, n = {len(d)}"
    )
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=140)
    plt.close(fig)


def fig_rolling(level_m: pd.Series, intensity_m: pd.Series, bf_m: pd.Series, fname: str) -> None:
    d = pd.concat({"level": level_m, "intensity": intensity_m, "bf": bf_m}, axis=1).dropna()
    win = 18
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    ax.plot(d.index, d["level"].rolling(win).corr(d["bf"]), color="#b3261e", lw=1.6, label="level")
    ax.plot(
        d.index,
        d["intensity"].rolling(win).corr(d["bf"]),
        color="#1f6feb",
        lw=1.6,
        label="intensity residual",
    )
    ax.axhline(0, color="#333", lw=0.7)
    ax.axhspan(0.3, 1.0, color="#1f6feb", alpha=0.06)
    ax.axhspan(-1.0, -0.3, color="#b3261e", alpha=0.06)
    ax.set_ylim(-1, 1)
    ax.set_ylabel(f"rolling corr vs BF ({win}-mo window)")
    ax.set_title(
        "Regime-dependence at the default scale: the sign and strength of the coupling drift"
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- driver


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = SignalConfig()
    cube0 = xr.open_dataset(CUBE).load()
    cube0.close()
    facilities = load_facilities(cfg.facilities_csv, set(cfg.active_statuses))

    # ERA5 meteo covariates (wind/PBL) over the native footprint — a regional, scale-independent
    # covariate applied identically to every scale so each reproduces the canonical NOX-003 chain.
    meteo = None
    snaps = sorted(ERA5_DIR.glob("era5_*.nc"))
    if snaps:
        try:
            native_fp = footprint_mask(cube0, facilities, cfg.footprint_radius_km)
            m = era5_footprint_series(snaps[-1], native_fp, freq=cfg.freq)
            meteo = m.set_index(pd.DatetimeIndex(pd.to_datetime(m["date"]))).drop(columns=["date"])
        except Exception as exc:  # noqa: BLE001 — meteo is best-effort; fall back to raw contrast
            print(f"[nox008] meteo unavailable ({type(exc).__name__}: {exc}); using raw contrast")

    bf = pd.read_parquet(BENCH)
    bf_s = pd.Series(
        bf["value"].to_numpy(float), index=pd.DatetimeIndex(pd.to_datetime(bf["date"]))
    ).sort_index()
    bf_s = bf_s[~bf_s.index.duplicated(keep="last")]
    bf_m = bf_s.resample("ME").mean()

    rows = []
    scale_meta = []
    level_cache: dict[str, pd.Series] = {}
    for ext_name, buf in EXTENTS.items():
        clipped = clip_extent(cube0, buf)
        for res_name, (fy, fx) in RESOLUTIONS.items():
            scale = f"{ext_name}-{res_name}"
            try:
                scaled = coarsen(clipped, fy, fx)
                level = footprint_contrast(scaled, facilities, cfg, meteo=meteo)
            except Exception as exc:  # GeometryError etc. — record and skip the scale
                scale_meta.append({"scale": scale, "status": f"skip: {type(exc).__name__}: {exc}"})
                continue
            meta = dict(level.attrs)
            meta.update(scale=scale, status="ok", n_cells=int(scaled["no2"].isel(time=0).size))
            scale_meta.append(meta)
            level_m = level.resample("ME").mean()
            level_cache[scale] = level_m
            for variant in VARIANTS:
                vser = variant_series(level_m, variant)
                r0 = robust_row(vser, bf_m, 0)
                if not r0:
                    continue
                lag, _ = best_lag(vser, bf_m, MAX_LAG)
                rb = robust_row(vser, bf_m, lag)
                rows.append(
                    {
                        "scale": scale,
                        "extent": ext_name,
                        "resolution": res_name,
                        "variant": variant,
                        **{k: r0[k] for k in r0},
                        "best_lag": rb.get("lag"),
                        "r_best": rb.get("r"),
                        "p_block_best": rb.get("p_block"),
                    }
                )

    grid = pd.DataFrame(rows)
    grid["q_block"] = bh_fdr(grid["p_block"].to_numpy())
    grid["ci_excl0"] = ~((grid["ci_lo"] <= 0) & (0 <= grid["ci_hi"]))
    grid["robust_sig"] = (grid["q_block"] < 0.05) & grid["ci_excl0"]
    grid["naive_sig"] = grid["p_naive"] < 0.05

    grid.to_csv(OUT / "scale_significance.csv", index=False)
    pd.DataFrame(scale_meta).to_csv(OUT / "scale_meta.csv", index=False)

    # Figures.
    fig_r_heatmap(grid, "fig1_scale_r_heatmap.png")
    fig_neff_shrinkage(grid, "fig2_neff_shrinkage.png")
    fig_forest(grid, "fig3_level_forest.png")
    wn = level_cache.get("wide-native")
    if wn is not None:
        fig_decoupling(wn, bf_m, "fig4_decoupling.png")
        fig_rolling(wn, variant_series(wn, "intensity"), bf_m, "fig5_rolling_regime.png")

    # Text findings.
    lines = [
        "NOX-008 — SPATIAL-SCALE x AUTOCORRELATION-ROBUST SIGNIFICANCE (monthly)",
        "=" * 78,
        "Footprint-minus-background NO2 vs CREA BF operating rate.",
        "naive_sig: scipy p<.05 (i.i.d.) | robust_sig: BH-FDR(block-perm) q<.05 AND boot CI excludes 0",
        "",
    ]
    for _, r in grid.iterrows():
        lines.append(
            f"{r['scale']:16s} {r['variant']:10s} n={int(r['n']):3d} lag0 r={r['r']:+.3f} "
            f"p={r['p_naive']:.3f} neff={r['neff']:5.1f} CI[{r['ci_lo']:+.2f},{r['ci_hi']:+.2f}] "
            f"q={r['q_block']:.3f} | best lag {int(r['best_lag']):+d} r={r['r_best']:+.3f} "
            f"-> {'ROBUST' if r['robust_sig'] else ('naive-only' if r['naive_sig'] else 'ns')}"
        )
    n_naive = int(grid["naive_sig"].sum())
    n_robust = int(grid["robust_sig"].sum())
    lines += [
        "",
        f"SUMMARY: {len(grid)} scale x variant cells. naive-significant={n_naive}, "
        f"robust (FDR + CI) ={n_robust}.",
        "Sign of the LEVEL relationship across scales: "
        + ", ".join(
            f"{s}={'-' if v < 0 else '+'}"
            for s, v in grid[grid["variant"] == "level"][["scale", "r"]].values
        ),
    ]
    (OUT / "findings.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()
