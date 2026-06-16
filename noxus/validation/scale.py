"""Spatial-scale sensitivity sweep of the NO2<->steel signal (NOX-008, REQ-010/011/040/041).

Sweeps two scale axes over the EXISTING analysis-ready cube — AOI extent (buffer 0.25 vs 0.10 deg,
``clip_cube_to_region``) and grid resolution (native vs block-averaged 0.10/0.15/0.25 deg,
``coarsen_cube``) — and at each scale re-derives the NO2 activity signal through the *existing*
footprint/deseasonalise pathway, aligns it to the CREA BF operating rate, and judges the correlation
with autocorrelation-robust, FDR-corrected significance (``noxus.validation.robust``).

The result is a tidy table (one row per extent x resolution x variant x lag-choice) that shows whether
the weak/null coupling is an artefact of one fixed scale — testing the Parubets & Naito (2025) scale
warning and the source-isolation intuition (a tighter AOI excludes Beijing-Tianjin-Hebei background).
Aggregation only, never interpolation (native-resolution decision, 2026-06-13). Honest null is valid.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from noxus.config.region import tangshan_aoi
from noxus.config.run import ScaleSweepConfig, SignalConfig
from noxus.data.gridding import COVERAGE, _native_spacing, clip_cube_to_region, coarsen_cube
from noxus.data.tropomi import NO2
from noxus.attribution.source import (
    GeometryError,
    background_ring,
    footprint_mask,
    footprint_signal,
    load_facilities,
)
from noxus.signal.index import deseasonalize
from noxus.validation.leadlag import lead_lag
from noxus.validation.robust import bh_fdr, block_length, robust_corr

# Variant name -> deseasonalize method ("level" is the raw signal, i.e. method "none").
_VARIANT_METHOD = {
    "level": "none",
    "intensity-model": "intensity-model",
    "yoy": "yoy",
    "stl": "stl",
}
_PERIOD = {"W": 52, "ME": 12}


@dataclass(frozen=True)
class ScaleSignal:
    """The NO2 signal re-derived at one (extent x resolution) scale."""

    series: pd.Series  # date-indexed NO2 signal at the cube's native frequency
    label: str  # "footprint" | "aoi-mean-fallback"
    n_cells: int  # footprint cells (or AOI cells when fallback)
    realised_deg: float  # realised cell spacing after coarsening


def _benchmark_series(path: Path) -> pd.Series:
    df = pd.read_parquet(path)
    s = pd.Series(df["value"].to_numpy(float), index=pd.DatetimeIndex(pd.to_datetime(df["date"])))
    return s[~s.index.duplicated(keep="last")].sort_index()


def _aoi_mean(cube: xr.Dataset) -> pd.Series:
    spatial = [d for d in cube[NO2].dims if d != "time"]
    m = cube[NO2].mean(dim=spatial, skipna=True)
    return pd.Series(
        m.values, index=pd.DatetimeIndex(pd.to_datetime(cube["time"].values))
    ).sort_index()


def derive_scale_signal(
    cube: xr.Dataset, buffer: float, resolution, facilities: pd.DataFrame, cfg: ScaleSweepConfig
) -> ScaleSignal:
    """Clip + coarsen the cube to (extent, resolution), then re-derive the footprint NO2 signal.

    Falls back to the AOI spatial mean (labelled) when the coarsened scale leaves too few cells for a
    footprint/background contrast, or the ring geometry is degenerate (REQ-011, EDGE-002).
    """
    region = tangshan_aoi(buffer)
    scaled = clip_cube_to_region(cube, region)
    if resolution != "native":
        scaled = coarsen_cube(scaled, float(resolution))
    lon_name = "x" if "x" in scaled.coords else "lon"
    realised = _native_spacing(scaled, lon_name)

    try:
        fp = footprint_mask(scaled, facilities, SignalConfig().footprint_radius_km)
        n_cells = int(fp.values.sum())
        if n_cells < cfg.min_footprint_cells:
            raise GeometryError(f"only {n_cells} footprint cells (< {cfg.min_footprint_cells})")
        sig_cfg = SignalConfig()
        bg = background_ring(
            scaled,
            fp,
            sig_cfg.background_inner_km,
            sig_cfg.background_outer_km,
            facilities=facilities,
        )
        df = footprint_signal(scaled, fp, bg, mode=sig_cfg.background_mode)
        series = pd.Series(
            df["no2_corrected"].to_numpy(float), index=pd.DatetimeIndex(df["date"])
        ).sort_index()
        return ScaleSignal(series=series, label="footprint", n_cells=n_cells, realised_deg=realised)
    except GeometryError:
        spatial = [d for d in scaled[NO2].dims if d != "time"]
        n_cells = (
            int(np.isfinite(scaled[NO2].isel(time=0)).sum()) if scaled.sizes.get("time") else 0
        )
        _ = spatial
        return ScaleSignal(
            series=_aoi_mean(scaled),
            label="aoi-mean-fallback",
            n_cells=n_cells,
            realised_deg=realised,
        )


def _variant_series(signal: pd.Series, variant: str, freq: str) -> pd.Series:
    """Resample the signal to ``freq`` and apply the variant's deseasonalisation."""
    period = _PERIOD[freq]
    s = signal.resample(freq).mean()
    method = _VARIANT_METHOD[variant]
    if method == "none":
        return s
    return deseasonalize(s, method=method, period=period, cfg=SignalConfig())


def scale_sweep(
    cube: xr.Dataset, benchmark: pd.Series, cfg: ScaleSweepConfig | None = None
) -> pd.DataFrame:
    """Run the full extent x resolution sweep and return the tidy, FDR-corrected result table.

    One row per (buffer, resolution, freq, variant, lag_kind) where ``lag_kind`` is ``lag0`` or
    ``peak`` (peak-|r| lead-lag). FDR is applied separately within each ``lag_kind`` family on the
    block-permutation p (Q3 default: two families, never pooled to inflate discoveries).
    """
    cfg = cfg or ScaleSweepConfig()
    facilities = load_facilities(cfg.facilities_csv)
    rows: list[dict] = []

    for buffer in cfg.buffers:
        for resolution in cfg.resolutions:
            sig = derive_scale_signal(cube, buffer, resolution, facilities, cfg)
            for freq in cfg.freqs:
                period = _PERIOD[freq]
                bench_f = benchmark.resample(freq).mean()
                for variant in cfg.variants:
                    try:
                        v = _variant_series(sig.series, variant, freq)
                    except Exception:  # noqa: BLE001 - a variant that can't fit at this scale is skipped
                        continue
                    paired = pd.concat({"v": v, "b": bench_f}, axis=1).dropna()
                    if len(paired) < cfg.min_overlap:
                        continue
                    base = dict(
                        buffer=buffer,
                        resolution=str(resolution),
                        realised_deg=round(sig.realised_deg, 4),
                        signal=sig.label,
                        n_cells=sig.n_cells,
                        freq=freq,
                        variant=variant,
                        period=period,
                    )
                    # lag 0
                    rows.append(
                        _row(base, paired["v"], paired["b"], lag=0, lag_kind="lag0", cfg=cfg)
                    )
                    # peak-|r| lag (variant LEADS benchmark for +lag)
                    cc = lead_lag(paired["v"], paired["b"], max_lag=cfg.max_lag)
                    if cc.peak_lag != 0:
                        shifted = paired["v"].shift(cc.peak_lag)
                        pk = pd.concat({"v": shifted, "b": paired["b"]}, axis=1).dropna()
                        if len(pk) >= cfg.min_overlap:
                            rows.append(
                                _row(
                                    base,
                                    pk["v"],
                                    pk["b"],
                                    lag=cc.peak_lag,
                                    lag_kind="peak",
                                    cfg=cfg,
                                )
                            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # FDR within each lag_kind family (Q3 default), on the headline robust (permutation) p.
    df["p_fdr"] = np.nan
    df["fdr_reject"] = False
    for kind, idx in df.groupby("lag_kind").groups.items():
        rej, adj = bh_fdr(df.loc[idx, "p_perm"].to_numpy(), alpha=cfg.fdr_alpha)
        df.loc[idx, "p_fdr"] = adj
        df.loc[idx, "fdr_reject"] = rej
        _ = kind
    df["verdict"] = df.apply(_verdict, axis=1)
    return df.reset_index(drop=True)


def _row(
    base: dict, x: pd.Series, b: pd.Series, *, lag: int, lag_kind: str, cfg: ScaleSweepConfig
) -> dict:
    rc = robust_corr(
        x, b, n_boot=cfg.n_boot, n_perm=cfg.n_perm, seed=cfg.seed, order=cfg.neff_order
    )
    return {
        **base,
        "lag": lag,
        "lag_kind": lag_kind,
        "n": rc.n,
        "block": rc.block if rc.block else block_length(rc.n),
        "r": round(rc.r, 4),
        "p_naive": rc.p_naive,
        "n_eff_first": round(rc.n_eff_first, 1),
        "p_eff_first": rc.p_eff_first,
        "n_eff_nw": round(rc.n_eff_nw, 1),
        "p_eff_nw": rc.p_eff_nw,
        "boot_lo": round(rc.boot_lo, 4),
        "boot_hi": round(rc.boot_hi, 4),
        "p_perm": rc.p_perm,
    }


def _verdict(r: pd.Series) -> str:
    """Robust only if the FDR-adjusted permutation p clears alpha AND the bootstrap CI excludes 0."""
    ci_excludes_0 = not (np.isnan(r["boot_lo"]) or (r["boot_lo"] <= 0.0 <= r["boot_hi"]))
    robust = bool(r["fdr_reject"]) and ci_excludes_0
    naive_sig = bool(r["p_naive"] < 0.05)
    if robust:
        return "robust"
    if naive_sig:
        return "fragile (naive-only)"
    return "ns"


def run_scale_sweep(cfg: ScaleSweepConfig | None = None) -> Path:
    """End-to-end: load the cube + benchmark, run the sweep, write the tidy CSV. Returns its path."""
    cfg = cfg or ScaleSweepConfig()
    cube_path = Path(cfg.cube_path)
    if not cube_path.exists():
        raise FileNotFoundError(
            f"NO2 cube not found: {cube_path}. Run 'noxus grid' first to build the weekly cube (ERR-001)."
        )
    cube = xr.open_dataset(cube_path).load()
    cube.close()
    if COVERAGE not in cube and NO2 not in cube:  # pragma: no cover - defensive
        raise ValueError(f"Cube {cube_path} has no '{NO2}' variable; cannot run the sweep.")
    benchmark = _benchmark_series(Path(cfg.benchmark_path))
    df = scale_sweep(cube, benchmark, cfg)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / cfg.results_name
    df.to_csv(out_path, index=False)
    return out_path
