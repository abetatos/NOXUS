"""Meteo regress-out, deseasonalisation, and the relative steel-activity index (NOX-003).

This module turns the background-corrected footprint signal (``noxus.attribution.source``) into a
**relative, unitless** activity index, in three configurable, recorded steps:

1. ``regress_out_meteo`` (REQ-011) — residualise the dominant ventilation confounder (ERA5 wind /
   PBL) out of the signal. Linear OLS residualisation is the default; a LOESS-style local filter is
   the optional alternative. The form is recorded so the run is reproducible and the choice is not
   hidden (researcher-degrees-of-freedom discipline).
2. ``deseasonalize`` (REQ-020/021/022) — remove seasonality (default year-over-year double
   differencing, Kondragunta 2021) while modelling the heating/non-heating season as an explicit
   **structural** covariate (the steel share of Tangshan NO2 shifts 42.5% → 29.1% across the heating
   season, Wen 2024 — so it is regressed, not merely differenced away) and curtailment periods as an
   exogenous control (Li 2024).
3. ``build_index`` (REQ-030/031) — normalise to a relative index (z-score or anchor-period base),
   **never** an absolute tonnage, and record the attributable-fraction cap (~0.30–0.43, Wen 2024).

All steps preserve NaN gaps from cloud screening — they are never interpolated (REQ-004). The index
is emitted to ``steel_activity_index.parquet`` with full provenance in the parquet/Arrow metadata.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- meteo regress-out (T6)


def regress_out_meteo(signal: pd.Series, meteo: pd.DataFrame, form: str = "linear") -> pd.Series:
    """Residualise the meteorological covariates out of ``signal`` (REQ-011).

    ``signal`` is the background-corrected footprint series indexed by date; ``meteo`` is a DataFrame
    of ERA5 covariates (e.g. ``u10``, ``v10``, ``blh``, ``wind_speed``) on the same index. The return
    is the meteorology-free residual on the same index, with the regression ``form`` recorded in
    ``Series.attrs['meteo_form']`` and the covariates used in ``Series.attrs['meteo_covariates']``.

    ``form="linear"`` (default): OLS of signal on the covariates (with intercept); the residual is
    ``signal - fitted``. ``form="loess"``: a LOESS local-regression of signal on the single dominant
    covariate (wind speed), residualising the smooth meteorological response. Rows where the signal or
    any used covariate is NaN are left as NaN in the residual (no interpolation, REQ-004).
    """
    s = signal.astype(float)
    cov = meteo.reindex(s.index)
    cov = cov.select_dtypes(include=[np.number])
    if cov.shape[1] == 0:
        raise ValueError("regress_out_meteo: no numeric meteorological covariates supplied.")

    used = list(cov.columns)
    valid = s.notna() & cov.notna().all(axis=1)
    resid = pd.Series(np.nan, index=s.index, name="signal_meteo_free")

    if int(valid.sum()) >= cov.shape[1] + 2:
        if form == "linear":
            resid.loc[valid] = _linear_residual(s[valid], cov.loc[valid])
        elif form == "loess":
            resid.loc[valid] = _loess_residual(s[valid], cov.loc[valid])
        else:
            raise ValueError(f"Unknown meteo regression form {form!r} (use 'linear'/'loess').")
    else:
        # Too few points to fit; pass the signal through unchanged on valid rows (recorded form).
        resid.loc[valid] = s[valid].to_numpy()

    resid.attrs["meteo_form"] = form
    resid.attrs["meteo_covariates"] = used
    return resid


def _linear_residual(y: pd.Series, x: pd.DataFrame) -> np.ndarray:
    """OLS residual of ``y`` on ``x`` (with intercept) via statsmodels."""
    import statsmodels.api as sm

    design = sm.add_constant(x.to_numpy(), has_constant="add")
    model = sm.OLS(y.to_numpy(), design).fit()
    return model.resid


def _loess_residual(y: pd.Series, x: pd.DataFrame) -> np.ndarray:
    """LOESS residual of ``y`` on the dominant covariate (wind speed if present, else the first)."""
    from statsmodels.nonparametric.smoothers_lowess import lowess

    driver = "wind_speed" if "wind_speed" in x.columns else x.columns[0]
    xv = x[driver].to_numpy()
    yv = y.to_numpy()
    order = np.argsort(xv)
    smoothed = lowess(yv[order], xv[order], frac=0.6, return_sorted=False)
    fitted = np.empty_like(yv)
    fitted[order] = smoothed
    return yv - fitted


# ----------------------------------------------------------------- deseasonalise + confounders (T7)


def heating_season_indicator(index: pd.DatetimeIndex, heating_months: tuple[int, ...]) -> pd.Series:
    """1.0 where the date's month is a heating-season month, else 0.0 (REQ-021)."""
    months = pd.DatetimeIndex(index).month
    return pd.Series(np.isin(months, heating_months).astype(float), index=index, name="heating")


def deseasonalize(
    s: pd.Series,
    method: str = "yoy",
    heating_season: pd.Series | None = None,
    curtailment: pd.Series | None = None,
    *,
    period: int = 52,
    cfg=None,
) -> pd.Series:
    """Deseasonalise ``s`` and model the heating-season + curtailment confounders (REQ-020/021/022).

    ``method``:
      - ``"yoy"`` (default): subtract the value one year prior (``period`` periods back). Removes the
        annual cycle and the slow secular trend (e.g. the ultra-low-emission retrofit decline that
        decouples NO2 from output, Li 2024) while preserving year-over-year activity changes. Chosen
        over double-differencing after the 2026-06-13 sensitivity run (double-diff erased the signal).
      - ``"stl"``: STL decomposition residual (seasonal + trend removed; Kim 2023 seasonal-adjustment
        analogue). Gaps are interpolated only to fit STL, then re-masked.
      - ``"yoy-double-diff"`` (Kondragunta 2021): year-prior subtraction then first-difference. Removes
        the trend most aggressively; demonstrably over-aggressive on this weekly series (kept for
        robustness comparison, not the default).
      - ``"intensity-model"`` (NOX-003.1): fit an explicit smoothness-controlled secular intensity
        trend ``s(t)`` and return the residual as the activity proxy (Li 2024 decomposition). The
        smoothness is chosen by cross-validation on the NO2 series alone — never against the benchmark
        (NFR-102). Parameters come from ``cfg`` (a ``SignalConfig``); defaults are used if ``cfg`` is
        ``None``. The selected df/criterion/estimator/cv_score are recorded in ``Series.attrs``.
      - ``"none"``: pass the series through (still applies the structural controls below).

    ``period`` defaults to 52 (weekly); use 12 for monthly. The **heating-season** term and
    **curtailment** control are then regressed out as *structural* covariates of the deseasonalised
    series (REQ-021/022) — modelled, not merely smoothed away, because the column's sensitivity to
    steel is itself season-dependent (Wen 2024) and curtailments can decouple emissions from output
    (Li 2024). The applied method and term names are recorded in ``Series.attrs``. NaN gaps are
    preserved (no interpolation, except internally to fit STL).
    """
    s = s.astype(float)
    intensity_attrs: dict = {}
    if method == "yoy":
        deseasoned = s.diff(period)
        applied = "yoy"
    elif method == "stl":
        from statsmodels.tsa.seasonal import STL

        si = s.interpolate(limit_direction="both")
        if int(si.notna().sum()) < 2 * period + 2:
            deseasoned = s * np.nan
        else:
            resid = STL(si, period=period, robust=True).fit().resid
            deseasoned = pd.Series(resid, index=s.index).where(s.notna())
        applied = "stl"
    elif method == "yoy-double-diff":
        deseasoned = s.diff(period).diff(1)
        applied = "yoy-double-diff"
    elif method == "intensity-model":
        from noxus.signal.intensity import fit_intensity_trend

        kw = _intensity_kwargs(cfg)
        fit = fit_intensity_trend(s, **kw)
        deseasoned = fit.residual.copy()
        applied = "intensity-model"
        intensity_attrs = {
            "intensity_df": fit.df,
            "intensity_criterion": fit.criterion,
            "intensity_estimator": fit.estimator,
            "intensity_cv_score": fit.cv_score,
        }
    elif method == "prophet":
        from noxus.signal.prophet_deseason import prophet_deseason

        fit = prophet_deseason(s, **_prophet_kwargs(cfg))
        deseasoned = fit.residual.reindex(s.index)
        applied = "prophet"
        intensity_attrs = {
            "prophet_weekly_amplitude": fit.weekly_amplitude,
            "prophet_variance_removed": fit.variance_removed,
            "prophet_params": fit.params,
        }
    elif method == "harmonic":
        from noxus.signal.prophet_deseason import harmonic_deseason

        k = cfg.harmonic_order if cfg is not None else 1
        deseasoned = harmonic_deseason(s, k=k).reindex(s.index)
        applied = "harmonic"
        intensity_attrs = {"harmonic_order": k}
    elif method == "none":
        deseasoned = s.copy()
        applied = "none"
    else:
        raise ValueError(
            f"Unknown deseasonalisation method {method!r} "
            "(use 'yoy'/'stl'/'yoy-double-diff'/'intensity-model'/'prophet'/'harmonic'/'none')."
        )

    terms: list[str] = []
    covariates: dict[str, pd.Series] = {}
    if heating_season is not None:
        covariates["heating"] = heating_season.reindex(deseasoned.index).astype(float)
        terms.append("heating_season")
    if curtailment is not None:
        covariates["curtailment"] = curtailment.reindex(deseasoned.index).astype(float)
        terms.append("curtailment")

    if covariates:
        deseasoned = _residualize_on_terms(deseasoned, pd.DataFrame(covariates))

    deseasoned = deseasoned.rename("index_deseasonalized")
    deseasoned.attrs["deseason_method"] = applied
    deseasoned.attrs["structural_terms"] = terms
    deseasoned.attrs.update(intensity_attrs)
    return deseasoned


def _intensity_kwargs(cfg) -> dict:
    """Pull intensity-model parameters from ``cfg`` (a ``SignalConfig``), or fall back to defaults."""
    if cfg is None:
        return {}
    return {
        "estimator": cfg.intensity_estimator,
        "df_grid": cfg.intensity_df_grid,
        "cv_folds": cfg.intensity_cv_folds,
        "criterion": cfg.intensity_criterion,
        "min_length": cfg.intensity_min_length,
    }


def _prophet_kwargs(cfg) -> dict:
    """Pull Prophet parameters from ``cfg`` (a ``SignalConfig``), or fall back to defaults (NOX-006)."""
    if cfg is None:
        return {}
    return {
        "growth": cfg.prophet_growth,
        "changepoint_prior": cfg.prophet_changepoint_prior,
        "yearly_order": cfg.prophet_yearly_order,
        "weekly_order": cfg.prophet_weekly_order,
        "min_valid_days": cfg.prophet_min_valid,
    }


def _residualize_on_terms(y: pd.Series, terms: pd.DataFrame) -> pd.Series:
    """OLS-residualise ``y`` on structural ``terms`` (with intercept); preserve NaN rows."""
    import statsmodels.api as sm

    out = y.copy()
    valid = y.notna() & terms.notna().all(axis=1)
    # A constant term carries no information once an intercept is added; drop zero-variance columns.
    usable = [c for c in terms.columns if terms.loc[valid, c].nunique() > 1]
    if int(valid.sum()) < len(usable) + 2 or not usable:
        return out
    design = sm.add_constant(terms.loc[valid, usable].to_numpy(), has_constant="add")
    model = sm.OLS(y[valid].to_numpy(), design).fit()
    out.loc[valid] = model.resid
    return out


# --------------------------------------------------------------------------- relative index (T8)


def build_index(
    corrected_meteo_free: pd.Series,
    anchor: str = "zscore",
    attributable_cap: tuple[float, float] = (0.30, 0.43),
) -> pd.DataFrame:
    """Normalise to a relative, unitless activity index and record the attributable cap (REQ-030/031).

    ``anchor="zscore"`` (default): ``(x - mean) / std`` over the valid sample. Any other ``anchor``
    value is treated as a baseline-period *label*: the series is divided by the mean of the values at
    dates whose ISO string starts with that label, then ×100 (an indexed-to-baseline series). The
    result is strictly relative — **no absolute steel-tonnage figure is produced anywhere** (REQ-030,
    the dominant statistical-discipline constraint).

    Returns a DataFrame with columns ``date``, ``index_value``, ``valid_coverage`` (NaN-filled when
    not supplied) and ``DataFrame.attrs`` provenance: ``anchor``, ``attributable_cap``, plus any
    ``meteo_form`` / ``deseason_method`` carried on the input series' ``attrs``.
    """
    s = corrected_meteo_free.astype(float)
    valid = s.notna()

    if anchor == "zscore":
        mu = float(s[valid].mean())
        sigma = float(s[valid].std(ddof=0))
        index_value = (s - mu) / sigma if sigma > 0 else s - mu
        anchor_label = "zscore"
    else:
        base_mask = pd.Index(s.index.astype(str)).str.startswith(anchor)
        base = s[valid.to_numpy() & np.asarray(base_mask)]
        base_mean = float(base.mean()) if len(base) else float(s[valid].mean())
        index_value = (s / base_mean) * 100.0 if base_mean != 0 else s
        anchor_label = anchor

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(s.index),
            "index_value": index_value.to_numpy(),
            "valid_coverage": np.full(len(s), np.nan),
        }
    ).reset_index(drop=True)

    df.attrs["anchor"] = anchor_label
    df.attrs["attributable_cap"] = list(attributable_cap)
    # Carry provenance forward from upstream stages so the index is fully self-describing.
    for key in (
        "meteo_form",
        "meteo_covariates",
        "deseason_method",
        "structural_terms",
        "intensity_df",
        "intensity_criterion",
        "intensity_estimator",
        "intensity_cv_score",
    ):
        if key in corrected_meteo_free.attrs:
            df.attrs[key] = corrected_meteo_free.attrs[key]
    return df


# --------------------------------------------------------------------------- orchestration + I/O


def write_index(df: pd.DataFrame, out_path: Path, extra_provenance: dict | None = None) -> Path:
    """Write the index parquet with provenance embedded in the Arrow schema metadata.

    parquet itself does not carry ``DataFrame.attrs``, so the provenance (anchor, cap, meteo form,
    deseason method, processor knobs) is serialised into the Arrow schema's key/value metadata under
    the ``noxus`` key. This keeps the emitted artifact self-describing (REQ-031, NFR-001).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    provenance = dict(df.attrs)
    if extra_provenance:
        provenance.update(extra_provenance)

    table = pa.Table.from_pandas(df, preserve_index=False)
    meta = dict(table.schema.metadata or {})
    meta[b"noxus"] = json.dumps(provenance, default=str).encode("utf-8")
    table = table.replace_schema_metadata(meta)
    pq.write_table(table, out_path)
    return out_path


def read_index_provenance(parquet_path: Path) -> dict:
    """Read back the ``noxus`` provenance metadata written by :func:`write_index`."""
    import pyarrow.parquet as pq

    schema = pq.read_schema(Path(parquet_path))
    meta = schema.metadata or {}
    raw = meta.get(b"noxus")
    return json.loads(raw.decode("utf-8")) if raw else {}


def build_activity_index(cfg=None, *, use_meteo: bool = True) -> Path:
    """End-to-end relative index: footprint signal (+ ERA5 regress-out) → deseason → index parquet.

    Reads ``steel_footprint_signal.parquet`` (emitted by ``noxus.attribution.source``); when
    ``use_meteo`` is set, reads the latest ERA5 snapshot, aggregates it to the footprint, and
    regresses it out (REQ-011, ERR-002 if the snapshot is missing). Then deseasonalises with the
    heating-season structural term + curtailment control (REQ-020/021/022) and normalises to the
    relative index (REQ-030/031). Writes ``steel_activity_index.parquet`` with full provenance and
    returns its path. Raises ``FileNotFoundError`` (ERR-001) if the footprint signal is missing.
    """
    from noxus.config.run import SignalConfig

    cfg = cfg or SignalConfig()
    out_dir = Path(cfg.out_dir)
    signal_path = out_dir / cfg.footprint_signal_name
    if not signal_path.exists():
        raise FileNotFoundError(
            f"Footprint signal not found: {signal_path}. Run 'noxus attribute' first (ERR-001)."
        )

    fp = pd.read_parquet(signal_path)
    series = pd.Series(
        fp["no2_corrected"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(pd.to_datetime(fp["date"])),
        name="no2_corrected",
    )

    if use_meteo:
        from noxus.data.era5 import era5_footprint_series
        from noxus.data.gridding import COVERAGE  # noqa: F401  (kept for schema parity)

        snapshot = _latest_era5_snapshot(Path(cfg.era5_snapshot_dir))
        footprint_mask = _load_footprint_mask(cfg)
        meteo = era5_footprint_series(snapshot, footprint_mask, freq=cfg.freq)
        meteo = meteo.set_index(pd.DatetimeIndex(pd.to_datetime(meteo["date"]))).drop(
            columns=["date"]
        )
        series = regress_out_meteo(series, meteo, form=cfg.meteo_form)

    heating = heating_season_indicator(series.index, cfg.heating_season_months)
    curtailment = _load_curtailment(cfg, series.index)
    deseasoned = deseasonalize(
        series,
        method=cfg.deseason_method,
        heating_season=heating,
        curtailment=curtailment,
        cfg=cfg,
    )

    # NOX-003.1: when the explicit intensity model runs, emit the decomposition (signal, trend,
    # residual) as a reportable diagnostic — the secular trend *is* the decoupling (REQ-104).
    if cfg.deseason_method == "intensity-model":
        _write_intensity_decomposition(series, deseasoned, cfg, out_dir)

    df = build_index(deseasoned, anchor=cfg.index_anchor, attributable_cap=cfg.attributable_cap)
    df["valid_coverage"] = fp["valid_coverage"].to_numpy()
    extra = {
        "footprint_radius_km": cfg.footprint_radius_km,
        "background_geom": [cfg.background_inner_km, cfg.background_outer_km],
        "curtailment_source": cfg.curtailment_source,
        # Honesty: record whether the curtailment control was *actually applied*, not just configured.
        # _load_curtailment returns None by default (no calendar artifact yet, Q6a), so the control is
        # absent end-to-end even though a source is named — downstream provenance must not overstate it.
        "curtailment_applied": curtailment is not None,
    }
    return write_index(df, out_dir / cfg.index_name, extra_provenance=extra)


def _write_intensity_decomposition(
    signal: pd.Series, deseasoned: pd.Series, cfg, out_dir: Path
) -> Path:
    """Emit the intensity decomposition diagnostic: signal, s(t) trend, activity residual (REQ-104).

    Recomputes the trend deterministically from ``signal`` (same CV-selected smoothness as the index
    run) so the secular intensity decline is recorded as a standalone, reportable series — it is the
    decoupling itself, not discarded. Written under the gitignored derived NO2 directory.
    """
    from noxus.signal.intensity import fit_intensity_trend

    fit = fit_intensity_trend(signal, **_intensity_kwargs(cfg))
    decomp = pd.DataFrame(
        {
            "date": pd.to_datetime(signal.index),
            "signal": signal.to_numpy(dtype=float),
            "trend_s_t": fit.trend.to_numpy(dtype=float),
            "residual_activity": fit.residual.to_numpy(dtype=float),
        }
    ).reset_index(drop=True)
    decomp.attrs.update(
        intensity_df=fit.df,
        intensity_criterion=fit.criterion,
        intensity_estimator=fit.estimator,
        intensity_cv_score=fit.cv_score,
    )
    out_path = out_dir / cfg.decomposition_name
    return write_index(decomp, out_path)


def _latest_era5_snapshot(snapshot_dir: Path) -> Path:
    """Return the most recent ``era5_<date>.nc`` snapshot, or raise ERR-002 if none exists."""
    from noxus.data.era5 import ERA5SnapshotError

    snaps = sorted(Path(snapshot_dir).glob("era5_*.nc"))
    if not snaps:
        raise ERA5SnapshotError(
            f"No ERA5 snapshot under {snapshot_dir}. Run 'noxus ingest-era5' first, or build the "
            "index with meteo normalisation disabled (ERR-002)."
        )
    return snaps[-1]


def _load_footprint_mask(cfg):
    """Rebuild the footprint mask from the cube + facilities (for ERA5 spatial aggregation)."""
    import xarray as xr

    from noxus.attribution.source import footprint_mask, load_facilities

    cube = xr.open_dataset(cfg.cube_path).load()
    cube.close()
    facilities = load_facilities(cfg.facilities_csv, set(cfg.active_statuses))
    return footprint_mask(cube, facilities, cfg.footprint_radius_km)


def _load_curtailment(cfg, index: pd.DatetimeIndex) -> pd.Series | None:
    """Best-effort curtailment exogenous control (REQ-022).

    Default behaviour (curtailment_source='crea'): no external fetch is performed unprompted; in the
    absence of a curtailment calendar artifact we return ``None`` so the control is simply absent and
    recorded as such, rather than inventing data. A future T15 wiring can read a CREA-derived
    curtailment series; this keeps the limitation explicit (open question Q6a).
    """
    return None
