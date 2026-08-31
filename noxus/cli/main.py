"""NOXUS command-line interface.

A thin dispatcher over the pipeline stages. ``ingest-benchmark``, ``fetch`` (TROPOMI NO2 acquisition)
and ``verify-no2`` are implemented; ``attribute``/``index``/``validate`` remain scaffolds so the
intended end-to-end shape stays visible.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path

from noxus import __version__
from noxus.config.region import TANGSHAN, tangshan_aoi
from noxus.config.run import AcquisitionConfig, BenchmarkConfig
from noxus.data.benchmark import emit_benchmark, fetch_benchmark_snapshot, load_benchmark

DERIVED = Path("data/derived")
RAW_BENCHMARK = Path("data/raw/benchmark")
DEFAULT_BENCHMARK_PARQUET = DERIVED / "benchmark_tangshan_bf_operating_rate.parquet"
DEFAULT_FACILITIES = DERIVED / "tangshan_steel_facilities.csv"
VERIFICATION_DIR = DERIVED / "verification"


def _cmd_ingest_benchmark(args: argparse.Namespace) -> int:
    """Fetch (or load a snapshot of) the CREA benchmark and emit the tidy parquet."""
    cfg = BenchmarkConfig()
    if args.from_snapshot:
        snapshot = Path(args.from_snapshot)
    else:
        snapshot = fetch_benchmark_snapshot(cfg.source_url, RAW_BENCHMARK, date.today())
        print(f"[noxus] fetched snapshot -> {snapshot}")

    wide = load_benchmark(snapshot, column=cfg.primary_column, aux_columns=cfg.aux_columns)
    out = Path(args.out) if args.out else DEFAULT_BENCHMARK_PARQUET
    aux_out = out.with_name("benchmark_auxiliary.parquet")
    emit_benchmark(
        wide,
        out,
        primary_column=cfg.primary_column,
        source=cfg.source_label,
        snapshot_date=_snapshot_date(snapshot),
        aux_out_path=aux_out,
    )
    n = int(wide[cfg.primary_column].notna().sum())
    print(f"[noxus] wrote {out} ({n} non-missing weekly observations); aux -> {aux_out}")
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    """Acquire TROPOMI NO2 over the AOI via openEO (server-side subset, resumable)."""
    from noxus.data.tropomi import acquire_no2

    aoi = tangshan_aoi(args.buffer) if args.buffer is not None else TANGSHAN
    cfg = AcquisitionConfig()
    cfg = replace(
        cfg,
        start=date.fromisoformat(args.start) if args.start else cfg.start,
        end=date.fromisoformat(args.end) if args.end else cfg.end,
        qa_threshold=args.qa if args.qa is not None else cfg.qa_threshold,
        raw_dir=Path(args.raw_dir) if getattr(args, "raw_dir", None) else cfg.raw_dir,
    )
    report = acquire_no2(aoi, cfg)
    print(
        f"[noxus] acquired: fetched={report.fetched} skipped={report.skipped} "
        f"failed_batches={report.failed_batches} -> {report.raw_dir}"
    )
    if report.has_version_discontinuity:
        print(f"[noxus] WARNING: processor-version discontinuity across {set(report.versions)}")
    return 0


def _cmd_flux_divergence(args: argparse.Namespace) -> int:
    """Flux-divergence attribution. Only ``--probe`` (T1 gate) is wired; the full run is T2+."""
    from noxus.config.run import FluxDivergenceConfig
    from noxus.fluxdiv.probe import probe

    if not args.probe:
        print(
            "[noxus] flux-divergence: only the T1 feasibility gate is available "
            "(run with --probe). The full attribution is gated behind a GO verdict (NOX-005 T2+)."
        )
        return 1

    cfg = FluxDivergenceConfig()
    cfg = replace(
        cfg,
        subset_start=date.fromisoformat(args.start) if args.start else cfg.subset_start,
        subset_end=date.fromisoformat(args.end) if args.end else cfg.subset_end,
    )
    v = probe(cfg)
    print(f"[noxus] T1 flux-divergence probe VERDICT={v.verdict} escalate={v.escalate}")
    print(f"[noxus]   Q1 free-CDSE access_ok={v.access_ok}: {v.access_note}")
    if v.scaling:
        curve = "  ".join(f"N={p.n}:SNR={p.snr:+.2f}/coloc={p.coloc_ratio:.2f}" for p in v.scaling)
        print(f"[noxus]   SNR-vs-N: {curve}")
    if v.evidence is not None and v.compute is not None:
        e, c = v.evidence, v.compute
        print(
            f"[noxus]   all-N coherence: coloc_ratio={e.coloc_ratio:.2f} SNR={e.snr:+.2f} "
            f"(overpasses={c.n_overpasses_used}, pixels={e.n_pixels})"
        )
        print(
            f"[noxus]   compute: {c.downloaded_bytes / 1e9:.2f} GB dl, "
            f"~{c.per_granule_mb:.0f} MB/granule, wall={c.wall_clock_s:.1f}s, "
            f"extrapolated full download ~{c.extrapolated_full_download_gb:.0f} GB"
        )
    print(f"[noxus]   report -> {v.report_path}")
    # NO-GO / NO-GO-DATA stop the build (ERR-002): non-zero exit so callers/CI can gate on it.
    return 0 if v.verdict in ("GO", "GO-degraded") else 2


def _cmd_verify_no2(args: argparse.Namespace) -> int:
    """Render NO2 over the AOI with facilities overlaid for clear-sky high-NO2 days."""
    from noxus.data.verify_no2 import fetch_optical_thumbnail, render_day, select_clear_high_days

    raw_dir = Path(args.raw_dir) if args.raw_dir else AcquisitionConfig().raw_dir
    picks = select_clear_high_days(raw_dir, n=args.days, max_cloud=args.max_cloud)
    if not picks:
        print(
            f"[noxus] no clear high-NO2 days found in {raw_dir} (acquire first with 'noxus fetch')."
        )
        return 1

    facilities = Path(args.facilities) if args.facilities else DEFAULT_FACILITIES
    out_dir = Path(args.out_dir) if args.out_dir else VERIFICATION_DIR
    for p in picks:
        png = render_day(p["path"], facilities, out_dir / f"{p['id']}.png", aoi=TANGSHAN)
        print(f"[noxus] {p['id']}  no2_mean={p['no2_mean']}  cloud_mean={p['cloud_mean']} -> {png}")
        if args.optical:
            fetch_optical_thumbnail(TANGSHAN, p["id"][:10], out_dir / f"{p['id']}_optical.png")
    return 0


def _cmd_grid(args: argparse.Namespace) -> int:
    """Composite per-overpass NO2 into an analysis-ready cube (+ interim AOI-mean series)."""
    from dataclasses import replace

    from noxus.config.run import GriddingConfig
    from noxus.data.gridding import build_cube

    cfg = GriddingConfig()
    cfg = replace(
        cfg,
        freq=args.freq or cfg.freq,
        min_period_coverage=args.min_coverage
        if args.min_coverage is not None
        else cfg.min_period_coverage,
        raw_dir=Path(args.raw_dir) if args.raw_dir else cfg.raw_dir,
        out_dir=Path(args.out_dir) if args.out_dir else cfg.out_dir,
        coverage_bbox=tangshan_aoi(args.coverage_buffer).as_bbox()
        if getattr(args, "coverage_buffer", None) is not None
        else cfg.coverage_bbox,
    )
    rep = build_cube(cfg)
    print(f"[noxus] cube: {rep.n_periods} periods -> {rep.cube_path}")
    if rep.series_path:
        print(f"[noxus] interim AOI-mean series ({rep.n_series_rows} rows) -> {rep.series_path}")
    return 0


def _cmd_attribute(args: argparse.Namespace) -> int:
    """Footprint sample + regional background -> background-corrected footprint signal (NOX-003)."""
    from dataclasses import replace

    from noxus.attribution.source import GeometryError, build_footprint_signal
    from noxus.config.run import SignalConfig

    cfg = SignalConfig()
    if args.radius is not None:
        cfg = replace(cfg, footprint_radius_km=args.radius)
    try:
        out = build_footprint_signal(cfg)
    except (FileNotFoundError, GeometryError) as exc:
        print(f"[noxus] attribute failed: {exc}")
        return 1
    print(f"[noxus] footprint signal -> {out}")
    return 0


def _cmd_index(args: argparse.Namespace) -> int:
    """Meteo regress-out + deseasonalise + relative activity index (NOX-003)."""
    from dataclasses import replace

    from noxus.config.run import SignalConfig
    from noxus.data.era5 import ERA5SnapshotError
    from noxus.signal.index import build_activity_index

    cfg = SignalConfig()
    if getattr(args, "deseason", None):
        cfg = replace(cfg, deseason_method=args.deseason)
    try:
        out = build_activity_index(cfg, use_meteo=not args.no_meteo)
    except (FileNotFoundError, ERA5SnapshotError) as exc:
        print(f"[noxus] index failed: {exc}")
        return 1
    print(f"[noxus] activity index -> {out}")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    """Align + sign + r/p + lead-lag -> report incl. the null (NOX-003)."""
    from dataclasses import replace

    from noxus.config.run import SignalConfig, ValidationConfig
    from noxus.validation.report import InsufficientOverlapError, run_validation

    val_cfg = ValidationConfig()
    if args.max_lag is not None:
        val_cfg = replace(val_cfg, max_lag=args.max_lag)
    try:
        artifacts = run_validation(SignalConfig(), val_cfg)
    except (FileNotFoundError, InsufficientOverlapError) as exc:
        print(f"[noxus] validate failed: {exc}")
        return 1
    print(f"[noxus] conclusion={artifacts.results['conclusion']} -> {artifacts.results_path}")
    print(f"[noxus] summary -> {artifacts.summary_path}")
    return 0


def _cmd_ingest_era5(args: argparse.Namespace) -> int:
    """Fetch an ERA5 snapshot from the Copernicus CDS (server-side AOI subset)."""
    from noxus.config.run import SignalConfig
    from noxus.data.era5 import ingest_era5

    cfg = SignalConfig()
    try:
        out = ingest_era5(cfg)
    except Exception as exc:  # network / auth / licence / CDS-availability -> clean CLI error
        print(f"[noxus] ingest-era5 failed ({type(exc).__name__}): {exc}")
        return 1
    print(f"[noxus] ERA5 snapshot -> {out}")
    return 0


def _cmd_detect_events(args: argparse.Namespace) -> int:
    """Detect coverage- + meteo-screened NO2 production events on the NOX-003.1 residual (NOX-004)."""
    from dataclasses import replace

    from noxus.catalyst.report import _load_meteo
    from noxus.config.run import CatalystConfig

    cfg = CatalystConfig()
    if args.z_thresh is not None:
        cfg = replace(cfg, z_thresh=args.z_thresh)
    import pandas as pd

    from noxus.catalyst.events import detect_events

    decomp_path = Path(cfg.decomposition_path)
    if not decomp_path.exists():
        print(f"[noxus] detect-events failed: {decomp_path} missing (run 'noxus index' first).")
        return 1
    decomp = pd.read_parquet(decomp_path)
    residual = pd.Series(
        decomp["residual_activity"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(pd.to_datetime(decomp["date"])),
    )
    cov = (
        pd.Series(decomp["valid_coverage"].to_numpy(), index=residual.index)
        if "valid_coverage" in decomp.columns
        else None
    )
    meteo = _load_meteo(cfg) if cfg.meteo_screen else None
    events = detect_events(
        residual,
        cov,
        meteo,
        z_thresh=cfg.z_thresh,
        method=cfg.detector,
        min_periods=cfg.detect_min_periods,
        min_coverage=cfg.min_coverage,
        meteo_screen=cfg.meteo_screen,
        ventilation_z=cfg.ventilation_z,
    )
    Path(cfg.events_out).parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(cfg.events_out, index=False)
    print(
        f"[noxus] {len(events)} NO2 events (meteo_screen={meteo is not None}) -> {cfg.events_out}"
    )
    return 0


def _cmd_ingest_market(args: argparse.Namespace) -> int:
    """Fetch free daily prices (miners + steel ETF + benchmark) -> dated snapshot (NOX-004)."""
    from noxus.catalyst.market import ingest_prices
    from noxus.config.run import CatalystConfig

    cfg = CatalystConfig()
    try:
        out = ingest_prices(cfg, start=args.start or "2019-01-01", end=args.end)
    except Exception as exc:  # network / yfinance availability -> clean CLI error
        print(f"[noxus] ingest-market failed ({type(exc).__name__}): {exc}")
        return 1
    print(f"[noxus] market snapshot -> {out}")
    return 0


def _cmd_catalyst(args: argparse.Namespace) -> int:
    """Detect events + match vs production + market event-study -> report incl. null (NOX-004)."""
    from dataclasses import replace

    from noxus.catalyst.report import InsufficientEventsError, run_catalyst
    from noxus.config.run import CatalystConfig

    cfg = CatalystConfig()
    if args.window is not None:
        cfg = replace(cfg, study_window=args.window)
    if args.latency is not None:
        cfg = replace(cfg, overpass_latency_days=args.latency)
    try:
        art = run_catalyst(cfg)
    except (FileNotFoundError, InsufficientEventsError) as exc:
        print(f"[noxus] catalyst failed: {exc}")
        return 1
    print(
        f"[noxus] conclusion={art.results['conclusion']} market={art.results['market']['verdict']}"
    )
    print(f"[noxus] results -> {art.results_path}; summary -> {art.summary_path}")
    return 0


def _cmd_scale_sweep(args: argparse.Namespace) -> int:
    """Sweep AOI extent x grid resolution + autocorrelation-robust significance (NOX-008)."""
    from dataclasses import replace

    from noxus.config.run import ScaleSweepConfig
    from noxus.validation.scale import run_scale_sweep

    cfg = ScaleSweepConfig()
    if args.buffers:
        cfg = replace(cfg, buffers=tuple(float(b) for b in args.buffers.split(",")))
    if args.resolutions:
        cfg = replace(
            cfg,
            resolutions=tuple(
                r.strip() if r.strip() == "native" else float(r)
                for r in args.resolutions.split(",")
            ),
        )
    if args.draws is not None:
        cfg = replace(cfg, n_boot=args.draws, n_perm=args.draws)
    if args.seed is not None:
        cfg = replace(cfg, seed=args.seed)
    if args.alpha is not None:
        cfg = replace(cfg, fdr_alpha=args.alpha)
    try:
        out = run_scale_sweep(cfg)
    except FileNotFoundError as exc:
        print(f"[noxus] scale-sweep failed: {exc}")
        return 1
    print(f"[noxus] scale-sensitivity table -> {out}")
    return 0


def _cmd_scale_contrast(args: argparse.Namespace) -> int:
    """Provincial/national benchmark arm: the scale contrast (NOX-016). --probe = Arm-B T1 gate."""
    from dataclasses import replace

    from noxus.config.run import ScaleContrastConfig
    from noxus.data.nbs import probe_nbs
    from noxus.validation.scale_contrast import run_scale_contrast

    cfg = ScaleContrastConfig()
    if args.draws is not None:
        cfg = replace(cfg, n_boot=args.draws, n_perm=args.draws)
    if args.seed is not None:
        cfg = replace(cfg, seed=args.seed)
    if args.probe:
        res = probe_nbs(Path(cfg.out_dir) / cfg.probe_name)
        print(res.render())
        return 0 if res.go else 2  # 2 = a valid NO-GO, not an error
    try:
        out = run_scale_contrast(cfg)
    except FileNotFoundError as exc:
        print(f"[noxus] scale-contrast failed: {exc}")
        return 1
    print(f"[noxus] scale-contrast table -> {out}")
    print(f"[noxus] findings -> {Path(cfg.out_dir) / cfg.findings_name}")
    return 0


def _cmd_ring_sweep(args: argparse.Namespace) -> int:
    """Background-ring radius sensitivity of the change-variance decomposition (NOX-015)."""
    from dataclasses import replace

    from noxus.config.run import RingSweepConfig, WideRingSweepConfig
    from noxus.data.era5 import ERA5SnapshotError
    from noxus.signal.ring_sweep import run_ring_sweep, run_wide_ring_sweep

    wide = bool(getattr(args, "wide", False))
    cfg = WideRingSweepConfig() if wide else RingSweepConfig()
    if args.rings:
        pairs = []
        for chunk in args.rings.split(";"):
            lo, hi = (float(v) for v in chunk.split(","))
            pairs.append((lo, hi))
        cfg = replace(cfg, ring_grid=tuple(pairs))
    if args.resolutions:
        cfg = replace(cfg, resolutions=tuple(r.strip() for r in args.resolutions.split(",")))
    if args.weightings:
        cfg = replace(cfg, weightings=tuple(w.strip() for w in args.weightings.split(",")))
    if args.draws is not None:
        cfg = replace(cfg, n_boot=args.draws)
    if args.seed is not None:
        cfg = replace(cfg, seed=args.seed)
    try:
        out = run_wide_ring_sweep(cfg) if wide else run_ring_sweep(cfg)
    except (FileNotFoundError, ERA5SnapshotError) as exc:
        print(f"[noxus] ring-sweep failed: {exc}")
        return 1
    print(f"[noxus] ring-sensitivity table -> {out}")
    print(f"[noxus] findings -> {Path(cfg.out_dir) / cfg.findings_name}")
    return 0


def _cmd_variance(args: argparse.Namespace) -> int:
    """Change-variance decomposition: steel-local share + implied change ceiling (NOX-009)."""
    from dataclasses import replace

    from noxus.config.run import VarianceDecompConfig
    from noxus.data.era5 import ERA5SnapshotError
    from noxus.signal.variance import run_decompose

    cfg = VarianceDecompConfig()
    if args.bases:
        cfg = replace(cfg, bases=tuple(b.strip() for b in args.bases.split(",")))
    if args.resolutions:
        cfg = replace(cfg, resolutions=tuple(r.strip() for r in args.resolutions.split(",")))
    if args.draws is not None:
        cfg = replace(cfg, n_boot=args.draws)
    if args.seed is not None:
        cfg = replace(cfg, seed=args.seed)
    if args.deseason:
        cfg = replace(cfg, deseason_method=args.deseason)
    try:
        out = run_decompose(cfg)
    except (FileNotFoundError, ERA5SnapshotError) as exc:
        print(f"[noxus] variance failed: {exc}")
        return 1
    print(f"[noxus] variance-decomposition table -> {out}")
    return 0


def _cmd_fetch_viirs(args: argparse.Namespace) -> int:
    """Acquire VIIRS thermal detections over the AOI (VNF primary / FIRMS cross-check); T1 --probe."""
    from dataclasses import replace

    from noxus.config.run import ViirsAcquisitionConfig
    from noxus.data.viirs import (
        ViirsAcquisitionError,
        acquire_viirs,
        probe_viirs,
    )
    from noxus.signal.thermal import load_facilities

    cfg = ViirsAcquisitionConfig()
    cfg = replace(
        cfg,
        source=args.source or cfg.source,
        start=date.fromisoformat(args.start) if args.start else cfg.start,
        end=date.fromisoformat(args.end) if args.end else cfg.end,
    )
    aoi = tangshan_aoi(cfg.aoi_buffer_deg)
    try:
        if args.probe:
            facilities = (
                load_facilities(DEFAULT_FACILITIES) if DEFAULT_FACILITIES.exists() else None
            )
            probe = probe_viirs(aoi, cfg, facilities=facilities)
            print(
                f"[noxus] T1 probe source={probe.source} endpoint={probe.endpoint} "
                f"auth_required={probe.auth_required}"
            )
            print(
                f"[noxus]   window={probe.window[0]}..{probe.window[1]} "
                f"archive_lag_days={probe.archive_lag_days}"
            )
            print(
                f"[noxus]   nights_with_detections={probe.nights_with_detections}/{probe.nights_probed} "
                f"detections={probe.n_detections} colocated={probe.facilities_colocated}"
            )
            print(
                f"[noxus]   per_night={probe.per_night_mb:.3f} MB implied_full_era="
                f"{probe.implied_full_era_mb:.1f} MB within_budget={probe.within_budget}"
            )
            print(f"[noxus]   fields={list(probe.fields_available)}")
            print(f"[noxus]   {'GO' if probe.go else 'NO-GO'}: {probe.notes}")
            return 0 if probe.go else 2
        report = acquire_viirs(aoi, cfg)
    except ViirsAcquisitionError as exc:
        print(f"[noxus] fetch-viirs failed: {exc}")
        return 1
    print(
        f"[noxus] acquired VIIRS ({report.source}): fetched={report.fetched} "
        f"skipped={report.skipped} failed={report.failed} empty_nights={report.empty_nights} "
        f"detections={report.n_detections} -> {report.raw_dir}"
    )
    return 0


def _cmd_build_thermal_series(args: argparse.Namespace) -> int:
    """Facility-radius join + weekly/daily thermal-activity series (benchmark-blind, NOX-011)."""
    from dataclasses import replace

    from noxus.config.run import ThermalTriangulationConfig
    from noxus.data.viirs import load_detections
    from noxus.signal.thermal import (
        FacilitiesError,
        build_thermal_series,
        join_to_facilities,
        load_facilities,
    )

    cfg = ThermalTriangulationConfig()
    cfg = replace(
        cfg,
        radius_km=args.radius_km if args.radius_km is not None else cfg.radius_km,
        freq=args.freq or cfg.freq,
    )
    source = args.source or "firms"
    try:
        facilities = load_facilities(cfg.facilities_csv)
    except FacilitiesError as exc:
        print(f"[noxus] build-thermal-series failed: {exc}")
        return 1
    detections = load_detections(cfg.raw_dir, source)
    if detections.empty:
        print(f"[noxus] no detections in {cfg.raw_dir}/{source} (run 'noxus fetch-viirs' first).")
        return 1
    joined = join_to_facilities(detections, facilities, cfg.radius_km)
    series = build_thermal_series(
        joined,
        freq=cfg.freq,
        aggregate=cfg.primary_aggregate,
        sparse_min_nights=cfg.sparse_min_nights,
    )
    out = Path(cfg.out_dir) / cfg.series_name
    out.parent.mkdir(parents=True, exist_ok=True)
    series.to_parquet(out)
    print(f"[noxus] thermal series ({source}, {len(series)} {cfg.freq}-periods) -> {out}")
    return 0


def _cmd_triangulate_thermal(args: argparse.Namespace) -> int:
    """NO2<->VIIRS + CREA<->VIIRS under robust significance -> feature-vs-target diagnosis (NOX-011)."""
    from analysis.viirs_triangulation import run_triangulation
    from noxus.config.run import ThermalTriangulationConfig

    cfg = ThermalTriangulationConfig()
    try:
        out = run_triangulation(cfg)
    except FileNotFoundError as exc:
        print(f"[noxus] triangulate-thermal failed: {exc}")
        return 1
    import pandas as pd

    df = pd.read_csv(out)
    print(f"[noxus] triangulation table -> {out}")
    if len(df):
        print(f"[noxus] diagnosis: {df['diagnosis'].iloc[0]}")
    return 0


def _cmd_assemble_calendar(args: argparse.Namespace) -> int:
    """Assemble/validate the curtailment calendar; --probe = T1 feasibility spike (NOX-012)."""
    from noxus.catalyst.calendar import (
        assemble_calendar,
        probe_sources,
        write_calendar,
        write_sources_md,
    )
    from noxus.config.run import RegimeStudyConfig

    cfg = RegimeStudyConfig()
    if args.probe:
        report = probe_sources(cfg)
        write_sources_md(report, cfg.sources_path)
        print(f"[noxus] T1 probe verdict={report.verdict}")
        print(
            f"[noxus]   candidates={report.n_candidates} securely_dated={report.n_securely_dated} "
            f"event_level={report.n_event_level} spans_era={report.spans_era}"
        )
        print(
            f"[noxus]   peak_event={report.has_peak_event} collapse_event={report.has_collapse_event} "
            f"live_survey_pending={report.live_survey_pending}"
        )
        print(f"[noxus]   {report.reason}")
        print(f"[noxus]   provenance -> {cfg.sources_path}")
        return 0 if report.go else 2
    df = assemble_calendar(cfg)
    out = write_calendar(df, cfg.calendar_path)
    report = probe_sources(cfg)
    write_sources_md(report, cfg.sources_path)
    print(f"[noxus] curated calendar ({len(df)} events) -> {out}; sources -> {cfg.sources_path}")
    return 0


def _cmd_regime_study(args: argparse.Namespace) -> int:
    """Rolling regime + event study + jackknife + scan-max null + BH-FDR -> verdict (NOX-012)."""
    from dataclasses import replace

    from noxus.catalyst.regime import regime_study
    from noxus.config.run import RegimeStudyConfig

    cfg = RegimeStudyConfig()
    if args.window_months is not None:
        cfg = replace(cfg, window_months=args.window_months)
    if args.draws is not None:
        cfg = replace(cfg, n_boot=args.draws, n_perm=args.draws, scanmax_draws=args.draws)
    if args.seed is not None:
        cfg = replace(cfg, seed=args.seed)
    try:
        art = regime_study(cfg)
    except FileNotFoundError as exc:
        print(f"[noxus] regime-study failed: {exc}")
        return 1
    print(f"[noxus] verdict={art.verdict} n_tests={art.n_tests}")
    print(
        f"[noxus]   scan-max observed={art.scanmax.observed_scanmax:.3f} p={art.scanmax.p_value} "
        f"beyond_band={art.scanmax.beyond_band}; collapse={art.collapse.verdict}"
    )
    print(f"[noxus]   table -> {art.results_path}; summary -> {art.summary_path}")
    return 0


def _cmd_verify_viirs(args: argparse.Namespace) -> int:
    """Render VIIRS detections over the AOI with facilities overlaid (co-location check, NOX-011)."""
    from noxus.config.run import ThermalTriangulationConfig
    from noxus.data.verify_viirs import render_from_store

    cfg = ThermalTriangulationConfig()
    source = args.source or "firms"
    facilities = Path(args.facilities) if args.facilities else DEFAULT_FACILITIES
    out_png = Path(args.out) if args.out else cfg.figures_dir / f"viirs_{source}_colocation.png"
    png = render_from_store(cfg.raw_dir, facilities, out_png, source=source, aoi=TANGSHAN)
    print(f"[noxus] co-location render -> {png}")
    return 0


def _snapshot_date(snapshot: Path) -> date | None:
    """Parse the YYYY-MM-DD date out of a ``crea_wind_<date>.csv`` snapshot name, if present."""
    stem = Path(snapshot).stem
    try:
        return datetime.strptime(stem.rsplit("_", 1)[-1], "%Y-%m-%d").date()
    except ValueError:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="noxus", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"noxus {__version__}")

    sub = parser.add_subparsers(dest="command")

    p_attr = sub.add_parser("attribute", help="footprint sample + background -> footprint signal")
    p_attr.add_argument("--radius", type=float, help="footprint radius in km (default: config 15)")
    p_attr.set_defaults(func=_cmd_attribute)

    p_index = sub.add_parser("index", help="meteo regress-out + deseason + relative activity index")
    p_index.add_argument(
        "--no-meteo", action="store_true", help="skip ERA5 meteo regress-out (no snapshot needed)"
    )
    p_index.add_argument(
        "--deseason",
        default=None,
        help="deseason basis method (e.g. intensity-model); default from SignalConfig. "
        "'intensity-model' emits steel_intensity_decomposition.parquet (NOX-003.1 residual).",
    )
    p_index.set_defaults(func=_cmd_index)

    p_val = sub.add_parser("validate", help="align + sign + r/p + lead-lag -> report (incl. null)")
    p_val.add_argument("--max-lag", type=int, help="lag window for the CCF (default: config 8)")
    p_val.set_defaults(func=_cmd_validate)

    p_era5 = sub.add_parser("ingest-era5", help="fetch an ERA5 snapshot from the Copernicus CDS")
    p_era5.set_defaults(func=_cmd_ingest_era5)

    p_grid = sub.add_parser("grid", help="composite per-overpass NO2 into an analysis-ready cube")
    p_grid.add_argument("--freq", help="composite frequency (pandas alias; default W)")
    p_grid.add_argument("--min-coverage", type=float, help="min period coverage before masking")
    p_grid.add_argument("--raw-dir", help="acquired store (default: data/raw/tropomi)")
    p_grid.add_argument("--out-dir", help="output dir (default: data/derived/no2)")
    p_grid.add_argument(
        "--coverage-buffer",
        type=float,
        help="evaluate the PERIOD coverage mask over the AOI at this buffer in degrees instead of "
        "the whole cube. A wider acquisition must pass the committed 0.25 so it keeps the same "
        "temporal mask, or the retained period set moves at the same time as the ring (NOX-017 "
        "REQ-004)",
    )
    p_grid.set_defaults(func=_cmd_grid)

    p_ing = sub.add_parser("ingest-benchmark", help="fetch + clean the CREA Tangshan benchmark")
    p_ing.add_argument(
        "--from-snapshot", help="parse an existing dated CSV snapshot instead of fetching"
    )
    p_ing.add_argument("--out", help="output parquet path")
    p_ing.set_defaults(func=_cmd_ingest_benchmark)

    p_fetch = sub.add_parser("fetch", help="acquire TROPOMI NO2 over the AOI via openEO")
    p_fetch.add_argument("--start", help="ISO start date (default: config 2019-01-01)")
    p_fetch.add_argument("--end", help="ISO end date (default: today)")
    p_fetch.add_argument("--buffer", type=float, help="AOI buffer in degrees (default: 0.25)")
    p_fetch.add_argument("--qa", type=float, help="qa_value threshold (default: 0.75)")
    p_fetch.add_argument(
        "--raw-dir",
        help="raw store to write into (default: data/raw/tropomi). A wider --buffer MUST use its "
        "own store: the manifest is keyed by date only, so mixing buffers would silently merge two "
        "geometries into one record (NOX-017 REQ-001/002)",
    )
    p_fetch.set_defaults(func=_cmd_fetch)

    p_fdiv = sub.add_parser(
        "flux-divergence",
        help="Beirle flux-divergence attribution; --probe = T1 feasibility gate (NOX-005)",
    )
    p_fdiv.add_argument(
        "--probe",
        action="store_true",
        help="T1 feasibility spike on a bounded subset: three-way go/no-go + compute profile",
    )
    p_fdiv.add_argument(
        "--start", help="ISO start of the bounded subset window (default 2023-06-01)"
    )
    p_fdiv.add_argument("--end", help="ISO end of the bounded subset window (default 2023-06-05)")
    p_fdiv.set_defaults(func=_cmd_flux_divergence)

    p_ver = sub.add_parser("verify-no2", help="render NO2 vs facilities on clear high-NO2 days")
    p_ver.add_argument("--raw-dir", help="acquired store (default: data/raw/tropomi)")
    p_ver.add_argument("--days", type=int, default=5, help="number of days to render")
    p_ver.add_argument("--max-cloud", type=float, default=0.2, help="max mean cloud fraction")
    p_ver.add_argument("--facilities", help="facilities CSV (default: data/derived/...)")
    p_ver.add_argument("--out-dir", help="output dir (default: data/derived/verification)")
    p_ver.add_argument("--optical", action="store_true", help="also fetch a Sentinel-2 thumbnail")
    p_ver.set_defaults(func=_cmd_verify_no2)

    p_det = sub.add_parser(
        "detect-events", help="detect NO2 production events on the residual (NOX-004)"
    )
    p_det.add_argument("--z-thresh", type=float, dest="z_thresh", help="robust-z event threshold")
    p_det.set_defaults(func=_cmd_detect_events)

    p_mkt = sub.add_parser(
        "ingest-market", help="fetch free daily prices -> dated snapshot (NOX-004)"
    )
    p_mkt.add_argument("--start", help="ISO start date (default 2019-01-01)")
    p_mkt.add_argument("--end", help="ISO end date (default today)")
    p_mkt.set_defaults(func=_cmd_ingest_market)

    p_cat = sub.add_parser(
        "catalyst", help="events + production match + market study -> report (NOX-004)"
    )
    p_cat.add_argument("--window", type=int, help="+/- trading-day event-study window (default 5)")
    p_cat.add_argument("--latency", type=int, help="overpass+processing latency days (default 2)")
    p_cat.set_defaults(func=_cmd_catalyst)

    p_scale = sub.add_parser(
        "scale-sweep",
        help="AOI extent x grid resolution sensitivity, robust significance (NOX-008)",
    )
    p_scale.add_argument("--buffers", help="comma AOI buffers in deg (default 0.25,0.10)")
    p_scale.add_argument(
        "--resolutions", help="comma targets: 'native' or deg (default native,0.10,0.15,0.25)"
    )
    p_scale.add_argument("--draws", type=int, help="bootstrap/permutation draws (default 5000)")
    p_scale.add_argument("--seed", type=int, help="RNG seed (default 20260614)")
    p_scale.add_argument("--alpha", type=float, help="Benjamini-Hochberg FDR level (default 0.05)")
    p_scale.set_defaults(func=_cmd_scale_sweep)

    p_var = sub.add_parser(
        "variance",
        help="NO2 change-variance decomposition -> steel-local share + change ceiling (NOX-009)",
    )
    p_var.add_argument("--bases", help="comma bases: first-diff,deseason (default both)")
    p_var.add_argument("--resolutions", help="comma resolutions: W,D (default W,D)")
    p_var.add_argument("--draws", type=int, help="moving-block bootstrap draws (default 5000)")
    p_var.add_argument("--seed", type=int, help="RNG seed (default 20260706)")
    p_var.add_argument("--deseason", help="deseason basis method: harmonic|intensity-model")
    p_var.set_defaults(func=_cmd_variance)

    p_sc = sub.add_parser(
        "scale-contrast",
        help="score NO2 features against cluster AND national steel targets (NOX-016)",
    )
    p_sc.add_argument(
        "--probe", action="store_true", help="Arm-B NBS feasibility gate only (REQ-030)"
    )
    p_sc.add_argument("--draws", type=int, help="bootstrap/permutation draws (default 5000)")
    p_sc.add_argument("--seed", type=int, help="RNG seed (default 20260828)")
    p_sc.set_defaults(func=_cmd_scale_contrast)

    p_ring = sub.add_parser(
        "ring-sweep",
        help="background-ring radius sensitivity of the change-variance decomposition (NOX-015)",
    )
    p_ring.add_argument(
        "--rings",
        help="semicolon-separated inner,outer km pairs (default 25,60;40,80;50,90;60,100;60,120)",
    )
    p_ring.add_argument("--resolutions", help="comma resolutions: W,D (default W,D)")
    p_ring.add_argument("--weightings", help="comma weightings: plain,octant (default both)")
    p_ring.add_argument("--draws", type=int, help="moving-block bootstrap draws (default 5000)")
    p_ring.add_argument("--seed", type=int, help="RNG seed (default 20260706)")
    p_ring.add_argument(
        "--wide",
        action="store_true",
        help="NOX-017 wide-AOI sweep: extended ring grid out to 90-150 km over the 1.25-deg cube, "
        "with the three pre-registered rules (level, rise, plateau). Writes its own artefacts",
    )
    p_ring.set_defaults(func=_cmd_ring_sweep)

    p_fviirs = sub.add_parser(
        "fetch-viirs",
        help="acquire VIIRS thermal detections (FIRMS SP-archive primary; vnf dropped); --probe = T1 spike",
    )
    p_fviirs.add_argument(
        "--source",
        choices=("vnf", "firms"),
        help="thermal source (default firms; vnf is dropped/licence-gated)",
    )
    p_fviirs.add_argument("--start", help="ISO start date (default: config 2019-01-01)")
    p_fviirs.add_argument("--end", help="ISO end date (default: today)")
    p_fviirs.add_argument(
        "--probe",
        action="store_true",
        help="T1 feasibility spike: measure volume/fields + go/no-go",
    )
    p_fviirs.set_defaults(func=_cmd_fetch_viirs)

    p_thermal = sub.add_parser(
        "build-thermal-series",
        help="facility-radius join + weekly/daily thermal-activity series (benchmark-blind)",
    )
    p_thermal.add_argument(
        "--source",
        choices=("vnf", "firms"),
        help="thermal source (default firms; vnf is dropped/licence-gated)",
    )
    p_thermal.add_argument("--freq", help="aggregation frequency (pandas alias; default W)")
    p_thermal.add_argument(
        "--radius-km", type=float, dest="radius_km", help="facility join radius km"
    )
    p_thermal.set_defaults(func=_cmd_build_thermal_series)

    p_tri = sub.add_parser(
        "triangulate-thermal",
        help="NO2<->VIIRS + CREA<->VIIRS robust significance -> feature-vs-target diagnosis",
    )
    p_tri.add_argument(
        "--source",
        choices=("vnf", "firms"),
        help="thermal source (default firms; vnf is dropped/licence-gated)",
    )
    p_tri.set_defaults(func=_cmd_triangulate_thermal)

    p_cal = sub.add_parser(
        "assemble-calendar",
        help="assemble/validate the curtailment calendar; --probe = T1 feasibility spike (NOX-012)",
    )
    p_cal.add_argument(
        "--probe",
        action="store_true",
        help="T1 feasibility spike: coverage+provenance report + go/no-go (no build)",
    )
    p_cal.set_defaults(func=_cmd_assemble_calendar)

    p_reg = sub.add_parser(
        "regime-study",
        help="2023 rolling regime + event study + jackknife + scan-max null + FDR (NOX-012)",
    )
    p_reg.add_argument(
        "--window-months",
        type=int,
        dest="window_months",
        help="rolling window in months (default 18)",
    )
    p_reg.add_argument("--draws", type=int, help="bootstrap/permutation/scan-max draws")
    p_reg.add_argument("--seed", type=int, help="RNG seed (default 20260720)")
    p_reg.set_defaults(func=_cmd_regime_study)

    p_vviirs = sub.add_parser(
        "verify-viirs", help="render VIIRS detections over the AOI with facilities overlaid"
    )
    p_vviirs.add_argument(
        "--source",
        choices=("vnf", "firms"),
        help="thermal source (default firms; vnf is dropped/licence-gated)",
    )
    p_vviirs.add_argument("--facilities", help="facilities CSV (default: data/derived/...)")
    p_vviirs.add_argument("--out", help="output PNG path")
    p_vviirs.set_defaults(func=_cmd_verify_viirs)

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if hasattr(args, "func"):
        return args.func(args)
    print(f"[noxus] '{args.command}' over region '{TANGSHAN.name}' is not yet implemented.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
