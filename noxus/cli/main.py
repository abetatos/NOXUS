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
    )
    report = acquire_no2(aoi, cfg)
    print(
        f"[noxus] acquired: fetched={report.fetched} skipped={report.skipped} "
        f"failed_batches={report.failed_batches} -> {report.raw_dir}"
    )
    if report.has_version_discontinuity:
        print(f"[noxus] WARNING: processor-version discontinuity across {set(report.versions)}")
    return 0


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
    from noxus.config.run import SignalConfig
    from noxus.data.era5 import ERA5SnapshotError
    from noxus.signal.index import build_activity_index

    cfg = SignalConfig()
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
    p_fetch.set_defaults(func=_cmd_fetch)

    p_ver = sub.add_parser("verify-no2", help="render NO2 vs facilities on clear high-NO2 days")
    p_ver.add_argument("--raw-dir", help="acquired store (default: data/raw/tropomi)")
    p_ver.add_argument("--days", type=int, default=5, help="number of days to render")
    p_ver.add_argument("--max-cloud", type=float, default=0.2, help="max mean cloud fraction")
    p_ver.add_argument("--facilities", help="facilities CSV (default: data/derived/...)")
    p_ver.add_argument("--out-dir", help="output dir (default: data/derived/verification)")
    p_ver.add_argument("--optical", action="store_true", help="also fetch a Sentinel-2 thumbnail")
    p_ver.set_defaults(func=_cmd_verify_no2)

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
