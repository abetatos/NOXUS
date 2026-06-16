"""Run configuration for benchmark ingestion.

Centralises the knobs the benchmark ingestion needs (source URL, the column to extract, auxiliary
series to retain) so every ingested artifact is reproducible from a recorded configuration. The
validation/correlation configuration will be added alongside the validation module, once the NO2
predictor (attribution stage) exists to validate against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from noxus.config.region import DEFAULT_AOI_BUFFER_DEG, TIGHT_AOI_BUFFER_DEG

# Public, free source of record for the physical-output benchmark: CREA's WIND-sourced steel sheet,
# exported as CSV. The Tangshan blast-furnace operating rate is one column among several steel
# series. Upstream provenance is commercial (WIND / China United Steel Network), republished by CREA.
CREA_BENCHMARK_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/1-LumT7h3GSryFLFYzxK_QedmhI6X4y58/export?format=csv"
)

# Exact header of the benchmark column (selected by name, not position, to survive column drift).
TANGSHAN_BF_COLUMN = "China: Tangshan: Operating Rate of Blast Furnaces"


@dataclass(frozen=True)
class BenchmarkConfig:
    """Where the benchmark comes from and which columns to keep."""

    source_url: str = CREA_BENCHMARK_CSV_URL
    primary_column: str = TANGSHAN_BF_COLUMN
    # Auxiliary steel series in the same CSV, retained for downstream robustness.
    aux_columns: tuple[str, ...] = (
        "China: Estimated Daily Average Output: Pig Iron",
        "China: Estimated Daily Average Output: Crude Steel",
        "China: Blast Furnace Starting Rate (247)",
    )
    source_label: str = "CREA (WIND / China United Steel Network)"


# CDSE OIDC token endpoint for the password grant (mint an access token from CDSE_USERNAME/PASSWORD).
CDSE_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
)


@dataclass(frozen=True)
class AcquisitionConfig:
    """Configuration for TROPOMI NO2 acquisition over the AOI via openEO (NOX-002a).

    Collection/band ids are the CDSE openEO Sentinel-5P defaults and are verified at runtime
    (open question Q1) before any bulk fetch.
    """

    openeo_url: str = "openeo.dataspace.copernicus.eu"
    collection_id: str = "SENTINEL_5P_L2"
    no2_band: str = "NO2"
    cloud_band: str = "CLOUD_FRACTION"
    qa_band: str = "qa_value"
    qa_threshold: float = 0.75
    start: date = date(2019, 1, 1)
    end: date | None = None  # None => today at runtime
    aoi_buffer_deg: float = DEFAULT_AOI_BUFFER_DEG
    raw_dir: Path = Path("data/raw/tropomi")
    batch_freq: str = "MS"  # month-start batches, to respect the openEO credit budget
    # Auth: password grant from .env first, then fall back to interactive openEO OIDC.
    cdse_client_id: str = "cdse-public"
    token_url: str = CDSE_TOKEN_URL
    env_username: str = "CDSE_USERNAME"
    env_password: str = "CDSE_PASSWORD"
    # Sentinel-2 collection for the optional optical verification thumbnail.
    optical_collection_id: str = "SENTINEL2_L2A"
    optical_bands: tuple[str, ...] = field(default_factory=lambda: ("B04", "B03", "B02"))


@dataclass(frozen=True)
class GriddingConfig:
    """Configuration for compositing per-overpass NO2 into an analysis-ready cube (NOX-002b)."""

    freq: str = "W"  # pandas resample alias; weekly matches the benchmark
    min_cell_obs: int = 1  # a cell-period needs >= this many valid overpasses
    min_period_coverage: float = 0.25  # else the whole period is masked (Kondragunta-style)
    raw_dir: Path = Path("data/raw/tropomi")
    out_dir: Path = Path("data/derived/no2")
    emit_aoi_series: bool = True  # interim naive AOI-mean series (pre-attribution)


# Status values in tangshan_steel_facilities.csv that indicate an active integrated plant. "operating"
# is the clear active case; "operating pre-retirement" is still physically running (REQ-001). Other
# values ("retired", "cancelled") are excluded.
ACTIVE_FACILITY_STATUSES = ("operating", "operating pre-retirement")


@dataclass(frozen=True)
class SignalConfig:
    """Configuration for the steel-sector activity signal (NOX-003).

    Drives the whole attribution → index chain so every emitted artifact is reproducible from a
    recorded configuration (REQ-050). Phase-1 fields (footprint, background, paths) are consumed now
    by ``noxus/attribution/source.py``; the later-phase fields (ERA5, deseasonalisation, index) are
    declared here so the config is complete, but their consumers arrive in later phases (T5+).
    """

    # Composite frequency of the signal/index (pandas alias; weekly matches the NOX-002b cube).
    freq: str = "W"

    # --- Footprint sampling (REQ-001/002) -----------------------------------------------------
    # Radius around each operating facility, in km, whose cells form the footprint. Default to be
    # confirmed against real coverage (open question); 15 km is a conservative cluster-scale start.
    footprint_radius_km: float = 15.0
    # Facility statuses treated as active; everything else is excluded.
    active_statuses: tuple[str, ...] = ACTIVE_FACILITY_STATUSES

    # --- Regional background ring (REQ-003) ---------------------------------------------------
    # Annular ring around the cluster, outside the footprint but inside the AOI, used as background.
    background_inner_km: float = 25.0
    background_outer_km: float = 60.0
    # How the footprint is corrected against the background ("subtract" default; "normalise" ratio).
    background_mode: str = "subtract"

    # --- Meteorological normalisation via ERA5 (REQ-010/011) — consumed in T5+ ----------------
    era5_vars: tuple[str, ...] = (
        "u10",
        "v10",
        "blh",
    )  # 10 m wind components + boundary-layer height
    meteo_form: str = "linear"  # "linear" residualisation (default) | "loess"
    # ERA5 ingest is via the Copernicus CDS (server-side AOI subset; decision 2026-06-13, specs Q3).
    # Era of record for the ERA5 snapshot (TROPOMI era start → today); matches the NO2 cube span.
    era5_start: str = "2019-01-01"

    # --- Deseasonalisation & confounders (REQ-020/021/022) — consumed in T7+ ------------------
    # "yoy" default (2026-06-13 sensitivity): removes annual cycle + secular retrofit trend (Li 2024)
    # while keeping year-over-year activity. "stl" / "yoy-double-diff" / "none" also available;
    # double-diff erased the signal on this weekly series, so it is no longer the default.
    # "intensity-model" (NOX-003.1) models the secular intensity decline explicitly (CV-selected
    # smooth trend; residual = activity proxy) instead of differencing blindly.
    deseason_method: str = "yoy"

    # --- Explicit emission-intensity (decoupling) model (NOX-003.1, REQ-101/102) --------------
    # Active only when deseason_method == "intensity-model". Smoothness (effective df) is chosen by
    # cross-validation on the NO2 series ALONE (never against the benchmark, NFR-102).
    intensity_estimator: str = "spline"  # "spline" (df = basis columns, exact) | "loess"
    intensity_df_grid: tuple[float, ...] = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
    intensity_cv_folds: int = 5
    intensity_criterion: str = "blocked-cv"  # "blocked-cv" (time-respecting) | "gcv" (spline only)
    intensity_min_length: int = 24  # refuse to fit a trend on a shorter valid sample (ERR-101)
    # Heating season as (start_month, end_month) inclusive; Tangshan heating ~Nov 15 → Mar 15.
    heating_season_months: tuple[int, ...] = (11, 12, 1, 2, 3)
    # Source of the production-curtailment calendar exogenous control (REQ-022).
    curtailment_source: str = "crea"

    # --- Index construction (REQ-030/031) — consumed in T8+ ----------------------------------
    # Normalisation anchor for the relative index ("zscore" | a baseline period label).
    index_anchor: str = "zscore"
    # Attributable-fraction cap (~30–43% of the column is steel; Wen 2024) recorded in provenance.
    attributable_cap: tuple[float, float] = (0.30, 0.43)

    # --- Paths --------------------------------------------------------------------------------
    cube_path: Path = Path("data/derived/no2/no2_cube_w.nc")
    facilities_csv: Path = Path("data/derived/tangshan_steel_facilities.csv")
    era5_snapshot_dir: Path = Path("data/raw/era5")
    out_dir: Path = Path("data/derived/no2")
    footprint_signal_name: str = "steel_footprint_signal.parquet"
    index_name: str = "steel_activity_index.parquet"
    # Intensity decomposition diagnostic (signal, s(t) trend, activity residual); NOX-003.1, REQ-104.
    decomposition_name: str = "steel_intensity_decomposition.parquet"

    # --- Prophet / harmonic deseasonalisation (NOX-006) ---------------------------------------
    # Active when deseason_method == "prophet" (trend + yearly + weekly Fourier) or "harmonic"
    # (annual Fourier on the intensity residual, dependency-free fallback). Prophet is lazy-imported.
    prophet_growth: str = "linear"
    prophet_changepoint_prior: float = 0.05  # trend flexibility (higher -> wigglier; Q1)
    prophet_yearly_order: int = (
        4  # annual Fourier order (reduced from Prophet's 10; K>=3 overfits, Q2)
    )
    prophet_weekly_order: int = (
        3  # weekly (day-of-week) Fourier order — the source-separation handle
    )
    prophet_min_valid: int = 120  # refuse the fit below this many valid days (ERR-003)
    harmonic_order: int = 1  # annual harmonics removed in the "harmonic" fallback (K=1 sweet spot)
    # Daily NO2 products (NOX-006; gitignored, NO2-derived).
    cube_daily_name: str = "no2_cube_d.nc"
    footprint_daily_name: str = "steel_footprint_daily.parquet"
    prophet_decomposition_name: str = "steel_prophet_decomposition.parquet"


@dataclass(frozen=True)
class ValidationConfig:
    """Configuration for validating the relative index against the CREA benchmark (NOX-003).

    Phase-1 declares this for completeness; its consumers (the revived validation engine) arrive in
    later phases (T9+).
    """

    benchmark_path: Path = Path("data/derived/benchmark_tangshan_bf_operating_rate.parquet")
    freq: str = "W"  # common frequency the index and benchmark are aligned to (REQ-040)
    min_coverage: float = 0.25  # coverage screening floor carried from NOX-002b
    max_lag: int = 8  # lag window (in periods) for the cross-correlation profile (REQ-042)
    min_overlap: int = (
        26  # refuse the statistical tests below this many overlapping periods (ERR-004)
    )
    # Literature success bar on the deseasonalised series (Kim 2023 / Kondragunta 2021).
    success_bar: tuple[float, float] = (0.50, 0.75)
    out_dir: Path = Path("data/derived")
    results_name: str = "steel_validation_results.json"
    summary_name: str = "steel_validation_summary.txt"


@dataclass(frozen=True)
class ScaleSweepConfig:
    """Configuration for the spatial-scale sensitivity sweep (NOX-008).

    Sweeps two scale axes over the EXISTING analysis-ready cube and recomputes the NO2<->BF-rate
    correlation at each scale with autocorrelation-robust, FDR-corrected significance. Both axes are
    pure re-aggregations of the committed cube (clip = subset; coarsen = block-mean) — never
    interpolation (REQ-001/002; native-resolution decision 2026-06-13). Defaults are fixed here so the
    sweep is reproducible and no scale is selected to maximise the benchmark correlation (NFR-003).
    """

    # --- Scale axes ---------------------------------------------------------------------------
    # AOI extent (buffer deg around the facility envelope): default wide vs tight (REQ-001).
    buffers: tuple[float, ...] = (DEFAULT_AOI_BUFFER_DEG, TIGHT_AOI_BUFFER_DEG)
    # Grid resolution targets: "native" keeps the cube; floats coarsen by block-mean to ~that deg
    # spacing (Q1 resolved 2026-06-14: native, 0.10, 0.15, 0.25; mirrors Parubets + the 0.1->0.25 gap).
    resolutions: tuple[object, ...] = ("native", 0.10, 0.15, 0.25)
    # Deseasonalisation variants compared at each scale (reuse noxus.signal.index.deseasonalize).
    variants: tuple[str, ...] = ("level", "intensity-model", "yoy", "stl")
    # Frequencies to align/correlate at (weekly primary; monthly low-power, Q5).
    freqs: tuple[str, ...] = ("W", "ME")

    # --- Robust significance (REQ-020..022) ---------------------------------------------------
    n_boot: int = 5000  # moving-block bootstrap draws for the CI of r
    n_perm: int = 5000  # block-permutation null draws
    seed: int = 20260614  # fixed RNG seed (NFR-001)
    neff_order: str = "auto"  # "first" (Bayley-Hammersley) | "newey-west" | "auto" (report both)
    # Moving-block length rule: round(n ** block_exponent), floored at block_floor (Q2).
    block_exponent: float = 1.0 / 3.0
    block_floor: int = 2

    # --- Multiple testing (REQ-030) -----------------------------------------------------------
    fdr_alpha: float = 0.05  # Benjamini-Hochberg level across the scale x variant x lag family

    # --- Lead-lag + alignment (REQ-040) -------------------------------------------------------
    max_lag: int = 8  # +/- lag window for the cross-correlation profile (periods)
    min_overlap: int = 26  # refuse a scale below this many overlapping periods
    # Coarse scales may leave too few cells for the footprint/background contrast -> AOI-mean
    # fallback, labelled (REQ-011, EDGE-002). Require >= this many footprint cells to keep the contrast.
    min_footprint_cells: int = 4

    # --- Paths --------------------------------------------------------------------------------
    cube_path: Path = Path("data/derived/no2/no2_cube_w.nc")
    benchmark_path: Path = Path("data/derived/benchmark_tangshan_bf_operating_rate.parquet")
    facilities_csv: Path = Path("data/derived/tangshan_steel_facilities.csv")
    out_dir: Path = Path("data/derived")
    results_name: str = "scale_sensitivity.csv"
    figures_dir: Path = Path("docs/figures/exploration")
    findings_name: str = "scale_sensitivity_findings.txt"


# Default market instruments (NOX-004). Free/reproducible via yfinance: global miners + a steel ETF.
# Chinese ferrous futures (SHFE rebar / DCE iron ore / coking coal) lack a clean free API (Q1) and are
# left out of the default set — added behind a best-effort exchange-settlement snapshot when available.
DEFAULT_MARKET_INSTRUMENTS = ("BHP", "RIO", "VALE", "SLX")
MARKET_BENCHMARK = "ACWI"  # broad global-equity benchmark for abnormal-return computation


@dataclass(frozen=True)
class CatalystConfig:
    """Configuration for the NO2 event catalyst (NOX-004).

    Drives event detection → production-event matching → market event-study so every artifact is
    reproducible from a recorded config (REQ-050). Detector + study defaults are fixed here so they are
    chosen BEFORE looking at market returns (anti-overfitting / multiple-testing discipline, REQ-042).
    """

    # --- Event detection (REQ-002..005) -------------------------------------------------------
    freq: str = "W"  # the residual cadence (weekly cube)
    detector: str = "zscore"  # "zscore" (robust MAD-z) | "cusum" | "both"
    z_thresh: float = 2.0  # |robust z| above which a residual deviation is an event
    detect_min_periods: int = 12  # causal expanding baseline needs >= this many past points
    min_coverage: float = 0.25  # event period coverage floor (inherited NOX-002b)
    # Strengthened meteo control: reject an event explained by a same-sign ventilation anomaly.
    meteo_screen: bool = True
    ventilation_z: float = 1.5  # |ventilation-index z| above which weather is the likely cause

    # --- Ground-truth production events (REQ-010..012) ----------------------------------------
    bf_event_z: float = 1.5  # |robust z| of the weekly BF-rate change that marks a production event
    curtailment_calendar: Path | None = None  # public MEE/CREA calendar (Q4); None => BF jumps only

    # --- Matching + event study (REQ-020..033, 041, 042) -------------------------------------
    match_window: int = 2  # periods within which an NO2 event matches a production event
    study_window: int = 5  # +/- trading days for the market cumulative-abnormal-return window
    overpass_latency_days: int = (
        2  # overpass + processing latency before the first tradeable session
    )
    min_events: int = 5  # refuse the study below this many (confirmed) events (ERR-003)
    instruments: tuple[str, ...] = DEFAULT_MARKET_INSTRUMENTS
    market_benchmark: str = MARKET_BENCHMARK

    # --- Paths --------------------------------------------------------------------------------
    decomposition_path: Path = Path("data/derived/no2/steel_intensity_decomposition.parquet")
    benchmark_path: Path = Path("data/derived/benchmark_tangshan_bf_operating_rate.parquet")
    era5_snapshot_dir: Path = Path("data/raw/era5")
    market_snapshot_dir: Path = Path("data/raw/market")
    events_out: Path = Path("data/derived/no2/steel_no2_events.parquet")
    out_dir: Path = Path("data/derived")
    results_name: str = "catalyst_results.json"
    summary_name: str = "catalyst_summary.txt"
