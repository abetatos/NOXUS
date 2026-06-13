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

from noxus.config.region import DEFAULT_AOI_BUFFER_DEG

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
