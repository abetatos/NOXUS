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


@dataclass(frozen=True)
class VarianceDecompConfig:
    """Configuration for the NO2 change-variance decomposition (NOX-009, REQ-060).

    Measures — benchmark-blind — what fraction of the *variance of changes* in the meteo-normalised
    footprint NO2 signal is attributable to a four-term budget (regional / meteo / steel-local /
    noise), on two change bases (first differences; deseasonalised residual) at two resolutions
    (weekly cube; NOX-006 daily cube), and converts the steel-local share into an implied SNR ceiling
    for a *change* detector, contrasted with the imported mean-level (Wen) ceiling. Defaults are fixed
    here so the run is reproducible and no basis/resolution is selected to flatter the conclusion
    (NFR-003/004). The CREA benchmark is *never* referenced (NFR-001).
    """

    # --- Change bases & resolutions (REQ-010/011) ---------------------------------------------
    # "first-diff" = gap-aware first differences; "deseason" = the deseasonalised residual.
    bases: tuple[str, ...] = ("first-diff", "deseason")
    # Cube resolutions to run: "W" weekly cube, "D" NOX-006 daily cube.
    resolutions: tuple[str, ...] = ("W", "D")
    # Deseasonalisation variant for the "deseason" basis (reuses noxus.signal.index.deseasonalize):
    # "harmonic" = NOX-003.1 intensity trend + K=1 annual harmonic (dependency-free); "intensity-model"
    # = the NOX-003.1 intensity residual. Never selected against the benchmark (NFR-001).
    deseason_method: str = "harmonic"

    # --- Partition (REQ-020/021) --------------------------------------------------------------
    # ERA5 covariates forming the meteo term (linear regress-out, A5). "wind_speed" is derived.
    meteo_covariates: tuple[str, ...] = ("u10", "v10", "blh", "wind_speed")
    # Fixed nested entry order (A2): regional common mode -> meteo -> steel-local footprint residual.
    order: tuple[str, ...] = ("regional", "meteo", "local")
    # Imported mean-level attributable cap (Wen 2024) used ONLY for the level-ceiling contrast.
    attributable_cap: tuple[float, float] = (0.30, 0.43)

    # --- Autocorrelation-aware CI (REQ-040, A4) -----------------------------------------------
    n_boot: int = 5000  # moving-block bootstrap draws for each share/ceiling CI
    seed: int = 20260706  # fixed RNG seed (NFR-004)
    block_exponent: float = 1.0 / 3.0  # moving-block length rule (reused from NOX-008, A4)
    block_floor: int = 2

    # --- Paths --------------------------------------------------------------------------------
    cube_path_weekly: Path = Path("data/derived/no2/no2_cube_w.nc")
    cube_path_daily: Path = Path("data/derived/no2/no2_cube_d.nc")
    facilities_csv: Path = Path("data/derived/tangshan_steel_facilities.csv")
    era5_snapshot_dir: Path = Path("data/raw/era5")
    out_dir: Path = Path("data/derived")
    results_name: str = "variance_decomposition.csv"
    figures_dir: Path = Path("docs/figures/exploration")
    findings_name: str = "variance_decomposition_findings.txt"
    # Optional gitignored per-basis change-series cache (NO2-derived, NFR-006).
    series_cache_dir: Path = Path("data/derived/no2/variance")


@dataclass(frozen=True)
class ViirsAcquisitionConfig:
    """Configuration for VIIRS thermal-anomaly acquisition over the AOI (NOX-011, REQ-001..006/040).

    Mirrors :class:`AcquisitionConfig` (TROPOMI): a resumable, provenance-tagged, credentials-by-
    reference acquisition. The PRIMARY physical proxy is now ``firms`` (NASA FIRMS active fire, key
    ``FIRMS_MAP_KEY`` by reference) — its SP-archive ``type=2`` static-land detections over the steel
    cluster are the space-based thermometer. ``vnf`` (VIIRS Nightfire, EOG/CSM) was the original
    primary but is DROPPED as of the 2026-08-22 pivot: it is no longer public (VIIRS Nightfire Data
    Use License eff. 2025-01-10; EOG programmatic access paid-subscriber-only eff. 2026-06-01), which
    conflicts with the free/public-data principle — the ``vnf`` source now surfaces a clear
    licence/pay-gated error (see :class:`VnfFetcher`). Defaults are recorded in the spec and verified
    by the T1 ``--probe`` before any bulk fetch.
    """

    source: str = "firms"  # "firms" (PRIMARY, public) | "vnf" (DROPPED — licence/pay-gated error)
    # VNF (EOG / Colorado School of Mines): retained only so ``source="vnf"`` can raise a clear
    # licence/pay-gated error. No longer fetched (2026-08-22 pivot; not public/free).
    vnf_endpoint_base: str = "https://eogdata.mines.edu/wwwdata/viirs_products/vnf"
    vnf_file_template: str = "VNF_npp_d{yyyymmdd}_noaa_v30.csv.gz"
    # NASA FIRMS area+date CSV API. product = collection id; key read from the env var by name only.
    firms_endpoint_base: str = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
    # SP (Standard-Processing) archive products: they carry the `type` column needed for the type=2
    # industrial filter and cover the 2019+ historical era. NRT products (VIIRS_*_NRT) are recent but
    # have NO `type` column, so they are NOT used for the historical series. The SP archive LAGS the
    # present by ~`firms_archive_lag_days` (recent dates return 0 rows).
    firms_products: tuple[str, ...] = ("VIIRS_SNPP_SP", "VIIRS_NOAA20_SP")
    firms_env_key: str = (
        "FIRMS_MAP_KEY"  # referenced by NAME only; value never read into logs/output
    )
    # FIRMS detection "type" code kept for the industrial proxy: 2 == "other static land source"
    # (0 presumed vegetation fire, 1 active volcano, 3 offshore are excluded — EDGE-004). Applies to
    # the SP archive only (NRT products lack the `type` column).
    firms_static_land_type: int = 2
    # SP-archive publication lag (days). The archive trails the present by ~2 months; the probe
    # anchors its window back by this offset and the acquisition clamps a "chase-today" end by it so
    # neither chases empty recent SP territory. Safe default ~2.5 months; override for a fresher run.
    firms_archive_lag_days: int = 75
    # Optional explicit probe-window anchor date (window end). None => today - firms_archive_lag_days
    # for FIRMS (an SP-archive-appropriate window), or today for other sources.
    probe_anchor: date | None = None

    start: date = date(2019, 1, 1)  # TROPOMI/CREA overlap era
    end: date | None = None  # None => (today - archive lag) for FIRMS at runtime (clamped)
    aoi_buffer_deg: float = DEFAULT_AOI_BUFFER_DEG
    raw_dir: Path = Path("data/raw/viirs")

    # Politeness / resumability (NFR-007): batch by date window, retry with backoff.
    batch_window_days: int = 30
    max_retries: int = 3
    backoff_seconds: float = 2.0
    request_timeout: float = 120.0

    # Download budget gate (REQ-000/ERR-004): the T1 spike measures per-night volume; if the implied
    # full-era volume exceeds this, the spike records a no-go and no bulk download starts.
    download_budget_mb: float = 2048.0
    probe_days: int = 7  # size of the T1 feasibility window

    # Optional quality screen retained per detection (REQ-002); None => keep all detections.
    min_temperature_k: float | None = None


@dataclass(frozen=True)
class ThermalTriangulationConfig:
    """Configuration for the VIIRS facility join, thermal series and triangulation (NOX-011).

    Pre-registration knobs (A2/A3/A9, NFR-006): the facility radius, aggregation frequency and primary
    aggregate are fixed here **before** the triangulation is viewed, so the thermal index cannot be
    tuned to the correlation. The triangulation reuses ``noxus.validation.robust`` unchanged (REQ-020).
    """

    # --- Facility-radius join (REQ-010, A2) ---------------------------------------------------
    radius_km: float = 3.0  # pre-registered; brackets the VIIRS pixel footprint + plant extent (Q3)

    # --- Aggregation (REQ-011, A3/A5) ---------------------------------------------------------
    freq: str = "W"  # weekly primary (matches the CREA benchmark + NO2 cube); "D" secondary
    primary_aggregate: str = "radiant_heat"  # summed radiant heat (MW) is the headline index
    sparse_min_nights: int = 2  # weeks with fewer detection-nights than this are flagged (EDGE-005)

    # --- Robust significance (REQ-020..022; reused from NOX-008) -------------------------------
    n_boot: int = 5000  # moving-block bootstrap draws for the CI of r
    n_perm: int = 5000  # block-permutation null draws
    seed: int = 20260712  # fixed RNG seed (NFR-001)
    max_lag: int = 8  # +/- lead/lag window (periods)
    min_overlap: int = 26  # refuse a correlation below this many overlapping periods
    fdr_alpha: float = 0.05  # Benjamini-Hochberg level across the correlation family (REQ-022)

    # --- Inputs (read-only) + committable outputs (REQ-031) -----------------------------------
    facilities_csv: Path = Path("data/derived/tangshan_steel_facilities.csv")
    # Existing NO2-footprint series + CREA benchmark, aligned on the common window (EDGE-003).
    # NO2 variant of record for the triangulation: the *background-corrected attributed footprint NO2
    # level* — i.e. the `no2_corrected` column of `steel_footprint_signal.parquet`, the canonical
    # series `noxus attribute` (attribution.source.build_footprint_signal) actually writes. This keeps
    # `triangulate-thermal` reproducible from the CLI with no hand-materialised intermediate.
    no2_series_path: Path = Path("data/derived/no2/steel_footprint_signal.parquet")
    no2_series_column: str = "no2_corrected"
    benchmark_path: Path = Path("data/derived/benchmark_tangshan_bf_operating_rate.parquet")
    raw_dir: Path = Path("data/raw/viirs")
    out_dir: Path = Path("data/derived")
    series_name: str = "viirs_thermal_series.parquet"
    results_name: str = "viirs_triangulation.csv"
    figures_dir: Path = Path("data/derived/verification/viirs")


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


# Chinese public-holiday windows that securely bound the coarse-floor curtailment calendar
# (NOX-012 T1/T2). These are matters of public record (State Council holiday arrangements /
# fixed National-Day Golden Week) — used as the *safe coarse floor* (A3/EDGE-001), never as
# fabricated event-level curtailment campaigns. Dates are (start, end) inclusive, ISO strings.
# Spring Festival (Lunar New Year) — steel demand + logistics slow; a documented production dip.
SPRING_FESTIVAL_WINDOWS = (
    ("2019-02-04", "2019-02-10"),
    ("2020-01-24", "2020-01-30"),
    ("2021-02-11", "2021-02-17"),
    ("2022-01-31", "2022-02-06"),
    ("2023-01-21", "2023-01-27"),
    ("2024-02-10", "2024-02-16"),
    ("2025-01-28", "2025-02-03"),
)
# National Day Golden Week — fixed Oct 1–7 every year (public record).
GOLDEN_WEEK_YEARS = tuple(range(2019, 2026))


@dataclass(frozen=True)
class RegimeStudyConfig:
    """Configuration for the 2023-regime event study + curtailment calendar (NOX-012, REQ-051).

    Drives calendar assembly (T1 probe + T2 build), rolling-regime identification, the
    regime-conditioned event study, the leave-one-event-out jackknife, the 2024-collapse dual
    diagnostic, the rolling-window scan-maximum null and the BH-FDR grid. Every knob is fixed here
    **before** any CREA outcome is viewed (anti-p-hacking, NFR-005): no window, threshold, deseason
    variant or regime boundary is selected to maximise agreement with the benchmark. The estimators
    expose **no** CREA-agreement objective (ERR-004). Reuses ``noxus.validation.robust`` and the
    NOX-004 catalyst primitives unchanged.
    """

    # --- Rolling-regime identification (REQ-010/011) ------------------------------------------
    freq: str = "W"  # residual cadence (weekly cube); "ME" monthly secondary
    window_months: int = 18  # the rolling-correlation window of record (~78 weeks)
    window_weeks_secondary: int = 52  # a 52-week secondary window reported alongside
    # Deseason variants in the fixed grid; the harmonic-K1 method of record headlines (Q3).
    variants: tuple[str, ...] = ("harmonic", "intensity-model")
    headline_variant: str = "harmonic"
    # Annual harmonics removed in the "harmonic" variant; K=1 is the method of record (mirrors
    # SignalConfig.harmonic_order). Consumed by deseasonalize() on the real file path (regime.py).
    harmonic_order: int = 1

    # --- Regime segmentation (fixed pre-registered rule, Q4/A5) -------------------------------
    # The "2023 peak" regime is the ``window_months`` window centred on the located rolling-r peak;
    # the "2024 collapse" is a fixed calendar-year split. Boundaries are NEVER tuned to CREA.
    collapse_year: int = 2024
    pre_period: tuple[str, str] = ("2022-01-01", "2023-12-31")  # cadence baseline for the collapse
    boundary_robustness_periods: tuple[int, ...] = (-4, 0, 4)  # +/- shift sweep on the peak centre

    # --- Regime-conditioned event study (REQ-020) --------------------------------------------
    z_thresh: float = 2.0  # NO2 event detector threshold (reused detect_events default)
    detect_min_periods: int = 12
    min_coverage: float = 0.25
    meteo_screen: bool = True
    ventilation_z: float = 1.5
    bf_event_z: float = 1.5  # BF-rate jump threshold for ground-truth production events
    match_windows: tuple[int, ...] = (2,)  # +/- periods for the NO2<->production match grid

    # --- Event-magnitude-threshold grid (REQ-031) --------------------------------------------
    thresholds: tuple[float, ...] = (1.5, 2.0, 2.5)

    # --- Event-bound jackknife (REQ-021, ERR-003) --------------------------------------------
    jackknife_event_window: int = 4  # +/- periods removed around each large event
    min_events: int = 3  # refuse the event-level jackknife below this many (ERR-003 -> coarse)
    # Fraction of the peak r that must be lost by removing the top events to call it "event-bound".
    event_bound_drop_frac: float = 0.5

    # --- Scan-maximum null (REQ-030, A6) -----------------------------------------------------
    scanmax_draws: int = 2000  # block-permutation draws for the scan-max distribution
    n_boot: int = 5000  # moving-block bootstrap draws for the robust CI of r
    n_perm: int = 5000  # block-permutation null draws per regime robust_corr
    seed: int = 20260720  # fixed RNG seed (NFR-001)
    block_exponent: float = 1.0 / 3.0  # moving-block length rule (reused from NOX-008)
    block_floor: int = 2

    # --- Multiple testing (REQ-031/032) ------------------------------------------------------
    fdr_alpha: float = 0.05  # Benjamini-Hochberg level across the fixed grid

    # --- Calendar assembly + T1 probe (REQ-001/002) ------------------------------------------
    calendar_start_year: int = 2019
    calendar_end_year: int = 2025
    # Heating-season restriction window months (Tangshan/Hebei ~Nov 15 -> Mar 15).
    heating_season: tuple[str, str] = ("11-15", "03-15")
    # T1 go/no-go thresholds (REQ-001): event-level needs >= this many securely-dated day/week events
    # spanning the era with >=1 in the 2023-peak window and >=1 in 2024.
    min_event_level: int = 6
    peak_window: tuple[str, str] = ("2023-01-01", "2023-12-31")
    collapse_window: tuple[str, str] = ("2024-01-01", "2024-12-31")

    # --- Inputs (read-only) ------------------------------------------------------------------
    decomposition_path: Path = Path("data/derived/no2/steel_intensity_decomposition.parquet")
    benchmark_path: Path = Path("data/derived/benchmark_tangshan_bf_operating_rate.parquet")
    era5_snapshot_dir: Path = Path("data/raw/era5")
    calendar_path: Path = Path("data/curtailment/tangshan_hebei_curtailment.csv")
    sources_path: Path = Path("data/curtailment/SOURCES.md")
    raw_calendar_dir: Path = Path("data/curtailment/_raw")  # gitignored bulk scrapes (if any)

    # --- Outputs (small stats committable; NO2-derived gitignored) ---------------------------
    out_dir: Path = Path("data/derived")
    results_name: str = "regime_study.csv"
    event_study_name: str = (
        "regime_event_study.csv"  # per-regime precision/recall/lead (REQ-020/050)
    )
    summary_name: str = "regime_study_summary.txt"
    events_out: Path = Path("data/derived/no2/regime_events.parquet")  # gitignored (NO2-derived)
    figures_dir: Path = Path("docs/figures/exploration")
    findings_name: str = "regime_2023_findings.txt"

    def window_periods(self) -> int:
        """The rolling window length in observation periods for the configured cadence."""
        if self.freq.upper().startswith("W"):
            return int(round(self.window_months * 52.0 / 12.0))
        if self.freq.upper() in ("M", "ME", "MS"):
            return int(self.window_months)
        return int(round(self.window_months * 52.0 / 12.0))


@dataclass(frozen=True)
class FluxDivergenceConfig:
    """Minimal frozen config for the NOX-005 T1 feasibility gate (REQ-015, AT7).

    This is intentionally the *smallest* surface the T1 probe needs — the raw-L2 bounded fetch, the
    minimal oversample -> flux -> divergence(+V/tau) chain, and the fixed benchmark-blind verdict
    thresholds. T2 builds out the full config surface (full tau/wind sweeps, freq, compute-budget
    knobs for the multi-year run); do not add those here. Frozen so no default can be tuned against
    an outcome (anti-p-hacking, NFR-002). The CREA benchmark is NEVER referenced by this config.

    Q1 (raw-L2 free access) is resolved live at T1: the CDSE OData catalogue search + password-grant
    token + authenticated download are exercised on a BOUNDED subset (``max_probe_granules``). No
    paid/licence-gated source is permitted (ERR-005); if the free route is unavailable the probe
    emits NO-GO-DATA without a heavy download.
    """

    # --- AOI + bounded subset window (laptop-feasible; never the full multi-year record) --------
    aoi_buffer_deg: float = DEFAULT_AOI_BUFFER_DEG
    subset_start: date = date(2023, 6, 1)
    subset_end: date = date(2023, 6, 15)  # ~2-week window → enough daytime overpasses to power N
    max_probe_granules: int = 1  # single-overpass baseline (retained for the n=1 comparison)

    # --- Extended multi-overpass probe (powers the gate; noise should fall ~1/sqrt(N) with N) ----
    # Default N=4 is what the committed artifact was produced from: the free CDSE zipper endpoint
    # throttles sustained bulk pulls (~1.6-7 MB/s here), so the developer-approved 10-12 overpasses
    # (~6-8 GB) were impractical in this environment; N=4 (~2.4 GB) already gives a decisive curve
    # (noise averages down while SNR stays flat). Raise n_overpasses / download_budget_gb where
    # bandwidth allows — the budget cap keeps the run laptop-feasible either way.
    n_overpasses: int = 4  # cap on granules for the powered probe (~0.6 GB each → ~2.4 GB)
    download_budget_gb: float = 3.0  # hard download cap (laptop-feasible; refuse to exceed)
    one_per_day: bool = True  # one overpass per calendar day → varied winds cancel advection dipoles
    scaling_ns: tuple[int, ...] = (1, 2, 4, 8)  # SNR-vs-N subset sizes (auto-clamped to available)

    # --- Oversampling + flux + divergence operators (fixed pre-outcome) ------------------------
    resolution_km: float = 5.0  # fine-grid spacing for the oversampled TVCD field
    qa_threshold: float = 0.75  # L2 qa_value screen (Beirle/CDSE recommended floor)
    min_pixels: int = 30  # refuse the coherence assessment below this many QA-valid AOI pixels
    wind_level: str = "10m"  # "10m" (ERA5 u10/v10) default; boundary-layer mean is a T2 option (Q3)
    wind_min_ms: float = 1.0  # calm-wind floor; below this the flux is flagged (EDGE-003)
    tau_hours: float = 4.0  # fixed literature NO2 lifetime central value (Q4); sink = V / tau
    coloc_radius_km: float = 15.0  # facility footprint radius for the hotspot co-location test

    # --- Three-way verdict thresholds (fixed constants; benchmark-blind) -----------------------
    # A source term is "coherent + localised" when positive-E hotspots preferentially fall over the
    # plant footprints (co-location ratio > 1) AND the footprint source rises above the field noise.
    snr_go: float = 3.0  # footprint-source signal-to-noise for a clean GO
    snr_degraded: float = 1.5  # partial/noisy but usable -> GO-degraded
    coloc_go: float = 1.5  # hotspot-over-footprint enrichment vs area expectation for GO
    coloc_degraded: float = 1.0  # any positive enrichment -> at least GO-degraded
    hotspot_top_frac: float = 0.05  # top fraction of positive-E cells treated as "hotspots"

    # --- CDSE free access route (OData catalogue + password-grant token + zipper download) ------
    odata_url: str = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
    zipper_url: str = "https://zipper.dataspace.copernicus.eu/odata/v1/Products"
    token_url: str = CDSE_TOKEN_URL
    cdse_client_id: str = "cdse-public"
    env_username: str = "CDSE_USERNAME"
    env_password: str = "CDSE_PASSWORD"
    # Hostnames that are part of the FREE public CDSE route. Any other download host is refused
    # (ERR-005): the raw-L2 path must never reach a paid/licence-gated source (the NOX-011 lesson).
    free_hosts: tuple[str, ...] = (
        "catalogue.dataspace.copernicus.eu",
        "zipper.dataspace.copernicus.eu",
        "download.dataspace.copernicus.eu",
        "identity.dataspace.copernicus.eu",
        "eodata.dataspace.copernicus.eu",
    )
    request_timeout: float = 600.0

    # --- Paths ---------------------------------------------------------------------------------
    raw_dir: Path = Path("data/raw/tropomi_l2")  # gitignored raw swath store (data/raw/)
    era5_snapshot_dir: Path = Path("data/raw/era5")
    facilities_csv: Path = Path("data/derived/tangshan_steel_facilities.csv")
    active_statuses: tuple[str, ...] = ACTIVE_FACILITY_STATUSES
    out_dir: Path = Path("data/derived/fluxdiv")
    probe_report_name: str = "fluxdiv_probe.txt"

    def probe_report_path(self) -> Path:
        """Committable T1 GO/GO-degraded/NO-GO(/NO-GO-DATA) verdict report path."""
        return self.out_dir / self.probe_report_name


@dataclass(frozen=True)
class NowcastConfig:
    """Configuration for the investment-grade activity nowcast (NOX-013).

    Frozen so the fixed defaults cannot be tuned against outcomes (anti-p-hacking, NFR-002). This
    is the *minimal* config the T1 feasibility spike (``noxus.nowcast.probe``) needs; T2 extends it
    (expectation-baseline, surprise, triangulation knobs) without restructuring these fields.

    Feature priority encodes the roadmap: the VIIRS physical proxy (NOX-011) is UNAVAILABLE (VNF
    paywalled; FIRMS active-fire empirically invalid for this cluster, finding 2026-08-27), so the
    main path is the NO2 event markers (NOX-004/012) + the NO2 intensity residual covariate
    (NOX-003.1). The VIIRS entry stays first only as a dormant branch that fires solely if a
    *valid* thermal series is ever supplied; the probe never fabricates one.
    """

    # --- Feature roadmap (VIIRS dormant/unavailable → events → NO2 residual covariate) --------
    feature_priority: tuple[str, ...] = ("viirs", "events", "no2_residual")

    # --- Cadence + causal-emission knobs (fixed defaults) -------------------------------------
    freq: str = "W"  # weekly primary (matches the CREA benchmark + NO2 cube)
    latency_days: int = 2  # overpass + processing latency → tradable next session (REQ-005)
    min_history: int = 26  # warm-up periods before a z-score / correlation is trusted (EDGE-006)

    # --- Incremental-information quick pass (reused NOX-008 robust inference; no tuning) -------
    seed: int = 20260712  # fixed RNG seed (NFR-001) — shared with the thermal triangulation
    n_boot: int = 5000  # moving-block bootstrap draws
    n_perm: int = 5000  # block-permutation null draws
    max_lag: int = 8  # +/- lead/lag window (periods) for lead_lag
    min_coverage: float = 0.25  # coverage screen inherited by align_series (NOX-003)
    min_overlap: int = 26  # refuse an incremental-info verdict below this many common weeks
    sig_alpha: float = 0.05  # robust-significance / BH-FDR level (REQ-011)

    # --- Triangulation sources (QC only; resolved + snapshotted at T7, placeholder here) ------
    triangulation_sources: tuple[str, ...] = (
        "iron_ore_shipments",
        "capesize_freight",
        "coking_coal_exports",
    )

    # --- Inputs (read-only, real on-disk artefacts) ------------------------------------------
    benchmark_path: Path = Path("data/derived/benchmark_tangshan_bf_operating_rate.parquet")
    # NO2 intensity residual of record (the background-corrected attributed footprint level).
    no2_residual_path: Path = Path("data/derived/no2/steel_footprint_signal.parquet")
    no2_residual_column: str = "no2_corrected"
    # NO2 intensity decomposition residual covariate (NOX-003.1).
    intensity_decomp_path: Path = Path("data/derived/no2/steel_intensity_decomposition.parquet")
    intensity_residual_column: str = "residual_activity"
    # NO2 event/catalyst markers (NOX-004/012) — presence signals event-feature availability.
    events_results_path: Path = Path("data/derived/catalyst_results.json")
    # VIIRS thermal proxy: the series file may exist but is empirically INVALID; the triangulation
    # CSV's significance verdict is the validity gate for the dormant GO-primary branch.
    thermal_series_path: Path = Path("data/derived/viirs_thermal_series.parquet")
    thermal_triangulation_path: Path = Path("data/derived/viirs_triangulation.csv")

    # --- Outputs ------------------------------------------------------------------------------
    out_dir: Path = Path("data/derived")
    probe_report_name: str = "nowcast_probe.txt"

    def probe_report_path(self) -> Path:
        """Absolute-relative path of the committable T1 go/no-go verdict report."""
        return self.out_dir / self.probe_report_name
