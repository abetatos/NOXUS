"""T1 feasibility spike — the go/no-go gate that GATES THE WHOLE flux-divergence build (NOX-005).

Mirroring the NOX-011/012/013 T1 pattern, this spike runs the **minimal** Beirle chain on a
**bounded, laptop-feasible** subset of REAL raw L2 data and emits a benchmark-blind, three-way
verdict before any of the full pipeline is built. It answers, without ever reading the CREA benchmark:

* Does the FREE public CDSE raw-L2 access path actually work end-to-end (Q1)? If not — no credentials,
  no network, or only a paid source — the spike STOPS with **NO-GO-DATA** and does *not* download
  anything heavy (NFR-001, ERR-005).
* On the bounded subset, does a coherent, physically-plausible, spatially-localised source term appear
  over the steel cluster **at all**: (a) do positive divergence/source hotspots co-locate with the
  committed facility footprints, and (b) is the footprint source above the field noise?

Verdict (three-way + the data-gate):

* ``GO``           — a coherent co-located source term rises above noise → build the full attribution.
* ``GO-degraded``  — a partial/noisy but usable source term → build with caveats.
* ``NO-GO``        — no coherent local source (honest null; the expected-hard case in high-background
  NE China) → stop and escalate to the developer (ERR-002).
* ``NO-GO-DATA``   — the free raw-L2 route is unavailable → stop; the compute/method question is moot
  until access is resolved (never a fabricated GO).

The verdict, its evidence, and the MEASURED compute/storage profile (download volume, wall-clock,
peak memory of the bounded run, extrapolated to a full multi-year run so the developer can make the
Q2 resource decision) are written to ``data/derived/fluxdiv/fluxdiv_probe.txt``.

Scope: the oversample -> flux -> divergence(+V/tau) operators here are the *minimal* T1 chain
(private helpers), not the full T3/T4/T5 modules. The CREA benchmark is never referenced.
"""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np

from noxus.config.region import Region, tangshan_aoi
from noxus.config.run import FluxDivergenceConfig
from noxus.data.tropomi_l2 import (
    CDSEFetcher,
    L2AccessError,
    L2Field,
    L2Fetcher,
    PaidSourceError,
    fetch_l2,
    read_l2_field,
)

_SCALE_CAVEAT = (
    "Scale caveat: Tangshan is ~13% of national steel; this is a cluster signal, "
    "Hebei-provincial statistics preferred where available."
)

_EARTH_R_M = 6_371_000.0
# TROPOMI/CREA overlap era used only to EXTRAPOLATE the bounded-subset cost to a full run (Q2).
_ERA_START = date(2019, 1, 1)


# ==================================================================================================
# Minimal T1 operator chain (oversample -> flux -> divergence + chemical-loss sink). Pure numpy.
# ==================================================================================================


def _regular_grid(aoi: Region, resolution_km: float) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Build a regular lon/lat grid over the AOI at ~``resolution_km`` spacing.

    Returns ``(lon_centres, lat_centres, dx_m, dy_m)``. Longitude spacing in metres is taken at the
    AOI centre latitude (adequate for a small cluster AOI; the full method refines this in T3).
    """
    x0, y0, x1, y1 = aoi.as_bbox()
    lat_c = 0.5 * (y0 + y1)
    dlat = resolution_km / 111.0
    dlon = resolution_km / (111.0 * max(0.1, np.cos(np.radians(lat_c))))
    lats = np.arange(y0, y1 + dlat, dlat)
    lons = np.arange(x0, x1 + dlon, dlon)
    dx_m = resolution_km * 1000.0
    dy_m = resolution_km * 1000.0
    return lons, lats, dx_m, dy_m


def _oversample(
    field: L2Field, lons: np.ndarray, lats: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate native L2 footprints onto the regular grid → per-cell mean TVCD + valid count.

    Cells with no contributing pixel are NaN (never interpolated/fabricated, EDGE-002).
    """
    lon_edges = _edges(lons)
    lat_edges = _edges(lats)
    total, _, _ = np.histogram2d(
        field.lat, field.lon, bins=[lat_edges, lon_edges], weights=field.no2
    )
    count, _, _ = np.histogram2d(field.lat, field.lon, bins=[lat_edges, lon_edges])
    with np.errstate(invalid="ignore"):
        mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return mean, count


def _edges(centres: np.ndarray) -> np.ndarray:
    step = centres[1] - centres[0] if len(centres) > 1 else 1.0
    return np.concatenate([centres - step / 2.0, [centres[-1] + step / 2.0]])


def _flux(tvcd: np.ndarray, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal NO2 flux F = V * wind (uniform effective wind over the small AOI). mol m-1 s-1."""
    return tvcd * u, tvcd * v


def _divergence(
    fx: np.ndarray, fy: np.ndarray, tvcd: np.ndarray, dx_m: float, dy_m: float, tau_s: float
) -> np.ndarray:
    """Emission-rate field E = div(F) + V/tau (chemical-loss corrected). mol m-2 s-1.

    Gradients are taken on the filled field (NaN → field mean) so finite differences are defined,
    then E is masked back to NaN wherever the underlying TVCD cell was NaN — the divergence is never
    reported where there is no observation.
    """
    valid = np.isfinite(tvcd)
    fill = np.nanmean(tvcd) if valid.any() else 0.0
    fx_f = np.where(np.isfinite(fx), fx, fill * 0.0)
    fy_f = np.where(np.isfinite(fy), fy, fill * 0.0)
    # axis 0 = latitude (y), axis 1 = longitude (x)
    dfy_dy = np.gradient(fy_f, dy_m, axis=0)
    dfx_dx = np.gradient(fx_f, dx_m, axis=1)
    sink = np.where(valid, tvcd, 0.0) / tau_s
    e = dfx_dx + dfy_dy + sink
    return np.where(valid, e, np.nan)


# ==================================================================================================
# Facility footprint mask + coherence assessment (benchmark-blind)
# ==================================================================================================


def _haversine_km(lon1: np.ndarray, lat1: np.ndarray, lon2: float, lat2: float) -> np.ndarray:
    lon1r, lat1r, lon2r, lat2r = map(np.radians, (lon1, lat1, lon2, lat2))
    dlon = lon2r - lon1r
    dlat = lat2r - lat1r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * _EARTH_R_M / 1000.0 * np.arcsin(np.sqrt(a))


def _footprint_mask(
    lons: np.ndarray, lats: np.ndarray, facilities: np.ndarray, radius_km: float
) -> np.ndarray:
    """Boolean grid: cells within ``radius_km`` of any facility (lon, lat) row in ``facilities``."""
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    mask = np.zeros(lon_grid.shape, dtype=bool)
    for flon, flat in facilities:
        mask |= _haversine_km(lon_grid, lat_grid, float(flon), float(flat)) <= radius_km
    return mask


@dataclass(frozen=True)
class CoherenceEvidence:
    """Benchmark-blind spatial-coherence measurement of the source-term field over the cluster."""

    n_pixels: int
    n_valid_cells: int
    n_footprint_cells: int
    wind_u: float
    wind_v: float
    wind_speed: float
    calm: bool
    e_footprint_median: float
    e_background_median: float
    e_noise: float
    snr: float
    coloc_ratio: float
    note: str = ""


def assess_coherence(
    field: L2Field,
    u: float,
    v: float,
    facilities: np.ndarray,
    cfg: FluxDivergenceConfig,
) -> CoherenceEvidence:
    """Run the minimal chain on one real field and measure source co-location + signal-vs-noise.

    Pure w.r.t. IO (field + wind + facilities in), so it is unit-testable with a synthetic planted
    point source. Never reads CREA; all thresholds are fixed ``cfg`` constants (anti-p-hacking).
    """
    lons, lats, dx_m, dy_m = _regular_grid(_aoi_from_cfg(cfg), cfg.resolution_km)
    tvcd, _count = _oversample(field, lons, lats)
    fx, fy = _flux(tvcd, u, v)
    e = _divergence(fx, fy, tvcd, dx_m, dy_m, cfg.tau_hours * 3600.0)

    fmask = _footprint_mask(lons, lats, facilities, cfg.coloc_radius_km)
    valid = np.isfinite(e)
    n_valid = int(valid.sum())
    fp = valid & fmask
    bg = valid & ~fmask

    speed = float(np.hypot(u, v))
    calm = speed < cfg.wind_min_ms
    if n_valid == 0 or not fp.any() or not bg.any():
        return CoherenceEvidence(
            n_pixels=int(field.no2.size),
            n_valid_cells=n_valid,
            n_footprint_cells=int(fp.sum()),
            wind_u=float(u),
            wind_v=float(v),
            wind_speed=speed,
            calm=calm,
            e_footprint_median=float("nan"),
            e_background_median=float("nan"),
            e_noise=float("nan"),
            snr=float("nan"),
            coloc_ratio=float("nan"),
            note="insufficient valid/footprint/background cells for a coherence assessment",
        )

    e_fp = float(np.median(e[fp]))
    e_bg = float(np.median(e[bg]))
    # Robust field-noise scale (MAD → sigma) from the background source term.
    mad = float(np.median(np.abs(e[bg] - e_bg)))
    noise = 1.4826 * mad if mad > 0 else float(np.std(e[bg]))
    snr = (e_fp - e_bg) / noise if noise > 0 else float("nan")

    # Hotspot co-location enrichment: of the strongest positive-E cells, what fraction sit over a
    # footprint, vs the fraction expected from the footprint's share of the valid grid area.
    e_valid = e[valid]
    pos = e_valid[e_valid > 0]
    coloc_ratio = float("nan")
    if pos.size >= 5:
        thresh = float(np.quantile(pos, 1.0 - cfg.hotspot_top_frac))
        hotspot = valid & (e >= thresh) & (e > 0)
        n_hot = int(hotspot.sum())
        if n_hot > 0:
            observed = float((hotspot & fmask).sum()) / n_hot
            expected = float(fp.sum()) / float(n_valid)
            if expected > 0:
                coloc_ratio = observed / expected

    return CoherenceEvidence(
        n_pixels=int(field.no2.size),
        n_valid_cells=n_valid,
        n_footprint_cells=int(fp.sum()),
        wind_u=float(u),
        wind_v=float(v),
        wind_speed=speed,
        calm=calm,
        e_footprint_median=e_fp,
        e_background_median=e_bg,
        e_noise=noise,
        snr=float(snr),
        coloc_ratio=coloc_ratio,
    )


# ==================================================================================================
# Verdict + compute profile + report
# ==================================================================================================


@dataclass(frozen=True)
class ComputeProfile:
    """Measured cost of the bounded run, extrapolated to a full multi-year run (Q2, NFR-008)."""

    n_granules: int
    downloaded_bytes: int
    reused_granules: int
    per_granule_mb: float
    wall_clock_s: float
    peak_mem_mb: float
    granules_per_day: float
    kept_field_pixels: int
    extrapolated_full_download_gb: float
    era_days: int


@dataclass(frozen=True)
class ProbeVerdict:
    """The T1 go/no-go verdict, its evidence, and the measured compute profile."""

    verdict: str  # "GO" | "GO-degraded" | "NO-GO" | "NO-GO-DATA"
    escalate: bool  # ERR-002: NO-GO / NO-GO-DATA stop the build and escalate
    access_ok: bool  # Q1: was the free CDSE raw-L2 route reached end-to-end?
    access_note: str
    evidence: CoherenceEvidence | None
    compute: ComputeProfile | None
    report_path: Path | None = None


def decide_verdict(
    access_ok: bool,
    evidence: CoherenceEvidence | None,
    cfg: FluxDivergenceConfig,
) -> tuple[str, bool]:
    """Pure verdict logic → ``(verdict, escalate)``. Benchmark-blind, fixed thresholds.

    NO-GO-DATA if the free route failed (data gate dominates). Otherwise GO if a co-located source
    rises clearly above noise, GO-degraded if it is partial/noisy but present, else NO-GO honest null.
    """
    if not access_ok:
        return "NO-GO-DATA", True
    if evidence is None or not np.isfinite(evidence.snr) or not np.isfinite(evidence.coloc_ratio):
        return "NO-GO", True
    if evidence.snr >= cfg.snr_go and evidence.coloc_ratio >= cfg.coloc_go:
        return "GO", False
    if evidence.snr >= cfg.snr_degraded and evidence.coloc_ratio >= cfg.coloc_degraded:
        return "GO-degraded", False
    return "NO-GO", True


def _aoi_from_cfg(cfg: FluxDivergenceConfig) -> Region:
    return tangshan_aoi(cfg.aoi_buffer_deg)


def _load_facilities(cfg: FluxDivergenceConfig) -> np.ndarray:
    """Committed steel-facility (lon, lat) rows, active statuses only (reuse attribution loader)."""
    from noxus.attribution.source import load_facilities

    df = load_facilities(cfg.facilities_csv, active_status=set(cfg.active_statuses))
    return df[["longitude", "latitude"]].to_numpy(dtype=float)


def _sample_wind(cfg: FluxDivergenceConfig, overpass: datetime, aoi: Region) -> tuple[float, float]:
    """AOI-mean ERA5 ``u10``/``v10`` at the ERA5 time nearest the overpass (effective transport wind).

    Returns ``(0.0, 0.0)`` if no ERA5 snapshot is available — the coherence assessment then flags the
    calm/degraded case rather than fabricating a wind.
    """
    import pandas as pd
    import xarray as xr

    snaps = sorted(Path(cfg.era5_snapshot_dir).glob("era5_*.nc"))
    if not snaps:
        return 0.0, 0.0
    ds = xr.open_dataset(snaps[-1]).load()
    try:
        if "time" not in ds.coords or {"u10", "v10"} - set(ds.data_vars):
            return 0.0, 0.0
        times = pd.DatetimeIndex(ds["time"].values)
        idx = int(np.argmin(np.abs(times - pd.Timestamp(overpass))))
        snap = ds.isel(time=idx)
        u = float(np.nanmean(snap["u10"].values))
        v = float(np.nanmean(snap["v10"].values))
    finally:
        ds.close()
    return u, v


def probe(cfg: FluxDivergenceConfig | None = None, fetcher: L2Fetcher | None = None) -> ProbeVerdict:
    """Run the T1 feasibility spike against REAL bounded L2 data and write the verdict report.

    Resolves Q1 live (free CDSE raw-L2 access), runs the minimal oversample→flux→divergence(+V/tau)
    chain on the bounded subset, measures spatial coherence over the cluster (benchmark-blind) and the
    compute/storage profile, and emits GO / GO-degraded / NO-GO / NO-GO-DATA. Never reads CREA.
    """
    cfg = cfg or FluxDivergenceConfig()
    aoi = _aoi_from_cfg(cfg)
    fetcher = fetcher or CDSEFetcher(cfg)

    tracemalloc.start()
    t0 = time.perf_counter()

    access_ok = True
    access_note = ""
    evidence: CoherenceEvidence | None = None
    compute: ComputeProfile | None = None

    try:
        result = fetch_l2(aoi, cfg, fetcher)
        access_note = (
            f"free CDSE OData/zipper route reached: {len(result.granules)} granule(s) selected, "
            f"{result.reused} reused, {result.downloaded_bytes / 1e6:.1f} MB downloaded this run."
        )
        field = read_l2_field(result.paths[0], aoi, cfg.qa_threshold)
        if field.no2.size < cfg.min_pixels:
            access_ok = True
            evidence = CoherenceEvidence(
                n_pixels=int(field.no2.size),
                n_valid_cells=0,
                n_footprint_cells=0,
                wind_u=0.0,
                wind_v=0.0,
                wind_speed=0.0,
                calm=True,
                e_footprint_median=float("nan"),
                e_background_median=float("nan"),
                e_noise=float("nan"),
                snr=float("nan"),
                coloc_ratio=float("nan"),
                note=f"only {field.no2.size} QA-valid AOI pixels (< min_pixels {cfg.min_pixels})",
            )
        else:
            u, v = _sample_wind(cfg, field.overpass_utc, aoi)
            facilities = _load_facilities(cfg)
            evidence = assess_coherence(field, u, v, facilities, cfg)
        compute = _profile(cfg, result, field, t0)
    except (L2AccessError, PaidSourceError) as exc:
        access_ok = False
        access_note = f"FREE raw-L2 route unavailable → NO-GO-DATA: {exc}"

    peak = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    if compute is not None:
        compute = ComputeProfile(**{**compute.__dict__, "peak_mem_mb": peak})

    verdict, escalate = decide_verdict(access_ok, evidence, cfg)
    result_verdict = ProbeVerdict(
        verdict=verdict,
        escalate=escalate,
        access_ok=access_ok,
        access_note=access_note,
        evidence=evidence,
        compute=compute,
    )

    report_path = cfg.probe_report_path()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(result_verdict, cfg))
    return ProbeVerdict(
        verdict=verdict,
        escalate=escalate,
        access_ok=access_ok,
        access_note=access_note,
        evidence=evidence,
        compute=compute,
        report_path=report_path,
    )


def _profile(cfg, result, field: L2Field, t0: float) -> ComputeProfile:
    n = len(result.granules)
    sizes = [g.size_bytes for g in result.granules if g.size_bytes]
    per_granule_mb = (sum(sizes) / len(sizes) / 1e6) if sizes else 0.0
    # Extrapolate on the TRUE AOI-intersecting rate (all granules found in the window), not just the
    # bounded subset actually downloaded — the honest (higher) figure for the Q2 resource decision.
    granules_per_day = result.n_found / max(1, result.window_days)
    era_days = max(1, (date.today() - _ERA_START).days)
    full_download_gb = per_granule_mb / 1000.0 * granules_per_day * era_days
    return ComputeProfile(
        n_granules=n,
        downloaded_bytes=result.downloaded_bytes,
        reused_granules=result.reused,
        per_granule_mb=per_granule_mb,
        wall_clock_s=time.perf_counter() - t0,
        peak_mem_mb=0.0,  # filled by the caller after tracemalloc stops
        granules_per_day=granules_per_day,
        kept_field_pixels=int(field.no2.size),
        extrapolated_full_download_gb=full_download_gb,
        era_days=era_days,
    )


def _render_report(v: ProbeVerdict, cfg: FluxDivergenceConfig) -> str:
    lines: list[str] = []
    lines.append("NOX-005 T1 feasibility spike — flux-divergence go/no-go gate (GATES THE BUILD)")
    lines.append("=" * 82)
    lines.append(f"Generated: {datetime.now(UTC).isoformat(timespec='seconds')}")
    lines.append("Benchmark-blind: the CREA benchmark is NEVER read/tuned to in this gate.")
    lines.append("")
    lines.append(f"VERDICT: {v.verdict}")
    lines.append(f"  escalate (ERR-002 stop-and-escalate): {v.escalate}")
    lines.append("")
    lines.append("Q1 — FREE CDSE raw-L2 access (resolved live)")
    lines.append("-" * 82)
    lines.append(f"  access_ok: {v.access_ok}")
    lines.append(f"  {v.access_note}")
    lines.append(
        f"  route: OData catalogue {cfg.odata_url}  +  zipper download {cfg.zipper_url}  "
        "(password-grant token; credentials by env-var name only, never logged)"
    )
    lines.append("")
    lines.append("Spatial coherence over the steel cluster (benchmark-blind)")
    lines.append("-" * 82)
    e = v.evidence
    if e is None:
        lines.append("  (not assessed — the free data route was unavailable)")
    else:
        lines.append(
            f"  QA-valid AOI pixels: {e.n_pixels}  valid grid cells: {e.n_valid_cells}  "
            f"footprint cells: {e.n_footprint_cells}"
        )
        lines.append(
            f"  effective wind (ERA5 {cfg.wind_level}): u={e.wind_u:+.2f} v={e.wind_v:+.2f} "
            f"|U|={e.wind_speed:.2f} m/s calm={e.calm} (calm floor {cfg.wind_min_ms} m/s)"
        )
        lines.append(
            f"  fixed operators: resolution={cfg.resolution_km} km  tau={cfg.tau_hours} h  "
            f"qa>={cfg.qa_threshold}  coloc_radius={cfg.coloc_radius_km} km"
        )
        if e.note:
            lines.append(f"  NOTE: {e.note}")
        lines.append(
            f"  source term E: footprint_median={e.e_footprint_median:+.3e} "
            f"background_median={e.e_background_median:+.3e} noise(MAD)={e.e_noise:.3e} mol m-2 s-1"
        )
        lines.append(
            f"  (a) hotspot co-location enrichment vs area: coloc_ratio={e.coloc_ratio:.2f} "
            f"(GO>={cfg.coloc_go}, degraded>={cfg.coloc_degraded})"
        )
        lines.append(
            f"  (b) footprint source vs field noise: SNR={e.snr:+.2f} "
            f"(GO>={cfg.snr_go}, degraded>={cfg.snr_degraded})"
        )
    lines.append("")
    lines.append("Q2 — MEASURED compute/storage profile (bounded run → full-run extrapolation)")
    lines.append("-" * 82)
    c = v.compute
    if c is None:
        lines.append("  (not measured — the free data route was unavailable)")
    else:
        lines.append(
            f"  bounded run: {c.n_granules} granule(s), {c.reused_granules} reused, "
            f"{c.downloaded_bytes / 1e6:.1f} MB downloaded, wall_clock={c.wall_clock_s:.1f}s, "
            f"peak_mem={c.peak_mem_mb:.0f} MB"
        )
        lines.append(
            f"  per full-orbit granule ~{c.per_granule_mb:.0f} MB (OData delivers the WHOLE orbit; "
            f"the AOI window is clipped AFTER download — kept field here = {c.kept_field_pixels} pixels)"
        )
        lines.append(
            f"  ~{c.granules_per_day:.1f} AOI-intersecting granule(s)/day over the bounded window"
        )
        lines.append(
            f"  EXTRAPOLATED full record ({c.era_days} days since {_ERA_START}): "
            f"~{c.extrapolated_full_download_gb:.0f} GB of DOWNLOAD transfer (kept AOI storage is far "
            "smaller after clipping). Fine oversampling+divergence over the full record is "
            "server-class; the bounded subset here is laptop-feasible."
        )
    lines.append("")
    lines.append("Interpretation")
    lines.append("-" * 82)
    if v.verdict == "NO-GO-DATA":
        lines.append(
            "  NO-GO-DATA. The free CDSE raw-L2 route could not be exercised end-to-end (missing "
            "credentials/network or a paid-only path). Per NFR-001/ERR-005 nothing heavy was "
            "downloaded and no paid source was introduced. The build STOPS; resolve access before "
            "the compute/method question is revisited (ERR-002 escalate)."
        )
    elif v.verdict == "NO-GO":
        lines.append(
            "  HONEST NULL. On the bounded real subset no coherent, spatially-localised source term "
            "co-locates with the plant footprints above the field noise — the expected-hard outcome "
            "for flux-divergence in high-background/complex-wind NE China. Per ERR-002 the build "
            "STOPS and escalates. A null here is a first-class, designed-for result (it closes the "
            "DL-A revisit condition), not a pipeline failure. NOTE: a single-overpass bounded probe "
            "has low power; a NO-GO warrants confirming on a few more overpasses before closing DL-A."
        )
    elif v.verdict == "GO-degraded":
        lines.append(
            "  GO-degraded. A partial/noisy but co-located source term is present on the bounded "
            "subset. Proceed to build the full attribution WITH caveats (report sensitivity across "
            "tau and wind level; every output carries the reduced-confidence + scale caveats)."
        )
    else:
        lines.append(
            "  GO. A coherent, spatially-localised source term co-locates with the plant footprints "
            "and rises above the field noise on the bounded subset. Proceed to build the full "
            "flux-divergence attribution (T2+)."
        )
    lines.append("")
    lines.append(_SCALE_CAVEAT)
    lines.append("")
    return "\n".join(lines)
