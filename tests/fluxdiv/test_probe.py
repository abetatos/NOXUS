"""Unit + guard tests for the NOX-005 T1 flux-divergence feasibility gate.

Covers the pure logic the gate turns on: the three-way + data-gate verdict thresholds (AT7), the
hotspot-co-location / signal-vs-noise measurement on a planted source vs a background-only field
(AT2/AT7), the NO-GO-DATA path when the free CDSE route is unavailable (AT7/ERR-005), and the
public-only refusal of a paid source (AT-ERR-3). No network or credentials are used — the free-route
fetch is exercised through the :class:`L2Fetcher` seam with a synthetic granule.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import xarray as xr

from noxus.config.region import tangshan_aoi
from noxus.config.run import FluxDivergenceConfig
from noxus.data.tropomi_l2 import (
    GranuleMeta,
    L2AccessError,
    L2Field,
    PaidSourceError,
    _assert_free_source,
)
from noxus.fluxdiv.probe import (
    CoherenceEvidence,
    ScalingPoint,
    assess_coherence,
    assess_coherence_multi,
    decide_powered_verdict,
    decide_verdict,
    probe,
    scaling_curve,
)


# --------------------------------------------------------------------------- verdict thresholds


def _evidence(snr: float, coloc: float) -> CoherenceEvidence:
    return CoherenceEvidence(
        n_pixels=500,
        n_valid_cells=400,
        n_footprint_cells=12,
        wind_u=2.0,
        wind_v=0.0,
        wind_speed=2.0,
        calm=False,
        e_footprint_median=1e-8,
        e_background_median=0.0,
        e_noise=1e-9,
        snr=snr,
        coloc_ratio=coloc,
    )


def test_verdict_go_requires_coherent_colocated_source_above_noise():
    cfg = FluxDivergenceConfig()
    assert decide_verdict(True, _evidence(3.5, 2.0), cfg) == ("GO", False)


def test_verdict_go_degraded_on_partial_noisy_source():
    cfg = FluxDivergenceConfig()
    verdict, escalate = decide_verdict(True, _evidence(2.0, 1.2), cfg)
    assert verdict == "GO-degraded"
    assert escalate is False


def test_verdict_no_go_honest_null_when_no_coherent_source():
    cfg = FluxDivergenceConfig()
    verdict, escalate = decide_verdict(True, _evidence(0.5, 0.9), cfg)
    assert verdict == "NO-GO"
    assert escalate is True  # ERR-002 stop-and-escalate


def test_verdict_no_go_data_when_free_route_unavailable():
    cfg = FluxDivergenceConfig()
    # The data gate dominates: even a would-be strong signal cannot yield GO without verified access.
    assert decide_verdict(False, _evidence(9.0, 9.0), cfg) == ("NO-GO-DATA", True)


def test_verdict_no_go_when_evidence_is_undefined():
    cfg = FluxDivergenceConfig()
    assert decide_verdict(True, None, cfg) == ("NO-GO", True)
    assert decide_verdict(True, _evidence(float("nan"), 2.0), cfg)[0] == "NO-GO"


# ----------------------------------------------------------- hotspot co-location / signal-vs-noise


def _synthetic_field(cfg: FluxDivergenceConfig, *, plant_lon: float, plant_lat: float, bump: bool):
    """A synthetic L2 field over the AOI: smooth background + (optionally) a bump at the plant."""
    aoi = tangshan_aoi(cfg.aoi_buffer_deg)
    x0, y0, x1, y1 = aoi.as_bbox()
    rng = np.random.default_rng(20260827)
    n = 4000
    lon = rng.uniform(x0, x1, n)
    lat = rng.uniform(y0, y1, n)
    base = 8e-5
    no2 = base + rng.normal(0.0, 2e-6, n)
    if bump:
        # ~10 km Gaussian enhancement co-located with the plant (a localised NO2 source column).
        dkm = np.hypot((lon - plant_lon) * 85.0, (lat - plant_lat) * 111.0)
        no2 = no2 + 1.5e-4 * np.exp(-(dkm**2) / (2 * 10.0**2))
    return L2Field(
        lat=lat,
        lon=lon,
        no2=no2,
        overpass_utc=None,  # unused by assess_coherence
        granule_name="synthetic",
    )


def test_planted_source_colocates_with_footprint_above_noise():
    cfg = FluxDivergenceConfig()
    aoi = tangshan_aoi(cfg.aoi_buffer_deg)
    x0, y0, x1, y1 = aoi.as_bbox()
    plant_lon, plant_lat = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    facilities = np.array([[plant_lon, plant_lat]])
    field = _synthetic_field(cfg, plant_lon=plant_lon, plant_lat=plant_lat, bump=True)

    ev = assess_coherence(field, u=2.0, v=0.5, facilities=facilities, cfg=cfg)

    # Hotspots of the source term are enriched over the footprint, and the footprint source rises
    # above the background field noise — this is the GO/GO-degraded evidence (not NO-GO).
    assert ev.coloc_ratio > 1.0
    assert ev.snr > 0.0
    assert ev.e_footprint_median > ev.e_background_median
    assert decide_verdict(True, ev, cfg)[0] in {"GO", "GO-degraded"}


def test_background_only_field_yields_no_go():
    cfg = FluxDivergenceConfig()
    aoi = tangshan_aoi(cfg.aoi_buffer_deg)
    x0, y0, x1, y1 = aoi.as_bbox()
    plant_lon, plant_lat = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    facilities = np.array([[plant_lon, plant_lat]])
    field = _synthetic_field(cfg, plant_lon=plant_lon, plant_lat=plant_lat, bump=False)

    ev = assess_coherence(field, u=2.0, v=0.5, facilities=facilities, cfg=cfg)

    # No coherent local source: the footprint source does not rise meaningfully above the noise.
    assert ev.snr < cfg.snr_go
    assert decide_verdict(True, ev, cfg)[0] == "NO-GO"


# -------------------------------------------------- multi-overpass SNR-vs-N scaling (powered gate)


def _synthetic_overpasses(cfg, *, plant_lon, plant_lat, bump, n, noise=1.3e-5, speed=3.0, seed0=100):
    """N synthetic overpasses with independent pixel noise and winds spread around the circle.

    Winds spread over all directions have ~zero mean, so the advected-background dipole cancels under
    oversampling while a persistent source (the bump's V/tau term) survives and its noise falls
    ~1/sqrt(N) — exactly the separation the real method relies on.
    """
    aoi = tangshan_aoi(cfg.aoi_buffer_deg)
    x0, y0, x1, y1 = aoi.as_bbox()
    ops = []
    for i in range(n):
        rng = np.random.default_rng(seed0 + i)
        npix = 4000
        lon = rng.uniform(x0, x1, npix)
        lat = rng.uniform(y0, y1, npix)
        no2 = 8e-5 + rng.normal(0.0, noise, npix)
        if bump:
            dkm = np.hypot((lon - plant_lon) * 85.0, (lat - plant_lat) * 111.0)
            no2 = no2 + 1.5e-4 * np.exp(-(dkm**2) / (2 * 10.0**2))
        angle = 2 * np.pi * i / n
        field = L2Field(lat=lat, lon=lon, no2=no2, overpass_utc=None, granule_name=f"op{i}")
        ops.append((field, speed * np.cos(angle), speed * np.sin(angle)))
    return ops


def test_multi_overpass_oversampling_separates_source_as_noise_falls():
    cfg = FluxDivergenceConfig()
    aoi = tangshan_aoi(cfg.aoi_buffer_deg)
    x0, y0, x1, y1 = aoi.as_bbox()
    plant_lon, plant_lat = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    facilities = np.array([[plant_lon, plant_lat]])
    ops = _synthetic_overpasses(cfg, plant_lon=plant_lon, plant_lat=plant_lat, bump=True, n=8)

    curve = scaling_curve(ops, facilities, cfg)
    ns = [p.n for p in curve]
    assert ns[0] == 1 and ns[-1] == 8  # N=1 baseline through all-N

    # The decisive behaviour: field noise averages DOWN with N and the source SNR RISES with N.
    assert curve[-1].e_noise < curve[0].e_noise
    assert curve[-1].snr > curve[0].snr
    ev = assess_coherence_multi(ops, facilities, cfg)
    assert ev.coloc_ratio > 1.0


def test_multi_overpass_background_only_stays_flat_and_powers_no_go():
    cfg = FluxDivergenceConfig()
    aoi = tangshan_aoi(cfg.aoi_buffer_deg)
    x0, y0, x1, y1 = aoi.as_bbox()
    plant_lon, plant_lat = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    facilities = np.array([[plant_lon, plant_lat]])
    ops = _synthetic_overpasses(cfg, plant_lon=plant_lon, plant_lat=plant_lat, bump=False, n=8)

    curve = tuple(scaling_curve(ops, facilities, cfg))
    ev = assess_coherence_multi(ops, facilities, cfg)
    # No source: oversampling cannot manufacture one → a powered NO-GO (not an n=1 artefact).
    assert ev.snr < cfg.snr_degraded
    assert decide_powered_verdict(True, curve, ev, cfg)[0] == "NO-GO"


def test_decide_powered_verdict_thresholds():
    cfg = FluxDivergenceConfig()
    curve = (
        ScalingPoint(1, 0.3, 2.0, 3e-8, 400),
        ScalingPoint(8, 3.5, 2.0, 1e-8, 3200),  # noise fell, SNR rose
    )
    assert decide_powered_verdict(True, curve, _evidence(3.5, 2.0), cfg) == ("GO", False)
    assert decide_powered_verdict(True, curve, _evidence(2.0, 1.2), cfg)[0] == "GO-degraded"
    assert decide_powered_verdict(True, curve, _evidence(0.3, 1.0), cfg) == ("NO-GO", True)
    assert decide_powered_verdict(False, curve, _evidence(9.0, 9.0), cfg) == ("NO-GO-DATA", True)
    assert decide_powered_verdict(True, (), None, cfg) == ("NO-GO", True)


# ------------------------------------------------------------------------------- public-only guard


def test_paid_source_is_refused():
    cfg = FluxDivergenceConfig()
    with pytest.raises(PaidSourceError):
        _assert_free_source("https://paid.example.com/odata/Products", cfg)
    # The free CDSE hosts are accepted.
    _assert_free_source(cfg.odata_url, cfg)
    _assert_free_source(f"{cfg.zipper_url}(abc)/$value", cfg)


# ------------------------------------------------------------- end-to-end probe via the fetcher seam


class _RaisingFetcher:
    """A fetcher whose search fails as if the free route were unreachable (no creds / no network)."""

    def search(self, aoi, start, end):
        raise L2AccessError("simulated: CDSE credentials not set / network unavailable")

    def download(self, meta, dest):  # pragma: no cover - never reached
        raise AssertionError("download must not be attempted on an access failure")


def test_probe_reports_no_go_data_without_heavy_download(tmp_path):
    cfg = replace(FluxDivergenceConfig(), out_dir=tmp_path / "fluxdiv", raw_dir=tmp_path / "raw")
    v = probe(cfg, fetcher=_RaisingFetcher())
    assert v.verdict == "NO-GO-DATA"
    assert v.escalate is True
    assert v.access_ok is False
    assert v.compute is None  # nothing measured; no heavy download happened
    assert v.report_path.exists()
    assert "NO-GO-DATA" in v.report_path.read_text()


def _write_synthetic_granule(path, cfg, *, plant_lon, plant_lat):
    """Write a minimal S5P-L2-shaped netCDF (PRODUCT group) with a planted NO2 source over a plant."""
    aoi = tangshan_aoi(cfg.aoi_buffer_deg)
    x0, y0, x1, y1 = aoi.as_bbox()
    ny, nx = 120, 60
    lats = np.linspace(y0, y1, ny)
    lons = np.linspace(x0, x1, nx)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    dkm = np.hypot((lon_grid - plant_lon) * 85.0, (lat_grid - plant_lat) * 111.0)
    no2 = 8e-5 + 1.5e-4 * np.exp(-(dkm**2) / (2 * 10.0**2))
    shape = (1, ny, nx)
    ds = xr.Dataset(
        {
            "nitrogendioxide_tropospheric_column": (("time", "scanline", "ground_pixel"), no2[None]),
            "qa_value": (("time", "scanline", "ground_pixel"), np.ones(shape)),
            "latitude": (("time", "scanline", "ground_pixel"), lat_grid[None]),
            "longitude": (("time", "scanline", "ground_pixel"), lon_grid[None]),
        }
    )
    # No time_utc: the overpass time then falls back to the filename token (as in read_l2_field).
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path, group="PRODUCT", engine="h5netcdf")
    return path.stat().st_size


class _LocalGranuleFetcher:
    """Serves a pre-written local granule through the fetch seam (no network); download is a no-op."""

    def __init__(self, name, size):
        self._name = name
        self._size = size

    def search(self, aoi, start, end):
        return [GranuleMeta(id="synthetic", name=self._name, size_bytes=self._size,
                            content_date_start="2023-06-01T05:30:00Z")]

    def download(self, meta, dest):  # pragma: no cover - reuse path is taken instead
        raise AssertionError("granule already present; fetch_l2 should reuse, not download")


def test_probe_runs_end_to_end_on_a_real_granule_file(tmp_path):
    """The probe fetches (reuses) a granule, reads native footprints, runs the chain, emits a verdict.

    Exercises the SAME code path as the live run — only the network is stubbed — so the verdict,
    compute profile and report are produced from a real on-disk L2-shaped granule.
    """
    cfg = replace(
        FluxDivergenceConfig(),
        out_dir=tmp_path / "fluxdiv",
        raw_dir=tmp_path / "raw",
        era5_snapshot_dir=tmp_path / "no_era5",  # absent → wind falls back to (0,0), flagged calm
    )
    aoi = tangshan_aoi(cfg.aoi_buffer_deg)
    x0, y0, x1, y1 = aoi.as_bbox()
    plant_lon, plant_lat = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    name = "S5P_OFFL_L2__NO2____20230601T053000_synthetic.nc"
    size = _write_synthetic_granule(cfg.raw_dir / name, cfg, plant_lon=plant_lon, plant_lat=plant_lat)

    # Point the facilities loader at a one-row CSV co-located with the planted source.
    fac_csv = tmp_path / "facilities.csv"
    fac_csv.write_text(
        "name,latitude,longitude,technology,status\n"
        f"Synthetic Plant,{plant_lat},{plant_lon},BF; BOF,operating\n"
    )
    cfg = replace(cfg, facilities_csv=fac_csv)

    v = probe(cfg, fetcher=_LocalGranuleFetcher(name, size))

    assert v.access_ok is True
    assert v.verdict in {"GO", "GO-degraded", "NO-GO"}
    assert v.compute is not None
    assert v.compute.n_granules == 1
    assert v.compute.downloaded_bytes == 0  # reused, not re-downloaded (resume)
    assert v.evidence is not None and v.evidence.n_pixels > cfg.min_pixels
    assert v.report_path.exists()
    assert "compute/storage profile" in v.report_path.read_text()
