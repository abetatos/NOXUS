"""Tests for VIIRS thermal acquisition (REQ-001..006, ERR-001..004). Fetchers are faked; no network.

Covers AT0 (probe), AT1 (acquire + clip + provenance), AT2 (resumable/idempotent), AT4 (FIRMS
static-land-source filter), AT7 (verification render), AT-ERR-1 (key/endpoint/budget failures).
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from noxus.config.region import tangshan_aoi
from noxus.config.run import ThermalTriangulationConfig, ViirsAcquisitionConfig
from noxus.data import viirs as V

AOI = tangshan_aoi()
# A point inside the AOI (near a Tangshan facility) and one far outside it.
IN_LAT, IN_LON = 39.63, 118.18
OUT_LAT, OUT_LON = 10.0, 10.0


def _detection(day, lat, lon, rh=5.0, temp=1500.0, did="d0"):
    return {
        V.TEMPERATURE_K: temp,
        V.SOURCE_AREA: 0.01,
        V.RADIANT_HEAT_MW: rh,
        V.LATITUDE: lat,
        V.LONGITUDE: lon,
        V.TIMESTAMP: pd.Timestamp(f"{day.isoformat()}T18:00"),
        V.DETECTION_ID: did,
        V.QUALITY_FLAG: 1,
    }


class FakeFetcher:
    """Returns synthetic per-night detections keyed by date; empty frame for unlisted nights."""

    def __init__(self, by_day):
        self.by_day = by_day
        self.calls = 0

    def fetch_night(self, aoi, day):
        self.calls += 1
        rows = self.by_day.get(day, [])
        return pd.DataFrame(rows) if rows else V.standardise(V._empty_frame())


def _cfg(tmp_path, **kw):
    return ViirsAcquisitionConfig(raw_dir=tmp_path / "viirs", **kw)


# ------------------------------------------------------------------------------------ AT1


def test_acquire_clips_aoi_retains_fields_and_provenance(tmp_path):
    d = date(2023, 6, 5)
    by_day = {
        d: [
            _detection(d, IN_LAT, IN_LON, did="in"),
            _detection(d, OUT_LAT, OUT_LON, did="out"),  # outside AOI -> must be dropped
        ]
    }
    cfg = _cfg(tmp_path, start=d)
    report = V.acquire_viirs(AOI, cfg, fetcher=FakeFetcher(by_day), today=d)

    assert report.fetched == 1
    assert report.n_detections == 1  # only the in-AOI detection persisted
    manifest = V.load_manifest(cfg.raw_dir, "vnf")
    assert manifest["nights"][d.isoformat()]["n_detections"] == 1

    stored = V.load_detections(cfg.raw_dir, "vnf")
    assert list(stored[V.LATITUDE]) == [IN_LAT]
    for col in V.STANDARD_COLUMNS:
        assert col in stored.columns
    prov = next((cfg.raw_dir / "vnf" / "2023").glob("*.provenance.json"))
    assert '"source": "vnf"' in prov.read_text()
    assert '"aoi_bbox"' in prov.read_text()


def test_empty_night_recorded_as_zero_not_error(tmp_path):
    d = date(2023, 6, 5)
    cfg = _cfg(tmp_path, start=d)
    report = V.acquire_viirs(AOI, cfg, fetcher=FakeFetcher({}), today=d)
    assert report.fetched == 1
    assert report.empty_nights == 1
    assert report.n_detections == 0


# ------------------------------------------------------------------------------------ AT2


def test_resumable_second_run_fetches_nothing_new(tmp_path):
    d = date(2023, 6, 5)
    by_day = {d: [_detection(d, IN_LAT, IN_LON)]}
    cfg = _cfg(tmp_path, start=d)
    f = FakeFetcher(by_day)
    V.acquire_viirs(AOI, cfg, fetcher=f, today=d)
    assert f.calls == 1
    report2 = V.acquire_viirs(AOI, cfg, fetcher=f, today=d)
    assert f.calls == 1  # night already in the manifest -> skipped, no fetch
    assert report2.fetched == 0
    assert report2.skipped == 1


class FlakyFetcher:
    """Raises on the second night once, then succeeds on resume (models a mid-range failure)."""

    def __init__(self, fail_on):
        self.fail_on = fail_on
        self.raised = False

    def fetch_night(self, aoi, day):
        if day == self.fail_on and not self.raised:
            self.raised = True
            raise V.ViirsAcquisitionError("boom")
        return pd.DataFrame([_detection(day, IN_LAT, IN_LON)])


def test_failed_night_recorded_then_resumes(tmp_path):
    start, end = date(2023, 6, 5), date(2023, 6, 6)
    cfg = _cfg(tmp_path, start=start, max_retries=1)
    f = FlakyFetcher(fail_on=date(2023, 6, 6))
    r1 = V.acquire_viirs(AOI, cfg, fetcher=f, today=end)
    assert r1.fetched == 1 and r1.failed == 1
    manifest = V.load_manifest(cfg.raw_dir, "vnf")
    assert "2023-06-06" in manifest["batch_errors"]
    # Resume: the good night is skipped, the failed night now succeeds.
    r2 = V.acquire_viirs(AOI, cfg, fetcher=f, today=end)
    assert r2.fetched == 1 and r2.skipped == 1
    assert "2023-06-06" not in V.load_manifest(cfg.raw_dir, "vnf")["batch_errors"]


# ------------------------------------------------------------------------------------ AT0 (probe)


def test_probe_reports_go_when_detections_colocate(tmp_path):
    from noxus.signal.thermal import load_facilities

    d = date(2023, 6, 7)
    by_day = {d: [_detection(d, IN_LAT, IN_LON)]}
    facilities = pd.DataFrame({"name": ["p"], "latitude": [IN_LAT], "longitude": [IN_LON]})
    # Give the facility a real name so co-location join succeeds.
    cfg = _cfg(tmp_path, probe_days=3, start=date(2023, 6, 1))
    probe = V.probe_viirs(AOI, cfg, fetcher=FakeFetcher(by_day), today=d, facilities=facilities)
    assert probe.n_detections == 1
    assert probe.facilities_colocated == 1
    assert probe.go is True
    assert V.RADIANT_HEAT_MW in probe.fields_available
    load_facilities  # imported to confirm the module wires together


def test_probe_no_go_over_budget(tmp_path):
    d = date(2023, 6, 7)
    by_day = {d: [_detection(d, IN_LAT, IN_LON)]}
    cfg = _cfg(tmp_path, probe_days=3, start=date(2019, 1, 1), download_budget_mb=1e-12)
    probe = V.probe_viirs(AOI, cfg, fetcher=FakeFetcher(by_day), today=d)
    assert probe.within_budget is False
    assert probe.go is False


def test_raise_if_over_budget():
    cfg = ViirsAcquisitionConfig(download_budget_mb=100.0)
    with pytest.raises(V.BudgetExceededError):
        V.raise_if_over_budget(500.0, cfg)
    V.raise_if_over_budget(50.0, cfg)  # under budget -> no raise


# ------------------------------------------------------------------------------------ AT4 (FIRMS)


def test_firms_keeps_only_static_land_source():
    cfg = ViirsAcquisitionConfig(source="firms")
    fetcher = V.FirmsFetcher(cfg)
    raw = pd.DataFrame(
        {
            "latitude": [IN_LAT, IN_LAT, IN_LAT],
            "longitude": [IN_LON, IN_LON, IN_LON],
            "frp": [8.0, 3.0, 5.0],
            "bright_ti4": [1400.0, 900.0, 1200.0],
            "acq_date": ["2023-06-05", "2023-06-05", "2023-06-05"],
            "acq_time": [1830, 1830, 1830],
            "type": [2, 0, 3],  # static-land-source, vegetation fire, offshore
            "confidence": ["n", "l", "h"],
        }
    )
    out = fetcher.filter_and_standardise(raw, AOI, date(2023, 6, 5))
    assert len(out) == 1  # only type==2 kept (EDGE-004)
    assert float(out[V.RADIANT_HEAT_MW].iloc[0]) == 8.0


def test_firms_missing_key_references_env_var_name(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    monkeypatch.delenv("FIRMS_MAP_KEY", raising=False)
    cfg = ViirsAcquisitionConfig(source="firms")
    fetcher = V.FirmsFetcher(cfg)
    with pytest.raises(V.ViirsAuthError) as ei:
        fetcher.fetch_night(AOI, date(2023, 6, 5))
    assert "FIRMS_MAP_KEY" in str(ei.value)


def test_scrub_url_hides_map_key():
    url = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/SECRETKEY/VIIRS_SNPP_SP/1,2,3,4/1/2023-06-05"
    scrubbed = V._scrub_url(url)
    assert "SECRETKEY" not in scrubbed
    assert "<MAP_KEY>" in scrubbed


# ------------------------------------------------------------------------------------ AT-ERR-1


def test_standardise_missing_required_field_raises():
    bad = pd.DataFrame({V.LATITUDE: [1.0], V.LONGITUDE: [2.0]})  # no radiant heat / timestamp
    with pytest.raises(V.ViirsAcquisitionError) as ei:
        V.standardise(bad)
    assert V.RADIANT_HEAT_MW in str(ei.value)


# ------------------------------------------------------------------------------------ AT7 (render)


def test_verify_render_writes_png_with_facilities(tmp_path):
    from noxus.data.verify_viirs import render_detections

    d = date(2023, 6, 5)
    detections = pd.DataFrame([_detection(d, IN_LAT, IN_LON)])
    fac_csv = tmp_path / "fac.csv"
    pd.DataFrame({"name": ["p"], "latitude": [IN_LAT], "longitude": [IN_LON]}).to_csv(
        fac_csv, index=False
    )
    out = render_detections(detections, fac_csv, tmp_path / "viirs.png", aoi=AOI)
    assert out.exists() and out.stat().st_size > 0


def test_config_defaults_are_sane():
    cfg = ThermalTriangulationConfig()
    assert cfg.radius_km == 3.0  # pre-registered (A2)
    assert cfg.primary_aggregate == "radiant_heat"  # pre-registered (A3)
    assert cfg.freq == "W"
