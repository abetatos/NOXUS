"""Tests for VIIRS thermal acquisition (REQ-001..006, ERR-001..004). Fetchers are faked; no network.

Covers AT0 (probe), AT1 (acquire + clip + provenance), AT2 (resumable/idempotent), AT4 (FIRMS
static-land-source filter), AT7 (verification render), AT-ERR-1 (key/endpoint/budget failures).
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta

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
    cfg = _cfg(tmp_path, start=d, end=d)  # explicit end => respected as-is (no SP-lag clamp)
    report = V.acquire_viirs(AOI, cfg, fetcher=FakeFetcher(by_day), today=d)

    assert report.fetched == 1
    assert report.n_detections == 1  # only the in-AOI detection persisted
    assert report.source == "firms"  # FIRMS is the default/primary source (2026-08-22 pivot)
    manifest = V.load_manifest(cfg.raw_dir, "firms")
    assert manifest["nights"][d.isoformat()]["n_detections"] == 1

    stored = V.load_detections(cfg.raw_dir, "firms")
    assert list(stored[V.LATITUDE]) == [IN_LAT]
    for col in V.STANDARD_COLUMNS:
        assert col in stored.columns
    prov = next((cfg.raw_dir / "firms" / "2023").glob("*.provenance.json"))
    assert '"source": "firms"' in prov.read_text()
    assert '"aoi_bbox"' in prov.read_text()


def test_empty_night_recorded_as_zero_not_error(tmp_path):
    d = date(2023, 6, 5)
    cfg = _cfg(tmp_path, start=d, end=d)
    report = V.acquire_viirs(AOI, cfg, fetcher=FakeFetcher({}), today=d)
    assert report.fetched == 1
    assert report.empty_nights == 1
    assert report.n_detections == 0


# ------------------------------------------------------------------------------------ AT2


def test_resumable_second_run_fetches_nothing_new(tmp_path):
    d = date(2023, 6, 5)
    by_day = {d: [_detection(d, IN_LAT, IN_LON)]}
    cfg = _cfg(tmp_path, start=d, end=d)
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
    cfg = _cfg(tmp_path, start=start, end=end, max_retries=1)
    f = FlakyFetcher(fail_on=date(2023, 6, 6))
    r1 = V.acquire_viirs(AOI, cfg, fetcher=f, today=end)
    assert r1.fetched == 1 and r1.failed == 1
    manifest = V.load_manifest(cfg.raw_dir, "firms")
    assert "2023-06-06" in manifest["batch_errors"]
    # Resume: the good night is skipped, the failed night now succeeds.
    r2 = V.acquire_viirs(AOI, cfg, fetcher=f, today=end)
    assert r2.fetched == 1 and r2.skipped == 1
    assert "2023-06-06" not in V.load_manifest(cfg.raw_dir, "firms")["batch_errors"]


# ------------------------------------------------------------------------------------ AT0 (probe)


def test_probe_reports_go_when_detections_colocate(tmp_path):
    from noxus.signal.thermal import load_facilities

    d = date(
        2023, 6, 7
    )  # the (archive-lagged) date whose SP detections we want the probe to land on
    by_day = {d: [_detection(d, IN_LAT, IN_LON)]}
    facilities = pd.DataFrame({"name": ["p"], "latitude": [IN_LAT], "longitude": [IN_LON]})
    # Give the facility a real name so co-location join succeeds.
    cfg = _cfg(tmp_path, probe_days=3, start=date(2023, 6, 1))
    # FIRMS SP lags ~firms_archive_lag_days, so the probe anchors back by that lag: set "today" so the
    # anchored window (today - lag) lands on the detection date d.
    today = d + timedelta(days=cfg.firms_archive_lag_days)
    probe = V.probe_viirs(AOI, cfg, fetcher=FakeFetcher(by_day), today=today, facilities=facilities)
    assert probe.n_detections == 1
    assert probe.facilities_colocated == 1
    assert probe.go is True
    assert probe.archive_lag_days == cfg.firms_archive_lag_days
    assert probe.auth_required is True  # FIRMS needs FIRMS_MAP_KEY (N1)
    assert V.RADIANT_HEAT_MW in probe.fields_available
    load_facilities  # imported to confirm the module wires together


def test_probe_go_on_archive_window_recent_window_would_miss(tmp_path):
    """Regression (item 1): the SP archive lags ~2 months, so detections live in an offset window.

    WITH the fix the probe anchors back by the archive lag and returns GO; the OLD recent-window
    behaviour (lag 0) sees only empty recent nights and would return NO-GO.
    """
    facilities = pd.DataFrame({"name": ["p"], "latitude": [IN_LAT], "longitude": [IN_LON]})
    today = date(2024, 3, 1)
    lag = 60
    archive_day = today - timedelta(days=lag)  # a date inside the SP archive window
    by_day = {archive_day: [_detection(archive_day, IN_LAT, IN_LON)]}
    cfg = _cfg(
        tmp_path, source="firms", probe_days=1, firms_archive_lag_days=lag, start=date(2024, 1, 1)
    )

    # PASS-WITH: probe anchors back by the archive lag -> lands on the detection day -> GO.
    probe = V.probe_viirs(AOI, cfg, fetcher=FakeFetcher(by_day), today=today, facilities=facilities)
    assert probe.n_detections == 1
    assert probe.go is True
    assert probe.archive_lag_days == lag
    assert probe.window == (archive_day.isoformat(), archive_day.isoformat())

    # FAIL-WITHOUT: the old recent-window behaviour (lag 0) probes empty recent nights -> NO-GO.
    cfg_recent = replace(cfg, firms_archive_lag_days=0)
    probe_recent = V.probe_viirs(
        AOI, cfg_recent, fetcher=FakeFetcher(by_day), today=today, facilities=facilities
    )
    assert probe_recent.n_detections == 0
    assert probe_recent.go is False


def test_acquire_clamps_chase_today_end_by_archive_lag(tmp_path):
    """Item 4: with cfg.end=None the effective end is clamped back by the SP archive lag and recorded."""
    start = date(2024, 1, 1)
    today = date(2024, 3, 1)
    lag = 40
    cfg = _cfg(tmp_path, source="firms", start=start, end=None, firms_archive_lag_days=lag)
    V.acquire_viirs(AOI, cfg, fetcher=FakeFetcher({}), today=today)
    manifest = V.load_manifest(cfg.raw_dir, "firms")
    clamp = manifest["end_clamp"]
    assert clamp["clamped_end"] == (today - timedelta(days=lag)).isoformat()
    assert clamp["skipped_tail_nights"] == lag
    # No night on/after the clamped end was fetched.
    assert (today - timedelta(days=lag - 1)).isoformat() not in manifest["nights"]


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
    scrubbed = V._scrub_url(url)  # path-segment fallback (no key value to hand)
    assert "SECRETKEY" not in scrubbed
    assert "<MAP_KEY>" in scrubbed


def test_scrub_redacts_key_by_value_anywhere():
    """N3: the actual key value must be redacted wherever it appears, not just one /csv/ segment."""
    key = "abc123DEFsecretkeyvalue0000000000"
    text = (
        f"FIRMS fetch failed at https://x/api/area/csv/{key}/PROD/1,2,3,4/1/2023-06-05: "
        f"ConnectError for url https://x/csv/{key}/PROD..."
    )
    scrubbed = V._scrub_url(text, key)
    assert key not in scrubbed  # value redacted in BOTH the URL and the trailing error text
    assert "<MAP_KEY>" in scrubbed


def test_firms_fetch_error_never_leaks_key(monkeypatch):
    """Fail-without/pass-with (N3): an httpx error embedding the key URL must not surface the value."""
    import httpx

    key = "SUPERSECRETKEY_32_characters_xxxx"
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: True)
    monkeypatch.setenv("FIRMS_MAP_KEY", key)

    def boom(url, **kwargs):
        raise RuntimeError(f"connection failed for {url}")  # url embeds the key path segment

    monkeypatch.setattr(httpx, "get", boom)
    fetcher = V.FirmsFetcher(ViirsAcquisitionConfig(source="firms"))
    with pytest.raises(V.ViirsAcquisitionError) as ei:
        fetcher.fetch_night(AOI, date(2023, 6, 5))
    assert key not in str(ei.value)
    assert "<MAP_KEY>" in str(ei.value)


def test_vnf_source_raises_clear_licence_error():
    """Item 3: source='vnf' surfaces a clear licence/pay-gated error pointing at FIRMS (no network)."""
    fetcher = V.VnfFetcher(ViirsAcquisitionConfig(source="vnf"))
    with pytest.raises(V.VnfUnavailableError) as ei:
        fetcher.fetch_night(AOI, date(2023, 6, 5))
    msg = str(ei.value)
    assert "licen" in msg.lower()  # licence/license
    assert "FIRMS" in msg
    assert "2025-01-10" in msg and "2026-06-01" in msg
    # A dropped VNF must not be retried by the acquisition loop -> it is a ViirsAuthError subclass.
    assert isinstance(ei.value, V.ViirsAuthError)


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
