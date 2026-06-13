"""Tests for TROPOMI NO2 acquisition (REQ-001..007, ERR-001, EDGE-002). openEO is faked."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from noxus.config.region import tangshan_aoi
from noxus.config.run import AcquisitionConfig
from noxus.data import tropomi as T


def make_ds(times, ny=4, nx=5, qa=0.9, versions=None):
    t = pd.to_datetime(times)
    shape = (len(t), ny, nx)
    ds = xr.Dataset(
        {
            T.NO2: (("time", "y", "x"), np.full(shape, 2.0)),
            T.QA: (("time", "y", "x"), np.full(shape, qa)),
            T.CLOUD: (("time", "y", "x"), np.full(shape, 0.1)),
        },
        coords={"time": t, "y": np.arange(ny), "x": np.arange(nx)},
    )
    if versions is not None:
        ds = ds.assign_coords(processor_version=("time", versions))
    return ds


class FakeFetcher:
    def __init__(self, ds):
        self.ds = ds
        self.calls = 0

    def fetch_window(self, aoi, t0, t1):
        self.calls += 1
        return self.ds.sel(time=slice(str(t0), str(t1)))


def _cfg(tmp_path, **kw):
    return AcquisitionConfig(raw_dir=tmp_path / "tropomi", **kw)


def test_acquire_writes_overpasses_with_provenance(tmp_path):
    ds = make_ds(["2023-06-05T05:00", "2023-06-06T05:00"])
    cfg = _cfg(tmp_path, start=date(2023, 6, 1))
    report = T.acquire_no2(tangshan_aoi(), cfg, fetcher=FakeFetcher(ds), today=date(2023, 6, 30))

    assert report.fetched == 2
    manifest = T.load_manifest(cfg.raw_dir)
    assert len(manifest["overpasses"]) == 2
    entry = next(iter(manifest["overpasses"].values()))
    assert entry["qa_threshold"] == 0.75
    assert "valid_coverage" in entry and "processor_version" in entry
    # Files written under year subfolder.
    assert list((cfg.raw_dir / "2023").glob("*.nc"))


def test_qa_filter_and_coverage():
    ds = make_ds(["2023-06-05T05:00"], ny=2, nx=2)
    # Half the pixels below the QA threshold.
    ds[T.QA].values[:] = np.array([[0.9, 0.4], [0.9, 0.4]])
    filtered = T.apply_qa(ds, 0.75)
    sub = next(T.iter_overpasses(filtered))[2]
    assert T.coverage_fraction(sub) == pytest.approx(0.5)


def test_resumable_skips_done_batch(tmp_path):
    ds = make_ds(["2023-06-05T05:00", "2023-06-06T05:00"])
    cfg = _cfg(tmp_path, start=date(2023, 6, 1))
    f = FakeFetcher(ds)
    T.acquire_no2(tangshan_aoi(), cfg, fetcher=f, today=date(2023, 6, 30))
    assert f.calls == 1
    # Second run: batch already done -> no fetch, nothing new.
    report2 = T.acquire_no2(tangshan_aoi(), cfg, fetcher=f, today=date(2023, 6, 30))
    assert f.calls == 1
    assert report2.fetched == 0


def test_overpass_level_idempotency(tmp_path):
    ds = make_ds(["2023-06-05T05:00", "2023-06-06T05:00"])
    cfg = _cfg(tmp_path, start=date(2023, 6, 1))
    # Pre-seed one overpass id as already fetched (batch not marked done).
    raw = cfg.raw_dir
    raw.mkdir(parents=True, exist_ok=True)
    T._save_manifest(
        raw,
        {
            "overpasses": {"2023-06-05T050000": {"processor_version": "v2.x"}},
            "batches_done": [],
            "batch_errors": {},
        },
    )
    report = T.acquire_no2(tangshan_aoi(), cfg, fetcher=FakeFetcher(ds), today=date(2023, 6, 30))
    assert report.fetched == 1  # only the not-yet-seen overpass
    assert report.skipped == 1


def test_version_discontinuity_flagged(tmp_path):
    ds = make_ds(["2022-06-10T05:00", "2022-06-20T05:00"], versions=["v1.x", "v2.x"])
    cfg = _cfg(tmp_path, start=date(2022, 6, 1))
    report = T.acquire_no2(tangshan_aoi(), cfg, fetcher=FakeFetcher(ds), today=date(2022, 6, 30))
    assert report.has_version_discontinuity is True
    assert T.version_discontinuity(cfg.raw_dir) is True


def test_batches_cover_range():
    b = T._batches(date(2023, 1, 1), date(2023, 3, 15), "MS")
    assert b[0][0] == date(2023, 1, 1)
    assert b[-1][1] == date(2023, 3, 15)
    # Contiguous, non-empty.
    for a, c in b:
        assert c > a


def test_mint_token_without_credentials_returns_none(monkeypatch):
    # Neutralise .env loading so the dev machine's real credentials don't leak into the test.
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    monkeypatch.delenv("CDSE_USERNAME", raising=False)
    monkeypatch.delenv("CDSE_PASSWORD", raising=False)
    assert T.mint_cdse_token(AcquisitionConfig()) is None


def test_mint_token_uses_password_grant(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    monkeypatch.setenv("CDSE_USERNAME", "u")
    monkeypatch.setenv("CDSE_PASSWORD", "p")
    captured = {}

    class FakeResp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"access_token": "TOKEN123"}

    def fake_post(url, data=None, timeout=None):
        captured["url"] = url
        captured["data"] = data
        return FakeResp()

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)
    tok = T.mint_cdse_token(AcquisitionConfig())
    assert tok == "TOKEN123"
    assert captured["data"]["grant_type"] == "password"
    assert captured["data"]["client_id"] == "cdse-public"
    # The secret password is passed to CDSE only, not returned/logged by us.
    assert captured["data"]["password"] == "p"
