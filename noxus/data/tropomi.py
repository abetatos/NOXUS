"""Acquire Sentinel-5P/TROPOMI tropospheric NO2 over the study AOI via openEO.

Downloading full orbit granules is infeasible (~3,734 overpasses x ~609 MB ~= 2.2 TB), so this module
asks **openEO on CDSE** for a **server-side spatial subset** of the AOI and keeps only that window
(~0.2-1 GB). Authentication reuses the credentials in ``.env``: a CDSE token is minted via the OIDC
password grant and handed to openEO; if that fails it falls back to openEO's interactive OIDC (refresh
token, else a device flow that opens the web) — see ``decisions/architecture-decisions.md``.

The network/openEO interaction sits behind the :class:`Fetcher` seam so the QA, coverage, version and
persistence logic is unit-testable without credentials or network (tests inject a fake fetcher).

Scope: this produces a per-overpass, AOI-clipped, QA-filtered NO2 store. Gridding/compositing is
NOX-002b; attribution is NOX-003. At TROPOMI's ~5.5x3.5 km footprint a plant is sub-pixel, so the
signal is cluster/sub-cluster level, not per-factory (see the spec assumptions).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterator, Protocol

import pandas as pd
import xarray as xr

from noxus.config.region import Region
from noxus.config.run import AcquisitionConfig

# Standardised variable names used downstream, regardless of the source band names.
NO2 = "no2"
QA = "qa"
CLOUD = "cloud"
# Fallback processor-version cutoff used only when the source does not expose a version. The S5P
# tropospheric NO2 algorithm moved v1.x -> v2.x around mid-2021/2022; treated as approximate.
_VERSION_CUTOFF = date(2022, 7, 1)


class AcquisitionError(RuntimeError):
    """Acquisition could not complete (collection/job problem)."""


class AcquisitionAuthError(AcquisitionError):
    """CDSE/openEO authentication failed."""


class Fetcher(Protocol):
    """Returns an AOI window as an xarray Dataset with vars ``no2``/``qa``/``cloud`` over ``time``."""

    def fetch_window(self, aoi: Region, t0: date, t1: date) -> xr.Dataset: ...


# --------------------------------------------------------------------------- auth + openEO fetcher


def mint_cdse_token(cfg: AcquisitionConfig) -> str | None:
    """Mint a CDSE access token via the OIDC password grant using ``.env`` credentials.

    Returns the access token, or ``None`` if the credentials are not set. Secrets are never logged.
    """
    from dotenv import load_dotenv

    load_dotenv()  # load .env (names per .env.example); values stay in the environment only
    user = os.environ.get(cfg.env_username)
    password = os.environ.get(cfg.env_password)
    if not (user and password):
        return None

    import httpx

    resp = httpx.post(
        cfg.token_url,
        data={
            "username": user,
            "password": password,
            "grant_type": "password",
            "client_id": cfg.cdse_client_id,
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.json().get("access_token")


def connect_openeo(cfg: AcquisitionConfig):
    """Connect and authenticate to CDSE openEO: password-grant token first, then interactive OIDC."""
    import openeo

    conn = openeo.connect(cfg.openeo_url)
    try:
        token = mint_cdse_token(cfg)
    except Exception as exc:  # noqa: BLE001 - surface a clean auth error, never the secret
        raise AcquisitionAuthError(
            f"Failed to mint a CDSE token from ${cfg.env_username}/${cfg.env_password}: {exc}"
        ) from exc

    if token:
        try:
            conn.authenticate_oidc_access_token(token)
            return conn
        except Exception:  # noqa: BLE001 - fall back to interactive OIDC below
            pass
    try:
        conn.authenticate_oidc()  # refresh token, else device flow (opens the web)
    except Exception as exc:  # noqa: BLE001
        raise AcquisitionAuthError(
            "openEO authentication failed. Set "
            f"{cfg.env_username}/{cfg.env_password} in .env (CDSE account) or complete the "
            "interactive login."
        ) from exc
    return conn


@dataclass
class OpenEOFetcher:
    """Real :class:`Fetcher`: builds the openEO subset graph and downloads the AOI window."""

    cfg: AcquisitionConfig
    _conn: object | None = field(default=None, repr=False)

    def _connection(self):
        if self._conn is None:
            import openeo

            self._conn = openeo.connect(self.cfg.openeo_url)
        return self._conn

    def _authenticate(self) -> None:
        """Refresh auth before each batch: CDSE access tokens expire (~1 h), which would otherwise
        kill a long run with 401s. Re-minting per batch (one HTTP POST) keeps the token fresh."""
        conn = self._connection()
        try:
            token = mint_cdse_token(self.cfg)
        except Exception as exc:  # noqa: BLE001
            raise AcquisitionAuthError(
                f"Failed to mint a CDSE token from ${self.cfg.env_username}/${self.cfg.env_password}: {exc}"
            ) from exc
        if token:
            try:
                conn.authenticate_oidc_access_token(token)
                return
            except Exception:  # noqa: BLE001
                pass
        try:
            conn.authenticate_oidc()  # refresh token, else device flow (opens the web)
        except Exception as exc:  # noqa: BLE001
            raise AcquisitionAuthError(
                f"openEO authentication failed. Set {self.cfg.env_username}/{self.cfg.env_password} "
                "in .env (CDSE account) or complete the interactive login."
            ) from exc

    def fetch_window(self, aoi: Region, t0: date, t1: date) -> xr.Dataset:
        cfg = self.cfg
        self._authenticate()  # fresh token each batch (avoids mid-run token expiry)
        # The CDSE SENTINEL_5P_L2 collection serves ONE band per request, so NO2 and CLOUD_FRACTION
        # are fetched separately and merged. NO2 is already quality-screened by the collection (no
        # qa_value band is exposed); cloud_fraction is used for clear-day selection downstream.
        try:
            no2 = self._download_band(aoi, t0, t1, cfg.no2_band, NO2)
            try:
                cloud = self._download_band(aoi, t0, t1, cfg.cloud_band, CLOUD)
                no2[CLOUD] = cloud[CLOUD]
            except Exception:  # noqa: BLE001 - cloud band is optional context, not fatal
                pass
            return no2
        except AcquisitionError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise AcquisitionError(
                f"openEO fetch failed for {cfg.collection_id} {t0}..{t1}: {exc}"
            ) from exc

    def _download_band(self, aoi: Region, t0: date, t1: date, band: str, name: str) -> xr.Dataset:
        cfg = self.cfg
        conn = self._connection()
        cube = conn.load_collection(
            cfg.collection_id,
            spatial_extent={
                "west": aoi.min_lon,
                "south": aoi.min_lat,
                "east": aoi.max_lon,
                "north": aoi.max_lat,
            },
            temporal_extent=[t0.isoformat(), t1.isoformat()],
            bands=[band],
        )
        tmp = Path(cfg.raw_dir) / f"_tmp_{band}_{t0.isoformat()}.nc"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        cube.download(str(tmp))
        ds = xr.open_dataset(tmp).load()
        ds.close()  # release the file handle before deleting (Windows locks open files)
        tmp.unlink(missing_ok=True)
        # Rename the single data variable to the standard name, and the time dim ('t') to 'time'.
        data_vars = [v for v in ds.data_vars if v != "crs"]
        src = band if band in ds.data_vars else (data_vars[0] if data_vars else band)
        if src in ds.data_vars:
            ds = ds.rename({src: name})
        if "t" in ds.dims:
            ds = ds.rename({"t": "time"})
        return ds


# ----------------------------------------------------------------------------------- processing


def apply_qa(ds: xr.Dataset, threshold: float) -> xr.Dataset:
    """Mask NO2 where the quality value is below ``threshold`` (REQ-003)."""
    if QA in ds:
        ds = ds.copy()
        ds[NO2] = ds[NO2].where(ds[QA] >= threshold)
    return ds


def _safe_mean(overpass: xr.Dataset, var: str) -> float | None:
    """Mean of a variable over the AOI window, ignoring NaN; None if absent/empty."""
    if var not in overpass:
        return None
    value = overpass[var].mean(skipna=True).item()
    return None if pd.isna(value) else float(value)


def coverage_fraction(overpass: xr.Dataset) -> float:
    """Fraction of AOI pixels with a valid (non-NaN) NO2 value after QA (REQ-006)."""
    total = int(overpass[NO2].size)
    if total == 0:
        return 0.0
    valid = int(overpass[NO2].notnull().sum())
    return valid / total


def _infer_version(d: date) -> str:
    """Fallback processor-version label when the source does not expose one."""
    return "v1.x" if d < _VERSION_CUTOFF else "v2.x"


def _overpass_version(overpass: xr.Dataset, when: date) -> str:
    if "processor_version" in overpass.coords:
        return str(overpass["processor_version"].values)
    if "processor_version" in overpass.attrs:
        return str(overpass.attrs["processor_version"])
    return _infer_version(when)


def iter_overpasses(ds: xr.Dataset) -> Iterator[tuple[str, datetime, xr.Dataset]]:
    """Yield (overpass_id, timestamp, single-time Dataset) for each time step."""
    for t in ds["time"].values:
        ts = pd.Timestamp(t).to_pydatetime()
        opid = ts.strftime("%Y-%m-%dT%H%M%S")
        yield opid, ts, ds.sel(time=t)


# ------------------------------------------------------------------------------- persistence


def _manifest_path(raw_dir: Path) -> Path:
    return Path(raw_dir) / "manifest.json"


def load_manifest(raw_dir: Path) -> dict:
    path = _manifest_path(raw_dir)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"overpasses": {}, "batches_done": [], "batch_errors": {}}


def _save_manifest(raw_dir: Path, manifest: dict) -> None:
    _manifest_path(raw_dir).write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )


def _write_overpass(
    raw_dir: Path,
    opid: str,
    overpass: xr.Dataset,
    cfg: AcquisitionConfig,
    version: str,
    cov: float,
    aoi_bbox: tuple[float, float, float, float],
) -> Path:
    out = Path(raw_dir) / opid[:4] / f"{opid}.nc"
    out.parent.mkdir(parents=True, exist_ok=True)
    overpass = overpass.copy()
    # Drop the version string coord (kept in attrs) and the CRS grid-mapping var; clear inherited
    # encodings (openEO marks time as an unlimited dim, which h5netcdf cannot resize on a scalar slice).
    overpass = overpass.drop_vars(
        [c for c in ("processor_version", "crs") if c in overpass.variables], errors="ignore"
    )
    for var in list(overpass.variables):
        overpass[var].encoding = {}
    overpass.encoding = {}
    overpass.attrs.update(
        {
            "collection": cfg.collection_id,
            "processor_version": version,
            "qa_threshold": cfg.qa_threshold,
            "aoi_bbox": list(aoi_bbox),
            "valid_coverage": cov,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    overpass.to_netcdf(out, engine="h5netcdf")
    return out


# --------------------------------------------------------------------------------- batching


def _batches(start: date, end: date, freq: str) -> list[tuple[date, date]]:
    """Split [start, end] into [t0, t1) periods at the given pandas offset (e.g. 'MS')."""
    edges = list(pd.date_range(start=start, end=end, freq=freq))
    if not edges or edges[0].date() > start:
        edges = [pd.Timestamp(start), *edges]
    edges.append(pd.Timestamp(end))
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        a_d, b_d = a.date(), b.date()
        if b_d > a_d:
            out.append((a_d, b_d))
    return out


@dataclass(frozen=True)
class AcquireReport:
    """Summary of an acquisition run."""

    fetched: int
    skipped: int
    failed_batches: int
    versions: tuple[str, ...]
    raw_dir: str

    @property
    def has_version_discontinuity(self) -> bool:
        return len(set(self.versions)) > 1


def acquire_no2(
    aoi: Region,
    cfg: AcquisitionConfig | None = None,
    fetcher: Fetcher | None = None,
    today: date | None = None,
) -> AcquireReport:
    """Acquire AOI-clipped, QA-filtered NO2 over [cfg.start, end], resumable via a manifest.

    Server-side subsetting + QA via the injected ``fetcher`` (default: openEO). Each overpass is
    written once; re-runs skip work already recorded in the manifest (REQ-001..007).
    """
    cfg = cfg or AcquisitionConfig()
    raw_dir = Path(cfg.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    aoi_bbox = aoi.as_bbox()
    manifest = load_manifest(raw_dir)
    end = cfg.end or (today or date.today())
    fetcher = fetcher or OpenEOFetcher(cfg)

    fetched = skipped = failed = 0
    versions: list[str] = list({v["processor_version"] for v in manifest["overpasses"].values()})

    for t0, t1 in _batches(cfg.start, end, cfg.batch_freq):
        bkey = t0.strftime("%Y-%m")
        if bkey in manifest["batches_done"]:
            continue
        try:
            ds = fetcher.fetch_window(aoi, t0, t1)
        except Exception as exc:  # noqa: BLE001 - record and continue; resume later
            failed += 1
            manifest["batch_errors"][bkey] = str(exc)
            _save_manifest(raw_dir, manifest)
            continue

        ds = apply_qa(ds, cfg.qa_threshold)
        for opid, ts, sub in iter_overpasses(ds):
            if opid in manifest["overpasses"]:
                skipped += 1
                continue
            cov = coverage_fraction(sub)
            version = _overpass_version(sub, ts.date())
            path = _write_overpass(raw_dir, opid, sub, cfg, version, cov, aoi_bbox)
            manifest["overpasses"][opid] = {
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "processor_version": version,
                "valid_coverage": cov,
                "cloud_mean": _safe_mean(sub, CLOUD),
                "no2_mean": _safe_mean(sub, NO2),
                "path": str(path),
                "qa_threshold": cfg.qa_threshold,
            }
            versions.append(version)
            fetched += 1

        manifest["batches_done"].append(bkey)
        manifest["batch_errors"].pop(bkey, None)
        _save_manifest(raw_dir, manifest)

    return AcquireReport(
        fetched=fetched,
        skipped=skipped,
        failed_batches=failed,
        versions=tuple(versions),
        raw_dir=str(raw_dir),
    )


def version_discontinuity(raw_dir: Path | str) -> bool:
    """True if the stored overpasses span more than one processor version (EDGE-002)."""
    manifest = load_manifest(Path(raw_dir))
    return len({v["processor_version"] for v in manifest["overpasses"].values()}) > 1
