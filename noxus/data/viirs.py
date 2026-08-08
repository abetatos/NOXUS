"""Acquire VIIRS thermal-anomaly detections over the study AOI (NOX-011, REQ-000..006).

An **independent, hard-to-falsify physical activity proxy** for the Tangshan steel cluster: VIIRS
thermal anomalies over the catalogued steel facilities. Blast furnaces, sinter plants and coke ovens
are high-temperature combustion sources VIIRS detects from orbit, so a space-based thermometer cannot
be dressed up by any statistical office. Two free/public sources share one :class:`ViirsFetcher` seam:

  * **VNF** — VIIRS Nightfire (NOAA / Colorado School of Mines Earth Observation Group). Per-detection
    temperature (K), source area and radiant heat (MW); the industrial-combustion-tuned *primary*.
  * **FIRMS** — NASA FIRMS VIIRS active fire (VNP14IMGT + VJ114). Active-fire-tuned; its
    *static-land-source* detections are kept as an independent *cross-check* (REQ-006, EDGE-004).

Mirrors ``noxus/data/tropomi.py``: the network interaction sits behind the fetcher seam so the AOI
clip, field retention, per-night persistence and resumable manifest are unit-testable without network
or credentials (tests inject a fake fetcher). VIIRS-derived raw detections live under gitignored
``data/raw/viirs/`` (NFR-003); only small derived stats + figures are committable.

Credentials (FIRMS ``FIRMS_MAP_KEY``) are read from the environment **by variable name only** — never
read into logs/output, never committed (REQ-005, ERR-001). VNF was defaulted keyless, but the 2026-07-12
T1 probe found the EOG host answers a Keycloak login page for the guessed path — VNF auth is a T1
open item (Q1); :class:`VnfFetcher` surfaces that as a clear :class:`ViirsAuthError`.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol

import pandas as pd

from noxus.config.region import Region
from noxus.config.run import ViirsAcquisitionConfig

# Standardised per-detection columns kept downstream, whatever the source band/column names (REQ-002).
TEMPERATURE_K = "temperature_k"
SOURCE_AREA = "source_area"
RADIANT_HEAT_MW = "radiant_heat_mw"
LATITUDE = "latitude"
LONGITUDE = "longitude"
TIMESTAMP = "timestamp"
DETECTION_ID = "detection_id"
QUALITY_FLAG = "quality_flag"

STANDARD_COLUMNS = (
    TEMPERATURE_K,
    SOURCE_AREA,
    RADIANT_HEAT_MW,
    LATITUDE,
    LONGITUDE,
    TIMESTAMP,
    DETECTION_ID,
    QUALITY_FLAG,
)
# The minimum a detection frame must expose to be usable (radiant heat is the primary aggregate).
REQUIRED_COLUMNS = (RADIANT_HEAT_MW, LATITUDE, LONGITUDE, TIMESTAMP)


class ViirsAcquisitionError(RuntimeError):
    """Acquisition could not complete (endpoint/format/field problem)."""


class ViirsAuthError(ViirsAcquisitionError):
    """A required credential (referenced by env-var name) is missing or invalid."""


class BudgetExceededError(ViirsAcquisitionError):
    """The measured/implied download volume exceeds the configured budget (REQ-000, ERR-004)."""


class ViirsFetcher(Protocol):
    """Returns the AOI thermal detections for one night as a standardised DataFrame."""

    def fetch_night(self, aoi: Region, day: date) -> pd.DataFrame: ...


# --------------------------------------------------------------------- clip + standardise helpers


def clip_to_aoi(df: pd.DataFrame, aoi: Region) -> pd.DataFrame:
    """Keep only detections whose lat/lon fall inside the AOI bbox (client-side clip, REQ-001)."""
    if df.empty:
        return df
    lat, lon = df[LATITUDE].astype(float), df[LONGITUDE].astype(float)
    keep = (lon >= aoi.min_lon) & (lon <= aoi.max_lon) & (lat >= aoi.min_lat) & (lat <= aoi.max_lat)
    return df.loc[keep].reset_index(drop=True)


def standardise(df: pd.DataFrame, column_map: dict[str, str] | None = None) -> pd.DataFrame:
    """Rename source columns to the standard names and validate the required fields (ERR-002).

    Frames that already use the standard names (e.g. test fixtures) pass through. Missing *required*
    fields raise a descriptive error naming the fields; missing *optional* fields are filled with NA so
    the schema is stable.
    """
    out = df.rename(columns=column_map) if column_map else df.copy()
    missing = [c for c in REQUIRED_COLUMNS if c not in out.columns]
    if missing:
        raise ViirsAcquisitionError(
            f"VIIRS detection frame is missing required field(s) {missing}; "
            f"present columns were {list(df.columns)}."
        )
    for col in STANDARD_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out[TIMESTAMP] = pd.to_datetime(out[TIMESTAMP])
    return out[list(STANDARD_COLUMNS)].reset_index(drop=True)


def _scrub_url(url: str) -> str:
    """Remove a FIRMS MAP_KEY path segment from a URL before it appears in any error/log (REQ-005)."""
    return url.split("/csv/", 1)[0] + "/csv/<MAP_KEY>/..." if "/csv/" in url else url


# ------------------------------------------------------------------------------- real fetchers


@dataclass
class VnfFetcher:
    """Real VNF :class:`ViirsFetcher`: download one per-night VNF file and clip to the AOI.

    The exact endpoint/filename/format and whether any auth is needed are the T1 unknowns (Q1); the
    config carries the safe defaults. Column mapping to the standard names is applied in
    :func:`standardise`; unknown VNF headers should be added to ``column_map`` once confirmed at T1.
    """

    cfg: ViirsAcquisitionConfig
    column_map: dict[str, str] = field(default_factory=dict)

    def _url(self, day: date) -> str:
        fname = self.cfg.vnf_file_template.format(yyyymmdd=day.strftime("%Y%m%d"))
        return f"{self.cfg.vnf_endpoint_base.rstrip('/')}/{day.strftime('%Y')}/{fname}"

    def fetch_night(self, aoi: Region, day: date) -> pd.DataFrame:
        import httpx

        url = self._url(day)
        try:
            resp = httpx.get(url, timeout=self.cfg.request_timeout, follow_redirects=True)
            if resp.status_code == 404:
                return standardise(_empty_frame())  # no VNF product this night -> empty (EDGE-001)
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001 - descriptive, endpoint named, no secret in VNF URLs
            raise ViirsAcquisitionError(f"VNF fetch failed for {day} at {url}: {exc}") from exc
        # T1 finding (2026-07-12): the EOG host answers a 200 HTML Keycloak login page for the guessed
        # path, i.e. VNF is NOT keyless as defaulted (Q1). Detect a login/HTML body and surface a clear
        # auth error instead of a confusing CSV parser error (ERR-002), pending T1 endpoint/auth pinning.
        ctype = resp.headers.get("content-type", "")
        if "html" in ctype.lower() or resp.content[:15].lstrip().lower().startswith(
            b"<!doctype html"
        ):
            raise ViirsAuthError(
                f"VNF endpoint returned an HTML/login page for {day} at {url} (content-type={ctype!r}). "
                "VNF appears to require EOG authentication — resolve the exact endpoint/filename/auth at "
                "the T1 spike (open-questions Q1) before the acquisition loop is run."
            )
        from io import BytesIO

        raw = pd.read_csv(BytesIO(resp.content), compression="infer")
        return clip_to_aoi(standardise(raw, self.column_map or None), aoi)


@dataclass
class FirmsFetcher:
    """FIRMS :class:`ViirsFetcher` cross-check: area+date CSV API, static-land-source only (REQ-006).

    Reads ``FIRMS_MAP_KEY`` from the environment **by name** (never logged). Keeps only the FIRMS
    detection ``type`` code for a static land source (EDGE-004); vegetation-fire/volcano/offshore are
    dropped. FIRMS frp -> radiant heat (MW); bright_ti4 -> temperature (K).
    """

    cfg: ViirsAcquisitionConfig
    product: str | None = None

    _FIRMS_MAP = {
        "frp": RADIANT_HEAT_MW,
        "bright_ti4": TEMPERATURE_K,
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
    }

    def _map_key(self) -> str:
        from dotenv import load_dotenv

        load_dotenv()  # names per .env.example; values stay in the environment only
        key = os.environ.get(self.cfg.firms_env_key)
        if not key:
            raise ViirsAuthError(
                f"FIRMS acquisition requires the ${self.cfg.firms_env_key} environment variable "
                "(request a MAP_KEY from https://firms.modaps.eosdis.nasa.gov/api/map_key/ and set it "
                "in .env). The value is never logged or committed."
            )
        return key

    def _url(self, aoi: Region, day: date) -> str:
        key = self._map_key()
        product = self.product or self.cfg.firms_products[0]
        w, s, e, n = aoi.as_bbox()
        bbox = f"{w},{s},{e},{n}"
        return (
            f"{self.cfg.firms_endpoint_base.rstrip('/')}/{key}/{product}/{bbox}/1/"
            f"{day.strftime('%Y-%m-%d')}"
        )

    def fetch_night(self, aoi: Region, day: date) -> pd.DataFrame:
        import httpx

        url = self._url(aoi, day)
        try:
            resp = httpx.get(url, timeout=self.cfg.request_timeout, follow_redirects=True)
            resp.raise_for_status()
        except ViirsAuthError:
            raise
        except Exception as exc:  # noqa: BLE001 - scrub the MAP_KEY out of the URL before surfacing
            raise ViirsAcquisitionError(
                f"FIRMS fetch failed for {day} at {_scrub_url(url)}: {exc}"
            ) from exc
        from io import StringIO

        raw = pd.read_csv(StringIO(resp.text))
        return self.filter_and_standardise(raw, aoi, day)

    def filter_and_standardise(self, raw: pd.DataFrame, aoi: Region, day: date) -> pd.DataFrame:
        """Keep static-land-source detections and map FIRMS columns to the standard schema."""
        if raw.empty:
            return standardise(_empty_frame())
        if "type" in raw.columns:
            raw = raw.loc[raw["type"] == self.cfg.firms_static_land_type].copy()
        ts = _firms_timestamp(raw, day)
        mapped = raw.rename(columns=self._FIRMS_MAP)
        mapped[TIMESTAMP] = ts
        mapped[DETECTION_ID] = [f"firms-{day:%Y%m%d}-{i}" for i in range(len(mapped))]
        if QUALITY_FLAG not in mapped.columns and "confidence" in raw.columns:
            mapped[QUALITY_FLAG] = raw["confidence"].to_numpy()
        return clip_to_aoi(standardise(mapped), aoi)


def _firms_timestamp(raw: pd.DataFrame, day: date) -> pd.Series:
    if "acq_date" in raw.columns:
        times = raw.get("acq_time")
        if times is not None:
            hhmm = times.astype(int).astype(str).str.zfill(4)
            return pd.to_datetime(
                raw["acq_date"].astype(str) + " " + hhmm.str[:2] + ":" + hhmm.str[2:],
                errors="coerce",
            )
        return pd.to_datetime(raw["acq_date"], errors="coerce")
    return pd.Series([pd.Timestamp(day)] * len(raw), index=raw.index)


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in REQUIRED_COLUMNS})


def _default_fetcher(cfg: ViirsAcquisitionConfig) -> ViirsFetcher:
    return FirmsFetcher(cfg) if cfg.source == "firms" else VnfFetcher(cfg)


# ------------------------------------------------------------------------------- persistence


def _manifest_path(raw_dir: Path, source: str) -> Path:
    return Path(raw_dir) / source / "manifest.json"


def load_manifest(raw_dir: Path, source: str = "vnf") -> dict:
    path = _manifest_path(raw_dir, source)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"source": source, "nights": {}, "batches_done": [], "batch_errors": {}}


def _save_manifest(raw_dir: Path, source: str, manifest: dict) -> None:
    path = _manifest_path(raw_dir, source)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


def _write_night(
    raw_dir: Path,
    source: str,
    day: date,
    detections: pd.DataFrame,
    aoi_bbox: tuple[float, float, float, float],
    endpoint: str,
) -> Path:
    out = Path(raw_dir) / source / day.strftime("%Y") / f"{day.isoformat()}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = detections.copy()
    # Provenance carried as frame attrs + a sidecar (parquet attrs are not always preserved).
    provenance = {
        "source": source,
        "aoi_bbox": list(aoi_bbox),
        "endpoint": endpoint,
        "date": day.isoformat(),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "n_detections": int(len(frame)),
    }
    frame.attrs.update(provenance)
    frame.to_parquet(out, index=False)
    out.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    return out


# --------------------------------------------------------------------------------- batching


def _nights(start: date, end: date) -> list[date]:
    n = (end - start).days
    return [start + timedelta(days=i) for i in range(n + 1)]


def _implied_full_era_mb(per_night_mb: float, start: date, end: date) -> float:
    return per_night_mb * max(1, (end - start).days + 1)


def raise_if_over_budget(estimated_mb: float, cfg: ViirsAcquisitionConfig) -> None:
    """Surface a no-go rather than silently starting an oversized download (ERR-004)."""
    if cfg.download_budget_mb is not None and estimated_mb > cfg.download_budget_mb:
        raise BudgetExceededError(
            f"Estimated download {estimated_mb:.1f} MB exceeds the configured budget "
            f"{cfg.download_budget_mb:.1f} MB — stopping (record the T1 no-go)."
        )


@dataclass(frozen=True)
class AcquireReport:
    """Summary of a VIIRS acquisition run."""

    source: str
    fetched: int
    skipped: int
    failed: int
    empty_nights: int
    n_detections: int
    raw_dir: str


def _fetch_with_retry(
    fetcher: ViirsFetcher, aoi: Region, day: date, cfg: ViirsAcquisitionConfig
) -> pd.DataFrame:
    last: Exception | None = None
    for attempt in range(cfg.max_retries):
        try:
            return fetcher.fetch_night(aoi, day)
        except ViirsAuthError:
            raise  # a missing key will not fix itself on retry
        except Exception as exc:  # noqa: BLE001 - retry with backoff, then surface
            last = exc
            if attempt < cfg.max_retries - 1:
                time.sleep(cfg.backoff_seconds * (attempt + 1))
    raise ViirsAcquisitionError(
        f"VIIRS fetch failed for {day} after {cfg.max_retries} tries: {last}"
    )


def acquire_viirs(
    aoi: Region,
    cfg: ViirsAcquisitionConfig | None = None,
    fetcher: ViirsFetcher | None = None,
    today: date | None = None,
) -> AcquireReport:
    """Acquire AOI-clipped VIIRS detections over [cfg.start, end], resumable via a manifest.

    Per night: fetch -> clip to AOI -> retain the standard fields -> per-night parquet + provenance ->
    manifest update. Re-runs skip nights already recorded; a failed night is recorded and retried on a
    later resume without re-downloading done nights (REQ-004, ERR-003). Zero-detection nights are
    recorded as zero-activity, not errors and not interpolated (EDGE-001).
    """
    cfg = cfg or ViirsAcquisitionConfig()
    raw_dir = Path(cfg.raw_dir)
    source = cfg.source
    fetcher = fetcher or _default_fetcher(cfg)
    endpoint = cfg.firms_endpoint_base if source == "firms" else cfg.vnf_endpoint_base
    aoi_bbox = aoi.as_bbox()
    manifest = load_manifest(raw_dir, source)
    end = cfg.end or (today or date.today())

    fetched = skipped = failed = empty = total_detections = 0
    for day in _nights(cfg.start, end):
        key = day.isoformat()
        if key in manifest["nights"]:
            skipped += 1
            continue
        try:
            detections = _fetch_with_retry(fetcher, aoi, day, cfg)
        except Exception as exc:  # noqa: BLE001 - record and continue; resume later (ERR-003)
            failed += 1
            manifest["batch_errors"][key] = str(exc)
            _save_manifest(raw_dir, source, manifest)
            continue
        detections = clip_to_aoi(standardise(detections), aoi)
        if cfg.min_temperature_k is not None and TEMPERATURE_K in detections:
            temp = pd.to_numeric(detections[TEMPERATURE_K], errors="coerce")
            detections = detections.loc[temp.isna() | (temp >= cfg.min_temperature_k)]
            detections = detections.reset_index(drop=True)
        path = _write_night(raw_dir, source, day, detections, aoi_bbox, endpoint)
        n = int(len(detections))
        total_detections += n
        if n == 0:
            empty += 1
        manifest["nights"][key] = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "n_detections": n,
            "path": str(path),
        }
        manifest["batch_errors"].pop(key, None)
        fetched += 1
        _save_manifest(raw_dir, source, manifest)

    return AcquireReport(
        source=source,
        fetched=fetched,
        skipped=skipped,
        failed=failed,
        empty_nights=empty,
        n_detections=total_detections,
        raw_dir=str(raw_dir),
    )


def load_detections(raw_dir: Path | str, source: str = "vnf") -> pd.DataFrame:
    """Concatenate all persisted per-night AOI-clipped detections for a source (join input)."""
    manifest = load_manifest(Path(raw_dir), source)
    frames = []
    for entry in manifest["nights"].values():
        p = Path(entry["path"])
        if p.exists():
            frames.append(pd.read_parquet(p))
    if not frames:
        return standardise(_empty_frame())
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------- T1 feasibility spike (probe)


@dataclass(frozen=True)
class ViirsProbe:
    """T1 feasibility-spike measurement + go/no-go (REQ-000, AT0)."""

    source: str
    endpoint: str
    auth_required: bool
    nights_probed: int
    nights_with_detections: int
    n_detections: int
    fields_available: tuple[str, ...]
    per_night_mb: float
    implied_full_era_mb: float
    facilities_colocated: int
    within_budget: bool
    go: bool
    notes: str


def probe_viirs(
    aoi: Region,
    cfg: ViirsAcquisitionConfig | None = None,
    fetcher: ViirsFetcher | None = None,
    today: date | None = None,
    facilities: pd.DataFrame | None = None,
    radius_km: float = 3.0,
) -> ViirsProbe:
    """T1 spike: fetch a small window, measure volume/fields/coverage/co-location, record go/no-go.

    Fetches ``cfg.probe_days`` nights, measures per-night data volume (implying the full-era volume),
    the available detection fields, temporal coverage (nights with >=1 detection), and how many
    detections co-locate with the facilities (within ``radius_km``). ``go`` is True only when the
    implied full-era volume is within ``cfg.download_budget_mb`` (ERR-004). No bulk download is started.
    """
    cfg = cfg or ViirsAcquisitionConfig()
    fetcher = fetcher or _default_fetcher(cfg)
    endpoint = cfg.firms_endpoint_base if cfg.source == "firms" else cfg.vnf_endpoint_base
    end = today or date.today()
    start = end - timedelta(days=cfg.probe_days - 1)

    per_night_bytes: list[float] = []
    nights_with = n_det = colocated = 0
    fields: set[str] = set()
    auth_required = cfg.source == "firms"

    for day in _nights(start, end):
        detections = clip_to_aoi(standardise(fetcher.fetch_night(aoi, day)), aoi)
        fields.update(c for c in detections.columns if detections[c].notna().any())
        k = int(len(detections))
        n_det += k
        if k > 0:
            nights_with += 1
        per_night_bytes.append(float(detections.memory_usage(deep=True).sum()))
        if k and facilities is not None and not facilities.empty:
            colocated += _count_colocated(detections, facilities, radius_km)

    per_night_mb = (sum(per_night_bytes) / max(1, len(per_night_bytes))) / 1e6
    implied_mb = _implied_full_era_mb(per_night_mb, cfg.start, end)
    within_budget = cfg.download_budget_mb is None or implied_mb <= cfg.download_budget_mb
    go = within_budget and n_det > 0
    notes = (
        "GO — VNF/FIRMS detections co-locate with the facilities and the implied volume is within "
        "budget."
        if go
        else (
            "NO-GO — implied full-era volume exceeds the budget."
            if not within_budget
            else "NO-GO — no detections over the AOI in the probe window (check endpoint/date/auth)."
        )
    )
    return ViirsProbe(
        source=cfg.source,
        endpoint=endpoint,
        auth_required=auth_required,
        nights_probed=cfg.probe_days,
        nights_with_detections=nights_with,
        n_detections=n_det,
        fields_available=tuple(sorted(fields)),
        per_night_mb=per_night_mb,
        implied_full_era_mb=implied_mb,
        facilities_colocated=colocated,
        within_budget=within_budget,
        go=go,
        notes=notes,
    )


def _count_colocated(detections: pd.DataFrame, facilities: pd.DataFrame, radius_km: float) -> int:
    from noxus.signal.thermal import join_to_facilities

    return int(len(join_to_facilities(detections, facilities, radius_km)))
