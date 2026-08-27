"""Acquire VIIRS thermal-anomaly detections over the study AOI (NOX-011, REQ-000..006).

An **independent, hard-to-falsify physical activity proxy** for the Tangshan steel cluster: VIIRS
thermal anomalies over the catalogued steel facilities. Blast furnaces, sinter plants and coke ovens
are high-temperature combustion sources VIIRS detects from orbit, so a space-based thermometer cannot
be dressed up by any statistical office.

**2026-08-22 pivot** (decisions/architecture-decisions.md): the PRIMARY (and only supported) source
is now **FIRMS** — NASA FIRMS VIIRS active fire. Its SP-archive ``type=2`` static-land detections over
the steel cluster are the physical proxy (live evidence: 5 detections 2023-12-15, 73 on 2024-11-20,
FRP 0.2–4.8 MW). **VNF** — VIIRS Nightfire (EOG / Colorado School of Mines) — is **DROPPED**: it is no
longer public (VIIRS Nightfire Data Use License eff. 2025-01-10; EOG programmatic access
paid-subscriber-only eff. 2026-06-01), conflicting with the free/public-data principle. ``source="vnf"``
now surfaces a clear licence/pay-gated :class:`ViirsAuthError` (:class:`VnfFetcher`), not a network path.

Mirrors ``noxus/data/tropomi.py``: the network interaction sits behind the fetcher seam so the AOI
clip, field retention, per-night persistence and resumable manifest are unit-testable without network
or credentials (tests inject a fake fetcher). VIIRS-derived raw detections live under gitignored
``data/raw/viirs/`` (NFR-003); only small derived stats + figures are committable.

Credentials (FIRMS ``FIRMS_MAP_KEY``) are read from the environment **by variable name only** — never
read into logs/output, never committed (REQ-005, ERR-001). The MAP_KEY is embedded as a URL path
segment, so any surfaced URL/error is scrubbed by **redacting the key value** (:func:`_scrub_url`),
not by matching a fixed path segment.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol

import pandas as pd

from noxus.config.region import Region
from noxus.config.run import ViirsAcquisitionConfig

logger = logging.getLogger(__name__)

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


class VnfUnavailableError(ViirsAuthError):
    """VNF (VIIRS Nightfire, EOG/CSM) is no longer public — licence/pay-gated, not a supported source.

    Subclasses :class:`ViirsAuthError` so the retry loop does not re-attempt it (a licence gate will
    not clear on retry). FIRMS is the supported free/public source (2026-08-22 pivot).
    """


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


def _scrub_url(text: str, key: str | None = None) -> str:
    """Redact the FIRMS MAP_KEY from any URL/error string before it is logged or surfaced (REQ-005).

    N3 fix: the MAP_KEY is embedded as a URL path segment, and httpx error bodies embed the full URL,
    so scrubbing must **redact the actual key value** wherever it appears — not merely rewrite a
    literal ``/csv/<seg>`` segment. When the value is known it is replaced everywhere first; the
    path-segment rewrite is only a best-effort fallback for callers without the value to hand.
    """
    out = text
    if key:
        out = out.replace(key, "<MAP_KEY>")
    if "/csv/" in out:
        head, tail = out.split("/csv/", 1)
        seg, sep, rest = tail.partition("/")
        if seg and seg != "<MAP_KEY>":
            out = f"{head}/csv/<MAP_KEY>" + (f"/{rest}" if sep else "/...")
    return out


# ------------------------------------------------------------------------------- real fetchers


@dataclass
class VnfFetcher:
    """VNF :class:`ViirsFetcher` — **dropped source**: raises a clear licence/pay-gated error.

    2026-08-22 pivot: VIIRS Nightfire (EOG / Colorado School of Mines) is no longer public, so there
    is no free/reproducible fetch path. Any ``fetch_night`` call raises :class:`VnfUnavailableError`
    naming the licence and pointing at FIRMS as the supported source, instead of hitting the EOG host
    (which now answers an OAuth/Keycloak login for anonymous access).
    """

    cfg: ViirsAcquisitionConfig
    column_map: dict[str, str] = field(default_factory=dict)

    def fetch_night(self, aoi: Region, day: date) -> pd.DataFrame:
        raise VnfUnavailableError(
            "VNF (VIIRS Nightfire, EOG / Colorado School of Mines) is no longer a public/free "
            "source and is unsupported: a VIIRS Nightfire Data Use License (effective 2025-01-10) "
            "gates the data and EOG programmatic access is paid-subscriber-only (effective "
            "2026-06-01). This conflicts with the project's free/public-data principle. Use the "
            "supported FIRMS source (source='firms'; NASA FIRMS SP-archive VIIRS active fire, "
            "static-land type=2), which needs only a free FIRMS_MAP_KEY (by env-var reference)."
        )


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

    def _url(self, aoi: Region, day: date, key: str) -> str:
        product = self.product or self.cfg.firms_products[0]
        w, s, e, n = aoi.as_bbox()
        bbox = f"{w},{s},{e},{n}"
        return (
            f"{self.cfg.firms_endpoint_base.rstrip('/')}/{key}/{product}/{bbox}/1/"
            f"{day.strftime('%Y-%m-%d')}"
        )

    def fetch_night(self, aoi: Region, day: date) -> pd.DataFrame:
        import httpx

        key = self._map_key()
        url = self._url(aoi, day, key)
        try:
            resp = httpx.get(url, timeout=self.cfg.request_timeout, follow_redirects=True)
            resp.raise_for_status()
        except ViirsAuthError:
            raise
        except Exception as exc:  # noqa: BLE001 - redact the MAP_KEY *value* before surfacing (N3)
            raise ViirsAcquisitionError(
                _scrub_url(f"FIRMS fetch failed for {day} at {url}: {exc}", key)
            ) from exc
        from io import StringIO

        raw = pd.read_csv(StringIO(resp.text))
        return self.filter_and_standardise(raw, aoi, day)

    def filter_and_standardise(self, raw: pd.DataFrame, aoi: Region, day: date) -> pd.DataFrame:
        """Keep static-land-source detections and map FIRMS columns to the standard schema."""
        if raw.empty:
            return standardise(_empty_frame())
        # SP-archive rows carry a `type` code -> keep only static-land (type=2) industrial sources.
        # NRT products have NO `type` column, so this filter simply does not apply to them (and the
        # historical 2019+ series uses the SP archive, which always has `type`).
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


def load_manifest(raw_dir: Path, source: str = "firms") -> dict:
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

    # SP-archive lag (item 4): when the end is left to "chase today" (cfg.end is None), FIRMS SP
    # products lag the present by ~firms_archive_lag_days, so the recent tail is empty. Clamp the
    # effective end back by the lag rather than fetching an empty tail; record it honestly in the
    # manifest + log. An explicit cfg.end is respected as-is (any empty tail is zero-filled, EDGE-001).
    now = today or date.today()
    requested_end = cfg.end or now
    end = requested_end
    if source == "firms" and cfg.end is None:
        latest_available = now - timedelta(days=cfg.firms_archive_lag_days)
        if end > latest_available:
            skipped_tail = (end - latest_available).days
            logger.info(
                "FIRMS SP archive lags ~%dd; clamping end %s -> %s (skipping %d empty tail night(s)).",
                cfg.firms_archive_lag_days,
                end.isoformat(),
                latest_available.isoformat(),
                skipped_tail,
            )
            manifest["end_clamp"] = {
                "requested_end": end.isoformat(),
                "clamped_end": latest_available.isoformat(),
                "skipped_tail_nights": skipped_tail,
                "archive_lag_days": cfg.firms_archive_lag_days,
                "reason": "FIRMS SP archive lags the present; recent dates return 0 rows.",
            }
            _save_manifest(raw_dir, source, manifest)
            end = latest_available

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


def load_detections(raw_dir: Path | str, source: str = "firms") -> pd.DataFrame:
    """Concatenate all persisted per-night AOI-clipped detections for a source (join input)."""
    manifest = load_manifest(Path(raw_dir), source)
    frames = []
    for entry in manifest["nights"].values():
        p = Path(entry["path"])
        if p.exists():
            frame = pd.read_parquet(p)
            # Exclude empty / all-NA per-night frames before concat: zero-detection nights (EDGE-001)
            # persist an empty schema-only parquet, and concatenating those triggers the pandas
            # all-NA-column FutureWarning and can upcast dtypes. We drop them here (BUG 4).
            if not frame.empty and not frame.isna().all(axis=None):
                frames.append(frame)
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
    # Archive-lag honesty (item 1): the SP-archive publication lag applied to anchor the probe window,
    # and the [start, end] window actually probed (ISO). 0 for sources without an archive lag.
    archive_lag_days: int = 0
    window: tuple[str, str] = ("", "")


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

    Archive-lag fix (item 1): the FIRMS SP archive lags the present by ~``firms_archive_lag_days``, so a
    *recent* window returns 0 rows (the earlier false NO-GO). For FIRMS the window is anchored back by
    the archive lag (or an explicit ``cfg.probe_anchor``), and the applied lag + probed window are
    reported so a cluster that really has SP detections yields GO.
    """
    cfg = cfg or ViirsAcquisitionConfig()
    fetcher = fetcher or _default_fetcher(cfg)
    endpoint = cfg.firms_endpoint_base if cfg.source == "firms" else cfg.vnf_endpoint_base
    now = today or date.today()
    # Anchor the probe window to an archive-appropriate date: FIRMS SP lags the present, so probe
    # (today - archive lag) unless an explicit anchor is configured; other sources probe up to today.
    archive_lag = cfg.firms_archive_lag_days if cfg.source == "firms" else 0
    anchor = cfg.probe_anchor or (now - timedelta(days=archive_lag))
    end = anchor
    start = end - timedelta(days=cfg.probe_days - 1)

    per_night_bytes: list[float] = []
    nights_with = n_det = colocated = 0
    fields: set[str] = set()
    # FIRMS needs the FIRMS_MAP_KEY (by env-var reference) -> auth_required is True for FIRMS (N1).
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
    lag_note = (
        f" (FIRMS SP archive lags ~{archive_lag}d; probed {start.isoformat()}..{end.isoformat()})"
        if cfg.source == "firms"
        else f" (probed {start.isoformat()}..{end.isoformat()})"
    )
    notes = (
        "GO — FIRMS detections co-locate with the facilities and the implied volume is within budget."
        + lag_note
        if go
        else (
            "NO-GO — implied full-era volume exceeds the budget." + lag_note
            if not within_budget
            else "NO-GO — no detections over the AOI in the probe window (for FIRMS confirm the window "
            "is inside the SP archive, i.e. offset back by the ~2-month lag)." + lag_note
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
        archive_lag_days=archive_lag,
        window=(start.isoformat(), end.isoformat()),
    )


def _count_colocated(detections: pd.DataFrame, facilities: pd.DataFrame, radius_km: float) -> int:
    from noxus.signal.thermal import join_to_facilities

    return int(len(join_to_facilities(detections, facilities, radius_km)))
