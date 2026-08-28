"""Raw Sentinel-5P/TROPOMI **L2 __NO2___ swath** acquisition via the FREE CDSE OData route (NOX-005).

NOX-002a delivered the *gridded* openEO product; the flux-divergence method (Beirle) needs the raw L2
swath with **native footprints, geolocation and QA** for fine oversampling, which openEO does not
expose. This module is that new, distinct acquisition path.

Q1 (open question) is resolved **live at the T1 gate**: the CDSE OData catalogue is queried for
``S5P_*_L2__NO2___`` granules intersecting the Tangshan AOI over a *bounded* window; a CDSE access
token is minted via the OIDC password grant (reusing the NOX-002a ``CDSE_USERNAME``/``CDSE_PASSWORD``
convention, by env-var **name** only — secrets are never read into logs/output); and the granule is
streamed from the CDSE ``zipper`` endpoint into a gitignored, resumable store.

Scope discipline (T1): only the *bounded* fetch the feasibility probe needs is implemented here. The
full multi-year fetch, the S3 ``eodata`` throughput alternative, and the ``fetch-l2`` CLI are T2. Full
L2 granules are large (~0.6 GB each, full orbit); the AOI window is clipped **after** download, so the
kept field is small but the *download* is per-granule heavy — a compute-profile fact the T1 probe
measures and reports (NFR-008).

**Free-only (NFR-001, ERR-005).** Every request host is checked against the CDSE free route; any other
(paid/licence-gated) host is refused. This is the NOX-011 VNF lesson encoded.

The network interaction sits behind the :class:`L2Fetcher` seam so the search/download/read logic is
unit-testable without credentials or network (tests inject a fake fetcher / a synthetic granule).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

import numpy as np

from noxus.config.region import Region
from noxus.config.run import FluxDivergenceConfig

# S5P L2 NO2 native product layout (verified against a real OFFL granule, 2026-08-27):
# group "PRODUCT" holds nitrogendioxide_tropospheric_column (mol m-2), qa_value, latitude, longitude,
# time_utc — all dimensioned (time=1, scanline, ground_pixel).
PRODUCT_GROUP = "PRODUCT"
NO2_VAR = "nitrogendioxide_tropospheric_column"
QA_VAR = "qa_value"


class L2StoreError(RuntimeError):
    """The raw L2 swath store is missing/empty (ERR-001)."""


class L2AccessError(RuntimeError):
    """The FREE CDSE raw-L2 route could not be reached/authenticated (→ NO-GO-DATA, not a fake GO)."""


class PaidSourceError(RuntimeError):
    """A non-free (paid/licence-gated) source was requested for a raw-L2 input (ERR-005)."""


@dataclass(frozen=True)
class GranuleMeta:
    """One CDSE OData catalogue hit for an AOI-intersecting L2 NO2 granule."""

    id: str
    name: str
    size_bytes: int
    content_date_start: str


@dataclass(frozen=True)
class L2Field:
    """A QA-screened, AOI-clipped bundle of native L2 footprints from one granule."""

    lat: np.ndarray
    lon: np.ndarray
    no2: np.ndarray  # tropospheric NO2 vertical column density, mol m-2
    overpass_utc: datetime
    granule_name: str


@dataclass(frozen=True)
class FetchResult:
    """Outcome of a bounded raw-L2 fetch (what was found, downloaded, and reused)."""

    granules: tuple[GranuleMeta, ...]
    paths: tuple[Path, ...]
    downloaded_bytes: int
    reused: int
    endpoint: str
    n_found: int  # total AOI-intersecting granules in the window (before the bounded limit)
    window_days: int


# --------------------------------------------------------------------------------- free-source guard


def _assert_free_source(url: str, cfg: FluxDivergenceConfig) -> None:
    """Refuse any host that is not part of the FREE CDSE route (ERR-005, NFR-001)."""
    host = (urlsplit(url).hostname or "").lower()
    if host not in cfg.free_hosts:
        raise PaidSourceError(
            f"Refusing a non-free raw-L2 source host {host!r}. The raw L2 __NO2___ swath must come "
            f"from the free CDSE route only ({', '.join(cfg.free_hosts)}); paid/licence-gated "
            "sources are not permitted (ERR-005, NFR-001)."
        )


# ---------------------------------------------------------------------------------- fetcher seam


class L2Fetcher(Protocol):
    """Search + download seam so the module is testable without credentials or network."""

    def search(self, aoi: Region, start: date, end: date) -> list[GranuleMeta]: ...

    def download(self, meta: GranuleMeta, dest: Path) -> int: ...


def _aoi_polygon(aoi: Region) -> str:
    """A closed WGS84 polygon ring for the OData ``OData.CSC.Intersects`` spatial filter."""
    x0, y0, x1, y1 = aoi.as_bbox()
    return f"POLYGON(({x0} {y0},{x1} {y0},{x1} {y1},{x0} {y1},{x0} {y0}))"


@dataclass
class CDSEFetcher:
    """Real :class:`L2Fetcher`: FREE CDSE OData catalogue search + password-grant token + download.

    Secrets are referenced by env-var name only (``cfg.env_username``/``cfg.env_password``); the token
    and credentials are never logged. Every request host is checked against the free route.
    """

    cfg: FluxDivergenceConfig
    _token: str | None = field(default=None, repr=False)

    def _mint_token(self) -> str:
        from dotenv import load_dotenv

        load_dotenv()
        user = os.environ.get(self.cfg.env_username)
        password = os.environ.get(self.cfg.env_password)
        if not (user and password):
            raise L2AccessError(
                f"CDSE credentials not set (${self.cfg.env_username}/${self.cfg.env_password}); "
                "cannot reach the free raw-L2 route. Set them in .env (CDSE account)."
            )
        import httpx

        _assert_free_source(self.cfg.token_url, self.cfg)
        try:
            resp = httpx.post(
                self.cfg.token_url,
                data={
                    "username": user,
                    "password": password,
                    "grant_type": "password",
                    "client_id": self.cfg.cdse_client_id,
                },
                timeout=60.0,
            )
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001 - surface a clean auth error, never the secret
            raise L2AccessError(f"CDSE token mint failed (free route): {exc}") from exc
        token = resp.json().get("access_token")
        if not token:
            raise L2AccessError("CDSE token endpoint returned no access_token (free route).")
        return token

    def search(self, aoi: Region, start: date, end: date) -> list[GranuleMeta]:
        import httpx

        _assert_free_source(self.cfg.odata_url, self.cfg)
        flt = (
            "Collection/Name eq 'SENTINEL-5P' and contains(Name,'L2__NO2') "
            "and contains(Name,'OFFL') "
            f"and OData.CSC.Intersects(area=geography'SRID=4326;{_aoi_polygon(aoi)}') "
            f"and ContentDate/Start gt {start.isoformat()}T00:00:00.000Z "
            f"and ContentDate/Start lt {end.isoformat()}T00:00:00.000Z"
        )
        params = {"$filter": flt, "$orderby": "ContentDate/Start asc", "$top": "50"}
        try:
            resp = httpx.get(self.cfg.odata_url, params=params, timeout=60.0)
            resp.raise_for_status()
            values = resp.json().get("value", [])
        except Exception as exc:  # noqa: BLE001
            raise L2AccessError(f"CDSE OData catalogue search failed (free route): {exc}") from exc
        return [
            GranuleMeta(
                id=v["Id"],
                name=v["Name"],
                size_bytes=int(v.get("ContentLength") or 0),
                content_date_start=str(v.get("ContentDate", {}).get("Start", "")),
            )
            for v in values
        ]

    def download(self, meta: GranuleMeta, dest: Path) -> int:
        import httpx

        url = f"{self.cfg.zipper_url}({meta.id})/$value"
        _assert_free_source(url, self.cfg)
        if self._token is None:
            self._token = self._mint_token()
        dest.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        try:
            with httpx.stream(
                "GET",
                url,
                headers={"Authorization": f"Bearer {self._token}"},
                timeout=self.cfg.request_timeout,
                follow_redirects=True,
            ) as resp:
                resp.raise_for_status()
                with open(dest, "wb") as fh:
                    for chunk in resp.iter_bytes(chunk_size=1 << 20):
                        fh.write(chunk)
                        n += len(chunk)
        except Exception as exc:  # noqa: BLE001 - clean error, never the token
            raise L2AccessError(f"CDSE granule download failed (free route): {exc}") from exc
        return n


# ---------------------------------------------------------------------------------- persistence


def _manifest_path(raw_dir: Path) -> Path:
    return Path(raw_dir) / "manifest.json"


def load_manifest(raw_dir: Path) -> dict:
    path = _manifest_path(raw_dir)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"granules": {}}


def _save_manifest(raw_dir: Path, manifest: dict) -> None:
    _manifest_path(raw_dir).write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )


# ---------------------------------------------------------------------------------- bounded fetch


def _one_per_day(granules: list[GranuleMeta]) -> list[GranuleMeta]:
    """Keep the first granule per calendar day (varied daily winds cancel advection dipoles)."""
    seen: set[str] = set()
    out: list[GranuleMeta] = []
    for g in granules:
        day = g.content_date_start[:10] or g.name.split("L2__NO2____")[-1][:8]
        if day not in seen:
            seen.add(day)
            out.append(g)
    return out


def fetch_l2(
    aoi: Region,
    cfg: FluxDivergenceConfig | None = None,
    fetcher: L2Fetcher | None = None,
    *,
    max_granules: int | None = None,
    one_per_day: bool | None = None,
) -> FetchResult:
    """Bounded raw-L2 fetch: search the free CDSE catalogue over ``[subset_start, subset_end)`` and
    download up to ``max_granules`` full granules into the gitignored store, resumably.

    A granule already present with its catalogue ContentLength is reused (resume), not re-downloaded.
    Downloading stops once ``cfg.download_budget_gb`` of NEW transfer would be exceeded (reused
    granules are free) — a hard laptop-feasible cap. With ``one_per_day`` the selection is spread one
    overpass per calendar day so the oversample sees varied wind directions.

    Raises :class:`L2AccessError` if the free route cannot be reached/authenticated (→ NO-GO-DATA at
    the gate) and :class:`PaidSourceError` if a non-free host is ever requested (ERR-005).
    """
    cfg = cfg or FluxDivergenceConfig()
    fetcher = fetcher or CDSEFetcher(cfg)
    limit = cfg.max_probe_granules if max_granules is None else max_granules
    per_day = cfg.one_per_day if one_per_day is None else one_per_day
    raw_dir = Path(cfg.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(raw_dir)

    found = fetcher.search(aoi, cfg.subset_start, cfg.subset_end)
    if not found:
        raise L2AccessError(
            "No L2 __NO2___ granules intersect the AOI in the bounded window "
            f"{cfg.subset_start}..{cfg.subset_end} on the free CDSE route."
        )

    ranked = _one_per_day(found) if per_day else found
    selected = ranked[: max(1, limit)]
    budget_bytes = cfg.download_budget_gb * 1e9
    paths: list[Path] = []
    kept: list[GranuleMeta] = []
    downloaded = reused = 0
    endpoint = cfg.zipper_url if isinstance(fetcher, CDSEFetcher) else "unknown"
    for meta in selected:
        dest = raw_dir / meta.name
        if dest.exists() and (meta.size_bytes == 0 or dest.stat().st_size == meta.size_bytes):
            reused += 1
        elif downloaded + (meta.size_bytes or 0) > budget_bytes:
            break  # hard download-budget cap reached; stop before exceeding it
        else:
            downloaded += fetcher.download(meta, dest)
        paths.append(dest)
        kept.append(meta)
        manifest["granules"][meta.name] = {
            "id": meta.id,
            "size_bytes": meta.size_bytes,
            "content_date_start": meta.content_date_start,
            "path": str(dest),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
    _save_manifest(raw_dir, manifest)
    return FetchResult(
        granules=tuple(kept),
        paths=tuple(paths),
        downloaded_bytes=downloaded,
        reused=reused,
        endpoint=endpoint,
        n_found=len(found),
        window_days=max(1, (cfg.subset_end - cfg.subset_start).days),
    )


# ------------------------------------------------------------------------------- native read path


def read_l2_field(path: Path, aoi: Region, qa_threshold: float) -> L2Field:
    """Read one L2 granule's native footprints, QA-screen, and clip to the AOI (Q1 read path).

    Returns flat arrays of ``lat``/``lon``/``no2`` (mol m-2) for QA-valid pixels inside the AOI plus
    the granule overpass timestamp. Never interpolates or fabricates pixels.
    """
    import xarray as xr

    path = Path(path)
    if not path.exists():
        raise L2StoreError(
            f"Raw L2 granule not found: {path}. Fetch the bounded subset first (fetch_l2 / the T1 "
            "probe) — nothing is fabricated (ERR-001)."
        )
    ds = xr.open_dataset(path, group=PRODUCT_GROUP)
    try:
        lat = np.asarray(ds["latitude"].values).reshape(-1)
        lon = np.asarray(ds["longitude"].values).reshape(-1)
        no2 = np.asarray(ds[NO2_VAR].values).reshape(-1)
        qa = np.asarray(ds[QA_VAR].values).reshape(-1)
        pixel_time = _pixel_time_utc(ds)  # per-pixel time_utc (broadcast over ground_pixel), or None
        fallback_overpass = _overpass_time(ds)
    finally:
        ds.close()

    x0, y0, x1, y1 = aoi.as_bbox()
    keep = (
        (lon >= x0)
        & (lon <= x1)
        & (lat >= y0)
        & (lat <= y1)
        & (qa >= qa_threshold)
        & np.isfinite(no2)
    )
    # Overpass time OVER THE AOI (not the whole global orbit segment) → accurate per-overpass wind.
    overpass = fallback_overpass
    if pixel_time is not None and keep.any():
        import pandas as pd

        aoi_times = pd.to_datetime(pixel_time[keep], utc=True, errors="coerce").dropna()
        if len(aoi_times):
            mid = aoi_times[len(aoi_times) // 2]
            overpass = mid.tz_convert("UTC").tz_localize(None).to_pydatetime()
    return L2Field(
        lat=lat[keep],
        lon=lon[keep],
        no2=no2[keep],
        overpass_utc=overpass,
        granule_name=path.name,
    )


def _pixel_time_utc(ds) -> np.ndarray | None:
    """Per-pixel ``time_utc`` broadcast to the (scanline, ground_pixel) grid, flattened; else None.

    ``time_utc`` is stored per scanline; broadcasting to ground pixels lets the AOI-clip mask select
    the timestamps of exactly the kept pixels, so the overpass time reflects the AOI, not the orbit.
    """
    if "time_utc" not in ds.variables:
        return None
    tu = ds["time_utc"].values  # (time=1, scanline)
    arr = np.asarray(tu).reshape(tu.shape[-1])  # (scanline,)
    ground = ds.sizes.get("ground_pixel")
    if ground is None:
        return arr
    return np.repeat(arr[:, None], ground, axis=1).reshape(-1)


def _overpass_time(ds) -> datetime:
    """Median ``time_utc`` of the granule as a naive-UTC ``datetime`` (best-effort, defensive)."""
    import pandas as pd

    if "time_utc" in ds.variables:
        vals = pd.to_datetime(np.asarray(ds["time_utc"].values).reshape(-1), utc=True, errors="coerce")
        vals = vals[~pd.isna(vals)]
        if len(vals):
            mid = vals[len(vals) // 2]
            return mid.tz_convert("UTC").tz_localize(None).to_pydatetime()
    # Fall back to the acquisition start encoded in the filename: S5P_..._L2__NO2____YYYYMMDDTHHMMSS_
    parts = str(ds.encoding.get("source", "")).split("_")
    for token in parts:
        if len(token) == 15 and token[8] == "T":
            try:
                return datetime.strptime(token, "%Y%m%dT%H%M%S")
            except ValueError:
                pass
    return datetime(1970, 1, 1)
