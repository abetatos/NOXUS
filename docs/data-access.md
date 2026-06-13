# Data access — sources, credentials, and how NOXUS fetches each

All NOXUS inputs are **public and free**; some require a (free) registration. Credentials live in
`.env` (gitignored, never committed). Copy `.env.example` → `.env` and fill in the values. This doc
references **variable names only** — never put secret values in the repo, in logs, or in commits.

| Source | What | Access | Credentials (`.env`) |
|---|---|---|---|
| Sentinel-5P / TROPOMI NO₂ | the predictor substrate | CDSE API (free account) or GEE mirror | `CDSE_*` or `GEE_PROJECT` |
| ERA5 reanalysis (wind + BLH) | meteorology for regress-out | Copernicus CDS API (free account + licence) | `CDSAPI_KEY` (+ optional `CDSAPI_URL`) |
| CREA steel CSV | physical-output benchmark | public Google-Sheet CSV | none |
| GEM Global Iron & Steel Tracker | facility locations (AOI) | free download (form) | none |

---

## 1. Sentinel-5P / TROPOMI NO₂ — Copernicus Data Space Ecosystem (CDSE)

### Register (free)

1. Go to <https://dataspace.copernicus.eu/> → avatar (top-right) → **Register**, fill the form, accept terms.
2. That account is all you need for the **password grant** (Option A below). No paid tier for our usage
   (free quotas apply).

### How NOXUS authenticates (Option A — password grant)

NOXUS reads the account login from `.env` (`CDSE_USERNAME`, `CDSE_PASSWORD`, `CDSE_CLIENT_ID=cdse-public`)
via `python-dotenv` and mints a **short-lived access token** from the CDSE OpenID token endpoint. The
raw call (values come from environment variables — never hard-code them):

```bash
curl -s -X POST \
  https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=$CDSE_USERNAME" \
  -d "password=$CDSE_PASSWORD" \
  -d "grant_type=password" \
  -d "client_id=$CDSE_CLIENT_ID"     # cdse-public
```

The response contains an `access_token` (valid ~10 min) and a `refresh_token`. Use the access token as
`Authorization: Bearer <token>` against the OData catalogue
(`https://catalogue.dataspace.copernicus.eu/odata/v1/...`) to search and download the
`S5P_*_L2__NO2___` products, then clip to the study AOI. In Python, the same flow uses `httpx` (already
a dependency) to POST the form and reuse the token; credentials are loaded with `python-dotenv` and are
never logged.

> Option B (machine-to-machine) — create a dedicated OAuth client in the CDSE dashboard and set
> `CDSE_CLIENT_SECRET`. Only needed if you prefer client-credentials over the account login.

### Alternatives

- **S3** — generate an S3 key/secret on the CDSE *s3-credentials* page (secret shown once); endpoint
  `https://eodata.dataspace.copernicus.eu/`. Good for high-throughput parallel pulls.
- **Google Earth Engine mirror** — set `GEE_PROJECT` and use the `geo` extra (`earthengine-api`). The
  GEE S5P collection is already gridded/clipped server-side; convenient for prototyping, less control
  over the QA/oversampling than raw L2.

See the CDSE docs: [Token](https://documentation.dataspace.copernicus.eu/APIs/Token.html) ·
[S3](https://documentation.dataspace.copernicus.eu/APIs/S3.html) ·
[Registration](https://documentation.dataspace.copernicus.eu/Registration.html).

### How NOXUS acquires NO₂ (NOX-002a)

`noxus fetch` uses **openEO server-side subsetting**: it asks CDSE for only the AOI window per overpass
(avoiding ~2.2 TB of full granules → ~0.2–1 GB kept). Auth is automatic from `.env`: it mints a CDSE
token via the password grant (`CDSE_USERNAME`/`CDSE_PASSWORD`, `client_id=cdse-public`) and hands it to
openEO via `authenticate_oidc_access_token()`; if that fails it falls back to `authenticate_oidc()`
(refresh token, else an interactive device flow that opens the web). It applies `qa_value ≥ 0.75`,
keeps NO₂/cloud (+coverage, processor version) per overpass under `data/raw/tropomi/`, and is
**resumable** via a manifest (batched monthly to respect the 10k credits/month).

```bash
uv run noxus fetch --start 2023-06-01 --end 2023-06-30      # small range first
uv run noxus verify-no2 --days 5 --max-cloud 0.2 --optical  # NO₂ over AOI + facilities (+ S2 thumb)
uv run noxus fetch --start 2019-01-01                        # full record (resumable; spans credit cycles)
```

`verify-no2` is the location/sanity check: on clear-sky high-NO₂ days it renders the NO₂ field over the
AOI with the steel facilities overlaid (PNGs under `data/derived/verification/`), optionally with a
Sentinel-2 optical thumbnail to confirm a plant is physically there.

## 2. ERA5 reanalysis (meteorology) — Copernicus Climate Data Store (CDS)

Meteorology — 10 m wind components (`u10`, `v10`) and boundary-layer height (`blh`) — is the dominant
week-to-week confounder of the NO₂ column: ventilation, not activity, drives much of the variance. NOXUS
ingests ERA5 over the AOI and regresses it out of the footprint signal in the index stage. Note the CDS is
a **separate service from the CDSE** used for TROPOMI — it needs its own account and credential.

### Register (free) and accept the licence

1. Create a free account at <https://cds.climate.copernicus.eu/> (the Climate Data Store; distinct from the
   Data Space Ecosystem in §1).
2. Open the **ERA5 hourly data on single levels** dataset page and **accept its licence** under the
   *Download* / *Terms of use* tab — this is a **one-time** step. Without it the API returns a
   `403 "required licences not accepted"`.
3. From your CDS profile, copy your API key.

### How NOXUS authenticates

`noxus ingest-era5` uses `cdsapi`. The credential is read **by reference only** from `.env`
(`CDSAPI_KEY`, plus optional `CDSAPI_URL`, default `https://cds.climate.copernicus.eu/api`) via
`python-dotenv`; if `CDSAPI_KEY` is unset it falls back to a `~/.cdsapirc` file. The key value is never
read into logs, printed, or committed (protected-area policy). `.env.example` documents the variable names.

### How NOXUS acquires ERA5 (NOX-003)

`noxus ingest-era5` requests an **AOI / era / overpass-hour subset of `reanalysis-era5-single-levels`**,
subsetted **server-side** by the CDS, so only a few MB cross the wire for the whole 2019→present record. It
fetches **one request per year** (a single multi-year request exceeds the CDS per-request cost limit),
concatenates and de-duplicates the parts, and writes a dated NetCDF snapshot to
`data/raw/era5/era5_<YYYY-MM-DD>.nc` (gitignored). Analysis then reads only the snapshot — no live fetch at
analysis time.

```bash
uv run noxus ingest-era5          # fetch the AOI/era ERA5 subset → dated .nc snapshot (no flags)
uv run noxus index                # meteo regress-out + deseason + relative activity index
uv run noxus index --no-meteo     # skip the ERA5 regress-out (no snapshot needed)
```

> The anonymous **ARCO-ERA5 Zarr** store was considered and rejected: its chunk-1 layout downloads a full
> global field per timestep (~30–760 GB for a small AOI over a long era — the wrong access pattern here).
> Decision recorded in `decisions/architecture-decisions.md` (2026-06-13).

## 3. CREA benchmark (steel operating rates)

Public CSV export of a CREA-maintained Google Sheet (provenance: WIND / China United Steel Network,
republished by CREA). No credentials. Ingested by `noxus ingest-benchmark`, which pins a dated snapshot
under `data/raw/benchmark/`. See the README "Data sources" and `noxus/data/benchmark.py`.

## 4. GEM Global Iron & Steel Tracker (facility locations)

Free download (behind a short form) from
<https://globalenergymonitor.org/projects/global-iron-and-steel-tracker/> → **Download data**. Place the
zip under `data/raw/gem/` (gitignored). It contains three workbooks: plant-level, iron-unit, steel-unit.

NOXUS derives the study facilities from the **plant-level** workbook (`Plant data` sheet): filter
`Country/area == China` and `Municipality` in the Tangshan-prefecture set, keep plants whose
`Main production equipment` contains `BF` (integrated blast-furnace plants), join the authoritative
`Status` from the `Plant capacities and status` sheet, and parse `Coordinates` → lat/lon. The result is
committed at `data/derived/tangshan_steel_facilities.csv` and defines the area of interest (AOI).

---

## Reproducibility & secrets

- `.env` is gitignored and **protected** — never read, write, or log secret values; reference variable
  names only (the names live in `.env.example`).
- Large raw downloads (`data/raw/`, `*.nc`, `*.tif`, the GEM zip) are gitignored and never committed;
  derived series under `data/derived/` are committed.
- Access tokens are short-lived and held in memory only — never written to disk or logs.
