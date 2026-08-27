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
| NASA FIRMS active fire | independent physical thermal benchmark (primary) | FIRMS area/CSV API (free MAP_KEY) | `FIRMS_MAP_KEY` |
| VIIRS Nightfire (VNF) | **dropped** — no longer public (licence/pay-gated, see §5) | — (retained only to raise a clear error) | — |
| Tangshan/Hebei curtailment calendar | ground-truth production-event calendar | public State Council holiday notices + MEE / Hebei DEE heating-season notices | none |

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

## 5. VIIRS thermal anomalies — NASA FIRMS primary (VNF dropped) (NOX-011)

The VIIRS thermal channel was intended as an **independent physical benchmark**: blast furnaces, sinter
plants and coke ovens are high-temperature combustion sources that VIIRS detects from orbit, over the
catalogued Tangshan steel facilities (`data/derived/tangshan_steel_facilities.csv`), to **triangulate**
the NO₂ signal against a hard-to-falsify heat measurement. The source is public/free, acquired behind a
fetcher seam, AOI-clipped **client-side** (each per-night product is global, so NOXUS filters to the AOI
bbox after download), written per night with a provenance sidecar under `data/raw/viirs/` (gitignored,
like the NO₂ raw store), and resumed via a manifest.

- **Primary — NASA FIRMS active fire** (VIIRS SNPP, SP-archive). Free with a `MAP_KEY`. Detections are
  filtered to the **static-land-source** type (`type == 2`) to exclude vegetation fires, volcanoes and
  offshore detections. The **SP (standard-processing) archive lags ~2 months** behind real time, so a
  probe/fetch must target an archive window, not the last few nights (an early recent-window probe was a
  false-negative for exactly this reason).
- **Dropped — VIIRS Nightfire (VNF)**, from the Earth Observation Group (EOG, Colorado School of Mines).
  VNF is industrial-combustion-tuned (temperature, source area, radiant heat) and *would* have been the
  ideal series, but it is **no longer public**: a VIIRS Nightfire Data Use License (effective 2025-01-10)
  plus paid-subscriber-only EOG programmatic access (effective 2026-06-01) conflict with the free/public
  -data principle. It was dropped in the 2026-08-22 pivot; `--source vnf` now raises a clear
  licence/pay-gated error (`VnfUnavailableError`) that names the licence and points to FIRMS, rather than
  hitting the EOG host.

### Credentials (by reference only)

- **FIRMS** — request a free `MAP_KEY` at <https://firms.modaps.eosdis.nasa.gov/api/map_key/> and set
  `FIRMS_MAP_KEY` in `.env`. NOXUS references this variable **by name only** and never logs or commits
  the value; the key path segment is scrubbed from any error/URL before it surfaces.

### Commands

```bash
uv run noxus fetch-viirs --probe                # feasibility spike: per-night volume + fields + go/no-go
uv run noxus fetch-viirs                         # resumable bulk FIRMS acquisition → data/raw/viirs/
uv run noxus build-thermal-series                # facility-radius join → benchmark-blind thermal series
uv run noxus verify-viirs                        # detections + facilities overlay (co-location PNG)
uv run noxus triangulate-thermal                 # NO2↔VIIRS + CREA↔VIIRS robust significance
```

### Result — FIRMS active-fire is not a valid steel thermometer here (negative result)

The full FIRMS SP-archive was acquired over the AOI (2019-01-01 → 2026-06-12, 2720 nights, 90,895
detections) and the triangulation run (weekly, n=373 common weeks, NOX-008 autocorrelation-robust
significance): **NO₂↔VIIRS r=−0.054** and **CREA↔VIIRS r=−0.049**, neither significant. The proxy tracks
**neither** series, so it cannot arbitrate feature-vs-target (diagnosis `DIAG_PROXY_INVALID`; this is
**not** a "feature weak" claim). **Root cause:** FIRMS active-fire over this AOI is dominated by
**seasonal agricultural crop-residue burning** (North China Plain / Hebei), not steel furnaces — the
detection climatology peaks spring–autumn and is ≈0 in winter (Jan 0.00 · May 0.87 · Sep 1.00 · Dec
0.09), low FRP (~1.4 MW median), and the same summer-peaked shape holds even for detections **<2 km from
the facilities** (42% of all). Blast furnaces run year-round and are winter-curtailed (opposite phase),
so VNP14 active-fire cannot see them; the sensor that could (VNF Nightfire) is the licence/pay-gated one
above. So critique #1 (target-vs-feature ambiguity) remains **unresolved by public VIIRS**. Full write-up
in `docs/exploration-results.html` §11 and `decisions/architecture-decisions.md` (2026-08-27).

---

## Market prices — yfinance (NOX-004 catalyst)

The event catalyst studies steel-exposed market instruments around NO₂ production events. Daily prices
come from **free public sources via `yfinance`** (Yahoo Finance), written to a dated snapshot under
`data/raw/market/` (gitignored; analysis reads only the snapshot, reproducible per NFR-001):

- **Default instruments** (free, reproducible): global miners `BHP`, `RIO`, `VALE` + a steel ETF `SLX`,
  with `ACWI` as the broad-market benchmark for abnormal returns.
- **Chinese ferrous futures** (SHFE rebar / DCE iron ore / coking coal) — the closest instrument to
  Tangshan output, but **no clean free API** (Yahoo does not carry them reliably). Open question Q1:
  v1 leads with the free instruments above; ferrous can be added as a best-effort exchange daily-
  settlement snapshot. **Never add a paywalled data dependency** without explicit developer approval.

Run: `uv run noxus ingest-market [--start --end]`. No credentials needed (public endpoint).

---

## Tangshan/Hebei curtailment calendar — public holiday + heating-season notices (NOX-012)

The regime/event study (`noxus regime-study`) matches NO₂ production events against a ground-truth
**production-event calendar** of periods when Tangshan/Hebei steel output is administratively
constrained. The calendar is assembled from **public, free, citable** sources only — no credentials,
no paid feed — and committed at `data/curtailment/tangshan_hebei_curtailment.csv` (21 rows) with
per-event provenance in `data/curtailment/SOURCES.md`. Assemble/validate it with:

```bash
uv run noxus assemble-calendar --probe     # T1 feasibility spike: coverage + provenance + go/no-go
uv run noxus assemble-calendar             # build/validate the committed calendar
```

### Coarse-floor vs event-level — and the pending live survey

The calendar is currently a **coarse structural floor**, not an event-level campaign log. Two
granularities are distinguished:

- **Coarse floor (committed).** Securely-datable *structural* windows that recur every year and are
  each backed by a public source: Spring Festival and National-Day Golden Week holidays (PRC State
  Council public-holiday arrangements) and the winter heating-season industrial-restriction windows
  (~Nov 15 → Mar 15, MEE / Hebei DEE heating-season notices). Every one of the 21 committed rows
  carries a source; unsourced candidates are dropped by the loader/validator.
- **Event-level campaign calendar (PENDING a live run).** A richer log of *specific* dated
  curtailment campaigns (e.g. air-quality-driven or event-driven production limits) requires a
  **live MEE / Hebei DEE / Tangshan / CREA / Mysteel / news survey** that was not performed in this
  offline environment. The T1 go/no-go verdict is therefore **`GO-coarse-floor`** with
  `live_survey_pending=True` (the NOX-007/NOX-011 pending-live-run pattern).

No event-level curtailment dates were fabricated: specific campaign dates that could not be sourced
offline are recorded as a **documented gap** in `SOURCES.md`, not invented events. Re-running the
probe with a live survey is what could upgrade the verdict to `GO-event-level`.

**Status:** the calendar-assembly and regime/event-study machinery has landed and been tested
offline, and the real-data `regime-study` run (on the 2023 rolling-correlation regime / 2024 breakdown)
has now **completed** on the full-series NOX-003.1 intensity residual. Verdict: the 2023 peak (r=0.276 at
2023-08-27) is **consistent with a rolling-window multiple-comparisons artefact** under the scan-maximum
null (p=0.325) — a valid, first-class finding; the coupling present is **continuous, not event-bound**
(jackknife), and the 2024 breakdown is **intensity-decoupling**, not a curtailment-regime change. Full
write-up in `docs/exploration-results.html` §10. The coarse-floor calendar was the event ground truth; the
richer **event-level** campaign survey (above) remains the one outstanding data gap.

---

## Reproducibility & secrets

- `.env` is gitignored and **protected** — never read, write, or log secret values; reference variable
  names only (the names live in `.env.example`).
- Large raw downloads (`data/raw/`, `*.nc`, `*.tif`, the GEM zip) are gitignored and never committed;
  derived series under `data/derived/` are committed.
- Access tokens are short-lived and held in memory only — never written to disk or logs.
