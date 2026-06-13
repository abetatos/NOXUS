# NOXUS — project map

Orientation artifact for Claude Code and developers: where things live, how
the project runs, and what must not be touched without approval.

Rules for this file:

- Keep it concise — aim for one or two screens. It exists to avoid loading
  many files into context, not to replace them (see the context-economy
  policy).
- Keep the tree shallow (2–3 levels) and annotated; omit generated and
  vendored directories. Do not list every file.
- Never record secrets, credentials, or tokens here.
- **Maintenance rule:** update this map when the structure changes
  significantly — new top-level directories, moved entrypoints, renamed
  build/test commands, or new protected areas.

## Directory tree

```text
noxus/                 # Python package
  config/              # region.py: Tangshan AOI derived from steel facilities + buffer; run.py: Benchmark/Acquisition/Gridding/Signal/ValidationConfig
  data/                # benchmark.py + tropomi.py (openEO acquisition) + verify_no2.py + gridding.py + era5.py (CDS ingest) — implemented
  attribution/         # source.py: footprint sampling + regional-background subtraction — implemented
  signal/              # index.py (meteo regress-out + deseason + relative index) + intensity.py (explicit emission-intensity/decoupling model, NOX-003.1) — implemented
  validation/          # preprocess.py + leadlag.py + report.py: align/sign/r·p + lead-lag vs the benchmark — implemented
  cli/                 # command-line entry point; all subcommands implemented (see CLI table)
analysis/              # preliminary_signal.py: reproducible preliminary run → docs/figures/preliminary/
docs/                  # design notes, preprint motivation, data-access.md, preliminary-results.html (+ figures/)
tests/                 # pytest unit tests
specs/                 # SDD specs, one folder per feature-slug
decisions/             # SDD decision logs + onboarding answers
scripts/               # SDD validation / run helpers (bash)
data/raw/              # (gitignored) EO downloads: tropomi/ (NO2 overpasses+manifest), era5/ (.nc snapshots), benchmark/, gem/
data/derived/          # derived parquet series (e.g. benchmark_tangshan_bf_operating_rate) — committed
```

## CLI commands

| Command | Purpose |
|---|---|
| `uv run noxus ingest-benchmark [--from-snapshot CSV] [--out PARQUET]` | Fetch/clean the CREA Tangshan benchmark → dated snapshot + tidy parquet |
| `uv run noxus fetch [--start --end --buffer --qa]` | Acquire TROPOMI NO2 over the AOI via openEO (server-side subset, resumable) → `data/raw/tropomi/` |
| `uv run noxus verify-no2 [--days --max-cloud --optical]` | Render NO2-over-AOI + facility overlay for clear high-NO2 days → `data/derived/verification/` |
| `uv run noxus grid [--freq --min-coverage]` | Composite per-overpass NO2 → weekly cube + interim AOI-mean series → `data/derived/no2/` (gitignored) |
| `uv run noxus ingest-era5` | Fetch the AOI/era ERA5 subset from the Copernicus CDS (per-year, server-side) → dated `data/raw/era5/*.nc` snapshot (gitignored) |
| `uv run noxus attribute [--radius KM]` | Footprint sampling + regional-background subtraction → background-corrected footprint signal |
| `uv run noxus index [--no-meteo]` | ERA5 meteo regress-out + deseason + relative activity index (deseason method is config-only) |
| `uv run noxus validate [--max-lag N]` | Align + sign + r/p + lead-lag vs the benchmark → report (reports the null) |

## Key entrypoints

| Entrypoint | Purpose |
|---|---|
| `noxus/cli/main.py` (`main`) | CLI entry point; exposed as the `noxus` console script |

## Commands

| Action | Command |
|---|---|
| Install / validate environment | `uv sync --extra dev` |
| Run all tests | `uv run pytest` |
| Run targeted tests | `uv run pytest <path>::<test>` |
| Lint | `uv run ruff check .` |
| Format | `uv run ruff format .` |
| Typecheck | none configured (no mypy) |
| Build | `uv build` (hatchling) |

If a command is unknown, keep it as a `TODO: ask the developer` entry — do
not invent one.

## Framework and runtime assumptions

- Language: Python ≥ 3.12.
- Package / environment manager: `uv` (lockfile `uv.lock` committed for reproducibility).
- Build backend: hatchling.
- Test framework: pytest. Lint/format: ruff (line length 100, target py312).
- Core deps: numpy, pandas, xarray, pyarrow, httpx, pyyaml, python-dotenv, rich, openeo (TROPOMI
  acquisition via CDSE), cdsapi (ERA5 via the Copernicus CDS), scipy + statsmodels (signal/validation
  stats), matplotlib (verification render), h5netcdf+h5py (per-overpass NetCDF I/O).
- Optional `geo` extra (heavier EO backends): netCDF4, rasterio, earthengine-api.
- Data sources are public/free: Sentinel-5P/TROPOMI via Copernicus Data Space Ecosystem (and a GEE
  mirror under the `geo` extra); ERA5 reanalysis via the Copernicus Climate Data Store (CDS).

## Important documentation

| Document | Path |
|---|---|
| Project README (motivation + usage) | `README.md` |
| Preprint-oriented motivation | `docs/motivation.md` |
| Data access (sources + credentials) | `docs/data-access.md` |
| Preliminary results writeup (+ figures) | `docs/preliminary-results.html` |
| SDD onboarding answers | `decisions/answers.md` |

## Protected areas

Files or directories that require explicit approval before changes:

- `.env` and any secret/credential files (CDSE / GEE / CDS `CDSAPI_KEY`). Never read, write, or log
  secret values; reference variable names only. `.env.example` documents the variable names.

Handle with care (reproducibility conventions, not hard-blocked):

- `data/raw/` and large EO downloads (`*.nc`, `*.tif`) — gitignored, never commit.
- `uv.lock` — committed on purpose; change only via `uv`.

## SDD locations

| Artifact | Path |
|---|---|
| Task state | `tasks.json` |
| Specs | `specs/<feature-slug>/` |
| History | `history.html` |
| Decisions | `decisions/` |

## Generated files and do-not-edit

| Path | Rule |
|---|---|
| `data/raw/`, `*.nc`, `*.tif` | Reproducible raw downloads — gitignored, do not commit |
| `.venv/`, `*.egg-info/`, caches | Generated — do not edit or commit |
