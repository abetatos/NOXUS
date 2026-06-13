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
  config/              # run configuration, study-region definitions (e.g. Tangshan bbox)
  data/                # public TROPOMI NO2 ingestion (Sentinel-5P) + benchmark series
  attribution/         # source attribution of the NO2 column to the cluster
  signal/              # construction of the activity index from attributed NO2
  validation/          # lead/lag tests against the physical-output benchmark
  cli/                 # command-line entry point (noxus.cli.main:main)
docs/                  # design notes and preprint motivation
tests/                 # pytest unit tests
specs/                 # SDD specs, one folder per feature-slug
decisions/             # SDD decision logs + onboarding answers
scripts/               # SDD validation / run helpers (bash)
data/raw/              # (gitignored) large raw EO downloads — never commit
data/derived/          # derived parquet series — meant to be committed
```

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
- Core deps: numpy, pandas, xarray, pyarrow, httpx, pyyaml, python-dotenv, rich.
- Optional `geo` extra (heavier EO backends): netCDF4, rasterio, earthengine-api.
- Data sources are public/free: Sentinel-5P/TROPOMI via Copernicus Data Space Ecosystem
  (and a GEE mirror under the `geo` extra).

## Important documentation

| Document | Path |
|---|---|
| Project README (motivation + usage) | `README.md` |
| Preprint-oriented motivation | `docs/motivation.md` |
| SDD onboarding answers | `decisions/answers.md` |

## Protected areas

Files or directories that require explicit approval before changes:

- `.env` and any secret/credential files (CDSE / GEE). Never read, write, or log secret
  values; reference variable names only. `.env.example` documents the variable names.

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
