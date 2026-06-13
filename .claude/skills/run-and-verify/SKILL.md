---
name: run-and-verify
description: Run NOXUS and verify behavior using its real commands. Use when validating an implementation, reproducing a bug, or confirming acceptance criteria before review.
---

# Run and verify — NOXUS

How to run, inspect, and verify this project. Implementation claims
("it works") must be backed by observed behavior from the commands below,
not by assumptions.

Rules for this file:

- Commands here come from repository inspection and developer answers.
  If a command is unknown, keep it as `TODO: ask the developer` — never
  invent one.
- Record environment variables by **name only**. Never write secret
  values, tokens, or credentials in this file.
- Update this file when commands, services, or verification steps change.

## When to use

- Before marking an implementation task ready for review.
- When the reviewer validates an implementation against its spec.
- Reproducing a reported bug before fixing it.
- Confirming acceptance criteria that describe runtime behavior.

## When not to use

- Pure documentation or comment changes with no runtime effect.
- When required services or credentials are unavailable — record a TODO
  and report that verification was not performed, rather than faking it.

## Project commands

| Action | Command |
|---|---|
| Run locally (CLI) | `uv run noxus --help` (subcommands are added as the pipeline is implemented) |
| Run all tests | `uv run pytest` |
| Run targeted tests | `uv run pytest <path>::<test>` |
| Lint | `uv run ruff check .` |
| Format | `uv run ruff format .` |
| Typecheck | none configured (no mypy) |
| Build | `uv build` |

## Required services

NOXUS is a local data pipeline; no databases, queues, or containers are
required to run it. Earth-observation data is fetched from public HTTP APIs
(Copernicus Data Space Ecosystem; optional Google Earth Engine mirror under
the `geo` extra). Network access and valid credentials are needed only for
the ingestion stage that downloads NO₂ data.

## Required environment variables

Names only — values come from the developer's local environment or `.env`
(see `.env.example`), never from this file:

- `CDSE_CLIENT_ID` — Copernicus Data Space Ecosystem client id (Sentinel-5P access).
- `CDSE_CLIENT_SECRET` — Copernicus Data Space Ecosystem client secret.
- `GEE_PROJECT` — optional; Google Earth Engine project id (only with the `geo` extra / GEE mirror).

## How to verify behavior

### UI verification

Not applicable — NOXUS has no browser UI.

### API / CLI verification

- Smoke-check the CLI loads: `uv run noxus --help` exits 0 and lists available commands.
- For a stage that produces a derived series, run its command and confirm the expected
  output artifact appears under `data/derived/` (parquet) and matches the spec's acceptance
  criteria (shape, columns, date range).
- For numerical/scientific changes, assert against fixtures in `tests/` rather than eyeballing;
  add or update a pytest test that encodes the acceptance criterion.
- Network-dependent ingestion: prefer cached/fixture inputs in tests; only hit the live CDSE
  API with credentials present, and never against production with destructive intent (these
  APIs are read-only anyway).

## Procedure

1. Confirm required environment variables are set (by name; do not echo values).
2. Identify the narrowest check that exercises the change (targeted test,
   single CLI invocation) and run it; capture actual output.
3. Run the broader applicable checks the project defines (tests, lint).
4. Verify runtime behavior against the spec's acceptance criteria using
   the CLI verification steps above.
5. Report honestly: what ran, what passed, what failed (with output), and
   what could not be verified and why.

## Output artifact

Verification evidence in the task's review notes (`review.html`) or the
implementation summary: commands run, results, and unverified items.

## Safety constraints

- Never invent commands; unknown commands are `TODO: ask the developer`.
- Do not run destructive operations (data deletion, deploys) without
  explicit approval.
- Never store secrets in this skill, specs, or evidence; reference
  variable names only.
- Prefer local/test environments; never verify against production without
  explicit instruction.
