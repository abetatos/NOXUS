# SDD onboarding decisions

Records the developer's answers from the SDD onboarding (2026-06-13).

## Project

- Project name: NOXUS
- Project summary: Public, reproducible pipeline from satellite NO₂ (Sentinel-5P/TROPOMI) to a steel-sector activity signal for a single industrial cluster (Tangshan), validated against a physical-output benchmark.
- Main language/framework: Python ≥ 3.12 (CLI + library; no web framework)
- Package manager: uv (lockfile committed)

## Commands

- Init command: `uv sync --extra dev`
- Test command: `uv run pytest`
- Targeted test command: `uv run pytest <path>::<test>`
- Lint command: `uv run ruff check .`
- Typecheck command: none configured (no mypy)
- Format command: `uv run ruff format .`

## SDD policy

- Scope of SDD: every new feature (spec before code)
- Tasks that may skip SDD: documentation-only changes; trivial fixes (typos, formatting); bug fixes use a lighter flow (short spec encouraged, full set only when non-trivial); emergency/hotfix may bypass and be backfilled
- Human approval required: yes, mandatory before implementation
- Requirements format: EARS (default)
- Task storage: local `tasks.json`
- Spec storage: `specs/<feature-slug>/`
- History storage: `history.html`

## State machine

- Statuses: pending → spec_draft → spec_ready → human_approved → in_progress → review → done (plus `blocked`, `rejected`)
- Approval transition rule: only the developer moves `spec_ready` → `human_approved`; approval recorded in `tasks.json`
- Done transition rule: reviewer approval required; not `done` while required docs pending or tests failing

## Git policy

- Branch creation: one branch per SDD feature (suggested `feature/<slug>`, `fix/<slug>`)
- Branch naming: `feature/<feature-slug>` / `fix/<slug>`
- Commit policy: commit only when the developer asks; reference task ID
- Pull request policy: open PRs with `gh` only on developer request; no push/merge/force without explicit per-action permission

## Hooks

- Enabled hooks: block-implementation-before-approval (PreToolUse Edit|Write); validate-spec-before-status-change (PreToolUse Edit|Write)
- Hooks left as examples: run-tests-after-edit, spec-drift, targeted-validation, pre-compact-capture, failure-learning, block-destructive-commands, notify-for-approval
- Hook failure mode: fail open (warn and allow) when `jq` is missing. NOTE: `jq` is not yet installed — developer agreed to install it so the guards become effective.

## MCPs

- Configured MCPs: none (local-first)
- Read-only MCPs: none
- Read/write MCPs: none
- External task mapping: none (tasks in local `tasks.json`)
- Note: `gh` CLI is available and preferred over an MCP for GitHub operations.

## Protected areas

- Protected files: `.env` and any secret/credential files (CDSE / GEE). Never read, write, or log secret values; variable names only.
- Protected directories: none hard-blocked beyond secrets.
- Requires explicit approval / handle with care: `data/raw/` and large EO downloads (`*.nc`, `*.tif`, gitignored, never commit); `uv.lock` (committed for reproducibility, change only via `uv`).

## Other defaults applied (safe profile)

- Documentation phase: enabled (task not `done` while required docs pending).
- Dependency/API freshness: required for high-risk categories, advisory otherwise — relevant here for external SDKs/APIs (CDSE, earthengine-api, xarray/rasterio/netCDF4). The `dependency-freshness` *pack* was declined; the policy is still enforced via the design template, spec-author rules, and reviewer checklist.
- Browser/Playwright (§18): option 1 — no browser UI; nothing installed.
- Failure learning (§19): proposals enabled; accepted lessons go to project memory only; no memory write without explicit approval of the exact text. The `failure-learning` pack was declined.
- Deep review (§21): recommended (not required) for high-risk categories; paid/cloud modes only with explicit per-invocation approval.
- Autonomy (§22): disabled except documented read-only monitoring with explicit stop conditions.
- Session recovery (§23): rule kept in `CLAUDE.md`; pre-compact hook not enabled.

## Optional skill packs

- Installed: `run-and-verify`, `project-map`, `decision-log`.
- Declined (do not re-propose): `context-audit`, `dependency-freshness`, `git-discipline`, `documentation-update`, `failure-learning`, `ui-qa`, `spec-from-screenshot`.

## Decision-log locations (§17)

- Architecture decisions: `decisions/architecture-decisions.md`
- Rejected options: `decisions/rejected-options.md`
- Workflow decisions: `decisions/workflow-decisions.md`

## Open TODOs

- TODO (developer): install `jq` (e.g. `winget install jqlang.jq`) so the enabled hooks enforce instead of failing open.
- TODO (optional): if a typecheck step is later desired, add `mypy` to the `dev` extra and a `uv run mypy noxus` command, then update `CLAUDE.md`, the project map, and `run-and-verify`.
