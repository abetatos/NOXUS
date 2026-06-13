# NOXUS — Claude Code project instructions

This project uses **Spec Driven Development (SDD)** for non-trivial implementation work.

## Project summary

NOXUS is a public, reproducible pipeline from satellite NO₂ (Sentinel-5P/TROPOMI) to a
steel-sector activity signal for a single industrial cluster (Tangshan, Hebei), tested
transparently against a physical-output benchmark. Python package (`noxus`) with a CLI; uses
only public, free data so the work is reproducible. Currently an early scaffold: module
skeleton in place, pipeline stages not yet implemented.

## Project map

Repository structure, entrypoints, commands, important docs, and protected
areas are described in the project map:

```text
.claude/context/project-map.md
```

Read it when you need orientation; do not duplicate its directory tree into
this file. Update it when the structure changes significantly (new
top-level directories, moved entrypoints, changed commands, new protected
areas).

## Commands

Use these project commands:

```bash
# Install or validate environment
uv sync --extra dev

# Run all tests
uv run pytest

# Run lint
uv run ruff check .

# Run typecheck
# (no typecheck configured — no mypy in this project)

# Format code
uv run ruff format .
```

If any command is unknown, ask the developer before inventing one.

## SDD policy

SDD applies to **every new feature**: a spec (requirements → design → tasks) is created and
approved before any implementation code is written. Bug fixes use a lighter flow (a short spec
is encouraged but the full requirements/design/tasks set is not mandatory unless the fix is
non-trivial). Documentation-only changes do not require a spec. Tasks are tracked in
`tasks.json`; mark feature tasks with `"sdd": true`.

Default state machine:

```text
pending → spec_draft → spec_ready → human_approved → in_progress → review → done
```

Do not implement an SDD task unless it is in `human_approved` or `in_progress`.

If a task is marked `spec_ready`, stop and ask for human approval.

## Task storage

Task state is stored in:

```text
tasks.json
```

## Spec storage

Specs are stored in:

```text
specs/<feature-slug>/
├── requirements.html
├── design.html
├── tasks.html
├── review.html
├── spec.css
└── spec.js
```

Spec files are self-contained HTML. Open any of them in a browser to read the spec.
Structured state (task status, approval) lives in `tasks.json`, not in the HTML files.

## Required SDD workflow

For each SDD task:

1. Read the task.
2. Create or update the spec.
3. Stop for human approval.
4. Implement only after approval.
5. Run tests and validation.
6. Run reviewer.
7. If the reviewer requires documentation updates, run documenter, then the reviewer re-checks the docs.
8. Mark done only if requirements, implementation, tests and required docs are aligned.
9. Append a summary to `history.html`.

## SDD skill

Use the project skill:

```text
.claude/skills/sdd-workflow/SKILL.md
```

Invoke it directly when needed:

```text
/sdd-workflow <task-id-or-feature-description>
```

## Optional skills installed

- `run-and-verify` — how to run the pipeline/CLI and verify behavior before review.
- `project-map` — generate or refresh the project map when the repo structure changes.
- `decision-log` — record durable architectural/methodological choices and rejected options in `decisions/`.

Invoke them by name when their trigger applies. Their full instructions live in `.claude/skills/<name>/SKILL.md` and load only when used — do not duplicate them here.

## Subagents

Use project subagents when appropriate:

- `leader`: inspects task state and recommends which agent to invoke next (subagents cannot invoke each other; the main conversation orchestrates).
- `spec-author`: creates requirements, design and implementation tasks.
- `implementer`: writes code from approved specs.
- `reviewer`: validates implementation against specs; decides whether documentation updates are required.
- `documenter`: updates affected documentation after review approval, before a task is marked done.

## Human approval policy

Human approval is **mandatory** before implementation. After the spec is complete, stop at
`spec_ready` and wait. Only the developer moves a task to `human_approved` (record it in
`tasks.json` under `approval`). Claude may revise specs after feedback without resetting the
task, but must not start implementation until approval is recorded.

## Hooks policy

Two SDD-enforcement hooks are enabled (see `.claude/settings.json` and `.claude/hooks/README.md`):

- **block-implementation-before-approval** — blocks edits to source files while an `sdd: true`
  task is `pending`/`spec_draft`/`spec_ready` (spec files, `tasks.json`, `.claude/` always allowed).
- **validate-spec-before-status-change** — blocks moving a task to `spec_ready` unless its
  `requirements.html`, `design.html` and `tasks.html` exist.

Both are bash + `jq` scripts and **fail open** (warn and allow) when `jq` is missing. `jq` is
not yet installed in this environment — install it (Git Bash / `winget install jqlang.jq`) for
the guards to actually enforce. Do not enable further hooks without developer approval.

## MCP policy

Local-first: no external MCPs are configured. Durable state lives in local artifacts
(`tasks.json`, `specs/`, `history.html`, `decisions/`). The `gh` CLI is available and may be
used directly for GitHub operations; remote-mutating commands need explicit per-action
permission. Do not add MCPs without developer approval.

## Git policy

- Create one branch per SDD feature (suggested naming: `feature/<feature-slug>` or
  `fix/<slug>` for bug fixes).
- Commit only when the developer asks; reference the task ID in the commit message.
- Open pull requests with `gh` only when the developer requests it.
- No push/merge/force operations without explicit per-action permission.

## Protected areas

Require explicit approval before editing:

- `.env` and any secret/credential files (CDSE / GEE credentials). Never read, write, or log
  secret values; reference variable names only.

Additional reproducibility conventions (not hard-blocked, but treat with care): `data/raw/` and
large Earth-observation downloads (`*.nc`, `*.tif`) are gitignored and must not be committed;
`uv.lock` is committed on purpose for reproducibility — change it only through `uv` and with
reason.

## When SDD may be skipped

- Documentation-only or comment-only changes.
- Trivial fixes (typos, formatting, one-line obvious corrections).
- Emergency/hotfix work — allowed to bypass the full flow, but record afterward in
  `history.html` and backfill a short spec if the change was non-trivial.

For bug fixes, prefer at least a short requirements + design note unless the fix is trivial.

## Context economy

The context window is a scarce resource. Rules:

- Keep this file short; link to deeper docs instead of duplicating them.
- Long procedures belong in skills, not here.
- Use `/context` to inspect usage; use `/compact` between unrelated tasks, with focus instructions to preserve decisions, task status, architecture constraints, and unresolved questions.
- Delegate noisy, many-file exploration to subagents; keep only conclusions in the main conversation.
- Durable truth lives in artifacts (`tasks.json`, `specs/`, decision logs, `history.html`), not in chat history.

## Session recovery

If unsure after resuming a session, a rewind, or a compaction: inspect the durable artifacts before continuing — `tasks.json` for task state, the active spec for what is approved, the latest review for open findings, `git status` for actual changes. When conversation memory and an artifact disagree, the artifact wins.

## Final rule

Prefer a clear spec over a clever implementation. If the spec is ambiguous, ask or revise the spec before coding.

## Functional documents

Functional documents, PRDs, tickets, user stories, or informal feature descriptions are source material.

They are not approved implementation specs.

When a functional document is provided, Claude Code must first generate an SDD spec under:

specs/<feature-slug>/

Required files:

- requirements.html
- design.html
- tasks.html
- assumptions.html
- open-questions.html
- acceptance-tests.html
- spec.css (copy from .claude/skills/sdd-workflow/templates/spec.css)
- spec.js (copy from .claude/skills/sdd-workflow/templates/spec.js)

Claude Code must stop after generating the spec and wait for human approval before implementation.
