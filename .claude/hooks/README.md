# Project hooks

This directory documents project-specific Claude Code hooks for the SDD harness.

Hooks are enabled from `.claude/settings.json`, not by placing scripts here automatically.

## Enabled hooks

| Event | Matcher | Script | Purpose | Failure mode |
|---|---|---|---|---|
| `PreToolUse` | `Edit\|Write` | `.claude/hooks/block-implementation-before-approval.sh` | Blocks edits to source files while any `sdd: true` task is `pending`/`spec_draft`/`spec_ready`. Spec files, `tasks.json`, `history.html`, and `.claude/` are always allowed. | Fails **open** (warns, exits 0) when `jq` is missing. |
| `PreToolUse` | `Edit\|Write` | `.claude/hooks/validate-spec-before-status-change.sh` | Blocks moving a task to `spec_ready` in `tasks.json` unless its `requirements.html`, `design.html`, and `tasks.html` exist. | Fails **open** (warns, exits 0) when `jq` is missing. |

## Available hook scripts

Copied into `.claude/hooks/`:

- `block-implementation-before-approval.sh` (enabled)
- `validate-spec-before-status-change.sh` (enabled)

More example scripts (run-tests-after-edit, spec-drift, targeted-validation,
pre-compact-capture, failure-learning, etc.) remain in
`sdd-onboarding-kit/hooks/examples/` and are **not** enabled. Enable them only with
developer approval.

## Requirements

These hooks are bash scripts that depend on `jq`. On Windows they need Git Bash (or WSL)
and `jq` on `PATH`. **`jq` is not yet installed in this environment** — until it is, both
hooks fail open (warn and allow) and provide no enforcement. Install `jq`
(`winget install jqlang.jq`, or via Git Bash) to make the guards effective.

## Policy

- Do not enable new hooks without developer approval.
- Prefer warning mode before blocking mode unless the developer approved strict enforcement.
- Keep hook scripts small and deterministic.
- Document every hook's purpose and failure mode.
