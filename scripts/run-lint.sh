#!/usr/bin/env bash
set -euo pipefail

# NOXUS lint command (ruff). No separate typecheck is configured (no mypy).
uv run ruff check .
