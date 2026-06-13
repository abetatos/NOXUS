#!/usr/bin/env bash
set -euo pipefail

# NOXUS test command.
uv run pytest "$@"
