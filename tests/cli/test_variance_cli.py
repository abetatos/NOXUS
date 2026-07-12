"""CLI wiring + actionable-error tests for `noxus variance` (NOX-009 AT-ERR-2, AT-REG-1).

The error paths are pure existence checks (missing weekly/daily cube, missing ERA5 snapshot) so they
need no valid NetCDF — only presence/absence of the artefacts under a temp working dir. No network,
no credentials, no benchmark.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from noxus.cli.main import main


def test_variance_help_parses():
    # AT-REG-1: the new subcommand parses --help cleanly (argparse SystemExit(0)); additive surface.
    with pytest.raises(SystemExit) as exc:
        main(["variance", "--help"])
    assert exc.value.code == 0


def test_existing_subcommands_unchanged(monkeypatch, tmp_path):
    # AT-REG-1: pre-existing verbs still parse --help (no regression from adding `variance`).
    for cmd in ("attribute", "index", "validate", "grid", "scale-sweep"):
        with pytest.raises(SystemExit) as exc:
            main([cmd, "--help"])
        assert exc.value.code == 0


def _cubes(tmp_path: Path):
    d = tmp_path / "data" / "derived" / "no2"
    d.mkdir(parents=True, exist_ok=True)
    return d / "no2_cube_w.nc", d / "no2_cube_d.nc"


def test_missing_weekly_cube_names_grid(monkeypatch, tmp_path, capsys):
    # AT-ERR-2 (a) / ERR-001: no weekly cube -> actionable failure naming 'noxus grid'.
    monkeypatch.chdir(tmp_path)
    rc = main(["variance"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "noxus grid" in out


def test_missing_daily_cube_names_daily_command(monkeypatch, tmp_path, capsys):
    # AT-ERR-2 (b) / ERR-002: weekly present, daily requested but absent -> names the NOX-006 daily cmd.
    monkeypatch.chdir(tmp_path)
    weekly, _daily = _cubes(tmp_path)
    weekly.write_bytes(b"")  # existence-only; the daily check precedes any open
    rc = main(["variance"])  # default resolutions include D
    out = capsys.readouterr().out
    assert rc == 1
    assert "noxus grid --freq D" in out


def test_missing_era5_names_ingest(monkeypatch, tmp_path, capsys):
    # AT-ERR-2 (c) / ERR-003: cubes present, ERA5 snapshot absent -> names the ERA5 ingest command.
    monkeypatch.chdir(tmp_path)
    weekly, daily = _cubes(tmp_path)
    weekly.write_bytes(b"")
    daily.write_bytes(b"")
    rc = main(["variance"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "ingest-era5" in out
