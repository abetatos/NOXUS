"""CLI wiring + actionable-error tests for `assemble-calendar` / `regime-study` (NOX-012).

Maps to AT10 (CLI wired), AT-REG-1 (existing subcommands unchanged), AT-ERR-1 (missing residual ->
actionable message). No network, no credentials.
"""

from __future__ import annotations

import pytest

from noxus.cli.main import main


def test_assemble_calendar_help_parses():
    with pytest.raises(SystemExit) as exc:
        main(["assemble-calendar", "--help"])
    assert exc.value.code == 0


def test_regime_study_help_parses():
    with pytest.raises(SystemExit) as exc:
        main(["regime-study", "--help"])
    assert exc.value.code == 0


def test_existing_subcommands_unchanged():
    """AT-REG-1: pre-existing verbs still parse cleanly after adding the NOX-012 verbs."""
    for cmd in ("attribute", "index", "validate", "grid", "scale-sweep", "variance", "catalyst"):
        with pytest.raises(SystemExit) as exc:
            main([cmd, "--help"])
        assert exc.value.code == 0


def test_assemble_calendar_probe_runs(monkeypatch, tmp_path, capsys):
    """AT1/AT10: `assemble-calendar --probe` prints the go/no-go verdict and writes SOURCES.md."""
    monkeypatch.chdir(tmp_path)
    rc = main(["assemble-calendar", "--probe"])
    out = capsys.readouterr().out
    assert rc == 0  # GO-coarse-floor is a go
    assert "T1 probe verdict=GO-coarse-floor" in out
    assert (tmp_path / "data" / "curtailment" / "SOURCES.md").exists()


def test_assemble_calendar_writes_csv(monkeypatch, tmp_path, capsys):
    """AT2/AT10: `assemble-calendar` writes the curated, loader-compatible CSV."""
    monkeypatch.chdir(tmp_path)
    rc = main(["assemble-calendar"])
    out = capsys.readouterr().out
    assert rc == 0
    csv = tmp_path / "data" / "curtailment" / "tangshan_hebei_curtailment.csv"
    assert csv.exists()
    assert "curated calendar" in out
    # Loads through the unchanged NOX-004 loader.
    from noxus.catalyst.groundtruth import load_curtailment_calendar

    loaded = load_curtailment_calendar(csv)
    assert list(loaded.columns) == ["start", "end", "direction", "cause"]


def test_regime_study_missing_residual_is_actionable(monkeypatch, tmp_path, capsys):
    """AT-ERR-1 / ERR-001: absent NOX-003.1 residual -> failure naming the producing command."""
    monkeypatch.chdir(tmp_path)
    rc = main(["regime-study"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "regime-study failed" in out
    assert "noxus index" in out  # names the producing command
