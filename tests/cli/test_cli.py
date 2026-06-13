"""CLI smoke tests for benchmark ingestion, TROPOMI fetch wiring, and verification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from noxus.cli.main import main
from noxus.data import tropomi as T

FIXTURE = Path(__file__).parent.parent / "fixtures" / "crea_wind_sample.csv"


def test_ingest_benchmark_from_snapshot(tmp_path):
    out = tmp_path / "benchmark.parquet"
    rc = main(["ingest-benchmark", "--from-snapshot", str(FIXTURE), "--out", str(out)])
    assert rc == 0
    assert out.exists()
    df = pd.read_parquet(out)
    assert list(df.columns) == ["date", "value", "source", "snapshot_date"]


def test_no_command_prints_help_and_succeeds():
    assert main([]) == 0


# --------------------------------------------------------------------------- NOX-003 subcommands (AT-REG-1)


def test_attribute_index_validate_subcommands_exist():
    # The wired subcommands parse --help without error (argparse SystemExit(0)).
    for cmd in ("attribute", "index", "validate", "ingest-era5"):
        with pytest.raises(SystemExit) as exc:
            main([cmd, "--help"])
        assert exc.value.code == 0


def test_existing_subcommands_unchanged_help():
    # Pre-existing commands still parse --help (no regression, AT-REG-1).
    for cmd in ("grid", "fetch", "ingest-benchmark", "verify-no2"):
        with pytest.raises(SystemExit) as exc:
            main([cmd, "--help"])
        assert exc.value.code == 0


def test_attribute_missing_cube_returns_1(tmp_path, monkeypatch):
    # No cube on disk under a fresh cwd -> actionable ERR-001 failure, return code 1 (not a crash).
    monkeypatch.chdir(tmp_path)
    assert main(["attribute"]) == 1


def test_index_missing_signal_returns_1(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["index", "--no-meteo"]) == 1


def test_validate_missing_artifacts_returns_1(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["validate"]) == 1


def test_ingest_era5_fetch_failure_returns_1(tmp_path, monkeypatch):
    # A CDS fetch failure (network/auth/licence) surfaces as a clean exit 1, not a traceback.
    monkeypatch.chdir(tmp_path)
    from noxus.data import era5 as E

    def _boom(*a, **k):
        raise RuntimeError("simulated CDS failure")

    monkeypatch.setattr(E, "ingest_era5", _boom)
    assert main(["ingest-era5"]) == 1


def test_grid_builds_cube_from_store(tmp_path):
    raw = tmp_path / "tropomi"
    out = tmp_path / "no2"
    for date_str, val in [("2023-06-05", 1.0), ("2023-06-07", 3.0)]:
        opid = pd.Timestamp(date_str).strftime("%Y-%m-%dT%H%M%S")
        p = raw / opid[:4] / f"{opid}.nc"
        p.parent.mkdir(parents=True, exist_ok=True)
        xr.Dataset(
            {T.NO2: (("y", "x"), np.full((2, 2), val))},
            coords={"time": pd.Timestamp(date_str), "y": np.arange(2), "x": np.arange(2)},
        ).to_netcdf(p, engine="h5netcdf")
    T._save_manifest(
        raw,
        {
            "overpasses": {
                pd.Timestamp(d).strftime("%Y-%m-%dT%H%M%S"): {
                    "path": str(
                        raw
                        / pd.Timestamp(d).strftime("%Y")
                        / (pd.Timestamp(d).strftime("%Y-%m-%dT%H%M%S") + ".nc")
                    ),
                    "processor_version": "v2.x",
                }
                for d in ["2023-06-05", "2023-06-07"]
            },
            "batches_done": [],
            "batch_errors": {},
        },
    )
    rc = main(["grid", "--raw-dir", str(raw), "--out-dir", str(out)])
    assert rc == 0
    assert (out / "no2_cube_w.nc").exists()


def test_verify_no2_empty_store_returns_1(tmp_path):
    assert main(["verify-no2", "--raw-dir", str(tmp_path / "empty")]) == 1


def test_verify_no2_renders_from_seeded_store(tmp_path):
    raw = tmp_path / "tropomi"
    (raw / "2023").mkdir(parents=True)
    nc = raw / "2023" / "op.nc"
    xr.Dataset(
        {T.NO2: (("y", "x"), np.random.default_rng(0).random((5, 5)))},
        coords={"y": np.arange(5), "x": np.arange(5)},
    ).to_netcdf(nc, engine="h5netcdf")
    T._save_manifest(
        raw,
        {
            "overpasses": {
                "2023-06-05T050000": {
                    "cloud_mean": 0.1,
                    "no2_mean": 5.0,
                    "valid_coverage": 0.9,
                    "path": str(nc),
                }
            },
            "batches_done": [],
            "batch_errors": {},
        },
    )
    fac = tmp_path / "fac.csv"
    pd.DataFrame({"name": ["p"], "latitude": [39.6], "longitude": [118.2]}).to_csv(fac, index=False)
    out = tmp_path / "verif"
    rc = main(
        [
            "verify-no2",
            "--raw-dir",
            str(raw),
            "--facilities",
            str(fac),
            "--out-dir",
            str(out),
        ]
    )
    assert rc == 0
    assert list(out.glob("*.png"))
