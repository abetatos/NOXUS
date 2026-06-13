"""Tests for CREA benchmark ingestion (REQ-001..006, EDGE-001/002/003, ERR-001/002)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from noxus.config.run import TANGSHAN_BF_COLUMN
from noxus.data.benchmark import (
    BenchmarkColumnError,
    emit_benchmark,
    load_benchmark,
    snapshot_path,
)

FIXTURE = Path(__file__).parent.parent / "fixtures" / "crea_wind_sample.csv"
PIG_IRON = "China: Estimated Daily Average Output: Pig Iron"


def test_load_selects_column_by_name_and_parses_dates():
    df = load_benchmark(FIXTURE, column=TANGSHAN_BF_COLUMN)
    assert df.index.name == "date"
    assert TANGSHAN_BF_COLUMN in df.columns
    # Metadata header rows (Frequency, Unit, ID, ...) must not leak in as data.
    assert df.index.is_monotonic_increasing
    assert isinstance(df.index, pd.DatetimeIndex)


def test_zeros_become_na_not_real_observations():
    # REQ-004 / EDGE-001: 0.00 placeholders on non-reporting (daily-cadence) dates -> NA.
    df = load_benchmark(FIXTURE, column=TANGSHAN_BF_COLUMN)
    assert pd.isna(df.loc[pd.Timestamp("2018-03-12"), TANGSHAN_BF_COLUMN])
    assert pd.isna(df.loc[pd.Timestamp("2018-03-23"), TANGSHAN_BF_COLUMN])
    # And no zero leaks into the cleaned series.
    assert (df[TANGSHAN_BF_COLUMN].dropna() != 0.0).all()


def test_trailing_whitespace_coerced():
    # EDGE-002: values like "70.50 " must coerce to float.
    df = load_benchmark(FIXTURE, column=TANGSHAN_BF_COLUMN)
    assert df.loc[pd.Timestamp("2018-03-02"), TANGSHAN_BF_COLUMN] == pytest.approx(70.5)


def test_duplicate_dates_deduplicated_keep_last():
    df = load_benchmark(FIXTURE, column=TANGSHAN_BF_COLUMN)
    assert df.index.is_unique
    # Fixture has two 2018-03-16 rows (72.80 then 73.10); keep last.
    assert df.loc[pd.Timestamp("2018-03-16"), TANGSHAN_BF_COLUMN] == pytest.approx(73.10)


def test_auxiliary_columns_retained():
    # REQ-006: auxiliary series kept, same zero-as-missing rule.
    df = load_benchmark(FIXTURE, column=TANGSHAN_BF_COLUMN, aux_columns=(PIG_IRON,))
    assert PIG_IRON in df.columns
    assert df.loc[pd.Timestamp("2018-03-12"), PIG_IRON] == pytest.approx(229.0)
    assert pd.isna(df.loc[pd.Timestamp("2018-03-02"), PIG_IRON])


def test_missing_column_raises_descriptive_error():
    # ERR-002 / EDGE-003: selection by name; a missing expected column is an error.
    with pytest.raises(BenchmarkColumnError) as exc:
        load_benchmark(FIXTURE, column="China: Nonexistent Series")
    assert "China: Nonexistent Series" in str(exc.value)


def test_emit_primary_parquet_schema(tmp_path):
    # REQ-005: tidy parquet with date,value,source,snapshot_date.
    df = load_benchmark(FIXTURE, column=TANGSHAN_BF_COLUMN, aux_columns=(PIG_IRON,))
    out = tmp_path / "benchmark.parquet"
    aux = tmp_path / "aux.parquet"
    path = emit_benchmark(
        df,
        out,
        primary_column=TANGSHAN_BF_COLUMN,
        snapshot_date=date(2026, 6, 13),
        aux_out_path=aux,
    )
    got = pd.read_parquet(path)
    assert list(got.columns) == ["date", "value", "source", "snapshot_date"]
    assert got["date"].is_monotonic_increasing
    assert got["value"].dropna().gt(0).all()
    assert aux.exists()
    aux_df = pd.read_parquet(aux)
    assert set(aux_df.columns) == {"date", "series", "value", "source", "snapshot_date"}


def test_snapshot_path_is_dated():
    p = snapshot_path(Path("/tmp/raw"), date(2026, 6, 13))
    assert p.name == "crea_wind_2026-06-13.csv"
