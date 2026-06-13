"""Ingest the physical-output benchmark used for validation.

The benchmark is the weekly **blast-furnace operating rate for the Tangshan cluster** — a physical
production measure used deliberately in place of a diffusion index such as the PMI, which measures
sentiment rather than production. It is sourced from CREA's public, WIND-sourced steel CSV.

Pipeline: ``fetch_benchmark_snapshot`` writes a dated raw snapshot (the reproducible source of
record); ``load_benchmark`` parses and cleans it (skip metadata headers, select the column by exact
name, treat ``0.00`` placeholders as missing); ``emit_benchmark`` writes a tidy parquet series.

Two data hazards motivate the careful parser:

1. The CSV is the union of weekly and daily series, so a weekly column carries ``0.00`` placeholders
   on dates belonging to other cadences. A 0% cluster operating rate is implausible, so ``0.00`` is
   treated as not-reported (``NA``), never as a real observation.
2. The file has several metadata header rows (``#ERROR!``, ``Name``, ``Frequency``, ``Unit``,
   ``ID``, ``Time Period``, ``Source``, ``Update``) before the dated observation rows.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from noxus.config.run import TANGSHAN_BF_COLUMN

# Label rows that precede the dated observations in the CREA CSV (first-column values to drop).
_METADATA_LABELS = frozenset({"Frequency", "Unit", "ID", "Time Period", "Source", "Update", "Name"})


class BenchmarkSourceError(RuntimeError):
    """The benchmark source could not be fetched or did not look like the expected CSV."""


class BenchmarkColumnError(KeyError):
    """The expected benchmark column was absent from the fetched CSV."""


def snapshot_path(raw_dir: Path, today: date) -> Path:
    """Return the dated snapshot path ``<raw_dir>/crea_wind_<YYYY-MM-DD>.csv``."""
    return Path(raw_dir) / f"crea_wind_{today.isoformat()}.csv"


def fetch_benchmark_snapshot(url: str, raw_dir: Path, today: date) -> Path:
    """Fetch the CREA CSV and persist the unmodified bytes to a dated snapshot.

    The fetch is the only network step; everything downstream reads the snapshot, so analysis is
    reproducible (REQ-001, NFR-001). On any failure the function raises and does not write a partial
    or empty snapshot, so an existing good snapshot is never clobbered (ERR-001).
    """
    import httpx

    try:
        response = httpx.get(url, follow_redirects=True, timeout=60.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:  # network / status failure
        raise BenchmarkSourceError(f"Could not fetch benchmark from {url}: {exc}") from exc

    content = response.content
    text = response.text
    if not content or "Name," not in text:
        raise BenchmarkSourceError(
            f"Benchmark source at {url} did not return the expected CSV "
            "(missing the 'Name' header row or empty body)."
        )

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = snapshot_path(raw_dir, today)
    out.write_bytes(content)
    return out


def _clean_numeric(series: pd.Series) -> pd.Series:
    """Strip whitespace, coerce to float, and map exact zeros to NA (0.00 == not reported)."""
    cleaned = series.astype("string").str.strip()
    numeric = pd.to_numeric(cleaned, errors="coerce")
    return numeric.where(numeric != 0.0)


def load_benchmark(
    snapshot_csv: Path,
    column: str = TANGSHAN_BF_COLUMN,
    aux_columns: tuple[str, ...] | list[str] = (),
) -> pd.DataFrame:
    """Parse a CREA snapshot into a clean wide frame indexed by date.

    Selects ``column`` (and any ``aux_columns``) by exact header name — never by position — so a
    change in the source's column order or set is caught rather than silently mis-read (EDGE-003).
    Returns a DataFrame indexed by ``date`` (ascending, de-duplicated) with one float column per
    requested series, with ``0.00`` placeholders converted to ``NA`` (REQ-003..006).
    """
    # skiprows=1 drops the leading "#ERROR!" cell row; the next row ("Name,...") becomes the header.
    raw = pd.read_csv(snapshot_csv, skiprows=1, dtype=str)
    raw = raw.rename(columns={raw.columns[0]: "label"})

    # The primary column is mandatory (ERR-002); auxiliary columns are best-effort — absent ones are
    # skipped so the ingest survives source column drift (EDGE-003).
    if column not in raw.columns:
        raise BenchmarkColumnError(
            f"Expected benchmark column {column!r} not found in {Path(snapshot_csv).name}. "
            f"Columns present: {list(raw.columns)}"
        )
    wanted = [column, *(c for c in aux_columns if c in raw.columns)]

    # Drop metadata rows (Frequency, Unit, ID, Time Period, Source, Update); keep dated rows.
    data = raw[~raw["label"].isin(_METADATA_LABELS)].copy()
    dates = pd.to_datetime(data["label"], errors="coerce")
    data = data.assign(date=dates).dropna(subset=["date"])

    out = pd.DataFrame({"date": data["date"].dt.normalize().to_numpy()})
    for col in wanted:
        out[col] = _clean_numeric(data[col]).to_numpy()

    out = out.sort_values("date").drop_duplicates(subset="date", keep="last").set_index("date")
    return out


def emit_benchmark(
    wide: pd.DataFrame,
    out_path: Path,
    primary_column: str = TANGSHAN_BF_COLUMN,
    source: str = "CREA (WIND / China United Steel Network)",
    snapshot_date: date | None = None,
    aux_out_path: Path | None = None,
) -> Path:
    """Write the tidy primary benchmark series to parquet; optionally write auxiliary series too.

    Primary schema (REQ-005): ``date``, ``value`` (float, percent), ``source``, ``snapshot_date``.
    Rows are sorted ascending by date and de-duplicated. If ``aux_out_path`` is given and the frame
    carries auxiliary columns, those are written as a long parquet (``date``, ``series``, ``value``,
    ``source``, ``snapshot_date``) for downstream robustness use (REQ-006).
    """
    if primary_column not in wide.columns:
        raise BenchmarkColumnError(
            f"Primary column {primary_column!r} not present in the frame to emit; "
            f"have {list(wide.columns)}."
        )

    snap = snapshot_date.isoformat() if snapshot_date is not None else None
    primary = (
        wide[[primary_column]]
        .rename(columns={primary_column: "value"})
        .reset_index()
        .rename(columns={"index": "date"})
        .sort_values("date")
        .drop_duplicates(subset="date", keep="last")
    )
    primary["source"] = source
    primary["snapshot_date"] = pd.to_datetime(snap) if snap else pd.NaT

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    primary.to_parquet(out_path, index=False)

    aux_columns = [c for c in wide.columns if c != primary_column]
    if aux_out_path is not None and aux_columns:
        long = (
            wide[aux_columns]
            .reset_index()
            .rename(columns={"index": "date"})
            .melt(id_vars="date", var_name="series", value_name="value")
        )
        long["source"] = source
        long["snapshot_date"] = pd.to_datetime(snap) if snap else pd.NaT
        aux_out_path = Path(aux_out_path)
        aux_out_path.parent.mkdir(parents=True, exist_ok=True)
        long.to_parquet(aux_out_path, index=False)

    return out_path
