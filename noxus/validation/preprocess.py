"""Align the relative index to the CREA benchmark for validation (NOX-003, REQ-040).

Revives the alignment stage designed in NOX-001 (crea-benchmark-validation) and reconciles its
contract with the new attribution stage. The key contract change: the predictor column is now the
**relative index** (``index_value`` from ``steel_activity_index.parquet``), not the NOX-001 naive
AOI mean (``no2_attributed``) — there was no real predictor when NOX-001 was written, so the module
was reverted to a scaffold. ``align_series`` accepts the index frame's ``index_value`` column.

Alignment composites both series to a common frequency (default weekly), applies the inherited
coverage screening (rows below ``min_coverage`` are dropped, never interpolated, REQ-004/EDGE-003),
and returns the inner-joined overlap so the downstream statistics see only jointly-observed periods.
"""

from __future__ import annotations

import pandas as pd

# Accepted predictor column names, newest first. ``index_value`` is the NOX-003 contract; the older
# names are tolerated so a NOX-001-era artifact still aligns.
_PREDICTOR_COLUMNS = ("index_value", "no2_attributed", "no2_corrected")
_BENCHMARK_COLUMNS = ("value", "operating_rate")


class AlignmentError(RuntimeError):
    """The index and benchmark could not be aligned (missing columns / no overlap)."""


def _pick_column(df: pd.DataFrame, candidates: tuple[str, ...], role: str) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise AlignmentError(
        f"Could not find a {role} column in {list(df.columns)}; expected one of {candidates}."
    )


def _to_dated_series(df: pd.DataFrame, value_col: str) -> pd.Series:
    """Return a date-indexed float Series from a frame carrying a ``date`` column or index."""
    frame = df.copy()
    if "date" in frame.columns:
        idx = pd.to_datetime(frame["date"])
    else:
        idx = pd.to_datetime(frame.index)
    s = pd.Series(frame[value_col].to_numpy(dtype=float), index=pd.DatetimeIndex(idx))
    return s[~s.index.duplicated(keep="last")].sort_index()


def align_series(
    index: pd.DataFrame,
    benchmark: pd.DataFrame,
    freq: str = "W",
    min_coverage: float = 0.25,
) -> pd.DataFrame:
    """Align the index and benchmark to a common frequency on their overlap (REQ-040).

    ``index`` is the ``steel_activity_index.parquet`` frame (``date``, ``index_value``, and
    optionally ``valid_coverage``); ``benchmark`` is the CREA parquet (``date``, ``value``). Both are
    resampled to ``freq`` (period-end), coverage-screened (``valid_coverage`` < ``min_coverage`` →
    dropped, no interpolation), and inner-joined.

    Returns a DataFrame indexed by ``date`` with columns ``index`` and ``benchmark`` over the jointly
    observed periods only. The caller checks the overlap length against ``min_overlap`` (ERR-004).
    """
    pred_col = _pick_column(index, _PREDICTOR_COLUMNS, "predictor")
    bench_col = _pick_column(benchmark, _BENCHMARK_COLUMNS, "benchmark")

    idx_s = _to_dated_series(index, pred_col)
    if "valid_coverage" in index.columns:
        cov = _to_dated_series(index, "valid_coverage")
        # Only screen where coverage is actually recorded (non-NaN); NaN coverage is left to the
        # value's own NaN to decide, so a fully-derived index without coverage is not dropped wholesale.
        below = cov.reindex(idx_s.index)
        idx_s = idx_s.mask(below.notna() & (below < min_coverage))

    bench_s = _to_dated_series(benchmark, bench_col)

    idx_w = idx_s.resample(freq).mean()
    bench_w = bench_s.resample(freq).mean()

    joined = pd.concat({"index": idx_w, "benchmark": bench_w}, axis=1).dropna()
    joined.index.name = "date"
    return joined
