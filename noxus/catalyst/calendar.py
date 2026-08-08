"""Public curtailment / air-quality-control calendar assembly + T1 feasibility spike (NOX-012).

Two deliverables live here (design "Technical summary", REQ-001..004):

- :func:`probe_sources` — the **T1 feasibility spike that GATES the build**. It surveys candidate
  public/free source records (mocked at test time; a live MEE / Hebei DEE / Tangshan / CREA / Mysteel
  survey at run time), classifies coverage + provenance per candidate (source, date, *granularity*,
  direction, securely-dated confidence) and emits a **three-way go/no-go verdict**:
  ``GO-event-level`` (>= the configured minimum securely-dated day/week events spanning the era, with
  >=1 in the 2023-peak window and >=1 in 2024) vs ``GO-coarse-floor`` (only heating-season windows +
  holidays + a few campaigns datable) vs ``NO-GO`` (cannot even date heating seasons — a blocking
  escalation, EDGE-001). Nothing is invented: an unsourced candidate is excluded/flagged, never
  fabricated (REQ-003).

- :func:`assemble_calendar` / :func:`validate_calendar` — curate the dated, source-cited
  ``tangshan_hebei_curtailment.csv`` (``start,end,direction,cause`` + ``source,url,granularity,
  confidence``). The output conforms to the **existing** ``load_curtailment_calendar`` four-column
  contract; the provenance columns are additive and ignored by that loader, so NOX-004 is untouched
  (A4/REQ-004).

The **coarse floor** (:func:`coarse_floor_candidates`) is built only from securely-datable public
record — the annual heating-season restriction windows and the fixed public-holiday windows (Spring
Festival + National-Day Golden Week). It is the safe default (A3/EDGE-001), not an event-level
curtailment campaign set; specific campaign dates that cannot be sourced offline are left as a
documented gap, never invented.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from noxus.config.run import (
    GOLDEN_WEEK_YEARS,
    SPRING_FESTIVAL_WINDOWS,
    RegimeStudyConfig,
)

# Schema: the four columns the NOX-004 loader consumes, plus additive provenance columns (REQ-003/004).
CORE_COLUMNS = ("start", "end", "direction", "cause")
PROVENANCE_COLUMNS = ("source", "url", "granularity", "confidence")
CALENDAR_COLUMNS = CORE_COLUMNS + PROVENANCE_COLUMNS

_VALID_DIRECTIONS = frozenset({"up", "down"})
_VALID_GRANULARITY = frozenset({"day", "week", "month", "season"})
_EVENT_LEVEL_GRANULARITY = frozenset({"day", "week"})  # fine enough for the event-bound test
_VALID_CONFIDENCE = frozenset({"high", "medium", "low"})

# The *structural coarse floor* causes (heating seasons + public holidays). These are securely datable
# public record but are NOT the specific curtailment *campaigns* (blue-sky/anti-smog enforcement,
# sudden shutdowns/restarts) the event-bound test needs. A GO-event-level verdict requires genuine
# campaign events beyond this floor, so the offline coarse floor cannot, by itself, claim event-level.
STRUCTURAL_FLOOR_CAUSES = frozenset(
    {"heating_season_restriction", "spring_festival", "golden_week"}
)

# Verdict labels (stable strings tests key on).
GO_EVENT = "GO-event-level"
GO_COARSE = "GO-coarse-floor"
NO_GO = "NO-GO"


class CalendarValidationError(ValueError):
    """A curated calendar row is unsourced, mis-typed, or missing a required column (REQ-003)."""


@dataclass(frozen=True)
class CandidateEvent:
    """One candidate curtailment/AQ-control event with provenance (REQ-001/003)."""

    cause: str
    start: str
    end: str
    direction: str  # up | down
    source: str
    url: str
    granularity: str  # day | week | month | season
    confidence: str  # high | medium | low

    @property
    def securely_dated(self) -> bool:
        """A candidate is securely dated when it has a source and a non-``low`` confidence."""
        return bool(self.source) and self.confidence in {"high", "medium"}

    def as_row(self) -> dict[str, str]:
        return {
            "start": self.start,
            "end": self.end,
            "direction": self.direction,
            "cause": self.cause,
            "source": self.source,
            "url": self.url,
            "granularity": self.granularity,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class ProbeReport:
    """T1 coverage+provenance report + three-way go/no-go verdict (REQ-001, AT1)."""

    verdict: str  # GO-event-level | GO-coarse-floor | NO-GO
    reason: str
    n_candidates: int
    n_securely_dated: int
    n_event_level: int  # securely-dated candidates at day/week granularity
    granularity_counts: dict[str, int]
    spans_era: bool
    has_peak_event: bool  # >=1 securely-dated event in the 2023-peak window
    has_collapse_event: bool  # >=1 securely-dated event in 2024
    live_survey_pending: bool  # True when the empirical event-level source survey was not run
    candidates: tuple[CandidateEvent, ...] = field(default_factory=tuple)

    @property
    def go(self) -> bool:
        return self.verdict != NO_GO


# --------------------------------------------------------------------------- coarse floor (datable)


def _golden_week_windows(years: Sequence[int]) -> list[tuple[str, str]]:
    """Fixed National-Day Golden Week (Oct 1–7) windows per year (public record)."""
    return [(f"{y}-10-01", f"{y}-10-07") for y in years]


def _heating_season_windows(
    start_year: int, end_year: int, season: tuple[str, str]
) -> list[tuple[str, str]]:
    """Annual heating-season restriction windows (~Nov 15 -> next-year Mar 15)."""
    start_md, end_md = season
    windows: list[tuple[str, str]] = []
    for y in range(start_year, end_year + 1):
        windows.append((f"{y}-{start_md}", f"{y + 1}-{end_md}"))
    return windows


def coarse_floor_candidates(cfg: RegimeStudyConfig | None = None) -> list[CandidateEvent]:
    """The securely-datable coarse floor: heating-season windows + public holidays (A3/EDGE-001).

    A ``down`` marks reduced production (heating-season restriction, holiday slowdown); the interval
    *end* is the opposite-direction restart edge, handled downstream by the NOX-004 edge-matcher
    (``_calendar_edge_events``, EDGE-002). Every row is public record and cited in ``SOURCES.md``;
    none is a fabricated event-level campaign.
    """
    cfg = cfg or RegimeStudyConfig()
    out: list[CandidateEvent] = []

    for start, end in _heating_season_windows(
        cfg.calendar_start_year, cfg.calendar_end_year, cfg.heating_season
    ):
        out.append(
            CandidateEvent(
                cause="heating_season_restriction",
                start=start,
                end=end,
                direction="down",
                source="MEE/Hebei DEE heating-season industrial-restriction notices",
                url="https://www.mee.gov.cn/",
                granularity="season",
                confidence="medium",
            )
        )

    for start, end in SPRING_FESTIVAL_WINDOWS:
        y = int(start[:4])
        if cfg.calendar_start_year <= y <= cfg.calendar_end_year:
            out.append(
                CandidateEvent(
                    cause="spring_festival",
                    start=start,
                    end=end,
                    direction="down",
                    source="PRC State Council public-holiday arrangement",
                    url="https://www.gov.cn/",
                    granularity="day",
                    confidence="high",
                )
            )

    gw_years = [
        y for y in GOLDEN_WEEK_YEARS if cfg.calendar_start_year <= y <= cfg.calendar_end_year
    ]
    for start, end in _golden_week_windows(gw_years):
        out.append(
            CandidateEvent(
                cause="golden_week",
                start=start,
                end=end,
                direction="down",
                source="PRC State Council National-Day holiday arrangement",
                url="https://www.gov.cn/",
                granularity="day",
                confidence="high",
            )
        )

    return sorted(out, key=lambda c: c.start)


# --------------------------------------------------------------------------- T1 probe (REQ-001)


def probe_sources(
    cfg: RegimeStudyConfig | None = None,
    *,
    candidates: Sequence[CandidateEvent] | None = None,
    live_survey: bool = False,
) -> ProbeReport:
    """T1 feasibility spike: classify coverage/provenance + emit a three-way go/no-go (REQ-001, AT1).

    ``candidates`` is the surveyed candidate set. When ``None`` and ``live_survey`` is False, the
    probe measures the **offline coarse floor** (heating seasons + holidays) — enough to reach at least
    ``GO-coarse-floor`` — and records ``live_survey_pending=True`` to flag that the empirical
    event-level source survey (a live MEE/DEE/Mysteel fetch) was **not** performed here (mirrors the
    NOX-007/NOX-011 T1 pattern: build the machinery, run the offline part, do not fabricate). Pass an
    explicit ``candidates`` list (mocked in tests) to classify a richer set.

    The verdict is ``GO-event-level`` only when >= ``cfg.min_event_level`` securely-dated day/week
    candidates span the era with >=1 in the 2023-peak window and >=1 in 2024; ``GO-coarse-floor`` when
    only the coarse floor is datable; ``NO-GO`` when not even a heating season can be dated (an
    escalation, never a silent stop).
    """
    cfg = cfg or RegimeStudyConfig()
    live_pending = candidates is None and not live_survey
    cand = list(candidates) if candidates is not None else coarse_floor_candidates(cfg)

    secure = [c for c in cand if c.securely_dated]
    # Event-level = securely-dated day/week *campaign* events beyond the structural coarse floor.
    # Heating seasons + holidays (the floor) never, on their own, upgrade the verdict to event-level.
    event_level = [
        c
        for c in secure
        if c.granularity in _EVENT_LEVEL_GRANULARITY and c.cause not in STRUCTURAL_FLOOR_CAUSES
    ]

    granularity_counts: dict[str, int] = {g: 0 for g in sorted(_VALID_GRANULARITY)}
    for c in cand:
        granularity_counts[c.granularity] = granularity_counts.get(c.granularity, 0) + 1

    peak_lo, peak_hi = pd.Timestamp(cfg.peak_window[0]), pd.Timestamp(cfg.peak_window[1])
    coll_lo, coll_hi = pd.Timestamp(cfg.collapse_window[0]), pd.Timestamp(cfg.collapse_window[1])

    def _in(c: CandidateEvent, lo: pd.Timestamp, hi: pd.Timestamp) -> bool:
        s = pd.Timestamp(c.start)
        return lo <= s <= hi

    has_peak = any(_in(c, peak_lo, peak_hi) for c in event_level)
    has_collapse = any(_in(c, coll_lo, coll_hi) for c in event_level)
    starts = [pd.Timestamp(c.start).year for c in secure]
    spans_era = (
        bool(starts) and min(starts) <= cfg.calendar_start_year and max(starts) >= cfg.collapse_year
    )

    can_date_heating = any(
        c.cause == "heating_season_restriction" and c.securely_dated for c in cand
    )

    if not secure and not can_date_heating:
        verdict = NO_GO
        reason = (
            "NO-GO: cannot securely date even the heating-season restriction windows from the "
            "surveyed sources — escalate to the developer (EDGE-001)."
        )
    elif len(event_level) >= cfg.min_event_level and spans_era and has_peak and has_collapse:
        verdict = GO_EVENT
        reason = (
            f"GO-event-level: {len(event_level)} securely-dated day/week events span the era with "
            ">=1 in the 2023-peak window and >=1 in 2024 — the event-bound test runs at event "
            "granularity (REQ-021)."
        )
    else:
        verdict = GO_COARSE
        pend = (
            " The empirical event-level source survey (live MEE/DEE/Mysteel fetch) is PENDING a "
            "live run; only the offline coarse floor was measured here."
            if live_pending
            else ""
        )
        reason = (
            "GO-coarse-floor: the heating-season restriction windows + Golden Week / Spring Festival "
            "holidays are securely datable; the study proceeds at coarse granularity with reduced "
            "power for the event-bound test (EDGE-001)." + pend
        )

    return ProbeReport(
        verdict=verdict,
        reason=reason,
        n_candidates=len(cand),
        n_securely_dated=len(secure),
        n_event_level=len(event_level),
        granularity_counts=granularity_counts,
        spans_era=spans_era,
        has_peak_event=has_peak,
        has_collapse_event=has_collapse,
        live_survey_pending=live_pending,
        candidates=tuple(cand),
    )


# --------------------------------------------------------------------------- assemble + validate


def assemble_calendar(
    cfg: RegimeStudyConfig | None = None,
    *,
    candidates: Sequence[CandidateEvent] | None = None,
) -> pd.DataFrame:
    """Curate the source-cited calendar frame (REQ-002/003).

    Defaults to the coarse floor; pass ``candidates`` to curate a richer surveyed set. Unsourced or
    mis-typed rows are dropped by :func:`validate_calendar` before the frame is returned, so the
    output is always loader-clean and never carries a fabricated event.
    """
    cfg = cfg or RegimeStudyConfig()
    cand = list(candidates) if candidates is not None else coarse_floor_candidates(cfg)
    df = pd.DataFrame([c.as_row() for c in cand], columns=list(CALENDAR_COLUMNS))
    df = _drop_unsourced(df)
    df = df.sort_values("start").reset_index(drop=True)
    validate_calendar(df)
    return df


def _drop_unsourced(df: pd.DataFrame) -> pd.DataFrame:
    """Exclude rows lacking a source or flagged low-confidence (never fabricate; REQ-003)."""
    if df.empty:
        return df
    keep = df["source"].astype(str).str.strip().ne("") & df["confidence"].isin({"high", "medium"})
    return df[keep].copy()


def validate_calendar(df: pd.DataFrame) -> None:
    """Enforce schema + provenance + granularity flags; raise on an unsourced/fabricated row (REQ-003).

    Checks the core four-column contract and the additive provenance columns, valid direction /
    granularity / confidence vocabularies, and that every retained row carries a non-empty source.
    """
    missing = set(CORE_COLUMNS) - set(df.columns)
    if missing:
        raise CalendarValidationError(
            f"calendar missing required columns {sorted(missing)} (need {list(CORE_COLUMNS)})."
        )
    missing_prov = set(PROVENANCE_COLUMNS) - set(df.columns)
    if missing_prov:
        raise CalendarValidationError(
            f"calendar missing provenance columns {sorted(missing_prov)} "
            f"(need {list(PROVENANCE_COLUMNS)}; REQ-003)."
        )
    for _, r in df.iterrows():
        if r["direction"] not in _VALID_DIRECTIONS:
            raise CalendarValidationError(
                f"row cause={r['cause']!r}: direction {r['direction']!r} not in {sorted(_VALID_DIRECTIONS)}."
            )
        if r["granularity"] not in _VALID_GRANULARITY:
            raise CalendarValidationError(
                f"row cause={r['cause']!r}: granularity {r['granularity']!r} not in "
                f"{sorted(_VALID_GRANULARITY)}."
            )
        if r["confidence"] not in _VALID_CONFIDENCE:
            raise CalendarValidationError(
                f"row cause={r['cause']!r}: confidence {r['confidence']!r} not in "
                f"{sorted(_VALID_CONFIDENCE)}."
            )
        if not str(r["source"]).strip():
            raise CalendarValidationError(
                f"row cause={r['cause']!r} is unsourced — exclude or source it, never fabricate "
                "(REQ-003)."
            )
        # Dates must parse and be ordered.
        s, e = pd.Timestamp(r["start"]), pd.Timestamp(r["end"])
        if pd.isna(s) or pd.isna(e) or e < s:
            raise CalendarValidationError(
                f"row cause={r['cause']!r}: bad interval start={r['start']!r} end={r['end']!r}."
            )


def write_calendar(df: pd.DataFrame, path: Path) -> Path:
    """Write the curated calendar CSV (committed deliverable, A2)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def write_sources_md(report: ProbeReport, path: Path, *, today: date | None = None) -> Path:
    """Write the T1 coverage verdict + per-event provenance narrative (REQ-001/003)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    today = today or date.today()
    lines = [
        "# Tangshan/Hebei steel curtailment calendar — sources & T1 coverage verdict",
        "",
        f"_Generated {today.isoformat()} (NOX-012 T1 feasibility spike)._",
        "",
        "## T1 go/no-go verdict",
        "",
        f"- **Verdict:** `{report.verdict}`",
        f"- **Reason:** {report.reason}",
        f"- Candidates surveyed: {report.n_candidates}; securely dated: {report.n_securely_dated}; "
        f"event-level (day/week): {report.n_event_level}.",
        f"- Spans {report.spans_era and 'the era' or 'a partial era'}; "
        f"2023-peak event: {report.has_peak_event}; 2024 event: {report.has_collapse_event}.",
        f"- Granularity counts: {report.granularity_counts}.",
    ]
    if report.live_survey_pending:
        lines += [
            "",
            "> **PENDING A LIVE RUN.** The empirical event-level source survey (a live MEE / Hebei "
            "DEE / Tangshan / CREA / Mysteel / news fetch) was **not** performed in this offline run. "
            "Only the securely-datable coarse floor (heating-season windows + public holidays) was "
            "assembled. No event-level curtailment campaign was invented; specific campaign dates "
            "that could not be sourced offline are a documented gap, not a fabricated event "
            "(REQ-003, EDGE-001). Re-run with a live survey to attempt `GO-event-level`.",
        ]
    lines += [
        "",
        "## Per-event provenance",
        "",
        "| cause | start | end | direction | granularity | confidence | source |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in report.candidates:
        lines.append(
            f"| {c.cause} | {c.start} | {c.end} | {c.direction} | {c.granularity} | "
            f"{c.confidence} | {c.source} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
