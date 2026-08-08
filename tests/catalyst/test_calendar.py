"""Calendar assembly + T1 feasibility spike tests (NOX-012, AT1/AT2, ERR-002/003).

Fully offline + deterministic. Source probing is mocked via injected candidate lists (no network).
"""

from __future__ import annotations

import pandas as pd
import pytest

from noxus.catalyst import calendar as C
from noxus.catalyst.groundtruth import load_curtailment_calendar
from noxus.config.run import RegimeStudyConfig


def _campaign(cause, start, end, direction="down", granularity="day", confidence="high"):
    return C.CandidateEvent(
        cause=cause,
        start=start,
        end=end,
        direction=direction,
        source="Reuters / MEE notice (test)",
        url="https://example.test/",
        granularity=granularity,
        confidence=confidence,
    )


# --------------------------------------------------------------------------- AT1 (T1 probe + go/no-go)


def test_probe_offline_coarse_floor_is_go_coarse_with_live_pending():
    """Offline: only the securely-datable coarse floor -> GO-coarse-floor, live survey pending (AT1)."""
    report = C.probe_sources(RegimeStudyConfig())
    assert report.verdict == C.GO_COARSE
    assert report.go is True
    assert report.live_survey_pending is True
    # The structural floor (holidays + heating seasons) never, by itself, claims event-level.
    assert report.n_event_level == 0
    assert report.n_securely_dated == report.n_candidates > 0


def test_probe_event_level_when_campaign_events_present():
    """A rich campaign set spanning the era with 2023 + 2024 events -> GO-event-level (AT1)."""
    cfg = RegimeStudyConfig(min_event_level=3)
    cands = C.coarse_floor_candidates(cfg) + [
        _campaign("blue_sky_campaign", "2019-06-01", "2019-06-10"),
        _campaign("sudden_shutdown", "2023-06-05", "2023-06-12"),
        _campaign("anti_smog_enforcement", "2024-03-01", "2024-03-08"),
    ]
    report = C.probe_sources(cfg, candidates=cands, live_survey=True)
    assert report.verdict == C.GO_EVENT
    assert report.has_peak_event and report.has_collapse_event
    assert report.n_event_level >= 3
    assert report.live_survey_pending is False


def test_probe_no_go_when_nothing_datable_is_escalation_not_silent():
    """No securely-dated candidates at all -> NO-GO surfaced as an escalation (AT1 negative)."""
    unsourced = C.CandidateEvent(
        cause="rumoured_cut",
        start="2023-06-01",
        end="2023-06-05",
        direction="down",
        source="",  # unsourced -> not securely dated
        url="",
        granularity="day",
        confidence="low",
    )
    report = C.probe_sources(RegimeStudyConfig(), candidates=[unsourced], live_survey=True)
    assert report.verdict == C.NO_GO
    assert report.go is False
    assert "escalate" in report.reason.lower()


def test_probe_invents_no_event():
    """The probe never fabricates: candidate count equals what was surveyed (AT1 negative)."""
    cfg = RegimeStudyConfig()
    floor = C.coarse_floor_candidates(cfg)
    report = C.probe_sources(cfg, candidates=floor, live_survey=True)
    assert report.n_candidates == len(floor)


def test_sources_md_records_verdict_and_pending(tmp_path):
    report = C.probe_sources(RegimeStudyConfig())
    out = C.write_sources_md(report, tmp_path / "SOURCES.md")
    text = out.read_text()
    assert "GO-coarse-floor" in text
    assert "PENDING A LIVE RUN" in text
    assert "fabricated" in text.lower()


# --------------------------------------------------------------------------- AT2 (assemble + loader compat)


def test_assemble_calendar_loads_through_unchanged_loader(tmp_path):
    """The curated CSV loads cleanly through the UNCHANGED NOX-004 loader (four-column) (AT2)."""
    cfg = RegimeStudyConfig()
    df = C.assemble_calendar(cfg)
    for col in C.CALENDAR_COLUMNS:
        assert col in df.columns  # provenance columns present
    path = tmp_path / "cal.csv"
    C.write_calendar(df, path)
    loaded = load_curtailment_calendar(path)
    # The loader selects exactly its four-column contract; provenance columns are ignored.
    assert list(loaded.columns) == ["start", "end", "direction", "cause"]
    assert len(loaded) == len(df)
    assert pd.api.types.is_datetime64_any_dtype(loaded["start"])


def test_validate_rejects_unsourced_row():
    """An unsourced row is excluded/flagged, never fabricated/kept (AT2, REQ-003)."""
    bad = pd.DataFrame(
        [
            {
                "start": "2023-06-01",
                "end": "2023-06-05",
                "direction": "down",
                "cause": "mystery",
                "source": "",  # unsourced
                "url": "",
                "granularity": "day",
                "confidence": "high",
            }
        ]
    )
    with pytest.raises(C.CalendarValidationError):
        C.validate_calendar(bad)


def test_assemble_drops_unsourced_candidate():
    """assemble_calendar silently drops the unsourced candidate rather than keeping it (AT2)."""
    cfg = RegimeStudyConfig()
    cands = C.coarse_floor_candidates(cfg) + [
        C.CandidateEvent("mystery", "2023-06-01", "2023-06-05", "down", "", "", "day", "low")
    ]
    df = C.assemble_calendar(cfg, candidates=cands)
    assert "mystery" not in set(df["cause"])


def test_validate_rejects_bad_direction_and_granularity():
    base = {
        "start": "2023-06-01",
        "end": "2023-06-05",
        "direction": "down",
        "cause": "x",
        "source": "s",
        "url": "u",
        "granularity": "day",
        "confidence": "high",
    }
    bad_dir = pd.DataFrame([{**base, "direction": "sideways"}])
    with pytest.raises(C.CalendarValidationError):
        C.validate_calendar(bad_dir)
    bad_gran = pd.DataFrame([{**base, "granularity": "hour"}])
    with pytest.raises(C.CalendarValidationError):
        C.validate_calendar(bad_gran)


def test_loader_raises_on_malformed_calendar(tmp_path):
    """A calendar missing start/end/direction raises the existing actionable ValueError (ERR-002)."""
    p = tmp_path / "bad.csv"
    pd.DataFrame({"cause": ["x"], "start": ["2023-01-01"]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_curtailment_calendar(p)


def test_loader_absent_calendar_degrades_to_empty(tmp_path):
    """An absent calendar path returns a typed empty frame (BF-jumps-only degrade, ERR-002)."""
    empty = load_curtailment_calendar(tmp_path / "nope.csv")
    assert list(empty.columns) == ["start", "end", "direction", "cause"]
    assert len(empty) == 0
