# Tangshan/Hebei steel curtailment calendar — sources & T1 coverage verdict

_Generated 2026-07-23 (NOX-012 T1 feasibility spike)._

## T1 go/no-go verdict

- **Verdict:** `GO-coarse-floor`
- **Reason:** GO-coarse-floor: the heating-season restriction windows + Golden Week / Spring Festival holidays are securely datable; the study proceeds at coarse granularity with reduced power for the event-bound test (EDGE-001). The empirical event-level source survey (live MEE/DEE/Mysteel fetch) is PENDING a live run; only the offline coarse floor was measured here.
- Candidates surveyed: 21; securely dated: 21; event-level (day/week): 0.
- Spans the era; 2023-peak event: False; 2024 event: False.
- Granularity counts: {'day': 14, 'month': 0, 'season': 7, 'week': 0}.

> **PENDING A LIVE RUN.** The empirical event-level source survey (a live MEE / Hebei DEE / Tangshan / CREA / Mysteel / news fetch) was **not** performed in this offline run. Only the securely-datable coarse floor (heating-season windows + public holidays) was assembled. No event-level curtailment campaign was invented; specific campaign dates that could not be sourced offline are a documented gap, not a fabricated event (REQ-003, EDGE-001). Re-run with a live survey to attempt `GO-event-level`.

## Per-event provenance

| cause | start | end | direction | granularity | confidence | source |
| --- | --- | --- | --- | --- | --- | --- |
| spring_festival | 2019-02-04 | 2019-02-10 | down | day | high | PRC State Council public-holiday arrangement |
| golden_week | 2019-10-01 | 2019-10-07 | down | day | high | PRC State Council National-Day holiday arrangement |
| heating_season_restriction | 2019-11-15 | 2020-03-15 | down | season | medium | MEE/Hebei DEE heating-season industrial-restriction notices |
| spring_festival | 2020-01-24 | 2020-01-30 | down | day | high | PRC State Council public-holiday arrangement |
| golden_week | 2020-10-01 | 2020-10-07 | down | day | high | PRC State Council National-Day holiday arrangement |
| heating_season_restriction | 2020-11-15 | 2021-03-15 | down | season | medium | MEE/Hebei DEE heating-season industrial-restriction notices |
| spring_festival | 2021-02-11 | 2021-02-17 | down | day | high | PRC State Council public-holiday arrangement |
| golden_week | 2021-10-01 | 2021-10-07 | down | day | high | PRC State Council National-Day holiday arrangement |
| heating_season_restriction | 2021-11-15 | 2022-03-15 | down | season | medium | MEE/Hebei DEE heating-season industrial-restriction notices |
| spring_festival | 2022-01-31 | 2022-02-06 | down | day | high | PRC State Council public-holiday arrangement |
| golden_week | 2022-10-01 | 2022-10-07 | down | day | high | PRC State Council National-Day holiday arrangement |
| heating_season_restriction | 2022-11-15 | 2023-03-15 | down | season | medium | MEE/Hebei DEE heating-season industrial-restriction notices |
| spring_festival | 2023-01-21 | 2023-01-27 | down | day | high | PRC State Council public-holiday arrangement |
| golden_week | 2023-10-01 | 2023-10-07 | down | day | high | PRC State Council National-Day holiday arrangement |
| heating_season_restriction | 2023-11-15 | 2024-03-15 | down | season | medium | MEE/Hebei DEE heating-season industrial-restriction notices |
| spring_festival | 2024-02-10 | 2024-02-16 | down | day | high | PRC State Council public-holiday arrangement |
| golden_week | 2024-10-01 | 2024-10-07 | down | day | high | PRC State Council National-Day holiday arrangement |
| heating_season_restriction | 2024-11-15 | 2025-03-15 | down | season | medium | MEE/Hebei DEE heating-season industrial-restriction notices |
| spring_festival | 2025-01-28 | 2025-02-03 | down | day | high | PRC State Council public-holiday arrangement |
| golden_week | 2025-10-01 | 2025-10-07 | down | day | high | PRC State Council National-Day holiday arrangement |
| heating_season_restriction | 2025-11-15 | 2026-03-15 | down | season | medium | MEE/Hebei DEE heating-season industrial-restriction notices |
