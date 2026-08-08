# Rejected options

Designs, tools, and approaches this project considered and rejected.
Recording them prevents future sessions from re-litigating settled
questions ("why don't we just use X?" — because we tried, and here is why
not).

Rules:

- Append new entries below the marker, newest first.
- Log only options that were seriously considered and meaningfully rejected.
- Never log secrets, credentials, or sensitive operational data.
- If the conditions that caused the rejection change, add a new entry that
  reopens or reverses the question; do not delete the old one.
- Propose entries to the developer before writing them.

Entry format:

```markdown
## <YYYY-MM-DD> — <option that was rejected>

- Rejected in favor of: <the chosen approach; link the architecture/workflow entry if one exists>
- Reason: <why it lost>
- Revisit when: <conditions that would reopen the question, or "unlikely">
- Source: <task/spec/review>
```

---

## 2026-07-20 — Attributing the NO₂↔steel null to a NO₂-vs-benchmark frequency mismatch (critique #3)

- Rejected in favor of: the existing weekly compositing that already matches the CREA weekly benchmark cadence; no finer-frequency rescue is available or warranted.
- Reason: The pipeline composites per-overpass NO₂ to the CREA benchmark's native weekly cadence, so there is no uncorrected frequency mismatch to blame. Going finer does not help — TROPOMI has a single ~13:30 daily overpass, and NOX-006 built the daily footprint + Prophet decomposition and found no exploitable sub-weekly structure. Going coarser to the auxiliary national series (10-day 旬 cadence) correlates no better (see 2026-06-13 entry). The near-null (weekly r≈0.07) is not a sampling-frequency artefact.
- Revisit when: a genuinely higher-frequency NO₂ predictor exists (e.g. GEMS geostationary ~8 scans/day, NOX-007) AND a higher-frequency Tangshan physical-output benchmark exists to match it.
- Source: critique roadmap 2026-07-06 (critique #3); evidence NOX-006 (daily Prophet), NOX-009 (change-variance).

## 2026-07-20 — Attributing the null to annual deseasonalization mishandling a weekly/diurnal traffic cycle (critique #7)

- Rejected in favor of: treating the footprint signal as steel-baseload; no weekly/diurnal traffic de-cycling is needed because the cycle is empirically absent in the observable signal.
- Reason: The premise is that annual deseasonalization leaves a weekly/diurnal traffic cycle contaminating the residual. NOX-006's daily Prophet decomposition with a weekly Fourier term measured that weekly component to be FLAT (~0.5% of variance, weekday profile ≈0) — steel is baseload, so there is no weekly cycle for annual deseasonalization to mishandle. The single ~13:30 TROPOMI overpass cannot observe the diurnal traffic cycle at all. NOX-009 independently found the steel-local change-variance share ≈0, so there is no source-separation headroom a de-cycling step could recover.
- Revisit when: a diurnally-resolving sensor (GEMS, NOX-007) makes the weekday/diurnal cycle observable, reviving the flat-baseload-vs-cyclical-traffic source-separation lever.
- Source: critique roadmap 2026-07-06 (critique #7); evidence NOX-006 (weekly term flat), NOX-009 (steel-local change-variance ≈0).

## 2026-06-13 — Swapping the validation target from Tangshan BF operating rate to the national pig-iron / crude-steel output series

- Rejected in favor of: keeping the CREA Tangshan BF operating rate as the benchmark (it is the only Tangshan-specific public series).
- Reason: The auxiliary CREA series ("China: Estimated Daily Average Output: Pig Iron / Crude Steel", "BF Starting Rate (247)") are **national**, not Tangshan — a spatial mismatch against a single-cluster footprint signal — and are published on a ~10-day (旬) cadence, coarser than the weekly NO₂, not finer. Empirically they correlate no better: residual lag-0 r ∈ [−0.07, +0.00] (all ns); the only "significant" hit (crude steel yoy-sym r=−0.21, p=0.005) carries a physically absurd negative sign (more steel → less NO₂), i.e. leftover spurious trend, not activity signal.
- Revisit when: a Tangshan-specific (not national) higher-frequency physical-output series becomes available on public/free terms.
- Source: methodology audit 2026-06-13.
