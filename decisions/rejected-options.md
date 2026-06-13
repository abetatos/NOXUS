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

## 2026-06-13 — Swapping the validation target from Tangshan BF operating rate to the national pig-iron / crude-steel output series

- Rejected in favor of: keeping the CREA Tangshan BF operating rate as the benchmark (it is the only Tangshan-specific public series).
- Reason: The auxiliary CREA series ("China: Estimated Daily Average Output: Pig Iron / Crude Steel", "BF Starting Rate (247)") are **national**, not Tangshan — a spatial mismatch against a single-cluster footprint signal — and are published on a ~10-day (旬) cadence, coarser than the weekly NO₂, not finer. Empirically they correlate no better: residual lag-0 r ∈ [−0.07, +0.00] (all ns); the only "significant" hit (crude steel yoy-sym r=−0.21, p=0.005) carries a physically absurd negative sign (more steel → less NO₂), i.e. leftover spurious trend, not activity signal.
- Revisit when: a Tangshan-specific (not national) higher-frequency physical-output series becomes available on public/free terms.
- Source: methodology audit 2026-06-13.
