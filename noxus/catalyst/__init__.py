"""Steel NO2 event catalyst (NOX-004).

Uses the cluster NO2 as an event marker, not a continuous tracker: detect discrete production events
on the NOX-003.1 residual, match them against ground-truth production events (CREA BF-rate jumps +
curtailment calendar), and study market abnormal returns around them. Strict no-look-ahead and
multiple-testing discipline; an honest null is a valid outcome.
"""
