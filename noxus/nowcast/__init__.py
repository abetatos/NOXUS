"""Investment-grade activity nowcast (NOX-013).

A weekly, latency-aware *surprise* nowcast of Tangshan steel activity plus its QC and
incremental-information diagnostics. The package is additive: it reuses the NOX-008 robust
inference (``noxus.validation.robust``), the lead/lag profile (``noxus.validation.leadlag``) and
the NOX-003/008 weekly-resample-then-align convention (``noxus.validation.preprocess.align_series``)
byte-for-byte.

Only the T1 feasibility spike (``noxus.nowcast.probe``) is implemented so far; it gates the rest of
the build with a go/no-go verdict.
"""

from noxus.nowcast.probe import FeatureEvidence, ProbeVerdict, probe

__all__ = ["FeatureEvidence", "ProbeVerdict", "probe"]
