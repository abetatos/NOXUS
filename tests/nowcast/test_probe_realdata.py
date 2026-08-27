"""Real-data smoke test for the T1 spike (NOX-013).

Offline/synthetic tests have repeatedly missed real bugs in this project, so the probe must also be
exercised against the actual on-disk artefacts. This test skips gracefully when the derived stores
are absent (fresh checkout without data), but runs the full probe when they exist.
"""

from __future__ import annotations

import pytest

from noxus.config.run import NowcastConfig
from noxus.nowcast.probe import probe

cfg = NowcastConfig()
_have_data = cfg.benchmark_path.exists() and cfg.no2_residual_path.exists()

pytestmark = pytest.mark.skipif(
    not _have_data, reason="derived CREA/NO2 stores not present in this environment"
)


def test_probe_runs_on_real_data_and_writes_report():
    v = probe()
    assert v.verdict in {"GO-primary", "GO-degraded", "NO-GO"}
    assert v.report_path is not None and v.report_path.exists()
    # The VIIRS proxy present in this project is the empirically-invalid FIRMS series.
    assert v.thermal_valid is False
    # NO2 features are available on disk.
    assert v.residual_available is True
    # Given NOX-009/012, the realistic real-data outcome is the honest null.
    assert v.verdict == "NO-GO"
    assert v.escalate is True
    for e in v.evidence:
        assert e.n_common >= 0
