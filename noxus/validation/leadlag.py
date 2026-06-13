"""Test whether the activity index leads the official-output benchmark.

The objective is an honest test, not a positive finding. A rigorous null — the index does not lead
the benchmark once seasonality, mandated production curtailments, and cloud-driven data gaps are
controlled for — is a valid result and is reported as such.
"""

from __future__ import annotations


def test_lead(index, benchmark):
    """Estimate the lead/lag relationship between `index` and `benchmark`.

    Not yet implemented — this is a scaffold. The implementation will deseasonalise both series,
    control for known curtailment periods, and report cross-correlation / Granger-style lead
    statistics with confidence intervals — including the null case.
    """
    raise NotImplementedError("Lead/lag validation is not yet implemented")
