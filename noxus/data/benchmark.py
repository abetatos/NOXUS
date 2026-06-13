"""Load the physical-output benchmark used for validation.

A physical-output series — monthly crude-steel production and/or blast-furnace operating rates for
the study region — is used deliberately in place of a diffusion index such as the PMI, which
measures sentiment rather than production.
"""

from __future__ import annotations

from datetime import date


def load_benchmark(start: date, end: date):
    """Load the monthly physical-output benchmark for [start, end].

    Returns a monthly series (intended as a `pandas.Series` indexed by month-end).

    Not yet implemented — this is a scaffold.
    """
    raise NotImplementedError("Benchmark loading is not yet implemented")
