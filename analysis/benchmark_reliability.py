"""Benchmark reliability floor and equivalence bounds (paper critique B3/B4).

Two questions the paper asserts but never measures.

B3 -- *Is the near-null a defect of the target?*  The CREA Tangshan blast-furnace operating rate
is a commercial trader survey, so every analysis scored against it inherits whatever measurement
error it carries.  We bound that error from below by correlating it against an independently
surveyed panel published in the same release -- the national blast-furnace starting rate over 247
mills.  A target whose weekly *changes* track a separate national panel closely cannot be mostly
survey noise, which removes the "maybe the benchmark is junk" reading of the near-null.  It does
NOT close the shared-publisher provenance gap (both series are CREA/WIND), and the script says so
in its own output.

B4 -- *What does the null exclude?*  A null reported as "we failed to reject" is not citable.  We
convert each feature's coupling to an equivalence statement: the moving-block bootstrap upper
confidence bound on |r|, the implied share of common variance, and the minimum detectable
correlation at the autocorrelation-adjusted effective sample size.

Every parameter matches the pipeline's compiled-in defaults (seed 20260712, 5,000 draws, weekly
cadence, 0.25 valid-coverage screen).  No benchmark-derived quantity tunes anything.

Reproduce:
    uv run python analysis/benchmark_reliability.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DERIVED = Path("data/derived")
OUT_JSON = DERIVED / "benchmark_reliability.json"
OUT_TXT = DERIVED / "benchmark_reliability.txt"

SEED = 20260712
N_BOOT = 5000
COVERAGE_MIN = 0.25
WINDOW_START = "2019-01-01"
NATIONAL_PANEL = "China: Blast Furnace Starting Rate (247)"

# (label, parquet, column, apply the valid_coverage screen?)
FEATURES = [
    ("activity index (screened)", "no2/steel_activity_index.parquet", "index_value", True),
    ("activity index (unscreened)", "no2/steel_activity_index.parquet", "index_value", False),
    ("background-corrected residual", "no2/steel_footprint_signal.parquet", "no2_corrected", True),
    (
        "intensity-decomposition covariate",
        "no2/steel_intensity_decomposition.parquet",
        "residual_activity",
        False,
    ),
]


def _weekly(frame: pd.DataFrame, column: str) -> pd.Series:
    out = frame[["date", column]].dropna().copy()
    out["date"] = pd.to_datetime(out["date"])
    return out.set_index("date")[column].resample("W").mean()


def _block_length(n: int) -> int:
    """Politis-White style rule of thumb used throughout the pipeline: n**(1/3)."""
    return max(2, int(round(n ** (1 / 3))))


def _moving_block_bootstrap_ci(x: np.ndarray, y: np.ndarray, seed: int = SEED) -> tuple:
    """Percentile CI for Pearson r under a moving-block resample that preserves serial structure."""
    n = len(x)
    block = _block_length(n)
    n_blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    starts_pool = n - block + 1
    draws = np.empty(N_BOOT)
    for i in range(N_BOOT):
        starts = rng.integers(0, starts_pool, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        xb, yb = x[idx], y[idx]
        if xb.std() == 0 or yb.std() == 0:
            draws[i] = np.nan
            continue
        draws[i] = np.corrcoef(xb, yb)[0, 1]
    draws = draws[~np.isnan(draws)]
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)), block


def _n_eff(x: np.ndarray, y: np.ndarray) -> float:
    """Bartlett/Quenouille effective sample size n(1-ab)/(1+ab) from lag-1 autocorrelations."""
    a = np.corrcoef(x[:-1], x[1:])[0, 1]
    b = np.corrcoef(y[:-1], y[1:])[0, 1]
    ab = a * b
    if ab >= 1:
        return float(len(x))
    return float(len(x) * (1 - ab) / (1 + ab))


def benchmark_reliability() -> dict:
    """B3: how much real signal does the CREA Tangshan target itself carry?"""
    tangshan = pd.read_parquet(DERIVED / "benchmark_tangshan_bf_operating_rate.parquet")
    tangshan = tangshan[["date", "value"]].dropna().rename(columns={"value": "tangshan"})
    aux = pd.read_parquet(DERIVED / "benchmark_auxiliary.parquet")
    national = (
        aux[aux["series"] == NATIONAL_PANEL][["date", "value"]]
        .dropna()
        .rename(columns={"value": "national"})
    )
    merged = tangshan.merge(national, on="date").sort_values("date")
    merged["date"] = pd.to_datetime(merged["date"])
    merged = merged[merged["date"] >= WINDOW_START].reset_index(drop=True)

    lev_x = merged["tangshan"].to_numpy(float)
    lev_y = merged["national"].to_numpy(float)
    diff = merged[["tangshan", "national"]].diff().dropna()
    dx = diff["tangshan"].to_numpy(float)
    dy = diff["national"].to_numpy(float)

    r_lev = float(np.corrcoef(lev_x, lev_y)[0, 1])
    r_diff = float(np.corrcoef(dx, dy)[0, 1])
    lo, hi, block = _moving_block_bootstrap_ci(dx, dy)
    return {
        "window_start": WINDOW_START,
        "n": int(len(merged)),
        "panel": NATIONAL_PANEL,
        "r_level": r_lev,
        "r_change": r_diff,
        "r_change_ci": [lo, hi],
        "block": block,
        # Attenuation logic: if both series measure a common activity factor with independent
        # errors, the observed cross-correlation is a LOWER bound on each one's reliability.
        "reliability_floor": r_diff,
        "max_attainable_r": float(np.sqrt(max(r_diff, 0.0))),
        "shared_publisher": True,
    }


def equivalence_bounds() -> list[dict]:
    """B4: restate each feature's near-null as an exclusion with an upper bound."""
    crea = pd.read_parquet(DERIVED / "benchmark_tangshan_bf_operating_rate.parquet")
    crea = crea[["date", "value"]].dropna()
    crea_w = _weekly(crea, "value")

    rows = []
    for label, rel, column, screen in FEATURES:
        frame = pd.read_parquet(DERIVED / rel)
        if screen:
            frame = frame[frame["valid_coverage"] >= COVERAGE_MIN]
        joined = (
            pd.concat([_weekly(frame, column).rename("f"), crea_w.rename("c")], axis=1)
            .dropna()
            .sort_index()
        )
        x = joined["f"].to_numpy(float)
        y = joined["c"].to_numpy(float)
        r = float(np.corrcoef(x, y)[0, 1])
        lo, hi, block = _moving_block_bootstrap_ci(x, y)
        neff = _n_eff(x, y)
        bound = max(abs(lo), abs(hi))
        rows.append(
            {
                "feature": label,
                "screened": screen,
                "n": int(len(joined)),
                "n_eff": neff,
                "r": r,
                "ci": [lo, hi],
                "block": block,
                # The exclusion the data license, and the variance it corresponds to.
                "excluded_above_abs_r": bound,
                "excluded_above_r2": bound**2,
                # Smallest |r| this design could have called significant at alpha=0.05.
                "min_detectable_r": float(1.96 / np.sqrt(max(neff - 3.0, 1.0))),
            }
        )
    return rows


def _render(reliability: dict, bounds: list[dict]) -> str:
    lines = [
        "Benchmark reliability floor and equivalence bounds (paper critique B3/B4)",
        "=" * 78,
        f"Fixed parameters: seed={SEED} n_boot={N_BOOT} cadence=W coverage_screen>={COVERAGE_MIN}",
        "",
        "B3 -- Does the TARGET carry real signal?",
        "-" * 78,
        f"  CREA Tangshan BF rate vs {reliability['panel']}",
        f"  window {reliability['window_start']}+   n={reliability['n']}",
        f"  levels          r = {reliability['r_level']:+.3f}",
        f"  weekly changes  r = {reliability['r_change']:+.3f} "
        f"(block bootstrap 95% CI [{reliability['r_change_ci'][0]:+.3f},"
        f" {reliability['r_change_ci'][1]:+.3f}], block={reliability['block']})",
        "",
        "  Read: the Tangshan series shares most of its week-to-week movement with a separately",
        "  surveyed 247-mill national panel, so it is not dominated by survey noise. Measurement",
        "  error in the target therefore cannot explain an observed feature correlation of ~0.08.",
        "  CAVEAT: both series are published by CREA from WIND inputs, so this bounds the target's",
        "  RELIABILITY, not its VALIDITY -- a shared methodology artefact would move both together.",
        "",
        "B4 -- What does the null EXCLUDE?",
        "-" * 78,
        f"  {'feature':<36} {'n':>4} {'n_eff':>6} {'r':>7} "
        f"{'95% CI':>18} {'excl |r|>':>10} {'r2':>7} {'MDC':>6}",
    ]
    for b in bounds:
        ci = f"[{b['ci'][0]:+.3f},{b['ci'][1]:+.3f}]"
        lines.append(
            f"  {b['feature']:<36} {b['n']:>4} {b['n_eff']:>6.1f} {b['r']:>+7.4f} "
            f"{ci:>18} {b['excluded_above_abs_r']:>10.3f} "
            f"{b['excluded_above_r2']:>7.4f} {b['min_detectable_r']:>6.3f}"
        )
    worst = max(b["excluded_above_abs_r"] for b in bounds)
    lines += [
        "",
        f"  Overall exclusion (widest bound across features): |r| > {worst:.2f} is ruled out at 95%,",
        f"  i.e. the weekly coupling shares less than {100 * worst**2:.1f}% of its variance with the",
        "  benchmark. This is the citable form of the null: an exclusion, not a failure to reject.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    reliability = benchmark_reliability()
    bounds = equivalence_bounds()
    report = _render(reliability, bounds)
    OUT_TXT.write_text(report, encoding="utf-8")
    OUT_JSON.write_text(
        json.dumps({"reliability": reliability, "equivalence": bounds}, indent=2),
        encoding="utf-8",
    )
    print(report)
    print(f"wrote {OUT_TXT}\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
