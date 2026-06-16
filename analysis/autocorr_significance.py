"""How much of the NO2<->BF 'significance' survives autocorrelation? (exploratory follow-up).

Motivation (developer-directed 2026-06-14): the validation engine reports Pearson r with the
scipy p-value (noxus/validation/leadlag.py) and the lead-lag profile uses a per-lag white-noise band
(+-1.96/sqrt(n)). Both assume i.i.d. observations. NO2 and the BF operating rate are smooth, strongly
autocorrelated series, so the naive p-value/band overstate significance: the effective number of
independent observations is far below n. This probe quantifies the gap and shows which 'SIG' findings
in docs/figures/exploration/findings.txt survive an autocorrelation-aware test.

For each NO2 variant vs the BF rate (weekly and monthly), at lag 0 and at the peak-|r| lag, it reports:
  - naive Pearson r, p, and the naive 95% CI;
  - the lag-1 autocorrelation of each series and the effective sample size n_eff
    (Bartlett / Bayley-Hammersley first-order: n_eff = n * (1 - a*b)/(1 + a*b));
  - the p-value recomputed on n_eff degrees of freedom (autocorrelation-adjusted t-test);
  - a moving-block bootstrap 95% CI for r (block length ~ n^(1/3), preserves within-block memory);
  - a block-permutation p-value (shuffle blocks of x against y => null that keeps each series'
    autocorrelation but destroys the cross-correlation).

Reproduce (after the upstream artifacts exist):
    uv run python analysis/autocorr_significance.py
Writes docs/figures/exploration/autocorr_significance.txt. Reads the same gitignored artifacts as
analysis/deep_exploration.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from analysis.deep_exploration import build_master, load_all

OUT = Path("docs/figures/exploration/autocorr_significance.txt")
RNG = np.random.default_rng(20260614)
N_BOOT = 5000


def _lag1(x: np.ndarray) -> float:
    """Lag-1 autocorrelation of a 1-D array (0 if degenerate)."""
    if len(x) < 3 or np.std(x) == 0:
        return 0.0
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def n_eff_first_order(x: np.ndarray, y: np.ndarray) -> float:
    """Effective sample size under first-order autocorrelation (Bayley & Hammersley 1946).

    n_eff = n * (1 - a*b) / (1 + a*b), with a,b the lag-1 autocorrelations. Clamped to [2, n].
    """
    n = len(x)
    a, b = _lag1(x), _lag1(y)
    denom = 1.0 + a * b
    if denom <= 0:
        return 2.0
    return float(np.clip(n * (1.0 - a * b) / denom, 2.0, n))


def p_from_r(r: float, n_eff: float) -> float:
    """Two-sided p-value for Pearson r using an effective n (t-test on n_eff-2 df)."""
    df = n_eff - 2.0
    if df <= 0 or abs(r) >= 1.0:
        return float("nan")
    t = r * np.sqrt(df / (1.0 - r * r))
    return float(2.0 * stats.t.sf(abs(t), df))


def moving_block_bootstrap_ci(x: np.ndarray, y: np.ndarray, block: int) -> tuple[float, float]:
    """95% CI for Pearson r via moving-block bootstrap of the *paired* series (preserves memory)."""
    n = len(x)
    n_blocks = int(np.ceil(n / block))
    starts_pool = np.arange(0, n - block + 1)
    rs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        starts = RNG.choice(starts_pool, size=n_blocks, replace=True)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        xb, yb = x[idx], y[idx]
        rs[i] = np.corrcoef(xb, yb)[0, 1] if np.std(xb) > 0 and np.std(yb) > 0 else 0.0
    return float(np.nanpercentile(rs, 2.5)), float(np.nanpercentile(rs, 97.5))


def block_permutation_p(x: np.ndarray, y: np.ndarray, block: int, r_obs: float) -> float:
    """Two-sided p under a block-permutation null: shuffle blocks of x, keep y fixed.

    Cutting x into contiguous blocks and reordering them destroys the x<->y cross-correlation while
    approximately preserving x's own autocorrelation, giving a null calibrated for serial dependence.
    """
    n = len(x)
    edges = list(range(0, n, block)) + [n]
    blocks = [x[edges[i] : edges[i + 1]] for i in range(len(edges) - 1)]
    count = 0
    for _ in range(N_BOOT):
        order = RNG.permutation(len(blocks))
        xp = np.concatenate([blocks[k] for k in order])
        r = np.corrcoef(xp, y)[0, 1] if np.std(xp) > 0 else 0.0
        if abs(r) >= abs(r_obs):
            count += 1
    return float((count + 1) / (N_BOOT + 1))


def _shift_pair(x: pd.Series, b: pd.Series, lag: int) -> tuple[np.ndarray, np.ndarray]:
    d = pd.concat({"x": x.shift(lag), "b": b}, axis=1).dropna()
    return d["x"].to_numpy(float), d["b"].to_numpy(float)


def best_lag(x: pd.Series, b: pd.Series, max_lag: int) -> tuple[int, float]:
    best_k, best_r = 0, 0.0
    for k in range(-max_lag, max_lag + 1):
        xv, bv = _shift_pair(x, b, k)
        if len(xv) < 6 or np.std(xv) == 0 or np.std(bv) == 0:
            continue
        r = float(np.corrcoef(xv, bv)[0, 1])
        if abs(r) > abs(best_r):
            best_k, best_r = k, r
    return best_k, best_r


def analyse(master: pd.DataFrame, variants: list[str], freq_label: str, max_lag: int) -> list[str]:
    lines = [f"\n{'=' * 78}", f"{freq_label.upper()}  (NO2 variant vs BF operating rate)", "=" * 78]
    header = (
        f"{'variant':10s} {'lag':>4s} {'n':>4s} {'r':>7s} "
        f"{'p_naive':>9s} {'neff':>5s} {'p_neff':>9s} {'boot95_CI':>16s} {'p_block':>8s}  verdict"
    )
    for at_best in (False, True):
        lines.append(f"\n-- {'peak-|r| lag' if at_best else 'lag 0'} --")
        lines.append(header)
        for v in variants:
            d = master[[v, "bf"]].dropna()
            if len(d) < 12:
                continue
            if at_best:
                lag, _ = best_lag(d[v], d["bf"], max_lag)
            else:
                lag = 0
            x, y = _shift_pair(d[v], d["bf"], lag)
            n = len(x)
            if n < 8 or np.std(x) == 0 or np.std(y) == 0:
                continue
            res = stats.pearsonr(x, y)
            r, p_naive = float(res.statistic), float(res.pvalue)
            neff = n_eff_first_order(x, y)
            p_neff = p_from_r(r, neff)
            block = max(2, int(round(n ** (1 / 3))))
            lo, hi = moving_block_bootstrap_ci(x, y, block)
            p_block = block_permutation_p(x, y, block, r)

            naive_sig = p_naive < 0.05
            robust_sig = (p_block < 0.05) and not (lo <= 0.0 <= hi)
            if naive_sig and robust_sig:
                verdict = "robust"
            elif naive_sig and not robust_sig:
                verdict = "FRAGILE (naive-only)"
            elif not naive_sig and robust_sig:
                verdict = "robust (naive missed)"
            else:
                verdict = "ns"
            lines.append(
                f"{v:10s} {lag:+4d} {n:4d} {r:+7.3f} {p_naive:9.4f} "
                f"{neff:5.1f} {p_neff:9.4f} [{lo:+.2f},{hi:+.2f}]{'':4s} {p_block:8.4f}  {verdict}"
            )
        lines.append(
            f"(naive CI for reference, lag0 only shown above; block len = round(n^1/3); "
            f"{N_BOOT} bootstrap/permutation draws)"
        )
    return lines


def main() -> None:
    no2_level, no2_resid, bench, aux_series, mkt_wide = load_all()
    mw = build_master("W", 52, no2_level, no2_resid, bench, aux_series, mkt_wide)
    mm = build_master("ME", 12, no2_level, no2_resid, bench, aux_series, mkt_wide)
    variants = ["no2_level", "no2_resid", "no2_yoy", "no2_stl"]

    out = [
        "AUTOCORRELATION-AWARE SIGNIFICANCE OF NO2 <-> BF RATE",
        "Does the naive Pearson p-value survive serial dependence?",
        "naive = scipy.pearsonr (i.i.d. assumption, what leadlag.py reports today)",
        "neff   = first-order effective-N t-test | boot95 = moving-block bootstrap CI of r",
        "p_block= block-permutation null p-value | 'robust' needs p_block<.05 AND boot CI excludes 0",
    ]
    out += analyse(mw, variants, "weekly", max_lag=16)
    out += analyse(mm, variants, "monthly", max_lag=8)
    text = "\n".join(out) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(text)
    print(f"written -> {OUT}")


if __name__ == "__main__":
    main()
