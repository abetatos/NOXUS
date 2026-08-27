"""NOX-011 — decisive VIIRS triangulation: NO2 vs CREA vs the physical heat (REQ-020..024, 031).

The one lever against critique #1 (the "fraudulent / unverifiable target"): correlate an independent,
hard-to-falsify **physical** activity proxy — VIIRS thermal activity over the steel facilities —
against both the FEATURE (footprint NO2) and the TARGET (CREA benchmark), under the NOX-008
autocorrelation-robust significance. The diagnosis localises the weak link either way:

    NO2<->VIIRS | CREA<->VIIRS | diagnosis
    -----------------------------------------------------------------------------------------
       no       |     yes      | NO2 blind at this scale; TARGET sound (VIIRS<->CREA holds)
       yes      |     no       | TARGET (CREA) was the weak link (re-interprets every prior null)
       yes      |     yes      | coherent: NO2 tracks the physical heat and CREA is sound
       no       |     no       | inconclusive: VIIRS tracks NEITHER series — the proxy may be
                |              | uninformative/invalid here and cannot arbitrate feature-vs-target

Significance is **reused, never re-implemented**: ``robust_corr`` (effective-N, moving-block bootstrap
CI, block-permutation null) + ``bh_fdr`` across the correlation family + ``lead_lag`` for the profile,
all from ``noxus.validation``. The thermal series is built benchmark-blind upstream (``signal.thermal``);
this module only *reads* the pre-registered series and aligns it on the common overlap window (EDGE-003).

Reproduce (after ``noxus fetch-viirs`` + ``noxus build-thermal-series`` have written the raw store):
    uv run python analysis/viirs_triangulation.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from noxus.config.run import ThermalTriangulationConfig
from noxus.validation.leadlag import lead_lag
from noxus.validation.robust import RobustCorr, bh_fdr, robust_corr

# Diagnosis labels (feature-vs-target). Kept as constants so tests key on stable strings.
DIAG_TARGET_WEAK = (
    "TARGET (CREA) was the weak link: NO2<->VIIRS holds but CREA<->VIIRS does not — the physical heat "
    "tracks NO2, so a prior near-null points at the benchmark, retroactively re-interpreting it."
)
DIAG_NO2_BLIND = (
    "NO2 blind at this scale; TARGET sound: VIIRS<->CREA holds while NO2 does not track the physical "
    "heat — the FEATURE (NO2 at this scale) is the weak link, consistent with the SNR-bound finding."
)
DIAG_COHERENT = (
    "Coherent: NO2<->VIIRS and CREA<->VIIRS both hold — NO2 is a valid physical proxy and the target "
    "is sound; any prior near-null lies elsewhere in the chain."
)
DIAG_PROXY_INVALID = (
    "Inconclusive — the VIIRS thermal proxy does not track NO2 or CREA (neither NO2<->VIIRS nor "
    "CREA<->VIIRS is significant), so the proxy may be uninformative/invalid for this cluster (e.g. "
    "active-fire dominated by non-target sources such as seasonal crop-residue burning rather than "
    "the year-round furnaces). It therefore cannot arbitrate feature-vs-target — this is NOT a "
    "'feature weak' claim. Informative as a proxy-validity result."
)
# Back-compat alias: prior name for the both-null branch (kept so any external reference resolves).
DIAG_HONEST_NULL = DIAG_PROXY_INVALID


@dataclass(frozen=True)
class Link:
    """One correlation link (r + robust significance + lead/lag) for the triangulation table."""

    name: str
    corr: RobustCorr
    peak_lag: int
    peak_r: float
    p_adj: float
    significant: bool


@dataclass(frozen=True)
class TriangulationReport:
    """Feature-vs-target triangulation result on the common overlap window (REQ-023)."""

    n_common: int
    links: tuple[Link, ...]
    diagnosis: str
    firms_diagnosis: str | None = None
    firms_consistent: bool | None = None

    def link(self, name: str) -> Link | None:
        return next((link_ for link_ in self.links if link_.name == name), None)


def _align_common(**series: pd.Series) -> pd.DataFrame:
    """Inner-join the named series on their common (jointly non-NaN) window (EDGE-003)."""
    frame = pd.concat({k: _as_series(v) for k, v in series.items()}, axis=1)
    return frame.dropna()


def _as_series(s: pd.Series) -> pd.Series:
    if not isinstance(s.index, pd.DatetimeIndex):
        s = s.copy()
        s.index = pd.to_datetime(s.index)
    return s[~s.index.duplicated(keep="last")].sort_index()


def _link(name: str, x: pd.Series, y: pd.Series, cfg: ThermalTriangulationConfig) -> Link:
    """Compute one robust correlation + lead/lag link (p_adj/significant filled in later)."""
    rc = robust_corr(x, y, n_boot=cfg.n_boot, n_perm=cfg.n_perm, seed=cfg.seed)
    ccf = lead_lag(x, y, max_lag=cfg.max_lag)
    return Link(
        name=name,
        corr=rc,
        peak_lag=ccf.peak_lag,
        peak_r=ccf.peak_r,
        p_adj=float("nan"),
        significant=False,
    )


def interpret(no2_significant: bool, crea_significant: bool) -> str:
    """Map the (NO2<->VIIRS, CREA<->VIIRS) significance pattern to the feature-vs-target diagnosis."""
    if no2_significant and crea_significant:
        return DIAG_COHERENT
    if no2_significant and not crea_significant:
        return DIAG_TARGET_WEAK
    if crea_significant and not no2_significant:
        return DIAG_NO2_BLIND
    return DIAG_PROXY_INVALID


def triangulate(
    thermal: pd.Series,
    no2: pd.Series,
    crea: pd.Series,
    cfg: ThermalTriangulationConfig | None = None,
    firms: pd.Series | None = None,
) -> TriangulationReport:
    """Triangulate the thermal series against NO2 and CREA under robust significance (REQ-020..024).

    Aligns each pair on its common window, computes ``robust_corr`` + ``lead_lag``, applies BH-FDR
    across the correlation family (including the FIRMS cross-check pair when supplied), and emits the
    feature-vs-target diagnosis (incl. the honest-null branch). Significance is reused from
    ``robust.py`` — not re-implemented.
    """
    cfg = cfg or ThermalTriangulationConfig()

    primary = _align_common(thermal=thermal, no2=no2, crea=crea)
    n_common = int(len(primary))
    links = [
        _link("no2_viirs", primary["no2"], primary["thermal"], cfg),
        _link("crea_viirs", primary["crea"], primary["thermal"], cfg),
    ]

    firms_links: list[Link] = []
    if firms is not None:
        fpaired = _align_common(firms=firms, no2=no2, crea=crea)
        if len(fpaired):
            firms_links = [
                _link("no2_firms", fpaired["no2"], fpaired["firms"], cfg),
                _link("crea_firms", fpaired["crea"], fpaired["firms"], cfg),
            ]

    all_links = links + firms_links
    p_family = [link_.corr.p_perm for link_ in all_links]
    rejected, p_adj = bh_fdr(p_family, alpha=cfg.fdr_alpha)
    all_links = [
        Link(
            name=link_.name,
            corr=link_.corr,
            peak_lag=link_.peak_lag,
            peak_r=link_.peak_r,
            p_adj=float(p_adj[i]),
            significant=bool(rejected[i]) and n_common >= cfg.min_overlap,
        )
        for i, link_ in enumerate(all_links)
    ]
    links = all_links[:2]
    firms_links = all_links[2:]

    no2_sig = links[0].significant
    crea_sig = links[1].significant
    diagnosis = interpret(no2_sig, crea_sig)

    firms_diag = firms_consistent = None
    if firms_links:
        firms_diag = interpret(firms_links[0].significant, firms_links[1].significant)
        firms_consistent = firms_diag == diagnosis

    return TriangulationReport(
        n_common=n_common,
        links=tuple(all_links),
        diagnosis=diagnosis,
        firms_diagnosis=firms_diag,
        firms_consistent=firms_consistent,
    )


def report_to_frame(report: TriangulationReport) -> pd.DataFrame:
    """Tidy one-row-per-link table for the committable triangulation CSV (REQ-031)."""
    rows = []
    for link_ in report.links:
        rc = link_.corr
        rows.append(
            {
                "link": link_.name,
                "n": rc.n,
                "n_common": report.n_common,
                "r": rc.r,
                "p_naive": rc.p_naive,
                "n_eff_nw": rc.n_eff_nw,
                "p_eff_nw": rc.p_eff_nw,
                "boot_lo": rc.boot_lo,
                "boot_hi": rc.boot_hi,
                "p_perm": rc.p_perm,
                "p_adj_bh": link_.p_adj,
                "significant": link_.significant,
                "peak_lag": link_.peak_lag,
                "peak_r": link_.peak_r,
                "diagnosis": report.diagnosis,
                "firms_diagnosis": report.firms_diagnosis or "",
                "firms_consistent": report.firms_consistent,
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- orchestration (I/O)


def _load_series_from_parquet(path: Path, column: str, date_col: str = "date") -> pd.Series:
    df = pd.read_parquet(path)
    if date_col not in df.columns:
        # A DatetimeIndex frame (e.g. the thermal series) — first data column by default.
        col = column if column in df.columns else df.columns[0]
        return _as_series(df[col])
    idx = pd.DatetimeIndex(pd.to_datetime(df[date_col]))
    col = column if column in df.columns else [c for c in df.columns if c != date_col][0]
    return _as_series(pd.Series(df[col].to_numpy(float), index=idx))


def _resample(series: pd.Series, freq: str) -> pd.Series:
    return series.resample(freq).mean()


def run_triangulation(cfg: ThermalTriangulationConfig | None = None) -> Path:
    """Load the thermal series + NO2 footprint + CREA, triangulate, write the committable CSV.

    Developer-run (needs the real thermal series produced by ``build-thermal-series``). A null is a
    valid, informative outcome (NFR-005). Never reads the benchmark during the series build — only here,
    at the correlation step.
    """
    from noxus.signal.thermal import thermal_activity

    cfg = cfg or ThermalTriangulationConfig()
    series_path = Path(cfg.out_dir) / cfg.series_name
    thermal_df = pd.read_parquet(series_path)
    thermal = _resample(thermal_activity(thermal_df), cfg.freq)

    no2 = _resample(
        _load_series_from_parquet(Path(cfg.no2_series_path), cfg.no2_series_column), cfg.freq
    )
    crea = _resample(_load_series_from_parquet(Path(cfg.benchmark_path), "value"), cfg.freq)

    report = triangulate(thermal, no2, crea, cfg)
    out = Path(cfg.out_dir) / cfg.results_name
    out.parent.mkdir(parents=True, exist_ok=True)
    report_to_frame(report).to_csv(out, index=False)
    return out


def main() -> int:
    cfg = ThermalTriangulationConfig()
    series_path = Path(cfg.out_dir) / cfg.series_name
    if not series_path.exists():
        print(f"[noxus] {series_path} missing — run 'noxus build-thermal-series' first.")
        return 1
    out = run_triangulation(cfg)
    df = pd.read_csv(out)
    print(f"[noxus] triangulation table -> {out}")
    if len(df):
        print(f"[noxus] diagnosis: {df['diagnosis'].iloc[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
