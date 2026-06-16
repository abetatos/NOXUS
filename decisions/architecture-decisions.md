# Architecture decisions

Durable architectural decisions for this project. This file is project
memory: it answers "why is it built this way?" for future sessions and new
contributors.

Rules:

- Append new entries below the marker, newest first.
- Log only significant decisions. Trivial choices do not belong here.
- Never log secrets, credentials, or sensitive operational data.
- When a decision replaces an earlier one, mark the old entry
  `Status: superseded by <date — title>` instead of deleting it.
- Propose entries to the developer before writing them.

Entry format:

```markdown
## <YYYY-MM-DD> — <short decision title>

- Status: accepted
- Context: <why a decision was needed>
- Decision: <what was decided>
- Alternatives rejected: <what was considered and why it lost; details may go to rejected-options.md>
- Consequences: <what this commits the project to>
- Source: <task/spec/review that settled it>
```

---

## 2026-06-14 — Autocorrelation-robust + FDR significance is the standard for index↔benchmark correlations

- Status: accepted
- Context: The validation engine (`noxus/validation/leadlag.py`) reported the Pearson p-value from `scipy.stats.pearsonr`, which assumes i.i.d. observations, plus a per-lag ±1.96/√n white-noise band. NO₂ and the steel benchmark are smooth, strongly autocorrelated series, so that p-value overstates significance: for the highly-autocorrelated level the effective sample size is ~225 of 377 (weekly) / ~39 of 90 (monthly). The 2026-06-14 probe (`analysis/autocorr_significance.py`) showed every positive NO₂-deseasonalised↔BF "SIG" finding from the deep exploration collapses once serial dependence is accounted for.
- Decision: Judge index↔benchmark correlations with **autocorrelation-robust** significance — effective-N (first-order Bayley–Hammersley **and** full-order Newey-West), a **moving-block bootstrap** CI of r, and a **block-permutation** null p — plus a **Benjamini–Hochberg FDR** correction across any family of tests (scales × variants × lags). A finding is "robust" only if the FDR-adjusted permutation p clears α **and** the bootstrap CI excludes 0. This lives in `noxus/validation/robust.py` as tested, reusable code; the existing `correlate`/`lead_lag`/`verify_sign` contracts are unchanged (additive).
- Alternatives rejected: keeping the naïve Pearson p (over-rejects under autocorrelation); first-order effective-N only (under-corrects the seasonal/high-order autocorrelation of the level); reporting per-test significance without multiplicity control (invites scale-shopping / p-hacking).
- Consequences: any new correlation claim in the project must pass the robust + FDR bar; the naïve p is shown only alongside, to expose the gap. The lead-lag white-noise band stays as an eyeballing aid, not a significance test.
- Source: NOX-008 (specs/spatial-scale-sensitivity), T3/T4; probe analysis/autocorr_significance.py.

## 2026-06-14 — No robust NO₂↔steel coupling exists at any reasonable spatial scale (Tangshan)

- Status: accepted
- Context: Every prior result used one fixed spatial scale (0.25° AOI buffer, native ~5 km grid). Parubets & Naito (2025) warn that the NO₂↔activity relationship can be significant at 0.25° but vanish/flip at 0.1°; a competing intuition said a *tighter* AOI might improve source isolation (steel is only ~30–43% of Tangshan NO₂, Wen 2024). NOX-008 swept AOI extent (0.25 vs 0.10° buffer, clip) × grid resolution (native vs block-averaged 0.10/0.15/0.25°, aggregation only — never interpolation) over the existing full-series cube and judged each scale with the robust + FDR standard above.
- Decision: Report a **hardened null**: of 128 (scale × variant × lag) tests, **zero** are robust after FDR; the only naïve-significant cells are monthly peak-lag artefacts that all fall to "fragile (naive-only)". The tight 0.10° AOI does **not** rescue the signal (Δ|r| ≈ 0.02–0.05, never significant), and the small native↔coarse sign flips are noise (|r| < 0.13). The conclusion is therefore *no robust coupling at any reasonable scale*, not *no coupling at one scale*.
- Alternatives rejected: presenting the most favourable scale (e.g. tight-AOI level r≈−0.13 weekly) as a finding without the FDR/multiplicity caveat (p-hacking); interpreting coarse↔fine sign flips as a Parubets-style scale effect (they are noise-level here).
- Consequences: the spatial-scale lever is closed for the relative-index paradigm; remaining avenues stay flux divergence (NOX-005), a diurnal sensor (GEMS, NOX-007), or the regime/event framing — not finer/coarser aggregation. Coarsening adds no information; it only changes background dilution.
- Source: NOX-008 real run (data/derived/scale_sensitivity.csv; docs/figures/exploration/scale_sensitivity_findings.txt), T8.

## 2026-06-13 — The negative levels NO₂↔BF-rate correlation is spurious; never report it as a finding

- Status: accepted
- Context: The intensity-model decomposition tempted a "decoupling" headline: footprint NO₂ levels correlate −0.218 with the CREA BF operating rate, and the fitted secular trend s(t) correlates −0.728 (naive p≈1e-63). A methodological audit (2026-06-13) showed this is a textbook spurious-regression artifact: both series are non-stationary (ADF p≈0.13–0.14, unit root) and not cointegrated (Engle–Granger p=0.058); lag-1 autocorrelation is 0.99/0.92 so the effective sample size is ~18, not 377 (autocorr-corrected p≈0.001 with a huge CI); and first-differencing BOTH sides collapses the relationship to r=−0.03 (p=0.55).
- Decision: Do **not** report the negative levels/trend correlation as a statistical finding. The decoupling phenomenon may be stated only **qualitatively**, anchored in external literature (Li 2024 emission-intensity decline), explicitly flagged as *consistent with* — not *established by* — these data. The single defensible quantitative result remains the **NULL on the activity proxy**, which is robust because it is the part that survives detrending (week-over-week symmetric r=+0.006; yoy-symmetric r=+0.11 marginal).
- Alternatives rejected: promoting r=−0.73 as a "decoupling" headline (spurious; reviewer-fatal). See rejected-options.md.
- Consequences: validation reporting must lead with the robust null, not the levels correlation; any levels relationship shown must carry the non-stationarity/non-cointegration caveat. The intensity-model's real contribution is framed as removing the difference-filter degree of freedom (flat smoothness_sweep), not as detecting decoupling.
- Source: methodology audit 2026-06-13 (post NOX-003.1 / NOX-004); reproduced on data/derived/no2/steel_intensity_decomposition.parquet vs benchmark_tangshan_bf_operating_rate.parquet.

## 2026-06-13 — NO₂ at native ~5 km resolution; gridding = temporal compositing, not swath oversampling

- Status: accepted
- Context: The openEO product is delivered already gridded at ~0.05°×0.035° (≈5 km, ≈TROPOMI native), and its swath-footprint geometry is not exposed. Papers that show ~1 km NO₂ maps (Beirle, Kim) achieve that by **oversampling many raw L2 overpasses** onto a fine grid — which needs L2 footprints we do not have from openEO. A single overpass is genuinely ~5 km.
- Decision: Work at **native ~5 km** resolution. NOX-002b "gridding" is therefore **temporal compositing** (stack overpasses → weekly/monthly mean over valid cells + coverage threshold), not swath→grid oversampling. This is adequate for the project's **cluster-level** activity signal (per-plant separation is already infeasible: 14/26 plants share a pixel). Fine ~1 km oversampling is **deferred** to a future raw-L2 path (CDSE OData/S3), only worth it for per-source attribution which is out of scope here.
- Alternatives rejected: (a) `resample_spatial` to a finer grid in openEO — pure interpolation, adds no real resolution; (b) pivoting now to raw-L2 oversampling — much heavier, only pays off for sub-cluster/per-plant work that isn't the goal.
- Consequences: the analysis-ready cube and any predictor are ~5 km; claims stay at cluster/sub-cluster scale. The L2/kernel path remains the route if finer attribution is ever required (NOX-003).
- Source: NOX-002b (specs/tropomi-no2-gridding); developer chose option A on 2026-06-13 after inspecting the native grid.

## 2026-06-13 — TROPOMI acquisition via openEO server-side subsetting (auth by CDSE password grant, web fallback)

- Status: accepted
- Context: Over the Tangshan AOI there are ~3,734 Sentinel-5P L2 NO₂ overpasses (2019→2026) at ~609 MB/granule; downloading full orbits ≈ 2.2 TB, infeasible on a free account. Only the AOI window is needed (~0.2–1 GB).
- Decision: Acquire NO₂ through **openEO on CDSE with server-side spatial subsetting** (filter to the AOI bbox before download), not full-granule download. **Authentication** uses the credentials already in `.env`: mint a CDSE access token via the **OIDC password grant** (`CDSE_USERNAME`/`CDSE_PASSWORD`, `client_id=cdse-public`) and hand it to openEO via `authenticate_oidc_access_token()`; if that fails, **fall back to `authenticate_oidc()`** (refresh token, else interactive device flow that opens the web). Adds the `openeo` Python package.
- Alternatives rejected: full-granule download (2.2 TB); Sentinel Hub or GEE (return L3 rasters, less L2 control; Sentinel Hub additionally requires a separate OAuth client). For auth, a dedicated machine OAuth client (client-credentials) is kept optional, not required — the existing account password suffices.
- Consequences: kept dataset is ~0.2–1 GB; processing uses the free 10k openEO credits/month (batch across months for the full record); secrets stay in `.env` (never logged/committed). L2 fidelity (averaging kernel) depends on what the openEO collection exposes — verified at implementation (NOX-002a open question Q1).
- Source: NOX-002a (specs/tropomi-no2-acquisition); volume check + auth confirmed against CDSE/openEO docs 2026-06-13.

## 2026-06-13 — Tangshan steel as the calibration case; phased extrapolation with decreasing ground-truth anchor

- Status: accepted (strategic / roadmap)
- Context: The nearest precedent (Morris–Zhang 2019) used national OMI NO₂ to validate GDP; NOXUS needs a differentiated, defensible contribution beyond a higher-resolution re-run. Tangshan steel uniquely combines three things that are rarely available together: located point sources (GEM Global Iron & Steel Tracker — 25 operating integrated plants, ~120 Mtpa), a cluster-level **physical** benchmark (CREA blast-furnace operating rate), and a NOₓ budget dominated by iron-and-steel. That makes it a rare "supervised" case where the NO₂→activity mapping can actually be learned and checked against ground truth.
- Decision: Treat Tangshan steel as the **calibration/validation anchor**: learn the NO₂→activity transfer function and its uncertainty there and validate it out-of-sample. Frame the contribution as a **validated, transferable pipeline**, and extrapolate in later phases with explicitly **decreasing ground-truth anchoring**: (1) Tangshan steel [now]; (2) other steel clusters / national steel cross-check; (3) other point-source sectors (cement, power, smelters — Beirle-resolvable); (4) regions with poor official data (the payoff, where NO₂ becomes the estimate, not the validated quantity); (5) a sector-activity leading-indicator index. Each extrapolation must carry validation where a benchmark exists and **explicit uncertainty where it does not**; the transfer function is NOT assumed constant across sector/region.
- Alternatives rejected: (a) positioning the novelty solely on "single cluster vs national" — weaker, closer to a re-run of Morris–Zhang; (b) blind extrapolation assuming a universal NO₂↔activity relationship — rejected per Montgomery–Holloway (the link is context/income-dependent, not universal).
- Consequences: commits later phases to re-validation rather than blind transfer; strengthens the differentiation from Morris–Zhang (a reusable validated method, not a one-off audit). Attribution-based steps remain limited to large, isolated emitters.
- Source: strategy discussion 2026-06-13 (follow-on to the NO₂-attribution roadmap).

## 2026-06-13 — Out-of-sample Diebold–Mariano skill is the standard for any "leading indicator" claim

- Status: accepted (principle; to be applied when the validation module is built)
- Context: The literature already shows NO₂ correlates with economic activity in sample; that bar is not the contribution of NOXUS. The project's stated posture is an honest test of *leading-indicator value*, with the null reported if found. In-sample correlation / R² is easy to produce and is not evidence that a signal leads.
- Decision: When the validation module is built, any claim that the NO₂ signal leads an official series must rest on **out-of-sample forecast skill** — a pseudo-real-time comparison of an AR-only baseline against AR + lagged NO₂, judged by the Diebold–Mariano test — not on in-sample correlation. CCF/Granger are descriptive diagnostics only, and the null must be reported explicitly when AR+NO₂ does not beat AR.
- Alternatives rejected: in-sample correlation / R² or contemporaneous CCF as the headline evidence — rejected: does not demonstrate lead and is already cleared in the literature.
- Consequences: the future validation module reports RMSE(AR) vs RMSE(AR+NO₂) and a DM statistic; a null is a first-class, reportable outcome. (Recorded now while fresh; the validation code is intentionally deferred until the NO₂ predictor/attribution stage exists — see history NOX-001.)
- Source: NOX-001 scope discussion (specs/crea-benchmark-validation).

## 2026-06-13 — Benchmark source-of-record: CREA public CSV snapshot

- Status: accepted
- Context: The physical-output benchmark for the Tangshan cluster (weekly blast-furnace operating rate) is needed at cluster resolution. The only cluster-level series is sourced from a commercial terminal (WIND / China United Steel Network), but CREA republishes it freely as a Google-Sheet CSV export. The project constraint is public, free, reproducible data.
- Decision: Ingest the benchmark from CREA's public CSV export, pinning results to **dated snapshots** committed under `data/raw/benchmark/`, and select the benchmark column by **exact header name** (never by position) to survive source column drift. Treat `0.00` as not-reported (NA), and skip absent auxiliary columns (only the primary column is mandatory).
- Alternatives rejected: (a) HTML scraping of CREA reports — fragile, no clean series; (b) commercial APIs (CEIC / Mysteel) — paywalled, violates the public/free criterion; (c) trusting a live endpoint at analysis time — not reproducible. The commercial provenance (WIND/Custeel) is documented in `references.bib` and the README; the public/free criterion is met at the access layer.
- Consequences: reproducibility relies on committed snapshots; if CREA's sheet moves, past results still reproduce from snapshots while future refreshes need a new source. Implemented in `noxus/data/benchmark.py`.
- Source: NOX-001 (specs/crea-benchmark-validation), review.html.

## 2026-06-13 — Cloud-gap handling: aggregation + minimum-coverage threshold as core; reconstruction only as robustness check

- Status: accepted
- Context: TROPOMI NO₂ is retrieved in the solar-backscatter UV-Vis (DOAS, ~405–465 nm); clouds optically shield the boundary layer where the steel cluster's NO₂ accumulates. There is no cloud-penetrating channel (microwave/IR) that measures boundary-layer NO₂ the way passive microwave measures snow depth, so the cloud problem is one of **non-random sampling**, not of choosing a different band. A literature review found that every economic/attribution precedent in our register (Kim 2023, Kondragunta 2021, Mao 2025, Beirle 2021/2023, Li–Zheng 2023, Parubets–Naito 2025) handles clouds by **QA filtering + temporal aggregation**, not by ML/statistical reconstruction; the only reconstruction precedent we hold (Park 2025, GEMS-GAN) is filed P3.
- Decision: The core signal pipeline handles cloud gaps with the field-standard approach: (1) QA filtering (`qa_value ≥ 0.75` recommended for NO₂; relax to ≥0.5 only with justification), (2) temporal aggregation (oversampling / monthly or moving-average means over the Tangshan AOI), and (3) a **minimum valid-coverage threshold** before a period's mean is admitted (cf. Kondragunta's ≥25% valid-pixels / ≥25% valid-days rule). Gap-filling/reconstruction (e.g. EOF+XGBoost, partial-convolution NN, GAN) is **not** part of the production signal; it is used only as a sensitivity/robustness layer to test whether the signal survives the cloud-sampling bias. Reconstructed fields must never feed the dependent variable used in validation against physical output.
- Alternatives rejected: (a) ML/statistical reconstruction as the core signal — rejected: injects model-prior artifacts into the dependent variable, confounding the validation against steel output. (b) Cloud-slicing — rejected: yields free-/upper-tropospheric NO₂, not the boundary-layer combustion signal we need. (c) Naive temporal interpolation across gaps — rejected: cloud gaps correlate with season/weather (already flagged as a known limitation in README). See `rejected-options.md`.
- Consequences: Commits the pipeline to a coverage-threshold parameter (to be set in `noxus/config/`) and to reporting per-period valid-coverage alongside the signal. Robustness work must keep reconstruction code paths isolated from the production signal. Geostationary GEMS sampling (more daily overpasses → fewer gaps at source) is the preferred mitigation over modeled gap-filling.
- Source: literature review on cloud handling (2026-06-13 session); README "Known limitations — Cloud gaps".

## 2026-06-13 — ERA5 meteorology source: Copernicus CDS (server-side AOI subset); ARCO-ERA5 Zarr rejected

- Status: accepted
- Context: NOX-003 meteo normalisation needs ERA5 10 m wind + boundary-layer height over the small Tangshan AOI for the whole 2019→present era (a long time-series over a tiny spatial box). Two public/free paths: the anonymous ARCO-ERA5 Zarr store on Google Cloud (no account) vs the Copernicus CDS API (`reanalysis-era5-single-levels`, free account + one-time licence).
- Decision: Ingest ERA5 from the **Copernicus CDS**, subsetting AOI/era/overpass-hour **server-side**, written to a dated local `.nc` snapshot (analysis reads only the snapshot, NFR-001). The full era is fetched **one CDS request per year** (a single multi-year request exceeds the CDS per-request cost limit). Credentials read **by name only** from `.env` (`CDSAPI_KEY`, optional `CDSAPI_URL`; `~/.cdsapirc` fallback) — never read/logged/committed. Verified live 2026-06-13 (3-day probe 42 KB; full era 8148 hourly steps, 3.8 MB).
- Alternatives rejected: **ARCO-ERA5 Zarr** — its `chunk-1` layout (1 timestep × full global 721×1440 field per chunk, ~4.15 MB) forces downloading a full global field per timestep to extract a small AOI: ~30 GB (overpass hour only) to ~760 GB (hourly) for our series, the wrong access pattern. ARCO is excellent for whole-map-per-time access, poor for a point/small-AOI over a long era. (The CDS itself flags a sibling `reanalysis-era5-single-levels-timeseries` ARCO dataset optimised for point-over-time — a possible future optimisation.) The ARCO code path was removed entirely (no residual fallback), and `gcsfs`/`zarr` deps dropped.
- Consequences: adds one authenticated egress (free CDS account) handled by reference; `cdsapi` dependency; a one-time licence acceptance is a prerequisite (403 "required licences not accepted" otherwise). Snapshot is tiny and cached.
- Source: NOX-003 T5/T15 (specs/steel-activity-attribution, open question Q3); live verification 2026-06-13.

## 2026-06-13 — Deseasonalisation default: year-over-year (yoy); double-differencing demoted

- Status: accepted
- Context: The relative-index validation needs a deseasonalisation that removes the annual cycle AND the secular policy/retrofit trend (which decouples NO₂ from steel output, Li 2024) without destroying the activity signal. A real sensitivity run (2026-06-13, full 2019–2026) compared methods × frequency × meteo.
- Decision: Default `deseason_method = "yoy"` (single year-over-year change). Findings: raw **levels** give a spurious **negative** correlation with the BF rate (retrofit decoupling, monthly r=−0.35); **yoy** and **stl** recover a faint **positive** activity signal (monthly yoy r≈0.19, peak +0.29 at lag +6 mo); **yoy-double-diff erased** it (r≈0, over-aggressive — Kim 2023 warned, they used seasonal adjustment not double-diff). Monthly > weekly (Morris–Zhang: weekly noisy). All four methods (`yoy`/`stl`/`yoy-double-diff`/`none`) remain config-selectable for replicability; nothing removed.
- Alternatives rejected: `yoy-double-diff` as default (demonstrably erases the signal here); `none`/level (dominated by the decoupling trend, wrong sign). Anchor/event normalisation (Li–Zheng) not implemented (deferred).
- Consequences: the headline result remains a weak, below-bar (0.5–0.75) positive after detrending — an honest, mechanistically-explained near-null (Montgomery 2018; Morris–Zhang 2019). Tighter footprint (5 km) beats 15 km. Reproduce via `analysis/preliminary_signal.py`; see `docs/preliminary-results.html`.
- Source: NOX-003 sensitivity run 2026-06-13 (specs/steel-activity-attribution, open question Q4).

## 2026-06-13 — Recovering activity: explicit emission-intensity model (CV-selected trend); blind differencing kept only for replicability

- Status: accepted
- Context: NOX-003 showed the footprint NO₂ correlates **negatively** with the BF rate in levels (monthly r=−0.35, the retrofit/decoupling signature, Li 2024) and only weakly positive after blind year-over-year detrending (r≈0.19), while double-differencing **erased** the signal. The blind difference filters either under-remove the secular intensity decline or over-remove it; the trend and the activity partly share timescales.
- Decision: Recover activity by modelling the emission-intensity decline **explicitly** (NOX-003.1): fit a smoothness-controlled secular trend `s(t)` to the meteo-normalised footprint signal and take the **residual as the activity proxy** (Li 2024 activity/intensity decomposition). The trend smoothness (effective df) is selected by **blocked time-series cross-validation on the NO₂ series alone — the benchmark is never consulted** (anti-p-hacking, the central guard). A **smoothness-sensitivity sweep** (residual r vs df) is mandatory output so robustness is shown, not assumed; `s(t)` is emitted as a reportable diagnostic (the decoupling itself). Default estimator is a regression spline (df = basis columns, exact, numpy-only); LOESS optional. The method is an **additive** new `deseason_method = "intensity-model"`; all blind methods (`yoy`/`stl`/`yoy-double-diff`/`none`) are retained verbatim for replicability.
- Alternatives rejected: (a) selecting trend df by maximising correlation with the benchmark — rejected, manufactures a correlation; (b) replacing yoy/double-diff — rejected, replicability (keep all modes); (c) a full multi-term econometric decomposition / change-point trend — deferred (one trend term + CV + sweep is the bounded scope); (d) a new GAM dependency — avoided (numpy spline) unless approved.
- Consequences: on the partial cube the explicit model confirms the decoupling (levels r=−0.35) and recovers a weak residual (monthly r≈0.14, comparable to yoy) — the gain is methodological honesty (one fewer researcher degree of freedom; the decoupling reported as a finding), not a higher number. CV selects a near-linear trend (df=2); the sweep decays monotonically with df (no high-df plateau). The full-series re-run (after the complete TROPOMI fetch) remains the go/no-go test for NOX-005. Implemented in `noxus/signal/intensity.py`; see `docs/preliminary-results.html` §5.
- Source: NOX-003.1 (specs/steel-intensity-model); follows the NOX-003 deseasonalisation decision above.

## 2026-06-13 — Catalyst paradigm: use cluster NO2 as an event marker, not a continuous tracker (NOX-004)

- Status: accepted
- Context: NOX-003/003.1 established the cluster NO2 is a weak continuous tracker of the BF operating rate (monthly r~0.14-0.19, below the 0.5-0.75 bar). The developer reframed the use case (2026-06-13): not a time-series correlation but a CATALYST / event marker for steel-exposed markets.
- Decision: Detect discrete production events (sharp surges/drops) on the NOX-003.1 intensity-detrended + meteo-normalised residual, match them to ground-truth production events (CREA BF-rate jumps COMBINED WITH a public curtailment calendar), and study market abnormal returns around them for global miners (BHP/RIO/VALE) + a steel ETF (SLX) (Chinese ferrous futures deferred, Q1, no free API). Headline metrics are precision/recall/lead-time/CAR with CIs, NOT Pearson r. Disciplines baked in: strict NO-LOOK-AHEAD (causal expanding median/MAD baselines; first tradeable session = overpass + latency), MULTIPLE-TESTING transparency (defaults fixed before viewing returns; n_tests recorded), strengthened meteo control (ventilation index + stagnation flag, the cheap in-framework lever instead of flux-divergence NOX-005), public/free data only, honest NULL valid. Market prices via yfinance (approved). Detector = robust MAD-z + CUSUM (in-house, no new dep). NO2 events + price snapshots gitignored.
- Alternatives rejected: (a) continuing to chase a continuous correlation (weak, below bar); (b) flux-divergence to clean events (NOX-005, expensive, not needed for step-change events); (c) tuning detector/window/instrument to maximise CAR (p-hacking); (d) a paywalled ferrous-futures feed.
- Consequences: first real run on the PARTIAL cube is an honest NULL -- 25 NO2 events detected but only 4/47 production events matched (precision 0.16, recall 0.09, median lead -2 d), market CARs all straddle zero. The machinery works; the event-marker is weak on partial/noisy data. The full TROPOMI fetch (NOX-003.1 T10) + threshold tuning is the real test. Implemented in noxus/catalyst/; report at data/derived/catalyst_summary.txt.
- Source: NOX-004 (specs/steel-event-catalyst); developer reframe + scoping answers 2026-06-13.

## 2026-06-13 — Daily Prophet decomposition: implemented; harmonic-K1 (weekly) stays the method of record (NOX-006)

- Status: accepted
- Context: To beat the ~40% steel mean-share SNR ceiling without flux divergence (NOX-005), the one temporal lever is the day-of-week cycle (steel baseload vs traffic weekly cycle), visible only at daily resolution. Built a daily NO2 footprint (re-aggregate per-overpass at freq=D; 1984/2720 valid days, 73%, weekdays balanced) + a Prophet decomposition (trend + yearly + weekly Fourier, deterministic MAP, lazy-imported, harmonic fallback).
- Decision: Keep the capability (deseason_method='prophet' + 'harmonic'; daily path) but make HARMONIC-K1 (annual Fourier on the intensity residual, weekly cadence) the METHOD OF RECORD. Empirical result on the real (partial) data: (a) the weekly component is FLAT — variance removed 0.5%, weekday profile ~0 -> steel is baseload, but precisely therefore there is NO weekly cycle to separate (also limited by TROPOMI's single ~13:30 overpass, which can't resolve a traffic commute cycle); (b) the daily-Prophet residual couples to BF at best-regime r=0.65, BELOW harmonic-K1 weekly (0.78) and intensity (0.73) — daily adds per-point noise without adding weekly signal; (c) yearly-only ~= yearly+weekly (0.648); (d) no usable quarterly/four-month cycle — the BF rate's sub-annual power is ~semiannual (heating-season curtailment on/off, already an annual K=2 harmonic), not aligned with the NO2 residual's ~2-3.6-month wiggles; adding quarterly/four-month Prophet terms removes ~0.5% variance and does not improve coupling.
- Alternatives rejected: daily-Prophet as the method of record (worse coupling + noisier); a weekly source-separation claim (no weekly cycle in the data); a quarterly/four-month seasonality (no usable shared power).
- Consequences: Prophet is available + tested (deterministic, lazy-imported) for future use (e.g. a geostationary/multi-overpass sensor like GEMS could revive the weekly lever), but the pipeline default + reporting stand on harmonic-K1 / intensity. Honest null on the marginal value; the full TROPOMI fetch + the event/regime framing remain the path. Implemented in noxus/signal/prophet_deseason.py; probe analysis/daily_prophet_probe.py.
- Source: NOX-006 (specs/daily-prophet-deseason); real partial-cube run 2026-06-13.
