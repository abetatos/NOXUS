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
