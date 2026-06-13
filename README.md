<p align="center">
  <strong>NOXUS</strong>
</p>

<p align="center">
  A public, reproducible pipeline from satellite NO₂ to a steel-sector activity signal
</p>

<p align="center">
  <sub>NOₓ (the combustion tracer we measure) + nexus (the bridge from the signal to the macro)</sub>
</p>

---

NOXUS attributes the Sentinel-5P/TROPOMI NO₂ column over a single industrial cluster to its
combustion activity and tests, transparently, whether that signal leads the official output
statistic — reporting the null result if that is what the data show. It uses only public, free
data so the work is reproducible by non-institutional researchers.

This README opens with the research motivation; build and usage instructions follow.

## Contents

- [Motivation and Scope](#motivation-and-scope)
  - [Why this project](#why-this-project)
  - [The chosen signal and the actual gap](#the-chosen-signal-and-the-actual-gap)
  - [Case study](#case-study)
  - [Posture on the outcome](#posture-on-the-outcome)
  - [Known limitations](#known-limitations)
  - [References](#references)
- [Installation](#installation)
- [Repository layout](#repository-layout)
- [Data sources](#data-sources)
- [Status](#status)
- [License](#license)

---

# Motivation and Scope

## Why this project

Remote-sensing-derived alternative data for financial and macroeconomic signals is an established
field (Donaldson & Storeygard, 2016), but its most-cited applications—counting vehicles in retail
parking lots, gauging oil-tank fill levels, monitoring port traffic—are mature. They have been
studied extensively and, where they carried predictive value, that value has plausibly been competed
away as the methods became standard. Replicating them adds little.

We adopt an explicit selection criterion: a candidate signal is worth pursuing only if it satisfies
three conditions simultaneously. It must be (a) extractable from **public, free** data, so that the
work is reproducible and accessible to non-institutional researchers rather than gated behind paid
feeds; (b) potentially **leading** with respect to the corresponding official statistic; and (c)
**not already broadly exploited** through some other channel.

Several candidates were evaluated and set aside. Narrative macro and credit-risk signals have no
physical satellite footprint—they are macro analysis, not remote sensing, and fall outside scope.
Aggregate regional NO₂ as a proxy for economic activity is already documented, including the natural
experiment of the COVID-19 lockdowns (Zheng et al., 2021; Kondragunta et al., 2021, who correlate
Q2-2020 U.S. unemployment with the TROPOMI NO₂ column) and work relating NO₂ to economic output
(Montgomery & Holloway, 2018; Li & Zheng, 2023; Parubets & Naito, 2025; Ezran et al., 2023); the
aggregate angle is therefore not novel. Mining output via optical imagery is valid but is already exploited
commercially using paid high-resolution data, and the pixel-to-tonnage chain is indirect because it
depends on ore grade, which is not observable from orbit. A further domain consideration rules out
reading a specific miner's output from smelting activity: extraction and smelting are geographically
decoupled—concentrate is shipped, and smelting is concentrated in China—so the satellite signature
of a smelter does not map back to any one mine.

## The chosen signal and the actual gap

NO₂ (Sentinel-5P/TROPOMI) measures **combustion**, not objects, which makes it the appropriate
instrument for thermal processes such as integrated steelmaking rather than for extraction. The
source-attribution technique itself already exists in the atmospheric literature: point-source NOₓ
catalogues derived from the divergence of the TROPOMI NO₂ flux, which already resolve power plants,
metal smelters, and industrial areas (Beirle et al., 2021, 2023).

The gap is therefore **not** the attribution technique, and it is **not** the general idea of turning
pollution into an activity indicator—commercial nowcasting vendors already market
industrial-pollution and steel-sector indices framed as leading signals, and a thin academic
literature already links NO₂ to official activity measures (Ezran et al., 2023, who identify only a
small number of prior studies of this kind). What is absent from the open literature is the specific
intersection: a **public, reproducible** pipeline that attributes the NO₂ signal to a **single
industrial cluster** and tests, transparently, whether it leads the official statistic—**with the
null result reported if that is what the data show.** The commercial products are closed: they sell
estimates, not methods, and they do not publish failures. The closest open methodological precedent
attributes clustered TROPOMI NO₂ to an economic series for a different sector—maritime shipping
validated against a container-throughput index (Kim et al., 2023)—which this work adapts rather than
reproduces.

The closest precedent **in intent** is Morris & Zhang (2019), who validate China's official output
statistics directly against satellite NO₂—the same use of pollution as an audit on Chinese
production that motivates this project. Their study, however, operates at the **national** level
using the older OMI instrument and validates **GDP**. NOXUS narrows the unit of analysis to a single
steel cluster, uses the higher-resolution **TROPOMI** record, and is framed as a candidate **leading
indicator** operable against metals-sector instruments rather than as a retrospective validation of
national accounts. Stating this delta explicitly is deliberate: without it, the contribution would
read as a re-run of Morris & Zhang.

## Case study

We focus on the Tangshan (Hebei) steel cluster, chosen for its high geographic concentration and
strong, continuous thermal signature; the iron-and-steel sector is a dominant NOₓ source in this
region (Scientific Reports, 2024, doi:10.1038/s41598-024-63338-8). Tangshan is China's "steel capital":
the Global Iron & Steel Tracker (March 2026) places **25 operating integrated blast-furnace plants**
(~120 Mtpa of crude-steel capacity) within the prefecture, packed into roughly a 1.5°×1° box — a
genuinely dense, steel-dominated cluster, which is what makes single-cluster attribution and a
steel-specific NO₂ signal tractable here rather than at national scale. A secondary motivation is data
integrity: official Chinese output figures for this region have been documented as subject to
misreporting (S&P Global Market Intelligence, 2021), and satellite NO₂ improves on conventional
proxies precisely where data manipulation degrades national accounts (Ezran et al., 2023).
Validation is against a physical-output benchmark—monthly crude-steel production (World Steel
Association; China NBS) and/or blast-furnace operating rates (CREA)—rather than a diffusion index
such as the PMI, which measures sentiment rather than production. The intended unit of analysis is
macro (steel-sector activity and metals-related instruments), not the prediction of any single
equity.

## Posture on the outcome

The objective is an honest test, not a positive finding. A rigorous null—the signal does not lead
the official statistic once seasonality, the confounding effect of mandated production curtailments,
and cloud-driven data gaps are controlled for—is a valid result and will be reported as such.

## Known limitations

- **Temporal sampling.** Sentinel-5P/TROPOMI provides roughly one overpass per day at local early
  afternoon, which constrains the achievable signal frequency and aliases sub-daily variation.
- **Cloud gaps.** Cloud cover removes usable observations non-randomly, producing gaps that are
  themselves correlated with season and weather and must not be naively interpolated.
- **Attribution validity.** Source attribution is only reliable for large, isolated emitters; coarse
  pixel size (relative to, e.g., nighttime lights) and the presence of co-located traffic, heating,
  and other industry mean the steel-specific component cannot be cleanly isolated everywhere, and the
  pixel-to-tonnage relationship is not linear in crude-steel output.
- **Confounding by policy.** In this region, mandated emission curtailments tied to air-quality
  targets are strong and seasonal, and can mimic a production signal.
- **Income-dependent, non-universal link.** The NO₂↔economic-activity relationship is neither
  universal nor of fixed sign. Across the 100 most populous global cities, Montgomery & Holloway
  (2018) find a positive NO₂↔urban-product relationship in only 38 of 56 low-income cities, while
  most middle-income cities show no clear relationship—consistent with an environmental Kuznets
  pattern in which the sign depends on income level. The case for a usable signal therefore rests on
  the specific industrial setting (a concentrated, high-income, heavy-combustion cluster), not on a
  general law.

## References

Beirle, S., Borger, C., Dörner, S., Eskes, H., Kumar, V., de Laat, A., et al. (2021). Catalog of NOₓ
emissions from point sources as derived from the divergence of the NO₂ flux for TROPOMI. *Earth
System Science Data*, 13, 2995–3015. https://doi.org/10.5194/essd-13-2995-2021

Beirle, S., et al. (2023). Improved catalog of NOₓ point source emissions (version 2). *Earth System
Science Data*, 15, 3051–3073. https://doi.org/10.5194/essd-15-3051-2023

Donaldson, D., & Storeygard, A. (2016). The view from above: Applications of satellite data in
economics. *Journal of Economic Perspectives*, 30(4), 171–198. https://doi.org/10.1257/jep.30.4.171

Ezran, I. A. S., Morris, S. D., Rama, M. G., & Riera-Crichton, D. (2023). Measuring Global Economic
Activity Using Air Pollution. *Policy Research Working Paper 10445*, World Bank.
https://ideas.repec.org/p/wbk/wbrwps/10445.html

Hersbach, H., Bell, B., Berrisford, P., Hirahara, S., Horányi, A., Muñoz-Sabater, J., et al. (2020).
The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society*, 146(730),
1999–2049. https://doi.org/10.1002/qj.3803 *(ERA5 reference paper; NOXUS uses ERA5 single-levels wind
+ boundary-layer height, accessed via the Copernicus Climate Data Store, as the meteorology regressed
out of the cluster NO₂ signal.)*

Kim, J., Emmerich, M. T. M., Voors, R., Ording, B., & Lee, J.-S. (2023). A Systematic Approach to
Identify Shipping Emissions Using Spatio-Temporally Resolved TROPOMI Data. *Remote Sensing*, 15(13),
3453. https://doi.org/10.3390/rs15133453

Kondragunta, S., Wei, Z., McDonald, B. C., Goldberg, D. L., & Tong, D. Q. (2021). COVID-19 Induced
Fingerprints of a New Normal Urban Air Quality in the United States. *Journal of Geophysical
Research: Atmospheres*, 126(17), e2021JD034797. https://doi.org/10.1029/2021JD034797 *(peer-reviewed
publication of the same group's COVID NO₂↔activity work; supersedes the unrefereed conference
abstract Wei et al., 2020, AGU Fall Meeting, which lacked a journal/volume.)*

Li, H., & Zheng, B. (2023). TROPOMI NO₂ Shows a Fast Recovery of China's Economy in the First
Quarter of 2023. *Environmental Science & Technology Letters*.
https://doi.org/10.1021/acs.estlett.3c00386

Montgomery, A., & Holloway, T. (2018). Assessing the relationship between satellite-derived NO₂ and
economic growth over the 100 most populous global cities. *Journal of Applied Remote Sensing*, 12(4),
042607. https://doi.org/10.1117/1.JRS.12.042607 *(Positive NO₂↔urban-product link found in 38 of 56
low-income cities; ~36 middle-income cities show no clear relationship—an environmental-Kuznets
pattern, i.e. the link is income-dependent, not universal.)*

Morris, S. D., & Zhang, J. (2019). Validating China's Output Data Using Satellite Observations.
*Macroeconomic Dynamics*, 23(8), 3327–3354. https://doi.org/10.1017/S1365100518000056 *(National-level
OMI NO₂ used to validate Chinese output/GDP; the closest precedent in intent to this project, which
differs by working at single-cluster scale with TROPOMI and as a leading indicator. Often cited as
"2018" after its online-first / SSRN working-paper version; the journal article is 2019.)*

Parubets, S., & Naito, H. (2025). Predicting economic activity using atmospheric nitrogen dioxide
(NO₂) satellite data: Evidence from local economic indicators in Japan. *PLOS ONE*, 20(12),
e0337901. https://doi.org/10.1371/journal.pone.0337901

Zheng, B., et al. (2021). Changes in China's anthropogenic emissions and air quality during the
COVID-19 pandemic in 2020. *Earth System Science Data*, 13, 2895–2907.
https://doi.org/10.5194/essd-13-2895-2021

[Scientific Reports paper, 2024] Analysis of the synergistic benefits of typical technologies for
pollution reduction and carbon reduction in the iron and steel industry in the Beijing–Tianjin–Hebei
region. *Scientific Reports*, 14. https://doi.org/10.1038/s41598-024-63338-8 *(author list not
captured — verify)*

S&P Global Market Intelligence (2021). China ramps up efforts to cut carbon emissions from steel,
aluminum industries.

CREA — Centre for Research on Energy and Clean Air. China Energy and Emissions Trends (blast-furnace
operating rates, Tangshan). https://energyandcleanair.org/

---

# Installation

NOXUS uses [`uv`](https://docs.astral.sh/uv/) for environment and dependency management.

```bash
# create the virtual environment and install the project (incl. dev tools)
uv sync --extra dev

# run the CLI
uv run noxus --help
```

# Repository layout

```
noxus/
  config/        # run configuration, study-region definitions (e.g. Tangshan bounding box)
  data/          # public TROPOMI NO2 ingestion (Sentinel-5P), ERA5 meteorology (CDS), and benchmark series
  attribution/   # source attribution of the NO2 column to the cluster
  signal/        # construction of the activity index from attributed NO2
  validation/    # lead/lag tests against the physical-output benchmark
  cli/           # command-line entry point
analysis/        # reproducible preliminary run (figures → docs/figures/preliminary/)
docs/            # design notes, the preprint motivation, and the preliminary-results writeup
tests/           # unit tests
```

# Data sources

All inputs are intended to be public and free:

- **NO₂** — Sentinel-5P/TROPOMI tropospheric NO₂ column (Copernicus Data Space Ecosystem; also
  mirrored on Google Earth Engine for research use). See [docs/data-access.md](docs/data-access.md) for
  how to register, obtain API credentials, and how NOXUS authenticates (CDSE token via the password
  grant; credentials in `.env` per `.env.example`).
- **Meteorology** — ERA5 reanalysis (10 m wind + boundary-layer height) from the Copernicus Climate
  Data Store (CDS), used to regress meteorological ventilation out of the NO₂ signal. `noxus ingest-era5`
  fetches a server-side AOI subset into a dated `.nc` snapshot. Needs a free CDS account and a one-time
  licence acceptance (`CDSAPI_KEY` in `.env`); see [docs/data-access.md](docs/data-access.md).
- **Facility locations** — the Tangshan integrated steel plants (Global Energy Monitor's Global Iron &
  Steel Tracker), used to define the study area of interest. Derived list:
  [`data/derived/tangshan_steel_facilities.csv`](data/derived/tangshan_steel_facilities.csv).
- **Benchmark** — the weekly **Tangshan blast-furnace operating rate**, ingested from CREA's public,
  WIND-sourced steel CSV (a Google-Sheet export). Run `noxus ingest-benchmark`: it writes a dated raw
  snapshot under `data/raw/benchmark/` (the reproducible source of record) and emits a tidy series to
  `data/derived/benchmark_tangshan_bf_operating_rate.parquet`. Two source quirks are handled in
  ingestion: several metadata header rows precede the data, and `0.00` is a *not-reported* placeholder
  (a 0% cluster operating rate is implausible) so it is treated as missing, never as a real zero.
  Auxiliary steel series in the same file (daily pig-iron / crude-steel output) are retained for
  robustness. Secondary national benchmarks (crude-steel production from the World Steel Association
  or China NBS) are coarser (national/provincial, not cluster-level) and used only as cross-checks.

  *Provenance:* the underlying series originate from a commercial terminal (WIND / China United Steel
  Network) and are **republished freely by CREA**; NOXUS consumes CREA's public artifact, so the
  "public, free" criterion is met at the access layer. Because the endpoint is a CREA-maintained sheet
  rather than a contracted API, results are pinned to committed dated snapshots and the benchmark
  column is selected by exact name to survive any change in the source's column set.

# Status

The pipeline now spans the full chain: acquisition → gridding → attribution → signal/index → validation.
Implemented: **CREA benchmark ingestion** (`noxus ingest-benchmark`); **TROPOMI NO₂ acquisition** over the
Tangshan AOI via openEO with a verification step (`noxus fetch` / `noxus verify-no2`); the **study
facilities/AOI** (Global Iron & Steel Tracker); **gridding/compositing** to a weekly cube (`noxus grid`);
**ERA5 meteorology ingest** from the Copernicus CDS (`noxus ingest-era5`); cluster **attribution**
(`noxus attribute`), the relative **activity index** with meteo regress-out and deseasonalisation
(`noxus index`), and the lead/lag **validation** against the benchmark (`noxus validate`, which reports
the null when that is what the data show). A reproducible preliminary run (`analysis/preliminary_signal.py`)
produces the first end-to-end result — see [docs/preliminary-results.html](docs/preliminary-results.html).
See `docs/data-access.md` for data access and `docs/motivation.md` for the preprint-oriented motivation.

# License

MIT — see [LICENSE](LICENSE).
