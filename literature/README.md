# Literature

Reference register for NOXUS. Keep in sync with the **References** section of the top-level
[README](../README.md) and with [`references.bib`](references.bib).

## What lives here

- **`references.bib`** — BibTeX for every source. **Committed.**
- **`papers/`** — local PDFs. **Gitignored, not committed** (publisher copyright).
- **`extracted/`** — plain-text Markdown per PDF (`pdftotext -layout`) for fast grep/source-checking.
  **Gitignored** (same copyright as the PDFs).

Both folders are listed in [`.gitignore`](../.gitignore). The bibliography and this README travel
with the repo; the PDFs and extracts do not. Clone fresh → re-download from the DOIs below and
re-run the extraction snippet at the bottom.

## File-naming convention

```
P<priority>_<Author(s)>_<Year>_<Venue>_<short-topic>.pdf
```

e.g. `P1_Mao_2025_ACP_Ukraine-war-NOx-TROPOMI.pdf`. The `extracted/*.md` mirror the PDF names.

## Priority legend

- **P1 — read first.** Defines the thesis, the attribution method, or is a closest precedent.
- **P2 — key context.** Methods, confounders, case-study justification, foundational reviews.
- **P3 — supporting / peripheral.** Reinforces a premise or a single limitation; nice-to-have.

---

## Register

### P1 — core (thesis, method, closest precedents)

| File in `papers/` | Citation | DOI | In repo |
|---|---|---|---|
| `P1_Morris-Zhang_2019_SSRN_validating-China-output.pdf` | Morris & Zhang (2019), *Macroeconomic Dynamics* 23(8) — validate China output with satellite NO₂. **Closest precedent in intent.** | [10.1017/S1365100518000056](https://doi.org/10.1017/S1365100518000056) | ✅ SSRN preprint (journal paywalled) |
| `P1_Li-Zheng_2023_ESTLett_FastRecovery-China_SI.pdf` | Li & Zheng (2023), *Env. Sci. Technol. Lett.* 10(8) — NO₂ nowcast of China's post-CNY rebound; meteorology removed via CTM. | [10.1021/acs.estlett.3c00386](https://doi.org/10.1021/acs.estlett.3c00386) | ⚠ SI only (CC BY-NC); main paywalled · abstract in `extracted/..._FastRecovery-China_ABSTRACT.md` |
| `P1_Mao_2025_ACP_Ukraine-war-NOx-TROPOMI.pdf` | Mao et al. (2025), *Atmos. Chem. Phys.* 25 — TROPOMI NO₂ inversion of a war-driven industrial shock (industry −34%). **Closest published analog.** | [10.5194/acp-25-14187-2025](https://doi.org/10.5194/acp-25-14187-2025) | ✅ full (open access) |
| `P1_Kim_2023_RemoteSensing_shipping-NO2-TROPOMI.pdf` | Kim et al. (2023), *Remote Sensing* 15(13) — clustered TROPOMI NO₂ vs. a sector throughput index. **Closest open methodological precedent.** | [10.3390/rs15133453](https://doi.org/10.3390/rs15133453) | ✅ full (open access) |
| `P1_Beirle_2021_ESSD_NOx-point-source-catalog.pdf` | Beirle et al. (2021), *Earth Syst. Sci. Data* 13 — point-source NOₓ from NO₂ flux divergence. **The attribution technique we build on.** | [10.5194/essd-13-2995-2021](https://doi.org/10.5194/essd-13-2995-2021) | ✅ full (open access) |
| `P1_Ezran_2023_WorldBankWP10445_NO2-global-economic-activity.pdf` | Ezran et al. (2023), World Bank WP 10445 — NO₂ to measure economic activity; China data integrity (Morris coauthor). | [ideas.repec](https://ideas.repec.org/p/wbk/wbrwps/10445.html) | ✅ full (open access) |

### P2 — key context (methods, confounders, foundations)

| File in `papers/` | Citation | DOI | In repo |
|---|---|---|---|
| `P2_Beirle_2023_ESSD_NOx-catalog-v2.pdf` | Beirle et al. (2023), *ESSD* 15 — improved point-source catalog (v2). | [10.5194/essd-15-3051-2023](https://doi.org/10.5194/essd-15-3051-2023) | ✅ full |
| `P2_Montgomery-Holloway_2018_JARS_NO2-economic-growth-cities.pdf` | Montgomery & Holloway (2018), *J. Appl. Remote Sens.* 12(4) — link is income-dependent (env. Kuznets), not universal. | [10.1117/1.JRS.12.042607](https://doi.org/10.1117/1.JRS.12.042607) | ✅ full |
| `P2_Li_2024_EnvSciEcotech_NOx-trends-drivers-China.pdf` | Li et al. (2024), *Environ. Sci. Ecotechnol.* 21 — decomposes China NOₓ into activity vs. policy vs. meteorology. **Confounder handling.** | [10.1016/j.ese.2024.100425](https://doi.org/10.1016/j.ese.2024.100425) | ✅ full (open access) |
| `P2_Zheng_2021_ESSD_China-COVID-emissions-airquality.pdf` | Zheng et al. (2021), *ESSD* 13 — China emissions/air quality, COVID natural experiment. | [10.5194/essd-13-2895-2021](https://doi.org/10.5194/essd-13-2895-2021) | ✅ full |
| `P2_Kondragunta_2021_JGR_COVID-NO2-US-unemployment.pdf` | Kondragunta et al. (2021), *JGR Atmos.* 126 — Q2-2020 U.S. unemployment vs. TROPOMI NO₂. (Refereed successor to Wei et al. 2020 abstract.) | [10.1029/2021JD034797](https://doi.org/10.1029/2021JD034797) | ✅ full (extract uses plain `pdftotext`; `-layout` mis-parses its STIX fonts) |
| `P2_SciRep_2024_steel-NOx-pollution-carbon-BTH.pdf` | Scientific Reports (2024) 14 — iron-and-steel as dominant NOₓ source in Beijing–Tianjin–Hebei. **Case-study justification.** | [10.1038/s41598-024-63338-8](https://doi.org/10.1038/s41598-024-63338-8) | ✅ full (author list TODO) |
| `P2_Donaldson-Storeygard_2016_JEP_satellite-data-in-economics.pdf` | Donaldson & Storeygard (2016), *J. Econ. Perspect.* 30(4) — foundational review of satellite data in economics. | [10.1257/jep.30.4.171](https://doi.org/10.1257/jep.30.4.171) | ✅ full |

### P3 — supporting / peripheral

| File in `papers/` | Citation | DOI | In repo |
|---|---|---|---|
| `P3_Parubets-Naito_2025_PLOSONE_NO2-activity-Japan.pdf` | Parubets & Naito (2025), *PLOS ONE* 20(12) — local NO₂↔activity in Japan. | [10.1371/journal.pone.0337901](https://doi.org/10.1371/journal.pone.0337901) | ✅ full (CC BY) |
| `P3_Li_2025_FrontEnvSciEng_NO2-CO2-monitoring-review.pdf` | Li et al. (2025), *Front. Environ. Sci. Eng.* 19 — review of NO₂→emissions monitoring (progress/challenges). | [10.1007/s11783-025-1922-x](https://doi.org/10.1007/s11783-025-1922-x) | ✅ full |
| `P3_Liao-Ruan_2023_FrontEnergyRes_carbon-network-Chengdu-Chongqing.pdf` | Liao & Ruan (2023), *Front. Energy Res.* 11 — spatial-network analysis of carbon emissions, Chengdu–Chongqing. (Developer-requested; spatial-econometrics angle.) | [10.3389/fenrg.2023.1280715](https://doi.org/10.3389/fenrg.2023.1280715) | ✅ full (open access) |
| `P3_Li-Zheng_2024_OneEarth_daily-anthropogenic-CO2-monitoring.pdf` | Li & Zheng (2024), *One Earth* 7 — daily anthropogenic CO₂ from NO₂ sensors. | [10.1016/j.oneear.2024.08.019](https://doi.org/10.1016/j.oneear.2024.08.019) | ✅ full |
| _(no PDF — abstract saved)_ | Park et al. (2025), *Atmos. Pollut. Res.* 16(10) — GAN nowcasting of GEMS NO₂ (cloud-gap method). | [10.1016/j.apr.2025.102631](https://doi.org/10.1016/j.apr.2025.102631) | ❌ Elsevier paywall · highlights+abstract+intro in `extracted/P3_Park_2025_APR_GEMS-GAN_WEB-EXCERPT.md` |
| _(no PDF — abstract saved)_ | Huang et al. (2025), *Remote Sens. Lett.* 16(5), 472–482 — COVID NO₂ in Yangtze River Basin (GF-5 02 EMI-II). | [10.1080/2150704X.2025.2470908](https://doi.org/10.1080/2150704X.2025.2470908) | ❌ T&F paywall (abstract only public) · in `extracted/P3_Huang_2025_RSL_YRB-COVID_ABSTRACT-only.md` |

### Grey / data sources (no PDF)

| Citation | Link |
|---|---|
| S&P Global Market Intelligence (2021) — China steel/aluminum carbon-cut efforts. | — |
| CREA — China Energy and Emissions Trends (blast-furnace operating rates, Tangshan). | https://energyandcleanair.org/ |

---

## Notes on tricky citations

- **Wei et al. (2020) → Kondragunta et al. (2021).** The WP's "Wei et al. (2020)" is an unrefereed
  AGU abstract (no journal/volume). Cite the peer-reviewed successor — same group, same finding
  (Q2-2020 U.S. unemployment vs. TROPOMI NO₂) — Kondragunta et al. (2021), JGR Atmospheres.
- **Morris & Zhang.** Closest precedent in intent (national OMI NO₂ → Chinese output/GDP). The DOI
  in the source WP (`…517001140`) **does not resolve**; correct DOI is `10.1017/S1365100518000056`,
  and the journal article is **2019** (often cited 2018 after the SSRN/online-first version 2920650).
- **Li & Zheng (2023).** Main text paywalled (ACS); the open **Supporting Information** (CC BY-NC) is
  what's in `papers/`. Author is **Hui** Li (not "Hao"). Vol. 10(8), 635–641.
- **Mao et al. (2025).** The egusphere preprint titled "28% drop" is the same work; the published
  ACP version refines it to −15% (2022) / −8% (2023). Cite the ACP version.

## Regenerating the text extracts

From `literature/`, with `pdftotext` (poppler) on PATH:

```bash
rm -f extracted/*.md
for f in papers/*.pdf; do
  b=$(basename "$f" .pdf)
  pdftotext -layout "$f" "extracted/$b.md"
done
```

Extracts are text only — figures, tables and equations may be garbled. Use them to verify wording,
then cite from the PDF of record.
