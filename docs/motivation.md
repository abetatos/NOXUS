# Motivation and Scope

> Preprint-oriented version of the motivation. The README carries the same text; this file is the
> canonical source for the arXiv intro.

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
experiment of the COVID-19 lockdowns (Zheng et al., 2021) and work relating NO₂ to regional output
(Li & Zheng, 2023; Parubets & Naito, 2025; Ezran et al., 2023); the aggregate angle is therefore not
novel. Mining output via optical imagery is valid but is already exploited commercially using paid
high-resolution data, and the pixel-to-tonnage chain is indirect because it depends on ore grade,
which is not observable from orbit. A further domain consideration rules out reading a specific
miner's output from smelting activity: extraction and smelting are geographically decoupled—
concentrate is shipped, and smelting is concentrated in China—so the satellite signature of a
smelter does not map back to any one mine.

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

## Case study

We focus on the Tangshan (Hebei) steel cluster, chosen for its high geographic concentration and
strong, continuous thermal signature; the iron-and-steel sector is a dominant NOₓ source in this
region (Scientific Reports, 2024, doi:10.1038/s41598-024-63338-8). A secondary motivation is data
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

Kim, J., Emmerich, M. T. M., Voors, R., Ording, B., & Lee, J.-S. (2023). A Systematic Approach to
Identify Shipping Emissions Using Spatio-Temporally Resolved TROPOMI Data. *Remote Sensing*, 15(13),
3453. https://doi.org/10.3390/rs15133453

Li, H., & Zheng, B. (2023). TROPOMI NO₂ Shows a Fast Recovery of China's Economy in the First
Quarter of 2023. *Environmental Science & Technology Letters*.
https://doi.org/10.1021/acs.estlett.3c00386

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
