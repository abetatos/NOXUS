# Motivation and Scope

> Preprint-oriented version of the motivation. The README carries the same text; this file is the
> canonical source for the arXiv intro. `[CITE]` markers indicate where references will be added.

## Why this project

Remote-sensing-derived alternative data for financial and macroeconomic signals is an established
field, but its most-cited applications—counting vehicles in retail parking lots, gauging oil-tank
fill levels, monitoring port traffic—are mature. They have been studied extensively and, where they
carried predictive value, that value has plausibly been competed away as the methods became
standard [CITE]. Replicating them adds little.

We adopt an explicit selection criterion: a candidate signal is worth pursuing only if it satisfies
three conditions simultaneously. It must be (a) extractable from **public, free** data, so that the
work is reproducible and accessible to non-institutional researchers rather than gated behind paid
feeds; (b) potentially **leading** with respect to the corresponding official statistic; and (c)
**not already broadly exploited** through some other channel.

Several candidates were evaluated and set aside. Narrative macro and credit-risk signals have no
physical satellite footprint—they are macro analysis, not remote sensing, and fall outside scope.
Aggregate regional NO₂ as a proxy for economic activity is already documented, including the natural
experiment of the COVID-19 lockdowns and work relating NO₂ to regional GDP [CITE]; the aggregate
angle is therefore not novel. Mining output via optical imagery is valid but is already exploited
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
metal smelters, and industrial areas [CITE], alongside plume-detection work at 10–60 m using
Landsat/Sentinel-2 [CITE].

The gap is therefore **not** the attribution technique, and it is **not** the general idea of turning
pollution into an activity indicator—commercial nowcasting vendors already market
industrial-pollution and steel-sector indices framed as leading signals, and a thin academic
literature already links NO₂ to official activity measures [CITE]. What is absent from the open
literature is the specific intersection: a **public, reproducible** pipeline that attributes the NO₂
signal to a **single industrial cluster** and tests, transparently, whether it leads the official
statistic—**with the null result reported if that is what the data show.** The commercial products
are closed: they sell estimates, not methods, and they do not publish failures.

## Case study

We focus on the Tangshan (Hebei) steel cluster, chosen for its high geographic concentration and
strong, continuous thermal signature. A secondary motivation is data integrity: official Chinese
output figures for this region have been documented as subject to misreporting [CITE], whereas the
satellite signal is not reported by the producer and is in that sense tamper-resistant. Validation
is against a physical-output benchmark—monthly crude-steel production and/or blast-furnace operating
rates [CITE]—rather than a diffusion index such as the PMI, which measures sentiment rather than
production. The intended unit of analysis is macro (steel-sector activity and metals-related
instruments), not the prediction of any single equity.

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

<!-- Replace each [CITE] marker above with a numbered reference here. -->
