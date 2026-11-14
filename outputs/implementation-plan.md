# TIMEFRAME/102 backend implementation plan

## Scope and gate

Backend work only: `/pipeline`, `/tests`, scientific processing, `/schemas/case.schema.json`, and validated exports. No frontend work. The first release gate is the user's steps 1–5: establish archive coverage, select an asteroid and stationary star, retrieve small cutouts, inspect real FITS structure, and extract time/wavelength/flags/variance/PSF or ePSF/WCS. Stop after that gate and report feasibility before implementing registration or source-analysis modules.

## Sequence

1. **Archive feasibility.** Query IRSA SIA2/TAP for proposed targets and candidate control stars. Preserve query, UTC timestamp, response, product IDs, data release, and access URLs. Select only on measured coverage and usable data, not a plausible-sounding target name.
2. **Schema freeze.** Schema version 1.2.0 is published in `/schemas/case.schema.json` following the accepted 1.1.0 frontend baseline. Every exported case, including synthetic fixtures, must validate against it; the case index has its own schema. Image paths are relative to their containing case JSON. Any breaking change requires an explicit migration and notice first.
3. **Retrieval and product inspection.** Implement a cache keyed by canonical product URL plus request parameters; download target-centered FITS cutouts while retaining all returned HDUs and headers. Inspect actual headers, units, WCS, `FLAGS`, `VARIANCE`, PSF/ePSF, and calibrated wavelength products. Do not infer the effective wavelength from detector number or band labels.
4. **Stop-gate report.** Report each candidate's confirmed coverage, usable observation count, UTC span, source-position wavelength coverage, product IDs, quality exclusions, and unresolved issues. Recommend one asteroid and one stationary-star control only if both are supported by the inspected data.
5. **P0 analysis.** Establish a documented verdict policy before reading target results; register images; use reference stars; measure centroid and uncertainty; request Horizons at the actual epochs; compare predictions and measured coordinates; explicitly record the frame, epoch, light-time correction, observer, and geocenter-vs-spacecraft offset. Then make a cautious verdict.
6. **P1 spectral consistency.** Compare flux only after wavelength and quality alignment; report wavelength-dependent change as such without treating it as intrinsic variability by default.
7. **P2 injection/recovery.** Add only after P0 works. Keep all synthetic measurements marked `is_synthetic: true` and clearly distinct from the real case.
8. **Export and reproducibility.** Produce schema-validated JSON and 128–256 px PNG cutouts with the stretch parameters recorded. Pin dependencies and provide one reproduction command per case.

## Predeclared motion policy (draft for review before analysis)

Use a weighted fit in the local east/north tangent plane to measured-minus-predicted positions, propagating centroid and prediction uncertainties. A case may be `moving` only when there are at least three usable epochs, the fitted non-zero angular velocity is at least 5 sigma from zero, and the fitted displacement over the observation baseline is at least 3 times the uncertainty of the endpoint displacement. `no significant motion detected` requires at least three usable epochs and a 95% upper bound on displacement below the predeclared project resolution threshold (initial proposal: 1 SPHEREx pixel, 6.15 arcsec, over the observed baseline). All other cases, including failed detections, inadequate epochs, or insufficient precision, are `insufficient evidence`. Do not tune these thresholds after seeing target results; document any approved change before analysis.

## Initial repository findings

The workspace had no source files, tests, dependency lock, or candidate-target list at start. Python 3.11 is not installed in the available execution environment; the available local interpreter is Python 3.14.7 and does not have `jsonschema`. Direct DNS access to IRSA failed. Therefore no scientific processing has started and no real target coverage is claimed.
