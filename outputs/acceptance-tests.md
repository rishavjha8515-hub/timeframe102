# Backend acceptance tests

These are observable release checks. Unit tests must be added with every scientific module, and every exported case must pass the shared schema validator.

## Schema and provenance

- Case schema is draft 2020-12, version `1.2.0`, with the required scientific fields and strict rejection of unknown top-level/observation fields.
- The case index validates against `schemas/case-index.schema.json`; every indexed relative path resolves to a case JSON whose ID, kind, and synthetic flag match the index.
- Every per-observation image path is relative to its case JSON file and resolves to a preview file when non-null.
- Legacy top-level `image_preview` is marked deprecated; new records use `observations[].image`.
- A valid minimal case passes; missing `schema_version`, `is_synthetic`, product provenance, observation UTC, detector, wavelength, flags, position uncertainties, flux uncertainty, or verdict evidence fails validation.
- Verdict labels are limited to `moving`, `no significant motion detected`, and `insufficient evidence`.
- Case `time_definition` is either `start` or `midpoint`; Horizons epochs use the corresponding observation instant.
- Every observation carries image metadata (or explicit null when no preview is available), registration metadata, wavelength width, and an exclusion reason when unusable.
- Motion thresholds are exported as actual values in `threshold_policy`, including an identifier.
- Optional spectral analysis and spectral verdict fields use the schema's wavelength-aware per-observation shape and closed verdict vocabulary.
- Optional spectral model curves contain wavelength/flux pairs; parameter names and units are documented by the selected model.
- A non-synthetic case cannot be represented by an example or fabricated NASA product ID; provenance includes archive, real product ID/type, release where available, access URL, retrieval UTC, and checksum when available.
- Synthetic tests/cases carry `is_synthetic: true`; real observed cases carry `false`.

## Archive retrieval and FITS preservation

- SIA2 query results are retained verbatim with query parameters, query time, release, product IDs, and URLs.
- A cached retrieval is byte-identical to the stored product for the same canonical URL and request; failed or partial responses are not cached as successful data.
- Cutout retrieval preserves every HDU and relevant header from IRSA; tests compare original/cutout extension names and required science extensions.
- Archive unavailability is surfaced as an explicit acquisition failure, never replaced with generated data.

## Product metadata

- UTC timestamps parse unambiguously and preserve their time scale.
- Effective wavelength is evaluated at the source pixel using the verified calibration product (`CWAVE`/`CBAND` where available); wavelength lookup and coordinate transforms are tested against a fixture taken from a real product.
- `FLAGS`, `VARIANCE`, PSF/ePSF, and WCS are preserved and reported. Masking does not erase excluded or missing observations from output.
- PSF/ePSF selection accounts for detector position and FITS 1-based versus Python 0-based conventions.

## Astrometry and verdict

- Registration tests recover known transforms on synthetic fixtures and preserve uncertainty.
- Reference-star selection excludes moving/contaminated/low-quality sources using documented criteria.
- Centroid tests recover planted positions within stated uncertainties over tested flux and quality regimes.
- Horizons requests use actual exposure epochs; output states frame, epoch scale, light-time handling, observer location, and quantified spacecraft-vs-geocenter offset.
- Residuals use a local tangent plane and propagate measurement and ephemeris uncertainties.
- Verdict policy is fixed and documented before target results are examined. Under-threshold or inadequate data yield `insufficient evidence`; failed detection never yields a stationary verdict.

## Spectral comparison and outputs

- Flux comparisons use wavelength-aware samples and flux uncertainties; simulated wavelength trends do not become automatic intrinsic-variability claims.
- PNG previews are 128–256 px per side, have a consistent recorded stretch, and derive from the identified archive cutout.
- JSON output validates against `/schemas/case.schema.json`; referenced files exist; all observations, including flagged and missing measurements, remain visible in the record.
- One documented, pinned-dependency command reproduces each real case. Synthetic injection/recovery outputs are explicitly separate from real observations.
