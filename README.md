# TIMEFRAME/102

NASA Space Apps Challenge 2026 · Challenge 12: Planet X and SPHEREx  
An individual project.

## Question

Is a source in SPHEREx data moving, changing in brightness, or appearing at a
different colour? Each SPHEREx exposure has its own date and wavelength, so a
flux change can come from either.

## Project

- `pipeline/` contains the Python backend. It measures the flux and position
  of control star Gaia DR3 1633306316700113024 in 80 planned SPHEREx cutouts
  from IRSA, applies the written quality policy, and exports a validated case.
- `schemas/case.schema.json` defines the case format, version 1.8.0.
- `outputs/control-star/` contains the case bundle, quality report, and slide
  figures.
- The single-page viewer is at `docs/index.html` in the site build. It shows
  date and wavelength side by side, a blink comparator, spectral plot, all 80
  observations with excluded frames and reasons, provenance, and a Planet X
  calculator. It loads `case_bundle.json` automatically when served with
  GitHub Pages, or accepts the file when opened locally.

## Result

- 80 observations planned; 61 usable over 52 UTC nights, spanning 0.757 to
  4.987 micrometres.
- 19 observations are excluded: 15 with hard quality flags, 2 by the
  robust-residual screen, and 2 outside the cutout edge.
- Verdict: **insufficient evidence** for motion and for spectral variability.
- No systematic photometric floor or astrometric threshold could be measured.
  Each frame had 1 to 7 vetted comparison stars, below the fixed minimum of 20.
  The project makes no claim of variability or constancy.
- Formal-error-only chi-squared is 20863.06 for 58 degrees of freedom
  (reduced chi-squared 359.7). The residuals greatly exceed the formal-error
  model; this is not evidence by itself that the star varies.

## Limits

- The 5-scaled-MAD outlier screen was written after exploratory inspection of
  these measurements. It is a screening result, not a preregistered test.
- One outlier, `2025W32_2D_0603_4` D5, measured 0.479 Jy and has a centroid
  25.07 arcseconds from the Gaia position. Its cause remains unexplained.
- The public IRSA calibration-map cross-check covered 7 of 8 preselected
  frames. Cutout WCS-WAVE values differed from the standalone map by less than
  0.7 nm. Cutout EPSF wavelengths differed by up to 14.5 nm, about 40% of the
  channel width, mainly on detectors 2, 5, and 6. The cause is unidentified.
- The injection test is a simulation on copies of real cutouts. It tests the
  measurement code only; it is not a measured sensitivity and does not set a
  systematic floor.
- The Planet X calculator is an educational approximation using a circular
  orbit and opposition. It is not an orbit fit or a detection.

## Run

Create a Python 3.11 environment and install the pinned dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the contract and scientific helper tests, then export the case from the
locally cached FITS and catalog inputs:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
.\.venv\Scripts\python.exe -m pipeline.export_control_star_case
```

## Data sources

SPHEREx data and calibration maps: NASA/IPAC IRSA. 2MASS and AllWISE: IRSA.
Gaia DR3: ESA. VizieR: CDS.

## Use of AI

The backend code was written with Codex (OpenAI); the viewer, slides, and
documentation were prepared with Claude (Anthropic). The analysis rules and
thresholds were set in advance, and the outputs were reviewed by the project
author, who is responsible for the conclusions. No AI image generation was
used; all plots and frames come from the data.
