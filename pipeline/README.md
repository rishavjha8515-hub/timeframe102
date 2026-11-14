# TIMEFRAME/102 pipeline commands

Use Python 3.11 and the pinned dependencies in `requirements-dev.txt`.

From the repository root, install the environment with:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Process the cached real FITS products offline. This writes reprojected 128×128
FITS cutouts, PNG previews, and a processing report:

```powershell
.\.venv\Scripts\python.exe -m pipeline.process_cutouts --manifest pipeline/cache/fits-inspection/inspection_manifest.json --output-dir outputs/processed-cutouts
```

When IRSA is reachable, select ten Victoria epochs on distinct UTC nights,
download their small FITS cutouts, and inspect flags at the predicted position:

```powershell
.\.venv\Scripts\python.exe -m pipeline.fetch_cutouts --table pipeline/cache/probe/most_20000012_Victoria_smaller-body-screen.ecsv --count 10 --output-dir pipeline/cache/victoria-cutouts
```

When JPL Horizons is reachable, compare the MOST positions with Horizons at
observer 500 at midpoint UTC epochs:

```powershell
.\.venv\Scripts\python.exe -m pipeline.compare_horizons_geocenter --table pipeline/cache/probe/most_20000012_Victoria_smaller-body-screen.ecsv --output pipeline/cache/victoria_horizons_500.ecsv
```

When Gaia and IRSA are reachable, search for a low-proper-motion control star
in the deep fields and download the qualifying candidate cutouts:

```powershell
.\.venv\Scripts\python.exe -m pipeline.probe_control_star --cache-dir pipeline/cache/control-star --max-candidates 12
```

Re-evaluate the cached control-star SIA tables offline (including metre-to-
micron unit conversion and per-release maxrec lower bounds):

```powershell
.\.venv\Scripts\python.exe -m pipeline.recheck_control_star_cache --cache-dir pipeline/cache/control-star --output outputs/control-star-cache-review.json
```

On a machine with IRSA access, query 2MASS/AllWISE magnitudes for the reviewed
Gaia candidates. The faintest candidate is processed first: the script plans
40 products across D1--D6 and time, directly GETs four server-side cutouts with
a 50 MB cap and transient-error retries, inspects the first FITS MEF, then
checks flags, source location, wavelength, and a PNG preview. After at least
three failed cutout downloads it tries four full products, capped at 150 MB
each, and cuts them locally. Other candidates are attempted only if the
faintest candidate completes:

```powershell
.\.venv\Scripts\python.exe -m pipeline.probe_star_flux_and_cutouts --cache-dir pipeline/cache/control-star --plan-count 40 --overflow-count 4 --output-dir pipeline/cache/star-candidates
```

The catalog query stores its ECSV response tables in the output directory.
HTTP failures include status and response body. Individual failed cutouts do
not stop later observations; processing stops after five consecutive failures.
A failed download is recorded separately and never counted as a zero-overflow
observation. AllWISE `var_flg`
is reported as catalog metadata only; it is not used as evidence for
SPHEREx-epoch variability.

For the larger control-star series, select 60--100 epochs across detectors and
dates, download 0.25-degree cutouts with retry/cache-resume, measure each
source wavelength and aperture flux, then group by the measured source
wavelength (not SIA's full-band bounds):

```powershell
.\.venv\Scripts\python.exe -m pipeline.fetch_control_star_series --count 80 --size-deg 0.25 --cache-dir pipeline/cache/control-star --output-dir pipeline/cache/control-star-series
```

Run the offline contract and scientific helper tests with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
