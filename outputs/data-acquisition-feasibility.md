# Data acquisition feasibility report — initial stop-gate status

**Status: blocked before candidate-specific step 1.** No proposed targets or target coordinates were present in the workspace or request. Direct HTTPS checks from this runtime failed for both `irsa.ipac.caltech.edu` and `ssd-api.jpl.nasa.gov`: `curl` returned exit code 6, `Could not resolve host`. `Resolve-DnsName` returned `Access is denied` for both hosts. The TAP/SIA and Horizons API endpoints were also not accessible through the browser retrieval tool. No SIA/TAP response, FITS file, product identifier, epoch, or usable-observation count could therefore be verified. No asteroid or control star is recommended yet. No result has been substituted or inferred from a secondary project.

| Target | Confirmed IRSA coverage | Usable observations | Time span | Wavelength range | Unverified |
|---|---|---:|---|---|---|
| Asteroid candidate | None; no candidate was supplied and live IRSA query was unreachable | Unknown | Unknown | Unknown | Target identity/coordinates, SIA/TAP matches, cutout availability, flags, calibrated wavelength map, measured signal and usable epochs |
| Stationary-star control | None; no candidate was supplied and live IRSA query was unreachable | Unknown | Unknown | Unknown | Star identity/coordinates, SIA/TAP matches, isolation, varied source-position wavelengths, cutout availability, flags and usable epochs |

## What the current archive documentation supports

IRSA's SPHEREx archive documentation describes searchable DR1 and QR3 spectral-image collections and recommends querying both for a current wide-survey search. It documents SIA2 at `https://irsa.ipac.caltech.edu/SIA?`, with `COLLECTION=spherex_dr1` or `spherex_qr3`, and states that each result includes an `access_url`. It also documents small FITS cutouts via the IRSA cutout service.

The Level 2 MEF documentation lists `IMAGE`, `FLAGS`, `VARIANCE`, `ZODI`, `PSF`, and `WCS-WAVE`. For science wavelength, it says to use the `CWAVE` and `CBAND` arrays from the detector's Spectral WCS calibration product rather than assuming from detector/band. DR1 and QR3 also provide spatially sampled ePSFs. These are product-format facts from documentation, not a claim that any target has a usable observation.

Sources: [IRSA SPHEREx Data Access](https://caltech-ipac.github.io/spherex-archive-documentation/spherex-data-access/), [IRSA SPHEREx Data Products](https://github.com/Caltech-IPAC/spherex-archive-documentation/blob/main/documentation/spherex_data_products.md), and [IRSA SPHEREx releases](https://caltech-ipac.github.io/spherex-archive-documentation/spherex-data-releases/).

## Required evidence to pass the gate

For every target and proposed epoch: the actual SIA/TAP query and timestamp; IRSA product ID and release; a retrievable product URL; primary-header and extension inventory; UTC time; detector; WCS sky coverage; source-pixel `CWAVE`/`CBAND`; raw quality flags and variance; PSF/ePSF identity and selection; usable-observation rationale; and total UTC span and wavelength range. The archive response and inspected FITS products must be saved with hashes before recommending either case.

## User-run preliminary SIA probe (2026-11-14; partial results)

The user ran the probe locally and supplied its output. The corresponding raw ECSV files are present in `pipeline/cache/probe/`. The four deep-field SIA requests used 5-degree pole cones and `maxrec=10000`; each returned exactly 10,000 rows. Treat these as capped partial result sets, not total row or observation counts. Unique `obs_id` counts below describe only the returned partial sets; they are not usable observation counts. The time fields below are raw ObsCore MJD ranges from those results, not verified UTC conversions.

| Query region and collection | Returned product rows | Unique `obs_id` in partial rows | Returned `t_min` range (MJD) | Returned `t_max` range (MJD) | Union of returned bandpass metadata |
|---|---:|---:|---:|---:|---|
| North ecliptic pole, `spherex_dr1_deep` | 10,000 (limit reached) | 2,129 | 60789.808196–60834.856519 | 60789.809528–60834.857851 | 0.7455–5.0096 µm |
| South ecliptic pole, `spherex_dr1_deep` | 10,000 (limit reached) | 2,428 | 60789.771067–60846.098765 | 60789.772400–60846.100098 | 0.7455–5.0096 µm |
| North ecliptic pole, `spherex_qr3_deep` | 10,000 (limit reached) | 2,149 | 61174.532694–61307.380852 | 61174.534026–61307.382184 | 0.7455–5.0096 µm |
| South ecliptic pole, `spherex_qr3_deep` | 10,000 (limit reached) | 2,515 | 61174.568579–61310.942562 | 61174.569911–61310.943895 | 0.7455–5.0096 µm |

The reported bandpass is the union of ObsCore `em_min`/`em_max` metadata across returned products. It is not an effective wavelength measured at a source pixel. These archive matches confirm that SPHEREx products overlap the queried pole cones, but do not establish source detections, usability, or a suitable control star.

Two ordinary-sky queries at RA 0°, Dec 0° also completed: `spherex_dr1` returned 287 product rows across 144 `obs_id` values (returned MJD bounds `t_min` 60842.377974–61031.567270; `t_max` 60842.379306–61031.568602); `spherex_qr3` returned 143 product rows across 72 `obs_id` values (`t_min` 61204.586212–61230.048381; `t_max` 61204.587544–61230.049714). The RA 120° and 240° ordinary-sky queries are absent from the supplied/cache results.

All asteroid MOST requests in the supplied output failed before querying because this Astroquery version accepts `input_mode="nafid_input"` (one fewer `i` than the earlier probe supplied). The probe has been corrected and its offline MOST validator now accepts this value. Thus no asteroid matches, candidate star, confirmed usable-observation count, UTC span, saturation risk, or source-position wavelength span has been established yet. The rerun uses a 1-degree ecliptic-pole cone and reports explicitly if a response reaches its 10,000-row limit.
