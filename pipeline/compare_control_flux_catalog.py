"""Compare cached SPHEREx control-star fluxes to 2MASS/AllWISE SED points."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table

TWOMASS = (("J", "j", 1.235, 1594.0, 27.8), ("H", "h", 1.662, 1024.0, 20.0),
           ("Ks", "k", 2.159, 666.7, 12.6))
WISE = (("W1", "w1", 3.4, 309.54), ("W2", "w2", 4.6, 171.79),
        ("W3", "w3", 12.0, 31.676), ("W4", "w4", 22.0, 8.3635))


def catalog_flux_points(tmass, allwise) -> list[dict]:
    points = []
    for band, key, wavelength, zero, zero_error in TWOMASS:
        magnitude, mag_error = float(tmass[key + "_m"]), float(tmass[key + "_msigcom"])
        flux = zero * 10 ** (-0.4 * magnitude)
        error = math.hypot(flux * math.log(10) / 2.5 * mag_error, flux * zero_error / zero)
        points.append({"band": band, "wavelength_um": wavelength, "flux_jy": flux,
                       "uncertainty_jy": error, "magnitude": magnitude,
                       "magnitude_uncertainty": mag_error, "zero_point_jy": zero,
                       "zero_point_uncertainty_jy": zero_error})
    for band, key, wavelength, zero in WISE:
        magnitude, mag_error = float(allwise[key + "mpro"]), float(allwise[key + "sigmpro"])
        flux = zero * 10 ** (-0.4 * magnitude)
        error = flux * math.log(10) / 2.5 * mag_error
        points.append({"band": band, "wavelength_um": wavelength, "flux_jy": flux,
                       "uncertainty_jy": error, "magnitude": magnitude,
                       "magnitude_uncertainty": mag_error, "zero_point_jy": zero,
                       "zero_point_uncertainty_jy": None})
    return sorted(points, key=lambda item: item["wavelength_um"])


def compare_fluxes(case: dict, points: list[dict]) -> list[dict]:
    wavelengths = np.array([p["wavelength_um"] for p in points], dtype=float)
    logw = np.log(wavelengths)
    logf = np.log([p["flux_jy"] for p in points])
    fracerr = np.array([p["uncertainty_jy"] / p["flux_jy"] for p in points])
    output = []
    for observation in case["observations"]:
        wavelength = observation["effective_wavelength_um"]
        x = math.log(wavelength)
        hi = int(np.searchsorted(logw, x, side="right"))
        lo = max(0, min(len(points) - 2, hi - 1))
        hi = lo + 1
        fraction = (x - logw[lo]) / (logw[hi] - logw[lo])
        catalog_flux = math.exp(logf[lo] + fraction * (logf[hi] - logf[lo]))
        catalog_relative_error = math.hypot((1 - fraction) * fracerr[lo], fraction * fracerr[hi])
        catalog_error = catalog_flux * catalog_relative_error
        measured = observation["flux"]["value"]
        measured_error = observation["flux"]["uncertainty"]
        ratio = measured / catalog_flux
        ratio_error = ratio * math.hypot(measured_error / measured, catalog_error / catalog_flux)
        output.append({
            "observation_id": observation["observation_id"],
            "wavelength_um": wavelength,
            "catalog_interpolation_bands": [points[lo]["band"], points[hi]["band"]],
            "catalog_interpolated_flux_jy": catalog_flux,
            "catalog_interpolated_uncertainty_jy": catalog_error,
            "spherex_flux_jy": measured,
            "spherex_uncertainty_jy": measured_error,
            "flux_ratio_spherex_to_catalog": ratio,
            "flux_ratio_uncertainty": ratio_error,
        })
    return output


def main() -> int:
    base = Path("pipeline/cache/star-candidates")
    sid = "1633306316700113024"
    case = json.loads(Path("outputs/control-star/case.json").read_text(encoding="utf-8"))
    tmass = Table.read(base / f"{sid}_fp_psc.ecsv", format="ascii.ecsv")
    allwise = Table.read(base / f"{sid}_allwise_p3as_psd.ecsv", format="ascii.ecsv")
    center = SkyCoord(case["target"]["coordinates"]["ra_deg"] * u.deg,
                      case["target"]["coordinates"]["dec_deg"] * u.deg)
    tcoord = SkyCoord(np.asarray(tmass["ra"], float) * u.deg, np.asarray(tmass["dec"], float) * u.deg)
    wcoord = SkyCoord(np.asarray(allwise["ra"], float) * u.deg, np.asarray(allwise["dec"], float) * u.deg)
    ti, wi = int(np.argmin(center.separation(tcoord))), int(np.argmin(center.separation(wcoord)))
    points = catalog_flux_points(tmass[ti], allwise[wi])
    result = {
        "confirmed_catalog_matches": {
            "2mass_designation": str(tmass[ti]["designation"]),
            "2mass_separation_arcsec": float(center.separation(tcoord[ti]).arcsec),
            "2mass_ph_qual": str(tmass[ti]["ph_qual"]),
            "allwise_designation": str(allwise[wi]["designation"]),
            "allwise_separation_arcsec": float(center.separation(wcoord[wi]).arcsec),
            "allwise_var_flg": str(allwise[wi]["var_flg"]),
            "allwise_ph_qual": str(allwise[wi]["ph_qual"]),
            "allwise_cc_flags": str(allwise[wi]["cc_flags"]),
            "allwise_ext_flg": int(allwise[wi]["ext_flg"]),
        },
        "catalog_flux_points": points,
        "comparisons": compare_fluxes(case, points),
        "assumptions": [
            "Fluxes are converted with catalog magnitude zero points; 2MASS zero-point errors and catalog magnitude errors are propagated.",
            "AllWISE catalog magnitude errors are propagated; separate AllWISE zero-point uncertainty is not included.",
            "Catalog SED is interpolated linearly in log flux versus log wavelength, with end-segment extrapolation where needed.",
            "No bandpass color correction, aperture correction, or catalog epoch adjustment is applied.",
        ],
    }
    out = Path("outputs/control-star/catalog_flux_comparison.json")
    out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    for row in result["comparisons"]:
        print(f"FLUX_RATIO obs={row['observation_id']} wavelength_um={row['wavelength_um']:.6f} "
              f"bands={row['catalog_interpolation_bands']} ratio={row['flux_ratio_spherex_to_catalog']:.6f} "
              f"uncertainty={row['flux_ratio_uncertainty']:.6f}")
    print(f"CATALOG_MATCH 2MASS={result['confirmed_catalog_matches']['2mass_separation_arcsec']:.3f} arcsec "
          f"AllWISE={result['confirmed_catalog_matches']['allwise_separation_arcsec']:.3f} arcsec "
          f"var_flg={result['confirmed_catalog_matches']['allwise_var_flg']} output={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
