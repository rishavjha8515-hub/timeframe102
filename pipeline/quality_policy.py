"""Predeclared control-star image-quality and spectral-outlier policy."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import least_squares
import astropy.units as u
from astropy.coordinates import SkyCoord


POLICY = {
    "id": "control-star-quality-and-spectral-policy-1",
    "quality_flags_exclude": ["TRANSIENT", "SUR_ERROR", "NONFUNC", "OVERFLOW", "SATURATED"],
    "context_only_flags": ["HALO", "PERSIST", "DICHROIC"],
    "mad_residual_threshold_sigma": 5.0,
    "spectral_fit_model": "degree-2 polynomial of log(flux_Jy) versus centered log(wavelength_um), robust soft-L1 loss",
    "wavelength_bin_width_um": 0.1,
    "minimum_comparison_stars": 20,
    "minimum_floor_epochs": 3,
    "comparison_star_criteria": {
        "catalogs": ["Gaia DR3", "2MASS PSC", "AllWISE"],
        "gaia_g_mag_range": [11.12, 14.12],
        "two_mass_match_max_arcsec": 1.0,
        "two_mass_ph_qual": "AAA",
        "allwise_match_max_arcsec": 1.0,
        "allwise_ph_qual_w1_w2": "AA",
        "allwise_cc_flags_w1_w2": "00",
        "allwise_ext_flg": 0,
        "gaia_isolation_radius_arcsec": 60.0,
        "maximum_total_proper_motion_mas_yr": 10.0,
        "maximum_ruwe": 1.4,
        "exclude_phot_variable_flag": True,
        "allowed_phot_variable_flag_values": ["CONSTANT", "NOT_AVAILABLE"],
        "minimum_cutout_edge_margin_arcsec": 60.0,
        "exclude_hard_source_flags": ["TRANSIENT", "SUR_ERROR", "NONFUNC", "OVERFLOW", "SATURATED"],
        "minimum_stars_per_frame": 20,
        "maximum_stars_per_frame": 50,
        "aperture_radius_px": 3.0,
        "annulus_inner_px": 6.0,
        "annulus_outer_px": 10.0,
        "photometric_floor_method": "In each 0.1 um source-wavelength bin, calculate 1.4826 times the MAD of log(measured/catalog-interpolated flux ratios) across comparison stars. Convert the resulting fractional scatter to Jy using the control-star smooth-model flux at that wavelength, then add it in quadrature to the formal uncertainty.",
        "astrometric_threshold_method": "Per frame, subtract the median east/north Gaia residual across selected comparison stars; set the displacement threshold to three times 1.4826 times the MAD of radial residuals after removing the common frame offset. Require at least 20 comparison stars in a frame.",
    },
}


def preselect_gaia_comparisons(table: Any, target_ra_deg: float, target_dec_deg: float,
                              target_source_id: str, criteria: dict[str, Any] | None = None) -> list[str]:
    """Return Gaia-only candidates passing the fixed cuts and isolation test.

    This intentionally does not establish 2MASS/AllWISE matches, per-frame
    edge clearance, or image quality; callers must keep those as later gates.
    """
    criteria = criteria or POLICY["comparison_star_criteria"]
    coords = SkyCoord(np.asarray(table["ra"], float) * u.deg,
                      np.asarray(table["dec"], float) * u.deg)
    target = SkyCoord(target_ra_deg * u.deg, target_dec_deg * u.deg)
    sep = target.separation(coords).arcsec
    gmag = np.asarray(table["phot_g_mean_mag"], float)
    pm = np.hypot(np.asarray(table["pmra"], float), np.asarray(table["pmdec"], float))
    ruwe = np.asarray(table["ruwe"], float)
    variable = np.asarray(table["phot_variable_flag"]).astype(str)
    source_ids = np.asarray(table["source_id"]).astype(str)
    mask = (
        (sep < 12 * 60) & (gmag >= criteria["gaia_g_mag_range"][0]) &
        (gmag <= criteria["gaia_g_mag_range"][1]) &
        (pm <= criteria["maximum_total_proper_motion_mas_yr"]) &
        (ruwe < criteria["maximum_ruwe"]) &
        np.isin(variable, criteria["allowed_phot_variable_flag_values"]) &
        (source_ids != str(target_source_id))
    )
    candidates = np.flatnonzero(mask)
    selected = []
    for idx in candidates:
        others = candidates[candidates != idx]
        if not np.any(coords[idx].separation(coords[others]).arcsec < criteria["gaia_isolation_radius_arcsec"]):
            selected.append(str(source_ids[idx]))
    return sorted(selected)


def classify_quality_flags(flag_names: list[str]) -> tuple[bool, list[str]]:
    """Return inclusion and hard-exclusion flags; context flags alone are retained."""
    flags = sorted(set(flag_names))
    excluded = [name for name in POLICY["quality_flags_exclude"] if name in flags]
    return not excluded, excluded


def fit_spectral_residuals(observations: list[dict[str, Any]]) -> dict[str, Any]:
    """Fit a robust smooth spectrum and flag >5 scaled-MAD residual outliers.

    Uses positive finite fluxes with positive wavelengths. It does not use formal
    errors as a substitute for a measured cross-star systematic floor.
    """
    usable = [o for o in observations if o.get("wavelength_um") and
              (o.get("flux") or {}).get("flux_jy") is not None and
              float(o["wavelength_um"]) > 0 and float(o["flux"]["flux_jy"]) > 0]
    if len(usable) < 4:
        return {"model": None, "center_log_wavelength": None, "coefficients": None,
                "residual_mad_log": None, "outliers": set(), "per_observation": {}}
    wavelength = np.asarray([float(o["wavelength_um"]) for o in usable])
    flux = np.asarray([float(o["flux"]["flux_jy"]) for o in usable])
    x = np.log(wavelength)
    center = float(np.median(x))
    z, y = x - center, np.log(flux)
    result = least_squares(lambda c: np.polyval(c, z) - y,
                           [0.0, 0.0, float(np.median(y))], loss="soft_l1", f_scale=0.03)
    residual = y - np.polyval(result.x, z)
    residual_center = float(np.median(residual))
    mad = float(1.4826 * np.median(np.abs(residual - residual_center)))
    if not np.isfinite(mad) or mad <= 0:
        mad = float(np.finfo(float).eps)
    threshold = float(POLICY["mad_residual_threshold_sigma"])
    per_observation = {}
    outliers = set()
    for item, log_wl, observed_log_flux, res in zip(usable, x, y, residual):
        model_flux = float(np.exp(np.polyval(result.x, log_wl - center)))
        sigma_residual = float((res - residual_center) / mad)
        per_observation[(item["observation_id"], int(item["detector"]))] = {
            "model_flux": model_flux,
            "log_residual": float(res),
            "mad_residual_sigma": sigma_residual,
        }
        if abs(sigma_residual) > threshold:
            outliers.add((item["observation_id"], int(item["detector"])))
    return {"model": POLICY["spectral_fit_model"], "center_log_wavelength": center,
            "coefficients": [float(v) for v in result.x], "residual_mad_log": mad,
            "residual_center_log": residual_center, "outliers": outliers,
            "per_observation": per_observation}
