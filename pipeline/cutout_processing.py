"""Offline SPHEREx cutout reprojection, flag decoding, and centroid utilities."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area
from photutils.detection import DAOStarFinder
from reproject import reproject_interp
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import maximum_filter

GRID_SHAPE = (128, 128)
GRID_PIXEL_SCALE_ARCSEC = 6.0
PREVIEW_STRETCH = {"type": "asinh", "vmin": -0.6, "vmax": 50.0}
FLAG_HEADER_RE = re.compile(r"^MSKN(\d{4})$")


def north_up_tan_wcs(
    ra_deg: float,
    dec_deg: float,
    *,
    shape: tuple[int, int] = GRID_SHAPE,
    pixel_scale_arcsec: float = GRID_PIXEL_SCALE_ARCSEC,
) -> WCS:
    """Make a zero-rotation TAN grid: x increases west and y increases north."""
    if pixel_scale_arcsec <= 0:
        raise ValueError("pixel_scale_arcsec must be positive")
    ny, nx = shape
    output = WCS(naxis=2)
    output.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    output.wcs.cunit = ["deg", "deg"]
    output.wcs.radesys = "ICRS"
    output.wcs.crval = [float(ra_deg), float(dec_deg)]
    output.wcs.crpix = [(nx + 1) / 2, (ny + 1) / 2]
    scale = pixel_scale_arcsec / 3600.0
    output.wcs.cd = np.array([[-scale, 0.0], [0.0, scale]])
    return output


def decode_flag_definitions(header: fits.Header) -> dict[int, tuple[str, str]]:
    """Read flag names and descriptions from FITS MSKN/MSKD cards."""
    found: dict[int, tuple[str, str]] = {}
    for key, name in header.items():
        match = FLAG_HEADER_RE.match(key)
        if not match:
            continue
        bit = int(match.group(1))
        description = str(header.get(f"MSKD{bit:04d}", ""))
        found[bit] = (str(name), description)
    return found


def decode_flag_value(raw_value: int, definitions: dict[int, tuple[str, str]]) -> list[dict[str, Any]]:
    """Return all set bit names using the definitions carried by this FITS."""
    value = int(raw_value)
    result = []
    for bit in sorted(definitions):
        if value & (1 << bit):
            name, description = definitions[bit]
            result.append({"bit": bit, "name": name, "description": description})
    for bit in range(max(value.bit_length(), 0)):
        if value & (1 << bit) and bit not in definitions:
            result.append({"bit": bit, "name": f"UNKNOWN_BIT_{bit}", "description": "No MSKN/MSKD header definition"})
    return result


def sample_wavelength_metadata(
    hdul: fits.HDUList,
    original_x: float,
    original_y: float,
) -> dict[str, float | str | None]:
    """Sample wavelength tables after mapping 0-based image pixels to detector pixels.

    IRSA spatial cutouts can retain full-frame EPSF and WCS-WAVE tables. Their
    tabulated coordinates are detector FITS pixels, while IMAGE CRPIX is
    shifted for the cutout. Use LTM/LTV when supplied; otherwise derive the
    full-frame offset from the WCS-WAVE detector grid and CRPIX values.
    """
    answer: dict[str, float | str | None] = {
        "effective_wavelength_um": None,
        "wavelength_width_um": None,
        "wavelength_uncertainty_um": None,
        "wavelength_source": None,
        "epsf_cwave_um": None,
        "wcs_wave_um": None,
    }
    if "EPSF" not in hdul or "WCS-WAVE" not in hdul:
        return answer

    full_x, full_y, transform = cutout_to_detector_pixels(hdul, original_x, original_y)
    epsf = hdul["EPSF"].data
    epsf_header = hdul["EPSF"].header
    centers = np.column_stack((np.asarray(epsf["XCENTER"], float), np.asarray(epsf["YCENTER"], float)))
    nearest = int(np.argmin(np.sum((centers - np.array([full_x, full_y])) ** 2, axis=1)))
    epsf_wave = float(epsf["CWAVE"][nearest])
    epsf_width = float(epsf["CBAND"][nearest])

    wave_hdu = hdul["WCS-WAVE"]
    row = wave_hdu.data[0]
    x_grid = np.asarray(row["X"], dtype=float)
    y_grid = np.asarray(row["Y"], dtype=float)
    values = np.asarray(row["VALUES"], dtype=float)
    if not (float(np.min(x_grid)) <= full_x <= float(np.max(x_grid))
            and float(np.min(y_grid)) <= full_y <= float(np.max(y_grid))):
        answer.update({
            "wavelength_source": "unavailable: mapped source pixel lies outside full-detector WCS-WAVE grid",
            "wavelength_detector_pixel_xy_fits": [full_x, full_y],
            "wavelength_coordinate_transform": transform,
        })
        return answer
    if values.shape != (len(y_grid), len(x_grid), 2):
        raise ValueError(f"Unexpected WCS-WAVE VALUES shape {values.shape} for {x_grid.size}x{y_grid.size} grid")
    # The table X/Y samples are FITS one-based detector coordinates.
    interpolator = RegularGridInterpolator(
        (y_grid, x_grid), values, bounds_error=False, fill_value=np.nan
    )
    wcs_values = np.asarray(interpolator([[full_y, full_x]])[0], dtype=float)
    wcs_wave = float(wcs_values[0]) if np.isfinite(wcs_values[0]) else None
    uncertainty = abs(epsf_wave - wcs_wave) if wcs_wave is not None else None
    answer.update({
        "effective_wavelength_um": epsf_wave,
        "wavelength_width_um": epsf_width,
        "wavelength_uncertainty_um": uncertainty,
        "wavelength_source": (
            "EPSF.CWAVE/CBAND nearest detector zone "
            f"(CWAVESRC={epsf_header.get('CWAVESRC', 'header unspecified')}; "
            f"CBANDSRC={epsf_header.get('CBANDSRC', 'header unspecified')}); "
            "WCS-WAVE bilinear cross-check"
        ),
        "epsf_cwave_um": epsf_wave,
        "wcs_wave_um": wcs_wave,
        "wcs_wave_bandwidth_um": float(wcs_values[1]) if np.isfinite(wcs_values[1]) else None,
        "wavelength_detector_pixel_xy_fits": [full_x, full_y],
        "wavelength_coordinate_transform": transform,
    })
    return answer


def cutout_to_detector_pixels(hdul: fits.HDUList, x_zero: float, y_zero: float) -> tuple[float, float, str]:
    """Map zero-based IMAGE coordinates to one-based full-detector FITS pixels."""
    image_header = hdul["IMAGE"].header
    x_fits, y_fits = float(x_zero) + 1.0, float(y_zero) + 1.0
    ltm = np.array([
        [float(image_header.get("LTM1_1", 1.0)), float(image_header.get("LTM1_2", 0.0))],
        [float(image_header.get("LTM2_1", 0.0)), float(image_header.get("LTM2_2", 1.0))],
    ])
    if "LTV1" in image_header or "LTV2" in image_header:
        ltv = np.array([float(image_header.get("LTV1", 0.0)), float(image_header.get("LTV2", 0.0))])
        full = ltm @ np.array([x_fits, y_fits]) + ltv
        return float(full[0]), float(full[1]), "IMAGE LTM/LTV affine transform"

    row = hdul["WCS-WAVE"].data[0]
    x_grid, y_grid = np.asarray(row["X"], dtype=float), np.asarray(row["Y"], dtype=float)
    full_crpix_x = (float(np.nanmin(x_grid)) + float(np.nanmax(x_grid))) / 2.0
    full_crpix_y = (float(np.nanmin(y_grid)) + float(np.nanmax(y_grid))) / 2.0
    cut_crpix_x = float(image_header.get("CRPIX1", full_crpix_x))
    cut_crpix_y = float(image_header.get("CRPIX2", full_crpix_y))
    full_x = x_fits + full_crpix_x - cut_crpix_x
    full_y = y_fits + full_crpix_y - cut_crpix_y
    return full_x, full_y, "WCS CRPIX offset from WCS-WAVE full detector grid"


def aperture_photometry_image_zodi(
    image: np.ndarray, zodi: np.ndarray, variance: np.ndarray, wcs: WCS,
    source_xy: tuple[float, float], *, aperture_radius_px: float = 3.0,
    annulus_inner_px: float = 6.0, annulus_outer_px: float = 10.0,
) -> dict[str, float | int | str | None]:
    """Integrate IMAGE-ZODI in Jy and propagate VARIANCE plus annulus error.

    Assumes independent input pixels; this does not include an encircled-energy
    correction or covariance from resampling. The annulus median is a local
    consistency check and its standard error enters the reported uncertainty.
    """
    if image.shape != zodi.shape or image.shape != variance.shape:
        raise ValueError("IMAGE, ZODI, and VARIANCE shapes must match")
    if not (0 < aperture_radius_px < annulus_inner_px < annulus_outer_px):
        raise ValueError("Require 0 < aperture radius < annulus inner < annulus outer")
    yy, xx = np.indices(image.shape)
    distance = np.hypot(xx - source_xy[0], yy - source_xy[1])
    science = np.asarray(image, float) - np.asarray(zodi, float)
    var = np.asarray(variance, float)
    aperture = (distance <= aperture_radius_px) & np.isfinite(science) & np.isfinite(var) & (var >= 0)
    annulus = ((distance >= annulus_inner_px) & (distance <= annulus_outer_px)
               & np.isfinite(science) & np.isfinite(var) & (var >= 0))
    n_ap, n_ann = int(aperture.sum()), int(annulus.sum())
    if n_ap == 0 or n_ann < 10:
        raise ValueError(f"Insufficient valid pixels: aperture={n_ap}, annulus={n_ann}")
    local_background = float(np.median(science[annulus]))
    annulus_mad = float(1.4826 * np.median(np.abs(science[annulus] - local_background)))
    flux_surface_sum = float(np.sum(science[aperture] - local_background))
    var_sum = float(np.sum(var[aperture]))
    background_variance = max(annulus_mad**2, float(np.median(var[annulus])) / 2.0) / n_ann
    total_variance_surface = var_sum + n_ap**2 * background_variance
    pixel_area_sr = float(proj_plane_pixel_area(wcs)) * (np.pi / 180.0) ** 2
    conversion_jy_per_mjysr = pixel_area_sr * 1e6
    return {
        "flux_jy": flux_surface_sum * conversion_jy_per_mjysr,
        "uncertainty_jy": float(np.sqrt(total_variance_surface) * conversion_jy_per_mjysr),
        "background_mjy_sr": local_background,
        "background_scatter_mjy_sr": annulus_mad,
        "aperture_pixels": n_ap,
        "annulus_pixels": n_ann,
        "pixel_area_sr": pixel_area_sr,
        "method": "IMAGE-ZODI circular aperture (r=3 px), local median annulus (6-10 px), VARIANCE propagation plus annulus background uncertainty; no aperture correction; assumes independent pixels",
    }


def _reproject_plane(data: np.ndarray, input_wcs: WCS, output_wcs: WCS, shape: tuple[int, int], *, flags: bool = False):
    order = "nearest-neighbor" if flags else "bilinear"
    result, footprint = reproject_interp((data, input_wcs), output_wcs, shape_out=shape, order=order)
    return result, footprint


def _measure_local_source(
    image: np.ndarray,
    zodi: np.ndarray,
    variance: np.ndarray,
    predicted_xy: tuple[float, float],
    *,
    search_radius_px: float = 20.0,
    association_radius_px: float = 5.0,
) -> dict[str, Any]:
    """Find a local S/N peak and estimate a variance-weighted centroid."""
    ny, nx = image.shape
    x0, y0 = predicted_xy
    yy, xx = np.indices(image.shape)
    distance = np.hypot(xx - x0, yy - y0)
    science = image - zodi
    annulus = (distance >= 8.0) & (distance <= 15.0) & np.isfinite(science) & np.isfinite(variance) & (variance > 0)
    background = float(np.nanmedian(science[annulus])) if np.any(annulus) else float(np.nanmedian(science))
    noise = float(np.sqrt(np.nanmedian(variance[annulus]))) if np.any(annulus) else float(np.sqrt(np.nanmedian(variance[np.isfinite(variance) & (variance > 0)])))
    if not np.isfinite(noise) or noise <= 0:
        noise = float("nan")
    snr = (science - background) / np.sqrt(np.where(variance > 0, variance, np.nan))
    in_search = distance <= search_radius_px
    if not np.any(in_search & np.isfinite(snr)):
        return {"candidate_xy": None, "source_xy": None, "peak_snr": None, "candidate_distance_px": None,
                "peak_distance_px": None,
                "centroid_uncertainty_xy_px": None, "background_image_minus_zodi": background,
                "local_noise_from_variance": noise, "association": "no finite pixels"}
    work = np.where(in_search & np.isfinite(snr), snr, -np.inf)
    peak_y, peak_x = np.unravel_index(int(np.argmax(work)), work.shape)
    peak_snr = float(snr[peak_y, peak_x])
    peak_distance_px = float(np.hypot(peak_x - x0, peak_y - y0))
    yy, xx = np.indices(image.shape)
    aperture = np.hypot(xx - peak_x, yy - peak_y) <= 2.5
    weights = np.where(aperture, np.maximum(science - background, 0.0), 0.0)
    total = float(np.sum(weights))
    if total > 0 and peak_snr >= 3.0:
        cx = float(np.sum(weights * xx) / total)
        cy = float(np.sum(weights * yy) / total)
        safe_var = np.where(np.isfinite(variance) & (variance > 0), variance, 0.0)
        sx2 = float(np.sum((xx - cx) ** 2 * safe_var * aperture) / total**2)
        sy2 = float(np.sum((yy - cy) ** 2 * safe_var * aperture) / total**2)
        centroid = (cx, cy)
        sigma = (float(np.sqrt(max(sx2, 0.0))), float(np.sqrt(max(sy2, 0.0))))
    else:
        centroid, sigma = (float(peak_x), float(peak_y)), None
    candidate_distance_px = float(np.hypot(centroid[0] - x0, centroid[1] - y0))
    association = "associated" if peak_snr >= 5.0 and candidate_distance_px <= association_radius_px else "unresolved"
    source_xy = [float(centroid[0]), float(centroid[1])] if association == "associated" else None
    return {"candidate_xy": [float(centroid[0]), float(centroid[1])], "source_xy": source_xy,
            "peak_snr": peak_snr, "candidate_distance_px": candidate_distance_px,
            "peak_distance_px": peak_distance_px,
            "centroid_uncertainty_xy_px": list(sigma) if sigma is not None else None,
            "background_image_minus_zodi": background, "local_noise_from_variance": noise,
            "association": association}


def _reference_candidates(image: np.ndarray, zodi: np.ndarray, variance: np.ndarray, predicted_xy: tuple[float, float]) -> list[dict[str, float]]:
    """Find morphological point-source candidates; without Gaia, these are not star identifications."""
    science = image - zodi
    valid = np.isfinite(science) & np.isfinite(variance) & (variance > 0)
    if np.count_nonzero(valid) < 20:
        return []
    background = float(np.nanmedian(science[valid]))
    sigma = float(np.sqrt(np.nanmedian(variance[valid])))
    if not np.isfinite(sigma) or sigma <= 0:
        return []
    data = np.where(valid, science - background, 0.0)
    finder = DAOStarFinder(fwhm=2.5, threshold=5.0 * sigma, exclude_border=True)
    detections = finder(data, mask=~valid)
    if detections is None:
        return []
    x_column = "x_centroid" if "x_centroid" in detections.colnames else "xcentroid"
    y_column = "y_centroid" if "y_centroid" in detections.colnames else "ycentroid"
    px, py = predicted_xy
    result = []
    for row in detections:
        x, y = float(row[x_column]), float(row[y_column])
        if np.hypot(x - px, y - py) < 10.0:
            continue
        result.append({"x": x, "y": y, "peak": float(row["peak"]), "sharpness": float(row["sharpness"]),
                       "roundness1": float(row["roundness1"])})
    return result


def process_fits_product(
    fits_path: Path,
    *,
    output_dir: Path,
    ra_deg: float,
    dec_deg: float,
    observation_id: str,
    target_name: str,
    detector: int | None,
    time_utc: str | None = None,
    mjd_midpoint: float | None = None,
    shape: tuple[int, int] = GRID_SHAPE,
    pixel_scale_arcsec: float = GRID_PIXEL_SCALE_ARCSEC,
) -> dict[str, Any]:
    """Reproject one real product to the shared grid and measure its target neighborhood."""
    fits_path, output_dir = Path(fits_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = fits_path.stem
    output_wcs = north_up_tan_wcs(ra_deg, dec_deg, shape=shape, pixel_scale_arcsec=pixel_scale_arcsec)
    expected_xy = tuple(map(float, output_wcs.world_to_pixel(SkyCoord(ra_deg, dec_deg, unit="deg", frame="icrs"))))
    with fits.open(fits_path, memmap=True) as hdul:
        image_hdu, flags_hdu, var_hdu, zodi_hdu = (hdul[name] for name in ("IMAGE", "FLAGS", "VARIANCE", "ZODI"))
        input_wcs = WCS(image_hdu.header, hdul).celestial
        original_x, original_y = map(float, input_wcs.world_to_pixel(SkyCoord(ra_deg, dec_deg, unit="deg", frame="icrs")))
        image, footprint = _reproject_plane(np.asarray(image_hdu.data, float), input_wcs, output_wcs, shape)
        variance, variance_footprint = _reproject_plane(np.asarray(var_hdu.data, float), input_wcs, output_wcs, shape)
        zodi, _ = _reproject_plane(np.asarray(zodi_hdu.data, float), input_wcs, output_wcs, shape)
        flag_map, _ = _reproject_plane(np.asarray(flags_hdu.data, float), input_wcs, output_wcs, shape, flags=True)
        flag_map = np.where(np.isfinite(flag_map), np.rint(flag_map), -1).astype(np.int64)
        flag_defs = decode_flag_definitions(flags_hdu.header)
        original_ix, original_iy = int(round(original_x)), int(round(original_y))
        radius = 2
        ys = slice(max(original_iy - radius, 0), min(original_iy + radius + 1, flags_hdu.data.shape[0]))
        xs = slice(max(original_ix - radius, 0), min(original_ix + radius + 1, flags_hdu.data.shape[1]))
        target_flag_values = np.asarray(flags_hdu.data[ys, xs], dtype=np.int64)
        central_flag = int(flags_hdu.data[original_iy, original_ix])
        decoded_flags = decode_flag_value(central_flag, flag_defs)
        flags_in_aperture = sorted({item["name"] for raw in np.unique(target_flag_values)
                                    for item in decode_flag_value(int(raw), flag_defs)})
        overflow = any(item["name"] == "OVERFLOW" for item in decoded_flags)
        overflow_aperture = "OVERFLOW" in flags_in_aperture
        saturated_aperture = any("SATURAT" in flag for flag in flags_in_aperture)
        saturated = any("SATURAT" in item["name"] for item in decoded_flags)
        # Preserve the original calibration tables while marking their pixel coordinates as input-frame coordinates.
        epsf_copy = hdul["EPSF"].copy()
        epsf_copy.name = "EPSF_SRC"
        epsf_copy.header["PIXREF"] = ("ORIGINAL", "Coordinates use input image pixels")
        wave_copy = hdul["WCS-WAVE"].copy()
        wave_copy.name = "WAVE_SRC"
        wave_copy.header["PIXREF"] = ("ORIGINAL", "Coordinates use input image pixels")
        flag_header = flags_hdu.header.copy()
        centroid = _measure_local_source(image, zodi, variance, expected_xy)
        if (centroid["association"] == "associated" and centroid["candidate_xy"] is not None
                and not overflow_aperture):
            sample_world = output_wcs.pixel_to_world(*centroid["candidate_xy"])
            sample_x, sample_y = map(float, input_wcs.world_to_pixel(sample_world))
            wavelength_basis = "measured_source_centroid"
        else:
            sample_x, sample_y = original_x, original_y
            wavelength_basis = "predicted_position_unresolved"
        wavelength = sample_wavelength_metadata(hdul, sample_x, sample_y)
        wavelength["wavelength_pixel_xy"] = [sample_x, sample_y]
        wavelength["wavelength_sample_basis"] = wavelength_basis
        photometry = aperture_photometry_image_zodi(
            np.asarray(image_hdu.data, float), np.asarray(zodi_hdu.data, float),
            np.asarray(var_hdu.data, float), input_wcs, (sample_x, sample_y),
        )

    reference_candidates = _reference_candidates(image, zodi, variance, expected_xy)
    if centroid["source_xy"] is not None and (overflow or overflow_aperture or saturated or saturated_aperture):
        centroid["association"] = "quality-flagged candidate; target association unresolved"
        centroid["source_xy"] = None
    measured_xy = centroid["source_xy"]
    measured_world = None
    uncertainty_ra_arcsec = uncertainty_dec_arcsec = None
    if measured_xy:
        coord = output_wcs.pixel_to_world(*measured_xy)
        measured_world = {"ra_deg": float(coord.ra.deg), "dec_deg": float(coord.dec.deg)}
        if centroid["centroid_uncertainty_xy_px"]:
            sigma_x, sigma_y = centroid["centroid_uncertainty_xy_px"]
            dec_cos = max(abs(float(np.cos(np.deg2rad(coord.dec.deg)))), 1e-8)
            uncertainty_ra_arcsec = float(sigma_x * pixel_scale_arcsec / dec_cos)
            uncertainty_dec_arcsec = float(sigma_y * pixel_scale_arcsec)

    output_fits = output_dir / f"{base}_northup.fits"
    output_png = output_dir / f"{base}_northup.png"
    primary = fits.PrimaryHDU()
    primary.header["ORIGFILE"] = fits_path.name
    primary.header["OBSID"] = observation_id
    primary.header["TARGET"] = target_name
    primary.header["DETECTOR"] = detector if detector is not None else -1
    primary.header["TARGRA"] = float(ra_deg)
    primary.header["TARGDEC"] = float(dec_deg)
    primary.header["PIXSCALE"] = (float(pixel_scale_arcsec), "arcsec/pixel; north-up TAN grid")
    image_header = output_wcs.to_header(relax=True)
    image_header["BUNIT"] = image_hdu.header.get("BUNIT", "MJy / sr")
    image_header["EXTTYPE"] = "IMAGE"
    image_header["OBSID"] = observation_id
    image_header["DATE-OBS"] = time_utc or "UNKNOWN"
    image_header["MJD-MID"] = float(mjd_midpoint) if mjd_midpoint is not None else -1.0
    image_header["TIMEDFN"] = "midpoint"
    image_header["WCSNAME"] = "TAN north-up shared"
    image_out = fits.ImageHDU(np.asarray(image, dtype=np.float32), header=image_header, name="IMAGE")
    flags_out_header = output_wcs.to_header(relax=True)
    flags_out_header["EXTTYPE"] = "MASK"
    for key, value in flag_header.items():
        if re.match(r"^MSK[ND]\d{4}$", key):
            flags_out_header[key] = value
    flags_out = fits.ImageHDU(flag_map.astype(np.int32), header=flags_out_header, name="FLAGS")
    var_header = output_wcs.to_header(relax=True)
    var_header["BUNIT"] = var_hdu.header.get("BUNIT", "MJy2 / sr2")
    var_out = fits.ImageHDU(np.asarray(variance, dtype=np.float32), header=var_header, name="VARIANCE")
    zodi_header = output_wcs.to_header(relax=True)
    zodi_header["BUNIT"] = zodi_hdu.header.get("BUNIT", "MJy / sr")
    zodi_out = fits.ImageHDU(np.asarray(zodi, dtype=np.float32), header=zodi_header, name="ZODI")
    for hdu in (flags_out, var_out, zodi_out):
        hdu.header["OBSID"] = observation_id
    fits.HDUList([primary, image_out, flags_out, var_out, zodi_out, epsf_copy, wave_copy]).writeto(output_fits, overwrite=True, checksum=True)

    from PIL import Image, ImageDraw
    lo, hi = PREVIEW_STRETCH["vmin"], PREVIEW_STRETCH["vmax"]
    normalized = np.arcsinh(np.clip(np.nan_to_num(image, nan=lo), lo, hi) - lo) / np.arcsinh(hi - lo)
    gray = np.flipud(np.rint(np.clip(normalized, 0, 1) * 255).astype(np.uint8))
    preview = Image.fromarray(gray, mode="L").convert("RGB")
    draw = ImageDraw.Draw(preview)
    px, py = expected_xy
    draw.line((px - 4, 127 - py, px + 4, 127 - py), fill=(0, 255, 255), width=1)
    draw.line((px, 127 - py - 4, px, 127 - py + 4), fill=(0, 255, 255), width=1)
    candidate_xy = centroid["candidate_xy"]
    if candidate_xy:
        mx, my = candidate_xy
        marker = (255, 64, 64) if measured_xy else (255, 196, 0)
        draw.ellipse((mx - 3, 127 - my - 3, mx + 3, 127 - my + 3), outline=marker, width=1)
    preview.save(output_png, optimize=True)

    return {
        "observation_id": observation_id,
        "target": target_name,
        "is_synthetic": False,
        "time_definition": "midpoint",
        "time_utc": time_utc,
        "fits_path": str(fits_path),
        "reprojected_fits_path": str(output_fits),
        "preview_path": str(output_png),
        "input_pixel_position_xy_0based": [original_x, original_y],
        "predicted_xy": [float(expected_xy[0]), float(expected_xy[1])],
        "measured_source_xy": measured_xy,
        "measured_world": measured_world,
        "centroid_candidate_xy": centroid["candidate_xy"],
        "candidate_distance_from_prediction_px": centroid["candidate_distance_px"],
        "peak_distance_from_prediction_px": centroid["peak_distance_px"],
        "peak_snr": centroid["peak_snr"],
        "centroid_uncertainty_xy_px": centroid["centroid_uncertainty_xy_px"],
        "centroid_uncertainty_method": "formal diagonal VARIANCE propagation on reprojected pixels; assumes independent pixels and excludes interpolation covariance, PSF and astrometric systematics",
        "uncertainty_ra_arcsec": uncertainty_ra_arcsec,
        "uncertainty_dec_arcsec": uncertainty_dec_arcsec,
        "association": centroid["association"],
        "central_flag_raw_value": central_flag,
        "central_flag_decoded": decoded_flags,
        "flags_within_2_pixel_radius_box": flags_in_aperture,
        "overflow_at_predicted_pixel": overflow,
        "overflow_within_2_pixel_radius_box": overflow_aperture,
        "saturated_at_predicted_pixel": saturated,
        "saturated_within_2_pixel_radius_box": saturated_aperture,
        "variance_at_predicted_pixel": float(variance[round(expected_xy[1]), round(expected_xy[0])]),
        "zodi_at_predicted_pixel": float(zodi[round(expected_xy[1]), round(expected_xy[0])]),
        "photometry": photometry,
        **wavelength,
        "reference_source_candidates_unconfirmed": reference_candidates,
        "registration": {"n_reference_stars": 0, "rms_arcsec": None,
                         "method": "north-up TAN WCS reprojection; no Gaia/reference-star astrometric refinement"},
        "grid": {"shape_px": list(shape), "pixel_scale_arcsec": pixel_scale_arcsec,
                 "orientation": "north up; x increases west, y increases north; origin is zero-based",
                 "stretch": PREVIEW_STRETCH},
    }


def load_inspection_manifest(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, list):
        raise ValueError("Inspection manifest must be a JSON list")
    return value
