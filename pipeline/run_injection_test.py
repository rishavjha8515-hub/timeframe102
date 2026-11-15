"""Run the preregistered, simulation-only injection/recovery experiment.

The experiment reads only cached reprojections and original ePSF calibration
tables. It writes results to a separate JSON file and never modifies observed
FITS products or case bundles.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area
from scipy.ndimage import map_coordinates

from .cutout_processing import _measure_local_source, aperture_photometry_image_zodi, cutout_to_detector_pixels

ROOT = Path(__file__).resolve().parents[1]
CASE_PATH = ROOT / "outputs/control-star/case.json"
SERIES_DIR = ROOT / "pipeline/cache/control-star-series"
OUTPUT_PATH = ROOT / "outputs/control-star/injection_test_results.json"
PIXEL_ARCSEC = 6.0


def native_psf_stamp(psf_oversampled: np.ndarray, source_xy: tuple[float, float],
                     shape: tuple[int, int], *, oversample_x: int, oversample_y: int,
                     radius_px: int = 8) -> tuple[np.ndarray, tuple[int, int]]:
    """Sample an oversampled ePSF at native pixel centers and normalize its stamp."""
    if oversample_x < 1 or oversample_y < 1:
        raise ValueError("EPSF oversampling must be positive")
    psf = np.asarray(psf_oversampled, dtype=float)
    if psf.ndim != 2 or not np.isfinite(psf).all() or np.nansum(psf) <= 0:
        raise ValueError("EPSF must be finite, 2-D, and have positive sum")
    x, y = source_xy
    ny, nx = shape
    xlo, xhi = max(0, int(np.floor(x)) - radius_px), min(nx, int(np.floor(x)) + radius_px + 2)
    ylo, yhi = max(0, int(np.floor(y)) - radius_px), min(ny, int(np.floor(y)) + radius_px + 2)
    yy, xx = np.indices((yhi - ylo, xhi - xlo), dtype=float)
    yy += ylo
    xx += xlo
    py = (psf.shape[0] - 1) / 2.0 + (yy - y) * oversample_y
    px = (psf.shape[1] - 1) / 2.0 + (xx - x) * oversample_x
    stamp = map_coordinates(psf, [py, px], order=1, mode="constant", cval=0.0, prefilter=False)
    if not np.isfinite(stamp).all() or stamp.sum() <= 0:
        raise ValueError("EPSF has no valid native-pixel samples at source location")
    stamp /= stamp.sum()
    out = np.zeros(shape, dtype=float)
    out[ylo:yhi, xlo:xhi] = stamp
    return out, (xlo, ylo)


def scale_stamp_to_aperture_flux(stamp: np.ndarray, source_xy: tuple[float, float],
                                flux_jy: float, pixel_area_sr: float) -> np.ndarray:
    """Scale a unit-sum ePSF stamp so its 3-pixel aperture sums to flux_jy."""
    yy, xx = np.indices(stamp.shape)
    aperture = np.hypot(xx - source_xy[0], yy - source_xy[1]) <= 3.0
    fraction = float(stamp[aperture].sum())
    if not np.isfinite(fraction) or fraction <= 0 or pixel_area_sr <= 0:
        raise ValueError("ePSF aperture fraction or pixel area is invalid")
    return stamp * (flux_jy / (fraction * pixel_area_sr * 1e6))


def _detector_psf(original_path: Path, predicted_ra: float, predicted_dec: float,
                  measured_xy: tuple[float, float], common_wcs: WCS) -> tuple[np.ndarray, int, int, float]:
    with fits.open(original_path, memmap=True) as hdus:
        image_wcs = WCS(hdus["IMAGE"].header, hdus).celestial
        sky = common_wcs.pixel_to_world(*measured_xy)
        original_x, original_y = map(float, image_wcs.world_to_pixel(sky))
        detector_x, detector_y, _ = cutout_to_detector_pixels(hdus, original_x, original_y)
        epsf_hdu = hdus["EPSF"]
        epsf_data = epsf_hdu.data
        distance = (np.asarray(epsf_data["XCENTER"], float) - detector_x) ** 2 + (
            np.asarray(epsf_data["YCENTER"], float) - detector_y) ** 2
        row = epsf_data[int(np.argmin(distance))]
        psf = np.asarray(row["EPSF"], float)
        osx = int(epsf_hdu.header.get("OVSMPX", 5))
        osy = int(epsf_hdu.header.get("OVSMPY", 5))
        return psf, osx, osy, float(row["CWAVE"])


def _measure(image: np.ndarray, zodi: np.ndarray, variance: np.ndarray, wcs: WCS,
             expected_xy: tuple[float, float]) -> dict[str, Any]:
    centroid = _measure_local_source(image, zodi, variance, expected_xy)
    detected = (centroid["association"] == "associated" and
                centroid["candidate_xy"] is not None and centroid["peak_snr"] is not None and
                centroid["peak_snr"] >= 5.0 and centroid["candidate_distance_px"] <= 5.0)
    result: dict[str, Any] = {"detected": bool(detected), "peak_snr": centroid["peak_snr"],
                              "centroid_xy": centroid["candidate_xy"],
                              "centroid_uncertainty_xy_px": centroid["centroid_uncertainty_xy_px"]}
    if centroid["candidate_xy"] is not None:
        try:
            phot = aperture_photometry_image_zodi(image, zodi, variance, wcs,
                                                  tuple(centroid["candidate_xy"]))
            result["flux_jy"] = float(phot["flux_jy"])
            result["flux_uncertainty_jy"] = float(phot["uncertainty_jy"])
        except ValueError:
            result["flux_jy"] = None
            result["flux_uncertainty_jy"] = None
    else:
        result["flux_jy"] = None
        result["flux_uncertainty_jy"] = None
    return result


def _trial_level(kind: str, size: float, trials: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in trials if row.get("attempted") is True and row.get("recovered") is not None]
    recovered = [row for row in completed if row["recovered"] is True]
    unavailable = [row for row in trials if row.get("attempted") is False]
    errors = [row for row in trials if row.get("attempted") is True and row.get("recovered") is None]
    ratios = [row["recovered_over_injected_ratio"] for row in recovered
              if row["recovered_over_injected_ratio"] is not None and np.isfinite(row["recovered_over_injected_ratio"])]
    return {"simulated": True, "kind": kind, "injected_size": size,
            "n_trials": len(trials), "n_attempted": len(trials) - len(unavailable),
            "n_completed": len(completed), "n_unavailable": len(unavailable),
            "n_processing_errors": len(errors), "n_recovered": len(recovered),
            "recovery_rate": len(recovered) / len(completed) if completed else None,
            "median_recovered_over_injected_ratio": float(np.median(ratios)) if ratios else None,
            "trials": trials}


def run_injection_test() -> dict[str, Any]:
    case = json.loads(CASE_PATH.read_text(encoding="utf-8"))
    plan = case["injection_test_plan"]
    if plan["status"] not in ("planned", "partial") or plan["simulated"] is not True:
        raise ValueError("Injection plan must be predeclared and marked simulated before execution")
    observations = [o for o in case["observations"] if o["usable"]]
    expected_trials = plan["trial_count_per_level"]
    flux_sizes = plan["flux_change_fractions"]
    position_sizes = plan["position_shift_arcsec"]["magnitudes"]
    if len(observations) != expected_trials:
        raise ValueError(f"Frozen usable-frame count changed: expected {expected_trials}, got {len(observations)}")

    flux_trials: dict[float, list[dict[str, Any]]] = {float(size): [] for size in flux_sizes}
    position_trials: dict[float, list[dict[str, Any]]] = {float(size): [] for size in position_sizes}
    frame_failures = []
    for obs in observations:
        obsid, detector = obs["observation_id"], int(obs["detector"])
        base = SERIES_DIR / "northup" / f"{obsid}_D{detector}_northup.fits"
        original = SERIES_DIR / f"{obsid}_D{detector}.fits"
        expected_xy_list = obs["image"].get("predicted_xy")
        measured_xy_list = obs["image"].get("source_xy")
        if not base.is_file() or not original.is_file() or not expected_xy_list:
            frame_failures.append({"simulated": True, "observation_id": obsid, "detector": detector,
                                   "reason": "required cached cutout, common-grid prediction, or image is missing"})
            for size in flux_sizes:
                flux_trials[float(size)].append({"simulated": True, "observation_id": obsid, "detector": detector,
                                                  "attempted": False, "recovered": None,
                                                  "recovered_over_injected_ratio": None,
                                                  "reason": "frame inputs unavailable"})
            for size in position_sizes:
                position_trials[float(size)].append({"simulated": True, "observation_id": obsid, "detector": detector,
                                                      "attempted": False, "recovered": None,
                                                      "recovered_over_injected_ratio": None,
                                                      "reason": "frame inputs unavailable"})
            continue
        expected_xy = tuple(map(float, expected_xy_list))
        target_xy = tuple(map(float, measured_xy_list or expected_xy_list))
        try:
            with fits.open(base, memmap=True) as hdus:
                image = np.asarray(hdus["IMAGE"].data, float)
                zodi = np.asarray(hdus["ZODI"].data, float)
                variance = np.asarray(hdus["VARIANCE"].data, float)
                flags = np.asarray(hdus["FLAGS"].data, int)
                wcs = WCS(hdus["IMAGE"].header).celestial
                area = float(proj_plane_pixel_area(wcs)) * (np.pi / 180.0) ** 2
                base_measurement = _measure(image, zodi, variance, wcs, expected_xy)
            if base_measurement["flux_jy"] is None or base_measurement["flux_jy"] <= 0:
                raise ValueError("base common-grid aperture flux is unavailable or nonpositive")
            psf, osx, osy, epsf_wave = _detector_psf(original, obs["ra_deg"], obs["dec_deg"], target_xy, wcs)
            stamp, _ = native_psf_stamp(psf, target_xy, image.shape, oversample_x=osx, oversample_y=osy)
            source_model = scale_stamp_to_aperture_flux(stamp, target_xy,
                                                        float(base_measurement["flux_jy"]), area)

            for fraction in flux_sizes:
                fraction = float(fraction)
                delta_jy = fraction * float(base_measurement["flux_jy"])
                injected = image + source_model * fraction
                measurement = _measure(injected, zodi, variance, wcs, expected_xy)
                # Differential recovery uses the same aperture method on the base and injected copy.
                ratio = ((measurement["flux_jy"] - base_measurement["flux_jy"]) / delta_jy
                         if measurement["flux_jy"] is not None else None)
                recovered = bool(measurement["detected"] and ratio is not None and np.isfinite(ratio))
                flux_trials[fraction].append({"simulated": True, "attempted": True,
                    "observation_id": obsid, "detector": detector,
                    "effective_wavelength_um": obs["effective_wavelength_um"], "epsf_cwave_um": epsf_wave,
                    "injected_flux_change_fraction": fraction, "injected_delta_flux_jy": delta_jy,
                    "recovered": recovered, "recovered_over_injected_ratio": float(ratio) if ratio is not None else None,
                    "peak_snr": measurement["peak_snr"], "centroid_xy": measurement["centroid_xy"],
                    "quality_flags": obs["quality_flags"]["flag_names"], "reason": None})

            for offset_arcsec in position_sizes:
                offset_arcsec = float(offset_arcsec)
                # North-up grid x increases west, so an eastward shift has negative dx.
                shifted_xy = (target_xy[0] - offset_arcsec / PIXEL_ARCSEC, target_xy[1])
                moved_stamp, _ = native_psf_stamp(psf, shifted_xy, image.shape,
                                                  oversample_x=osx, oversample_y=osy)
                moved_model = scale_stamp_to_aperture_flux(moved_stamp, shifted_xy,
                                                            float(base_measurement["flux_jy"]), area)
                injected = image - source_model + moved_model
                measurement = _measure(injected, zodi, variance, wcs, shifted_xy)
                recovered_shift = None
                ratio = None
                if measurement["centroid_xy"] is not None:
                    # Convert common-grid westward x change into eastward tangent-plane arcseconds.
                    recovered_shift = (target_xy[0] - float(measurement["centroid_xy"][0])) * PIXEL_ARCSEC
                    ratio = recovered_shift / offset_arcsec
                recovered = bool(measurement["detected"] and ratio is not None and np.isfinite(ratio))
                position_trials[offset_arcsec].append({"simulated": True, "attempted": True,
                    "observation_id": obsid,
                    "detector": detector, "effective_wavelength_um": obs["effective_wavelength_um"],
                    "epsf_cwave_um": epsf_wave, "injected_eastward_shift_arcsec": offset_arcsec,
                    "recovered_eastward_shift_arcsec": recovered_shift, "recovered": recovered,
                    "recovered_over_injected_ratio": float(ratio) if ratio is not None else None,
                    "peak_snr": measurement["peak_snr"], "centroid_xy": measurement["centroid_xy"],
                    "quality_flags": obs["quality_flags"]["flag_names"], "reason": None})
        except Exception as exc:
            frame_failures.append({"simulated": True, "observation_id": obsid, "detector": detector,
                                   "reason": f"{type(exc).__name__}: {exc}"})
            for size in flux_sizes:
                flux_trials[float(size)].append({"simulated": True, "attempted": True,
                    "observation_id": obsid, "detector": detector,
                    "recovered": None, "recovered_over_injected_ratio": None,
                    "reason": f"{type(exc).__name__}: {exc}"})
            for size in position_sizes:
                position_trials[float(size)].append({"simulated": True, "attempted": True,
                    "observation_id": obsid, "detector": detector,
                    "recovered": None, "recovered_over_injected_ratio": None,
                    "reason": f"{type(exc).__name__}: {exc}"})

    result = {
        "simulated": True,
        "status": "complete" if not frame_failures else "partial",
        "plan": plan,
        "method": "Use the frozen north-up IMAGE/ZODI/VARIANCE/FLAGS products. Sample the per-frame ePSF at native-pixel centers using the FITS OVSMPX/OVSMPY factors (5x), normalize it to the measured common-grid aperture flux, and inject only into in-memory copies. Run the same local SNR/centroid association and 3-pixel aperture photometry functions used by the real pipeline. The original files are read-only.",
        "quality_gate": "A trial is recovered when the source is detected at peak SNR >= 5, the local centroid is associated within 5 pixels of that trial's prediction, the frame is usable in the frozen case, and the measured flux/centroid is finite.",
        "recovery_denominator": "Completed attempted trials only. Missing frame inputs are marked attempted=false and recovered=null, are listed as unavailable, and are excluded from the recovery-rate denominator; processing errors are reported separately.",
        "flux_change_results": [_trial_level("flux_change", size, trials) for size, trials in flux_trials.items()],
        "position_shift_results": [_trial_level("eastward_position_shift", size, trials) for size, trials in position_trials.items()],
        "failed_frames": frame_failures,
        "cannot_show": "These injections are simulations on real image backgrounds. They retain some detector noise, sky background, flags, and resampling, but do not reproduce unknown cross-frame calibration, PSF, or correlated-resampling systematics. They cannot measure the real systematic floor or justify a motion/variability claim.",
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"simulated": True, "status": result["status"],
                      "flux_levels": [{"fraction": r["injected_size"], "n_trials": r["n_trials"],
                                       "n_completed": r["n_completed"], "n_unavailable": r["n_unavailable"],
                                       "recovery_rate": r["recovery_rate"],
                                       "median_ratio": r["median_recovered_over_injected_ratio"]}
                                      for r in result["flux_change_results"]],
                      "position_levels": [{"arcsec": r["injected_size"], "n_trials": r["n_trials"],
                                           "n_completed": r["n_completed"], "n_unavailable": r["n_unavailable"],
                                           "recovery_rate": r["recovery_rate"],
                                           "median_ratio": r["median_recovered_over_injected_ratio"]}
                                          for r in result["position_shift_results"]],
                      "failed_frames": len(frame_failures),
                      "output": OUTPUT_PATH.relative_to(ROOT).as_posix()}, separators=(",", ":")))
    return result


if __name__ == "__main__":
    run_injection_test()
