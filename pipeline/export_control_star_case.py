"""Apply the declared quality policy and export the cached control-star case.

Run from the repository root with the project Python environment:
    python -m pipeline.export_control_star_case
"""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.wcs import WCS
from jsonschema import Draft202012Validator, FormatChecker
from PIL import Image

from .cutout_processing import process_fits_product
from .probe_control_star import cutout_url
from .quality_policy import POLICY, classify_quality_flags, fit_spectral_residuals, preselect_gaia_comparisons

ROOT = Path(__file__).resolve().parents[1]
SERIES_DIR = ROOT / "pipeline/cache/control-star-series"
CATALOG_DIR = ROOT / "pipeline/cache/control-star"
OUTPUT_DIR = ROOT / "outputs/control-star"
CASE_ID = "control-star-1633306316700113024"
SOURCE_ID = "1633306316700113024"
HARD_FLAGS = set(POLICY["quality_flags_exclude"])
PLANNED_GAIA_SEARCH_RADIUS_DEG = 0.25
GAIA_SEARCH_RADIUS_DEG = 0.30


def _flatten(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [o for group in report["groups_by_measured_source_wavelength"].values() for o in group]


def _git_head() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _cached_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except (OSError, json.JSONDecodeError):
        return None


def _merge_cloud_calibration_attempt(
    result: dict[str, Any], cloud_report: dict[str, Any] | None
) -> dict[str, Any]:
    """Attach a retrieval/comparison record without asserting a calibration match."""
    if cloud_report is None:
        return result
    comparison = result.get("comparison")
    if not isinstance(comparison, dict):
        comparison = {}
        result["comparison"] = comparison
    comparison["cloud_calibration_attempt"] = cloud_report
    provenance = cloud_report.get("calibration_provenance", {})
    if cloud_report.get("status") == "compared" and provenance.get("matches_cached_mef_source"):
        result["status"] = "pass"
        result["reason"] = (
            "Standalone DR1 Spectral WCS CWAVE/CBAND were retrieved from the public S3 object and sampled at "
            "the mapped source pixel. Its WCS-WAVE HISTORY source filename matches the cached MEF, and the "
            "detector matches. This pass records successful provenance-matched comparison; no tolerance-based "
            "acceptance threshold was applied. See comparison.cloud_calibration_attempt for the measured values."
        )
    elif cloud_report.get("status") == "failed":
        result["reason"] += (
            " The public S3 map was listed, but the FITS download/comparison failed in the "
            "current execution environment; see comparison.cloud_calibration_attempt."
        )
    elif cloud_report.get("status") == "compared":
        result["status"] = "not applicable"
        result["reason"] += " The public S3 calibration map was downloaded and compared at the mapped source pixel."
    return result


def _catalog_rows() -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted(CATALOG_DIR.glob(f"sia_{SOURCE_ID}_*.ecsv")):
        table = Table.read(path, format="ascii.ecsv")
        for row in table:
            url = str(row["access_url"])
            try:
                detector = int(url.split("D")[-1].split("_")[0])
            except (ValueError, IndexError):
                continue
            key = (str(row["obs_id"]), detector)
            collection = path.stem.removeprefix(f"sia_{SOURCE_ID}_")
            rows.setdefault(key, {"row": row, "release": collection})
    return rows


def _time_from_sia(row: Any) -> tuple[str, float]:
    mid = (float(row["t_min"]) + float(row["t_max"])) / 2
    return Time(mid, format="mjd", scale="utc").isot + "Z", mid


def _propagated_gaia(ra: float, dec: float, pmra: float, pmdec: float, time_utc: str) -> tuple[float, float]:
    dt_year = (Time(time_utc).jyear - 2016.0)
    dec_new = dec + (pmdec * dt_year) / 3_600_000.0
    ra_new = ra + (pmra * dt_year) / (3_600_000.0 * np.cos(np.deg2rad(dec)))
    return float(ra_new % 360.0), float(dec_new)


def build_case() -> tuple[dict[str, Any], dict[str, Any]]:
    report = json.loads((SERIES_DIR / "control_star_series.json").read_text(encoding="utf-8"))
    series = _flatten(report)
    series_by_key = {(o["observation_id"], int(o["detector"])): o for o in series}
    failed_by_key = {(f["observation_id"], int(f["detector"])): f for f in report["failures"]}
    catalog = _catalog_rows()
    gaia_table = Table.read(CATALOG_DIR / "gaia_north.ecsv", format="ascii.ecsv")
    gaia = next(r for r in gaia_table if str(r["source_id"]) == SOURCE_ID)
    ra0, dec0, pmra, pmdec = map(float, (gaia["ra"], gaia["dec"], gaia["pmra"], gaia["pmdec"]))
    # This is a Gaia-only pre-screen, not a vetted comparison-star sample.
    # The fixed cuts are recorded in POLICY before any candidate filtering.
    criteria = POLICY["comparison_star_criteria"]
    gaia_candidate_ids_at_planned_radius = preselect_gaia_comparisons(
        gaia_table, ra0, dec0, SOURCE_ID, criteria,
        search_radius_arcsec=PLANNED_GAIA_SEARCH_RADIUS_DEG * 3600.0,
    )
    gaia_candidate_ids = preselect_gaia_comparisons(
        gaia_table, ra0, dec0, SOURCE_ID, criteria,
        search_radius_arcsec=GAIA_SEARCH_RADIUS_DEG * 3600.0,
    )
    comparison_query_report = _cached_json(
        ROOT / "pipeline/cache/control-star/comparison_catalogs/comparison_catalog_vetting.json") or {}
    comparison_query_error = comparison_query_report.get("error")
    vizier_query_report = _cached_json(
        ROOT / "pipeline/cache/control-star/comparison_catalogs/vizier/comparison_catalog_vetting_vizier.json") or {}
    frame_vetting_report = _cached_json(
        ROOT / "pipeline/cache/control-star/comparison_catalogs/vizier/per_frame_vetting.json") or {}
    catalog_qualified_candidates = [
        row for row in vizier_query_report.get("candidates", []) if row.get("passed_catalog_gates")
    ] if vizier_query_report.get("status") == "complete" else []
    catalog_qualified_ids = sorted(row["source_id"] for row in catalog_qualified_candidates)
    frame_vetting_by_key = {
        (row["observation_id"], int(row["detector"])): row
        for row in frame_vetting_report.get("per_frame_counts", [])
    }
    catalog_row_counts = {name: table["row_count"]
                          for name, table in vizier_query_report.get("raw_tables", {}).items()}
    frame_count_min = frame_vetting_report.get("minimum_candidates_in_frame")
    frame_count_max = frame_vetting_report.get("maximum_candidates_in_frame")
    frames_with_required_stars = frame_vetting_report.get("frames_with_minimum_candidates", 0)
    usable_vetting_frame_count = frame_vetting_report.get("usable_frame_count", 0)
    dr1_report = _cached_json(ROOT / "pipeline/cache/control-star/dr1-cal-check/dr1_calibration_check.json") or {}
    direct_dr1_report = _cached_json(ROOT / "pipeline/cache/control-star/dr1-cal-check/direct_sia_query_report.json")
    cloud_calibration_report = _cached_json(
        ROOT / "pipeline/cache/control-star/dr1-cal-check/cloud/cloud_calibration_comparison.json")
    catalog_flux_comparison = _cached_json(OUTPUT_DIR / "catalog_flux_comparison.json")

    # Re-run the two cached edge failures using the repaired boundary-aware code.
    # They become explicit excluded observations; only four network failures remain.
    for key, failure in list(failed_by_key.items()):
        obs_id, detector = key
        cached = SERIES_DIR / f"{obs_id}_D{detector}.fits"
        if not cached.exists():
            continue
        sia = catalog.get(key)
        if sia is None:
            continue
        utc, mjd = _time_from_sia(sia["row"])
        result = process_fits_product(
            cached, output_dir=SERIES_DIR / "northup", ra_deg=ra0, dec_deg=dec0,
            observation_id=obs_id, target_name=f"Gaia DR3 {SOURCE_ID}", detector=detector,
            time_utc=utc, mjd_midpoint=mjd,
        )
        series_by_key[key] = {
            "observation_id": obs_id, "detector": detector, "time_utc": utc,
            "mjd_midpoint": mjd, "access_url": cutout_url(str(sia["row"]["access_url"]), ra0, dec0, size_deg=0.25),
            "sha256": hashlib.sha256(cached.read_bytes()).hexdigest(),
            "wavelength_um": result["effective_wavelength_um"],
            "wavelength_width_um": result["wavelength_width_um"],
            "wavelength_uncertainty_um": result["wavelength_uncertainty_um"],
            "flux": result["photometry"], "flags_5x5": result["flags_within_2_pixel_radius_box"],
            "measured_position": result["measured_world"], "position_uncertainty_arcsec": [None, None],
            "preview_path": result["preview_path"], "usable": result["usable"],
            "exclusion_reason": (
                f"source outside cutout edge at input pixels {result['input_pixel_position_xy_0based']}; {result['exclusion_reason']}"
                if not result["usable"] else None
            ),
        }
        series.append(series_by_key[key])
        del failed_by_key[key]

    # Apply the spectrum residual cut to all measured observations first,
    # including quality-flagged frames, so simultaneous causes remain visible.
    spectral = fit_spectral_residuals(series)
    all_keys = sorted(set(series_by_key) | set(failed_by_key), key=lambda k: (
        series_by_key.get(k, {}).get("mjd_midpoint", 1e9), k[0], k[1]))
    fit_keys = set(spectral["outliers"])
    clean_series = [o for o in series
                    if classify_quality_flags(o.get("flags_5x5", []))[0]
                    and (o["observation_id"], int(o["detector"])) not in fit_keys]
    clean_spectral = fit_spectral_residuals(clean_series)
    counts = {name: 0 for name in [*POLICY["quality_flags_exclude"], "MAD_RESIDUAL", "EDGE_OR_MISSING"]}
    context_counts = {name: 0 for name in POLICY["context_only_flags"]}
    observations: list[dict[str, Any]] = []
    products: list[dict[str, Any]] = []
    thumbs: dict[tuple[str, int], str] = {}
    null_fields: set[str] = set()
    usable_rows = []

    image_out = OUTPUT_DIR / "images"
    image_out.mkdir(parents=True, exist_ok=True)
    for obs_id, detector in all_keys:
        record = series_by_key.get((obs_id, detector))
        failure = failed_by_key.get((obs_id, detector))
        sia_item = catalog.get((obs_id, detector))
        sia_row = sia_item["row"] if sia_item else None
        release = sia_item["release"] if sia_item else None
        time_utc = record["time_utc"] if record else None
        mjd = record.get("mjd_midpoint") if record else None
        url = record["access_url"] if record else (str(sia_row["access_url"]) if sia_row else None)
        if not time_utc and sia_row is not None:
            time_utc, mjd = _time_from_sia(sia_row)
        if not time_utc:
            time_utc = "2000-01-01T00:00:00Z"
            null_fields.add(f"observations[{obs_id}/D{detector}].time_utc (not recovered)")
        predicted_ra, predicted_dec = _propagated_gaia(ra0, dec0, pmra, pmdec, time_utc)
        fit = spectral["per_observation"].get((obs_id, detector))
        flags = list(record.get("flags_5x5", [])) if record else []
        flags_usable, flag_reasons = classify_quality_flags(flags)
        in_mad = (obs_id, detector) in fit_keys
        file_path = SERIES_DIR / f"{obs_id}_D{detector}.fits"
        sha256 = record.get("sha256") if record else None
        retrieved = None
        if file_path.exists():
            sha256 = hashlib.sha256(file_path.read_bytes()).hexdigest()
            retrieved = datetime.fromtimestamp(file_path.stat().st_mtime, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

        usable = bool(record and record.get("usable", True) and flags_usable and not in_mad)
        reasons = []
        if not record:
            reason = (failure or {}).get("response_body", "observation was not present in the successful cache")
            status = (failure or {}).get("http_status")
            if status:
                reason = f"download not recovered after HTTP {status}: {reason}"
            else:
                reason = f"processing excluded at cutout edge: {reason}"
            reasons.append(reason)
            counts["EDGE_OR_MISSING"] += 1
        elif record.get("exclusion_reason"):
            reasons.append(str(record["exclusion_reason"]))
            counts["EDGE_OR_MISSING"] += 1
        for flag in flag_reasons:
            counts[flag] += 1
            reasons.append(f"source-aperture quality flag: {flag}")
        for flag in context_counts:
            if flag in flags:
                context_counts[flag] += 1
        if in_mad:
            counts["MAD_RESIDUAL"] += 1
            score = fit["mad_residual_sigma"] if fit else None
            reasons.append(f"unexplained outlier: robust spectral residual exceeds {POLICY['mad_residual_threshold_sigma']} scaled MAD (observed {score} scaled MAD)")
        if obs_id == "2025W32_2D_0603_4" and detector == 5:
            reasons.append("measured centroid is displaced from Gaia prediction; no contaminant cause confirmed")
        if obs_id == "2026W33_2A_0208_3" and detector == 5:
            reasons.append("centroid is not reliable because source-aperture flags are present")

        measured = record.get("measured_position") if record else None
        pos_unc = record.get("position_uncertainty_arcsec") if record else None
        if measured:
            dra = ((float(measured["ra_deg"]) - predicted_ra + 180.0) % 360.0 - 180.0) * np.cos(np.deg2rad(predicted_dec)) * 3600
            ddec = (float(measured["dec_deg"]) - predicted_dec) * 3600
            measured_position = {"ra_deg": float(measured["ra_deg"]), "dec_deg": float(measured["dec_deg"]),
                                 "uncertainty_ra_arcsec": float(pos_unc[0]) if pos_unc and pos_unc[0] is not None else None,
                                 "uncertainty_dec_arcsec": float(pos_unc[1]) if pos_unc and pos_unc[1] is not None else None,
                                 "method": "measured centroid on common north-up TAN grid; formal diagonal VARIANCE propagation"}
            residual = [float(dra), float(ddec)]
        else:
            measured_position = {"ra_deg": None, "dec_deg": None, "uncertainty_ra_arcsec": None,
                                 "uncertainty_dec_arcsec": None, "method": "no reliable centroid available"}
            residual = None
            null_fields.update({f"observations[{obs_id}/D{detector}].measured_position.ra_deg",
                                f"observations[{obs_id}/D{detector}].measured_position.dec_deg",
                                f"observations[{obs_id}/D{detector}].position_residual_arcsec"})
        flux = record.get("flux") if record else None
        flux_value = float(flux["flux_jy"]) if flux else None
        flux_unc = float(flux["uncertainty_jy"]) if flux else None
        if flux_value is None:
            null_fields.add(f"observations[{obs_id}/D{detector}].flux.value")
        wavelength = record.get("wavelength_um") if record else None
        width = record.get("wavelength_width_um") if record else None
        if wavelength is None:
            null_fields.add(f"observations[{obs_id}/D{detector}].effective_wavelength_um")
        preview_path = None
        png = None
        special = OUTPUT_DIR / "outlier-diagnostics" / f"{obs_id}_D{detector}_aperture-annulus.png"
        if special.exists():
            png = special
        elif record and record.get("preview_path"):
            png = ROOT / record["preview_path"]
        elif (SERIES_DIR / "northup" / f"{obs_id}_D{detector}_northup.png").exists():
            png = SERIES_DIR / "northup" / f"{obs_id}_D{detector}_northup.png"
        image = None
        if png and png.exists():
            name = f"{obs_id}_D{detector}_128.png"
            destination = image_out / name
            with Image.open(png) as im:
                im.convert("RGB").save(destination, format="PNG", optimize=True)
            preview_path = destination
            data = base64.b64encode(destination.read_bytes()).decode("ascii")
            uri = f"data:image/png;base64,{data}"
            source_xy = None
            predicted_xy = None
            if measured:
                northup = SERIES_DIR / "northup" / f"{obs_id}_D{detector}_northup.fits"
                if northup.exists():
                    with fits.open(northup, memmap=True) as hdul:
                        wcs = WCS(hdul["IMAGE"].header).celestial
                        mx, my = wcs.world_to_pixel(SkyCoord(measured["ra_deg"], measured["dec_deg"], unit="deg"))
                        source_xy = [float(mx), float(my)]
                        px, py = wcs.world_to_pixel(SkyCoord(predicted_ra, predicted_dec, unit="deg"))
                        predicted_xy = [float(px), float(py)]
            image = {"path": uri, "width_px": 128, "height_px": 128,
                     "stretch": {"type": "asinh", "vmin": -0.6, "vmax": 50.0},
                     "pixel_scale_arcsec": 6.0, "source_xy": source_xy, "predicted_xy": predicted_xy}
        else:
            null_fields.add(f"observations[{obs_id}/D{detector}].image")

        obs = {
            "observation_id": obs_id, "time_utc": time_utc,
            "ra_deg": float(measured["ra_deg"]) if measured else None,
            "dec_deg": float(measured["dec_deg"]) if measured else None,
            "detector": detector, "effective_wavelength_um": float(wavelength) if wavelength is not None else None,
            "wavelength_width_um": float(width) if width is not None else None,
            "wavelength_uncertainty_um": float(record["wavelength_uncertainty_um"]) if record and record.get("wavelength_uncertainty_um") is not None else None,
            "wavelength_source": "mapped-pixel EPSF.CWAVE/CBAND and WCS-WAVE" if record and wavelength is not None else None,
            "wavelength_pixel_xy": None,
            "quality_flags": {"raw_value": None, "flag_names": flags, "accepted_for_science": usable},
            "measured_position": measured_position,
            "predicted_position": {"ra_deg": predicted_ra, "dec_deg": predicted_dec,
                                   "uncertainty_ra_arcsec": None, "uncertainty_dec_arcsec": None,
                                   "method": "Gaia DR3 J2016.0 position propagated linearly to observation midpoint; catalog uncertainty and frame systematics not included"},
            "position_residual_arcsec": residual,
            "flux": {"value": flux_value, "uncertainty": flux_unc, "unit": "Jy",
                     "method": flux["method"] if flux else None},
            "image": image,
            "registration": {"n_reference_stars": None, "rms_arcsec": None,
                             "method": "north-up TAN reprojection; no multi-star astrometric refinement"},
            "usable": usable,
            "metadata_status": "confirmed" if record else "missing",
        }
        if not usable:
            obs["exclusion_reason"] = "; ".join(reasons) if reasons else "not usable; cause unresolved"
        observations.append(obs)
        if usable and wavelength is not None and flux_value is not None and flux_unc is not None:
            usable_rows.append((float(wavelength), flux_value, flux_unc, obs_id, detector))
        if url:
            product_id = Path(urlparse(url).path).name if url else f"{obs_id}/D{detector}"
            products.append({"archive": "NASA/IPAC IRSA SPHEREx", "product_id": product_id,
                             "product_type": "SPHEREx L2 spectral image cutout",
                             "release": release, "access_url": url, "retrieved_utc": retrieved,
                             "sha256": sha256})

    if clean_spectral["coefficients"] is not None:
        xlo, xhi = min(r[0] for r in usable_rows), max(r[0] for r in usable_rows)
        curve = [[float(w), float(np.exp(np.polyval(clean_spectral["coefficients"], np.log(w) - clean_spectral["center_log_wavelength"])))]
                 for w in np.linspace(xlo, xhi, 60)]
        model_parameters = {"log_wavelength_center": clean_spectral["center_log_wavelength"],
                            "coefficient_logw2": clean_spectral["coefficients"][0],
                            "coefficient_logw1": clean_spectral["coefficients"][1],
                            "coefficient_constant": clean_spectral["coefficients"][2],
                            "residual_mad_log_flux": clean_spectral["residual_mad_log"]}
    else:
        curve, model_parameters = [], {}
    chi2_without_floor = 0.0
    chi2_n = 0
    for item in clean_series:
        key = (item["observation_id"], int(item["detector"]))
        fit = clean_spectral["per_observation"].get(key)
        uncertainty = (item.get("flux") or {}).get("uncertainty_jy")
        if fit and uncertainty and float(uncertainty) > 0:
            chi2_without_floor += ((float(item["flux"]["flux_jy"]) - fit["model_flux"]) / float(uncertainty)) ** 2
            chi2_n += 1
    chi2_dof = max(chi2_n - 3, 0)
    reduced_chi2_without_floor = chi2_without_floor / chi2_dof if chi2_dof else None
    per_observation = []
    for o in observations:
        fit = spectral["per_observation"].get((o["observation_id"], o["detector"]))
        per_observation.append({"observation_id": f"{o['observation_id']}/D{o['detector']}",
                                "model_flux": fit["model_flux"] if fit else None,
                                "normalized_residual": fit["mad_residual_sigma"] if fit else None})
    valid_wavelengths = [r[0] for r in usable_rows]
    usable_count = sum(o["usable"] for o in observations)
    measured_frame_count = sum(1 for o in observations if o["flux"]["value"] is not None)
    dr1_calibration_result = ({
        "status": "not applicable",
        "product": "SPHEREx standalone Spectral WCS calibration maps (CWAVE/CBAND)",
        "reason": (
            f"A direct SIA query to {direct_dr1_report['collection']} at CALIB={direct_dr1_report['calib_level']} "
            f"succeeded with HTTP {direct_dr1_report['http_status']} and returned {direct_dr1_report['row_count']} "
            f"wide-survey image rows ({direct_dr1_report['distinct_obs_ids']} observation IDs), but "
            f"{direct_dr1_report['case_observation_detector_matches']} rows matched the control-case "
            "observation-ID/detector pairs. Standalone calibration files are not available through SIA; the "
            "CWAVE/CBAND file has not yet been retrieved or compared."
        ),
        "comparison": {
            "direct_sia_http_status": direct_dr1_report["http_status"],
            "direct_sia_collection": direct_dr1_report["collection"],
            "direct_sia_calib_level": direct_dr1_report["calib_level"],
            "direct_sia_row_count": direct_dr1_report["row_count"],
            "distinct_obs_ids": direct_dr1_report["distinct_obs_ids"],
            "detectors": direct_dr1_report["detectors"],
            "case_observation_detector_matches": direct_dr1_report["case_observation_detector_matches"],
            "case_usable_observation_detector_matches": direct_dr1_report["case_usable_observation_detector_matches"],
            "band_bounds_m": direct_dr1_report["em_min_max_m"]
        }
    } if direct_dr1_report else {
        "status": dr1_report.get("status", "not applicable"),
        "product": "SPHEREx standalone Spectral WCS calibration maps (CWAVE/CBAND)",
        "reason": ((dr1_report.get("reason") or "No cached calibration-check result is available.")
                   + (f" Error: {dr1_report['error']}" if dr1_report.get("error") else "")),
        "comparison": dr1_report.get("comparison")
    })
    dr1_calibration_result = _merge_cloud_calibration_attempt(
        dr1_calibration_result, cloud_calibration_report
    )

    case = {
        "schema_version": "1.7.0", "case_id": CASE_ID, "is_synthetic": False,
        "time_definition": "midpoint",
        "target": {"name": f"Gaia DR3 {SOURCE_ID}", "kind": "star", "designation": None,
                   "catalog_id": SOURCE_ID,
                   "coordinates": {"ra_deg": ra0, "dec_deg": dec0, "frame": "ICRS", "epoch": "J2016.0"},
                   "proper_motion": {"pm_ra_cosdec_mas_yr": pmra, "pm_dec_mas_yr": pmdec, "reference_epoch": "J2016.0"}},
        "observations": observations,
        "comparison_star_analysis": {
            "selection_status": (
                "insufficient stars" if frame_vetting_report.get("status") == "insufficient stars"
                else "blocked" if not catalog_qualified_candidates or not frame_vetting_by_key
                else "insufficient stars"
            ),
            "selection_reason": (
                f"{len(gaia_candidate_ids)} Gaia candidates passed the initial fixed cuts; "
                f"{len(catalog_qualified_ids)} passed the fixed 2MASS/AllWISE positional and quality gates. "
                f"Across {frame_vetting_report.get('usable_frame_count', 0)} usable cutouts, "
                f"the pre-photometry edge/5x5-hard-flag eligible count ranged from "
                f"{frame_count_min} to {frame_count_max}; "
                f"{frames_with_required_stars} frames met the required "
                f"minimum of {criteria['minimum_stars_per_frame']}. No comparison-star detection or photometry "
                "was run because the minimum sample size was not met. This pre-photometry count does not "
                "assert that every listed candidate was detected. Search radius was widened by a count-only "
                f"decision from {PLANNED_GAIA_SEARCH_RADIUS_DEG:.2f} to {GAIA_SEARCH_RADIUS_DEG:.2f} deg: "
                f"{len(gaia_candidate_ids_at_planned_radius)} to {len(gaia_candidate_ids)} preliminary Gaia candidates."
            ),
            "n_gaia_candidates": len(gaia_candidate_ids),
            "gaia_candidate_source_ids": gaia_candidate_ids,
            "catalog_query": {
                "status": vizier_query_report.get("status", "failed"),
                "catalog_source": vizier_query_report.get("catalog_source", "CDS VizieR public mirrors"),
                "query_utc": vizier_query_report.get("query_utc"),
                "servers": {name: table["server"] for name, table in vizier_query_report.get("raw_tables", {}).items()},
                "row_counts": catalog_row_counts,
                "reason": (None if vizier_query_report.get("status") == "complete"
                           else vizier_query_report.get("error", "No completed VizieR catalog query is cached.")),
            },
            "catalog_matched_candidate_source_ids": catalog_qualified_ids,
            "n_catalog_matched": len(catalog_qualified_ids) if vizier_query_report.get("status") == "complete" else None,
            "n_stars_used": 0,
            "stars_used": [],
            "per_frame_counts": [
                ({"observation_id": o["observation_id"], "detector": o["detector"],
                  "n_stars": frame_vetting_by_key[(o["observation_id"], int(o["detector"]))]["n_stars"],
                  "reason": frame_vetting_by_key[(o["observation_id"], int(o["detector"]))]["reason"],
                  "n_catalog_matched": frame_vetting_by_key[(o["observation_id"], int(o["detector"]))]["n_catalog_matched"],
                  "n_inside_cutout": frame_vetting_by_key[(o["observation_id"], int(o["detector"]))]["n_inside_cutout"],
                  "n_edge_safe": frame_vetting_by_key[(o["observation_id"], int(o["detector"]))]["n_edge_safe"],
                  "n_hard_flagged": frame_vetting_by_key[(o["observation_id"], int(o["detector"]))]["n_hard_flagged"],
                  "prephotometry_candidate_source_ids": frame_vetting_by_key[(o["observation_id"], int(o["detector"]))]["eligible_source_ids"],
                  "hard_flagged_source_ids": frame_vetting_by_key[(o["observation_id"], int(o["detector"]))]["hard_flagged_source_ids"]}
                 if (o["observation_id"], int(o["detector"])) in frame_vetting_by_key else
                 {"observation_id": o["observation_id"], "detector": o["detector"], "n_stars": None,
                  "reason": "frame-level cached FITS vetting did not complete"})
                for o in observations if o["usable"]
            ],
            "photometric_floor": {
                "status": "unavailable",
                "method": criteria["photometric_floor_method"],
                "sample_size": 0,
                "fractional_scatter": None,
                "floor_jy": None,
                "reason": (f"No comparison-star flux ratios were measured. Although {len(catalog_qualified_ids)} sources passed catalog gates, "
                           f"{frames_with_required_stars} of {usable_vetting_frame_count} usable cutouts met the required "
                           "20-star pre-photometry count; comparison photometry was therefore not started.")
                          if vizier_query_report.get("status") == "complete" and frame_vetting_report
                          else ("No 2MASS/AllWISE matched comparison-star flux ratios were measured; "
                                + (f"IRSA TAP error: {comparison_query_error}" if comparison_query_error else "No catalog query result is available."))
            },
            "astrometric_calibration": {
                "status": "unavailable",
                "method": criteria["astrometric_threshold_method"],
                "n_star_measurements": 0,
                "common_offset_east_arcsec": None,
                "common_offset_north_arcsec": None,
                "residual_scatter_arcsec": None,
                "displacement_threshold_arcsec": None,
                "reason": "No comparison-star centroids were measured; the common frame offset and residual scatter are unknown."
            }
        },
        "catalog_flux_comparison": catalog_flux_comparison,
        "spectral_analysis": {"model": POLICY["spectral_fit_model"], "parameters": model_parameters,
                              "chi2": float(chi2_without_floor), "n_observations": chi2_n,
                              "n_parameters": 3, "dof": chi2_dof,
                              "reduced_chi2": reduced_chi2_without_floor,
                              "chi2_with_floor": None, "reduced_chi2_with_floor": None,
                              "chi2_error_model": "Formal flux uncertainties only; no systematic floor is available.",
                              "constant_flux_test": {
                                  "status": "skipped", "statistic": None,
                                  "n_observations": None, "n_parameters": None, "dof": None,
                                  "p_value": None,
                                  "reason": "Skipped because the required comparison-star sample and systematic floor are unavailable.",
                                  "cannot_detect": "Even with a floor, this sparse wavelength-dependent sampling cannot exclude variability between SPHEREx epochs or variability faster than an individual exposure."
                              },
                              "model_curve": curve,
                              "per_observation": per_observation},
        "dr1_calibration_check": dr1_calibration_result,
        "spectral_verdict": {"label": "insufficient evidence", "evidence": {
            "summary": "A smooth empirical spectrum and robust residual screen are available, but the per-frame comparison-star sample is below the fixed minimum and no systematic floor was measured; no variability inference is made.",
            "n_observations": len(observations), "n_usable": usable_count,
            "wavelength_range_um": [float(min(valid_wavelengths)), float(max(valid_wavelengths))] if valid_wavelengths else None,
            "reasons": [f"Only {frame_count_min} to {frame_count_max} catalog-qualified candidates per usable cutout pass the edge and hard-flag gates, below the required 20.",
                        "No comparison-star residual MAD floor was measured.",
                        "A single control-star wavelength-dependent fit is not a temporal variability test."]}},
        "verdict": {"label": "insufficient evidence", "evidence": {
            "summary": "The control-star case has measured source positions but the cutouts contain fewer than the required 20 comparison candidates per frame, so no comparison-star astrometric solution or calibrated residual scatter was measured.",
            "n_observations": len(observations), "n_usable": usable_count,
            "baseline_days": float(max(Time(o["time_utc"]).mjd for o in observations) - min(Time(o["time_utc"]).mjd for o in observations)),
            "residual_rms_arcsec": None, "motion_significance_sigma": None,
            "reasons": [f"{len(catalog_qualified_ids)} candidates pass catalog gates, but no usable cutout has 20 candidates after footprint, edge, and hard-flag checks.",
                        "Common frame north offset and corrected scatter are unverified."]},
            "threshold_policy": {"id": "control-star-quality-and-spectral-policy-1", "minimum_usable_epochs": 3,
                "moving_significance_sigma": 5.0, "minimum_displacement_uncertainty_ratio": 3.0,
                "no_motion_confidence_level": 0.95, "no_motion_max_displacement_arcsec": None,
                **POLICY}},
        "confirmed_versus_assumed": {
            "confirmed": [f"{measured_frame_count} real IRSA SPHEREx cutouts have measured fluxes from cached FITS.",
                          "Two cached edge cutouts were reprocessed and now return unusable with explicit edge reasons.",
                          "The two specified high-flux frames are retained and rejected by the declared flag/MAD policy.",
                          "The provenance SHA-256 values are taken from cache or computed from the cached FITS.",
                          f"VizieR mirrors returned {catalog_row_counts.get('2mass')} 2MASS rows and {catalog_row_counts.get('allwise')} AllWISE rows; {len(catalog_qualified_ids)} of {len(gaia_candidate_ids)} candidates passed the fixed catalog gates.",
                          f"All {usable_vetting_frame_count} usable cutouts were checked for comparison-star footprint, edge margin, and decoded 5x5 hard flags; {frames_with_required_stars} met the required 20-candidate threshold."],
            "assumed": ["Gaia proper motion is propagated linearly; perspective acceleration is ignored.",
                       "The degree-2 log-log spectrum is an empirical description, not a physical stellar model.",
                       "Reprojected pixels are treated as independent in formal measurement errors."],
            "unverified": ([f"{len(failed_by_key)} transient download failures remain without cached FITS in this network-restricted environment."] if failed_by_key else []) + [
                           "The written 5-scaled-MAD policy followed exploratory inspection of these cached frames; classifications are screening results, not a preregistered test.",
                           "The 24 catalog-qualified candidates are pre-photometry selections; none were measured because every usable cutout has at most 7 eligible candidates.",
                           *([f"IRSA TAP comparison-catalog query failed: {comparison_query_error}; the CDS VizieR mirror fallback succeeded."]
                             if comparison_query_error and vizier_query_report.get("status") == "complete"
                             else ([f"VizieR catalog query failed: {vizier_query_report.get('error')}"]
                                   if vizier_query_report.get("status") != "complete" else [])),
                           "No systematic photometric floor; no common frame offset or comparison-star astrometric scatter.",
                           "No variability claim; spectral verdict remains insufficient evidence.",
                           "No motion threshold recalibration without comparison-star residuals."]},
        "provenance": {"products": products, "software": {"pipeline_version": "0.1.0",
                         "environment_lock": "requirements.txt; requirements-dev.txt",
                         "git_commit": _git_head()}},
    }
    hard_flag_union = sum(1 for o in series if set(o.get("flags_5x5", [])) & HARD_FLAGS)
    diagnostics = {"policy": POLICY, "rule_hit_counts": counts,
                   "retry_backoff_seconds": [5, 15, 45, 90],
                   "comparison_candidates_at_planned_radius": len(gaia_candidate_ids_at_planned_radius),
                   "comparison_candidates_at_selected_radius": len(gaia_candidate_ids),
                   "catalog_qualified_comparison_candidates": len(catalog_qualified_ids),
                   "comparison_frames_with_minimum_candidates": frames_with_required_stars,
                   "comparison_frame_min_candidates": frame_count_min,
                   "comparison_frame_max_candidates": frame_count_max,
                   "context_flag_frame_counts": context_counts,
                   "frames_removed_by_context_only_flags": 0,
                   "unique_frames_with_hard_flags": hard_flag_union,
                   "usable_frames": usable_count, "total_frames": len(observations),
                   "unrecovered_network_failures": [
                       {"observation_id": key[0], "detector": key[1],
                        "http_status": value.get("http_status"),
                        "response_body": value.get("response_body")}
                       for key, value in sorted(failed_by_key.items())],
                   "robust_spectrum": {k: v for k, v in spectral.items() if k not in {"outliers", "per_observation"}},
                   "outlier_keys": [list(k) for k in sorted(fit_keys)],
                   "null_fields": sorted(null_fields),
                   "comparison_star_floor": None,
                   "reduced_chi_square_with_floor": None,
                   "reduced_chi_square_without_floor": reduced_chi2_without_floor,
                   "chi2": chi2_without_floor, "chi2_n_observations": chi2_n,
                   "chi2_n_parameters": 3, "chi2_dof": chi2_dof,
                   "usable_unique_utc_nights": len({o["time_utc"][:10] for o in observations if o["usable"]}),
                   "usable_wavelength_range_um": [float(min(valid_wavelengths)), float(max(valid_wavelengths))] if valid_wavelengths else None}
    return case, diagnostics


def _null_paths(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if value is None:
        return [prefix]
    if isinstance(value, dict):
        for key, child in value.items():
            found.extend(_null_paths(child, f"{prefix}.{key}" if prefix else key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_null_paths(child, f"{prefix}[{index}]"))
    return found


def _reason_for_null(case: dict[str, Any], path: str) -> str:
    if path == "target.designation":
        return "No separate common-name designation is used; target identity is supplied by Gaia DR3 catalog_id."
    if path.startswith("observations["):
        index = int(path.split("[", 1)[1].split("]", 1)[0])
        observation = case["observations"][index]
        tail = path.split("].", 1)[1] if "]." in path else ""
        exclusion = observation.get("exclusion_reason")
        if tail == "wavelength_pixel_xy":
            return "The wavelength lookup was performed at the mapped detector pixel, but that raw pixel coordinate was not retained in this case export."
        if tail == "quality_flags.raw_value":
            return "The packed raw flag integer was not retained; decoded flag names are recorded in quality_flags.flag_names."
        if tail.startswith("predicted_position.uncertainty_"):
            return "A propagated positional covariance was not available; the recorded Gaia prediction omits catalog uncertainty and frame systematics."
        if tail.startswith("registration."):
            return "No multi-star astrometric refinement was performed; registration.method records the reprojection-only method."
        if tail.endswith("uncertainty_ra_arcsec") or tail.endswith("uncertainty_dec_arcsec"):
            return "A centroid positional uncertainty was not calculated by the current source-position export."
        if tail in ("ra_deg", "dec_deg", "measured_position.ra_deg", "measured_position.dec_deg",
                    "position_residual_arcsec", "image.source_xy", "image.predicted_xy"):
            return (f"No reliable source centroid was available for this frame; {exclusion}" if exclusion
                    else "No reliable source centroid was available, so this coordinate or residual was not computed.")
        if tail.startswith("flux.") or tail.startswith("effective_wavelength_") or tail in ("wavelength_width_um", "wavelength_uncertainty_um", "wavelength_source"):
            return (f"The FITS product could not provide this measurement; {exclusion}" if exclusion
                    else "The measurement was not available from the processed FITS product.")
        if tail.startswith("image."):
            return (f"No usable source position was associated with the image; {exclusion}" if exclusion
                    else "The source position was not available for this image field.")
    if path.startswith("comparison_star_analysis.per_frame_counts["):
        return "Per-frame comparison-star catalog matching and photometry were not run because no catalog-qualified sample was confirmed."
    if path == "comparison_star_analysis.n_catalog_matched":
        return "The IRSA TAP and VizieR catalog queries failed in this execution; catalog-match count is unknown."
    if path.startswith("comparison_star_analysis.photometric_floor."):
        return case["comparison_star_analysis"]["photometric_floor"].get("reason") or "No comparison-star flux-ratio sample was measured."
    if path.startswith("comparison_star_analysis.astrometric_calibration."):
        return case["comparison_star_analysis"]["astrometric_calibration"].get("reason") or "No comparison-star astrometric sample was measured."
    if path.startswith("catalog_flux_comparison.catalog_flux_points[") and path.endswith("zero_point_uncertainty_jy"):
        return "A separate AllWISE zero-point uncertainty was not supplied; the catalog magnitude uncertainty is recorded and propagated."
    if path.startswith("spectral_analysis.per_observation["):
        return "No model residual is available because the corresponding observation lacks a fitted usable flux."
    if path in ("spectral_analysis.chi2_with_floor", "spectral_analysis.reduced_chi2_with_floor"):
        return "A comparison-star systematic floor was not measured, so a with-floor statistic cannot be computed."
    if path.startswith("spectral_analysis.constant_flux_test."):
        return case["spectral_analysis"]["constant_flux_test"].get("reason") or "The constant-flux test was skipped because no systematic floor is available."
    if path.endswith("hdu_summary[0].shape"):
        return "The FITS PRIMARY HDU has no image array; its shape is null by product structure."
    if path in ("verdict.evidence.residual_rms_arcsec", "verdict.evidence.motion_significance_sigma"):
        return "No comparison-star astrometric calibration was measured, so residual significance is unknown."
    if path == "verdict.threshold_policy.no_motion_max_displacement_arcsec":
        return "The displacement threshold awaits the comparison-star residual scatter and remains unmeasured."
    return "This value was not measured or supplied; it remains null rather than being inferred."


def export() -> tuple[int, list[str]]:
    case, diagnostics = build_case()
    null_fields = _null_paths(case)
    case["null_field_reasons"] = {path: _reason_for_null(case, path) for path in null_fields}
    schema = json.loads((ROOT / "schemas/case.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(case)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = json.loads(json.dumps(case))
    # A bundle is self-contained: replace local paths with encoded PNGs.
    for obs in bundle["observations"]:
        if obs["image"]:
            path = obs["image"]["path"]
            if not path.startswith("data:image/png;base64,"):
                raise ValueError("expected generated data URI")
    case_path = OUTPUT_DIR / "case.json"
    bundle_path = OUTPUT_DIR / "case_bundle.json"
    path_case = json.loads(json.dumps(case))
    for obs in path_case["observations"]:
        if obs["image"] is not None:
            obs["image"]["path"] = f"images/{obs['observation_id']}_D{obs['detector']}_128.png"
    case_path.write_text(json.dumps(path_case, indent=2, allow_nan=False), encoding="utf-8")
    bundle_path.write_text(json.dumps(bundle, indent=2, allow_nan=False), encoding="utf-8")
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(json.loads(bundle_path.read_text(encoding="utf-8")))
    diagnostics["null_fields"] = null_fields
    diagnostics["bundle_size_bytes"] = bundle_path.stat().st_size
    usable_count = sum(bool(o["usable"]) for o in case["observations"])
    diagnostics_path = OUTPUT_DIR / "quality_report.json"
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2, allow_nan=False), encoding="utf-8")
    utc_dates = sorted({o["time_utc"][:10] for o in case["observations"] if o["usable"] and o["time_utc"]})
    usable_wavelengths = [o["effective_wavelength_um"] for o in case["observations"]
                          if o["usable"] and o["effective_wavelength_um"] is not None]
    flag_counts: dict[str, int] = {}
    for observation in case["observations"]:
        for flag in observation["quality_flags"]["flag_names"]:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
    mad_frames = [f"{obs_id}/D{detector}" for obs_id, detector in diagnostics["outlier_keys"]]
    analysis = case["spectral_analysis"]
    comparison = case["comparison_star_analysis"]
    slide_numbers = {
        "case_id": CASE_ID,
        "figures": {
            "planned_frames": {"value": len(case["observations"]), "bundle_path": "observations.length"},
            "usable_frames": {"value": usable_count, "bundle_path": "observations[*].usable"},
            "excluded_frames": {"value": len(case["observations"]) - usable_count, "bundle_path": "observations[*].usable"},
            "distinct_usable_utc_dates": {"value": len(utc_dates), "dates": utc_dates,
                                            "bundle_path": "observations[*].time_utc; count unique UTC calendar dates"},
            "wavelength_range_um": {"value": [float(min(usable_wavelengths)), float(max(usable_wavelengths))] if usable_wavelengths else None,
                                     "bundle_path": "observations[*].effective_wavelength_um where usable=true"},
            "quality_flag_counts": {"value": flag_counts, "bundle_path": "observations[*].quality_flags.flag_names"},
            "mad_residual_hits": {"value": len(mad_frames), "frame_ids": mad_frames,
                                  "bundle_path": "observations[*].exclusion_reason and spectral_analysis.per_observation[*].normalized_residual"},
            "chi2": {"value": analysis["chi2"], "bundle_path": "spectral_analysis.chi2"},
            "n_observations": {"value": analysis["n_observations"], "bundle_path": "spectral_analysis.n_observations"},
            "n_parameters": {"value": analysis["n_parameters"], "bundle_path": "spectral_analysis.n_parameters"},
            "dof": {"value": analysis["dof"], "bundle_path": "spectral_analysis.dof"},
            "reduced_chi2": {"value": analysis["reduced_chi2"], "bundle_path": "spectral_analysis.reduced_chi2"},
            "chi2_with_floor": {"value": analysis["chi2_with_floor"], "bundle_path": "spectral_analysis.chi2_with_floor"},
            "reduced_chi2_with_floor": {"value": analysis["reduced_chi2_with_floor"], "bundle_path": "spectral_analysis.reduced_chi2_with_floor"},
            "photometric_floor": {"value": comparison["photometric_floor"], "bundle_path": "comparison_star_analysis.photometric_floor"},
            "astrometric_displacement_threshold_arcsec": {"value": comparison["astrometric_calibration"]["displacement_threshold_arcsec"],
                                                           "bundle_path": "comparison_star_analysis.astrometric_calibration.displacement_threshold_arcsec"},
            "comparison_stars_used": {"value": comparison["n_stars_used"], "candidate_count": comparison["n_gaia_candidates"],
                                       "bundle_path": "comparison_star_analysis.n_stars_used; comparison_star_analysis.n_gaia_candidates"},
            "catalog_qualified_comparison_stars": {"value": comparison["n_catalog_matched"],
                                                    "source_ids": comparison["catalog_matched_candidate_source_ids"],
                                                    "bundle_path": "comparison_star_analysis.catalog_matched_candidate_source_ids"},
            "comparison_candidates_per_frame": {"minimum": diagnostics["comparison_frame_min_candidates"],
                                                  "maximum": diagnostics["comparison_frame_max_candidates"],
                                                  "frames_meeting_minimum": diagnostics["comparison_frames_with_minimum_candidates"],
                                                  "usable_frames": len(comparison["per_frame_counts"]),
                                                  "bundle_path": "comparison_star_analysis.per_frame_counts[*]"},
            "comparison_catalog_query": {"value": comparison["catalog_query"],
                                          "bundle_path": "comparison_star_analysis.catalog_query"},
            "comparison_star_criteria": {"value": case["verdict"]["threshold_policy"]["comparison_star_criteria"],
                                          "bundle_path": "verdict.threshold_policy.comparison_star_criteria"},
            "comparison_star_radius_deviation": {
                "value": {"planned_radius_deg": PLANNED_GAIA_SEARCH_RADIUS_DEG,
                          "selected_radius_deg": GAIA_SEARCH_RADIUS_DEG,
                          "candidate_count_at_planned_radius": diagnostics["comparison_candidates_at_planned_radius"],
                          "candidate_count_at_selected_radius": diagnostics["comparison_candidates_at_selected_radius"],
                          "decision_basis": "Count only, before catalog matching or floor/statistic calculation."},
                "bundle_path": "comparison_star_analysis.selection_reason"},
            "dr1_calibration_check": {"value": case["dr1_calibration_check"], "bundle_path": "dr1_calibration_check"},
            "catalog_flux_ratios": {"value": [row["flux_ratio_spherex_to_catalog"] for row in case["catalog_flux_comparison"]["comparisons"]],
                                     "signed_deviation_percent": [(row["flux_ratio_spherex_to_catalog"] - 1.0) * 100.0 for row in case["catalog_flux_comparison"]["comparisons"]],
                                     "absolute_deviation_percent": [abs(row["flux_ratio_spherex_to_catalog"] - 1.0) * 100.0 for row in case["catalog_flux_comparison"]["comparisons"]],
                                     "bundle_path": "catalog_flux_comparison.comparisons[*].flux_ratio_spherex_to_catalog"},
            "five_mad_exploratory_caveat": {"value": "The 5-scaled-MAD policy was set after exploratory inspection.",
                                             "bundle_path": "confirmed_versus_assumed.unverified"}
        }
    }
    (OUTPUT_DIR / "slide_numbers.json").write_text(json.dumps(slide_numbers, indent=2, allow_nan=False), encoding="utf-8")
    return bundle_path.stat().st_size, diagnostics["null_fields"]


if __name__ == "__main__":
    size, null_fields = export()
    print(f"schema=1.7.0 validated=true bundle_bytes={size}")
    print(f"null_fields_count={len(null_fields)} full_list=outputs/control-star/quality_report.json:null_fields")
