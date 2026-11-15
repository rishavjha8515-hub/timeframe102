"""Compare a DR1 calibrated-level cutout wavelength map with the case value.

Run from the repository root where IRSA is reachable:
    python -m pipeline.check_dr1_calibration

The script queries the advertised ``spherex_dr1`` SIA collection at calibration
level 2, selects the first usable case epoch available at the same observation
ID and detector, and inspects EPSF.CWAVE/CBAND and WCS-WAVE. It does not use the
unadvertised ``spherex_dr1_cal`` collection name or substitute another epoch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astroquery.ipac.irsa import Irsa

from .cutout_processing import process_fits_product
from .probe_control_star import cutout_url, detector_from_url
from .probe_star_flux_and_cutouts import _download

ROOT = Path(__file__).resolve().parents[1]
CASE_PATH = ROOT / "outputs/control-star/case.json"
CACHE = ROOT / "pipeline/cache/control-star/dr1-cal-check"
COLLECTION = "spherex_dr1"
CALIB_LEVEL = 2


def _query_sia(ra_deg: float, dec_deg: float) -> Table:
    return Irsa.query_sia(pos=(ra_deg, dec_deg, 0.001), collection=COLLECTION,
                          calib_level=CALIB_LEVEL, maxrec=10000)


def _matching_epoch(case: dict[str, Any], table: Table) -> tuple[dict[str, Any] | None, Any | None]:
    rows = []
    for _, row in enumerate(table):
        obs_id = str(row["obs_id"])
        url = str(row["access_url"])
        detector = detector_from_url(url)
        if detector is None:
            continue
        rows.append((obs_id, detector, row))
    by_key = {(obs, det): row for obs, det, row in rows}
    for observation in case["observations"]:
        if not observation["usable"]:
            continue
        row = by_key.get((observation["observation_id"], observation["detector"]))
        if row is not None:
            return observation, row
    return None, None


def main() -> int:
    CACHE.mkdir(parents=True, exist_ok=True)
    case = json.loads(CASE_PATH.read_text(encoding="utf-8"))
    coord = case["target"]["coordinates"]
    result: dict[str, Any] = {
        "status": "not applicable", "collection": COLLECTION, "calib_level": CALIB_LEVEL,
        "case_path": "outputs/control-star/case.json", "matched_observation": None,
        "comparison": None, "error": None,
        "http_status": None, "query_status": None, "sia_row_count": None,
    }
    query_succeeded = False
    try:
        table = _query_sia(coord["ra_deg"], coord["dec_deg"])
        query_succeeded = True
        result["http_status"] = 200
        result["query_status"] = "OK"
        result["sia_row_count"] = len(table)
        table.write(CACHE / "spherex_dr1_cal_sia.ecsv", format="ascii.ecsv", overwrite=True)
        observation, row = _matching_epoch(case, table)
        if observation is None:
            result["reason"] = (
                f"The advertised {COLLECTION} SIA query at CALIB={CALIB_LEVEL} returned {len(table)} rows, "
                "but none matched a usable case observation ID and detector; no epoch was substituted."
            )
        else:
            result["matched_observation"] = {
                "observation_id": observation["observation_id"],
                "detector": observation["detector"],
            }
            url = cutout_url(str(row["access_url"]), coord["ra_deg"], coord["dec_deg"], size_deg=0.25)
            fits_path = CACHE / f"{observation['observation_id']}_D{observation['detector']}.fits"
            with requests.Session() as session:
                size, digest, status, content_length = _download(
                    session, url, fits_path, max_bytes=50 * 1024 * 1024, announce_length=True)
            with fits.open(fits_path, memmap=True) as hdul:
                extensions = [{"name": hdu.name, "shape": list(hdu.data.shape) if hdu.data is not None else None}
                              for hdu in hdul]
                hdr_obs = str(hdul["IMAGE"].header.get("OBSID", ""))
                hdr_detector = int(hdul["IMAGE"].header.get("DETECTOR", -1))
            if hdr_obs != observation["observation_id"] or hdr_detector != observation["detector"]:
                raise ValueError(f"download product identity mismatch: OBSID={hdr_obs}, DETECTOR={hdr_detector}")
            mjd = float(row["t_min"] + row["t_max"]) / 2.0
            time_utc = Time(mjd, format="mjd", scale="utc").isot + "Z"
            sampled = process_fits_product(
                fits_path, output_dir=CACHE / "northup", ra_deg=coord["ra_deg"], dec_deg=coord["dec_deg"],
                observation_id=observation["observation_id"], target_name=case["target"]["name"],
                detector=observation["detector"], time_utc=time_utc, mjd_midpoint=mjd,
            )
            values = [float(o["effective_wavelength_um"]) for o in case["observations"]
                      if o["observation_id"] == observation["observation_id"] and
                      o["detector"] == observation["detector"] and o["effective_wavelength_um"] is not None]
            case_value = values[0] if values else None
            dr1_value = sampled.get("effective_wavelength_um")
            difference = abs(float(dr1_value) - case_value) if dr1_value is not None and case_value is not None else None
            result.update({
                "status": "pass" if difference is not None else "fail",
                "matched_observation": {"observation_id": observation["observation_id"],
                                        "detector": observation["detector"], "time_utc": time_utc},
                "product": {"access_url": url, "sha256": digest, "bytes": size,
                            "content_length": content_length, "http_status": status,
                            "extensions": extensions},
                "comparison": {"case_effective_wavelength_um": case_value,
                               "dr1_cal_epsf_cwave_um": sampled.get("epsf_cwave_um"),
                               "dr1_cal_wcs_wave_um": sampled.get("wcs_wave_um"),
                               "dr1_cal_band_width_um": sampled.get("wavelength_width_um"),
                               "absolute_difference_um": difference},
                "reason": "Matched same observation ID and detector; sampled with the mapped-pixel wavelength method."
                          if difference is not None else "A source-position wavelength could not be measured in the calibrated cutout.",
            })
    except Exception as exc:
        result["status"] = "fail" if query_succeeded else "not applicable"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["reason"] = "IRSA query, product retrieval, or product processing failed; no calibration comparison is asserted."
    output = CACHE / "dr1_calibration_check.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, allow_nan=False))
    print(f"OUTPUT {output.relative_to(ROOT).as_posix()}")
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
