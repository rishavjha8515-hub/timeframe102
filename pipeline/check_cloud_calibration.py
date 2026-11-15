"""Download and compare one SPHEREx spectral-WCS cloud calibration map.

Run from the repository root on a network-connected machine:
    python -m pipeline.check_cloud_calibration --release dr1 --detector 1

The default observation is the first usable cached case observation for the
requested detector. The script checks the cached cutout's EPSF and WCS-WAVE
values against the standalone S3 CWAVE/CBAND map at the measured source pixel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

import numpy as np
import requests
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates

from .cutout_processing import cutout_to_detector_pixels, sample_wavelength_metadata


ROOT = Path(__file__).resolve().parents[1]
CASE_PATH = ROOT / "outputs/control-star/case.json"
FRAME_CACHE = ROOT / "pipeline/cache/control-star-series"
CACHE = ROOT / "pipeline/cache/control-star/dr1-cal-check/cloud"
BUCKET_URL = "https://nasa-irsa-spherex.s3.us-east-1.amazonaws.com"
CALIBRATION_VERSION = "cal-swcs-v5-2026-191"
MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024
EXPECTED_MAP_BYTES = 33_312_960


def _select_observation(case: dict[str, Any], detector: int, observation_id: str | None) -> dict[str, Any]:
    candidates = [o for o in case["observations"] if o.get("detector") == detector and o.get("usable")]
    if observation_id:
        candidates = [o for o in candidates if o.get("observation_id") == observation_id]
    if not candidates:
        raise ValueError(f"No usable case observation found for detector {detector}"
                         + (f" and ID {observation_id}" if observation_id else ""))
    return candidates[0]


def _download_map(key: str, path: Path, *, session: Any = requests) -> dict[str, Any]:
    url = f"{BUCKET_URL}/{key}"
    response = session.get(url, stream=True, timeout=(15, 90))
    status = int(response.status_code)
    length_header = response.headers.get("Content-Length")
    declared_size = int(length_header) if length_header and length_header.isdigit() else None
    if status != 200:
        body = response.text[:2000]
        response.close()
        raise RuntimeError(json.dumps({"http_status": status, "url": url, "body": body}))
    if declared_size is not None and declared_size > MAX_DOWNLOAD_BYTES:
        response.close()
        raise RuntimeError(f"Map exceeds {MAX_DOWNLOAD_BYTES} byte cap: Content-Length={declared_size}")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    digest = hashlib.sha256()
    size = 0
    try:
        with temporary.open("wb") as output:
            for block in response.iter_content(chunk_size=1024 * 1024):
                if not block:
                    continue
                size += len(block)
                if size > MAX_DOWNLOAD_BYTES:
                    raise RuntimeError(f"Map exceeded {MAX_DOWNLOAD_BYTES} byte streaming cap")
                digest.update(block)
                output.write(block)
        if declared_size is not None and size != declared_size:
            raise RuntimeError(f"Incomplete download: received {size} bytes; Content-Length={declared_size}")
        os.replace(temporary, path)
    finally:
        response.close()
        if temporary.exists():
            temporary.unlink()
    return {"url": url, "http_status": status, "content_length": declared_size,
            "bytes_downloaded": size, "sha256": digest.hexdigest(),
            "etag": response.headers.get("ETag", "").strip('"') or None}


def sample_calibration_map(hdu_data: np.ndarray, detector_x_fits: float, detector_y_fits: float) -> float:
    """Bilinearly sample a full-detector calibration image in 1-based pixels."""
    image = np.asarray(hdu_data, dtype=float)
    if image.ndim != 2:
        raise ValueError(f"Calibration map must be 2-D; got shape {image.shape}")
    ny, nx = image.shape
    if not (1.0 <= detector_x_fits <= nx and 1.0 <= detector_y_fits <= ny):
        raise ValueError("Mapped source lies outside the calibration image")
    value = float(map_coordinates(image, [[detector_y_fits - 1], [detector_x_fits - 1]],
                                  order=1, mode="nearest", prefilter=False)[0])
    if not np.isfinite(value):
        raise ValueError("Calibration map has no finite value at the mapped source pixel")
    return value


def compare_one(*, release: str = "dr1", detector: int = 1,
                observation_id: str | None = None, session: Any = requests) -> dict[str, Any]:
    if release not in ("dr1", "qr3"):
        raise ValueError("release must be dr1 or qr3")
    case = json.loads(CASE_PATH.read_text(encoding="utf-8"))
    observation = _select_observation(case, detector, observation_id)
    obsid = observation["observation_id"]
    frame_path = FRAME_CACHE / f"{obsid}_D{detector}.fits"
    if not frame_path.is_file():
        raise FileNotFoundError(f"Cached cutout is missing for {obsid} detector {detector}")

    filename = f"spectral_wcs_D{detector}_spx_{CALIBRATION_VERSION}.fits"
    key = f"{release}/spectral_wcs/{CALIBRATION_VERSION}/{detector}/{filename}"
    map_path = CACHE / release / filename
    download = _download_map(key, map_path, session=session)
    if download["bytes_downloaded"] != EXPECTED_MAP_BYTES:
        raise RuntimeError(f"Unexpected calibration-map size: {download['bytes_downloaded']} bytes; "
                           f"bucket inventory reported {EXPECTED_MAP_BYTES}")

    with fits.open(frame_path, memmap=False) as cutout:
        sky_wcs = WCS(cutout["IMAGE"].header).celestial
        source_ra, source_dec = observation["measured_position"]["ra_deg"], observation["measured_position"]["dec_deg"]
        source_x, source_y = sky_wcs.world_to_pixel_values(source_ra, source_dec)
        sampled = sample_wavelength_metadata(cutout, float(source_x), float(source_y))
        detector_x, detector_y, transform = cutout_to_detector_pixels(cutout, float(source_x), float(source_y))
        with fits.open(map_path, memmap=True) as calibration:
            required = ("CWAVE", "CBAND", "WCS-WAVE")
            missing = [name for name in required if name not in calibration]
            if missing:
                raise ValueError(f"Downloaded calibration product lacks HDUs: {missing}")
            cal_wave = sample_calibration_map(calibration["CWAVE"].data, detector_x, detector_y)
            cal_band = sample_calibration_map(calibration["CBAND"].data, detector_x, detector_y)
            calibration_hdu_summary = [
                {"name": hdu.name, "shape": list(hdu.data.shape) if hdu.data is not None else None}
                for hdu in calibration
            ]
            calibration_version = calibration[0].header.get("VERSION")
            calibration_processing_date = calibration[0].header.get("DATE")

    return {
        "status": "compared",
        "release": release,
        "observation_id": obsid,
        "detector": detector,
        "time_utc": observation["time_utc"],
        "calibration_product": {"key": key, **download,
                                "version_header": calibration_version,
                                "processing_date_header": calibration_processing_date,
                                "hdu_summary": calibration_hdu_summary},
        "source_position": {"ra_deg": source_ra, "dec_deg": source_dec,
                            "cutout_xy_zero_based": [float(source_x), float(source_y)],
                            "detector_xy_fits_one_based": [detector_x, detector_y],
                            "coordinate_transform": transform},
        "wavelength_comparison_um": {
            "epsf_cwave": sampled["epsf_cwave_um"],
            "epsf_cband": sampled["wavelength_width_um"],
            "mef_wcs_wave": sampled["wcs_wave_um"],
            "mef_wcs_wave_cband": sampled["wcs_wave_bandwidth_um"],
            "standalone_cwave": cal_wave,
            "standalone_cband": cal_band,
            "standalone_minus_epsf_cwave": cal_wave - float(sampled["epsf_cwave_um"]),
            "standalone_minus_mef_wcs_wave": (cal_wave - float(sampled["wcs_wave_um"])
                                              if sampled["wcs_wave_um"] is not None else None),
            "standalone_minus_epsf_cband": cal_band - float(sampled["wavelength_width_um"]),
        },
        "confirmed_versus_assumed": {
            "confirmed": ["S3 key was returned by the user's public-bucket inventory.",
                          "Map HDUs were sampled at the cached cutout's mapped detector pixel."],
            "assumed": ["The listed DR1/QR3 calibration version is the intended match for the cached case product; verify from the FITS provenance headers and IRSA response."],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", choices=("dr1", "qr3"), default="dr1")
    parser.add_argument("--detector", type=int, choices=range(1, 7), default=1)
    parser.add_argument("--observation-id")
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    output = CACHE / "cloud_calibration_comparison.json"
    obs_id = args.observation_id
    key = f"{args.release}/spectral_wcs/{CALIBRATION_VERSION}/{args.detector}/spectral_wcs_D{args.detector}_spx_{CALIBRATION_VERSION}.fits"
    try:
        selected = _select_observation(json.loads(CASE_PATH.read_text(encoding="utf-8")),
                                       args.detector, args.observation_id)
        obs_id = selected["observation_id"]
        report = compare_one(release=args.release, detector=args.detector,
                             observation_id=obs_id)
    except Exception as exc:
        report = {"status": "failed", "release": args.release, "detector": args.detector,
                  "observation_id": obs_id, "requested_key": key,
                  "requested_url": f"{BUCKET_URL}/{key}",
                  "expected_bytes_from_inventory": EXPECTED_MAP_BYTES,
                  "attempted_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                  "error_type": type(exc).__name__, "error": str(exc)}
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    print(f"OUTPUT {output.relative_to(ROOT).as_posix()}")
    return 0 if report.get("status") == "compared" else 1


if __name__ == "__main__":
    raise SystemExit(main())
