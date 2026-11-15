"""Retry only prior transient control-star downloads from cached SIA metadata.

Run on a network-enabled machine from the project root after the pipeline has
recorded failures::

    .venv\\Scripts\\python.exe -m pipeline.retry_control_star_failures
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import requests
from astropy.table import Table
from astropy.time import Time

from .cutout_processing import process_fits_product
from .probe_control_star import cutout_url
from .probe_star_flux_and_cutouts import _download

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "pipeline/cache/control-star"
SERIES = ROOT / "pipeline/cache/control-star-series"
SOURCE_ID = "1633306316700113024"
RETRY_IDS = {
    ("2025W17_4B_0165_1", 5),
    ("2025W25_1B_0045_3", 4),
    ("2025W31_2A_0240_4", 1),
    ("2025W40_1A_0274_1", 5),
}


def retry_failures() -> dict:
    catalog = {}
    for path in sorted(CACHE.glob(f"sia_{SOURCE_ID}_*.ecsv")):
        for row in Table.read(path, format="ascii.ecsv"):
            url = str(row["access_url"])
            for obs_id, detector in RETRY_IDS:
                if str(row["obs_id"]) == obs_id and f"D{detector}_spx" in url:
                    catalog[(obs_id, detector)] = row
    gaia = Table.read(CACHE / "gaia_north.ecsv", format="ascii.ecsv")
    star = next(row for row in gaia if str(row["source_id"]) == SOURCE_ID)
    ra, dec = float(star["ra"]), float(star["dec"])
    SERIES.mkdir(parents=True, exist_ok=True)
    results = []
    with requests.Session() as session:
        session.headers.update({"User-Agent": "TIMEFRAME-102-control-star-series/1.0"})
        for obs_id, detector in sorted(RETRY_IDS):
            row = catalog.get((obs_id, detector))
            if row is None:
                results.append({"observation_id": obs_id, "detector": detector,
                                "status": "failed", "reason": "no cached SIA row found"})
                continue
            url = cutout_url(str(row["access_url"]), ra, dec, size_deg=0.25)
            path = SERIES / f"{obs_id}_D{detector}.fits"
            try:
                size, digest, status, content_length = _download(session, url, path, max_bytes=50 * 1024 * 1024)
                mjd = (float(row["t_min"]) + float(row["t_max"])) / 2
                utc = Time(mjd, format="mjd", scale="utc").isot + "Z"
                measured = process_fits_product(
                    path, output_dir=SERIES / "northup", ra_deg=ra, dec_deg=dec,
                    observation_id=obs_id, target_name=f"Gaia DR3 {SOURCE_ID}", detector=detector,
                    time_utc=utc, mjd_midpoint=mjd,
                )
                results.append({"observation_id": obs_id, "detector": detector,
                                "status": "downloaded_and_processed", "http_status": status,
                                "bytes": size, "content_length": content_length,
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                "usable": measured["usable"], "exclusion_reason": measured["exclusion_reason"]})
            except Exception as exc:
                results.append({"observation_id": obs_id, "detector": detector,
                                "status": "failed", "http_status": getattr(exc, "status", None),
                                "body": str(getattr(exc, "body", str(exc)))[:2000]})
    output = {"is_synthetic": False, "retry_policy_seconds": [5, 15, 45, 90], "results": results}
    (SERIES / "retry_report.json").write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    return output


if __name__ == "__main__":
    for item in retry_failures()["results"]:
        print(json.dumps(item, separators=(",", ":")))
