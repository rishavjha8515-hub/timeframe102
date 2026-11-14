"""Fetch and measure a time/detector-spread control-star series; resumes cache.

Run where IRSA is reachable::

    python -m pipeline.fetch_control_star_series --count 80 --size-deg 0.25

Observation selection uses cached SIA rows, detector, and time only. Wavelength
grouping is performed after each downloaded frame is sampled at the mapped
source pixel; SIA em_min/em_max are not used to group observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astroquery.ipac.irsa import Irsa

from .cutout_processing import process_fits_product
from .probe_control_star import SIA_RADIUS_DEG, cutout_url, detector_from_url
from .probe_star_flux_and_cutouts import _download, plan_observations

SOURCE_ID = "1633306316700113024"
COLLECTIONS = ("spherex_dr1_deep", "spherex_qr3_deep")


def _read_or_query(cache_dir: Path) -> list[Table]:
    gaia = Table.read(cache_dir / "gaia_north.ecsv", format="ascii.ecsv")
    star = next(row for row in gaia if str(row["source_id"]) == SOURCE_ID)
    ra, dec = float(star["ra"]), float(star["dec"])
    tables = []
    for collection in COLLECTIONS:
        path = cache_dir / f"sia_{SOURCE_ID}_{collection}.ecsv"
        if path.exists():
            table = Table.read(path, format="ascii.ecsv")
        else:
            table = Irsa.query_sia(pos=(ra, dec, SIA_RADIUS_DEG), collection=collection, maxrec=10000)
            table.write(path, format="ascii.ecsv", overwrite=True)
        tables.append(table)
    return tables


def _verify_cached(path: Path, obs_id: str, detector: int) -> bool:
    try:
        with fits.open(path, memmap=True) as hdul:
            hdr = hdul["IMAGE"].header
            return str(hdr.get("OBSID")) == obs_id and int(hdr.get("DETECTOR", -1)) == detector
    except Exception:
        return False


def fetch_series(count: int, size_deg: float, cache_dir: Path, output_dir: Path) -> dict:
    if not 0.25 <= size_deg <= 0.30:
        raise ValueError("Requested larger cutout size must be within 0.25--0.30 degree")
    tables = _read_or_query(cache_dir)
    plan = plan_observations(tables, count)
    gaia = Table.read(cache_dir / "gaia_north.ecsv", format="ascii.ecsv")
    star = next(row for row in gaia if str(row["source_id"]) == SOURCE_ID)
    ra, dec = float(star["ra"]), float(star["dec"])
    output_dir.mkdir(parents=True, exist_ok=True)
    results, failures = [], []
    import requests
    with requests.Session() as session:
        session.headers.update({"User-Agent": "TIMEFRAME-102-control-star-series/1.0"})
        for index, row in enumerate(plan, 1):
            obs_id, detector = row["obs_id"], row["detector"]
            if detector_from_url(row["url"]) != detector:
                failures.append({"observation_id": obs_id, "detector": detector, "error": "SIA URL detector mismatch"})
                continue
            url = cutout_url(row["url"], ra, dec, size_deg=size_deg)
            path = output_dir / f"{obs_id}_D{detector}.fits"
            try:
                if path.exists() and _verify_cached(path, obs_id, detector):
                    content = path.read_bytes()
                    digest, status, size = hashlib.sha256(content).hexdigest(), "cache", len(content)
                else:
                    if path.exists():
                        path.unlink()
                    size, digest, status_code, _ = _download(
                        session, url, path, max_bytes=50 * 1024 * 1024)
                    status = str(status_code)
                if not _verify_cached(path, obs_id, detector):
                    raise ValueError("downloaded FITS OBSID/DETECTOR does not match selected row")
                mjd = float(row["mjd_midpoint"])
                utc = Time(mjd, format="mjd", scale="utc").isot + "Z"
                result = process_fits_product(
                    path, output_dir=output_dir / "northup", ra_deg=ra, dec_deg=dec,
                    observation_id=obs_id, target_name=f"Gaia DR3 {SOURCE_ID}", detector=detector,
                    time_utc=utc, mjd_midpoint=mjd,
                )
                results.append({
                    "observation_id": obs_id, "time_utc": utc, "mjd_midpoint": mjd,
                    "utc_night": row["utc_night"], "detector": detector,
                    "access_url": url, "sha256": digest, "bytes": size, "http_status": status,
                    "wavelength_um": result["effective_wavelength_um"],
                    "wavelength_width_um": result["wavelength_width_um"],
                    "wavelength_uncertainty_um": result["wavelength_uncertainty_um"],
                    "flux": result["photometry"],
                    "flags_at_star_pixel": result["central_flag_decoded"],
                    "flags_5x5": result["flags_within_2_pixel_radius_box"],
                    "measured_position": result["measured_world"],
                    "position_uncertainty_arcsec": [result["uncertainty_ra_arcsec"], result["uncertainty_dec_arcsec"]],
                    "preview_path": result["preview_path"],
                })
                print(f"SERIES {index}/{count} {obs_id} D{detector} night={row['utc_night']} "
                      f"status={status} bytes={size} wavelength_um={result['effective_wavelength_um']} "
                      f"flux_jy={result['photometry']['flux_jy']} flags={result['central_flag_decoded']}")
            except Exception as exc:
                status_code = getattr(exc, "status", None) or getattr(getattr(exc, "response", None), "status_code", None)
                body = getattr(exc, "body", str(exc))
                failures.append({"observation_id": obs_id, "detector": detector,
                                 "http_status": status_code, "response_body": str(body)[:2000]})
                print(f"SERIES_FAILED {obs_id} D{detector} HTTP={status_code} body={str(body)[:2000]!r}")
    results.sort(key=lambda r: (float("inf") if r["wavelength_um"] is None else r["wavelength_um"], r["mjd_midpoint"]))
    grouped = {}
    for result in results:
        wl = result["wavelength_um"]
        group = "unavailable" if wl is None else f"{wl:.1f} um"
        grouped.setdefault(group, []).append(result)
    report = {"source_id": SOURCE_ID, "requested_count": count, "cutout_size_deg": size_deg,
              "observations": results, "groups_by_measured_source_wavelength": grouped,
              "failures": failures, "is_synthetic": False}
    path = output_dir / "control_star_series.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"SERIES_COMPLETE requested={count} succeeded={len(results)} failed={len(failures)} report={path}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, choices=range(60, 101), default=80)
    parser.add_argument("--size-deg", type=float, default=0.25)
    parser.add_argument("--cache-dir", type=Path, default=Path("pipeline/cache/control-star"))
    parser.add_argument("--output-dir", type=Path, default=Path("pipeline/cache/control-star-series"))
    args = parser.parse_args()
    result = fetch_series(args.count, args.size_deg, args.cache_dir, args.output_dir)
    return 0 if not result["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
