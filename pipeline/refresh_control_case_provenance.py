"""Reconcile the four exported control-star products with cached FITS files."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table
from astropy.time import Time

SOURCE_ID = "1633306316700113024"


def detector_from_access_url(url: str) -> int:
    match = re.search(r"D([1-6])_spx_l2b", url)
    if not match:
        raise ValueError(f"Could not parse detector from product URL: {url}")
    return int(match.group(1))


def validate_binding(observation: dict, url: str, file_observation_id: str,
                     file_detector: int, expected_sha256: str, actual_sha256: str) -> None:
    url_detector = detector_from_access_url(url)
    if url_detector != observation["detector"] or file_detector != observation["detector"]:
        raise ValueError(f"detector mismatch for {observation['observation_id']}: "
                         f"observation={observation['detector']} url={url_detector} FITS={file_detector}")
    if file_observation_id != observation["observation_id"]:
        raise ValueError(f"observation ID mismatch: case={observation['observation_id']} FITS={file_observation_id}")
    if expected_sha256 != actual_sha256:
        raise ValueError(f"SHA256 mismatch for {observation['observation_id']}")


def refresh(case_path: Path, cache_dir: Path, sia_dir: Path, gaia_path: Path) -> dict:
    case = json.loads(case_path.read_text(encoding="utf-8"))
    source_paths = sorted(cache_dir.glob(f"{SOURCE_ID}_*_D*.fits"))
    products = []
    for obs in case["observations"]:
        matches = []
        for path in source_paths:
            with fits.open(path, memmap=True) as hdul:
                image = hdul["IMAGE"].header
                if (str(image.get("OBSID")) == obs["observation_id"]
                        and int(image.get("DETECTOR", -1)) == obs["detector"]):
                    matches.append((path, image.copy()))
        if len(matches) != 1:
            raise ValueError(f"Expected one FITS for {obs['observation_id']} D{obs['detector']}; found {len(matches)}")
        path, image_header = matches[0]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        matched_product = None
        matched_release = None
        for release in ("spherex_dr1_deep", "spherex_qr3_deep"):
            sia_path = sia_dir / f"sia_{SOURCE_ID}_{release}.ecsv"
            table = Table.read(sia_path, format="ascii.ecsv")
            for row in table:
                url = str(row["access_url"])
                if str(row["obs_id"]) == obs["observation_id"] and detector_from_access_url(url) == obs["detector"]:
                    matched_product, matched_release = url, release
                    break
            if matched_product:
                break
        if not matched_product:
            raise ValueError(f"No SIA URL matched {obs['observation_id']} D{obs['detector']}")
        validate_binding(obs, matched_product, str(image_header["OBSID"]), int(image_header["DETECTOR"]), digest, digest)
        retrieved = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        products.append({
            "archive": "NASA/IPAC IRSA SPHEREx",
            "product_id": obs["observation_id"],
            "product_type": "SPHEREx L2 spectral image server-side cutout",
            "release": matched_release,
            "access_url": matched_product,
            "retrieved_utc": retrieved,
            "sha256": digest,
        })
    case["provenance"]["products"] = products
    gaia = Table.read(gaia_path, format="ascii.ecsv")
    target = next(row for row in gaia if str(row["source_id"]) == SOURCE_ID)
    gaia_coord = SkyCoord(
        ra=float(target["ra"]) * u.deg, dec=float(target["dec"]) * u.deg,
        pm_ra_cosdec=float(target["pmra"]) * u.mas / u.yr,
        pm_dec=float(target["pmdec"]) * u.mas / u.yr,
        obstime=Time(2016.0, format="jyear"), frame="icrs",
    )
    for observation in case["observations"]:
        predicted = gaia_coord.apply_space_motion(new_obstime=Time(observation["time_utc"]))
        measured = SkyCoord(ra=observation["ra_deg"] * u.deg, dec=observation["dec_deg"] * u.deg)
        east = ((measured.ra - predicted.ra).wrap_at(180 * u.deg) * np.cos(predicted.dec)).to_value(u.arcsec)
        north = (measured.dec - predicted.dec).to_value(u.arcsec)
        observation["predicted_position"] = {
            "ra_deg": float(predicted.ra.deg), "dec_deg": float(predicted.dec.deg),
            "uncertainty_ra_arcsec": None, "uncertainty_dec_arcsec": None,
            "method": "Gaia DR3 position propagated linearly from J2016.0 using pm_ra_cosdec and pm_dec; Gaia coordinate uncertainties not propagated",
        }
        observation["position_residual_arcsec"] = [float(east), float(north)]
    return case


def main() -> int:
    case_path = Path("outputs/control-star/case.json")
    refreshed = refresh(case_path, Path("pipeline/cache/star-candidates"), Path("pipeline/cache/control-star"),
                        Path("pipeline/cache/control-star/gaia_north.ecsv"))
    case_path.write_text(json.dumps(refreshed, indent=2), encoding="utf-8")
    print(f"PROVENANCE_REFRESHED products={len(refreshed['provenance']['products'])} output={case_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
