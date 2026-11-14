"""Download a time-spread sample of Victoria SPHEREx cutouts and screen flags.

Run locally with IRSA connectivity from the project root::

    python -m pipeline.fetch_cutouts \
      --table pipeline/cache/probe/most_20000012_Victoria_smaller-body-screen.ecsv \
      --count 10 --output-dir pipeline/cache/victoria-cutouts

The selection is one row per UTC night, spaced across the available nights,
with detector diversity preferred. It reports OVERFLOW at the MOST-predicted
position; the bit is a quality warning, not by itself proof of image clipping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import numpy as np
import requests
from astropy.table import Table
from astropy.time import Time

from .cutout_processing import process_fits_product
from .fetch_and_inspect import resolve_product_url


def _night(row) -> str:
    return str(row["start_date"])[:10]


def _detector(row) -> int | None:
    match = re.search(r"D([1-6])", str(row.get("productid", "")))
    if match:
        return int(match.group(1))
    match = re.search(r"D([1-6])_spx_l2b", str(row.get("uri", "")))
    return int(match.group(1)) if match else None


def select_epoch_rows(table: Table, count: int = 10) -> list[dict]:
    """Select distinct UTC nights across the baseline and balance detectors."""
    required = {"start_date", "mjd_start", "mjd_end", "ra_obj", "dec_obj", "observationid", "uri"}
    missing = required.difference(table.colnames)
    if missing:
        raise ValueError(f"Victoria table is missing columns: {sorted(missing)}")
    by_night: dict[str, list[dict]] = {}
    seen_products: set[str] = set()
    for row in table:
        uri = str(row["uri"]).strip()
        if not uri or uri in seen_products:
            continue
        seen_products.add(uri)
        item = {key: (value.item() if hasattr(value, "item") else value) for key, value in zip(table.colnames, row)}
        item["_night"] = _night(row)
        item["_detector"] = _detector(row)
        by_night.setdefault(item["_night"], []).append(item)
    nights = sorted(by_night, key=lambda n: min(float(r["mjd_start"]) for r in by_night[n]))
    if count < 1 or len(nights) < count:
        raise ValueError(f"Need at least {count} distinct UTC nights; found {len(nights)}")
    chosen_nights = [nights[i] for i in np.linspace(0, len(nights) - 1, count, dtype=int)]
    detector_counts: dict[int, int] = {}
    selected = []
    for night in chosen_nights:
        candidates = sorted(
            by_night[night],
            key=lambda row: (detector_counts.get(row["_detector"], 0), float(row["mjd_start"]), str(row["observationid"])),
        )
        chosen = candidates[0]
        selected.append(chosen)
        if chosen["_detector"] is not None:
            detector_counts[chosen["_detector"]] = detector_counts.get(chosen["_detector"], 0) + 1
    return selected


def make_cutout_url(url: str, ra_deg: float, dec_deg: float, size_deg: float) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "irsa.ipac.caltech.edu":
        raise ValueError("Expected an IRSA HTTPS product URL")
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.update(center=f"{ra_deg:.10f},{dec_deg:.10f}", size=f"{size_deg:g}")
    return urlunparse(parsed._replace(query=urlencode(query)))


def _download(session: requests.Session, url: str, destination: Path) -> tuple[int, str]:
    head = session.head(url, allow_redirects=True, timeout=45)
    print(f"HEAD status={head.status_code} content_length={head.headers.get('Content-Length')}")
    head.raise_for_status()
    expected = int(head.headers["Content-Length"]) if head.headers.get("Content-Length", "").isdigit() else None
    get = session.get(url, stream=True, allow_redirects=True, timeout=180)
    print(f"GET status={get.status_code}")
    get.raise_for_status()
    digest, total = hashlib.sha256(), 0
    with destination.open("wb") as output:
        for block in get.iter_content(1024 * 1024):
            if block:
                output.write(block)
                digest.update(block)
                total += len(block)
    if expected is not None and total != expected:
        destination.unlink(missing_ok=True)
        raise IOError(f"Downloaded {total} bytes; HEAD reported {expected}")
    return total, digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=Path("pipeline/cache/probe/most_20000012_Victoria_smaller-body-screen.ecsv"))
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--size-deg", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, default=Path("pipeline/cache/victoria-cutouts"))
    args = parser.parse_args(argv)
    table = Table.read(args.table, format="ascii.ecsv")
    rows = select_epoch_rows(table, args.count)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"SELECTION n={len(rows)} nights={len({row['_night'] for row in rows})} detectors={sorted({row['_detector'] for row in rows if row['_detector']})}")
    results = []
    failures = 0
    with requests.Session() as session:
        session.headers.update({"User-Agent": "TIMEFRAME-102-SPHEREx-cutout/1.0"})
        for index, row in enumerate(rows, 1):
            try:
                center_ra, center_dec = float(row["ra_obj"]), float(row["dec_obj"])
                url = make_cutout_url(resolve_product_url(str(row["uri"])), center_ra, center_dec, args.size_deg)
                dest = args.output_dir / f"{index:02d}_{row['_night']}_{row['_detector'] or 'D?'}_{Path(urlparse(url).path).name}"
                print(f"\nEPOCH {index}/{len(rows)} night={row['_night']} obsid={row['observationid']} detector={row['_detector']} "
                      f"mid_mjd={(float(row['mjd_start']) + float(row['mjd_end'])) / 2:.9f} center=({center_ra:.8f},{center_dec:.8f})")
                byte_count, digest = _download(session, url, dest)
                result = process_fits_product(
                    dest,
                    output_dir=args.output_dir / "northup",
                    ra_deg=center_ra,
                    dec_deg=center_dec,
                    observation_id=str(row["observationid"]),
                    target_name="Victoria",
                    detector=row["_detector"],
                    time_utc=Time((float(row["mjd_start"]) + float(row["mjd_end"])) / 2,
                                  format="mjd", scale="utc").isot,
                    mjd_midpoint=(float(row["mjd_start"]) + float(row["mjd_end"])) / 2,
                )
                result.update({"download_url": url, "downloaded_bytes": byte_count, "sha256": digest,
                               "utc_night": row["_night"], "product_id": str(row.get("productid", "")),
                               "overflow_flag_at_predicted_position": result["overflow_at_predicted_pixel"]})
                results.append(result)
                print(f"OVERFLOW_CHECK obsid={result['observation_id']} raw={result['central_flag_raw_value']} "
                      f"decoded={[f['name'] for f in result['central_flag_decoded']]} "
                      f"overflow_at_position={result['overflow_at_predicted_pixel']} "
                      f"overflow_near_position={result['overflow_within_2_pixel_radius_box']}")
            except Exception as exc:
                failures += 1
                print(f"EPOCH_FAILED night={row['_night']} obsid={row['observationid']} {type(exc).__name__}: {exc}")
    report_path = args.output_dir / "victoria_cutout_report.json"
    report_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"CUTOUT_FETCH_COMPLETE selected={len(rows)} succeeded={len(results)} failures={failures} report={report_path}")
    return 0 if failures == 0 and results else 1


if __name__ == "__main__":
    sys.exit(main())
