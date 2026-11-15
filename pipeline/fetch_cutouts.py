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
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import numpy as np
import requests
from astropy.table import Table
from astropy.time import Time

from .cutout_processing import process_fits_product
from .fetch_and_inspect import resolve_product_url

MAX_CUTOUT_BYTES = 50 * 1024 * 1024
RETRY_BACKOFF_SECONDS = (5, 15, 45, 90)
MAX_CONSECUTIVE_FAILURES = 5


class CutoutDownloadFailure(RuntimeError):
    def __init__(self, message: str, *, http_status: int | None = None,
                 response_body: str | None = None, attempts: int = 1):
        super().__init__(message)
        self.http_status = http_status
        self.response_body = response_body
        self.attempts = attempts


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


def _retry_delay(headers: dict, attempt: int) -> float:
    fallback = RETRY_BACKOFF_SECONDS[min(attempt - 1, len(RETRY_BACKOFF_SECONDS) - 1)]
    raw = headers.get("Retry-After")
    if not raw:
        return float(fallback)
    try:
        return max(float(fallback), float(raw))
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(raw)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(float(fallback), (retry_at - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return float(fallback)


def _download(session: requests.Session, url: str, destination: Path, *,
              max_bytes: int = MAX_CUTOUT_BYTES, sleep=time.sleep) -> tuple[int, str, int, int]:
    """GET a small cutout directly; retry transient failures and cap its size."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    retryable = (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                 requests.exceptions.ChunkedEncodingError, OSError)
    last_status: int | None = None
    last_body: str | None = None
    for attempt in range(1, len(RETRY_BACKOFF_SECONDS) + 2):
        response = None
        try:
            response = session.get(url, stream=True, allow_redirects=True, timeout=180)
            last_status = int(response.status_code)
            if response.status_code in (429, 503):
                last_body = response.text[:4000]
                raise requests.exceptions.HTTPError(f"HTTP {response.status_code}: {last_body}")
            if response.status_code >= 400:
                last_body = response.text[:4000]
                raise CutoutDownloadFailure(f"HTTP {response.status_code}: {last_body}",
                    http_status=last_status, response_body=last_body, attempts=attempt)
            content_length = response.headers.get("Content-Length")
            expected = int(content_length) if content_length and content_length.isdigit() else None
            if expected is not None and expected > max_bytes:
                raise CutoutDownloadFailure(
                    f"Content-Length {expected} exceeds {max_bytes}-byte cap",
                    http_status=last_status, response_body="", attempts=attempt)
            digest, total = hashlib.sha256(), 0
            with destination.open("wb") as output:
                for block in response.iter_content(1024 * 1024):
                    if not block:
                        continue
                    total += len(block)
                    if total > max_bytes:
                        raise CutoutDownloadFailure(
                            f"stream exceeded {max_bytes}-byte cap",
                            http_status=last_status, response_body="", attempts=attempt)
                    output.write(block)
                    digest.update(block)
            if expected is not None and total != expected:
                raise requests.exceptions.ChunkedEncodingError(
                    f"incomplete stream: received {total} of {expected} bytes")
            return total, digest.hexdigest(), last_status, attempt
        except CutoutDownloadFailure:
            destination.unlink(missing_ok=True)
            raise
        except (requests.exceptions.HTTPError, *retryable) as exc:
            destination.unlink(missing_ok=True)
            if attempt >= len(RETRY_BACKOFF_SECONDS) + 1:
                if response is not None and last_body is None:
                    last_body = response.text[:4000]
                raise CutoutDownloadFailure(f"{type(exc).__name__}: {exc}",
                    http_status=last_status, response_body=last_body, attempts=attempt) from exc
            delay = _retry_delay(response.headers if response is not None else {}, attempt)
            print(f"DOWNLOAD_RETRY attempt={attempt} status={last_status} wait_seconds={delay:g}")
            sleep(delay)
        finally:
            if response is not None:
                response.close()
    raise CutoutDownloadFailure("retry loop ended unexpectedly", http_status=last_status,
                                response_body=last_body, attempts=len(RETRY_BACKOFF_SECONDS) + 1)


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
    consecutive_failures = 0
    with requests.Session() as session:
        session.headers.update({"User-Agent": "TIMEFRAME-102-SPHEREx-cutout/1.0"})
        for index, row in enumerate(rows, 1):
            url = None
            byte_count = None
            digest = None
            http_status = None
            attempts = None
            try:
                center_ra, center_dec = float(row["ra_obj"]), float(row["dec_obj"])
                url = make_cutout_url(resolve_product_url(str(row["uri"])), center_ra, center_dec, args.size_deg)
                dest = args.output_dir / f"{index:02d}_{row['_night']}_{row['_detector'] or 'D?'}_{Path(urlparse(url).path).name}"
                print(f"\nEPOCH {index}/{len(rows)} night={row['_night']} obsid={row['observationid']} detector={row['_detector']} "
                      f"mid_mjd={(float(row['mjd_start']) + float(row['mjd_end'])) / 2:.9f} center=({center_ra:.8f},{center_dec:.8f})")
                byte_count, digest, http_status, attempts = _download(session, url, dest)
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
                               "http_status": http_status, "download_attempts": attempts,
                               "utc_night": row["_night"], "product_id": str(row.get("productid", "")),
                               "overflow_flag_at_predicted_position": result["overflow_at_predicted_pixel"]})
                results.append(result)
                consecutive_failures = 0
                print(f"OVERFLOW_CHECK obsid={result['observation_id']} raw={result['central_flag_raw_value']} "
                      f"decoded={[f['name'] for f in result['central_flag_decoded']]} "
                      f"overflow_at_position={result['overflow_at_predicted_pixel']} "
                      f"overflow_near_position={result['overflow_within_2_pixel_radius_box']}")
            except Exception as exc:
                failures += 1
                consecutive_failures += 1
                failure = {"observation_id": str(row["observationid"]),
                           "product_id": str(row.get("productid", "")), "detector": row["_detector"],
                           "utc_night": row["_night"], "usable": False,
                           "status": "processing_failed" if http_status == 200 else "download_failed",
                           "access_url": url,
                           "downloaded_bytes": byte_count, "sha256": digest,
                           "http_status": getattr(exc, "http_status", http_status),
                           "download_attempts": getattr(exc, "attempts", attempts),
                           "response_body": getattr(exc, "response_body", None),
                           "exclusion_reason": f"{type(exc).__name__}: {exc}"}
                results.append(failure)
                print(f"EPOCH_FAILED night={row['_night']} obsid={row['observationid']} detector={row['_detector']} "
                      f"http_status={failure['http_status']} attempts={failure['download_attempts']} "
                      f"{type(exc).__name__}: {exc}; response_body={failure['response_body']!r}")
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    for remaining in rows[index:]:
                        results.append({"observation_id": str(remaining["observationid"]),
                            "product_id": str(remaining.get("productid", "")), "detector": remaining["_detector"],
                            "utc_night": remaining["_night"], "usable": False, "status": "not_attempted",
                            "exclusion_reason": "not attempted after five consecutive download/processing failures"})
                    print(f"STOPPING after {MAX_CONSECUTIVE_FAILURES} consecutive failures; remaining planned frames recorded.")
                    break
    report_path = args.output_dir / "victoria_cutout_report.json"
    report_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    succeeded = sum(item.get("status") not in ("download_failed", "processing_failed", "not_attempted")
                    for item in results)
    print(f"CUTOUT_FETCH_COMPLETE selected={len(rows)} processed={succeeded} failures={failures} report={report_path}")
    return 0 if failures == 0 and succeeded else 1


if __name__ == "__main__":
    sys.exit(main())
