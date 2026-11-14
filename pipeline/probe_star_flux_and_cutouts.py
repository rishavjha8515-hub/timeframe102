"""Query near-IR catalogs and screen Gaia-star SPHEREx cutouts.

Run on a machine with IRSA network access from the repository root::

    .venv\\Scripts\\python.exe -m pipeline.probe_star_flux_and_cutouts \
      --cache-dir pipeline/cache/control-star --plan-count 40 --overflow-count 4

The faintest candidate is processed first. It GETs four time-spread cutouts,
retries transient failures, checks a 50 MB cap, inspects the first FITS MEF,
and continues after individual failures. After at least three failed cutouts,
it tries four capped full products and cuts them locally. Other stars are
attempted only after the faintest candidate succeeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import astropy.units as u
import numpy as np
import requests
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astroquery.ipac.irsa import Irsa

from .cutout_processing import process_fits_product
from .probe_control_star import cutout_url, detector_from_url

TARGETS = {
    "1633306316700113024": {"g_mag": 12.62},
}
RELEASES = ("spherex_dr1_deep", "spherex_qr3_deep")
MAX_CUTOUT_BYTES = 50 * 1024 * 1024
MAX_FULL_PRODUCT_BYTES = 150 * 1024 * 1024
RETRY_DELAYS_SECONDS = (5, 15, 45, 90)
REQUIRED_CUTOUT_EXTENSIONS = {"IMAGE", "FLAGS", "VARIANCE", "ZODI", "EPSF", "WCS-WAVE"}


def _coord_lookup(path: Path) -> dict[str, tuple[float, float]]:
    table = Table.read(path, format="ascii.ecsv")
    return {str(r["source_id"]): (float(r["ra"]), float(r["dec"])) for r in table}


def _night(mjd: float) -> str:
    return Time(mjd, format="mjd", scale="utc").isot[:10]


def plan_observations(tables: list[Table], count: int = 40) -> list[dict]:
    """Choose time-spread products, balanced across D1--D6, retaining repeat dates."""
    by_det: dict[int, dict[str, dict]] = {i: {} for i in range(1, 7)}
    for table in tables:
        required = {"access_url", "obs_id", "t_min", "t_max"}
        if not required.issubset(table.colnames):
            raise ValueError(f"SIA result lacks required columns {sorted(required - set(table.colnames))}")
        for row_index, row in enumerate(table):
            url = str(row["access_url"])
            det = detector_from_url(url)
            obs_id = str(row["obs_id"])
            if det not in by_det or not obs_id or obs_id == "masked":
                continue
            midpoint = (float(row["t_min"]) + float(row["t_max"])) / 2
            by_det[det].setdefault(obs_id, {
                "detector": det, "obs_id": obs_id, "url": url,
                "mjd_midpoint": midpoint, "utc_night": _night(midpoint),
            })
    quota, extra = divmod(count, 6)
    selected = []
    availability = {det: len(items) for det, items in by_det.items()}
    if any(availability[d] < quota + (d <= extra) for d in by_det):
        raise ValueError(f"Insufficient products for balanced {count}-observation plan; per-detector available={availability}")
    for det in range(1, 7):
        rows = sorted(by_det[det].values(), key=lambda x: x["mjd_midpoint"])
        n = quota + (det <= extra)
        indexes = np.linspace(0, len(rows) - 1, n, dtype=int)
        selected.extend(rows[i] for i in indexes)
    selected.sort(key=lambda x: (x["mjd_midpoint"], x["detector"]))
    detector_nights = {d: {r["utc_night"] for r in selected if r["detector"] == d} for d in by_det}
    deficient = {d: sorted(nights) for d, nights in detector_nights.items() if len(nights) < 2}
    if deficient:
        raise ValueError(f"Plan cannot provide same-detector repeats on distinct UTC dates: {deficient}")
    selected.sort(key=lambda r: r["mjd_midpoint"])
    return selected


def overflow_sample(plan: list[dict], count: int) -> list[dict]:
    if count < 4 or count > 6:
        raise ValueError("overflow-count must be 4, 5, or 6")
    if len(plan) < count:
        raise ValueError(f"Need {count} planned rows; only {len(plan)} available")
    # Pick at evenly spaced epochs, preferring a new detector at each epoch.
    ordered = sorted(plan, key=lambda r: r["mjd_midpoint"])
    desired_times = np.linspace(ordered[0]["mjd_midpoint"], ordered[-1]["mjd_midpoint"], count)
    chosen, used_detectors = [], set()
    for desired in desired_times:
        eligible = [r for r in ordered if r not in chosen and r["detector"] not in used_detectors]
        if not eligible:
            eligible = [r for r in ordered if r not in chosen]
        selected = min(eligible, key=lambda r: abs(r["mjd_midpoint"] - desired))
        chosen.append(selected)
        used_detectors.add(selected["detector"])
    chosen.sort(key=lambda r: r["mjd_midpoint"])
    return chosen


def _catalog_query(source_id: str, ra: float, dec: float, catalog: str, out_dir: Path) -> bool:
    try:
        table = Irsa.query_region(SkyCoord(ra * u.deg, dec * u.deg), catalog=catalog, radius=5 * u.arcsec)
        out = out_dir / f"{source_id}_{catalog}.ecsv"
        table.write(out, format="ascii.ecsv", overwrite=True)
        print(f"CATALOG source_id={source_id} catalog={catalog} rows={len(table)} raw={out}")
        return True
    except Exception as exc:
        print(f"CATALOG_QUERY_FAILED source_id={source_id} catalog={catalog} {type(exc).__name__}: {exc}")
        return False


def inspect_cutout_hdus(path: Path) -> list[dict]:
    """Print and validate the cutout MEF structure before subsequent downloads."""
    listing = []
    with fits.open(path, memmap=True) as hdul:
        for index, hdu in enumerate(hdul):
            shape = list(hdu.data.shape) if hdu.data is not None else None
            cards = {key: hdu.header.get(key) for key in (
                "BUNIT", "NAXIS", "CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2", "CD1_1", "CD2_2"
            ) if key in hdu.header}
            entry = {"index": index, "name": hdu.name, "shape": shape, "key_header_cards": cards}
            listing.append(entry)
            print("CUTOUT_HDU " + json.dumps(entry, default=str))
            if hdu.name == "EPSF" and hdu.data is not None and getattr(hdu.data, "names", None):
                print(f"CUTOUT_EPSF_COLUMNS {list(hdu.data.names)}")
            if hdu.name == "WCS-WAVE" and hdu.data is not None and getattr(hdu.data, "names", None):
                print(f"CUTOUT_WCS_WAVE_COLUMNS {list(hdu.data.names)}")
        names = {hdu.name for hdu in hdul}
        missing = sorted(REQUIRED_CUTOUT_EXTENSIONS - names)
        if missing:
            raise ValueError(f"Cutout MEF is missing expected extensions {missing}; observed {sorted(names)}")
        if hdul["IMAGE"].data is None or hdul["IMAGE"].data.ndim != 2:
            raise ValueError("Cutout IMAGE is absent or not a 2-D image")
        if not fits.getheader(path, "IMAGE").get("CTYPE1"):
            raise ValueError("Cutout IMAGE lacks celestial WCS CTYPE1")
        epsf_cols = set(getattr(hdul["EPSF"].data, "names", []) or [])
        wave_cols = set(getattr(hdul["WCS-WAVE"].data, "names", []) or [])
        if not {"XCENTER", "YCENTER", "CWAVE", "CBAND"}.issubset(epsf_cols):
            raise ValueError(f"Cutout EPSF has unexpected columns {sorted(epsf_cols)}")
        if not {"X", "Y", "VALUES"}.issubset(wave_cols):
            raise ValueError(f"Cutout WCS-WAVE has unexpected columns {sorted(wave_cols)}")
    print("CUTOUT_HDU_INSPECTION_PASS extensions=" + ",".join(x["name"] for x in listing))
    return listing


class CutoutDownloadError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None, body: str = ""):
        self.status = status
        self.body = body
        super().__init__(f"{message}; HTTP_status={status}; response_body={body[:2000]!r}")


def _retry_after(response: requests.Response | None) -> float | None:
    if response is None:
        return None
    value = response.headers.get("Retry-After")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            return max(0.0, (parsedate_to_datetime(value).astimezone(timezone.utc) - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def _download(session: requests.Session, url: str, path: Path, *, max_bytes: int = MAX_CUTOUT_BYTES,
              retries: bool = True, announce_length: bool = False) -> tuple[int, str, int, int | None]:
    delays = RETRY_DELAYS_SECONDS if retries else ()
    attempt = 0
    while True:
        attempt += 1
        try:
            response = session.get(url, stream=True, allow_redirects=True, timeout=(15, 120))
        except requests.RequestException as exc:
            response = getattr(exc, "response", None)
            status = response.status_code if response is not None else None
            body = response.text[:2000] if response is not None else str(exc)
            if attempt <= len(delays):
                retry_wait = _retry_after(response)
                time.sleep(retry_wait if retry_wait is not None else delays[attempt - 1])
                continue
            raise CutoutDownloadError("GET request failed", status=status, body=body) from exc
        if not response.ok:
            status, body = response.status_code, response.text[:2000]
            should_retry = status in {429, 503} or 500 <= status <= 599
            wait = _retry_after(response)
            response.close()
            if should_retry and attempt <= len(delays):
                time.sleep(wait if wait is not None else delays[attempt - 1])
                continue
            raise CutoutDownloadError("GET returned non-success status", status=status, body=body)
        break
    content_length = response.headers.get("Content-Length")
    if announce_length:
        print(f"FULL_PRODUCT_RESPONSE HTTP={response.status_code} Content-Length={content_length or 'unknown'} url={url}")
    if content_length and content_length.isdigit() and int(content_length) > max_bytes:
        response.close()
        raise CutoutDownloadError(f"Content-Length {content_length} exceeds {max_bytes} byte cap",
                                  status=response.status_code,
                                  body=f"not read: Content-Length {content_length} exceeds configured cap")
    digest, size = hashlib.sha256(), 0
    try:
        with path.open("wb") as stream:
            for block in response.iter_content(1024 * 1024):
                if block:
                    size += len(block)
                    if size > max_bytes:
                        raise CutoutDownloadError(f"streamed response exceeded {max_bytes} byte cap",
                                                  status=response.status_code,
                                                  body="aborted while streaming; no response error body")
                    stream.write(block)
                    digest.update(block)
    except CutoutDownloadError:
        path.unlink(missing_ok=True)
        response.close()
        raise
    except requests.RequestException as exc:
        path.unlink(missing_ok=True)
        status = response.status_code
        body = response.text[:2000] if response.content else str(exc)
        response.close()
        raise CutoutDownloadError("GET stream failed", status=status, body=body) from exc
    except Exception:
        path.unlink(missing_ok=True)
        response.close()
        raise
    response.close()
    return size, digest.hexdigest(), response.status_code, int(content_length) if content_length and content_length.isdigit() else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("pipeline/cache/control-star"))
    parser.add_argument("--plan-count", type=int, default=40)
    parser.add_argument("--overflow-count", type=int, choices=(4,), default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("pipeline/cache/star-candidates"))
    args = parser.parse_args(argv)
    coords = _coord_lookup(args.cache_dir / "gaia_north.ecsv")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "overflow_results": [], "download_failures": [], "processing_failures": [],
        "var_flg_use": "not used as evidence; AllWISE var_flg is a four-character W1/W2/W3/W4 likelihood indicator derived from historical single-exposure flux distributions, not the SPHEREx-epoch light curve",
        "assumed": ["catalog association is nearest positional match within the queried 5 arcsec radius", "SPHEREx cutout size 0.1 deg"],
    }
    with requests.Session() as session:
        session.headers.update({"User-Agent": "TIMEFRAME-102-SPHEREx-control-star/1.0"})
        first_star_complete = False
        for target_index, (source_id, meta) in enumerate(TARGETS.items()):
            if target_index > 0 and not first_star_complete:
                print(f"STAR_SEQUENCE_STOP first_candidate_not_complete source_id={next(iter(TARGETS))}")
                break
            if source_id not in coords:
                print(f"TARGET_CACHE_MISSING source_id={source_id}; no coordinates available, skipping")
                report["processing_failures"].append({"source_id": source_id, "stage": "coordinates"})
                if target_index == 0:
                    break
                continue
            ra, dec = coords[source_id]
            print(f"\nTARGET source_id={source_id} G={meta['g_mag']} RA={ra:.9f} Dec={dec:.9f}")
            for catalog in ("fp_psc", "allwise_p3as_psd"):
                if not _catalog_query(source_id, ra, dec, catalog, args.output_dir):
                    report["processing_failures"].append({"source_id": source_id, "stage": "catalog", "catalog": catalog})
            tables = []
            for release in RELEASES:
                path = args.cache_dir / f"sia_{source_id}_{release}.ecsv"
                if path.exists():
                    tables.append(Table.read(path, format="ascii.ecsv"))
                else:
                    print(f"SIA_CACHE_MISSING source_id={source_id} release={release}; no live substitute issued")
            try:
                plan = plan_observations(tables, args.plan_count)
            except Exception as exc:
                print(f"PLAN_FAILED source_id={source_id} {type(exc).__name__}: {exc}")
                report["processing_failures"].append({"source_id": source_id, "stage": "plan", "error": str(exc)})
                if target_index == 0:
                    break
                continue
            repeat_counts = {f"D{d}": len({r["utc_night"] for r in plan if r["detector"] == d}) for d in range(1, 7)}
            print(f"PLAN source_id={source_id} count={len(plan)} detectors={sorted({r['detector'] for r in plan})} "
                  f"date_range={min(r['utc_night'] for r in plan)}..{max(r['utc_night'] for r in plan)} "
                  f"repeat_dates_per_detector={repeat_counts} grouping=none")
            cutouts = overflow_sample(plan, args.overflow_count)
            successful = 0
            failed_rows = []
            consecutive_failures = 0
            inspected_mef = False
            for item_index, row in enumerate(cutouts):
                url = cutout_url(row["url"], ra, dec, size_deg=0.1)
                name = f"{source_id}_{row['utc_night']}_D{row['detector']}_{row['obs_id']}.fits"
                path = args.output_dir / name
                try:
                    nbytes, sha, status, content_length = _download(session, url, path)
                    if not inspected_mef:
                        hdu_listing = inspect_cutout_hdus(path)
                        report.setdefault("hdu_inspections", {})[source_id] = hdu_listing
                        inspected_mef = True
                    processed = process_fits_product(
                        path, output_dir=args.output_dir / "northup", ra_deg=ra, dec_deg=dec,
                        observation_id=row["obs_id"], target_name=f"Gaia-{source_id}", detector=row["detector"],
                        time_utc=Time(row["mjd_midpoint"], format="mjd", scale="utc").isot,
                        mjd_midpoint=row["mjd_midpoint"],
                    )
                    successful += 1
                    consecutive_failures = 0
                    record = {
                        "source_id": source_id, "obs_id": row["obs_id"], "detector": row["detector"],
                        "utc_night": row["utc_night"], "bytes": nbytes, "sha256": sha,
                        "flags_at_star_pixel": processed["central_flag_decoded"],
                        "flags_within_2_pixel_radius_box": processed["flags_within_2_pixel_radius_box"],
                        "overflow_at_star_pixel": processed["overflow_at_predicted_pixel"],
                        "overflow_within_2px": processed["overflow_within_2_pixel_radius_box"],
                        "input_pixel_position_xy_0based": processed["input_pixel_position_xy_0based"],
                        "predicted_xy": processed["predicted_xy"], "measured_source_xy": processed["measured_source_xy"],
                        "association": processed["association"], "peak_snr": processed["peak_snr"],
                        "effective_wavelength_um": processed["effective_wavelength_um"],
                        "wavelength_uncertainty_um": processed["wavelength_uncertainty_um"],
                        "wavelength_source": processed["wavelength_source"],
                        "preview_path": processed["preview_path"],
                    }
                    report["overflow_results"].append(record)
                    print(f"CUTOUT source_id={source_id} obs_id={row['obs_id']} D{row['detector']} "
                          f"date={row['utc_night']} HTTP={status} Content-Length={content_length or 'unknown'} bytes={nbytes} "
                          f"flags={record['flags_at_star_pixel']} "
                          f"lambda_um={record['effective_wavelength_um']} "
                          f"lambda_unc_um={record['wavelength_uncertainty_um']}")
                except Exception as exc:
                    stage = "download" if isinstance(exc, CutoutDownloadError) else "inspect_or_process"
                    if isinstance(exc, CutoutDownloadError):
                        detail = {"error": str(exc), "http_status": exc.status, "response_body": exc.body}
                        print(f"CUTOUT_DOWNLOAD_FAILED source_id={source_id} obs_id={row['obs_id']} "
                              f"status={exc.status} body={exc.body[:2000]!r} error={exc}")
                        report["download_failures"].append({"source_id": source_id, "obs_id": row["obs_id"], **detail})
                        failed_rows.append(row)
                        consecutive_failures += 1
                    else:
                        print(f"CUTOUT_INSPECT_OR_PROCESS_FAILED source_id={source_id} obs_id={row['obs_id']} "
                              f"{type(exc).__name__}: {exc}")
                        report["processing_failures"].append({"source_id": source_id, "obs_id": row["obs_id"],
                                                               "stage": stage, "error": str(exc)})
                    if item_index == 0:
                        pass
                    if consecutive_failures >= 5:
                        print(f"STAR_SEQUENCE_STOP source_id={source_id} consecutive_failures={consecutive_failures}")
                        break
            if len(failed_rows) >= 3:
                print(f"FULL_PRODUCT_FALLBACK source_id={source_id} observations={min(4, len(cutouts))}")
                fallback_rows = cutouts[:4]
                fallback_success = 0
                for fallback_index, row in enumerate(fallback_rows):
                    name = f"{source_id}_{row['utc_night']}_D{row['detector']}_{row['obs_id']}_full.fits"
                    path = args.output_dir / name
                    try:
                        nbytes, sha, status, content_length = _download(
                            session, row["url"], path, max_bytes=MAX_FULL_PRODUCT_BYTES, announce_length=True)
                        if not inspected_mef:
                            report.setdefault("hdu_inspections", {})[source_id] = inspect_cutout_hdus(path)
                            inspected_mef = True
                        processed = process_fits_product(
                            path, output_dir=args.output_dir / "northup", ra_deg=ra, dec_deg=dec,
                            observation_id=row["obs_id"], target_name=f"Gaia-{source_id}", detector=row["detector"],
                            time_utc=Time(row["mjd_midpoint"], format="mjd", scale="utc").isot,
                            mjd_midpoint=row["mjd_midpoint"],
                        )
                        fallback_success += 1
                        report["overflow_results"].append({
                            "source_id": source_id, "obs_id": row["obs_id"], "detector": row["detector"],
                            "utc_night": row["utc_night"], "download_kind": "full_product_local_cutout",
                            "bytes": nbytes, "sha256": sha, "http_status": status,
                            "content_length": content_length,
                            "flags_at_star_pixel": processed["central_flag_decoded"],
                            "flags_within_2_pixel_radius_box": processed["flags_within_2_pixel_radius_box"],
                            "overflow_at_star_pixel": processed["overflow_at_predicted_pixel"],
                            "effective_wavelength_um": processed["effective_wavelength_um"],
                            "wavelength_uncertainty_um": processed["wavelength_uncertainty_um"],
                            "preview_path": processed["preview_path"],
                        })
                        print(f"CUTOUT source_id={source_id} obs_id={row['obs_id']} D{row['detector']} "
                          f"date={row['utc_night']} HTTP={status} mode=full_product_local_crop "
                          f"Content-Length={content_length or 'unknown'} bytes={nbytes} "
                              f"flags={processed['central_flag_decoded']} "
                              f"lambda_um={processed['effective_wavelength_um']} "
                              f"lambda_unc_um={processed['wavelength_uncertainty_um']}")
                    except Exception as exc:
                        status = exc.status if isinstance(exc, CutoutDownloadError) else None
                        body = exc.body if isinstance(exc, CutoutDownloadError) else str(exc)
                        print(f"FULL_PRODUCT_FAILED source_id={source_id} obs_id={row['obs_id']} HTTP={status} body={body[:2000]!r}")
                        report["download_failures"].append({"source_id": source_id, "obs_id": row["obs_id"],
                            "mode": "full_product_fallback", "http_status": status, "response_body": body})
            if target_index == 0:
                first_star_complete = successful == len(cutouts)
                if not first_star_complete:
                    print(f"STAR_SEQUENCE_STOP faintest_candidate_completed={successful}/{len(cutouts)}")
    output = args.output_dir / "star_candidate_screen.json"
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"COMPLETE report={output} examined_overflow_rows={len(report['overflow_results'])} "
          f"download_failures={len(report['download_failures'])} "
          f"processing_failures={len(report['processing_failures'])}")
    return 0 if not report["download_failures"] and not report["processing_failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
