"""Download and inspect a small sample of SPHEREx FITS products from MOST ECSV.

This utility is intentionally an inspection step: it preserves and reports all
HDU extensions, but does not assert that a name-matched extension is the
science, variance, flag, PSF, or wavelength product. Review the printed headers
before assigning scientific meaning.

Example (from the project root)::

    python -m pipeline.fetch_and_inspect \
      --table pipeline/cache/probe/most_20000002_Pallas_large-brightness-risk.ecsv \
      --table pipeline/cache/probe/most_20000010_Hygiea_large-brightness-risk.ecsv \
      --count 3 --output-dir pipeline/cache/fits-inspection
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import numpy as np
import requests
from astropy.io import fits
from astropy.table import Table

IRSA_ORIGIN = "https://irsa.ipac.caltech.edu/"
KEY_CARDS = (
    "EXTNAME", "EXTVER", "EXTTYPE", "BUNIT", "BTYPE", "DATE-OBS", "MJD-OBS", "MJD-BEG",
    "MJD-END", "OBSID", "PLANEID", "PRODID", "OBJECT", "FILTER", "BAND",
    "WAVELEN", "WAVELENGTH", "EFFWAVE", "LAM_CEN", "LAM_MIN", "LAM_MAX",
    "CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
    "CDELT1", "CDELT2", "CD1_1", "CD1_2", "CD2_1", "CD2_2",
    "PC1_1", "PC1_2", "PC2_1", "PC2_2", "RADESYS", "EQUINOX",
    "WCSNAMEA", "CTYPE1A", "CTYPE2A", "CUNIT1A", "CUNIT2A", "CRPIX1A", "CRPIX2A",
    "WCSNAMEW", "CTYPE1W", "CTYPE2W", "CUNIT1W", "CUNIT2W", "PS1_0W", "PS2_0W",
    "PS1_1W", "PS2_1W", "PS1_2W", "PS2_2W", "PV1_3W", "PV2_3W",
    "QUALITY", "MASK", "CHECKSUM", "DATASUM",
)
ROLE_TOKENS = {
    "image": ("SCI", "IMAGE", "FLUX", "INTENSITY"),
    "flags": ("FLAG", "MASK", "DQ", "QUALITY"),
    "variance": ("VAR", "VARIANCE", "ERR", "ERROR", "UNC"),
    "psf": ("PSF", "EPSF", "POINTSPREAD"),
    "wavelength_map": ("WAVE", "WAVELENGTH", "LAMBDA", "EFFWAVE", "LAM_CEN"),
}


def resolve_product_url(uri: str, origin: str = IRSA_ORIGIN) -> str:
    """Resolve an IRSA relative URI while rejecting unexpected hosts."""
    value = uri.strip()
    if not value:
        raise ValueError("Empty product URI")
    url = value if urlparse(value).scheme else urljoin(origin, value.lstrip("/"))
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != urlparse(origin).hostname:
        raise ValueError(f"Product URI is outside the expected IRSA HTTPS host: {uri!r}")
    return url


def choose_spaced_rows(table: Table, count: int) -> list[dict[str, Any]]:
    """Select distinct product URIs spaced across the table's observation times."""
    if count < 1:
        raise ValueError("count must be at least one")
    if "uri" not in table.colnames:
        raise ValueError("MOST table has no 'uri' column")
    time_col = "mjd_start" if "mjd_start" in table.colnames else None
    rows: dict[str, dict[str, Any]] = {}
    for row in table:
        uri = str(row["uri"]).strip()
        if not uri or uri.lower() in {"masked", "none", "nan"}:
            continue
        item = {str(key): row[key].item() if hasattr(row[key], "item") else row[key] for key in table.colnames}
        rows.setdefault(uri, item)
    candidates = list(rows.values())
    candidates.sort(key=lambda r: float(r[time_col]) if time_col and np.isfinite(float(r[time_col])) else 0.0)
    if count > len(candidates):
        raise ValueError(f"Requested {count} FITS files but table contains only {len(candidates)} distinct URIs")
    indexes = np.linspace(0, len(candidates) - 1, num=count, dtype=int)
    return [candidates[i] for i in indexes]


def _safe_filename(uri: str) -> str:
    base = Path(urlparse(uri).path).name or "spherex_product.fits"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", base)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hdu_metadata(hdu: fits.hdu.base.ExtensionHDU | fits.PrimaryHDU) -> dict[str, Any]:
    header = hdu.header
    name = str(header.get("EXTNAME", hdu.name or "PRIMARY"))
    tokens = f"{name} {header.get('BTYPE', '')} {header.get('OBJECT', '')}".upper()
    possible_roles = [role for role, words in ROLE_TOKENS.items() if any(word in tokens for word in words)]
    if hdu.data is not None and np.ndim(hdu.data) >= 2 and "image" not in possible_roles:
        possible_roles.append("image")
    columns = None
    if isinstance(hdu, fits.BinTableHDU):
        columns = [
            {"name": column.name, "format": column.format, "dim": column.dim}
            for column in hdu.columns
        ]
    summaries: dict[str, Any] = {}
    if name == "WCS-WAVE" and hdu.data is not None and "VALUES" in hdu.columns.names:
        values = np.asarray(hdu.data["VALUES"][0], dtype=float)
        summaries["central_wavelength_um_min_max_visualization_lookup"] = [
            float(np.nanmin(values[..., 0])), float(np.nanmax(values[..., 0]))
        ]
        summaries["bandwidth_um_min_max_visualization_lookup"] = [
            float(np.nanmin(values[..., 1])), float(np.nanmax(values[..., 1]))
        ]
    if name == "EPSF" and hdu.data is not None and getattr(hdu.data, "dtype", None) is not None:
        field_names = hdu.data.dtype.names or ()
        for key in ("CWAVE", "CBAND"):
            if key in field_names:
                values = np.asarray(hdu.data[key], dtype=float)
                summaries[f"{key.lower()}_um_min_max"] = [float(np.nanmin(values)), float(np.nanmax(values))]
        if "EPSF" in field_names:
            summaries["epsf_array_shape_per_row"] = list(hdu.data["EPSF"].shape[1:])
    return {
        "name": name,
        "shape": list(hdu.data.shape) if hdu.data is not None else None,
        "cards": {key: header[key] for key in KEY_CARDS if key in header},
        "table_columns": columns,
        "data_summaries": summaries,
        "possible_roles_by_name_or_header": possible_roles,
    }


def inspect_fits(path: Path) -> list[dict[str, Any]]:
    """Print all HDUs, key header cards, and name/header role candidates."""
    metadata: list[dict[str, Any]] = []
    with fits.open(path, memmap=False) as hdul:
        print(f"\nFITS_FILE {path}")
        print(f"HDU_COUNT {len(hdul)}")
        for index, hdu in enumerate(hdul):
            item = _hdu_metadata(hdu)
            metadata.append(item)
            print(f"HDU index={index} name={item['name']!r} shape={item['shape']}")
            print(f"  KEY_CARDS {item['cards']!r}")
            if item["table_columns"] is not None:
                print(f"  TABLE_COLUMNS {item['table_columns']!r}")
            if item["data_summaries"]:
                print(f"  DATA_SUMMARIES {item['data_summaries']!r}")
            print(f"  POSSIBLE_ROLES_BY_NAME_OR_HEADER {item['possible_roles_by_name_or_header']}")
    for role in ROLE_TOKENS:
        matches = [m["name"] for m in metadata if role in m["possible_roles_by_name_or_header"]]
        print(f"ROLE_CANDIDATES {role} {matches}")
    return metadata


def fetch_one(
    session: requests.Session,
    uri: str,
    output_dir: Path,
    *,
    head_timeout: float,
    get_timeout: float,
) -> dict[str, Any]:
    url = resolve_product_url(uri)
    print(f"\nPRODUCT_URI {uri}")
    print(f"PRODUCT_URL {url}")
    response = session.head(url, allow_redirects=True, timeout=head_timeout)
    size_header = response.headers.get("Content-Length")
    expected_size = int(size_header) if size_header and size_header.isdigit() else None
    print(f"HEAD_STATUS {response.status_code} content_length={expected_size} content_type={response.headers.get('Content-Type')}")
    response.raise_for_status()

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / _safe_filename(url)
    if destination.exists() and (expected_size is None or destination.stat().st_size == expected_size):
        print(f"DOWNLOAD_CACHE_HIT {destination} bytes={destination.stat().st_size}")
    else:
        get_response = session.get(url, stream=True, allow_redirects=True, timeout=get_timeout)
        print(f"GET_STATUS {get_response.status_code}")
        get_response.raise_for_status()
        fd, temp_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".partial", dir=output_dir)
        os.close(fd)
        temp_path = Path(temp_name)
        try:
            total = 0
            with temp_path.open("wb") as stream:
                for block in get_response.iter_content(chunk_size=1024 * 1024):
                    if block:
                        stream.write(block)
                        total += len(block)
            if expected_size is not None and total != expected_size:
                raise IOError(f"Downloaded {total} bytes but HEAD reported {expected_size}")
            os.replace(temp_path, destination)
        finally:
            temp_path.unlink(missing_ok=True)
        print(f"DOWNLOADED {destination} bytes={total}")

    hdus = inspect_fits(destination)
    return {
        "source_uri": uri,
        "url": url,
        "path": str(destination),
        "head_content_length_bytes": expected_size,
        "downloaded_size_bytes": destination.stat().st_size,
        "etag": response.headers.get("ETag"),
        "last_modified": response.headers.get("Last-Modified"),
        "sha256": _sha256(destination),
        "hdus": hdus,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, action="append", required=True, help="MOST ECSV table; repeat for multiple asteroids.")
    parser.add_argument("--count", type=int, default=3, help="Total distinct FITS products to inspect (default: 3).")
    parser.add_argument("--output-dir", type=Path, default=Path("pipeline/cache/fits-inspection"))
    parser.add_argument("--head-timeout", type=float, default=30.0)
    parser.add_argument("--get-timeout", type=float, default=180.0)
    args = parser.parse_args(argv)
    if args.count < len(args.table):
        parser.error("--count must allow at least one file from each --table")

    tables: list[tuple[Path, Table]] = []
    for path in args.table:
        tables.append((path, Table.read(path, format="ascii.ecsv")))
    counts = [args.count // len(tables)] * len(tables)
    for i in range(args.count % len(tables)):
        counts[i] += 1

    selections: list[tuple[Path, dict[str, Any]]] = []
    for (path, table), count in zip(tables, counts):
        for row in choose_spaced_rows(table, count):
            selections.append((path, row))
    print(f"SELECTION_COUNT {len(selections)}")

    manifest: list[dict[str, Any]] = []
    failures = 0
    with requests.Session() as session:
        session.headers.update({"User-Agent": "TIMEFRAME-102-SPHEREx-probe/1.0"})
        for index, (source_table, row) in enumerate(selections, start=1):
            print(f"\n=== PRODUCT {index}/{len(selections)} source_table={source_table} ===")
            print(f"MOST_ROW_FIELDS { {k: row.get(k) for k in ('observationid','mjd_start','mjd_end','ra_obj','dec_obj','vmag','planeid','productid')}!r}")
            try:
                info = fetch_one(
                    session,
                    str(row["uri"]),
                    args.output_dir,
                    head_timeout=args.head_timeout,
                    get_timeout=args.get_timeout,
                )
                info["source_table"] = str(source_table)
                info["most_row"] = row
                manifest.append(info)
            except Exception as exc:
                failures += 1
                print(f"PRODUCT_FAILED {type(exc).__name__}: {exc}")
    manifest_path = args.output_dir / "inspection_manifest.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"MANIFEST {manifest_path}")
    print(f"INSPECTION_COMPLETE products={len(selections)} failures={failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
