"""List public SPHEREx Spectral WCS calibration objects in the IRSA S3 bucket.

Run from the repository root on a machine with network access:
    python -m pipeline.list_cloud_spectral_wcs --release both

This uses the unauthenticated S3 ListObjectsV2 API. It only inventories object
keys and sizes; it does not select or download a calibration file.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from typing import Any

import requests


BUCKET_URL = "https://nasa-irsa-spherex.s3.us-east-1.amazonaws.com/"
RELEASES = ("dr1", "qr3")
S3_NAMESPACE = "http://s3.amazonaws.com/doc/2006-03-01/"


def parse_list_response(xml_text: str) -> dict[str, Any]:
    """Parse one ListObjectsV2 XML page into object metadata and paging state."""
    root = ET.fromstring(xml_text)

    def value(name: str) -> str | None:
        node = root.find(f"{{{S3_NAMESPACE}}}{name}")
        return node.text if node is not None else None

    objects = []
    for node in root.findall(f"{{{S3_NAMESPACE}}}Contents"):
        def child(name: str) -> str | None:
            item = node.find(f"{{{S3_NAMESPACE}}}{name}")
            return item.text if item is not None else None

        key = child("Key")
        if not key:
            continue
        objects.append({
            "key": key,
            "size_bytes": int(child("Size") or 0),
            "last_modified": child("LastModified"),
            "etag": (child("ETag") or "").strip('"') or None,
        })

    return {
        "objects": objects,
        "is_truncated": (value("IsTruncated") or "false").lower() == "true",
        "next_token": value("NextContinuationToken"),
    }


def list_spectral_wcs(release: str, session: Any = requests) -> list[dict[str, Any]]:
    """Return every spectral-WCS FITS object for one release, across pages."""
    if release not in RELEASES:
        raise ValueError(f"release must be one of {RELEASES}")

    prefix = f"{release}/spectral_wcs/"
    token = None
    found: list[dict[str, Any]] = []
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if token:
            params["continuation-token"] = token
        response = session.get(BUCKET_URL, params=params, timeout=(10, 45))
        if response.status_code != 200:
            body = response.text[:2000]
            raise RuntimeError(f"S3 listing failed: HTTP {response.status_code}: {body}")
        page = parse_list_response(response.text)
        found.extend(obj for obj in page["objects"] if obj["key"].lower().endswith(".fits"))
        if not page["is_truncated"]:
            return found
        token = page["next_token"]
        if not token:
            raise RuntimeError("S3 listing marked truncated but returned no continuation token")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", choices=(*RELEASES, "both"), default="both")
    args = parser.parse_args()
    releases = RELEASES if args.release == "both" else (args.release,)
    failures = 0
    for release in releases:
        try:
            products = list_spectral_wcs(release)
            print(json.dumps({"release": release, "status": "ok", "count": len(products),
                              "products": products}, separators=(",", ":")))
        except (requests.RequestException, RuntimeError, ET.ParseError, ValueError) as exc:
            failures += 1
            print(json.dumps({"release": release, "status": "failed", "error": str(exc)},
                             separators=(",", ":")))
    return 1 if failures == len(releases) else 0


if __name__ == "__main__":
    raise SystemExit(main())
