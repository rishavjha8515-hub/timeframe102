"""Check catalog-qualified comparison stars against cached usable cutouts.

This is a pre-photometry gate only. It applies Gaia proper motion, the declared
60 arcsec cutout-edge margin, and hard flags in a 5x5 FLAGS window. It does not
claim detection or measure comparison-star fluxes.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from .cutout_processing import decode_flag_definitions, decode_flag_value
from .quality_policy import POLICY


ROOT = Path(__file__).resolve().parents[1]
CASE_PATH = ROOT / "outputs/control-star/case.json"
GAIA_PATH = ROOT / "pipeline/cache/control-star/gaia_north.ecsv"
CATALOG_REPORT_PATH = ROOT / "pipeline/cache/control-star/comparison_catalogs/vizier/comparison_catalog_vetting_vizier.json"
SERIES_DIR = ROOT / "pipeline/cache/control-star-series"
OUTPUT_PATH = ROOT / "pipeline/cache/control-star/comparison_catalogs/vizier/per_frame_vetting.json"
HARD_FLAGS = set(POLICY["comparison_star_criteria"]["exclude_hard_source_flags"])


def propagate_gaia_position(row: Any, time_utc: str) -> tuple[float, float]:
    """Propagate Gaia DR3 ICRS position linearly from its J2016.0 reference epoch."""
    ra, dec = float(row["ra"]), float(row["dec"])
    dt_year = Time(time_utc).jyear - 2016.0
    pmra, pmdec = float(row["pmra"]), float(row["pmdec"])
    dec_at_epoch = dec + pmdec * dt_year / 3_600_000.0
    ra_at_epoch = (ra + pmra * dt_year / (3_600_000.0 * np.cos(np.deg2rad(dec)))) % 360.0
    return float(ra_at_epoch), float(dec_at_epoch)


def classify_candidate_pixel(
    x: float,
    y: float,
    flags: np.ndarray,
    definitions: dict[int, tuple[str, str]],
    *,
    pixel_scale_arcsec: float,
    edge_margin_arcsec: float = 60.0,
) -> dict[str, Any]:
    """Return footprint/edge/5x5-hard-flag eligibility for one predicted position."""
    ny, nx = flags.shape
    if not (np.isfinite(x) and np.isfinite(y) and 0 <= x < nx and 0 <= y < ny):
        return {"inside_cutout": False, "edge_safe": False, "hard_flags": [], "eligible": False,
                "reason": "predicted position outside cutout"}
    margin = min(x, y, nx - 1 - x, ny - 1 - y) * pixel_scale_arcsec
    if margin < edge_margin_arcsec:
        return {"inside_cutout": True, "edge_safe": False, "edge_margin_arcsec": float(margin),
                "hard_flags": [], "eligible": False, "reason": "less than 60 arcsec from cutout edge"}
    ix, iy = int(np.floor(x + 0.5)), int(np.floor(y + 0.5))
    window = flags[max(0, iy - 2):min(ny, iy + 3), max(0, ix - 2):min(nx, ix + 3)]
    names: set[str] = set()
    for raw in np.unique(window):
        names.update(flag["name"] for flag in decode_flag_value(int(raw), definitions))
    hard = sorted(names & HARD_FLAGS)
    return {"inside_cutout": True, "edge_safe": True, "edge_margin_arcsec": float(margin),
            "hard_flags": hard, "eligible": not hard,
            "reason": "hard flag in 5x5 window: " + ", ".join(hard) if hard else None}


def vet_case_frames() -> dict[str, Any]:
    case = json.loads(CASE_PATH.read_text(encoding="utf-8"))
    catalog = json.loads(CATALOG_REPORT_PATH.read_text(encoding="utf-8"))
    if catalog.get("status") != "complete":
        raise RuntimeError(f"catalog query is not complete: {catalog.get('error', catalog.get('status'))}")
    matched_ids = sorted(c["source_id"] for c in catalog["candidates"] if c["passed_catalog_gates"])
    gaia = Table.read(GAIA_PATH, format="ascii.ecsv")
    by_id = {str(row["source_id"]): row for row in gaia}
    missing = [sid for sid in matched_ids if sid not in by_id]
    if missing:
        raise ValueError(f"Catalog-qualified sources absent from cached Gaia table: {missing}")

    criteria = POLICY["comparison_star_criteria"]
    frame_results = []
    for obs in case["observations"]:
        if not obs["usable"]:
            continue
        path = SERIES_DIR / f"{obs['observation_id']}_D{obs['detector']}.fits"
        result: dict[str, Any] = {
            "observation_id": obs["observation_id"], "detector": obs["detector"],
            "n_catalog_matched": len(matched_ids), "n_inside_cutout": 0,
            "n_edge_safe": 0, "n_hard_flagged": 0, "n_stars": None,
            "reason": None, "eligible_source_ids": [], "hard_flagged_source_ids": [],
            "status": "not processed",
        }
        if not path.is_file():
            result.update(status="unavailable", reason="cached FITS product is missing")
            frame_results.append(result)
            continue
        try:
            with fits.open(path, memmap=True) as hdus:
                image, flag_hdu = hdus["IMAGE"], hdus["FLAGS"]
                if image.data is None or flag_hdu.data is None:
                    raise ValueError("IMAGE or FLAGS extension has no data")
                definitions = decode_flag_definitions(flag_hdu.header)
                if not definitions:
                    raise ValueError("FLAGS header contains no MSKN/MSKD definitions")
                wcs = WCS(image.header).celestial
                scale = float(np.mean(proj_plane_pixel_scales(wcs)) * 3600.0)
                for source_id in matched_ids:
                    row = by_id[source_id]
                    ra, dec = propagate_gaia_position(row, obs["time_utc"])
                    x, y = wcs.world_to_pixel_values(ra, dec)
                    decision = classify_candidate_pixel(
                        float(x), float(y), flag_hdu.data, definitions,
                        pixel_scale_arcsec=scale,
                        edge_margin_arcsec=criteria["minimum_cutout_edge_margin_arcsec"],
                    )
                    if decision["inside_cutout"]:
                        result["n_inside_cutout"] += 1
                    if decision["edge_safe"]:
                        result["n_edge_safe"] += 1
                    if decision["hard_flags"]:
                        result["n_hard_flagged"] += 1
                        result["hard_flagged_source_ids"].append({
                            "source_id": source_id, "flag_names": decision["hard_flags"]
                        })
                    if decision["eligible"]:
                        result["eligible_source_ids"].append(source_id)
                result["n_stars"] = len(result["eligible_source_ids"])
                result["status"] = "pre-photometry eligible"
                result["reason"] = (
                    "Count is after catalog, footprint, 60 arcsec edge, and 5x5 hard-flag gates, "
                    "but before detection and photometry. "
                    f"{result['n_stars']} candidates remain; required minimum is "
                    f"{criteria['minimum_stars_per_frame']}."
                )
        except Exception as exc:
            result.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
        frame_results.append(result)

    measured_counts = [r["n_stars"] for r in frame_results if isinstance(r["n_stars"], int)]
    required = criteria["minimum_stars_per_frame"]
    count_ge_min = sum(n >= required for n in measured_counts)
    result = {
        "status": "ready for photometry" if measured_counts and count_ge_min == len(frame_results)
                  else "insufficient stars",
        "catalog_query_utc": catalog["query_utc"],
        "catalog_servers": {name: table["server"] for name, table in catalog["raw_tables"].items()},
        "catalog_row_counts": {name: table["row_count"] for name, table in catalog["raw_tables"].items()},
        "catalog_qualified_count": len(matched_ids),
        "catalog_qualified_source_ids": matched_ids,
        "minimum_per_frame_required": required,
        "usable_frame_count": len(frame_results),
        "frames_with_minimum_candidates": count_ge_min,
        "minimum_candidates_in_frame": min(measured_counts) if measured_counts else None,
        "maximum_candidates_in_frame": max(measured_counts) if measured_counts else None,
        "photometry_started": False,
        "method": "Gaia DR3 positions propagated linearly from J2016.0; WCS projection into each usable cached IMAGE; declared 60 arcsec edge margin; hard flags decoded from FITS MSKN/MSKD cards in a 5x5 FLAGS window.",
        "per_frame_counts": frame_results,
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, separators=(",", ":")))
    print(f"OUTPUT {OUTPUT_PATH.relative_to(ROOT).as_posix()}")
    return result


if __name__ == "__main__":
    vet_case_frames()
