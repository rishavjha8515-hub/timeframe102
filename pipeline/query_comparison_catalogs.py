"""Locally query IRSA 2MASS/AllWISE and vet the preselected Gaia candidates.

Run from the repository root with the project Python environment:
    python -m pipeline.query_comparison_catalogs

This only determines catalog matches. It does not claim comparison stars are
usable in the SPHEREx frames; per-frame edge, flag, and aperture checks remain
required and the predeclared minimum is not relaxed.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astroquery.ipac.irsa import Irsa

from .quality_policy import POLICY, preselect_gaia_comparisons

ROOT = Path(__file__).resolve().parents[1]
GAIA_PATH = ROOT / "pipeline/cache/control-star/gaia_north.ecsv"
OUT_DIR = ROOT / "pipeline/cache/control-star/comparison_catalogs"
TARGET_ID = "1633306316700113024"
TARGET_RA_DEG = 269.77410473768515
TARGET_DEC_DEG = 66.46748335101324
PLANNED_SEARCH_RADIUS_DEG = 0.25
GAIA_SEARCH_RADIUS_DEG = 0.30
SEARCH_RADIUS_DEG = GAIA_SEARCH_RADIUS_DEG


def _value(row: Any, key: str) -> Any:
    value = row[key]
    if np.ma.is_masked(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def vet_catalog_matches(gaia: Table, tmass: Table, allwise: Table,
                        *, search_radius_deg: float = SEARCH_RADIUS_DEG) -> list[dict[str, Any]]:
    """Apply fixed Gaia, positional, and catalog-quality criteria to each candidate."""
    criteria = POLICY["comparison_star_criteria"]
    candidate_ids = preselect_gaia_comparisons(
        gaia, TARGET_RA_DEG, TARGET_DEC_DEG, TARGET_ID, criteria,
        search_radius_arcsec=search_radius_deg * 3600.0,
    )
    tmass_coords = SkyCoord(np.asarray(tmass["ra"], float) * u.deg,
                            np.asarray(tmass["dec"], float) * u.deg)
    allwise_coords = SkyCoord(np.asarray(allwise["ra"], float) * u.deg,
                              np.asarray(allwise["dec"], float) * u.deg)
    by_id = {str(row["source_id"]): row for row in gaia}
    vetted = []
    for source_id in candidate_ids:
        row = by_id[source_id]
        position = SkyCoord(float(row["ra"]) * u.deg, float(row["dec"]) * u.deg)
        ti = int(np.argmin(position.separation(tmass_coords).arcsec))
        wi = int(np.argmin(position.separation(allwise_coords).arcsec))
        sep_2mass = float(position.separation(tmass_coords[ti]).arcsec)
        sep_allwise = float(position.separation(allwise_coords[wi]).arcsec)
        ph_qual_2mass = str(_value(tmass[ti], "ph_qual") or "")
        ph_qual_allwise = str(_value(allwise[wi], "ph_qual") or "")
        cc_flags = str(_value(allwise[wi], "cc_flags") or "")
        ext_flg_value = _value(allwise[wi], "ext_flg")
        ext_flg = int(ext_flg_value) if ext_flg_value is not None else None
        reasons = []
        if sep_2mass > criteria["two_mass_match_max_arcsec"]:
            reasons.append("no 2MASS match within the declared radius")
        if ph_qual_2mass != criteria["two_mass_ph_qual"]:
            reasons.append("2MASS ph_qual does not satisfy the declared AAA requirement")
        if sep_allwise > criteria["allwise_match_max_arcsec"]:
            reasons.append("no AllWISE match within the declared radius")
        if ph_qual_allwise[:2] != criteria["allwise_ph_qual_w1_w2"]:
            reasons.append("AllWISE W1/W2 ph_qual does not satisfy the declared AA requirement")
        if cc_flags[:2] != criteria["allwise_cc_flags_w1_w2"]:
            reasons.append("AllWISE W1/W2 cc_flags do not satisfy the declared 00 requirement")
        if ext_flg != criteria["allwise_ext_flg"]:
            reasons.append("AllWISE ext_flg does not satisfy the declared point-source requirement")
        vetted.append({
            "source_id": source_id,
            "ra_deg": float(row["ra"]), "dec_deg": float(row["dec"]),
            "g_mag": float(row["phot_g_mean_mag"]),
            "two_mass_designation": str(_value(tmass[ti], "designation") or ""),
            "two_mass_separation_arcsec": sep_2mass,
            "two_mass_ph_qual": ph_qual_2mass,
            "allwise_designation": str(_value(allwise[wi], "designation") or ""),
            "allwise_separation_arcsec": sep_allwise,
            "allwise_ph_qual": ph_qual_allwise,
            "allwise_cc_flags": cc_flags,
            "allwise_ext_flg": ext_flg,
            "passed_catalog_gates": not reasons,
            "rejection_reasons": reasons,
        })
    return vetted


def _query(table: str, columns: str) -> Table:
    query = (
        f"SELECT {columns} FROM {table} WHERE "
        "CONTAINS(POINT('ICRS', ra, dec), "
        f"CIRCLE('ICRS', {TARGET_RA_DEG}, {TARGET_DEC_DEG}, {SEARCH_RADIUS_DEG})) = 1"
    )
    return Irsa.query_tap(query=query, maxrec=100000)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "target_gaia_source_id": TARGET_ID,
        "query_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "search_radius_deg": SEARCH_RADIUS_DEG,
        "search_radius_deviation": {
            "planned_radius_deg": PLANNED_SEARCH_RADIUS_DEG,
            "selected_radius_deg": SEARCH_RADIUS_DEG,
            "reason": "Count-only decision before catalog matching or floor calculation; counts are computed from the cached Gaia table.",
        },
        "criteria": POLICY["comparison_star_criteria"],
        "status": "failed",
        "error": None,
    }
    try:
        gaia = Table.read(GAIA_PATH, format="ascii.ecsv")
        planned_ids = preselect_gaia_comparisons(
            gaia, TARGET_RA_DEG, TARGET_DEC_DEG, TARGET_ID, POLICY["comparison_star_criteria"],
            search_radius_arcsec=PLANNED_SEARCH_RADIUS_DEG * 3600.0,
        )
        selected_ids = preselect_gaia_comparisons(
            gaia, TARGET_RA_DEG, TARGET_DEC_DEG, TARGET_ID, POLICY["comparison_star_criteria"],
            search_radius_arcsec=SEARCH_RADIUS_DEG * 3600.0,
        )
        result["search_radius_deviation"]["candidate_count_at_planned_radius"] = len(planned_ids)
        result["search_radius_deviation"]["candidate_count_at_selected_radius"] = len(selected_ids)
        result["search_radius_deviation"]["reason"] = (
            "Count-only decision before catalog matching or floor calculation: the cached Gaia table yielded "
            f"{len(planned_ids)} candidates at {PLANNED_SEARCH_RADIUS_DEG:.2f} deg and "
            f"{len(selected_ids)} at {SEARCH_RADIUS_DEG:.2f} deg after unchanged fixed Gaia cuts."
        )
        tmass = _query("fp_psc", "designation, ra, dec, ph_qual")
        allwise = _query("allwise_p3as_psd", "designation, ra, dec, ph_qual, cc_flags, ext_flg")
        tmass.write(OUT_DIR / "2mass_fp_psc.ecsv", format="ascii.ecsv", overwrite=True)
        allwise.write(OUT_DIR / "allwise_p3as_psd.ecsv", format="ascii.ecsv", overwrite=True)
        candidates = vet_catalog_matches(gaia, tmass, allwise, search_radius_deg=SEARCH_RADIUS_DEG)
        result.update({
            "status": "complete",
            "gaia_candidate_count": len(candidates),
            "two_mass_rows": len(tmass),
            "allwise_rows": len(allwise),
            "catalog_matched_candidate_count": sum(c["passed_catalog_gates"] for c in candidates),
            "candidates": candidates,
            "per_frame_photometry_started": False,
            "note": "Per-frame quality/edge/photometry gates are intentionally not run by this catalog-only probe.",
        })
        print(f"STATUS complete Gaia={len(candidates)} catalog_qualified={result['catalog_matched_candidate_count']} "
              f"2MASS_rows={len(tmass)} AllWISE_rows={len(allwise)}")
    except Exception as exc:  # Keep the exact network/service failure in the artifact.
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(f"STATUS failed error={result['error']}")
    output = OUT_DIR / "comparison_catalog_vetting.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(f"OUTPUT {output.relative_to(ROOT).as_posix()}")
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
