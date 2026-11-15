"""Use CDS VizieR as a catalog-only fallback when IRSA TAP is unavailable.

The Gaia DR3 table is the existing cached input; this script queries the 2MASS
PSC and AllWISE catalogs from VizieR within the predeclared widened 0.30 deg
search radius. It applies the unchanged Gaia, positional-match, and catalog
quality cuts. Per-frame edge/flag/photometry gates are still required later.

Run from the repository root:
    python -m pipeline.query_comparison_catalogs_vizier
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astroquery.vizier import Vizier

from .quality_policy import POLICY, preselect_gaia_comparisons
from .query_comparison_catalogs import (
    GAIA_SEARCH_RADIUS_DEG,
    PLANNED_SEARCH_RADIUS_DEG,
    TARGET_DEC_DEG,
    TARGET_ID,
    TARGET_RA_DEG,
    vet_catalog_matches,
)


ROOT = Path(__file__).resolve().parents[1]
GAIA_PATH = ROOT / "pipeline/cache/control-star/gaia_north.ecsv"
OUT_DIR = ROOT / "pipeline/cache/control-star/comparison_catalogs/vizier"
CATALOGS = {"tmass": "II/246/out", "allwise": "II/328/allwise"}
# CDS currently reports the Strasbourg primary service as degraded and lists
# these public mirrors. Try Harvard first, then IDIA, without changing the cuts.
VIZIER_SERVERS = ("vizier.cfa.harvard.edu", "vizier.idia.ac.za")


def _find_column(table: Table, *names: str) -> str:
    normalized = {name.casefold(): name for name in table.colnames}
    for name in names:
        if name.casefold() in normalized:
            return normalized[name.casefold()]
    raise ValueError(f"None of the required columns {names} found; returned columns={table.colnames}")


def normalize_tmass(table: Table) -> Table:
    return Table({
        "designation": np.asarray(table[_find_column(table, "2MASS", "designation")]).astype(str),
        "ra": np.asarray(table[_find_column(table, "RAJ2000", "_RAJ2000", "ra")], dtype=float),
        "dec": np.asarray(table[_find_column(table, "DEJ2000", "_DEJ2000", "dec")], dtype=float),
        "ph_qual": np.asarray(table[_find_column(table, "Qflg", "ph_qual")]).astype(str),
    })


def normalize_allwise(table: Table) -> Table:
    return Table({
        "designation": np.asarray(table[_find_column(table, "AllWISE", "designation")]).astype(str),
        "ra": np.asarray(table[_find_column(table, "RAJ2000", "_RAJ2000", "ra")], dtype=float),
        "dec": np.asarray(table[_find_column(table, "DEJ2000", "_DEJ2000", "dec")], dtype=float),
        "ph_qual": np.asarray(table[_find_column(table, "qph", "ph_qual")]).astype(str),
        "cc_flags": np.asarray(table[_find_column(table, "ccf", "cc_flags")]).astype(str),
        "ext_flg": np.asarray(table[_find_column(table, "ex", "ext_flg")], dtype=int),
    })


def _query_vizier(catalog: str, center: SkyCoord, radius_deg: float) -> tuple[Table, str]:
    failures = []
    for server in VIZIER_SERVERS:
        try:
            client = Vizier(columns=["*"], row_limit=100000, vizier_server=server)
            # The returned catalogs are persisted as ECSV below; avoid
            # Astroquery's default cache under the user's home directory.
            tables = client.query_region(center, radius=radius_deg * u.deg,
                                         catalog=catalog, cache=False)
            if not tables or len(tables[0]) == 0:
                raise RuntimeError(f"VizieR returned no rows for {catalog}")
            return tables[0], server
        except Exception as exc:
            failures.append(f"{server}: {type(exc).__name__}: {exc}")
    raise RuntimeError("All configured VizieR mirrors failed for " + catalog + "; " + " | ".join(failures))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "status": "failed",
        "catalog_source": "CDS VizieR public mirrors",
        "mirror_attempt_order": list(VIZIER_SERVERS),
        "catalogs": CATALOGS,
        "target_gaia_source_id": TARGET_ID,
        "search_radius_deg": GAIA_SEARCH_RADIUS_DEG,
        "search_radius_deviation": {
            "planned_radius_deg": PLANNED_SEARCH_RADIUS_DEG,
            "selected_radius_deg": GAIA_SEARCH_RADIUS_DEG,
            "reason": "Count-only decision before 2MASS/AllWISE matching or any floor calculation.",
        },
        "criteria": POLICY["comparison_star_criteria"],
        "query_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "raw_tables": {},
    }
    try:
        gaia = Table.read(GAIA_PATH, format="ascii.ecsv")
        planned_ids = preselect_gaia_comparisons(
            gaia, TARGET_RA_DEG, TARGET_DEC_DEG, TARGET_ID, POLICY["comparison_star_criteria"],
            search_radius_arcsec=PLANNED_SEARCH_RADIUS_DEG * 3600.0,
        )
        selected_ids = preselect_gaia_comparisons(
            gaia, TARGET_RA_DEG, TARGET_DEC_DEG, TARGET_ID, POLICY["comparison_star_criteria"],
            search_radius_arcsec=GAIA_SEARCH_RADIUS_DEG * 3600.0,
        )
        result["search_radius_deviation"]["candidate_count_at_planned_radius"] = len(planned_ids)
        result["search_radius_deviation"]["candidate_count_at_selected_radius"] = len(selected_ids)
        result["search_radius_deviation"]["reason"] = (
            "Count-only decision before 2MASS/AllWISE matching or any floor calculation: the cached Gaia table yielded "
            f"{len(planned_ids)} candidates at {PLANNED_SEARCH_RADIUS_DEG:.2f} deg and "
            f"{len(selected_ids)} at {GAIA_SEARCH_RADIUS_DEG:.2f} deg after unchanged fixed Gaia cuts."
        )
        center = SkyCoord(TARGET_RA_DEG * u.deg, TARGET_DEC_DEG * u.deg, frame="icrs")
        raw_tmass, tmass_server = _query_vizier(CATALOGS["tmass"], center, GAIA_SEARCH_RADIUS_DEG)
        raw_allwise, allwise_server = _query_vizier(CATALOGS["allwise"], center, GAIA_SEARCH_RADIUS_DEG)
        tmass_path = OUT_DIR / "2mass_psc_vizier.ecsv"
        allwise_path = OUT_DIR / "allwise_vizier.ecsv"
        raw_tmass.write(tmass_path, format="ascii.ecsv", overwrite=True)
        raw_allwise.write(allwise_path, format="ascii.ecsv", overwrite=True)
        result["raw_tables"] = {
            "2mass": {"server": tmass_server, "row_count": len(raw_tmass), "columns": raw_tmass.colnames,
                      "path": tmass_path.relative_to(ROOT).as_posix()},
            "allwise": {"server": allwise_server, "row_count": len(raw_allwise), "columns": raw_allwise.colnames,
                        "path": allwise_path.relative_to(ROOT).as_posix()},
        }
        candidates = vet_catalog_matches(
            gaia, normalize_tmass(raw_tmass), normalize_allwise(raw_allwise),
            search_radius_deg=GAIA_SEARCH_RADIUS_DEG,
        )
        result.update({
            "status": "complete",
            "gaia_candidate_count": len(candidates),
            "catalog_qualified_candidate_count": sum(item["passed_catalog_gates"] for item in candidates),
            "candidates": candidates,
            "per_frame_vetting_started": False,
            "note": "Catalog quality and positional matches only; edge, flags, and photometry remain to be checked in each usable frame.",
        })
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    output = OUT_DIR / "comparison_catalog_vetting_vizier.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, separators=(",", ":")))
    print(f"OUTPUT {output.relative_to(ROOT).as_posix()}")
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
