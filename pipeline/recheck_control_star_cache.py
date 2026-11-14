"""Re-evaluate cached Gaia/SIA control-star results offline with unit-aware wavelengths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from astropy.table import Table
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import SkyCoord

from .probe_control_star import (
    COLLECTIONS,
    MIN_DETECTORS,
    MIN_OBS_IDS,
    MIN_WAVELENGTH_SPAN_UM,
    _wavelength_bounds,
    detector_from_url,
)
from .probe_control_star import G_MIN, G_MAX, PM_MAX_MAS_YR, RUWE_MAX, ISOLATION_ARCSEC, BRIGHT_NEIGHBOR_DELTA


def _candidate_index(gaia_tables: list[Table], source_ids: set[str]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for table in gaia_tables:
        ids = np.asarray(table["source_id"]).astype(str)
        gmag = np.ma.asarray(table["phot_g_mean_mag"], dtype=float).filled(np.nan)
        pmra = np.ma.asarray(table["pmra"], dtype=float).filled(np.nan)
        pmdec = np.ma.asarray(table["pmdec"], dtype=float).filled(np.nan)
        pm = np.hypot(pmra, pmdec)
        ruwe = np.ma.asarray(table["ruwe"], dtype=float).filled(np.nan)
        duplicated = np.ma.asarray(table["duplicated_source"], dtype=bool).filled(True)
        variable = np.ma.asarray(table["phot_variable_flag"]).filled("UNKNOWN").astype(str)
        coords = SkyCoord(np.asarray(table["ra"], dtype=float) * u.deg,
                          np.asarray(table["dec"], dtype=float) * u.deg, frame="icrs")
        for candidate_id in source_ids.intersection(ids.tolist()):
            i = int(np.flatnonzero(ids == candidate_id)[0])
            if not (np.isfinite(pm[i]) and np.isfinite(gmag[i]) and np.isfinite(ruwe[i])
                    and pm[i] < PM_MAX_MAS_YR and G_MIN <= gmag[i] <= G_MAX
                    and ruwe[i] < RUWE_MAX and not duplicated[i] and variable[i] != "VARIABLE"):
                continue
            neighbors = np.flatnonzero(coords[i].separation(coords) < ISOLATION_ARCSEC * u.arcsec)
            contaminated = any(j != i and gmag[j] <= gmag[i] + BRIGHT_NEIGHBOR_DELTA for j in neighbors)
            if not contaminated:
                index[candidate_id] = {
                    "source_id": candidate_id, "ra": float(table["ra"][i]), "dec": float(table["dec"][i]),
                    "pmra": float(pmra[i]), "pmdec": float(pmdec[i]), "pm_total": float(pm[i]),
                    "gmag": float(gmag[i]), "ruwe": float(ruwe[i]),
                }
    return index


def recheck(cache_dir: Path) -> dict[str, Any]:
    gaia_paths = sorted(cache_dir.glob("gaia_*.ecsv"))
    gaia_tables = [Table.read(path, format="ascii.ecsv") for path in gaia_paths]
    source_ids = {path.name.split("_")[1] for path in cache_dir.glob("sia_*.ecsv")}
    candidates = _candidate_index(gaia_tables, source_ids)
    results = []
    for source_id in sorted({path.name.split("_")[1] for path in cache_dir.glob("sia_*.ecsv")}):
        if source_id not in candidates:
            continue
        by_release = {}
        release_obs = {}
        release_detectors = {}
        release_wave = {}
        for collection in COLLECTIONS:
            path = cache_dir / f"sia_{source_id}_{collection}.ecsv"
            if not path.is_file():
                continue
            table = Table.read(path, format="ascii.ecsv")
            row_cap_hit = len(table) >= 10000
            obs_ids = {str(value) for value in table["obs_id"]
                       if str(value) not in {"", "None", "masked"}}
            urls = [str(value) for value in table["access_url"]
                    if str(value).startswith("https://")]
            detectors = sorted({d for url in urls if (d := detector_from_url(url)) is not None})
            wave_bounds = _wavelength_bounds([(collection, table)])
            if "t_min" in table.colnames and "t_max" in table.colnames:
                start = np.asarray(table["t_min"], dtype=float)
                end = np.asarray(table["t_max"], dtype=float)
                mjd_lo, mjd_hi = float(np.nanmin(start)), float(np.nanmax(end))
                time_range = [Time(mjd_lo, format="mjd", scale="utc").isot,
                              Time(mjd_hi, format="mjd", scale="utc").isot]
            else:
                time_range = None
            by_release[collection] = {
                "rows_returned": len(table),
                "maxrec_cap": 10000,
                "row_count_is_lower_bound": row_cap_hit,
                "distinct_obs_id_count": len(obs_ids),
                "distinct_obs_id_count_is_lower_bound": row_cap_hit,
                "utc_time_range": time_range,
                "detectors": detectors,
                "wavelength_range_um": list(wave_bounds) if wave_bounds else None,
            }
            release_obs[collection] = obs_ids
            release_detectors[collection] = set(detectors)
            release_wave[collection] = wave_bounds
        union_obs = set().union(*release_obs.values()) if release_obs else set()
        union_detectors = set().union(*release_detectors.values()) if release_detectors else set()
        available_waves = [bounds for bounds in release_wave.values() if bounds]
        wave_bounds = ((min(bounds[0] for bounds in available_waves),
                        max(bounds[1] for bounds in available_waves)) if available_waves else None)
        wave_span = wave_bounds[1] - wave_bounds[0] if wave_bounds else 0.0
        suitable_releases = {
            collection: (len(release_obs[collection]) >= MIN_OBS_IDS
                         and len(release_detectors[collection]) >= MIN_DETECTORS
                         and release_wave[collection] is not None
                         and release_wave[collection][1] - release_wave[collection][0] >= MIN_WAVELENGTH_SPAN_UM)
            for collection in by_release
        }
        gaia = candidates[source_id]
        results.append({
            "source_id": source_id,
            "ra_j2016_deg": gaia["ra"], "dec_j2016_deg": gaia["dec"],
            "pmra_mas_yr": gaia["pmra"], "pmdec_mas_yr": gaia["pmdec"],
            "pm_total_mas_yr": gaia["pm_total"], "gmag": gaia["gmag"], "ruwe": gaia["ruwe"],
            "per_release": by_release,
            "distinct_obs_id_union": len(union_obs),
            "detectors_union": sorted(union_detectors),
            "wavelength_bounds_um": list(wave_bounds) if wave_bounds else None,
            "wavelength_span_um": wave_span,
            "suitable_by_unchanged_criteria_per_release": suitable_releases,
            "suitable_in_any_single_release": any(suitable_releases.values()),
        })
    return {
        "offline": True,
        "gaia_sources_cached": len(candidates),
        "thresholds_unchanged": {"minimum_distinct_obs_ids": MIN_OBS_IDS,
                                 "minimum_detectors": MIN_DETECTORS,
                                 "minimum_wavelength_span_um": MIN_WAVELENGTH_SPAN_UM},
        "candidates": sorted(results, key=lambda item: (-item["distinct_obs_id_union"], item["gmag"])),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("pipeline/cache/control-star"))
    parser.add_argument("--output", type=Path, default=Path("outputs/control-star-cache-review.json"))
    args = parser.parse_args(argv)
    result = recheck(args.cache_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for item in result["candidates"]:
        print(f"CACHED_CANDIDATE source_id={item['source_id']} G={item['gmag']:.3f} "
              f"distinct_obs_id_union={item['distinct_obs_id_union']} detectors={item['detectors_union']} "
              f"wavelength_um={item['wavelength_bounds_um']} suitable_in_any_single_release={item['suitable_in_any_single_release']}")
        for collection, release in item["per_release"].items():
            qualifier = ">=" if release["row_count_is_lower_bound"] else "="
            print(f"  RELEASE {collection} rows{qualifier}{release['rows_returned']} "
                  f"distinct_obs_id{qualifier}{release['distinct_obs_id_count']} "
                  f"time={release['utc_time_range']} detectors={release['detectors']} "
                  f"wavelength_um={release['wavelength_range_um']}")
    print(f"OFFLINE_RECHECK_COMPLETE candidates={len(result['candidates'])} output={args.output}")
    return 0 if result["candidates"] else 1


if __name__ == "__main__":
    sys.exit(main())
