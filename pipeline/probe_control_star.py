"""Find a low-proper-motion, isolated Gaia star with SPHEREx deep coverage.

Run locally where Gaia and IRSA are reachable::

    python -m pipeline.probe_control_star --cache-dir pipeline/cache/control-star

Before querying, the script fixes these criteria: Gaia DR3 G=10--14, total
proper motion <10 mas/yr, RUWE <1.4, not duplicated or flagged VARIABLE; no
neighbor within 15 arcsec with G <= candidate G+3. It queries point-position
SIA using a (RA, Dec, radius) tuple, counts distinct obs_id values across the
deep collections, requires an actual wavelength span, and downloads up to three
0.1-degree FITS cutouts across detector IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import astropy.units as u
import numpy as np
import requests
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.table import Table
from astroquery.gaia import Gaia
from astroquery.ipac.irsa import Irsa

POLES = (("north", 270.0, 66.560708), ("south", 90.0, -66.560708))
GAIA_QUERY_RADIUS_DEG = 1.0
G_MIN, G_MAX, G_CATALOG_MAX = 10.0, 14.0, 18.0
PM_MAX_MAS_YR, RUWE_MAX = 10.0, 1.4
ISOLATION_ARCSEC, BRIGHT_NEIGHBOR_DELTA = 15.0, 3.0
SIA_RADIUS_DEG = 0.001
COLLECTIONS = ("spherex_dr1_deep", "spherex_qr3_deep")
MIN_OBS_IDS, MIN_DETECTORS, MAX_CANDIDATES = 10, 3, 30
MIN_WAVELENGTH_SPAN_UM = 0.5
CUTOUT_SIZE_DEG = 0.1


def proper_motion_total(pmra: float, pmdec: float) -> float:
    return float(np.hypot(pmra, pmdec))


def detector_from_url(url: str) -> int | None:
    match = re.search(r"/([1-6])/[^/]*D[1-6]_spx_l2b", url)
    if match:
        return int(match.group(1))
    match = re.search(r"D([1-6])_spx_l2b", url)
    return int(match.group(1)) if match else None


def cutout_url(access_url: str, ra_deg: float, dec_deg: float, size_deg: float = CUTOUT_SIZE_DEG) -> str:
    parsed = urlparse(access_url)
    if parsed.scheme != "https" or parsed.hostname != "irsa.ipac.caltech.edu":
        raise ValueError("Expected an on-premises IRSA HTTPS access_url")
    params = dict(parse_qsl(parsed.query, keep_blank_values=True))
    params.update(center=f"{ra_deg:.10f},{dec_deg:.10f}", size=f"{size_deg:g}")
    return urlunparse(parsed._replace(query=urlencode(params)))


def _gaia_candidates(table: Table) -> list[dict]:
    def numeric(column: str) -> np.ndarray:
        return np.ma.asarray(table[column], dtype=float).filled(np.nan)

    ra, dec = numeric("ra"), numeric("dec")
    pmra, pmdec = numeric("pmra"), numeric("pmdec")
    gmag, ruwe = numeric("phot_g_mean_mag"), numeric("ruwe")
    pm = np.hypot(pmra, pmdec)
    variables = np.ma.asarray(table["phot_variable_flag"]).filled("UNKNOWN").astype(str)
    duplicated = np.ma.asarray(table["duplicated_source"], dtype=bool).filled(True)
    mask = (np.isfinite(pm) & np.isfinite(gmag) & np.isfinite(ruwe) &
            (pm < PM_MAX_MAS_YR) & (gmag >= G_MIN) & (gmag <= G_MAX) &
            (ruwe < RUWE_MAX) & ~duplicated & (variables != "VARIABLE"))
    ids = np.flatnonzero(mask)
    if not len(ids):
        return []
    all_coords = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    cand_idx, near_idx, sep, _ = search_around_sky(
        all_coords[ids], all_coords, ISOLATION_ARCSEC * u.arcsec
    )
    has_bright_neighbor: set[int] = set()
    for ci, ni in zip(cand_idx, near_idx):
        candidate_index, neighbor_index = int(ids[int(ci)]), int(ni)
        if candidate_index != neighbor_index and gmag[neighbor_index] <= gmag[candidate_index] + BRIGHT_NEIGHBOR_DELTA:
            has_bright_neighbor.add(candidate_index)
    rows = []
    for i in ids:
        if int(i) in has_bright_neighbor:
            continue
        rows.append({
            "source_id": str(table["source_id"][i]), "ra": float(ra[i]), "dec": float(dec[i]),
            "pmra": float(pmra[i]), "pmdec": float(pmdec[i]), "pm_total": float(pm[i]),
            "gmag": float(gmag[i]), "ruwe": float(ruwe[i]),
        })
    return sorted(rows, key=lambda r: (r["pm_total"], r["gmag"]))


def _query_gaia(pole: str, ra: float, dec: float, cache: Path) -> Table:
    query = f"""SELECT TOP 50000 source_id, ra, dec, pmra, pmdec, phot_g_mean_mag,
ruwe, duplicated_source, phot_variable_flag
FROM gaiadr3.gaia_source
WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {GAIA_QUERY_RADIUS_DEG}))=1
AND phot_g_mean_mag < {G_CATALOG_MAX}"""
    table = Gaia.launch_job_async(query, verbose=False).get_results()
    path = cache / f"gaia_{pole}.ecsv"
    table.write(path, format="ascii.ecsv", overwrite=True)
    print(f"GAIA pole={pole} rows={len(table)} columns={table.colnames} raw={path}")
    if len(table) >= 50000:
        raise RuntimeError(f"Gaia result reached TOP 50000 cap for {pole}; isolation screen is incomplete")
    return table


def _query_sia(candidate: dict, collection: str, cache: Path) -> Table:
    table = Irsa.query_sia(
        pos=(candidate["ra"], candidate["dec"], SIA_RADIUS_DEG),
        collection=collection, maxrec=10000,
    )
    path = cache / f"sia_{candidate['source_id']}_{collection}.ecsv"
    table.write(path, format="ascii.ecsv", overwrite=True)
    obs = {str(v) for v in table["obs_id"] if str(v) not in {"", "None", "masked"}}
    urls = [str(v) for v in table["access_url"] if str(v).startswith("https://")]
    detectors = sorted({d for url in urls if (d := detector_from_url(url)) is not None})
    def numeric_range(column: str) -> tuple[float, float] | None:
        if column not in table.colnames:
            return None
        try:
            values = np.asarray(table[column], dtype=float)
            values = values[np.isfinite(values)]
            return (float(values.min()), float(values.max())) if len(values) else None
        except (TypeError, ValueError):
            return None

    t_min_range, t_max_range = numeric_range("t_min"), numeric_range("t_max")
    wave_range = _wavelength_bounds([(collection, table)])
    print(f"SIA collection={collection} rows={len(table)} distinct_obs_id={len(obs)} "
          f"detectors={detectors} t_min_range_mjd={t_min_range} t_max_range_mjd={t_max_range} "
          f"em_min_to_max_range={wave_range} columns={table.colnames}")
    if len(table) >= 10000:
        print(f"SIA_ROW_CAP collection={collection} maxrec=10000; count may be truncated")
    return table


def _wavelength_bounds(tables: list[tuple[str, Table]]) -> tuple[float, float] | None:
    lows, highs = [], []
    for _, table in tables:
        for column, values in (("em_min", lows), ("em_max", highs)):
            if column not in table.colnames:
                continue
            quantity = getattr(table[column], "quantity", None)
            unit = getattr(table[column], "unit", None)
            if quantity is None or unit is None:
                raise ValueError(f"{column} has no unit; refusing to assume wavelength units")
            arr = np.asarray(quantity.to_value(u.um), dtype=float)
            arr = arr[np.isfinite(arr)]
            if len(arr):
                values.extend(arr.tolist())
    if not lows or not highs:
        return None
    return float(min(lows)), float(max(highs))


def _cutouts(candidate: dict, tables: list[tuple[str, Table]], cache: Path) -> list[dict]:
    by_detector: dict[int, tuple[str, Any]] = {}
    for collection, table in tables:
        for row in table:
            url = str(row["access_url"])
            detector = detector_from_url(url)
            if detector is not None:
                by_detector.setdefault(detector, (collection, row))
    detectors = sorted(by_detector)
    if not detectors:
        return []
    indexes = np.linspace(0, len(detectors)-1, min(3, len(detectors)), dtype=int)
    cut_dir = cache / "cutouts"
    cut_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    with requests.Session() as session:
        for index in indexes:
            detector = detectors[int(index)]
            collection, row = by_detector[detector]
            url = cutout_url(str(row["access_url"]), candidate["ra"], candidate["dec"])
            path = cut_dir / f"gaia-{candidate['source_id']}-{collection}-D{detector}.fits"
            response = session.get(url, stream=True, timeout=180)
            response.raise_for_status()
            digest, size = hashlib.sha256(), 0
            with path.open("wb") as stream:
                for block in response.iter_content(1024 * 1024):
                    if block:
                        stream.write(block); digest.update(block); size += len(block)
            print(f"CUTOUT detector=D{detector} status={response.status_code} bytes={size} path={path}")
            outputs.append({"detector": detector, "collection": collection, "obs_id": str(row["obs_id"]),
                            "path": str(path), "size_bytes": size, "sha256": digest.hexdigest()})
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("pipeline/cache/control-star"))
    parser.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES)
    args = parser.parse_args(argv)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    print("DECLARED_CRITERIA " + json.dumps({
        "gaia_dr3_g_mag": [G_MIN, G_MAX], "total_pm_mas_yr_lt": PM_MAX_MAS_YR,
        "ruwe_lt": RUWE_MAX, "exclude_duplicated": True, "exclude_variable_flag": "VARIABLE",
        "isolated_within_arcsec": ISOLATION_ARCSEC, "reject_neighbor_g_le_candidate_plus": BRIGHT_NEIGHBOR_DELTA,
        "sia_pos_tuple_radius_deg": SIA_RADIUS_DEG, "minimum_distinct_obs_ids": MIN_OBS_IDS,
        "minimum_detectors": MIN_DETECTORS, "cutout_size_deg": CUTOUT_SIZE_DEG,
        "minimum_wavelength_span_um": MIN_WAVELENGTH_SPAN_UM,
    }, sort_keys=True))

    candidates = []
    for pole, ra, dec in POLES:
        try:
            raw = _query_gaia(pole, ra, dec, args.cache_dir)
            selected = _gaia_candidates(raw)[:args.max_candidates]
            for row in selected:
                row["pole"] = pole
            print(f"GAIA_FILTER pole={pole} isolated_candidates={len(selected)}")
            candidates.extend(selected)
        except Exception as exc:
            print(f"GAIA_QUERY_FAILED pole={pole} {type(exc).__name__}: {exc}")
    candidates.sort(key=lambda r: (r["pm_total"], r["gmag"]))
    if not candidates:
        print("NO_GAIA_CANDIDATES")
        return 2

    best = None
    for candidate in candidates:
        tables = []
        for collection in COLLECTIONS:
            try:
                tables.append((collection, _query_sia(candidate, collection, args.cache_dir)))
            except Exception as exc:
                print(f"SIA_QUERY_FAILED source_id={candidate['source_id']} collection={collection} {type(exc).__name__}: {exc}")
        obs_ids = {str(row["obs_id"]) for collection, table in tables
                   for row in table if str(row["obs_id"]) not in {"", "None", "masked"}}
        detector_ids = {detector_from_url(str(row["access_url"])) for _, table in tables for row in table}
        detector_ids.discard(None)
        wave_bounds = _wavelength_bounds(tables)
        wave_span = wave_bounds[1] - wave_bounds[0] if wave_bounds else 0.0
        score = (len(obs_ids), wave_span, len(detector_ids))
        print(f"GAIA_SIA_CANDIDATE source_id={candidate['source_id']} pole={candidate['pole']} "
              f"ra={candidate['ra']:.8f} dec={candidate['dec']:.8f} pm={candidate['pm_total']:.4f} "
              f"G={candidate['gmag']:.3f} distinct_obs_id={len(obs_ids)} detectors={sorted(detector_ids)} "
              f"wavelength_bounds_um={wave_bounds} span_um={wave_span:.4f}")
        if best is None or score > best[0]:
            best = (score, candidate, tables, len(obs_ids), sorted(detector_ids))
        if (len(obs_ids) >= MIN_OBS_IDS and len(detector_ids) >= MIN_DETECTORS
                and wave_span >= MIN_WAVELENGTH_SPAN_UM):
            best = (score, candidate, tables, len(obs_ids), sorted(detector_ids))
            break
    assert best is not None
    _, candidate, tables, obs_count, detector_ids = best
    wave_bounds = _wavelength_bounds(tables)
    wave_span = wave_bounds[1] - wave_bounds[0] if wave_bounds else 0.0
    suitable = (obs_count >= MIN_OBS_IDS and len(detector_ids) >= MIN_DETECTORS
                and wave_span >= MIN_WAVELENGTH_SPAN_UM)
    print(f"STAR_CANDIDATE suitable_by_declared_screen={suitable} source_id={candidate['source_id']} "
          f"pole={candidate['pole']} ra_j2016_deg={candidate['ra']:.10f} dec_j2016_deg={candidate['dec']:.10f} "
          f"pmra={candidate['pmra']:.4f} pmdec={candidate['pmdec']:.4f} pm_total={candidate['pm_total']:.4f} "
          f"G={candidate['gmag']:.4f} RUWE={candidate['ruwe']:.4f} distinct_obs_id={obs_count} "
          f"detectors={detector_ids} wavelength_bounds_um={wave_bounds} span_um={wave_span:.4f}")
    cutouts = _cutouts(candidate, tables, args.cache_dir) if suitable else []
    manifest = {"candidate": candidate, "distinct_obs_id_union": obs_count, "detectors": detector_ids,
                "wavelength_bounds_um": wave_bounds, "wavelength_span_um": wave_span,
                "suitable_by_declared_screen": suitable, "cutouts": cutouts}
    (args.cache_dir / "control_star_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not suitable:
        print("NO_CANDIDATE_MET_COVERAGE_CRITERIA; raw Gaia/SIA tables retained; criteria were not relaxed")
        return 2
    return 0 if cutouts else 1


if __name__ == "__main__":
    sys.exit(main())
