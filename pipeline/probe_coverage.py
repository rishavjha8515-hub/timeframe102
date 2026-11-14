"""Probe IRSA SPHEREx coverage for candidate asteroids and sky regions.

This script only queries archive metadata. It does not download FITS products,
measure sources, or make scientific suitability claims. Query results are saved
as raw ECSV tables under ``pipeline/cache/probe`` by default.

Run with network access from the project root::

    python -m pipeline.probe_coverage

Use ``--dry-run`` for an offline check of installed Astroquery signatures and
the planned query set. No network request is made in dry-run mode.
"""

from __future__ import annotations

import argparse
import inspect
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astroquery.ipac.irsa import Irsa
from astroquery.ipac.irsa.most import Most


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "cache" / "probe"


@dataclass(frozen=True)
class AsteroidCandidate:
    name: str
    naif_id: str
    screening_group: str


# The set intentionally spans large, potentially saturation-prone bodies and
# smaller main-belt candidates. The group labels are screening strata, not
# measured brightness or archive-result claims.
ASTEROIDS = (
    AsteroidCandidate("Ceres", "20000001", "large-brightness-risk"),
    AsteroidCandidate("Pallas", "20000002", "large-brightness-risk"),
    AsteroidCandidate("Juno", "20000003", "large-brightness-risk"),
    AsteroidCandidate("Vesta", "20000004", "large-brightness-risk"),
    AsteroidCandidate("Hygiea", "20000010", "large-brightness-risk"),
    AsteroidCandidate("Iris", "20000007", "intermediate-screen"),
    AsteroidCandidate("Eunomia", "20000015", "intermediate-screen"),
    AsteroidCandidate("Psyche", "20000016", "intermediate-screen"),
    AsteroidCandidate("Flora", "20000008", "intermediate-screen"),
    AsteroidCandidate("Hebe", "20000006", "smaller-body-screen"),
    AsteroidCandidate("Victoria", "20000012", "smaller-body-screen"),
    AsteroidCandidate("Amphitrite", "20000029", "smaller-body-screen"),
    AsteroidCandidate("Europa", "20000052", "smaller-body-screen"),
)

POLE_REGIONS = (
    ("north_ecliptic_pole", 270.0, 66.560708),
    ("south_ecliptic_pole", 90.0, -66.560708),
)

# Widely separated ordinary-sky query points provide a basic non-deep-field
# coverage comparison. They are sampling positions, not proposed targets.
ORDINARY_REGIONS = (
    ("ordinary_ra000_dec000", 0.0, 0.0),
    ("ordinary_ra120_dec000", 120.0, 0.0),
    ("ordinary_ra240_dec000", 240.0, 0.0),
)

TIME_COLUMN_NAMES = (
    "t_min", "t_max", "start_date", "end_date", "mjd_start", "mjd_end",
    "date_obs", "obs_date", "observation_time", "time_utc",
)
WAVELENGTH_COLUMN_NAMES = (
    "em_min", "em_max", "band", "wavelength", "wavelength_um",
    "lambda_min", "lambda_max", "eff_wave",
)
SIA_MAXREC = 10000
POLE_RADIUS_DEG = 1.0
ORDINARY_RADIUS_DEG = 0.5


def _column_name_map(table: Any) -> dict[str, str]:
    return {str(name).lower(): str(name) for name in table.colnames}


def _nonmasked_values(column: Any) -> list[Any]:
    values: list[Any] = []
    for value in column:
        if np.ma.is_masked(value):
            continue
        values.append(value)
    return values


def _print_column_ranges(table: Any, candidate_names: tuple[str, ...], label: str) -> None:
    names = _column_name_map(table)
    found = [names[name] for name in candidate_names if name in names]
    print(f"{label}_COLUMNS {found}")
    for name in found:
        values = _nonmasked_values(table[name])
        if not values:
            print(f"{label}_RANGE {name}: no non-masked values")
            continue
        try:
            low, high = min(values), max(values)
        except TypeError:
            print(f"{label}_RANGE {name}: not orderable")
            continue
        unit = getattr(table[name], "unit", None)
        unit_text = f" {unit}" if unit is not None else ""
        print(f"{label}_RANGE {name}: {low} .. {high}{unit_text}")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.-") or "query"


def _save_raw_table(table: Any, cache_dir: Path, query_name: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{_safe_name(query_name)}.ecsv"
    table.write(path, format="ascii.ecsv", overwrite=True)
    print(f"RAW_RESULT {path}")
    return path


def _report_result(
    query_name: str,
    result: Any,
    cache_dir: Path,
    *,
    first_rows: int = 5,
    maxrec: int | None = None,
) -> None:
    print(f"\n=== RESULT {query_name} ===")
    if result is None:
        print("ROWS 0 (service returned no table)")
        return
    if not hasattr(result, "colnames"):
        print(f"UNEXPECTED_RESULT_TYPE {type(result).__name__}")
        print(repr(result)[:2000])
        return

    print(f"ROWS {len(result)}")
    if maxrec is not None and len(result) >= maxrec:
        print(f"ROW_LIMIT_REACHED maxrec={maxrec}; returned rows may be truncated and are not a total count")
    print(f"COLUMNS {list(result.colnames)}")
    _print_column_ranges(result, TIME_COLUMN_NAMES, "OBSERVATION_TIME")
    _print_column_ranges(result, WAVELENGTH_COLUMN_NAMES, "WAVELENGTH")
    print(f"FIRST_ROWS\n{result[:first_rows]}")
    _save_raw_table(result, cache_dir, query_name)


def _run_one(query_name: str, query: Any, cache_dir: Path) -> bool:
    print(f"\n=== QUERY {query_name} ===")
    try:
        result = query()
    except Exception as exc:  # Each query is isolated so one failure cannot stop the batch.
        print(f"QUERY_FAILED {type(exc).__name__}: {exc}")
        response = getattr(exc, "response", None)
        if response is not None:
            print(f"QUERY_ERROR_HTTP_STATUS {getattr(response, 'status_code', 'unknown')}")
            body = getattr(response, "text", "")
            if body:
                print(f"QUERY_ERROR_RESPONSE_BODY {body[:4000]!r}")
        return False
    _report_result(
        query_name,
        result,
        cache_dir,
        maxrec=SIA_MAXREC if query_name.startswith("sia_") else None,
    )
    return True


def _check_installed_signatures() -> None:
    most_parameters = inspect.signature(Most.query_object).parameters
    sia_parameters = inspect.signature(Irsa.query_sia).parameters
    tap_parameters = inspect.signature(Irsa.query_tap).parameters

    required = {
        "Most.query_object": {"catalog", "input_mode", "obj_nafid", "obs_begin", "obs_end", "output_mode"},
        "Irsa.query_sia": {"pos", "collection", "maxrec"},
        "Irsa.query_tap": {"query", "async_job", "maxrec"},
    }
    actual = {
        "Most.query_object": set(most_parameters),
        "Irsa.query_sia": set(sia_parameters),
        "Irsa.query_tap": set(tap_parameters),
    }
    for name, expected in required.items():
        missing = expected - actual[name]
        if missing:
            raise RuntimeError(f"Installed API signature for {name} lacks expected parameters: {sorted(missing)}")

    print("OFFLINE_API_SIGNATURES")
    print(f"Most.query_object{inspect.signature(Most.query_object)}")
    print(f"Irsa.query_sia{inspect.signature(Irsa.query_sia)}")
    print(f"Irsa.query_tap{inspect.signature(Irsa.query_tap)}")
    Most._validate_input({
        "catalog": "spherex",
        "input_type": "name_input",
        "output_mode": "Brief",
        "ephem_step": 0.25,
        "obj_name": "Ceres",
    })
    print("OFFLINE_MOST_NAME_VALIDATION_OK input_mode='name_input' maps to service input_type='name_input'")
    print("SIA_POSITION_ARGUMENT: tuple (ra_deg, dec_deg, radius_deg), not a 'CIRCLE ...' string")
    print("MOST_INPUT_ARGUMENT: input_mode='name_input' with obj_name=<NAIF ID>:NID")
    print("MOST_NAIF_NOTE: the installed Astroquery NAIF mode emits input_type='nafid_input'; IRSA MOST rejected that value in the user's live response")


def _query_specs(obs_begin: str, obs_end: str):
    for asteroid in ASTEROIDS:
        yield (
            f"most_{asteroid.naif_id}_{asteroid.name}_{asteroid.screening_group}",
            lambda asteroid=asteroid: Most.query_object(
                catalog="spherex",
                input_mode="name_input",
                obj_name=f"{asteroid.naif_id}:NID",
                obs_begin=obs_begin,
                obs_end=obs_end,
                output_mode="Brief",
            ),
        )

    for label, ra, dec in POLE_REGIONS:
        for collection in ("spherex_dr1_deep", "spherex_qr3_deep"):
            yield (
                f"sia_{collection}_{label}_r{POLE_RADIUS_DEG:g}deg_mr{SIA_MAXREC}",
                lambda ra=ra, dec=dec, collection=collection: Irsa.query_sia(
                    pos=(ra, dec, POLE_RADIUS_DEG),
                    collection=collection,
                    maxrec=SIA_MAXREC,
                ),
            )

    for label, ra, dec in ORDINARY_REGIONS:
        for collection in ("spherex_dr1", "spherex_qr3"):
            yield (
                f"sia_{collection}_{label}_r{ORDINARY_RADIUS_DEG:g}deg_mr{SIA_MAXREC}",
                lambda ra=ra, dec=dec, collection=collection: Irsa.query_sia(
                    pos=(ra, dec, ORDINARY_RADIUS_DEG),
                    collection=collection,
                    maxrec=SIA_MAXREC,
                ),
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obs-begin", default="2025-05-01", help="Earliest MOST UTC date (YYYY-MM-DD).")
    parser.add_argument("--obs-end", default="2026-11-14", help="Latest MOST UTC date (YYYY-MM-DD).")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory for raw ECSV query results (default: pipeline/cache/probe).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Check local API signatures and print query plan without network access.")
    parser.add_argument("--skip-most", action="store_true", help="Skip asteroid queries.")
    parser.add_argument("--skip-sia", action="store_true", help="Skip sky-region SIA queries.")
    args = parser.parse_args(argv)

    _check_installed_signatures()
    specs = list(_query_specs(args.obs_begin, args.obs_end))
    if args.skip_most:
        specs = [(name, query) for name, query in specs if not name.startswith("most_")]
    if args.skip_sia:
        specs = [(name, query) for name, query in specs if not name.startswith("sia_")]

    print(f"QUERY_WINDOW_UTC {args.obs_begin} .. {args.obs_end}")
    print(f"CACHE_DIR {args.cache_dir}")
    print(
        f"SIA_CONFIGURATION pole_radius_deg={POLE_RADIUS_DEG} "
        f"ordinary_radius_deg={ORDINARY_RADIUS_DEG} maxrec={SIA_MAXREC}"
    )
    if not args.skip_most:
        sample_payload = Most.query_object(
            catalog="spherex",
            input_mode="name_input",
            obj_name=f"{ASTEROIDS[0].naif_id}:NID",
            obs_begin=args.obs_begin,
            obs_end=args.obs_end,
            output_mode="Brief",
            get_query_payload=True,
        )
        print(f"MOST_SAMPLE_REQUEST_PAYLOAD {sample_payload!r}")
    if args.dry_run:
        print(f"DRY_RUN_QUERY_COUNT {len(specs)}")
        for name, _ in specs:
            print(f"PLANNED_QUERY {name}")
        print("NETWORK_REQUESTS 0")
        return 0

    failures = 0
    for name, query in specs:
        if not _run_one(name, query, args.cache_dir):
            failures += 1
    print(f"\nPROBE_COMPLETE queries={len(specs)} failures={failures} raw_results_dir={args.cache_dir}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
