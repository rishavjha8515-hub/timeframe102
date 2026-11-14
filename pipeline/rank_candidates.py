"""Rank MOST asteroid coverage tables using archive cadence and ephemerides.

The reported motion is the change in MOST's predicted coordinates within each
UTC observing night. It is not a measured SPHEREx centroid or motion detection.

Run from the project root::

    python -m pipeline.rank_candidates --input-dir pipeline/cache/probe

Timestamps within 0.864 seconds are treated as duplicate product rows for the
coordinate-step calculation only. Rates use the actual MJD difference between
the remaining timestamps.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table

DUPLICATE_EPOCH_DECIMALS = 5  # 1e-5 day = 0.864 seconds


@dataclass(frozen=True)
class CandidateSummary:
    target: str
    source_file: str
    product_rows: int
    distinct_observation_ids: int
    distinct_utc_nights: int
    baseline_days: float
    median_gap_days: float | None
    max_gap_days: int | None
    vmag_min: float | None
    vmag_max: float | None
    intranight_coordinate_pairs: int
    median_step_arcsec: float | None
    median_rate_arcsec_per_day: float | None


def _values(column: Any, dtype: Any = float) -> np.ndarray:
    values = np.ma.asarray(column).compressed()
    return np.asarray(values, dtype=dtype)


def _night_labels(table: Table) -> np.ndarray:
    if "start_date" in table.colnames:
        return np.asarray([str(value)[:10] for value in table["start_date"]])
    if "mjd_start" not in table.colnames:
        raise ValueError("MOST table requires start_date or mjd_start")
    return np.floor(_values(table["mjd_start"])).astype(int).astype(str)


def summarize_table(table: Table, *, target: str, source_file: str = "") -> CandidateSummary:
    required = {"observationid", "mjd_start", "ra_obj", "dec_obj", "vmag"}
    missing = required - set(table.colnames)
    if missing:
        raise ValueError(f"MOST table is missing required columns: {sorted(missing)}")
    mjd = np.asarray(table["mjd_start"], dtype=float)
    valid = np.isfinite(mjd)
    if not np.any(valid):
        raise ValueError("MOST table has no valid mjd_start values")

    nights_by_row = _night_labels(table)
    night_dates = sorted(set(nights_by_row[valid]))
    gaps = np.diff(np.asarray(night_dates, dtype="datetime64[D]")).astype(int) if (
        len(night_dates) and not night_dates[0].isdigit()
    ) else np.diff(np.asarray(night_dates, dtype=int))

    obsids = {str(value) for value in table["observationid"] if str(value) not in {"", "masked", "None"}}
    vmag = _values(table["vmag"])
    vmag = vmag[np.isfinite(vmag)]

    ra = np.asarray(table["ra_obj"], dtype=float)
    dec = np.asarray(table["dec_obj"], dtype=float)
    pairs: list[tuple[float, float]] = []
    for night in night_dates:
        indices = np.flatnonzero(valid & (nights_by_row == night))
        indices = indices[np.argsort(mjd[indices])]
        epoch_key = np.round(mjd[indices], DUPLICATE_EPOCH_DECIMALS)
        keep: list[int] = []
        previous: float | None = None
        for index, key in zip(indices, epoch_key):
            if previous is None or key != previous:
                keep.append(int(index))
                previous = float(key)
        if len(keep) < 2:
            continue
        dt = np.diff(mjd[keep])
        coords = SkyCoord(ra[keep] * u.deg, dec[keep] * u.deg, frame="icrs")
        steps = coords[:-1].separation(coords[1:]).arcsec
        for step, delta_days in zip(steps, dt):
            if delta_days > 0 and np.isfinite(step):
                pairs.append((float(step), float(step / delta_days)))

    pair_array = np.asarray(pairs, dtype=float).reshape((-1, 2)) if pairs else np.empty((0, 2))
    return CandidateSummary(
        target=target,
        source_file=source_file,
        product_rows=len(table),
        distinct_observation_ids=len(obsids),
        distinct_utc_nights=len(night_dates),
        baseline_days=float(np.max(mjd[valid]) - np.min(mjd[valid])),
        median_gap_days=float(np.median(gaps)) if len(gaps) else None,
        max_gap_days=int(np.max(gaps)) if len(gaps) else None,
        vmag_min=float(np.min(vmag)) if len(vmag) else None,
        vmag_max=float(np.max(vmag)) if len(vmag) else None,
        intranight_coordinate_pairs=len(pair_array),
        median_step_arcsec=float(np.median(pair_array[:, 0])) if len(pair_array) else None,
        median_rate_arcsec_per_day=float(np.median(pair_array[:, 1])) if len(pair_array) else None,
    )


def summarize_directory(input_dir: Path) -> list[CandidateSummary]:
    summaries = []
    for path in sorted(input_dir.glob("most_*.ecsv")):
        table = Table.read(path, format="ascii.ecsv")
        # probe file names are most_<NAIF ID>_<name>_<screening group>.ecsv
        parts = path.stem.split("_")
        target = parts[2] if len(parts) > 2 else path.stem
        summaries.append(summarize_table(table, target=target, source_file=str(path)))
    return sorted(
        summaries,
        key=lambda row: (
            -row.distinct_observation_ids,
            -row.distinct_utc_nights,
            -row.baseline_days,
            row.max_gap_days if row.max_gap_days is not None else float("inf"),
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("pipeline/cache/probe"))
    parser.add_argument("--json", type=Path, help="Optional path for machine-readable ranking output.")
    args = parser.parse_args(argv)
    summaries = summarize_directory(args.input_dir)
    if not summaries:
        print(f"No MOST tables found under {args.input_dir}", file=sys.stderr)
        return 1
    writer = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
    writer.writerow(CandidateSummary.__dataclass_fields__.keys())
    for summary in summaries:
        writer.writerow([getattr(summary, key) for key in CandidateSummary.__dataclass_fields__])
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps([asdict(row) for row in summaries], indent=2), encoding="utf-8")
        print(f"JSON_RESULT {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
