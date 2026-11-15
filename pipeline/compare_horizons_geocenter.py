"""Compare MOST's Victoria coordinates with JPL Horizons at observer 500.

Run locally with Horizons connectivity::

    python -m pipeline.compare_horizons_geocenter \
      --table pipeline/cache/probe/most_20000012_Victoria_smaller-body-screen.ecsv \
      --output pipeline/cache/victoria_horizons_500.ecsv

MOST's image metadata provide start/end UTC times; this script uses their
midpoint, matching the midpoint convention in the MOST instructions. It queries
the geocentre (`location='500'`) using ICRF and writes both astrometric and
apparent RA/Dec residuals. The comparison can identify whether MOST is
consistent with observer 500; it does not independently validate the orbit,
because both products use Horizons ephemerides.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.config.paths import set_temp_cache
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from astropy.time import Time
from astroquery.jplhorizons import Horizons


def _separation_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    return float(SkyCoord(ra1 * u.deg, dec1 * u.deg).separation(SkyCoord(ra2 * u.deg, dec2 * u.deg)).arcsec)


def midpoint_epochs(table: Table) -> list[dict]:
    required = {"mjd_start", "mjd_end", "ra_obj", "dec_obj", "observationid"}
    missing = required.difference(table.colnames)
    if missing:
        raise ValueError(f"MOST table is missing columns: {sorted(missing)}")
    epochs = []
    for row_index, row in enumerate(table):
        mjd = (float(row["mjd_start"]) + float(row["mjd_end"])) / 2.0
        epoch = Time(mjd, format="mjd", scale="utc")
        epochs.append({"row_index": row_index, "mjd_midpoint": mjd, "epoch_utc": epoch.isot,
                       "observation_id": str(row["observationid"]),
                       "product_id": str(row["productid"]) if "productid" in table.colnames else "",
                       "most_ra_deg": float(row["ra_obj"]), "most_dec_deg": float(row["dec_obj"])})
    return epochs


def query_epochs(target: str, epochs: list[dict], *, chunk_size: int = 50) -> Table:
    outputs = []
    for start in range(0, len(epochs), chunk_size):
        chunk = epochs[start:start + chunk_size]
        strings = [item["epoch_utc"] for item in chunk]
        print(f"HORIZONS_QUERY observer=500 target={target!r} refsystem=ICRF refraction=False "
              f"time_scale=UTC n_epochs={len(strings)} first={strings[0]} last={strings[-1]}")
        query = Horizons(id=target, id_type="smallbody", location="500", epochs=strings)
        outputs.append(query.ephemerides(refsystem="ICRF", refraction=False, quantities="1,2"))
    return vstack(outputs, metadata_conflicts="silent") if len(outputs) > 1 else outputs[0]


def query_epochs_with_workspace_cache(target: str, epochs: list[dict], *, chunk_size: int = 50) -> Table:
    """Run Horizons with Astropy's response cache in the project workspace."""
    cache_dir = Path(__file__).resolve().parents[1] / "pipeline/cache/astroquery"
    with set_temp_cache(cache_dir):
        return query_epochs(target, epochs, chunk_size=chunk_size)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=Path("pipeline/cache/probe/most_20000012_Victoria_smaller-body-screen.ecsv"))
    parser.add_argument("--target", default="12", help="Horizons small-body identifier; default is asteroid number 12 (Victoria).")
    parser.add_argument("--output", type=Path, default=Path("pipeline/cache/victoria_horizons_500.ecsv"))
    parser.add_argument("--chunk-size", type=int, default=50)
    args = parser.parse_args(argv)
    most = Table.read(args.table, format="ascii.ecsv")
    epochs = midpoint_epochs(most)
    try:
        # Managed desktops may deny access to the user's default cache directory.
        horizons = query_epochs_with_workspace_cache(args.target, epochs, chunk_size=args.chunk_size)
    except Exception as exc:
        print(f"HORIZONS_QUERY_FAILED {type(exc).__name__}: {exc}")
        return 2
    if len(horizons) != len(epochs):
        print(f"HORIZONS_ROW_COUNT_MISMATCH expected={len(epochs)} got={len(horizons)}")
        return 3

    rows = []
    astrometric_residuals, apparent_residuals = [], []
    has_apparent = all(key in horizons.colnames for key in ("RA_app", "DEC_app"))
    for input_epoch, hrow in zip(epochs, horizons):
        ra, dec = float(hrow["RA"]), float(hrow["DEC"])
        astrometric = _separation_arcsec(input_epoch["most_ra_deg"], input_epoch["most_dec_deg"], ra, dec)
        astrometric_residuals.append(astrometric)
        apparent = None
        if has_apparent and np.isfinite(float(hrow["RA_app"])) and np.isfinite(float(hrow["DEC_app"])):
            apparent = _separation_arcsec(input_epoch["most_ra_deg"], input_epoch["most_dec_deg"],
                                          float(hrow["RA_app"]), float(hrow["DEC_app"]))
            apparent_residuals.append(apparent)
        rows.append({**input_epoch, "horizons_500_ra_deg": ra, "horizons_500_dec_deg": dec,
                     "horizons_500_ra_app_deg": float(hrow["RA_app"]) if has_apparent else None,
                     "horizons_500_dec_app_deg": float(hrow["DEC_app"]) if has_apparent else None,
                     "most_minus_horizons_500_astrometric_arcsec": astrometric,
                     "most_minus_horizons_500_apparent_arcsec": apparent,
                     "observer": "500 (geocentre)", "reference_frame": "ICRF",
                     "epoch_scale": "UTC", "light_time_handling": "Horizons observer-table quantities 1 and 2; service defaults"})
    result = Table(rows=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.write(args.output, format="ascii.ecsv", overwrite=True)
    def summary(values):
        data = np.asarray(values, float)
        return {"n": int(len(data)), "median_arcsec": float(np.nanmedian(data)),
                "p95_arcsec": float(np.nanpercentile(data, 95)), "max_arcsec": float(np.nanmax(data))}
    print("MOST_MINUS_HORIZONS_500_ASTROMETRIC " + str(summary(astrometric_residuals)))
    if apparent_residuals:
        print("MOST_MINUS_HORIZONS_500_APPARENT " + str(summary(apparent_residuals)))
    print(f"HORIZONS_COMPARISON_COMPLETE rows={len(result)} output={args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
