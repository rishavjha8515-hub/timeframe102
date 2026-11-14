"""Reproject cached SPHEREx products and inspect predicted source positions offline.

Example::

    python -m pipeline.process_cutouts \
      --manifest pipeline/cache/fits-inspection/inspection_manifest.json \
      --output-dir outputs/processed-cutouts
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

from astropy.time import Time

from .cutout_processing import load_inspection_manifest, process_fits_product


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("pipeline/cache/fits-inspection/inspection_manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/processed-cutouts"))
    parser.add_argument("--size-px", type=int, default=128)
    parser.add_argument("--pixel-scale-arcsec", type=float, default=6.0)
    args = parser.parse_args(argv)
    manifest = load_inspection_manifest(args.manifest)
    results = []
    failures = 0
    for item in manifest:
        fits_path = Path(item["path"])
        if not fits_path.is_absolute() and not fits_path.exists():
            fits_path = Path.cwd() / fits_path
        row = item["most_row"]
        mjd_midpoint = (float(row["mjd_start"]) + float(row["mjd_end"])) / 2
        source_table = Path(str(item.get("source_table", ""))).stem
        target_match = re.search(r"most_\d+_([^_]+)", source_table, re.IGNORECASE)
        target_name = target_match.group(1) if target_match else str(row.get("target_name", "unknown"))
        try:
            result = process_fits_product(
                fits_path,
                output_dir=args.output_dir,
                ra_deg=float(row["ra_obj"]),
                dec_deg=float(row["dec_obj"]),
                observation_id=str(row["observationid"]),
                target_name=target_name,
                detector=int(str(row.get("productid", "D0")).lstrip("D")) or None,
                time_utc=Time(mjd_midpoint, format="mjd", scale="utc").isot,
                mjd_midpoint=mjd_midpoint,
                shape=(args.size_px, args.size_px),
                pixel_scale_arcsec=args.pixel_scale_arcsec,
            )
            results.append(result)
            print(f"PROCESSED obsid={result['observation_id']} target={result['target']} "
                  f"association={result['association']} predicted_xy={result['predicted_xy']} "
                  f"source_xy={result['measured_source_xy']} peak_snr={result['peak_snr']} "
                  f"wave_epsf={result['epsf_cwave_um']} wave_wcs={result['wcs_wave_um']} "
                  f"wave_diff={result['wavelength_uncertainty_um']} overflow={result['overflow_within_2_pixel_radius_box']} "
                  f"reference_candidates={len(result['reference_source_candidates_unconfirmed'])}")
        except Exception as exc:
            failures += 1
            print(f"PROCESS_FAILED path={fits_path} {type(exc).__name__}: {exc}")
    report = args.output_dir / "processing_report.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(_json_safe(results), indent=2, allow_nan=False), encoding="utf-8")
    print(f"PROCESSING_COMPLETE products={len(results)} failures={failures} report={report}")
    return 0 if failures == 0 and results else 1


if __name__ == "__main__":
    sys.exit(main())
