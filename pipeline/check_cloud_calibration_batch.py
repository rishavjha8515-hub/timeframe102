"""Run the predeclared calibration-map check for selected control-star frames.

The frame list must already be frozen in outputs/control-star/case.json under
calibration_crosscheck_plan. Each frame is isolated so one failure does not
discard the other actual results. This script never edits the case or frozen
bundle; it writes a separate JSON report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .check_cloud_calibration import compare_one

ROOT = Path(__file__).resolve().parents[1]
CASE_PATH = ROOT / "outputs/control-star/case.json"
OUTPUT_PATH = ROOT / "outputs/control-star/calibration_crosscheck_results.json"


def run_batch(compare: Callable[..., dict[str, Any]] = compare_one) -> dict[str, Any]:
    case = json.loads(CASE_PATH.read_text(encoding="utf-8"))
    plan = case["calibration_crosscheck_plan"]
    frames = plan["frames"]
    if plan["status"] not in ("planned", "partial"):
        raise ValueError(f"Calibration plan is not runnable: status={plan['status']}")
    if len(frames) != plan["selected_frame_count"]:
        raise ValueError("Calibration frame count differs from the predeclared plan")

    results = []
    for item in frames:
        key = (item["observation_id"], int(item["detector"]))
        try:
            record = compare(release="dr1", detector=key[1], observation_id=key[0])
            record = {"selection_basis": item["selection_basis"], **record}
            record["status"] = "compared" if record.get("status") == "compared" else record.get("status", "failed")
            if record["status"] == "compared":
                comparison = record["wavelength_comparison_um"]
                comparison["standalone_minus_epsf_cwave_nm"] = comparison["standalone_minus_epsf_cwave"] * 1000.0
                comparison["standalone_minus_epsf_cband_nm"] = comparison["standalone_minus_epsf_cband"] * 1000.0
                epsf_width = comparison["epsf_cband"]
                comparison["absolute_cwave_difference_fraction_channel_width"] = (
                    abs(comparison["standalone_minus_epsf_cwave"]) / epsf_width if epsf_width else None
                )
                comparison["absolute_cband_difference_fraction_channel_width"] = (
                    abs(comparison["standalone_minus_epsf_cband"]) / epsf_width if epsf_width else None
                )
                provenance = record["calibration_provenance"]
                expected_source = provenance["cached_mef_wcs_wave_source_file"]
                record["calibration_version_status"] = (
                    "confirmed" if provenance["matches_cached_mef_source"] and expected_source
                    else "mismatch" if expected_source and provenance["standalone_map_source_file"]
                    else "unverified"
                )
                record["version_evidence"] = {
                    "cutout_fits_primary.VERSION": record.get("cutout_fits_primary_version"),
                    "cutout.WCS-WAVE HISTORY Calibration source file": expected_source,
                    "standalone.WCS-WAVE HISTORY Calibration source file": provenance["standalone_map_source_file"],
                    "standalone.CWAVE DETECTOR": provenance["standalone_detector"],
                    "standalone_product_key_version": provenance["standalone_product_release_key"],
                    "IRSA_response.product_id": (record.get("irsa_response_product") or {}).get("product_id"),
                    "IRSA_response.release": (record.get("irsa_response_product") or {}).get("release"),
                    "IRSA_response_has_cal-swcs-v5-2026-191_field": False,
                    "IRSA_response_calibration_version_status": "unverified; SIA response has L2 product ID/release but no standalone calibration-map version field",
                }
        except Exception as exc:
            record = {"status": "failed", "selection_basis": item["selection_basis"],
                      "observation_id": key[0], "detector": key[1],
                      "error": f"{type(exc).__name__}: {exc}"}
        results.append(record)

    compared = [item for item in results if item.get("status") == "compared"]
    cwave_abs = [abs(item["wavelength_comparison_um"]["standalone_minus_epsf_cwave"])
                 for item in compared]
    cband_abs = [abs(item["wavelength_comparison_um"]["standalone_minus_epsf_cband"])
                 for item in compared]
    widths = [item["wavelength_comparison_um"]["epsf_cband"] for item in compared
              if item["wavelength_comparison_um"].get("epsf_cband")]
    result = {
        "status": "complete" if len(compared) == len(frames) else "partial" if compared else "failed",
        "plan_id": "control-star-1.7.0-followup-01",
        "selection_rule": plan["selection_rule"],
        "selected_frame_count": len(frames),
        "compared_frame_count": len(compared),
        "failed_frame_count": len(frames) - len(compared),
        "frames": results,
        "summary": {
            "max_absolute_cwave_difference_nm": max(cwave_abs) * 1000.0 if cwave_abs else None,
            "max_absolute_cband_difference_nm": max(cband_abs) * 1000.0 if cband_abs else None,
            "max_absolute_cwave_difference_fraction_channel_width": max(
                (abs(item["wavelength_comparison_um"]["standalone_minus_epsf_cwave"]) /
                 item["wavelength_comparison_um"]["epsf_cband"] for item in compared
                 if item["wavelength_comparison_um"].get("epsf_cband")), default=None),
            "max_absolute_cband_difference_fraction_channel_width": max(
                (abs(item["wavelength_comparison_um"]["standalone_minus_epsf_cband"]) /
                 item["wavelength_comparison_um"]["epsf_cband"] for item in compared
                 if item["wavelength_comparison_um"].get("epsf_cband")), default=None),
            "all_compared_cutout_map_versions_confirmed": bool(compared) and all(
                item.get("calibration_version_status") == "confirmed" for item in compared),
            "IRSA_response_calibration_version_status": "unverified; SIA response exposes no standalone calibration-map version field",
        },
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "selected": len(frames),
                      "compared": len(compared), "failed": result["failed_frame_count"],
                      "summary": result["summary"], "output": (OUTPUT_PATH.relative_to(ROOT).as_posix()
                                                                  if OUTPUT_PATH.is_relative_to(ROOT)
                                                                  else OUTPUT_PATH.name)},
                     separators=(",", ":")))
    return result


if __name__ == "__main__":
    run_batch()
