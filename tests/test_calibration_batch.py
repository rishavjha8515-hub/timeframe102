import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.check_cloud_calibration_batch import run_batch


class CalibrationBatchTests(unittest.TestCase):
    def test_runs_every_predeclared_frame_and_summarizes_units_and_version_evidence(self):
        frame = {"selection_basis": "earliest_usable_utc_for_detector", "observation_id": "obs1",
                 "detector": 1, "time_utc": "2025-01-01T00:00:00Z", "effective_wavelength_um": 1.0}
        case = {"calibration_crosscheck_plan": {"status": "planned", "selected_frame_count": 1,
                "selection_rule": "fixed", "version_match_rule": "match", "frames": [frame]}}
        comparison = {
            "status": "compared", "release": "dr1", "observation_id": "obs1", "detector": 1,
            "time_utc": frame["time_utc"], "cutout_fits_primary_version": "7.0.5",
            "irsa_response_product": {"product_id": "product.fits", "release": "dr1"},
            "calibration_provenance": {
                "cached_mef_wcs_wave_source_file": "spectral_wcs_1_v2.2.fits",
                "standalone_map_source_file": "spectral_wcs_1_v2.2.fits",
                "standalone_detector": "1", "matches_cached_mef_source": True,
                "standalone_product_release_key": "cal-swcs-v5-2026-191"},
            "wavelength_comparison_um": {
                "epsf_cwave": 1.0, "epsf_cband": 0.02, "mef_wcs_wave": 1.01,
                "standalone_cwave": 1.001, "standalone_cband": 0.021,
                "standalone_minus_epsf_cwave": 0.001,
                "standalone_minus_mef_wcs_wave": -0.009,
                "standalone_minus_epsf_cband": 0.001},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_path, output_path = root / "case.json", root / "result.json"
            case_path.write_text(json.dumps(case), encoding="utf-8")
            with patch("pipeline.check_cloud_calibration_batch.CASE_PATH", case_path), \
                 patch("pipeline.check_cloud_calibration_batch.OUTPUT_PATH", output_path):
                result = run_batch(compare=lambda **kwargs: comparison.copy())
            self.assertEqual(result["status"], "complete")
            item = result["frames"][0]
            self.assertAlmostEqual(item["wavelength_comparison_um"]["standalone_minus_epsf_cwave_nm"], 1.0)
            self.assertAlmostEqual(item["wavelength_comparison_um"]["absolute_cwave_difference_fraction_channel_width"], .05)
            self.assertEqual(item["calibration_version_status"], "confirmed")
            self.assertEqual(result["summary"]["IRSA_response_calibration_version_status"],
                             "unverified; SIA response exposes no standalone calibration-map version field")
            self.assertTrue(output_path.is_file())

    def test_continues_after_one_frame_error(self):
        frames = [{"selection_basis": "a", "observation_id": "bad", "detector": 1},
                  {"selection_basis": "b", "observation_id": "good", "detector": 1}]
        case = {"calibration_crosscheck_plan": {"status": "planned", "selected_frame_count": 2,
                "selection_rule": "fixed", "version_match_rule": "match", "frames": frames}}
        def compare(**kwargs):
            if kwargs["observation_id"] == "bad":
                raise OSError("network down")
            return {
                "status": "compared", "observation_id": "good", "detector": 1,
                "wavelength_comparison_um": {"epsf_cwave": 1.0, "epsf_cband": 0.02,
                    "mef_wcs_wave": 1.0, "standalone_cwave": 1.0, "standalone_cband": 0.02,
                    "standalone_minus_epsf_cwave": 0.0, "standalone_minus_mef_wcs_wave": 0.0,
                    "standalone_minus_epsf_cband": 0.0},
                "calibration_provenance": {"cached_mef_wcs_wave_source_file": "spectral_wcs_1_v2.2.fits",
                    "standalone_map_source_file": "spectral_wcs_1_v2.2.fits", "standalone_detector": "1",
                    "matches_cached_mef_source": True, "standalone_product_release_key": "version"},
                "irsa_response_product": None, "cutout_fits_primary_version": "7.0.5",
            }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cp, op = root / "case.json", root / "result.json"
            cp.write_text(json.dumps(case), encoding="utf-8")
            with patch("pipeline.check_cloud_calibration_batch.CASE_PATH", cp), \
                 patch("pipeline.check_cloud_calibration_batch.OUTPUT_PATH", op):
                result = run_batch(compare=compare)
        self.assertEqual(result["status"], "partial")
        self.assertEqual([row["status"] for row in result["frames"]], ["failed", "compared"])


if __name__ == "__main__":
    unittest.main()
