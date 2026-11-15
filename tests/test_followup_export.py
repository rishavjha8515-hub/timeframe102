import json
import tempfile
import unittest
from pathlib import Path

from pipeline.export_control_star_case import _attach_followup_analysis


class FollowupExportTests(unittest.TestCase):
    def test_adds_calibration_and_simulation_as_schema_18_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            plan = {
                "calibration_crosscheck": {
                    "status": "planned", "selection_rule": "fixed before comparison",
                    "selected_frame_count": 1, "frames": [{"observation_id": "o1", "detector": 1}],
                    "version_match_rule": "source filename and detector must match",
                },
                "injection_test_plan": {
                    "status": "planned", "simulated": True, "frames": "one frame",
                    "flux_change_fractions": [0.1], "position_shift_arcsec": {"magnitudes": [1.0]},
                    "trial_count_per_level": 1, "injection_method": "in memory",
                    "detection_criterion": "fixed", "systematics_limit": "not a floor",
                },
            }
            calibration = {"status": "partial", "selected_frame_count": 1,
                           "compared_frame_count": 0, "failed_frame_count": 1,
                           "frames": [{"status": "failed", "error": "offline"}], "summary": {}}
            injection = {"status": "partial", "simulated": True, "failed_frames": [{"reason": "missing"}],
                         "flux_change_results": [], "position_shift_results": []}
            (out / "case_analysis_plan.json").write_text(json.dumps(plan), encoding="utf-8")
            (out / "calibration_crosscheck_results.json").write_text(json.dumps(calibration), encoding="utf-8")
            (out / "injection_test_results.json").write_text(json.dumps(injection), encoding="utf-8")
            case = {"schema_version": "1.7.0", "verdict": {"label": "insufficient evidence"},
                    "confirmed_versus_assumed": {"confirmed": [], "assumed": [], "unverified": []}}

            result = _attach_followup_analysis(case, out)

        self.assertEqual(result["schema_version"], "1.8.0")
        self.assertEqual(result["calibration_crosscheck_plan"]["status"], "partial")
        self.assertEqual(result["calibration_crosscheck_results"]["failed_frame_count"], 1)
        self.assertEqual(result["injection_test_plan"]["status"], "partial")
        self.assertIs(result["injection_test"]["simulated"], True)
        self.assertEqual(result["verdict"]["label"], "insufficient evidence")
        self.assertTrue(any("not counted as non-recoveries" in x
                            for x in result["confirmed_versus_assumed"]["unverified"]))


if __name__ == "__main__":
    unittest.main()
