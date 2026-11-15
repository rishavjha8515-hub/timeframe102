"""Contract tests for TIMEFRAME/102 case records."""

import json
import unittest
from pathlib import Path
from pathlib import PurePosixPath

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schemas" / "case.schema.json"
INDEX_SCHEMA_PATH = ROOT / "schemas" / "case-index.schema.json"
CASES_DIR = ROOT / "cases"
INDEX_PATH = CASES_DIR / "index.json"


def load_schema():
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


class CaseSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = load_schema()
        cls.index_schema = json.loads(INDEX_SCHEMA_PATH.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(cls.schema)
        Draft202012Validator.check_schema(cls.index_schema)
        cls.validator = Draft202012Validator(cls.schema, format_checker=FormatChecker())
        cls.index_validator = Draft202012Validator(cls.index_schema, format_checker=FormatChecker())

    def test_schema_is_versioned(self):
        self.assertEqual(self.schema["properties"]["schema_version"]["const"], "1.6.0")

    def test_index_is_versioned_and_valid(self):
        index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        self.index_validator.validate(index)
        self.assertEqual(index["schema_version"], "1.3.0")

    def test_verdict_vocabulary_is_closed(self):
        labels = self.schema["properties"]["verdict"]["properties"]["label"]["enum"]
        self.assertEqual(
            labels,
            ["moving", "no significant motion detected", "insufficient evidence"],
        )

    def test_all_exported_cases_validate(self):
        case_files = sorted(path for path in CASES_DIR.glob("*.json") if path.name != "index.json")
        case_files += [ROOT / "outputs/control-star/case.json", ROOT / "outputs/control-star/case_bundle.json"]
        for case_path in case_files:
            with self.subTest(case=case_path.name):
                instance = json.loads(case_path.read_text(encoding="utf-8"))
                self.validator.validate(instance)
                analysis = instance.get("spectral_analysis")
                if isinstance(analysis, dict) and analysis["dof"] is not None:
                    self.assertEqual(analysis["dof"], analysis["n_observations"] - analysis["n_parameters"])
                    self.assertAlmostEqual(analysis["reduced_chi2"], analysis["chi2"] / analysis["dof"])

    def test_control_bundle_is_self_contained_and_under_hosting_limit(self):
        bundle = ROOT / "outputs/control-star/case_bundle.json"
        self.assertLess(bundle.stat().st_size, 2 * 1024 * 1024)
        case = json.loads(bundle.read_text(encoding="utf-8"))
        self.assertFalse(case["is_synthetic"])
        self.assertTrue(all(o["image"] is None or o["image"]["path"].startswith("data:image/png;base64,")
                            for o in case["observations"]))

    def test_control_analysis_keeps_unavailable_floor_and_skipped_test_explicit(self):
        case = json.loads((ROOT / "outputs/control-star/case_bundle.json").read_text(encoding="utf-8"))
        null_paths = json.loads((ROOT / "outputs/control-star/quality_report.json").read_text(encoding="utf-8"))["null_fields"]
        self.assertEqual(set(null_paths), set(case["null_field_reasons"]))
        self.assertTrue(all(reason.strip() for reason in case["null_field_reasons"].values()))
        comparison = case["comparison_star_analysis"]
        self.assertEqual(comparison["n_stars_used"], 0)
        self.assertEqual(comparison["photometric_floor"]["status"], "unavailable")
        self.assertIsNone(comparison["photometric_floor"]["floor_jy"])
        self.assertTrue(comparison["photometric_floor"]["reason"])
        test = case["spectral_analysis"]["constant_flux_test"]
        self.assertEqual(test["status"], "skipped")
        self.assertIsNone(test["p_value"])
        self.assertTrue(test["reason"])
        self.assertEqual(case["dr1_calibration_check"]["status"], "pass")
        dr1 = case["dr1_calibration_check"]
        self.assertEqual(dr1["comparison"]["direct_sia_http_status"], 200)
        self.assertEqual(dr1["comparison"]["direct_sia_row_count"], 467)
        self.assertEqual(dr1["comparison"]["case_observation_detector_matches"], 0)
        self.assertIn("no tolerance-based acceptance threshold was applied", dr1["reason"])
        cloud_attempt = dr1["comparison"]["cloud_calibration_attempt"]
        self.assertEqual(cloud_attempt["status"], "compared")
        self.assertEqual(cloud_attempt["calibration_product"]["key"],
                         "dr1/spectral_wcs/cal-swcs-v5-2026-191/1/spectral_wcs_D1_spx_cal-swcs-v5-2026-191.fits")
        self.assertTrue(cloud_attempt["calibration_provenance"]["matches_cached_mef_source"])
        self.assertAlmostEqual(
            cloud_attempt["wavelength_comparison_um"]["standalone_minus_epsf_cwave"],
            0.0004682155901392093,
        )
        flux = case["catalog_flux_comparison"]["comparisons"]
        self.assertEqual([row["flux_ratio_spherex_to_catalog"] for row in flux],
                         [0.9489875344110771, 0.9467459958984408,
                          1.0822292439202874, 0.955228966847731])

    def test_case_index_entries_resolve_to_matching_records(self):
        index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        for entry in index["cases"]:
            with self.subTest(case_id=entry["case_id"]):
                case_path = (INDEX_PATH.parent / entry["case_path"]).resolve()
                self.assertTrue(case_path.is_relative_to(INDEX_PATH.parent.resolve()))
                case = json.loads(case_path.read_text(encoding="utf-8"))
                self.assertEqual(entry["case_id"], case["case_id"])
                self.assertEqual(entry["kind"], case["target"]["kind"])
                self.assertEqual(entry["is_synthetic"], case["is_synthetic"])

    def test_fixture_is_explicitly_synthetic_and_image_paths_are_case_relative(self):
        index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        for entry in index["cases"]:
            case_path = INDEX_PATH.parent / entry["case_path"]
            case = json.loads(case_path.read_text(encoding="utf-8"))
            if case["case_id"] == "synthetic-ui-demo":
                self.assertIs(case["is_synthetic"], True)
                self.assertIs(entry["is_synthetic"], True)
                for observation in case["observations"]:
                    image = observation["image"]
                    if image is None:
                        continue
                    image_path = PurePosixPath(image["path"])
                    self.assertFalse(image_path.is_absolute())
                    self.assertTrue((case_path.parent / Path(*image_path.parts)).is_file())

    def test_schema_rejects_a_case_without_provenance(self):
        invalid = {
            "schema_version": "1.6.0",
            "case_id": "fixture",
            "is_synthetic": True,
            "time_definition": "midpoint",
            "target": {"name": "synthetic fixture", "kind": "other"},
            "observations": [],
            "verdict": {
                "label": "insufficient evidence",
                "evidence": {
                    "summary": "Synthetic schema fixture only",
                    "n_observations": 0,
                    "n_usable": 0,
                    "baseline_days": None,
                },
                "threshold_policy": {
                    "id": "test-policy",
                    "minimum_usable_epochs": 3,
                    "moving_significance_sigma": 5,
                    "minimum_displacement_uncertainty_ratio": 3,
                    "no_motion_confidence_level": 0.95,
                    "no_motion_max_displacement_arcsec": 6.15,
                    "quality_flags_exclude": ["TRANSIENT", "SUR_ERROR", "NONFUNC", "OVERFLOW", "SATURATED"],
                    "context_only_flags": ["HALO", "PERSIST", "DICHROIC"],
                    "mad_residual_threshold_sigma": 5.0,
                    "spectral_fit_model": "degree-2 robust log spectrum",
                    "wavelength_bin_width_um": 0.1,
                    "minimum_comparison_stars": 20,
                    "minimum_floor_epochs": 3,
                    "comparison_star_criteria": {
                        "catalogs": ["Gaia DR3", "2MASS PSC", "AllWISE"],
                        "gaia_g_mag_range": [11.12, 14.12],
                        "two_mass_match_max_arcsec": 1.0,
                        "two_mass_ph_qual": "AAA",
                        "allwise_match_max_arcsec": 1.0,
                        "allwise_ph_qual_w1_w2": "AA",
                        "allwise_cc_flags_w1_w2": "00",
                        "allwise_ext_flg": 0,
                        "gaia_isolation_radius_arcsec": 60.0,
                        "maximum_total_proper_motion_mas_yr": 10.0,
                        "maximum_ruwe": 1.4,
                        "exclude_phot_variable_flag": True,
                        "allowed_phot_variable_flag_values": ["CONSTANT", "NOT_AVAILABLE"],
                        "minimum_cutout_edge_margin_arcsec": 60.0,
                        "exclude_hard_source_flags": ["TRANSIENT", "SUR_ERROR", "NONFUNC", "OVERFLOW", "SATURATED"],
                        "minimum_stars_per_frame": 20,
                        "maximum_stars_per_frame": 50,
                        "aperture_radius_px": 3.0,
                        "annulus_inner_px": 6.0,
                        "annulus_outer_px": 10.0,
                        "photometric_floor_method": "predeclared method",
                        "astrometric_threshold_method": "predeclared method",
                    },
                },
            },
            "confirmed_versus_assumed": {"confirmed": [], "assumed": [], "unverified": []},
            "provenance": {
                "products": [],
                "software": {"pipeline_version": "test", "environment_lock": "test"},
            },
        }
        del invalid["provenance"]
        with self.assertRaises(ValidationError):
            self.validator.validate(invalid)

    def test_format_checker_rejects_invalid_date_time(self):
        validator = Draft202012Validator(
            {"type": "string", "format": "date-time"},
            format_checker=FormatChecker(),
        )
        with self.assertRaises(ValidationError):
            validator.validate("not-a-date")


if __name__ == "__main__":
    unittest.main()
