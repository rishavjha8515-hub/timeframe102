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
        self.assertEqual(self.schema["properties"]["schema_version"]["const"], "1.3.0")

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
        for case_path in case_files:
            with self.subTest(case=case_path.name):
                instance = json.loads(case_path.read_text(encoding="utf-8"))
                self.validator.validate(instance)

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
            "schema_version": "1.3.0",
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
