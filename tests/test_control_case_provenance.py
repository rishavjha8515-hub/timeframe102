import unittest
import hashlib
import json
import re
from pathlib import Path

from astropy.io import fits

from pipeline.refresh_control_case_provenance import validate_binding


class ControlCaseProvenanceTests(unittest.TestCase):
    def test_rejects_detector_mismatch_between_case_and_url(self):
        observation = {"observation_id": "obs-1", "detector": 4}
        url = "https://irsa.ipac.caltech.edu/ibe/data/5/level2_obsD5_spx_l2b.fits"
        with self.assertRaisesRegex(ValueError, "detector mismatch"):
            validate_binding(observation, url, "obs-1", 4, "a" * 64, "a" * 64)

    def test_rejects_detector_mismatch_between_case_and_file(self):
        observation = {"observation_id": "obs-1", "detector": 4}
        url = "https://irsa.ipac.caltech.edu/ibe/data/4/level2_obsD4_spx_l2b.fits"
        with self.assertRaisesRegex(ValueError, "detector mismatch"):
            validate_binding(observation, url, "obs-1", 5, "a" * 64, "a" * 64)

    def test_rejects_file_hash_mismatch(self):
        observation = {"observation_id": "obs-1", "detector": 4}
        url = "https://irsa.ipac.caltech.edu/ibe/data/4/level2_obsD4_spx_l2b.fits"
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            validate_binding(observation, url, "obs-1", 4, "a" * 64, "b" * 64)

    def test_exported_case_urls_detectors_and_hashes_match_cached_cutouts(self):
        root = Path(__file__).resolve().parents[1]
        case = json.loads((root / "outputs/control-star/case.json").read_text(encoding="utf-8"))
        cache = root / "pipeline/cache/star-candidates"
        for observation, product in zip(case["observations"], case["provenance"]["products"]):
            detector = int(re.search(r"D([1-6])_spx_l2b", product["access_url"]).group(1))
            candidates = list(cache.glob(f"{case['target']['catalog_id']}_*_{observation['observation_id']}.fits"))
            self.assertEqual(len(candidates), 1)
            path = candidates[0]
            with fits.open(path) as hdul:
                image = hdul["IMAGE"].header
                validate_binding(observation, product["access_url"], str(image["OBSID"]),
                                 int(image["DETECTOR"]), product["sha256"],
                                 hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(detector, observation["detector"])


if __name__ == "__main__":
    unittest.main()
