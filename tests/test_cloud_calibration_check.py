import tempfile
import unittest
from pathlib import Path
import hashlib
import json
from unittest.mock import patch

import numpy as np

from pipeline.check_cloud_calibration import (
    _download_map, _download_or_reuse_map, calibration_source_from_history, sample_calibration_map,
)
from pipeline.export_control_star_case import _merge_cloud_calibration_attempt


class CloudCalibrationCheckTests(unittest.TestCase):
    def test_bilinear_sampling_uses_one_based_detector_coordinates(self):
        image = np.array([[10.0, 20.0], [30.0, 40.0]])
        self.assertEqual(sample_calibration_map(image, 1.0, 1.0), 10.0)
        self.assertEqual(sample_calibration_map(image, 2.0, 2.0), 40.0)
        self.assertEqual(sample_calibration_map(image, 1.5, 1.5), 25.0)

    def test_sampling_rejects_out_of_bounds_pixels(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            sample_calibration_map(np.ones((2, 2)), 2.1, 1.0)

    def test_calibration_source_is_extracted_from_fits_history(self):
        self.assertEqual(
            calibration_source_from_history([
                "Headers standardized by SSDC",
                "Calibration source file: spectral_wcs_1_v2.2.fits",
            ]),
            "spectral_wcs_1_v2.2.fits",
        )
        self.assertIsNone(calibration_source_from_history(["no source card"]))

    def test_download_checks_size_hash_and_etag(self):
        class Response:
            status_code = 200
            headers = {"Content-Length": "4", "ETag": '"abc"'}

            def iter_content(self, chunk_size):
                yield b"fits"

            def close(self):
                pass

        class Session:
            def get(self, url, stream, timeout):
                return Response()

        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "map.fits"
            result = _download_map("dr1/spectral_wcs/map.fits", destination, session=Session())
            self.assertEqual(destination.read_bytes(), b"fits")
            self.assertEqual(result["bytes_downloaded"], 4)
            self.assertEqual(result["etag"], "abc")
            self.assertEqual(result["sha256"], "658ace74701b4d47c76b8caba284b3c55c9c68eb3afe48a6abf97df4dfb15bec")

    def test_reuses_cached_calibration_only_when_prior_digest_matches(self):
        class NoNetwork:
            def get(self, *args, **kwargs):
                raise AssertionError("verified cache should avoid network")

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache = root / "cloud"
            cache.mkdir()
            target = cache / "map.fits"
            payload = b"fits"
            target.write_bytes(payload)
            key = "dr1/spectral_wcs/map.fits"
            previous = {"status": "compared", "calibration_product": {
                "key": key, "url": "https://bucket/map.fits", "http_status": 200,
                "content_length": 4, "bytes_downloaded": 4,
                "sha256": hashlib.sha256(payload).hexdigest(), "etag": "etag-1",
            }}
            (cache / "cloud_calibration_comparison.json").write_text(json.dumps(previous), encoding="utf-8")
            with patch("pipeline.check_cloud_calibration.CACHE", cache), \
                 patch("pipeline.check_cloud_calibration.EXPECTED_MAP_BYTES", 4):
                result = _download_or_reuse_map(key, target, session=NoNetwork())
            self.assertTrue(result["cached"])
            self.assertEqual(result["http_status"], 200)
            self.assertEqual(result["etag"], "etag-1")

    def test_export_records_failed_cloud_attempt_without_claiming_comparison(self):
        result = {"status": "not applicable", "reason": "No standalone map was compared.",
                  "comparison": {"direct_sia_row_count": 467}}
        failure = {"status": "failed", "error_type": "ConnectionError",
                   "requested_key": "dr1/spectral_wcs/map.fits"}
        merged = _merge_cloud_calibration_attempt(result, failure)
        self.assertEqual(merged["comparison"]["cloud_calibration_attempt"], failure)
        self.assertEqual(merged["comparison"]["direct_sia_row_count"], 467)
        self.assertIn("failed", merged["reason"])
        self.assertEqual(merged["status"], "not applicable")


if __name__ == "__main__":
    unittest.main()
