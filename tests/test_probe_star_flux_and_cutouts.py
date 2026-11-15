"""Synthetic offline tests for control-star observation planning."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock
from unittest.mock import patch

import requests
from astropy.io import fits
from astropy.table import Table

from pipeline.probe_star_flux_and_cutouts import (
    CutoutDownloadError,
    _download,
    inspect_cutout_hdus,
    plan_observations,
    overflow_sample,
)


class StarObservationPlanTests(unittest.TestCase):
    def test_plan_balances_six_detectors_and_repeats_dates(self):
        rows = {"access_url": [], "obs_id": [], "t_min": [], "t_max": []}
        for day in range(12):
            for detector in range(1, 7):
                rows["access_url"].append(
                    f"https://irsa.ipac.caltech.edu/ibe/data/D{detector}_spx_l2b_product.fits"
                )
                rows["obs_id"].append(f"obs-{day}-{detector}")
                rows["t_min"].append(60000 + day)
                rows["t_max"].append(60000 + day + 0.001)
        plan = plan_observations([Table(rows)], count=12)
        self.assertEqual(len(plan), 12)
        self.assertEqual({row["detector"] for row in plan}, set(range(1, 7)))
        self.assertTrue(all(len({r["utc_night"] for r in plan if r["detector"] == det}) >= 2 for det in range(1, 7)))
        sample = overflow_sample(plan, 4)
        self.assertEqual(len(sample), 4)
        self.assertEqual(len({row["detector"] for row in sample}), 4)
        self.assertGreater(sample[-1]["mjd_midpoint"] - sample[0]["mjd_midpoint"], 5)

    def test_plan_does_not_relax_missing_detector_coverage(self):
        rows = {"access_url": [], "obs_id": [], "t_min": [], "t_max": []}
        for day in range(3):
            for detector in range(1, 6):
                rows["access_url"].append(f"https://irsa.ipac.caltech.edu/D{detector}_spx_l2b.fits")
                rows["obs_id"].append(f"obs-{day}-{detector}")
                rows["t_min"].append(60000 + day)
                rows["t_max"].append(60000 + day + 0.001)
        with self.assertRaisesRegex(ValueError, "per-detector available"):
            plan_observations([Table(rows)], count=12)

    def test_cutout_inspector_reports_expected_cutout_mef_extensions(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "cutout.fits"
            image_header = fits.Header({"CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN"})
            epsf = fits.BinTableHDU.from_columns([
                fits.Column(name="XCENTER", format="E", array=[1.0]),
                fits.Column(name="YCENTER", format="E", array=[1.0]),
                fits.Column(name="CWAVE", format="E", array=[1.2]),
                fits.Column(name="CBAND", format="E", array=[0.1]),
            ], name="EPSF")
            wave = fits.BinTableHDU.from_columns([
                fits.Column(name="X", format="2E", array=[[1.0, 2.0]]),
                fits.Column(name="Y", format="2E", array=[[1.0, 2.0]]),
                fits.Column(name="VALUES", format="8E", dim="(2,2,2)",
                            array=[[[[1.0, .1], [1.1, .1]], [[1.2, .1], [1.3, .1]]]]),
            ], name="WCS-WAVE")
            fits.HDUList([
                fits.PrimaryHDU(),
                fits.ImageHDU(data=[[1.0, 2.0], [3.0, 4.0]], header=image_header, name="IMAGE"),
                fits.ImageHDU(data=[[0, 0], [0, 0]], name="FLAGS"),
                fits.ImageHDU(data=[[1.0, 1.0], [1.0, 1.0]], name="VARIANCE"),
                fits.ImageHDU(data=[[0.0, 0.0], [0.0, 0.0]], name="ZODI"), epsf, wave,
            ]).writeto(path)
            listing = inspect_cutout_hdus(path)
            expected = {"IMAGE", "FLAGS", "VARIANCE", "ZODI", "EPSF", "WCS-WAVE"}
            self.assertTrue(expected.issubset({item["name"] for item in listing}))

    def test_download_refuses_declared_oversize_without_saving(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "large.fits"
            response = Mock(ok=True, status_code=200, headers={"Content-Length": "1024"})
            session = Mock()
            session.get.return_value = response
            with self.assertRaises(CutoutDownloadError) as raised:
                _download(session, "https://irsa.ipac.caltech.edu/cutout", path, max_bytes=10)
            self.assertEqual(raised.exception.status, 200)
            self.assertIn("Content-Length", raised.exception.body)
            self.assertFalse(path.exists())

    def test_download_error_preserves_http_status_and_body(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "error.fits"
            response = Mock(ok=False, status_code=405, headers={}, text="HEAD is not supported")
            session = Mock()
            session.get.return_value = response
            with self.assertRaises(CutoutDownloadError) as raised:
                _download(session, "https://irsa.ipac.caltech.edu/cutout", path)
            self.assertEqual(raised.exception.status, 405)
            self.assertEqual(raised.exception.body, "HEAD is not supported")
            self.assertFalse(path.exists())

    def test_download_aborts_if_stream_exceeds_cap(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "stream.fits"
            response = Mock(ok=True, status_code=200, headers={})
            response.iter_content.return_value = [b"123456", b"789"]
            session = Mock()
            session.get.return_value = response
            with self.assertRaises(CutoutDownloadError) as raised:
                _download(session, "https://irsa.ipac.caltech.edu/cutout", path, max_bytes=8)
            self.assertEqual(raised.exception.status, 200)
            self.assertFalse(path.exists())

    def test_download_retries_503_and_honors_retry_after(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "retry.fits"
            busy = Mock(ok=False, status_code=503, headers={"Retry-After": "7"}, text="Server busy")
            success = Mock(ok=True, status_code=200, headers={"Content-Length": "4"})
            success.iter_content.return_value = [b"fits"]
            session = Mock()
            session.get.side_effect = [busy, success]
            with patch("pipeline.probe_star_flux_and_cutouts.time.sleep") as sleep:
                size, digest, status, length = _download(session, "https://irsa.ipac.caltech.edu/cutout", path)
            self.assertEqual(size, 4)
            self.assertEqual(status, 200)
            self.assertEqual(length, 4)
            sleep.assert_called_once_with(7.0)

    def test_download_retries_incomplete_stream_read(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "stream-retry.fits"
            interrupted = Mock(ok=True, status_code=200, headers={"Content-Length": "4"})
            interrupted.iter_content.side_effect = requests.exceptions.ChunkedEncodingError("IncompleteRead")
            success = Mock(ok=True, status_code=200, headers={"Content-Length": "4"})
            success.iter_content.return_value = [b"fits"]
            session = Mock()
            session.get.side_effect = [interrupted, success]
            with patch("pipeline.probe_star_flux_and_cutouts.time.sleep") as sleep:
                size, digest, status, length = _download(session, "https://irsa.ipac.caltech.edu/cutout", path)
            self.assertEqual((size, status, length), (4, 200, 4))
            self.assertEqual(path.read_bytes(), b"fits")
            self.assertEqual(session.get.call_count, 2)
            sleep.assert_called_once_with(5)


if __name__ == "__main__":
    unittest.main()
