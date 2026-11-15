import unittest

import numpy as np

from pipeline.run_injection_test import _trial_level, native_psf_stamp, scale_stamp_to_aperture_flux


class InjectionHelperTests(unittest.TestCase):
    def test_oversampled_psf_is_sampled_at_fractional_native_pixel_and_normalized(self):
        y, x = np.indices((41, 41), dtype=float)
        psf = np.exp(-((x - 20.0) ** 2 + (y - 20.0) ** 2) / (2 * 3.0 ** 2))
        stamp, origin = native_psf_stamp(
            psf, (20.25, 19.75), (32, 32), oversample_x=5, oversample_y=5,
        )
        self.assertEqual(origin, (12, 11))
        self.assertAlmostEqual(float(stamp.sum()), 1.0, places=12)
        self.assertGreater(stamp[20, 20], 0)

    def test_flux_scaling_matches_requested_three_pixel_aperture_flux(self):
        stamp = np.zeros((15, 15), dtype=float)
        stamp[7, 7] = 0.8
        stamp[7, 8] = 0.2
        pixel_area_sr = 1.0e-10
        scaled = scale_stamp_to_aperture_flux(stamp, (7.0, 7.0), 2.0, pixel_area_sr)
        yy, xx = np.indices(stamp.shape)
        aperture = np.hypot(xx - 7.0, yy - 7.0) <= 3.0
        # The measurement pipeline reports sum(image * pixel_area_sr) in Jy.
        self.assertAlmostEqual(float(scaled[aperture].sum()) * pixel_area_sr * 1.0e6, 2.0)

    def test_invalid_psf_and_oversampling_fail_explicitly(self):
        with self.assertRaises(ValueError):
            native_psf_stamp(np.ones((3, 3)), (2, 2), (5, 5), oversample_x=0, oversample_y=1)
        with self.assertRaises(ValueError):
            native_psf_stamp(np.zeros((3, 3)), (2, 2), (5, 5), oversample_x=1, oversample_y=1)

    def test_unavailable_frames_do_not_count_as_failed_recoveries(self):
        result = _trial_level("flux_change", 0.1, [
            {"attempted": True, "recovered": True, "recovered_over_injected_ratio": 1.0},
            {"attempted": True, "recovered": False, "recovered_over_injected_ratio": 0.2},
            {"attempted": False, "recovered": None, "recovered_over_injected_ratio": None},
            {"attempted": True, "recovered": None, "recovered_over_injected_ratio": None},
        ])
        self.assertEqual(result["n_trials"], 4)
        self.assertEqual(result["n_completed"], 2)
        self.assertEqual(result["n_unavailable"], 1)
        self.assertEqual(result["n_processing_errors"], 1)
        self.assertEqual(result["recovery_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
