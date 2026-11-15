import unittest

import numpy as np
from astropy.table import Table

from pipeline.vet_comparison_stars import classify_candidate_pixel, propagate_gaia_position


class ComparisonStarFrameVettingTests(unittest.TestCase):
    def setUp(self):
        self.flags = np.zeros((25, 25), dtype=np.int32)
        self.definitions = {1: ("TRANSIENT", "known transient source")}

    def test_position_inside_cutout_and_edge_margin_is_eligible_without_hard_flags(self):
        result = classify_candidate_pixel(12.0, 12.0, self.flags, self.definitions,
                                          pixel_scale_arcsec=6.0)
        self.assertTrue(result["inside_cutout"])
        self.assertTrue(result["edge_safe"])
        self.assertTrue(result["eligible"])

    def test_edge_position_is_excluded_before_flag_check(self):
        result = classify_candidate_pixel(3.0, 12.0, self.flags, self.definitions,
                                          pixel_scale_arcsec=6.0)
        self.assertFalse(result["edge_safe"])
        self.assertFalse(result["eligible"])
        self.assertIn("edge", result["reason"])

    def test_hard_flag_in_5x5_window_excludes_candidate(self):
        self.flags[10, 11] = 1 << 1
        result = classify_candidate_pixel(12.0, 12.0, self.flags, self.definitions,
                                          pixel_scale_arcsec=6.0)
        self.assertEqual(result["hard_flags"], ["TRANSIENT"])
        self.assertFalse(result["eligible"])

    def test_gaia_position_propagates_using_pmra_cosdec_convention(self):
        row = Table({"ra": [10.0], "dec": [30.0], "pmra": [100.0], "pmdec": [-50.0]})[0]
        ra, dec = propagate_gaia_position(row, "2026-01-01T00:00:00Z")
        self.assertGreater(ra, 10.0)
        self.assertLess(dec, 30.0)


if __name__ == "__main__":
    unittest.main()
