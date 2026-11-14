"""Offline, explicitly synthetic tests for Gaia/SIA candidate helper logic."""

import unittest

from astropy.table import Table
import astropy.units as u

from pipeline.probe_control_star import (
    _gaia_candidates,
    _wavelength_bounds,
    cutout_url,
    detector_from_url,
)


class ControlStarProbeTests(unittest.TestCase):
    def test_synthetic_gaia_filter_keeps_isolated_low_pm_candidate(self):
        table = Table({
            "source_id": [101, 102, 103],
            "ra": [10.0, 10.001, 20.0],
            "dec": [0.0, 0.0, 0.0],
            "pmra": [1.0, 0.5, 2.0],
            "pmdec": [1.0, 0.5, 2.0],
            "phot_g_mean_mag": [12.0, 11.0, 13.0],
            "ruwe": [1.0, 1.0, 1.0],
            "duplicated_source": [False, False, False],
            "phot_variable_flag": ["NOT_AVAILABLE", "NOT_AVAILABLE", "NOT_AVAILABLE"],
        })

        candidates = _gaia_candidates(table)

        # Source 101 has a nearby brighter contaminant (102) and is rejected.
        self.assertEqual([row["source_id"] for row in candidates], ["103"])
        self.assertAlmostEqual(candidates[0]["pm_total"], 2.0 * 2.0**0.5)

    def test_synthetic_access_url_helpers(self):
        url = "https://irsa.ipac.caltech.edu/ibe/data/spherex/dr1/level2/a/4/xD4_spx_l2b.fits"
        self.assertEqual(detector_from_url(url), 4)
        cutout = cutout_url(url, 123.4, -5.6)
        self.assertIn("center=123.4000000000%2C-5.6000000000", cutout)
        self.assertIn("size=0.1", cutout)

    def test_synthetic_sia_table_reports_wavelength_span(self):
        table = Table({
            "em_min": [0.80, 2.20] * u.um,
            "em_max": [1.10, 2.60] * u.um,
        })
        self.assertEqual(_wavelength_bounds([("synthetic", table)]), (0.8, 2.6))

    def test_synthetic_sia_metre_wavelengths_are_converted_to_microns(self):
        table = Table({
            "em_min": ([0.7455, 1.10, 3.81] * u.um).to(u.m),
            "em_max": ([1.1167, 1.65, 5.0096] * u.um).to(u.m),
        })
        self.assertAlmostEqual(_wavelength_bounds([("synthetic", table)])[0], 0.7455, places=7)
        self.assertAlmostEqual(_wavelength_bounds([("synthetic", table)])[1], 5.0096, places=7)


if __name__ == "__main__":
    unittest.main()
