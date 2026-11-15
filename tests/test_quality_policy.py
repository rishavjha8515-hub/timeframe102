import unittest

import numpy as np
from astropy.table import Table

from pipeline.quality_policy import POLICY, classify_quality_flags, fit_spectral_residuals, preselect_gaia_comparisons


class QualityPolicyTests(unittest.TestCase):
    def test_predeclared_flags(self):
        usable, excluded = classify_quality_flags(["SOURCE", "PERSIST", "HALO"])
        self.assertTrue(usable)
        self.assertEqual(excluded, [])
        usable, excluded = classify_quality_flags(["SOURCE", "TRANSIENT", "SUR_ERROR"])
        self.assertFalse(usable)
        self.assertEqual(excluded, ["TRANSIENT", "SUR_ERROR"])
        self.assertEqual(POLICY["mad_residual_threshold_sigma"], 5.0)

    def test_robust_spectral_mad_marks_large_flux_outlier(self):
        observations = [
            {"observation_id": f"obs-{i}", "detector": 1, "wavelength_um": 1 + i * 0.05,
             "flux": {"flux_jy": 0.05 * (1 + i * 0.05) ** -1.2}}
            for i in range(12)
        ]
        observations.append({"observation_id": "bright", "detector": 5, "wavelength_um": 1.25,
                             "flux": {"flux_jy": 1.0}})
        result = fit_spectral_residuals(observations)
        self.assertIn(("bright", 5), result["outliers"])
        self.assertEqual(result["model"], POLICY["spectral_fit_model"])

    def test_gaia_preselection_applies_fixed_cuts_and_isolation(self):
        table = Table({
            "source_id": ["target", "isolated", "crowded", "neighbor", "variable", "high_pm", "bad_ruwe"],
            "ra": [10.0, 10.02, 10.04, 10.0401, 10.08, 10.10, 10.12],
            "dec": [20.0] * 7,
            "phot_g_mean_mag": [12.0] * 7,
            "pmra": [0.0, 1.0, 1.0, 1.0, 1.0, 11.0, 1.0],
            "pmdec": [0.0] * 7,
            "ruwe": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5],
            "phot_variable_flag": ["CONSTANT", "CONSTANT", "CONSTANT", "CONSTANT", "VARIABLE", "CONSTANT", "CONSTANT"],
        })
        selected = preselect_gaia_comparisons(table, 10.0, 20.0, "target")
        self.assertEqual(selected, ["isolated"])
        self.assertEqual(POLICY["comparison_star_criteria"]["minimum_stars_per_frame"], 20)


if __name__ == "__main__":
    unittest.main()
