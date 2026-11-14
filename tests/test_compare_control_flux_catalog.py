import unittest

from pipeline.compare_control_flux_catalog import compare_fluxes


class ControlCatalogFluxTests(unittest.TestCase):
    def test_interpolated_catalog_flux_ratio_uses_log_wavelength_and_errors(self):
        points = [
            {"band": "J", "wavelength_um": 1.0, "flux_jy": 10.0, "uncertainty_jy": 1.0},
            {"band": "H", "wavelength_um": 4.0, "flux_jy": 1.0, "uncertainty_jy": 0.2},
        ]
        case = {"observations": [{"observation_id": "obs", "effective_wavelength_um": 2.0,
                                  "flux": {"value": 2.0, "uncertainty": 0.1}}]}
        result = compare_fluxes(case, points)[0]
        self.assertAlmostEqual(result["catalog_interpolated_flux_jy"], (10.0 * 1.0) ** 0.5)
        self.assertAlmostEqual(result["flux_ratio_spherex_to_catalog"], 2.0 / (10.0 * 1.0) ** 0.5)
        self.assertGreater(result["flux_ratio_uncertainty"], 0)


if __name__ == "__main__":
    unittest.main()
