import unittest

from astropy.table import Table

from pipeline.query_comparison_catalogs_vizier import normalize_allwise, normalize_tmass


class VizierComparisonCatalogTests(unittest.TestCase):
    def test_normalizes_2mass_vizier_fields_for_fixed_match_gates(self):
        raw = Table({"2MASS": ["17590579+6628028"], "RAJ2000": [269.7741],
                     "DEJ2000": [66.4675], "Qflg": ["AAA"]})
        result = normalize_tmass(raw)
        self.assertEqual(result.colnames, ["designation", "ra", "dec", "ph_qual"])
        self.assertEqual(result[0]["ph_qual"], "AAA")
        self.assertEqual(float(result[0]["ra"]), 269.7741)

    def test_normalizes_allwise_vizier_fields_for_fixed_match_gates(self):
        raw = Table({"AllWISE": ["J175905.79+662802.9"], "RAJ2000": [269.7741],
                     "DEJ2000": [66.4675], "qph": ["AAAC"], "ccf": ["00OD"], "ex": [0]})
        result = normalize_allwise(raw)
        self.assertEqual(result.colnames, ["designation", "ra", "dec", "ph_qual", "cc_flags", "ext_flg"])
        self.assertEqual(result[0]["cc_flags"], "00OD")
        self.assertEqual(int(result[0]["ext_flg"]), 0)


if __name__ == "__main__":
    unittest.main()
