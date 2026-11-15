import unittest
from unittest.mock import Mock, call, patch

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from requests import HTTPError

from pipeline.query_comparison_catalogs_vizier import (
    VIZIER_SERVERS, _query_vizier, normalize_allwise, normalize_tmass,
)


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

    def test_tries_cds_mirror_after_primary_endpoint_failure(self):
        primary = Mock()
        primary.query_region.side_effect = HTTPError("404 primary endpoint degraded")
        mirror = Mock()
        expected = Table({"source": ["catalog row"]})
        mirror.query_region.return_value = [expected]
        with patch("pipeline.query_comparison_catalogs_vizier.Vizier",
                   side_effect=[primary, mirror]) as vizier_factory:
            table, server = _query_vizier(
                "II/246/out", SkyCoord(269.7 * u.deg, 66.4 * u.deg), 0.3
            )
        self.assertEqual(server, VIZIER_SERVERS[1])
        self.assertEqual(table[0]["source"], "catalog row")
        self.assertEqual(vizier_factory.call_args_list, [
            call(columns=["*"], row_limit=100000, vizier_server=VIZIER_SERVERS[0]),
            call(columns=["*"], row_limit=100000, vizier_server=VIZIER_SERVERS[1]),
        ])


if __name__ == "__main__":
    unittest.main()
