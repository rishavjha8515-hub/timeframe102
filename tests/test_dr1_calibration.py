import unittest
from unittest.mock import patch

from astropy.table import Table

from pipeline.check_dr1_calibration import CALIB_LEVEL, COLLECTION, _matching_epoch, _query_sia


class Dr1CalibrationTests(unittest.TestCase):
    @patch("pipeline.check_dr1_calibration.Irsa.query_sia")
    def test_query_uses_advertised_dr1_collection_and_calibration_level(self, query_sia):
        query_sia.return_value = Table()
        _query_sia(269.7, 66.4)
        query_sia.assert_called_once_with(
            pos=(269.7, 66.4, 0.001), collection="spherex_dr1",
            calib_level=2, maxrec=10000,
        )
        self.assertEqual(COLLECTION, "spherex_dr1")
        self.assertEqual(CALIB_LEVEL, 2)

    def test_selects_same_usable_observation_and_detector(self):
        case = {"observations": [
            {"observation_id": "obs-a", "detector": 1, "usable": True},
            {"observation_id": "obs-b", "detector": 3, "usable": True},
        ]}
        table = Table({
            "obs_id": ["obs-a", "obs-b", "obs-b"],
            "access_url": [
                "https://irsa.ipac.caltech.edu/data/D2_spx_l2b.fits",
                "https://irsa.ipac.caltech.edu/data/D3_spx_l2b.fits",
                "https://irsa.ipac.caltech.edu/data/D4_spx_l2b.fits",
            ],
        })
        observation, row = _matching_epoch(case, table)
        self.assertEqual(observation["observation_id"], "obs-b")
        self.assertEqual(observation["detector"], 3)
        self.assertEqual(str(row["obs_id"]), "obs-b")

    def test_does_not_substitute_different_epoch_or_detector(self):
        case = {"observations": [{"observation_id": "obs-a", "detector": 1, "usable": True}]}
        table = Table({"obs_id": ["other"],
                       "access_url": ["https://irsa.ipac.caltech.edu/data/D1_spx_l2b.fits"]})
        self.assertEqual(_matching_epoch(case, table), (None, None))


if __name__ == "__main__":
    unittest.main()
