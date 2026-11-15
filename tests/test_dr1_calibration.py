import unittest

from astropy.table import Table

from pipeline.check_dr1_calibration import _matching_epoch


class Dr1CalibrationTests(unittest.TestCase):
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
