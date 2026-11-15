"""Synthetic offline tests for the user-run Victoria download/Horizons scripts."""

import unittest
from pathlib import Path
from unittest.mock import patch

from astropy.table import Table
from astropy.config.paths import get_cache_dir_path

from pipeline.compare_horizons_geocenter import midpoint_epochs, query_epochs_with_workspace_cache
from pipeline.fetch_cutouts import select_epoch_rows


class LocalEpochScriptTests(unittest.TestCase):
    def test_selection_uses_distinct_nights_and_varies_detectors(self):
        table = Table({
            "start_date": [f"2025-01-{day:02d}T00:00:00" for day in [1, 1, 2, 2, 3, 3, 4, 4]],
            "mjd_start": [60000, 60000, 60001, 60001, 60002, 60002, 60003, 60003],
            "mjd_end": [60000.001, 60000.001, 60001.001, 60001.001, 60002.001, 60002.001, 60003.001, 60003.001],
            "ra_obj": [10.0] * 8,
            "dec_obj": [20.0] * 8,
            "observationid": [f"obs{day}" for day in [1, 1, 2, 2, 3, 3, 4, 4]],
            "productid": ["D1", "D4", "D2", "D5", "D3", "D6", "D1", "D4"],
            "uri": [f"product_{i}_D{det}.fits" for i, det in enumerate([1, 4, 2, 5, 3, 6, 1, 4])],
        })
        selected = select_epoch_rows(table, count=4)
        self.assertEqual(len({row["_night"] for row in selected}), 4)
        self.assertGreaterEqual(len({row["_detector"] for row in selected}), 3)

    def test_horizons_epochs_use_mjd_interval_midpoints(self):
        table = Table({
            "mjd_start": [60000.0], "mjd_end": [60000.002],
            "ra_obj": [10.0], "dec_obj": [20.0],
            "observationid": ["obs1"], "productid": ["D4"],
        })
        epoch = midpoint_epochs(table)[0]
        self.assertAlmostEqual(epoch["mjd_midpoint"], 60000.001, places=8)
        self.assertTrue(epoch["epoch_utc"].startswith("2023-02-25T"))

    def test_horizons_query_uses_workspace_cache_without_network_in_test(self):
        expected = Table({"RA": [10.0]})
        def inspect_cache(*args, **kwargs):
            cache_path = Path(get_cache_dir_path()).resolve()
            self.assertTrue(cache_path.is_relative_to((Path(__file__).resolve().parents[1] / "pipeline/cache").resolve()))
            return expected
        with patch("pipeline.compare_horizons_geocenter.query_epochs", side_effect=inspect_cache):
            actual = query_epochs_with_workspace_cache("12", [], chunk_size=50)
        self.assertIs(actual, expected)


if __name__ == "__main__":
    unittest.main()
