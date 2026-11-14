import unittest

from astropy.table import Table

from pipeline.rank_candidates import summarize_table


class CandidateRankingTests(unittest.TestCase):
    def test_counts_nights_gaps_and_uses_actual_mjd_delta_for_rate(self):
        table = Table(
            rows=[
                ("o1", "2025-01-01 00:00:00", 60000.000, 10.0, 0.0, 12.0),
                ("o2", "2025-01-01 00:01:26", 60000.001, 10.0 + 1 / 3600, 0.0, 12.1),
                ("o3", "2025-01-03 00:00:00", 60002.000, 11.0, 0.0, 12.2),
                ("o3", "2025-01-03 00:01:26", 60002.001, 11.0 + 1 / 3600, 0.0, 12.3),
            ],
            names=("observationid", "start_date", "mjd_start", "ra_obj", "dec_obj", "vmag"),
        )
        summary = summarize_table(table, target="Test", source_file="synthetic-fixture-only")
        self.assertEqual(summary.product_rows, 4)
        self.assertEqual(summary.distinct_observation_ids, 3)
        self.assertEqual(summary.distinct_utc_nights, 2)
        self.assertAlmostEqual(summary.baseline_days, 2.001)
        self.assertEqual(summary.max_gap_days, 2)
        self.assertEqual(summary.intranight_coordinate_pairs, 2)
        self.assertAlmostEqual(summary.median_rate_arcsec_per_day, 1000.0, delta=0.01)

    def test_collapses_subsecond_duplicate_product_rows(self):
        table = Table(
            rows=[
                ("o1", "2025-01-01 00:00:00", 60000.000000000, 10.0, 0.0, 12.0),
                ("o1", "2025-01-01 00:00:00", 60000.000000010, 10.0, 0.0, 12.0),
                ("o2", "2025-01-01 00:01:26", 60000.001000000, 10.0 + 1 / 3600, 0.0, 12.0),
            ],
            names=("observationid", "start_date", "mjd_start", "ra_obj", "dec_obj", "vmag"),
        )
        summary = summarize_table(table, target="Test")
        self.assertEqual(summary.intranight_coordinate_pairs, 1)


if __name__ == "__main__":
    unittest.main()
