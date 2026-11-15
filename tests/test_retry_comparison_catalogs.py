import unittest

from pipeline.retry_comparison_catalogs import retry_delays


class RetryComparisonCatalogTests(unittest.TestCase):
    def test_backoff_starts_at_fifteen_minutes_and_caps_at_twenty(self):
        self.assertEqual(retry_delays(120), [15, 20, 20, 20, 20, 20])

    def test_schedule_does_not_start_attempt_past_window(self):
        self.assertEqual(retry_delays(35), [15])

    def test_invalid_schedule_is_rejected(self):
        with self.assertRaises(ValueError):
            retry_delays(0)


if __name__ == "__main__":
    unittest.main()
