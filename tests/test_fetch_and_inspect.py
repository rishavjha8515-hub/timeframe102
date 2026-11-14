import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from astropy.table import Table
from astropy.io import fits

from pipeline.fetch_and_inspect import choose_spaced_rows, inspect_fits, resolve_product_url


class FetchAndInspectTests(unittest.TestCase):
    def test_resolve_relative_irsa_product_uri(self):
        self.assertEqual(
            resolve_product_url("ibe/data/spherex/example.fits"),
            "https://irsa.ipac.caltech.edu/ibe/data/spherex/example.fits",
        )

    def test_reject_product_uri_outside_irsa(self):
        with self.assertRaisesRegex(ValueError, "outside the expected IRSA"):
            resolve_product_url("https://example.org/example.fits")

    def test_choose_spaced_rows_selects_distinct_products_in_time_order(self):
        table = Table(
            rows=[
                ("late.fits", 3.0),
                ("early.fits", 1.0),
                ("middle.fits", 2.0),
                ("middle.fits", 2.0),
            ],
            names=("uri", "mjd_start"),
        )
        selected = choose_spaced_rows(table, 3)
        self.assertEqual([row["uri"] for row in selected], ["early.fits", "middle.fits", "late.fits"])

    def test_choose_spaced_rows_rejects_more_than_available(self):
        table = Table(rows=[("one.fits", 1.0)], names=("uri", "mjd_start"))
        with self.assertRaisesRegex(ValueError, "only 1 distinct URIs"):
            choose_spaced_rows(table, 2)

    def test_inspector_reports_every_hdu_and_shape(self):
        with TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "synthetic_test_only.fits"
            fits.HDUList(
                [
                    fits.PrimaryHDU(np.zeros((3, 4), dtype=np.float32)),
                    fits.ImageHDU(np.ones((2, 2), dtype=np.uint8), name="QUALITY"),
                ]
            ).writeto(path)
            captured = StringIO()
            with redirect_stdout(captured):
                metadata = inspect_fits(path)
            self.assertEqual([hdu["name"] for hdu in metadata], ["PRIMARY", "QUALITY"])
            self.assertEqual([hdu["shape"] for hdu in metadata], [[3, 4], [2, 2]])
            self.assertIn("HDU_COUNT 2", captured.getvalue())


if __name__ == "__main__":
    unittest.main()
