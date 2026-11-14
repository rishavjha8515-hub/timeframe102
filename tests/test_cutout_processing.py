"""Synthetic helper and integration tests for the cutout-processing module."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from pipeline.cutout_processing import (
    decode_flag_value,
    decode_flag_definitions,
    aperture_photometry_image_zodi,
    north_up_tan_wcs,
    process_fits_product,
    sample_wavelength_metadata,
)


def synthetic_mef(path: Path) -> None:
    size = 32
    y, x = np.indices((size, size))
    image = (100.0 * np.exp(-((x - 15.5) ** 2 + (y - 15.5) ** 2) / (2 * 1.0**2))).astype("f4")
    zodi = np.full((size, size), 0.5, dtype="f4")
    variance = np.ones((size, size), dtype="f4")
    flags = np.zeros((size, size), dtype="i4")
    flags[16, 16] = 2
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.cunit = ["deg", "deg"]
    wcs.wcs.radesys = "ICRS"
    wcs.wcs.crval = [10.0, 20.0]
    wcs.wcs.crpix = [16.5, 16.5]
    wcs.wcs.cd = np.array([[-6 / 3600, 0], [0, 6 / 3600]])
    image_hdu = fits.ImageHDU(image, header=wcs.to_header(), name="IMAGE")
    image_hdu.header["BUNIT"] = "MJy / sr"
    flags_hdu = fits.ImageHDU(flags, name="FLAGS")
    flags_hdu.header["MSKN0001"] = "OVERFLOW"
    flags_hdu.header["MSKD0001"] = "Synthetic overflow flag"
    var_hdu = fits.ImageHDU(variance, name="VARIANCE")
    var_hdu.header["BUNIT"] = "MJy2 / sr2"
    zodi_hdu = fits.ImageHDU(zodi, name="ZODI")
    zodi_hdu.header["BUNIT"] = "MJy / sr"
    epsf = fits.BinTableHDU.from_columns([
        fits.Column(name="XCENTER", format="E", array=[15.5]),
        fits.Column(name="YCENTER", format="E", array=[15.5]),
        fits.Column(name="CWAVE", format="E", array=[2.5]),
        fits.Column(name="CBAND", format="E", array=[0.1]),
    ], name="EPSF")
    values = np.zeros((1, 2, 2, 2), dtype="f4")
    values[..., 0] = 2.48
    values[..., 1] = 0.11
    wave = fits.BinTableHDU.from_columns([
        fits.Column(name="X", format="2J", array=[[1, 32]]),
        fits.Column(name="Y", format="2J", array=[[1, 32]]),
        fits.Column(name="VALUES", format="8E", dim="(2,2,2)", array=values.reshape(1, 8)),
    ], name="WCS-WAVE")
    fits.HDUList([fits.PrimaryHDU(), image_hdu, flags_hdu, var_hdu, zodi_hdu, epsf, wave]).writeto(path)


class CutoutProcessingTests(unittest.TestCase):
    def test_north_up_tan_axis_convention(self):
        wcs = north_up_tan_wcs(120.0, 20.0, shape=(128, 128))
        center = wcs.pixel_to_world(63.5, 63.5)
        west = wcs.pixel_to_world(64.5, 63.5)
        north = wcs.pixel_to_world(63.5, 64.5)
        self.assertAlmostEqual(center.ra.deg, 120.0, places=7)
        self.assertLess(west.ra.deg, center.ra.deg)
        self.assertGreater(north.dec.deg, center.dec.deg)

    def test_flag_decoder_uses_header_cards(self):
        header = fits.Header()
        header["MSKN0001"] = "OVERFLOW"
        header["MSKD0001"] = "Overflow from this synthetic product"
        definitions = decode_flag_definitions(header)
        self.assertEqual(decode_flag_value(2, definitions), [
            {"bit": 1, "name": "OVERFLOW", "description": "Overflow from this synthetic product"}
        ])
        self.assertEqual(decode_flag_value(6, definitions)[-1]["name"], "UNKNOWN_BIT_2")

    def test_synthetic_wavelength_crosscheck(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wavelength.fits"
            synthetic_mef(path)
            with fits.open(path) as hdul:
                result = sample_wavelength_metadata(hdul, 15.5, 15.5)
            self.assertEqual(result["effective_wavelength_um"], 2.5)
            self.assertAlmostEqual(result["wcs_wave_um"], 2.48, places=6)
            self.assertAlmostEqual(result["wavelength_uncertainty_um"], 0.02, places=6)
            self.assertAlmostEqual(result["wavelength_width_um"], 0.1, places=6)

    def test_synthetic_full_and_cutout_wavelength_coordinates_agree_and_vary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full_path = root / "full.fits"
            updated_full_path = root / "full-updated.fits"
            cutout_path = root / "cutout.fits"
            synthetic_mef(full_path)
            with fits.open(full_path) as full:
                epsf = fits.BinTableHDU.from_columns([
                    fits.Column(name="XCENTER", format="E", array=[8.5, 24.5]),
                    fits.Column(name="YCENTER", format="E", array=[8.5, 24.5]),
                    fits.Column(name="CWAVE", format="E", array=[1.2, 3.2]),
                    fits.Column(name="CBAND", format="E", array=[0.1, 0.1]),
                ], name="EPSF")
                wave_values = np.zeros((1, 2, 2, 2), dtype="f4")
                wave_values[0, :, :, 0] = [[1.0, 2.0], [3.0, 4.0]]
                wave_values[0, :, :, 1] = 0.1
                wave = fits.BinTableHDU.from_columns([
                    fits.Column(name="X", format="2J", array=[[1, 32]]),
                    fits.Column(name="Y", format="2J", array=[[1, 32]]),
                    fits.Column(name="VALUES", format="8E", dim="(2,2,2)", array=wave_values.reshape(1, 8)),
                ], name="WCS-WAVE")
                source_hdus = [h.copy() for h in full]
                source_hdus[-2:] = [epsf, wave]
                fits.HDUList(source_hdus).writeto(updated_full_path)
            full_path = updated_full_path
            with fits.open(full_path) as full:
                # Crop at full-frame x/y offset 16. WCS tables intentionally remain full-frame.
                crop = fits.HDUList([h.copy() for h in full])
                crop["IMAGE"].data = full["IMAGE"].data[16:32, 16:32].copy()
                crop["FLAGS"].data = full["FLAGS"].data[16:32, 16:32].copy()
                crop["VARIANCE"].data = full["VARIANCE"].data[16:32, 16:32].copy()
                crop["ZODI"].data = full["ZODI"].data[16:32, 16:32].copy()
                crop["IMAGE"].header["CRPIX1"] -= 16
                crop["IMAGE"].header["CRPIX2"] -= 16
                crop.writeto(cutout_path)
            with fits.open(full_path) as full, fits.open(cutout_path) as cutout:
                first = sample_wavelength_metadata(full, 7.5, 7.5)
                second_full = sample_wavelength_metadata(full, 23.5, 23.5)
                second_cutout = sample_wavelength_metadata(cutout, 7.5, 7.5)
            self.assertNotEqual(first["effective_wavelength_um"], second_full["effective_wavelength_um"])
            self.assertAlmostEqual(second_full["effective_wavelength_um"], second_cutout["effective_wavelength_um"], places=6)
            self.assertAlmostEqual(second_full["wcs_wave_um"], second_cutout["wcs_wave_um"], places=6)

    def test_aperture_flux_uses_image_minus_zodi_and_variance(self):
        shape = (32, 32)
        yy, xx = np.indices(shape)
        image = np.full(shape, 3.0)
        zodi = np.ones(shape)
        image += np.where(np.hypot(xx - 16, yy - 16) <= 2.0, 5.0, 0.0)
        variance = np.ones(shape)
        wcs = north_up_tan_wcs(10.0, 20.0, shape=shape)
        result = aperture_photometry_image_zodi(image, zodi, variance, wcs, (16, 16))
        self.assertGreater(result["flux_jy"], 0)
        self.assertGreater(result["uncertainty_jy"], 0)
        self.assertEqual(result["annulus_pixels"] > 10, True)

    def test_synthetic_fits_reprojection_centroid_and_flagged_association(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "synthetic-mef.fits"
            outdir = Path(tmp) / "out"
            synthetic_mef(source)
            result = process_fits_product(
                source, output_dir=outdir, ra_deg=10.0, dec_deg=20.0,
                observation_id="synthetic-obs", target_name="synthetic",
                detector=4, shape=(32, 32),
            )
            self.assertAlmostEqual(result["predicted_xy"][0], 15.5, places=7)
            self.assertAlmostEqual(result["predicted_xy"][1], 15.5, places=7)
            self.assertIsNone(result["measured_source_xy"])
            self.assertIsNotNone(result["centroid_candidate_xy"])
            self.assertIn("OVERFLOW", result["flags_within_2_pixel_radius_box"])
            self.assertIn("quality-flagged", result["association"])
            self.assertEqual(result["wavelength_sample_basis"], "predicted_position_unresolved")
            with fits.open(result["reprojected_fits_path"]) as output:
                self.assertEqual([h.name for h in output], ["PRIMARY", "IMAGE", "FLAGS", "VARIANCE", "ZODI", "EPSF_SRC", "WAVE_SRC"])
                self.assertEqual(output["IMAGE"].data.shape, (32, 32))


if __name__ == "__main__":
    unittest.main()
