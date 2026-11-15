"""Recreate aperture/annulus figures and bright-source diagnostics for two frames."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from photutils.detection import DAOStarFinder
from PIL import Image, ImageDraw

from .cutout_processing import PREVIEW_STRETCH

ROOT = Path(__file__).resolve().parents[1]
SERIES = ROOT / "pipeline/cache/control-star-series"
OUT = ROOT / "outputs/control-star/outlier-diagnostics"
TARGET = SkyCoord(269.77410473768515, 66.46748335101324, unit="deg")
FRAMES = [("2025W32_2D_0603_4", 5), ("2026W33_2A_0208_3", 5)]


def create_diagnostics() -> list[dict]:
    OUT.mkdir(parents=True, exist_ok=True)
    report = []
    for obs_id, detector in FRAMES:
        path = SERIES / "northup" / f"{obs_id}_D{detector}_northup.fits"
        with fits.open(path, memmap=True) as hdul:
            image = np.asarray(hdul["IMAGE"].data, dtype=float)
            variance = np.asarray(hdul["VARIANCE"].data, dtype=float)
            wcs = WCS(hdul["IMAGE"].header).celestial
            tx, ty = map(float, wcs.world_to_pixel(TARGET))
            aperture_x, aperture_y = tx, ty
            if obs_id == "2025W32_2D_0603_4":
                measured_for_aperture = SkyCoord(269.78679966011134, 66.46270913072287, unit="deg")
                aperture_x, aperture_y = map(float, wcs.world_to_pixel(measured_for_aperture))
            valid = np.isfinite(image) & np.isfinite(variance) & (variance > 0)
            noise = float(np.sqrt(np.nanmedian(variance[valid])))
            background = float(np.nanmedian(image[valid]))
            finder = DAOStarFinder(fwhm=2.5, threshold=5 * noise, exclude_border=True)
            detections = finder(np.where(valid, image - background, 0.0), mask=~valid)
            sources = []
            if detections is not None:
                for row in detections:
                    x, y = float(row["x_centroid"]), float(row["y_centroid"])
                    sources.append({"x": x, "y": y, "peak_mjy_sr": float(row["peak"]),
                                    "separation_from_target_px": float(np.hypot(x - tx, y - ty)),
                                    "sharpness": float(row["sharpness"])})
        sources.sort(key=lambda s: s["peak_mjy_sr"], reverse=True)
        low, high = PREVIEW_STRETCH["vmin"], PREVIEW_STRETCH["vmax"]
        stretch = np.arcsinh(np.clip(np.nan_to_num(image, nan=low), low, high) - low) / np.arcsinh(high - low)
        pixels = np.flipud(np.rint(np.clip(stretch, 0, 1) * 255).astype(np.uint8))
        canvas = Image.fromarray(pixels, mode="L").convert("RGB")
        draw = ImageDraw.Draw(canvas)
        # The resampled image has 128 px. Radius 3 is the aperture; radii 6 and 10 bound the annulus.
        px, py = aperture_x, 127 - aperture_y
        draw.ellipse((px - 3, py - 3, px + 3, py + 3), outline=(0, 255, 255), width=1)
        draw.ellipse((px - 6, py - 6, px + 6, py + 6), outline=(255, 255, 0), width=1)
        draw.ellipse((px - 10, py - 10, px + 10, py + 10), outline=(255, 255, 0), width=1)
        draw.rectangle((1, 1, 89, 12), fill=(0, 0, 0))
        draw.text((3, 2), "cyan r=3; yellow r=6-10", fill=(255, 255, 255))
        output_name = f"{obs_id}_D{detector}_aperture-annulus.png"
        canvas.save(OUT / output_name, optimize=True)
        if obs_id == "2025W32_2D_0603_4":
            measured = SkyCoord(269.78679966011134, 66.46270913072287, unit="deg")
            mx, my = map(float, wcs.world_to_pixel(measured))
            centroid_offset_px = float(np.hypot(mx - tx, my - ty))
            centroid_offset_arcsec = centroid_offset_px * 6.0
            candidate = {"x": mx, "y": my}
            centroid_status = "measured centroid displaced from Gaia prediction; cause unresolved"
        else:
            centroid_offset_px = centroid_offset_arcsec = None
            candidate = None
            centroid_status = "no reliable centroid; source-aperture TRANSIENT/SUR_ERROR/NONFUNC flags"
        report.append({"observation_id": obs_id, "detector": detector,
                       "figure": f"outlier-diagnostics/{output_name}",
                       "figure_marks": {"cyan_circle_radius_px": 3, "yellow_annulus_inner_radius_px": 6,
                                        "yellow_annulus_outer_radius_px": 10},
                       "centroid": candidate, "centroid_offset_pixels": centroid_offset_px,
                       "centroid_offset_arcsec": centroid_offset_arcsec,
                       "centroid_assessment": centroid_status,
                       "bright_sources_inside_frame": sources})
    (ROOT / "outputs/control-star/outlier_diagnostics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(create_diagnostics(), indent=2))
