"""The amber wash the viewer paints over the texels a front end fired on.

These exist because the first version sliced the detector mask with the
viewer's *float* visible rectangle.  PIL's resize box takes floats, numpy does
not, so the IndexError threw out of the render and the canvas went blank with
the overlay ticked.  Every case here passes floats.
"""

from __future__ import annotations

import unittest

import numpy as np

from mesh_segmentation_transform.annotate_texture_tuning_app import (
    DETECTOR_TEXEL_COLOUR,
    texel_overlay,
)


def ring(size: int = 64) -> np.ndarray:
    """A one-texel loop, the thing the overlay exists to show."""
    mask = np.zeros((size, size), dtype=bool)
    low, high = size // 4, size - size // 4 - 1
    mask[low, low:high + 1] = mask[high, low:high + 1] = True
    mask[low:high + 1, low] = mask[low:high + 1, high] = True
    return mask


class TexelOverlayTests(unittest.TestCase):
    def test_a_fractional_view_rectangle_does_not_raise(self) -> None:
        # The viewer's rectangle is max(origin, 0.0) -- a float, always.
        result = texel_overlay(ring(), (3.5, 7.25, 60.75, 61.5), 3.5, 7.25, 1.0)

        self.assertIsNotNone(result)
        tint, alpha, at = result
        self.assertEqual(tint.size, alpha.size)
        self.assertEqual(at, (0, 0))

    def test_the_rectangle_snaps_outwards_to_whole_texels(self) -> None:
        _tint, alpha, at = texel_overlay(
            ring(), (10.9, 20.1, 30.2, 40.8), 0.0, 0.0, 1.0
        )
        # floor(10.9)=10 .. ceil(30.2)=31 is 21 texels, not 20.
        self.assertEqual(alpha.size, (21, 21))
        self.assertEqual(at, (10, 20))

    def test_it_is_clamped_to_the_mask(self) -> None:
        result = texel_overlay(ring(32), (-5.0, -5.0, 999.0, 999.0), -5.0, -5.0, 1.0)
        _tint, alpha, at = result
        self.assertEqual(alpha.size, (32, 32))
        self.assertEqual(at, (5, 5))

    def test_a_one_texel_loop_survives_being_zoomed_out(self) -> None:
        """Sampling would drop it; area-averaging plus the lift keeps it."""
        _tint, alpha, _at = texel_overlay(
            ring(), (0.0, 0.0, 64.0, 64.0), 0.0, 0.0, 0.125
        )
        self.assertGreater(np.asarray(alpha).max(), 0)

    def test_it_declines_when_there_is_nothing_to_show(self) -> None:
        self.assertIsNone(texel_overlay(None, (0.0, 0.0, 8.0, 8.0), 0.0, 0.0, 1.0))
        self.assertIsNone(
            texel_overlay(np.zeros((8, 8), bool), (0.0, 0.0, 8.0, 8.0), 0.0, 0.0, 1.0)
        )
        # Entirely outside the mask, and a degenerate scale.
        self.assertIsNone(
            texel_overlay(ring(), (100.0, 100.0, 120.0, 120.0), 0.0, 0.0, 1.0)
        )
        self.assertIsNone(texel_overlay(ring(), (0.0, 0.0, 8.0, 8.0), 0.0, 0.0, 0.0))

    def test_the_wash_is_the_amber_the_checkbox_names(self) -> None:
        tint, _alpha, _at = texel_overlay(
            ring(), (0.0, 0.0, 64.0, 64.0), 0.0, 0.0, 1.0
        )
        self.assertEqual(tint.getpixel((0, 0)), DETECTOR_TEXEL_COLOUR)

    def test_the_alpha_marks_the_loop_and_not_its_interior(self) -> None:
        _tint, alpha, _at = texel_overlay(
            ring(), (0.0, 0.0, 64.0, 64.0), 0.0, 0.0, 1.0
        )
        values = np.asarray(alpha)
        self.assertEqual(values[16, 20], 255)   # on the loop
        self.assertEqual(values[30, 30], 0)     # inside it
        self.assertEqual(values[2, 2], 0)       # outside it


if __name__ == "__main__":
    unittest.main()
