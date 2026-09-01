"""Final display padding retains region ownership."""

from __future__ import annotations

import unittest

import numpy as np

from glyph_lab.region_padding import pad_regions
from glyph_lab.regions import Region


class RegionPaddingTests(unittest.TestCase):
    def test_padding_grows_a_region_by_the_requested_margin(self) -> None:
        source = Region((4, 4, 3, 3), np.ones((3, 3), dtype=bool), 1)

        found = pad_regions([source], 2, np.ones((12, 12), dtype=bool), (0, 0))

        self.assertEqual(found[0].bounds, (2, 2, 7, 7))
        self.assertTrue(found[0].mask.all())

    def test_padding_never_merges_two_neighbouring_regions(self) -> None:
        first = Region((2, 4, 2, 2), np.ones((2, 2), dtype=bool), 1)
        second = Region((6, 4, 2, 2), np.ones((2, 2), dtype=bool), 1)

        found = pad_regions(
            [first, second], 2, np.ones((12, 12), dtype=bool), (0, 0),
        )

        canvas = np.zeros((12, 12), dtype=np.uint8)
        for region in found:
            x, y, width, height = region.bounds
            canvas[y:y + height, x:x + width] += region.mask
        self.assertEqual(int(canvas.max()), 1)
        self.assertTrue(canvas[4:6, 2:4].all())
        self.assertTrue(canvas[4:6, 6:8].all())
