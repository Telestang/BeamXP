"""The normal-map rectangle fit is tight, rotated, and enclosing."""

from __future__ import annotations

import unittest

import numpy as np

from glyph_lab.region_rectangles import (
    fit_region_rectangles,
    longest_straight_hull_chain_axis,
)
from glyph_lab.regions import Region


class RegionRectangleTests(unittest.TestCase):
    def test_a_run_of_hull_edges_beats_one_longer_individual_edge(self) -> None:
        points = np.asarray((
            (0, 0), (4, 0), (8, 0.5), (12, 1), (12, 6), (0, 6),
        ), dtype=float)

        axis = longest_straight_hull_chain_axis(points, 8.0, 1.25)

        self.assertIsNotNone(axis)
        np.testing.assert_allclose(axis, np.asarray((12, 1)) / np.hypot(12, 1))

    def test_fitted_rectangle_keeps_all_source_texels_and_fills_the_fit(self) -> None:
        mask = np.zeros((11, 11), dtype=bool)
        for index in range(2, 9):
            mask[index, index - 1:index + 2] = True
        source = Region((20, 40, 11, 11), mask, 3)

        found = fit_region_rectangles([source], 8.0, 1.25)

        self.assertEqual(found[0].domain, source.domain)
        left = source.x - found[0].x
        top = source.y - found[0].y
        source_mask = found[0].mask[top:top + source.height, left:left + source.width]
        self.assertTrue(source_mask[source.mask].all())
        self.assertGreaterEqual(found[0].texels, source.texels)

    def test_ambiguous_diamond_keeps_its_original_region(self) -> None:
        diamond = np.zeros((9, 9), dtype=bool)
        diamond[0, 4] = diamond[4, 0] = diamond[4, 8] = diamond[8, 4] = True
        source = Region((20, 40, 9, 9), diamond, 3)

        found = fit_region_rectangles([source], 8.0, 1.25)

        self.assertEqual(found, [source])
