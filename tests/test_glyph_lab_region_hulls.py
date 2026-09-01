"""The post-grouping convex envelopes retain ownership boundaries."""

from __future__ import annotations

import unittest

import numpy as np

from glyph_lab.region_hulls import convex_region_hulls
from glyph_lab.regions import Region


class RegionHullTests(unittest.TestCase):
    def test_a_hull_fills_the_concavity_of_one_region(self) -> None:
        mask = np.zeros((5, 5), dtype=bool)
        mask[0, 0] = mask[0, 4] = mask[4, 0] = True

        found = convex_region_hulls([Region((10, 20, 5, 5), mask, 1)])

        self.assertTrue(found[0].mask[2, 1])
        self.assertTrue(found[0].mask[1, 2])

    def test_a_hull_gives_another_region_its_original_texels_back(self) -> None:
        first = np.zeros((5, 5), dtype=bool)
        first[0, 0] = first[0, 4] = first[4, 0] = first[4, 4] = True
        second = np.ones((3, 2), dtype=bool)

        found = convex_region_hulls([
            Region((10, 20, 5, 5), first, 1),
            Region((14, 21, 2, 3), second, 1),
        ])

        self.assertFalse(found[0].mask[1:4, 4].any())
        self.assertTrue(found[1].mask.all())

    def test_a_region_inside_another_regions_enclosed_hole_is_absorbed(self) -> None:
        outer = np.zeros((7, 7), dtype=bool)
        outer[0, :] = outer[-1, :] = True
        outer[:, 0] = outer[:, -1] = True
        inner = np.ones((1, 1), dtype=bool)

        found = convex_region_hulls([
            Region((10, 20, 7, 7), outer, 1),
            Region((13, 23, 1, 1), inner, 1),
        ])

        self.assertEqual(len(found), 1)
        self.assertTrue(found[0].mask[3, 3])

    def test_a_region_inside_another_regions_convex_hull_is_absorbed(self) -> None:
        corners = np.zeros((7, 7), dtype=bool)
        corners[0, 0] = corners[0, 6] = corners[6, 0] = corners[6, 6] = True
        inner = np.ones((1, 1), dtype=bool)

        found = convex_region_hulls([
            Region((10, 20, 7, 7), corners, 1),
            Region((13, 23, 1, 1), inner, 1),
        ])

        self.assertEqual(len(found), 1)
        self.assertTrue(found[0].mask[3, 3])
