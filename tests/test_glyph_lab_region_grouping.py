"""The final shared grouping pass joins only demonstrably flat corridors."""

from __future__ import annotations

import unittest

import numpy as np

from glyph_lab.region_grouping import (
    GroupingConfig,
    group_regions_across_flat_ground,
    mesh_region_labels,
)
from glyph_lab.regions import Region


def box(x: int, y: int, width: int, height: int) -> Region:
    return Region((x, y, width, height), np.ones((height, width), dtype=bool), 1)


class FlatGroundGroupingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.colour = np.zeros((64, 96, 3), dtype=np.uint8)
        self.normal_flat = np.ones((64, 96), dtype=bool)
        self.config = GroupingConfig(
            group_max_gap_px=24, group_min_shared_span_px=4,
            group_path_margin_px=3, group_colour_edge_threshold=24,
            group_close_gap_px=0,
        )

    def test_a_shared_flat_horizontal_corridor_makes_one_region(self) -> None:
        found = group_regions_across_flat_ground(
            [box(10, 20, 8, 12), box(34, 22, 8, 12)],
            self.colour, self.normal_flat, self.config,
        )
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].bounds, (10, 20, 32, 14))
        self.assertTrue(found[0].mask[4:12, 8:24].all())

    def test_overlapping_bounds_do_not_hide_a_real_horizontal_gap(self) -> None:
        left_mask = np.zeros((8, 12), dtype=bool)
        left_mask[:, :8] = True
        right_mask = np.zeros((8, 12), dtype=bool)
        right_mask[:, 4:] = True
        found = group_regions_across_flat_ground(
            [
                Region((10, 20, 12, 8), left_mask, 1),
                Region((20, 20, 12, 8), right_mask, 1),
            ],
            self.colour, self.normal_flat, self.config,
        )
        self.assertEqual(len(found), 1)

    def test_a_direct_colour_divider_refuses_the_same_corridor(self) -> None:
        self.colour[:, 25] = 80
        found = group_regions_across_flat_ground(
            [box(10, 20, 8, 12), box(34, 22, 8, 12)],
            self.colour, self.normal_flat, self.config,
        )
        self.assertEqual(len(found), 2)

    def test_a_normal_slope_refuses_the_same_corridor(self) -> None:
        self.normal_flat[22:34, 25] = False
        found = group_regions_across_flat_ground(
            [box(10, 20, 8, 12), box(34, 22, 8, 12)],
            self.colour, self.normal_flat, self.config,
        )
        self.assertEqual(len(found), 2)

    def test_diagonal_neighbours_have_no_shared_axis_run(self) -> None:
        found = group_regions_across_flat_ground(
            [box(10, 10, 8, 8), box(30, 30, 8, 8)],
            self.colour, self.normal_flat, self.config,
        )
        self.assertEqual(len(found), 2)

    def test_a_vertical_flat_corridor_joins_the_same_way(self) -> None:
        found = group_regions_across_flat_ground(
            [box(30, 10, 12, 8), box(32, 34, 12, 8)],
            self.colour, self.normal_flat, self.config,
        )
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].bounds, (30, 10, 14, 32))

    def test_colour_only_evidence_groups_a_texture_without_a_normal_map(self) -> None:
        found = group_regions_across_flat_ground(
            [box(10, 20, 8, 12), box(34, 22, 8, 12)],
            self.colour, None, self.config,
        )
        self.assertEqual(len(found), 1)

    def test_a_tight_uniform_gap_groups_despite_the_two_endpoint_slopes(self) -> None:
        flat = np.zeros_like(self.normal_flat)
        found = group_regions_across_flat_ground(
            [box(10, 20, 8, 12), box(20, 20, 8, 12)],
            self.colour, flat, self.config,
        )
        self.assertEqual(len(found), 1)

    def test_a_close_gap_joins_despite_contrast_and_slopes(self) -> None:
        self.colour[:, 20] = 80
        self.normal_flat[20:32, 20] = False
        config = GroupingConfig(
            group_max_gap_px=24, group_min_shared_span_px=4,
            group_path_margin_px=3, group_colour_edge_threshold=24,
            group_close_gap_px=6,
        )
        found = group_regions_across_flat_ground(
            [box(10, 20, 8, 12), box(21, 20, 8, 12)],
            self.colour, self.normal_flat, config,
        )
        self.assertEqual(len(found), 1)

    def test_a_brief_close_approach_uses_the_shared_range_not_four_close_rows(self) -> None:
        first = box(10, 20, 8, 8)
        staggered = np.zeros((8, 10), dtype=bool)
        staggered[:, 7] = True
        staggered[3:5, 0] = True
        second = Region((20, 20, 10, 8), staggered, 1)
        config = GroupingConfig(
            group_max_gap_px=24, group_min_shared_span_px=4,
            group_path_margin_px=3, group_colour_edge_threshold=24,
            group_close_gap_px=3,
        )
        found = group_regions_across_flat_ground(
            [first, second], self.colour, np.zeros_like(self.normal_flat), config,
        )
        self.assertEqual(len(found), 1)
        self.assertTrue(found[0].mask[3:7, 8:18].any())

    def test_a_contrasting_divider_in_a_tight_gap_still_refuses(self) -> None:
        self.colour[:, 20] = 80
        found = group_regions_across_flat_ground(
            [box(10, 20, 8, 12), box(23, 20, 8, 12)],
            self.colour, self.normal_flat, self.config,
        )
        self.assertEqual(len(found), 2)

    def test_a_mesh_seam_between_two_regions_refuses_their_link(self) -> None:
        mesh_regions = np.ones(self.normal_flat.shape, dtype=np.uint32)
        mesh_regions[:, 24:] = 2
        found = group_regions_across_flat_ground(
            [box(10, 20, 8, 12), box(34, 22, 8, 12)],
            self.colour, self.normal_flat, self.config, mesh_regions=mesh_regions,
        )
        self.assertEqual(len(found), 2)

    def test_a_region_mostly_inside_one_mesh_region_can_join_from_its_edge(self) -> None:
        mesh_regions = np.ones(self.normal_flat.shape, dtype=np.uint32)
        mesh_regions[20:32, 16:18] = 2
        found = group_regions_across_flat_ground(
            [box(10, 20, 8, 12), box(22, 20, 8, 12)],
            self.colour, self.normal_flat, self.config, mesh_regions=mesh_regions,
        )
        self.assertEqual(len(found), 1)

    def test_mesh_labels_split_a_coalesced_domain_at_its_seam(self) -> None:
        domains = np.ones((16, 32), dtype=np.uint16)
        seams = np.zeros(domains.shape, dtype=bool)
        seams[:, 16] = True
        labels = mesh_region_labels(domains, seams)
        self.assertNotEqual(labels[8, 8], 0)
        self.assertNotEqual(labels[8, 24], 0)
        self.assertNotEqual(labels[8, 8], labels[8, 24])

    def test_a_cropped_domain_uses_its_atlas_offset_for_evidence(self) -> None:
        found = group_regions_across_flat_ground(
            [box(30, 35, 8, 12), box(54, 37, 8, 12)],
            self.colour, self.normal_flat, self.config, offset=(20, 15),
        )
        self.assertEqual(len(found), 1)
