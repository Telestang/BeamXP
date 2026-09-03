"""The independent slope-free normal-map detector."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from glyph_lab.corpus import CorpusTexture
from glyph_lab.detect import Region
from glyph_lab.normal_detect import (
    NORMAL_ABSENT_SLOPES,
    NORMAL_CLEAR_OF_CROWD,
    NORMAL_CLEAR_OF_FRAME,
    NORMAL_CLEAR_OF_PANEL,
    NORMAL_CLEAR_OF_RIM,
    NORMAL_CLEAR_OF_STRIPE,
    NORMAL_CONVEX_HULLED,
    NORMAL_FITTED_RECTANGLE,
    NORMAL_GROUPED_FLAT_GROUND,
    NORMAL_PADDED,
    NORMAL_PAST_SIZE_FLOOR,
    NormalMapConfig,
    clear_of_normal_crowd,
    clear_of_normal_frame,
    clear_of_normal_panel_face,
    clear_of_normal_rim,
    clear_of_normal_stripe,
    detect_normal,
    past_normal_size_floor,
)


def edge_texture() -> CorpusTexture:
    size = 48
    colour = np.zeros((size, size, 3), dtype=np.uint8)
    # A high-steepness rectangular stroke surrounded by slope-free ground.
    colour[16:32, 16:18] = 255
    colour[16:32, 30:32] = 255
    colour[16:18, 16:32] = 255
    colour[30:32, 16:32] = 255
    domains = np.zeros((size, size), dtype=np.uint16)
    domains[4:-4, 4:-4] = 1
    return CorpusTexture("edges", colour, domains, None, {})


class NormalSlopeDetectorTests(unittest.TestCase):
    def test_it_paints_regions_around_non_slope_ground_holes(self) -> None:
        found = detect_normal(
            edge_texture(),
            NormalMapConfig(
                min_width_px=4, min_height_px=4, min_area_px=36,
                panel_sparse_min_span_px=0, panel_solid_min_minor_span_px=0,
            ),
            capture=True,
        )
        self.assertGreater(len(found.regions), 0)
        self.assertTrue(found.fired.any())
        self.assertTrue(found.sparse.any())
        self.assertIn(NORMAL_ABSENT_SLOPES, found.stages)
        self.assertEqual(
            found.stage_counts[NORMAL_ABSENT_SLOPES], len(found.regions)
        )

    def test_flat_normal_input_leaves_no_painted_regions(self) -> None:
        texture = edge_texture()
        texture = CorpusTexture(
            texture.name, np.zeros_like(texture.colour), texture.domains,
            texture.normal, texture.source,
        )
        found = detect_normal(texture)
        self.assertEqual(found.regions, [])
        self.assertFalse(found.fired.any())

    def test_its_rim_filter_refuses_a_shallow_edge_band(self) -> None:
        sample = np.ones((32, 32), dtype=bool)
        band = Region((1, 10, 3, 12), np.ones((12, 3), dtype=bool), 1)
        kept = clear_of_normal_rim(
            [band], sample, (0, 0),
            NormalMapConfig(
                min_rim_margin_px=4, max_rim_margin_px=4,
                rim_margin_share=0.0, min_rim_contact=0.0,
            ),
        )
        self.assertEqual(kept, [])

    def test_the_rim_stage_is_captured_separately(self) -> None:
        found = detect_normal(edge_texture(), NormalMapConfig(), capture=True)
        self.assertIn(NORMAL_CLEAR_OF_RIM, found.stages)
        self.assertIn(NORMAL_CLEAR_OF_RIM, found.stage_outlines)

    def test_its_crowd_filter_refuses_small_neighbours(self) -> None:
        regions = [
            Region((x, 10, 2, 2), np.ones((2, 2), dtype=bool), 1)
            for x in (10, 15, 20)
        ]
        kept = clear_of_normal_crowd(
            regions, regions,
            NormalMapConfig(
                crowd_radius_px=12, max_neighbours=1,
                crowd_large_region_area_px=0,
            ),
        )
        self.assertEqual(kept, [])

    def test_the_crowd_stage_is_captured_separately(self) -> None:
        found = detect_normal(edge_texture(), NormalMapConfig(), capture=True)
        self.assertIn(NORMAL_CLEAR_OF_CROWD, found.stages)
        self.assertIn(NORMAL_CLEAR_OF_CROWD, found.stage_outlines)

    def test_its_size_floor_refuses_a_small_region(self) -> None:
        small = Region((10, 10, 3, 3), np.ones((3, 3), dtype=bool), 1)
        kept = past_normal_size_floor(
            [small], NormalMapConfig(
                min_width_px=4, min_height_px=4, min_area_px=16,
            ),
        )
        self.assertEqual(kept, [])

    def test_the_size_floor_stage_is_captured_separately(self) -> None:
        found = detect_normal(edge_texture(), NormalMapConfig(), capture=True)
        self.assertIn(NORMAL_PAST_SIZE_FLOOR, found.stages)
        self.assertIn(NORMAL_PAST_SIZE_FLOOR, found.stage_outlines)

    def test_its_frame_filter_refuses_a_large_simple_bezel(self) -> None:
        mask = np.zeros((80, 80), dtype=bool)
        mask[:6] = True
        mask[-6:] = True
        mask[:, :6] = True
        mask[:, -6:] = True
        frame = Region((10, 10, 80, 80), mask, 1)
        kept = clear_of_normal_frame([frame], NormalMapConfig())
        self.assertEqual(kept, [])

    def test_the_frame_stage_is_captured_separately(self) -> None:
        found = detect_normal(edge_texture(), NormalMapConfig(), capture=True)
        self.assertIn(NORMAL_CLEAR_OF_FRAME, found.stages)
        self.assertIn(NORMAL_CLEAR_OF_FRAME, found.stage_outlines)

    def test_its_stripe_filter_refuses_a_large_parallel_line(self) -> None:
        stripe = Region((10, 10, 160, 20), np.ones((20, 160), dtype=bool), 1)
        compact = Region((10, 10, 80, 24), np.ones((24, 80), dtype=bool), 1)
        kept = clear_of_normal_stripe([stripe, compact], NormalMapConfig())
        self.assertEqual(kept, [compact])

    def test_its_stripe_filter_refuses_a_thin_oblique_line(self) -> None:
        diagonal = np.zeros((80, 80), dtype=bool)
        for index in range(5, 75):
            diagonal[index, max(index - 1, 0):index + 2] = True
        mark = Region((10, 10, 80, 80), diagonal, 1)
        broad = np.zeros((80, 80), dtype=bool)
        for index in range(8, 72):
            broad[index, max(index - 8, 0):index + 9] = True
        thick_mark = Region((10, 10, 80, 80), broad, 1)

        kept = clear_of_normal_stripe([mark, thick_mark], NormalMapConfig())

        self.assertEqual(kept, [thick_mark])

    def test_the_stripe_stage_is_captured_separately(self) -> None:
        found = detect_normal(edge_texture(), NormalMapConfig(), capture=True)
        self.assertIn(NORMAL_CLEAR_OF_STRIPE, found.stages)
        self.assertIn(NORMAL_CLEAR_OF_STRIPE, found.stage_outlines)

    def test_its_panel_filter_refuses_only_well_separated_panel_shapes(self) -> None:
        control = Region((10, 10, 50, 60), np.ones((60, 50), dtype=bool), 1)
        perimeter = np.zeros((160, 160), dtype=bool)
        perimeter[:8, :] = perimeter[-8:, :] = True
        perimeter[:, :8] = True
        panel_edge = Region((10, 10, 160, 160), perimeter, 1)
        word = Region((10, 10, 160, 29), np.ones((29, 160), dtype=bool), 1)

        kept = clear_of_normal_panel_face(
            [control, panel_edge, word], NormalMapConfig(),
        )

        self.assertEqual(kept, [word])

    def test_the_panel_stage_is_captured_separately(self) -> None:
        found = detect_normal(edge_texture(), NormalMapConfig(), capture=True)
        self.assertIn(NORMAL_CLEAR_OF_PANEL, found.stages)
        self.assertIn(NORMAL_CLEAR_OF_PANEL, found.stage_outlines)

    def test_the_shared_grouping_stage_is_captured_separately(self) -> None:
        found = detect_normal(edge_texture(), NormalMapConfig(), capture=True)
        self.assertIn(NORMAL_GROUPED_FLAT_GROUND, found.stages)
        self.assertIn(NORMAL_GROUPED_FLAT_GROUND, found.stage_outlines)
        self.assertIn(NORMAL_CONVEX_HULLED, found.stages)
        self.assertIn(NORMAL_CONVEX_HULLED, found.stage_outlines)
        self.assertIn(NORMAL_FITTED_RECTANGLE, found.stages)
        self.assertIn(NORMAL_FITTED_RECTANGLE, found.stage_outlines)
        self.assertIn(NORMAL_PADDED, found.stages)
        self.assertIn(NORMAL_PADDED, found.stage_outlines)

    def test_fitted_rectangle_is_visualisation_not_the_final_flip_mask(self) -> None:
        config = NormalMapConfig(
            min_width_px=4,
            min_height_px=4,
            min_area_px=36,
            panel_sparse_min_span_px=0,
            panel_solid_min_minor_span_px=0,
        )
        # 2026-09-03: a rectangle fit is still computed for the harness, but
        # the final region must retain its hull. Returning no fitted shapes
        # must not erase the mark production hands to correction.
        with patch(
            "glyph_lab.normal_detect.fit_region_rectangles", return_value=[]
        ):
            found = detect_normal(edge_texture(), config, capture=True)

        self.assertGreater(len(found.regions), 0)
        self.assertTrue(found.fired.any())
