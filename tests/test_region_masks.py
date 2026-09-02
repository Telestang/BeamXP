"""Carrying a mark's own texels alongside its rectangle, through the planner.

The invariant these protect: a plan whose regions are plain rectangles must
behave exactly as it did before masks existed.  Everything else here is about
what a mask buys once a detector supplies one.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mesh_segmentation_transform import mirror_texture_for_rhd as rhd
from mesh_segmentation_transform.region_masks import (
    paint_region_mask,
    region_mask_in_bounds,
    scale_region_mask,
    union_region_masks,
)


class PaintTests(unittest.TestCase):
    def test_a_none_mask_paints_its_whole_rectangle(self) -> None:
        target = np.zeros((8, 8), dtype=bool)
        self.assertTrue(paint_region_mask(target, (2, 1, 3, 4), None))
        self.assertEqual(int(target.sum()), 12)
        self.assertTrue(target[1, 2])
        self.assertFalse(target[0, 2])

    def test_the_buffer_is_cleared_between_regions(self) -> None:
        target = np.zeros((8, 8), dtype=bool)
        paint_region_mask(target, (0, 0, 4, 4), None)
        paint_region_mask(target, (4, 4, 2, 2), None)
        self.assertEqual(int(target.sum()), 4)
        self.assertFalse(target[0, 0])

    def test_a_region_clipped_off_the_atlas_paints_nothing(self) -> None:
        target = np.zeros((8, 8), dtype=bool)
        self.assertFalse(paint_region_mask(target, (20, 20, 4, 4), None))

    def test_a_region_hanging_off_the_edge_keeps_the_part_on_it(self) -> None:
        target = np.zeros((8, 8), dtype=bool)
        mask = np.zeros((4, 4), dtype=bool)
        mask[:, 3] = True                      # the column that runs off
        mask[0, 0] = True
        self.assertTrue(paint_region_mask(target, (5, 0, 4, 4), mask))
        self.assertTrue(target[0, 5])
        self.assertEqual(int(target.sum()), 1)

    def test_a_mask_of_the_wrong_shape_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            paint_region_mask(
                np.zeros((8, 8), dtype=bool), (0, 0, 4, 4),
                np.zeros((3, 3), dtype=bool),
            )

    def test_none_materialises_as_a_full_rectangle(self) -> None:
        self.assertTrue(region_mask_in_bounds(None, (0, 0, 3, 2)).all())
        self.assertEqual(region_mask_in_bounds(None, (0, 0, 3, 2)).shape, (2, 3))


class UnionTests(unittest.TestCase):
    def test_rectangles_only_stay_a_rectangle(self) -> None:
        """The invariant that keeps a maskless plan bit-identical."""
        self.assertIsNone(
            union_region_masks((0, 0, 8, 8), [((0, 0, 4, 4), None), ((4, 4, 4, 4), None)])
        )

    def test_a_masked_part_makes_the_union_carry_texels(self) -> None:
        left = np.zeros((4, 4), dtype=bool)
        left[1, 1] = True
        union = union_region_masks(
            (0, 0, 8, 4), [((0, 0, 4, 4), left), ((4, 0, 4, 4), None)]
        )
        self.assertIsNotNone(union)
        self.assertEqual(union.shape, (4, 8))
        self.assertTrue(union[1, 1])
        self.assertFalse(union[0, 0])
        # The maskless part contributes its whole rectangle.
        self.assertTrue(union[:, 4:].all())

    def test_two_masks_are_placed_in_the_union_frame(self) -> None:
        first = np.zeros((2, 2), dtype=bool)
        first[0, 0] = True
        second = np.zeros((2, 2), dtype=bool)
        second[1, 1] = True
        union = union_region_masks(
            (0, 0, 6, 6), [((0, 0, 2, 2), first), ((4, 4, 2, 2), second)]
        )
        self.assertTrue(union[0, 0])
        self.assertTrue(union[5, 5])
        self.assertEqual(int(union.sum()), 2)


class ScaleTests(unittest.TestCase):
    def test_a_none_mask_stays_none(self) -> None:
        self.assertIsNone(scale_region_mask(None, (0, 0, 8, 8), (0, 0, 4, 4), 2, 1))

    def test_a_mask_halves_with_its_bounds(self) -> None:
        mask = np.zeros((8, 8), dtype=bool)
        mask[4:, 4:] = True
        scaled = scale_region_mask(mask, (0, 0, 8, 8), (0, 0, 4, 4), 2, 1)
        self.assertEqual(scaled.shape, (4, 4))
        self.assertTrue(scaled[2:, 2:].all())
        self.assertFalse(scaled[:2, :2].any())

    def test_a_mask_that_subsamples_to_nothing_falls_back_to_its_box(self) -> None:
        """An empty stencil would silently decline the flip; None does not."""
        mask = np.zeros((8, 8), dtype=bool)
        mask[0, 0] = True          # never sampled at offset 1, stride 2
        self.assertIsNone(scale_region_mask(mask, (0, 0, 8, 8), (0, 0, 4, 4), 2, 1))


class FlatFlipTests(unittest.TestCase):
    """``apply_masked_flip`` with and without a mark's texels."""

    @staticmethod
    def scene() -> tuple[np.ndarray, np.ndarray]:
        image = np.zeros((4, 8, 3), dtype=np.uint8)
        # A mark on the left of the box and a bystander on the right, both
        # inside the same rectangle and both on mirrored geometry.
        image[1, 1] = (255, 0, 0)
        image[3, 1] = (0, 255, 0)
        stencil = np.ones((4, 8), dtype=bool)
        return image, stencil

    def test_without_a_mask_the_whole_rectangle_turns_over(self) -> None:
        image, stencil = self.scene()
        rhd.apply_masked_flip(image, stencil, (0, 0, 8, 4), "horizontal")
        self.assertEqual(tuple(image[1, 6]), (255, 0, 0))
        self.assertEqual(tuple(image[3, 6]), (0, 255, 0))

    def test_a_mask_moves_the_mark_and_leaves_the_bystander(self) -> None:
        image, stencil = self.scene()
        marked = np.zeros((4, 8), dtype=bool)
        marked[1, 1] = True                     # only the red mark
        rhd.apply_masked_flip(
            image, stencil, (0, 0, 8, 4), "horizontal", region_mask=marked,
        )
        self.assertEqual(tuple(image[1, 6]), (255, 0, 0))
        self.assertEqual(tuple(image[1, 1]), (0, 0, 0))
        # The green bystander was never part of the mark and has not moved.
        self.assertEqual(tuple(image[3, 1]), (0, 255, 0))
        self.assertEqual(tuple(image[3, 6]), (0, 0, 0))

    def test_a_mask_reflecting_onto_ground_still_carries_the_ground_back(self) -> None:
        """The union with the flip is what makes this a reflection, not a copy."""
        image, stencil = self.scene()
        image[1, 6] = (9, 9, 9)
        marked = np.zeros((4, 8), dtype=bool)
        marked[1, 1] = True
        rhd.apply_masked_flip(
            image, stencil, (0, 0, 8, 4), "horizontal", region_mask=marked,
        )
        self.assertEqual(tuple(image[1, 6]), (255, 0, 0))
        self.assertEqual(tuple(image[1, 1]), (9, 9, 9))

    def test_the_stencil_still_bounds_a_masked_flip(self) -> None:
        image, stencil = self.scene()
        stencil[:, 4:] = False                  # right half is not mirrored
        marked = np.zeros((4, 8), dtype=bool)
        marked[1, 1] = True
        moved = rhd.apply_masked_flip(
            image, stencil, (0, 0, 8, 4), "horizontal", region_mask=marked,
        )
        self.assertEqual(moved, 0)
        self.assertEqual(tuple(image[1, 1]), (255, 0, 0))


class ExchangeableShareTests(unittest.TestCase):
    def test_an_overhanging_box_is_judged_on_the_mark_not_the_overhang(self) -> None:
        """The Andronisk gear-selector "2" case, measured both ways.

        A rectangle whose right column hangs off the mirrored domain scores
        badly over the whole rectangle, while the mark inside it has a partner
        for every texel it owns.
        """
        mirror = np.zeros((4, 8), dtype=bool)
        mirror[:, :7] = True
        domain = np.ones((4, 8), dtype=bool)
        bounds = (0, 0, 8, 4)

        over_the_box = rhd.exchangeable_share(mirror, bounds, "horizontal", domain)

        marked = np.zeros((4, 8), dtype=bool)
        marked[1, 2:5] = True                   # sits clear of both edges
        over_the_mark = rhd.exchangeable_share(
            mirror, bounds, "horizontal", domain & marked,
        )
        self.assertLess(over_the_box, 1.0)
        self.assertEqual(over_the_mark, 1.0)

    def test_a_mark_that_genuinely_tears_still_scores_low(self) -> None:
        """Narrowing to the mark must not become a way to pass everything."""
        mirror = np.zeros((4, 8), dtype=bool)
        mirror[:, :4] = True
        domain = np.ones((4, 8), dtype=bool)
        marked = np.zeros((4, 8), dtype=bool)
        marked[1, 0:8] = True                   # spans the mirrored/rigid split
        share = rhd.exchangeable_share(
            mirror, (0, 0, 8, 4), "horizontal", domain & marked,
        )
        self.assertLess(share, 0.6)


class PlanReplayTests(unittest.TestCase):
    def test_a_step_mask_survives_a_half_resolution_replay(self) -> None:
        mask = np.zeros((8, 8), dtype=bool)
        mask[4:, 4:] = True
        steps = [rhd.FlipStep((0, 0, 8, 8), "horizontal", mask=mask)]
        rescaled = rhd.rescale_plan(
            steps, np.ones((16, 16), dtype=bool), None, (8, 8),
        )
        self.assertIsNotNone(rescaled)
        scaled_steps, _mirror, _labels, _domain = rescaled
        self.assertEqual(scaled_steps[0].bounds, (0, 0, 4, 4))
        self.assertEqual(scaled_steps[0].mask.shape, (4, 4))

    def test_a_maskless_step_replays_maskless(self) -> None:
        steps = [rhd.FlipStep((0, 0, 8, 8), "horizontal")]
        scaled_steps, _mirror, _labels, _domain = rhd.rescale_plan(
            steps, np.ones((16, 16), dtype=bool), None, (8, 8),
        )
        self.assertIsNone(scaled_steps[0].mask)

    def test_an_island_flip_ignores_a_mask(self) -> None:
        """A whole chart turns over; a mark's texels have no say in that."""
        image = np.zeros((4, 8, 3), dtype=np.uint8)
        image[1, 1] = (255, 0, 0)
        image[3, 1] = (0, 255, 0)
        labels = np.ones((4, 8), dtype=np.int32)
        marked = np.zeros((4, 8), dtype=bool)
        marked[1, 1] = True
        rhd.apply_flip_plan(
            image,
            [rhd.FlipStep((0, 0, 8, 4), "horizontal", island_label=1, mask=marked)],
            np.ones((4, 8), dtype=bool),
            labels,
        )
        self.assertEqual(tuple(image[1, 6]), (255, 0, 0))
        self.assertEqual(tuple(image[3, 6]), (0, 255, 0))


class DetectionPlumbingTests(unittest.TestCase):
    def test_a_detection_without_masks_keeps_its_regions_through_a_merge(self) -> None:
        """A short mask list must not truncate the zip and drop regions."""
        merged, contributed, _rotations, masks = (
            rhd.merge_region_sets_with_rotations(
                [], [(0, 1, 4, 3)], rhd.DEFAULT_RHD_CONFIG, [], [None],
            )
        )
        self.assertEqual(merged, [(0, 1, 4, 3)])
        self.assertEqual(contributed, 1)
        self.assertEqual(masks, [None])

    def test_a_dedupe_union_unions_the_texels_too(self) -> None:
        first = np.zeros((4, 4), dtype=bool)
        first[0, 0] = True
        second = np.zeros((4, 4), dtype=bool)
        second[3, 3] = True
        regions, _rotations, removed, masks = rhd.deduplicate_region_detections(
            [(0, 0, 4, 4), (2, 2, 4, 4)], [None, None], [first, second],
        )
        self.assertEqual(removed, 1)
        self.assertEqual(regions, [(0, 0, 6, 6)])
        self.assertEqual(masks[0].shape, (6, 6))
        self.assertTrue(masks[0][0, 0])
        self.assertTrue(masks[0][5, 5])
        self.assertEqual(int(masks[0].sum()), 2)

    def test_a_dedupe_of_rectangles_stays_rectangles(self) -> None:
        _regions, _rotations, _removed, masks = rhd.deduplicate_region_detections(
            [(0, 0, 4, 4), (2, 2, 4, 4)], [None, None],
        )
        self.assertEqual(masks, [None])


if __name__ == "__main__":
    unittest.main()
