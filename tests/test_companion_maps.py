"""Replaying an RHD flip plan onto a material's non-colour maps.

The base colour and the normal, roughness, AO and mask maps of one material
share a UV layout, so a mark that is both printed and moulded only survives the
correction if every map is turned over by the same plan.  A tangent-space
normal map needs one thing more: it stores a direction, so the component along
the flipped axis has to be negated or the relief comes out inverted.
"""

from __future__ import annotations

import json
import struct
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from mesh_segmentation_transform.relief_from_normals import (
    relief_field,
    shade,
    DEFAULT_RELIEF_CONFIG,
    height_from_normals,
    region_relief_report,
    render_relief,
)
from mesh_segmentation_transform.mirror_texture_for_rhd import (
    DEFAULT_RHD_CONFIG,
    CompanionResult,
    FlipStep,
    RhdTextureResult,
    apply_flip_plan,
    apply_masked_flip,
    apply_masked_rotated_flip,
    companion_boundary_blend_px,
    deduplicate_region_detections,
    exchangeable_share,
    merge_region_sets,
    normal_map_relief,
    reconstruct_normal_z,
    rescale_plan,
    rotated_axis_alignment_degrees,
    rotated_axis_for_surface_axis,
    rotated_exchangeable_share,
    source_dds_codec,
    write_blender_preview,
    write_dds,
)


def flat_normal(height: int, width: int) -> np.ndarray:
    """An RGBA normal map with every texel pointing straight out."""
    image = np.zeros((height, width, 4), dtype=np.uint8)
    image[:, :, 0] = 128
    image[:, :, 1] = 128
    image[:, :, 3] = 255
    return image


class NormalChannelNegationTests(unittest.TestCase):
    def test_horizontal_flip_negates_x_and_leaves_y(self) -> None:
        image = flat_normal(8, 8)
        image[2:6, 1:3, 0] = 200  # a slope facing +x
        image[2:6, 1:3, 1] = 60
        stencil = np.zeros((8, 8), dtype=bool)
        stencil[2:6, 0:8] = True

        apply_masked_flip(image, stencil, (0, 2, 8, 4), "horizontal", negate_channel=0)

        # The slope has moved to the mirrored columns and now faces -x, while
        # its y component is untouched: reflecting about x negates x alone.
        self.assertTrue((image[2:6, 5:7, 0] == 255 - 200).all())
        self.assertTrue((image[2:6, 5:7, 1] == 60).all())
        # Where it came from is flat again, mirrored from the flat side.
        self.assertTrue((image[2:6, 1:3, 0] == 255 - 128).all())

    def test_vertical_flip_negates_y_and_leaves_x(self) -> None:
        image = flat_normal(8, 8)
        image[1:3, 2:6, 0] = 200
        image[1:3, 2:6, 1] = 60
        stencil = np.zeros((8, 8), dtype=bool)
        stencil[0:8, 2:6] = True

        apply_masked_flip(image, stencil, (2, 0, 4, 8), "vertical", negate_channel=1)

        self.assertTrue((image[5:7, 2:6, 1] == 255 - 60).all())
        self.assertTrue((image[5:7, 2:6, 0] == 200).all())

    def test_neutral_stays_neutral_through_a_negation(self) -> None:
        # 128 and 127 are the two encodings either side of zero, so a flat
        # normal map must not drift towards a tilt when it is turned over.
        image = flat_normal(4, 4)
        stencil = np.ones((4, 4), dtype=bool)
        apply_masked_flip(image, stencil, (0, 0, 4, 4), "horizontal", negate_channel=0)
        self.assertTrue((image[:, :, 0] == 127).all())
        decoded = image[:, :, 0].astype(float) / 127.5 - 1.0
        self.assertLess(abs(float(decoded.max())), 0.01)

    def test_scalar_companion_moves_without_negation(self) -> None:
        image = np.zeros((8, 8, 4), dtype=np.uint8)
        image[:, :, 3] = 255
        image[2:6, 1:3, 0] = 200
        stencil = np.zeros((8, 8), dtype=bool)
        stencil[2:6, 0:8] = True
        steps = [FlipStep((0, 2, 8, 4), "horizontal")]

        apply_flip_plan(image, steps, stencil, None, "scalar")

        self.assertTrue((image[2:6, 5:7, 0] == 200).all())

    def test_one_plan_moves_every_map_the_same_way(self) -> None:
        colour = np.zeros((8, 8, 4), dtype=np.uint8)
        colour[:, :, 3] = 255
        colour[2:6, 1:3, :3] = 255
        normal = flat_normal(8, 8)
        normal[2:6, 1:3, 0] = 200
        stencil = np.zeros((8, 8), dtype=bool)
        stencil[2:6, 0:8] = True
        steps = [FlipStep((0, 2, 8, 4), "horizontal")]

        apply_flip_plan(colour, steps, stencil, None, "colour")
        apply_flip_plan(normal, steps, stencil, None, "normal")

        # The print and the relief land on the same texels: registration is
        # what makes replaying one plan better than detecting on each map.
        # Flat is 128 or its negation 127, so relief is judged by amplitude.
        printed = colour[:, :, 0] == 255
        relief = np.abs(normal[:, :, 0].astype(float) - 127.5) > 10
        self.assertTrue((printed == relief).all())

    def test_normal_plan_can_move_without_vector_reflection(self) -> None:
        normal = flat_normal(8, 8)
        normal[2:6, 1:3, 0] = 200
        stencil = np.zeros((8, 8), dtype=bool)
        stencil[2:6, 0:8] = True
        steps = [FlipStep((0, 2, 8, 4), "horizontal")]

        apply_flip_plan(
            normal, steps, stencil, None, "normal", reflect_normal_vectors=False
        )

        self.assertTrue((normal[2:6, 5:7, 0] == 200).all())

    def test_domain_stencil_can_flip_a_region_across_a_mask_split(self) -> None:
        image = np.zeros((1, 8, 4), dtype=np.uint8)
        image[0, :, 0] = np.arange(8, dtype=np.uint8)
        image[0, :, 3] = 255
        mirror = np.zeros((1, 8), dtype=bool)
        mirror[:, 2:6] = True
        domain = np.ones((1, 8), dtype=bool)
        steps = [FlipStep((0, 0, 8, 1), "horizontal", stencil="domain")]

        apply_flip_plan(image, steps, mirror, None, "scalar", domain)

        self.assertEqual(image[0, :, 0].tolist(), list(reversed(range(8))))

    def test_boundary_blend_softens_only_the_write_edge(self) -> None:
        image = np.zeros((6, 6, 4), dtype=np.uint8)
        image[:, :, 0] = np.arange(6, dtype=np.uint8)[None, :] * 10
        image[:, :, 3] = 255
        stencil = np.ones((6, 6), dtype=bool)

        apply_masked_flip(
            image, stencil, (0, 0, 6, 6), "horizontal", boundary_blend_px=1.5
        )

        self.assertGreater(int(image[2, 0, 0]), 0)
        self.assertLess(int(image[2, 0, 0]), 50)
        self.assertEqual(int(image[2, 3, 0]), 20)

    def test_normal_detail_gate_leaves_quiet_background_in_place(self) -> None:
        image = flat_normal(16, 16)
        image[:, :, 0] = 100 + np.arange(16, dtype=np.uint8)[None, :]
        image[6:10, 3:5, 0] = 220
        stencil = np.ones((16, 16), dtype=bool)

        apply_masked_flip(
            image,
            stencil,
            (0, 0, 16, 16),
            "horizontal",
            negate_channel=0,
            normal_detail_gate=True,
            normal_detail_floor=8.0,
        )

        self.assertEqual(int(image[1, 8, 0]), 108)
        self.assertLess(int(image[7, 11, 0]), 80)
        self.assertNotEqual(int(image[7, 3, 0]), 220)

    def test_default_normal_flip_does_not_recrop_shallow_detected_relief(self) -> None:
        image = flat_normal(16, 16)
        image[6:10, 3:5, 0] = 136
        stencil = np.ones((16, 16), dtype=bool)

        apply_masked_flip(
            image,
            stencil,
            (0, 0, 16, 16),
            "horizontal",
            negate_channel=0,
            normal_detail_gate=DEFAULT_RHD_CONFIG.normal_region_detail_gate,
            normal_detail_floor=DEFAULT_RHD_CONFIG.normal_region_detail_floor,
        )

        self.assertFalse(DEFAULT_RHD_CONFIG.normal_region_detail_gate)
        self.assertTrue((image[6:10, 11:13, 0] == 119).all())
        self.assertTrue((image[6:10, 3:5, 0] == 127).all())

    def test_normal_background_correction_preserves_quiet_mean(self) -> None:
        image = flat_normal(16, 16)
        image[:, :, 0] = 100
        image[6:10, 3:5, 0] = 220
        stencil = np.ones((16, 16), dtype=bool)

        apply_masked_flip(
            image,
            stencil,
            (0, 0, 16, 16),
            "horizontal",
            negate_channel=0,
            correct_normal_background=True,
            normal_detail_floor=8.0,
        )

        self.assertAlmostEqual(int(image[1, 8, 0]), 100, delta=1)
        self.assertLess(int(image[7, 11, 0]), 70)

    def test_scalar_detail_gate_leaves_quiet_background_in_place(self) -> None:
        image = np.zeros((16, 16, 4), dtype=np.uint8)
        image[:, :, 0] = 80 + np.arange(16, dtype=np.uint8)[None, :]
        image[:, :, 3] = 255
        image[6:10, 3:5, 0] = 180
        stencil = np.ones((16, 16), dtype=bool)

        apply_masked_flip(
            image,
            stencil,
            (0, 0, 16, 16),
            "horizontal",
            scalar_detail_gate=True,
            scalar_detail_floor=6.0,
        )

        self.assertEqual(int(image[1, 8, 0]), 88)
        self.assertGreater(int(image[7, 11, 0]), 150)
        self.assertNotEqual(int(image[7, 3, 0]), 180)

    def test_rotated_flip_moves_along_the_rectangle_axis(self) -> None:
        image = np.zeros((16, 16, 4), dtype=np.uint8)
        image[:, :, 3] = 255
        image[6, 6, 0] = 200
        stencil = np.ones((16, 16), dtype=bool)
        corners = ((5.0, 3.0), (13.0, 11.0), (11.0, 13.0), (3.0, 5.0))

        moved = apply_masked_rotated_flip(image, stencil, corners, "long")

        self.assertGreater(moved, 0)
        self.assertEqual(int(image[10, 10, 0]), 200)

    def test_rotated_flip_resamples_between_texels(self) -> None:
        image = np.zeros((16, 16, 4), dtype=np.uint8)
        image[:, :, 3] = 255
        image[5, 13, 0] = 255
        stencil = np.ones((16, 16), dtype=bool)
        corners = ((2.0, 4.0), (13.0, 5.0), (13.0, 8.0), (2.0, 7.0))

        apply_masked_rotated_flip(image, stencil, corners, "long")

        # Destination (3,4) reflects to roughly (12.29, 4.84).  Nearest-neighbour
        # sampling would read the dark texel at (12,5); filtered resampling
        # picks up part of the bright texel at (13,5).
        self.assertGreater(int(image[4, 3, 0]), 20)

    def test_rotated_normal_flip_reflects_the_vector_direction(self) -> None:
        image = flat_normal(16, 16)
        image[6, 6, 0] = 200
        image[6, 6, 1] = 200
        stencil = np.ones((16, 16), dtype=bool)
        corners = ((5.0, 3.0), (13.0, 11.0), (11.0, 13.0), (3.0, 5.0))

        apply_masked_rotated_flip(image, stencil, corners, "long", True)

        self.assertLess(int(image[10, 10, 0]), 128)
        self.assertLess(int(image[10, 10, 1]), 128)

    def test_near_axis_rotated_rectangle_can_be_snapped(self) -> None:
        corners = ((0.0, 0.0), (100.0, 1.0), (100.0, 11.0), (0.0, 10.0))
        rotated_axis = rotated_axis_for_surface_axis(corners, "horizontal")
        self.assertEqual(rotated_axis, "long")
        alignment = rotated_axis_alignment_degrees(
            corners, "horizontal", rotated_axis
        )
        self.assertIsNotNone(alignment)
        self.assertLess(alignment, DEFAULT_RHD_CONFIG.rotated_axis_snap_degrees)


class ExchangeableShareTests(unittest.TestCase):
    """What a region risks tearing is measured over what it can write.

    ``apply_masked_flip`` exchanges only texels whose partner is also inside the
    stencil, and never touches anything outside it, so a texel the material
    does not paint at all cannot tear. Counting the region's whole rectangle
    instead cost the Andronisk door panel its gear-selector "2": the detection
    box overhangs that glyph's UV island by 5%, and the overhang alone put it
    under the 0.98 floor while only 11 of its 2,397 painted texels really
    lacked a partner.
    """

    def island_overhung_by_its_region(self) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """A symmetric island a little smaller than the box detected over it."""
        domain = np.zeros((20, 20), dtype=bool)
        domain[3:17, 3:17] = True
        return domain, (1, 1, 18, 18)

    def test_an_overhanging_region_is_judged_on_the_island_it_paints(self) -> None:
        domain, bounds = self.island_overhung_by_its_region()
        self.assertLess(exchangeable_share(domain, bounds, "horizontal"), 0.98)
        self.assertEqual(
            exchangeable_share(domain, bounds, "horizontal", domain), 1.0
        )

    def test_a_region_split_between_mirrored_and_rigid_still_scores_low(self) -> None:
        # The half outside the mirror mask is inside the domain and does tear,
        # so narrowing the denominator must not excuse it.
        domain = np.zeros((20, 20), dtype=bool)
        domain[3:17, 3:17] = True
        mirror = np.zeros_like(domain)
        mirror[3:17, 3:10] = True
        share = exchangeable_share(mirror, (1, 1, 18, 18), "horizontal", domain)
        self.assertLess(share, 0.98)

    def test_a_stencil_that_paints_nothing_here_exchanges_nothing(self) -> None:
        domain = np.zeros((20, 20), dtype=bool)
        domain[3:17, 3:17] = True
        self.assertEqual(
            exchangeable_share(domain, (0, 0, 2, 2), "horizontal", domain), 0.0
        )

    def test_the_rotated_measure_narrows_the_same_way(self) -> None:
        domain = np.zeros((20, 20), dtype=bool)
        domain[3:17, 3:17] = True
        # Centred on the island, so only the overhang separates the two answers.
        corners = ((1.0, 1.0), (18.0, 1.0), (18.0, 18.0), (1.0, 18.0))
        self.assertLess(rotated_exchangeable_share(domain, corners, "long"), 0.98)
        self.assertEqual(
            rotated_exchangeable_share(domain, corners, "long", domain), 1.0
        )


class PlanRescalingTests(unittest.TestCase):
    def test_exact_half_scale_is_restated(self) -> None:
        mask = np.zeros((16, 16), dtype=bool)
        mask[4:12, 4:12] = True
        steps = [FlipStep((4, 4, 8, 8), "horizontal")]

        rescaled = rescale_plan(steps, mask, None, (8, 8))

        self.assertIsNotNone(rescaled)
        scaled_steps, scaled_mask, _labels, _domain = rescaled  # type: ignore[misc]
        self.assertEqual(scaled_steps[0].bounds, (2, 2, 4, 4))
        self.assertEqual(scaled_steps[0].axis, "horizontal")
        self.assertEqual(scaled_mask.shape, (8, 8))
        self.assertEqual(int(scaled_mask.sum()), 16)

    def test_rotated_corners_are_rescaled_with_the_plan(self) -> None:
        mask = np.ones((16, 16), dtype=bool)
        corners = ((4.0, 2.0), (12.0, 10.0), (10.0, 12.0), (2.0, 4.0))
        steps = [FlipStep((2, 2, 12, 12), "horizontal", None, corners, "long")]

        rescaled = rescale_plan(steps, mask, None, (8, 8))

        self.assertIsNotNone(rescaled)
        scaled_steps, _mask, _labels, _domain = rescaled  # type: ignore[misc]
        self.assertEqual(
            scaled_steps[0].rotated_corners,
            ((2.0, 1.0), (6.0, 5.0), (5.0, 6.0), (1.0, 2.0)),
        )
        self.assertEqual(scaled_steps[0].rotated_axis, "long")

    def test_same_size_passes_the_plan_through_untouched(self) -> None:
        mask = np.ones((8, 8), dtype=bool)
        steps = [FlipStep((1, 1, 4, 4), "vertical")]
        rescaled = rescale_plan(steps, mask, None, (8, 8))
        self.assertIsNotNone(rescaled)
        self.assertIs(rescaled[0], steps)  # type: ignore[index]

    def test_rescale_keeps_the_step_stencil_and_scales_domain_mask(self) -> None:
        mirror = np.zeros((16, 16), dtype=bool)
        mirror[4:12, 4:12] = True
        domain = np.ones((16, 16), dtype=bool)
        steps = [FlipStep((4, 4, 8, 8), "horizontal", stencil="domain")]

        rescaled = rescale_plan(steps, mirror, None, (8, 8), domain)

        self.assertIsNotNone(rescaled)
        scaled_steps, _mirror, _labels, scaled_domain = rescaled  # type: ignore[misc]
        self.assertEqual(scaled_steps[0].stencil, "domain")
        self.assertIsNotNone(scaled_domain)
        self.assertEqual(scaled_domain.shape, (8, 8))
        self.assertTrue(scaled_domain.all())

    def test_companion_boundary_blend_keeps_a_low_res_minimum(self) -> None:
        self.assertEqual(
            companion_boundary_blend_px(DEFAULT_RHD_CONFIG, (2048, 2048), (4096, 4096)),
            DEFAULT_RHD_CONFIG.companion_boundary_blend_min_px,
        )

    def test_a_ratio_that_is_not_exact_is_declined(self) -> None:
        # 12 does not divide 16, so the reflection axis would fall between
        # texels.  Refusing beats flipping about the wrong line.
        mask = np.ones((16, 16), dtype=bool)
        self.assertIsNone(rescale_plan([FlipStep((0, 0, 4, 4), "horizontal")],
                                       mask, None, (12, 12)))

    def test_a_ratio_differing_per_axis_is_declined(self) -> None:
        mask = np.ones((16, 16), dtype=bool)
        self.assertIsNone(rescale_plan([FlipStep((0, 0, 4, 4), "horizontal")],
                                       mask, None, (8, 16)))


class ReliefRenderTests(unittest.TestCase):
    """The render the RHD run would detect on, if relief detection were on.

    It is off by default and its tuning is unfinished; these pin the contract
    the harness and the production path share, not any particular tuning.
    """

    def _flat(self, size: int = 64) -> np.ndarray:
        normal = np.zeros((size, size, 3), dtype=np.uint8)
        normal[:, :, 0] = 128
        normal[:, :, 1] = 128
        return normal

    def test_the_rhd_run_renders_through_the_shared_module(self) -> None:
        # The harness tunes relief_from_normals; production must be that same
        # code, or what gets tuned is not what ships.
        normal = self._flat()
        normal[20:40, 20:40, 0] = 190
        self.assertTrue(
            (
                normal_map_relief(normal, DEFAULT_RHD_CONFIG)
                == render_relief(normal, DEFAULT_RHD_CONFIG.relief)
            ).all()
        )

    def test_slope_mode_leaves_flat_material_at_the_base_level(self) -> None:
        config = replace(
            DEFAULT_RELIEF_CONFIG, mode="slope", grain_blur_sigma=0.0,
            form_blur_sigma=0.0, global_scale_percentile=100.0, gain=127.0,
            invert=False,
        )
        normal = self._flat(8)
        normal[3:5, 3:5, 0] = 128 + 90
        relief = render_relief(normal, config)
        self.assertEqual(relief.shape, (8, 8, 3))
        self.assertGreater(int(relief[3, 3, 0]), int(relief[0, 0, 0]))

    def test_height_mode_renders_a_step_away_from_the_flat_level(self) -> None:
        # A moulded mark must come out as a plateau distinguishable from the
        # material around it; that is the whole reason for integrating.
        normal = self._flat()
        normal[24:40, 24:40, 0] = 170
        relief = render_relief(normal, replace(DEFAULT_RELIEF_CONFIG, mode="height"))
        self.assertEqual(relief.shape, (64, 64, 3))
        self.assertNotEqual(int(relief[32, 32, 0]), int(relief[2, 2, 0]))

    def test_the_three_channels_are_identical(self) -> None:
        normal = self._flat()
        normal[20:40, 20:40, 1] = 200
        relief = render_relief(normal, DEFAULT_RELIEF_CONFIG)
        self.assertTrue((relief[:, :, 0] == relief[:, :, 1]).all())
        self.assertTrue((relief[:, :, 1] == relief[:, :, 2]).all())

    def test_invert_mirrors_the_render_about_the_flat_level(self) -> None:
        # Engraving and embossing differ only in sign, so a detector keyed to
        # one direction needs the other offered to it.
        normal = self._flat()
        normal[24:40, 24:40, 0] = 170
        config = replace(DEFAULT_RELIEF_CONFIG, invert=False)
        straight = render_relief(normal, config).astype(int)
        flipped = render_relief(
            normal, replace(config, invert=True)
        ).astype(int)
        self.assertLessEqual(int(np.abs((straight - 128) + (flipped - 128)).max()), 1)

    def test_every_stage_can_be_switched_off(self) -> None:
        config = replace(
            DEFAULT_RELIEF_CONFIG, grain_blur_sigma=0.0, form_blur_sigma=0.0,
            local_scale_sigma=0.0, clahe_clip_limit=0.0,
        )
        render_relief(self._flat(), config)  # must not raise


class ShadedReliefTests(unittest.TestCase):
    """Shading is the mode that works, because the shadow is the signal.

    A moulded stroke is not a different colour and, to look at, not a different
    height: it is a bright edge beside a dark one.
    """

    def _embossed(self, size: int = 128) -> np.ndarray:
        """A normal map for a raised square, as an authoring tool would emit."""
        truth = np.zeros((size, size), dtype=np.float32)
        truth[48:80, 48:80] = 1.0
        truth = cv2.GaussianBlur(truth, (0, 0), 2.0)
        dy, dx = np.gradient(truth)
        x, y, z = -dx, -dy, np.ones_like(truth)
        length = np.sqrt(x * x + y * y + z * z)
        normal = np.zeros((size, size, 3), dtype=np.uint8)
        normal[:, :, 0] = np.clip((x / length + 1) * 127.5, 0, 255).astype(np.uint8)
        normal[:, :, 1] = np.clip((y / length + 1) * 127.5, 0, 255).astype(np.uint8)
        return normal

    def test_flat_material_shades_to_zero(self) -> None:
        # Flat has to cancel exactly, or the ring-background model that every
        # downstream filter rests on has no background to find.
        flat = np.zeros((32, 32, 3), dtype=np.uint8)
        flat[:, :, 0] = 128
        flat[:, :, 1] = 128
        self.assertLess(float(np.abs(shade(flat, DEFAULT_RELIEF_CONFIG)).max()), 0.02)

    def test_an_edge_is_lit_on_one_side_and_shadowed_on_the_other(self) -> None:
        field = shade(self._embossed(), DEFAULT_RELIEF_CONFIG)
        self.assertGreater(float(field.max()), 0.05)   # a face towards the light
        self.assertLess(float(field.min()), -0.05)     # and one turned away

    # Deliberately not tested here: that shading beats height on a mark.  It
    # does on the ardente -- prominence 0.30 against 0.03 -- but that gap comes
    # from the trim seam sitting beside the lettering, and on a synthetic
    # square with nothing else in frame integration wins instead (1.37 to
    # 1.26).  Asserting the ordering on a fixture would pin a claim that is
    # only true of real, cluttered texture.

    def test_more_lights_rescue_a_stroke_aligned_with_the_first(self) -> None:
        # One light leaves a stroke running along it unlit; the sweep exists so
        # no stroke is missed for its direction alone.
        ridge = np.zeros((64, 64), dtype=np.float32)
        ridge[:, 28:36] = 1.0                      # a vertical ridge
        ridge = cv2.GaussianBlur(ridge, (0, 0), 2.0)
        dy, dx = np.gradient(ridge)
        x, y, z = -dx, -dy, np.ones_like(ridge)
        length = np.sqrt(x * x + y * y + z * z)
        normal = np.zeros((64, 64, 3), dtype=np.uint8)
        normal[:, :, 0] = np.clip((x / length + 1) * 127.5, 0, 255).astype(np.uint8)
        normal[:, :, 1] = np.clip((y / length + 1) * 127.5, 0, 255).astype(np.uint8)

        along = replace(DEFAULT_RELIEF_CONFIG, light_azimuth_degrees=90.0)
        swept = replace(along, light_count=4)
        self.assertGreater(
            float(np.abs(shade(normal, swept)).max()),
            float(np.abs(shade(normal, along)).max()),
        )

    def test_shading_needs_no_integration(self) -> None:
        # Worth keeping honest: shaded is ~0.8s on a 4k map where height is ~6s,
        # because it never takes an FFT.
        normal = self._embossed()
        config = replace(DEFAULT_RELIEF_CONFIG, mode="shaded")
        self.assertTrue(
            (
                relief_field(normal, config)
                == shade(normal, config)
            ).all()
        )


class ReliefFieldTests(unittest.TestCase):
    def test_height_recovers_a_plateau_from_the_normals_it_would_produce(self) -> None:
        """Round-trip a known moulded shape: height -> normals -> height.

        A globally constant gradient is deliberately not the test.  The
        integration is periodic (Frankot-Chellappa), so a gradient that never
        returns puts all its energy at DC, which is unconstrained and zeroed --
        correct behaviour, and irrelevant, because a moulded mark is by
        definition local.
        """
        size = 128
        truth = np.zeros((size, size), dtype=np.float32)
        truth[40:88, 40:88] = 1.0
        truth = cv2.GaussianBlur(truth, (0, 0), 3.0)

        # Encode the surface that height implies as a tangent-space normal map.
        dy, dx = np.gradient(truth)
        x, y, z = -dx, -dy, np.ones_like(truth)
        length = np.sqrt(x * x + y * y + z * z)
        normal = np.zeros((size, size, 3), dtype=np.uint8)
        normal[:, :, 0] = np.clip((x / length + 1) * 127.5, 0, 255).astype(np.uint8)
        normal[:, :, 1] = np.clip((y / length + 1) * 127.5, 0, 255).astype(np.uint8)

        height = height_from_normals(normal)
        inside = float(height[56:72, 56:72].mean())
        outside = float(
            np.concatenate([height[:16, :16].ravel(), height[-16:, -16:].ravel()]).mean()
        )
        self.assertGreater(inside - outside, 0.5)

    def test_report_measures_a_mark_against_its_surround(self) -> None:
        normal = np.zeros((256, 256, 3), dtype=np.uint8)
        normal[:, :, 0] = 128
        normal[:, :, 1] = 128
        normal[120:136, 100:160, 0] = 190
        report = region_relief_report(normal, (100, 120, 60, 16))
        self.assertIn("prominence", report)
        self.assertGreater(report["inner_spread"], 0.0)

    def test_report_on_an_off_image_rectangle_says_nothing(self) -> None:
        normal = np.full((32, 32, 3), 128, dtype=np.uint8)
        self.assertEqual(region_relief_report(normal, (100, 100, 4, 4)), {})


class RegionMergeTests(unittest.TestCase):
    def test_duplicate_atlas_regions_from_multiple_uv_domains_collapse(self) -> None:
        regions, rotations, removed = deduplicate_region_detections(
            [(163, 241, 58, 56), (162, 241, 59, 56), (320, 200, 20, 20)]
        )

        self.assertEqual(removed, 1)
        self.assertEqual(regions, [(320, 200, 20, 20), (162, 241, 59, 56)])
        self.assertEqual(rotations, [None, None])

    def test_distinct_overlapping_atlas_regions_are_collapsed(self) -> None:
        regions, _rotations, removed = deduplicate_region_detections(
            [(100, 100, 40, 20), (120, 100, 40, 20)]
        )

        self.assertEqual(removed, 1)
        self.assertEqual(regions, [(100, 100, 60, 20)])

    def test_region_collapse_repeats_when_union_creates_new_overlap(self) -> None:
        regions, rotations, removed = deduplicate_region_detections(
            [(0, 0, 10, 10), (8, 8, 10, 10), (16, 0, 5, 7)]
        )

        self.assertEqual(removed, 2)
        self.assertEqual(regions, [(0, 0, 21, 18)])
        self.assertEqual(rotations, [None])

    def test_a_relief_mark_of_its_own_is_added(self) -> None:
        merged, added = merge_region_sets(
            [(0, 0, 10, 10)], [(50, 50, 10, 10)], DEFAULT_RHD_CONFIG
        )
        self.assertEqual(added, 1)
        self.assertEqual(sorted(merged), [(0, 0, 10, 10), (50, 50, 10, 10)])

    def test_an_overlapping_relief_mark_unions_rather_than_flipping_twice(self) -> None:
        merged, added = merge_region_sets(
            [(10, 10, 20, 20)], [(20, 20, 20, 20)], DEFAULT_RHD_CONFIG
        )
        self.assertEqual(added, 1)
        self.assertEqual(merged, [(10, 10, 30, 30)])

    def test_relief_far_larger_than_the_print_leaves_the_print_alone(self) -> None:
        # A moulded pad around a small printed icon: the union is the pad, and
        # flipping that would move the pad's edges and still not fix the icon.
        merged, added = merge_region_sets(
            [(100, 100, 10, 10)], [(0, 0, 300, 300)], DEFAULT_RHD_CONFIG
        )
        self.assertEqual(added, 0)
        self.assertEqual(merged, [(100, 100, 10, 10)])


class NormalPreviewTests(unittest.TestCase):
    """A two-channel normal map needs its z back before Blender can shade it."""

    def test_flat_normal_reconstructs_to_straight_out(self) -> None:
        preview = reconstruct_normal_z(flat_normal(4, 4))
        self.assertTrue((preview[:, :, 2] == 255).all())

    def test_a_slope_reconstructs_to_a_unit_vector(self) -> None:
        image = flat_normal(4, 4)
        image[:, :, 0] = 200
        preview = reconstruct_normal_z(image)
        vector = preview[0, 0, :3].astype(float) / 127.5 - 1.0
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=2)

    def test_x_and_y_are_carried_through_untouched(self) -> None:
        image = flat_normal(4, 4)
        image[:, :, 0] = 200
        image[:, :, 1] = 60
        preview = reconstruct_normal_z(image)
        self.assertTrue((preview[:, :, 0] == 200).all())
        self.assertTrue((preview[:, :, 1] == 60).all())


class BlenderPreviewTests(unittest.TestCase):
    """Blender's COLLADA importer takes the diffuse texture and drops the rest,
    so the maps have to be attached after import, from a manifest."""

    def _result(self, directory: Path) -> RhdTextureResult:
        return RhdTextureResult(
            texture_member="vehicles/x/x_b.color.dds",
            size=(64, 64),
            parts_analysed=1,
            mirrored_triangles=1,
            rigid_triangles=0,
            mirror_coverage=0.5,
            rigid_coverage=0.0,
            conflict_coverage=0.0,
            glyph_regions=1,
            mirrored_glyph_regions=1,
            material_aliases=("x_interior", "x_interior-material"),
            png_path=directory / "x_b.color_rhd.png",
            companions=[
                CompanionResult(
                    member="vehicles/x/x_nm.normal.dds",
                    stage_key="normalMap",
                    kind="normal",
                    codec="bc5",
                    texels_moved=10,
                    png_path=directory / "x_nm.normal_rhd.png",
                    preview_path=directory / "x_nm.normal_rhd.preview.png",
                ),
            ],
        )

    def test_manifest_lists_the_aliases_and_the_maps(self) -> None:
        directory = Path(tempfile.mkdtemp(prefix="beamxp_blender_test_"))
        script = write_blender_preview(directory, [self._result(directory)],
                                       log=lambda *_a: None)
        self.assertIsNotNone(script)
        manifest = json.loads((directory / "rhd_materials.json").read_text())
        entry = manifest["materials"][0]
        self.assertEqual(entry["aliases"], ["x_interior", "x_interior-material"])
        self.assertEqual(entry["maps"]["baseColorMap"], "x_b.color_rhd.png")
        # The z-reconstructed preview, not the exact two-channel PNG: Blender's
        # Normal Map node reads blue as given and would shade it flat.
        self.assertEqual(
            entry["maps"]["normalMap"], "x_nm.normal_rhd.preview.png"
        )

    def test_the_emitted_script_is_valid_python(self) -> None:
        directory = Path(tempfile.mkdtemp(prefix="beamxp_blender_test_"))
        write_blender_preview(directory, [self._result(directory)],
                              log=lambda *_a: None)
        source = (directory / "blender_preview.py").read_text(encoding="utf-8")
        compile(source, "blender_preview.py", "exec")

    def test_a_run_with_no_material_aliases_writes_nothing(self) -> None:
        directory = Path(tempfile.mkdtemp(prefix="beamxp_blender_test_"))
        self.assertIsNone(
            write_blender_preview(directory, [], log=lambda *_a: None)
        )
        self.assertFalse((directory / "rhd_materials.json").exists())


class DdsCodecTests(unittest.TestCase):
    """BeamNG authors one channel per purpose; the rebuild has to match it."""

    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp(prefix="beamxp_dds_test_"))
        self.image = np.zeros((16, 16, 4), dtype=np.uint8)
        self.image[:, :, 0] = 128
        self.image[:, :, 1] = 128
        self.image[:, :, 3] = 255
        self.image[4:12, 4:12, 0] = 200
        self.image[4:12, 4:12, 1] = 60

    def test_bc5_carries_both_normal_channels_exactly(self) -> None:
        path = self.directory / "normal.dds"
        info = write_dds(path, self.image, "bc5")
        self.assertEqual(info["codec"], "bc5")
        back = np.asarray(Image.open(path).convert("RGBA"))
        # BC5 stores flat blocks exactly, so a two-value map round-trips.
        self.assertTrue((back[:, :, 0] == self.image[:, :, 0]).all())
        self.assertTrue((back[:, :, 1] == self.image[:, :, 1]).all())

    def test_bc4_carries_the_single_channel_exactly(self) -> None:
        path = self.directory / "data.dds"
        info = write_dds(path, self.image, "bc4")
        self.assertEqual(info["codec"], "bc4")
        back = np.asarray(Image.open(path).convert("L"))
        self.assertTrue((back == self.image[:, :, 0]).all())

    def test_every_codec_names_itself_on_the_way_back_in(self) -> None:
        for codec in ("bc7", "bc7_srgb", "bc5", "bc4"):
            path = self.directory / f"{codec}.dds"
            write_dds(path, self.image, codec)
            self.assertEqual(source_dds_codec(path), codec, codec)

    def test_an_unreadable_source_falls_back_to_bc7(self) -> None:
        path = self.directory / "not.dds"
        path.write_bytes(b"not a dds at all")
        self.assertEqual(source_dds_codec(path), "bc7")

    def test_bc4_is_half_the_size_of_bc5(self) -> None:
        four = write_dds(self.directory / "four.dds", self.image, "bc4")
        five = write_dds(self.directory / "five.dds", self.image, "bc5")
        self.assertLess(int(four["bytes"]), int(five["bytes"]))

    def test_header_and_mip_payload_match_beamng_expectations(self) -> None:
        path = self.directory / "colour.dds"
        write_dds(path, self.image, "bc7_srgb")
        data = path.read_bytes()
        header = struct.unpack("<31I", data[4:128])
        fourcc = header[20].to_bytes(4, "little")

        self.assertEqual(header[5], 1)  # depth
        self.assertEqual(header[6], 5)  # 16, 8, 4, 2, 1
        self.assertEqual(header[26], 0x401008)  # texture | complex | mipmap
        self.assertEqual(fourcc, b"DX10")
        self.assertEqual(struct.unpack("<5I", data[128:148])[0], 99)
        self.assertEqual(len(data), 148 + 256 + 64 + 16 + 16 + 16)


if __name__ == "__main__":
    unittest.main()


class BlenderDiscoveryTests(unittest.TestCase):
    """Baking a .blend is a convenience; its absence must not fail an export."""

    def test_an_explicit_override_is_used(self) -> None:
        import os
        from unittest import mock
        from mesh_segmentation_transform.mirror_texture_for_rhd import (
            BLENDER_ENVIRONMENT_VARIABLE, find_blender,
        )
        directory = Path(tempfile.mkdtemp(prefix="beamxp_blender_"))
        fake = directory / "blender.exe"
        fake.write_text("", encoding="utf-8")
        with mock.patch.dict(os.environ, {BLENDER_ENVIRONMENT_VARIABLE: str(fake)}):
            self.assertEqual(find_blender(), fake)

    def test_a_bad_override_falls_through_rather_than_raising(self) -> None:
        import os
        from unittest import mock
        from mesh_segmentation_transform.mirror_texture_for_rhd import (
            BLENDER_ENVIRONMENT_VARIABLE, find_blender,
        )
        with mock.patch.dict(
            os.environ, {BLENDER_ENVIRONMENT_VARIABLE: "/no/such/blender"}
        ):
            found = find_blender()
        self.assertTrue(found is None or found.is_file())

    def test_a_missing_blender_reports_the_manual_command(self) -> None:
        from mesh_segmentation_transform.mirror_texture_for_rhd import (
            bake_blender_scene,
        )
        directory = Path(tempfile.mkdtemp(prefix="beamxp_blender_"))
        lines: list[str] = []
        result = bake_blender_scene(
            directory / "blender_preview.py", directory / "part.dae",
            directory / "out.blend",
            blender=Path("/nonexistent/blender"), log=lines.append,
        )
        self.assertIsNone(result)
        self.assertTrue(any("did not run" in line or "--dae" in line for line in lines))
