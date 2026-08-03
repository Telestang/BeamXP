"""The edge front-end, for relief that MSER cannot see.

MSER looks for maximally stable extremal regions -- patches of one intensity
against another -- which is exactly what a painted glyph is and exactly what a
moulded one is not.  The trim inside a moulded stroke is the same material, at
the same height, as the trim outside it; what exists is the pair of edges where
the surface steps up and back down.  So MSER has nothing to be stable about and
returns nothing, while a kernel gradient keys straight onto the thing that is
there, for one convolution instead of a region search.
"""

from __future__ import annotations

import unittest
from dataclasses import replace

import cv2
import numpy as np

from mesh_segmentation_transform.annotate_texture_regions import (
    DEFAULT_CONFIG,
    detect_edge_boxes,
    detect_mser_boxes,
    edge_response,
    run_detection,
)


LETTER_BAND = (40, 200)  # rows the lettering occupies in the fixtures below


def embossed_text(
    size: int = 256,
    amplitude: float = 3.0,
    seam: bool = False,
) -> np.ndarray:
    """A shaded-relief image with the amplitudes the real ones have.

    Built the way the relief render delivers one: flat material at a constant
    level, and a stroke present only as a bright edge beside a dark one with
    untouched material between them.  Filling the strokes in would make it a
    picture of text and test nothing, because that is the case MSER already
    handles.

    ``amplitude`` is levels either side of flat.  The default of 3 is taken
    from the ardente, where the "ARDENTE" lettering spans about 4 levels of the
    relief field.  ``seam`` adds a trim edge at 120 levels, the order the real
    one answers at, which is what makes a single global threshold hopeless.
    """
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.putText(mask, "ABC", (40, size // 2), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 255, 6)
    outline = cv2.morphologyEx(
        mask, cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    ).astype(np.float32) / 255.0
    lit = cv2.GaussianBlur(outline, (0, 0), 1.2)
    shading = cv2.Scharr(lit, cv2.CV_32F, 1, 0)
    shading /= max(float(np.abs(shading).max()), 1e-6)
    field = shading * amplitude
    if seam:
        field[:12, :] = 120.0  # a trim edge, far louder than any lettering
    grey = np.clip(128 + field, 0, 255).astype(np.uint8)
    return np.repeat(grey[:, :, None], 3, axis=2)


def in_letter_band(boxes) -> list:
    low, high = LETTER_BAND
    return [b for b in boxes if low < b[1] + b[3] // 2 < high]


EDGE_CONFIG = replace(
    DEFAULT_CONFIG, box_source="edge", edge_local_window_px=64, edge_local_k=1.6
)


class EdgeResponseTests(unittest.TestCase):
    def test_flat_material_has_no_gradient(self) -> None:
        flat = np.full((64, 64), 128, dtype=np.uint8)
        self.assertLess(float(edge_response(flat, DEFAULT_CONFIG).max()), 1.0)

    def test_a_step_answers_at_the_step(self) -> None:
        image = np.full((64, 64), 128, dtype=np.uint8)
        image[:, 32:] = 200
        response = edge_response(image, DEFAULT_CONFIG)
        self.assertGreater(float(response[:, 28:36].max()), 10.0)
        self.assertLess(float(response[:, :20].max()), 1.0)

    def test_scharr_answers_alike_to_a_stroke_at_any_angle(self) -> None:
        # Moulded lettering has strokes at every angle, so an operator that
        # answers differently by direction finds some letters and loses others.
        answers = []
        for angle in (0, 30, 45, 60, 90):
            image = np.zeros((128, 128), dtype=np.uint8)
            centre = (64, 64)
            length = 40
            radians = np.radians(angle)
            offset = (int(length * np.cos(radians)), int(length * np.sin(radians)))
            cv2.line(
                image,
                (centre[0] - offset[0], centre[1] - offset[1]),
                (centre[0] + offset[0], centre[1] + offset[1]),
                255, 3,
            )
            answers.append(float(edge_response(image, DEFAULT_CONFIG).max()))
        self.assertLess(max(answers) / min(answers), 1.25)


class EdgeBoxTests(unittest.TestCase):
    def test_edges_find_moulded_text_where_mser_finds_none(self) -> None:
        # At the amplitude real moulded lettering has, MSER's delta of 12 spans
        # the whole signal, so there is no threshold range over which anything
        # is stable and it returns nothing at all.  A gradient does not care
        # how faint the step is, only that there is one.
        image = embossed_text()
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        edges = detect_edge_boxes(grey, None, EDGE_CONFIG)
        mser = detect_mser_boxes(grey, DEFAULT_CONFIG)

        self.assertGreater(len(in_letter_band(edges)), 0,
                           "the edge detector found no strokes")
        self.assertEqual(len(in_letter_band(mser)), 0,
                         "MSER was expected to find nothing at this amplitude")

    def test_flat_material_yields_nothing(self) -> None:
        flat = np.full((128, 128), 128, dtype=np.uint8)
        self.assertEqual(len(detect_edge_boxes(flat, None, EDGE_CONFIG)), 0)

    def test_the_uv_domain_is_respected(self) -> None:
        image = embossed_text()
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        domain = np.zeros(grey.shape, dtype=bool)
        domain[:, :64] = True
        for x, _y, w, _h in detect_edge_boxes(grey, domain, EDGE_CONFIG):
            self.assertLess(x, 64 + w, "a box was found outside the UV domain")

    def test_closing_joins_a_stroke_into_one_mark(self) -> None:
        # Without it every letter arrives as a pair of thin rings and grouping
        # has twice as much to join.
        image = embossed_text()
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        joined = detect_edge_boxes(grey, None, replace(EDGE_CONFIG, edge_close_px=9))
        split = detect_edge_boxes(grey, None, replace(EDGE_CONFIG, edge_close_px=0))
        self.assertLessEqual(len(joined), len(split))

    def test_a_local_threshold_survives_something_louder_nearby(self) -> None:
        """The ardente case: shallow lettering beside a very loud trim seam.

        A global threshold set high enough to exclude the seam also excludes the
        lettering; measured on the real map, a 97th-percentile cut found the
        word along with thousands of others and a 99th found neither it nor
        AIRBAG.  Compared against its own neighbourhood the word still stands.
        """
        grey = cv2.cvtColor(embossed_text(seam=True), cv2.COLOR_BGR2GRAY)

        local = detect_edge_boxes(grey, None, EDGE_CONFIG)
        globally = detect_edge_boxes(
            grey, None,
            replace(EDGE_CONFIG, edge_local_window_px=0, edge_threshold_percentile=99.0),
        )
        self.assertEqual(len(in_letter_band(globally)), 0,
                         "a global threshold was expected to lose the lettering")
        self.assertGreater(len(in_letter_band(local)), 0,
                           "a local threshold should still find it")


class PipelineIntegrationTests(unittest.TestCase):
    def test_the_colour_path_is_untouched_by_default(self) -> None:
        # box_source defaults to mser, so every existing colour run must be
        # bit-for-bit what it was before the edge front-end existed.
        self.assertEqual(DEFAULT_CONFIG.box_source, "mser")
        image = embossed_text()
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        run = run_detection(image, None, DEFAULT_CONFIG)
        self.assertEqual(run.stages[0].title, "MSER boxes")
        self.assertEqual(
            len(run.stages[0].kept), len(detect_mser_boxes(grey, DEFAULT_CONFIG))
        )

    def test_the_edge_source_renames_the_first_stage(self) -> None:
        run = run_detection(embossed_text(), None, EDGE_CONFIG)
        self.assertEqual(run.stages[0].title, "Edge boxes")
        self.assertIn("gradient", run.stages[0].detail)

    def test_changing_an_edge_parameter_reruns_from_the_first_step(self) -> None:
        # Otherwise the harness would show stale boxes while it is being tuned.
        image = embossed_text()
        first = run_detection(image, None, EDGE_CONFIG)
        second = run_detection(
            image, None, replace(EDGE_CONFIG, edge_local_k=3.0), first
        )
        self.assertEqual(second.resumed_from, 0)


if __name__ == "__main__":
    unittest.main()


class StrokeWidthTests(unittest.TestCase):
    """SWT is the text/not-text test: a letter is drawn with one pen, a seam
    is not, however strong an edge it makes."""

    def _bar(self, width_px: int, size: int = 128) -> np.ndarray:
        """A raised band of known width on flat background, height-like."""
        image = np.zeros((size, size), dtype=np.uint8)
        image[:, 40 : 40 + width_px] = 200
        return image

    def test_a_wider_bar_reads_wider(self) -> None:
        """Monotonicity, not absolute accuracy, is what this has to have.

        The measured width under-reads: the gradient band around an edge is
        several texels thick, so a ray starts partway in and a 10 px bar comes
        back at about 7.  That does not matter, because nothing here compares a
        width against a real-world number -- the decision rests on how
        *consistent* the widths within one region are.
        """
        from mesh_segmentation_transform.annotate_texture_regions import (
            stroke_width_stats, stroke_width_transform,
        )
        config = replace(EDGE_CONFIG, enable_stroke_width_filter=True)
        measured = []
        for width_px in (6, 12, 20):
            widths = stroke_width_transform(self._bar(width_px), None, config)
            median, _variation, coverage = stroke_width_stats(
                widths, (30, 20, 40, 80)
            )
            self.assertGreater(coverage, 0.0, f"no stroke found for {width_px} px")
            measured.append(median)
        self.assertEqual(measured, sorted(measured))
        self.assertLess(measured[0], measured[-1])

    def test_a_consistent_stroke_varies_less_than_a_tapering_one(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            stroke_width_stats, stroke_width_transform,
        )
        config = replace(EDGE_CONFIG, enable_stroke_width_filter=True)
        even = self._bar(10)
        wedge = np.zeros((128, 128), dtype=np.uint8)
        for row in range(128):  # a taper, like a bevel rather than a letter
            wedge[row, 40 : 40 + 2 + row // 6] = 200
        box = (35, 10, 40, 100)
        _m, even_variation, _c = stroke_width_stats(
            stroke_width_transform(even, None, config), box
        )
        _m, wedge_variation, _c = stroke_width_stats(
            stroke_width_transform(wedge, None, config), box
        )
        self.assertLess(even_variation, wedge_variation)

    def test_the_stage_is_a_no_op_until_enabled(self) -> None:
        image = embossed_text()
        run = run_detection(image, None, EDGE_CONFIG)
        stage = next(s for s in run.stages if s.key == "stroke_width")
        self.assertEqual(stage.detail, "disabled")
        self.assertEqual(len(stage.rejected), 0)
        self.assertEqual(len(stage.kept), len(run.stages[0].kept))

    def test_every_parameter_maps_to_a_pipeline_step(self) -> None:
        # A parameter left out of the table resumes from the wrong step and
        # shows stale boxes while it is being tuned.
        from mesh_segmentation_transform.annotate_texture_regions import (
            MserConfig, PARAMETER_STEP,
        )
        from dataclasses import fields as dataclass_fields
        missing = [f.name for f in dataclass_fields(MserConfig)
                   if f.name not in PARAMETER_STEP]
        self.assertEqual(missing, [])

    def test_the_step_table_and_the_name_table_agree(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            PIPELINE_STEPS, STEP_INDEX,
        )
        self.assertEqual(len(STEP_INDEX), len(PIPELINE_STEPS))
        self.assertEqual(STEP_INDEX["boxes"], 0)
        self.assertLess(STEP_INDEX["stroke_width"], STEP_INDEX["box_filter"])


class RingSmoothnessTests(unittest.TestCase):
    """What surrounds a region, not the region itself.

    Inside its own bounds a patch of speaker grille looks much like a glyph:
    strong, closed, stroke-like edges.  One ring further out, the grille
    carries on and a fascia does not.
    """

    def _mark_on_plain_trim(self, size: int = 256) -> np.ndarray:
        image = np.full((size, size), 128, dtype=np.uint8)
        cv2.putText(image, "AB", (90, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 168, 4)
        return image

    def _patch_of_grille(self, size: int = 256) -> np.ndarray:
        image = np.full((size, size), 128, dtype=np.uint8)
        for y in range(8, size, 12):
            for x in range(8, size, 12):
                cv2.circle(image, (x, y), 3, 168, -1)
        return image

    def _reference(self, image, config):
        from mesh_segmentation_transform.annotate_texture_regions import edge_response
        return float(np.median(edge_response(image, config)))

    def test_plain_trim_around_a_mark_reads_smooth(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            edge_response, ring_roughness,
        )
        image = self._mark_on_plain_trim()
        response = edge_response(image, DEFAULT_CONFIG)
        roughness = ring_roughness(
            response, None, (88, 108, 96, 40), DEFAULT_CONFIG,
            self._reference(image, DEFAULT_CONFIG),
        )
        self.assertIsNotNone(roughness)
        self.assertLess(roughness, DEFAULT_CONFIG.max_ring_roughness)

    def test_a_grille_reads_rough_around_any_part_of_itself(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            edge_response, ring_roughness,
        )
        image = self._patch_of_grille()
        response = edge_response(image, DEFAULT_CONFIG)
        roughness = ring_roughness(
            response, None, (88, 108, 96, 40), DEFAULT_CONFIG,
            self._reference(image, DEFAULT_CONFIG),
        )
        self.assertIsNotNone(roughness)
        self.assertGreater(roughness, DEFAULT_CONFIG.max_ring_roughness)

    def test_a_ring_with_too_little_domain_concludes_nothing(self) -> None:
        # Nothing measured, nothing concluded: a region at an island edge is
        # looking at the void rather than at material.
        from mesh_segmentation_transform.annotate_texture_regions import (
            edge_response, ring_roughness,
        )
        image = self._mark_on_plain_trim()
        response = edge_response(image, DEFAULT_CONFIG)
        domain = np.zeros(image.shape, dtype=bool)
        domain[100:140, 100:140] = True
        self.assertIsNone(
            ring_roughness(response, domain, (100, 100, 40, 40), DEFAULT_CONFIG, 1.0)
        )

    def test_the_margin_keeps_the_mark_out_of_its_own_ring(self) -> None:
        # Without a gap the mark's outer edge counts as surrounding busyness
        # and every real mark measures rough.
        from mesh_segmentation_transform.annotate_texture_regions import (
            edge_response, ring_roughness,
        )
        image = self._mark_on_plain_trim()
        response = edge_response(image, DEFAULT_CONFIG)
        reference = self._reference(image, DEFAULT_CONFIG)
        tight = replace(DEFAULT_CONFIG, ring_smoothness_margin_px=0)
        bounds = (92, 112, 88, 32)
        self.assertLessEqual(
            ring_roughness(response, None, bounds, DEFAULT_CONFIG, reference),
            ring_roughness(response, None, bounds, tight, reference),
        )

    def test_the_stage_is_a_no_op_until_enabled(self) -> None:
        run = run_detection(embossed_text(), None, EDGE_CONFIG)
        stage = next(s for s in run.stages if s.key == "ring_smoothness")
        self.assertEqual(stage.detail, "disabled")
        self.assertEqual(len(stage.rejected), 0)

    def test_it_runs_after_grouping_and_before_final_size(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import STEP_INDEX
        self.assertLess(STEP_INDEX["grouped"], STEP_INDEX["ring_smoothness"])
        self.assertLess(STEP_INDEX["ring_smoothness"], STEP_INDEX["size"])


class RegionFlatnessTests(unittest.TestCase):
    """The other half of the pair: rejects a region for holding nothing.

    The ring test asks whether a region is isolated, which says nothing about
    one that is isolated and also empty -- a bolster roll or a fascia curve
    raises an edge without carrying a mark.
    """

    def _reference(self, image):
        from mesh_segmentation_transform.annotate_texture_regions import edge_response
        return float(np.median(edge_response(image, DEFAULT_CONFIG)))

    def test_a_region_holding_a_mark_has_relief(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            edge_response, region_flatness,
        )
        image = np.full((256, 256), 128, dtype=np.uint8)
        cv2.putText(image, "AB", (90, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 168, 4)
        response = edge_response(image, DEFAULT_CONFIG)
        relief = region_flatness(
            response, None, (88, 108, 96, 40), DEFAULT_CONFIG, self._reference(image)
        )
        self.assertIsNotNone(relief)
        self.assertGreater(relief, DEFAULT_CONFIG.min_region_relief)

    def test_a_featureless_patch_holds_none(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            edge_response, region_flatness,
        )
        # Texture over most of the atlas, so the reference means something,
        # with one plain patch in it.  A bare ramp will not do: quantised to
        # 8 bits a shallow one becomes a staircase whose median gradient is
        # zero, and every ratio against it is meaningless.
        rng = np.random.default_rng(0)
        image = (128 + rng.normal(0, 12, (256, 256))).clip(0, 255).astype(np.uint8)
        image[100:160, 80:200] = 128
        response = edge_response(image, DEFAULT_CONFIG)
        relief = region_flatness(
            response, None, (88, 108, 96, 40), DEFAULT_CONFIG, self._reference(image)
        )
        self.assertIsNotNone(relief)
        self.assertLess(relief, DEFAULT_CONFIG.min_region_relief)

    def test_a_high_percentile_sees_a_mark_a_median_would_miss(self) -> None:
        # A glyph does not fill its box, so the middle of the distribution is
        # background either way; the 90th percentile is what notices anything
        # is there.
        from mesh_segmentation_transform.annotate_texture_regions import (
            edge_response, region_flatness,
        )
        image = np.full((256, 256), 128, dtype=np.uint8)
        cv2.putText(image, "A", (96, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 200, 5)
        response = edge_response(image, DEFAULT_CONFIG)
        reference = self._reference(image)
        box = (92, 108, 60, 52)
        high = region_flatness(response, None, box, DEFAULT_CONFIG, reference)
        middle = region_flatness(
            response, None, box,
            replace(DEFAULT_CONFIG, region_flatness_percentile=50.0), reference,
        )
        self.assertGreater(high, middle)

    def test_a_mark_far_smaller_than_its_box_is_missed(self) -> None:
        """A real limit, recorded rather than papered over.

        The percentile only reaches a mark that occupies more than
        100 - region_flatness_percentile of its box.  A speck in a generous
        box is invisible to it at any setting short of the maximum, so this
        filter relies on grouping having fitted the box to the mark.
        """
        from mesh_segmentation_transform.annotate_texture_regions import (
            edge_response, region_flatness,
        )
        image = np.full((256, 256), 128, dtype=np.uint8)
        cv2.putText(image, ".", (120, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 200, 4)
        response = edge_response(image, DEFAULT_CONFIG)
        self.assertEqual(
            region_flatness(response, None, (60, 60, 140, 140), DEFAULT_CONFIG,
                            self._reference(image)),
            0.0,
        )

    def test_too_little_domain_concludes_nothing(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            edge_response, region_flatness,
        )
        image = np.full((128, 128), 128, dtype=np.uint8)
        response = edge_response(image, DEFAULT_CONFIG)
        domain = np.zeros(image.shape, dtype=bool)
        domain[:4, :4] = True
        self.assertIsNone(
            region_flatness(response, domain, (0, 0, 8, 8), DEFAULT_CONFIG, 1.0)
        )

    def test_the_stage_is_a_no_op_until_enabled(self) -> None:
        run = run_detection(embossed_text(), None, EDGE_CONFIG)
        stage = next(s for s in run.stages if s.key == "region_flatness")
        self.assertEqual(stage.detail, "disabled")
        self.assertEqual(len(stage.rejected), 0)

    def test_the_two_region_filters_run_in_order(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import STEP_INDEX
        self.assertLess(STEP_INDEX["pattern_group"], STEP_INDEX["region_flatness"])
        self.assertLess(STEP_INDEX["region_flatness"], STEP_INDEX["ring_smoothness"])
        self.assertLess(STEP_INDEX["ring_smoothness"], STEP_INDEX["size"])

    def test_the_default_floor_is_reachable(self) -> None:
        # A filter whose threshold sits below anything it will ever see is a
        # dead knob.  10 is chosen because the edge front-end returns nothing
        # flatter than about 4.
        self.assertGreater(DEFAULT_CONFIG.min_region_relief, 4.0)


class RotatedBoundsTests(unittest.TestCase):
    """The feature's true shape, rather than the box it happens to sit in."""

    def _mask_with(self, draw) -> np.ndarray:
        image = np.zeros((256, 256), dtype=np.uint8)
        draw(image)
        return image > 0

    def test_a_tilted_bar_measures_its_own_width_not_its_box(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            rotated_bounds_measures, rotated_feature_bounds,
        )
        mask = self._mask_with(
            lambda im: cv2.line(im, (60, 60), (180, 180), 255, 6)
        )
        box = (55, 55, 130, 130)   # the axis-aligned box is mostly background
        rectangle = rotated_feature_bounds(mask, box, DEFAULT_CONFIG)
        self.assertIsNotNone(rectangle)
        _centre, (rw, rh), angle = rectangle
        self.assertAlmostEqual(min(rw, rh), 6.0, delta=3.0)
        self.assertAlmostEqual(abs(angle) % 90, 45.0, delta=5.0)
        fill, aspect = rotated_bounds_measures(rectangle, box)
        self.assertLess(fill, 0.2)          # a diagonal streak barely fills it
        self.assertGreater(aspect, 10.0)    # and is long and thin in truth

    def test_a_square_mark_fills_its_box(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            rotated_bounds_measures, rotated_feature_bounds,
        )
        mask = self._mask_with(
            lambda im: cv2.rectangle(im, (80, 80), (160, 160), 255, -1)
        )
        box = (78, 78, 85, 85)
        fill, aspect = rotated_bounds_measures(
            rotated_feature_bounds(mask, box, DEFAULT_CONFIG), box
        )
        self.assertGreater(fill, 0.8)
        self.assertLess(aspect, 1.3)

    def test_too_few_points_fits_nothing(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            rotated_feature_bounds,
        )
        mask = np.zeros((64, 64), dtype=bool)
        mask[10, 10] = True
        self.assertIsNone(
            rotated_feature_bounds(mask, (0, 0, 40, 40), DEFAULT_CONFIG)
        )

    def test_the_stage_is_a_no_op_until_enabled(self) -> None:
        run = run_detection(embossed_text(), None, EDGE_CONFIG)
        stage = next(s for s in run.stages if s.key == "rotated_bounds")
        self.assertEqual(stage.detail, "disabled")
        self.assertEqual(len(stage.rejected), 0)

    def test_it_runs_immediately_before_region_flatness(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import STEP_INDEX
        self.assertEqual(
            STEP_INDEX["rotated_bounds"] + 1, STEP_INDEX["region_flatness"]
        )

    def test_the_defaults_admit_real_lettering(self) -> None:
        # Measured on the ardente: ARDENTE fills 0.78 of its box at a true
        # aspect of 9.1, AIRBAG 0.67 at 5.3.  A default that excluded either
        # would be worse than no filter.
        self.assertLess(DEFAULT_CONFIG.min_rotated_fill, 0.67)
        self.assertGreater(DEFAULT_CONFIG.max_rotated_aspect, 9.1)

    def test_the_stage_carries_corners_for_the_viewer(self) -> None:
        """The outline drawn must be the rectangle that was measured.

        A filter judged on one shape while another is drawn cannot be tuned by
        eye, which is the whole point of the harness.
        """
        image = embossed_text()
        config = replace(EDGE_CONFIG, enable_rotated_bounds_filter=True)
        run = run_detection(image, None, config)
        stage = next(s for s in run.stages if s.key == "rotated_bounds")
        self.assertEqual(len(stage.rotations), len(stage.kept))
        for corners in stage.rotations:
            if corners is not None:
                self.assertEqual(len(corners), 4)

    def test_a_region_with_nothing_fitted_carries_no_corners(self) -> None:
        # Kept, because nothing was measured and so nothing concluded -- but
        # with no outline to draw, so the axis-aligned box shows instead.
        image = embossed_text()
        config = replace(EDGE_CONFIG, enable_rotated_bounds_filter=True,
                         rotated_bounds_min_points=10 ** 6)
        run = run_detection(image, None, config)
        stage = next(s for s in run.stages if s.key == "rotated_bounds")
        self.assertEqual(len(stage.rejected), 0)
        self.assertTrue(all(c is None for c in stage.rotations))

    def test_corners_are_absolute_texture_coordinates(self) -> None:
        # Fitted inside a crop, so they have to be offset back out or every
        # outline lands at the top left of the atlas.
        from mesh_segmentation_transform.annotate_texture_regions import (
            rotated_feature_bounds,
        )
        mask = np.zeros((256, 256), dtype=bool)
        mask[180:200, 150:210] = True
        rectangle = rotated_feature_bounds(mask, (140, 170, 90, 50), DEFAULT_CONFIG)
        self.assertIsNotNone(rectangle)
        (cx, cy), _size, _angle = rectangle
        self.assertAlmostEqual(cx, 180.0, delta=6.0)
        self.assertAlmostEqual(cy, 190.0, delta=6.0)

    def test_edge_aligned_rotation_uses_a_near_parallel_uv_edge(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            edge_aligned_feature_outline,
            feature_shape,
        )

        feature = np.zeros((256, 256), np.uint8)
        cv2.line(feature, (70, 95), (150, 135), 255, 6)
        mask = feature > 0
        uv = np.zeros((256, 256), np.uint8)
        cv2.fillPoly(
            uv,
            [np.asarray([(0, 76), (255, 204), (255, 255), (0, 255)], np.int32)],
            255,
        )
        config = replace(
            DEFAULT_CONFIG,
            enable_edge_aligned_rotation=True,
            rotation_edge_search_px=18,
            rotation_edge_band_px=5,
            max_rotation_edge_angle_degrees=10.0,
        )
        box = (65, 88, 95, 55)
        shape = feature_shape(mask, box, config)
        self.assertIsNotNone(shape)
        alignment = edge_aligned_feature_outline(mask, uv > 0, box, shape, config)
        self.assertIsNotNone(alignment)
        self.assertLess(alignment.distance_px, config.rotation_edge_search_px)
        self.assertLess(
            alignment.angle_delta_degrees,
            config.max_rotation_edge_angle_degrees,
        )

        corners = alignment.outline
        side_angle = np.degrees(
            np.arctan2(
                corners[1][1] - corners[0][1],
                corners[1][0] - corners[0][0],
            )
        ) % 180.0
        self.assertAlmostEqual(side_angle, alignment.edge_angle_degrees, delta=0.01)

    def test_edge_aligned_rotation_rejects_an_angle_mismatch(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            edge_aligned_feature_outline,
            feature_shape,
        )

        feature = np.zeros((128, 128), np.uint8)
        cv2.line(feature, (58, 45), (58, 105), 255, 6)
        mask = feature > 0
        uv = np.zeros((128, 128), np.uint8)
        cv2.fillPoly(
            uv,
            [np.asarray([(0, 36), (127, 100), (127, 127), (0, 127)], np.int32)],
            255,
        )
        config = replace(
            DEFAULT_CONFIG,
            enable_edge_aligned_rotation=True,
            rotation_edge_search_px=24,
            rotation_edge_band_px=5,
            max_rotation_edge_angle_degrees=10.0,
        )
        box = (50, 38, 18, 75)
        shape = feature_shape(mask, box, config)
        self.assertIsNotNone(shape)
        self.assertIsNone(
            edge_aligned_feature_outline(mask, uv > 0, box, shape, config)
        )

    def test_edge_aligned_rotation_rejects_a_distant_edge(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            edge_aligned_feature_outline,
            feature_shape,
        )

        feature = np.zeros((128, 128), np.uint8)
        cv2.line(feature, (20, 55), (100, 95), 255, 6)
        mask = feature > 0
        uv = np.zeros((128, 128), np.uint8)
        cv2.fillPoly(
            uv,
            [np.asarray([(0, 20), (127, 84), (127, 127), (0, 127)], np.int32)],
            255,
        )
        config = replace(
            DEFAULT_CONFIG,
            enable_edge_aligned_rotation=True,
            rotation_edge_search_px=4,
            rotation_edge_band_px=3,
            max_rotation_edge_angle_degrees=10.0,
        )
        box = (15, 48, 95, 55)
        shape = feature_shape(mask, box, config)
        self.assertIsNotNone(shape)
        self.assertIsNone(
            edge_aligned_feature_outline(mask, uv > 0, box, shape, config)
        )

    def test_edge_aligned_rotation_rejects_a_touching_edge(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            edge_aligned_feature_outline,
            feature_shape,
        )

        feature = np.zeros((128, 128), np.uint8)
        cv2.line(feature, (20, 46), (100, 86), 255, 6)
        mask = feature > 0
        uv = np.zeros((128, 128), np.uint8)
        cv2.fillPoly(
            uv,
            [np.asarray([(0, 36), (127, 100), (127, 127), (0, 127)], np.int32)],
            255,
        )
        config = replace(
            DEFAULT_CONFIG,
            enable_edge_aligned_rotation=True,
            rotation_edge_min_gap_px=2.0,
            rotation_edge_search_px=18,
            rotation_edge_band_px=5,
            max_rotation_edge_angle_degrees=10.0,
        )
        box = (15, 39, 95, 55)
        shape = feature_shape(mask, box, config)
        self.assertIsNotNone(shape)
        self.assertIsNone(
            edge_aligned_feature_outline(mask, uv > 0, box, shape, config)
        )

    def test_edge_aligned_rotation_rejects_a_two_sided_uv_strip(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            edge_aligned_feature_outline,
            feature_shape,
        )

        feature = np.zeros((160, 160), np.uint8)
        cv2.rectangle(feature, (72, 25), (88, 135), 255, -1)
        mask = feature > 0
        uv = np.zeros((160, 160), np.uint8)
        cv2.rectangle(uv, (52, 0), (108, 159), 255, -1)
        config = replace(
            DEFAULT_CONFIG,
            enable_edge_aligned_rotation=True,
            rotation_edge_min_gap_px=2.0,
            rotation_edge_search_px=24,
            rotation_edge_band_px=5,
            rotation_edge_min_points=8,
            max_opposite_rotation_edge_fraction=0.5,
        )
        box = (68, 20, 25, 120)
        shape = feature_shape(mask, box, config)
        self.assertIsNotNone(shape)
        self.assertIsNone(
            edge_aligned_feature_outline(mask, uv > 0, box, shape, config)
        )


def separated_marks(size: int = 512) -> np.ndarray:
    """Several well-spaced marks that survive the whole pipeline.

    The embossed_text fixture cannot be used here: at the amplitude real
    lettering has, box filtering rejects every one of its boxes, so nothing
    reaches the later stages and any assertion about them passes on empty
    lists.  Plumbing has to be tested on something that actually flows.
    """
    image = np.full((size, size), 128, dtype=np.uint8)
    cv2.rectangle(image, (40, 40), (160, 90), 200, -1)      # wide bar
    cv2.line(image, (260, 40), (380, 160), 200, 14)         # diagonal
    cv2.rectangle(image, (60, 240), (150, 330), 200, -1)    # square
    cv2.rectangle(image, (300, 300), (320, 320), 200, -1)   # small square
    cv2.line(image, (60, 420), (300, 430), 200, 10)         # long thin bar
    # Elongated, solidly filled and axis-aligned: the one mark here that earns
    # a rotated outline, so the adoption rules have something to accept.
    cv2.rectangle(image, (330, 220), (496, 246), 200, -1)
    return np.repeat(image[:, :, None], 3, axis=2)


MARKS_CONFIG = replace(
    DEFAULT_CONFIG, box_source="edge", edge_local_window_px=64, edge_local_k=1.6,
    enable_box_feature_filter=False, enable_rotated_bounds_filter=True,
)


class RotatedBoundsPersistenceTests(unittest.TestCase):
    """The outline has to survive every filter that runs after it.

    Refitting per stage would cost a gradient pass apiece and could quietly
    disagree with what the earlier stage decided on, so it is carried.
    """

    def _run(self, **overrides):
        return run_detection(
            separated_marks(), None, replace(MARKS_CONFIG, **overrides)
        )

    def test_outlines_reach_the_final_stage(self) -> None:
        run = self._run()
        after = [s for s in run.stages
                 if s.key in ("region_flatness", "ring_smoothness", "size",
                              "final_padding")]
        self.assertEqual(len(after), 4)
        for stage in after:
            self.assertGreater(len(stage.kept), 0, f"{stage.title} kept nothing")
            self.assertEqual(len(stage.rotations), len(stage.kept), stage.title)
        self.assertGreater(
            sum(1 for r in after[-1].rotations if r), 0,
            "no outline survived to the end",
        )

    def test_outlines_stay_aligned_when_a_later_stage_rejects(self) -> None:
        # The failure this guards against is silent: a misaligned list draws
        # each outline against somebody else's region.  Forced at the size
        # stage because that is the one realigning through a helper rather
        # than tracking indices as it goes.
        run = self._run(final_min_area_px=3000)
        size = next(s for s in run.stages if s.key == "size")
        self.assertGreater(len(size.rejected), 0, "expected the size test to bite")
        self.assertGreater(len(size.kept), 0, "expected something to survive")
        self.assertEqual(len(size.rotations), len(size.kept))

        # And the survivors keep their own outlines, not their neighbours'.
        before = next(s for s in run.stages if s.key == "ring_smoothness")
        carried = dict(zip(before.kept, before.rotations))
        for group, outline in zip(size.kept, size.rotations):
            self.assertEqual(outline, carried[group])

    def test_padding_does_not_grow_the_outline(self) -> None:
        # The outline describes the feature, not the box grown around it.
        run = self._run(final_region_padding_px=6)
        size = next(s for s in run.stages if s.key == "size")
        padding = next(s for s in run.stages if s.key == "final_padding")
        self.assertNotEqual(size.kept, padding.kept, "padding changed nothing")
        self.assertEqual(size.rotations, padding.rotations)

    def test_a_subset_realignment_survives_duplicate_bounds(self) -> None:
        # Two regions can share bounds; matching by lookup would give them the
        # same outline, so the walk has to advance monotonically.
        from mesh_segmentation_transform.annotate_texture_regions import (
            _rotations_for_subset,
        )
        original = [(0, 0, 4, 4), (0, 0, 4, 4), (9, 9, 4, 4)]
        rotations = [("a",), ("b",), ("c",)]
        self.assertEqual(
            _rotations_for_subset(rotations, original, [(0, 0, 4, 4), (9, 9, 4, 4)]),
            [("a",), ("c",)],
        )


class FeatureShapeTests(unittest.TestCase):
    """Which boundary describes the feature, and whether it is worth adopting."""

    def _shape(self, mask, box, **overrides):
        from mesh_segmentation_transform.annotate_texture_regions import feature_shape
        return feature_shape(mask, box, replace(DEFAULT_CONFIG, **overrides))

    def test_the_hull_hugs_tighter_than_the_rectangle(self) -> None:
        # The substantive reason to prefer it.  Measured on the ardente the
        # hull is about four fifths of the rectangle's area; a cross is a
        # sharper case of the same thing.
        image = np.zeros((256, 256), np.uint8)
        cv2.line(image, (60, 128), (200, 128), 255, 12)
        cv2.line(image, (128, 60), (128, 200), 255, 12)
        shape = self._shape(image > 0, (55, 55, 150, 150))
        self.assertIsNotNone(shape)
        self.assertLess(shape.hull_area, shape.rectangle_area)
        self.assertGreater(shape.tightness("hull"), shape.tightness("rotated"))

    def test_the_rectangle_from_the_hull_matches_the_one_from_every_point(self) -> None:
        # Which is what makes taking both nearly free: rotating calipers over
        # a handful of hull points, not over thousands of texels.
        image = np.zeros((256, 256), np.uint8)
        cv2.line(image, (60, 60), (190, 170), 255, 14)
        mask = image > 0
        shape = self._shape(mask, (55, 55, 145, 125))
        direct = cv2.minAreaRect(cv2.findNonZero(mask[55:180, 55:200].astype(np.uint8)))
        self.assertAlmostEqual(
            shape.rectangle_area,
            float(direct[1][0] * direct[1][1]),
            delta=1.0,
        )

    def test_a_solid_block_is_almost_entirely_feature(self) -> None:
        image = np.zeros((128, 128), np.uint8)
        cv2.rectangle(image, (40, 40), (90, 90), 255, -1)
        shape = self._shape(image > 0, (38, 38, 55, 55))
        self.assertGreater(shape.tightness("hull"), 0.9)

    def test_scattered_speckle_wraps_mostly_nothing(self) -> None:
        # The case the bar exists for: a boundary can be fitted, but it
        # describes empty space rather than a mark.
        rng = np.random.default_rng(0)
        image = np.zeros((128, 128), np.uint8)
        for _ in range(30):
            x, y = rng.integers(20, 108, 2)
            image[y, x] = 255
        shape = self._shape(image > 0, (10, 10, 110, 110))
        self.assertIsNotNone(shape)
        self.assertLess(shape.tightness("hull"), DEFAULT_CONFIG.min_feature_tightness)

    def test_tightness_over_one_is_normal(self) -> None:
        # The hull is a polygon through pixel centres while the feature is
        # counted in whole pixels, so a small solid mark can exceed 1.
        image = np.zeros((64, 64), np.uint8)
        cv2.rectangle(image, (28, 28), (34, 34), 255, -1)
        shape = self._shape(image > 0, (26, 26, 11, 11))
        self.assertGreater(shape.tightness("hull"), 1.0)

    def test_the_shape_choice_changes_the_outline(self) -> None:
        image = np.zeros((256, 256), np.uint8)
        cv2.line(image, (60, 128), (200, 128), 255, 12)
        cv2.line(image, (128, 60), (128, 200), 255, 12)
        mask = image > 0
        box = (55, 55, 150, 150)
        self.assertEqual(len(self._shape(mask, box).outline("rotated")), 4)
        self.assertGreater(len(self._shape(mask, box).outline("hull")), 4)

    def test_the_region_is_kept_whichever_shape_is_adopted(self) -> None:
        # Declining the outline must not remove the region: how well a shape
        # describes a mark is not evidence about whether it is one.
        loose = run_detection(
            separated_marks(), None,
            replace(MARKS_CONFIG, min_feature_tightness=0.0),
        )
        strict = run_detection(
            separated_marks(), None,
            replace(MARKS_CONFIG, min_feature_tightness=1.5),
        )
        a = next(s for s in loose.stages if s.key == "rotated_bounds")
        b = next(s for s in strict.stages if s.key == "rotated_bounds")
        self.assertEqual(a.kept, b.kept)
        self.assertGreater(sum(1 for r in a.rotations if r), 0)
        self.assertEqual(sum(1 for r in b.rotations if r), 0)

    def test_the_default_bar_admits_real_lettering(self) -> None:
        # Measured on the ardente: ARDENTE's hull is 0.41 feature and
        # AIRBAG's 0.66.  A bar excluding either would be worse than none.
        self.assertLess(DEFAULT_CONFIG.min_feature_tightness, 0.41)


class TextLineTests(unittest.TestCase):
    """Word or symbol -- the question none of the other filters asks.

    A horn or seat pictogram is a genuine mark on plain trim, so it passes
    every test about whether a region holds something.  What separates it from
    the AIRBAG text is that text is several similar marks in a row.
    """

    def _measures(self, image, box, **overrides):
        from mesh_segmentation_transform.annotate_texture_regions import (
            text_line_measures,
        )
        return text_line_measures(
            image > 0, box, replace(DEFAULT_CONFIG, **overrides)
        )

    def _word(self, size: int = 256) -> np.ndarray:
        image = np.zeros((size, size), np.uint8)
        for index, x in enumerate(range(30, 200, 28)):
            cv2.rectangle(image, (x, 110), (x + 16, 145), 255, -1)
        return image

    def _pictogram(self, size: int = 256) -> np.ndarray:
        image = np.zeros((size, size), np.uint8)
        cv2.ellipse(image, (128, 128), (46, 30), 0, 0, 360, 255, -1)
        return image

    def test_a_row_of_characters_reads_as_text(self) -> None:
        characters, scatter = self._measures(self._word(), (20, 100, 200, 60))
        self.assertGreaterEqual(characters, DEFAULT_CONFIG.text_min_characters)
        self.assertLess(scatter, DEFAULT_CONFIG.max_baseline_scatter)

    def test_a_single_pictogram_does_not(self) -> None:
        result = self._measures(self._pictogram(), (75, 90, 110, 80))
        self.assertIsNotNone(result)
        characters, _scatter = result
        self.assertLess(characters, DEFAULT_CONFIG.text_min_characters)

    def test_marks_scattered_about_are_not_a_line(self) -> None:
        image = np.zeros((256, 256), np.uint8)
        for x, y in ((40, 40), (150, 60), (70, 170), (200, 200), (120, 110)):
            cv2.rectangle(image, (x, y), (x + 18, y + 18), 255, -1)
        _characters, scatter = self._measures(image, (30, 30, 200, 200))
        self.assertGreater(scatter, DEFAULT_CONFIG.max_baseline_scatter)

    def test_two_components_are_never_a_line(self) -> None:
        # Two points lie on a perfect line by definition, so scatter below
        # three components would wave every two-part icon through.
        image = np.zeros((128, 128), np.uint8)
        cv2.rectangle(image, (30, 60), (48, 78), 255, -1)
        cv2.rectangle(image, (80, 60), (98, 78), 255, -1)
        characters, scatter = self._measures(image, (20, 50, 90, 40))
        self.assertEqual(characters, 2)
        self.assertEqual(scatter, float("inf"))

    def test_speckle_is_not_counted_as_characters(self) -> None:
        image = self._pictogram()
        rng = np.random.default_rng(0)
        for _ in range(40):
            x, y = rng.integers(10, 246, 2)
            image[y, x] = 255
        result = self._measures(image, (5, 5, 246, 246), text_min_component_px=20)
        self.assertIsNotNone(result)
        characters, _scatter = result
        self.assertLess(characters, DEFAULT_CONFIG.text_min_characters)

    def test_an_empty_region_says_nothing(self) -> None:
        self.assertIsNone(
            self._measures(np.zeros((64, 64), np.uint8), (0, 0, 40, 40))
        )

    def test_the_stage_is_a_no_op_until_enabled(self) -> None:
        run = run_detection(separated_marks(), None, MARKS_CONFIG)
        stage = next(s for s in run.stages if s.key == "text_line")
        self.assertEqual(stage.detail, "disabled")
        self.assertEqual(len(stage.rejected), 0)

    def test_it_runs_after_the_region_tests_and_before_final_size(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import STEP_INDEX
        self.assertLess(STEP_INDEX["ring_smoothness"], STEP_INDEX["text_line"])
        self.assertLess(STEP_INDEX["text_line"], STEP_INDEX["size"])

    def test_outlines_survive_the_stage(self) -> None:
        run = run_detection(
            separated_marks(), None,
            replace(MARKS_CONFIG, enable_text_line_filter=True,
                    text_min_characters=1, max_baseline_scatter=1e9),
        )
        stage = next(s for s in run.stages if s.key == "text_line")
        self.assertEqual(len(stage.rotations), len(stage.kept))
