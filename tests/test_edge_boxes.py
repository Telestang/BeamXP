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

import random
import unittest
from dataclasses import replace

import cv2
import numpy as np

from mesh_segmentation_transform.annotate_texture_regions import (
    DEFAULT_CONFIG,
    detect_edge_boxes,
    detect_foreground_boxes,
    detect_opacity_mask_boxes,
    detect_mser_boxes,
    edge_response,
    run_detection,
    SHAPE_HULL,
    SHAPE_ROTATED,
    STEP_INDEX,
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
RING_CONFIG = replace(
    DEFAULT_CONFIG,
    ring_smoothness_width_px=12,
    ring_smoothness_percentile=75.0,
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

    def test_gpu_laplacian_uses_the_shared_context_and_tracks_cpu_response(self) -> None:
        from mesh_segmentation_transform.texture_local_contrast_gpu import (
            LocalContrastGpuUnavailable,
            gpu_renderer,
        )

        image = embossed_text(size=96)[:, :, 0]
        cpu_config = replace(
            DEFAULT_CONFIG, box_source="edge", edge_operator="laplacian",
        )
        gpu_config = replace(cpu_config, box_source="edge_gpu")
        try:
            renderer = gpu_renderer()
            gpu = edge_response(image, gpu_config)
        except LocalContrastGpuUnavailable as exc:
            self.skipTest(str(exc))
        cpu = edge_response(image, cpu_config)
        self.assertEqual(gpu.shape, cpu.shape)
        self.assertEqual(gpu_renderer(), renderer)
        self.assertGreater(float(np.corrcoef(cpu.ravel(), gpu.ravel())[0, 1]), 0.99)
        self.assertAlmostEqual(
            float(np.percentile(gpu, 90)), float(np.percentile(cpu, 90)), delta=3.0,
        )


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
    def test_opacity_mask_keeps_a_glyph_filling_its_uv_island(self) -> None:
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        domain = np.zeros((32, 32), dtype=bool)
        domain[8:24, 10:22] = True
        image[8:24, 10:22] = 239
        config = replace(
            DEFAULT_CONFIG,
            box_source="opacity_mask",
            foreground_min_component_px=4,
            foreground_merge_gap_px=0,
        )

        boxes = detect_opacity_mask_boxes(image, domain, config)
        run = run_detection(image, domain, config)

        self.assertEqual(boxes.tolist(), [[10, 8, 12, 16]])
        self.assertEqual(run.stages[0].title, "Opacity-mask boxes")
        self.assertEqual(run.stages[0].kept, ((10, 8, 12, 16),))

    def test_the_colour_path_uses_the_foreground_front_end_by_default(self) -> None:
        # Authored UI/emissive atlases are now detected from their foreground
        # mask; MSER remains an explicit comparison mode.
        self.assertEqual(DEFAULT_CONFIG.box_source, "foreground")
        image = embossed_text()
        run = run_detection(image, None, DEFAULT_CONFIG)
        self.assertEqual(run.stages[0].title, "Foreground-mask boxes")
        self.assertEqual(
            len(run.stages[0].kept), len(detect_foreground_boxes(image, None, DEFAULT_CONFIG))
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


class ConservativeGroupingTests(unittest.TestCase):
    def test_diagonal_neighbours_do_not_group(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            union_region_group_candidates,
        )

        image = np.full((80, 80, 3), 128, dtype=np.uint8)
        boxes = np.asarray(
            [
                (20, 20, 8, 8),
                (34, 34, 8, 8),
            ],
            dtype=np.int32,
        )
        config = replace(
            DEFAULT_CONFIG,
            merge_distance_px=18,
            enable_circular_groups=False,
        )

        groups = union_region_group_candidates(boxes, image, config)

        self.assertEqual(
            [group.bounds for group in groups],
            [(20, 20, 8, 8), (34, 34, 8, 8)],
        )

    def test_corner_touching_neighbours_do_not_group(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            union_region_group_candidates,
        )

        image = np.full((80, 80, 3), 128, dtype=np.uint8)
        boxes = np.asarray(
            [
                (20, 20, 8, 8),
                (28, 28, 8, 8),
            ],
            dtype=np.int32,
        )
        config = replace(
            DEFAULT_CONFIG,
            merge_distance_px=18,
            enable_circular_groups=False,
        )

        groups = union_region_group_candidates(boxes, image, config)

        self.assertEqual(
            [group.bounds for group in groups],
            [(20, 20, 8, 8), (28, 28, 8, 8)],
        )

    def test_vertical_neighbours_still_group(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            union_region_group_candidates,
        )

        image = np.full((80, 80, 3), 128, dtype=np.uint8)
        boxes = np.asarray(
            [
                (20, 20, 8, 8),
                (20, 34, 8, 8),
            ],
            dtype=np.int32,
        )
        config = replace(
            DEFAULT_CONFIG,
            merge_distance_px=18,
            enable_circular_groups=False,
        )

        groups = union_region_group_candidates(boxes, image, config)

        self.assertEqual([group.bounds for group in groups], [(20, 20, 8, 22)])

    def test_grouping_tolerates_one_raster_pixel_at_the_expansion_boundary(self) -> None:
        """A half-open box boundary must not split an otherwise aligned run."""
        from mesh_segmentation_transform.annotate_texture_regions import (
            union_region_group_candidates,
        )

        image = np.full((96, 64, 3), 128, dtype=np.uint8)
        # The raw gap is 21 px.  A 10-px expansion on both sides misses by a
        # single raster cell without the grouping tolerance.
        boxes = np.asarray(((20, 20, 8, 8), (20, 49, 8, 8)), dtype=np.int32)
        config = replace(
            DEFAULT_CONFIG,
            merge_distance_px=10,
            enable_circular_groups=False,
        )

        groups = union_region_group_candidates(boxes, image, config)

        self.assertEqual([group.bounds for group in groups], [(20, 20, 8, 37)])

    def test_overlap_grouping_tolerates_one_pixel_near_containment(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            overlap_region_group_candidates,
        )

        image = np.full((192, 384, 3), 128, dtype=np.uint8)
        # The inner edge response begins one texel above the enclosing response,
        # matching the Scintilla badge atlas.  They still describe one mark.
        boxes = np.asarray(
            ((21, 8, 291, 145), (67, 7, 197, 77)), dtype=np.int32
        )
        config = replace(
            DEFAULT_CONFIG,
            merge_distance_px=0,
            enable_circular_groups=False,
        )

        groups = overlap_region_group_candidates(boxes, image, config)

        self.assertEqual([group.bounds for group in groups], [(21, 7, 291, 146)])
        self.assertEqual(len(groups[0].members), 2)

    def test_significant_overlap_groups_before_proximity(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            overlap_region_group_candidates,
        )

        image = np.full((80, 80, 3), 128, dtype=np.uint8)
        boxes = np.asarray(
            [
                (20, 20, 28, 28),
                (30, 30, 8, 8),
            ],
            dtype=np.int32,
        )
        config = replace(
            DEFAULT_CONFIG,
            merge_distance_px=0,
            min_group_union_region_px=10_000,
            enable_circular_groups=False,
        )

        groups = overlap_region_group_candidates(boxes, image, config)

        self.assertEqual([group.bounds for group in groups], [(20, 20, 28, 28)])

    def test_any_positive_overlap_groups_but_edge_contact_does_not(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            overlap_region_group_candidates,
        )

        image = np.full((80, 80, 3), 128, dtype=np.uint8)
        boxes = np.asarray(
            [
                (20, 20, 20, 20),
                (35, 35, 20, 20),
                (55, 35, 10, 20),
            ],
            dtype=np.int32,
        )
        config = replace(
            DEFAULT_CONFIG,
            merge_distance_px=0,
            min_group_union_region_px=10_000,
            enable_circular_groups=False,
        )

        groups = overlap_region_group_candidates(boxes, image, config)

        self.assertEqual(
            [group.bounds for group in groups],
            [(20, 20, 35, 35), (55, 35, 10, 20)],
        )

    def test_overlap_grouping_runs_before_initial_grouping(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import STEP_INDEX

        self.assertLess(STEP_INDEX["box_filter"], STEP_INDEX["overlap_box_group"])
        self.assertLess(STEP_INDEX["overlap_box_group"], STEP_INDEX["grouped"])

    def test_circles_begin_at_initial_grouping_not_overlap_collapse(self) -> None:
        """Overlap collapse is rectilinear housekeeping, not a shape fit."""
        image = np.full((80, 80, 3), 128, dtype=np.uint8)
        # Four detected marks, because a circle is a claim about a cluster:
        # see ``circular_group_min_regions``.  Their union is unchanged, so
        # this still measures *when* the circle appears.
        boxes = np.asarray(
            (
                (20, 20, 32, 32),
                (25, 25, 12, 12),
                (22, 42, 8, 8),
                (42, 22, 8, 8),
            ),
            dtype=np.int32,
        )
        config = replace(
            DEFAULT_CONFIG,
            enable_box_feature_filter=False,
            enable_circular_groups=True,
        )

        run = run_detection(image, None, config, initial_boxes=boxes)

        self.assertEqual(run.stages[STEP_INDEX["overlap_box_group"]].key, "overlap_box_group")
        self.assertEqual(run.stages[STEP_INDEX["overlap_box_group"]].circles, ())
        self.assertEqual(run.stages[STEP_INDEX["grouped"]].key, "grouped")
        self.assertEqual(run.stages[STEP_INDEX["grouped"]].circles, (16,))

    def test_local_contrast_also_collapses_nested_overlap_before_proximity(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            DetectionState,
            _step_overlap_box_group,
        )

        image = np.full((80, 80, 3), 128, dtype=np.uint8)
        boxes = np.asarray([(20, 20, 28, 28), (30, 30, 8, 8)], dtype=np.int32)
        config = replace(
            DEFAULT_CONFIG,
            box_source="contrast",
            enable_circular_groups=False,
        )

        state, stage = _step_overlap_box_group(
            image, None, config, DetectionState(boxes, []),
        )

        self.assertEqual(stage.kept, ((20, 20, 28, 28),))
        self.assertEqual(state.candidates[0].members, ((20, 20, 28, 28), (30, 30, 8, 8)))

    def test_overlap_grouping_repeats_when_a_union_creates_a_new_overlap(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            DetectionState,
            _step_overlap_box_group,
        )

        image = np.full((64, 64, 3), 128, dtype=np.uint8)
        boxes = np.asarray(
            ((0, 0, 10, 10), (8, 8, 10, 10), (16, 0, 5, 7)),
            dtype=np.int32,
        )
        config = replace(DEFAULT_CONFIG, enable_circular_groups=False)

        state, stage = _step_overlap_box_group(
            image, None, config, DetectionState(boxes, [])
        )

        self.assertEqual(stage.kept, ((0, 0, 21, 18),))
        self.assertEqual(stage.adjusted, 2)
        self.assertEqual(len(state.candidates[0].members), 3)

    def test_initial_grouping_continues_from_overlap_candidates(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            _step_grouped,
            _step_overlap_box_group,
            DetectionState,
        )

        image = np.full((96, 96, 3), 128, dtype=np.uint8)
        boxes = np.asarray(
            [
                (20, 20, 28, 28),
                (30, 30, 8, 8),
                (58, 20, 8, 28),
            ],
            dtype=np.int32,
        )
        config = replace(
            DEFAULT_CONFIG,
            merge_distance_px=18,
            enable_circular_groups=False,
        )

        overlap_state, overlap_stage = _step_overlap_box_group(
            image,
            None,
            config,
            DetectionState(boxes, []),
        )
        grouped_state, grouped_stage = _step_grouped(
            image, None, config, overlap_state
        )

        self.assertEqual(overlap_stage.kept, ((20, 20, 28, 28), (58, 20, 8, 28)))
        self.assertEqual(overlap_stage.adjusted, 1)
        self.assertEqual(grouped_stage.kept, ((20, 20, 46, 28),))
        self.assertEqual(
            grouped_state.candidates[0].members,
            ((20, 20, 28, 28), (30, 30, 8, 8), (58, 20, 8, 28)),
        )
        self.assertEqual(
            grouped_state.candidates[0].units,
            ((20, 20, 28, 28), (58, 20, 8, 28)),
        )

    def test_proximity_grouping_rejects_a_thin_box_on_a_tall_boxs_edge(self) -> None:
        """Shared-axis overlap alone falsely treats this as one text row."""
        image = np.full((128, 256, 3), 128, dtype=np.uint8)
        boxes = np.asarray(
            ((10, 10, 100, 80), (120, 80, 50, 10)), dtype=np.int32,
        )
        config = replace(
            DEFAULT_CONFIG,
            merge_distance_px=20,
            enable_box_feature_filter=False,
            enable_circular_groups=False,
            enable_contrast_continuity_grouping=False,
            enable_relief_edge_bridge_grouping=False,
        )

        run = run_detection(image, None, config, initial_boxes=boxes)

        self.assertEqual(run.stages[STEP_INDEX["grouped"]].kept, tuple(tuple(box) for box in boxes))

    def test_proximity_grouping_keeps_a_thin_box_on_the_same_centreline(self) -> None:
        image = np.full((128, 256, 3), 128, dtype=np.uint8)
        boxes = np.asarray(
            ((10, 10, 100, 80), (120, 45, 50, 10)), dtype=np.int32,
        )
        config = replace(
            DEFAULT_CONFIG,
            merge_distance_px=20,
            enable_box_feature_filter=False,
            enable_circular_groups=False,
            enable_contrast_continuity_grouping=False,
            enable_relief_edge_bridge_grouping=False,
        )

        run = run_detection(image, None, config, initial_boxes=boxes)

        self.assertEqual(run.stages[STEP_INDEX["grouped"]].kept, ((10, 10, 160, 80),))

    def test_nested_detail_is_absorbed_only_by_overlap_grouping(self) -> None:
        """Initial grouping receives the already-collapsed overlap candidate."""
        image = np.full((160, 256, 3), 128, dtype=np.uint8)
        boxes = np.asarray(
            ((10, 10, 180, 100), (130, 70, 20, 12)), dtype=np.int32,
        )
        config = replace(
            DEFAULT_CONFIG,
            merge_distance_px=20,
            enable_box_feature_filter=False,
            enable_circular_groups=False,
            enable_contrast_continuity_grouping=False,
            enable_relief_edge_bridge_grouping=False,
        )

        run = run_detection(image, None, config, initial_boxes=boxes)

        self.assertEqual(run.stages[STEP_INDEX["overlap_box_group"]].kept, ((10, 10, 180, 100),))
        self.assertEqual(run.stages[STEP_INDEX["overlap_box_group"]].adjusted, 1)
        self.assertEqual(run.stages[STEP_INDEX["grouped"]].kept, ((10, 10, 180, 100),))

    def test_domain_recovery_splits_failed_initial_groups_to_overlap_groups(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            DetectionState,
            _step_grouped,
            _step_overlap_box_group,
            _step_region_domain,
        )

        image = np.full((96, 96, 3), 128, dtype=np.uint8)
        uv = np.zeros((96, 96), dtype=bool)
        uv[10:30, 10:30] = True
        uv[10:30, 42:62] = True
        uv[19:20, 30:42] = True
        boxes = np.asarray(
            [
                (10, 10, 20, 20),
                (14, 14, 7, 7),
                (42, 10, 20, 20),
                (46, 14, 7, 7),
            ],
            dtype=np.int32,
        )
        config = replace(
            DEFAULT_CONFIG,
            merge_distance_px=12,
            min_box_uv_coverage=0.75,
            min_region_uv_coverage=1.0,
            enable_circular_groups=False,
        )

        overlap_state, _overlap_stage = _step_overlap_box_group(
            image,
            uv,
            config,
            DetectionState(boxes, []),
        )
        grouped_state, grouped_stage = _step_grouped(
            image, uv, config, overlap_state
        )
        recovered_state, recovered_stage = _step_region_domain(
            image, uv, config, grouped_state
        )

        self.assertEqual(overlap_state.groups, [(10, 10, 20, 20), (42, 10, 20, 20)])
        self.assertEqual(grouped_stage.kept, ((10, 10, 52, 20),))
        self.assertEqual(
            recovered_state.groups,
            [(10, 10, 20, 20), (42, 10, 20, 20)],
        )
        self.assertEqual(recovered_stage.rejected[0], (10, 10, 52, 20))
        self.assertIn("2 strict recovery groups kept", recovered_stage.detail)
        self.assertIn("0 broad rebuilds", recovered_stage.detail)

    def test_domain_recovery_regroups_overlap_units_until_coverage_would_break(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            DetectionState,
            _step_grouped,
            _step_overlap_box_group,
            _step_region_domain,
        )

        image = np.full((104, 104, 3), 128, dtype=np.uint8)
        uv = np.zeros((104, 104), dtype=bool)
        uv[10:30, 10:54] = True
        uv[10:30, 66:86] = True
        uv[19:20, 54:66] = True
        boxes = np.asarray(
            [
                (10, 10, 20, 20),
                (34, 10, 20, 20),
                (66, 10, 20, 20),
            ],
            dtype=np.int32,
        )
        config = replace(
            DEFAULT_CONFIG,
            merge_distance_px=12,
            min_box_uv_coverage=0.75,
            min_region_uv_coverage=1.0,
            enable_circular_groups=False,
        )

        overlap_state, _overlap_stage = _step_overlap_box_group(
            image, uv, config, DetectionState(boxes, [])
        )
        grouped_state, grouped_stage = _step_grouped(
            image, uv, config, overlap_state
        )
        recovered_state, recovered_stage = _step_region_domain(
            image, uv, config, grouped_state
        )

        self.assertEqual(
            overlap_state.groups,
            [(10, 10, 20, 20), (34, 10, 20, 20), (66, 10, 20, 20)],
        )
        self.assertEqual(grouped_stage.kept, ((10, 10, 76, 20),))
        self.assertEqual(
            recovered_state.groups,
            [(10, 10, 44, 20), (66, 10, 20, 20)],
        )
        self.assertIn("2 strict recovery groups kept", recovered_stage.detail)

    def test_domain_recovery_can_split_an_invalid_overlap_group_to_valid_members(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            DetectionState,
            _step_grouped,
            _step_overlap_box_group,
            _step_region_domain,
        )

        image = np.full((64, 64, 3), 128, dtype=np.uint8)
        uv = np.zeros((64, 64), dtype=bool)
        uv[15:40, 10:40] = True
        boxes = np.asarray(
            [(10, 10, 30, 30), (15, 20, 10, 10)],
            dtype=np.int32,
        )
        config = replace(
            DEFAULT_CONFIG,
            min_box_uv_coverage=0.8,
            min_region_uv_coverage=1.0,
            merge_distance_px=0,
            enable_circular_groups=False,
        )

        overlap_state, _overlap_stage = _step_overlap_box_group(
            image, uv, config, DetectionState(boxes, []),
        )
        grouped_state, _grouped_stage = _step_grouped(
            image, uv, config, overlap_state,
        )
        recovered_state, recovered_stage = _step_region_domain(
            image, uv, config, grouped_state,
        )

        self.assertEqual(overlap_state.groups, [(10, 10, 30, 30)])
        self.assertEqual(recovered_state.groups, [(15, 20, 10, 10)])
        self.assertIn("1 overlap member recovered", recovered_stage.detail)

    def test_post_circle_forced_merge_reconnects_cardinal_control(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            DetectionState,
            _step_overlap_group,
        )

        image = np.full((220, 220, 3), 24, dtype=np.uint8)
        # Four D-pad glyphs: no pair is horizontally or vertically aligned,
        # but their enclosing square is an unambiguous, UV-valid circle.
        groups = [(92, 40, 20, 20), (40, 92, 20, 20),
                  (144, 92, 20, 20), (92, 144, 20, 20)]
        # Also the originally detected boxes: four marks, which is what earns
        # the circle in the first place.
        raw = np.asarray(groups, dtype=np.int32)
        config = replace(
            DEFAULT_CONFIG,
            merge_distance_px=100,
            enable_region_domain_filter=True,
            min_region_uv_coverage=1.0,
        )

        state, stage = _step_overlap_group(
            image, np.ones(image.shape[:2], dtype=bool), config,
            DetectionState(raw, groups),
        )

        self.assertEqual(stage.kept, ((40, 40, 124, 124),))
        self.assertEqual(stage.adjusted, 3)
        self.assertEqual(state.groups, [(40, 40, 124, 124)])

    def test_post_circle_forced_merge_reconnects_overlapping_cross_arms(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            DetectionState,
            _step_overlap_group,
        )

        image = np.full((220, 220, 3), 24, dtype=np.uint8)
        # The same D-pad can arrive as one vertical and one horizontal edge
        # component.  They overlap, but only their square union earns a circle.
        groups = [(92, 40, 20, 124), (40, 92, 124, 20)]
        # The arms are an intermediate grouping; the detection underneath them
        # is still four marks, and that is the level the circle rule counts at.
        raw = np.asarray(
            [(92, 40, 20, 20), (40, 92, 20, 20),
             (144, 92, 20, 20), (92, 144, 20, 20)],
            dtype=np.int32,
        )
        config = replace(DEFAULT_CONFIG, merge_distance_px=100)

        state, stage = _step_overlap_group(
            image, np.ones(image.shape[:2], dtype=bool), config,
            DetectionState(raw, groups),
        )

        self.assertEqual(stage.kept, ((40, 40, 124, 124),))
        self.assertEqual(stage.adjusted, 1)
        self.assertEqual(state.groups, [(40, 40, 124, 124)])

    def test_post_circle_forced_merge_does_not_join_only_two_marks(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            DetectionState,
            _step_overlap_group,
        )

        image = np.full((160, 160, 3), 24, dtype=np.uint8)
        groups = [(40, 40, 20, 20), (100, 100, 20, 20)]
        config = replace(DEFAULT_CONFIG, merge_distance_px=100)

        _state, stage = _step_overlap_group(
            image, np.ones(image.shape[:2], dtype=bool), config,
            DetectionState(np.empty((0, 4), dtype=np.int32), groups),
        )

        self.assertEqual(stage.kept, tuple(groups))

    def test_weakly_overlapping_vertical_neighbours_do_not_group(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            union_region_group_candidates,
        )

        image = np.full((90, 90, 3), 128, dtype=np.uint8)
        boxes = np.asarray(
            [
                (20, 20, 30, 8),
                (45, 34, 30, 8),
            ],
            dtype=np.int32,
        )
        config = replace(
            DEFAULT_CONFIG,
            merge_distance_px=18,
            enable_circular_groups=False,
        )

        groups = union_region_group_candidates(boxes, image, config)

        self.assertEqual(
            [group.bounds for group in groups],
            [(20, 20, 30, 8), (45, 34, 30, 8)],
        )

    def test_weakly_overlapping_horizontal_neighbours_do_not_group(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            union_region_group_candidates,
        )

        image = np.full((90, 90, 3), 128, dtype=np.uint8)
        boxes = np.asarray(
            [
                (20, 20, 8, 30),
                (34, 45, 8, 30),
            ],
            dtype=np.int32,
        )
        config = replace(
            DEFAULT_CONFIG,
            merge_distance_px=18,
            enable_circular_groups=False,
        )

        groups = union_region_group_candidates(boxes, image, config)

        self.assertEqual(
            [group.bounds for group in groups],
            [(20, 20, 8, 30), (34, 45, 8, 30)],
        )

    def test_accumulated_row_or_column_group_gets_follow_up_merge(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            union_region_group_candidates,
        )

        image = np.full((128, 160, 3), 128, dtype=np.uint8)
        boxes = np.asarray(
            [
                (20, 20, 100, 8),
                (25, 40, 8, 30),
                (107, 40, 8, 30),
                (65, 85, 20, 12),
            ],
            dtype=np.int32,
        )
        config = replace(
            DEFAULT_CONFIG,
            merge_distance_px=20,
            enable_circular_groups=False,
            # This fixture deliberately exercises the follow-up/accumulated
            # union rather than the default strict same-row/column policy.
            # Its cross-shaped layout is intentionally not centre-aligned.
            group_axis_center_tolerance=10.0,
        )

        groups = union_region_group_candidates(boxes, image, config)

        self.assertEqual([group.bounds for group in groups], [(20, 20, 100, 77)])

    def test_box_uv_coverage_uses_one_island_not_the_total_domain(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            box_uv_coverage,
            build_uv_domain_index,
        )

        uv = np.zeros((80, 100), dtype=bool)
        uv[20:40, 10:30] = True
        uv[20:40, 50:70] = True
        domain = build_uv_domain_index(uv)

        coverage = box_uv_coverage(uv, (10, 20, 60, 20), uv.shape, domain)

        self.assertAlmostEqual(coverage, 400 / 1200)

    def test_box_coverage_gate_cannot_sum_multiple_uv_islands(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            union_region_group_candidates,
        )

        image = np.full((80, 100, 3), 128, dtype=np.uint8)
        uv = np.zeros((80, 100), dtype=bool)
        uv[20:40, 10:30] = True
        uv[20:40, 50:70] = True
        boxes = np.asarray(
            [
                (12, 24, 10, 8),
                (56, 24, 10, 8),
            ],
            dtype=np.int32,
        )
        config = replace(
            DEFAULT_CONFIG,
            merge_distance_px=24,
            min_box_uv_coverage=0.75,
            enable_island_bounded_grouping=False,
            enable_circular_groups=False,
        )

        groups = union_region_group_candidates(boxes, image, config, uv)

        self.assertEqual([group.bounds for group in groups], [(12, 24, 10, 8), (56, 24, 10, 8)])

    def test_domain_breaking_merge_keeps_the_smaller_valid_groups(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            union_region_group_candidates,
        )

        image = np.full((96, 96, 3), 128, dtype=np.uint8)
        uv_pixels = np.zeros((96, 96), dtype=np.uint8)
        uv_pixels[10:32, 10:34] = 255
        uv_pixels[42:64, 42:66] = 255
        cv2.line(uv_pixels, (33, 31), (42, 42), 255, 1)
        uv = uv_pixels > 0
        boxes = np.asarray(
            [
                (13, 14, 7, 8),
                (24, 14, 7, 8),
                (45, 46, 7, 8),
                (56, 46, 7, 8),
            ],
            dtype=np.int32,
        )
        config = replace(
            DEFAULT_CONFIG,
            merge_distance_px=24,
            enable_circular_groups=False,
            min_region_uv_coverage=1.0,
        )

        groups = union_region_group_candidates(boxes, image, config, uv)

        self.assertEqual(
            [group.bounds for group in groups],
            [(13, 14, 18, 8), (45, 46, 18, 8)],
        )

    def test_grouping_uses_box_coverage_before_strict_domain_recovery(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            DetectionState,
            _step_region_domain,
            build_uv_domain_index,
            union_region_group_candidates,
        )

        image = np.full((96, 96, 3), 128, dtype=np.uint8)
        uv = np.zeros((96, 96), dtype=bool)
        uv[20:28, 20:28] = True
        uv[20:21, 28:48] = True
        uv[20:28, 48:56] = True
        boxes = np.asarray(
            [
                (20, 20, 8, 8),
                (48, 20, 8, 8),
            ],
            dtype=np.int32,
        )
        config = replace(
            DEFAULT_CONFIG,
            merge_distance_px=20,
            min_box_uv_coverage=0.4,
            min_region_uv_coverage=1.0,
            enable_circular_groups=False,
        )
        domain = build_uv_domain_index(uv)

        candidates = union_region_group_candidates(boxes, image, config, uv, domain)
        self.assertEqual(
            [candidate.bounds for candidate in candidates],
            [(20, 20, 36, 8)],
        )

        state, stage = _step_region_domain(
            image,
            uv,
            config,
            DetectionState(
                boxes,
                [candidate.bounds for candidate in candidates],
                candidates,
                domain,
            ),
        )

        self.assertEqual(state.groups, [(20, 20, 8, 8), (48, 20, 8, 8)])
        self.assertEqual(stage.rejected[0], (20, 20, 36, 8))


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
        response = edge_response(image, RING_CONFIG)
        roughness = ring_roughness(
            response, None, (88, 108, 96, 40), RING_CONFIG,
            self._reference(image, RING_CONFIG),
        )
        self.assertIsNotNone(roughness)
        self.assertLess(roughness, RING_CONFIG.max_ring_roughness)

    def test_a_grille_reads_rough_around_any_part_of_itself(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            edge_response, ring_roughness,
        )
        image = self._patch_of_grille()
        response = edge_response(image, RING_CONFIG)
        roughness = ring_roughness(
            response, None, (88, 108, 96, 40), RING_CONFIG,
            self._reference(image, RING_CONFIG),
        )
        self.assertIsNotNone(roughness)
        self.assertGreater(roughness, RING_CONFIG.max_ring_roughness)

    def test_a_ring_with_too_little_domain_concludes_nothing(self) -> None:
        # Nothing measured, nothing concluded: a region at an island edge is
        # looking at the void rather than at material.
        from mesh_segmentation_transform.annotate_texture_regions import (
            edge_response, ring_roughness,
        )
        image = self._mark_on_plain_trim()
        response = edge_response(image, RING_CONFIG)
        domain = np.zeros(image.shape, dtype=bool)
        domain[100:140, 100:140] = True
        self.assertIsNone(
            ring_roughness(response, domain, (100, 100, 40, 40), RING_CONFIG, 1.0)
        )

    def test_the_margin_keeps_the_mark_out_of_its_own_ring(self) -> None:
        # Without a gap the mark's outer edge counts as surrounding busyness
        # and every real mark measures rough.
        from mesh_segmentation_transform.annotate_texture_regions import (
            edge_response, ring_roughness,
        )
        image = self._mark_on_plain_trim()
        response = edge_response(image, RING_CONFIG)
        reference = self._reference(image, RING_CONFIG)
        tight = replace(RING_CONFIG, ring_smoothness_margin_px=0)
        bounds = (92, 112, 88, 32)
        self.assertLessEqual(
            ring_roughness(response, None, bounds, RING_CONFIG, reference),
            ring_roughness(response, None, bounds, tight, reference),
        )

    def test_a_hull_ring_measures_the_material_near_the_feature(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            RegionFeatureHull,
            edge_response,
            ring_roughness,
        )

        image = np.full((160, 160), 128, dtype=np.uint8)
        cv2.rectangle(image, (50, 76), (110, 82), 168, -1)
        for x in range(48, 113, 6):
            cv2.line(image, (x, 62), (x + 4, 72), 168, 2)
        response = edge_response(image, RING_CONFIG)
        config = replace(
            RING_CONFIG,
            ring_smoothness_margin_px=2,
            ring_smoothness_width_px=12,
        )
        reference = self._reference(image, RING_CONFIG)
        bounds = (38, 54, 84, 48)
        hull = RegionFeatureHull(
            points=((50, 76), (110, 76), (110, 82), (50, 82)),
            feature_area_px=427,
            hull_area_px=360.0,
            colour_variation=0.0,
        )

        rectangular = ring_roughness(response, None, bounds, config, reference)
        hull_based = ring_roughness(response, None, bounds, config, reference, hull)

        self.assertIsNotNone(rectangular)
        self.assertIsNotNone(hull_based)
        self.assertGreater(hull_based, rectangular)

    def test_the_stage_is_a_no_op_until_enabled(self) -> None:
        run = run_detection(embossed_text(), None, EDGE_CONFIG)
        stage = next(s for s in run.stages if s.key == "ring_smoothness")
        self.assertEqual(stage.detail, "disabled")
        self.assertEqual(len(stage.rejected), 0)

    def test_it_runs_after_blob_shape_and_before_final_size(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import STEP_INDEX
        self.assertLess(STEP_INDEX["grouped"], STEP_INDEX["ring_smoothness"])
        self.assertLess(STEP_INDEX["blob_shape"], STEP_INDEX["ring_smoothness"])
        self.assertLess(STEP_INDEX["ring_smoothness"], STEP_INDEX["size"])


class MagicFeatureFilterTests(unittest.TestCase):
    def _image(self) -> np.ndarray:
        return np.full((128, 128, 3), 160, dtype=np.uint8)

    def _soft_fringe_fixture(
        self, scale: int = 1
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """A solid mark with a three-step antialiased edge outside its box."""
        image = np.full((128 * scale, 128 * scale, 3), 160, dtype=np.uint8)
        # Draw outer-to-inner so every reference-pixel ring scales exactly.
        for low, high, value in (
            (49, 79, 140),
            (50, 78, 110),
            (51, 77, 70),
            (52, 76, 30),
        ):
            cv2.rectangle(
                image,
                (low * scale, low * scale),
                ((high + 1) * scale - 1, (high + 1) * scale - 1),
                (value, value, value),
                -1,
            )
        return image, (52 * scale, 52 * scale, 25 * scale, 25 * scale)

    def test_a_filled_mark_reads_as_a_blob(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import blob_hull_fill

        image = self._image()
        cv2.circle(image, (64, 64), 18, (30, 30, 30), -1)

        fill = blob_hull_fill(image, None, (46, 46, 36, 36), DEFAULT_CONFIG)

        self.assertIsNotNone(fill)
        self.assertGreater(fill, DEFAULT_CONFIG.max_blob_hull_fill)

    def test_a_blob_with_internal_detail_is_not_flat(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            blob_shape_measures,
        )

        image = self._image()
        cv2.circle(image, (64, 64), 22, (20, 20, 20), -1)
        cv2.putText(image, "23", (45, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 3)

        measures = blob_shape_measures(image, None, (40, 40, 48, 48), DEFAULT_CONFIG)

        self.assertIsNotNone(measures)
        fill, colour_variation = measures
        self.assertGreater(fill, DEFAULT_CONFIG.max_blob_hull_fill)
        self.assertGreater(
            colour_variation,
            DEFAULT_CONFIG.min_blob_internal_colour_variation,
        )

    def test_a_distinctive_shape_leaves_empty_hull_space(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import blob_hull_fill

        image = self._image()
        cv2.line(image, (42, 64), (86, 64), (30, 30, 30), 6)
        cv2.line(image, (64, 42), (64, 86), (30, 30, 30), 6)

        fill = blob_hull_fill(image, None, (38, 38, 52, 52), DEFAULT_CONFIG)

        self.assertIsNotNone(fill)
        self.assertLess(fill, DEFAULT_CONFIG.max_blob_hull_fill)

    def test_connected_feature_extension_is_measured(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            feature_extension_measure,
        )

        image = self._image()
        cv2.line(image, (20, 64), (108, 64), (30, 30, 30), 6)
        config = replace(DEFAULT_CONFIG, feature_extension_context_px=28)

        measure = feature_extension_measure(image, None, (50, 59, 28, 12), config)

        self.assertIsNotNone(measure)
        self.assertGreater(measure.extension_area_px, measure.feature_area_px)
        self.assertGreater(
            measure.extension_ratio,
            config.feature_extension_min_ratio,
        )

    def test_isolated_feature_has_no_extension(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            feature_extension_measure,
        )

        image = self._image()
        cv2.rectangle(image, (55, 58), (73, 70), (30, 30, 30), -1)
        config = replace(DEFAULT_CONFIG, feature_extension_context_px=28)

        measure = feature_extension_measure(image, None, (50, 52, 30, 28), config)

        self.assertIsNotNone(measure)
        self.assertEqual(measure.extension_area_px, 0)
        self.assertEqual(measure.extension_ratio, 0.0)

    def test_feature_extension_accepts_a_soft_antialias_collar(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            feature_extension_measure,
        )

        image, group = self._soft_fringe_fixture()
        without_grace = feature_extension_measure(
            image,
            None,
            group,
            replace(DEFAULT_CONFIG, feature_extension_grace_px=0),
        )
        with_grace = feature_extension_measure(
            image,
            None,
            group,
            replace(DEFAULT_CONFIG, feature_extension_grace_px=4),
        )

        self.assertIsNotNone(without_grace)
        self.assertIsNotNone(with_grace)
        self.assertGreater(without_grace.extension_area_px, 0)
        self.assertEqual(with_grace.extension_area_px, 0)
        self.assertEqual(with_grace.extension_ratio, 0.0)
        self.assertGreater(with_grace.soft_fringe_area_px, 0)
        self.assertEqual(with_grace.expanded_bounds, (49, 49, 31, 31))

    def test_feature_extension_rejects_a_hard_continuation_inside_the_grace(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            DetectionState,
            _step_feature_extension,
            feature_extension_measure,
        )

        ratios: list[float] = []
        for scale in (1, 2):
            image = np.full(
                (128 * scale, 128 * scale, 3), 160, dtype=np.uint8
            )
            group = (52 * scale, 52 * scale, 25 * scale, 25 * scale)
            # The same solid paint carries four reference pixels beyond the
            # detector box.  It fits spatially inside the V60 allowance but is
            # not antialiasing, at either native layer resolution.
            cv2.rectangle(
                image,
                (52 * scale, 52 * scale),
                (81 * scale - 1, 77 * scale - 1),
                (30, 30, 30),
                -1,
            )

            measure = feature_extension_measure(
                image, None, group, DEFAULT_CONFIG
            )
            self.assertIsNotNone(measure)
            self.assertEqual(measure.soft_fringe_area_px, 0)
            self.assertGreater(
                measure.extension_ratio,
                DEFAULT_CONFIG.feature_extension_min_ratio,
            )
            ratios.append(measure.extension_ratio)
            _state, stage = _step_feature_extension(
                image,
                None,
                DEFAULT_CONFIG,
                DetectionState(np.empty((0, 4), np.int32), [group]),
            )
            self.assertEqual(stage.rejected, (group,))
        self.assertAlmostEqual(ratios[0], ratios[1])

    def test_feature_extension_decision_is_invariant_at_double_resolution(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            feature_extension_measure,
        )

        image_1x, group_1x = self._soft_fringe_fixture(1)
        image_2x, group_2x = self._soft_fringe_fixture(2)
        measure_1x = feature_extension_measure(
            image_1x, None, group_1x, DEFAULT_CONFIG
        )
        measure_2x = feature_extension_measure(
            image_2x, None, group_2x, DEFAULT_CONFIG
        )

        self.assertIsNotNone(measure_1x)
        self.assertIsNotNone(measure_2x)
        self.assertEqual(measure_1x.extension_ratio, 0.0)
        self.assertEqual(measure_2x.extension_ratio, 0.0)
        self.assertEqual(measure_2x.grace_px, measure_1x.grace_px * 2)
        self.assertEqual(measure_2x.search_px, measure_1x.search_px * 2)
        self.assertEqual(
            measure_2x.expanded_bounds,
            tuple(value * 2 for value in measure_1x.expanded_bounds),
        )

    def test_feature_extension_grace_is_fractionally_bounded_on_tiny_marks(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            feature_extension_geometry,
        )

        tiny = feature_extension_geometry((0, 0, 4, 4), DEFAULT_CONFIG)
        reference = feature_extension_geometry((0, 0, 25, 25), DEFAULT_CONFIG)

        self.assertEqual(tiny.grace_px, 0)
        self.assertEqual(reference.grace_px, 4)
        self.assertLessEqual(
            reference.grace_px / 25,
            DEFAULT_CONFIG.feature_extension_grace_max_fraction,
        )

    def test_accepted_soft_fringe_expands_the_final_output_bounds(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            DetectionState,
            _step_feature_extension,
            _step_final_padding,
        )

        image, group = self._soft_fringe_fixture()
        initial = DetectionState(np.empty((0, 4), np.int32), [group])
        extended, extension_stage = _step_feature_extension(
            image, None, DEFAULT_CONFIG, initial
        )
        final, _padding_stage = _step_final_padding(
            image,
            None,
            replace(DEFAULT_CONFIG, enable_circular_groups=False),
            extended,
        )

        self.assertEqual(extension_stage.kept, ((49, 49, 31, 31),))
        self.assertEqual(extension_stage.adjusted, 1)
        # Final padding is additional; every soft pixel from 49..79 is inside.
        self.assertEqual(final.groups, [(47, 47, 35, 35)])

    def test_accepted_soft_fringe_expands_a_rotated_output_stencil(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            DetectionState,
            _step_feature_extension,
            _step_final_padding,
        )
        from mesh_segmentation_transform.mirror_texture_for_rhd import (
            apply_masked_rotated_flip,
        )

        image, group = self._soft_fringe_fixture()
        original = tuple(
            (float(x), float(y))
            for x, y in cv2.boxPoints(((64.0, 64.0), (25.0, 25.0), 17.0))
        )
        initial = DetectionState(
            np.empty((0, 4), np.int32), [group], rotations=[original]
        )

        extended, _extension_stage = _step_feature_extension(
            image, None, DEFAULT_CONFIG, initial
        )
        final, _padding_stage = _step_final_padding(
            image,
            None,
            replace(DEFAULT_CONFIG, enable_circular_groups=False),
            extended,
        )

        expanded = extended.rotations[0]
        self.assertIsNotNone(expanded)
        self.assertNotEqual(expanded, original)
        # Padding grows it further rather than leaving it alone, so the
        # relationship to assert is containment, not equality.
        (padded,) = final.rotations
        self.assertIsNotNone(padded)
        padded_polygon = np.asarray(padded, dtype=np.float32)
        for point in expanded:
            self.assertGreaterEqual(
                cv2.pointPolygonTest(padded_polygon, tuple(point), True), -1e-4
            )
        expanded_polygon = np.asarray(expanded, dtype=np.float32)
        for point in ((49, 49), (79, 49), (79, 79), (49, 79)):
            self.assertGreaterEqual(
                cv2.pointPolygonTest(expanded_polygon, point, True), -1e-4
            )

        # Exercise the production write shape, not just its reported outline.
        rows, columns = np.indices(image.shape[:2])
        source = np.zeros_like(image)
        source[:, :, 0] = columns
        source[:, :, 1] = rows
        old_output = source.copy()
        expanded_output = source.copy()
        stencil = np.ones(image.shape[:2], dtype=bool)
        apply_masked_rotated_flip(old_output, stencil, original, "long")
        apply_masked_rotated_flip(expanded_output, stencil, expanded, "long")
        self.assertTrue(np.array_equal(old_output[49, 49], source[49, 49]))
        self.assertFalse(np.array_equal(expanded_output[49, 49], source[49, 49]))

    def test_blob_stage_rejects_only_blob_like_regions(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            DetectionState,
            _step_blob_shape,
        )

        image = self._image()
        cv2.circle(image, (36, 64), 14, (30, 30, 30), -1)
        cv2.line(image, (78, 64), (114, 64), (30, 30, 30), 5)
        cv2.line(image, (96, 46), (96, 82), (30, 30, 30), 5)
        groups = [(22, 50, 28, 28), (74, 42, 44, 44)]
        config = replace(DEFAULT_CONFIG, enable_blob_shape_filter=True)

        _state, stage = _step_blob_shape(
            image, None, config, DetectionState(np.empty((0, 4), np.int32), groups)
        )

        self.assertEqual(stage.rejected, (groups[0],))
        self.assertEqual(stage.kept, (groups[1],))

    def _dark_trim(self) -> np.ndarray:
        """Near-black trim, the ground both a stitch and a legend sit on."""
        return np.full((128, 128, 3), 20, dtype=np.uint8)

    def test_blob_stage_judges_regions_the_size_of_a_stitch_dash(self) -> None:
        """The area floor used to exempt the one shape that is plainly a blob.

        At 512 px^2 the floor was larger than most stitch dashes, so the filter
        never looked at them: measured over both V60 stitch atlases, 97 of 97
        dashes fill their hull at 1.09-1.15 with a colour range of 0.0-1.0, and
        every one was skipped on size alone.
        """
        from mesh_segmentation_transform.annotate_texture_regions import (
            DetectionState,
            _step_blob_shape,
        )

        image = self._dark_trim()
        # Flat-painted, the way a stitch is drawn: one solid colour, no blend
        # into the trim at any edge.
        image[56:78, 60:70] = (70, 70, 70)
        group = (57, 53, 16, 28)
        config = replace(DEFAULT_CONFIG, enable_blob_shape_filter=True)

        self.assertLess(group[2] * group[3], 512)
        _state, stage = _step_blob_shape(
            image, None, config, DetectionState(np.empty((0, 4), np.int32), [group])
        )

        self.assertEqual(stage.rejected, (group,))
        self.assertNotIn("skipped", stage.detail)

    def test_a_small_antialiased_mark_survives_the_lowered_floor(self) -> None:
        """What keeps real marks is the colour range, never the area floor.

        A cursor arrowhead is exactly as hull-filling as a stitch dash -- the
        scintilla cluster's four all read 1.09 -- and is kept because its edges
        blend into the trim behind it, ranging 82-84 where a flat-painted dash
        ranges 0-1.
        """
        from mesh_segmentation_transform.annotate_texture_regions import (
            DetectionState,
            _step_blob_shape,
        )

        image = self._dark_trim()
        arrow = np.array([[64, 56], [73, 73], [55, 73]], dtype=np.int32)
        cv2.fillPoly(image, [arrow], (235, 235, 235), lineType=cv2.LINE_AA)
        group = (54, 55, 21, 19)  # the size the real ones are
        config = replace(DEFAULT_CONFIG, enable_blob_shape_filter=True)

        self.assertLess(group[2] * group[3], 512)
        _state, stage = _step_blob_shape(
            image, None, config, DetectionState(np.empty((0, 4), np.int32), [group])
        )

        self.assertEqual(stage.kept, (group,))
        self.assertEqual(stage.rejected, ())

    def test_blob_stage_keeps_blob_like_regions_with_internal_detail(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            DetectionState,
            _step_blob_shape,
        )

        image = self._image()
        cv2.circle(image, (64, 64), 22, (20, 20, 20), -1)
        cv2.putText(image, "23", (45, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 3)
        group = (40, 40, 48, 48)
        config = replace(DEFAULT_CONFIG, enable_blob_shape_filter=True)

        _state, stage = _step_blob_shape(
            image, None, config, DetectionState(np.empty((0, 4), np.int32), [group])
        )

        self.assertEqual(stage.rejected, ())
        self.assertEqual(stage.kept, (group,))

    def test_blob_stage_skips_regions_below_the_area_floor(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            DetectionState,
            _step_blob_shape,
        )

        image = self._image()
        cv2.circle(image, (64, 64), 8, (30, 30, 30), -1)
        group = (56, 56, 16, 16)
        config = replace(
            DEFAULT_CONFIG,
            enable_blob_shape_filter=True,
            min_blob_region_area_px=512,
        )

        _state, stage = _step_blob_shape(
            image, None, config, DetectionState(np.empty((0, 4), np.int32), [group])
        )

        self.assertEqual(stage.rejected, ())
        self.assertEqual(stage.kept, (group,))

    def test_blob_stage_caches_feature_hulls_for_ring_smoothness(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            DetectionState,
            _step_blob_shape,
        )

        image = self._image()
        cv2.rectangle(image, (44, 60), (84, 68), (30, 30, 30), -1)
        group = (36, 50, 60, 28)
        config = replace(
            DEFAULT_CONFIG,
            enable_blob_shape_filter=False,
            enable_ring_smoothness_filter=True,
        )

        state, stage = _step_blob_shape(
            image, None, config, DetectionState(np.empty((0, 4), np.int32), [group])
        )

        self.assertEqual(stage.kept, (group,))
        self.assertEqual(len(state.feature_hulls), 1)
        self.assertIsNotNone(state.feature_hulls[0])

    def test_feature_extension_stage_rejects_partial_larger_features(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            DetectionState,
            _step_feature_extension,
        )

        image = self._image()
        cv2.line(image, (12, 42), (80, 42), (30, 30, 30), 6)
        cv2.rectangle(image, (86, 78), (106, 94), (30, 30, 30), -1)
        groups = [(34, 36, 22, 14), (82, 72, 30, 28)]
        config = replace(
            DEFAULT_CONFIG,
            enable_feature_extension_filter=True,
            feature_extension_context_px=28,
            feature_extension_min_ratio=0.25,
        )

        _state, stage = _step_feature_extension(
            image, None, config, DetectionState(np.empty((0, 4), np.int32), groups)
        )

        self.assertEqual(stage.rejected, (groups[0],))
        self.assertEqual(stage.kept, (groups[1],))

    def test_feature_extension_stage_keeps_small_raw_extensions_on_large_features(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            DetectionState,
            _step_feature_extension,
        )

        image = self._image()
        cv2.rectangle(image, (35, 40), (85, 88), (30, 30, 30), -1)
        cv2.rectangle(image, (86, 60), (94, 68), (30, 30, 30), -1)
        group = (34, 39, 52, 50)
        config = replace(
            DEFAULT_CONFIG,
            enable_feature_extension_filter=True,
            feature_extension_context_px=16,
            feature_extension_min_ratio=0.25,
        )

        _state, stage = _step_feature_extension(
            image, None, config, DetectionState(np.empty((0, 4), np.int32), [group])
        )

        self.assertEqual(stage.rejected, ())
        self.assertEqual(stage.kept, (group,))

    def test_the_stages_are_no_ops_until_enabled(self) -> None:
        config = replace(
            EDGE_CONFIG,
            enable_blob_shape_filter=False,
            enable_feature_extension_filter=False,
        )
        run = run_detection(embossed_text(), None, config)
        blob = next(s for s in run.stages if s.key == "blob_shape")
        extension = next(s for s in run.stages if s.key == "feature_extension")
        self.assertEqual(blob.detail, "disabled")
        self.assertEqual(extension.detail, "disabled")
        self.assertEqual(len(blob.rejected), 0)
        self.assertEqual(len(extension.rejected), 0)

    def test_they_run_before_ring_smoothness_and_text_lines(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import STEP_INDEX

        self.assertLess(STEP_INDEX["blob_shape"], STEP_INDEX["feature_extension"])
        self.assertLess(STEP_INDEX["blob_shape"], STEP_INDEX["ring_smoothness"])
        self.assertLess(STEP_INDEX["ring_smoothness"], STEP_INDEX["feature_extension"])
        self.assertLess(STEP_INDEX["feature_extension"], STEP_INDEX["text_line"])


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
        self.assertLess(STEP_INDEX["region_flatness"], STEP_INDEX["blob_shape"])
        self.assertLess(STEP_INDEX["blob_shape"], STEP_INDEX["ring_smoothness"])
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

    def test_circular_regions_keep_their_circle_instead_of_a_rotated_outline(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            DetectionState,
            _step_rotated_bounds,
            inscribed_circle_radius,
        )

        image = np.full((96, 96, 3), 128, dtype=np.uint8)
        cv2.circle(image, (48, 48), 24, (220, 220, 220), -1)
        group = (22, 22, 52, 52)
        # Four detected marks inside it: a circle describes an arrangement, so
        # one squarish mark no longer earns one.  See circular_group_min_regions.
        boxes = np.asarray(
            [(44, 26, 8, 8), (26, 44, 8, 8), (62, 44, 8, 8), (44, 62, 8, 8)],
            dtype=np.int32,
        )
        config = replace(
            DEFAULT_CONFIG,
            enable_circular_groups=True,
            enable_rotated_bounds_filter=True,
            rotated_bounds_min_points=8,
            min_rotated_fill=0.0,
            max_rotated_aspect=10.0,
            min_feature_tightness=0.0,
            min_rotated_elongation=0.0,
        )

        self.assertIsNotNone(
            inscribed_circle_radius(image, None, group, config, None, boxes)
        )

        _state, stage = _step_rotated_bounds(
            image,
            None,
            config,
            DetectionState(boxes, [group]),
        )

        self.assertEqual(stage.kept, (group,))
        self.assertEqual(stage.rotations, (None,))

    def test_circle_domain_candidate_never_expands_past_its_rectangle(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            inscribed_circle_radius,
        )

        image = np.full((64, 64, 3), 128, dtype=np.uint8)
        group = (18, 18, 20, 20)
        config = replace(
            DEFAULT_CONFIG,
            enable_circular_groups=True,
        )

        radius = inscribed_circle_radius(image, None, group, config)

        self.assertEqual(radius, 10)
        self.assertLessEqual(radius * 2, min(group[2], group[3]))
        odd_radius = inscribed_circle_radius(
            image, None, (18, 18, 21, 21), config,
        )
        self.assertEqual(odd_radius, 10)
        self.assertLessEqual(odd_radius * 2, 21)

    def test_too_few_points_fits_nothing(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            rotated_feature_bounds,
        )
        mask = np.zeros((64, 64), dtype=bool)
        mask[10, 10] = True
        self.assertIsNone(
            rotated_feature_bounds(mask, (0, 0, 40, 40), DEFAULT_CONFIG)
        )

    def test_the_filter_rejects_nothing_until_enabled(self) -> None:
        """Shape fitting is not the filter, and is not gated behind it.

        ``enable_rotated_bounds_filter`` decides whether a region is *rejected*
        for filling its box badly.  Which shape describes a region it keeps is
        a different question, and a sheared mark needs its parallelogram
        whether or not that filter is on -- gating one on the other left the
        stage reporting "disabled" while the marks it exists for went out as
        plain boxes.
        """
        run = run_detection(embossed_text(), None, EDGE_CONFIG)
        stage = next(s for s in run.stages if s.key == "rotated_bounds")
        self.assertEqual(len(stage.rejected), 0)

    def test_the_stage_is_a_no_op_when_both_jobs_are_off(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            DetectionState, _step_rotated_bounds,
        )
        state = DetectionState(np.empty((0, 4), dtype=np.int32), [(4, 4, 40, 20)])
        _new, stage = _step_rotated_bounds(
            embossed_text(), None,
            replace(
                EDGE_CONFIG,
                enable_rotated_bounds_filter=False,
                enable_parallelogram_bounds=False,
            ),
            state,
        )
        self.assertIn("disabled", stage.detail)
        self.assertEqual(len(stage.rejected), 0)
        self.assertEqual(len(stage.kept), 1)

    def test_it_runs_before_the_filters_that_read_its_shape(self) -> None:
        """The repeating-pattern score is measured over the axis-aligned crop.

        For a tilted or sheared mark most of that crop is the material around
        it, so run ahead of the fit it scored the background's periodicity and
        called the answer the region's.
        """
        from mesh_segmentation_transform.annotate_texture_regions import STEP_INDEX
        self.assertEqual(
            STEP_INDEX["rotated_bounds"] + 1, STEP_INDEX["pattern_group"]
        )
        self.assertLess(
            STEP_INDEX["pattern_group"], STEP_INDEX["region_flatness"]
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

class SymmetryRotationTests(unittest.TestCase):
    """The angle a mark is mirrored about is its own axis of symmetry.

    Taking it from a nearby UV edge borrows a neighbour's direction; measuring
    the mark's own reflective symmetry asks the question directly.  Measured on
    the magic-wand feature and not on its convex hull -- over 15 ardente marks
    the hull scored 0.83-0.99, a spread of 0.17 with everything bunched at the
    top, because convexity forces high self-overlap whatever is inside it.
    """

    def _feature(self, draw, size: int = 128) -> np.ndarray:
        mask = np.zeros((size, size), dtype=np.uint8)
        draw(mask)
        return mask > 0

    def test_a_tilted_bar_reports_its_own_angle(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            feature_symmetry_axis,
        )
        feature = self._feature(lambda m: cv2.fillConvexPoly(
            m, cv2.boxPoints(((64.0, 64.0), (70.0, 16.0), 30.0)).astype(np.int32), 255
        ))

        axis = feature_symmetry_axis(feature, DEFAULT_CONFIG)

        self.assertIsNotNone(axis)
        self.assertAlmostEqual(axis.angle_degrees % 90.0, 30.0, delta=2.0)
        self.assertGreater(axis.score, 0.9)

    def test_a_disc_is_symmetric_about_everything_and_so_claims_nothing(self) -> None:
        """A round mark peaks at an angle chosen by noise; spread catches it."""
        from mesh_segmentation_transform.annotate_texture_regions import (
            feature_symmetry_axis,
        )
        feature = self._feature(lambda m: cv2.circle(m, (64, 64), 40, 255, -1))

        axis = feature_symmetry_axis(feature, DEFAULT_CONFIG)

        self.assertIsNotNone(axis)
        self.assertGreater(axis.score, 0.95)  # symmetric about every axis
        self.assertGreater(axis.spread, DEFAULT_CONFIG.max_rotation_symmetry_spread)

    def test_lettering_separates_by_whether_it_is_symmetric(self) -> None:
        """The shape real marks actually take, rather than a solid blob."""
        from mesh_segmentation_transform.annotate_texture_regions import (
            feature_symmetry_axis,
        )

        def score(letter: str) -> float:
            feature = self._feature(lambda m: cv2.putText(
                m, letter, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 3.0, 255, 8
            ))
            axis = feature_symmetry_axis(feature, DEFAULT_CONFIG)
            self.assertIsNotNone(axis, letter)
            return axis.score

        for letter in ("A", "T"):
            self.assertGreater(score(letter), DEFAULT_CONFIG.min_rotation_symmetry, letter)
        for letter in ("F", "L", "R"):
            self.assertLess(score(letter), DEFAULT_CONFIG.min_rotation_symmetry, letter)

    def test_a_solid_convex_blob_is_the_known_soft_spot(self) -> None:
        """Recorded rather than fixed, because nothing here can fix it.

        A convex shape overlaps its own reflection well however asymmetric it
        is -- a scalene triangle measures about 0.78 -- so the score alone will
        not reject one.  It does not need to: a solid convex blob is what
        ``enable_blob_shape_filter`` exists to remove, and an angle on a shape
        with no internal structure changes nothing when it is mirrored.
        """
        from mesh_segmentation_transform.annotate_texture_regions import (
            feature_symmetry_axis,
        )
        feature = self._feature(lambda m: cv2.fillConvexPoly(
            m, np.array([[64, 20], [100, 100], [50, 96]], dtype=np.int32), 255
        ))

        axis = feature_symmetry_axis(feature, DEFAULT_CONFIG)

        self.assertIsNotNone(axis)
        self.assertGreater(axis.score, 0.7)
        self.assertLess(axis.score, 0.85)

    def test_too_few_feature_pixels_conclude_nothing(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            feature_symmetry_axis,
        )
        feature = np.zeros((32, 32), dtype=bool)
        feature[4:6, 4:6] = True
        self.assertIsNone(feature_symmetry_axis(feature, DEFAULT_CONFIG))


    def test_the_numpy_hull_raster_matches_opencv(self) -> None:
        """The fill is half-plane intersection, not cv2.fillConvexPoly."""
        from mesh_segmentation_transform.annotate_texture_regions import (
            rasterise_convex_hull,
        )
        points = ((10, 12), (58, 10), (70, 44), (34, 60), (12, 40))

        filled, origin = rasterise_convex_hull(points, 4096)  # 1 cell per pixel

        reference = np.zeros(filled.shape, dtype=np.uint8)
        cv2.fillConvexPoly(
            reference,
            (np.asarray(points, dtype=np.int32) - np.asarray(origin, dtype=np.int32)),
            255,
        )
        overlap = (filled & (reference > 0)).sum()
        union = (filled | (reference > 0)).sum()
        self.assertGreater(overlap / union, 0.95)

    def test_a_degenerate_hull_rasterises_to_nothing(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            rasterise_convex_hull,
        )
        self.assertIsNone(rasterise_convex_hull(((1, 1), (2, 2)), 64))
        self.assertIsNone(rasterise_convex_hull(((1, 1), (1, 1), (1, 1)), 64))

    def test_the_outline_keeps_the_mark_the_way_round_it_was(self) -> None:
        """Extent is re-measured in the axis frame, never carried from the fit.

        A fitted rectangle's size is expressed along *its* axes, so pasting a
        different angle onto it leaves the two dimensions on the wrong sides;
        and the symmetry angle turns the mirror line off vertical while
        OpenCV's rectangle angle turns its width side off horizontal, which is
        another 90 degrees apart.  Together they drew a portrait box around a
        landscape mark.
        """
        from mesh_segmentation_transform.annotate_texture_regions import (
            feature_symmetry_axis, symmetry_aligned_outline,
        )

        for angle in (0.0, 20.0, 70.0, 155.0):
            mask = np.zeros((200, 200), dtype=np.uint8)
            cv2.fillConvexPoly(
                mask,
                cv2.boxPoints(((100.0, 100.0), (90.0, 24.0), angle)).astype(np.int32),
                255,
            )
            feature = mask > 0
            axis = feature_symmetry_axis(feature, DEFAULT_CONFIG)
            self.assertIsNotNone(axis, angle)

            outline = symmetry_aligned_outline(feature, (0, 0), axis)

            self.assertIsNotNone(outline, angle)
            points = np.asarray(outline)
            sides = [
                float(np.hypot(*(points[(i + 1) % 4] - points[i]))) for i in range(4)
            ]
            longest, shortest = max(sides), min(sides)
            # The bar is 90 x 24; the outline must come back that way round.
            self.assertAlmostEqual(longest, 90.0, delta=4.0, msg=str(angle))
            self.assertAlmostEqual(shortest, 24.0, delta=4.0, msg=str(angle))

    def test_a_squarish_mark_is_not_gated_by_edge_fit_elongation(self) -> None:
        """The case the symmetry path was adopted for.

        ``min_rotated_elongation`` exists because the *edge fit* cannot claim a
        direction for a squarish mark.  Symmetry can -- it carries its own
        degeneracy guard -- so it must be asked before that gate, or it never
        sees the marks it was brought in to handle.  The ardente's tilted
        footwell-vent icon is aspect 1.76 against a gate of 2.5, and reads a
        sharp axis 7.6 degrees off the atlas.
        """
        from mesh_segmentation_transform.annotate_texture_regions import (
            DetectionState, _step_rotated_bounds,
        )

        image = np.full((256, 256), 130, dtype=np.uint8)
        # A squarish but plainly one-way mark: a wide bar over a narrow one,
        # the whole thing turned well off axis.
        for size, offset in (((90.0, 18.0), -22.0), ((40.0, 16.0), 14.0)):
            cv2.fillConvexPoly(
                image,
                cv2.boxPoints(((128.0, 128.0 + offset), size, 8.0)).astype(np.int32),
                30,
            )
        image = np.repeat(image[:, :, None], 3, axis=2)
        group = (70, 80, 116, 96)
        config = replace(
            MARKS_CONFIG,
            enable_symmetry_rotation=True,
            min_rotation_angle_degrees=0.5,
            # Pinned rather than inherited: what is under test is that symmetry
            # is asked before the elongation gate, not where the bar happens to
            # sit today.
            min_rotation_symmetry=0.70,
            # A squarish region with quiet corners inscribes a circle, and that
            # branch answers before any outline question is asked.  Not what is
            # under test here.
            enable_circular_groups=False,
        )
        state = DetectionState(np.empty((0, 4), np.int32), [group])

        _state, stage = _step_rotated_bounds(image, None, config, state)

        self.assertEqual(len(stage.kept), 1)
        self.assertIsNotNone(
            stage.rotations[0],
            f"squarish mark got no outline: {stage.detail}",
        )


class NegligibleRotationTests(unittest.TestCase):
    """A rotation of a fraction of a degree is quantisation, not tilt.

    Measured over the 147 marks the shipped plans flipped, 55% sit within a
    quarter of a degree of an atlas axis; adopting such an angle resamples the
    region when it is mirrored, softening every edge in it to move it nowhere.
    """

    def test_distance_is_measured_to_the_nearest_axis(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            rotation_off_axis_degrees,
        )
        for angle, expected in (
            (0.0, 0.0), (0.3, 0.3), (89.7, 0.3), (90.0, 0.0),
            (179.6, 0.4), (45.0, 45.0), (-0.4, 0.4), (135.0, 45.0),
        ):
            self.assertAlmostEqual(
                rotation_off_axis_degrees(angle), expected, places=6, msg=str(angle)
            )

    def test_a_sub_degree_turn_is_declined(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            _rotation_is_negligible,
        )
        config = replace(DEFAULT_CONFIG, min_rotation_angle_degrees=0.5)
        self.assertTrue(_rotation_is_negligible(0.2, config))
        self.assertTrue(_rotation_is_negligible(89.9, config))
        self.assertFalse(_rotation_is_negligible(0.6, config))
        self.assertFalse(_rotation_is_negligible(20.0, config))

    def test_a_zero_floor_disables_the_gate(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            _rotation_is_negligible,
        )
        config = replace(DEFAULT_CONFIG, min_rotation_angle_degrees=0.0)
        self.assertFalse(_rotation_is_negligible(0.0, config))

    def test_an_axis_aligned_bar_keeps_its_box(self) -> None:
        """The end-to-end case the gate exists for."""
        image = np.full((512, 512), 128, dtype=np.uint8)
        cv2.rectangle(image, (330, 220), (496, 246), 200, -1)
        image = np.repeat(image[:, :, None], 3, axis=2)
        config = replace(MARKS_CONFIG, min_rotation_angle_degrees=0.5)

        run = run_detection(image, None, config)

        stage = run.stages[STEP_INDEX["rotated_bounds"]]
        self.assertGreater(len(stage.kept), 0)
        self.assertEqual([r for r in stage.rotations if r], [])
        self.assertIn("straightened back to their box", stage.detail)

    def test_the_same_bar_tilted_does_earn_an_outline(self) -> None:
        image = np.full((512, 512), 128, dtype=np.uint8)
        cv2.fillConvexPoly(
            image,
            cv2.boxPoints(((413.0, 233.0), (166.0, 26.0), 20.0)).astype(np.int32),
            200,
        )
        image = np.repeat(image[:, :, None], 3, axis=2)
        config = replace(MARKS_CONFIG, min_rotation_angle_degrees=0.5)

        run = run_detection(image, None, config)

        stage = run.stages[STEP_INDEX["rotated_bounds"]]
        self.assertTrue([r for r in stage.rotations if r])


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
    # Elongated, solidly filled and genuinely tilted: the one mark here that
    # earns a rotated outline, so the adoption rules have something to accept.
    # It has to be off-axis to earn one -- an axis-aligned bar fits a rectangle
    # turned by 0 degrees, which `min_rotation_angle_degrees` declines because
    # a rotation of nothing only costs a resample.
    cv2.fillConvexPoly(
        image,
        cv2.boxPoints(((413.0, 233.0), (166.0, 26.0), 20.0)).astype(np.int32),
        200,
    )
    return np.repeat(image[:, :, None], 3, axis=2)


MARKS_CONFIG = replace(
    DEFAULT_CONFIG, box_source="edge", edge_local_window_px=64, edge_local_k=1.6,
    enable_box_feature_filter=False, enable_rotated_bounds_filter=True,
    enable_blob_shape_filter=False, enable_feature_extension_filter=False,
    merge_distance_px=18,
    # These fixtures are flat-painted bars and strokes on a uniform ground, and
    # a long straight stroke matches itself at every shift along its own
    # length, so the recurrence filter reads them as repeating material.  These
    # tests are about rotated-outline plumbing, not about that judgement.
    enable_repeat_texture_filter=False,
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
                 if s.key in ("region_flatness", "blob_shape",
                              "ring_smoothness", "size", "final_padding")]
        self.assertEqual(len(after), 5)
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

    def test_padding_grows_the_outline_as_well_as_the_box(self) -> None:
        """The outline is the shape the mirror writes through.

        Growing only the box left every rotated and sheared region with no
        margin at all, which are the ones whose corners need it most: the
        padding exists so a glyph's antialiased edge is carried over with it.
        """
        run = self._run(final_region_padding_px=6)
        size = next(s for s in run.stages if s.key == "size")
        padding = next(s for s in run.stages if s.key == "final_padding")
        self.assertNotEqual(size.kept, padding.kept, "padding changed nothing")
        compared = 0
        for before, after in zip(size.rotations, padding.rotations):
            if before is None:
                self.assertIsNone(after)
                continue
            compared += 1
            self.assertNotEqual(before, after)
            grown = np.asarray(after, dtype=np.float32)
            for point in before:
                self.assertGreaterEqual(
                    cv2.pointPolygonTest(grown, tuple(point), True), -1e-4
                )
        self.assertGreater(compared, 0, "no outline was exercised")

    def test_padding_keeps_the_angle_it_was_given(self) -> None:
        """Clipping to the image would change it; reducing the margin does not."""
        run = self._run(final_region_padding_px=6)
        size = next(s for s in run.stages if s.key == "size")
        padding = next(s for s in run.stages if s.key == "final_padding")
        for before, after in zip(size.rotations, padding.rotations):
            if before is None:
                continue
            for index in range(4):
                first = np.asarray(before[(index + 1) % 4]) - np.asarray(before[index])
                second = np.asarray(after[(index + 1) % 4]) - np.asarray(after[index])
                cross = float(first[0] * second[1] - first[1] * second[0])
                self.assertAlmostEqual(
                    cross / (np.linalg.norm(first) * np.linalg.norm(second)),
                    0.0,
                    places=6,
                )

    def test_no_padding_leaves_every_shape_alone(self) -> None:
        """Zero padding grows nothing -- but the bounds may still widen.

        An outline can already reach slightly outside the box it was fitted
        in, and the bounds have to contain what gets written through them, so
        the union is applied whatever the padding is.  What must not happen is
        a shape changing, or a box shrinking.
        """
        run = self._run(final_region_padding_px=0)
        size = next(s for s in run.stages if s.key == "size")
        padding = next(s for s in run.stages if s.key == "final_padding")
        self.assertEqual(size.rotations, padding.rotations)
        for before, after in zip(size.kept, padding.kept):
            self.assertLessEqual(after[0], before[0])
            self.assertLessEqual(after[1], before[1])
            self.assertGreaterEqual(after[0] + after[2], before[0] + before[2])
            self.assertGreaterEqual(after[1] + after[3], before[1] + before[3])

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
        """Both descriptions are still available; the stage asks for one.

        Nothing configurable chooses between them any more -- the rotated
        rectangle is named at the call site -- but the shape can still describe
        itself either way, and the hull is what the symmetry axis is read off.
        """
        mask = np.zeros((64, 64), dtype=np.uint8)
        cv2.fillConvexPoly(
            mask, np.array([[20, 20], [44, 24], [40, 44], [22, 38]], np.int32), 255
        )
        shape = self._shape(mask > 0, (16, 16, 34, 34))

        self.assertIsNotNone(shape)
        self.assertLess(shape.hull_area, shape.rectangle_area)
        self.assertGreater(shape.tightness(SHAPE_HULL), shape.tightness(SHAPE_ROTATED))

    def test_the_shape_still_outlines_either_way(self) -> None:
        mask = np.zeros((64, 64), dtype=np.uint8)
        cv2.fillConvexPoly(
            mask, np.array([[20, 20], [44, 24], [40, 44], [22, 38]], np.int32), 255
        )
        box = (16, 16, 34, 34)

        self.assertEqual(len(self._shape(mask > 0, box).outline(SHAPE_ROTATED)), 4)
        self.assertGreater(len(self._shape(mask > 0, box).outline(SHAPE_HULL)), 4)

    def test_the_rectangle_is_what_the_shape_reports_by_default(self) -> None:
        mask = np.zeros((64, 64), dtype=np.uint8)
        cv2.rectangle(mask, (20, 20), (44, 44), 255, -1)
        shape = self._shape(mask > 0, (16, 16, 34, 34))

        self.assertEqual(shape.tightness(), shape.tightness(SHAPE_ROTATED))
        self.assertEqual(shape.outline(), shape.outline(SHAPE_ROTATED))
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
        self.assertGreater(shape.tightness(), 0.9)

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
        self.assertLess(shape.tightness(), DEFAULT_CONFIG.min_feature_tightness)

    def test_tightness_over_one_is_normal(self) -> None:
        # The rectangle is a polygon through pixel centres while the feature
        # is counted in whole pixels, so a small solid mark can exceed 1.
        image = np.zeros((64, 64), np.uint8)
        cv2.rectangle(image, (28, 28), (34, 34), 255, -1)
        shape = self._shape(image > 0, (26, 26, 11, 11))
        self.assertGreater(shape.tightness(), 1.0)

    def test_the_region_is_kept_whichever_shape_is_adopted(self) -> None:
        # Declining the outline must not remove the region: how well a shape
        # describes a mark is not evidence about whether it is one.
        #
        # The fit decides the shape now, so its own fill is the lever: a
        # boundary wrapping mostly empty space describes the mark badly, which
        # is a statement about the boundary and not about the mark.
        base = replace(MARKS_CONFIG, enable_symmetry_rotation=False)
        loose = run_detection(
            separated_marks(), None, replace(base, parallelogram_min_fill=0.0),
        )
        strict = run_detection(
            separated_marks(), None, replace(base, parallelogram_min_fill=1.5),
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


class SharedUvDomainIndexTests(unittest.TestCase):
    """Handing the circle test the shared labelling must decide nothing new.

    ``circle_uv_coverage`` relabelled the whole atlas whenever it was given no
    index, and the circle test never had one to give: on the V60's stitch
    normal that was a 4096-square connected-components pass 11,662 times over,
    11.9 ms each. ``UvDomainIndex`` exists to stop exactly that, and computes
    the identical ``cv2.connectedComponents(mask, connectivity=8)``.

    A filter that changes nothing on one vehicle can still change another, so
    this is pinned as the equivalence it is rather than by any one car's
    output.
    """

    def _island_mask(self, rng, height, width):
        mask = np.zeros((height, width), dtype=bool)
        for _ in range(rng.randint(1, 5)):
            x0 = rng.randrange(width - 20)
            y0 = rng.randrange(height - 20)
            mask[y0:y0 + rng.randint(8, 20), x0:x0 + rng.randint(8, 20)] = True
        return mask

    def test_the_index_decides_what_relabelling_decided(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            build_uv_domain_index,
            box_uv_coverage,
            circle_uv_coverage,
            inscribed_circle_radius,
        )

        rng = random.Random(20260815)
        values = np.random.default_rng(20260815)
        circles = 0
        for trial in range(120):
            extent = rng.choice((48, 96, 160))
            mask = self._island_mask(rng, extent, extent)
            image = values.integers(0, 255, (extent, extent, 3), dtype=np.uint8)
            if trial % 2:
                # a flat field is where a circle is actually findable
                image[:] = 40
            index = build_uv_domain_index(mask)
            for _ in range(6):
                size = rng.randint(6, 30)
                group = (
                    rng.randrange(max(extent - size, 1)),
                    rng.randrange(max(extent - size, 1)),
                    size,
                    size,
                )
                relabelled = inscribed_circle_radius(
                    image, mask, group, DEFAULT_CONFIG, None
                )
                shared = inscribed_circle_radius(
                    image, mask, group, DEFAULT_CONFIG, index
                )
                self.assertEqual(relabelled, shared, f"radius for {group}")
                if relabelled is not None:
                    circles += 1
                centre = (group[0] + group[2] / 2.0, group[1] + group[3] / 2.0)
                self.assertEqual(
                    circle_uv_coverage(mask, centre, group[2] / 2.0, None),
                    circle_uv_coverage(mask, centre, group[2] / 2.0, index),
                    f"circle coverage for {group}",
                )
                self.assertEqual(
                    box_uv_coverage(mask, group, image.shape, None),
                    box_uv_coverage(mask, group, image.shape, index),
                    f"box coverage for {group}",
                )
        # the equivalence is worthless if no circle was ever found
        self.assertGreater(circles, 0)

    def test_the_crop_local_mask_does_not_shadow_the_shared_index(self) -> None:
        # inscribed_circle_radius already binds `domain` to the cropped mask,
        # so an index parameter of that name is clobbered before the coverage
        # call reads it and the whole thing raises on an ndarray.
        from mesh_segmentation_transform.annotate_texture_regions import (
            build_uv_domain_index,
            inscribed_circle_radius,
        )

        image = np.full((64, 64, 3), 128, dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=bool)
        mask[10:50, 10:50] = True
        index = build_uv_domain_index(mask)
        config = replace(DEFAULT_CONFIG, enable_circular_groups=True)

        self.assertEqual(
            inscribed_circle_radius(image, mask, (18, 18, 20, 20), config, index),
            inscribed_circle_radius(image, mask, (18, 18, 20, 20), config, None),
        )
