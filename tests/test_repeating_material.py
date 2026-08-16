"""Marks that are made of, or embedded in, a repeating texture.

Three false positives motivated this filter, all of them things a person names
instantly and no earlier stage could: the dash of a stitched seam, which has
thirty identical siblings strung along the same line; a scrap of woven carbon,
which tiles the panel it sits on; and a moulding pip on a carbon panel, which
is unique in itself and given away only by what surrounds it.

What they share is recurrence.  Autocorrelation was tried first -- it is what
``max_pattern_autocorrelation`` already asks of a finished group -- and could
not separate them: measured over the volvo stitch atlas and both scintilla
interiors it scores plain trim and a woven panel within the same band, because
what it really measures is smoothness.  Counting recurrences does separate
them, so that is what these tests pin.
"""

from __future__ import annotations

import unittest
from dataclasses import fields, replace

import cv2
import numpy as np

from mesh_segmentation_transform.annotate_texture_regions import (
    DEFAULT_CONFIG,
    PARAMETER_STEP,
    STEP_INDEX,
    build_repeat_texture_index,
    count_match_peaks,
    filter_boxes_by_repeat_texture,
    moving_average,
    normalised_match,
    repeat_texture_evidence,
    run_detection,
)


class CorrelationTests(unittest.TestCase):
    """The normalised cross-correlation the whole filter rests on."""

    def test_a_template_is_found_at_every_planted_copy(self) -> None:
        rng = np.random.default_rng(0)
        template = (rng.normal(size=(16, 16)) * 30).astype(np.float32)
        search = (rng.normal(size=(128, 128)) * 6).astype(np.float32)
        for row, column in ((10, 10), (60, 20), (30, 80)):
            search[row : row + 16, column : column + 16] = template

        response = normalised_match(search, template)

        # The valid block starts at zero shift, so a copy planted at (r, c)
        # peaks at exactly (r, c).  Getting that offset wrong misaligns the
        # correlation against the local variance it is divided by, and the
        # scores stay plausible while meaning nothing.
        self.assertEqual(response.shape, (113, 113))
        for row, column in ((10, 10), (60, 20), (30, 80)):
            self.assertAlmostEqual(float(response[row, column]), 1.0, places=5)
        self.assertEqual(count_match_peaks(response, 0.9, 8, 32), 3)

    def test_an_empty_patch_does_not_match_every_other_empty_patch(self) -> None:
        """The trap that makes a naive normalised correlation useless here."""
        rng = np.random.default_rng(1)
        search = (rng.normal(size=(128, 128)) * 0.01).astype(np.float32)
        template = (rng.normal(size=(16, 16)) * 30).astype(np.float32)
        search[10:26, 10:26] = template

        response = normalised_match(search, template)

        # Without the local-energy guard the flat surround divides by nearly
        # zero and reports a perfect match almost everywhere, which would read
        # as a legend on plain fascia recurring a hundred times.
        self.assertEqual(count_match_peaks(response, 0.7, 8, 64), 1)

    def test_peaks_claim_a_neighbourhood(self) -> None:
        response = np.zeros((64, 64), dtype=np.float32)
        response[20:24, 20:24] = 0.95  # one broad match, not sixteen
        self.assertEqual(count_match_peaks(response, 0.7, 6, 32), 1)

    def test_the_peak_walk_stops_at_its_limit(self) -> None:
        response = np.ones((64, 64), dtype=np.float32)
        self.assertEqual(count_match_peaks(response, 0.5, 1, 5), 5)

    def test_a_template_larger_than_its_search_area_concludes_nothing(self) -> None:
        self.assertEqual(
            normalised_match(np.zeros((8, 8)), np.zeros((16, 16))).size, 0
        )

    def test_the_blur_matches_a_naive_mean(self) -> None:
        rng = np.random.default_rng(2)
        array = rng.normal(size=(24, 31)).astype(np.float32)
        padded = np.pad(array, 2, mode="edge")
        expected = np.empty_like(array)
        for row in range(array.shape[0]):
            for column in range(array.shape[1]):
                expected[row, column] = padded[row : row + 5, column : column + 5].mean()

        np.testing.assert_allclose(moving_average(array, 2), expected, atol=1e-5)


class RepeatingMaterialTests(unittest.TestCase):
    """What the filter keeps and what it throws away."""

    def _weave(self, size: int = 512, period: int = 16) -> np.ndarray:
        """A regular two-axis weave, the shape woven carbon presents."""
        rows = np.arange(size)[:, None]
        columns = np.arange(size)[None, :]
        cells = ((rows // period) + (columns // period)) % 2
        grey = np.where(cells == 1, 90, 170).astype(np.uint8)
        return np.repeat(grey[:, :, None], 3, axis=2)

    def _plain(self, size: int = 512, level: int = 130) -> np.ndarray:
        """Flat trim with a little grain.

        The grain is not decoration.  A perfectly uniform surround gives the
        local-energy guard nothing to weigh a match against, which is not a
        situation any real atlas presents.
        """
        rng = np.random.default_rng(3)
        image = np.full((size, size, 3), level, dtype=np.int16)
        image += rng.integers(-6, 7, image.shape)
        return np.clip(image, 0, 255).astype(np.uint8)

    def _mark(self, image: np.ndarray, x: int, y: int, w: int, h: int) -> None:
        image[y : y + h, x : x + w] = 20

    def _config(self, **overrides):
        return replace(
            DEFAULT_CONFIG,
            enable_repeat_texture_filter=True,
            repeat_texture_decimation=1,
            **overrides,
        )

    def test_a_scrap_of_weave_is_rejected_by_its_own_recurrence(self) -> None:
        image = self._weave()
        config = self._config()

        own, _directions = repeat_texture_evidence(
            build_repeat_texture_index(image, config),
            (240, 240, 32, 32),
            config,
        )

        self.assertGreaterEqual(own, config.repeat_texture_min_repeats)

    def test_a_lone_mark_on_plain_trim_survives(self) -> None:
        image = self._plain()
        self._mark(image, 240, 240, 32, 40)
        config = self._config()

        own, directions = repeat_texture_evidence(
            build_repeat_texture_index(image, config),
            (240, 240, 32, 40),
            config,
        )

        self.assertLess(own, config.repeat_texture_min_repeats)
        self.assertLess(directions, config.repeat_texture_context_directions)

    def test_a_unique_mark_in_weave_is_rejected_by_its_surroundings(self) -> None:
        """The pip on a carbon panel: only the context gives it away."""
        image = self._weave()
        self._mark(image, 240, 232, 32, 48)
        config = self._config()

        own, directions = repeat_texture_evidence(
            build_repeat_texture_index(image, config),
            (240, 232, 32, 48),
            config,
        )

        self.assertLess(own, config.repeat_texture_min_repeats)
        self.assertGreaterEqual(directions, config.repeat_texture_context_directions)

    def test_a_word_is_not_a_repeating_texture(self) -> None:
        """Four different letters in a row must not read as four recurrences."""
        image = self._plain()
        mask = np.zeros((512, 512), dtype=np.uint8)
        cv2.putText(mask, "PRND", (150, 280), cv2.FONT_HERSHEY_SIMPLEX, 3.0, 255, 8)
        image[mask > 0] = 20
        boxes = np.asarray(
            [[150 + index * 62, 235, 52, 52] for index in range(4)], dtype=np.int32
        )
        config = self._config()

        kept, rejected, _index = filter_boxes_by_repeat_texture(
            image, boxes, config, build_repeat_texture_index(image, config)
        )

        self.assertEqual(len(kept), 4)
        self.assertEqual(rejected, [])

    def test_a_row_of_identical_marks_is_rejected(self) -> None:
        """A stitched seam: the same dash, over and over, along one line."""
        image = self._plain()
        dashes = [(40 + step * 26, 250) for step in range(16)]
        for x, y in dashes:
            self._mark(image, x, y, 16, 6)
        boxes = np.asarray([[x, y, 16, 6] for x, y in dashes], dtype=np.int32)
        config = self._config()

        kept, rejected, _index = filter_boxes_by_repeat_texture(
            image, boxes, config, build_repeat_texture_index(image, config)
        )

        self.assertEqual(len(kept), 0)
        self.assertEqual(len(rejected), len(dashes))

    def test_a_probe_off_the_edge_of_the_atlas_concludes_nothing(self) -> None:
        """A negative slice bound wraps in Python rather than clipping."""
        image = self._weave()
        config = self._config()

        own, directions = repeat_texture_evidence(
            build_repeat_texture_index(image, config),
            (0, 0, 8, 8),
            config,
        )

        self.assertEqual(own, 0)
        self.assertGreaterEqual(directions, 0)

    def test_one_patterned_neighbour_is_not_enough(self) -> None:
        """A legend beside a woven strip keeps its place.

        Requiring more than one direction is the whole point of the context
        test: a mark embedded in material has patterned surroundings all round,
        a mark that merely sits next to some has them on one side.
        """
        image = self._plain()
        image[:, 340:] = self._weave()[:, 340:]
        self._mark(image, 250, 240, 32, 40)
        config = self._config()

        _own, directions = repeat_texture_evidence(
            build_repeat_texture_index(image, config),
            (250, 240, 32, 40),
            config,
        )

        self.assertLess(directions, config.repeat_texture_context_directions)


class WiringTests(unittest.TestCase):
    """Where the stage sits, and that it stays out of the way until asked."""

    def test_the_stage_is_a_no_op_when_switched_off(self) -> None:
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[20:40, 20:40] = 200

        run = run_detection(
            image, None, replace(DEFAULT_CONFIG, enable_repeat_texture_filter=False)
        )

        stage = next(s for s in run.stages if s.key == "repeat_texture")
        self.assertEqual(stage.detail, "disabled")
        self.assertEqual(stage.rejected, ())

    def test_it_is_on_by_default(self) -> None:
        """Replayed over every region the shipped plans flipped, the two it
        takes are a run of carpet stitching and a dead-flat patch of trim."""
        self.assertTrue(DEFAULT_CONFIG.enable_repeat_texture_filter)

    def test_it_runs_before_anything_groups(self) -> None:
        """A seam that reaches grouping merges into chains that read as text."""
        self.assertLess(STEP_INDEX["box_filter"], STEP_INDEX["repeat_texture"])
        self.assertLess(STEP_INDEX["repeat_texture"], STEP_INDEX["overlap_box_group"])
        self.assertLess(STEP_INDEX["repeat_texture"], STEP_INDEX["grouped"])

    def test_every_parameter_resumes_from_its_own_step(self) -> None:
        named = [
            entry.name
            for entry in fields(DEFAULT_CONFIG)
            if entry.name.startswith("repeat_texture")
            or entry.name == "enable_repeat_texture_filter"
        ]

        self.assertTrue(named)
        for name in named:
            self.assertEqual(
                PARAMETER_STEP.get(name), STEP_INDEX["repeat_texture"], name
            )

    def test_the_index_is_built_once_and_caches_its_probes(self) -> None:
        """Rebuilding the high-passed atlas per box would dominate the stage."""
        image = np.repeat(
            np.tile(np.array([90, 170], dtype=np.uint8), (512, 256))[:, :, None],
            3,
            axis=2,
        )
        config = replace(
            DEFAULT_CONFIG,
            enable_repeat_texture_filter=True,
            repeat_texture_decimation=1,
        )
        index = build_repeat_texture_index(image, config)
        boxes = np.asarray([[240, 240, 32, 32], [260, 260, 32, 32]], dtype=np.int32)

        _kept, _rejected, returned = filter_boxes_by_repeat_texture(
            image, boxes, config, index
        )

        self.assertIs(returned, index)
        self.assertTrue(index.matches)

    def test_decimation_keeps_every_length_in_atlas_pixels(self) -> None:
        rows = np.arange(512)[:, None]
        columns = np.arange(512)[None, :]
        cells = ((rows // 16) + (columns // 16)) % 2
        image = np.repeat(
            np.where(cells == 1, 90, 170).astype(np.uint8)[:, :, None], 3, axis=2
        )

        halved = build_repeat_texture_index(
            image,
            replace(
                DEFAULT_CONFIG,
                enable_repeat_texture_filter=True,
                repeat_texture_decimation=2,
            ),
        )
        self.assertEqual(halved.detail.shape, (256, 256))

        for decimation in (1, 2):
            config = replace(
                DEFAULT_CONFIG,
                enable_repeat_texture_filter=True,
                repeat_texture_decimation=decimation,
            )
            own, _directions = repeat_texture_evidence(
                build_repeat_texture_index(image, config),
                (240, 240, 32, 32),
                config,
            )
            self.assertGreaterEqual(own, config.repeat_texture_min_repeats, decimation)


class AtlasScopeTests(unittest.TestCase):
    """Detection runs on crops; recurrence has to be counted on the atlas.

    This is the bug that made the filter look inert on real parts while passing
    on whole atlases.  The harness detects one UV island at a time and
    production one view at a time, and both hand the pipeline a crop tight to
    that island's bounding box.  A stitched seam sliced that way shows one or
    two dashes per crop and reads as unique in every one of them: measured on
    the volvo stitch atlas, a seam scoring 13 recurrences over the atlas scores
    3 inside a 128 px crop and 1 inside a 96 px one.
    """

    def _striped_atlas(self, size: int = 512, pitch: int = 26) -> np.ndarray:
        """Plain trim carrying one long run of identical dashes."""
        rng = np.random.default_rng(4)
        image = np.full((size, size, 3), 130, dtype=np.int16)
        image += rng.integers(-6, 7, image.shape)
        image = np.clip(image, 0, 255).astype(np.uint8)
        for step in range(18):
            x = 20 + step * pitch
            image[250:256, x : x + 16] = 20
        return image

    def _config(self):
        return replace(
            DEFAULT_CONFIG,
            enable_repeat_texture_filter=True,
            repeat_texture_decimation=1,
        )

    def test_a_crop_too_small_to_show_the_repeat_is_judged_on_the_atlas(self) -> None:
        atlas = self._striped_atlas()
        config = self._config()
        index = build_repeat_texture_index(atlas, config)
        # The crop a single narrow UV island would produce: one dash and a
        # sliver of its neighbours.
        crop_origin = (150, 220)
        crop = atlas[crop_origin[1] : crop_origin[1] + 80,
                     crop_origin[0] : crop_origin[0] + 80]
        box_in_crop = (20, 30, 16, 6)

        without_atlas = repeat_texture_evidence(
            build_repeat_texture_index(crop, config), box_in_crop, config
        )
        with_atlas = repeat_texture_evidence(
            index.for_view(((( 0, 0, 80, 80), *crop_origin),)), box_in_crop, config
        )

        self.assertLess(without_atlas[0], config.repeat_texture_min_repeats)
        self.assertGreaterEqual(with_atlas[0], config.repeat_texture_min_repeats)

    def test_a_view_shares_the_atlas_field_and_match_cache(self) -> None:
        """Islands cut from the same trim must not each pay for the answer."""
        atlas = self._striped_atlas()
        config = self._config()
        index = build_repeat_texture_index(atlas, config)

        view = index.for_view((((0, 0, 80, 80), 150, 220),))

        self.assertIs(view.detail, index.detail)
        self.assertIs(view.matches, index.matches)
        repeat_texture_evidence(view, (20, 30, 16, 6), config)
        self.assertTrue(index.matches)

    def test_the_same_atlas_point_is_reached_from_any_placement(self) -> None:
        atlas = self._striped_atlas()
        config = self._config()
        index = build_repeat_texture_index(atlas, config)

        direct = repeat_texture_evidence(index, (170, 250, 16, 6), config)
        shifted = repeat_texture_evidence(
            index.for_view((((0, 0, 80, 80), 150, 220),)), (20, 30, 16, 6), config
        )

        self.assertEqual(direct, shifted)

    def test_a_resumed_run_takes_the_callers_index_not_the_stored_one(self) -> None:
        """Tuning decimation or the high-pass rebuilds the field it measures.

        A resumed run restores the state the previous one entered the stage
        with, and that state carries the previous run's index.  Without the
        override the harness would keep measuring the old field while showing
        the new number in the panel.
        """
        atlas = self._striped_atlas()
        config = self._config()
        boxes = np.asarray([[170, 250, 16, 6]], dtype=np.int32)
        first = run_detection(
            atlas, None, config, initial_boxes=boxes,
            initial_repeat_texture=build_repeat_texture_index(atlas, config),
        )
        self.assertEqual(
            first.entry_states[STEP_INDEX["repeat_texture"]].repeat_texture.decimation,
            1,
        )

        coarser = replace(config, repeat_texture_decimation=4)
        second = run_detection(
            atlas, None, coarser, previous=first, initial_boxes=boxes,
            initial_repeat_texture=build_repeat_texture_index(atlas, coarser),
        )

        self.assertEqual(
            second.entry_states[STEP_INDEX["repeat_texture"]].repeat_texture.decimation,
            4,
        )

    def test_a_box_outside_every_placed_rectangle_concludes_nothing(self) -> None:
        """A collage leaves gaps between its tiles; nothing lives there."""
        atlas = self._striped_atlas()
        config = self._config()
        view = build_repeat_texture_index(atlas, config).for_view(
            (((0, 0, 40, 40), 150, 220),)
        )

        self.assertEqual(repeat_texture_evidence(view, (200, 200, 16, 6), config), (0, 0))


if __name__ == "__main__":
    unittest.main()
