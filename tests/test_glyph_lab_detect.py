"""The pipeline's own stages, and the density gate that thins the seed.

Written against arrays rather than a corpus: every claim here is about one
kernel and can be stated exactly, which a real texture cannot do.
"""

from __future__ import annotations

import time
import unittest
from dataclasses import fields, replace
from unittest.mock import patch

import numpy as np
from scipy import ndimage

from glyph_lab.corpus import CorpusTexture
from glyph_lab.detect import (
    AT_THRESHOLD,
    BRIDGED,
    CONVEX_HULLED,
    DENSE_SEED,
    FLAT_GROUPED,
    GROUP_SIZED,
    LAST_STOP,
    MARKED,
    NEARBY_CONTRAST,
    OFF_BOUNDARY,
    PADDED,
    PATTERNED,
    PROXIMITY_GROUPED,
    SEEDED,
    SIZED,
    SPARSE,
    STAGE_FIRST,
    STAGE_OF,
    STAGES,
    TRANSFORMED,
    UNCROWDED,
    UNREPEATED,
    UNRIMMED,
    UNTRANSITIONED,
    ContrastConfig,
    LabConfig,
    RimCandidate,
    boundary_margin,
    build_repeat_texture_index,
    classify_rim_candidates,
    clear_of_a_crowd,
    clear_of_a_repeat,
    clear_of_boundary,
    clear_of_the_rim,
    confirmed_repeats,
    contrast_response,
    detect,
    gap_bridges,
    island_size,
    marks_from_seed,
    paint_holes,
    past_the_size_floor,
    pattern_candidates,
    recover_nearby_contrast,
    regions_from_holes,
    regions_from_marks,
    repeat_scores,
    repeat_texture_evidence,
    rim_contact,
    rim_perimeter_contact,
    sampling_mask,
    seed_density,
    sparse_field,
    sparse_field_with_mesh_cuts,
    structure_map,
    structure_seed,
)
from glyph_lab.regions import NEIGHBOURHOOD, Region


class SeedDensityTests(unittest.TestCase):
    def test_a_lone_seed_scores_one_window(self) -> None:
        """The centre counts itself, so an isolated seed is 1/k^2, not 0."""
        domain = np.ones((21, 21), dtype=bool)
        seed = np.zeros((21, 21), dtype=bool)
        seed[10, 10] = True
        self.assertAlmostEqual(
            float(seed_density(seed, domain, 9)[10, 10]), 1.0 / 81.0, places=6
        )

    def test_a_solid_patch_scores_one(self) -> None:
        domain = np.ones((21, 21), dtype=bool)
        seed = np.ones((21, 21), dtype=bool)
        self.assertAlmostEqual(
            float(seed_density(seed, domain, 9)[10, 10]), 1.0, places=6
        )

    def test_a_line_beats_a_speck_at_the_same_window(self) -> None:
        """The whole point: an edge seeds along its length and a fluke does not."""
        domain = np.ones((31, 31), dtype=bool)
        seed = np.zeros((31, 31), dtype=bool)
        seed[5, :] = True          # a drawn edge
        seed[20, 15] = True        # a chance pass
        density = seed_density(seed, domain, 9)
        self.assertGreater(float(density[5, 15]), float(density[20, 15]))

    def test_the_domain_is_what_the_share_is_taken_over(self) -> None:
        """A seed near a seam must not read as lonely because of the void.

        An island filled solid with seed reads 1.0 everywhere inside it, right
        up to its own edge.  Counting the emptiness outside instead would drag
        the rim down and take out exactly the texels a mark on a small chart is
        made of.
        """
        domain = np.zeros((31, 31), dtype=bool)
        domain[10:16, 10:16] = True
        seed = domain.copy()
        density = seed_density(seed, domain, 9)
        self.assertAlmostEqual(float(density[10, 10]), 1.0, places=5)
        self.assertAlmostEqual(float(density[12, 12]), 1.0, places=5)

    def test_an_even_window_is_widened_to_stay_centred(self) -> None:
        domain = np.ones((21, 21), dtype=bool)
        seed = np.zeros((21, 21), dtype=bool)
        seed[10, 10] = True
        np.testing.assert_allclose(
            seed_density(seed, domain, 10), seed_density(seed, domain, 11)
        )

    def test_seed_outside_the_domain_is_not_counted(self) -> None:
        """Nothing fires out there, so nothing may be credited to it either."""
        domain = np.zeros((21, 21), dtype=bool)
        domain[:, :10] = True
        seed = np.ones((21, 21), dtype=bool)
        inside = seed_density(seed & domain, domain, 9)
        both = seed_density(seed, domain, 9)
        np.testing.assert_allclose(inside, both)


def one_texture(colour: np.ndarray) -> CorpusTexture:
    domains = np.ones(colour.shape[:2], dtype=np.uint16)
    return CorpusTexture(
        name="probe", colour=colour, domains=domains, normal=None, source={}
    )


def whole_pipeline() -> LabConfig:
    """Every stage, including the front end the sparse field replaces.

    2026-08-28: the default stop is the size floor now, so a test about any
    stage after it has to clear the stop or the stage never runs.
    """
    base = LabConfig()
    return replace(base, contrast=replace(base.contrast, stop_after_stage=0))


class PerStageRegionsTests(unittest.TestCase):
    """Each stage records the regions *it* had, not the ones left at the end.

    2026-08-29: without this the viewer drew the finished outlines over every
    stage's texels, so a mark thrown away in the middle still had a box round
    it at every earlier stage and there was no way to see which stage dropped
    it.  Tracing one down meant re-running the whole pipeline once per stage
    and diffing the bounds by hand.
    """

    @staticmethod
    def two_marks() -> CorpusTexture:
        """A big mark and a small one, on ground blank enough to be blank."""
        colour = np.zeros((160, 220, 3), dtype=np.uint8)
        colour[40:70, 30:60] = 240
        colour[110:118, 150:158] = 240
        return one_texture(colour)

    @staticmethod
    def floor_of(width: int) -> LabConfig:
        base = LabConfig()
        return replace(base, contrast=replace(
            base.contrast, min_width_px=width, min_height_px=width,
        ))

    def test_a_dropped_region_leaves_the_later_stages_only(self) -> None:
        # 24 sits between the two holes these marks make: 38x38 and 16x16.
        found = detect(self.two_marks(), self.floor_of(24), capture=True)
        self.assertEqual(found.stage_counts[SPARSE], 2)
        self.assertEqual(found.stage_counts[UNCROWDED], 2)
        self.assertEqual(found.stage_counts[SIZED], 1, "the small one survived")
        self.assertEqual(len(found.regions), 1)
        for name in (UNREPEATED, UNRIMMED):
            self.assertEqual(found.stage_counts[name], 1)

    def test_the_earlier_stage_draws_the_region_the_later_one_lost(self) -> None:
        found = detect(self.two_marks(), self.floor_of(24), capture=True)
        early, late = found.stage_outline(SPARSE), found.stage_outline(SIZED)
        self.assertEqual(int((late & ~early).sum()), 0, "a path appeared later")
        self.assertGreater(
            int((early & ~late).sum()), 0, "the dropped region left no trace"
        )

    def test_the_texel_stage_reports_no_regions_rather_than_the_final_count(
        self,
    ) -> None:
        """Zero, not "unrecorded".  The threshold has no regions to speak of,
        and falling back to the finished tally there is the exact confusion
        the stage view exists to remove.
        """
        found = detect(self.two_marks(), self.floor_of(24), capture=True)
        self.assertEqual(found.stage_counts[AT_THRESHOLD], 0)
        self.assertIsNone(found.stage_outline(AT_THRESHOLD))

    def test_the_counts_only_ever_fall(self) -> None:
        found = detect(self.two_marks(), self.floor_of(24), capture=True)
        stages = [SPARSE, UNCROWDED, SIZED, UNREPEATED, UNRIMMED]
        counts = [found.stage_counts[name] for name in stages]
        self.assertEqual(counts, sorted(counts, reverse=True))
        self.assertEqual(counts[-1], len(found.regions))

    def test_nothing_is_recorded_when_nobody_is_watching(self) -> None:
        found = detect(self.two_marks(), self.floor_of(24))
        self.assertEqual(found.stage_counts, {})
        self.assertEqual(found.stage_outlines, {})
        self.assertIsNone(found.stage_outline(SIZED))


class SeedDensityInThePipelineTests(unittest.TestCase):
    """The gate as the detector applies it, not as a kernel."""

    @staticmethod
    def speckled() -> np.ndarray:
        """A drawn bar, and a scatter of single-texel flecks well away from it."""
        colour = np.full((96, 96, 3), 40, dtype=np.uint8)
        colour[20:26, 10:86] = 230                       # the mark
        flecks = np.random.default_rng(7).integers(0, 40, size=(60, 2))
        for y, x in flecks:
            colour[50 + y, 30 + x] = 210                 # one texel each
        return colour

    def test_the_gate_thins_the_seed_and_leaves_it_alone_when_off(self) -> None:
        texture = one_texture(self.speckled())
        off = whole_pipeline()
        on = replace(
            off, contrast=replace(off.contrast, min_seed_density=0.2)
        )
        loose = detect(texture, off, capture=True)
        tight = detect(texture, on, capture=True)
        seeded = loose.stage_mask(SEEDED)
        dense = tight.stage_mask(DENSE_SEED)
        self.assertIsNotNone(dense)
        self.assertLess(int(dense.sum()), int(seeded.sum()))
        # And it only ever removes: the gate is an AND against the seed.
        self.assertEqual(int((dense & ~seeded).sum()), 0)

    def test_off_leaves_the_two_seed_stages_identical(self) -> None:
        """0 disables it, which has to mean bit-for-bit and not nearly."""
        found = detect(one_texture(self.speckled()), whole_pipeline(), capture=True)
        np.testing.assert_array_equal(
            found.stage_mask(SEEDED), found.stage_mask(DENSE_SEED)
        )

    def test_a_scatter_of_flecks_goes_before_a_drawn_bar_does(self) -> None:
        """What the gate is for, put as an ordering rather than a threshold.

        Raise the bar until the mark starts to suffer and the flecks must
        already be gone: they are single texels and it is a solid edge.
        """
        texture = one_texture(self.speckled())
        base = whole_pipeline()
        base = replace(
            base, contrast=replace(base.contrast, nearby_contrast_radius_px=0),
        )
        bar = np.zeros((96, 96), dtype=bool)
        bar[18:28, 8:88] = True                          # the mark and its rim
        for level in (0.05, 0.1, 0.15, 0.2, 0.3):
            config = replace(
                base, contrast=replace(base.contrast, min_seed_density=level)
            )
            seed = detect(texture, config, capture=True).stage_mask(DENSE_SEED)
            if int((seed & ~bar).sum()) == 0:
                break
            self.assertGreater(
                int((seed & bar).sum()), 0,
                f"the bar went before the flecks did, at {level}",
            )
        else:
            self.fail("no level cleared the flecks")


class StructureSeedTests(unittest.TestCase):
    """Stage 3 asks the question of the piece, not of every texel in it."""

    @staticmethod
    def crossing() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One clean step, a lit run over it, and a lit run away from it.

        The run at row 20 crosses the step at column 40, so a handful of its
        texels read as structure and the rest sit over flat ground.  The run at
        row 30 never leaves the flat half.
        """
        colour = np.full((40, 80, 3), 40, dtype=np.uint8)
        colour[:, 40:] = 230
        lit = np.zeros((40, 80), dtype=bool)
        lit[20, 10:70] = True
        lit[30, 5:20] = True
        return colour, lit, np.ones((40, 80), dtype=bool)

    def test_a_run_that_reads_as_a_step_anywhere_is_kept_whole(self) -> None:
        """The failure this replaced: 6 of 75 texels, and the run in pieces."""
        colour, lit, domain = self.crossing()
        config = ContrastConfig()
        gated = lit & (
            structure_map(
                colour, domain, config.structure_px, config.structure_inner_px
            ) >= config.min_structure
        )
        self.assertEqual(int(gated.sum()), 6)
        kept = structure_seed(colour, lit, domain, config)
        self.assertTrue(bool(kept[20, 10:70].all()))
        self.assertEqual(int(kept.sum()), 60)

    def test_a_run_with_no_step_under_it_is_declined(self) -> None:
        """Growing is not passing: a piece still has to hold one step."""
        colour, lit, domain = self.crossing()
        kept = structure_seed(colour, lit, domain, ContrastConfig())
        self.assertFalse(bool(kept[30].any()))

    def test_the_grouping_does_not_reach_across_a_gap(self) -> None:
        """One empty texel is enough to keep a neighbour out.

        Measured on the corpus, this is the whole safety of the stage: growing
        across a single texel of gap takes ardente's confetti panel from 518
        texels to 4,927, and across two the field is one piece.
        """
        colour, lit, domain = self.crossing()
        lit[22, 10:36] = True          # row 21 left clear between them
        kept = structure_seed(colour, lit, domain, ContrastConfig())
        self.assertTrue(bool(kept[20, 10:70].all()))
        self.assertFalse(bool(kept[22, 10:36].any()))

    def test_a_diagonal_touch_is_one_piece(self) -> None:
        """Eight-connected, like every other grouping in the module."""
        colour, lit, domain = self.crossing()
        lit[21, 9] = True              # touches (20, 10) corner to corner
        kept = structure_seed(colour, lit, domain, ContrastConfig())
        self.assertTrue(bool(kept[21, 9]))

    def test_nothing_lit_is_nothing_kept(self) -> None:
        colour, _, domain = self.crossing()
        lit = np.zeros((40, 80), dtype=bool)
        self.assertFalse(bool(structure_seed(colour, lit, domain,
                                             ContrastConfig()).any()))

    def test_the_stage_is_a_no_op_at_zero(self) -> None:
        colour = self.crossing()[0]
        texture = one_texture(colour)
        config = LabConfig(
            contrast=replace(ContrastConfig(), min_structure=0.0,
                             stop_after_stage=STAGE_OF["min_structure"] + 1)
        )
        got = detect(texture, config, capture=True)
        np.testing.assert_array_equal(
            got.stage_mask(SEEDED), got.stage_mask(OFF_BOUNDARY)
        )


class StageOrderTests(unittest.TestCase):
    def test_every_stage_is_numbered_in_order(self) -> None:
        """The names carry their own position, so they must agree with it."""
        for index, name in enumerate(STAGES, start=1):
            self.assertEqual(
                name.split(" ", 1)[0], str(index), f"{name} is misnumbered"
            )

    def test_each_overlay_starts_inside_the_pipeline(self) -> None:
        for key, first in STAGE_FIRST.items():
            self.assertTrue(
                0 <= first < len(STAGES), f"{key} points outside STAGES"
            )

    def test_the_seed_stages_stay_inside_the_threshold(self) -> None:
        """Seeding, thinning and growing all happen within stage 1.

        Not every stage narrows -- hysteresis deliberately reaches *below* the
        threshold, which is the whole of what it is for -- so the claim is made
        only over the stretch where it holds: the seed is drawn from the
        threshold mask, the density gate ANDs against the seed, and the growth
        follows weak response bounded by the threshold mask it was handed.
        """
        colour = np.full((96, 96, 3), 40, dtype=np.uint8)
        colour[20:26, 10:86] = 230
        colour[40:70, 40:70] = np.random.default_rng(3).integers(
            30, 90, size=(30, 30, 1)
        )
        base = whole_pipeline()
        config = replace(
            base,
            contrast=replace(
                base.contrast, min_seed_density=0.1, stop_after_stage=0
            ),
        )
        found = detect(one_texture(colour), config, capture=True)
        # 2026-09-04: the threshold itself.  It was the recovery, which was
        # the widest mask only because it grew `lit`; the recovery now grows
        # marks instead, so stage 1 is what the seed stages are handed.
        first = found.stage_mask(AT_THRESHOLD)
        for name in (OFF_BOUNDARY, SEEDED, DENSE_SEED, PATTERNED):
            mask = found.stage_mask(name)
            self.assertIsNotNone(mask, name)
            self.assertEqual(
                int((mask & ~first).sum()), 0,
                f"{name} holds texels outside the threshold",
            )
        # The gate can only remove from the seed it is given.
        np.testing.assert_array_equal(
            found.stage_mask(DENSE_SEED) & found.stage_mask(SEEDED),
            found.stage_mask(DENSE_SEED),
        )

class SamplingMaskTests(unittest.TestCase):
    """Which texels count as a measurement of the surface."""

    def test_skip_zero_is_the_domain_itself(self) -> None:
        domain = np.zeros((16, 16), dtype=bool)
        domain[4:12, 4:12] = True
        self.assertIs(sampling_mask(domain, 0), domain)

    def test_one_step_takes_the_rim_including_the_diagonals(self) -> None:
        domain = np.zeros((16, 16), dtype=bool)
        domain[4:12, 4:12] = True
        inner = sampling_mask(domain, 1)
        self.assertFalse(bool(inner[4, 4]), "a corner survived")
        self.assertFalse(bool(inner[4, 7]), "an edge texel survived")
        self.assertTrue(bool(inner[5, 5]))
        self.assertEqual(int(inner.sum()), 6 * 6)

    def test_the_edge_of_the_crop_counts_as_outside(self) -> None:
        """A domain arrives cropped to its bounds, so its extremes are rim."""
        domain = np.ones((8, 8), dtype=bool)
        inner = sampling_mask(domain, 1)
        self.assertEqual(int(inner.sum()), 6 * 6)
        self.assertFalse(bool(inner[0, 0]))

    def test_a_two_wide_ribbon_keeps_nothing(self) -> None:
        """It is partial coverage all the way through; there is no surface."""
        domain = np.zeros((16, 16), dtype=bool)
        domain[7:9, 2:14] = True
        self.assertEqual(int(sampling_mask(domain, 1).sum()), 0)


class BoundaryContrastTests(unittest.TestCase):
    """The response with the island rim taken out of the sample."""

    @staticmethod
    def bled_island() -> tuple[np.ndarray, np.ndarray]:
        """A flat panel whose rim texels are a partial-coverage blend.

        This is what the UV rasteriser leaves: the outermost texel of a chart
        is written wherever a triangle covers any part of it, so its colour is
        the surface mixed with whatever lies outside.  The panel itself is one
        flat colour and holds nothing anybody would want to detect.
        """
        domain = np.zeros((40, 40), dtype=bool)
        domain[8:32, 8:32] = True
        rim = domain & ~sampling_mask(domain, 1)
        colour = np.zeros((40, 40, 3), dtype=np.uint8)
        colour[domain] = 200
        colour[rim] = 110          # half the surface, half the black outside
        return colour, domain

    def test_the_rim_is_not_answered_for(self) -> None:
        colour, domain = self.bled_island()
        config = ContrastConfig()
        response = contrast_response(colour, domain, config)
        rim = domain & ~sampling_mask(domain, config.boundary_skip_px)
        self.assertEqual(float(response[rim].max()), 0.0)
        self.assertEqual(float(response[~domain].max()), 0.0)

    def test_a_flat_panel_stops_ringing_at_its_own_edge(self) -> None:
        """The bug, stated as a measurement.

        With the rim in the sample, every texel whose kernel reaches it is
        compared against a mean dragged toward the padding, so a panel with
        nothing on it draws a bright ring one texel inside its own border.
        """
        colour, domain = self.bled_island()
        config = ContrastConfig()
        keeping = contrast_response(
            colour, domain, replace(config, boundary_skip_px=0)
        )
        dropping = contrast_response(colour, domain, config)
        self.assertGreater(
            float(keeping.max()), 20.0, "the fixture does not reproduce it"
        )
        # Not exactly zero: the box filters run in float32, so a flat panel
        # comes back as a few parts in 10^5 rather than as nothing.
        self.assertLess(
            float(dropping.max()), 1e-3,
            "the flat panel still responds somewhere",
        )

    def test_nothing_lights_up_outside_the_sample(self) -> None:
        """Not merely zero response -- never a candidate, at any later stage."""
        colour, domain = self.bled_island()
        colour[18:22, 12:28] = 40                    # something real to find
        texture = CorpusTexture(
            name="probe", colour=colour, domains=domain.astype(np.uint16),
            normal=None, source={},
        )
        found = detect(texture, LabConfig(), capture=True)
        rim = domain & ~sampling_mask(
            domain, ContrastConfig().boundary_skip_px
        )
        # Up to the grouping.  The marks stage dilates by `mark_gap_px` and
        # is *meant* to reach the rim -- it joins marks, and a mark near an
        # island edge is still one mark.
        for name in STAGES[:STAGES.index(MARKED)]:
            mask = found.stage_mask(name)
            if mask is None:
                continue
            self.assertEqual(
                int((mask & rim).sum()), 0, f"{name} lit a boundary texel"
            )

    def test_zero_goes_back_to_sampling_the_whole_domain(self) -> None:
        colour, domain = self.bled_island()
        response = contrast_response(
            colour, domain, replace(ContrastConfig(), boundary_skip_px=0)
        )
        self.assertGreater(float(response[domain].max()), 0.0)


class NearbyContrastRecoveryTests(unittest.TestCase):
    def test_only_the_neighbourhood_of_the_first_pass_is_recovered(self) -> None:
        response = np.zeros((15, 15), dtype=np.float32)
        first_pass = np.zeros((15, 15), dtype=bool)
        first_pass[7, 7] = True
        response[7, 10] = 30.0
        response[1, 1] = 90.0

        recovered = recover_nearby_contrast(
            response, first_pass, np.ones_like(first_pass), 3, 24.0,
        )

        self.assertTrue(recovered[7, 7])
        self.assertTrue(recovered[7, 10])
        self.assertFalse(recovered[1, 1])

    def test_a_non_positive_setting_disables_recovery(self) -> None:
        response = np.full((7, 7), 99.0, dtype=np.float32)
        first_pass = np.zeros((7, 7), dtype=bool)
        first_pass[3, 3] = True

        for radius, threshold in ((0, 24.0), (3, 0.0)):
            np.testing.assert_array_equal(
                recover_nearby_contrast(
                    response, first_pass, np.ones_like(first_pass), radius, threshold,
                ),
                first_pass,
            )

    def test_the_recovery_grows_a_mark_and_reaches_nothing_else(self) -> None:
        """2026-09-04: it grows marks now, and only along their own contrast.

        The old stage dilated the whole threshold and admitted any texel over
        the bar, so two marks with contrast between them became one -- which
        is what buried seven glyphs on etkc's alcantara interior.  A mark can
        still recover its own faint edge; a speck across the panel is not part
        of it however high it reads.
        """
        response = np.zeros((15, 15), dtype=np.float32)
        response[7, 7] = 100.0
        response[7, 8:11] = 25.0          # the mark's own faint edge
        response[1, 1] = 25.0             # a speck, reachable by neither
        colour = np.zeros((15, 15, 3), dtype=np.uint8)
        base = LabConfig()
        config = replace(
            base,
            contrast=replace(
                base.contrast, local_floor_multiple=0.0, min_response=50.0,
                nearby_contrast_radius_px=3, nearby_contrast_min_response=24.0,
                stop_after_stage=STAGES.index(NEARBY_CONTRAST) + 1,
            ),
        )
        with patch("glyph_lab.detect.contrast_response", return_value=response):
            found = detect(one_texture(colour), config, capture=True)

        self.assertTrue(found.stage_mask(AT_THRESHOLD)[7, 7])
        self.assertFalse(found.stage_mask(AT_THRESHOLD)[7, 10])
        recovered = found.stage_mask(NEARBY_CONTRAST)
        self.assertIsNotNone(recovered)
        self.assertFalse(bool(recovered[1, 1]), "recovered a disconnected speck")


class BoundaryReachTests(unittest.TestCase):
    """Touching the island edge is not held against a mark; only hugging it is."""

    @staticmethod
    def island() -> np.ndarray:
        domain = np.zeros((60, 60), dtype=bool)
        domain[10:50, 10:50] = True
        return domain

    def test_a_ribbon_along_the_edge_goes_entirely(self) -> None:
        """The case this exists for: a response tracing a chart's outline."""
        sample = self.island()
        lit = sample & ~ndimage.binary_erosion(
            sample, structure=np.ones((3, 3), bool), iterations=2
        )
        self.assertGreater(int(lit.sum()), 0)
        self.assertEqual(int(clear_of_boundary(lit, sample, 4).sum()), 0)

    def test_a_mark_that_merely_meets_the_edge_is_kept_whole(self) -> None:
        """Not trimmed at the border -- kept, every texel of it.

        This is the half the old margin band could not do: it took the rim off
        anything that reached one, so a legend running to a chart edge lost
        its outermost rows whether or not it was a legend.
        """
        sample = self.island()
        lit = np.zeros(sample.shape, dtype=bool)
        lit[10:40, 20:26] = True          # starts on the rim, runs 30 in
        kept = clear_of_boundary(lit, sample, 4)
        np.testing.assert_array_equal(kept, lit)

    def test_a_mark_that_never_touches_is_not_examined(self) -> None:
        sample = self.island()
        lit = np.zeros(sample.shape, dtype=bool)
        lit[30:32, 30:40] = True          # deep inside, and shallow
        np.testing.assert_array_equal(clear_of_boundary(lit, sample, 12), lit)

    def test_the_reach_is_where_the_line_falls(self) -> None:
        """Depth is measured from the edge, so the bar is a distance."""
        sample = self.island()
        lit = np.zeros(sample.shape, dtype=bool)
        lit[10:16, 20:26] = True          # touches, reaches 6 texels in
        self.assertGreater(int(clear_of_boundary(lit, sample, 6).sum()), 0)
        self.assertEqual(int(clear_of_boundary(lit, sample, 7).sum()), 0)

    def test_it_judges_a_component_whole(self) -> None:
        """A ribbon joined to a deep mark is kept: one component, one verdict.

        Stated because it is the honest limit of the test rather than a
        surprise -- what a run of texels *is* is a property of the run, and a
        ribbon welded to a glyph by the threshold is not separable here.
        """
        sample = self.island()
        lit = np.zeros(sample.shape, dtype=bool)
        lit[10:12, 12:48] = True          # a ribbon along the top rim
        lit[10:40, 20:22] = True          # with a stalk running deep
        np.testing.assert_array_equal(clear_of_boundary(lit, sample, 6), lit)

    def test_zero_disables_it(self) -> None:
        sample = self.island()
        lit = sample & ~ndimage.binary_erosion(
            sample, structure=np.ones((3, 3), bool), iterations=2
        )
        np.testing.assert_array_equal(clear_of_boundary(lit, sample, 0), lit)

    def test_the_stage_sits_between_the_threshold_and_the_seed(self) -> None:
        """Where it was asked to go, pinned against the pipeline itself.

        2026-09-04: the field runs first, then the crowd filter, and the
        recovery last.  Growing the threshold let one mark's contrast reach
        another's; growing a mark the crowd filter is about to drop is work
        for nothing.  The recovery now runs on the survivors alone.
        """
        self.assertEqual(STAGES[STAGES.index(AT_THRESHOLD) + 1], SPARSE)
        self.assertEqual(STAGES[STAGES.index(SPARSE) + 1], UNCROWDED)
        self.assertEqual(STAGES[STAGES.index(UNCROWDED) + 1], NEARBY_CONTRAST)
        self.assertEqual(STAGES[STAGES.index(NEARBY_CONTRAST) + 1], SIZED)
        self.assertEqual(STAGES[STAGES.index(SIZED) + 1], UNREPEATED)
        self.assertEqual(STAGES[STAGES.index(UNREPEATED) + 1], UNRIMMED)
        self.assertEqual(STAGES[STAGES.index(UNRIMMED) + 1], UNTRANSITIONED)
        self.assertEqual(
            STAGES[STAGES.index(UNTRANSITIONED) + 1], PROXIMITY_GROUPED,
        )
        self.assertEqual(
            STAGES[STAGES.index(PROXIMITY_GROUPED) + 1], GROUP_SIZED,
        )
        self.assertEqual(
            STAGES[STAGES.index(GROUP_SIZED) + 1], FLAT_GROUPED,
        )
        self.assertEqual(STAGES[STAGES.index(FLAT_GROUPED) + 1], CONVEX_HULLED)
        self.assertEqual(STAGES[STAGES.index(CONVEX_HULLED) + 1], PADDED)
        self.assertEqual(STAGES[STAGES.index(PADDED) + 1], OFF_BOUNDARY)
        self.assertEqual(STAGES[STAGES.index(OFF_BOUNDARY) + 1], SEEDED)

    def test_the_stage_only_ever_removes(self) -> None:
        colour = np.full((60, 60, 3), 40, dtype=np.uint8)
        colour[10:50, 10:50] = 150
        colour[24:30, 16:44] = 230
        texture = CorpusTexture(
            name="probe", colour=colour,
            domains=(self.island()).astype(np.uint16), normal=None, source={},
        )
        found = detect(texture, whole_pipeline(), capture=True)
        before = found.stage_mask(NEARBY_CONTRAST)
        after = found.stage_mask(OFF_BOUNDARY)
        self.assertEqual(int((after & ~before).sum()), 0)


class PatternCandidateTests(unittest.TestCase):
    """Which groups are worth putting a transform over.

    2026-08-28: big, dense, and covering enough of its island -- and nothing
    more.  Deciding here whether something repeats is the transform's job; two
    attempts to do it with cheaper measures are recorded on
    `ContrastConfig.min_pattern_texels`, and both were wrong.
    """

    @staticmethod
    def config(**changes) -> ContrastConfig:
        """Loose floors unless a test is about one of them."""
        loose = {
            "min_pattern_texels": 1500, "min_pattern_fill": 0.02,
            "min_pattern_area": 0, "min_pattern_coverage": 0.0,
        }
        return replace(ContrastConfig(), **{**loose, **changes})

    @staticmethod
    def field(shape=(300, 300), step=5):
        mask = np.zeros(shape, dtype=bool)
        mask[::step, ::step] = True
        return mask

    def test_a_big_group_qualifies(self) -> None:
        seed = self.field()
        sample = np.ones(seed.shape, dtype=bool)
        np.testing.assert_array_equal(
            pattern_candidates(seed, sample, 2, self.config()), seed
        )

    def test_a_small_group_does_not(self) -> None:
        seed = np.zeros((300, 300), dtype=bool)
        seed[100:120, 100:160] = True                # 1,200 texels
        sample = np.ones(seed.shape, dtype=bool)
        self.assertEqual(
            int(pattern_candidates(seed, sample, 2, self.config()).sum()), 0
        )

    def test_a_group_is_taken_whole(self) -> None:
        """Candidates are groups, so a group is in or out entire."""
        seed = self.field()
        seed[10:14, 10:40] = True                    # a solid mark in the field
        sample = np.ones(seed.shape, dtype=bool)
        np.testing.assert_array_equal(
            pattern_candidates(seed, sample, 2, self.config()), seed
        )

    def test_a_sprawling_wisp_does_not_qualify(self) -> None:
        """What the density floor is for, and the whole of what it is for."""
        seed = np.zeros((600, 600), dtype=bool)
        seed[np.arange(600), np.arange(600)] = True          # a diagonal
        sample = np.ones(seed.shape, dtype=bool)
        thin = self.config(min_pattern_texels=500, min_pattern_fill=0.05)
        loose = self.config(min_pattern_texels=500, min_pattern_fill=0.0)
        self.assertEqual(int(pattern_candidates(seed, sample, 2, thin).sum()), 0)
        self.assertGreater(
            int(pattern_candidates(seed, sample, 2, loose).sum()), 0
        )

    def test_the_area_is_the_rectangle_inside_the_island(self) -> None:
        """A rectangle that hangs off the chart must not count as area.

        One group on a triangular island, measured twice: against the whole
        rectangle it spans, and against the part of that rectangle which is
        surface.  Half of the rectangle is off the island, so counting it makes
        the group read as half as dense -- for no reason but the shape of the
        chart it was packed on.
        """
        half = np.arange(300)[:, None] >= np.arange(300)[None, :]
        seed = self.field((300, 300), 5) & half
        config = self.config(min_pattern_texels=1000, min_pattern_fill=0.03)
        self.assertEqual(
            int(pattern_candidates(
                seed, np.ones(seed.shape, dtype=bool), 2, config).sum()),
            0, "counting the overhang should read it as sparse",
        )
        self.assertGreater(
            int(pattern_candidates(seed, half, 2, config).sum()), 0,
            "measured against the island it is the same density it looks",
        )

    def test_a_mark_in_a_corner_of_its_island_is_not_a_candidate(self) -> None:
        """Coverage, and why it is the floor that does the work.

        A repeating material is the surface of a panel, wall to wall.  A glyph
        sits on a panel and takes a corner of it.  Measured over the corpus,
        every genuine pattern covers at least 0.387 of its island while the
        groups the transform wrongly confirmed covered 0.008 to 0.109.
        """
        sample = np.ones((600, 600), dtype=bool)
        seed = np.zeros((600, 600), dtype=bool)
        seed[20:120, 20:120] = self.field((100, 100), 5)     # a corner of it
        loose = self.config(min_pattern_texels=250)
        self.assertGreater(
            int(pattern_candidates(seed, sample, 2, loose).sum()), 0
        )
        self.assertEqual(
            int(pattern_candidates(
                seed, sample, 2, replace(loose, min_pattern_coverage=0.30)
            ).sum()), 0,
        )

    def test_filling_a_tiny_island_is_not_enough_on_its_own(self) -> None:
        """What the area floor guards, and the only thing it guards.

        Coverage cannot see the difference between a panel of hatch and a small
        mark that happens to fill the whole of a small chart -- both read 1.00.
        """
        sample = np.zeros((300, 300), dtype=bool)
        sample[100:130, 100:130] = True              # a 900-texel island
        seed = self.field((300, 300), 4) & sample
        config = self.config(min_pattern_texels=25, min_pattern_coverage=0.30)
        self.assertGreater(
            int(pattern_candidates(seed, sample, 2, config).sum()), 0
        )
        self.assertEqual(
            int(pattern_candidates(
                seed, sample, 2, replace(config, min_pattern_area=3000)
            ).sum()), 0,
        )

    def test_the_gap_has_to_reach_across_the_pattern(self) -> None:
        """Its elements stand much further apart than a letter's strokes do.

        The measured case: a pedal pad's dot grid arrives as 69 separate groups
        of forty texels at the mark stage's distance of 2, and as one group of
        1,526 at 6.  A candidate stage grouping at 2 never sees an area at all.
        """
        seed = self.field((300, 300), 12)            # dots 12 texels apart
        sample = np.ones(seed.shape, dtype=bool)
        config = self.config(min_pattern_texels=500, min_pattern_fill=0.0)
        self.assertEqual(
            int(pattern_candidates(seed, sample, 2, config).sum()), 0,
            "at 2 the dots are separate specks",
        )
        self.assertGreater(
            int(pattern_candidates(seed, sample, 8, config).sum()), 0,
            "at 8 they are one area",
        )

    def test_zero_switches_it_off(self) -> None:
        seed = self.field()
        sample = np.ones(seed.shape, dtype=bool)
        self.assertEqual(
            int(pattern_candidates(
                seed, sample, 2, self.config(min_pattern_texels=0)).sum()),
            0,
        )

    def test_an_empty_seed_is_no_candidates(self) -> None:
        seed = np.zeros((60, 60), dtype=bool)
        self.assertEqual(
            int(pattern_candidates(seed, ~seed, 2, self.config()).sum()), 0
        )

    def test_the_stage_removes_nothing_itself(self) -> None:
        """It marks; the transform after it is what removes.

        With the transform switched off the seed reaches the mark stage
        untouched, however much the screen flagged.
        """
        colour = np.full((600, 600, 3), 40, dtype=np.uint8)
        colour[40:90, 40:160] = 230
        colour[200:560, 200:560][::7, ::7] = 220
        base = whole_pipeline()
        quiet = replace(
            base, contrast=replace(base.contrast, min_repeat_energy=0.0))
        texture = one_texture(colour)
        found = detect(texture, quiet, capture=True)
        self.assertGreater(int(found.stage_mask(PATTERNED).sum()), 0)
        np.testing.assert_array_equal(
            found.stage_mask(DENSE_SEED), found.stage_mask(TRANSFORMED)
        )
        # And what it marked is a subset of the seed, never an addition.
        self.assertEqual(
            int((found.stage_mask(PATTERNED)
                 & ~found.stage_mask(DENSE_SEED)).sum()), 0
        )


class RepeatScoreTests(unittest.TestCase):
    """The transform's two readings of a group."""

    @staticmethod
    def lattice(shape=(240, 240), step=12, size=3):
        mask = np.zeros(shape, dtype=bool)
        mask[::step, ::step] = True
        return ndimage.binary_dilation(
            mask, structure=np.ones((size, size), bool)
        )

    @staticmethod
    def bar(shape=(240, 240)):
        mask = np.zeros(shape, dtype=bool)
        mask[100:140, 20:220] = True
        return mask

    def test_a_lattice_scores_on_all_three(self) -> None:
        energy, overlap, dip = repeat_scores(self.lattice())
        self.assertGreater(energy, 0.07)
        self.assertGreater(overlap, 0.40)
        self.assertGreater(dip, 0.35)

    def test_a_straight_bar_has_no_off_axis_energy(self) -> None:
        """The case that sank peak height alone.

        A bar's transform is a ridge along one axis through the origin, so by
        peak size it is the most periodic thing there is.  Off the axes it has
        nothing, which is why the energy reading ignores them.
        """
        energy, overlap, _dip = repeat_scores(self.bar())
        self.assertLess(energy, 0.07)
        self.assertGreater(
            overlap, 0.40, "a bar does shift onto itself -- that is the point"
        )

    def test_a_diagonal_bar_is_caught_by_the_dip(self) -> None:
        """What the two axes could not cover.

        Tilt the bar and its ridge leaves both axes, so the energy reading
        stops helping -- but shifting an edge along itself only ever overlaps
        less, so there is no trough on the way to its best lag.  This is
        etk800's steering-wheel panels, which took every legend on them.
        """
        # The two edges of a wide bar, which is what the response of one
        # actually is -- and far enough apart relative to the crop that one
        # landing on the other is the only repeat there is.  A thin synthetic
        # stroke will not do: in a square crop it runs corner to corner and
        # troughs against the crop's own edges, scoring 0.58.
        ys, xs = np.mgrid[0:300, 0:300]
        mask = np.abs(np.abs(ys - xs * 0.6 - 150) - 70) < 2
        energy, overlap, dip = repeat_scores(mask)
        self.assertGreater(energy, 0.07, "the fixture does not reproduce it")
        self.assertGreater(
            overlap, 0.40, "a bar does overlap itself -- that is the point"
        )
        self.assertLess(dip, 0.35)

    def test_a_coarse_pattern_in_a_small_crop_still_scores(self) -> None:
        """Why neither measure may count how many times a thing repeats.

        Five cells across is five cycles, which is a fact about how much was
        cropped rather than about the thing.  Ardente's honeycomb pedal faces
        are exactly this and were being left behind.
        """
        mask = self.lattice((100, 100), step=20, size=7)
        _energy, _overlap, dip = repeat_scores(mask)
        self.assertGreater(dip, 0.35)

    def test_a_single_drawn_mark_has_no_peak(self) -> None:
        """A rectangle outline -- the shape a glyph's response actually makes.

        Not a solid square in a wide empty crop: the pipeline crops tight to
        the group, so a fixture with a lot of surrounding nothing is measuring
        the nothing.  A filled crop is degenerate the other way -- subtract its
        mean and there is no signal left at all.
        """
        mask = np.zeros((240, 240), dtype=bool)
        mask[60:180, 60:180] = True
        mask[70:170, 70:170] = False
        energy, _overlap, _dip = repeat_scores(mask)
        self.assertLess(energy, 0.07)

    def test_something_too_small_to_transform_scores_nothing(self) -> None:
        self.assertEqual(
            repeat_scores(np.ones((10, 10), dtype=bool)), (0.0, 0.0, 0.0)
        )

    def test_reducing_a_big_crop_keeps_the_verdict(self) -> None:
        """A period of many texels survives losing half the resolution."""
        big = self.lattice((600, 600), 20, 5)
        full = repeat_scores(big, longest=1024)
        cut = repeat_scores(big, longest=256)
        for whole, reduced in zip(full, cut):
            self.assertGreater(reduced, whole * 0.5, "the reading collapsed")


class ConfirmedRepeatTests(unittest.TestCase):
    """The verdict, applied to whole groups."""

    @staticmethod
    def config(**changes) -> ContrastConfig:
        return replace(ContrastConfig(), **changes)

    def test_a_lattice_is_confirmed_and_a_glyph_is_not(self) -> None:
        mask = np.zeros((400, 700), dtype=bool)
        mask[40:340, 40:340][::12, ::12] = True      # a lattice
        mask[150:200, 420:660] = True                # a solid mark
        confirmed = confirmed_repeats(mask, 6, self.config())
        self.assertGreater(int(confirmed[:, :400].sum()), 0, "the lattice")
        self.assertEqual(int(confirmed[:, 400:].sum()), 0, "the mark")

    def test_a_group_goes_whole_or_not_at_all(self) -> None:
        mask = np.zeros((400, 400), dtype=bool)
        mask[40:340, 40:340][::12, ::12] = True
        confirmed = confirmed_repeats(mask, 6, self.config())
        np.testing.assert_array_equal(confirmed, mask)

    def test_either_bar_at_zero_switches_it_off(self) -> None:
        mask = np.zeros((400, 400), dtype=bool)
        mask[40:340, 40:340][::12, ::12] = True
        for changes in ({"min_repeat_energy": 0.0},
                        {"min_repeat_overlap": 0.0},
                        {"min_repeat_dip": 0.0}):
            self.assertEqual(
                int(confirmed_repeats(mask, 6, self.config(**changes)).sum()),
                0, str(changes),
            )

    def test_nothing_in_is_nothing_out(self) -> None:
        empty = np.zeros((60, 60), dtype=bool)
        self.assertEqual(
            int(confirmed_repeats(empty, 6, self.config()).sum()), 0
        )

    def test_the_pipeline_takes_a_confirmed_pattern_out(self) -> None:
        colour = np.full((600, 600, 3), 40, dtype=np.uint8)
        colour[40:90, 40:160] = 230                  # a drawn mark
        colour[200:560, 200:560][::7, ::7] = 220     # a repeating field
        found = detect(one_texture(colour), whole_pipeline(), capture=True)
        before = found.stage_mask(DENSE_SEED)
        after = found.stage_mask(TRANSFORMED)
        field = np.zeros(colour.shape[:2], dtype=bool)
        field[200:560, 200:560] = True
        drawn = np.zeros(colour.shape[:2], dtype=bool)
        drawn[36:94, 36:164] = True
        self.assertLess(
            int((after & field).sum()), int((before & field).sum()) * 0.5,
            "the field survived the transform",
        )
        np.testing.assert_array_equal(after & drawn, before & drawn)


class MarksFromSeedTests(unittest.TestCase):
    """Grouping the seed into marks, and refusing the ones that are only dust.

    2026-08-27: no size test here any more -- see `ContrastConfig.mark_gap_px`.
    Grouping says what a mark *is*; whether it is worth correcting is a later
    question, and settling it here would settle it before anything has looked
    at the mark's shape, its ground or its company.
    """

    @staticmethod
    def dots(shape, step, size=1):
        """A field of identical specks -- perforation, weave, fleck."""
        mask = np.zeros(shape, dtype=bool)
        mask[::step, ::step] = True
        if size > 1:
            mask = ndimage.binary_dilation(
                mask, structure=np.ones((size, size), bool)
            )
        return mask

    def test_a_field_of_specks_has_no_stroke_in_it(self) -> None:
        """The case the stroke test exists for: a perforated panel.

        Thousands of texels and thousands of fragments, and not one of them
        big enough to have been drawn.
        """
        seed = self.dots((80, 80), 3)
        self.assertGreater(int(seed.sum()), 36, "the fixture is too small")
        self.assertEqual(int(marks_from_seed(seed, 2, 8).sum()), 0)

    def test_one_solid_stroke_carries_the_whole_mark(self) -> None:
        """A glyph is dust plus a stroke, and the stroke is what says so.

        Deliberately the same dust as the field above -- what differs is the
        one unbroken run through it, which is the whole of the measurement.
        """
        seed = self.dots((40, 40), 3)
        seed[20, 5:35] = True
        kept = marks_from_seed(seed, 2, 8)
        self.assertGreater(int(kept.sum()), int(seed.sum()) * 0.9)

    def test_the_stroke_is_measured_before_the_gap_joins_anything(self) -> None:
        """Otherwise the test measures the gap rather than the mark.

        At gap 2 the specks below weld into one body, and a stroke read off
        *that* would be enormous.  Read off the seed, there is no stroke.
        """
        seed = self.dots((60, 60), 3)
        joined = ndimage.binary_dilation(
            seed, structure=NEIGHBOURHOOD, iterations=2
        )
        self.assertGreater(int(joined.sum()), int(seed.sum()))
        self.assertEqual(int(marks_from_seed(seed, 2, 8).sum()), 0)

    def test_a_small_mark_is_not_thrown_out_for_being_small(self) -> None:
        """The size test that used to sit here would have taken this."""
        seed = np.zeros((40, 40), dtype=bool)
        seed[10, 10:22] = True                      # 12 texels, one stroke
        np.testing.assert_array_equal(marks_from_seed(seed, 2, 8), seed)

    def test_zero_disables_the_stroke_test(self) -> None:
        seed = self.dots((80, 80), 3)
        np.testing.assert_array_equal(marks_from_seed(seed, 2, 0), seed)

    def test_an_empty_seed_survives_the_trip(self) -> None:
        seed = np.zeros((20, 20), dtype=bool)
        self.assertEqual(int(marks_from_seed(seed, 2, 8).sum()), 0)

    def test_a_mark_is_kept_or_dropped_whole(self) -> None:
        """Never thinned: what is being judged is what the run of texels is."""
        seed = np.zeros((60, 60), dtype=bool)
        seed[10:14, 10:40] = True                   # a solid mark
        seed |= self.dots((60, 60), 5) & (
            np.arange(60)[:, None] > 30
        )                                            # dust, well clear of it
        kept = marks_from_seed(seed, 2, 8)
        block = np.zeros((60, 60), dtype=bool)
        block[10:14, 10:40] = True
        np.testing.assert_array_equal(kept, block)

    def test_the_stage_sits_after_the_transform(self) -> None:
        """Seed, screen, transform, bridge, and only then group."""
        self.assertEqual(STAGES[STAGES.index(DENSE_SEED) + 1], PATTERNED)
        self.assertEqual(STAGES[STAGES.index(PATTERNED) + 1], TRANSFORMED)
        self.assertEqual(STAGES[STAGES.index(TRANSFORMED) + 1], BRIDGED)
        self.assertEqual(STAGES[STAGES.index(BRIDGED) + 1], MARKED)

    def test_the_stage_only_ever_removes(self) -> None:
        """Against the stage before it, which is the one that adds."""
        colour = np.full((96, 96, 3), 40, dtype=np.uint8)
        colour[20:44, 20:50] = 230
        base = whole_pipeline()
        found = detect(
            one_texture(colour),
            replace(base, contrast=replace(base.contrast, stop_after_stage=0)),
            capture=True,
        )
        before = found.stage_mask(BRIDGED)
        after = found.stage_mask(MARKED)
        self.assertEqual(int((after & ~before).sum()), 0)

    def test_a_bridge_is_not_a_stroke(self) -> None:
        """Two specks joined by a long bridge still hold no stroke.

        The bridge says the two belong together; it does not say either was
        drawn, and reading it as stroke would let a bridged pair clear a bar
        neither could.
        """
        drawn = np.zeros((40, 40), dtype=bool)
        drawn[20, 8:11] = True
        drawn[20, 26:29] = True
        seed = drawn.copy()
        seed[20, 11:26] = True                       # the bridge between them
        self.assertGreater(int(marks_from_seed(seed, 2, 8).sum()), 0)
        self.assertEqual(int(marks_from_seed(seed, 2, 8, drawn).sum()), 0)


class BoundaryMarginTests(unittest.TestCase):
    """A margin that slides with the island, held between a floor and a cap."""

    @staticmethod
    def panel(height: int, width: int) -> np.ndarray:
        return np.ones((height, width), dtype=bool)

    def test_the_size_is_the_mean_of_the_bounds(self) -> None:
        """Neither side alone: one calls a long panel small, the other calls a
        thin strip large."""
        self.assertEqual(island_size(self.panel(200, 200)), 200.0)
        self.assertEqual(island_size(self.panel(8, 200)), 104.0)

    def test_the_margin_is_a_share_of_that_size(self) -> None:
        self.assertEqual(
            boundary_margin(island_size(self.panel(200, 200)), 0, 999, 0.08), 16
        )

    def test_the_floor_holds_a_small_island_open(self) -> None:
        self.assertEqual(
            boundary_margin(island_size(self.panel(20, 20)), 3, 16, 0.08), 3
        )

    def test_the_ceiling_stops_a_large_panel_being_hollowed(self) -> None:
        size = island_size(self.panel(900, 900))
        self.assertGreater(0.08 * size, 16)
        self.assertEqual(boundary_margin(size, 3, 16, 0.08), 16)

    def test_a_share_of_zero_is_the_floor_everywhere(self) -> None:
        for shape in ((20, 200), (200, 200), (600, 600)):
            self.assertEqual(
                boundary_margin(island_size(self.panel(*shape)), 3, 16, 0.0), 3
            )

    def test_a_bigger_island_gets_a_wider_margin(self) -> None:
        margins = [
            boundary_margin(island_size(self.panel(side, side)), 3, 40, 0.08)
            for side in (120, 400)
        ]
        self.assertLess(margins[0], margins[1])

    def test_the_floor_wins_when_it_is_above_the_ceiling(self) -> None:
        size = island_size(self.panel(200, 200))
        self.assertEqual(boundary_margin(size, 20, 5, 0.08), 20)

    def test_the_margin_is_whole_because_it_is_an_erosion_count(self) -> None:
        margin = boundary_margin(island_size(self.panel(157, 157)), 0, 999, 0.08)
        self.assertIsInstance(margin, int)
        self.assertEqual(margin, 12)


class RimContactTests(unittest.TestCase):
    """How much of the island's rim a region runs along, over its own length.

    The mask is a region's own tight crop, as the pipeline passes it, and the
    rim is that crop's slice of the island's edge.
    """

    @staticmethod
    def rim_under(shape, rows) -> np.ndarray:
        """A straight length of rim along ``rows`` of a region's crop."""
        rim = np.zeros(shape, dtype=bool)
        rim[rows] = True
        return rim

    def test_a_band_lying_along_the_rim_reads_about_one(self) -> None:
        mask = np.ones((4, 80), dtype=bool)
        share = rim_contact(mask, self.rim_under(mask.shape, 0))
        self.assertGreater(share, 0.9)

    def test_a_wiggly_band_reads_the_same_as_a_straight_one(self) -> None:
        """The reason for measuring length and not the region's own outline."""
        straight = np.ones((6, 80), dtype=bool)
        scalloped = straight.copy()
        scalloped[4:, ::2] = False          # a comb along the far side
        rim = self.rim_under(straight.shape, 0)
        self.assertAlmostEqual(
            rim_contact(straight, rim), rim_contact(scalloped, rim), places=6
        )

    def test_a_mark_that_merely_touches_reads_little(self) -> None:
        """Tall and narrow, with only its short end against the rim."""
        mask = np.ones((60, 12), dtype=bool)
        self.assertLess(rim_contact(mask, self.rim_under(mask.shape, 0)), 0.3)

    def test_perimeter_contact_distinguishes_an_edge_run_from_a_touch(self) -> None:
        run = np.ones((4, 80), dtype=bool)
        touch = np.ones((60, 12), dtype=bool)
        self.assertGreater(
            rim_perimeter_contact(run, self.rim_under(run.shape, 0)), 0.4
        )
        self.assertLess(
            rim_perimeter_contact(touch, self.rim_under(touch.shape, 0)), 0.2
        )

    def test_a_mark_away_from_the_rim_reads_nothing(self) -> None:
        mask = np.ones((20, 20), dtype=bool)
        self.assertEqual(rim_contact(mask, np.zeros(mask.shape, dtype=bool)), 0.0)

    def test_it_may_pass_one_when_the_rim_wraps_the_region(self) -> None:
        """Left uncapped: such a region is deep, and the margin keeps it."""
        mask = np.ones((30, 30), dtype=bool)
        rim = np.zeros(mask.shape, dtype=bool)
        rim[0], rim[-1], rim[:, 0], rim[:, -1] = True, True, True, True
        self.assertGreater(rim_contact(mask, rim), 1.0)

    def test_an_empty_mask_is_not_a_division_by_zero(self) -> None:
        empty = np.zeros((10, 10), dtype=bool)
        self.assertEqual(rim_contact(empty, np.ones((10, 10), dtype=bool)), 0.0)


class ClearOfTheRimTests(unittest.TestCase):
    """Refused only when the whole of it lies inside the island's margin.

    Domains arrive cropped to their own bounds, which is what makes the size
    measure free, so the fixtures here are tight islands and not padded ones.
    """

    @staticmethod
    def island(height: int = 180, width: int = 180) -> np.ndarray:
        return np.ones((height, width), dtype=bool)

    @staticmethod
    def bar(x: int, y: int, width: int, height: int) -> Region:
        return Region(
            bounds=(x, y, width, height),
            mask=np.ones((height, width), dtype=bool), domain=1,
        )

    def keep(self, regions, domain=None, floor=3, ceiling=16, share=0.08,
             offset=(0, 0), contact=0.0, max_band_span=0,
             min_perimeter_contact=0.30):
        return clear_of_the_rim(
            regions, self.island() if domain is None else domain, offset,
            floor, ceiling, share, contact, max_band_span,
            min_perimeter_contact,
        )

    def test_a_ribbon_along_the_rim_is_refused(self) -> None:
        """Size is 180, so the margin is 0.08 of that: 14 texels."""
        ribbon = self.bar(10, 0, 160, 8)
        self.assertEqual(self.keep([ribbon]), [])

    def test_a_mark_that_starts_at_the_rim_and_leaves_it_is_kept(self) -> None:
        """The whole point: touching costs nothing, only never leaving does."""
        straddling = self.bar(10, 0, 160, 40)
        self.assertEqual(self.keep([straddling]), [straddling])

    def test_a_mark_well_inside_is_never_looked_at(self) -> None:
        middle = self.bar(70, 70, 40, 40)
        self.assertEqual(self.keep([middle]), [middle])

    def test_the_margin_slides_with_the_island(self) -> None:
        """The same ribbon, on a narrow island and on a wide one.

        The narrow island is 16 x 180, so its mean side is 98 and its margin
        7; the ribbon reaches a depth of 8 and stays.  On the 180-square
        island the margin is 14 and the same ribbon does not.
        """
        ribbon = self.bar(10, 0, 160, 8)
        self.assertEqual(
            self.keep([ribbon], domain=self.island(16, 180)), [ribbon]
        )
        self.assertEqual(self.keep([ribbon]), [])

    def test_a_band_along_the_rim_needs_the_contact_too(self) -> None:
        """With the second condition on, it still goes: it runs along the rim."""
        ribbon = self.bar(10, 0, 160, 8)
        self.assertEqual(self.keep([ribbon], contact=0.35), [])

    def test_the_sampling_boundary_catches_the_detector_inset(self) -> None:
        """The response starts inside the raw UV edge excluded at sampling."""
        domain = self.island()
        sample = sampling_mask(domain, 1)
        inset = self.bar(10, 2, 160, 8)
        # One dilation reaches the sampling rim at row 1, but not the raw
        # domain rim at row 0.
        self.assertEqual(self.keep([inset], domain=domain, contact=0.35), [inset])
        self.assertEqual(self.keep([inset], domain=sample, contact=0.35), [])

    def test_low_fill_stitching_is_a_rim_artifact(self) -> None:
        sample = sampling_mask(self.island(), 1)
        band = self.bar(10, 2, 160, 8)
        stitch_mask = np.zeros((12, 120), dtype=bool)
        stitch_mask[:4] = True
        stitch = Region(
            bounds=(40, 2, 120, 12), mask=stitch_mask, domain=1,
        )
        self.assertEqual(
            self.keep(
                [band], domain=sample, contact=0.35,
                max_band_span=32,
            ), [],
        )
        self.assertEqual(
            self.keep(
                [stitch], domain=sample, contact=0.35,
                max_band_span=32,
            ), [],
        )

    def test_a_large_low_fill_rim_trace_can_bypass_the_span_guard(self) -> None:
        """2026-08-29: a sparse perimeter trace is an edge artefact, not a control."""
        frame = RimCandidate(
            region=self.bar(0, 0, 234, 162), margin_px=30,
            minor_span_px=162, major_span_px=234, fill=0.054,
            perimeter_texels=802, rim_touch_texels=396,
            length_contact=1.69, perimeter_contact=0.49,
            anchor_x=120.0, anchor_y=10.0, width_eligible=False,
            contact_eligible=True, perimeter_eligible=True,
            large_low_fill_eligible=True,
        )
        control = RimCandidate(
            region=self.bar(0, 0, 102, 102), margin_px=20,
            minor_span_px=102, major_span_px=102, fill=0.228,
            perimeter_texels=748, rim_touch_texels=406,
            length_contact=3.98, perimeter_contact=0.54,
            anchor_x=50.0, anchor_y=10.0, width_eligible=False,
            contact_eligible=True, perimeter_eligible=True,
            large_low_fill_eligible=False,
        )
        classify_rim_candidates([frame, control], 0.2, 32)
        self.assertTrue(frame.discarded)
        self.assertFalse(control.discarded)

    def test_a_stitch_train_is_refused_but_a_control_cluster_is_kept(self) -> None:
        sample = sampling_mask(self.island(80, 80), 1)
        stitch = [
            self.bar(x, 2, 20, 8)
            for x in (8, 34, 60)
        ]
        self.assertEqual(
            self.keep(
                stitch, domain=sample, floor=16, share=0.0,
                contact=0.35, max_band_span=32,
            ), [],
        )
        controls = [
            self.bar(30, 2, 20, 8),
            self.bar(30, 70, 20, 8),
            self.bar(2, 30, 8, 20),
            self.bar(70, 30, 8, 20),
        ]
        self.assertEqual(
            self.keep(
                controls, domain=sample, floor=16, share=0.0,
                contact=0.35, max_band_span=32, min_perimeter_contact=0.0,
            ), controls,
        )

    def test_a_mark_inside_the_margin_but_off_the_rim_is_kept(self) -> None:
        """The case the contact condition exists for.

        A compact mark sitting wholly inside the margin band but touching the
        rim over only a short stretch -- a legend near an edge, not the edge.
        """
        legend = self.bar(60, 2, 30, 10)
        self.assertEqual(self.keep([legend], contact=0.0), [])
        self.assertEqual(self.keep([legend], contact=0.35), [legend])

    def test_contact_at_zero_lets_the_margin_decide_alone(self) -> None:
        legend = self.bar(60, 2, 30, 10)
        self.assertEqual(self.keep([legend], contact=0.0), [])

    def test_the_offset_is_taken_off_before_the_crop(self) -> None:
        ribbon = self.bar(1010, 2000, 160, 8)
        self.assertEqual(self.keep([ribbon], offset=(1000, 2000)), [])

    def test_the_stage_switches_off(self) -> None:
        ribbon = self.bar(20, 11, 160, 8)
        self.assertEqual(self.keep([ribbon], floor=0, share=0.0), [ribbon])

    def test_no_regions_at_all_is_not_a_crash(self) -> None:
        self.assertEqual(self.keep([]), [])

    def test_the_stage_only_ever_removes(self) -> None:
        """Through the pipeline: stage 6 is a subset of stage 5."""
        colour = np.full((200, 200, 3), 40, dtype=np.uint8)
        colour[20:44, 20:150] = 230
        found = detect(one_texture(colour), LabConfig(), capture=True)
        before = found.stage_mask(UNREPEATED)
        after = found.stage_mask(UNRIMMED)
        self.assertEqual(int((after & ~before).sum()), 0)


class SparseFieldTests(unittest.TestCase):
    """The blank ground: where a kernel holds at most so many lit texels."""

    def test_an_empty_surface_is_all_blank(self) -> None:
        sample = np.ones((30, 30), dtype=bool)
        lit = np.zeros((30, 30), dtype=bool)
        np.testing.assert_array_equal(sparse_field(lit, sample, 9, 1), sample)

    def test_a_stroke_punches_a_hole_the_kernel_wide(self) -> None:
        """The hole reaches the disc's radius either side of what fired.

        This is what does the grouping: the hole is wider than the mark, so
        two strokes closer than a kernel share one hole without anything
        joining them.
        """
        sample = np.ones((40, 40), dtype=bool)
        lit = np.zeros((40, 40), dtype=bool)
        lit[20, 10:30] = True
        blank = sparse_field(lit, sample, 9, 0)
        self.assertFalse(bool(blank[20, 20]))     # on the stroke
        self.assertFalse(bool(blank[16, 20]))     # four off, the disc reaches
        self.assertTrue(bool(blank[15, 20]))      # five, and past it

    def test_the_window_is_a_disc_and_not_the_square_around_it(self) -> None:
        """One lit texel, and the ground it clears is round.

        A square reaches sqrt(2) further on the diagonal than on the axes, so
        it answers "is anything near me" over a different distance depending
        on which way the neighbour lies.  The corner below is inside the
        square and outside the disc, and that is the whole of the change.
        """
        sample = np.ones((41, 41), dtype=bool)
        lit = np.zeros((41, 41), dtype=bool)
        lit[20, 20] = True
        solid = ~sparse_field(lit, sample, 9, 0)
        span = np.arange(-20, 21)
        disc = (span[:, None] ** 2 + span[None, :] ** 2) <= 4 * 4
        np.testing.assert_array_equal(solid, disc)
        self.assertTrue(bool(solid[16, 20]))      # four along an axis
        self.assertFalse(bool(solid[17, 17]))     # the square's corner

    def test_the_count_is_what_erodes_a_scatter(self) -> None:
        """A stroke is solid, so raising the count only eats the scatter."""
        sample = np.ones((40, 40), dtype=bool)
        scatter = np.zeros((40, 40), dtype=bool)
        scatter[::6, ::6] = True
        stroke = np.zeros((40, 40), dtype=bool)
        stroke[20, 10:30] = True
        lit = scatter | stroke
        for count in (1, 2, 3):
            blank = sparse_field(lit, sample, 9, count)
            self.assertFalse(bool(blank[20, 20]), "the stroke went blank")
        self.assertGreater(
            int(sparse_field(lit, sample, 9, 3).sum()),
            int(sparse_field(lit, sample, 9, 1).sum()),
        )

    def test_it_is_held_to_the_island_but_counted_across_it(self) -> None:
        """Outside is never painted, and the edge is genuinely blank ground."""
        sample = np.zeros((30, 30), dtype=bool)
        sample[10:20, 10:20] = True
        lit = np.zeros((30, 30), dtype=bool)
        blank = sparse_field(lit, sample, 9, 1)
        self.assertEqual(int((blank & ~sample).sum()), 0)
        self.assertTrue(bool(blank[10, 10]))

    def test_either_knob_at_zero_paints_nothing(self) -> None:
        sample = np.ones((20, 20), dtype=bool)
        lit = np.zeros((20, 20), dtype=bool)
        self.assertEqual(int(sparse_field(lit, sample, 0, 1).sum()), 0)

    def test_a_mesh_seam_is_removed_from_the_sparse_fields_ground(self) -> None:
        sample = np.ones((21, 21), dtype=bool)
        mesh_regions = np.zeros((21, 21), dtype=np.uint32)
        mesh_regions[:, :10] = 1
        mesh_regions[:, 11:] = 2
        lit = np.zeros((21, 21), dtype=bool)
        lit[10, 8] = lit[10, 12] = True

        blank, field_sample = sparse_field_with_mesh_cuts(
            lit, sample, mesh_regions, 9, 0,
        )

        self.assertFalse(field_sample[:, 10].any())
        self.assertFalse((field_sample & ~blank)[:, 10].any())

    def test_the_stage_lets_the_threshold_past_untouched(self) -> None:
        """A marker, not a filter -- the boundary stage sees the whole of it."""
        colour = np.full((96, 96, 3), 40, dtype=np.uint8)
        colour[20:44, 20:50] = 230
        base = whole_pipeline()
        found = detect(
            one_texture(colour),
            replace(base, contrast=replace(base.contrast, stop_after_stage=0)),
            capture=True,
        )
        lit = found.stage_mask(NEARBY_CONTRAST)
        boundary = found.stage_mask(OFF_BOUNDARY)
        self.assertEqual(int((boundary & ~lit).sum()), 0)
        # The stage answers with the inverse of what it computes: `sparse` is
        # the blank ground for the overlay, the stage mask is the holes in it,
        # and between them they are exactly the part of the island the detector
        # answers for.
        holes = found.stage_mask(SPARSE)
        self.assertEqual(int((holes & found.sparse).sum()), 0)
        answered = sampling_mask(
            np.ones(found.sparse.shape, dtype=bool),
            base.contrast.boundary_skip_px,
        )
        np.testing.assert_array_equal(holes | found.sparse, answered)
        self.assertEqual(int((found.sparse & lit).sum()), 0)


class RegionsFromHolesTests(unittest.TestCase):
    """The inverse of the sparse field: a hole is a region."""

    @staticmethod
    def ground(shape=(60, 90)):
        return np.ones(shape, dtype=bool)

    def holes(self, blank, offset=(0, 0), label=1, **floors):
        found = regions_from_holes(
            blank, self.ground(blank.shape), label, offset
        )
        if floors:
            found = past_the_size_floor(
                found, floors.get("min_width", 0), floors.get("min_height", 0),
                floors.get("min_area", 0),
            )
        outline, kept = paint_holes(
            found, blank.shape, offset, ContrastConfig()
        )
        return found, outline, kept

    def test_a_hole_becomes_one_region_with_its_own_bounds(self) -> None:
        blank = np.ones((60, 90), dtype=bool)
        blank[10:30, 20:50] = False
        regions, outline, kept = self.holes(blank)
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].bounds, (20, 10, 30, 20))
        self.assertEqual(regions[0].texels, 600)
        self.assertEqual(int(kept.sum()), 600)
        self.assertGreater(int(outline.sum()), 0)

    def test_two_holes_are_two_regions_however_close(self) -> None:
        """Nothing here groups: the field already decided what one mark is."""
        blank = np.ones((60, 90), dtype=bool)
        blank[10:30, 20:40] = False
        blank[10:30, 41:60] = False        # a single texel of ground between
        self.assertEqual(len(self.holes(blank)[0]), 2)

    def test_the_offset_puts_a_region_on_the_atlas(self) -> None:
        blank = np.ones((40, 40), dtype=bool)
        blank[5:15, 6:16] = False
        regions, _outline, _kept = self.holes(blank, offset=(100, 200), label=3)
        self.assertEqual(regions[0].bounds, (106, 205, 10, 10))
        self.assertEqual(regions[0].domain, 3)

    def test_the_floors_are_each_their_own_refusal(self) -> None:
        blank = np.ones((60, 90), dtype=bool)
        blank[10:14, 20:60] = False        # 40 wide, 4 high, 160 texels
        self.assertEqual(len(self.holes(blank)[0]), 1)
        self.assertEqual(len(self.holes(blank, min_width=41)[0]), 0)
        self.assertEqual(len(self.holes(blank, min_height=5)[0]), 0)
        self.assertEqual(len(self.holes(blank, min_area=161)[0]), 0)
        self.assertEqual(
            len(self.holes(blank, min_width=40, min_height=4,
                           min_area=160)[0]), 1
        )

    def test_the_area_is_the_hole_and_not_its_box(self) -> None:
        """A diagonal is not credited with the emptiness either side of it."""
        blank = np.ones((40, 40), dtype=bool)
        for step in range(20):
            blank[10 + step, 10 + step] = False
        regions, _outline, _kept = self.holes(blank)
        self.assertEqual(regions[0].bounds[2:], (20, 20))
        self.assertEqual(regions[0].texels, 20)
        self.assertEqual(len(self.holes(blank, min_area=21)[0]), 0)

    def test_a_ring_keeps_the_ground_it_encloses(self) -> None:
        """And gets its inner rim drawn, the way a mark's own void does."""
        blank = np.ones((60, 60), dtype=bool)
        blank[10:50, 10:50] = False
        blank[20:40, 20:40] = True         # blank ground inside the ring
        regions, outline, _kept = self.holes(blank)
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].texels, 40 * 40 - 20 * 20)
        # a path on the outer edge and another on the inner one
        self.assertTrue(bool(outline[10, 30]))
        self.assertTrue(bool(outline[19, 30]))

    def test_nothing_but_blank_ground_yields_nothing(self) -> None:
        blank = np.ones((30, 30), dtype=bool)
        regions, outline, kept = self.holes(blank)
        self.assertEqual(regions, [])
        self.assertEqual(int(outline.sum()), 0)
        self.assertEqual(int(kept.sum()), 0)

    def test_the_stage_answers_with_the_holes_it_kept(self) -> None:
        """Through the pipeline: `fired` is the regions, `sparse` the ground."""
        colour = np.full((96, 96, 3), 40, dtype=np.uint8)
        colour[20:44, 20:50] = 230
        found = detect(one_texture(colour), LabConfig())
        self.assertNotEqual(found.regions, [])
        self.assertEqual(
            sum(int(region.texels) for region in found.regions),
            int(found.fired.sum()),
        )
        self.assertGreater(int((found.fired & found.sparse).sum()), 0)


class ClearOfACrowdTests(unittest.TestCase):
    """Company, not shape: a quilted panel is hundreds of ordinary regions."""

    @staticmethod
    def spot(x: int, y: int, side: int = 6) -> Region:
        return Region(
            bounds=(x, y, side, side),
            mask=np.ones((side, side), dtype=bool), domain=1,
        )

    def keep(self, regions, radius=120, limit=8, company=None, large_area=0):
        return clear_of_a_crowd(
            regions, regions if company is None else company, radius, limit,
            large_area,
        )

    def test_a_field_of_them_goes_and_a_lone_mark_stays(self) -> None:
        crowd = [self.spot(100 + 20 * (index % 6), 100 + 20 * (index // 6))
                 for index in range(36)]
        alone = self.spot(500, 500)
        kept = self.keep([*crowd, alone])
        self.assertEqual(kept, [alone])

    def test_company_is_only_counted_on_the_mark_s_own_chart(self) -> None:
        """2026-09-04: atlas distance is not surface distance.

        Two charts packed against each other in the atlas are a few texels
        apart and no part of the same surface, so a busy chart used to
        suppress the marks on a quiet one beside it.
        """
        busy = [self.spot(100 + 20 * (index % 6), 100 + 20 * (index // 6))
                for index in range(36)]
        quiet = self.spot(160, 160)          # sat inside the busy cluster
        regions = [*busy, quiet]
        # A Region holds a mask, so identity is the only safe comparison here.
        held = lambda kept: {id(region) for region in kept}
        # Without ids the quiet mark is buried by its neighbours.
        self.assertNotIn(id(quiet), held(self.keep(regions)))
        # Put it on its own chart and its company drops to nothing.
        ids = [1] * len(busy) + [2]
        kept = clear_of_a_crowd(
            regions, regions, 120, 8, 0, region_ids=ids, company_ids=ids,
        )
        self.assertEqual(held(kept), {id(quiet)})

    def test_a_mark_spanning_two_charts_has_no_company(self) -> None:
        """None means no chart owns it, so nothing can crowd it out."""
        busy = [self.spot(100 + 20 * (index % 6), 100 + 20 * (index // 6))
                for index in range(36)]
        spanning = self.spot(160, 160)
        regions = [*busy, spanning]
        ids = [1] * len(busy) + [None]
        kept = clear_of_a_crowd(
            regions, regions, 120, 8, 0, region_ids=ids, company_ids=ids,
        )
        self.assertEqual({id(region) for region in kept}, {id(spanning)})

    def test_without_ids_the_stage_behaves_as_it_always_did(self) -> None:
        crowd = [self.spot(100 + 20 * (index % 6), 100 + 20 * (index // 6))
                 for index in range(36)]
        alone = self.spot(500, 500)
        self.assertEqual(self.keep([*crowd, alone]), [alone])

    def test_a_word_of_letters_is_not_a_crowd(self) -> None:
        """The case the bar has to clear: eight legends in a row."""
        row = [self.spot(100 + 30 * index, 300) for index in range(8)]
        self.assertEqual(self.keep(row), row)

    def test_the_limit_counts_neighbours_and_not_the_region_itself(self) -> None:
        """Six spots, all within reach of each other: five neighbours each."""
        row = [self.spot(100 + 20 * index, 300) for index in range(6)]
        self.assertEqual(self.keep(row, radius=200, limit=5), row)
        self.assertEqual(self.keep(row, radius=200, limit=4), [])

    def test_the_radius_decides_who_is_near(self) -> None:
        pair = [self.spot(100, 300), self.spot(260, 300)]
        self.assertEqual(self.keep(pair, radius=200, limit=0), [])
        self.assertEqual(self.keep(pair, radius=80, limit=0), pair)

    def test_the_radius_is_a_disc_and_not_a_box(self) -> None:
        """160 apart on one axis is inside 200; 160 on both is not."""
        axis = [self.spot(100, 300), self.spot(260, 300)]
        diagonal = [self.spot(100, 100), self.spot(260, 260)]
        self.assertEqual(self.keep(axis, radius=200, limit=0), [])
        self.assertEqual(self.keep(diagonal, radius=200, limit=0), diagonal)

    def test_the_radius_at_zero_switches_the_stage_off(self) -> None:
        """And the limit at zero does not -- it is the strictest setting."""
        crowd = [self.spot(100 + 20 * (index % 6), 100 + 20 * (index // 6))
                 for index in range(36)]
        self.assertEqual(self.keep(crowd, radius=0), crowd)
        self.assertEqual(self.keep(crowd, limit=0), [])

    def test_atlas_coordinates_need_no_mapping_back(self) -> None:
        """Distances are measured where the bounds already live."""
        row = [self.spot(1000 + 50 * index, 2000) for index in range(11)]
        self.assertEqual(self.keep(row, radius=40, limit=0), row)
        self.assertEqual(self.keep(row, radius=60, limit=0), [])
        # At 60 the ends have one neighbour and the middles have two.
        self.assertEqual(self.keep(row, radius=60, limit=1), [row[0], row[-1]])

    def test_company_can_be_a_wider_set_than_the_regions(self) -> None:
        """The field is judged on every hole, not only those that survived.

        A lone survivor standing in a crowd of holes the size floor threw out
        is still standing in a crowd.
        """
        survivor = self.spot(300, 300)
        field = [self.spot(280 + 12 * (index % 5), 280 + 12 * (index // 5))
                 for index in range(25)]
        self.assertEqual(self.keep([survivor]), [survivor])
        self.assertEqual(self.keep([survivor], company=field), [])

    def test_birdshot_cannot_refuse_a_large_region(self) -> None:
        large = self.spot(300, 300, side=12)
        birdshot = [
            self.spot(260 + 12 * (index % 6), 260 + 12 * (index // 6))
            for index in range(36)
        ]
        company = [large, *birdshot]
        self.assertEqual(self.keep([large], company=company, limit=8), [])
        self.assertEqual(
            self.keep([large], company=company, limit=8, large_area=100),
            [large],
        )

    def test_no_regions_at_all_is_not_a_crash(self) -> None:
        self.assertEqual(self.keep([]), [])

    def test_it_only_ever_refuses(self) -> None:
        """Never reshapes a region and never invents one: a subset, in order."""
        row = [self.spot(100 + 20 * index, 300) for index in range(6)]
        kept = self.keep(row, radius=90, limit=2)
        self.assertLess(len(kept), len(row))
        self.assertEqual(kept, [region for region in row if region in kept])

    def test_the_stage_only_ever_removes(self) -> None:
        """Through the pipeline: stage 4 is a subset of stage 3."""
        colour = np.full((200, 200, 3), 40, dtype=np.uint8)
        colour[40:60, 40:160] = 230
        for index in range(36):
            top = 100 + 12 * (index // 6)
            left = 100 + 12 * (index % 6)
            colour[top:top + 6, left:left + 6] = 230
        base = LabConfig()
        found = detect(one_texture(colour), replace(base, contrast=replace(
            base.contrast, min_width_px=0, min_height_px=0, min_area_px=0,
            crowd_radius_px=40, max_neighbours=3,
        )), capture=True)
        before = found.stage_mask(UNCROWDED)
        after = found.stage_mask(SIZED)
        self.assertEqual(int((after & ~before).sum()), 0)


class ClearOfARepeatTests(unittest.TestCase):
    """Material recurrence, rather than a region's own silhouette."""

    @staticmethod
    def region_at(x: int, y: int, width: int, height: int) -> Region:
        return Region(
            bounds=(x, y, width, height),
            mask=np.ones((height, width), dtype=bool), domain=1,
        )

    @staticmethod
    def weave(size=512, period=16):
        rows = np.arange(size)[:, None]
        columns = np.arange(size)[None, :]
        cells = ((rows // period) + (columns // period)) % 2
        grey = np.where(cells == 1, 90, 170).astype(np.uint8)
        return np.repeat(grey[:, :, None], 3, axis=2)

    @staticmethod
    def plain(size=512):
        rng = np.random.default_rng(3)
        image = np.full((size, size, 3), 130, dtype=np.int16)
        image += rng.integers(-6, 7, image.shape)
        return np.clip(image, 0, 255).astype(np.uint8)

    @staticmethod
    def config(**changes):
        return replace(ContrastConfig(), **changes)

    def keep(self, image, region, **changes):
        config = self.config(**changes)
        return clear_of_a_repeat(
            [region], build_repeat_texture_index(image, config), config
        )

    def test_a_scrap_of_weave_is_refused(self) -> None:
        image = self.weave()
        region = self.region_at(240, 240, 32, 32)
        self.assertEqual(self.keep(image, region), [])

    def test_a_word_is_not_a_repeating_texture(self) -> None:
        image = self.plain()
        image[230:282, 160:180] = 20  # O
        image[230:282, 198:204] = 20  # N
        image[230:282, 232:238] = 20  # slash
        image[230:282, 260:280] = 20  # O
        image[230:282, 298:304] = 20  # F
        image[230:282, 320:326] = 20  # F
        region = self.region_at(150, 220, 190, 72)
        config = self.config()
        index = build_repeat_texture_index(image, config)
        own, directions = repeat_texture_evidence(index, region.bounds, config)
        self.assertLess(own, 4)
        self.assertLess(directions, 2)
        self.assertEqual(clear_of_a_repeat([region], index, config), [region])

    def test_a_thin_glyph_fragment_is_not_a_material_patch(self) -> None:
        image = self.weave()
        region = self.region_at(304, 65, 8, 12)
        self.assertEqual(self.keep(image, region), [region])

    def test_a_small_hole_does_not_pay_for_a_texture_search(self) -> None:
        image = self.weave()
        region = self.region_at(240, 240, 31, 32)
        self.assertEqual(self.keep(image, region), [region])

    def test_compact_periodic_insert_uses_the_confirmed_small_tile_fallback(self) -> None:
        image = self.weave()
        mask = np.zeros((80, 80), dtype=bool)
        for row in range(8, 72, 16):
            for column in range(8, 56, 16):
                mask[row:row + 10, column:column + 10] = True
        region = Region(bounds=(220, 220, 80, 80), mask=mask, domain=1)
        config = self.config(repeat_texture_tile_px=48)
        index = build_repeat_texture_index(image, config)
        with patch("glyph_lab.detect.repeat_texture_evidence", side_effect=[(0, 0), (4, 0)]):
            self.assertEqual(clear_of_a_repeat([region], index, config), [])

    def test_a_unique_mark_inside_weave_is_refused_by_its_context(self) -> None:
        image = self.weave()
        image[232:280, 240:272] = 20
        region = self.region_at(240, 232, 32, 48)
        config = self.config()
        index = build_repeat_texture_index(image, config)
        own, directions = repeat_texture_evidence(index, region.bounds, config)
        self.assertLess(own, config.repeat_texture_min_repeats)
        # The result is capped at the configured verdict, so the two unused
        # directions never build their correlation maps.
        self.assertEqual(directions, config.repeat_texture_context_directions)
        self.assertEqual(clear_of_a_repeat([region], index, config), [])

    def test_the_switch_turns_the_stage_off(self) -> None:
        image = self.weave()
        region = self.region_at(240, 240, 32, 32)
        self.assertEqual(
            self.keep(image, region, enable_repeat_texture_filter=False), [region]
        )

    def test_no_regions_at_all_is_not_a_crash(self) -> None:
        self.assertEqual(clear_of_a_repeat([], None, self.config()), [])

    def test_the_stage_only_ever_removes(self) -> None:
        """Through the pipeline: stage 5 is a subset of stage 4."""
        colour = np.full((200, 200, 3), 40, dtype=np.uint8)
        colour[40:160, 40:160][::10, ::10] = 230
        found = detect(one_texture(colour), LabConfig(), capture=True)
        before = found.stage_mask(UNCROWDED)
        after = found.stage_mask(UNREPEATED)
        self.assertEqual(int((after & ~before).sum()), 0)


class GapBridgesTests(unittest.TestCase):
    """The stage that completes a mark broken across a gap.

    ``ratio`` is the response over the bar it was held to, so the seed reads 1
    and up and everything this stage looks at reads below it.  Written as
    arrays because every claim here is about one geometry.
    """

    SPAN, FLOOR, GAP, SHARE = 10, 0.6, 2, 0.0

    def bridge(self, seed, ratio, span=None, floor=None, share=None,
               void=0.0, clear=5, window=81, threshold=None):
        return gap_bridges(
            seed, ratio, np.ones(seed.shape, dtype=bool),
            self.SPAN if span is None else span,
            self.FLOOR if floor is None else floor, self.GAP,
            self.SHARE if share is None else share,
            void, clear, window, threshold,
        )

    @staticmethod
    def broken_stroke(gap_px: int) -> tuple[np.ndarray, np.ndarray]:
        """One stroke cut in two, with the trace still showing in the cut."""
        seed = np.zeros((40, 60), dtype=bool)
        seed[20, 6:24] = True
        seed[20, 24 + gap_px:54] = True
        ratio = np.zeros((40, 60), dtype=np.float32)
        ratio[seed] = 2.0
        ratio[20, 24:24 + gap_px] = 0.8              # under the bar, over the floor
        return seed, ratio

    def groups(self, mask: np.ndarray) -> int:
        """How many marks the stage after this one would make of ``mask``."""
        return ndimage.label(
            ndimage.binary_dilation(
                mask, structure=NEIGHBOURHOOD, iterations=self.GAP
            ),
            structure=NEIGHBOURHOOD,
        )[1]

    def test_a_trace_across_the_cut_joins_the_two_halves(self) -> None:
        """The bridge is taken from outside the closing, so it is the cut
        minus what the grouping distance already reaches from either side."""
        seed, ratio = self.broken_stroke(6)
        joined = self.bridge(seed, ratio)
        self.assertEqual(int(joined.sum()), 6 - self.GAP * 2)
        self.assertEqual(self.groups(seed), 2)
        self.assertEqual(self.groups(seed | joined), 1)

    def test_a_crossing_longer_than_the_span_is_left_alone(self) -> None:
        seed, ratio = self.broken_stroke(20)
        self.assertEqual(int(self.bridge(seed, ratio).sum()), 0)
        self.assertEqual(self.groups(seed), 2)

    def test_a_trace_under_the_floor_is_not_evidence(self) -> None:
        seed, ratio = self.broken_stroke(6)
        ratio[20, 24:30] = self.FLOOR / 2
        self.assertEqual(int(self.bridge(seed, ratio).sum()), 0)

    def test_a_spur_that_leads_nowhere_is_not_a_bridge(self) -> None:
        """The case the crossing length exists for.

        A stroke's own falloff throws traces off sideways into blank ground.
        They touch the mark and go nowhere, and admitting them draws every
        stroke wider instead of completing anything.
        """
        seed = np.zeros((40, 60), dtype=bool)
        seed[20, 6:54] = True
        ratio = np.zeros((40, 60), dtype=np.float32)
        ratio[seed] = 2.0
        ratio[14:20, 30] = 0.8                       # a spur, straight out
        self.assertEqual(int(self.bridge(seed, ratio).sum()), 0)

    def test_a_trace_beside_a_stroke_is_not_a_bridge_either(self) -> None:
        """Nor is the skirt that runs the whole length of one."""
        seed = np.zeros((40, 60), dtype=bool)
        seed[20, 6:54] = True
        ratio = np.zeros((40, 60), dtype=np.float32)
        ratio[seed] = 2.0
        ratio[17, 6:54] = 0.8
        self.assertEqual(int(self.bridge(seed, ratio).sum()), 0)

    def test_two_marks_with_blank_ground_between_them_stay_apart(self) -> None:
        seed, ratio = self.broken_stroke(6)
        ratio[20, 24:30] = 0.0
        self.assertEqual(int(self.bridge(seed, ratio).sum()), 0)

    def test_either_knob_at_zero_switches_the_stage_off(self) -> None:
        seed, ratio = self.broken_stroke(6)
        self.assertGreater(int(self.bridge(seed, ratio).sum()), 0)
        self.assertEqual(int(self.bridge(seed, ratio, span=0).sum()), 0)
        self.assertEqual(int(self.bridge(seed, ratio, floor=0.0).sum()), 0)

    def test_a_lone_piece_has_nothing_to_bridge_to(self) -> None:
        seed = np.zeros((40, 40), dtype=bool)
        seed[20, 10:30] = True
        ratio = np.zeros((40, 40), dtype=np.float32)
        ratio[seed] = 2.0
        ratio[18:23, 8:32] = np.maximum(ratio[18:23, 8:32], 0.8)
        self.assertEqual(int(self.bridge(seed, ratio).sum()), 0)

    def test_an_empty_seed_survives_the_trip(self) -> None:
        empty = np.zeros((20, 20), dtype=bool)
        self.assertEqual(
            int(self.bridge(empty, np.zeros((20, 20), dtype=np.float32)).sum()), 0
        )

    @staticmethod
    def blob_between(seed: np.ndarray, rows: slice, columns: slice) -> np.ndarray:
        """A trace with area rather than a hairline, which is what grain leaves."""
        ratio = np.zeros(seed.shape, dtype=np.float32)
        ratio[seed] = 2.0
        ratio[rows, columns] = 0.8
        return ratio

    def test_a_repair_may_not_outweigh_what_it_repairs(self) -> None:
        """The case the share cap exists for: a scatter of specks in grain.

        Two specks of three texels, six apart, with as much trace between them
        as there is speck either side.  Every other test here passes -- the
        crossing is short, the trace is over the floor, the run arrives at two
        pieces -- and joining them builds more mark than there was.
        """
        seed = np.zeros((40, 40), dtype=bool)
        seed[20, 10:13] = True
        seed[20, 19:22] = True
        ratio = self.blob_between(seed, slice(19, 22), slice(13, 19))
        self.assertEqual(int(self.bridge(seed, ratio, share=0.0).sum()), 6)
        self.assertEqual(int(self.bridge(seed, ratio, share=0.6).sum()), 0)

    def test_a_stroke_can_afford_the_same_repair(self) -> None:
        """The same gap and the same trace, between two things worth joining.

        Six texels of bridge against 42 of stroke rather than 6 of speck, so
        the cap that refuses the scatter above never binds here.
        """
        seed = np.zeros((40, 60), dtype=bool)
        seed[20, 4:24] = True
        seed[20, 30:52] = True
        ratio = self.blob_between(seed, slice(19, 22), slice(24, 30))
        for share in (0.0, 0.3, 0.6):
            self.assertEqual(int(self.bridge(seed, ratio, share=share).sum()), 6)

    def test_zero_switches_the_cap_off(self) -> None:
        seed = np.zeros((40, 40), dtype=bool)
        seed[20, 10:13] = True
        seed[20, 19:22] = True
        ratio = self.blob_between(seed, slice(19, 22), slice(13, 19))
        self.assertGreater(int(self.bridge(seed, ratio, share=0.0).sum()), 0)

    def test_a_surface_with_no_clear_ground_is_not_bridged(self) -> None:
        """The gate: the place is judged, not the bridge.

        The same stroke, the same gap and the same trace in both halves.  What
        differs is only the ground around them -- blank in the first, and in
        the second speckled everywhere so that no part of the surface is ever
        five texels clear of something that fired.
        """
        seed, ratio = self.broken_stroke(6)
        blank = self.bridge(seed, ratio, void=0.1, threshold=seed)
        self.assertGreater(int(blank.sum()), 0)
        speckled = np.zeros(seed.shape, dtype=bool)
        speckled[::4, ::4] = True
        self.assertEqual(
            int(self.bridge(seed, ratio, void=0.1,
                            threshold=seed | speckled).sum()), 0
        )
        # ...and the bridge itself is identical in both, which is the point.
        np.testing.assert_array_equal(blank, self.bridge(seed, ratio, void=0.0))

    def test_the_gate_needs_all_of_its_inputs(self) -> None:
        """Any of the three at zero, or no threshold mask, and it sits out."""
        seed, ratio = self.broken_stroke(6)
        speckled = seed | np.ones(seed.shape, dtype=bool)
        for kwargs in ({"void": 0.0, "threshold": speckled},
                       {"void": 0.1, "clear": 0, "threshold": speckled},
                       {"void": 0.1, "window": 0, "threshold": speckled},
                       {"void": 0.1, "threshold": None}):
            self.assertGreater(int(self.bridge(seed, ratio, **kwargs).sum()), 0)

    def test_the_stage_only_ever_adds(self) -> None:
        colour = np.full((96, 96, 3), 40, dtype=np.uint8)
        colour[20:44, 20:50] = 230
        base = whole_pipeline()
        found = detect(
            one_texture(colour),
            replace(base, contrast=replace(base.contrast, stop_after_stage=0)),
            capture=True,
        )
        before = found.stage_mask(TRANSFORMED)
        after = found.stage_mask(BRIDGED)
        self.assertEqual(int((before & ~after).sum()), 0)

    def test_the_gate_reads_the_voids_in_the_threshold(self) -> None:
        """A place is judged on clear ground, not on what fired or survived.

        Two surfaces holding the same broken stroke: one blank, one speckled
        so evenly that no window is ever five texels clear.  The speckled one
        has *more* threshold texels and the same seed, so any reading of
        presence would rank it at least as high; the void reading refuses it.
        """
        seed, ratio = self.broken_stroke(6)
        speckled = np.zeros(seed.shape, dtype=bool)
        speckled[::4, ::4] = True
        self.assertGreater(int((seed | speckled).sum()), int(seed.sum()))
        self.assertGreater(
            int(self.bridge(seed, ratio, void=0.1, threshold=seed).sum()), 0
        )
        self.assertEqual(
            int(self.bridge(seed, ratio, void=0.1,
                            threshold=seed | speckled).sum()), 0
        )


class RegionsFromMarksTests(unittest.TestCase):
    """The marks stage's answer: one region per constellation of texels."""

    @staticmethod
    def two_blocks() -> np.ndarray:
        mask = np.zeros((60, 90), dtype=bool)
        mask[10:30, 10:30] = True
        mask[10:30, 50:70] = True          # 20 texels of ground between them
        return mask

    def test_the_gap_decides_how_many_regions_there_are(self) -> None:
        apart, _paths = regions_from_marks(
            self.two_blocks(), 2, 1, (0, 0), ContrastConfig()
        )
        together, _paths = regions_from_marks(
            self.two_blocks(), 12, 1, (0, 0), ContrastConfig()
        )
        self.assertEqual(len(apart), 2)
        self.assertEqual(len(together), 1)

    def test_the_bounds_are_the_mark_not_the_dilation(self) -> None:
        """`find_objects` gives the grown bounds, which are `gap` too wide."""
        regions, _paths = regions_from_marks(
            self.two_blocks(), 6, 1, (0, 0), ContrastConfig()
        )
        for region in regions:
            _x, _y, width, height = region.bounds
            self.assertEqual((width, height), (20, 20))

    def test_the_offset_places_the_mark_in_the_atlas(self) -> None:
        """A domain is detected on its own crop and answers in atlas terms."""
        regions, _paths = regions_from_marks(
            self.two_blocks(), 2, 3, (100, 200), ContrastConfig()
        )
        self.assertEqual(
            sorted(r.bounds[:2] for r in regions), [(110, 210), (150, 210)]
        )
        self.assertEqual({r.domain for r in regions}, {3})

    def test_the_path_is_drawn_inside_the_mark(self) -> None:
        mask = self.two_blocks()
        _regions, paths = regions_from_marks(mask, 2, 1, (0, 0), ContrastConfig())
        self.assertGreater(int(paths.sum()), 0)
        self.assertEqual(int((paths & ~mask).sum()), 0)

    def test_nothing_detected_is_no_regions(self) -> None:
        regions, paths = regions_from_marks(
            np.zeros((20, 20), dtype=bool), 2, 1, (0, 0), ContrastConfig()
        )
        self.assertEqual(regions, [])
        self.assertEqual(int(paths.sum()), 0)


class StopAfterStageTests(unittest.TestCase):
    """The pipeline cut short while the front end is being rebuilt."""

    @staticmethod
    def probe() -> CorpusTexture:
        """A block, not a bar.

        These tests need the far end of the pipeline to produce something when
        it is allowed to run, and a long thin bar is refused there by
        `min_noodle_ratio` -- correctly, since a ribbon is what that test is
        for.  A block's edge response is a fat ring and survives.
        """
        colour = np.full((96, 96, 3), 40, dtype=np.uint8)
        colour[20:44, 20:50] = 230
        colour[60:90, 55:85] = np.random.default_rng(5).integers(
            30, 90, size=(30, 30, 1)
        )
        return one_texture(colour)

    def test_the_default_ends_on_the_last_stage_that_is_built(self) -> None:
        """2026-08-28: the size floor, not the marks.

        The default states how far the rebuild has reached, and it has reached
        the sparse field and the floor that sorts its holes.  The stages after
        it are the front end this replaces; they still run if the stop is
        cleared, which is what makes the two comparable.
        """
        found = detect(self.probe(), LabConfig(), capture=True)
        self.assertEqual(
            STAGES[LabConfig().contrast.stop_after_stage - 1], PADDED
        )
        np.testing.assert_array_equal(found.stage_mask(PADDED), found.fired)
        self.assertNotEqual(found.regions, [])
        self.assertGreater(int(found.outlines.sum()), 0)

    def test_the_stages_past_the_stop_never_happen(self) -> None:
        found = detect(self.probe(), LabConfig(), capture=True)
        for name in STAGES[:STAGES.index(PADDED) + 1]:
            self.assertIsNotNone(found.stage_mask(name), name)
        for name in STAGES[STAGES.index(PADDED) + 1:]:
            self.assertIsNone(found.stage_mask(name), f"{name} still ran")

    def test_zero_puts_the_whole_pipeline_back(self) -> None:
        """The reversibility claim, as a test rather than as a comment."""
        base = LabConfig()
        whole = replace(
            base, contrast=replace(base.contrast, stop_after_stage=0)
        )
        found = detect(self.probe(), whole, capture=True)
        self.assertNotEqual(found.regions, [])
        for name in STAGES:
            self.assertIsNotNone(found.stage_mask(name), name)

    def test_a_number_outside_the_pipeline_runs_all_of_it(self) -> None:
        """Including the region pass, which has no seam to cut at."""
        base = LabConfig()
        for value in (-1, LAST_STOP + 1, len(STAGES) + 5):
            config = replace(
                base, contrast=replace(base.contrast, stop_after_stage=value)
            )
            self.assertNotEqual(
                detect(self.probe(), config).regions, [], f"stop {value}"
            )

    def test_the_marks_stage_answers_with_regions(self) -> None:
        """What the stage is for: a green outline round each constellation.

        Grouped by `mark_gap_px` -- the same distance the stage grouped by, so
        the region and the mask agree about what one mark is.
        """
        found = detect(self.probe(), LabConfig())
        self.assertGreater(len(found.regions), 0)
        # The regions partition what survived: every texel is in exactly one,
        # which is what makes a count of them mean anything.
        self.assertEqual(
            sum(int(region.texels) for region in found.regions),
            int(found.fired.sum()),
        )
        # And every path lies inside some region's bounds, nowhere else.
        painted = np.zeros(found.outlines.shape, dtype=bool)
        for x, y, width, height in (r.bounds for r in found.regions):
            painted[y:y + height, x:x + width] = True
        self.assertEqual(int((found.outlines & ~painted).sum()), 0)

    def test_the_rim_decision_filters_the_final_marks(self) -> None:
        """The early sparse-hole rim pass must not be the last use of it."""
        base = whole_pipeline()
        with patch("glyph_lab.detect.clear_of_the_rim", return_value=[]) as rim:
            found = detect(self.probe(), base)
        self.assertGreaterEqual(rim.call_count, 2)
        self.assertEqual(found.regions, [])
        self.assertEqual(int(found.fired.sum()), 0)
        self.assertEqual(int(found.outlines.sum()), 0)

    def test_the_regions_follow_the_grouping_distance(self) -> None:
        """Two marks a gap apart are one region; further apart, two."""
        colour = np.full((96, 96, 3), 40, dtype=np.uint8)
        colour[20:44, 12:32] = 230
        colour[20:44, 52:72] = 230           # 20 texels of ground between them
        base = whole_pipeline()
        near = replace(base, contrast=replace(base.contrast, mark_gap_px=12))
        far = replace(base, contrast=replace(base.contrast, mark_gap_px=1))
        self.assertEqual(len(detect(one_texture(colour), near).regions), 1)
        self.assertEqual(len(detect(one_texture(colour), far).regions), 2)

    def test_each_stop_leaves_its_own_stage_as_the_answer(self) -> None:
        base = whole_pipeline()
        for number in range(1, LAST_STOP + 1):
            config = replace(
                base, contrast=replace(base.contrast, stop_after_stage=number)
            )
            found = detect(self.probe(), config, capture=True)
            np.testing.assert_array_equal(
                found.stage_mask(STAGES[number - 1]), found.fired,
                f"stop {number} did not end on its own stage",
            )
            # The stages that group answer with regions; the rest stop with
            # a mask and no answer, which is the honest thing for a stage that
            # has not decided anything.  The sparse field and the size floor
            # group because a hole *is* a region, and the marks stage groups at
            # the end of the front end this replaces.
            if STAGES[number - 1] in (
                    SPARSE, NEARBY_CONTRAST, SIZED, UNCROWDED, UNREPEATED,
                    UNRIMMED,
                    UNTRANSITIONED, PATTERNED, MARKED, PROXIMITY_GROUPED,
                    GROUP_SIZED,
                    FLAT_GROUPED,
                    CONVEX_HULLED, PADDED):
                self.assertNotEqual(found.regions, [], f"stop {number}")
            else:
                self.assertEqual(
                    found.regions, [], f"stop {number} invented a region"
                )


class StageOfTests(unittest.TestCase):
    """The field-to-stage table, which the panel hides parameters by."""

    @staticmethod
    def every_field() -> set[str]:
        return {entry.name for entry in fields(ContrastConfig)}

    def test_every_parameter_names_a_stage(self) -> None:
        """A field missing from the table would be shown at every stop.

        Which is the quiet failure: the parameter looks live, takes a value,
        and does nothing -- exactly what the table exists to prevent.
        """
        self.assertEqual(self.every_field() - set(STAGE_OF), set())

    def test_the_table_names_no_parameter_that_is_gone(self) -> None:
        self.assertEqual(set(STAGE_OF) - self.every_field(), set())

    def test_every_stage_named_is_a_real_stage(self) -> None:
        for name, number in STAGE_OF.items():
            self.assertTrue(
                0 <= number <= len(STAGES), f"{name} points at stage {number}"
            )

    def test_only_the_stop_itself_belongs_to_no_stage(self) -> None:
        """0 means "always shown", and there is exactly one of those."""
        self.assertEqual(
            {name for name, number in STAGE_OF.items() if number == 0},
            {"stop_after_stage"},
        )

class StageCostTests(unittest.TestCase):
    """The per-stage clock.  Wall time is not assertable; its bookkeeping is."""

    @staticmethod
    def probe() -> CorpusTexture:
        colour = np.full((96, 96, 3), 40, dtype=np.uint8)
        colour[20:26, 10:86] = 230
        return one_texture(colour)

    def test_stages_are_timed_without_being_captured(self) -> None:
        """Cost is collected always: a number behind a switch is never read."""
        found = detect(self.probe(), LabConfig())
        self.assertEqual(found.stages, {}, "it captured without being asked")
        self.assertTrue(found.stage_times, "nothing was timed")

    def test_every_timed_key_is_a_stage(self) -> None:
        found = detect(self.probe(), LabConfig())
        for name in found.stage_times:
            self.assertIn(name, STAGES)

    def test_no_more_time_is_charged_than_elapsed(self) -> None:
        """Catches the double-count, which is the way this goes wrong."""
        texture = self.probe()
        started = time.perf_counter()
        found = detect(texture, LabConfig())
        elapsed = time.perf_counter() - started
        self.assertLessEqual(sum(found.stage_times.values()), elapsed)

    def test_capturing_does_not_bill_the_stages_for_it(self) -> None:
        """`group_outline` is dearer than most of the stages it draws.

        Charging it to them would make every stage look expensive in exactly
        the mode anyone uses to find out what a stage costs.
        """
        texture = self.probe()
        plain = detect(texture, whole_pipeline())
        watched = detect(texture, whole_pipeline(), capture=True)
        self.assertEqual(set(plain.stage_times), set(watched.stage_times))
        # The capture's own work is real and would show up several times over
        # if it were being billed; the stages themselves do the same work in
        # both runs.
        self.assertLess(
            sum(watched.stage_times.values()),
            4.0 * sum(plain.stage_times.values()) + 0.05,
        )

    def test_the_breakdown_is_in_pipeline_order_and_sums_to_one(self) -> None:
        found = detect(self.probe(), LabConfig())
        costs = found.stage_cost()
        self.assertEqual(
            [name for name, _s, _share in costs],
            [name for name in STAGES if name in found.stage_times],
        )
        self.assertAlmostEqual(sum(share for _n, _s, share in costs), 1.0)

    def test_an_empty_result_is_still_timed_to_the_end(self) -> None:
        """A texture with nothing in it still walks every stage."""
        flat = one_texture(np.full((64, 64, 3), 40, dtype=np.uint8))
        found = detect(flat, whole_pipeline())
        self.assertEqual(found.regions, [])
        self.assertIn(MARKED, found.stage_times)
        self.assertEqual(
            [name for name, _s, _share in found.stage_cost()], list(STAGES)
        )


class ConfigTests(unittest.TestCase):
    def test_the_tuned_seed_density_is_the_default(self) -> None:
        """The current corpus setting must be what a fresh session starts with."""
        self.assertEqual(ContrastConfig().min_seed_density, 0.02)

class GroupNearbyRegionsTests(unittest.TestCase):
    """The proximity stage: overlap first, then a gap, never across a seam."""

    @staticmethod
    def block(x: int, y: int, width: int = 4, height: int = 4, domain: int = 1):
        from glyph_lab.regions import Region

        return Region(
            bounds=(x, y, width, height),
            mask=np.ones((height, width), dtype=bool), domain=domain,
        )

    @staticmethod
    def group(regions, gap, mesh=None):
        from glyph_lab.region_grouping import group_nearby_regions

        return group_nearby_regions(regions, gap, (0, 0), mesh)

    def test_overlapping_regions_become_one(self) -> None:
        """The case the stage was added for: one mark claimed twice."""
        kept = self.group([self.block(10, 10), self.block(12, 10)], 0)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].bounds, (10, 10, 6, 4))

    def test_a_gap_of_zero_leaves_separated_marks_alone(self) -> None:
        kept = self.group([self.block(10, 10), self.block(30, 10)], 0)
        self.assertEqual(len(kept), 2)

    def test_the_gap_is_measured_between_the_texels(self) -> None:
        """Four blank columns between them, so they are 5 apart.

        Joined at 5 and left alone at 4 -- the Chebyshev distance between the
        nearest texels, which is one more than the blank count.
        """
        pair = [self.block(10, 10), self.block(18, 10)]
        self.assertEqual(len(self.group(pair, 4)), 2)
        self.assertEqual(len(self.group(pair, 5)), 1)

    def test_a_chain_joins_transitively(self) -> None:
        row = [self.block(10 + 8 * step, 10) for step in range(4)]
        kept = self.group(row, 5)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].bounds, (10, 10, 28, 4))

    def test_the_gap_between_members_is_not_painted_in(self) -> None:
        """Nearness is not evidence that the space between is part of either."""
        kept = self.group([self.block(10, 10), self.block(18, 10)], 5)
        self.assertEqual(len(kept), 1)
        self.assertEqual(int(kept[0].mask.sum()), 32)
        self.assertFalse(bool(kept[0].mask[0, 5]), "the corridor was filled in")

    def test_marks_on_different_charts_stay_apart(self) -> None:
        """Atlas distance is not surface distance; a seam is a different surface."""
        mesh = np.zeros((40, 40), dtype=np.uint32)
        mesh[:, :16] = 1
        mesh[:, 16:] = 2
        pair = [self.block(10, 10), self.block(18, 10)]
        self.assertEqual(len(self.group(pair, 5, mesh)), 2)
        # The same pair on one chart still joins, so it is the seam doing it.
        self.assertEqual(
            len(self.group(pair, 5, np.ones((40, 40), np.uint32))), 1,
        )

    def test_marks_in_different_uv_domains_stay_apart(self) -> None:
        pair = [self.block(10, 10), self.block(12, 10, domain=2)]
        self.assertEqual(len(self.group(pair, 5)), 2)

    def test_nested_boxes_with_distant_texels_are_not_joined(self) -> None:
        """Boxes overlap, texels do not: the reason the test is on the mask."""
        from glyph_lab.regions import Region

        left_mask = np.zeros((20, 20), dtype=bool)
        left_mask[:2, :2] = True
        right_mask = np.zeros((20, 20), dtype=bool)
        right_mask[18:, 18:] = True
        pair = [
            Region(bounds=(0, 0, 20, 20), mask=left_mask, domain=1),
            Region(bounds=(0, 0, 20, 20), mask=right_mask, domain=1),
        ]
        self.assertEqual(len(self.group(pair, 3)), 2)

    def test_a_single_region_is_returned_untouched(self) -> None:
        one = self.block(10, 10)
        self.assertIs(self.group([one], 8)[0], one)


if __name__ == "__main__":
    unittest.main()
