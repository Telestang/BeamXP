"""The outward seam test: two flat grounds meeting at a mark, or one ground.

The cases here are the ones the measurement was actually wrong about at some
point, kept so the next change to it has to answer them again:
a mark's own antialiasing read as the ground beside it, a lettered button read
as a clean seam because flatness was sampled in its blank margin, and a single
flat ground split into "two sides" by a two-means that will always return
something.
"""

from __future__ import annotations

import unittest

import numpy as np

from glyph_lab.regions import Region
from glyph_lab.seam_sides import (
    SideEvidence,
    clear_of_two_sided_seams,
    divides_two_grounds,
    measure_region_sides,
    side_evidence,
    side_features,
)

GAP, DETAIL, BALANCE, COHERENCE = 25.0, 8.0, 0.15, 0.40


def whole(shape: tuple[int, int]) -> np.ndarray:
    return np.ones(shape, dtype=bool)


def ring(shape: tuple[int, int], box: tuple[int, int, int, int],
         width: int = 4) -> np.ndarray:
    """A hollow rectangle: what the detector makes of an outlined feature."""
    mask = np.zeros(shape, dtype=bool)
    top, left, height, width_px = box
    mask[top:top + height, left:left + width_px] = True
    mask[top + width:top + height - width, left + width:left + width_px - width] = False
    return mask


class OneGroundTests(unittest.TestCase):
    """A mark on flat colour is the ordinary case and must never be a seam."""

    def test_a_mark_on_flat_ground_reports_no_gap(self) -> None:
        tone = np.full((80, 80), 40.0)
        mask = np.zeros((80, 80), dtype=bool)
        mask[34:46, 34:46] = True
        evidence = side_evidence(mask, tone, whole((80, 80)), reach=20)
        self.assertEqual(evidence.gap, 0.0)
        self.assertFalse(
            divides_two_grounds(evidence, GAP, DETAIL, BALANCE, COHERENCE)
        )

    def test_speckled_ground_is_not_two_sides(self) -> None:
        """Two-means always returns a split; balance and coherence refuse it."""
        rng = np.random.default_rng(4)
        tone = rng.normal(120.0, 18.0, size=(80, 80))
        mask = np.zeros((80, 80), dtype=bool)
        mask[34:46, 34:46] = True
        evidence = side_evidence(mask, tone, whole((80, 80)), reach=20)
        self.assertLess(evidence.coherence, COHERENCE)
        self.assertFalse(
            divides_two_grounds(evidence, GAP, DETAIL, BALANCE, COHERENCE)
        )


class TwoGroundTests(unittest.TestCase):
    """The case the stage is for."""

    @staticmethod
    def seam() -> tuple[np.ndarray, np.ndarray]:
        tone = np.full((80, 80), 30.0)
        tone[40:, :] = 200.0
        mask = np.zeros((80, 80), dtype=bool)
        mask[37:43, 20:60] = True
        return mask, tone

    def test_a_sliver_along_a_material_boundary_is_a_seam(self) -> None:
        mask, tone = self.seam()
        evidence = side_evidence(mask, tone, whole((80, 80)), reach=20)
        self.assertGreater(evidence.gap, GAP)
        self.assertLess(evidence.detail, DETAIL)
        self.assertTrue(
            divides_two_grounds(evidence, GAP, DETAIL, BALANCE, COHERENCE)
        )

    def test_an_outline_around_a_flat_inset_is_a_seam(self) -> None:
        tone = np.full((90, 90), 200.0)
        tone[30:60, 30:60] = 30.0
        evidence = side_evidence(
            ring((90, 90), (28, 28, 34, 34)), tone, whole((90, 90)), reach=20,
        )
        self.assertTrue(
            divides_two_grounds(evidence, GAP, DETAIL, BALANCE, COHERENCE)
        )


class PrintedInteriorTests(unittest.TestCase):
    """A button's rim encircles a legend, and a legend is not flat ground.

    2026-09-04: measuring flatness in a band a few texels inside the rim put
    the sample in the button's blank margin and never reached the legend, so
    eight lettered buttons across etk800 and etkc read as clean seams.  The
    geometry here is a real one's: a ~60 texel face with the print across its
    middle, which is what puts the legend out of a short reach.
    """

    @staticmethod
    def button(printed: bool) -> tuple[np.ndarray, np.ndarray]:
        tone = np.full((140, 140), 200.0)
        tone[42:98, 42:98] = 30.0
        if printed:
            tone[56:68, 46:84] = 210.0
        return ring((140, 140), (40, 40, 60, 60)), tone

    def test_a_lettered_button_is_not_a_seam(self) -> None:
        mask, tone = self.button(printed=True)
        evidence = side_evidence(mask, tone, whole((140, 140)), reach=20)
        self.assertGreater(evidence.detail, DETAIL)
        self.assertFalse(
            divides_two_grounds(evidence, GAP, DETAIL, BALANCE, COHERENCE)
        )

    def test_the_same_button_without_the_legend_is_a_seam(self) -> None:
        """The legend is doing the work, not the shape."""
        mask, tone = self.button(printed=False)
        evidence = side_evidence(mask, tone, whole((140, 140)), reach=20)
        self.assertTrue(
            divides_two_grounds(evidence, GAP, DETAIL, BALANCE, COHERENCE)
        )

    def test_a_reach_that_stops_short_of_the_legend_misses_it(self) -> None:
        """Why flatness is taken over the whole side and not near the rim.

        The failure this guards is silent -- the region simply reads flat --
        so it is pinned as a fact about the measurement rather than left to
        the corpus to catch again.
        """
        mask, tone = self.button(printed=True)
        short = side_evidence(mask, tone, whole((140, 140)), reach=6)
        self.assertLess(short.detail, DETAIL, "the short reach saw the legend")
        self.assertTrue(
            divides_two_grounds(short, GAP, DETAIL, BALANCE, COHERENCE),
            "a short reach calls a lettered button a seam",
        )


class EnclosedFloorTests(unittest.TestCase):
    """A mark closes around its own counters; a seam closes around a feature.

    2026-09-04: the ardente's fuel pictogram comes back as a ring with its own
    white body inside, which is a textbook two-plateau reading -- black
    outside, white inside, gap 183, detail 0.07.  What it is not is a material
    boundary, and the 142 texels it encloses are what say so, against the etkc
    slider bezel's 254.
    """

    @staticmethod
    def seam_like(enclosed: int) -> object:
        """Evidence that passes every other gate, varying only the interior.

        Built rather than rendered: reproducing the ardente's exact numbers in
        a synthetic needs a mark whose interior is large enough in extent to
        have a flat middle but small enough in count to be a counter, and a
        fixture contorted to hit both says less about the rule than the two
        real measurements in the docstring do.  What the floor is, is a clause
        in `divides_two_grounds`, and this is that clause.
        """
        return SideEvidence(
            gap=183.0, detail=0.07, shift=0.0, balance=0.20, coherence=1.0,
            two_sided=True, enclosed=enclosed,
        )

    def test_a_small_enclosed_body_is_not_a_material_boundary(self) -> None:
        """142 texels: the ardente pictogram's own white body."""
        evidence = self.seam_like(enclosed=142)
        self.assertTrue(
            divides_two_grounds(evidence, GAP, DETAIL, BALANCE, COHERENCE, 0),
            "without the floor it is taken for a seam",
        )
        self.assertFalse(
            divides_two_grounds(evidence, GAP, DETAIL, BALANCE, COHERENCE, 200)
        )

    def test_a_large_enclosed_feature_still_is_one(self) -> None:
        """254 texels: the etkc slider bezel's slot."""
        self.assertTrue(divides_two_grounds(
            self.seam_like(enclosed=254), GAP, DETAIL, BALANCE, COHERENCE, 200,
        ))

    def test_the_enclosed_count_comes_off_the_mask(self) -> None:
        """The count the floor reads is the void the mark closes around."""
        tone = np.full((120, 120), 200.0)
        tone[50:70, 50:70] = 30.0
        mask = np.zeros((120, 120), dtype=bool)
        mask[45:75, 45:75] = True
        mask[50:70, 50:70] = False
        evidence = side_evidence(mask, tone, whole((120, 120)), reach=20)
        self.assertEqual(evidence.enclosed, 400)

    def test_a_mark_enclosing_nothing_is_unaffected(self) -> None:
        """Six of the seams this removes are open slivers with no interior."""
        tone = np.full((80, 80), 30.0)
        tone[40:, :] = 200.0
        mask = np.zeros((80, 80), dtype=bool)
        mask[37:43, 20:60] = True
        evidence = side_evidence(mask, tone, whole((80, 80)), reach=20)
        self.assertEqual(evidence.enclosed, 0)
        self.assertTrue(
            divides_two_grounds(evidence, GAP, DETAIL, BALANCE, COHERENCE, 10_000),
            "a floor must not reach a region that encloses nothing",
        )


class HaloTests(unittest.TestCase):
    """A mark's own antialiasing belongs to the mark, not to the ground."""

    def test_the_halo_is_not_read_as_a_second_ground(self) -> None:
        tone = np.full((80, 80), 40.0)
        tone[32:48, 32:48] = 240.0        # the mark
        tone[31, 31:49] = tone[48, 31:49] = 140.0   # its halo
        tone[31:49, 31] = tone[31:49, 48] = 140.0
        mask = np.zeros((80, 80), dtype=bool)
        mask[32:48, 32:48] = True
        evidence = side_evidence(mask, tone, whole((80, 80)), reach=20)
        self.assertEqual(evidence.gap, 0.0)


class FilterTests(unittest.TestCase):
    """`clear_of_two_sided_seams` over a list of regions."""

    @staticmethod
    def regions() -> tuple[list[Region], np.ndarray]:
        colour = np.full((80, 120, 3), 30, dtype=np.uint8)
        colour[40:, :] = 200
        seam = Region((20, 37, 40, 6), np.ones((6, 40), dtype=bool), 1)
        colour[10:20, 80:95] = 210          # a mark on the upper ground alone
        mark = Region((80, 10, 15, 10), np.ones((10, 15), dtype=bool), 1)
        return [seam, mark], colour

    def test_it_keeps_the_mark_and_drops_the_seam(self) -> None:
        regions, colour = self.regions()
        kept = clear_of_two_sided_seams(
            regions, colour, whole((80, 120)), (0, 0), 20,
            GAP, DETAIL, BALANCE, COHERENCE,
        )
        self.assertEqual([r.bounds for r in kept], [(80, 10, 15, 10)])

    def test_a_zero_gap_disables_the_stage(self) -> None:
        regions, colour = self.regions()
        kept = clear_of_two_sided_seams(
            regions, colour, whole((80, 120)), (0, 0), 20,
            0.0, DETAIL, BALANCE, COHERENCE,
        )
        self.assertEqual(len(kept), len(regions))

    def test_supplied_measurements_are_not_recomputed(self) -> None:
        """The probe and the filter must agree, so the filter reuses them."""
        regions, colour = self.regions()
        kept = clear_of_two_sided_seams(
            regions, colour, whole((80, 120)), (0, 0), 20,
            GAP, DETAIL, BALANCE, COHERENCE,
            measurements={id(region): None for region in regions},
        )
        self.assertEqual(len(kept), len(regions), "None means unjudged, so kept")


class OffsetTests(unittest.TestCase):
    """Regions carry atlas coordinates; the colour is the domain's crop."""

    def test_the_offset_places_the_region_in_the_crop(self) -> None:
        colour = np.full((60, 60, 3), 30, dtype=np.uint8)
        colour[30:, :] = 200
        region = Region((520, 527, 30, 6), np.ones((6, 30), dtype=bool), 1)
        placed = measure_region_sides(
            region, colour, whole((60, 60)), (500, 500), 15,
        )
        self.assertIsNotNone(placed)
        self.assertGreater(placed.gap, GAP)

    def test_a_region_outside_the_crop_measures_nothing(self) -> None:
        colour = np.full((60, 60, 3), 30, dtype=np.uint8)
        region = Region((900, 900, 10, 10), np.ones((10, 10), dtype=bool), 1)
        self.assertIsNone(
            measure_region_sides(region, colour, whole((60, 60)), (0, 0), 15)
        )


class FeatureTests(unittest.TestCase):

    def test_no_evidence_reports_no_features(self) -> None:
        self.assertEqual(side_features(None), {})

    def test_the_features_name_every_measurement(self) -> None:
        tone = np.full((80, 80), 30.0)
        tone[40:, :] = 200.0
        mask = np.zeros((80, 80), dtype=bool)
        mask[37:43, 20:60] = True
        features = side_features(side_evidence(mask, tone, whole((80, 80)), 20))
        self.assertEqual(set(features), {
            "side gap", "side detail", "side shift", "side balance",
            "side coherence", "two sided", "enclosed texels",
        })


if __name__ == "__main__":
    unittest.main()
