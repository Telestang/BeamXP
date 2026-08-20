"""Which of the four shapes a region's own outline justifies claiming.

The cases here are the ones that separate the shapes from each other, not a
sample of textures.  Each is a filled polygon rasterised the way a detector
would hand it over, so the fit sees quantisation exactly as it does in the
pipeline -- a 20-degree rotation lands about half a degree off square, and the
snap has to absorb that rather than read it as shear.
"""

from __future__ import annotations

import math
import unittest

import cv2
import numpy as np

from mesh_segmentation_transform.annotate_texture_regions import (
    DEFAULT_CONFIG,
    FeatureShape,
    inscribed_circle_radius,
    sheared_outline,
)
from mesh_segmentation_transform.parallelogram_fit import (
    enclosing_parallelogram,
    pad_outline,
    fit_parallelogram,
    hull_edge_directions,
    measure_parallelogram,
)

RECTANGLE = [(0, 0), (100, 0), (100, 40), (0, 40)]
# A switch legend, square enough to fail any elongation test.
LOCK = [(0, 3), (3, 0), (17, 0), (20, 3), (20, 22), (0, 22)]


def rotated(polygon, degrees: float):
    radians = math.radians(degrees)
    cos, sin = math.cos(radians), math.sin(radians)
    return [(x * cos - y * sin, x * sin + y * cos) for x, y in polygon]


def sheared(polygon, shear: float):
    return [(x + shear * y, y) for x, y in polygon]


def shape_of(mask: np.ndarray) -> FeatureShape:
    """Wrap a rasterised mask the way the detector hands a region over."""
    filled = cv2.findNonZero(mask).reshape(-1, 2).astype(float)
    hull = cv2.convexHull(filled.astype(np.float32)).reshape(-1, 2).astype(float)
    rectangle = cv2.minAreaRect(hull.astype(np.float32))
    return FeatureShape(
        area_px=float(len(filled)),
        hull=tuple(tuple(float(v) for v in point) for point in hull),
        hull_area=float(cv2.contourArea(hull.astype(np.float32))),
        rectangle=rectangle,
        rectangle_area=float(rectangle[1][0] * rectangle[1][1]),
        points=filled,
    )


def word(text: str = "AUTO", shear: float = 0.0, degrees: float = 0.0) -> np.ndarray:
    """Real rendered glyphs.

    Hand-drawn polygons are not a fair stand-in here: it is easy to draw a
    "word" whose whole left boundary is one long ramp, and a shape like that
    genuinely is sheared.  What makes real text hard is the opposite -- its
    outline is nearly a box whatever the stems inside are doing.
    """
    canvas = np.zeros((160, 480), np.uint8)
    cv2.putText(
        canvas, text, (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 3.0, 1, 6, cv2.LINE_8
    )
    if shear:
        canvas = cv2.warpAffine(
            canvas,
            np.array([[1.0, -shear, shear * 160.0], [0.0, 1.0, 0.0]], np.float32),
            (480, 160),
            flags=cv2.INTER_NEAREST,
        )
    if degrees:
        canvas = cv2.warpAffine(
            canvas,
            cv2.getRotationMatrix2D((240.0, 80.0), degrees, 1.0),
            (480, 160),
            flags=cv2.INTER_NEAREST,
        )
    return canvas


def feature_of(polygon, pad: int = 6) -> FeatureShape:
    """Rasterise a polygon the way the detector would hand one over."""
    points = np.array(polygon, dtype=np.int32)
    points = points - points.min(axis=0) + pad
    canvas = np.zeros(
        (int(points[:, 1].max()) + pad, int(points[:, 0].max()) + pad), np.uint8
    )
    cv2.fillPoly(canvas, [points], 1)
    filled = cv2.findNonZero(canvas).reshape(-1, 2).astype(float)
    hull = cv2.convexHull(filled.astype(np.float32)).reshape(-1, 2).astype(float)
    rectangle = cv2.minAreaRect(hull.astype(np.float32))
    return FeatureShape(
        area_px=float(len(filled)),
        hull=tuple(tuple(float(v) for v in point) for point in hull),
        hull_area=float(cv2.contourArea(hull.astype(np.float32))),
        rectangle=rectangle,
        rectangle_area=float(rectangle[1][0] * rectangle[1][1]),
        points=filled,
    )


class ShapeLadderTests(unittest.TestCase):
    """Shear is the strictly stronger claim, so it has to be earned."""

    def assert_shape(self, polygon, expect_shear: bool, message: str) -> None:
        shape = feature_of(polygon)
        outline = sheared_outline(shape, DEFAULT_CONFIG)
        self.assertEqual(outline is not None, expect_shear, message)
        if outline is not None:
            self.assertEqual(len(outline), 4)

    def test_an_upright_rectangle_claims_no_shear(self) -> None:
        self.assert_shape(RECTANGLE, False, "an axis-aligned box is not sheared")

    def test_a_rotated_rectangle_claims_no_shear(self) -> None:
        # Rotation is not shear.  Rasterising a 20-degree turn leaves the fit
        # about half a degree off square, and reading that as a lean would
        # resample every rotated mark in the atlas for nothing.
        self.assert_shape(rotated(RECTANGLE, 20), False, "rotation is not shear")

    def test_a_sheared_rectangle_claims_shear(self) -> None:
        self.assert_shape(sheared(RECTANGLE, 0.35), True, "a leaning bar is sheared")

    def test_a_clearly_sheared_rectangle_claims_shear(self) -> None:
        self.assert_shape(sheared(RECTANGLE, 0.25), True, "a real lean counts")

    def test_a_shallow_shear_on_a_long_mark_is_squared_off(self) -> None:
        """Below the area test's resolution, and deliberately so.

        A parallelogram encloses in ``ab*sin(t)`` where the rectangle along a
        needs ``(a + b*cos(t))*b*sin(t)``, so the saving is ``a/(a + b*cos(t))``
        -- it shrinks as the mark gets longer against its lean.  On a 100x40
        bar, 0.20 saves only 6.7% and is squared off while 0.25 saves 8.4% and
        is kept.  This is the right way to be wrong: a shear that saves nothing
        also costs nothing to leave as a rectangle.
        """
        self.assert_shape(sheared(RECTANGLE, 0.20), False, "no saving, no claim")

    def test_the_cut_is_shape_aware_rather_than_a_fixed_angle(self) -> None:
        """A compact mark earns its shear at a shallower lean than a long one.

        0.15 on the switch legend is kept while the same lean on the 100x40 bar
        is not, because the same angle hides a larger share of a squat mark's
        enclosure than of an elongated one's.  A fixed angle threshold cannot
        express that; the area ratio does it without being told.
        """
        self.assert_shape(sheared(LOCK, 0.15), True, "a squat mark leans sooner")
        self.assert_shape(sheared(RECTANGLE, 0.15), False, "a long mark later")

    def test_upright_text_claims_no_shear(self) -> None:
        """Diagonal strokes offer themselves as hull edges.

        A parallelogram leaning to hug the diagonal of an A, a V, a W or a 4
        encloses the word in nearly the same area as the upright box, so a fit
        judged loosely reports a lean the word does not have.  Measured, W4
        fits 77 degrees and AVA 72.
        """
        for text in ("AUTO", "AVA", "W4"):
            with self.subTest(text=text):
                self.assertIsNone(
                    sheared_outline(shape_of(word(text)), DEFAULT_CONFIG)
                )

    def test_rotated_text_claims_no_shear(self) -> None:
        self.assertIsNone(
            sheared_outline(shape_of(word(degrees=20)), DEFAULT_CONFIG)
        )

    def test_a_word_presents_one_direction_and_so_claims_no_shear(self) -> None:
        """One close side is one real direction, which is a rectangle.

        A word offers its baseline and little else: the left and right sides
        of the fit are only the first and last glyph's extremes, so the angle
        between the two directions would be read off a side the word never
        reaches.  Sheared text is declined for the same reason it is declined
        upright, and keeps the rectangle it had.  Recorded because the two are
        genuinely indistinguishable from the outline -- measured, upright AUTO
        reaches 0.50 on its weak direction and sheared AUTO 0.55 -- so a future
        change that starts separating them is doing something new, not fixing
        this.
        """
        upright = shape_of(word())
        leaning = shape_of(word(shear=0.30))
        weak = [
            min(
                fit_parallelogram(
                    np.asarray(shape.hull, dtype=float), feature=shape.points
                ).direction_evidence
            )
            for shape in (upright, leaning)
        ]
        self.assertLess(abs(weak[0] - weak[1]), 0.15)
        for shape in (upright, leaning):
            self.assertIsNone(sheared_outline(shape, DEFAULT_CONFIG))

    def test_evidence_is_read_per_direction_not_per_side(self) -> None:
        """One ragged edge must not erase a direction the other side carries.

        A sheared mark with a ragged edge still has both directions, because
        the stronger side of each pair carries it.  Judging on every side would
        answer a stricter question than the one being asked -- whether the two
        directions are real -- and would discard marks that are plainly leaning.
        """
        shape = feature_of(sheared(LOCK, 0.30)[:-1] + [(-4.0, 14.0), (-1.0, 22.0)])
        fit = fit_parallelogram(
            np.asarray(shape.hull, dtype=float), feature=shape.points
        )
        self.assertLess(min(fit.side_evidence), min(fit.direction_evidence))
        self.assertGreaterEqual(
            min(fit.direction_evidence),
            DEFAULT_CONFIG.parallelogram_min_direction_evidence,
        )


class HullChordTests(unittest.TestCase):
    """The hull cannot be the thing that judges the hull's own edges."""

    def _stack(self) -> FeatureShape:
        """Five upright rows that narrow downwards, as one region.

        This is a *group* of separate marks, and the hull of a group is bounded
        by chords leaping from one mark's corner to another's.  Those chords
        are what turned the ardente's AUTO/headlight legend column into a
        fitted diamond.
        """
        canvas = np.zeros((130, 60), np.uint8)
        for index, (top, width) in enumerate(
            ((10, 46), (35, 42), (58, 36), (80, 30), (102, 24))
        ):
            left = 6 + (46 - width) // 2
            canvas[top : top + 14, left : left + width] = 1
        filled = cv2.findNonZero(canvas).reshape(-1, 2).astype(float)
        hull = cv2.convexHull(filled.astype(np.float32)).reshape(-1, 2).astype(float)
        return FeatureShape(
            area_px=float(len(filled)),
            hull=tuple(tuple(float(v) for v in p) for p in hull),
            hull_area=float(cv2.contourArea(hull.astype(np.float32))),
            rectangle=((0.0, 0.0), (1.0, 1.0), 0.0),
            rectangle_area=1.0,
            points=filled,
        )

    def test_a_stack_of_upright_rows_is_not_a_sheared_mark(self) -> None:
        self.assertIsNone(sheared_outline(self._stack(), DEFAULT_CONFIG))

    def test_the_bridging_chord_is_not_believed(self) -> None:
        """Judged against the hull the chord scores perfectly, by construction.

        Every side of an enclosing parallelogram is a supporting line, so a
        side laid along a hull edge has zero gap whatever that edge means.
        Judged against the feature the rows fall away from it in the middle.
        """
        shape = self._stack()
        hull = np.asarray(shape.hull, dtype=float)
        # The steepest chord the stack offers, which is the direction that
        # produced the diamond.
        chord = max(
            hull_edge_directions(hull, 2.0),
            key=lambda item: abs(item[0][0]),
        )[0]
        corners = enclosing_parallelogram(hull, chord, (0.0, 1.0))
        self.assertIsNotNone(corners)
        by_hull = measure_parallelogram(hull, corners, (chord, (0.0, 1.0)))
        by_feature = measure_parallelogram(
            hull, corners, (chord, (0.0, 1.0)), feature=shape.points
        )
        # The chord itself: perfectly supported by the hull it came from,
        # barely supported by the marks the hull was drawn around.
        self.assertGreater(by_feature.direction_evidence[0], 0.9)
        self.assertLess(min(by_feature.side_evidence), min(by_hull.side_evidence))
        self.assertLess(min(by_feature.side_evidence), 0.2)


class FitGeometryTests(unittest.TestCase):
    def test_the_fit_recovers_the_shear_it_was_given(self) -> None:
        # On a large shape, so this measures the fit rather than the raster.
        # A 40 px tall bar quantises its slanted side into a staircase whose
        # direction is only good to a couple of degrees; four times the size
        # is four times the angular resolution.
        large = [(x * 4, y * 4) for x, y in RECTANGLE]
        for shear in (0.15, 0.30, 0.45):
            with self.subTest(shear=shear):
                shape = feature_of(sheared(large, shear))
                fit = fit_parallelogram(
                    np.asarray(shape.hull, dtype=float), feature=shape.points
                )
                expected = 90.0 - math.degrees(math.atan(shear))
                self.assertAlmostEqual(
                    fit.interior_angle_degrees, expected, delta=1.0
                )

    def test_directions_are_deduplicated_modulo_a_half_turn(self) -> None:
        # A side and the side opposite it are one direction, so a rectangle
        # offers two, not four.
        square = np.array(
            [(0.0, 0.0), (50.0, 0.0), (50.0, 20.0), (0.0, 20.0)], dtype=float
        )
        self.assertEqual(len(hull_edge_directions(square, 2.0)), 2)

    def test_a_degenerate_hull_fits_nothing(self) -> None:
        self.assertIsNone(fit_parallelogram(np.array([(0.0, 0.0), (1.0, 1.0)])))


if __name__ == "__main__":
    unittest.main()


class CircleClusterTests(unittest.TestCase):
    """A circle is a claim about an arrangement, not about one squarish mark."""

    def _dial(self, marks: int) -> tuple[np.ndarray, np.ndarray, tuple]:
        image = np.full((220, 220, 3), 24, dtype=np.uint8)
        placed = [
            (92, 40, 20, 20), (40, 92, 20, 20),
            (144, 92, 20, 20), (92, 144, 20, 20),
        ][:marks]
        return (
            image,
            np.asarray(placed, dtype=np.int32),
            (40, 40, 124, 124),
        )

    def test_four_marks_earn_a_circle(self) -> None:
        image, boxes, bounds = self._dial(4)
        self.assertIsNotNone(
            inscribed_circle_radius(
                image, np.ones(image.shape[:2], bool), bounds,
                DEFAULT_CONFIG, None, boxes,
            )
        )

    def test_three_marks_do_not(self) -> None:
        image, boxes, bounds = self._dial(3)
        self.assertIsNone(
            inscribed_circle_radius(
                image, np.ones(image.shape[:2], bool), bounds,
                DEFAULT_CONFIG, None, boxes,
            )
        )

    def test_a_single_glyph_never_becomes_a_circle(self) -> None:
        """The ardente's lock switch, which used to take this path.

        It inscribed a circle and so was answered before any shape was fitted
        to what is actually printed on it.
        """
        image = np.full((60, 60, 3), 24, dtype=np.uint8)
        cv2.circle(image, (30, 30), 18, (230, 230, 230), -1)
        boxes = np.asarray([(12, 12, 36, 36)], dtype=np.int32)
        self.assertIsNone(
            inscribed_circle_radius(
                image, np.ones(image.shape[:2], bool), (12, 12, 36, 36),
                DEFAULT_CONFIG, None, boxes,
            )
        )

    def test_the_count_ignores_intermediate_groupings(self) -> None:
        """Merges are an artefact of the pipeline; the marks are not."""
        from mesh_segmentation_transform.annotate_texture_regions import (
            contained_region_count,
        )

        arms = [(92, 40, 20, 124), (40, 92, 124, 20)]
        marks = [
            (92, 40, 20, 20), (40, 92, 20, 20),
            (144, 92, 20, 20), (92, 144, 20, 20),
        ]
        bounds = (40, 40, 124, 124)
        self.assertEqual(contained_region_count(np.asarray(arms), bounds), 2)
        self.assertEqual(contained_region_count(np.asarray(marks), bounds), 4)


class AreaRatioTests(unittest.TestCase):
    """A lean that does not tighten the enclosure has explained nothing."""

    def _ratio(self, polygon) -> float:
        shape = feature_of(polygon)
        fit = fit_parallelogram(
            np.asarray(shape.hull, dtype=float), feature=shape.points
        )
        return fit.area / shape.rectangle_area

    def test_an_unsheared_mark_saves_nothing(self) -> None:
        for name, polygon in (
            ("upright", RECTANGLE),
            ("rotated", rotated(RECTANGLE, 20)),
        ):
            with self.subTest(shape=name):
                self.assertGreater(self._ratio(polygon), 0.99)

    def test_a_sheared_mark_saves_measurably(self) -> None:
        self.assertLess(self._ratio(sheared(RECTANGLE, 0.35)), 0.90)
        self.assertLess(self._ratio(sheared(LOCK, 0.30)), 0.85)

    def test_the_saving_grows_with_the_lean(self) -> None:
        ratios = [self._ratio(sheared(RECTANGLE, k)) for k in (0.10, 0.25, 0.45)]
        self.assertEqual(ratios, sorted(ratios, reverse=True))


class PadOutlineTests(unittest.TestCase):
    """Growing a fitted outline without changing what it is."""

    SQUARE = ((10.0, 10.0), (40.0, 10.0), (40.0, 30.0), (10.0, 30.0))
    LEANING = ((10.0, 10.0), (40.0, 10.0), (48.0, 30.0), (18.0, 30.0))

    def _edge_angles(self, corners):
        return [
            math.degrees(
                math.atan2(
                    corners[(i + 1) % 4][1] - corners[i][1],
                    corners[(i + 1) % 4][0] - corners[i][0],
                )
            )
            % 180
            for i in range(4)
        ]

    def test_no_padding_returns_the_outline_unchanged(self) -> None:
        self.assertEqual(pad_outline(self.LEANING, 0.0), self.LEANING)

    def test_the_angle_is_preserved(self) -> None:
        """Which is why the corners move along the edges rather than outwards."""
        for name, corners in (("square", self.SQUARE), ("leaning", self.LEANING)):
            with self.subTest(shape=name):
                for before, after in zip(
                    self._edge_angles(corners),
                    self._edge_angles(pad_outline(corners, 4.0)),
                ):
                    self.assertAlmostEqual(before, after, places=6)

    def test_the_centre_stays_put(self) -> None:
        for name, corners in (("square", self.SQUARE), ("leaning", self.LEANING)):
            with self.subTest(shape=name):
                before = np.asarray(corners, dtype=float).mean(axis=0)
                after = np.asarray(pad_outline(corners, 4.0), dtype=float).mean(axis=0)
                np.testing.assert_allclose(before, after, atol=1e-9)

    def test_the_margin_is_perpendicular_to_each_side(self) -> None:
        """Four texels means four texels across, not four along the edge.

        On a leaning outline those differ, and the one that matters is the
        perpendicular gap: it is what carries a glyph's soft edge over.
        """
        margin = 4.0
        for name, corners in (("square", self.SQUARE), ("leaning", self.LEANING)):
            with self.subTest(shape=name):
                grown = pad_outline(corners, margin)
                for index in range(4):
                    start = np.asarray(corners[index])
                    edge = np.asarray(corners[(index + 1) % 4]) - start
                    normal = np.array((-edge[1], edge[0])) / np.linalg.norm(edge)
                    offset = np.asarray(grown[index]) - start
                    self.assertAlmostEqual(
                        abs(float(offset @ normal)), margin, places=6
                    )

    def test_it_contains_what_it_grew_from(self) -> None:
        grown = np.asarray(pad_outline(self.LEANING, 4.0), dtype=np.float32)
        for point in self.LEANING:
            self.assertGreaterEqual(
                cv2.pointPolygonTest(grown, tuple(point), True), -1e-4
            )

    def test_a_near_edge_reduces_the_margin_rather_than_clipping_it(self) -> None:
        """Clipping a corner would change the angle, which is the one thing
        the mirror cannot survive: the shape it reflects about would no longer
        be the shape that was measured."""
        against_edge = ((1.0, 1.0), (30.0, 1.0), (38.0, 21.0), (9.0, 21.0))
        grown = pad_outline(against_edge, 12.0, (64, 64))
        points = np.asarray(grown, dtype=float)
        self.assertGreaterEqual(points[:, 0].min(), -1e-6)
        self.assertGreaterEqual(points[:, 1].min(), -1e-6)
        self.assertLessEqual(points[:, 0].max(), 63.0 + 1e-6)
        self.assertLessEqual(points[:, 1].max(), 63.0 + 1e-6)
        for before, after in zip(
            self._edge_angles(against_edge), self._edge_angles(grown)
        ):
            self.assertAlmostEqual(before, after, places=6)
