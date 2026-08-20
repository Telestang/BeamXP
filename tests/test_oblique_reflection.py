"""The reflection a sheared region is mirrored about.

An axis-aligned box, a rotated rectangle and a parallelogram are one family at
three levels of freedom, and one reflection serves all of them: reverse one of
the outline's own edge directions and keep the other.  For perpendicular edges
that is the ordinary mirror; for sheared ones it is an oblique reflection,
whose mirror line is the direction kept and whose texels move parallel to the
direction reversed.
"""

from __future__ import annotations

import math
import unittest

import cv2
import numpy as np

from mesh_segmentation_transform.mirror_texture_for_rhd import (
    apply_masked_rotated_flip,
    outline_anisotropy,
    outline_frame,
    oblique_supersample_factor,
    outline_reflection_matrix,
    rotated_axis_for_surface_axis,
)

SQUARE = ((2.0, 2.0), (14.0, 2.0), (14.0, 10.0), (2.0, 10.0))
LEANING = ((2.0, 2.0), (14.0, 2.0), (18.0, 10.0), (6.0, 10.0))


def _matrix(corners, axis: str = "long") -> np.ndarray:
    _origin, long_edge, short_edge = outline_frame(corners)
    return outline_reflection_matrix(long_edge, short_edge, axis)


class ReflectionAlgebraTests(unittest.TestCase):
    def test_it_is_an_involution(self) -> None:
        """Applied twice it is the identity, which is what lets texels swap.

        The whole partner/exchangeable scheme downstream depends on this: a
        texel's partner's partner has to be the texel itself, or a flip would
        have a source and a destination rather than being a swap.
        """
        for name, corners in (("square", SQUARE), ("leaning", LEANING)):
            for axis in ("long", "short"):
                with self.subTest(shape=name, axis=axis):
                    matrix = _matrix(corners, axis)
                    np.testing.assert_allclose(
                        matrix @ matrix, np.eye(2), atol=1e-9
                    )

    def test_it_preserves_area_and_reverses_orientation(self) -> None:
        for name, corners in (("square", SQUARE), ("leaning", LEANING)):
            with self.subTest(shape=name):
                self.assertAlmostEqual(
                    float(np.linalg.det(_matrix(corners))), -1.0, places=9
                )

    def test_it_maps_the_outline_onto_itself(self) -> None:
        """The point of choosing this reflection rather than any other."""
        origin, long_edge, short_edge = outline_frame(LEANING)
        matrix = _matrix(LEANING)
        moved = {
            tuple(
                np.round(origin + long_edge + matrix @ (np.asarray(c) - origin), 6)
            )
            for c in LEANING
        }
        self.assertEqual(moved, {tuple(np.round(c, 6)) for c in LEANING})

    def test_a_right_angled_outline_reduces_to_the_plain_mirror(self) -> None:
        np.testing.assert_allclose(
            _matrix(SQUARE), np.diag((-1.0, 1.0)), atol=1e-9
        )

    def test_a_leaning_outline_is_not_symmetric(self) -> None:
        """Which is why normals need the transpose rather than the matrix.

        A right-angled reflection is symmetric, so transpose and original
        coincide and the distinction never shows.  A sheared one separates
        them, and using the wrong one lights relief from a direction the
        geometry never turned towards.
        """
        matrix = _matrix(LEANING)
        self.assertFalse(np.allclose(matrix, matrix.T, atol=1e-6))

    def test_anisotropy_is_one_for_a_rectangle_and_grows_with_the_lean(self) -> None:
        self.assertAlmostEqual(outline_anisotropy(_matrix(SQUARE)), 1.0, places=6)
        leans = [
            outline_anisotropy(
                _matrix(((0.0, 0.0), (20.0, 0.0), (20.0 + k * 10.0, 10.0), (k * 10.0, 10.0)))
            )
            for k in (0.2, 0.4, 0.8)
        ]
        self.assertEqual(leans, sorted(leans))
        self.assertGreater(leans[0], 1.0)


class MeshDrivenAxisTests(unittest.TestCase):
    """Which direction is reversed still comes from the mesh, not the shape."""

    def test_the_edge_nearest_the_surface_axis_is_the_one_reversed(self) -> None:
        # The long edge runs across the image; the short one runs down it.
        self.assertEqual(rotated_axis_for_surface_axis(LEANING, "horizontal"), "long")
        self.assertEqual(rotated_axis_for_surface_axis(LEANING, "vertical"), "short")

    def test_a_leaning_outline_is_no_longer_refused(self) -> None:
        """It used to be, while only the right-angled sampler existed."""
        self.assertIsNotNone(rotated_axis_for_surface_axis(LEANING, "horizontal"))


class ObliqueFlipTests(unittest.TestCase):
    def _inside(self, corners, shape) -> np.ndarray:
        """Texels well inside the outline, away from its resampled boundary."""
        origin, long_edge, short_edge = outline_frame(corners)
        cross = long_edge[0] * short_edge[1] - long_edge[1] * short_edge[0]
        rows, columns = np.mgrid[0:shape[0], 0:shape[1]]
        dx, dy = columns - origin[0], rows - origin[1]
        along = (dx * short_edge[1] - dy * short_edge[0]) / cross
        across = (dy * long_edge[0] - dx * long_edge[1]) / cross
        return (
            (along > 0.15) & (along < 0.85) & (across > 0.15) & (across < 0.85)
        )

    def test_flipping_twice_restores_the_image(self) -> None:
        """The strongest statement that the geometry is self-consistent.

        Measured on smooth marks rather than white noise.  Noise is the worst
        case for any resampler and nothing a texture atlas contains -- the same
        round trip costs 32 levels on uniform noise and 2.5 on marks, and it is
        the second number that says whether a flipped glyph survives.
        """
        image = np.zeros((24, 24, 3), dtype=np.uint8)
        cv2.circle(image, (10, 6), 4, (240, 200, 160), -1)
        cv2.circle(image, (15, 7), 3, (40, 60, 90), -1)
        image = cv2.GaussianBlur(image, (5, 5), 1.2)
        stencil = np.ones((24, 24), dtype=bool)

        once = image.copy()
        apply_masked_rotated_flip(once, stencil, LEANING, "long")
        self.assertFalse(np.array_equal(once, image))
        twice = once.copy()
        apply_masked_rotated_flip(twice, stencil, LEANING, "long")

        inside = self._inside(LEANING, image.shape[:2])
        self.assertGreater(int(inside.sum()), 20)
        difference = np.abs(twice[inside].astype(int) - image[inside].astype(int))
        self.assertLess(float(difference.mean()), 8.0)

    def test_it_moves_texels_parallel_to_the_reversed_direction(self) -> None:
        """An oblique reflection displaces along the direction it reverses."""
        origin, long_edge, _short = outline_frame(LEANING)
        matrix = _matrix(LEANING)
        point = np.asarray((8.0, 6.0))
        moved = origin + long_edge + matrix @ (point - origin)
        displacement = moved - point
        cross = float(
            displacement[0] * long_edge[1] - displacement[1] * long_edge[0]
        )
        self.assertAlmostEqual(cross, 0.0, places=6)


class SupersamplingTests(unittest.TestCase):
    """Extra samples are spent only where an axis is actually compressed."""

    def test_a_right_angled_reflection_takes_a_single_sample(self) -> None:
        self.assertEqual(oblique_supersample_factor(_matrix(SQUARE)), 1)

    def test_a_leaning_reflection_takes_more(self) -> None:
        self.assertGreater(oblique_supersample_factor(_matrix(LEANING)), 1)

    def test_a_shallow_lean_is_not_worth_supersampling(self) -> None:
        # 5 degrees off square: anisotropy about 1.09, below the threshold.
        shallow = ((0.0, 0.0), (40.0, 0.0), (41.0, 12.0), (1.0, 12.0))
        self.assertEqual(oblique_supersample_factor(_matrix(shallow)), 1)


class NormalTransformTests(unittest.TestCase):
    """Relief has to end up lit the way the mirrored geometry would light it."""

    HEIGHT_SHAPE = (64, 64)

    def _height(self) -> np.ndarray:
        rows, columns = np.mgrid[0 : self.HEIGHT_SHAPE[0], 0 : self.HEIGHT_SHAPE[1]]
        # Anisotropic on purpose: a symmetric bump cannot tell the two
        # candidate transforms apart, which is how this went unnoticed.
        height = (
            120.0
            + 60.0 * np.sin(columns / 7.0)
            + 40.0 * np.cos(rows / 11.0)
            + 0.9 * columns
        )
        return np.dstack([height] * 3).clip(0, 255).astype(np.uint8)

    def _normals(self, height: np.ndarray) -> np.ndarray:
        gradient_x = cv2.Scharr(height.astype(np.float32), cv2.CV_32F, 1, 0) / 16.0
        gradient_y = cv2.Scharr(height.astype(np.float32), cv2.CV_32F, 0, 1) / 16.0
        vectors = np.dstack(
            [-gradient_x, -gradient_y, np.full_like(gradient_x, 20.0)]
        )
        return vectors / np.linalg.norm(vectors, axis=2, keepdims=True)

    def _interior(self) -> np.ndarray:
        origin, long_edge, short_edge = outline_frame(LEANING)
        cross = long_edge[0] * short_edge[1] - long_edge[1] * short_edge[0]
        rows, columns = np.mgrid[0 : self.HEIGHT_SHAPE[0], 0 : self.HEIGHT_SHAPE[1]]
        dx, dy = columns - origin[0], rows - origin[1]
        along = (dx * short_edge[1] - dy * short_edge[0]) / cross
        across = (dy * long_edge[0] - dx * long_edge[1]) / cross
        return (along > 0.2) & (along < 0.8) & (across > 0.2) & (across < 0.8)

    def test_flipping_a_normal_map_matches_flipping_the_surface(self) -> None:
        """The only check that does not require trusting the derivation.

        Flip a height field and read its normals, then flip that height
        field's normal map and compare.  They have to agree, because the two
        describe the same mirrored surface.
        """
        stencil = np.ones(self.HEIGHT_SHAPE, dtype=bool)
        height = self._height()

        flipped_height = height.copy()
        apply_masked_rotated_flip(flipped_height, stencil, LEANING, "long")
        expected = self._normals(flipped_height[:, :, 0])

        encoded = np.clip(
            np.rint((self._normals(height[:, :, 0]) + 1.0) * 127.5), 0, 255
        ).astype(np.uint8)
        apply_masked_rotated_flip(encoded, stencil, LEANING, "long", True)
        got = encoded.astype(np.float32) / 127.5 - 1.0

        inside = self._interior()
        first, second = got[inside][:, :2], expected[inside][:, :2]
        cosine = np.sum(first * second, axis=1) / (
            np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1) + 1e-9
        )
        error = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        self.assertLess(float(np.median(error)), 5.0)

    def test_a_right_angled_outline_cannot_tell_the_two_apart(self) -> None:
        """Which is why this was invisible until a mark actually leaned.

        A right-angled reflection is symmetric, so transpose or not gives the
        same answer and every existing test passed either way.
        """
        _origin, long_edge, short_edge = outline_frame(SQUARE)
        matrix = outline_reflection_matrix(long_edge, short_edge, "long")
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-9)


class EnvelopeTests(unittest.TestCase):
    """Texture inside a fitted shape has to stay inside it.

    This is the property the whole thing rests on: the mirror is applied about
    the mark's own axis so the mark lands back on itself.  A region that loses
    its outline anywhere between detection and the plan drops to the flat flip,
    which mirrors about the image axis and carries a leaning mark straight out
    of the envelope it was fitted in.
    """

    def _provenance(self, shape) -> np.ndarray:
        rows, columns = np.mgrid[0 : shape[0], 0 : shape[1]]
        image = np.zeros(shape + (3,), dtype=np.uint8)
        image[:, :, 0] = (columns * 5) % 256
        image[:, :, 1] = (rows * 5) % 256
        image[:, :, 2] = 180
        return image

    def _outside(self, corners, shape) -> np.ndarray:
        polygon = np.asarray(corners, dtype=np.float32)
        return ~np.array(
            [
                [
                    cv2.pointPolygonTest(polygon, (float(x), float(y)), True) >= -0.5
                    for x in range(shape[1])
                ]
                for y in range(shape[0])
            ]
        )

    def test_a_flip_writes_nothing_outside_the_shape(self) -> None:
        shape = (60, 100)
        for name, corners in (("square", SQUARE), ("leaning", LEANING)):
            for axis in ("long", "short"):
                with self.subTest(shape=name, axis=axis):
                    image = self._provenance(shape)
                    before = image.copy()
                    moved = apply_masked_rotated_flip(
                        image, np.ones(shape, dtype=bool), corners, axis
                    )
                    self.assertGreater(moved, 0)
                    changed = np.any(image != before, axis=2)
                    outside = self._outside(corners, shape)
                    self.assertEqual(int((changed & outside).sum()), 0)

    def test_merging_two_views_of_one_mark_keeps_its_envelope(self) -> None:
        """A mark found in colour and in relief arrives twice and is folded.

        Discarding the outline at that point is what sent the marks legible
        enough to be found twice out of their own envelope.
        """
        from mesh_segmentation_transform.mirror_texture_for_rhd import (
            DEFAULT_RHD_CONFIG,
            merge_region_sets_with_rotations,
        )

        _merged, _contributed, rotations = merge_region_sets_with_rotations(
            [(6, 2, 40, 32)],
            [(8, 4, 36, 28)],
            DEFAULT_RHD_CONFIG,
            [None],
            [LEANING],
        )
        (kept,) = rotations
        self.assertIsNotNone(kept, "the fitted outline was discarded on merge")
        # Same axis as the mark was fitted with...
        original = np.subtract(LEANING[1], LEANING[0])
        merged_edge = np.subtract(kept[1], kept[0])
        cross = float(
            original[0] * merged_edge[1] - original[1] * merged_edge[0]
        ) / (np.linalg.norm(original) * np.linalg.norm(merged_edge))
        self.assertAlmostEqual(cross, 0.0, places=6)
        # ...and wide enough to hold everything that was merged into it.
        polygon = np.asarray(kept, dtype=np.float32)
        for point in LEANING:
            self.assertGreaterEqual(
                cv2.pointPolygonTest(polygon, tuple(point), True), -1e-4
            )

    def test_deduplicating_two_views_keeps_the_envelope(self) -> None:
        from mesh_segmentation_transform.mirror_texture_for_rhd import (
            deduplicate_region_detections,
        )

        _regions, rotations, _removed = deduplicate_region_detections(
            [(6, 2, 40, 32), (7, 3, 39, 31)], [LEANING, None]
        )
        self.assertIsNotNone(rotations[0])


if __name__ == "__main__":
    unittest.main()
