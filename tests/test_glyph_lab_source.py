"""Assembling a detector input from an export layer rather than from disk."""

from __future__ import annotations

import unittest

import numpy as np

from glyph_lab.source import rgb_from_detection_bgr, texture_from_layer


def crop(x: int, y: int, width: int, height: int) -> tuple[int, int, np.ndarray]:
    return (x, y, np.ones((height, width), dtype=bool))


class ChannelOrderTests(unittest.TestCase):
    def test_a_colour_layer_is_flipped_back_to_rgb(self) -> None:
        bgr = np.zeros((4, 4, 3), dtype=np.uint8)
        bgr[:, :, 0] = 10          # blue in the export's order
        bgr[:, :, 2] = 200         # red
        rgb = rgb_from_detection_bgr(bgr)
        self.assertEqual(int(rgb[0, 0, 0]), 200)
        self.assertEqual(int(rgb[0, 0, 2]), 10)

    def test_an_alpha_evidence_layer_is_left_alone(self) -> None:
        """A grey stack is its own mirror; flipping it must not be detectable."""
        grey = np.repeat(
            np.arange(16, dtype=np.uint8).reshape(4, 4)[:, :, None], 3, axis=2
        )
        self.assertTrue(np.array_equal(rgb_from_detection_bgr(grey), grey))

    def test_an_rgba_layer_keeps_only_its_colour(self) -> None:
        rgba = np.zeros((4, 4, 4), dtype=np.uint8)
        rgba[:, :, 2] = 200
        rgba[:, :, 3] = 255
        self.assertEqual(rgb_from_detection_bgr(rgba).shape, (4, 4, 3))

    def test_a_flat_image_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            rgb_from_detection_bgr(np.zeros((4, 4), dtype=np.uint8))


class TextureFromLayerTests(unittest.TestCase):
    def test_the_crops_become_the_domain_labels(self) -> None:
        texture = texture_from_layer(
            "layer", np.zeros((16, 16, 3), dtype=np.uint8),
            [crop(0, 0, 4, 4), crop(8, 8, 4, 4)],
        )
        self.assertEqual(texture.domain_count, 2)
        self.assertEqual(int(texture.domains[0, 0]), 1)
        self.assertEqual(int(texture.domains[9, 9]), 2)
        self.assertEqual(int(texture.domains[6, 6]), 0)

    def test_overlapping_crops_are_refused(self) -> None:
        """Coalescing guarantees disjoint domains; a clash means it broke."""
        with self.assertRaises(ValueError):
            texture_from_layer(
                "layer", np.zeros((16, 16, 3), dtype=np.uint8),
                [crop(0, 0, 8, 8), crop(4, 4, 8, 8)],
            )

    def test_a_normal_of_another_resolution_is_dropped_not_resized(self) -> None:
        texture = texture_from_layer(
            "layer", np.zeros((16, 16, 3), dtype=np.uint8), [crop(0, 0, 4, 4)],
            normal_rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        )
        self.assertIsNone(texture.normal)

    def test_a_matching_normal_is_kept(self) -> None:
        texture = texture_from_layer(
            "layer", np.zeros((16, 16, 3), dtype=np.uint8), [crop(0, 0, 4, 4)],
            normal_rgb=np.full((16, 16, 3), 128, dtype=np.uint8),
        )
        self.assertIsNotNone(texture.normal)
        self.assertEqual(texture.normal.shape, (16, 16, 3))

    def test_seams_of_the_wrong_size_are_an_error_not_a_silent_drop(self) -> None:
        """Unlike a normal map, a mismatched seam map means a caller bug."""
        with self.assertRaises(ValueError):
            texture_from_layer(
                "layer", np.zeros((16, 16, 3), dtype=np.uint8), [crop(0, 0, 4, 4)],
                seam_extents=np.zeros((8, 8), dtype=np.uint16),
            )

    def test_seams_reach_the_texture_as_sizes(self) -> None:
        seams = np.zeros((16, 16), dtype=np.uint16)
        seams[3, :] = 64
        texture = texture_from_layer(
            "layer", np.zeros((16, 16, 3), dtype=np.uint8), [crop(0, 0, 8, 8)],
            seam_extents=seams,
        )
        self.assertIsNotNone(texture.seam_extents)
        self.assertEqual(int(texture.seam_extents[3, 0]), 64)
        self.assertTrue(bool(texture.seams[3, 0]))

    def test_no_seams_is_a_working_texture(self) -> None:
        texture = texture_from_layer(
            "layer", np.zeros((16, 16, 3), dtype=np.uint8), [crop(0, 0, 4, 4)],
        )
        self.assertIsNone(texture.seam_extents)
        self.assertIsNone(texture.seams)


if __name__ == "__main__":
    unittest.main()
