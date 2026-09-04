"""The seam between the export path and the rebuilt detector.

These cover the wiring, not the detector: what reaches ``glyph_lab``, what
comes back, which filters the new path keeps and which it drops.  Whether the
detector finds the right marks is a question for the corpus and the eye.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mesh_segmentation_transform import mirror_texture_for_rhd as rhd
from mesh_segmentation_transform.glyph_lab_detection import (
    DEFAULT_GLYPH_LAB_CONFIG,
    DetectedMark,
    detect_layer_marks,
)


def island(x: int, y: int, width: int, height: int) -> rhd.UvIslandCrop:
    return rhd.UvIslandCrop(
        (x, y, x + width, y + height), np.ones((height, width), dtype=bool)
    )


class DomainCropTests(unittest.TestCase):
    def test_islands_become_disjoint_crops(self) -> None:
        crops = rhd.glyph_lab_domain_crops(
            (island(0, 0, 4, 4), island(8, 8, 4, 4)),
            np.zeros((16, 16), dtype=bool),
        )
        self.assertEqual(len(crops), 2)
        self.assertEqual([(x, y) for x, y, _mask in crops], [(0, 0), (8, 8)])

    def test_overlapping_islands_coalesce_into_one_domain(self) -> None:
        """The invariant a corpus texture is built on, and the lab's too."""
        crops = rhd.glyph_lab_domain_crops(
            (island(0, 0, 8, 8), island(4, 4, 8, 8)),
            np.zeros((16, 16), dtype=bool),
        )
        self.assertEqual(len(crops), 1)

    def test_no_islands_falls_back_to_the_domain_mask(self) -> None:
        domain = np.zeros((16, 16), dtype=bool)
        domain[2:6, 3:9] = True
        crops = rhd.glyph_lab_domain_crops((), domain)
        self.assertEqual(len(crops), 1)
        x, y, mask = crops[0]
        self.assertEqual((x, y), (3, 2))
        self.assertEqual(mask.shape, (4, 6))

    def test_an_empty_domain_yields_nothing(self) -> None:
        self.assertEqual(
            rhd.glyph_lab_domain_crops((), np.zeros((8, 8), dtype=bool)), []
        )


class MirroredOverlapTests(unittest.TestCase):
    """The one filter the new path keeps, measured over texels not rectangles."""

    @staticmethod
    def run_with(marks, mirror):
        with patch.object(
            rhd, "detect_layer_marks", return_value=marks
        ):
            return rhd.detect_layer_regions_with_glyph_lab(
                np.zeros((16, 16, 3), dtype=np.uint8),
                mirror,
                np.ones((16, 16), dtype=bool),
                "colour",
                "colour",
                (island(0, 0, 16, 16),),
                rhd.DEFAULT_RHD_CONFIG,
                log=lambda *_a: None,
            )

    def test_a_mark_on_mirrored_geometry_is_kept(self) -> None:
        mirror = np.zeros((16, 16), dtype=bool)
        mirror[:, :8] = True
        mask = np.ones((4, 4), dtype=bool)
        found = self.run_with([DetectedMark((1, 1, 4, 4), mask, None)], mirror)
        self.assertEqual(found.detected, 1)
        self.assertEqual(found.regions, [(1, 1, 4, 4)])
        self.assertEqual(len(found.masks), 1)

    def test_a_mark_off_mirrored_geometry_is_dropped(self) -> None:
        mirror = np.zeros((16, 16), dtype=bool)
        mirror[:, :8] = True
        mask = np.ones((4, 4), dtype=bool)
        found = self.run_with([DetectedMark((10, 1, 4, 4), mask, None)], mirror)
        self.assertEqual(found.detected, 1)
        self.assertEqual(found.regions, [])

    def test_the_overlap_is_measured_over_the_marks_texels(self) -> None:
        """A mark wholly on mirrored geometry passes though its box is not.

        The rectangle overhangs the mirrored/rigid split by five of its eight
        columns, so the rectangle-wide measure scores 0.375 against a 0.5 floor
        and declines a mark that never touches the split.  Measured over the
        mark's own texels it scores 1.0, which is the truth about it.
        """
        mirror = np.zeros((16, 16), dtype=bool)
        mirror[:, :8] = True
        mask = np.zeros((8, 8), dtype=bool)
        mask[0:2, 0:2] = True          # atlas (5,0)-(6,1), all of it mirrored
        bounds = (5, 0, 8, 8)
        over_the_box = float(mirror[0:8, 5:13].mean())
        self.assertLess(over_the_box, rhd.DEFAULT_RHD_CONFIG.min_region_mirror_overlap)
        found = self.run_with([DetectedMark(bounds, mask, None)], mirror)
        self.assertEqual(found.regions, [bounds])

    def test_a_mark_with_no_texels_is_dropped_rather_than_divided_by_zero(self) -> None:
        found = self.run_with(
            [DetectedMark((1, 1, 4, 4), np.zeros((4, 4), dtype=bool), None)],
            np.ones((16, 16), dtype=bool),
        )
        self.assertEqual(found.regions, [])


class DroppedFilterTests(unittest.TestCase):
    def test_the_containment_and_blob_filters_do_not_run(self) -> None:
        """Both are rectangle-era shape arguments the rebuilt pipeline replaces."""
        mirror = np.ones((16, 16), dtype=bool)
        marks = [DetectedMark((1, 1, 4, 4), np.ones((4, 4), dtype=bool), None)]
        with (
            patch.object(rhd, "detect_layer_marks", return_value=marks),
            patch.object(
                rhd, "feature_escape",
                side_effect=AssertionError("containment filter must not run"),
            ),
            patch.object(
                rhd, "hull_solidity",
                side_effect=AssertionError("blob filter must not run"),
            ),
        ):
            found = rhd.detect_layer_regions_with_glyph_lab(
                np.zeros((16, 16, 3), dtype=np.uint8),
                mirror,
                np.ones((16, 16), dtype=bool),
                "colour",
                "colour",
                (island(0, 0, 16, 16),),
                rhd.DEFAULT_RHD_CONFIG,
                log=lambda *_a: None,
            )
        self.assertEqual(found.regions, [(1, 1, 4, 4)])


class LayerKindTests(unittest.TestCase):
    def test_production_uses_the_promoted_rim_contact_threshold(self) -> None:
        """The lab's tuned value, pinned where the exporter reads it.

        2026-09-04: 0.5, re-promoted from the lab session.  The number is what
        changed, not what this asks -- production still has to carry whatever
        the lab settled on, and this fails if the two drift apart.
        """
        self.assertEqual(DEFAULT_GLYPH_LAB_CONFIG.contrast.min_rim_contact, 0.5)

    def test_a_colour_layer_reports_no_fitted_shape(self) -> None:
        """The colour pipeline has no rectangle-fitting stage to report one."""
        colour = np.zeros((32, 32, 3), dtype=np.uint8)
        colour[8:16, 8:24] = 240
        crops = [(0, 0, np.ones((32, 32), dtype=bool))]
        marks = detect_layer_marks("t", colour, crops, "colour")
        self.assertTrue(all(mark.rotation is None for mark in marks))

    def test_a_normal_layer_reads_its_own_vectors(self) -> None:
        """No companion is supplied, so the layer must become its own normal."""
        normal = np.full((64, 64, 3), (128, 128, 255), dtype=np.uint8)
        # A slope-free plateau bounded by a ring of slope: the hole the
        # detector paints.
        normal[20:44, 20:44, 0] = 200
        normal[24:40, 24:40, 0] = 128
        crops = [(0, 0, np.ones((64, 64), dtype=bool))]
        marks = detect_layer_marks("t", normal, crops, "normal")
        self.assertIsInstance(marks, list)
        for mark in marks:
            self.assertEqual(mark.mask.shape, (mark.bounds[3], mark.bounds[2]))

    def test_no_domains_detects_nothing_rather_than_raising(self) -> None:
        self.assertEqual(
            detect_layer_marks("t", np.zeros((8, 8, 3), dtype=np.uint8), [], "colour"),
            [],
        )


class OpacityMaskTests(unittest.TestCase):
    def test_an_opacity_layer_stays_on_the_legacy_front_end(self) -> None:
        """An authored visibility mask is a declaration, not evidence."""
        config = rhd.RhdTextureConfig(detector="glyph_lab")
        self.assertEqual(config.detector, "glyph_lab")
        opacity = rhd._production_layer_detection_config(
            rhd.DEFAULT_CONFIG, False, authoritative_opacity_mask=True,
        )
        self.assertEqual(opacity.box_source, "opacity_mask")


class MeshSeamTests(unittest.TestCase):
    def test_a_symbol_with_no_triangles_yields_no_seams(self) -> None:
        class Loaded:
            class tree:
                @staticmethod
                def getroot():
                    return object()

        with patch.object(
            rhd, "discontinuous_edge_extents", side_effect=ValueError("no triangles")
        ):
            self.assertIsNone(
                rhd.material_mesh_seam_extents(Loaded(), {"a-material"}, (16, 16))
            )

    def test_the_smaller_chart_wins_a_shared_texel(self) -> None:
        """A margin at a seam is sized for the more delicate of its two sides."""
        first = np.zeros((4, 4), dtype=np.uint16)
        first[1, :] = 100
        second = np.zeros((4, 4), dtype=np.uint16)
        second[1, :] = 40
        second[2, :] = 70

        class Loaded:
            class tree:
                @staticmethod
                def getroot():
                    return object()

        with patch.object(
            rhd, "discontinuous_edge_extents",
            side_effect=[(first, {}), (second, {})],
        ):
            seams = rhd.material_mesh_seam_extents(
                Loaded(), ("a-material", "b-material"), (4, 4)
            )
        self.assertEqual(int(seams[1, 0]), 40)
        self.assertEqual(int(seams[2, 0]), 70)


if __name__ == "__main__":
    unittest.main()


class SwallowedDonationTests(unittest.TestCase):
    """A sibling's coarse box must not replace marks this layer resolved apart.

    The ardente door card: the roughness map reports the window-switch bezel as
    one component, 128x96 at the colour atlas's scale, over two 36x31 switch
    icons the colour layer had separated.  Unioned in, both turned over as one
    block -- more of the panel moved than anything had detected.
    """

    SWITCHES: ClassVar = [(3832, 2501, 36, 31), (3892, 2500, 36, 31)]

    def test_a_box_over_two_resolved_marks_is_refused(self) -> None:
        self.assertTrue(
            rhd._swallows_resolved_regions((3814, 2494, 128, 96), self.SWITCHES)
        )

    def test_a_box_over_one_is_kept(self) -> None:
        """The ledger's actual purpose: a mark seen faintly here, whole there."""
        self.assertFalse(
            rhd._swallows_resolved_regions((3830, 2388, 100, 45),
                                           [(3838, 2414, 17, 19)])
        )

    def test_a_box_over_none_is_kept(self) -> None:
        self.assertFalse(
            rhd._swallows_resolved_regions((10, 10, 40, 40), self.SWITCHES)
        )

    def test_a_box_merely_touching_two_marks_is_kept(self) -> None:
        """Refusal is about covering them, not about brushing their edges."""
        self.assertFalse(
            rhd._swallows_resolved_regions((3860, 2510, 60, 8), self.SWITCHES)
        )

    def test_a_donation_that_swallows_two_marks_never_reaches_the_plan(self) -> None:
        ledger = rhd.SharedLayerRegionLedger()
        ledger.record(
            "vehicles/x/interior_r.data.dds", (4096, 4096),
            [(3814, 2494, 128, 96)], [None],
        )
        regions, _rotations, sources, _added, _masks = ledger.combined(
            "vehicles/x/interior_b.color.dds", (4096, 4096),
            list(self.SWITCHES), [None, None],
        )
        self.assertEqual(sorted(regions), sorted(self.SWITCHES))
        self.assertEqual(sources, ())

    def test_a_donation_over_one_mark_still_widens_it(self) -> None:
        ledger = rhd.SharedLayerRegionLedger()
        ledger.record(
            "vehicles/x/interior_nm.normal.dds", (4096, 4096),
            [(3830, 2388, 100, 45)], [None],
        )
        regions, _rotations, sources, _added, _masks = ledger.combined(
            "vehicles/x/interior_b.color.dds", (4096, 4096),
            [(3838, 2414, 17, 19)], [None],
        )
        self.assertEqual(regions, [(3830, 2388, 100, 45)])
        self.assertEqual(len(sources), 1)
