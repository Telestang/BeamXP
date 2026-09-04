"""The labelled-region export: the artefact a filter gets fitted to.

Tested without a detector and without Tk, which is the point of keeping the
data model in its own module -- the export has to survive being read back a
week later, and that is a property of the file rather than of the window that
happened to write it.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from glyph_lab.labelling import (
    CROP_DIRECTORY,
    FALSE_POSITIVE,
    IGNORE,
    MANIFEST_NAME,
    TRUE_POSITIVE,
    LabelledRegion,
    crop_window,
    export_session,
    fittable,
    load_session,
    normalise_edges,
    placed_mask,
    region_statistics,
    write_region_crops,
)


def entry(label: str = TRUE_POSITIVE, **overrides) -> LabelledRegion:
    values = {
        "texture": "etkc_interior_alcantara", "stage": "2 the sparse field",
        "index": 0, "label": label, "bounds": (10, 20, 6, 4), "texels": 18,
        "fill": 0.75, "aspect": 1.5, "domain": 3,
    }
    values.update(overrides)
    return LabelledRegion(**values)


class RegionStatisticsTests(unittest.TestCase):
    """The numbers a candidate filter is a rule over."""

    def test_a_solid_box_fills_its_bounds(self) -> None:
        texels, fill, aspect = region_statistics((0, 0, 4, 4), np.ones((4, 4), bool))
        self.assertEqual(texels, 16)
        self.assertAlmostEqual(fill, 1.0)
        self.assertAlmostEqual(aspect, 1.0)

    def test_a_ring_reports_the_hole_it_leaves(self) -> None:
        mask = np.ones((5, 5), bool)
        mask[1:4, 1:4] = False
        texels, fill, _aspect = region_statistics((0, 0, 5, 5), mask)
        self.assertEqual(texels, 16)
        self.assertAlmostEqual(fill, 16 / 25)

    def test_aspect_is_the_long_side_over_the_short(self) -> None:
        """Orientation-free: a rule about thinness should not care which way."""
        wide = region_statistics((0, 0, 30, 3), np.ones((3, 30), bool))[2]
        tall = region_statistics((0, 0, 3, 30), np.ones((30, 3), bool))[2]
        self.assertAlmostEqual(wide, 10.0)
        self.assertAlmostEqual(wide, tall)

    def test_a_degenerate_box_does_not_divide_by_zero(self) -> None:
        texels, fill, aspect = region_statistics((0, 0, 0, 0), np.zeros((0, 0), bool))
        self.assertEqual(texels, 0)
        self.assertEqual(fill, 0.0)
        self.assertEqual(aspect, 0.0)


class CropWindowTests(unittest.TestCase):
    """Context around a mark, clipped to the atlas it came from."""

    def test_the_window_grows_by_the_context(self) -> None:
        self.assertEqual(crop_window((20, 30, 4, 6), (100, 100), 5), (15, 25, 14, 16))

    def test_it_clips_at_the_atlas_edge(self) -> None:
        self.assertEqual(crop_window((1, 1, 4, 4), (100, 100), 5), (0, 0, 10, 10))

    def test_it_clips_at_the_far_edge_too(self) -> None:
        left, top, width, height = crop_window((94, 94, 4, 4), (100, 100), 5)
        self.assertEqual((left, top), (89, 89))
        self.assertEqual((left + width, top + height), (100, 100))


class PlacedMaskTests(unittest.TestCase):
    """The mask has to land where the mark is, or the crops lie."""

    def test_the_mask_sits_at_its_offset_in_the_window(self) -> None:
        mask = np.ones((2, 3), bool)
        window = crop_window((10, 10, 3, 2), (40, 40), 4)
        drawn = placed_mask((10, 10, 3, 2), mask, window)
        left, top, _w, _h = window
        self.assertTrue(drawn[10 - top: 12 - top, 10 - left: 13 - left].all())
        self.assertEqual(int(drawn.sum()), 6)

    def test_a_region_overhanging_the_atlas_is_clipped_not_wrapped(self) -> None:
        """A padded region can reach past the edge; it must not reappear."""
        mask = np.ones((4, 4), bool)
        window = crop_window((38, 38, 4, 4), (40, 40), 0)
        drawn = placed_mask((38, 38, 4, 4), mask, window)
        self.assertEqual(drawn.shape, (2, 2))
        self.assertEqual(int(drawn.sum()), 4)


class NormaliseEdgesTests(unittest.TestCase):
    def test_the_response_is_scaled_by_its_own_peak(self) -> None:
        grey = normalise_edges(np.array([[0.0, 0.5], [1.0, 0.25]]))
        self.assertEqual(int(grey.max()), 255)
        self.assertEqual(int(grey.min()), 0)

    def test_a_flat_response_comes_back_black_rather_than_dividing(self) -> None:
        grey = normalise_edges(np.zeros((4, 4)))
        self.assertEqual(int(grey.max()), 0)


class LabelledRegionTests(unittest.TestCase):
    def test_an_unknown_verdict_is_refused_at_construction(self) -> None:
        """A typo must not reach the file, where it becomes a third class."""
        with self.assertRaises(ValueError):
            entry(label="probably")

    def test_the_three_verdicts_are_accepted(self) -> None:
        for label in (TRUE_POSITIVE, FALSE_POSITIVE, IGNORE):
            self.assertEqual(entry(label=label).label, label)


class ExportSessionTests(unittest.TestCase):
    """The manifest is the deliverable; it has to round-trip intact."""

    def test_the_parameters_travel_with_the_labels(self) -> None:
        """A verdict means nothing without the settings that produced it."""
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw)
            export_session(out, {"min_area_px": 12, "stage": "2 the sparse field"},
                           [entry()])
            parameters, entries = load_session(out)
            self.assertEqual(parameters["min_area_px"], 12)
            self.assertEqual(parameters["stage"], "2 the sparse field")
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].bounds, (10, 20, 6, 4))
            self.assertEqual(entries[0].texels, 18)

    def test_ignored_regions_are_written_and_then_dropped_by_the_reader(self) -> None:
        """Written so "looked at and passed over" is not lost; dropped for fitting."""
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw)
            export_session(out, {}, [
                entry(TRUE_POSITIVE, index=0),
                entry(IGNORE, index=1),
                entry(FALSE_POSITIVE, index=2),
            ])
            _parameters, entries = load_session(out)
            self.assertEqual(len(entries), 3)
            self.assertEqual([e.label for e in fittable(entries)],
                             [TRUE_POSITIVE, FALSE_POSITIVE])

    def test_the_manifest_carries_a_tally(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw)
            export_session(out, {}, [entry(TRUE_POSITIVE, index=0),
                                     entry(TRUE_POSITIVE, index=1),
                                     entry(IGNORE, index=2)])
            payload = json.loads((out / MANIFEST_NAME).read_text())
            self.assertEqual(payload["counts"][TRUE_POSITIVE], 2)
            self.assertEqual(payload["counts"][IGNORE], 1)
            self.assertEqual(payload["counts"][FALSE_POSITIVE], 0)


class WriteRegionCropsTests(unittest.TestCase):
    """Three pictures of one region, pixel-aligned so they can be stacked."""

    @staticmethod
    def atlas() -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(7)
        colour = rng.integers(0, 255, (40, 40, 3), dtype=np.uint8)
        edges = rng.random((40, 40)).astype(np.float32)
        return colour, edges

    def test_the_three_crops_share_one_window(self) -> None:
        from PIL import Image

        colour, edges = self.atlas()
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw)
            crops = write_region_crops(
                out, "tex", 3, (12, 12, 6, 6), np.ones((6, 6), bool),
                colour, edges, context_px=4,
            )
            sizes = {
                kind: Image.open(out / name).size for kind, name in crops.items()
            }
            self.assertEqual(len(sizes), 3)
            self.assertEqual(len(set(sizes.values())), 1, sizes)

    def test_the_mask_marks_the_region_and_not_its_context(self) -> None:
        from PIL import Image

        colour, edges = self.atlas()
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw)
            crops = write_region_crops(
                out, "tex", 0, (12, 12, 6, 6), np.ones((6, 6), bool),
                colour, edges, context_px=4,
            )
            mask = np.asarray(Image.open(out / crops["mask"]))
            self.assertEqual(int((mask > 0).sum()), 36)

    def test_the_names_are_relative_so_the_directory_can_move(self) -> None:
        colour, edges = self.atlas()
        with tempfile.TemporaryDirectory() as raw:
            crops = write_region_crops(
                Path(raw), "tex", 5, (12, 12, 4, 4), np.ones((4, 4), bool),
                colour, edges,
            )
            for name in crops.values():
                self.assertFalse(Path(name).is_absolute(), name)
                self.assertTrue(name.startswith(f"{CROP_DIRECTORY}/"), name)

    def test_a_texture_with_no_edge_map_still_exports(self) -> None:
        """Not every source has one, and that is not a reason to lose the label."""
        colour, _edges = self.atlas()
        with tempfile.TemporaryDirectory() as raw:
            crops = write_region_crops(
                Path(raw), "tex", 0, (12, 12, 4, 4), np.ones((4, 4), bool),
                colour, None,
            )
            self.assertIn("colour", crops)
            self.assertIn("mask", crops)
            self.assertNotIn("edges", crops)


if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    unittest.main()
