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


class MarkGateTests(unittest.TestCase):
    """Does a grouped region read as ink, or as a piece of the panel?

    Fitted to two labelling sessions; these pin the three signals it reads
    and the shapes of the cases each was added for.
    """

    @staticmethod
    def gate(region, colour, contrast=12.0, tone=45.0):
        from glyph_lab.region_grouping import looks_like_a_mark

        return looks_like_a_mark(region, colour, (0, 0), contrast, tone)

    @staticmethod
    def region(mask, x=8, y=8):
        from glyph_lab.regions import Region

        return Region(bounds=(x, y, mask.shape[1], mask.shape[0]),
                      mask=mask, domain=1)

    @staticmethod
    def panel(value=90):
        return np.full((40, 40, 3), value, dtype=np.uint8)

    def test_ink_on_a_panel_is_a_mark(self) -> None:
        colour = self.panel()
        colour[8:20, 8:20] = 240                 # a bright block on dark ground
        self.assertTrue(self.gate(self.region(np.ones((12, 12), bool)), colour))

    def test_a_piece_of_the_panel_is_not(self) -> None:
        """Same texels as its surroundings: no step, one tone, and so no mark."""
        colour = self.panel()
        self.assertFalse(self.gate(self.region(np.ones((12, 12), bool)), colour))

    def test_a_region_that_swallowed_its_background_is_kept(self) -> None:
        """The scintilla badge: the mask is the whole plaque, script and all.

        Its mean matches the ground outside it, so the boundary step reads as
        nothing -- all of the contrast is inside. Two tones in the interior is
        what keeps it, and 12 of 331 labelled marks depend on that clause.
        """
        # Sized so the plaque and its script average back to the ground
        # around them: 1/3 of the mask at 30 and 2/3 at 210 means 150, which
        # is the surround.  That is the badge's actual signature -- not that
        # it is low contrast, but that its contrast is entirely internal.
        colour = self.panel(150)
        colour[8:20, 8:20] = 210                 # the plaque
        colour[8:12, 8:20] = 30                  # the script across its top third
        region = self.region(np.ones((12, 12), bool))
        inside = colour[8:20, 8:20, 0].astype(float).mean()
        self.assertLess(
            abs(inside - 150.0), 12.0,
            "the fixture must not step against its surround",
        )
        self.assertTrue(self.gate(region, colour))
        # ...and it is the tone gap doing it, not the step: put the tone bar
        # out of reach and nothing is left to save it.
        self.assertFalse(self.gate(region, colour, tone=500.0))

    def test_a_diagonal_sliver_still_reads_as_ink(self) -> None:
        """Shape is not this gate's business.

        A trim arc is high-contrast ink by every appearance measure, and it
        is: the objection to it is that flipping it would put it somewhere
        it never was, which is a question about the *flip* and is answered by
        `GroupingConfig.min_lone_mirror`.  Asking this gate to drop it would
        also drop a row of glyphs that happens to follow a curve.
        """
        colour = self.panel()
        mask = np.zeros((24, 24), dtype=bool)
        for i in range(24):
            mask[i, max(i - 1, 0): i + 2] = True
        colour[8:32, 8:32][mask] = 240
        self.assertTrue(self.gate(self.region(mask), colour))

    def test_zero_on_either_appearance_bar_switches_the_pair_off(self) -> None:
        """Not one clause each: the two are ORed, so a zero bar passes both.

        Worth pinning because the natural reading -- "0 disables that half" --
        is wrong, and a config comment said so until this test was written.
        """
        colour = self.panel()
        region = self.region(np.ones((12, 12), bool))
        self.assertFalse(self.gate(region, colour))
        self.assertTrue(self.gate(region, colour, contrast=0.0))
        self.assertTrue(self.gate(region, colour, tone=0.0))

    def test_the_tone_gap_ignores_a_smooth_ramp(self) -> None:
        """A gradient splits into two halves as cleanly as a glyph does.

        Which is why the gate reads the distance between the tones and not
        how much variance the split explains -- scored the latter way a ramp
        was indistinguishable from ink.
        """
        from glyph_lab.region_grouping import interior_tone_gap

        ramp = np.linspace(100, 130, 400, dtype=np.float32)
        two_tone = np.concatenate([np.full(200, 40.0, np.float32),
                                   np.full(200, 210.0, np.float32)])
        self.assertLess(interior_tone_gap(ramp), 20.0)
        self.assertGreater(interior_tone_gap(two_tone), 150.0)


class MergeSessionsTests(unittest.TestCase):
    """Two passes over one region set, combined without losing either."""

    @staticmethod
    def session(directory, rows):
        export_session(directory, {"stage": "8 clear material transitions"}, [
            entry(label, index=index, texture=texture)
            for texture, index, label in rows
        ])

    def test_a_later_session_wins_on_the_regions_it_covers(self) -> None:
        """Re-labelling a range means putting the newer session last."""
        from glyph_lab.labelling import merge_sessions

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self.session(root / "first", [
                ("etkc", 0, TRUE_POSITIVE), ("etkc", 1, TRUE_POSITIVE),
            ])
            self.session(root / "second", [("etkc", 1, FALSE_POSITIVE)])
            _params, merged = merge_sessions(
                [root / "first", root / "second"], root / "all",
            )
            verdicts = {e.index: e.label for e in merged}
            self.assertEqual(verdicts, {0: TRUE_POSITIVE, 1: FALSE_POSITIVE})

    def test_the_halves_of_a_split_session_add_up(self) -> None:
        from glyph_lab.labelling import merge_sessions

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self.session(root / "a", [("etkc", i, TRUE_POSITIVE) for i in range(100)])
            self.session(root / "b", [("etkc", i, FALSE_POSITIVE)
                                      for i in range(100, 160)])
            _params, merged = merge_sessions([root / "a", root / "b"], root / "all")
            self.assertEqual(len(merged), 160)
            self.assertEqual(sum(1 for e in merged if e.label == TRUE_POSITIVE), 100)
            self.assertEqual(sum(1 for e in merged if e.label == FALSE_POSITIVE), 60)

    def test_sessions_at_different_stages_do_not_overwrite(self) -> None:
        """The same index means a different region at a different stage."""
        from glyph_lab.labelling import merge_sessions

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            export_session(root / "early", {}, [entry(TRUE_POSITIVE, index=0)])
            export_session(root / "late", {}, [
                entry(FALSE_POSITIVE, index=0, stage="13 final padding"),
            ])
            _params, merged = merge_sessions([root / "early", root / "late"],
                                             root / "all")
            self.assertEqual(len(merged), 2)
            self.assertEqual({e.stage for e in merged},
                             {"2 the sparse field", "13 final padding"})

    def test_the_merge_records_where_it_came_from(self) -> None:
        from glyph_lab.labelling import load_session, merge_sessions

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self.session(root / "first", [("etkc", 0, TRUE_POSITIVE)])
            merge_sessions([root / "first"], root / "all")
            params, _entries = load_session(root / "all")
            self.assertIn("merged_from", params)
            self.assertEqual(len(params["merged_from"]), 1)


class LabelWindowPagingTests(unittest.TestCase):
    """A session of hundreds of regions has to stay reachable.

    2026-09-04: it did not.  Tk addresses canvas coordinates as signed
    16-bit, so a body frame taller than 32,767 px silently cut off every row
    below that line -- on a 557-region session the frame reached 55,000 px
    and roughly 220 rows could not be scrolled to.  Nothing said so: the
    tally counted them, and a bulk verdict had labelled them.
    """

    @staticmethod
    def build(count: int):
        import tkinter as tk

        from glyph_lab.label_window import LabelSubject, LabelWindow
        from glyph_lab.regions import Region

        rng = np.random.default_rng(3)
        colour = rng.integers(0, 255, (1200, 1200, 3), dtype=np.uint8)
        regions = tuple(
            Region(bounds=(10 + (i % 12) * 90, 10 + (i // 12) % 12 * 90, 70, 70),
                   mask=np.ones((70, 70), dtype=bool), domain=1)
            for i in range(count)
        )
        root = tk.Tk()
        root.withdraw()
        window = LabelWindow(
            root, [LabelSubject("t", "s", colour, None, regions)], {},
        )
        return root, window

    @staticmethod
    def settle(window):
        while window._pending:
            window._build_batch()
        window.update_idletasks()

    def test_every_row_stays_inside_tk_s_coordinate_limit(self) -> None:
        root, window = self.build(557)
        try:
            self.settle(window)
            self.assertLess(window._body.winfo_reqheight(), 32767)
            for child in window._body.pack_slaves():
                self.assertLessEqual(child.winfo_y(), 32767)
        finally:
            window.destroy(); root.destroy()

    def test_every_region_is_reachable_across_the_pages(self) -> None:
        root, window = self.build(557)
        try:
            seen = 0
            for page in range(window.pages):
                window.show_page(page)
                self.settle(window)
                seen += len(window._body.pack_slaves())
            self.assertEqual(seen, 557)
        finally:
            window.destroy(); root.destroy()

    def test_a_verdict_survives_leaving_its_page_and_coming_back(self) -> None:
        root, window = self.build(150)
        try:
            self.settle(window)
            window._set_page(TRUE_POSITIVE)
            self.settle(window)
            window.next_page(); self.settle(window)
            window.previous_page(); self.settle(window)
            self.assertEqual(window._tally_all()[TRUE_POSITIVE], 60)
            self.assertEqual(window._choices[("t", 0)].get(), TRUE_POSITIVE)
        finally:
            window.destroy(); root.destroy()

    def test_a_bulk_verdict_reaches_only_the_page_in_front_of_you(self) -> None:
        """The failure that produced a corpus of unreviewed 'true'."""
        root, window = self.build(150)
        try:
            self.settle(window)
            window._set_page(TRUE_POSITIVE)
            self.settle(window)
            counts = window._tally_all()
            self.assertEqual(counts[TRUE_POSITIVE], 60)
            self.assertEqual(counts[""], 90, "it labelled regions off the page")
        finally:
            window.destroy(); root.destroy()
