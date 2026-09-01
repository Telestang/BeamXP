"""The Glyph Lab window, built and torn down for real.

Importing the module proves nothing about it: the first version shipped with a
``_options`` method that shadowed ``tkinter.Widget._options``, which ``configure``
calls with arguments, so the window died on its second line of ``__init__``.
Nothing short of constructing a Tk would have caught it.

These build against a tiny synthetic corpus so the run is instant, skip
themselves where no display exists, and always destroy the window.
"""

from __future__ import annotations

import gc
import json
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from glyph_lab.corpus import MANIFEST_NAME, UvDomainGeometry, write_texture


def tiny_corpus(directory: Path, count: int = 3) -> Path:
    """A corpus small enough that detection over it is instant."""
    records = []
    for index in range(count):
        size = 48 + index * 16
        colour = np.full((size, size, 3), 60, dtype=np.uint8)
        colour[12:20, 8:size - 8] = 220        # something to fire on
        domains = np.zeros((size, size), dtype=np.uint16)
        domains[4:size - 4, 4:size - 4] = 1
        normal = None
        if index == 0:
            normal = np.full((size, size, 3), (128, 128, 255), dtype=np.uint8)
            normal[12:20, 8:size - 8, 0] = 220
        records.append(
            write_texture(directory, f"tiny_{index}", colour, domains, normal)
        )
    (directory / MANIFEST_NAME).write_text(json.dumps({"textures": records}))
    return directory


class GlyphLabWindowTests(unittest.TestCase):
    """Each test owns its window, so a failure cannot strand a Tk."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            import tkinter as tk
        except ImportError as exc:                       # pragma: no cover
            raise unittest.SkipTest(f"no tkinter: {exc}") from exc
        try:
            probe = tk.Tk()
        except tk.TclError as exc:                       # pragma: no cover
            raise unittest.SkipTest(f"no display: {exc}") from exc
        probe.destroy()
        # tkinter keeps the first Tk it ever saw as the process-wide default
        # root and does not forget it on destroy, so a probe left in place
        # hands every later window a dead interpreter to hang widgets off.
        tk._default_root = None
        cls._temp = tempfile.TemporaryDirectory(prefix="glyph_lab_corpus_")
        cls.corpus = tiny_corpus(Path(cls._temp.name))
        # The app remembers its panel to disk on every teardown.  Left pointed
        # at the real file, a test run would overwrite whatever the user had
        # tuned -- and read it back in, so the assertions below would depend on
        # it.  Redirect it at the throwaway corpus directory.
        from glyph_lab import app as app_module

        cls._settings = app_module.SETTINGS_PATH
        app_module.SETTINGS_PATH = Path(cls._temp.name) / "params.json"

    @classmethod
    def tearDownClass(cls) -> None:
        from glyph_lab import app as app_module

        app_module.SETTINGS_PATH = cls._settings
        cls._temp.cleanup()

    def setUp(self) -> None:
        # Per test, not per class: the app writes the panel out on teardown, so
        # one test's settings would otherwise be the next one's starting point
        # -- which is exactly how a test asserting a default started failing
        # after another had saved over it.
        from glyph_lab import app as app_module

        app_module.SETTINGS_PATH.unlink(missing_ok=True)

    @staticmethod
    def new_window(corpus):
        """Build a window, retrying a transient Tcl library-load failure.

        2026-08-27: every ``Tk()`` gets its own interpreter, and every fresh
        interpreter re-reads ``init.tcl`` and ``tk.tcl`` from disk.  Under the
        disk load of the full suite one of those reads intermittently fails on
        Windows with ``couldn't read file ".../init.tcl": No error`` -- an
        errno of zero, which is a transient share violation rather than a
        missing file, and the same path reads back fine moments later in the
        same process.  It has nothing to do with the app or with what ran
        before: it hit whichever test happened to be building a window, about
        one run in three, never when this file ran on its own, and it went away
        entirely when something else had just read the file and warmed the
        cache.  Retrying is the mitigation; a window that genuinely will not
        build still fails, three times over.
        """
        import tkinter as tk

        from glyph_lab.app import GlyphLab

        last = None
        for attempt in range(3):
            try:
                return GlyphLab(corpus)
            except tk.TclError as exc:
                if "tcl" not in str(exc).lower():
                    raise               # a real Tk problem, not the flaky read
                last = exc
                time.sleep(0.1 * (attempt + 1))
        raise AssertionError(f"Tk would not start in three attempts: {last}")

    @staticmethod
    def whole_pipeline(app) -> None:
        """Run every stage, not just up to the seed.

        `stop_after_stage` ends the pipeline after "seed over density" while
        the front end is being rebuilt, so a test about a later stage has to
        ask for it -- the same way a test about the rim band asks for the band.
        """
        app.config_vars["stop_after_stage"].set("0")
        app.run_detection()

    def build(self):
        app = self.new_window(self.corpus)
        app.withdraw()          # keep it off the screen; it still lays out
        # LIFO, so these run as settle, destroy, collect.
        #
        # 2026-08-27: the collect is the one that is not obvious, and it is
        # what fixed a roughly one-in-three abort of the whole suite.
        # ``destroy()`` does not deallocate the interpreter -- the widget tree
        # is full of reference cycles, so ``Tkapp_Dealloc`` waits for the
        # cyclic collector, which runs on whichever thread next happens to
        # trigger it.  Reached from a worker, ``Tcl_DeleteInterp`` aborts the
        # process with ``Tcl_AsyncDelete: async handler deleted by the wrong
        # thread``, and it lands on some later, unrelated test.  Collecting
        # here pins that deallocation to the main thread while the window that
        # owns it is the one being torn down.  Measured: 80 build/destroy
        # cycles abort within 20 without it, and run clean with it.
        self.addCleanup(gc.collect)
        self.addCleanup(app.destroy)
        self.addCleanup(self.settle, app)
        app.update()
        # 2026-08-27: settle here, not in each test that happens to need it.
        # One `update()` used to be enough for the opening detection to land on
        # a corpus this small, which made every test that read `app.results`
        # depend on a race -- and they duly started failing the moment the
        # panel gained work to do at startup.
        self.settle(app)
        return app

    @staticmethod
    def settle(app, timeout: float = 10.0) -> None:
        """Wait out detection and raster workers before the window goes away.

        Releasing the Tk variables and the poll timer used to happen here, by
        name.  It belongs in ``GlyphLab.destroy``, which now does it by type --
        this list went stale every time the app gained a checkbox, and a name
        missed off it surfaced as a crash in some later, unrelated test.
        """
        deadline = time.monotonic() + timeout
        while (app._busy or app._render_busy) and time.monotonic() < deadline:
            app.update()
            app.after(10)
        app.update()
        gc.collect()

    def test_the_window_builds(self) -> None:
        """`configure` alone was enough to break this once."""
        app = self.build()
        self.assertEqual(len(app.textures), 3)

    def test_the_cursor_marks_background_work(self) -> None:
        app = self.build()
        app._busy = True
        app._refresh_busy_cursor()
        self.assertEqual(app.canvas.cget("cursor"), "wait")
        app._busy = False
        app._refresh_busy_cursor()
        self.assertEqual(app.canvas.cget("cursor"), "")

    def test_normal_source_uses_only_textures_with_normal_maps(self) -> None:
        from glyph_lab.app import SOURCE_NORMAL_EDGE_STEEPNESS

        app = self.build()
        app.source_var.set(SOURCE_NORMAL_EDGE_STEEPNESS)
        app._on_source_change()
        self.settle(app)
        self.assertEqual([texture.name for texture in app.textures], ["tiny_0"])

    def test_source_switch_shows_only_its_detector_parameters(self) -> None:
        from glyph_lab.app import (
            SOURCE_COLOUR,
            SOURCE_NORMAL_EDGE_STEEPNESS,
        )

        app = self.build()
        self.assertEqual(
            list(app._source_box.cget("values")),
            [SOURCE_COLOUR, SOURCE_NORMAL_EDGE_STEEPNESS],
        )
        self.assertTrue(any(
            row.winfo_ismapped() for row in app._config_rows.values()
        ))
        self.assertFalse(any(
            row.winfo_ismapped() for row in app._normal_config_rows.values()
        ))

        app.source_var.set(SOURCE_NORMAL_EDGE_STEEPNESS)
        app._on_source_change()
        self.settle(app)
        self.assertFalse(any(
            row.winfo_ismapped() for row in app._config_rows.values()
        ))
        self.assertTrue(any(
            row.winfo_ismapped() for row in app._normal_config_rows.values()
        ))


    def test_normal_edge_source_measures_normal_direction_changes(self) -> None:
        from glyph_lab.app import SOURCE_NORMAL_EDGE_STEEPNESS

        app = self.build()
        app.source_var.set(SOURCE_NORMAL_EDGE_STEEPNESS)
        app._on_source_change()
        self.settle(app)
        self.assertEqual([texture.name for texture in app.textures], ["tiny_0"])
        image = app.textures[0].colour
        self.assertGreater(int(image.max()), 0, "the normal change made no edge")
        np.testing.assert_array_equal(image[:, :, 0], image[:, :, 1])
        np.testing.assert_array_equal(image[:, :, 1], image[:, :, 2])

    def test_normal_parameters_are_separate_from_colour_and_their_cache(self) -> None:
        from glyph_lab import app as app_module

        app = self.build()
        colour_kernel = app.current_config().contrast.kernel_px
        app.normal_config_vars["slope_threshold"].set("19")
        app._on_normal_config_change()
        self.assertEqual(app.current_config().contrast.kernel_px, colour_kernel)
        self.assertEqual(app.current_normal_config().slope_threshold, 19)
        app.remember()
        saved = app_module.load_settings()
        self.assertEqual(saved["normal_slope_threshold"], 19)
        self.assertEqual(saved["kernel_px"], colour_kernel)

    def test_normal_source_has_its_own_stage_picker(self) -> None:
        from glyph_lab.app import NO_STAGE, SOURCE_NORMAL_EDGE_STEEPNESS
        from glyph_lab.normal_detect import NORMAL_STAGES

        app = self.build()
        app.source_var.set(SOURCE_NORMAL_EDGE_STEEPNESS)
        app._on_source_change()
        self.settle(app)
        self.assertEqual(
            list(app._stage_box.cget("values")), [NO_STAGE, *NORMAL_STAGES],
        )

    def test_the_contact_sheet_draws_a_tile_per_texture(self) -> None:
        app = self.build()
        app.canvas.configure(width=800, height=600)
        app.update()
        app.redraw()
        self.assertEqual(len(app._tiles), len(app.textures))

    def test_opening_a_tile_and_escaping_back(self) -> None:
        app = self.build()
        app.canvas.configure(width=800, height=600)
        app.update()
        app.redraw()
        name = app._tiles[0][4]

        app._open(name)
        self.assertEqual(app.zoomed, name)
        self.assertGreater(app.view_scale, 0.0)

        app._show_sheet()
        self.assertIsNone(app.zoomed)
        self.assertEqual(len(app._tiles), len(app.textures))

    def test_every_way_back_to_the_sheet_is_wired_up(self) -> None:
        """Each route is bound, and what it is bound to returns to the sheet.

        Deliberately not driven through ``event_generate``: a withdrawn
        toplevel takes no button events at all, and a keypress is routed by
        focus, which several Tk instances in one test process do not agree
        about -- the same assertion passes alone and fails in the suite.
        Whether Tk delivers an event is Tk's business; whether this app binds
        one and does the right thing with it is not.
        """
        app = self.build()
        app.canvas.configure(width=800, height=600)
        app.update()
        app.redraw()
        name = app._tiles[0][4]

        for description, widget, sequence, handler in (
            ("right-click", app.canvas, "<Button-3>", app._on_right_click),
            ("escape", app, "<Escape>", lambda _event: app._show_sheet()),
        ):
            self.assertTrue(
                widget.bind(sequence), f"nothing bound to {sequence} ({description})"
            )
            app._open(name)
            self.assertEqual(app.zoomed, name)
            handler(None)
            self.assertIsNone(app.zoomed, f"{description} did not go back")

        self.assertTrue(
            app.canvas.bind("<Double-Button-1>"), "nothing bound to Double-Button-1"
        )

    def test_double_click_refits_a_wandering_view(self) -> None:
        app = self.build()
        app.canvas.configure(width=800, height=600)
        app.update()
        app.redraw()
        app._open(app._tiles[0][4])
        fitted = app.view_scale
        app.view_scale *= 8
        app.view_origin = (500.0, 500.0)
        app._on_double_click(None)
        self.assertAlmostEqual(app.view_scale, fitted)
        self.assertEqual(app.view_origin, (0.0, 0.0))

    def test_the_view_options_read_the_checkboxes(self) -> None:
        app = self.build()
        self.assertTrue(app.view_options().show_texels)
        self.assertFalse(app.view_options().show_uv_edges)
        self.assertFalse(app.view_options().show_mesh_seams)
        self.assertFalse(app.view_options().show_uv_geometry)
        app.show_texels.set(False)
        app.dim_outside.set(False)
        app.show_uv_edges.set(True)
        app.show_mesh_seams.set(True)
        app.show_uv_geometry.set(True)
        app.geometry_vars["seam_parallelogram_opposite_edge_dot"].set("0.97")
        app.geometry_vars["seam_symmetry_min_score"].set("0.98")
        app.geometry_vars["parallelogram_min_edge_px"].set("3.5")
        options = app.view_options()
        self.assertFalse(options.show_texels)
        self.assertFalse(options.dim_outside_uv)
        self.assertTrue(options.show_uv_edges)
        self.assertTrue(options.show_mesh_seams)
        self.assertTrue(options.show_uv_geometry)
        self.assertEqual(app.geometry_config().seam_parallelogram_opposite_edge_dot, 0.97)
        self.assertEqual(app.geometry_config().seam_symmetry_min_score, 0.98)
        self.assertEqual(app.geometry_config().parallelogram_min_edge_px, 3.5)
        self.assertTrue(options.show_regions)

    def test_uv_boundaries_are_derived_once_per_texture(self) -> None:
        """They come from the DAE, so a parameter change must not redo them."""
        app = self.build()
        texture = app.textures[0]
        first = app.boundaries_for(texture)
        self.assertIs(app.boundaries_for(texture), first)
        self.assertEqual(first.shape, texture.shape)

    def test_symmetry_threshold_filters_scored_axes_without_rebuilding(self) -> None:
        app = self.build()
        texture = replace(app.textures[0], uv_geometry=(
            UvDomainGeometry(1, "symmetric", axis=((8.0, 4.0), (8.0, 44.0)),
                             symmetry_score=0.97),
            UvDomainGeometry(1, "symmetric", axis=((16.0, 4.0), (16.0, 44.0)),
                             symmetry_score=0.99),
        ))
        app.geometry_vars["seam_symmetry_min_score"].set("0.98")

        self.assertEqual(len(app.geometry_for(texture)), 1)
        self.assertEqual(app.geometry_for(texture)[0].symmetry_score, 0.99)

    def test_destroy_leaves_nothing_pointed_at_the_interpreter(self) -> None:
        """The cause of a one-in-three failure of a random test in the suite.

        A pending ``after`` fires into the dead interpreter, and a ``Variable``
        or ``PhotoImage`` collected later calls into it from whichever thread
        CPython happens to be on -- which aborts with ``Tcl_AsyncDelete``
        instead of raising, taking the next window's construction down with it.
        Neither shows up in the window that caused it; both show up in the next
        one, which is why this is asserted on state rather than on output.
        """
        import tkinter as tk

        app = self.new_window(self.corpus)
        app.withdraw()
        app.update()
        self.settle(app)
        self.assertIsNotNone(app._poll_job, "the poll timer was never armed")
        variables = [n for n, v in vars(app).items() if isinstance(v, tk.Variable)]
        self.assertTrue(variables, "no Tk variables to release")

        app.destroy()

        self.assertIsNone(app._poll_job, "the poll timer outlived the window")
        self.assertTrue(app._closing)
        self.assertIsNone(app._photo, "a PhotoImage outlived the window")
        self.assertFalse(
            [n for n, v in vars(app).items() if isinstance(v, tk.Variable)],
            "a Tk variable outlived the window",
        )
        self.assertEqual(app.config_vars, {})
        gc.collect()            # as `build` does; see the note there

    def test_the_panel_is_remembered_and_read_back(self) -> None:
        """A session of tuning has to survive closing the window."""
        from glyph_lab import app as app_module

        app = self.build()
        app.config_vars["min_response"].set("13.5")
        app.show_uv_edges.set(True)
        app.show_uv_geometry.set(True)
        app.geometry_vars["seam_corner_tolerance_px"].set("0.75")
        app.geometry_vars["seam_symmetry_min_score"].set("0.98")
        app.remember()
        stored = app_module.load_settings()
        self.assertEqual(stored["min_response"], 13.5)
        self.assertIs(stored["show_uv_edges"], True)
        self.assertIs(stored["show_uv_geometry"], True)
        self.assertEqual(stored["seam_corner_tolerance_px"], 0.75)
        self.assertEqual(stored["seam_symmetry_min_score"], 0.98)

    def test_a_saved_value_that_no_longer_fits_is_dropped(self) -> None:
        """The parameter set changes most sessions; a stale file must not
        break the window it is read into."""
        from glyph_lab import app as app_module

        app_module.SETTINGS_PATH.write_text(json.dumps({
            "min_response": "not a number",     # unparseable
            "a_knob_that_was_removed": 7,       # gone from the config
            "kernel_px": 9,                     # good
        }))
        stored = app_module.load_settings()
        self.assertEqual(stored, {"kernel_px": 9})
        app = self.build()
        self.assertEqual(app.current_config().contrast.kernel_px, 9)

    def test_the_short_lived_solid_rim_gate_migrates_to_the_stitch_limit(self) -> None:
        from glyph_lab import app as app_module

        app_module.SETTINGS_PATH.write_text(json.dumps({
            "rim_artifact_max_span_px": 16,
            "rim_artifact_min_fill": 0.8,
        }))
        stored = app_module.load_settings()
        self.assertEqual(stored["rim_artifact_max_span_px"], 32)
        self.assertNotIn("rim_artifact_min_fill", stored)

    def test_the_status_line_lives_in_the_side_panel(self) -> None:
        """Packed on the root after an expanding canvas, it gets squeezed."""
        app = self.build()
        self.assertIsNot(app.status.master, app)
        self.assertIs(app.status.master, app.counts.master)

    def test_the_config_panel_round_trips_a_value(self) -> None:
        """A float and an int, since they are parsed by different branches."""
        app = self.build()
        app.config_vars["min_response"].set("31.5")
        app.config_vars["kernel_px"].set("7")
        app.config_vars["min_repeat_energy"].set("0.25")
        config = app.current_config()
        self.assertEqual(config.contrast.min_response, 31.5)
        self.assertEqual(config.contrast.kernel_px, 7)
        self.assertEqual(config.contrast.min_repeat_energy, 0.25)

    def test_a_bool_field_round_trips_as_a_bool(self) -> None:
        """A bool gets a checkbox, not an entry box.

        ``bool`` is a subclass of ``int``, so a parser testing ``int`` first
        sends it through ``int()`` and hands back 0 or 1.  The overlay panel is
        where the bools live now that the filters are all numeric, but the
        parsing is the same code path.
        """
        app = self.build()
        app.view_vars["show_mesh_seams"].set(False)
        self.assertIs(app.view_options().show_mesh_seams, False)
        app.view_vars["show_mesh_seams"].set(True)
        self.assertIs(app.view_options().show_mesh_seams, True)

    def test_the_controls_scroll_and_the_readouts_do_not(self) -> None:
        """The panel outgrew the window; only the controls were made to scroll.

        The readouts are the output of every change made above them, so they
        are pinned outside the scrolling canvas rather than dragged out of
        sight by it.
        """
        import tkinter as tk

        app = self.build()
        shell = app._panel_shell
        self.assertIsInstance(shell, tk.Canvas)

        def inside(widget) -> bool:
            while widget is not None:
                if widget is shell:
                    return True
                widget = getattr(widget, "master", None)
            return False

        self.assertTrue(shell.winfo_children(), "nothing was put in the shell")
        self.assertFalse(inside(app.counts), "the counts box scrolls away")
        self.assertFalse(inside(app.status), "the status line scrolls away")

        app.update_idletasks()
        # A canvas with no scrollregion silently refuses to scroll, so this has
        # to have followed the content rather than stayed at its initial zero.
        region = [int(float(v)) for v in str(shell.cget("scrollregion")).split()]
        self.assertEqual(len(region), 4, "no scrollregion was ever set")
        self.assertGreater(region[3] - region[1], 0)
        self.assertGreater(
            int(shell.cget("width")), 1,
            "the canvas kept its default width and would clip the entries",
        )

    def test_the_wheel_scrolls_the_controls_and_stops_when_they_fit(self) -> None:
        """Both branches of the handler's size test.

        Driven through the handler rather than through a real wheel event,
        with the pointer stood in for: a withdrawn window has nothing under the
        cursor, so ``winfo_containing`` answers None and the handler correctly
        declines before it ever reaches the part under test.
        """
        app = self.build()
        shell = app._panel_shell
        app.winfo_containing = lambda _x, _y: shell     # the pointer is on it
        app.update_idletasks()
        box = shell.bbox("all")
        self.assertIsNotNone(box)
        content = box[3] - box[1]

        class Turn:
            delta = -120
            x_root = y_root = 0

        # Shorter than its content: a turn should move it.
        shell.configure(height=max(content // 2, 1))
        app.update_idletasks()
        before = shell.yview()
        app._on_panel_wheel(Turn())
        app.update_idletasks()
        self.assertNotEqual(shell.yview(), before, "a full panel would not scroll")

        # Taller than its content: nothing to scroll, so nothing should move --
        # otherwise a turn leaves the controls hanging off the top of a blank
        # column with no way to tell that is what happened.  Reported rather
        # than configured: the shell is packed to fill, so pack owns its real
        # height and `configure(height=...)` is only a request it overrides.
        shell.yview_moveto(0.0)
        app.update_idletasks()
        settled = shell.yview()
        shell.winfo_height = lambda: content + 100
        app._on_panel_wheel(Turn())
        app.update_idletasks()
        self.assertEqual(shell.yview(), settled, "it scrolled a panel that fits")

    def test_the_wheel_ignores_a_native_combobox_popup(self) -> None:
        app = self.build()

        class Turn:
            delta = -120
            x_root = y_root = 0

        def native_popup(_x, _y):
            raise KeyError("popdown")

        app.winfo_containing = native_popup
        app._on_panel_wheel(Turn())

    def test_the_stage_picker_offers_every_stage(self) -> None:
        """A stage missing from the picker is a stage nobody can look at."""
        from glyph_lab.app import NO_STAGE

        app = self.build()
        self.assertIsNone(app.stage, "it should start on the response")
        self.assertEqual(app.stage_var.get(), NO_STAGE)
        found = next(iter(app.results.values()))
        self.assertIs(app.stage_layer(found), found.fired)
        self.assertEqual(found.stages, {}, "it captured without being asked")

    def test_choosing_a_stage_captures_it_and_draws_it(self) -> None:
        """Capture is off until asked for, because it is not free."""
        from glyph_lab.detect import STAGES

        app = self.build()
        self.whole_pipeline(app)
        app.stage_var.set(STAGES[0])
        app._on_stage_change()
        self.settle(app)
        found = next(iter(app.results.values()))
        self.assertTrue(found.stages, "nothing was captured")
        layer = app.stage_layer(found)
        self.assertIsNotNone(layer)
        self.assertEqual(layer.shape, found.fired.shape)
        # 2026-08-27: not "a superset of what survives to the end".  It is
        # not one -- hysteresis deliberately promotes response *below* the
        # threshold -- and the claim only held here because this fixture is
        # too clean to have a dropout in it.  What is actually true is that
        # the seed is drawn from the threshold mask, and that the last texel
        # stage is `fired` itself.
        from glyph_lab.detect import NEARBY_CONTRAST, OFF_BOUNDARY

        seeded = found.stage_mask(OFF_BOUNDARY)
        recovered = found.stage_mask(NEARBY_CONTRAST)
        self.assertEqual(int((seeded & ~recovered).sum()), 0)
        np.testing.assert_array_equal(found.stage_mask(STAGES[-1]), found.fired)

    def test_switching_between_captured_stages_needs_no_rerun(self) -> None:
        """They were all recorded by the same pass, so this must be instant."""
        from glyph_lab.detect import STAGES

        app = self.build()
        self.whole_pipeline(app)
        app.stage_var.set(STAGES[0])
        app._on_stage_change()
        self.settle(app)
        before = next(iter(app.results.values()))
        app.stage_var.set(STAGES[-3])
        app._on_stage_change()
        self.assertIs(next(iter(app.results.values())), before, "it re-ran")
        self.assertIsNotNone(app.stage_layer(before))

    def test_an_early_stage_does_not_show_the_finished_overlays(self) -> None:
        """The point of a stage view: nothing on screen may be from later.

        Leaving the final outlines under an early stage's texels invites
        reading a late decision back onto an early picture, which is exactly
        the confusion this is meant to remove.  An overlay whose stage has not
        happened yet is drawn as nothing.
        """
        from glyph_lab.detect import STAGE_FIRST, STAGES

        app = self.build()
        app.stage_var.set(STAGES[0])            # over threshold, before anything
        app._on_stage_change()
        self.settle(app)
        found = next(iter(app.results.values()))
        early = app.stage_overlays(found)
        self.assertIsNone(early["outlines"], "the final regions leaked backwards")

        # Indexed by what the overlay says about itself rather than by a
        # number, so inserting a stage does not silently retarget the test.
        app.stage_var.set(STAGES[STAGE_FIRST["outlines"]])
        app._on_stage_change()
        middle = app.stage_overlays(found)
        self.assertIsNotNone(middle["outlines"], "regions exist by now")

        app.stage_var.set(STAGES[-1])           # the end: everything is real
        app._on_stage_change()
        self.assertIsNotNone(app.stage_overlays(found)["outlines"])

    def test_a_stop_from_before_a_new_stage_is_moved_forward(self) -> None:
        """Otherwise adding a stage hides it, on every machine that has run.

        The panel hides the knobs of any stage past the stop, and a settings
        file written before the stage existed names an earlier one -- so the
        stage arrives switched off and looks unbuilt.
        """
        from glyph_lab import app as app_module
        from glyph_lab.detect import ContrastConfig

        fresh = ContrastConfig().stop_after_stage
        self.assertGreater(fresh, 1, "this test wants a truncated pipeline")
        app_module.SETTINGS_PATH.write_text(
            json.dumps({"stop_after_stage": fresh - 1})
        )
        self.assertEqual(app_module.load_settings()["stop_after_stage"], fresh)

    def test_a_stop_that_is_not_stale_is_left_alone(self) -> None:
        """0 means run all of it, and a later stop is a deliberate choice."""
        from glyph_lab import app as app_module
        from glyph_lab.detect import ContrastConfig

        fresh = ContrastConfig().stop_after_stage
        for value in (0, fresh, fresh + 1):
            app_module.SETTINGS_PATH.write_text(
                json.dumps({"stop_after_stage": value})
            )
            self.assertEqual(
                app_module.load_settings()["stop_after_stage"], value
            )

    def test_a_stale_normal_stop_runs_the_new_normal_stage(self) -> None:
        """A normal pipeline extension must not arrive silently disabled."""
        from glyph_lab import app as app_module
        from glyph_lab.normal_detect import NORMAL_STAGES

        app_module.SETTINGS_PATH.write_text(
            json.dumps({"normal_stop_after_stage": len(NORMAL_STAGES) - 1})
        )
        self.assertEqual(
            app_module.load_settings()["normal_stop_after_stage"], 0
        )

    def test_a_stage_that_does_not_run_is_off_the_panel(self) -> None:
        """Half a job otherwise: the pipeline stops but the knobs remain.

        A box that still takes a value for a stage that no longer happens is
        worse than no box -- it invites tuning a number that cannot do
        anything.  The same goes for the stage picker, which would otherwise
        offer a blank picture with nothing to say why it is blank.
        """
        from glyph_lab.app import NO_STAGE
        from glyph_lab.detect import PADDED, STAGE_OF, STAGES

        app = self.build()
        stop = app.stage_stop()
        self.assertGreater(stop, 0, "this test wants a truncated pipeline")

        for name, row in app._config_rows.items():
            shown = bool(row.winfo_ismapped())
            self.assertEqual(
                shown, STAGE_OF[name] <= stop,
                f"{name} (stage {STAGE_OF[name]}) is "
                f"{'shown' if shown else 'hidden'} at stop {stop}",
            )
        self.assertEqual(
            list(app._stage_box.cget("values")), [NO_STAGE, *STAGES[:stop]]
        )
        self.assertEqual(STAGES[stop - 1], PADDED)
        # ...and the stage's own heading goes with its knobs, so the panel
        # does not end on a run of empty titles.
        for heading, names in app._colour_section_rows:
            shown = bool(heading.winfo_ismapped())
            reached = any(STAGE_OF[name] <= stop for name in names)
            self.assertEqual(
                shown, reached,
                f"heading {heading.cget('text')!r} is "
                f"{'shown' if shown else 'hidden'} at stop {stop}",
            )

    def test_the_panel_is_sectioned_by_stage(self) -> None:
        """Each parameter under the stage that reads it, in pipeline order.

        The panel is read while watching a stage, so a flat list in dataclass
        order makes tuning a hunt -- `mark_gap_px` used to sit above the
        pattern knobs purely because it was written first.
        """
        from dataclasses import fields

        from glyph_lab.app import SECTIONS, panel_groups
        from glyph_lab.detect import STAGE_OF, STAGES, ContrastConfig

        title, config_type, _attribute = SECTIONS[0]
        groups = panel_groups(title, config_type)
        # The pipeline control keeps the section's own title and comes first.
        self.assertEqual(groups[0], (title, ("stop_after_stage",)))
        # Then one heading per stage that owns a parameter, in order.
        self.assertEqual(
            [heading for heading, _names in groups[1:]],
            [name for index, name in enumerate(STAGES, start=1)
             if any(STAGE_OF.get(e.name) == index for e in fields(config_type))],
        )
        for heading, names in groups[1:]:
            for name in names:
                self.assertEqual(STAGES[STAGE_OF[name] - 1], heading)
        # Every parameter appears exactly once, so nothing is lost in grouping.
        listed = [name for _heading, names in groups for name in names]
        self.assertEqual(
            sorted(listed), sorted(e.name for e in fields(ContrastConfig))
        )
        self.assertEqual(len(listed), len(set(listed)))

    def test_a_config_with_no_stages_stays_one_section(self) -> None:
        """The grouping is not allowed to break a config the stages ignore."""
        from dataclasses import fields

        from glyph_lab.app import panel_groups
        from glyph_lab.render import ViewOptions

        groups = panel_groups("The view", ViewOptions)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0][0], "The view")
        self.assertEqual(
            list(groups[0][1]), [e.name for e in fields(ViewOptions)]
        )

    def test_clearing_the_stop_brings_every_knob_back(self) -> None:
        """The reversibility claim, at the panel rather than in the pipeline."""
        from glyph_lab.app import NO_STAGE
        from glyph_lab.detect import STAGES

        app = self.build()
        self.whole_pipeline(app)
        self.settle(app)
        for name, row in app._config_rows.items():
            self.assertTrue(row.winfo_ismapped(), f"{name} stayed hidden")
        self.assertEqual(
            list(app._stage_box.cget("values")), [NO_STAGE, *STAGES]
        )

    def test_a_stage_past_the_stop_is_dropped_when_the_stop_moves_in(self) -> None:
        """Picking a late stage and then truncating must not leave it selected."""
        from glyph_lab.app import NO_STAGE
        from glyph_lab.detect import STAGES

        app = self.build()
        self.whole_pipeline(app)
        self.settle(app)
        app.stage_var.set(STAGES[-1])
        app._on_stage_change()
        self.settle(app)
        self.assertEqual(app.stage, STAGES[-1])

        app.config_vars["stop_after_stage"].set("2")
        app.run_detection()
        self.settle(app)
        self.assertIsNone(app.stage, "a stage that no longer runs stayed picked")
        self.assertEqual(app.stage_var.get(), NO_STAGE)

    def test_the_status_line_says_when_the_pipeline_is_cut_short(self) -> None:
        """0 regions with no explanation reads as a broken parameter."""
        from glyph_lab.detect import STAGES

        app = self.build()
        self.assertIn("stopped after", app.status.cget("text"))
        self.assertIn(STAGES[app.stage_stop() - 1], app.status.cget("text"))

        self.whole_pipeline(app)
        self.settle(app)
        self.assertNotIn("stopped after", app.status.cget("text"))

    def test_the_panel_shows_what_each_stage_cost(self) -> None:
        """Without anyone having to switch capture on to see it."""
        from glyph_lab.detect import STAGES

        app = self.build()
        text = app.costs.get("1.0", "end")
        self.assertIn("stage cost", text)
        found = next(iter(app.results.values()))
        self.assertEqual(found.stages, {}, "it captured to time the stages")
        named = [name for name in STAGES if name in text]
        self.assertTrue(named, "no stage was priced")
        # 2026-08-28: the *last stage that runs*, not the last there is.  The
        # default stop is the size floor, and the stages past it are the front
        # end the sparse field replaces.
        from glyph_lab.detect import UNRIMMED

        self.assertIn(UNRIMMED, named, "the region pass went unaccounted")
        # Pipeline order, so the breakdown reads the way the pipeline runs.
        self.assertEqual(named, sorted(named, key=text.index))

    def test_no_stage_selected_shows_the_finished_detection(self) -> None:
        app = self.build()
        found = next(iter(app.results.values()))
        layers = app.stage_overlays(found)
        self.assertIs(layers["outlines"], found.outlines)

    def test_a_stage_draws_its_own_regions_not_the_finished_ones(self) -> None:
        """Picking a stage has to change the green, or it answers nothing.

        2026-08-29: it did not.  Every stage at or past the first one with
        regions drew ``found.outlines``, so stepping back through the stages
        showed where a mark's texels went but never which stage stopped
        calling it a region -- and that is the only question the picker is
        there to answer.
        """
        from glyph_lab.detect import SIZED, SPARSE

        app = self.build()
        # A floor high enough to throw something away between the two stages,
        # so there is a difference on screen to find.
        app.config_vars["min_width_px"].set("24")
        app.config_vars["min_height_px"].set("24")
        app.stage_var.set(SPARSE)
        app._on_stage_change()
        self.settle(app)
        found = next(iter(app.results.values()))
        early = app.stage_overlays(found)["outlines"]

        app.stage_var.set(SIZED)
        app._on_stage_change()
        late = app.stage_overlays(found)["outlines"]
        self.assertIsNotNone(early)
        self.assertIsNotNone(late)
        self.assertGreater(
            int(early.sum()), int(late.sum()),
            "the sparse field drew the size floor's regions",
        )
        self.assertGreaterEqual(app.stage_count(found), 0)

    def test_the_region_count_follows_the_selected_stage(self) -> None:
        from glyph_lab.detect import AT_THRESHOLD, SPARSE

        app = self.build()
        found = next(iter(app.results.values()))
        self.assertEqual(app.stage_count(found), len(found.regions))

        app.stage_var.set(SPARSE)
        app._on_stage_change()
        self.settle(app)
        found = next(iter(app.results.values()))
        self.assertEqual(app.stage_count(found), found.stage_counts[SPARSE])

        # The threshold has no regions, and must say so rather than borrowing
        # the finished tally.
        app.stage_var.set(AT_THRESHOLD)
        app._on_stage_change()
        self.assertEqual(app.stage_count(found), 0)

    def test_the_panel_offers_a_box_for_every_parameter(self) -> None:
        """A knob with no entry is a knob nobody can turn.

        ``config_vars`` is flat and keyed by field name, so this also catches a
        section added with a name that collides with an existing one.
        """
        from dataclasses import fields

        from glyph_lab.app import SECTIONS

        app = self.build()
        expected = [
            entry.name
            for _title, config_type, _attribute in SECTIONS
            for entry in fields(config_type)
        ]
        self.assertEqual(len(expected), len(set(expected)), "field names collide")
        self.assertEqual(set(app.config_vars), set(expected))

    def test_a_bad_parameter_is_reported_rather_than_raised(self) -> None:
        app = self.build()
        app.config_vars["kernel_px"].set("not a number")
        app.run_detection()                     # must not raise
        self.assertIn("bad parameter", app.status.cget("text"))

    def test_detection_reaches_the_canvas(self) -> None:
        """Drive the worker/queue handshake the way the event loop does."""
        app = self.build()
        app.canvas.configure(width=800, height=600)
        app.update()
        for _ in range(400):                    # ~4 s ceiling on a tiny corpus
            app.update()
            if app.results and not app._busy:
                break
            app.after(10)
        self.assertEqual(set(app.results), {t.name for t in app.textures})
        self.assertIn("regions over", app.status.cget("text"))
        self.assertIn("texels", app.status.cget("text"))
        # One mark per texture in the tiny corpus, so the breakdown should
        # name each one and give it a size.
        breakdown = self.counts_text(app)
        for texture in app.textures:
            self.assertIn(texture.name, breakdown)
        self.assertIn("median", breakdown)

    @staticmethod
    def counts_text(app) -> str:
        return app.counts.get("1.0", "end")


if __name__ == "__main__":
    unittest.main()
