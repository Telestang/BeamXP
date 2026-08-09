"""Q/W/E/R/T/Y set a transform on whichever table holds the selection.

The keys run along the top row in the order the parts dropdown offers, and
the triggers table answers the first three of them, since a trigger box only
takes Skip, Move or Mirror. One key press must reach exactly one table: the
trigger table keeps a focused row after the parts table has taken the
selection off it, so reading the focus instead of the selection would let a
long-dead trigger row swallow keys meant for a mesh.
"""

from __future__ import annotations

import unittest

from beamxp import hand_drive_core as core
from beamxp.hand_drive_ui.shared import (
    MODE_CYCLE_VALUES,
    MODE_HOTKEYS,
    MODE_VALUES_BY_LABEL,
    mode_label,
)
from beamxp.hand_drive_ui.triggers_workflow import (
    CONTROL_MASK,
    SHIFT_MASK,
    TriggersWorkflowMixin,
)


class _FakeWidget:
    def __init__(self, widget_class: str) -> None:
        self._class = widget_class

    def winfo_class(self) -> str:
        return self._class


class _FakeVar:
    def __init__(self) -> None:
        self.value = ""

    def get(self) -> str:
        return self.value

    def set(self, value: str) -> None:
        self.value = value


class FakeTree:
    def __init__(self) -> None:
        self._selection: list[str] = []
        self._focus = ""
        self.rows: list[str] = []
        self.click_row = ""
        self.click_column = "trigger"
        # Stands in for <<TreeviewSelect>>, which Tk QUEUES on every selection
        # change -- including the ones a rebuild makes putting its rows back.
        self.on_select: list = []
        self.pending: list = []

    def identify_row(self, _y):
        return self.click_row

    def identify_column(self, _x):
        return self.click_column

    def selection(self) -> list[str]:
        return list(self._selection)

    def selection_set(self, items) -> None:
        self._selection = list(items)
        self._queue_select()

    def selection_add(self, item) -> None:
        if item not in self._selection:
            self._selection.append(item)
        self._queue_select()

    def delete_all(self) -> None:
        """Empty the table the way Tk does: one row at a time, queueing the
        selection event on each."""
        for item in list(self.rows):
            self.rows.remove(item)
            if item in self._selection:
                self._selection.remove(item)
            self._queue_select()

    def _queue_select(self) -> None:
        """Tk QUEUES <<TreeviewSelect>> rather than sending it.

        Verified against a live ttk.Treeview: a handler bound to it runs after
        the call that changed the selection has returned. Modelling it as a
        synchronous callback is what let two wrong fixes pass their tests --
        any guard set around a selection change is long gone by the time the
        real handler reads it. Tests drain this explicitly, as the event loop
        would.
        """
        self.pending.extend(self.on_select)

    def drain(self) -> None:
        pending, self.pending = list(self.pending), []
        for hook in pending:
            hook()

    def focus(self, item: str | None = None) -> str:
        if item is not None:
            self._focus = item
        return self._focus

    def see(self, _item: str) -> None:
        pass

    def exists(self, item: str) -> bool:
        return item in self.rows


class Harness(TriggersWorkflowMixin):
    """The cross-table coordinator, with both tables' edges stubbed.

    The parts side lives in PartEditingMixin; what matters here is that the
    coordinator asks it for targets and hands it modes, so those two seams are
    recorded rather than reimplemented.
    """

    def __init__(self) -> None:
        self.trigger_tree = FakeTree()
        self.part_tree = FakeTree()
        self.trigger_rows_by_iid: dict[str, dict] = {}
        self.status_var = _FakeVar()
        self.conversion: dict[str, object] = {}
        self.context = object()  # only ever checked for None
        self.focused = None
        self.part_targets: list[str] = []
        self.part_modes: dict[str, str] = {}
        self.part_settings: dict[str, dict] = {}
        self.part_mode_calls: list[str] = []
        self.trigger_refreshes = 0
        self.part_refreshes = 0
        self.detail_updates = 0
        self.errors: list[tuple[str, str]] = []
        self.selection_events = 0

    def focus_get(self):
        return self.focused

    def _refresh_triggers(self) -> None:
        # A faithful miniature of the real refresh: rows are rebuilt from the
        # conversion, so a row's mode and Move X are whatever was just stored,
        # and the selection it was holding is put back -- which is the step
        # that used to be mistaken for the user claiming this table.
        self.trigger_refreshes += 1
        chosen = core.trigger_mode_map(self.conversion)
        offsets = core.trigger_offset_map(self.conversion)
        for row in self.trigger_rows_by_iid.values():
            key = (str(row["trigger"]), tuple(row["at"]))
            row["mode"] = chosen.get(key)
            row["offset"] = offsets.get(key)
        keep = self.trigger_tree.selection()
        rows = list(self.trigger_tree.rows)
        self.trigger_tree.delete_all()
        self.trigger_tree.rows.extend(rows)
        if keep:
            self.trigger_tree.selection_set(keep)

    def _refresh_parts(self) -> None:
        self.part_refreshes += 1
        keep = self.part_tree.selection()
        rows = list(self.part_tree.rows)
        self.part_tree.delete_all()
        self.part_tree.rows.extend(rows)
        if keep:
            self.part_tree.selection_set(keep)

    def drain_events(self) -> None:
        """Run the queued <<TreeviewSelect>> handlers, as the event loop would."""
        for _ in range(4):  # a handler may queue more
            if not (self.part_tree.pending or self.trigger_tree.pending):
                break
            self.part_tree.drain()
            self.trigger_tree.drain()

    def _refresh_delta_label(self) -> None:
        pass

    # the real one reschedules the 3D scene; here it only has to record that a
    # mutation asked for the preview to catch up
    def _update_detail(self) -> None:
        self.detail_updates += 1

    def _show_error(self, title: str, message: str) -> None:
        self.errors.append((title, message))

    # --- the PartEditingMixin seams the coordinator reaches through ---------
    def _selected_part_mode_targets(self) -> tuple[list[str], list[str]]:
        return list(self.part_targets), [f"row:{mesh}" for mesh in self.part_targets]

    def _apply_part_modes(self, targets: list[str], mode: str) -> None:
        # The real one writes settings["mode"], which is what _get_part_setting
        # reads back when Move X decides which rows can take an offset.
        self.part_mode_calls.append(mode)
        for object_id in targets:
            self.part_modes[object_id] = mode
            self.part_settings.setdefault(object_id, {})["mode"] = mode

    def _get_part_setting(self, object_id: str, key: str, default=None):
        return self.part_settings.get(object_id, {}).get(key, default)

    def _part_settings(self, object_id: str) -> dict:
        return self.part_settings.setdefault(object_id, {})

    def _part_child_override(self, _row_id):
        return None

    # PartEditingMixin's <<TreeviewSelect>> handler, minus the viewer refresh.
    # It deliberately decides nothing about ownership -- the real one does not
    # either, because Tk queues this event and a rebuild's re-selection is
    # indistinguishable from a click by the time it arrives.
    def _part_selection_changed_stub(self) -> None:
        self.selection_events += 1

    # --- the click plumbing _trigger_click sits on -------------------------
    def _tree_body_click(self, _tree, _event) -> bool:
        return True

    def _tree_column_name(self, _tree, column: str) -> str:
        return column

    def _close_tree_combo_editor(self) -> None:
        pass

    def _edit_tree_combo(self, *_args, **_kwargs) -> None:
        pass

    def _edit_tree_entry(self, *_args, **_kwargs) -> None:
        pass

    # --- fixtures -----------------------------------------------------------
    def add_trigger(self, row_id: str, trigger_id: str, at) -> dict:
        row = {"id": row_id, "trigger": trigger_id, "at": at, "label": trigger_id}
        self.trigger_rows_by_iid[row_id] = row
        self.trigger_tree.rows.append(row_id)
        return row

    def add_part(self, mesh: str, mode: str = core.MODE_TRANSLATE) -> str:
        self.part_targets.append(mesh)
        self.part_modes[mesh] = mode
        self.part_settings.setdefault(mesh, {})["mode"] = mode
        return mesh


ROW_ID = "trig|hood_int|0.3|1.2|0.6"
AT = (0.3, 1.2, 0.6)


class HotkeyOrderTests(unittest.TestCase):
    def test_the_keys_run_along_the_dropdown_in_order(self) -> None:
        self.assertEqual(
            [mode_label(mode) for mode in MODE_CYCLE_VALUES],
            ["Skip", "Move", "Mirror", "Swap Mesh", "Replace Source"],
        )
        self.assertEqual(list(MODE_HOTKEYS), list("qwery"))
        self.assertEqual(list(MODE_HOTKEYS.values()), MODE_CYCLE_VALUES)

    def test_mirror_move_cannot_be_chosen_from_the_ui(self) -> None:
        """It did not work as intended and is withdrawn, not fixed.

        MODE_CYCLE_VALUES is the single list behind the dropdown, the hotkeys
        and the label->mode map, so being absent from it is what makes the mode
        unreachable. The pipeline still understands it, because a project saved
        while it was offered has to keep building the way it always did.
        """
        self.assertNotIn(core.MODE_MIRROR_POSITION, MODE_CYCLE_VALUES)
        self.assertNotIn(core.MODE_MIRROR_POSITION, MODE_HOTKEYS.values())
        self.assertNotIn(core.MODE_MIRROR_POSITION, MODE_VALUES_BY_LABEL.values())
        # Still rendered and still converted for the projects that carry it.
        self.assertEqual(mode_label(core.MODE_MIRROR_POSITION), "Mirror Move")
        self.assertIn(core.MODE_MIRROR_POSITION, core.NUDGE_MODES)


class TriggerHotkeyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = Harness()
        self.app.add_trigger(ROW_ID, "hood_int", AT)

    def chosen(self):
        return core.trigger_mode_map(self.app.conversion)

    def test_the_first_three_keys_set_the_selected_box(self) -> None:
        for mode in (core.MODE_SKIP, core.MODE_TRANSLATE, core.MODE_MIRROR):
            with self.subTest(mode=mode):
                self.app.trigger_tree.selection_set([ROW_ID])
                self.assertEqual(self.app._set_selected_mode_shortcut(None, mode), "break")
                self.assertEqual(list(self.chosen().values()), [mode])
                self.assertEqual(self.app.part_mode_calls, [])

    def test_a_parts_only_transform_is_refused_not_passed_on(self) -> None:
        self.app.trigger_tree.selection_set([ROW_ID])
        self.assertEqual(
            self.app._set_selected_mode_shortcut(None, core.MODE_MIRROR_STRUCTURAL),
            "break",
        )
        self.assertEqual(self.chosen(), {})
        self.assertEqual(self.app.part_mode_calls, [])
        self.assertIn("Swap Mesh", self.app.status_var.get())

    def test_with_only_meshes_selected_the_key_goes_to_them(self) -> None:
        self.app.add_part("dash")
        self.app._set_selected_mode_shortcut(None, core.MODE_MIRROR)
        self.assertEqual(self.app.part_mode_calls, [core.MODE_MIRROR])
        self.assertEqual(self.chosen(), {})

    def test_nothing_selected_declines_the_key(self) -> None:
        self.app.trigger_tree.focus(ROW_ID)
        self.app.trigger_tree.selection_set([])
        self.assertIsNone(self.app._set_selected_mode_shortcut(None, core.MODE_TRANSLATE))
        self.assertEqual(self.app.part_mode_calls, [])
        self.assertEqual(self.chosen(), {})

    def test_typing_keeps_its_own_letters(self) -> None:
        # The filter box above the table is an Entry; an "e" typed into it is
        # part of a word, not a transform. (The parts handler it falls through
        # to stands down on the same test.)
        self.app.trigger_tree.selection_set([ROW_ID])
        self.app.focused = _FakeWidget("TEntry")
        self.app._set_selected_mode_shortcut(None, core.MODE_MIRROR)
        self.assertEqual(self.chosen(), {})
        self.assertEqual(self.app.status_var.get(), "")


class TriggerOffsetTests(unittest.TestCase):
    """The Triggers table's Move X: a per-box override of how far it travels."""

    def setUp(self) -> None:
        self.app = Harness()
        self.row = self.app.add_trigger(ROW_ID, "hood_int", AT)
        self.app.trigger_tree.selection_set([ROW_ID])
        self.app._set_trigger_mode(self.row, core.MODE_TRANSLATE)

    def offsets(self):
        return core.trigger_offset_map(self.app.conversion)

    def test_an_offset_is_stored_signed_against_the_box(self) -> None:
        self.app._set_trigger_offset(self.row, "-0.42")
        self.assertEqual(self.offsets(), {("hood_int", AT): -0.42})
        self.assertEqual(self.app.errors, [])

    def test_blanking_it_hands_the_box_back_to_the_conversion_delta(self) -> None:
        self.app._set_trigger_offset(self.row, "0.42")
        self.app._set_trigger_offset(self.row, "  ")
        self.assertEqual(self.offsets(), {})
        self.assertEqual(list(self.chosen_modes()), [("hood_int", AT)])  # still Move

    def chosen_modes(self):
        return core.trigger_mode_map(self.app.conversion)

    def test_junk_is_refused_and_changes_nothing(self) -> None:
        self.app._set_trigger_offset(self.row, "0.42")
        self.app._set_trigger_offset(self.row, "half a metre")
        self.assertEqual(self.offsets(), {("hood_int", AT): 0.42})
        self.assertEqual(len(self.app.errors), 1)

    def test_zero_is_refused_because_skip_is_what_that_means(self) -> None:
        self.app._set_trigger_offset(self.row, "0")
        self.assertEqual(self.offsets(), {})
        self.assertEqual(len(self.app.errors), 1)

    def test_the_offset_survives_every_change_of_transform(self) -> None:
        """Including a round trip through Skip.

        Skip is how you check what a box looks like untouched, so dropping the
        distance on the way through costs the work of dialling it back in --
        and a mesh has never behaved that way with its own Move X.
        """
        self.app._set_trigger_offset(self.row, "0.42")
        for mode in (core.MODE_SKIP, core.MODE_MIRROR, core.MODE_SKIP, core.MODE_TRANSLATE):
            self.app._set_trigger_mode(self.row, mode)
            self.assertEqual(self.offsets(), {("hood_int", AT): 0.42}, mode)

    def test_every_edit_asks_the_preview_to_catch_up(self) -> None:
        before = self.app.detail_updates
        self.app._set_trigger_offset(self.row, "0.42")
        self.assertEqual(self.app.detail_updates, before + 1)


SECOND_ROW_ID = "trig|shifter|0.4|1.1|0.5"
SECOND_AT = (0.4, 1.1, 0.5)


class CombinedSelectionTests(unittest.TestCase):
    """One edit reaching meshes and trigger boxes at once.

    The case this is for: a shifter is a handful of meshes plus the boxes that
    label them, and they all want the same Move and the same Move X. Gathering
    them with Ctrl-click and pressing W once beats two passes that have to
    agree with each other.
    """

    def setUp(self) -> None:
        self.app = Harness()
        self.row = self.app.add_trigger(ROW_ID, "hood_int", AT)
        self.second = self.app.add_trigger(SECOND_ROW_ID, "shifter", SECOND_AT)
        self.app.add_part("shifter_boot", core.MODE_SKIP)
        self.app.add_part("shifter_knob", core.MODE_SKIP)
        self.app.trigger_tree.selection_set([ROW_ID, SECOND_ROW_ID])

    def chosen(self):
        return core.trigger_mode_map(self.app.conversion)

    def test_one_key_moves_the_meshes_and_the_boxes_together(self) -> None:
        self.assertEqual(
            self.app._set_selected_mode_shortcut(None, core.MODE_TRANSLATE), "break"
        )
        self.assertEqual(
            self.app.part_modes,
            {"shifter_boot": core.MODE_TRANSLATE, "shifter_knob": core.MODE_TRANSLATE},
        )
        self.assertEqual(
            sorted(self.chosen().values()), [core.MODE_TRANSLATE, core.MODE_TRANSLATE]
        )
        status = self.app.status_var.get()
        self.assertIn("2 mesh(es)", status)
        self.assertIn("2 trigger(s)", status)

    def test_a_mode_only_meshes_take_still_reaches_them(self) -> None:
        # Mirror Move is not one of a box's three. The meshes still get it, and
        # the boxes are reported rather than passed over in silence.
        self.app._set_selected_mode_shortcut(None, core.MODE_MIRROR_POSITION)
        self.assertEqual(set(self.app.part_modes.values()), {core.MODE_MIRROR_POSITION})
        self.assertEqual(self.chosen(), {})
        self.assertIn("2 trigger(s) cannot take it", self.app.status_var.get())

    def test_a_pairing_mode_refuses_a_gathered_selection(self) -> None:
        # Swap Mesh needs a source picked for the one row it lands on, so it
        # cannot be sprayed across a selection.
        self.app._set_selected_mode_shortcut(None, core.MODE_MIRROR_STRUCTURAL)
        self.assertEqual(self.app.part_mode_calls, [])
        self.assertEqual(self.chosen(), {})
        self.assertIn("Swap Mesh", self.app.status_var.get())

    def test_one_move_x_lands_on_every_row_that_can_take_it(self) -> None:
        self.app._set_selected_mode_shortcut(None, core.MODE_TRANSLATE)
        self.app._apply_offset_to_selection("-0.42")
        self.assertEqual(
            {mesh: settings.get("translateOffset") for mesh, settings in self.app.part_settings.items()},
            {"shifter_boot": -0.42, "shifter_knob": -0.42},
        )
        self.assertEqual(
            sorted(core.trigger_offset_map(self.app.conversion).values()), [-0.42, -0.42]
        )

    def test_move_x_passes_over_rows_with_no_distance_to_take(self) -> None:
        # Boxes on Move, meshes left on Skip: only the boxes take the offset,
        # and the count says so rather than implying the meshes changed.
        self.app.trigger_tree.selection_set([ROW_ID, SECOND_ROW_ID])
        for row in (self.row, self.second):
            core.set_trigger_mode(self.app.conversion, str(row["trigger"]), row["at"], core.MODE_TRANSLATE)
        self.app._refresh_triggers()
        self.app._apply_offset_to_selection("0.3")
        self.assertEqual([s.get("translateOffset") for s in self.app.part_settings.values()], [None, None])
        self.assertEqual(len(core.trigger_offset_map(self.app.conversion)), 2)
        self.assertIn("2 trigger(s)", self.app.status_var.get())
        self.assertNotIn("mesh(es)", self.app.status_var.get())

    def test_a_zero_move_x_is_refused_before_anything_is_written(self) -> None:
        self.app._set_selected_mode_shortcut(None, core.MODE_TRANSLATE)
        self.app._apply_offset_to_selection("0")
        self.assertEqual(len(self.app.errors), 1)
        self.assertEqual(core.trigger_offset_map(self.app.conversion), {})
        self.assertEqual([s.get("translateOffset") for s in self.app.part_settings.values()], [None, None])


class TriggerClickFallthroughTests(unittest.TestCase):
    """A click the Treeview never sees cannot Ctrl-select.

    Extended selection lives in the Treeview's own class binding, which only
    runs when our handler declines the event. Returning "break" for every body
    click left the table stuck on one row no matter what selectmode said --
    which is exactly how multi-select shipped broken the first time.
    """

    class _Event:
        state = 0
        x = 0
        y = 0

    def setUp(self) -> None:
        self.app = Harness()
        self.row = self.app.add_trigger(ROW_ID, "hood_int", AT)
        self.app.trigger_tree.click_row = ROW_ID

    def click(self, column: str):
        self.app.trigger_tree.click_column = column
        return self.app._trigger_click(self._Event())

    def test_the_trigger_column_falls_through_to_the_treeview(self) -> None:
        self.assertIsNone(self.click("trigger"))

    def test_the_editable_columns_still_take_the_click(self) -> None:
        for column in ("mode", "offset"):
            with self.subTest(column=column):
                self.assertEqual(self.click(column), "break")

    def test_falling_through_leaves_the_selection_to_the_treeview(self) -> None:
        # Two rows already gathered; a click on the name column must not
        # reduce them here, because the Treeview has not run yet.
        second = self.app.add_trigger(SECOND_ROW_ID, "shifter", SECOND_AT)
        self.app.trigger_tree.selection_set([ROW_ID, SECOND_ROW_ID])
        self.click("trigger")
        self.assertEqual(
            sorted(self.app.trigger_tree.selection()), sorted([ROW_ID, SECOND_ROW_ID])
        )
        self.assertIsNotNone(second)


class EditKeepsTheGatheredSelectionTests(unittest.TestCase):
    """Applying an edit must not quietly shrink what it was applied to.

    Each table's refresh puts its own rows back, and that re-selection fires
    the same event a click does. Read as a click it means "this table now owns
    the selection", and the other table is cleared -- so setting Move X on a
    gathered set dropped every mesh out of it, the trigger refresh running
    first and taking the parts table with it.
    """

    def setUp(self) -> None:
        self.app = Harness()
        # Decoys first, and deliberately unselected. Emptying the table deletes
        # them before it reaches the selected row, so the selection is still
        # non-empty for those deletes -- which is the state that made a rebuild
        # look like a click. A table holding only the selected row never
        # reaches it, and a test built that way passes against the bug.
        self.app.add_trigger("trig|decoy|0|0|0", "decoy", (0.0, 0.0, 0.0))
        self.row = self.app.add_trigger(ROW_ID, "hood_int", AT)
        self.app.add_part("shifter_knob", core.MODE_TRANSLATE)
        self.app.part_tree.rows.extend(["row:decoy", "row:shifter_knob"])
        # The tables wired to the coordinator, as the real <<TreeviewSelect>>
        # bindings are.
        self.app.part_tree.on_select.append(self.app._part_selection_changed_stub)
        self.app.trigger_tree.on_select.append(self.app._trigger_selection_changed)
        # Gathered the way a user gathers: a click in one table, then a
        # Ctrl-click in the other.
        self.app.part_tree.selection_set(["row:shifter_knob"])
        self.app._selection_accumulating = True
        self.app.trigger_tree.selection_set([ROW_ID])
        core.set_trigger_mode(self.app.conversion, "hood_int", AT, core.MODE_TRANSLATE)
        self.app._refresh_triggers()
        # Opening the editor is a plain click on the cell, so accumulation is
        # over by the time the edit commits. The selection has to survive on
        # the strength of the restore guard alone -- which is the fix.
        self.app._selection_accumulating = False

    def test_setting_move_x_keeps_both_tables_selected(self) -> None:
        self.app._apply_offset_to_selection("-0.42")
        self.app.drain_events()  # the queued selection events land here
        self.assertEqual(self.app.part_tree.selection(), ["row:shifter_knob"])
        self.assertEqual(self.app.trigger_tree.selection(), [ROW_ID])

    def test_setting_a_mode_keeps_both_tables_selected(self) -> None:
        self.app._apply_mode_to_selection(core.MODE_MIRROR)
        self.app.drain_events()
        self.assertEqual(self.app.part_tree.selection(), ["row:shifter_knob"])
        self.assertEqual(self.app.trigger_tree.selection(), [ROW_ID])

    def test_a_plain_click_in_one_table_still_clears_the_other(self) -> None:
        # Ownership is decided at click time now, so this is the path that has
        # to keep working: a fresh click must not leave the other table armed.
        self.app._note_selection_modifier(SelectionAccumulationTests._Event(0))
        self.app._claim_selection("triggers", "trig|decoy|0|0|0")
        self.app.drain_events()
        self.assertEqual(self.app.part_tree.selection(), [])

    def test_clicking_inside_a_gathered_selection_is_an_edit_not_a_reselect(self) -> None:
        # Clicking the Move X cell of a row that is already selected must not
        # be read as starting a new selection in that table.
        self.app._note_selection_modifier(SelectionAccumulationTests._Event(0))
        self.app._claim_selection("triggers", ROW_ID)
        self.app.drain_events()
        self.assertEqual(self.app.part_tree.selection(), ["row:shifter_knob"])


class PreviewPickAccumulationTests(unittest.TestCase):
    """Ctrl-clicking in the 3D preview gathers, the same as in the tables.

    The viewer's pick carried no modifier at all, so every preview click was a
    fresh single selection however it was made.
    """

    def setUp(self) -> None:
        self.app = Harness()
        self.app.part_tree.rows.extend(["mesh_a", "mesh_b"])

    def toggle(self, item: str) -> None:
        self.app._toggle_tree_selection(self.app.part_tree, item)

    def test_a_second_pick_adds_rather_than_replaces(self) -> None:
        self.toggle("mesh_a")
        self.toggle("mesh_b")
        self.assertEqual(sorted(self.app.part_tree.selection()), ["mesh_a", "mesh_b"])

    def test_picking_a_gathered_row_again_drops_it(self) -> None:
        # A mis-pick is undone by picking it again, not by starting over.
        self.toggle("mesh_a")
        self.toggle("mesh_b")
        self.toggle("mesh_a")
        self.assertEqual(self.app.part_tree.selection(), ["mesh_b"])

    def test_the_viewer_reports_the_modifier(self) -> None:
        # The pick plumbing has to carry it, or nothing above can act on it.
        import inspect

        from beamxp import mesh_preview

        signature = inspect.signature(mesh_preview.MeshPreview._do_pick)
        self.assertIn("accumulate", signature.parameters)


class SelectionAccumulationTests(unittest.TestCase):
    """Which click keeps the other table's selection, and which clears it."""

    class _Event:
        def __init__(self, state: int) -> None:
            self.state = state

    def setUp(self) -> None:
        self.app = Harness()
        self.app.add_trigger(ROW_ID, "hood_int", AT)
        self.app.part_tree.rows.append("row:dash")
        self.app.part_tree.selection_set(["row:dash"])

    def test_a_plain_click_starts_a_fresh_selection(self) -> None:
        self.app._note_selection_modifier(self._Event(0))
        self.app._claim_selection("triggers")
        self.assertEqual(self.app.part_tree.selection(), [])

    def test_ctrl_and_shift_clicks_accumulate_across_both_tables(self) -> None:
        for state in (CONTROL_MASK, SHIFT_MASK, CONTROL_MASK | SHIFT_MASK):
            with self.subTest(state=state):
                self.app.part_tree.selection_set(["row:dash"])
                self.app._note_selection_modifier(self._Event(state))
                self.app._claim_selection("triggers")
                self.assertEqual(self.app.part_tree.selection(), ["row:dash"])

    def test_a_junk_event_state_is_treated_as_a_plain_click(self) -> None:
        self.app._note_selection_modifier(self._Event("not a bitmask"))
        self.app._claim_selection("triggers")
        self.assertEqual(self.app.part_tree.selection(), [])


if __name__ == "__main__":
    unittest.main()
