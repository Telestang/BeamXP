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
from beamxp.hand_drive_ui.shared import MODE_CYCLE_VALUES, MODE_HOTKEYS, mode_label
from beamxp.hand_drive_ui.triggers_workflow import TriggersWorkflowMixin


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

    def selection(self) -> list[str]:
        return list(self._selection)

    def selection_set(self, items) -> None:
        self._selection = list(items)

    def focus(self, item: str | None = None) -> str:
        if item is not None:
            self._focus = item
        return self._focus

    def see(self, _item: str) -> None:
        pass

    def exists(self, item: str) -> bool:
        return item in self.rows


class Harness(TriggersWorkflowMixin):
    def __init__(self) -> None:
        self.trigger_tree = FakeTree()
        self.trigger_rows_by_iid: dict[str, dict] = {}
        self.status_var = _FakeVar()
        self.conversion: dict[str, object] = {}
        self.focused = None
        self.part_mode_calls: list[str] = []
        self.trigger_refreshes = 0
        self.detail_updates = 0
        self.errors: list[tuple[str, str]] = []

    def focus_get(self):
        return self.focused

    def _refresh_triggers(self) -> None:
        self.trigger_refreshes += 1

    # the real one reschedules the 3D scene; here it only has to record that a
    # mutation asked for the preview to catch up
    def _update_detail(self) -> None:
        self.detail_updates += 1

    def _show_error(self, title: str, message: str) -> None:
        self.errors.append((title, message))

    # the real one lives in PartEditingMixin; here it only has to record that
    # the key made it through to the parts table
    def _set_selected_part_mode_shortcut(self, _event, mode: str) -> str | None:
        self.part_mode_calls.append(mode)
        return "break"

    def add_trigger(self, row_id: str, trigger_id: str, at) -> dict:
        row = {"id": row_id, "trigger": trigger_id, "at": at, "label": trigger_id}
        self.trigger_rows_by_iid[row_id] = row
        self.trigger_tree.rows.append(row_id)
        return row


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
        self.assertIn("Skip (Q)", self.app.status_var.get())

    def test_without_a_trigger_selection_the_key_reaches_the_parts_table(self) -> None:
        self.app._set_selected_mode_shortcut(None, core.MODE_MIRROR)
        self.assertEqual(self.app.part_mode_calls, [core.MODE_MIRROR])
        self.assertEqual(self.chosen(), {})

    def test_a_focused_but_unselected_row_does_not_swallow_the_key(self) -> None:
        # what the parts table taking the selection leaves behind
        self.app.trigger_tree.focus(ROW_ID)
        self.app.trigger_tree.selection_set([])
        self.app._set_selected_mode_shortcut(None, core.MODE_TRANSLATE)
        self.assertEqual(self.app.part_mode_calls, [core.MODE_TRANSLATE])
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

    def test_skipping_the_box_drops_the_offset(self) -> None:
        self.app._set_trigger_offset(self.row, "0.42")
        self.app._set_trigger_mode(self.row, core.MODE_SKIP)
        self.assertEqual(self.offsets(), {})
        # ...and it does not come back when the box returns to Move
        self.app._set_trigger_mode(self.row, core.MODE_TRANSLATE)
        self.assertEqual(self.offsets(), {})

    def test_the_offset_survives_a_move_between_transforms_that_use_it(self) -> None:
        self.app._set_trigger_offset(self.row, "0.42")
        for mode in (core.MODE_MIRROR, core.MODE_TRANSLATE):
            self.app._set_trigger_mode(self.row, mode)
            self.assertEqual(self.offsets(), {("hood_int", AT): 0.42}, mode)

    def test_every_edit_asks_the_preview_to_catch_up(self) -> None:
        before = self.app.detail_updates
        self.app._set_trigger_offset(self.row, "0.42")
        self.assertEqual(self.app.detail_updates, before + 1)


if __name__ == "__main__":
    unittest.main()
