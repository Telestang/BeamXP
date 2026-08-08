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

    def focus_get(self):
        return self.focused

    def _refresh_triggers(self) -> None:
        self.trigger_refreshes += 1

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
            ["Skip", "Move", "Mirror", "Swap Mesh", "Mirror Move", "Replace Source"],
        )
        self.assertEqual(list(MODE_HOTKEYS), list("qwerty"))
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


if __name__ == "__main__":
    unittest.main()
