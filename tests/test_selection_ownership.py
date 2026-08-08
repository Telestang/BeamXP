"""Only one of the two tables holds the selection at a time.

A mesh row and a trigger row have different detail panels and different
hotkeys, so holding both at once leaves it ambiguous which one a keystroke is
about. Clearing a treeview fires <<TreeviewSelect>>, which lands straight back
in this logic, so these also cover the reentrancy that guards against.
"""

from __future__ import annotations

import unittest

from beamxp.hand_drive_ui.triggers_workflow import TriggersWorkflowMixin


class _FakeVar:
    def get(self) -> str:
        return ""


class FakeTree:
    """Just enough treeview to exercise selection ownership.

    Both selecting and DELETING a selected row fire the <<TreeviewSelect>>
    callback, exactly as tk does -- the delete case is verified against real
    tk and is what made a rebuild steal the user's selection.
    """

    def __init__(self, on_select=None) -> None:
        self._selection: list[str] = []
        self._rows: list[str] = []
        self.on_select = on_select

    def selection(self) -> list[str]:
        return list(self._selection)

    def selection_set(self, items) -> None:
        self._selection = list(items)
        if self.on_select is not None:
            self.on_select()

    def get_children(self, item: str = "") -> list[str]:
        return list(self._rows)

    def insert(self, _parent, _index, iid=None, **_kwargs) -> None:
        self._rows.append(str(iid))

    def delete(self, item: str) -> None:
        if item in self._rows:
            self._rows.remove(item)
        if item in self._selection:
            self._selection.remove(item)
            if self.on_select is not None:
                self.on_select()  # tk fires with the now-empty selection

    def exists(self, item: str) -> bool:
        return item in self._rows


class Harness(TriggersWorkflowMixin):
    def __init__(self) -> None:
        self.part_tree = FakeTree(on_select=self._part_selection_changed)
        self.trigger_tree = FakeTree(on_select=self._trigger_selection_changed)
        self.trigger_rows_by_iid = {}
        self.viewer = None
        self.viewer_supports_scene = False
        self.detail_refreshes = 0
        self.context = object()
        self.trigger_rows: list[dict] = []
        self.trigger_filter_var = _FakeVar()

    # the real _part_selection_changed lives in PartEditingMixin; this is the
    # part of it that matters here
    def _part_selection_changed(self) -> None:
        if self.part_tree.selection():
            self._claim_selection("parts")

    def _update_detail(self) -> None:
        self.detail_refreshes += 1

    def _selected_preview_ids(self):
        return set(self.part_tree.selection())

    # stubs for the parts of _refresh_triggers that need the real app
    def _trigger_rows(self):
        return list(self.trigger_rows)

    def _row_tags(self, index):
        return ()

    def _trigger_label(self, row):
        return str(row["label"])

    def _trigger_mode_label(self, row):
        return "(auto: unattributed)"


class SelectionOwnershipTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = Harness()

    def test_selecting_a_trigger_drops_the_mesh_selection(self) -> None:
        self.app.part_tree.selection_set(["some_mesh"])
        self.app.trigger_tree.selection_set(["trig|hood_int|0.0|0.0|0.0"])
        self.assertEqual(self.app.part_tree.selection(), [])
        self.assertEqual(
            self.app.trigger_tree.selection(), ["trig|hood_int|0.0|0.0|0.0"]
        )

    def test_selecting_a_mesh_drops_the_trigger_selection(self) -> None:
        self.app.trigger_tree.selection_set(["trig|hood_int|0.0|0.0|0.0"])
        self.app.part_tree.selection_set(["some_mesh"])
        self.assertEqual(self.app.trigger_tree.selection(), [])
        self.assertEqual(self.app.part_tree.selection(), ["some_mesh"])

    def test_the_two_tables_do_not_hand_it_back_and_forth(self) -> None:
        # Without the guard, each clear fires the other table's handler and
        # the pair recurse until the stack gives out.
        self.app.part_tree.selection_set(["some_mesh"])
        self.app.trigger_tree.selection_set(["trig|a|0.0|0.0|0.0"])
        self.app.part_tree.selection_set(["another_mesh"])
        self.assertEqual(self.app.part_tree.selection(), ["another_mesh"])
        self.assertEqual(self.app.trigger_tree.selection(), [])

    def test_an_empty_preview_click_clears_both(self) -> None:
        self.app.part_tree.selection_set(["some_mesh"])
        self.app._clear_all_selection()
        self.assertEqual(self.app.part_tree.selection(), [])
        self.assertEqual(self.app.trigger_tree.selection(), [])
        self.assertGreater(self.app.detail_refreshes, 0)

        self.app.trigger_tree.selection_set(["trig|a|0.0|0.0|0.0"])
        self.app._clear_all_selection()
        self.assertEqual(self.app.trigger_tree.selection(), [])

    def test_rebuilding_the_trigger_table_does_not_steal_the_mesh_selection(self) -> None:
        """The bug: clicking a mesh while a trigger was selected.

        The click reaches a trigger-table rebuild, whose row deletion fires
        <<TreeviewSelect>> with an empty selection. Claiming on that event
        took the selection straight back off the mesh, so the first click
        appeared only to deselect the trigger.
        """
        row = "trig|hood_int|0.0|0.0|0.0"
        self.app.trigger_tree.insert("", "end", iid=row)
        self.app.trigger_tree.selection_set([row])

        # the user clicks a mesh
        self.app.part_tree.selection_set(["some_mesh"])
        self.assertEqual(self.app.part_tree.selection(), ["some_mesh"])

        # ... and the refresh that follows rebuilds the trigger table
        self.app.trigger_tree.delete(row)
        self.assertEqual(
            self.app.part_tree.selection(),
            ["some_mesh"],
            "rebuilding the trigger table stole the mesh selection",
        )

    def test_an_empty_select_event_never_claims(self) -> None:
        self.app.part_tree.selection_set(["some_mesh"])
        self.app.trigger_tree.selection_set([])  # e.g. rows cleared
        self.assertEqual(self.app.part_tree.selection(), ["some_mesh"])

    def test_the_scene_highlight_follows_whichever_table_holds_it(self) -> None:
        self.app.trigger_tree.selection_set(["trig|a|0.0|0.0|0.0"])
        self.app.trigger_rows_by_iid = {"trig|a|0.0|0.0|0.0": {}}
        self.assertEqual(
            self.app._selected_trigger_scene_ids(), {"trig|a|0.0|0.0|0.0"}
        )
        self.app.part_tree.selection_set(["some_mesh"])
        self.assertEqual(self.app._selected_trigger_scene_ids(), set())


if __name__ == "__main__":
    unittest.main()
