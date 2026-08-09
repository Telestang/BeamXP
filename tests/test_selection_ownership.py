"""Which table owns the selection, and when the other one is cleared.

The two tables used to be mutually exclusive, and the swap was driven from
their <<TreeviewSelect>> handlers. Both of those changed:

* A Ctrl- or Shift-click gathers rows across BOTH tables on purpose, so that
  a shifter's meshes and the trigger boxes that label them take one Move and
  one Move X together.
* Ownership is decided at click time and never from the selection handler.
  Tk *queues* <<TreeviewSelect>> rather than sending it, so a handler always
  runs after the call that changed the selection has returned -- which makes a
  rebuild's own re-selection indistinguishable from a user's click. Deciding
  in the handler meant every table refresh silently cleared the other table.

``FakeTree`` therefore queues its callbacks, as tk does. A synchronous fake
here is what let two wrong fixes pass their tests while the app stayed broken.
"""

from __future__ import annotations

import unittest

from beamxp.hand_drive_ui.triggers_workflow import TriggersWorkflowMixin


class _FakeVar:
    def get(self) -> str:
        return ""


class FakeTree:
    """Just enough treeview to exercise selection ownership.

    Selecting and deleting a selected row both fire <<TreeviewSelect>>, and
    both are QUEUED -- verified against a live ttk.Treeview, where a bound
    handler runs only after selection_set has returned.
    """

    def __init__(self, on_select=None) -> None:
        self._selection: list[str] = []
        self._rows: list[str] = []
        self.on_select = on_select
        self.pending = 0

    def selection(self) -> list[str]:
        return list(self._selection)

    def selection_set(self, items) -> None:
        self._selection = list(items)
        self._queue()

    def get_children(self, item: str = "") -> list[str]:
        return list(self._rows)

    def insert(self, _parent, _index, iid=None, **_kwargs) -> None:
        self._rows.append(str(iid))

    def delete(self, item: str) -> None:
        if item in self._rows:
            self._rows.remove(item)
        if item in self._selection:
            self._selection.remove(item)
            self._queue()

    def exists(self, item: str) -> bool:
        return item in self._rows

    def _queue(self) -> None:
        if self.on_select is not None:
            self.pending += 1

    def drain(self) -> None:
        pending, self.pending = self.pending, 0
        for _ in range(pending):
            self.on_select()


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

    # the real _part_selection_changed lives in PartEditingMixin; like the real
    # one, it decides nothing about ownership
    def _part_selection_changed(self) -> None:
        pass

    def drain(self) -> None:
        """Run the queued handlers, as the tk event loop would."""
        self.part_tree.drain()
        self.trigger_tree.drain()

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

    def _trigger_offset_display(self, row):
        return "N/A"


MESH = "some_mesh"
BOX = "trig|hood_int|0.0|0.0|0.0"


class ClickOwnershipTests(unittest.TestCase):
    """A plain click starts fresh; Ctrl and Shift gather."""

    def setUp(self) -> None:
        self.app = Harness()
        self.app.trigger_tree.insert("", "end", iid=BOX)

    def test_a_plain_click_on_a_trigger_drops_the_mesh_selection(self) -> None:
        self.app.part_tree.selection_set([MESH])
        self.app._selection_accumulating = False
        self.app._claim_selection("triggers", BOX)
        self.app.trigger_tree.selection_set([BOX])
        self.app.drain()
        self.assertEqual(self.app.part_tree.selection(), [])
        self.assertEqual(self.app.trigger_tree.selection(), [BOX])

    def test_a_plain_click_on_a_mesh_drops_the_trigger_selection(self) -> None:
        self.app.trigger_tree.selection_set([BOX])
        self.app._selection_accumulating = False
        self.app._claim_selection("parts", MESH)
        self.app.part_tree.selection_set([MESH])
        self.app.drain()
        self.assertEqual(self.app.trigger_tree.selection(), [])
        self.assertEqual(self.app.part_tree.selection(), [MESH])

    def test_a_gathering_click_leaves_both_tables_holding_rows(self) -> None:
        self.app.part_tree.selection_set([MESH])
        self.app._selection_accumulating = True  # Ctrl or Shift held
        self.app._claim_selection("triggers", BOX)
        self.app.trigger_tree.selection_set([BOX])
        self.app.drain()
        self.assertEqual(self.app.part_tree.selection(), [MESH])
        self.assertEqual(self.app.trigger_tree.selection(), [BOX])

    def test_clicking_inside_a_selection_this_table_holds_is_an_edit(self) -> None:
        # Opening the Move X editor on a row that is already selected must not
        # be read as starting a new selection, or the gathered meshes go.
        self.app.part_tree.selection_set([MESH])
        self.app.trigger_tree.selection_set([BOX])
        self.app._selection_accumulating = False
        self.app._claim_selection("triggers", BOX)
        self.app.drain()
        self.assertEqual(self.app.part_tree.selection(), [MESH])


class RebuildDoesNotStealTests(unittest.TestCase):
    """The bug this file exists for, in its current form.

    A table refresh deletes and re-inserts its rows, and every one of those
    changes queues a selection event. None of them may cost the other table
    the rows the user gathered.
    """

    def setUp(self) -> None:
        self.app = Harness()
        self.app.trigger_tree.insert("", "end", iid=BOX)

    def test_rebuilding_the_trigger_table_keeps_the_mesh_selection(self) -> None:
        self.app.trigger_tree.selection_set([BOX])
        self.app._selection_accumulating = True
        self.app.part_tree.selection_set([MESH])
        self.app.drain()

        self.app.trigger_tree.delete(BOX)          # the refresh empties it
        self.app.trigger_tree.insert("", "end", iid=BOX)
        self.app.trigger_tree.selection_set([BOX])  # and puts the rows back
        self.app.drain()

        self.assertEqual(self.app.part_tree.selection(), [MESH])
        self.assertEqual(self.app.trigger_tree.selection(), [BOX])

    def test_a_refresh_never_reaches_the_ownership_decision(self) -> None:
        # Nothing but a click may clear a table, so a rebuild cannot -- however
        # many events it queues, and whatever order they arrive in.
        self.app.part_tree.selection_set([MESH])
        self.app.trigger_tree.selection_set([BOX])
        self.app._refresh_triggers()
        self.app.drain()
        self.assertEqual(self.app.part_tree.selection(), [MESH])


class ClearAllTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = Harness()

    def test_an_empty_preview_click_clears_both(self) -> None:
        self.app.part_tree.selection_set([MESH])
        self.app.trigger_tree.selection_set([BOX])
        self.app._clear_all_selection()
        self.app.drain()
        self.assertEqual(self.app.part_tree.selection(), [])
        self.assertEqual(self.app.trigger_tree.selection(), [])
        self.assertGreater(self.app.detail_refreshes, 0)


class SceneHighlightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = Harness()

    def test_the_highlight_covers_both_tables_at_once(self) -> None:
        # Gathered rows halo together in the preview; the highlight is no
        # longer a question of which single table won.
        self.app.trigger_tree.selection_set([BOX])
        self.app.trigger_rows_by_iid = {BOX: {}}
        self.app._selection_accumulating = True
        self.app.part_tree.selection_set([MESH])
        self.app.drain()
        self.assertEqual(self.app._selected_trigger_scene_ids(), {BOX})
        self.assertEqual(self.app._selected_preview_ids(), {MESH})


if __name__ == "__main__":
    unittest.main()
