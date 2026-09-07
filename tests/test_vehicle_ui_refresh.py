"""2026-09-06: Vehicle switches must converge despite late table and preview workers."""

import unittest
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from beamxp.hand_drive_ui.parts_workflow import PartsWorkflowMixin
from beamxp.hand_drive_ui.vehicle_workflow import VehicleWorkflowMixin
from beamxp.hand_drive_ui.worker_handlers import WorkerHandlersMixin


class Tree:
    def __init__(self):
        self.rows = ["shared_mesh", "old_vehicle_only"]
        self.selected = ["shared_mesh"]
        self.focused = "shared_mesh"

    def get_children(self):
        return tuple(self.rows)

    def delete(self, *rows):
        self.rows = [row for row in self.rows if row not in rows]
        self.selected = [row for row in self.selected if row not in rows]

    def focus(self, row):
        self.focused = row


class VehicleSwitchTests(unittest.TestCase):
    def test_accepting_a_vehicle_clears_old_rows_before_any_refresh(self):
        app = Mock()
        app.vehicle_load_seq = 2
        app.part_refresh_seq = 3
        app.part_table_seq = 4
        app.mesh_scene_seq = 5
        app.part_refresh_running = app.part_table_running = app.mesh_scene_running = True
        app.part_refresh_after_id = "parts_idle"
        app.mesh_scene_after = "scene_idle"
        app._preview_hover_after = "hover_poll"
        app.box_preview_by_id = {"shared_mesh": "old_position"}
        app.settings = {}
        app.part_table_snapshot = {"rows": ["old_vehicle_only"]}
        app.part_table_snapshot_key = "old_table"
        app.preview_output_hover = "old_config"
        trees = [Tree() for _ in range(4)]
        app.variant_tree, app.part_tree, app.slot_tree, app.trigger_tree = trees
        app._reset_vehicle_ui_state = lambda: VehicleWorkflowMixin._reset_vehicle_ui_state(app)
        context = SimpleNamespace(variants={"trim": object()}, objects={},
                                  vehicle_id="new", project_dir=Path("new_project"))

        def check_refresh(**_kwargs):
            self.assertIs(app.context, context)
            for tree in trees:
                self.assertEqual(tree.rows, [])
                self.assertEqual(tree.selected, [])
                self.assertEqual(tree.focused, "")
            self.assertEqual(app.current_part_ids, [])
            self.assertEqual(app.part_row_mesh_ids, {})
            self.assertEqual(app.part_row_side_refs, {})
            self.assertEqual(app.part_row_positions, {})
            self.assertEqual(app.part_instance_rows, {})
            self.assertEqual(app.part_row_labels, {})
            self.assertEqual(app.part_child_overrides, {})
            self.assertEqual(app.side_pair_rows_by_iid, {})
            self.assertEqual(app.trigger_rows_by_iid, {})
            self.assertIsNone(app.side_pair_pick_target)
            self.assertIsNone(app.part_table_snapshot)
            self.assertIsNone(app.part_table_snapshot_key)
            self.assertEqual(app.box_preview_by_id, {})
            self.assertIsNone(app._box_preview_config)
            self.assertIsNone(app.preview_output_hover)

        app._refresh_all.side_effect = check_refresh
        with (
            patch("beamxp.hand_drive_ui.vehicle_workflow.core.load_cached_variant_hands",
                  return_value={"trim": "LHD"}),
            patch("beamxp.hand_drive_ui.vehicle_workflow.core.save_app_settings"),
        ):
            VehicleWorkflowMixin._handle_vehicle_load_success(
                app, (2, Path("new.zip"), "new", context, {}, True)
            )

        app._refresh_all.assert_called_once_with(reset_view=True)
        app._invalidate_slot_usage.assert_called_once()
        app._invalidate_trigger_cache.assert_called_once()
        self.assertEqual(app.part_refresh_seq, 4)
        self.assertEqual(app.part_table_seq, 5)
        self.assertEqual(app.mesh_scene_seq, 6)
        self.assertTrue(app.part_refresh_running)
        self.assertTrue(app.part_table_running)
        self.assertTrue(app.mesh_scene_running)
        self.assertCountEqual([call.args[0] for call in app.after_cancel.call_args_list],
                              ["parts_idle", "scene_idle", "hover_poll"])

    def test_rejected_vehicle_load_leaves_the_current_ui_intact(self):
        app = Mock(vehicle_load_seq=2)
        VehicleWorkflowMixin._handle_vehicle_load_success(app, (1, None, None, None, None, None))
        app._reset_vehicle_ui_state.assert_not_called()
        app._refresh_all.assert_not_called()


class PreviewTableReadinessTests(unittest.TestCase):
    def make_app(self):
        app = Mock()
        app.viewer.scene = SimpleNamespace(pick_to_row={}, extra=())
        app._resolved_visible_ids.return_value = {"body", "wheel"}
        app._selected_part_mesh_ids.return_value = set()
        app._selected_preview_ids.return_value = set()
        app._selected_trigger_scene_ids.return_value = set()
        app._part_row_mesh_id.side_effect = lambda row: row
        app._part_table_snapshot_key.return_value = "new_table"
        app.current_part_ids = []
        app.part_table_snapshot = None
        app.part_table_snapshot_key = None
        return app

    def test_scene_arriving_before_table_is_not_dimmed(self):
        app = self.make_app()
        PartsWorkflowMixin._refresh_viewer(app)
        app.viewer.set_dimmed_ids.assert_called_once_with(set())
        self.assertCountEqual(app.viewer.set_visible_ids.call_args.args[0], ["body", "wheel"])

    def test_stale_table_cannot_dim_the_current_scene(self):
        app = self.make_app()
        app.part_table_snapshot = {"rows": []}
        app.part_table_snapshot_key = "old_table"
        PartsWorkflowMixin._refresh_viewer(app)
        app.viewer.set_dimmed_ids.assert_called_once_with(set())

    def test_ready_table_still_dims_parts_excluded_by_filter(self):
        app = self.make_app()
        app.part_table_snapshot = {"rows": ["body", "wheel"]}
        app.part_table_snapshot_key = "new_table"
        for rows, dimmed in ((["body"], {"wheel"}), ([], {"body", "wheel"})):
            with self.subTest(rows=rows):
                app.current_part_ids = rows
                PartsWorkflowMixin._refresh_viewer(app)
                app.viewer.set_dimmed_ids.assert_called_with(dimmed)


class PartTableCompletionTests(unittest.TestCase):
    def make_app(self):
        app = Mock()
        app.context = object()
        app.part_table_seq = 2
        app.part_table_running = True
        app.part_table_requested_key = "old_table"
        app.part_table_snapshot = None
        app.part_table_snapshot_key = None
        app.part_table_pending = False
        app.part_table_pending_reset = True
        app.current_part_ids = []
        app._part_table_snapshot_key.return_value = "new_table"
        app._schedule_pending_part_table_snapshot = (
            lambda: PartsWorkflowMixin._schedule_pending_part_table_snapshot(app)
        )
        return app

    def test_stale_completion_requests_current_table_without_a_pending_flag(self):
        for previous_vehicle in (False, True):
            for failed in (False, True):
                with self.subTest(previous_vehicle=previous_vehicle, failed=failed):
                    app = self.make_app()
                    future = Future()
                    if failed:
                        future.set_exception(RuntimeError("old worker failed"))
                    else:
                        future.set_result({"rows": ["old_vehicle_only"]})
                    context = object() if previous_vehicle else app.context
                    WorkerHandlersMixin._handle_part_table_success(
                        app, (2, context, "old_table", future)
                    )
                    app._schedule_part_table_snapshot.assert_called_once_with(reset_view=True)
                    app._refresh_parts.assert_not_called()
                    self.assertIsNone(app.part_table_snapshot)
                    self.assertFalse(app.part_table_running)

    def test_current_completion_populates_the_table_and_finishes_loading(self):
        app = self.make_app()
        future = Future()
        snapshot = {"rows": ["new_mesh"]}
        future.set_result(snapshot)
        WorkerHandlersMixin._handle_part_table_success(app, (2, app.context, "new_table", future))
        self.assertIs(app.part_table_snapshot, snapshot)
        app._refresh_parts.assert_called_once_with(reset_view=True)
        app._schedule_part_table_snapshot.assert_not_called()

    def test_current_worker_failure_is_reported_without_a_retry_loop(self):
        app = self.make_app()
        future = Future()
        future.set_exception(RuntimeError("broken table"))
        WorkerHandlersMixin._handle_part_table_success(app, (2, app.context, "new_table", future))
        app.status_var.set.assert_called_once_with("Part table refresh failed: broken table")
        app._schedule_part_table_snapshot.assert_not_called()


if __name__ == "__main__":
    unittest.main()
