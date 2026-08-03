from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from beamxp.core.models import MeshTransformInstance
from beamxp.hand_drive_ui.parts_workflow import PartsWorkflowMixin


def instance(
    mesh: str,
    slot: str,
    path: str,
    *,
    count: int = 1,
    ordinal: int = 1,
) -> MeshTransformInstance:
    return MeshTransformInstance(
        instance_id=f"{mesh}@@{path}",
        mesh_id=mesh,
        part_id=slot,
        slot_id=slot,
        slot_path=path,
        position=(0.0, 0.0, 0.0),
        count_for_mesh=count,
        ordinal_for_mesh=ordinal,
    )


class _Workflow(PartsWorkflowMixin):
    def __init__(self, focused_config: str = "a") -> None:
        self.context = SimpleNamespace(
            variants={"a": object(), "b": object(), "c": object()},
            objects={"mesh": object()},
        )
        self.mesh_instance_numbering_key = None
        self.mesh_instance_numbering_cache = {}
        self.focused_config = focused_config

    def _mesh_scene_config(self) -> str:
        return self.focused_config


class _Tree:
    def __init__(self, selected: list[str]) -> None:
        self._selected = selected

    def selection(self) -> list[str]:
        return self._selected

    def exists(self, row_id: object) -> bool:
        return str(row_id) in self._selected


class PartsWorkflowNumberingTests(unittest.TestCase):
    def test_alternative_config_slots_do_not_create_single_hash_one_rows(self) -> None:
        workflow = _Workflow()
        by_config = {
            "a": [instance("mesh", "slot_a", "/slot_a/")],
            "b": [instance("mesh", "slot_b", "/slot_b/")],
            "c": [instance("mesh", "slot_a", "/slot_a/")],
        }

        with patch(
            "beamxp.hand_drive_ui.parts_workflow.core.selected_mesh_transform_instances_for_config",
            side_effect=lambda _context, config: by_config[config],
        ):
            rows = workflow._part_table_rows(["mesh"])

        self.assertEqual([row["row_id"] for row in rows], ["mesh"])
        self.assertEqual(rows[0]["display_count_for_mesh"], 1)

    def test_true_duplicate_instances_use_stable_numbered_rows(self) -> None:
        workflow = _Workflow()
        by_config = {
            "a": [
                instance("mesh", "slot_l", "/slot_l/", count=2, ordinal=1),
                instance("mesh", "slot_r", "/slot_r/", count=2, ordinal=2),
            ],
            "b": [
                instance("mesh", "slot_r", "/slot_r/", count=2, ordinal=1),
                instance("mesh", "slot_l", "/slot_l/", count=2, ordinal=2),
            ],
            "c": [
                instance("mesh", "slot_l", "/slot_l/", count=2, ordinal=1),
                instance("mesh", "slot_r", "/slot_r/", count=2, ordinal=2),
            ],
        }

        with patch(
            "beamxp.hand_drive_ui.parts_workflow.core.selected_mesh_transform_instances_for_config",
            side_effect=lambda _context, config: by_config[config],
        ):
            rows = workflow._part_table_rows(["mesh"])

        self.assertEqual([row["row_id"] for row in rows], ["mesh@@1", "mesh@@2"])
        self.assertEqual([row["display_count_for_mesh"] for row in rows], [2, 2])

    def test_single_instance_keeps_number_when_slot_belongs_to_duplicate_layout(self) -> None:
        workflow = _Workflow(focused_config="c")
        by_config = {
            "a": [
                instance("mesh", "slot_l", "/slot_l/", count=2, ordinal=1),
                instance("mesh", "slot_r", "/slot_r/", count=2, ordinal=2),
            ],
            "b": [
                instance("mesh", "slot_l", "/slot_l/", count=2, ordinal=1),
                instance("mesh", "slot_r", "/slot_r/", count=2, ordinal=2),
            ],
            "c": [instance("mesh", "slot_r", "/slot_r/")],
        }

        with patch(
            "beamxp.hand_drive_ui.parts_workflow.core.selected_mesh_transform_instances_for_config",
            side_effect=lambda _context, config: by_config[config],
        ):
            rows = workflow._part_table_rows(["mesh"])

        self.assertEqual([row["row_id"] for row in rows], ["mesh@@2"])
        self.assertEqual(rows[0]["display_count_for_mesh"], 2)

    def test_numbered_singleton_selection_highlights_base_mesh_only_when_single_span(self) -> None:
        workflow = _Workflow()
        workflow.part_tree = _Tree(["mesh@@1"])
        workflow.part_row_mesh_ids = {"mesh@@1": "mesh"}
        workflow.part_row_side_refs = {"mesh@@1": "mesh@@/slot_l/"}
        workflow.viewer = SimpleNamespace(
            scene=SimpleNamespace(groups={"mesh": [(0, 1, 0, 3)]}, pick_to_row={})
        )

        self.assertIn("mesh", workflow._selected_preview_ids())

    def test_numbered_duplicate_selection_does_not_highlight_base_mesh(self) -> None:
        workflow = _Workflow()
        workflow.part_tree = _Tree(["mesh@@1"])
        workflow.part_row_mesh_ids = {"mesh@@1": "mesh"}
        workflow.part_row_side_refs = {"mesh@@1": "mesh@@/slot_l/"}
        workflow.viewer = SimpleNamespace(
            scene=SimpleNamespace(
                groups={
                    "mesh": [(0, 1, 0, 3), (1, 2, 3, 6)],
                    "mesh@@/slot_l/": [(0, 1, 0, 3)],
                },
                pick_to_row={},
            )
        )

        selected = workflow._selected_preview_ids()
        self.assertIn("mesh@@/slot_l/", selected)
        self.assertNotIn("mesh", selected)


if __name__ == "__main__":
    unittest.main()
