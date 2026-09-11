"""2026-09-09: Trigger geometry must use the trim's assembled reference nodes."""

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from beamxp import hand_drive_core as core
from beamxp.hand_drive_parts.configuration import _migrated_trigger_modes
from beamxp.hand_drive_ui.triggers_workflow import TriggersWorkflowMixin


def scoped_context():
    triggers = [
        ["id", "idRef:", "idX:", "idY:", "type", "size", "baseTranslation"],
        ["switch", "ref", "x", "y", "sphere", 0.02, {"x": 0.2, "y": 0.3, "z": 0.4}],
        ["inactive", "absent", "x", "y", "sphere", 0.02, {"x": 0, "y": 0, "z": 0}],
    ]
    parts = {
        name: {
            "slotType": "main",
            "slots": [["type", "default", "description"],
                      ["frame", "frame", "Frame", {"nodeMove": {"y": offset}}],
                      ["controls", "controls", "Controls"]],
        }
        for name, offset in (("road", 0), ("raised", 2))
    }
    parts["frame"] = {
        "slotType": "frame",
        "nodes": [["id", "posX", "posY", "posZ"],
                  ["ref", 1, 0, 0], ["x", 2, 0, 0], ["y", 1, 1, 0]],
    }
    parts["controls"] = {"slotType": "controls", "triggers2": triggers}
    texts = {"vehicles/acme/test.jbeam": json.dumps(parts)}
    context = core.VehicleContext(
        source_zip=Path("unused.zip"), vehicle_id="acme", vehicle_path="vehicles/acme/",
        dae_paths=[], objects={}, preview_by_id={}, jbeam_texts=texts,
        node_positions={"ref": (9, 0, 0), "x": (10, 0, 0), "y": (9, 1, 0),
                        "absent": (8, 0, 0)},
        project_dir=Path("."), part_body_index=core.build_part_body_index(texts),
        variants={name: core.VariantInfo(name, name + ".pc", None, name)
                  for name in ("road", "road_copy", "raised")},
    )
    for name in context.variants:
        context.pc_cache[name + ".pc"] = {
            "mainPartName": "road" if name == "road_copy" else name,
            "parts": {},
        }
    return context


class TriggerPlacementScopeTests(unittest.TestCase):
    def test_preview_uses_selected_nodes_and_omits_inactive_refs(self):
        context = scoped_context()
        for name, expected_y in (("road", 0.3), ("raised", 2.3)):
            with self.subTest(config=name):
                selected = core.selected_parts_for_config(context, name)
                boxes = core.preview_trigger_boxes(context, {}, selected)
                self.assertEqual([box["id"] for box in boxes], ["switch"])
                centre = [sum(v[i] for v in boxes[0]["vertices_stock"]) /
                          len(boxes[0]["vertices_stock"]) for i in range(3)]
                for actual, expected in zip(centre, (1.2, expected_y, -0.4)):
                    self.assertAlmostEqual(actual, expected)

    def test_table_keeps_shared_part_at_distinct_trim_positions(self):
        context = scoped_context()
        boxes = TriggersWorkflowMixin._collect_trigger_boxes(
            context, ("road", "road_copy", "raised")
        )
        self.assertEqual(len(boxes), 2)
        self.assertEqual({box["at"] for box in boxes}, {(1.2, 0.3, -0.4), (1.2, 2.3, -0.4)})
        self.assertTrue(all(box["parts"] == ["controls"] for box in boxes))

    def test_table_choice_matches_preview_and_export_identity(self):
        context = scoped_context()
        table = TriggersWorkflowMixin._collect_trigger_boxes(context, ("road",))[0]
        conversion = {}
        core.set_trigger_mode(conversion, table["trigger"], table["at"], core.MODE_MIRROR)
        selected = core.selected_parts_for_config(context, "road")
        box = core.preview_trigger_boxes(context, conversion, selected)[0]
        self.assertEqual(box["action"], "mirror")
        for stock, converted in zip(box["vertices_stock"], box["vertices"]):
            self.assertEqual(converted, [-stock[0], stock[1], stock[2]])
        body = core.part_body_for_context(context, "controls")[0]
        positions = core.authored_trigger_positions(
            body, core.selected_node_positions_for_config(context, "road")
        )
        self.assertEqual(core.trigger_position_key("switch", positions["switch"]), table["key"])

    def test_owner_prediction_uses_the_same_trim_nodes(self):
        context = scoped_context()
        app = TriggersWorkflowMixin()
        app.context = context
        app.conversion = {}
        app._selected_variant_names = lambda: ["raised"]
        with patch.object(core, "trigger_owners_for_config", return_value=([], {}, [])) as owners:
            app._trigger_auto_owners()
        self.assertEqual(owners.call_args.args[-1]["ref"], (1.0, 2.0, 0.0))

    def test_old_choices_follow_each_resolved_position_without_losing_offsets(self):
        context = scoped_context()
        saved = [{"id": "switch", "at": [9.2, 0.3, -0.4],
                  "mode": core.MODE_TRANSLATE, "offset": -0.42}]
        migrated = _migrated_trigger_modes(context, saved)
        self.assertEqual({tuple(row["at"]) for row in migrated},
                         {(1.2, 0.3, -0.4), (1.2, 2.3, -0.4)})
        self.assertTrue(all(row["mode"] == core.MODE_TRANSLATE and row["offset"] == -0.42
                            for row in migrated))
        self.assertEqual(_migrated_trigger_modes(context, migrated), migrated)
        self.assertEqual(saved[0]["at"], [9.2, 0.3, -0.4])

    def test_migration_preserves_corrected_choices_and_unknown_records(self):
        context = scoped_context()
        current = {"id": "switch", "at": [1.2, 0.3, -0.4], "mode": core.MODE_SKIP}
        unknown = {"id": "other", "at": [8, 8, 8], "mode": core.MODE_MIRROR}
        saved = [{"id": "switch", "at": [9.2, 0.3, -0.4], "mode": core.MODE_MIRROR},
                 current, unknown]
        migrated = _migrated_trigger_modes(context, saved)
        self.assertIn(current, migrated)
        self.assertIn(unknown, migrated)
        self.assertEqual(len(migrated), 3)


if __name__ == "__main__":
    unittest.main()
