from __future__ import annotations

import unittest

from beamxp import hand_drive_core as core


ROOT = r"""
"veh": {
    "slotType": "main",
    "slots": [
        ["type", "default", "description"],
        ["mount_L", "shared", "Left", {
            "nodeMove": {"x": 2.0},
            "variables": {"$prefix": "a_"}
        }],
        ["mount_R", "shared", "Right", {
            "nodeMove": {"x": -2.0},
            "variables": {"$prefix": "b_"}
        }]
    ]
}
"""

SHARED = r"""
"shared": {
    "slotType": ["mount_L", "mount_R"],
    "nodes": [
        ["id", "posX", "posY", "posZ"],
        ["$=$prefix..'ref'", 0, 0, 0],
        ["$=$prefix..'x'", 1, 0, 0],
        ["$=$prefix..'y'", 0, 1, 0]
    ]
}
"""


class DuplicatePartInstanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = core.build_part_body_index(
            {"vehicles/veh/veh.jbeam": "{" + ROOT + "," + SHARED + "}"}
        )

    def test_two_occurrences_keep_separate_paths_options_and_nodes(self) -> None:
        selected = core.resolve_selected_parts(
            {"mainPartName": "veh", "parts": {}},
            {},
            vehicle_id="veh",
            part_body_index=self.index,
        )
        shared = [
            item for item in core.selected_part_instances(selected)
            if item["part_id"] == "shared"
        ]
        self.assertEqual(len(shared), 2)
        self.assertEqual({item["slot_path"] for item in shared}, {"/mount_L/", "/mount_R/"})
        self.assertEqual(
            {core.part_instance_variable_scope(selected, item)["$prefix"] for item in shared},
            {"a_", "b_"},
        )

        nodes = core.selected_node_positions_for_parts(selected, {}, self.index)
        self.assertEqual(nodes["a_ref"], (2.0, 0.0, 0.0))
        self.assertEqual(nodes["b_ref"], (-2.0, 0.0, 0.0))
        self.assertEqual(nodes["a_x"], (3.0, 0.0, 0.0))
        self.assertEqual(nodes["b_x"], (-1.0, 0.0, 0.0))

    def test_cycle_is_stopped(self) -> None:
        cyclic = core.build_part_body_index(
            {"vehicles/veh/cycle.jbeam": r"""{
                "veh":{"slotType":"main","slots":[
                    ["type","default","description"],
                    ["child","child","Child"]]},
                "child":{"slotType":"child","slots":[
                    ["type","default","description"],
                    ["back","veh","Back"]]}
            }"""}
        )
        selected = core.resolve_selected_parts(
            {"mainPartName": "veh", "parts": {}},
            {},
            vehicle_id="veh",
            part_body_index=cyclic,
        )
        self.assertTrue(selected["cycles"])
        self.assertEqual({item["part_id"] for item in selected["part_instances"]}, {"veh", "child"})


if __name__ == "__main__":
    unittest.main()
