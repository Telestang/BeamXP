"""Node group membership, and the flexbodies the engine drops because of it.

meshs.lua ignores a flexbody whose ``_group_nodes`` is empty, and links.lua
fills that from the assembled ``nodes`` table. Stock content leans on this:
deleting a part removes the nodes its group owned, and every mesh bound only to
that group silently stops spawning.
"""

from __future__ import annotations

import unittest

from beamxp import hand_drive_core as core


class GroupNameTests(unittest.TestCase):
    def test_a_string_names_one_group(self) -> None:
        self.assertEqual(core.jbeam_group_names("body"), ("body",))

    def test_a_table_names_several(self) -> None:
        self.assertEqual(core.jbeam_group_names(["a", "b"]), ("a", "b"))

    def test_empty_values_name_nothing(self) -> None:
        # {"group":""} is the documented reset.
        self.assertEqual(core.jbeam_group_names(""), ())
        self.assertEqual(core.jbeam_group_names([]), ())
        self.assertEqual(core.jbeam_group_names(["", "a"]), ("a",))
        self.assertEqual(core.jbeam_group_names(None), ())


class TableRowTests(unittest.TestCase):
    def test_modifier_rows_persist_until_changed(self) -> None:
        rows = list(
            core.iter_jbeam_table_rows(
                '[["id","posX"],{"group":"body"},["n1",1],["n2",2],'
                '{"group":"roof"},["n3",3],{"group":""},["n4",4]]'
            )
        )
        self.assertEqual(
            [(row["id"], row.get("group")) for row in rows],
            [("n1", "body"), ("n2", "body"), ("n3", "roof"), ("n4", "")],
        )

    def test_inline_options_apply_to_one_row_only(self) -> None:
        rows = list(
            core.iter_jbeam_table_rows(
                '[["id","posX"],["n1",1,{"group":"solo"}],["n2",2]]'
            )
        )
        self.assertEqual(rows[0].get("group"), "solo")
        self.assertIsNone(rows[1].get("group"))

    def test_unreadable_text_yields_nothing(self) -> None:
        self.assertEqual(list(core.iter_jbeam_table_rows("[[")), [])


class NodeGroupTests(unittest.TestCase):
    NODES = (
        '[["id","posX","posY","posZ"],'
        '{"group":["door","doorpanel2"]},'
        '["d1",0,0,0],'
        '{"group":""},'
        '["d2",1,1,1]]'
    )

    def test_groups_come_from_the_rows_they_cover(self) -> None:
        self.assertEqual(core.node_group_names(self.NODES), {"door", "doorpanel2"})

    def test_a_group_with_no_rows_after_it_owns_nothing(self) -> None:
        nodes = '[["id","posX","posY","posZ"],["n1",0,0,0],{"group":"late"}]'
        self.assertEqual(core.node_group_names(nodes), set())

    def test_the_header_row_is_not_a_node(self) -> None:
        nodes = '[["id","posX","posY","posZ"],{"group":"g"},["n1",0,0,0]]'
        self.assertEqual(core.node_group_names(nodes), {"g"})

    def test_a_dict_first_row_is_not_a_usable_table(self) -> None:
        # tableSchema.lua rejects it outright: "Invalid table header, must be a
        # list, not a dict", so there is no table to read groups from.
        nodes = '[{"group":"g"},["id","posX","posY","posZ"],["n1",0,0,0]]'
        self.assertEqual(core.node_group_names(nodes), set())


class WheelGroupTests(unittest.TestCase):
    def test_generated_wheel_nodes_count_as_populating_their_groups(self) -> None:
        # Brake discs and hubcaps bind to these; the nodes appear in no
        # ``nodes`` section, so without this they would look unbound.
        wheels = (
            '[["name","hubGroup","group","hubcapGroup"],'
            '["FL","hub_FL","wheel_FL","hubcap_FL"]]'
        )
        self.assertEqual(
            core.wheel_group_names(wheels), {"hub_FL", "wheel_FL", "hubcap_FL"}
        )


class FlexbodyRowTests(unittest.TestCase):
    def test_group_column_is_read_as_a_list_or_string(self) -> None:
        self.assertEqual(core.flexbody_row_groups('["m", ["a","b"]]'), ("a", "b"))
        self.assertEqual(core.flexbody_row_groups('["m", "a"]'), ("a",))

    def test_a_row_without_a_group_column_is_undetermined(self) -> None:
        # None means "do not filter on this".
        self.assertIsNone(core.flexbody_row_groups('["m"]'))
        self.assertIsNone(core.flexbody_row_groups("not a row"))


def part(part_id: str, slot_type: str, body: str) -> str:
    return f'"{part_id}": {{\n"slotType":"{slot_type}"{body}\n}}'


class FlexbodyDropTests(unittest.TestCase):
    """The BX case: a deleted part takes its node group, and the meshes with it."""

    def _context(self, doorpanel_installed: bool) -> core.VehicleContext:
        index = {
            "car": (
                part("car", "main", ',\n"slots":[["type","default","description"],'
                     '["dash","dash","Dash"],["panel","panel","Panel"]]'),
                "car.jbeam",
            ),
            # The dashboard hangs its controls mesh off the panel's node group.
            "dash": (
                part("dash", "dash", ',\n"flexbodies":[["mesh","[group]:"],'
                     '["dash_mesh",["cabin"]],["door_controls",["panel_nodes"]]]'
                     ',\n"nodes":[["id","posX","posY","posZ"],{"group":"cabin"},["c1",0,0,0]]'),
                "car.jbeam",
            ),
            # ...and only this part supplies nodes for that group.
            "panel": (
                part("panel", "panel", ',\n"flexbodies":[["mesh","[group]:"],'
                     '["panel_mesh",["panel_nodes"]]]'
                     ',\n"nodes":[["id","posX","posY","posZ"],'
                     '{"group":["panel_nodes"]},["p1",0,0,0]]'),
                "car.jbeam",
            ),
        }
        instances = [
            {"part_id": "car", "slot_id": "main", "slot_path": "/"},
            {"part_id": "dash", "slot_id": "dash", "slot_path": "/dash/"},
        ]
        parts = {"car", "dash"}
        if doorpanel_installed:
            instances.append({"part_id": "panel", "slot_id": "panel", "slot_path": "/panel/"})
            parts.add("panel")

        context = core.VehicleContext(
            source_zip=core.Path("test.zip"),
            vehicle_id="car",
            vehicle_path="vehicles/car",
            dae_paths=[],
            variants={"trim": core.VariantInfo("trim", "trim.pc", None, "Trim")},
            objects={},
            preview_by_id={},
            jbeam_texts={},
            node_positions={},
            project_dir=core.Path("project"),
            part_body_index=index,
        )
        context.selected_parts_cache["trim"] = {
            "parts": parts,
            "part_instances": instances,
            "selected_by_slot": {},
            "selected_by_path": {},
            "part_slot_options": {},
        }
        return context

    def test_meshes_survive_while_the_group_has_nodes(self) -> None:
        context = self._context(doorpanel_installed=True)
        self.assertIn("panel_nodes", core.populated_node_groups(context, "trim"))
        _flex, _props, meshes = core.mesh_roles_for_config(context, "trim")
        self.assertEqual(meshes, {"dash_mesh", "door_controls", "panel_mesh"})

    def test_deleting_the_part_drops_every_mesh_bound_to_its_group(self) -> None:
        context = self._context(doorpanel_installed=False)
        self.assertNotIn("panel_nodes", core.populated_node_groups(context, "trim"))
        _flex, _props, meshes = core.mesh_roles_for_config(context, "trim")
        # door_controls is declared by the still-installed dash, but its only
        # node group went with the panel, so the engine never spawns it.
        self.assertEqual(meshes, {"dash_mesh"})


if __name__ == "__main__":
    unittest.main()
