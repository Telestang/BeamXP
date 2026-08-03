from __future__ import annotations

import unittest

from beamxp import hand_drive_core as core
from beamxp import transform_helpers as th

# Stock content ships part keys with a stray comma between the colon and the
# brace ("bluebuck_bumper_F":, {...}); the game's lenient parser accepts it.
STRAY_COMMA_PART = """
{
"acme_bumper_F":, {
    "information":{
        "authors":"BeamNG",
        "name":"Front Bumper",
    },
    "slotType" : "acme_bumper_F",
    "slots": [
        ["type", "default", "description"],
        ["acme_bumperguards_F","", "Front Bumper Guards"],
    ],
    "flexbodies": [
         ["mesh", "[group]:"],
         ["acme_bumper_F", ["acme_bumper_F"]],
    ],
},
}
"""

# pickup_bumper_F_prefacelift.jbeam ships a commented-out slots2 row with an
# unbalanced quote; the whole vehicle used to fail with "Unclosed [] block".
MALFORMED_COMMENTED_ROW_ARRAY = """[
    ["name", "allowTypes", "denyTypes", "default", "description"],
    ["acme_lip_F", ["acme_lip_F"], [], "", "Front Lip"],
    //["acme_guards_F"","","Front Bumper Guards"]
]"""


class StrayCommaPartKeyTests(unittest.TestCase):
    def test_part_body_index_tolerates_stray_comma(self) -> None:
        index = core.build_part_body_index({"vehicles/acme/a.jbeam": STRAY_COMMA_PART})
        self.assertIn("acme_bumper_F", index)
        body, filename = index["acme_bumper_F"]
        self.assertEqual(filename, "vehicles/acme/a.jbeam")
        self.assertIn('"slotType"', body)

    def test_extract_keyed_object_tolerates_stray_comma(self) -> None:
        body = th.extract_keyed_object(STRAY_COMMA_PART, "acme_bumper_F")
        self.assertIsNotNone(body)
        self.assertIn('"slotType"', body)

    def test_named_array_tolerates_stray_comma(self) -> None:
        text = '{"nodes":, [\n["id", "posX", "posY", "posZ"],\n["n1", 0.1, 0.2, 0.3],\n]}'
        array = th.extract_named_array(text, "nodes")
        self.assertIsNotNone(array)
        self.assertEqual(
            core.extract_node_positions_from_array(array),
            {"n1": (0.1, 0.2, 0.3)},
        )

    def test_node_position_index_tolerates_stray_comma(self) -> None:
        text = '{"part": {"nodes":,[\n["id", "posX", "posY", "posZ"],\n["n1", 1.0, 2.0, 3.0],\n]}}'
        nodes = core.build_node_position_index({"vehicles/acme/a.jbeam": text})
        self.assertEqual(nodes, {"n1": (1.0, 2.0, 3.0)})


class StrayCommaBeforeColonTests(unittest.TestCase):
    # Stock etk800 844_police_A.pc ships a key with a comma between the key and
    # its colon ("lightbar_sign",:""); the game's lenient parser accepts it.
    def test_config_with_comma_before_colon_parses(self) -> None:
        text = (
            '{\n"format":2,\n"parts":{\n'
            '    "lightbar_floodlights":"",\n'
            '    "lightbar_sign",:"",\n'
            '    "lightbar_antenna":""\n'
            "}\n}\n"
        )
        parsed = core.parse_beamng_json(text, label="acme.pc")
        self.assertEqual(
            parsed["parts"],
            {"lightbar_floodlights": "", "lightbar_sign": "", "lightbar_antenna": ""},
        )

    def test_commas_inside_strings_are_preserved(self) -> None:
        text = '{"a": "x,: y", "b",: 1}'
        self.assertEqual(
            core.parse_beamng_json(text, label="acme.pc"),
            {"a": "x,: y", "b": 1},
        )


class MalformedRowTests(unittest.TestCase):
    def test_iter_top_level_rows_skips_unbalanced_commented_row(self) -> None:
        rows = core.iter_top_level_rows(MALFORMED_COMMENTED_ROW_ARRAY)
        self.assertEqual(len(rows), 2)
        self.assertIn('"acme_lip_F"', rows[1])

    def test_iter_active_top_level_rows_skips_unbalanced_commented_row(self) -> None:
        rows = core.iter_active_top_level_rows(MALFORMED_COMMENTED_ROW_ARRAY)
        self.assertEqual(len(rows), 2)
        self.assertIn('"acme_lip_F"', rows[1])

    def test_slot_demand_types_survives_malformed_slots2_row(self) -> None:
        body = f'"acme_part": {{\n"slotType": "main",\n"slots2": {MALFORMED_COMMENTED_ROW_ARRAY},\n}}'
        self.assertEqual(core.slot_demand_types(body), {"acme_lip_F"})

    def test_well_formed_commented_rows_are_still_returned(self) -> None:
        # The build path deliberately keeps commented-out rows verbatim.
        array = '[\n["a", 1],\n//["b", 2]\n]'
        rows = core.iter_top_level_rows(array)
        self.assertEqual(rows, ['["a", 1]', '["b", 2]'])

    def test_slot_defs_ignore_commented_out_rows(self) -> None:
        # bluebuck ships //["bluebuck_","bluebuck_", ""] slot rows; the game
        # does not load them, so they must not select phantom parts.
        body = (
            '"acme_bumperguards_F": {\n'
            '"slotType": "acme_bumperguards_F",\n'
            '"slots": [\n'
            '    ["type", "default", "description"],\n'
            '    //["acme_","acme_", ""],\n'
            '    ["acme_trim_F", "acme_trim_F_chrome", "Trim"],\n'
            "],\n"
            "}"
        )
        defs = core.extract_slot_defs(body)
        self.assertEqual(
            [(d.slot_type, d.default_part) for d in defs],
            [("acme_trim_F", "acme_trim_F_chrome")],
        )
        self.assertEqual(core.slot_demand_types(body), {"acme_trim_F"})


class MainPartResolutionTests(unittest.TestCase):
    # A mod whose root part is NOT named after the vehicle id, and whose .pc
    # omits mainPartName -- BeamNG finds the root by slotType "main", so we must
    # too rather than assuming a part literally named "acme" exists.
    ROOT = (
        '"acme_body": {\n'
        '"slotType": "main",\n'
        '"slots": [\n'
        '    ["type", "default", "description"],\n'
        '    ["acme_engine", "acme_engine_v8", "Engine"],\n'
        "],\n"
        "}"
    )
    ENGINE = '"acme_engine_v8": {\n"slotType": "acme_engine",\n}'

    def _index(self) -> dict[str, tuple[str, str]]:
        return core.build_part_body_index(
            {
                "vehicles/acme/body.jbeam": f"{{{self.ROOT},\n{self.ENGINE}}}",
            }
        )

    def test_main_part_found_by_slot_type(self) -> None:
        index = self._index()
        self.assertEqual(core.vehicle_namespace_main_part("acme", index), "acme_body")

    def test_common_part_is_never_the_root(self) -> None:
        # A common part with slotType main must not be picked as the vehicle root.
        index = core.build_part_body_index(
            {"vehicles/common/x.jbeam": '"shared": {"slotType": "main"}'}
        )
        self.assertIsNone(core.vehicle_namespace_main_part("acme", index))

    def test_resolve_uses_slot_type_main_when_pc_omits_it(self) -> None:
        selected = core.resolve_selected_parts(
            {"parts": {}},  # no mainPartName
            {},
            vehicle_id="acme",
            part_body_index=self._index(),
        )
        self.assertEqual(selected["main_part"], "acme_body")
        self.assertIn("acme_engine_v8", selected["parts"])  # default child resolved

    def test_pc_main_part_name_still_wins(self) -> None:
        selected = core.resolve_selected_parts(
            {"mainPartName": "acme_body", "parts": {}},
            {},
            vehicle_id="acme",
            part_body_index=self._index(),
        )
        self.assertEqual(selected["main_part"], "acme_body")


class SlotFitmentAndNoneTests(unittest.TestCase):
    ROOT = (
        '"acme": {\n'
        '"slotType": "main",\n'
        '"slots": [\n'
        '    ["type", "default", "description"],\n'
        '    ["acme_mirror_R", "acme_mirror_R_std", "Right Mirror"],\n'
        '    ["acme_engine", "acme_engine_i4", "Engine"],\n'
        "],\n"
        "}"
    )
    MIRROR = '"acme_mirror_R_std": {\n"slotType": "acme_mirror_R",\n}'
    I4 = '"acme_engine_i4": {\n"slotType": "acme_engine",\n}'
    V8 = '"acme_engine_v8": {\n"slotType": "acme_engine",\n}'
    # a part whose slotType does NOT match the engine slot
    WHEEL = '"acme_wheel": {\n"slotType": "acme_wheel",\n}'

    def _index(self):
        parts = (self.ROOT, self.MIRROR, self.I4, self.V8, self.WHEEL)
        body = "{" + ",".join(parts) + "}"
        return core.build_part_body_index({"vehicles/acme/a.jbeam": body})

    def _resolve(self, parts):
        return core.resolve_selected_parts(
            {"parts": parts}, {}, vehicle_id="acme", part_body_index=self._index()
        )

    def test_none_empties_slot_without_phantom_part(self) -> None:
        sel = self._resolve({"acme_mirror_R": "none"})
        self.assertNotIn("none", sel["parts"])
        self.assertNotIn("none", sel["missing_parts"])
        self.assertNotIn("acme_mirror_R_std", sel["parts"])  # default not used
        self.assertIn("acme_engine_i4", sel["parts"])  # other slot still defaults

    def test_user_choice_overrides_default(self) -> None:
        sel = self._resolve({"acme_engine": "acme_engine_v8"})
        self.assertIn("acme_engine_v8", sel["parts"])
        self.assertNotIn("acme_engine_i4", sel["parts"])

    def test_misfitting_choice_resets_to_default(self) -> None:
        # acme_wheel does not fit the acme_engine slot -> engine falls back.
        sel = self._resolve({"acme_engine": "acme_wheel"})
        self.assertIn("acme_engine_i4", sel["parts"])
        self.assertEqual(sel["selected_by_slot"]["acme_engine"], "acme_engine_i4")

    def test_slots2_allow_deny_parsing(self) -> None:
        body = (
            '"p": {\n"slotType": "main",\n"slots2": [\n'
            '    ["name", "allowTypes", "denyTypes", "default", "description"],\n'
            '    ["eng", ["acme_engine", "acme_engine_alt"], ["banned"], "d", "Engine"],\n'
            "]\n}"
        )
        (slot,) = core.extract_slot_defs(body)
        self.assertEqual(slot.slot_type, "eng")
        self.assertEqual(slot.allow_types, ("acme_engine", "acme_engine_alt"))
        self.assertEqual(slot.deny_types, ("banned",))
        self.assertTrue(core.part_fits_slot(["acme_engine_alt"], slot))
        self.assertFalse(core.part_fits_slot(["banned"], slot))
        self.assertFalse(core.part_fits_slot(["unrelated"], slot))


class SlotPathKeyTests(unittest.TestCase):
    # The same slot id (plate) appears under two different parents; a .pc keys
    # its pick by full path so the two are distinguishable (miramar's ute plate).
    PARTS = (
        (
            '"veh": {\n"slotType": "main",\n"slots": [\n'
            '    ["type", "default", "description"],\n'
            '    ["body", "body", "Body"],\n'
            "]\n}"
        ),
        (
            '"body": {\n"slotType": "body",\n"slots": [\n'
            '    ["type", "default", "description"],\n'
            '    ["plate", "", "Body Plate"],\n'
            '    ["tailgate", "tailgate", "Tailgate"],\n'
            "]\n}"
        ),
        (
            '"tailgate": {\n"slotType": "tailgate",\n"slots": [\n'
            '    ["type", "default", "description"],\n'
            '    ["plate", "", "Tailgate Plate"],\n'
            "]\n}"
        ),
        '"plate": {\n"slotType": "plate",\n}',
    )

    def _index(self):
        return core.build_part_body_index(
            {"vehicles/veh/veh.jbeam": "{" + ",\n".join(self.PARTS) + "}"}
        )

    def test_path_key_targets_the_right_slot_instance(self) -> None:
        # Plate on the tailgate path only; the body plate path stays empty.
        sel = core.resolve_selected_parts(
            {
                "parts": {
                    "/body/plate/": "",
                    "/body/tailgate/plate/": "plate",
                }
            },
            {},
            vehicle_id="veh",
            part_body_index=self._index(),
        )
        self.assertIn("plate", sel["parts"])
        # The raw path strings must not leak into selected_by_slot as slot types.
        self.assertNotIn("/body/plate/", sel["selected_by_slot"])

    def test_bare_id_pick_still_resolves(self) -> None:
        sel = core.resolve_selected_parts(
            {"parts": {"plate": "plate"}},
            {},
            vehicle_id="veh",
            part_body_index=self._index(),
        )
        self.assertIn("plate", sel["parts"])


class SlotOptionMergeTests(unittest.TestCase):
    # jbeam slotSystem merges slot options down the tree (tableMerge): a child
    # slot's nodeOffset overrides only the components it names and keeps the
    # parent's other components, rather than replacing the whole vector.
    def test_parent_and_child_offsets_merge_componentwise(self) -> None:
        parent = '{"nodeOffset":{"y":2.0}}'
        row = '["n1", 1.0, 0.0, 0.0, {"nodeOffset":{"x":5.0}}]'
        pos = core.pos_after_node_transforms(row, (1.0, 0.0, 0.0), (parent,))
        # x from the child (with +x sign), y retained from the parent.
        self.assertEqual(pos, (6.0, 2.0, 0.0))

    def test_child_component_overrides_parent_same_component(self) -> None:
        parent = '{"nodeOffset":{"x":1.0, "z":9.0}}'
        row = '["n1", 1.0, 0.0, 0.0, {"nodeOffset":{"x":5.0}}]'
        pos = core.pos_after_node_transforms(row, (1.0, 0.0, 0.0), (parent,))
        # x overridden to 5, z retained from parent.
        self.assertEqual(pos, (6.0, 0.0, 9.0))

    def test_empty_child_offset_resets(self) -> None:
        # An empty nodeOffset object clears the inherited one (jbeam reset row).
        parent = '{"nodeOffset":{"x":5.0}}'
        row = '["n1", 1.0, 0.0, 0.0, {"nodeOffset":{}}]'
        pos = core.pos_after_node_transforms(row, (1.0, 0.0, 0.0), (parent,))
        self.assertEqual(pos, (1.0, 0.0, 0.0))


class VariableTableTests(unittest.TestCase):
    def test_expression_evaluation(self) -> None:
        v = {"$a": 2.0, "$b": 10.0}
        self.assertEqual(core.evaluate_jbeam_expression("$=$a+0.5", v), 2.5)
        self.assertEqual(core.evaluate_jbeam_expression("$b", v), 10.0)
        self.assertEqual(core.evaluate_jbeam_expression("$=$a*3", v), 6.0)
        # Lua nil-guard idiom translates to Python truthiness
        self.assertEqual(core.evaluate_jbeam_expression("$= $c==nil and 1 or $c", {}), 1.0)
        # unknown variable in arithmetic -> unevaluable -> None (caller falls back)
        self.assertIsNone(core.evaluate_jbeam_expression("$=$missing*2", {}))

    def test_expression_number_falls_back_to_constant_sum(self) -> None:
        # No table: behaves like the old approximation.
        self.assertEqual(core.expression_number("$=$trackoffset+0.155"), 0.155)
        # With a table giving the variable a value: exact.
        self.assertAlmostEqual(
            core.expression_number("$=$trackoffset+0.155", {"$trackoffset": 0.05}), 0.205
        )

    def test_variable_defaults_and_pc_override(self) -> None:
        part = (
            '"veh": {\n"slotType": "main",\n"variables": [\n'
            '    ["name", "type", "unit", "category", "default", "min", "max"],\n'
            '    ["$trackoffset_R", "range", "+m", "Wheels", 0.0, -0.02, 0.05],\n'
            "]\n}"
        )
        index = core.build_part_body_index({"vehicles/veh/veh.jbeam": part})
        # default config: trackoffset defaults to 0
        sel = core.resolve_selected_parts({"parts": {}}, {}, vehicle_id="veh", part_body_index=index)
        self.assertEqual(core.part_variable_scope(sel, "veh")["$trackoffset_R"], 0.0)
        # .pc vars override, clamped to range (0.09 -> max 0.05)
        sel2 = core.resolve_selected_parts(
            {"parts": {}, "vars": {"$trackoffset_R": 0.09}},
            {},
            vehicle_id="veh",
            part_body_index=index,
        )
        self.assertEqual(core.part_variable_scope(sel2, "veh")["$trackoffset_R"], 0.05)

    def test_nodeoffset_uses_variable_value(self) -> None:
        # A wheel-mount nodeOffset expression resolves against the scope.
        offset_opt = '{"nodeOffset":{"x":"$=$trackoffset_R+0.5"}}'
        pos = core.pos_after_node_transforms(
            '["rw1l", 0.0, 0.0, 0.0]', (0.5, 0.0, 0.0), (offset_opt,), {"$trackoffset_R": 0.05}
        )
        self.assertAlmostEqual(pos[0], 0.5 + 0.55)  # +x side: base 0.5 + (0.05+0.5)


class MirrorPositionTransformTests(unittest.TestCase):
    def test_flexbody_mirror_position_preserves_rotation(self) -> None:
        row = (
            '["gearlever", ["grp"], '
            '{"pos":{"x":0.42,"y":1.1,"z":0.3}, '
            '"rot":{"x":10,"y":20,"z":30}}]'
        )
        rewritten = core.transform_flexbody_row(row, "mirrorPosition")
        self.assertEqual(core.vector_from_row(rewritten, "pos"), (-0.42, 1.1, 0.3))
        self.assertEqual(core.vector_from_row(rewritten, "rot"), (10.0, 20.0, 30.0))

    def test_prop_mirror_position_preserves_rotation(self) -> None:
        props = (
            "[\n"
            '  ["func", "mesh", "idRef:"],\n'
            '  ["gear", "gearlever", "n1", '
            '{"baseTranslationGlobal":{"x":0.42,"y":1.1,"z":0.3}, '
            '"baseRotationGlobal":{"x":10,"y":20,"z":30}}]\n'
            "]"
        )
        rewritten = core.rewrite_prop_meshes_with_globals(
            props,
            {"gearlever": "gearlever_xp_rhd"},
            {},
            {"gearlever": ("mirrorPosition", 0.0)},
            {},
        )
        row = next(row for row in core.iter_top_level_rows(rewritten) if "gearlever_xp_rhd" in row)
        self.assertEqual(core.vector_from_row(row, "baseTranslationGlobal"), (-0.42, 1.1, 0.3))
        self.assertEqual(core.vector_from_row(row, "baseRotationGlobal"), (10.0, 20.0, 30.0))


class NodeMergeOrderTests(unittest.TestCase):
    def test_iter_node_rows_skips_commented_and_is_last_wins_ready(self) -> None:
        arr = (
            "[\n"
            '  ["id", "posX", "posY", "posZ"],\n'
            '  //["c3", 0.0, -1.60, 0.42],\n'
            '  ["c3", 0.0, -1.59, 0.43],\n'
            "]"
        )
        rows = list(core.iter_node_rows(arr))
        # only the active c3 row is yielded (commented one skipped)
        self.assertEqual([(nid, pos) for nid, pos, _ in rows], [("c3", (0.0, -1.59, 0.43))])

    def test_in_part_redefinition_is_last_wins(self) -> None:
        # Same node redefined lower in the same array -> the later one wins.
        part = (
            '"veh": {\n"slotType": "main",\n"nodes": [\n'
            '  ["id", "posX", "posY", "posZ"],\n'
            '  ["n1", 0.0, 0.0, 0.0],\n'
            '  ["n1", 1.0, 2.0, 3.0],\n'
            "]\n}"
        )
        index = core.build_part_body_index({"vehicles/veh/veh.jbeam": part})
        sel = core.resolve_selected_parts({"parts": {}}, {}, vehicle_id="veh", part_body_index=index)
        nodes = core.selected_node_positions_for_parts(sel, {}, index)
        self.assertEqual(nodes["n1"], (1.0, 2.0, 3.0))

    def test_child_part_overrides_parent_node(self) -> None:
        # Tree order: the child (deeper) part's node definition wins.
        parent = (
            '"veh": {\n"slotType": "main",\n"slots": [\n'
            '  ["type", "default", "description"],\n'
            '  ["addon", "addon", "Addon"],\n'
            "],\n"
            '"nodes": [\n'
            '  ["id", "posX", "posY", "posZ"],\n'
            '  ["shared", 0.0, 0.0, 0.0],\n'
            "]\n}"
        )
        child = (
            '"addon": {\n"slotType": "addon",\n"nodes": [\n'
            '  ["id", "posX", "posY", "posZ"],\n'
            '  ["shared", 9.0, 9.0, 9.0],\n'
            "]\n}"
        )
        index = core.build_part_body_index(
            {"vehicles/veh/veh.jbeam": "{" + parent + ",\n" + child + "}"}
        )
        sel = core.resolve_selected_parts({"parts": {}}, {}, vehicle_id="veh", part_body_index=index)
        nodes = core.selected_node_positions_for_parts(sel, {}, index)
        self.assertEqual(nodes["shared"], (9.0, 9.0, 9.0))
        # parent appears before child in merge order
        order = core.selected_parts_in_merge_order(sel)
        self.assertLess(order.index("veh"), order.index("addon"))


if __name__ == "__main__":
    unittest.main()
