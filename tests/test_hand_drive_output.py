from __future__ import annotations

import unittest
from pathlib import Path

from beamxp import hand_drive_core as core
from beamxp.hand_drive_parts import generation as generation_impl


def part(
    part_id: str,
    slot_type: str,
    display_name: str,
    slots: tuple[tuple[str, str], ...] = (),
) -> str:
    slot_rows = ""
    if slots:
        rows = ",\n".join(
            f'["{slot_id}", "{default_part}", "{slot_id}"]'
            for slot_id, default_part in slots
        )
        slot_rows = (
            ',\n"slots": [\n'
            '    ["type", "default", "description"],\n'
            f"    {rows}\n"
            "]"
        )
    return (
        f'"{part_id}": {{\n'
        f'"information": {{"name": "{display_name}"}},\n'
        f'"slotType": "{slot_type}"'
        f"{slot_rows}\n"
        "}"
    )


def context_with_parts(
    part_index: dict[str, tuple[str, str]],
    selected: dict[str, object],
) -> core.VehicleContext:
    context = core.VehicleContext(
        source_zip=Path("test.zip"),
        vehicle_id="acme",
        vehicle_path="vehicles/acme",
        dae_paths=[],
        variants={"trim": core.VariantInfo("trim", "trim.pc", None, "Trim")},
        objects={},
        preview_by_id={},
        jbeam_texts={},
        node_positions={},
        project_dir=Path("project"),
        part_body_index=part_index,
    )
    context.selected_parts_cache["trim"] = selected
    return context


class GeneratedPartIdentityTests(unittest.TestCase):
    def test_generated_part_identity_is_not_trim_specific(self) -> None:
        first = core.generated_variant_part_name("acme_dash", core.HAND_RHD, "base")
        second = core.generated_variant_part_name("acme_dash", core.HAND_RHD, "sport")
        self.assertEqual(first, "acme_dash_xp_rhd")
        self.assertEqual(second, first)


class GeneratedRootSlotDefaultTests(unittest.TestCase):
    def test_slots_defaults_can_point_at_generated_children(self) -> None:
        array_text = (
            "[\n"
            '    ["type", "default", "description"],\n'
            '    ["acme_steer", "acme_steer", "Steering Wheel"],\n'
            '    ["acme_radio", "acme_radio", "Radio"]\n'
            "]"
        )
        rewritten = core.rewrite_child_slot_defaults(
            array_text,
            {"acme_steer": "acme_steer_xp_rhd"},
            1,
        )
        self.assertIn('"acme_steer", "acme_steer_xp_rhd"', rewritten)
        self.assertIn('"acme_radio", "acme_radio"', rewritten)

    def test_slots2_defaults_can_point_at_generated_children(self) -> None:
        array_text = (
            "[\n"
            '    ["name", "allowTypes", "denyTypes", "default", "description"],\n'
            '    ["acme_pedals", ["acme_pedals"], [], "acme_pedals", "Pedals"],\n'
            '    ["soundscape_horn", ["soundscape_horn"], [], "horn", "Horn"]\n'
            "]"
        )
        rewritten = core.rewrite_child_slot_defaults(
            array_text,
            {"acme_pedals": "acme_pedals_xp_rhd"},
            3,
        )
        self.assertIn(
            '"acme_pedals", ["acme_pedals"], [], "acme_pedals_xp_rhd"',
            rewritten,
        )
        self.assertIn('"soundscape_horn", ["soundscape_horn"], [], "horn"', rewritten)

    def test_slots2_generated_defaults_get_handed_slot_names(self) -> None:
        array_text = (
            "[\n"
            '    ["name", "allowTypes", "denyTypes", "default", "description"],\n'
            '    ["acme_dash", ["acme_dash"], [], "acme_dash", "Dashboard"],\n'
            '    ["acme_seat", ["acme_seat"], [], "acme_seat", "Seat"]\n'
            "]"
        )
        rewritten = core.rewrite_child_slot_defaults(
            array_text,
            {"acme_dash": "acme_dash_xp_rhd"},
            3,
            "_xp_rhd",
        )
        self.assertIn(
            '"acme_dash_xp_rhd", ["acme_dash"], [], "acme_dash_xp_rhd"',
            rewritten,
        )
        self.assertIn('"acme_seat", ["acme_seat"], [], "acme_seat"', rewritten)


class GeneratedLightPatternTests(unittest.TestCase):
    def test_rhd_clone_forces_rhd_light_pattern(self) -> None:
        body = (
            '"acme_headlight": {\n'
            '  "slots2": [\n'
            '    ["lowbeam", ["led"], [], "led", "Low Beam", {"variables":{"$lightPattern":"LHD"}}],\n'
            '    ["lowbeam_alt", ["led"], [], "led", "Low Beam", {"variables":{"$lightPattern":"US"}}]\n'
            "  ]\n"
            "}"
        )
        rewritten = core.rewrite_light_pattern_for_target(body, core.HAND_RHD)
        self.assertEqual(rewritten.count('"$lightPattern":"RHD"'), 2)
        self.assertNotIn('"LHD"', rewritten)
        self.assertNotIn('"US"', rewritten)

    def test_lhd_clone_forces_lhd_light_pattern(self) -> None:
        body = '"acme_headlight": {"$lightPattern":"RHD"}'
        rewritten = core.rewrite_light_pattern_for_target(body, core.HAND_LHD)
        self.assertIn('"$lightPattern":"LHD"', rewritten)

    def test_light_slots_seed_generated_bridge_ancestors(self) -> None:
        part_index = {
            "car": (part("car", "main", "Car", (("chassis", "chassis"),)), "car.jbeam"),
            "chassis": (
                part("chassis", "chassis", "Chassis", (("fender_L", "fender_L"),)),
                "chassis.jbeam",
            ),
            "fender_L": (
                part(
                    "fender_L",
                    "fender_L",
                    "Left Fender",
                    (("headlight_L", "headlight_L"),),
                ),
                "fender.jbeam",
            ),
            "headlight_L": (
                '"headlight_L": {\n'
                '"information": {"name": "Left Headlight"},\n'
                '"slotType": "headlight_L",\n'
                '"slots2": [\n'
                '  ["name", "allowTypes", "denyTypes", "default", "description"],\n'
                '  ["headlight_L_lowbeam", ["bulb"], [], "bulb", "Low Beam", {"variables":{"$electric":"lowbeam","$lightPattern":"US"}}],\n'
                '  ["headlight_L_highbeam", ["bulb"], [], "bulb", "High Beam", {"variables":{"$electric":"highbeam","$lightPattern":"US"}}]\n'
                "]\n"
                "}",
                "headlight.jbeam",
            ),
        }
        selected = {
            "parts": {"car", "chassis", "fender_L", "headlight_L"},
            "main_part": "car",
            "part_instances": [
                {
                    "instance_id": "/car",
                    "part_id": "car",
                    "slot_id": "main",
                    "slot_path": "/",
                    "parent_instance_id": None,
                },
                {
                    "instance_id": "/chassis/chassis",
                    "part_id": "chassis",
                    "slot_id": "chassis",
                    "slot_path": "/chassis/",
                    "parent_instance_id": "/car",
                },
                {
                    "instance_id": "/chassis/fender_L/fender_L",
                    "part_id": "fender_L",
                    "slot_id": "fender_L",
                    "slot_path": "/chassis/fender_L/",
                    "parent_instance_id": "/chassis/chassis",
                },
                {
                    "instance_id": "/chassis/fender_L/headlight_L/headlight_L",
                    "part_id": "headlight_L",
                    "slot_id": "headlight_L",
                    "slot_path": "/chassis/fender_L/headlight_L/",
                    "parent_instance_id": "/chassis/fender_L/fender_L",
                },
            ],
            "selected_by_slot": {
                "main": "car",
                "chassis": "chassis",
                "fender_L": "fender_L",
                "headlight_L": "headlight_L",
            },
            "part_slot_options": {},
        }
        context = context_with_parts(part_index, selected)

        plan = generation_impl._generated_clone_plan(
            context,
            selected,
            core.HAND_RHD,
            "trim",
            {},
            {},
            set(),
        )

        self.assertEqual(
            plan,
            {
                "chassis": "chassis_xp_rhd",
                "fender_L": "fender_L_xp_rhd",
                "headlight_L": "headlight_L_xp_rhd",
            },
        )


class HandAuthoredGroupTests(unittest.TestCase):
    def setUp(self) -> None:
        # Model the vanilla pattern: dashboard roots share one slot, while
        # related child slots and parts carry their own LHD/RHD suffixes.
        self.part_index = {
            "car": (
                part("car", "main", "Car", (("dashboard", "dash_lhd"),)),
                "car.jbeam",
            ),
            "dash_lhd": (
                part(
                    "dash_lhd",
                    "dashboard",
                    "Left-Hand-Drive Dashboard",
                    (
                        ("steer", "steer"),
                        ("shifter_lhd", "shifter_A_lhd"),
                        ("gauges_lhd", "gauges_lhd"),
                    ),
                ),
                "dash_lhd.jbeam",
            ),
            "dash_rhd": (
                part(
                    "dash_rhd",
                    "dashboard",
                    "Right-Hand-Drive Dashboard",
                    (
                        ("steer", "steer"),
                        ("shifter_rhd", "shifter_A_rhd"),
                        ("gauges_rhd", "gauges_rhd"),
                    ),
                ),
                "dash_rhd.jbeam",
            ),
            "steer": (part("steer", "steer", "Steering Wheel"), "dash.jbeam"),
            "shifter_M_lhd": (
                part("shifter_M_lhd", "shifter_lhd", "Manual Shifter LHD"),
                "dash_lhd.jbeam",
            ),
            "shifter_M_rhd": (
                part("shifter_M_rhd", "shifter_rhd", "Manual Shifter RHD"),
                "dash_rhd.jbeam",
            ),
            "shifter_A_lhd": (
                part("shifter_A_lhd", "shifter_lhd", "Automatic Shifter LHD"),
                "dash_lhd.jbeam",
            ),
            "shifter_A_rhd": (
                part("shifter_A_rhd", "shifter_rhd", "Automatic Shifter RHD"),
                "dash_rhd.jbeam",
            ),
            "gauges_lhd": (
                part("gauges_lhd", "gauges_lhd", "Gauges LHD"),
                "dash_lhd.jbeam",
            ),
            "gauges_rhd": (
                part("gauges_rhd", "gauges_rhd", "Gauges RHD"),
                "dash_rhd.jbeam",
            ),
        }
        self.selected = {
            "parts": {"car", "dash_lhd", "steer", "shifter_M_lhd", "gauges_lhd"},
            "part_instances": [
                {
                    "instance_id": "/car",
                    "part_id": "car",
                    "slot_id": "main",
                    "slot_path": "/",
                    "parent_instance_id": None,
                    "inherited_options": (),
                },
                {
                    "instance_id": "/dashboard/dash_lhd",
                    "part_id": "dash_lhd",
                    "slot_id": "dashboard",
                    "slot_path": "/dashboard/",
                    "parent_instance_id": "/car",
                    "inherited_options": (),
                },
                {
                    "instance_id": "/dashboard/steer/steer",
                    "part_id": "steer",
                    "slot_id": "steer",
                    "slot_path": "/dashboard/steer/",
                    "parent_instance_id": "/dashboard/dash_lhd",
                    "inherited_options": (),
                },
                {
                    "instance_id": "/dashboard/shifter_lhd/shifter_M_lhd",
                    "part_id": "shifter_M_lhd",
                    "slot_id": "shifter_lhd",
                    "slot_path": "/dashboard/shifter_lhd/",
                    "parent_instance_id": "/dashboard/dash_lhd",
                    "inherited_options": (),
                },
                {
                    "instance_id": "/dashboard/gauges_lhd/gauges_lhd",
                    "part_id": "gauges_lhd",
                    "slot_id": "gauges_lhd",
                    "slot_path": "/dashboard/gauges_lhd/",
                    "parent_instance_id": "/dashboard/dash_lhd",
                    "inherited_options": (),
                },
            ],
            "selected_by_slot": {
                "main": "car",
                "dashboard": "dash_lhd",
                "steer": "steer",
                "shifter_lhd": "shifter_M_lhd",
                "gauges_lhd": "gauges_lhd",
            },
            "selected_by_path": {
                "/": "car",
                "/dashboard/": "dash_lhd",
                "/dashboard/steer/": "steer",
                "/dashboard/shifter_lhd/": "shifter_M_lhd",
                "/dashboard/gauges_lhd/": "gauges_lhd",
            },
            "part_slot_options": {},
        }

    def test_detects_authored_dashboard_and_maps_related_choices(self) -> None:
        context = context_with_parts(self.part_index, self.selected)
        group = core.find_hand_authored_opposite_group(
            context, "trim", core.HAND_LHD, core.HAND_RHD
        )
        self.assertIsNotNone(group)
        assert group is not None
        self.assertEqual(group["sourcePart"], "dash_lhd")
        self.assertEqual(group["targetPart"], "dash_rhd")
        updates = {
            selection["slotId"]: selection["partId"]
            for selection in group["selections"]
        }
        self.assertEqual(
            updates,
            {
                "dashboard": "dash_rhd",
                "steer": "steer",
                "shifter_rhd": "shifter_M_rhd",
                "gauges_rhd": "gauges_rhd",
            },
        )

    def test_unmarked_default_can_pair_with_explicit_rhd_part(self) -> None:
        part_index = dict(self.part_index)
        part_index["dash"] = (
            part(
                "dash",
                "dashboard",
                "Dashboard",
                (("steer", "steer"), ("shifter_lhd", "shifter_A_lhd")),
            ),
            "dash_lhd.jbeam",
        )
        selected = dict(self.selected)
        selected["parts"] = set(self.selected["parts"])
        selected["parts"].discard("dash_lhd")
        selected["parts"].add("dash")
        selected["part_instances"] = [
            dict(instance) for instance in self.selected["part_instances"]
        ]
        selected["part_instances"][1]["part_id"] = "dash"
        selected["selected_by_slot"] = dict(self.selected["selected_by_slot"])
        selected["selected_by_slot"]["dashboard"] = "dash"
        selected["selected_by_path"] = dict(self.selected["selected_by_path"])
        selected["selected_by_path"]["/dashboard/"] = "dash"
        context = context_with_parts(part_index, selected)
        group = core.find_hand_authored_opposite_group(
            context, "trim", core.HAND_LHD, core.HAND_RHD
        )
        self.assertIsNotNone(group)
        assert group is not None
        self.assertEqual(group["targetPart"], "dash_rhd")

    def test_leaf_pair_alone_is_not_treated_as_a_group(self) -> None:
        part_index = {
            "car": (part("car", "main", "Car", (("wheel", "wheel_lhd"),)), "a.jbeam"),
            "wheel_lhd": (part("wheel_lhd", "wheel", "LHD Wheel"), "a.jbeam"),
            "wheel_rhd": (part("wheel_rhd", "wheel", "RHD Wheel"), "a.jbeam"),
        }
        selected = {
            "part_instances": [
                {"part_id": "car", "slot_id": "main", "slot_path": "/"},
                {
                    "part_id": "wheel_lhd",
                    "slot_id": "wheel",
                    "slot_path": "/wheel/",
                },
            ],
            "selected_by_slot": {"main": "car", "wheel": "wheel_lhd"},
            "selected_by_path": {"/": "car", "/wheel/": "wheel_lhd"},
        }
        context = context_with_parts(part_index, selected)
        group = core.find_hand_authored_opposite_group(
            context, "trim", core.HAND_LHD, core.HAND_RHD
        )
        self.assertIsNone(group)

    def test_group_application_preserves_path_specific_keys(self) -> None:
        context = context_with_parts(self.part_index, self.selected)
        group = core.find_hand_authored_opposite_group(
            context, "trim", core.HAND_LHD, core.HAND_RHD
        )
        assert group is not None
        pc: dict[str, object] = {
            "parts": {
                "/dashboard/": "dash_lhd",
                "/dashboard/shifter_lhd/": "shifter_M_lhd",
            }
        }
        core.apply_hand_authored_group(pc, group)
        self.assertEqual(
            pc["parts"],
            {
                "/dashboard/": "dash_rhd",
                "/dashboard/shifter_rhd/": "shifter_M_rhd",
                "steer": "steer",
                "gauges_rhd": "gauges_rhd",
            },
        )

    def test_group_application_rewrites_child_slot_namespace(self) -> None:
        context = context_with_parts(self.part_index, self.selected)
        group = core.find_hand_authored_opposite_group(
            context, "trim", core.HAND_LHD, core.HAND_RHD
        )
        assert group is not None
        pc: dict[str, object] = {
            "parts": {
                "dashboard": "dash_lhd",
                "steer": "steer",
                "shifter_lhd": "shifter_M_lhd",
                "gauges_lhd": "gauges_lhd",
                "paint": "red",
            }
        }
        core.apply_hand_authored_group(pc, group)
        self.assertEqual(
            pc["parts"],
            {
                "dashboard": "dash_rhd",
                "steer": "steer",
                "shifter_rhd": "shifter_M_rhd",
                "gauges_rhd": "gauges_rhd",
                "paint": "red",
            },
        )


if __name__ == "__main__":
    unittest.main()
