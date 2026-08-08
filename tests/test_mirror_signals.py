"""A wing mirror's turn-signal repeater under a mesh swap.

The lamp is not authored in the mirror part at all: it is a SPOTLIGHT prop in
a shared bulb part, and the mirror fills in every one of its fields from its
own slots2 row. Six of those fields say where the lamp sits and which way it
shines; the rest say which indicator it is. A swap has to move the first six
and leave the rest alone.
"""
from __future__ import annotations

import unittest
from pathlib import Path

from beamxp import hand_drive_core as core
from beamxp.core.geometry import (
    base_rotation_global_matrix3,
    mirrored_base_rotation_global,
)
from beamxp.hand_drive_ui import recommendation_common


def mirror_part(side: str, pos: tuple[float, float, float], rot: tuple[float, float, float]) -> str:
    """A wing mirror in the shape sunburst2, scintilla, etkc and the sbr use."""
    low = side.lower()
    return (
        f'"veh_mirror_{side}": {{\n'
        f'"slotType": "veh_mirror_{side}",\n'
        '"slots2": [\n'
        '    ["name", "allowTypes", "denyTypes", "default", "description"],\n'
        f'    ["veh_mirror_{side}_signal_bulb", ["incandescent_amber_5W"], [], '
        f'"incandescent_amber_5W", "Mirror Turn Signal Bulb",\n'
        '    {"coreSlot":true, "variables":{\n'
        f'        "$electric":"signal_{side}_filament",\n'
        f'        "$nodeRef":"mi4{low}", "$nodeX":"mi3{low}", "$nodeY":"mi2{low}",\n'
        f'        "$deformGroup":"mirrorsignal_{side}_break",\n'
        f'        "$cookieName":"/art/special/blinker_{low}.color.png",\n'
        f'        "$posX":{pos[0]}, "$posY":{pos[1]}, "$posZ":{pos[2]},\n'
        f'        "$rotX":{rot[0]}, "$rotY":{rot[1]}, "$rotZ":{rot[2]}\n'
        "    }}]\n"
        "],\n"
        '"flexbodies": [\n'
        '    ["mesh", "[group]:", "nonFlexMaterials"],\n'
        f'    ["veh_mirror_{side}", ["veh_mirror_{side}"]],\n'
        f'    {{"deformGroup":"mirrorsignal_{side}_break", "deformMaterialBase":"veh_lights", '
        '"deformMaterialDamaged":"veh_lights_dmg"},\n'
        f'    ["veh_mirrorsignal_{side}", ["veh_mirror_{side}"]],\n'
        '    {"deformGroup":""},\n'
        "],\n"
        "}"
    )


# Deliberately NOT an exact reflection: the left bulb sits 4 mm further out and
# aims 3 degrees differently, which is the vivace's real geometry and the whole
# reason the placement has to travel with the glass.
LEFT = mirror_part("L", (0.937, -0.641, 0.985), (-179.515, 172.713, 40.652))
RIGHT = mirror_part("R", (-0.941, -0.635, 0.985), (-179.93, -173.016, -43.76))

PARTS = {
    "veh_mirror_L": (LEFT, "mirrors.jbeam"),
    "veh_mirror_R": (RIGHT, "mirrors.jbeam"),
}


def context() -> core.VehicleContext:
    return core.VehicleContext(
        source_zip=Path("test.zip"),
        vehicle_id="veh",
        vehicle_path="vehicles/veh",
        dae_paths=[],
        variants={},
        objects={},
        preview_by_id={},
        jbeam_texts={},
        node_positions={},
        project_dir=Path("project"),
        part_body_index=PARTS,
    )


SWAPPED = {
    "veh_mirrorsignal_L": core.MODE_MIRROR_STRUCTURAL,
    "veh_mirrorsignal_R": core.MODE_MIRROR_STRUCTURAL,
}
SOURCES = {
    "veh_mirrorsignal_L": "veh_mirrorsignal_R",
    "veh_mirrorsignal_R": "veh_mirrorsignal_L",
}


class MirrorSignalRecommendationTests(unittest.TestCase):
    def test_both_spellings_of_a_signal_lens_join_the_mirror_family(self) -> None:
        # sunburst2/scintilla/etkc/sbr write "mirrorsignal"; vivace/roamer/
        # sv1ev3 write "mirror_signal". Only the second used to be recognised,
        # so on the first group the casing swapped and the lens did not.
        for name in (
            "sunburst2_mirrorsignal_l",
            "sbr_mirrorsignalglass_l",
            "vivace_mirror_signal_l",
            "roamer_mirror_signalglass_l_facelift",
            "scintilla_mirrorsignal_r",
        ):
            self.assertTrue(
                recommendation_common.recommendation_matches(
                    name, recommendation_common.HANDED_PATTERNS
                ),
                name,
            )

    def test_a_fender_repeater_is_not_a_wing_mirror(self) -> None:
        for name in ("etk800_fendersignal_l", "etk800_taillight_l"):
            self.assertFalse(
                recommendation_common.recommendation_matches(
                    name, recommendation_common.HANDED_PATTERNS
                ),
                name,
            )


class BulbPlacementTests(unittest.TestCase):
    def test_the_lamp_and_its_lens_are_linked_by_their_deform_group(self) -> None:
        self.assertEqual(
            core.deform_group_flexbody_meshes(LEFT),
            {"mirrorsignal_L_break": {"veh_mirrorsignal_L"}},
        )

    def test_a_slot_reports_the_circuit_it_drives(self) -> None:
        slots = core.light_slot_placements(LEFT)
        self.assertEqual(sorted(slots), ["signal_L_filament"])
        self.assertEqual(slots["signal_L_filament"]["pos"], (0.937, -0.641, 0.985))
        self.assertEqual(slots["signal_L_filament"]["deformGroup"], "mirrorsignal_L_break")

    def test_a_swapped_lens_takes_its_lamp_with_it(self) -> None:
        moved = core.swapped_light_slot_placements(context(), LEFT, SWAPPED, SOURCES)
        placement = moved["signal_L_filament"]
        # The right bulb's own placement, reflected: x negates, y and z carry.
        self.assertAlmostEqual(placement["pos"][0], 0.941)
        self.assertAlmostEqual(placement["pos"][1], -0.635)
        self.assertAlmostEqual(placement["pos"][2], 0.985)

    def test_the_beam_ends_up_aimed_where_the_far_side_aimed_it(self) -> None:
        moved = core.swapped_light_slot_placements(context(), LEFT, SWAPPED, SOURCES)
        got = base_rotation_global_matrix3(moved["signal_L_filament"]["rot"])
        want = base_rotation_global_matrix3(
            mirrored_base_rotation_global((-179.93, -173.016, -43.76))
        )
        for got_row, want_row in zip(got, want):
            for got_value, want_value in zip(got_row, want_row):
                self.assertAlmostEqual(got_value, want_value, places=9)

    def test_a_lens_left_alone_leaves_its_lamp_alone(self) -> None:
        self.assertEqual(core.swapped_light_slot_placements(context(), LEFT, {}, {}), {})

    def test_a_mirror_whose_casing_swaps_but_lens_does_not_keeps_its_lamp(self) -> None:
        # The deform group is what ties the lamp to the lens, so swapping the
        # casing alone is not a reason to move the light.
        casing_only = {"veh_mirror_L": core.MODE_MIRROR_STRUCTURAL}
        sources = {"veh_mirror_L": "veh_mirror_R"}
        self.assertEqual(
            core.swapped_light_slot_placements(context(), LEFT, casing_only, sources),
            {},
        )


class AmbiguousFarSideTests(unittest.TestCase):
    def test_a_lens_two_parts_share_leaves_the_lamp_alone(self) -> None:
        # Two right-hand parts declare the lens, their bulbs 12 mm apart, so
        # nothing says which one is this lamp's far side. No wing mirror poses
        # the question -- every shared signal lens in the fleet belongs to
        # parts with no lamp -- so refusing costs nothing and keeps this to
        # the case it was written for.
        wide = mirror_part("R", (-0.929, -0.635, 0.985), (-179.93, -173.016, -43.76))
        wide = wide.replace('"veh_mirror_R"', '"veh_mirror_R_wide"', 1)
        ctx = context()
        ctx.part_body_index = {**PARTS, "veh_mirror_R_wide": (wide, "mirrors.jbeam")}
        self.assertEqual(
            core.swapped_light_slot_placements(ctx, LEFT, SWAPPED, SOURCES), {}
        )

    def test_two_parts_agreeing_on_the_placement_is_not_ambiguous(self) -> None:
        same = RIGHT.replace('"veh_mirror_R"', '"veh_mirror_R_wide"', 1)
        ctx = context()
        ctx.part_body_index = {**PARTS, "veh_mirror_R_wide": (same, "mirrors.jbeam")}
        moved = core.swapped_light_slot_placements(ctx, LEFT, SWAPPED, SOURCES)
        self.assertAlmostEqual(moved["signal_L_filament"]["pos"][0], 0.941)


class ReplaceSourceTests(unittest.TestCase):
    def test_a_reskinned_lens_moves_its_lamp_too(self) -> None:
        # Replace Source takes its geometry from the same structural sources
        # and the build mirrors its flexbody row exactly as Swap Mesh does, so
        # a lamp behind either has equally moved.
        moved = core.swapped_light_slot_placements(
            context(),
            LEFT,
            {"veh_mirrorsignal_L": core.MODE_REPLACE_SOURCE},
            {"veh_mirrorsignal_L": "veh_mirrorsignal_R"},
        )
        self.assertAlmostEqual(moved["signal_L_filament"]["pos"][0], 0.941)

    def test_move_and_mirror_still_leave_the_lamp_alone(self) -> None:
        for mode in (core.MODE_MIRROR, core.MODE_TRANSLATE, core.MODE_SKIP):
            self.assertEqual(
                core.swapped_light_slot_placements(
                    context(),
                    LEFT,
                    {"veh_mirrorsignal_L": mode},
                    {"veh_mirrorsignal_L": "veh_mirrorsignal_R"},
                ),
                {},
                mode,
            )


class BulbRewriteTests(unittest.TestCase):
    def _rewritten(self) -> str:
        moved = core.swapped_light_slot_placements(context(), LEFT, SWAPPED, SOURCES)
        slots2 = core.transform_helpers.extract_named_array(LEFT, "slots2")
        return core.rewrite_light_slot_placements(slots2, moved)

    def test_only_the_placement_numbers_change(self) -> None:
        out = self._rewritten()
        self.assertIn('"$posX":0.941', out)
        self.assertIn('"$posY":-0.635', out)
        self.assertIn('"$rotZ":43.76', out)
        self.assertNotIn('"$posX":0.937', out)

    def test_the_indicator_keeps_its_own_circuit_nodes_and_cookie(self) -> None:
        # This is the whole point: the glass came from the right mirror, but
        # the lamp behind it is still the LEFT indicator.
        out = self._rewritten()
        self.assertIn('"$electric":"signal_L_filament"', out)
        self.assertIn('"$deformGroup":"mirrorsignal_L_break"', out)
        self.assertIn('"$nodeRef":"mi4l"', out)
        self.assertIn('"$nodeX":"mi3l"', out)
        self.assertIn('"$nodeY":"mi2l"', out)
        self.assertIn('blinker_l.color.png', out)
        self.assertNotIn("signal_R_filament", out)
        self.assertNotIn("blinker_r", out)

    def test_an_unlisted_circuit_is_not_touched(self) -> None:
        slots2 = core.transform_helpers.extract_named_array(LEFT, "slots2")
        self.assertEqual(core.rewrite_light_slot_placements(slots2, {}), slots2)


if __name__ == "__main__":
    unittest.main()
