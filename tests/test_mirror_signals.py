"""A wing mirror's turn-signal repeater under a mesh swap.

The lamp is not authored in the mirror part at all: it is a SPOTLIGHT prop in
a shared bulb part, and the mirror fills in every one of its fields from its
own slots2 row. Six of those fields say where the lamp sits and which way it
shines; the rest say which indicator it is. A swap has to move the first six
and leave the rest alone.
"""
from __future__ import annotations

import re
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


class MirrorPlaneSourceTests(unittest.TestCase):
    """Which side's reflection plane a converted wing mirror ends up with.

    The numbers are the Covet's own, because the Covet is the ground truth: the
    game ships both hands of it, and its RHD mirror parts differ from the LHD
    ones by the mesh name alone.
    """

    # Deliberately not symmetric, exactly as BeamNG authored them.
    ROWS = {
        "covet_mirror_L": (
            '["covet_mirror_L","mi4l","mi2l","mi3l",'
            '{"refBaseTranslation":{"x":-0.09,"y":0.00,"z":0.04},'
            '"baseRotationGlobal":{"x":0.2,"y":0.0,"z":-13.4}}],'
        ),
        "covet_mirror_R": (
            '["covet_mirror_R","mi4r","mi2r","mi3r",'
            '{"refBaseTranslation":{"x":0.095,"y":-0.02,"z":0.04},'
            '"baseRotationGlobal":{"x":0.2,"y":0.0,"z":13.4}}],'
        ),
    }
    MESHES = ["covet_mirror_L", "covet_mirror_R"]

    def test_a_swapped_wing_mirror_is_left_out_so_it_keeps_its_plane(self) -> None:
        swapped = dict.fromkeys(self.MESHES, core.MODE_MIRROR_STRUCTURAL)
        self.assertEqual(
            core.mirror_plane_sources_for_meshes(self.MESHES, swapped, self.ROWS), {}
        )

    def test_replace_source_is_left_out_for_the_same_reason(self) -> None:
        replaced = dict.fromkeys(self.MESHES, core.MODE_REPLACE_SOURCE)
        self.assertEqual(
            core.mirror_plane_sources_for_meshes(self.MESHES, replaced, self.ROWS), {}
        )

    def test_a_plain_mirror_crosses_the_car_and_reflects_its_own_plane(self) -> None:
        mirrored = dict.fromkeys(self.MESHES, core.MODE_MIRROR)
        sources = core.mirror_plane_sources_for_meshes(self.MESHES, mirrored, self.ROWS)
        # its own row, never the twin's
        self.assertEqual(sources["covet_mirror_L"], self.ROWS["covet_mirror_L"])
        self.assertEqual(sources["covet_mirror_R"], self.ROWS["covet_mirror_R"])

    def test_a_swap_leaves_the_row_reading_as_the_game_authored_the_rhd_one(self) -> None:
        """End to end: mesh renamed, plane untouched -- the Covet's own diff."""
        array_text = "[\n" + self.ROWS["covet_mirror_L"] + "\n]"
        swapped = dict.fromkeys(self.MESHES, core.MODE_MIRROR_STRUCTURAL)
        rewritten = core.rewrite_mirror_rows(
            array_text,
            {"covet_mirror_L": "covet_mirror_RHD_L"},
            core.mirror_plane_sources_for_meshes(self.MESHES, swapped, self.ROWS),
        )
        self.assertIn('["covet_mirror_RHD_L","mi4l"', rewritten)
        self.assertIn('"refBaseTranslation":{"x":-0.09,"y":0.00,"z":0.04}', rewritten)
        self.assertIn('"baseRotationGlobal":{"x":0.2,"y":0.0,"z":-13.4}', rewritten)


class SwappedBeamPreloadTests(unittest.TestCase):
    """A mirror's toe-in can live in its mount's beam pre-load.

    The hopper's numbers. ``baseRotationGlobal`` is (-5,0,0) on both sides and
    carries no lateral aim; the toe-in is ``beamPrecompression``, 0.94 left
    against 0.85 right. Leaving each side its own left the converted driver's
    mirror with the passenger's figure and about 22 degrees of error in what it
    showed.
    """

    def _part(self, side, precompression):
        low = side.lower()
        return (
            f'"veh_mirror_{side}": {{\n'
            f'"slotType": "veh_mirror_{side}",\n'
            '"flexbodies": [\n'
            '    ["mesh", "[group]:"],\n'
            f'    ["veh_mirror_{side}", ["veh_mirror_{side}"]],\n'
            '],\n'
            '"beams": [\n'
            '    ["id1:", "id2:"],\n'
            '    {"beamSpring":401000},\n'
            f'    ["mi1{low}", "mi2{low}"],\n'
            f'    ["mi2{low}", "d8{low}",  {{"beamPrecompression":{precompression}}}],\n'
            f'    ["mi4{low}", "d8{low}",  {{"beamPrecompression":1.0}}],\n'
            '],\n'
            '}'
        )

    def _context(self):
        return core.VehicleContext(
            source_zip=Path("test.zip"), vehicle_id="veh", vehicle_path="vehicles/veh",
            dae_paths=[], variants={}, objects={}, preview_by_id={}, jbeam_texts={},
            node_positions={}, project_dir=Path("project"),
            part_body_index={
                "veh_mirror_L": (self._part("L", 0.94), "mirrors.jbeam"),
                "veh_mirror_R": (self._part("R", 0.85), "mirrors.jbeam"),
            },
        )

    MODES = {"veh_mirror_L": core.MODE_MIRROR_STRUCTURAL,
             "veh_mirror_R": core.MODE_MIRROR_STRUCTURAL}
    SOURCES = {"veh_mirror_L": "veh_mirror_R", "veh_mirror_R": "veh_mirror_L"}

    def test_a_swapped_mirror_takes_the_far_side_pre_load(self) -> None:
        got = core.swapped_beam_options(
            self._context(), self._part("L", 0.94), self.MODES, self.SOURCES)
        # the left mount takes the right's 0.85, on its own nodes
        self.assertEqual(got, {("mi2l", "d8l"): '{"beamPrecompression":0.85}'})

    def test_rows_that_agree_between_the_sides_are_left_alone(self) -> None:
        """mi4 carries 1.0 on both sides, so nothing about it is handed."""
        got = core.swapped_beam_options(
            self._context(), self._part("L", 0.94), self.MODES, self.SOURCES)
        self.assertNotIn(("mi4l", "d8l"), got)
        self.assertNotIn(("mi1l", "mi2l"), got)

    def test_a_mirror_the_build_left_alone_keeps_its_pre_load(self) -> None:
        self.assertEqual(
            core.swapped_beam_options(self._context(), self._part("L", 0.94), {}, {}), {})

    def test_the_rewrite_puts_the_far_side_figure_in_the_row(self) -> None:
        beams = core.transform_helpers.extract_named_array(self._part("L", 0.94), "beams")
        out = core.rewrite_beam_options(
            beams, {("mi2l", "d8l"): '{"beamPrecompression":0.85}'})
        self.assertIn('["mi2l", "d8l", {"beamPrecompression":0.85}]', out)
        self.assertNotIn('0.94', out)
        # every other row untouched, options and all
        self.assertIn('["mi4l", "d8l",  {"beamPrecompression":1.0}]', out)
        self.assertIn('["mi1l", "mi2l"]', out)
        self.assertIn('{"beamSpring":401000}', out)


class MirrorRowReflectionTests(unittest.TestCase):
    """Where a converted mirror's glass ends up.

    The Covet is the one vehicle the game ships in both hands, so its own LHD
    and RHD rows are the specification. It keeps the node columns across the
    change and lets the offset carry the reflection, which works because it
    hangs its interior mirror off ``rf1``, a centreline node.
    """

    # rf1 is on the centreline; the hopper's wi3l is 0.28 m out on the left.
    NODES = {"rf1": (0.0, -0.2, 1.35), "wi3l": (0.28, -0.09, 1.6)}

    COVET_LHD = (
        '["covet_intmirror","rf1","rf1r","rf2",'
        '{"refBaseTranslation":{"x":0.108,"y":0.00,"z":-0.139},'
        '"baseRotationGlobal":{"x":0,"y":0,"z":12}}],'
    )
    HOPPER_LHD = (
        '["hopper_intmirror","wi3l","wi2l","wi3r",'
        '{"refBaseTranslation":{"x":-0.265,"y":0.049,"z":-0.077},'
        '"baseRotationGlobal":{"x":0,"y":0,"z":5}}],'
    )

    def _rewrite(self, row, mesh, mode, renamed=None, nodes=None):
        sources = core.mirror_plane_sources_for_meshes([mesh], {mesh: mode}, {mesh: row})
        out = core.rewrite_mirror_rows(
            "[\n" + row + "\n]",
            {mesh: renamed} if renamed else {},
            sources,
            self.NODES if nodes is None else nodes,
        )
        return out.strip().splitlines()[1].strip()

    def _world_x(self, row):
        ref = core.mirror_row_node_ids(row)[0]
        offset = re.search(r'"refBaseTranslation":\{"x":(-?[\d.]+)', row)
        return round(self.NODES[ref][0] + float(offset.group(1)), 4)

    def test_converting_the_covet_reproduces_the_rhd_row_the_game_authored(self) -> None:
        row = self._rewrite(self.COVET_LHD, "covet_intmirror", core.MODE_MIRROR,
                            "covet_intmirror_rhd")
        # covet_intmirror -> covet_intmirror_rhd: same nodes, x negated, z negated
        self.assertIn('["covet_intmirror_rhd","rf1","rf1r","rf2"', row)
        self.assertIn('"refBaseTranslation":{"x":-0.108,"y":0,"z":-0.139}', row)
        self.assertIn('"baseRotationGlobal":{"x":0,"y":0,"z":-12}', row)

    def test_an_off_centre_mount_still_reflects_the_glass(self) -> None:
        """A sign flip only reflects when the reference node is on the centreline.

        The engine places the glass at ``v.offset + nodes[v.idRef].pos``, so an
        offset measured from a node 0.28 m out on the left has to absorb that
        node too. Negating alone drove the hopper's mirror from 15 mm off the
        centreline to 545 mm out, against the door pillar.
        """
        authored_x = self._world_x(self.HOPPER_LHD)
        self.assertAlmostEqual(authored_x, 0.015)

        row = self._rewrite(self.HOPPER_LHD, "hopper_intmirror", core.MODE_MIRROR)
        self.assertEqual(core.mirror_row_node_ids(row), ["wi3l", "wi2l", "wi3r"])
        self.assertAlmostEqual(self._world_x(row), -authored_x)
        self.assertIn('"baseRotationGlobal":{"x":0,"y":0,"z":-5}', row)

    def test_without_node_positions_it_falls_back_to_the_sign_flip(self) -> None:
        row = self._rewrite(self.COVET_LHD, "covet_intmirror", core.MODE_MIRROR,
                            nodes={})
        self.assertIn('"refBaseTranslation":{"x":-0.108,"y":0,"z":-0.139}', row)

    def test_a_swap_moves_the_glass_nowhere(self) -> None:
        wing = (
            '["covet_mirror_L","mi4l","mi2l","mi3l",'
            '{"refBaseTranslation":{"x":-0.09,"y":0.00,"z":0.04},'
            '"baseRotationGlobal":{"x":0.2,"y":0.0,"z":-13.4}}],'
        )
        row = self._rewrite(
            wing, "covet_mirror_L", core.MODE_MIRROR_STRUCTURAL, "covet_mirror_RHD_L"
        )
        # covet_mirror_L -> covet_mirror_L_rhd is a mesh rename and nothing else
        self.assertIn('["covet_mirror_RHD_L","mi4l"', row)
        self.assertIn('"refBaseTranslation":{"x":-0.09,"y":0.00,"z":0.04}', row)
        self.assertIn('"baseRotationGlobal":{"x":0.2,"y":0.0,"z":-13.4}', row)

    def test_the_options_object_is_not_mistaken_for_a_node_column(self) -> None:
        self.assertEqual(
            core.mirror_row_node_ids(self.COVET_LHD), ["rf1", "rf1r", "rf2"]
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
