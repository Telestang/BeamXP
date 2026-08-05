"""Mirroring of interaction trigger boxes ("triggers2" sections).

The reference cases are taken from stock jbeam: etk800 ships a hand-authored
door_FR_int/door_FL_int pair, so mirroring one must reproduce the other.
"""

import unittest

from beamxp.hand_drive_parts.rewriting import (
    build_node_mirror_map,
    mirror_trigger_offset,
    rewrite_triggers,
    trigger_column_names,
    trigger_frame,
    triggers_needing_manual_review,
)

HEADER = (
    '        ["id", "idRef:", "idX:", "idY:", "type", "size", "baseRotation",'
    ' "rotation", "translation", "baseTranslation"],'
)

# etk800_doors_F.jbeam door node cage, right side; the left side is its mirror.
ETK_DOOR_NODES_R = {
    "d4r": (-0.91, -0.76, 0.60),
    "d6r": (-0.91, 0.27, 0.62),
    "d7r": (-0.85, -0.73, 0.86),
    "d8r": (-0.85, -0.28, 0.88),
    "d9r": (-0.85, 0.30, 0.90),
}
ETK_DOOR_NODES = dict(ETK_DOOR_NODES_R)
ETK_DOOR_NODES.update(
    {f"{node_id[:-1]}l": (-x, y, z) for node_id, (x, y, z) in ETK_DOOR_NODES_R.items()}
)

# ardente_interior.jbeam: dash and floor cage, plus the driver-side stalk.
ARDENTE_NODES = {
    "dshl": (0.355, -0.6203, 0.9431),
    "dshr": (-0.355, -0.6203, 0.9431),
    "dsh2l": (0.355, -0.5735, 0.7659),
    "dsh2r": (-0.355, -0.5735, 0.7659),
    "int_strw": (0.355, -0.4397, 0.8037),
    "int_stalk": (0.5656, -0.455, 0.8367),
    "f2l": (0.16, -0.2936, 0.233),
    "f2r": (-0.16, -0.2936, 0.233),
    "f3l": (0.16, 0.3295, 0.233),
    "f3r": (-0.16, 0.3295, 0.233),
}


def section(*rows: str) -> str:
    return "[\n" + HEADER + "\n" + "\n".join(rows) + "\n    "


def mirror(text: str, nodes: dict) -> str:
    return rewrite_triggers(text, nodes, build_node_mirror_map(nodes))


class TriggerFrameTest(unittest.TestCase):
    def test_frame_flips_only_in_z_under_reflection(self):
        source = trigger_frame(*(ETK_DOOR_NODES[n] for n in ("d7r", "d8r", "d4r")))
        target = trigger_frame(*(ETK_DOOR_NODES[n] for n in ("d7l", "d8l", "d4l")))
        for axis in (1, 2):  # x and y axes mirror straight across
            for component, (source_c, target_c) in enumerate(
                zip(source[axis], target[axis])
            ):
                expected = -source_c if component == 0 else source_c
                self.assertAlmostEqual(target_c, expected, places=9)
        # the z axis picks up the extra sign that makes the frame left-handed
        for component, (source_c, target_c) in enumerate(zip(source[3], target[3])):
            expected = source_c if component == 0 else -source_c
            self.assertAlmostEqual(target_c, expected, places=9)

    def test_offset_round_trips_to_the_vanilla_left_hand_value(self):
        source = trigger_frame(*(ETK_DOOR_NODES[n] for n in ("d7r", "d8r", "d4r")))
        target = trigger_frame(*(ETK_DOOR_NODES[n] for n in ("d7l", "d8l", "d4l")))
        result = mirror_trigger_offset((0.45, -0.02, 0.085), source, target)
        for value, expected in zip(result, (0.45, -0.02, -0.085)):
            self.assertAlmostEqual(value, expected, places=9)


class TriggerRewriteTest(unittest.TestCase):
    def test_reproduces_the_stock_etk800_door_pair(self):
        text = section(
            '        ["door_FR_int", "d7r","d8r","d4r", "box", {"x":0.15, "y":0.05, "z":0.06},'
            ' {"x":-12, "y":0, "z":-0.2}, {"x":0, "y":0, "z":0}, {"x":0, "y":0, "z":0},'
            ' {"x":0.45, "y":-0.02, "z":0.085}],'
        )
        out = mirror(text, ETK_DOOR_NODES)
        self.assertIn('"d7l","d8l","d4l"', out)
        # baseRotation.x negates, .z is kept; baseTranslation.z negates
        self.assertIn('{"x":12, "y":0, "z":-0.2}', out)
        self.assertIn('{"x":0.45, "y":-0.02, "z":-0.085}', out)
        # the box extents are untouched
        self.assertIn('{"x":0.15, "y":0.05, "z":0.06}', out)

    def test_header_row_is_not_treated_as_a_trigger(self):
        text = section(
            '        ["hazard", "dsh2l","dsh2r","dshl", "box", {"x":0.03, "y":0.025, "z":0.015},'
            ' {"x":45, "y":0, "z":0}, {"x":0, "y":0, "z":0}, {"x":0, "y":0, "z":0},'
            ' {"x":0.34, "y":-0.04, "z":-0.09}],'
        )
        out = mirror(text, ARDENTE_NODES)
        self.assertIn(HEADER, out)
        self.assertEqual(trigger_column_names(out)[1], "idRef")

    def test_console_buttons_swap_their_reversed_ref_triples(self):
        text = section(
            '        ["sw_ignition", "f2l","f2r","f3l", "sphere", 0.015,'
            ' {"x":0, "y":0, "z":0}, {"x":0, "y":0, "z":0}, {"x":0, "y":0, "z":0},'
            ' {"x":0.125, "y":0.052, "z":0.34}],',
            '        ["toggleESCMode", "f2r","f2l","f3r", "box", {"x":0.025, "y":0.025, "z":0.025},'
            ' {"x":0, "y":0, "z":0}, {"x":0, "y":0, "z":0}, {"x":0, "y":0, "z":0},'
            ' {"x":0.109, "y":0.084, "z":-0.33}],'
        )
        out = mirror(text, ARDENTE_NODES)
        self.assertIn('"f2r","f2l","f3r", "sphere"', out)
        self.assertIn('"f2l","f2r","f3l", "box"', out)
        # a sphere's radius is a scalar, not a vector, and must survive intact
        self.assertIn('"sphere", 0.015,', out)

    def test_mirrored_box_lands_on_the_mirrored_world_point(self):
        text = section(
            '        ["tailgate_int", "dsh2l","dsh2r","dshl", "box", {"x":0.025, "y":0.025, "z":0.025},'
            ' {"x":0, "y":0, "z":0}, {"x":0, "y":0, "z":0}, {"x":0, "y":0, "z":0},'
            ' {"x":-0.32, "y":0.01, "z":-0.04}],'
        )
        source_frame = trigger_frame(*(ARDENTE_NODES[n] for n in ("dsh2l", "dsh2r", "dshl")))
        before = [
            source_frame[0][i]
            + sum((-0.32, 0.01, -0.04)[a] * source_frame[a + 1][i] for a in range(3))
            for i in range(3)
        ]
        out = mirror(text, ARDENTE_NODES)
        columns = trigger_column_names(out)
        self.assertEqual(columns[-1], "baseTranslation")
        row = [line for line in out.splitlines() if "tailgate_int" in line][0]
        values = tuple(
            float(part) for part in row.rsplit("{", 1)[1].rstrip("}],").replace('"', "").split(",")
            for part in [part.split(":")[1]]
        )
        target_frame = trigger_frame(*(ARDENTE_NODES[n] for n in ("dsh2r", "dsh2l", "dshr")))
        after = [
            target_frame[0][i] + sum(values[a] * target_frame[a + 1][i] for a in range(3))
            for i in range(3)
        ]
        self.assertAlmostEqual(after[0], -before[0], places=6)
        self.assertAlmostEqual(after[1], before[1], places=6)
        self.assertAlmostEqual(after[2], before[2], places=6)

    def test_twinned_pairs_are_left_alone(self):
        text = section(
            '        ["sunvisor_L_open", "rf1","rf1l","rf2", "box", {"x":0.09, "y":0.04, "z":0.04},'
            ' {"x":0, "y":0, "z":0}, {"x":0, "y":0, "z":0}, {"x":0, "y":0, "z":0},'
            ' {"x":0.36, "y":-0.08, "z":0.02}],',
            '        ["sunvisor_R_open", "rf1","rf1r","rf2", "box", {"x":0.09, "y":0.04, "z":0.04},'
            ' {"x":0, "y":0, "z":0}, {"x":0, "y":0, "z":0}, {"x":0, "y":0, "z":0},'
            ' {"x":0.36, "y":-0.08, "z":-0.06}],'
        )
        nodes = {
            "rf1": (0.0, 0.1, 1.3),
            "rf1l": (0.35, 0.1, 1.3),
            "rf1r": (-0.35, 0.1, 1.3),
            "rf2": (0.0, 0.5, 1.3),
        }
        self.assertEqual(mirror(text, nodes), text)

    def test_unmappable_ref_node_leaves_the_row_and_flags_it(self):
        text = section(
            '        ["headlights", "int_strw","int_stalk","dshr", "sphere", 0.025,'
            ' {"x":0, "y":0, "z":0}, {"x":0, "y":0, "z":0}, {"x":0, "y":0, "z":0},'
            ' {"x":0.2, "y":0, "z":0}],'
        )
        out = mirror(text, ARDENTE_NODES)
        self.assertIn('"int_strw","int_stalk","dshr"', out)
        self.assertIn("//BeamXP: headlights left unmirrored", out)
        self.assertIn("int_stalk", out.split("//BeamXP:")[1].split("\n")[0])

    def test_manual_review_reports_the_unmappable_trigger(self):
        body = '{"triggers2":' + section(
            '        ["headlights", "int_strw","int_stalk","dshr", "sphere", 0.025,'
            ' {"x":0, "y":0, "z":0}, {"x":0, "y":0, "z":0}, {"x":0, "y":0, "z":0},'
            ' {"x":0.2, "y":0, "z":0}],'
        ) + "]}"
        findings = triggers_needing_manual_review(
            body, ARDENTE_NODES, build_node_mirror_map(ARDENTE_NODES)
        )
        self.assertEqual([name for name, _reason in findings], ["headlights"])


if __name__ == "__main__":
    unittest.main()
