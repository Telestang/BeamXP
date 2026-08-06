"""Mirroring of interaction trigger boxes ("triggers2" sections).

The reference cases are taken from stock jbeam: etk800 ships a hand-authored
door_FR_int/door_FL_int pair, so mirroring one must reproduce the other.
"""

import unittest

import math

from beamxp import transform_helpers
from beamxp.core import sjson
from beamxp.core.constants import HAND_RHD
from beamxp.hand_drive_parts.rewriting import (
    build_node_mirror_map,
    generate_trigger_frame_twins,
    hydro_driven_nodes,
    note_trigger_frames_in_part,
    mirror_trigger_offset,
    rewrite_triggers,
    trigger_column_names,
    trigger_frame,
    local_to_world,
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


# A trigger inherits the verdict of the geometry it is mounted on, so every
# case has to say what the config did. Owners are (prop rest pivots, anchor
# node -> transform); the default here puts every node on mirrored geometry.
def mirror(text: str, nodes: dict, owners: tuple | None = None) -> str:
    if owners is None:
        owners = ([], {node: ("mirror", 0.0) for node in nodes}, [])
    return rewrite_triggers(text, nodes, build_node_mirror_map(nodes), owners)


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
        # box extents behave like a signed local corner, so y changes sign
        self.assertIn('{"x":0.15, "y":-0.05, "z":0.06}', out)

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

    def test_unmirrorable_ref_nodes_keep_the_anchor_and_move_the_box(self):
        """A steering stalk exists on one side only, so there is nothing to
        repoint at. BeamXP relocates a mirrored prop on such nodes with
        baseTranslationGlobal and leaves its ref nodes alone; the trigger has to
        follow suit or it parts company with the geometry it labels."""
        text = section(
            '        ["headlights", "int_strw","int_stalk","dshr", "sphere", 0.025,'
            ' {"x":0, "y":0, "z":0}, {"x":0, "y":0, "z":0}, {"x":0, "y":0, "z":0},'
            ' {"x":0.2, "y":0, "z":0}],'
        )
        out = mirror(text, ARDENTE_NODES)
        self.assertIn('"int_strw","int_stalk","dshr"', out)
        # all-zero rotations need no approximation, so nothing is flagged
        self.assertNotIn("//BeamXP:", out)

        frame = trigger_frame(*(ARDENTE_NODES[n] for n in ("int_strw", "int_stalk", "dshr")))
        before = [
            frame[0][i] + sum((0.2, 0.0, 0.0)[a] * frame[a + 1][i] for a in range(3))
            for i in range(3)
        ]
        row = [line for line in out.splitlines() if "headlights" in line][0]
        values = [
            float(part.split(":")[1])
            for part in row.rsplit("{", 1)[1].rstrip("}],").replace('"', "").split(",")
        ]
        after = [
            frame[0][i] + sum(values[a] * frame[a + 1][i] for a in range(3)) for i in range(3)
        ]
        self.assertAlmostEqual(after[0], -before[0], places=6)
        self.assertAlmostEqual(after[1], before[1], places=6)
        self.assertAlmostEqual(after[2], before[2], places=6)

    def test_translation_column_is_reflected_as_a_free_vector(self):
        """baseTranslation positions the box so it carries the frame origin;
        translation stacks on top of it. Reflecting the origin twice only shows
        up once the anchor cannot be repointed, so pin it on the stalk case."""
        text = section(
            '        ["headlights", "int_strw","int_stalk","dshr", "sphere", 0.025,'
            ' {"x":0, "y":0, "z":0}, {"x":0, "y":0, "z":0}, {"x":0, "y":0, "z":0},'
            ' {"x":0.2, "y":0, "z":0}],'
        )
        out = mirror(text, ARDENTE_NODES)
        row = [line for line in out.splitlines() if "headlights" in line][0]
        # rotation, rotation, translation all stay zero; only baseTranslation moves
        self.assertEqual(row.count('{"x":0, "y":0, "z":0}'), 3)

    def test_rotation_is_kept_when_the_frame_does_not_reflect(self):
        """The stalk cage exists on one side only, so its frame is unchanged and
        the box keeps pointing the way it was authored. That is what the part
        does: BeamXP copies a mirrored prop's rotation across untouched and moves
        it with baseTranslationGlobal, because the mirroring is in the DAE."""
        text = section(
            '        ["headlights", "int_strw","int_stalk","dshr", "box", {"x":0.03, "y":0.03, "z":0.03},'
            ' {"x":30, "y":0, "z":0}, {"x":0, "y":0, "z":0}, {"x":0, "y":0, "z":0},'
            ' {"x":0.2, "y":0, "z":0}],'
        )
        out = mirror(text, ARDENTE_NODES)
        self.assertIn('{"x":30, "y":0, "z":0}', out)
        self.assertNotIn("//BeamXP:", out)

    def test_nothing_needs_manual_review_on_a_resolvable_part(self):
        body = '{"triggers2":' + section(
            '        ["headlights", "int_strw","int_stalk","dshr", "box", {"x":0.03, "y":0.03, "z":0.03},'
            ' {"x":30, "y":0, "z":0}, {"x":0, "y":0, "z":0}, {"x":0, "y":0, "z":0},'
            ' {"x":0.2, "y":0, "z":0}],'
        ) + "]}"
        findings = triggers_needing_manual_review(
            body, ARDENTE_NODES, build_node_mirror_map(ARDENTE_NODES)
        )
        self.assertEqual(findings, [])

    def test_trigger_on_a_translated_mesh_slides_instead_of_mirroring(self):
        """The Ardente's indicator stalk is mode translate: it slides across by
        the steering offset without being reflected. Mirroring its headlight
        trigger instead of sliding it lands the box 12.7 cm out."""
        text = section(
            '        ["headlights", "int_strw","int_stalk","dshr", "sphere", 0.025,'
            ' {"x":0, "y":0, "z":0}, {"x":0, "y":0, "z":0}, {"x":0, "y":0, "z":0},'
            ' {"x":0.2, "y":0, "z":0}],'
        )
        # ardente_signalstalk sits at x=+0.4186 and moves by the steering offset
        owners = ([((0.4186, -0.4588, 0.7969), "translate", -0.71, None)], {}, [])
        frame = trigger_frame(*(ARDENTE_NODES[n] for n in ("int_strw", "int_stalk", "dshr")))
        before = local_to_world(frame, (0.2, 0.0, 0.0))

        out = mirror(text, ARDENTE_NODES, owners)
        # the anchor and the rotations are untouched by a pure slide
        self.assertIn('"int_strw","int_stalk","dshr"', out)
        row = [line for line in out.splitlines() if "headlights" in line][0]
        values = tuple(
            float(part.split(":")[1])
            for part in row.rsplit("{", 1)[1].rstrip("}],").replace('"', "").split(",")
        )
        after = local_to_world(frame, values)
        self.assertAlmostEqual(after[0], before[0] - 0.71, places=6)
        self.assertAlmostEqual(after[1], before[1], places=6)
        self.assertAlmostEqual(after[2], before[2], places=6)

    def test_trigger_on_a_skipped_mesh_stays_put(self):
        text = section(
            '        ["headlights", "int_strw","int_stalk","dshr", "sphere", 0.025,'
            ' {"x":0, "y":0, "z":0}, {"x":0, "y":0, "z":0}, {"x":0, "y":0, "z":0},'
            ' {"x":0.2, "y":0, "z":0}],'
        )
        owners = ([((0.4186, -0.4588, 0.7969), "skip", 0.0, None)], {}, [])
        out = mirror(text, ARDENTE_NODES, owners)
        self.assertEqual(out, text)

    def test_nearest_mesh_wins_the_association(self):
        """Steering wheel, wiper stalk and indicator stalk all sit within 15 cm
        of the headlight trigger; only the indicator stalk should claim it."""
        text = section(
            '        ["headlights", "int_strw","int_stalk","dshr", "sphere", 0.025,'
            ' {"x":0, "y":0, "z":0}, {"x":0, "y":0, "z":0}, {"x":0, "y":0, "z":0},'
            ' {"x":0.2, "y":0, "z":0}],'
        )
        owners = (
            [
                ((0.4186, -0.4588, 0.7969), "translate", -0.71, None),
                ((0.2960, -0.4588, 0.7969), "skip", 0.0, None),
                ((0.3550, -0.4397, 0.8037), "mirror", 0.0, None),
            ],
            {},
            [],
        )
        out = mirror(text, ARDENTE_NODES, owners)
        self.assertNotEqual(out, text)  # not the skipped wiper stalk
        self.assertIn('"int_strw","int_stalk","dshr"', out)  # not the mirrored wheel
        row = [line for line in out.splitlines() if "headlights" in line][0]
        frame = trigger_frame(*(ARDENTE_NODES[n] for n in ("int_strw", "int_stalk", "dshr")))
        values = tuple(
            float(part.split(":")[1])
            for part in row.rsplit("{", 1)[1].rstrip("}],").replace('"', "").split(",")
        )
        self.assertAlmostEqual(
            local_to_world(frame, values)[0],
            local_to_world(frame, (0.2, 0.0, 0.0))[0] - 0.71,
            places=6,
        )

    def test_a_mesh_with_no_verdict_leaves_the_trigger_alone(self):
        """There is no default transform. If the config says nothing about the
        geometry, the box stays exactly where it was authored."""
        text = section(
            '        ["hazard", "dsh2l","dsh2r","dshl", "box", {"x":0.03, "y":0.025, "z":0.015},'
            ' {"x":45, "y":0, "z":0}, {"x":0, "y":0, "z":0}, {"x":0, "y":0, "z":0},'
            ' {"x":0.34, "y":-0.04, "z":-0.09}],'
        )
        self.assertEqual(mirror(text, ARDENTE_NODES, ([], {}, [])), text)

    def test_rotation_x_negates_when_the_frame_reflects(self):
        """Fitted from vanilla: across etk800, bx and covet the authored x always
        negates while y is always kept."""
        text = section(
            '        ["door_R_int", "d7r","d8r","d4r", "box", {"x":0.19, "y":0.03, "z":0.06},'
            ' {"x":-13, "y":1, "z":1}, {"x":0, "y":0, "z":0}, {"x":0, "y":0, "z":0},'
            ' {"x":0.51, "y":0.062, "z":0.085}],'
        )
        out = mirror(text, ETK_DOOR_NODES)
        self.assertIn('{"x":13, "y":1, "z":1}', out)


# ardente_interior.jbeam, trimmed to the rows the stalk frame is built from.
# The turn signal stalk hangs off int_strw and a torsionHydro swings int_stalk
# from the "turnsignal" input; the headlight trigger rides 0.2 m along that
# axis, which is what makes it follow the stalk.
ARDENTE_STALK_PART = """"ardente_dash": {
    "slotType": "ardente_dash",
    "triggers2":[
      ["id", "idRef:", "idX:", "idY:", "type", "size", "baseRotation", "rotation", "translation", "baseTranslation"],
      ["hazard", "dsh2l","dsh2r","dshl","box",{"x":0.030, "y":0.025, "z":0.015},{"x":45,"y":0, "z":0}, {"x":0, "y":0, "z":0}, {"x":0, "y":0, "z":0}, {"x": 0.340, "y":-0.040, "z":-0.090}],
      ["headlights", "int_strw","int_stalk","dshr","sphere", 0.025, {"x":0, "y":0, "z":0}, {"x":0, "y":0, "z":0}, {"x":0, "y":0, "z":0}, {"x": 0.200, "y": 0.000, "z": 0.000}],
    ],
    "nodes": [
         ["id", "posX", "posY", "posZ"],
         {"group":["ardente_dash"]},
         {"nodeWeight":15},
         ["dshr", -0.355, -0.6203, 0.9431],
         ["dshl", 0.355, -0.6203, 0.9431],
         ["dsh2r", -0.355, -0.5735, 0.7659],
         ["dsh2l", 0.355, -0.5735, 0.7659],
         {"group":""},
         {"nodeWeight":1},
         ["int_strw", 0.355, -0.4397, 0.8037],
         {"nodeWeight":0.2},
         ["int_stalk", 0.5656, -0.455, 0.8367],
    ],
    "torsionbars":[
       ["id1:", "id2:", "id3:", "id4:"],
        {"spring":20000, "damp":10, "deform":18000, "strength":28000},
        ["int_strw", "dsh2l", "dsh2r", "f5l"],
    ],
    "torsionHydros": [
        ["id1:","id2:","id3:","id4:"],
        {"spring":100, "damp":1, "deform":"FLT_MAX", "strength":1000},
        ["int_stalk","int_strw","dsh2l","dsh2r",  {"inputSource":"turnsignal","factor":-0.12}],
    ],
    "beams":[
          ["id1:", "id2:"],
          {"beamSpring":7001000,"beamDamp":150},
          ["dsh2r",    "int_strw"],
          ["dsh2l",    "int_strw"],
          {"beamSpring":160100,"beamDamp":142.73},
          ["int_stalk","dsh2l"],
          ["int_stalk","int_strw"],
    ],
}"""

STALK_NODES = dict(ARDENTE_NODES)
STALK_NODES["f5l"] = (0.33, -0.9, 0.30)
STALK_NODES["f5r"] = (-0.33, -0.9, 0.30)

# ardente_signalstalk sits at x=+0.4186; the config slides it across by the
# steering offset rather than reflecting it.
STALK_OWNERS = ([((0.4186, -0.4588, 0.7969), "translate", -0.71, None)], {}, [])


def build_twins(part=ARDENTE_STALK_PART, nodes=STALK_NODES, owners=STALK_OWNERS):
    return generate_trigger_frame_twins(
        part, nodes, build_node_mirror_map(nodes), owners, HAND_RHD
    )


def box_world(nodes, ref_ids, offset):
    return local_to_world(trigger_frame(*(nodes[node] for node in ref_ids)), offset)


def box_swing(nodes, ref_ids, offset, pivot, stalk, angle=0.12):
    """How far and which way the box moves when the stalk swings.

    The torsionHydro turns the stalk node about its column, so yawing that node
    about the column is a fair stand-in for what the input does at runtime.
    """
    origin = nodes[pivot]
    arm = [nodes[stalk][i] - origin[i] for i in range(3)]
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    swung = dict(nodes)
    swung[stalk] = (
        origin[0] + arm[0] * cos_a - arm[1] * sin_a,
        origin[1] + arm[0] * sin_a + arm[1] * cos_a,
        origin[2] + arm[2],
    )
    at_rest = box_world(nodes, ref_ids, offset)
    return tuple(box_world(swung, ref_ids, offset)[i] - at_rest[i] for i in range(3))


class TriggerFrameTwinTest(unittest.TestCase):
    """Triggers whose ref frame is animated need the frame itself moved.

    The box offset is a constant, so the only thing that can move a trigger at
    runtime is its ref nodes moving. Rewriting only the offset therefore keeps
    the authored pivot and stretches the arc; the frame has to be rebuilt on
    the converted side for the box to track the stalk it labels.
    """

    def test_the_driven_node_is_read_from_the_hydro_section(self):
        self.assertEqual(hydro_driven_nodes(ARDENTE_STALK_PART), {"int_stalk"})

    def test_only_the_animated_frame_gets_twins(self):
        _body, twins, positions, notes = build_twins()
        self.assertEqual(set(twins), {"int_strw", "int_stalk"})  # not dsh2l/dshl
        self.assertEqual(notes, [])
        # placed by the same transform the box itself is moved with
        self.assertAlmostEqual(positions[twins["int_strw"]][0], -0.355, places=6)
        self.assertAlmostEqual(positions[twins["int_stalk"]][0], 0.5656 - 0.71, places=6)
        for node in ("int_strw", "int_stalk"):
            for axis in (1, 2):
                self.assertAlmostEqual(
                    positions[twins[node]][axis], STALK_NODES[node][axis], places=6
                )

    def test_the_trigger_is_repointed_and_keeps_its_authored_offset(self):
        body, twins, positions, _notes = build_twins()
        nodes = {**STALK_NODES, **positions}
        out = rewrite_triggers(
            transform_helpers.extract_named_array(body, "triggers2"),
            nodes,
            build_node_mirror_map(STALK_NODES),
            STALK_OWNERS,
            twins,
        )
        row = [line for line in out.splitlines() if "headlights" in line][0]
        self.assertIn(f'"{twins["int_strw"]}","{twins["int_stalk"]}","dshr"', row)
        # the offset is the authored one again, the way vanilla bx keeps 0.2 m
        # on both hands rather than growing a half-metre lever
        self.assertIn('{"x": 0.2, "y": 0, "z": 0}', row)

    def test_the_box_lands_where_the_old_slide_put_it(self):
        """The frame moves under the same transform, so the rest position is
        unchanged -- only the arc it sweeps is different."""
        body, twins, positions, _notes = build_twins()
        nodes = {**STALK_NODES, **positions}
        before = box_world(STALK_NODES, ("int_strw", "int_stalk", "dshr"), (0.2, 0, 0))
        after = box_world(
            nodes, (twins["int_strw"], twins["int_stalk"], "dshr"), (0.2, 0, 0)
        )
        self.assertAlmostEqual(after[0], before[0] - 0.71, places=6)
        self.assertAlmostEqual(after[1], before[1], places=6)
        self.assertAlmostEqual(after[2], before[2], places=6)

    def test_the_arc_matches_the_authored_one_instead_of_stretching(self):
        body, twins, positions, _notes = build_twins()
        nodes = {**STALK_NODES, **positions}
        authored = box_swing(
            STALK_NODES, ("int_strw", "int_stalk", "dshr"), (0.2, 0, 0),
            "int_strw", "int_stalk",
        )
        # the offset the shipped vivace conversion carries: authored frame kept,
        # box slid across, which is a 0.51 m lever on the wrong side's column
        stretched = box_swing(
            STALK_NODES, ("int_strw", "int_stalk", "dshr"),
            (-0.499641, 0.114806, 0.037709),
            "int_strw", "int_stalk",
        )
        rebuilt = box_swing(
            nodes, (twins["int_strw"], twins["int_stalk"], "dshr"), (0.2, 0, 0),
            twins["int_strw"], twins["int_stalk"],
        )
        # a slide leaves free vectors alone, so a correct frame reproduces the
        # authored displacement exactly
        for value, expected in zip(rebuilt, authored):
            self.assertAlmostEqual(value, expected, places=9)
        # keeping the authored frame swings the box backwards -- the offset now
        # points behind the pivot -- and 1.6x further: 37 mm against 23.7 mm
        self.assertLess(sum(stretched[i] * authored[i] for i in range(3)), 0)
        self.assertGreater(
            math.dist(stretched, (0, 0, 0)), math.dist(authored, (0, 0, 0)) * 1.5
        )

    def test_the_twins_get_the_rows_that_hold_and_drive_them(self):
        body, twins, _positions, _notes = build_twins()
        strw, stalk = twins["int_strw"], twins["int_stalk"]
        nodes = transform_helpers.extract_named_array(body, "nodes")
        beams = transform_helpers.extract_named_array(body, "beams")
        hydros = transform_helpers.extract_named_array(body, "torsionHydros")
        bars = transform_helpers.extract_named_array(body, "torsionbars")
        # declared right after their source rows, so the 0.2 kg weight and the
        # empty node group carry over rather than being invented
        self.assertLess(nodes.index('["int_stalk"'), nodes.index(f'["{stalk}"'))
        self.assertLess(nodes.index(f'["{strw}"'), nodes.index('["int_stalk"'))
        for pair in (
            f'["dsh2r",    "{strw}"]',
            f'["dsh2l",    "{strw}"]',
            f'["{stalk}","dsh2l"]',
            f'["{stalk}","{strw}"]',
        ):
            self.assertIn(pair, beams)
        self.assertIn(f'["{strw}", "dsh2l", "dsh2r", "f5l"]', bars)
        # the driver is copied with its input and factor untouched: a slid stalk
        # keeps its prop's ref nodes, so it keeps its rotation sense too
        self.assertIn(
            f'["{stalk}","{strw}","dsh2l","dsh2r",'
            '  {"inputSource":"turnsignal","factor":-0.12}]',
            hydros,
        )
        # the twin rows go in as rows, not as text that happens to look like one
        part = sjson.decode("{" + body + "}")["ardente_dash"]
        self.assertIn([strw, -0.355, -0.4397, 0.8037], part["nodes"])
        self.assertIn([stalk, -0.1444, -0.455, 0.8367], part["nodes"])
        self.assertIn([stalk, strw], part["beams"])
        self.assertEqual(
            part["torsionHydros"][-1],
            [stalk, strw, "dsh2l", "dsh2r", {"inputSource": "turnsignal", "factor": -0.12}],
        )

    def test_a_frame_with_no_beams_to_copy_is_left_alone(self):
        part = ARDENTE_STALK_PART.replace('["dsh2r",    "int_strw"],', "").replace(
            '["dsh2l",    "int_strw"],', ""
        ).replace('["int_stalk","int_strw"],', "")
        body, twins, positions, notes = build_twins(part=part)
        self.assertEqual(body, part)
        self.assertEqual(twins, {})
        self.assertEqual(positions, {})
        self.assertEqual([reason for _id, reason in notes], ["ref node int_strw has no beams to copy"])

    def test_a_reflected_frame_repoints_its_anchors_at_the_mirrored_cage(self):
        """bx's authored RHD interior hangs its mirrored column off the
        right-hand dash nodes, so a reflecting verdict has to do the same."""
        owners = ([((0.4186, -0.4588, 0.7969), "mirror", 0.0, None)], {}, [])
        body, twins, positions, notes = build_twins(owners=owners)
        self.assertEqual(notes, [])
        self.assertAlmostEqual(positions[twins["int_stalk"]][0], -0.5656, places=6)
        beams = transform_helpers.extract_named_array(body, "beams")
        self.assertIn(f'["dsh2l",    "{twins["int_strw"]}"]', beams)
        self.assertIn(f'["{twins["int_stalk"]}","dsh2r"]', beams)
        bars = transform_helpers.extract_named_array(body, "torsionbars")
        self.assertIn(f'["{twins["int_strw"]}", "dsh2r", "dsh2l", "f5r"]', bars)

    def test_a_reflected_frame_with_no_mirrored_anchor_generates_nothing(self):
        """No guess is available for where an unmirrorable anchor should go, so
        the trigger keeps today's behaviour and says why."""
        nodes = {node: pos for node, pos in STALK_NODES.items() if node != "f5r"}
        owners = ([((0.4186, -0.4588, 0.7969), "mirror", 0.0, None)], {}, [])
        body, twins, _positions, notes = build_twins(nodes=nodes, owners=owners)
        self.assertEqual(body, ARDENTE_STALK_PART)
        self.assertEqual(twins, {})
        self.assertEqual(
            [reason for _id, reason in notes],
            ["a torsionbars row on the frame cannot be repointed"],
        )
        annotated = note_trigger_frames_in_part(body, notes)
        self.assertIn(
            "//BeamXP: trigger frame int_stalk/int_strw -- "
            "a torsionbars row on the frame cannot be repointed",
            annotated,
        )
        # the note goes in as a comment, so the part still parses
        self.assertEqual(
            sjson.decode("{" + annotated + "}")["ardente_dash"]["triggers2"],
            sjson.decode("{" + body + "}")["ardente_dash"]["triggers2"],
        )

    def test_a_still_frame_is_never_rebuilt(self):
        """Only the animated frame is worth two nodes and a hydro; a trigger on
        a static cage is already placed correctly by moving its offset."""
        part = ARDENTE_STALK_PART.replace(
            '["int_stalk","int_strw","dsh2l","dsh2r",  {"inputSource":"turnsignal","factor":-0.12}],',
            "",
        )
        body, twins, _positions, notes = build_twins(part=part)
        self.assertEqual((body, twins, notes), (part, {}, []))

    def test_twins_are_shared_by_every_trigger_on_the_same_frame(self):
        part = ARDENTE_STALK_PART.replace(
            '["headlights", "int_strw","int_stalk","dshr","sphere", 0.025,',
            '["foglights", "int_strw","int_stalk","dshr","sphere", 0.025,'
            ' {"x":0, "y":0, "z":0}, {"x":0, "y":0, "z":0}, {"x":0, "y":0, "z":0},'
            ' {"x": 0.180, "y": 0.000, "z": 0.000}],\n'
            '      ["headlights", "int_strw","int_stalk","dshr","sphere", 0.025,',
        )
        body, twins, positions, _notes = build_twins(part=part)
        self.assertEqual(len(positions), 2)
        nodes = transform_helpers.extract_named_array(body, "nodes")
        self.assertEqual(nodes.count(f'["{twins["int_stalk"]}"'), 1)

    def test_generated_names_do_not_collide_with_existing_nodes(self):
        nodes = dict(STALK_NODES)
        nodes["int_stalk_xp_rhd"] = (0.0, 0.0, 0.0)
        _body, twins, positions, _notes = build_twins(nodes=nodes)
        self.assertEqual(twins["int_stalk"], "int_stalk_xp_rhd_2")
        self.assertIn("int_stalk_xp_rhd_2", positions)


if __name__ == "__main__":
    unittest.main()
