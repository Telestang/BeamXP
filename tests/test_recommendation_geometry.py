"""2026-09-10: Geometry decisions must preserve controls and atlas assignments."""
from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from beamxp.hand_drive_ui.recommendation_controls import (
    driver_footrest,
    pedal_offsets,
    steering_mount_meshes,
    trigger_hosts,
)
from beamxp.hand_drive_ui.recommendation_engine import (
    _add_trigger_rows,
    _suppress_symmetric_reflections,
    _warning_cluster,
)
from beamxp.hand_drive_ui.recommendation_evidence import (
    VehicleEvidence,
    vehicle_evidence,
)
from beamxp.hand_drive_ui.recommendation_geometry import (
    appearance_symmetric,
    geometry_for_ids,
    projected_symmetry,
    small_component_symmetric,
    surface_distance,
)
from beamxp.hand_drive_ui.recommendation_pairing import primary_card_surface
from beamxp.hand_drive_ui.recommendations_ui import RecommendationsUIMixin
from tests.cabin_fixtures import mesh_context, part_body


def paired_faces():
    left = np.array([[[.1, 0, 0], [.4, 1, 0], [.2, 0, 1]]])
    right = left.copy()
    right[:, :, 0] *= -1
    uv = np.array([[[0., 0.], [1., 0.], [0., 1.]]])
    return [(left, uv, "panel"), (right[:, ::-1], uv[:, ::-1].copy(), "panel")]


class SymmetryTests(unittest.TestCase):
    def test_small_door_attachment_does_not_inherit_panel_lettering(self):
        context = mesh_context({"panel": (.5, 0, 0), "strap": (.5, 0, 0), "unrelated": (.5, 0, 0)})
        context.preview_by_id["panel"]["bounds"] = ((.4, 0, 0), (.6, 1, .6))
        context.preview_by_id["strap"]["bounds"] = ((.4, 0, 0), (.6, .15, .05))
        context.preview_by_id["unrelated"]["bounds"] = ((0, 0, 0), (10, 10, 10))
        evidence = VehicleEvidence(part_ids={"panel": frozenset({"door"}), "strap": frozenset({"door"})},
                                   meshes_by_part={"door": frozenset({"panel", "strap", "unrelated"})})
        handed = {"panel", "strap"}
        self.assertTrue(primary_card_surface(context, evidence, "panel", handed))
        self.assertFalse(primary_card_surface(context, evidence, "strap", handed))
        context.preview_by_id["strap"].pop("bounds")
        self.assertTrue(primary_card_surface(context, evidence, "strap", handed))

    def test_small_control_keeps_lettering_but_large_panel_requires_more_evidence(self):
        points = np.array([[x, y, z] for x in (-.04, .04) for y in (-.05, .05) for z in (-.02, .02)])
        self.assertTrue(small_component_symmetric(points))
        self.assertFalse(small_component_symmetric(points * 10))
        asymmetric = np.vstack([points, [.015, .025, .02]])
        self.assertFalse(small_component_symmetric(asymmetric))

    def test_geometry_reader_applies_centimetres_once_and_reuses_arrays(self):
        context = mesh_context({"panel": (0, 0, 0)})
        tree = ET.ElementTree(ET.fromstring('''
        <COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema">
          <asset><unit meter="0.01"/></asset>
          <library_geometries><geometry id="g"><mesh>
            <source id="p"><float_array id="pa" count="9">-100 0 0 100 0 0 0 100 0</float_array>
              <technique_common><accessor source="#pa" count="3" stride="3">
                <param name="X"/><param name="Y"/><param name="Z"/>
              </accessor></technique_common></source>
            <vertices id="v"><input semantic="POSITION" source="#p"/></vertices>
            <triangles count="1"><input semantic="VERTEX" source="#v" offset="0"/><p>0 1 2</p></triangles>
          </mesh></geometry></library_geometries>
          <library_visual_scenes><visual_scene><node id="panel">
            <matrix>1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</matrix><instance_geometry url="#g"/>
          </node></visual_scene></library_visual_scenes>
        </COLLADA>'''))
        with patch("beamxp.hand_drive_ui.recommendation_geometry.core.parse_dae", return_value=tree) as read:
            result = geometry_for_ids(context, ["panel"])["panel"]
            self.assertAlmostEqual(np.ptp(result["points"][:, 0]), 2.)
            self.assertIs(result, geometry_for_ids(context, ["panel"])["panel"])
            self.assertEqual(read.call_count, 1)

    def test_both_projections_must_agree(self):
        points = np.array([[-1, -1, 0], [-1, 1, 0], [1, -1, 0], [1, 1, 0], [1, 0, 2]])
        self.assertLess(projected_symmetry(points), .6)
        symmetric = np.vstack([points, points * [-1, 1, 1]])
        self.assertAlmostEqual(projected_symmetry(symmetric), 1.)
        symmetric[:, 0] += 2
        self.assertEqual(projected_symmetry(symmetric), 0.)
        self.assertAlmostEqual(projected_symmetry(symmetric, 2), 1.)

    def test_missing_or_degenerate_evidence_is_unknown(self):
        self.assertIsNone(projected_symmetry([]))
        self.assertIsNone(projected_symmetry(np.zeros((4, 3))))

    def test_reflected_faces_preserve_uv_and_material(self):
        faces = paired_faces()
        self.assertTrue(appearance_symmetric(faces, 0))
        faces[1][1][0, 0, 0] += .1
        self.assertFalse(appearance_symmetric(faces, 0))
        faces = paired_faces()
        faces[1] = (*faces[1][:2], "other")
        self.assertFalse(appearance_symmetric(faces, 0))
        faces = paired_faces()
        faces[1] = (faces[1][0][:, ::-1], faces[1][1][:, ::-1], "panel")
        self.assertFalse(appearance_symmetric(faces, 0))

    def test_outline_does_not_erase_asymmetric_internal_detail(self):
        faces = paired_faces()
        interior = np.array([[[.1, .2, .2], [.11, .2, .2], [.1, .21, .2]]])
        faces.append((interior, np.zeros((1, 3, 2)), "panel"))
        self.assertFalse(appearance_symmetric(faces, 0))

    def test_proven_skip_is_an_explicit_row_and_does_not_change_pairs(self):
        faces = paired_faces()
        points = np.concatenate([face[0].reshape(-1, 3) for face in faces])
        context = mesh_context({"panel": (0, 0, 0)})
        rows = [{"object_id": "panel", "mode": "mirror", "kind": "single", "reason": "cabin"},
                {"object_id": "pair", "mode": "mirrorStructural", "kind": "pair"}]
        with patch("beamxp.hand_drive_ui.recommendation_engine.geometry_for_ids",
                   return_value={"panel": {"points": points, "appearance": faces}}):
            _suppress_symmetric_reflections(context, rows, 0)
        self.assertEqual(rows[0]["mode"], "skip")
        self.assertEqual(rows[0]["trigger_mode"], "mirror")
        self.assertEqual(rows[1]["mode"], "mirrorStructural")


class ControlTests(unittest.TestCase):
    def test_footrest_requires_a_nearby_pedal_on_the_driver_side(self):
        centers = {"brake": (.4, -.8, .4), "footrest": (.6, -.8, .35),
                   "rear_footrest": (.6, 1., .35), "passenger_footrest": (-.6, -.8, .35)}
        context = mesh_context(centers)
        evidence = VehicleEvidence(prop_funcs={"brake": frozenset({"brake"})})
        self.assertTrue(driver_footrest(context, evidence, "footrest", centers, 0, 1))
        self.assertFalse(driver_footrest(context, evidence, "rear_footrest", centers, 0, 1))
        self.assertFalse(driver_footrest(context, evidence, "passenger_footrest", centers, 0, 1))

    def test_dedicated_steering_mount_includes_column_selector_and_cover(self):
        nodes = '"nodes":[["id","posX","posY","posZ"],{"group":"mount"},["a",0.4,0,1],["b",0.2,0,1],["c",0.4,0.2,1]]'
        parts = {
            "wheel": part_body("wheel", props=(("steering", "rim"),)),
            "selector": part_body("selector", props=(("gearModeIndex", "lever"),)),
            "column": '{' + nodes + ',"flexbodies":[["mesh","[group]:"],["cover",["mount"]],["dash",["body"]]]}',
        }
        centers = {"rim": (.4, 0, 1), "lever": (.3, 0, 1), "cover": (.4, 0, 1), "dash": (0, 0, 1)}
        context = mesh_context(centers, parts=parts)
        evidence = vehicle_evidence(context)
        self.assertEqual(steering_mount_meshes(context, evidence, "rim"), {"rim", "lever", "cover"})
        # 2026-09-10: A node contributed by another selected part makes this
        # a broader group, even though the wheel's declaring part has three.
        parts["extra"] = '{"nodes":[["id","posX","posY","posZ"],{"group":"mount"},["d",0,0,0]]}'
        context = mesh_context(centers, parts=parts)
        self.assertFalse(steering_mount_meshes(context, vehicle_evidence(context), "rim"))

    def test_compact_prop_rows_and_glow_channels_do_not_depend_on_names(self):
        parts = {"p": '''{"slotType":"body", "props":[["mesh","func","idY:","idRef:","idX:"],
            ["one","steering","c","a","b"],["two","oil","c","a","b"]],
            "glowMap":{"arbitrary_a":{"simpleFunction":"abs"},
                       "arbitrary_b":{"advancedFunction":{"triggers":["lowfuel"],"cmd":"lowfuel"}}}}'''}
        context = mesh_context({"one": (.4, 0, 1), "two": (.4, 0, 1)}, parts=parts)
        evidence = vehicle_evidence(context)
        self.assertEqual(evidence.funcs("two"), {"oil"})
        self.assertEqual(evidence.prop_frames["one"], {("a", "b", "c")})
        context.preview_by_id["two"]["materials"] = ("arbitrary_a", "arbitrary_b")
        self.assertTrue(_warning_cluster(context, evidence, "two", 0, 1))
        context.preview_by_id["two"]["materials"] = ("arbitrary_a",)
        self.assertFalse(_warning_cluster(context, evidence, "two", 0, 1))

    def test_applying_trigger_recommendation_writes_and_clears_its_offset(self):
        app = SimpleNamespace(conversion={"triggers": []})
        row = {"kind": "trigger", "object_id": "esc", "mode": "translate",
               "trigger_key": ("esc", (.1, 0, 0)), "offset": .068}
        RecommendationsUIMixin._apply_recommendations(app, [row])
        self.assertEqual(app.conversion["triggers"][0]["offset"], .068)
        row["offset"] = None
        RecommendationsUIMixin._apply_recommendations(app, [row])
        self.assertNotIn("offset", app.conversion["triggers"][0])
        row["offset"] = 0.
        RecommendationsUIMixin._apply_recommendations(app, [row])
        self.assertEqual(app.conversion["triggers"][0]["mode"], "skip")
        app._get_part_setting = lambda host, key, default: .81
        row.update(offset=None, host_id="footrest")
        RecommendationsUIMixin._apply_recommendations(app, [row])
        self.assertEqual(app.conversion["triggers"][0]["offset"], .81)

    def test_distance_measures_filled_face_and_degenerate_edge(self):
        triangle = np.array([[[0., 0., 0.], [10., 0., 0.], [0., 10., 0.]]])
        self.assertAlmostEqual(surface_distance([2, 2, .03], triangle), .03)
        self.assertAlmostEqual(surface_distance([2, -1, 0], triangle), 1)
        self.assertAlmostEqual(surface_distance([1, 1, 0], [[[0, 0, 0], [2, 0, 0], [2, 0, 0]]]), 1)
        self.assertEqual(surface_distance([0, 0, 0], []), float("inf"))

    def test_small_button_wins_over_nearby_large_fascia(self):
        context = mesh_context({"button": (0, 0, .004), "dash": (0, 0, 0)})
        context.preview_by_id["button"]["bounds"] = ((-.01, -.01, .004), (.01, .01, .004))
        context.preview_by_id["dash"]["bounds"] = ((-1, -1, 0), (1, 1, 0))
        geometry = {
            "button": {"triangles": [[[-.01, -.01, .004], [.01, -.01, .004], [0, .01, .004]]]},
            "dash": {"triangles": [[[-1, -1, 0], [1, -1, 0], [0, 1, 0]]]},
        }
        with patch("beamxp.hand_drive_ui.recommendation_controls.geometry_for_ids", return_value=geometry):
            hosts = trigger_hosts(context, ["button", "dash"], [{"at": (0, 0, 0)}])
        self.assertEqual(hosts[0], "button")

    def test_trigger_inherits_move_offset_and_unknown_is_not_skip(self):
        rows = [{"kind": "single", "object_id": "button", "mode": "translate", "offset": .068}]
        boxes = [{"trigger": "esc", "at": (.1, 0, 0), "key": ("esc", (.1, 0, 0))},
                 {"trigger": "hazard", "at": (.1, 0, 0), "key": ("hazard", (.1, 0, 0))}]
        with patch("beamxp.hand_drive_ui.recommendation_engine.trigger_hosts", return_value={0: "button", 1: "unknown"}):
            _add_trigger_rows(rows, boxes, None, None, object(), ["button", "unknown"])
        self.assertEqual(rows[1]["mode"], "translate")
        self.assertEqual(rows[1]["offset"], .068)
        self.assertEqual(rows[2]["mode"], "mirror")

    def test_pedal_channels_have_equal_weight_and_both_hands_work(self):
        centers = {"gas": (.2, 0, 0), "brake": (.4, 0, 0), "shaft": (.4, 0, 0)}
        evidence = VehicleEvidence(prop_funcs={"gas": frozenset({"throttle"}),
                                              "brake": frozenset({"brake"}), "shaft": frozenset({"brake"})})
        context = mesh_context(centers)
        self.assertEqual(set(pedal_offsets(context, evidence, centers, 0, 1).values()), {.6})
        for preview in context.preview_by_id.values():
            preview["center"] = tuple(-x if i == 0 else x for i, x in enumerate(preview["center"]))
        self.assertEqual(set(pedal_offsets(context, evidence, centers, 0, -1).values()), {.6})
        self.assertEqual(pedal_offsets(context, evidence, ["brake"], 0, -1), {})


if __name__ == "__main__":
    unittest.main()
