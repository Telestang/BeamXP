from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET

import numpy as np

import beamng_hand_drive_core as core
import beamng_transform_helpers as transform_helpers


class DaeSurfaceTriangleTests(unittest.TestCase):
    def test_extracts_triangles_and_triangulates_polylist(self) -> None:
        data = b'''<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema">
          <library_geometries><geometry id="g"><mesh>
            <source id="positions">
              <float_array id="positions-array" count="12">0 0 0  1 0 0  1 1 0  0 1 0</float_array>
              <technique_common><accessor source="#positions-array" count="4" stride="3">
                <param name="X"/><param name="Y"/><param name="Z"/>
              </accessor></technique_common>
            </source>
            <vertices id="vertices"><input semantic="POSITION" source="#positions"/></vertices>
            <triangles count="1"><input semantic="VERTEX" source="#vertices" offset="0"/><p>0 1 2</p></triangles>
            <polylist count="1"><input semantic="VERTEX" source="#vertices" offset="0"/><vcount>4</vcount><p>0 1 2 3</p></polylist>
          </mesh></geometry></library_geometries>
          <library_visual_scenes><visual_scene><node id="panel">
            <matrix>1 0 0 4  0 1 0 5  0 0 1 6  0 0 0 1</matrix>
            <instance_geometry url="#g"/>
          </node></visual_scene></library_visual_scenes>
        </COLLADA>'''
        tree = ET.ElementTree(ET.fromstring(data))
        geometry = tree.getroot().find("c:library_geometries/c:geometry", transform_helpers.NS)

        local = transform_helpers.geometry_surface_triangles(geometry)
        self.assertEqual(len(local), 3)
        surfaces = core.surface_triangles_from_tree(tree)
        self.assertEqual(surfaces["panel"].shape, (3, 3, 3))
        np.testing.assert_allclose(surfaces["panel"][0, 0], (4.0, 5.0, 6.0))


class DaeAliasCandidateTests(unittest.TestCase):
    """dae_alias_candidates replaces a combined mesh-name regex as the
    prefilter for common DAEs. It must never be narrower than the aliases
    dae_objects_from_tree can key an object by, or meshes go missing."""

    def test_collects_both_id_and_name_attributes(self) -> None:
        data = b'<node id="acme_hood" name="Acme Hood"><matrix/></node>'
        self.assertEqual(
            core.dae_alias_candidates(data),
            {"acme_hood", "Acme Hood"},
        )

    def test_includes_stripped_forms_like_dae_node_aliases(self) -> None:
        # dae_node_aliases yields the raw and stripped value, so a node whose
        # id carries padding is reachable under the trimmed mesh name.
        data = b'<node id="  acme_door  " name=""/>'
        candidates = core.dae_alias_candidates(data)
        self.assertIn("  acme_door  ", candidates)
        self.assertIn("acme_door", candidates)

    def test_matches_the_aliases_the_object_parser_produces(self) -> None:
        import xml.etree.ElementTree as ET

        data = (
            b'<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema">'
            b'<library_visual_scenes><visual_scene>'
            b'<node id="acme_wheel" name=" acme_wheel_display ">'
            b'<matrix>1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</matrix>'
            b'<instance_geometry url="#g1"/>'
            b"</node></visual_scene></library_visual_scenes></COLLADA>"
        )
        objects = core.dae_objects_from_tree(ET.ElementTree(ET.fromstring(data)), "a.dae")
        candidates = core.dae_alias_candidates(data)
        for alias in objects:
            self.assertIn(alias, candidates, f"prefilter would skip {alias!r}")

    def test_ignores_undecodable_bytes_without_raising(self) -> None:
        self.assertEqual(core.dae_alias_candidates(b'<node id="\xff\xfe"/>'), set())


class ReachableCommonOrderTests(unittest.TestCase):
    def test_reachable_common_index_order_is_deterministic(self) -> None:
        # slot_demand_types returns a set; iterating it unsorted made the
        # insertion order (and therefore preview sampling) vary per process.
        vehicle = {
            "acme": (
                '"acme": {"slotType":"main","slots":[\n'
                '["type","default","description"],\n'
                '["zeta_slot","zeta_part",""],\n'
                '["alpha_slot","alpha_part",""],\n'
                '["mid_slot","mid_part",""],\n'
                "]}",
                "acme.jbeam",
            )
        }
        common = {
            name: (f'"{name}": {{"slotType":"{slot}"}}', f"{name}.jbeam")
            for name, slot in (
                ("zeta_part", "zeta_slot"),
                ("alpha_part", "alpha_slot"),
                ("mid_part", "mid_slot"),
            )
        }
        order = list(core.reachable_common_part_index(vehicle, common))
        self.assertEqual(order, ["alpha_part", "mid_part", "zeta_part"])
        for _ in range(5):
            self.assertEqual(list(core.reachable_common_part_index(vehicle, common)), order)


class MaskCommentsCacheTests(unittest.TestCase):
    def test_cached_mask_still_matches_a_fresh_computation(self) -> None:
        import beamng_transform_helpers as th

        text = '{"a": 1, // note "b"\n "c": [1, 2], /* "d" */ "e": "f"}'
        first = th.mask_comments_preserve_offsets(text)
        second = th.mask_comments_preserve_offsets(text)
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(text))
        th.mask_comments_preserve_offsets.cache_clear()
        self.assertEqual(th.mask_comments_preserve_offsets(text), first)


if __name__ == "__main__":
    unittest.main()
