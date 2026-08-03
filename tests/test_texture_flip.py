from __future__ import annotations

import types
import unittest
from xml.etree import ElementTree as ET

from beamxp import hand_drive_core as core
from beamxp import transform_helpers as th


def build_geometry(uv_values: str) -> ET.Element:
    xml = f"""
    <geometry xmlns="{th.NS['c']}" id="Mesh_077-mesh" name="screen">
      <mesh>
        <source id="Mesh_077-mesh-positions">
          <float_array id="Mesh_077-mesh-positions-array" count="12">
            -0.5 0 0  0.5 0 0  0.5 0 1  -0.5 0 1
          </float_array>
          <technique_common>
            <accessor source="#Mesh_077-mesh-positions-array" count="4" stride="3">
              <param name="X" type="float"/>
              <param name="Y" type="float"/>
              <param name="Z" type="float"/>
            </accessor>
          </technique_common>
        </source>
        <source id="Mesh_077-mesh-map-0">
          <float_array id="Mesh_077-mesh-map-0-array" count="8">{uv_values}</float_array>
          <technique_common>
            <accessor source="#Mesh_077-mesh-map-0-array" count="4" stride="2">
              <param name="S" type="float"/>
              <param name="T" type="float"/>
            </accessor>
          </technique_common>
        </source>
        <vertices id="Mesh_077-mesh-vertices">
          <input semantic="POSITION" source="#Mesh_077-mesh-positions"/>
        </vertices>
        <triangles material="screen-material" count="2">
          <input semantic="VERTEX" source="#Mesh_077-mesh-vertices" offset="0"/>
          <input semantic="TEXCOORD" source="#Mesh_077-mesh-map-0" offset="1" set="0"/>
          <p>0 0 1 1 2 2 0 0 2 2 3 3</p>
        </triangles>
      </mesh>
    </geometry>
    """
    return ET.fromstring(xml)


def uv_pairs(geometry: ET.Element) -> tuple[list[float], list[float]]:
    mesh = geometry.find("c:mesh", th.NS)
    for source in mesh.findall("c:source", th.NS):
        accessor = source.find(".//c:accessor", th.NS)
        params = [p.get("name") for p in accessor.findall("c:param", th.NS)]
        if params[:2] == ["S", "T"]:
            values = [float(v) for v in source.find("c:float_array", th.NS).text.split()]
            return values[0::2], values[1::2]
    raise AssertionError("No TEXCOORD source found")


class TextureFlipTests(unittest.TestCase):
    def test_mirrored_geometry_leaves_uvs_alone_by_default(self) -> None:
        geometry = build_geometry("0.1 0.2  0.9 0.2  0.9 0.8  0.1 0.8")
        out = th.mirrored_geometry(geometry, "new")
        s, t = uv_pairs(out)
        self.assertEqual(s, [0.1, 0.9, 0.9, 0.1])
        self.assertEqual(t, [0.2, 0.2, 0.8, 0.8])

    def test_flip_texture_reflects_s_within_bounds(self) -> None:
        geometry = build_geometry("0.1 0.2  0.9 0.2  0.9 0.8  0.1 0.8")
        out = th.mirrored_geometry(geometry, "new", flip_texture=True)
        s, t = uv_pairs(out)
        for got, expected in zip(s, [0.9, 0.1, 0.1, 0.9]):
            self.assertAlmostEqual(got, expected)
        self.assertEqual(t, [0.2, 0.2, 0.8, 0.8])

    def test_flip_texture_preserves_offset_footprint(self) -> None:
        # UVs that live in a sub-region of an atlas (and outside 0..1, as the
        # stock etk800 screen does) must keep sampling the same region.
        geometry = build_geometry("-0.995 0.0  -0.4276 0.0  -0.4276 1.0  -0.995 1.0")
        out = th.mirrored_geometry(geometry, "new", flip_texture=True)
        s, _t = uv_pairs(out)
        self.assertAlmostEqual(min(s), -0.995)
        self.assertAlmostEqual(max(s), -0.4276)
        for got, expected in zip(s, [-0.4276, -0.995, -0.995, -0.4276]):
            self.assertAlmostEqual(got, expected)

    def test_flip_texture_still_mirrors_positions(self) -> None:
        geometry = build_geometry("0.0 0.0  1.0 0.0  1.0 1.0  0.0 1.0")
        out = th.mirrored_geometry(geometry, "new", flip_texture=True)
        mesh = out.find("c:mesh", th.NS)
        positions = next(
            source
            for source in mesh.findall("c:source", th.NS)
            if "positions" in source.get("id")
        )
        values = [float(v) for v in positions.find("c:float_array", th.NS).text.split()]
        self.assertEqual(values[0::3], [0.5, -0.5, -0.5, 0.5])

    def test_texture_flip_mesh_ids_is_display_screens_in_mirror_modes(self) -> None:
        # Derived from display detection (no manual flag) and gated to modes
        # that reflect geometry. Translate never reflects the geometry.
        context = types.SimpleNamespace(
            _display_texture_flip_scope={
                "screen": frozenset({"screen"}),
                "gauge": frozenset({"gauge_screen"}),
                "panel": frozenset({"panel_screen"}),
            }
        )
        modes = {
            "screen": core.MODE_MIRROR,     # nav + mirror -> flipped
            "dash": core.MODE_MIRROR,       # mirror but not a nav screen
            "gauge": core.MODE_TRANSLATE,   # nav but translated, not reflected
            "panel": core.MODE_SKIP,        # display but nothing reflects it
        }
        self.assertEqual(core.texture_flip_mesh_ids(context, modes), {"screen"})


class DisplayScreenDetectionTests(unittest.TestCase):
    """The beamNavigator screen material resolves to what the DAE binds, both
    directly (etk800) and through the same part's glowMap (sunburst2)."""

    def _context(self, part_bodies: dict[str, str], symbols: dict[str, tuple[str, ...]]):
        context = types.SimpleNamespace(
            part_body_index={name: (body, f"{name}.jbeam") for name, body in part_bodies.items()},
            preview_by_id={
                object_id: {"materials": mats} for object_id, mats in symbols.items()
            },
        )
        return context

    def test_direct_screen_material_match(self) -> None:
        # etk800: screenMaterialName IS the bound material.
        body = """
        "etk800_dash": { "slotType": "etk800_dash",
          "controller": [ ["fileName"],
            ["beamNavigator", {"screenMaterialName": "@etk800_screen", "name": "etk800_navi"}] ]
        }
        """
        context = self._context({"etk800_dash": body}, {"etk800_screen": ("etk800_screen",)})
        self.assertEqual(core.nav_screen_materials_for_context(context), frozenset({"etk800_screen"}))
        self.assertEqual(
            core.nav_screen_mesh_scope(context),
            {"etk800_screen": frozenset({"etk800_screen"})},
        )

    def test_glowmap_resolves_runtime_base_material(self) -> None:
        # sunburst2: the mesh binds sunburst2_display_nav; the glowMap swaps it
        # to the screenMaterialName when lit, so the base must be resolved back.
        body = """
        "sunburst2_nav": { "slotType": "sunburst2_radio",
          "controller": [ ["fileName"],
            ["beamNavigator", {"screenMaterialName": "@sunburst2_naviscreen_on", "name": "sunburst2_navi"}] ],
          "glowMap": {
            "sunburst2_display_nav": {"simpleFunction": {"ignitionLevel": 0.5},
               "off": "screen_off", "on": "sunburst2_naviscreen_accessory",
               "on_intense": "sunburst2_naviscreen_on"}
          }
        }
        """
        context = self._context(
            {"sunburst2_nav": body},
            {"sunburst2_nav": ("sunburst2_gauges", "sunburst2_display_nav")},
        )
        materials = core.nav_screen_materials_for_context(context)
        self.assertIn("sunburst2_display_nav", materials)
        # Only the screen island is scoped; the gauge cluster material is not.
        self.assertEqual(
            core.nav_screen_mesh_scope(context),
            {"sunburst2_nav": frozenset({"sunburst2_display_nav"})},
        )

    def test_grouped_nav_and_gauge_screen_materials_share_flip_scope(self) -> None:
        body = """
        "ardente_dash": { "slotType": "ardente_dash",
          "controller": [ ["fileName"],
            ["beamNavigator", {"screenMaterialName": "@ardente_gps_screen", "name": "ardente_navi"}] ],
          "glowMap": {
            "ardente_gauges_screen": {"simpleFunction": {"ignitionLevel": 0.5},
              "off": "screen_off", "on": "ardente_gauges_screen_accessory",
              "on_intense": "ardente_gauges_screen_accessory"},
            "ardente_gps_screen": {"simpleFunction": {"ignitionLevel": 0.5},
              "off": "screen_off", "on": "ardente_naviscreen_accessory",
              "on_intense": "ardente_naviscreen_accessory"}
          }
        }
        """
        context = self._context(
            {"ardente_dash": body},
            {"ardente_screens": ("ardente_gauges_screen", "ardente_gps_screen")},
        )
        context._material_flags = {
            "ardente_gauges_screen": {"emissive": True, "glass": False},
            "ardente_gps_screen": {"emissive": True, "glass": False},
        }
        self.assertEqual(
            core.display_texture_flip_scope(context),
            {"ardente_screens": frozenset({"ardente_gauges_screen", "ardente_gps_screen"})},
        )

    def test_no_navigator_yields_empty(self) -> None:
        body = '"plain_dash": { "slotType": "plain_dash", "flexbodies": [["mesh"]] }'
        context = self._context({"plain_dash": body}, {"plain_dash": ("dash_mat",)})
        context._material_flags = {}
        self.assertEqual(core.nav_screen_materials_for_context(context), frozenset())
        self.assertEqual(core.nav_screen_mesh_scope(context), {})
        self.assertEqual(core.display_texture_flip_scope(context), {})


class ScopedTextureFlipTests(unittest.TestCase):
    """A nav screen sharing a mesh (and one texcoord source) with a cluster,
    like sunburst2_nav: only the screen's UV island may be reflected."""

    def build_shared_geometry(self) -> ET.Element:
        # Two islands in one map source: cluster in U[0,1), nav screen in U[1,2).
        xml = f"""
        <geometry xmlns="{th.NS['c']}" id="nav-mesh" name="nav">
          <mesh>
            <source id="nav-mesh-positions">
              <float_array id="nav-mesh-positions-array" count="24">
                -0.5 0 0  0.5 0 0  0.5 0 1  -0.5 0 1
                -0.5 0 0  0.5 0 0  0.5 0 1  -0.5 0 1
              </float_array>
              <technique_common>
                <accessor source="#nav-mesh-positions-array" count="8" stride="3">
                  <param name="X" type="float"/>
                  <param name="Y" type="float"/>
                  <param name="Z" type="float"/>
                </accessor>
              </technique_common>
            </source>
            <source id="nav-mesh-map-0">
              <float_array id="nav-mesh-map-0-array" count="16">
                0.1 0.2  0.9 0.2  0.9 0.8  0.1 0.8
                1.1 0.2  1.9 0.2  1.9 0.8  1.1 0.8
              </float_array>
              <technique_common>
                <accessor source="#nav-mesh-map-0-array" count="8" stride="2">
                  <param name="S" type="float"/>
                  <param name="T" type="float"/>
                </accessor>
              </technique_common>
            </source>
            <vertices id="nav-mesh-vertices">
              <input semantic="POSITION" source="#nav-mesh-positions"/>
            </vertices>
            <triangles material="cluster-material" count="2">
              <input semantic="VERTEX" source="#nav-mesh-vertices" offset="0"/>
              <input semantic="TEXCOORD" source="#nav-mesh-map-0" offset="1" set="0"/>
              <p>0 0 1 1 2 2  0 0 2 2 3 3</p>
            </triangles>
            <triangles material="display_nav-material" count="2">
              <input semantic="VERTEX" source="#nav-mesh-vertices" offset="0"/>
              <input semantic="TEXCOORD" source="#nav-mesh-map-0" offset="1" set="0"/>
              <p>4 4 5 5 6 6  4 4 6 6 7 7</p>
            </triangles>
          </mesh>
        </geometry>
        """
        return ET.fromstring(xml)

    def test_scoped_flip_reflects_only_nav_island(self) -> None:
        geometry = self.build_shared_geometry()
        out = th.mirrored_geometry(
            geometry, "new", flip_texture=True, flip_materials={"display_nav"}
        )
        s, t = uv_pairs(out)
        # Cluster island (indices 0-3) is untouched.
        self.assertEqual(s[0:4], [0.1, 0.9, 0.9, 0.1])
        # Nav island (indices 4-7) reflects within its own tile: u' = 3.0 - u.
        for got, expected in zip(s[4:8], [1.9, 1.1, 1.1, 1.9]):
            self.assertAlmostEqual(got, expected)
        # T (vertical) is never touched.
        self.assertEqual(t, [0.2, 0.2, 0.8, 0.8, 0.2, 0.2, 0.8, 0.8])

    def test_scoped_flip_keeps_nav_island_in_its_own_tile(self) -> None:
        geometry = self.build_shared_geometry()
        out = th.mirrored_geometry(
            geometry, "new", flip_texture=True, flip_materials={"display_nav"}
        )
        s, _t = uv_pairs(out)
        # The reflection stays within U[1,2); it must not land on the cluster.
        self.assertGreaterEqual(min(s[4:8]), 1.0)
        self.assertLessEqual(max(s[4:8]), 2.0)


if __name__ == "__main__":
    unittest.main()
