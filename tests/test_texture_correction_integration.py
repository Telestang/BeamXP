from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from xml.etree import ElementTree as ET

from beamxp import hand_drive_core as core
from beamxp.core.dae import DaeObject
from beamxp.core.models import VehicleContext
from beamxp.hand_drive_parts import build_pipeline


def _screen_quad_dae(u_low: float, u_high: float) -> str:
    """A one-quad DAE whose screen material reads u_low..u_high of its page."""
    return f"""<?xml version="1.0"?>
    <COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema">
      <library_geometries>
        <geometry id="cluster">
          <mesh>
            <source id="cluster-map">
              <float_array id="cluster-map-array" count="8">
                {u_low} 0.0  {u_high} 0.0  {u_high} 1.0  {u_low} 1.0
              </float_array>
              <technique_common>
                <accessor source="#cluster-map-array" count="4" stride="2">
                  <param name="S" type="float"/><param name="T" type="float"/>
                </accessor>
              </technique_common>
            </source>
            <triangles material="lc500_screen_off-material" count="2">
              <input semantic="VERTEX" source="#cluster-vertices" offset="0"/>
              <input semantic="TEXCOORD" source="#cluster-map" offset="1" set="0"/>
              <p>0 0 1 1 2 2 0 0 2 2 3 3</p>
            </triangles>
          </mesh>
        </geometry>
      </library_geometries>
    </COLLADA>
    """


def _combined_screen_dae(
    first_material: str,
    second_material: str,
) -> str:
    """A generated-mesh fixture with two independently scoped UV islands."""
    return f"""<?xml version="1.0"?>
    <COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema">
      <asset><unit meter="1"/></asset>
      <library_geometries>
        <geometry id="combined_screen" name="combined_screen">
          <mesh>
            <source id="combined_screen-positions">
              <float_array id="combined_screen-positions-array" count="24">
                -1 0 0  0 0 0  0 0 1  -1 0 1
                 0 0 0  1 0 0  1 0 1   0 0 1
              </float_array>
              <technique_common>
                <accessor source="#combined_screen-positions-array" count="8" stride="3">
                  <param name="X" type="float"/><param name="Y" type="float"/>
                  <param name="Z" type="float"/>
                </accessor>
              </technique_common>
            </source>
            <source id="combined_screen-map">
              <float_array id="combined_screen-map-array" count="16">
                0.1 0.2  0.9 0.2  0.9 0.8  0.1 0.8
                1.1 0.2  1.9 0.2  1.9 0.8  1.1 0.8
              </float_array>
              <technique_common>
                <accessor source="#combined_screen-map-array" count="8" stride="2">
                  <param name="S" type="float"/><param name="T" type="float"/>
                </accessor>
              </technique_common>
            </source>
            <vertices id="combined_screen-vertices">
              <input semantic="POSITION" source="#combined_screen-positions"/>
            </vertices>
            <triangles material="{first_material}-material" count="2">
              <input semantic="VERTEX" source="#combined_screen-vertices" offset="0"/>
              <input semantic="TEXCOORD" source="#combined_screen-map" offset="1" set="0"/>
              <p>0 0 1 1 2 2  0 0 2 2 3 3</p>
            </triangles>
            <triangles material="{second_material}-material" count="2">
              <input semantic="VERTEX" source="#combined_screen-vertices" offset="0"/>
              <input semantic="TEXCOORD" source="#combined_screen-map" offset="1" set="0"/>
              <p>4 4 5 5 6 6  4 4 6 6 7 7</p>
            </triangles>
          </mesh>
        </geometry>
      </library_geometries>
      <library_visual_scenes>
        <visual_scene id="Scene">
          <node id="lc500_interior" name="lc500_interior">
            <matrix>1 0 0 0  0 1 0 0  0 0 1 0  0 0 0 1</matrix>
            <instance_geometry url="#combined_screen"/>
          </node>
        </visual_scene>
      </library_visual_scenes>
      <scene><instance_visual_scene url="#Scene"/></scene>
    </COLLADA>
    """


def _combined_screen_s_values(path: Path) -> list[float]:
    root = ET.parse(path).getroot()
    source = root.find(
        ".//c:source[@id='combined_screen-map']/c:float_array",
        core.NS,
    )
    if source is None or source.text is None:
        raise AssertionError("generated DAE lost the combined screen UV source")
    return [float(value) for value in source.text.split()][0::2]


def minimal_context(tmp: Path) -> VehicleContext:
    source_zip = tmp / "vehicle.zip"
    source_zip.write_bytes(b"placeholder")
    return VehicleContext(
        source_zip=source_zip,
        vehicle_id="scintilla",
        vehicle_path="vehicles/scintilla",
        dae_paths=["vehicles/scintilla/scintilla.dae"],
        variants={},
        objects={
            "scintilla_dashboard": DaeObject(
                id="scintilla_dashboard",
                name="scintilla_dashboard",
                dae_path="vehicles/scintilla/scintilla.dae",
                x=0.0,
                y=0.0,
                z=0.0,
                geometry_ids=("dash_geom",),
            ),
            "scintilla_dash_controls": DaeObject(
                id="scintilla_dash_controls",
                name="scintilla_dash_controls",
                dae_path="vehicles/scintilla/scintilla.dae",
                x=0.0,
                y=0.0,
                z=0.0,
                geometry_ids=("controls_geom",),
            ),
        },
        preview_by_id={},
        jbeam_texts={},
        node_positions={},
        project_dir=tmp,
    )


def _placeholder_dds(path: Path) -> Path:
    """Write a corrected-texture stand-in whose bytes are its own name.

    Distinct content matters here: staging folds byte-identical maps of one
    stage into a single file, so placeholders that were all ``b"dds"`` could
    no longer show which source a stage ended up wired to -- two opacity masks
    became indistinguishable, and the assertion that each stage keeps its own
    map could pass or fail on nothing.
    """
    path.write_bytes(b"dds:" + path.name.encode())
    return path


class TextureCorrectionIntegrationTests(unittest.TestCase):
    def test_texture_asset_search_excludes_its_installed_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            context = minimal_context(tmp)
            generated = tmp / build_pipeline.package_name_for_context(context)
            generated.write_bytes(b"generated")
            sibling = tmp / "shared_assets.zip"
            sibling.write_bytes(b"assets")

            with patch.object(build_pipeline, "beamng_game_common_zips", return_value=[]):
                archives = build_pipeline.texture_correction_asset_archives(context)

        self.assertIn(context.source_zip, archives)
        self.assertIn(sibling, archives)
        self.assertNotIn(generated, archives)

    def test_switched_navigator_gets_conversion_owned_runtime_resource(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            source = tmp / "lexlc500.zip"
            source_jbeam = r'''
            {
              "lc500": {
                "slotType":"main",
                "controller":[["fileName"], ["beamNavigator", {
                  "screenMaterialName":"@lc500_GPS",
                  "htmlFilePath":"local://local/vehicles/lc500/lc500_GPS.html",
                  "name":"lc500_GPS"
                }]],
                "glowMap":{
                  "lc500_centralscreen":{"simpleFunction":{"ignitionLevel":0.5},
                    "off":"lc500_screens_off", "on":"lc500_centralscreen_on",
                    "on_intense":"lc500_GPS"}
                }
              }
            }
            '''
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("vehicles/lc500/lc500.jbeam", source_jbeam)
                archive.writestr(
                    "vehicles/lc500/main.materials.json",
                    json.dumps(
                        {
                            "lc500_GPS": {
                                "name": "lc500_GPS",
                                "mapTo": "lc500_GPS",
                                "class": "Material",
                                "Stages": [
                                    {"colorMap": "@lc500_GPS", "emissive": True}
                                ],
                            }
                        }
                    ),
                )
            output = tmp / "out" / "vehicles" / "lc500"
            output.mkdir(parents=True)
            target_mesh = core.generated_mesh_name("lc500_screen", core.HAND_RHD)
            other_mesh = core.generated_mesh_name("lc500_body", core.HAND_RHD)
            dash_mesh = core.generated_mesh_name("lc500_dash", core.HAND_RHD)
            generated_jbeam = output / "handdrive_visual_conversion.jbeam"
            generated_jbeam.write_text(
                '{"lc500":{"slotType":"main","flexbodies":[["mesh","[group]:"],["'
                + dash_mesh
                + '",["lc500_body"]]]},'
                '"lc500_body_xp_rhd":{"slotType":"lc500_body",'
                '"flexbodies":[["mesh","[group]:"],["'
                + other_mesh
                + '",["lc500_body"]]]},'
                '"lc500_interior_xp_rhd":{"slotType":"lc500_interior",'
                '"flexbodies":[["mesh","[group]:"],["'
                + target_mesh
                + '",["lc500_body"]]],'
                # Texture correction runs first and has already claimed the
                # "off" state; isolating the live state must not undo that.
                '"glowMap":{"lc500_centralscreen":{"simpleFunction":{"ignitionLevel":0.5},'
                '"off":"lc500_screens_off_beamxp_tc", "on":"lc500_centralscreen_on",'
                '"on_intense":"lc500_GPS"}}}}',
                encoding="utf-8",
            )
            context = VehicleContext(
                source_zip=source,
                vehicle_id="lc500",
                vehicle_path="vehicles/lc500",
                dae_paths=["vehicles/lc500/lc500.dae"],
                variants={},
                objects={},
                preview_by_id={
                    "lc500_screen": {"materials": ["lc500_centralscreen"]}
                },
                jbeam_texts={"vehicles/lc500/lc500.jbeam": source_jbeam},
                node_positions={},
                project_dir=tmp,
                part_body_index={
                    "lc500": (source_jbeam, "vehicles/lc500/lc500.jbeam")
                },
            )

            report = build_pipeline.isolate_converted_runtime_screens(
                context,
                output,
                {"lc500_screen"},
                {core.HAND_RHD},
            )
            patched = generated_jbeam.read_text(encoding="utf-8")
            materials = json.loads(
                (output / "beamxp_runtime_screens.materials.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertTrue(report["enabled"])
        target_alias = report["materials"][0]
        self.assertIn('"screenMaterialName":"@' + target_alias + '"', patched)
        self.assertEqual(patched.count('"screenMaterialName":"@' + target_alias + '"'), 1)
        self.assertIn('"on_intense":"' + target_alias + '"', patched)
        self.assertEqual(materials[target_alias]["Stages"][0]["colorMap"], "@" + target_alias)
        self.assertIn('"off":"lc500_screens_off_beamxp_tc"', patched)
        self.assertNotIn('"on_intense":"lc500_GPS"', patched)
        # Mesh and part identities are independent: the screen mesh is carried
        # by the generated interior part, not by the part the navigator
        # controller was authored on, and that part owns the rebind.
        self.assertIn(target_mesh, patched)
        self.assertEqual(patched.count(target_mesh), 1)  # its one carrying row

    def test_direct_navigator_binding_survives_stale_child_and_clones_dim_state(self) -> None:
        """A child glowMap cannot hand a private screen back to the donor tag."""

        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            source = tmp / "vivace.zip"
            source_jbeam = r'''
            {
              "ardente_dash": {
                "slotType":"ardente_dash",
                "controller":[["fileName"], ["beamNavigator", {
                  "screenMaterialName":"@ardente_gps_screen",
                  "htmlFilePath":"local://local/vehicles/vivace/ardente/nav.html",
                  "name":"ardente_navi"
                }]],
                "glowMap":{
                  "ardente_gps_screen":{"simpleFunction":{"ignitionLevel":0.1},
                    "off":"screen_off", "on":"ardente_gps_screen",
                    "on_intense":"ardente_gps_screen_dim"}
                }
              },
              "ardente_screen_branding": {
                "slotType":"ardente_screen_branding",
                "glowMap":{
                  "ardente_gps_screen":{"simpleFunction":{"ignitionLevel":0.1},
                    "off":"screen_off", "on":"ardente_gps_screen",
                    "on_intense":"ardente_gps_screen_dim"}
                }
              }
            }
            '''
            source_materials = {
                "ardente_gps_screen": {
                    "name": "ardente_gps_screen",
                    "mapTo": "ardente_gps_screen",
                    "class": "Material",
                    "Stages": [{"emissiveMap": "@ardente_gps_screen"}],
                },
                "ardente_gps_screen_dim": {
                    "name": "ardente_gps_screen_dim",
                    "mapTo": "ardente_gps_screen_dim",
                    "class": "Material",
                    "Stages": [{"emissiveMap": "@ardente_gps_screen"}],
                },
                "screen_off": {
                    "name": "screen_off",
                    "mapTo": "screen_off",
                    "class": "Material",
                    "Stages": [{"baseColorMap": "screen_off.dds"}],
                },
            }
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("vehicles/vivace/ardente/interior.jbeam", source_jbeam)
                archive.writestr(
                    "vehicles/vivace/ardente/screens.materials.json",
                    json.dumps(source_materials),
                )

            output = tmp / "out" / "vehicles" / "vivace"
            output.mkdir(parents=True)
            source_mesh = "ardente_screens"
            generated_mesh = core.generated_mesh_name(source_mesh, core.HAND_RHD)
            generated_jbeam = output / "handdrive_visual_conversion.jbeam"
            generated_jbeam.write_text(
                f'''{{
                  "ardente_dash_xp_rhd": {{
                    "slotType":"ardente_dash",
                    "flexbodies":[["mesh","[group]:"],
                      ["{generated_mesh}",["ardente_dash"]]],
                    "controller":[["fileName"],["beamNavigator",{{
                      "screenMaterialName":"@ardente_gps_screen",
                      "htmlFilePath":"local://local/vehicles/vivace/ardente/nav.html",
                      "name":"ardente_navi"}}]],
                    "glowMap":{{"ardente_gps_screen":{{
                      "simpleFunction":{{"ignitionLevel":0.1}},
                      "off":"screen_off", "on":"ardente_gps_screen",
                      "on_intense":"ardente_gps_screen_dim"}}}}
                  }},
                  "ardente_screen_branding": {{
                    "slotType":"ardente_screen_branding",
                    "glowMap":{{"ardente_gps_screen":{{
                      "off":"screen_off", "on":"ardente_gps_screen",
                      "on_intense":"ardente_gps_screen_dim"}}}}
                  }}
                }}''',
                encoding="utf-8",
            )
            generated_dae = output / "ardente_handdrive.dae"
            generated_dae.write_text(
                f'''<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema">
                  <library_effects><effect id="ardente-screen-effect"/></library_effects>
                  <library_materials>
                    <material id="ardente_gps_screen-material" name="ardente_gps_screen">
                      <instance_effect url="#ardente-screen-effect"/>
                    </material>
                  </library_materials>
                  <library_geometries><geometry id="shared-screen"><mesh>
                    <triangles material="ardente_gps_screen-material" count="0"/>
                  </mesh></geometry></library_geometries>
                  <library_visual_scenes><visual_scene id="Scene">
                    <node id="{generated_mesh}" name="{generated_mesh}">
                      <instance_geometry url="#shared-screen"><bind_material>
                        <technique_common><instance_material
                          symbol="ardente_gps_screen-material"
                          target="#ardente_gps_screen-material"/>
                        </technique_common></bind_material>
                      </instance_geometry>
                    </node>
                    <node id="stock_screen" name="stock_screen">
                      <instance_geometry url="#shared-screen"><bind_material>
                        <technique_common><instance_material
                          symbol="ardente_gps_screen-material"
                          target="#ardente_gps_screen-material"/>
                        </technique_common></bind_material>
                      </instance_geometry>
                    </node>
                  </visual_scene></library_visual_scenes>
                </COLLADA>''',
                encoding="utf-8",
            )
            context = VehicleContext(
                source_zip=source,
                vehicle_id="vivace",
                vehicle_path="vehicles/vivace",
                dae_paths=[],
                variants={},
                objects={},
                preview_by_id={source_mesh: {"materials": ["ardente_gps_screen"]}},
                jbeam_texts={"vehicles/vivace/ardente/interior.jbeam": source_jbeam},
                node_positions={},
                project_dir=tmp,
                part_body_index={
                    "ardente_dash": (source_jbeam, "vehicles/vivace/ardente/interior.jbeam")
                },
            )

            report = build_pipeline.isolate_converted_runtime_screens(
                context,
                output,
                {source_mesh},
                {core.HAND_RHD},
            )
            patched = generated_jbeam.read_text(encoding="utf-8")
            materials = json.loads(
                (output / "beamxp_runtime_screens.materials.json").read_text(
                    encoding="utf-8"
                )
            )
            suffix = build_pipeline.mod_id_for_context(context).lower()
            target_alias = f"ardente_gps_screen_beamxp_{suffix}"
            target_dim = f"ardente_gps_screen_dim_beamxp_{suffix}"
            symbols = build_pipeline._node_material_symbols(
                generated_dae, {generated_mesh, "stock_screen"}
            )

        self.assertTrue(report["enabled"])
        self.assertEqual(symbols[generated_mesh], {target_alias})
        # The shared geometry was copied before retargeting; the stock consumer
        # remains on its authored material.
        self.assertEqual(symbols["stock_screen"], {"ardente_gps_screen"})
        self.assertEqual(
            materials[target_alias]["Stages"][0]["emissiveMap"], "@" + target_alias
        )
        self.assertEqual(
            materials[target_dim]["Stages"][0]["emissiveMap"], "@" + target_alias
        )
        self.assertIn('"' + target_alias + '":{', patched)
        self.assertIn('"on":"' + target_alias + '"', patched)
        self.assertIn('"on_intense":"' + target_dim + '"', patched)
        self.assertEqual(len(report["colladaRetargets"]), 1)
        # The ungenerated branding part deliberately remains stale.  It can no
        # longer affect the converted mesh because that mesh binds target_alias.
        branding = build_pipeline.transform_helpers.extract_keyed_object(
            patched, "ardente_screen_branding"
        )
        self.assertIsNotNone(branding)
        self.assertIn('"on_intense":"ardente_gps_screen_dim"', branding)

    def test_switched_navigator_rebinds_reachable_texture_correction_fork(self) -> None:
        """A split screen's private glow handle must own the live state too."""

        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            source = tmp / "vehicle.zip"
            source_jbeam = r'''
            {
              "car_interior": {
                "slotType":"car_interior",
                "controller":[["fileName"], ["beamNavigator", {
                  "screenMaterialName":"@car_nav_screen",
                  "htmlFilePath":"local://local/vehicles/car/nav.html",
                  "name":"car_nav"
                }]],
                "glowMap":{
                  "car_nav_screen":{"simpleFunction":{"ignitionLevel":0.5},
                    "off":"car_black", "on":"car_idle_screen",
                    "on_intense":"car_nav_screen"}
                }
              }
            }
            '''
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("vehicles/car/interior.jbeam", source_jbeam)
                archive.writestr(
                    "vehicles/car/screens.materials.json",
                    json.dumps(
                        {
                            "car_nav_screen": {
                                "name": "car_nav_screen",
                                "mapTo": "car_nav_screen",
                                "Stages": [{"colorMap": "@car_nav_screen"}],
                            }
                        }
                    ),
                )

            output = tmp / "out" / "vehicles" / "car"
            output.mkdir(parents=True)
            source_mesh = "car_screen"
            generated_mesh = core.generated_mesh_name(source_mesh, core.HAND_RHD)
            fork_material = "car_nav_screen_beamxp_tc"
            generated_jbeam = output / "handdrive_visual_conversion.jbeam"
            generated_jbeam.write_text(
                f'''{{
                  "car_body_xp_rhd": {{
                    "slotType":"car_body",
                    "flexbodies":[["mesh","[group]:"],
                      ["car_body_xp_rhd",["car_body"]]],
                    "glowMap":{{"{fork_material}":{{
                      "simpleFunction":{{"ignitionLevel":0.5}},
                      "off":"car_body_black_beamxp_tc",
                      "on":"car_body_idle_beamxp_tc",
                      "on_intense":"{fork_material}"}}}}
                  }},
                  "car_interior_xp_rhd": {{
                    "slotType":"car_interior",
                    "flexbodies":[["mesh","[group]:"],
                      ["{generated_mesh}",["car_body"]]],
                    "controller":[["fileName"],["beamNavigator",{{
                      "screenMaterialName":"@car_nav_screen",
                      "htmlFilePath":"local://local/vehicles/car/nav.html",
                      "name":"car_nav"}}]],
                    "glowMap":{{"{fork_material}":{{
                      "simpleFunction":{{"ignitionLevel":0.5}},
                      "off":"car_black",
                      "on":"car_idle_screen_beamxp_tc",
                      "on_intense":"{fork_material}"}}}}
                  }},
                  "car_door_xp_rhd": {{
                    "slotType":"car_door",
                    "flexbodies":[["mesh","[group]:"],
                      ["car_door_xp_rhd",["car_body"]]],
                    "glowMap":{{"{fork_material}":{{
                      "simpleFunction":{{"ignitionLevel":0.5}},
                      "off":"car_door_black_beamxp_tc",
                      "on":"car_door_idle_beamxp_tc",
                      "on_intense":"{fork_material}"}}}}
                  }}
                }}''',
                encoding="utf-8",
            )
            generated_dae = output / "car_handdrive.dae"
            generated_dae.write_text(
                f'''<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema">
                  <library_geometries>
                    <geometry id="screen-geometry"><mesh>
                      <triangles material="{fork_material}-material" count="0"/>
                    </mesh></geometry>
                  </library_geometries>
                  <library_visual_scenes><visual_scene id="Scene">
                    <node id="{generated_mesh}" name="{generated_mesh}">
                      <instance_geometry url="#screen-geometry"/>
                    </node>
                  </visual_scene></library_visual_scenes>
                </COLLADA>''',
                encoding="utf-8",
            )
            context = VehicleContext(
                source_zip=source,
                vehicle_id="car",
                vehicle_path="vehicles/car",
                dae_paths=[],
                variants={},
                objects={},
                preview_by_id={source_mesh: {"materials": ["car_nav_screen"]}},
                jbeam_texts={"vehicles/car/interior.jbeam": source_jbeam},
                node_positions={},
                project_dir=tmp,
                part_body_index={
                    "car_interior": (source_jbeam, "vehicles/car/interior.jbeam")
                },
            )

            report = build_pipeline.isolate_converted_runtime_screens(
                context,
                output,
                {source_mesh},
                {core.HAND_RHD},
                generated_switch_forks=[
                    {
                        "alias": "car_nav_screen",
                        "material": fork_material,
                        "partKeys": [source_mesh],
                        "states": {
                            "car_idle_screen": "car_idle_screen_beamxp_tc"
                        },
                    }
                ],
            )
            patched = generated_jbeam.read_text(encoding="utf-8")
            part_bodies = {
                part_id: build_pipeline.transform_helpers.extract_keyed_object(
                    patched, part_id
                )
                for part_id in (
                    "car_body_xp_rhd",
                    "car_interior_xp_rhd",
                    "car_door_xp_rhd",
                )
            }
            glow_entries_by_part = {}
            for part_id, part_body in part_bodies.items():
                self.assertIsNotNone(part_body)
                glow = build_pipeline.transform_helpers.extract_keyed_object(
                    part_body, "glowMap"
                )
                self.assertIsNotNone(glow)
                glow_entries_by_part[part_id] = {
                    key: value
                    for key, _start, _end, value in (
                        build_pipeline._top_level_jbeam_object_entries(glow)
                    )
                }
            bound_symbols = build_pipeline._node_material_symbols(
                generated_dae, {generated_mesh}
            )[generated_mesh]

        target_alias = report["materials"][0]
        self.assertEqual(bound_symbols, {fork_material})
        expected_static_states = {
            "car_body_xp_rhd": (
                "car_body_black_beamxp_tc",
                "car_body_idle_beamxp_tc",
            ),
            "car_interior_xp_rhd": ("car_black", "car_idle_screen_beamxp_tc"),
            "car_door_xp_rhd": (
                "car_door_black_beamxp_tc",
                "car_door_idle_beamxp_tc",
            ),
        }
        for part_id, (off_material, on_material) in expected_static_states.items():
            glow_entries = glow_entries_by_part[part_id]
            self.assertEqual(set(glow_entries), {fork_material})
            reachable = glow_entries[fork_material]
            self.assertIn('"off":"' + off_material + '"', reachable)
            self.assertIn('"on":"' + on_material + '"', reachable)
            self.assertIn('"on_intense":"' + target_alias + '"', reachable)

        # JBeam merges every selected part's glowMap into one table, so every
        # duplicate fork row must agree.  Controller ownership is narrower:
        # only the part carrying the converted screen mesh gets the navigator.
        self.assertIn('"screenMaterialName":"@' + target_alias + '"', patched)
        self.assertEqual(patched.count('"screenMaterialName":"@' + target_alias + '"'), 1)
        self.assertNotIn('"screenMaterialName":"@car_nav_screen"', patched)
        self.assertEqual(patched.count('["beamNavigator"'), 1)
        self.assertNotIn('"controller"', part_bodies["car_body_xp_rhd"])
        self.assertNotIn('"controller"', part_bodies["car_door_xp_rhd"])

    def test_runtime_screen_patch_adds_missing_parent_glowmap(self) -> None:
        target_mesh = "car_screen_xp_rhd"
        updated, changed = build_pipeline._patch_runtime_screen_parts(
            '{"car_interior_xp_rhd":{"slotType":"car_interior",'
            '"flexbodies":[["mesh","[group]:"],'
            f'["{target_mesh}",["car_body"]]]}}',
            [
                {
                    "sourceAlias": "car_nav_screen",
                    "targetAlias": "car_nav_screen_beamxp_conversion",
                    "controllerRow": (
                        '["beamNavigator", {'
                        '"screenMaterialName":"@car_nav_screen_beamxp_conversion"}]'
                    ),
                    "glowEntries": {
                        "car_nav_screen": (
                            '{"off":"car_black",'
                            '"on":"car_nav_screen_beamxp_conversion"}'
                        )
                    },
                    "targetMeshes": [target_mesh],
                    "switchForks": [],
                }
            ],
        )

        self.assertEqual(changed, 1)
        self.assertIn('"glowMap":{', updated)
        self.assertIn('"car_nav_screen":{', updated)
        self.assertIn('"on":"car_nav_screen_beamxp_conversion"', updated)

    def test_each_navigator_is_rebound_only_on_its_own_screen_parent(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            source = tmp / "two_navs.zip"
            source_jbeam = r'''
            {
              "nav_a": {
                "slotType":"nav_a",
                "controller":[["fileName"], ["beamNavigator", {
                  "screenMaterialName":"@canvas_a", "name":"nav_a"
                }]],
                "glowMap":{"screen_a":{"off":"black", "on":"canvas_a"}}
              },
              "nav_b": {
                "slotType":"nav_b",
                "controller":[["fileName"], ["beamNavigator", {
                  "screenMaterialName":"@canvas_b", "name":"nav_b"
                }]],
                "glowMap":{"screen_b":{"off":"black", "on":"canvas_b"}}
              }
            }
            '''
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("vehicles/two/navs.jbeam", source_jbeam)
                archive.writestr(
                    "vehicles/two/main.materials.json",
                    json.dumps(
                        {
                            alias: {
                                "name": alias,
                                "mapTo": alias,
                                "Stages": [{"colorMap": "@" + alias}],
                            }
                            for alias in ("canvas_a", "canvas_b")
                        }
                    ),
                )

            output = tmp / "out" / "vehicles" / "two"
            output.mkdir(parents=True)
            mesh_a = core.generated_mesh_name("screen_mesh_a", core.HAND_RHD)
            mesh_b = core.generated_mesh_name("screen_mesh_b", core.HAND_RHD)
            generated_jbeam = output / "handdrive_visual_conversion.jbeam"
            generated_jbeam.write_text(
                '{"interior_a_xp_rhd":{"slotType":"interior_a",'
                '"flexbodies":[["mesh","[group]:"],'
                f'["{mesh_a}",["body"]]],'
                '"glowMap":{"screen_a":'
                '{"off":"black_beamxp_tc", "on":"canvas_a"}}},'
                '"interior_b_xp_rhd":{"slotType":"interior_b",'
                '"flexbodies":[["mesh","[group]:"],'
                f'["{mesh_b}",["body"]]],'
                '"glowMap":{"screen_b":'
                '{"off":"black_beamxp_tc", "on":"canvas_b"}}}}',
                encoding="utf-8",
            )
            context = VehicleContext(
                source_zip=source,
                vehicle_id="two",
                vehicle_path="vehicles/two",
                dae_paths=[],
                variants={},
                objects={},
                preview_by_id={
                    "screen_mesh_a": {"materials": ["screen_a"]},
                    "screen_mesh_b": {"materials": ["screen_b"]},
                },
                jbeam_texts={"vehicles/two/navs.jbeam": source_jbeam},
                node_positions={},
                project_dir=tmp,
                part_body_index={
                    "nav_a": (
                        r'''"nav_a":{"slotType":"nav_a",
                        "controller":[["fileName"],["beamNavigator",{
                        "screenMaterialName":"@canvas_a","name":"nav_a"}]],
                        "glowMap":{"screen_a":{"off":"black","on":"canvas_a"}}}''',
                        "vehicles/two/navs.jbeam",
                    ),
                    "nav_b": (
                        r'''"nav_b":{"slotType":"nav_b",
                        "controller":[["fileName"],["beamNavigator",{
                        "screenMaterialName":"@canvas_b","name":"nav_b"}]],
                        "glowMap":{"screen_b":{"off":"black","on":"canvas_b"}}}''',
                        "vehicles/two/navs.jbeam",
                    ),
                },
            )

            build_pipeline.isolate_converted_runtime_screens(
                context,
                output,
                {"screen_mesh_a", "screen_mesh_b"},
                {core.HAND_RHD},
            )
            patched = generated_jbeam.read_text(encoding="utf-8")
            suffix = build_pipeline.mod_id_for_context(context).lower()

        alias_a = f"canvas_a_beamxp_{suffix}"
        alias_b = f"canvas_b_beamxp_{suffix}"
        split = patched.index('"interior_b_xp_rhd"')
        part_a, part_b = patched[:split], patched[split:]
        self.assertIn(alias_a, part_a)
        self.assertNotIn(alias_b, part_a)
        self.assertIn(alias_b, part_b)
        self.assertNotIn(alias_a, part_b)
        self.assertEqual(patched.count('"screenMaterialName":"@' + alias_a + '"'), 1)
        self.assertEqual(patched.count('"screenMaterialName":"@' + alias_b + '"'), 1)

    def test_lua_owned_runtime_tag_gets_a_conversion_copy_of_its_controller(self) -> None:
        # The LC500's cluster tag lives in the mod's own controller Lua, not in
        # a jbeam field, and obj:createWebView hands one global tag to whichever
        # vehicle asks first -- so the conversion needs its own controller.
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            source = tmp / "lexlc500.zip"
            source_jbeam = r'''
            {
              "lc500_interior": {
                "slotType":"lc500_interior",
                "controller":[["fileName"], ["LEX_LC500_21"], ["gauges/customModules/tireData"]],
                "glowMap":{
                  "lc500_screen_off":{"simpleFunction":{"ignitionLevel":0.5},
                    "off":"lc500_screen_off_off", "on":"lc500_screen_off_on"}
                }
              }
            }
            '''
            controller_lua = (
                'local settings = {\n'
                '  textureName = "@LEX_LC500_21_fh6_gauge",\n'
                '  width = 1280\n'
                '}\n'
                'local htmlPath = "local://local/vehicles/lc500/html/lc500.html"\n'
                'htmlTexture.create(settings.textureName, htmlPath, 1280, 720, 30)\n'
            )
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("vehicles/lc500/lc500_interior.jbeam", source_jbeam)
                archive.writestr(
                    "vehicles/lc500/lc500.dae",
                    _combined_screen_dae(
                        "lc500_screen_off",
                        "lc500_trim",
                    ),
                )
                archive.writestr(
                    "vehicles/lc500/lua/controller/LEX_LC500_21.lua", controller_lua
                )
                archive.writestr(
                    "vehicles/lc500/html/lc500.html",
                    "<html><head><title>gauge</title></head><body>0</body></html>",
                )
                archive.writestr(
                    "vehicles/lc500/main.materials.json",
                    json.dumps(
                        {
                            # This is the DAE-bound glow base.  Giving it an
                            # emissive screen stage puts this exact alias in
                            # display_texture_flip_scope, so the generated DAE
                            # below really does flip the cluster island.
                            "lc500_screen_off": {
                                "name": "lc500_screen_off",
                                "mapTo": "lc500_screen_off",
                                "class": "Material",
                                "Stages": [
                                    {
                                        "emissiveMap": (
                                            "/vehicles/lc500/cluster_boot.png"
                                        )
                                    }
                                ],
                            },
                            "lc500_screen_off_on": {
                                "name": "lc500_screen_off_on",
                                "mapTo": "lc500_screen_off_on",
                                "class": "Material",
                                "Stages": [
                                    {"emissiveMap": "@LEX_LC500_21_fh6_gauge"}
                                ],
                            },
                            "lc500_trim": {
                                "name": "lc500_trim",
                                "mapTo": "lc500_trim",
                                "class": "Material",
                                "Stages": [
                                    {"baseColorMap": "/vehicles/lc500/trim.png"}
                                ],
                            },
                        }
                    ),
                )
            output = tmp / "out" / "vehicles" / "lc500"
            output.mkdir(parents=True)
            generated_jbeam = output / "handdrive_visual_conversion.jbeam"
            # The generated part already carries a corrected "off" state; the
            # rebind must move the live "on" state without undoing that.
            generated_jbeam.write_text(
                '{"lc500_interior_xp_rhd":{"slotType":"lc500_interior",'
                '"controller":[["fileName"], ["LEX_LC500_21"], '
                '["gauges/customModules/tireData"]],'
                '"glowMap":{"lc500_screen_off":{"simpleFunction":{"ignitionLevel":0.5},'
                '"off":"lc500_screen_off_off_beamxp_tc", "on":"lc500_screen_off_on"}}}}',
                encoding="utf-8",
            )
            context = VehicleContext(
                source_zip=source,
                vehicle_id="lc500",
                vehicle_path="vehicles/lc500",
                dae_paths=["vehicles/lc500/lc500.dae"],
                variants={},
                objects={
                    "lc500_interior": DaeObject(
                        id="lc500_interior",
                        name="lc500_interior",
                        dae_path="vehicles/lc500/lc500.dae",
                        x=0.0,
                        y=0.0,
                        z=0.0,
                        geometry_ids=("combined_screen",),
                    )
                },
                preview_by_id={
                    "lc500_interior": {
                        "materials": [
                            "lc500_screen_off",
                            "lc500_trim",
                        ]
                    }
                },
                jbeam_texts={"vehicles/lc500/lc500_interior.jbeam": source_jbeam},
                node_positions={},
                project_dir=tmp,
                part_body_index={
                    "lc500_interior": (
                        source_jbeam,
                        "vehicles/lc500/lc500_interior.jbeam",
                    )
                },
            )

            generated_daes = build_pipeline.generate_daes(
                context,
                tmp / "out",
                output,
                {"lc500_interior": core.MODE_MIRROR},
                {},
                {core.HAND_RHD},
                {},
                set(),
                set(),
                set(),
                [],
                {"lc500_interior"},
            )
            self.assertEqual(len(generated_daes), 1)
            generated_s = _combined_screen_s_values(generated_daes[0])
            scoped_aliases = build_pipeline.display_texture_flip_scope(context)[
                "lc500_interior"
            ]

            report = build_pipeline.isolate_converted_runtime_screens(
                context,
                output,
                {"lc500_interior"},
                {core.HAND_RHD},
                reflected_geometry=True,
            )
            patched = generated_jbeam.read_text(encoding="utf-8")
            materials = json.loads(
                (output / "beamxp_runtime_screens.materials.json").read_text(
                    encoding="utf-8"
                )
            )
            suffix = build_pipeline.mod_id_for_context(context).lower()
            copied = output / "lua" / "controller" / f"LEX_LC500_21_beamxp_{suffix}.lua"
            copied_text = copied.read_text(encoding="utf-8")
            page = output / "html" / f"lc500_beamxp_{suffix}.html"
            page_written = page.exists()

        target_alias = f"lex_lc500_21_fh6_gauge_beamxp_{suffix}"
        target_material = f"lc500_screen_off_on_beamxp_{suffix}"

        self.assertTrue(report["enabled"])
        self.assertIn("lc500_screen_off", scoped_aliases)
        for got, expected in zip(generated_s[:4], (0.9, 0.1, 0.1, 0.9)):
            self.assertAlmostEqual(got, expected)
        # The conversion loads its own controller, and only that row moved.
        self.assertIn(f'["LEX_LC500_21_beamxp_{suffix}"]', patched)
        self.assertNotIn('["LEX_LC500_21"]', patched)
        self.assertIn('["gauges/customModules/tireData"]', patched)
        # That controller creates its own webview tag, not the donor's.
        self.assertIn(f'"@{target_alias}"', copied_text)
        self.assertNotIn('"@LEX_LC500_21_fh6_gauge"', copied_text)
        # The material drawing from the tag follows it, and the glow trigger
        # that switches the screen on now names the conversion's material.
        self.assertEqual(
            materials[target_material]["Stages"][0]["emissiveMap"], "@" + target_alias
        )
        self.assertIn(f'"on":"{target_material}"', patched)
        self.assertIn('"off":"lc500_screen_off_off_beamxp_tc"', patched)
        self.assertEqual(patched.count('"lc500_screen_off"'), 1)
        # The selected screen mesh already has material-scoped UV orientation
        # repair.  Mirrored carriers receive that one flip while rigid pieces
        # keep their authored UVs, so reflecting the HTML would reverse the
        # LC500 gauge a second time.
        self.assertIn('"local://local/vehicles/lc500/html/lc500.html"', copied_text)
        self.assertNotIn(f"lc500_beamxp_{suffix}.html", copied_text)
        self.assertNotIn("transform:scaleX(-1)", copied_text)
        self.assertFalse(page_written)
        self.assertEqual(report["screenPages"], [])

    def test_co_located_unscoped_display_does_not_suppress_html_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            source = tmp / "two_screens.zip"
            source_jbeam = r'''
            {
              "cluster": {
                "slotType":"cluster",
                "controller":[["fileName"], ["customCluster"]],
                "glowMap":{
                  "cluster_base":{"off":"cluster_off", "on":"cluster_on"}
                }
              }
            }
            '''
            controller_lua = (
                'local tag = "@cluster_canvas"\n'
                'local page = "local://local/vehicles/two/html/cluster.html"\n'
                "htmlTexture.create(tag, page, 800, 400, 30)\n"
            )
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("vehicles/two/cluster.jbeam", source_jbeam)
                archive.writestr(
                    "vehicles/two/lua/controller/customCluster.lua",
                    controller_lua,
                )
                archive.writestr(
                    "vehicles/two/html/cluster.html",
                    "<html><head></head><body>cluster</body></html>",
                )
                archive.writestr(
                    "vehicles/two/main.materials.json",
                    json.dumps(
                        {
                            "cluster_on": {
                                "name": "cluster_on",
                                "mapTo": "cluster_on",
                                "Stages": [{"emissiveMap": "@cluster_canvas"}],
                            },
                            "unrelated_nav_screen": {
                                "name": "unrelated_nav_screen",
                                "mapTo": "unrelated_nav_screen",
                                "Stages": [
                                    {
                                        "emissiveMap": (
                                            "/vehicles/two/unrelated.png"
                                        )
                                    }
                                ],
                            },
                        }
                    ),
                )

            output = tmp / "out" / "vehicles" / "two"
            output.mkdir(parents=True)
            generated_jbeam = output / "handdrive_visual_conversion.jbeam"
            generated_jbeam.write_text(
                '{"cluster_xp_rhd":{"slotType":"cluster",'
                '"controller":[["fileName"], ["customCluster"]],'
                '"glowMap":{"cluster_base":'
                '{"off":"cluster_off_beamxp_tc", "on":"cluster_on"}}}}',
                encoding="utf-8",
            )
            context = VehicleContext(
                source_zip=source,
                vehicle_id="two",
                vehicle_path="vehicles/two",
                dae_paths=[],
                variants={},
                objects={},
                preview_by_id={
                    "combined_display": {
                        "materials": ["unrelated_nav_screen", "cluster_base"]
                    }
                },
                jbeam_texts={"vehicles/two/cluster.jbeam": source_jbeam},
                node_positions={},
                project_dir=tmp,
                part_body_index={
                    "cluster": (source_jbeam, "vehicles/two/cluster.jbeam")
                },
            )

            # Both materials occupy one generated source mesh, but the DAE
            # exporter receives only the display scope below as flip_materials.
            # Co-location must not claim the cluster controller's UV island.
            scoped_aliases = build_pipeline.display_texture_flip_scope(context)[
                "combined_display"
            ]

            report = build_pipeline.isolate_converted_runtime_screens(
                context,
                output,
                {"combined_display"},
                {core.HAND_RHD},
                reflected_geometry=True,
            )
            suffix = build_pipeline.mod_id_for_context(context).lower()
            copied_controller = (
                output
                / "lua"
                / "controller"
                / f"customCluster_beamxp_{suffix}.lua"
            ).read_text(encoding="utf-8")
            copied_page = (
                output / "html" / f"cluster_beamxp_{suffix}.html"
            ).read_text(encoding="utf-8")

        self.assertTrue(report["enabled"])
        self.assertEqual(scoped_aliases, frozenset({"unrelated_nav_screen"}))
        self.assertIn(f"cluster_beamxp_{suffix}.html", copied_controller)
        self.assertIn("body{transform:scaleX(-1)", copied_page)
        self.assertNotIn("html{transform:scaleX(-1)", copied_page)
        self.assertEqual(len(report["screenPages"]), 1)

    def test_an_off_centre_screen_window_turns_about_its_own_middle(self) -> None:
        # A quad reading u 0.27..0.79 of its page: reflecting about the page
        # centre would slide the dial 12% of the quad's width out of place.
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            source = tmp / "lexlc500.zip"
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr(
                    "vehicles/lc500/lc500.dae", _screen_quad_dae(0.27493, 0.78493)
                )
            context = VehicleContext(
                source_zip=source,
                vehicle_id="lc500",
                vehicle_path="vehicles/lc500",
                dae_paths=["vehicles/lc500/lc500.dae"],
                variants={},
                objects={},
                preview_by_id={},
                jbeam_texts={},
                node_positions={},
                project_dir=tmp,
                part_body_index={},
            )
            centre = build_pipeline._sampled_u_centre(context, {"lc500_screen_off"})

        self.assertAlmostEqual(centre, 0.52993, places=5)
        page = build_pipeline._mirrored_screen_page("<html><head></head></html>", centre)
        self.assertIn(
            "body{transform:scaleX(-1);transform-origin:52.9930% 50%", page
        )
        self.assertNotIn("html{transform:scaleX(-1)", page)

    def test_a_screen_symbol_with_no_uvs_falls_back_to_the_page_centre(self) -> None:
        self.assertIn(
            "transform-origin:50.0000% 50%",
            build_pipeline._mirrored_screen_page("<html><head></head></html>"),
        )

    def test_unreadable_textures_are_announced_when_others_corrected(self) -> None:
        # A mod may reference a texture it does not ship, so this stays
        # non-fatal -- but it has to reach the caller, not just the log.
        notice = build_pipeline.unreadable_texture_notice(
            [{"texture": "vehicles/v/t/a.dds", "reason": "FileNotFoundError: a"}],
            corrected=5,
            label="v.dae",
        )

        self.assertIsNotNone(notice)
        self.assertIn("could not read 1 texture(s)", notice)
        self.assertIn("a.dds: FileNotFoundError: a", notice)

    def test_reading_none_of_the_textures_raises(self) -> None:
        # Andronisk's V60: 26 unreadable textures, zero corrected, and a build
        # that reported success while every glyph stayed mirrored.
        failures = [
            {"texture": f"vehicles/v/t/{name}.dds", "reason": "FileNotFoundError"}
            for name in "abcd"
        ]

        with self.assertRaises(RuntimeError) as caught:
            build_pipeline.unreadable_texture_notice(
                failures, corrected=0, label="v60_andronisk.dae"
            )

        message = str(caught.exception)
        self.assertIn("read none of the 4 texture(s)", message)
        self.assertIn("v60_andronisk.dae", message)
        self.assertIn("and 1 more", message)

    def test_no_failures_says_nothing(self) -> None:
        self.assertIsNone(
            build_pipeline.unreadable_texture_notice([], corrected=0, label="v.dae")
        )

    def test_a_corrected_texture_name_is_trimmed_only_when_it_will_not_fit(self) -> None:
        material = "v60_andronisk_int_stitch_beamxp_tc.skin_interior.amber_int"
        source = "v60_andronisk_int_stitch_amber_BC.color_rhd.dds"

        # A short project keeps the name it has always written.
        short = Path(r"C:\p\unpacked_output\vehicles\v")
        self.assertEqual(
            build_pipeline._corrected_texture_file_name(short, material, source),
            f"{material}_{source}",
        )

        # The V60's, staged 153 characters deep, came to exactly 260.
        deep = Path(
            r"C:\Users\x\AppData\Local\BeamXP\handedness_conversion_projects"
            r"\volvo_v60_andronisk_v5.1_01-08-26_v60_andronisk"
            r"\unpacked_output\vehicles\v60_andronisk"
        )
        name = build_pipeline._corrected_texture_file_name(deep, material, source)
        self.assertLess(len(str(deep / name)), 260)
        self.assertTrue(name.endswith(".dds"))

    def test_two_corrections_of_one_atlas_keep_distinct_trimmed_names(self) -> None:
        deep = Path(
            r"C:\Users\x\AppData\Local\BeamXP\handedness_conversion_projects"
            r"\volvo_v60_andronisk_v5.1_01-08-26_v60_andronisk"
            r"\unpacked_output\vehicles\v60_andronisk"
        )
        source = "v60_andronisk_int_stitch_amber_BC.color_rhd.dds"
        first = build_pipeline._corrected_texture_file_name(
            deep, "v60_andronisk_int_stitch_beamxp_tc.skin_interior.amber_int", source
        )
        second = build_pipeline._corrected_texture_file_name(
            deep, "v60_andronisk_int_stitch_beamxp_tc.skin_interior.amber_ext", source
        )

        self.assertNotEqual(first, second)

    def test_extraction_workspace_leaves_a_deep_mod_room_under_max_path(self) -> None:
        # Andronisk's V60 nests its textures 113 characters deep. Under the
        # project directory that came to 295 and Windows refused every one of
        # its 26 extractions, so the conversion shipped uncorrected.
        project = Path(
            r"C:\Users\x\AppData\Local\BeamXP\handedness_conversion_projects"
            r"\volvo_v60_andronisk_v5.1_01-08-26_v60_andronisk"
        )
        context = SimpleNamespace(project_dir=project)
        root = build_pipeline.texture_correction_workspace_root(context)
        archive_dir = root / build_pipeline._short_digest("some/mod/archive.zip")
        member = (
            "vehicles/v60_andronisk/texture/v60_andronisk_int_texture/wood"
            "/v60_andronisk_int_texture_wood_white_BC.color.dds"
        )

        self.assertLess(len(str(archive_dir / member)), 260)
        # Scratch, and deliberately not inside the project it belongs to.
        self.assertNotIn(str(project), str(root))
        # Two archives in one conversion still get separate directories.
        self.assertNotEqual(
            build_pipeline._short_digest("a.zip"),
            build_pipeline._short_digest("b.zip"),
        )

    def test_non_power_of_two_dds_falls_back_to_generated_png(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            job = tmp / "job"
            target = tmp / "vehicles/lc500"
            job.mkdir()
            header = bytearray(20)
            header[:4] = b"DDS "
            header[12:16] = (349).to_bytes(4, "little")
            header[16:20] = (922).to_bytes(4, "little")
            (job / "overlay_rhd.dds").write_bytes(header)
            (job / "overlay_rhd.png").write_bytes(b"png")
            (job / "rhd_materials.json").write_text(
                json.dumps(
                    {
                        "materials": [
                            {
                                "aliases": ["screen_overlay"],
                                "maps": {"baseColorMap": "overlay_rhd.png"},
                                "outputMaps": [
                                    {
                                        "stageKey": "baseColorMap",
                                        "member": "vehicles/lc500/overlay.png",
                                        "png": "overlay_rhd.png",
                                        "dds": "overlay_rhd.dds",
                                    }
                                ],
                                "sourceMaterials": [
                                    {
                                        "key": "screen_overlay",
                                        "aliases": ["screen_overlay"],
                                        "material": {
                                            "name": "screen_overlay",
                                            "mapTo": "screen_overlay",
                                            "Stages": [
                                                {
                                                    "baseColorMap": "/vehicles/lc500/overlay.png"
                                                }
                                            ],
                                        },
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            build_pipeline._prepare_texture_correction_materials(job, target, tmp)
            material = json.loads(
                (target / "beamxp_texture_correction.materials.json").read_text(
                    encoding="utf-8"
                )
            )["screen_overlay_beamxp_tc"]

        self.assertTrue(material["Stages"][0]["baseColorMap"].endswith("_overlay_rhd.png"))

    def test_texture_correction_flag_survives_merge_and_import(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            context = minimal_context(Path(raw))
            saved = core.base_conversion_config(context)
            saved["parts"]["scintilla_dashboard"][core.PART_TEXTURE_CORRECTION_KEY] = True

            merged = core.merge_with_current_inventory(context, saved)
            self.assertEqual(
                core.active_texture_correction_mesh_ids(merged),
                {"scintilla_dashboard"},
            )

            current = core.base_conversion_config(context)
            imported, counts = core.import_matching_conversion(context, current, saved)
            self.assertEqual(counts["partImported"], 2)
            self.assertEqual(
                core.active_texture_correction_mesh_ids(imported),
                {"scintilla_dashboard"},
            )

    def test_export_artifacts_batches_marked_meshes_per_dae(self) -> None:
        """Every marked mesh out of one DAE goes through a single export call.

        export_parts_preview groups its texture work by atlas across the parts
        it is given, so meshes sharing an atlas have their UV domains unioned
        and corrected once. Exporting them one at a time discards that grouping
        (its sweep/mask/companion caches are per call) and rebuilds a shared
        atlas once per mesh, shipping competing corrected copies of it.
        """
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            context = minimal_context(tmp)
            loaded = SimpleNamespace(
                parts=[
                    SimpleNamespace(key="scintilla_dashboard", node_id="scintilla_dashboard", node_name="Dashboard"),
                    SimpleNamespace(key="scintilla_dash_controls", node_id="scintilla_dash_controls", node_name="Controls"),
                    SimpleNamespace(key="scintilla_dashboard_race", node_id="scintilla_dashboard_race", node_name="Race Dashboard"),
                ]
            )
            previews = [
                SimpleNamespace(
                    report_path=tmp / "scintilla_rhd_preview.report.json",
                    dae_paths=(
                        tmp / "scintilla_dash_controls_rhd.dae",
                        tmp / "scintilla_dashboard_rhd.dae",
                    ),
                    textures=[object()],
                    seconds=1.5,
                ),
            ]
            progress_messages: list[str] = []

            with (
                patch("mesh_segmentation_transform.beamxp_transform_sym_mesh_POC.scan_vehicle_archive", return_value=object()) as scan,
                patch("mesh_segmentation_transform.beamxp_transform_sym_mesh_POC.extract_archive_member", return_value=tmp / "source.dae"),
                patch("mesh_segmentation_transform.beamxp_transform_sym_mesh_POC.load_dae", return_value=loaded) as load,
                patch("mesh_segmentation_transform.mirror_texture_for_rhd.export_parts_preview", side_effect=previews) as export,
            ):
                report = core.export_texture_correction_artifacts(
                    context,
                    tmp / "unpacked_output",
                    ["scintilla_dashboard", "scintilla_dash_controls"],
                    progress=progress_messages.append,
                )

            scan.assert_called_once()
            load.assert_called_once()
            export.assert_called_once()
            # Both marked meshes reach the one call; the unmarked race dashboard
            # sharing the DAE must not be dragged in with them.
            self.assertEqual(
                [part.key for part in export.call_args.args[2]],
                ["scintilla_dashboard", "scintilla_dash_controls"],
            )
            self.assertTrue(export.call_args.args[4].detect_on_normal_map)
            self.assertEqual(len(report["jobs"]), 1)
            self.assertEqual(
                report["jobs"][0]["meshes"],
                ["scintilla_dash_controls", "scintilla_dashboard"],
            )
            self.assertEqual(report["missing"], [])
            self.assertTrue(any("Texture correction:" in message for message in progress_messages))
            self.assertTrue(any("finished" in message for message in progress_messages))

    def test_export_auto_includes_only_mirrored_dependencies_sharing_an_atlas(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            context = minimal_context(tmp)
            for mesh_id in ("door_source", "unrelated_source"):
                context.objects[mesh_id] = DaeObject(
                    id=mesh_id,
                    name=mesh_id,
                    dae_path="vehicles/scintilla/scintilla.dae",
                    x=0.0,
                    y=0.0,
                    z=0.0,
                    geometry_ids=(f"{mesh_id}_geom",),
                )
            loaded = SimpleNamespace(
                parts=[
                    SimpleNamespace(
                        key="scintilla_dashboard",
                        node_id="scintilla_dashboard",
                        node_name="Dashboard",
                    ),
                    SimpleNamespace(
                        key="door_source", node_id="door_source", node_name="Door"
                    ),
                    SimpleNamespace(
                        key="unrelated_source",
                        node_id="unrelated_source",
                        node_name="Unrelated",
                    ),
                ]
            )
            preview = SimpleNamespace(
                report_path=tmp / "scintilla_rhd_preview.report.json",
                dae_paths=(tmp / "dashboard_rhd.dae", tmp / "door_rhd.dae"),
                textures=[object()],
                seconds=1.0,
                failed_parts=(),
            )

            def texture_bindings(_archive, _loaded, parts):
                keys = {part.key for part in parts}
                if "scintilla_dashboard" in keys or "door_source" in keys:
                    return {"vehicles/scintilla/shared.png": []}
                return {"vehicles/scintilla/unrelated.png": []}

            with (
                patch(
                    "mesh_segmentation_transform.beamxp_transform_sym_mesh_POC.scan_vehicle_archive",
                    return_value=object(),
                ),
                patch(
                    "mesh_segmentation_transform.beamxp_transform_sym_mesh_POC.extract_archive_member",
                    return_value=tmp / "source.dae",
                ),
                patch(
                    "mesh_segmentation_transform.beamxp_transform_sym_mesh_POC.load_dae",
                    return_value=loaded,
                ),
                patch(
                    "mesh_segmentation_transform.mirror_texture_for_rhd.texture_bindings_for_parts",
                    side_effect=texture_bindings,
                ),
                patch(
                    "mesh_segmentation_transform.mirror_texture_for_rhd.export_parts_preview",
                    return_value=preview,
                ) as export,
            ):
                report = core.export_texture_correction_artifacts(
                    context,
                    tmp / "artifacts",
                    ["scintilla_dashboard"],
                    shared_atlas_dependency_targets={
                        "door_source": {"door_target"},
                        "unrelated_source": {"unrelated_target"},
                    },
                    force_mirrored_dependency_ids={"door_source"},
                )

        self.assertEqual(len(export.call_args_list), 2)
        self.assertEqual(
            [part.key for part in export.call_args_list[0].args[2]],
            ["scintilla_dashboard"],
        )
        self.assertEqual(
            [part.key for part in export.call_args_list[1].args[2]], ["door_source"]
        )
        self.assertEqual(
            export.call_args_list[0].kwargs["texture_member_scope"],
            {"vehicles/scintilla/shared.png"},
        )
        self.assertEqual(
            export.call_args_list[1].kwargs["texture_member_scope"],
            {"vehicles/scintilla/shared.png"},
        )
        self.assertEqual(
            export.call_args_list[0].kwargs["force_mirrored_part_keys"],
            {"door_source"},
        )
        self.assertEqual(
            export.call_args_list[1].kwargs["force_mirrored_part_keys"],
            {"door_source"},
        )
        self.assertEqual(
            [part.key for part in export.call_args_list[0].kwargs["texture_part_scope"]],
            ["scintilla_dashboard", "door_source"],
        )
        self.assertEqual(
            [part.key for part in export.call_args_list[1].kwargs["texture_part_scope"]],
            ["door_source"],
        )
        self.assertEqual(report["autoIncludedTargets"], {"door_source": ["door_target"]})
        self.assertEqual(
            [job["meshes"] for job in report["jobs"]],
            [["scintilla_dashboard"], ["door_source"]],
        )

    def test_a_skipped_part_is_reported_per_mesh_not_per_dae(self) -> None:
        """One unbindable mesh must not be reported as ten failed ones.

        The exporter skips a part it cannot wire and carries on, so the job
        keeps the meshes it did correct and only the skipped mesh lands in
        failures. Reporting the whole DAE as failed is what hid ten silently
        uncorrected scintilla meshes behind one carbon-fibre console.
        """
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            context = minimal_context(tmp)
            loaded = SimpleNamespace(
                parts=[
                    SimpleNamespace(key="scintilla_dashboard", node_id="scintilla_dashboard", node_name="Dashboard"),
                    SimpleNamespace(key="scintilla_dash_controls", node_id="scintilla_dash_controls", node_name="Controls"),
                ]
            )
            preview = SimpleNamespace(
                report_path=tmp / "scintilla_rhd_preview.report.json",
                dae_paths=(tmp / "scintilla_dashboard_rhd.dae",),
                textures=[object()],
                seconds=1.5,
                failed_parts=(
                    {
                        "source_part": {
                            "key": "scintilla_dash_controls",
                            "label": "scintilla_dash_controls",
                            "node_id": "scintilla_dash_controls",
                            "node_name": "Controls",
                        },
                        "error": "ValueError: archive texture aliases did not match carbon",
                    },
                ),
            )
            progress_messages: list[str] = []

            with (
                patch("mesh_segmentation_transform.beamxp_transform_sym_mesh_POC.scan_vehicle_archive", return_value=object()),
                patch("mesh_segmentation_transform.beamxp_transform_sym_mesh_POC.extract_archive_member", return_value=tmp / "source.dae"),
                patch("mesh_segmentation_transform.beamxp_transform_sym_mesh_POC.load_dae", return_value=loaded),
                patch("mesh_segmentation_transform.mirror_texture_for_rhd.export_parts_preview", return_value=preview),
            ):
                report = core.export_texture_correction_artifacts(
                    context,
                    tmp / "unpacked_output",
                    ["scintilla_dashboard", "scintilla_dash_controls"],
                    progress=progress_messages.append,
                )

            self.assertEqual(report["jobs"][0]["meshes"], ["scintilla_dashboard"])
            self.assertEqual(len(report["failures"]), 1)
            self.assertEqual(report["failures"][0]["meshes"], ["scintilla_dash_controls"])
            self.assertIn("carbon", report["failures"][0]["error"])
            self.assertTrue(any("skipped 1" in message for message in progress_messages))

    def test_a_corrected_mesh_leaves_every_flexbody_row_as_it_found_it(self) -> None:
        """The ETK K-Series drift trim: five segments of a console it never had.

        ``etkc_dash_race_lower`` is commented out of the stripped interior, and
        while the correction expanded a row into one row per piece the ``//``
        stayed on the first piece alone -- the other four became live rows and
        the drift trim wore them on the floor. Grouping the pieces back into
        the mesh means no row is rewritten at all, so a commented one stays
        commented whatever the correction did to the mesh it names.
        """
        with tempfile.TemporaryDirectory() as raw:
            vehicle_dir = Path(raw)
            jbeam = vehicle_dir / "handdrive_visual_conversion.jbeam"
            source = """{
  "etkc_dash_xp_rhd": {
    "flexbodies": [
      ["etkc_dash_race_lower_xp_rhd", ["etkc_dash"]]
    ]
  },
  "etkc_dash_stripped_xp_rhd": {
    "flexbodies": [
      //["etkc_dash_race_lower_xp_rhd", ["etkc_dash"]],
      ["etkc_intcarpet_stripped_xp_rhd", ["etkc_body"]]
    ]
  }
}"""
            jbeam.write_text(source, encoding="utf-8")

            result = build_pipeline._patch_texture_correction_jbeams(
                vehicle_dir,
                {"etkc_dash_race_lower_xp_rhd"},
            )

            self.assertEqual(result["files"], [])
            self.assertEqual(jbeam.read_text(encoding="utf-8"), source)

    def test_jbeam_patch_clones_glowmap_for_corrected_material_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            vehicle_dir = Path(raw)
            jbeam = vehicle_dir / "ardente_body.jbeam"
            jbeam.write_text(
                """{
  "ardente_body": {
    "glowMap":{
      "ardente_interior":{"simpleFunction":{"lowhighbeam":0.49}, "off":"ardente_interior", "on":"ardente_interior_on", "on_intense":"ardente_interior_on"},
      "screen_runtime":{"simpleFunction":{"ignitionLevel":0.1}, "off":"screen_off", "on":"@lc500_GPS"}
    }
  }
}
""",
                encoding="utf-8",
            )

            result = build_pipeline._patch_texture_correction_jbeams(
                vehicle_dir,
                {},
                [
                    {
                        "ardente_interior": "ardente_interior_beamxp_tc",
                        "ardente_interior_on": "ardente_interior_on_beamxp_tc",
                    }
                ],
            )

            updated = jbeam.read_text(encoding="utf-8")

        self.assertEqual(result["files"], [str(jbeam)])
        self.assertIn('"ardente_interior_beamxp_tc"', updated)
        self.assertIn('"off":"ardente_interior_beamxp_tc"', updated)
        self.assertIn('"on":"ardente_interior_on_beamxp_tc"', updated)
        self.assertIn('"on_intense":"ardente_interior_on_beamxp_tc"', updated)
        self.assertIn('"on":"@lc500_GPS"', updated)

    def test_jbeam_patch_uses_combined_aliases_for_split_glow_state_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            vehicle_dir = Path(raw)
            jbeam = vehicle_dir / "lc500.jbeam"
            jbeam.write_text(
                """{
  "lc500": {
    "glowMap":{
      "lc500_screen_off":{"simpleFunction":"running", "off":"lc500_screen_off_off", "on":"lc500_screen_off_on"}
    }
  }
}
""",
                encoding="utf-8",
            )

            build_pipeline._patch_texture_correction_jbeams(
                vehicle_dir,
                {},
                [
                    {
                        "lc500_screen_off": "lc500_screen_off_off_beamxp_tc",
                        "lc500_screen_off_off": "lc500_screen_off_off_beamxp_tc",
                    },
                    {
                        "lc500_screen_off_on": "lc500_screen_off_on_beamxp_tc",
                    },
                ],
            )
            updated = jbeam.read_text(encoding="utf-8")

        self.assertIn('"lc500_screen_off_off_beamxp_tc"', updated)
        self.assertIn('"off":"lc500_screen_off_off_beamxp_tc"', updated)
        self.assertIn('"on":"lc500_screen_off_on_beamxp_tc"', updated)

    def test_jbeam_patch_retargets_preserved_switch_base_in_place(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            vehicle_dir = Path(raw)
            jbeam = vehicle_dir / "lc500.jbeam"
            jbeam.write_text(
                """{
  "lc500": {
    "glowMap":{
      "lc500_screen_off":{"simpleFunction":"running", "off":"lc500_screen_off_off", "on":"lc500_screen_off_on"}
    }
  }
}
""",
                encoding="utf-8",
            )

            build_pipeline._patch_texture_correction_jbeams(
                vehicle_dir,
                {},
                [
                    {
                        "lc500_screen_off": "lc500_screen_off_off_beamxp_tc",
                        "lc500_screen_off_off": "lc500_screen_off_off_beamxp_tc",
                    },
                    {
                        "lc500_screen_off_on": "lc500_screen_off_on_beamxp_tc",
                    },
                ],
                {"lc500_screen_off"},
            )
            updated = jbeam.read_text(encoding="utf-8")

        self.assertIn('"lc500_screen_off"', updated)
        self.assertNotIn('"lc500_screen_off_off_beamxp_tc":', updated)
        self.assertIn('"off":"lc500_screen_off_off_beamxp_tc"', updated)
        self.assertIn('"on":"lc500_screen_off_on_beamxp_tc"', updated)

    def test_jbeam_patch_leaves_a_rigid_only_switch_on_its_stock_states(self) -> None:
        """The LC500 gauge cluster: a switch on a piece nothing reflected.

        The cluster's 38 faces are a perimeter-symmetric candidate, so the
        split translates them rather than mirroring them, while the backing
        quad behind them binds the same off-state material directly and is
        mirrored.  Correcting that shared atlas is right for the quad; pointing
        the cluster's switch at it hands an unreflected mesh a pre-reversed
        image, which is how the cluster shipped reading backwards.
        """
        with tempfile.TemporaryDirectory() as raw:
            vehicle_dir = Path(raw)
            jbeam = vehicle_dir / "lc500.jbeam"
            jbeam.write_text(
                """{
  "lc500": {
    "glowMap":{
      "lc500_screen_off":{"simpleFunction":"running", "off":"lc500_screen_off_off", "on":"lc500_screen_off_on"}
    }
  }
}
""",
                encoding="utf-8",
            )

            build_pipeline._patch_texture_correction_jbeams(
                vehicle_dir,
                {},
                [
                    {
                        "lc500_screen_off_off": "lc500_screen_off_off_beamxp_tc",
                    }
                ],
                rigid_only_aliases={"lc500_screen_off"},
            )
            updated = jbeam.read_text(encoding="utf-8")

        self.assertIn('"off":"lc500_screen_off_off"', updated)
        self.assertNotIn("lc500_screen_off_off_beamxp_tc", updated)

    def test_rigid_only_aliases_exclude_materials_a_carrier_still_paints(self) -> None:
        pieces = {
            "lc500_interior__beamxp_mirrored_carrier": {
                "lc500_leather1",
                "lc500_screen_off_off_beamxp_tc",
            },
            "lc500_interior__beamxp_rigid_001": {
                "lc500_chrome",
                "lc500_screen_off",
            },
            "lc500_interior__beamxp_rigid_002": {"lc500_leather1"},
        }

        self.assertEqual(
            build_pipeline._rigid_only_material_aliases(pieces),
            {"lc500_chrome", "lc500_screen_off"},
        )

    def test_jbeam_patch_imports_source_glowmap_into_generated_part(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            vehicle_dir = Path(raw)
            jbeam = vehicle_dir / "handdrive_visual_conversion.jbeam"
            jbeam.write_text(
                '''{
                  "lc500_interior_xp_rhd": {
                    "flexbodies": [
                      ["mesh", "[group]:"],
                      ["lc500_interior_xp_rhd", ["dash"]]
                    ]
                  }
                }''',
                encoding="utf-8",
            )
            source_jbeam = '''{
              "lc500": {
                "glowMap": {
                  "lc500_screen_off": {
                    "simpleFunction":"running",
                    "off":"lc500_screen_off_off",
                    "on":"lc500_screen_off_on"
                  }
                }
              }
            }'''

            build_pipeline._patch_texture_correction_jbeams(
                vehicle_dir,
                {
                    "lc500_interior_xp_rhd": [
                        "lc500_interior__beamxp_mirrored_carrier"
                    ]
                },
                [
                    {
                        "lc500_screen_off": "lc500_screen_off_off_beamxp_tc",
                        "lc500_screen_off_off": "lc500_screen_off_off_beamxp_tc",
                        "lc500_screen_off_on": "lc500_screen_off_on_beamxp_tc",
                    }
                ],
                {"lc500_screen_off"},
                [source_jbeam],
            )
            updated = jbeam.read_text(encoding="utf-8")

        self.assertIn('"glowMap"', updated)
        self.assertIn('"lc500_screen_off"', updated)
        self.assertIn('"off":"lc500_screen_off_off_beamxp_tc"', updated)
        self.assertIn('"on":"lc500_screen_off_on_beamxp_tc"', updated)

    def test_jbeam_patch_retargets_runtime_switch_state_without_cloning_base(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            vehicle_dir = Path(raw)
            jbeam = vehicle_dir / "lc500.jbeam"
            jbeam.write_text(
                """{
  "lc500": {
    "glowMap":{
      "lc500_centralscreen":{"simpleFunction":{"ignitionLevel":0.5}, "off":"lc500_screens_off", "on":"lc500_centralscreen_on", "on_intense":"lc500_GPS"}
    }
  }
}
""",
                encoding="utf-8",
            )

            build_pipeline._patch_texture_correction_jbeams(
                vehicle_dir,
                {},
                [
                    {
                        "lc500_centralscreen_on": "lc500_centralscreen_on_beamxp_tc",
                    },
                ],
            )
            updated = jbeam.read_text(encoding="utf-8")

        self.assertIn('"lc500_centralscreen"', updated)
        self.assertNotIn('"lc500_centralscreen_on_beamxp_tc":', updated)
        self.assertIn('"on":"lc500_centralscreen_on_beamxp_tc"', updated)
        self.assertIn('"on_intense":"lc500_GPS"', updated)

    def test_prune_keeps_texture_corrected_materials_referenced_by_glowmap(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            vehicle_dir = tmp / "vehicles/car"
            vehicle_dir.mkdir(parents=True)
            for name in ("base.dds", "on.dds", "unused.dds"):
                _placeholder_dds(vehicle_dir / name)
            (vehicle_dir / "car.dae").write_text(
                """<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema">
  <library_materials>
    <material id="dash_beamxp_tc-material" name="dash_beamxp_tc"/>
  </library_materials>
  <library_geometries>
    <geometry id="geom"><mesh><triangles material="dash_beamxp_tc" count="0"/></mesh></geometry>
  </library_geometries>
</COLLADA>
""",
                encoding="utf-8",
            )
            (vehicle_dir / "car.jbeam").write_text(
                """{
  "dash": {
    "glowMap":{
      "dash_beamxp_tc":{"off":"dash_beamxp_tc", "on":"dash_on_beamxp_tc"}
    }
  }
}
""",
                encoding="utf-8",
            )
            (vehicle_dir / "beamxp_texture_correction.materials.json").write_text(
                json.dumps(
                    {
                        "dash_beamxp_tc": {
                            "Stages": [{"baseColorMap": "/vehicles/car/base.dds"}]
                        },
                        "dash_on_beamxp_tc": {
                            "Stages": [{"baseColorMap": "/vehicles/car/on.dds"}]
                        },
                        "dash_unused_beamxp_tc": {
                            "Stages": [{"baseColorMap": "/vehicles/car/unused.dds"}]
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = build_pipeline.prune_unused_texture_correction_assets(
                tmp,
                vehicle_dir,
            )
            materials = json.loads(
                (vehicle_dir / "beamxp_texture_correction.materials.json").read_text(
                    encoding="utf-8"
                )
            )
            kept_glow_texture = (vehicle_dir / "on.dds").is_file()
            removed_unused_texture = not (vehicle_dir / "unused.dds").exists()

        self.assertEqual(result["removedMaterials"], ["dash_unused_beamxp_tc"])
        self.assertIn("dash_beamxp_tc", materials)
        self.assertIn("dash_on_beamxp_tc", materials)
        self.assertTrue(kept_glow_texture)
        self.assertTrue(removed_unused_texture)

    def test_structural_texture_correction_retargets_generated_node_without_row_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            context = minimal_context(tmp)
            output_root = tmp / "unpacked_output"
            output_vehicle_dir = output_root / context.vehicle_path
            output_vehicle_dir.mkdir(parents=True)
            target_dae = output_vehicle_dir / "scintilla_handdrive.dae"
            target_dae.write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema"><library_materials>
<material id="door_mat-material" name="door_mat"/></library_materials><library_geometries>
<geometry id="geom_generated"><mesh><triangles material="door_mat-material" count="0"/></mesh></geometry>
</library_geometries><library_visual_scenes><visual_scene id="Scene">
<node id="doorpanel_FL_xp_rhd" name="doorpanel_FL_xp_rhd"><matrix>1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</matrix>
<instance_geometry url="#geom_generated"><bind_material><technique_common>
<instance_material symbol="door_mat-material" target="#door_mat-material"/>
</technique_common></bind_material></instance_geometry></node>
</visual_scene></library_visual_scenes></COLLADA>""",
                encoding="utf-8",
            )
            jbeam_dir = output_vehicle_dir / "jbeam"
            jbeam_dir.mkdir()
            (jbeam_dir / "handdrive_visual_conversion.jbeam").write_text(
                '{"doorpanel_FL_xp_rhd":{"flexbodies":[["mesh", "[group]:"],'
                '["doorpanel_FL_xp_rhd", ["door_FL"]]]}}',
                encoding="utf-8",
            )
            job = tmp / "texture_job"
            job.mkdir()
            source_dae = job / "doorpanel_FR_rhd.dae"
            source_dae.write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema"><library_materials>
<material id="door_mat-material" name="door_mat"/></library_materials><library_geometries/>
<library_visual_scenes><visual_scene id="Scene">
<node id="doorpanel_FR__beamxp_mirrored_carrier" name="doorpanel_FR__beamxp_mirrored_carrier"/>
</visual_scene></library_visual_scenes></COLLADA>""",
                encoding="utf-8",
            )
            (job / "rhd_materials.json").write_text(
                json.dumps(
                    {
                        "materials": [
                            {
                                "aliases": ["door_mat", "door_mat-material"],
                                "maps": {},
                            }
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            detail_path = job / "texture.report.json"
            detail_path.write_text(
                json.dumps(
                    {
                        "dae_exports": [
                            {
                                "source_part": {"key": "doorpanel_FR"},
                                "generated_flexbody_rows": [
                                    {"node_id": "doorpanel_FR__beamxp_mirrored_carrier"}
                                ],
                                "dae_path": str(source_dae),
                            }
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = build_pipeline.integrate_texture_correction_artifacts(
                context,
                output_root,
                output_vehicle_dir,
                {
                    "jobs": [
                        {
                            "dae": "vehicles/scintilla/scintilla.dae",
                            "outputDirectory": str(job),
                            "reportPath": str(detail_path),
                        }
                    ]
                },
                {core.HAND_RHD},
                texture_correction_targets={"doorpanel_FR": {"doorpanel_FL"}},
                structural_sources={"doorpanel_FL": "doorpanel_FR"},
            )

            jbeam = (jbeam_dir / "handdrive_visual_conversion.jbeam").read_text(encoding="utf-8")
            root = ET.parse(target_dae).getroot()

        self.assertEqual(result["groupedMeshes"], [])
        self.assertEqual(result["jbeamPatch"], {"files": [], "renamedRows": 0})
        self.assertEqual(result["daePatches"][0]["groupedNodes"], [])
        self.assertEqual(result["daePatches"][0]["retargetedNodes"], ["doorpanel_FL_xp_rhd"])
        self.assertIn('"doorpanel_FL_xp_rhd", ["door_FL"]', jbeam)
        self.assertNotIn("doorpanel_FR__beamxp_mirrored_carrier", jbeam)

        binding = root.find(".//c:node[@id='doorpanel_FL_xp_rhd']//c:instance_material", core.NS)
        triangle = root.find(".//c:geometry[@id='geom_generated']//c:triangles", core.NS)
        appended = root.find(".//c:node[@id='doorpanel_FR__beamxp_mirrored_carrier']", core.NS)
        material = root.find(".//c:material[@id='door_mat_beamxp_tc-material']", core.NS)
        self.assertIsNotNone(binding)
        assert binding is not None
        self.assertEqual(binding.get("symbol"), "door_mat_beamxp_tc")
        self.assertEqual(binding.get("target"), "#door_mat_beamxp_tc-material")
        self.assertIsNotNone(triangle)
        assert triangle is not None
        self.assertEqual(triangle.get("material"), "door_mat_beamxp_tc")
        self.assertIsNone(appended)
        self.assertIsNotNone(material)

    def test_a_prop_mesh_is_retargeted_on_its_baked_copy_not_split_into_rows(self) -> None:
        """The Andronisk signal stalk: corrected, then wired to nothing.

        A prop row names one mesh and the engine spawns exactly that one, so
        the split pieces have no row to ride and the corrected material has to
        land on the whole-mesh copy the row already names. That copy is baked
        per row -- ``signalstalk_xp_rhd__<config>__<part>__<index>`` -- so it
        answers to neither ``generated_mesh_name`` nor the pieces' names, and
        the stalk shipped mirrored on its uncorrected shipped atlas.
        """
        baked = "signalstalk_xp_rhd__cross_250__dash__0000"
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            context = minimal_context(tmp)
            output_root = tmp / "unpacked_output"
            output_vehicle_dir = output_root / context.vehicle_path
            output_vehicle_dir.mkdir(parents=True)
            target_dae = output_vehicle_dir / "scintilla_handdrive.dae"
            target_dae.write_text(
                f"""<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema"><library_materials>
<material id="buttons_off-material" name="buttons_off"/></library_materials><library_geometries>
<geometry id="geom_baked"><mesh><triangles material="buttons_off-material" count="0"/></mesh></geometry>
</library_geometries><library_visual_scenes><visual_scene id="Scene">
<node id="{baked}" name="{baked}"><matrix>1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</matrix>
<instance_geometry url="#geom_baked"><bind_material><technique_common>
<instance_material symbol="buttons_off-material" target="#buttons_off-material"/>
</technique_common></bind_material></instance_geometry></node>
</visual_scene></library_visual_scenes></COLLADA>""",
                encoding="utf-8",
            )
            jbeam_dir = output_vehicle_dir / "jbeam"
            jbeam_dir.mkdir()
            jbeam_path = jbeam_dir / "handdrive_visual_conversion.jbeam"
            jbeam_path.write_text(
                '{"signalstalk_xp_rhd":{"props":[["func", "mesh", "idRef:", "idX:", "idY:"],'
                f'["turnsignal", "{baked}", "f5l", "f5r", "dsh5"]]}}}}',
                encoding="utf-8",
            )
            job = tmp / "texture_job"
            job.mkdir()
            source_dae = job / "signalstalk_rhd.dae"
            source_dae.write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema"><library_materials>
<material id="buttons_off-material" name="buttons_off"/></library_materials><library_geometries/>
<library_visual_scenes><visual_scene id="Scene">
<node id="signalstalk__beamxp_mirrored_carrier" name="signalstalk__beamxp_mirrored_carrier"/>
</visual_scene></library_visual_scenes></COLLADA>""",
                encoding="utf-8",
            )
            (job / "rhd_materials.json").write_text(
                json.dumps(
                    {
                        "materials": [
                            {
                                "aliases": ["buttons_off", "buttons_off-material"],
                                "maps": {},
                            }
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            detail_path = job / "texture.report.json"
            detail_path.write_text(
                json.dumps(
                    {
                        "dae_exports": [
                            {
                                "source_part": {"key": "signalstalk"},
                                "generated_flexbody_rows": [
                                    {"node_id": "signalstalk__beamxp_mirrored_carrier"}
                                ],
                                "dae_path": str(source_dae),
                            }
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = build_pipeline.integrate_texture_correction_artifacts(
                context,
                output_root,
                output_vehicle_dir,
                {
                    "jobs": [
                        {
                            "dae": "vehicles/scintilla/scintilla.dae",
                            "outputDirectory": str(job),
                            "reportPath": str(detail_path),
                        }
                    ]
                },
                {core.HAND_RHD},
                texture_correction_targets={"signalstalk": {"signalstalk"}},
                prop_meshes={"signalstalk"},
                baked_mesh_copies={("signalstalk", core.HAND_RHD): [baked]},
            )

            jbeam = jbeam_path.read_text(encoding="utf-8")
            root = ET.parse(target_dae).getroot()

        patch_report = result["daePatches"][0]
        self.assertEqual(patch_report["propTargetMeshes"], ["signalstalk"])
        # Nothing grouped and no row rewritten: rebuilding the node would move
        # the frame the prop is placed from.
        self.assertEqual(patch_report["groupedNodes"], [])
        self.assertEqual(result["groupedMeshes"], [])
        self.assertIn(baked, patch_report["retargetedNodes"])
        self.assertIn(f'"turnsignal", "{baked}"', jbeam)
        self.assertIsNone(
            root.find(".//c:node[@id='signalstalk__beamxp_mirrored_carrier']", core.NS)
        )
        # The corrected material lands on the copy the row actually names.
        binding = root.find(f".//c:node[@id='{baked}']//c:instance_material", core.NS)
        triangle = root.find(".//c:geometry[@id='geom_baked']//c:triangles", core.NS)
        self.assertIsNotNone(binding)
        assert binding is not None
        self.assertEqual(binding.get("symbol"), "buttons_off_beamxp_tc")
        self.assertIsNotNone(triangle)
        assert triangle is not None
        self.assertEqual(triangle.get("material"), "buttons_off_beamxp_tc")

    def test_a_mesh_bound_as_both_keeps_a_copy_for_each_binding(self) -> None:
        """The Ardente's ``grp_shifter_knob_a``: a prop and a flexbody at once.

        Its race shifter renders the knob as a flexbody and its sequential
        shifter animates the same mesh as a prop, in different configs of one
        conversion. The prop has to keep the whole-mesh copy it is placed
        from, so the grouped correction is added beside it and only the
        flexbody and ``mirrors`` rows are moved onto it -- one name for one
        name, which is what leaves a commented row commented.
        """
        generated = core.generated_mesh_name("grp_shifter_knob_a", core.HAND_RHD)
        corrected = f"{generated}__beamxp_corrected"
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            context = minimal_context(tmp)
            output_root = tmp / "unpacked_output"
            output_vehicle_dir = output_root / context.vehicle_path
            output_vehicle_dir.mkdir(parents=True)
            target_dae = output_vehicle_dir / "scintilla_handdrive.dae"
            target_dae.write_text(
                f"""<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema"><library_materials>
<material id="knob_mat-material" name="knob_mat"/></library_materials><library_geometries>
<geometry id="geom_whole"><mesh><triangles material="knob_mat-material" count="0"/></mesh></geometry>
</library_geometries><library_visual_scenes><visual_scene id="Scene">
<node id="{generated}" name="{generated}"><matrix>1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</matrix>
<instance_geometry url="#geom_whole"><bind_material><technique_common>
<instance_material symbol="knob_mat-material" target="#knob_mat-material"/>
</technique_common></bind_material></instance_geometry></node>
</visual_scene></library_visual_scenes></COLLADA>""",
                encoding="utf-8",
            )
            jbeam_dir = output_vehicle_dir / "jbeam"
            jbeam_dir.mkdir()
            jbeam_path = jbeam_dir / "handdrive_visual_conversion.jbeam"
            jbeam_path.write_text(
                f"""{{
  "shifter_race_xp_rhd": {{
    "flexbodies": [
      ["mesh", "[group]:"],
      ["{generated}", ["shifter_lever"]],
    ],
    "mirrors": [
      ["mesh", "idRef:", "id1:", "id2:"],
      ["{generated}","rf1","rf1r","rf2"],
    ],
  }},
  "shiftknob_sq_xp_rhd": {{
    "props": [
      ["func", "mesh", "idRef:", "idX:", "idY:"],
      ["sequentialLeverY", "{generated}", "st_1r", "st_1l", "st_2r"],
    ],
  }},
}}""",
                encoding="utf-8",
            )
            job = tmp / "texture_job"
            job.mkdir()
            source_dae = job / "knob_rhd.dae"
            source_dae.write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema"><library_materials>
<material id="knob_mat-material" name="knob_mat"/></library_materials>
<library_geometries>
<geometry id="piece"><mesh><triangles material="knob_mat-material" count="0"/></mesh></geometry>
</library_geometries>
<library_visual_scenes><visual_scene id="Scene">
<node id="grp_shifter_knob_a__beamxp_mirrored_carrier" name="grp_shifter_knob_a__beamxp_mirrored_carrier">
<instance_geometry url="#piece"/></node>
</visual_scene></library_visual_scenes></COLLADA>""",
                encoding="utf-8",
            )
            (job / "rhd_materials.json").write_text(
                json.dumps(
                    {
                        "materials": [
                            {"aliases": ["knob_mat", "knob_mat-material"], "maps": {}}
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            detail_path = job / "texture.report.json"
            detail_path.write_text(
                json.dumps(
                    {
                        "dae_exports": [
                            {
                                "source_part": {"key": "grp_shifter_knob_a"},
                                "generated_flexbody_rows": [
                                    {"node_id": "grp_shifter_knob_a__beamxp_mirrored_carrier"}
                                ],
                                "dae_path": str(source_dae),
                            }
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = build_pipeline.integrate_texture_correction_artifacts(
                context,
                output_root,
                output_vehicle_dir,
                {
                    "jobs": [
                        {
                            "dae": "vehicles/scintilla/scintilla.dae",
                            "outputDirectory": str(job),
                            "reportPath": str(detail_path),
                        }
                    ]
                },
                {core.HAND_RHD},
                texture_correction_targets={"grp_shifter_knob_a": {"grp_shifter_knob_a"}},
                prop_meshes={"grp_shifter_knob_a"},
                flexbody_meshes={"grp_shifter_knob_a"},
            )

            jbeam = jbeam_path.read_text(encoding="utf-8")
            root = ET.parse(target_dae).getroot()

        self.assertEqual(result["renamedMeshes"], {generated: corrected})
        self.assertEqual(result["daePatches"][0]["groupedNodes"], [corrected])
        self.assertEqual(result["jbeamPatch"]["renamedRows"], 1)
        # The flexbody and the reflection follow; the prop stays on the copy
        # it is placed from.
        self.assertIn(f'["{corrected}", ["shifter_lever"]]', jbeam)
        self.assertIn(f'["{corrected}","rf1"', jbeam)
        self.assertIn(f'["sequentialLeverY", "{generated}"', jbeam)
        self.assertIsNotNone(root.find(f".//c:node[@id='{generated}']", core.NS))
        self.assertIsNotNone(root.find(f".//c:node[@id='{corrected}']", core.NS))

    def test_the_correction_pieces_are_grouped_into_the_mesh_they_came_from(self) -> None:
        """One node, under the name the flexbody row already had.

        The sweep hands back a mirrored carrier and a rigidly moved piece,
        each on its own node with its own placement. Both are baked flat and
        merged into a single mesh so the conversion ships exactly the mesh
        count it started with, and the whole-mesh copy they supersede goes
        with its geometry.
        """
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            target = tmp / "target.dae"
            source = tmp / "source.dae"
            target.write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema">
  <library_geometries>
    <geometry id="geom_whole"><mesh><triangles material="dash_mat-material" count="0"/></mesh></geometry>
  </library_geometries>
  <library_visual_scenes><visual_scene id="Scene">
    <node id="etkc_dash_xp_rhd" name="etkc_dash_xp_rhd">
      <matrix>1 0 0 9 0 1 0 9 0 0 1 9 0 0 0 1</matrix>
      <instance_geometry url="#geom_whole"/>
    </node>
  </visual_scene></library_visual_scenes>
</COLLADA>
""",
                encoding="utf-8",
            )
            source.write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema">
  <library_effects><effect id="dash_mat-effect"/></library_effects>
  <library_materials>
    <material id="dash_mat-material" name="dash_mat">
      <instance_effect url="#dash_mat-effect"/>
    </material>
    <material id="trim_mat-material" name="trim_mat"/>
  </library_materials>
  <library_geometries>
    <geometry id="dash__beamxp_mirrored_carrier"><mesh>
      <source id="dash__beamxp_mirrored_carrier-positions">
        <float_array id="dash__beamxp_mirrored_carrier-positions-array" count="9">1.0 1.0 1.0 2.0 2.0 2.0 3.0 3.0 3.0</float_array>
        <technique_common><accessor source="#dash__beamxp_mirrored_carrier-positions-array" count="3" stride="3">
          <param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/>
        </accessor></technique_common>
      </source>
      <source id="dash__beamxp_mirrored_carrier-normals">
        <float_array id="dash__beamxp_mirrored_carrier-normals-array" count="9">0 1 0 0 1 0 0 1 0</float_array>
        <technique_common><accessor source="#dash__beamxp_mirrored_carrier-normals-array" count="3" stride="3">
          <param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/>
        </accessor></technique_common>
      </source>
      <vertices id="dash__beamxp_mirrored_carrier-vertices"><input semantic="POSITION" source="#dash__beamxp_mirrored_carrier-positions"/></vertices>
      <triangles material="dash_mat-material" count="1">
        <input semantic="VERTEX" offset="0" source="#dash__beamxp_mirrored_carrier-vertices"/>
        <input semantic="NORMAL" offset="1" source="#dash__beamxp_mirrored_carrier-normals"/>
        <p>0 0 1 1 2 2</p>
      </triangles>
    </mesh></geometry>
    <geometry id="dash__beamxp_rigid_001"><mesh>
      <source id="dash__beamxp_rigid_001-positions">
        <float_array id="dash__beamxp_rigid_001-positions-array" count="9">10.0 10.0 10.0 11.0 11.0 11.0 12.0 12.0 12.0</float_array>
        <technique_common><accessor source="#dash__beamxp_rigid_001-positions-array" count="3" stride="3">
          <param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/>
        </accessor></technique_common>
      </source>
      <source id="dash__beamxp_rigid_001-normals">
        <float_array id="dash__beamxp_rigid_001-normals-array" count="9">0 1 0 0 1 0 0 1 0</float_array>
        <technique_common><accessor source="#dash__beamxp_rigid_001-normals-array" count="3" stride="3">
          <param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/>
        </accessor></technique_common>
      </source>
      <vertices id="dash__beamxp_rigid_001-vertices"><input semantic="POSITION" source="#dash__beamxp_rigid_001-positions"/></vertices>
      <triangles material="trim_mat-material" count="1">
        <input semantic="VERTEX" offset="0" source="#dash__beamxp_rigid_001-vertices"/>
        <input semantic="NORMAL" offset="1" source="#dash__beamxp_rigid_001-normals"/>
        <p>0 0 1 1 2 2</p>
      </triangles>
    </mesh></geometry>
  </library_geometries>
  <library_visual_scenes>
    <visual_scene id="Scene">
      <node id="dash__beamxp_mirrored_carrier" name="dash__beamxp_mirrored_carrier">
        <matrix>1 0 0 2 0 1 0 3 0 0 1 4 0 0 0 1</matrix>
        <instance_geometry url="#dash__beamxp_mirrored_carrier">
          <bind_material><technique_common>
            <instance_material symbol="dash_mat-material" target="#dash_mat-material"/>
          </technique_common></bind_material>
        </instance_geometry>
      </node>
      <node id="dash__beamxp_rigid_001" name="dash__beamxp_rigid_001">
        <matrix>1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</matrix>
        <instance_geometry url="#dash__beamxp_rigid_001">
          <bind_material><technique_common>
            <instance_material symbol="trim_mat-material" target="#trim_mat-material"/>
          </technique_common></bind_material>
        </instance_geometry>
      </node>
    </visual_scene>
  </library_visual_scenes>
</COLLADA>
""",
                encoding="utf-8",
            )

            grouped, piece_materials = build_pipeline._group_texture_correction_dae(
                target,
                source,
                {"dash__beamxp_mirrored_carrier", "dash__beamxp_rigid_001"},
                {"dash_mat": "dash_beamxp_tc"},
                ["etkc_dash_xp_rhd"],
            )

            self.assertEqual(grouped, ["etkc_dash_xp_rhd"])
            self.assertEqual(
                piece_materials,
                {
                    "dash__beamxp_mirrored_carrier": {"dash_beamxp_tc"},
                    "dash__beamxp_rigid_001": {"trim_mat"},
                },
            )
            root = ET.parse(target).getroot()
            ns = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
            nodes = root.findall(".//c:visual_scene/c:node", ns)
            self.assertEqual([node.get("id") for node in nodes], ["etkc_dash_xp_rhd"])
            instances = nodes[0].findall("c:instance_geometry", ns)
            self.assertEqual(len(instances), 1)
            self.assertEqual(
                nodes[0].find("c:matrix", ns).text, "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"
            )
            self.assertIsNone(root.find(".//c:geometry[@id='geom_whole']", ns))

            merged = root.find(
                ".//c:geometry[@id='" + instances[0].get("url").lstrip("#") + "']", ns
            )
            positions = merged.find(".//c:vertices/c:input", ns).get("source").lstrip("#")
            array = merged.find(".//c:source[@id='" + positions + "']/c:float_array", ns)
            # The carrier's own placement is baked in; the rigid piece sat at
            # the origin and keeps its coordinates.
            self.assertEqual(
                array.text, "3 4 5 4 5 6 5 6 7 10 10 10 11 11 11 12 12 12"
            )
            self.assertEqual(array.get("count"), "18")
            triangles = merged.findall("c:mesh/c:triangles", ns)
            self.assertEqual(
                [element.get("material") for element in triangles],
                ["dash_beamxp_tc", "trim_mat-material"],
            )
            # The second piece's vertices moved to the back of the shared pool.
            self.assertEqual(triangles[0].find("c:p", ns).text, "0 0 1 1 2 2")
            self.assertEqual(triangles[1].find("c:p", ns).text, "3 0 4 1 5 2")
            self.assertEqual(
                {
                    binding.get("symbol")
                    for binding in nodes[0].findall(".//c:instance_material", ns)
                },
                {"dash_beamxp_tc", "trim_mat-material"},
            )
            self.assertIsNotNone(
                root.find(".//c:material[@id='dash_beamxp_tc-material']", ns)
            )
            self.assertIsNotNone(root.find(".//c:effect[@id='dash_mat-effect']", ns))

    def test_texture_correction_material_preserves_source_stages(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            job = tmp / "job"
            target = tmp / "vehicles/scintilla"
            job.mkdir()
            for name in (
                "dash_b.color_rhd.dds",
                "dash_nm.normal_rhd.dds",
                "dash_r.data_rhd.dds",
                "dash_m.data_rhd.dds",
                "dash_ao.data_rhd.dds",
                "dash_leather_o.data_rhd.dds",
                "dash_carpet_o.data_rhd.dds",
            ):
                _placeholder_dds(job / name)
            (job / "rhd_materials.json").write_text(
                json.dumps(
                    {
                        "materials": [
                            {
                                "aliases": ["dash_mat", "dash_mat-material"],
                                "maps": {
                                    "baseColorMap": "dash_b.color_rhd.dds",
                                    "normalMap": "dash_nm.normal_rhd.dds",
                                },
                                "outputMaps": [
                                    {"stageKey": "baseColorMap", "member": "vehicles/car/dash_b.color.dds", "dds": "dash_b.color_rhd.dds"},
                                    {"stageKey": "normalMap", "member": "vehicles/car/dash_nm.normal.dds", "dds": "dash_nm.normal_rhd.dds"},
                                    {"stageKey": "roughnessMap", "member": "vehicles/car/dash_r.data.dds", "dds": "dash_r.data_rhd.dds"},
                                    {"stageKey": "metallicMap", "member": "vehicles/car/dash_m.data.dds", "dds": "dash_m.data_rhd.dds"},
                                    {"stageKey": "ambientOcclusionMap", "member": "vehicles/car/dash_ao.data.dds", "dds": "dash_ao.data_rhd.dds"},
                                    {"stageKey": "opacityMap", "member": "vehicles/car/dash_leather_o.data.dds", "dds": "dash_leather_o.data_rhd.dds"},
                                    {"stageKey": "opacityMap", "member": "vehicles/car/dash_carpet_o.data.dds", "dds": "dash_carpet_o.data_rhd.dds"},
                                ],
                                "sourceMaterials": [
                                    {
                                        "key": "dash_mat",
                                        "aliases": ["dash_mat"],
                                        "materialsMember": "vehicles/car/main.materials.json",
                                        "material": {
                                            "name": "dash_mat",
                                            "mapTo": "dash_mat",
                                            "class": "Material",
                                            "Stages": [
                                                {
                                                    "baseColorMap": "/vehicles/car/dash_b.color.png",
                                                    "normalMap": "/vehicles/car/dash_nm.normal.png",
                                                    "roughnessMap": "/vehicles/car/dash_r.data.png",
                                                    "metallicMap": "/vehicles/car/dash_m.data.png",
                                                    "ambientOcclusionMap": "/vehicles/car/dash_ao.data.png",
                                                    "metallicFactor": 1,
                                                },
                                                {
                                                    "baseColorFactor": [0.1, 0.1, 0.1, 1],
                                                    "normalMap": "/vehicles/car/dash_nm.normal.png",
                                                    "roughnessMap": "/vehicles/car/dash_r.data.png",
                                                    "ambientOcclusionMap": "/vehicles/car/dash_ao.data.png",
                                                    "opacityMap": "/vehicles/car/dash_leather_o.data.png",
                                                    "detailNormalMap": "/vehicles/common/detail_nm.normal.png",
                                                },
                                                {
                                                    "normalMap": "/vehicles/car/dash_nm.normal.png",
                                                    "opacityMap": "/vehicles/car/dash_carpet_o.data.png",
                                                },
                                            ],
                                            "activeLayers": 3,
                                            "dynamicCubemap": True,
                                            "version": 1.5,
                                        },
                                    }
                                ],
                            }
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            mapping = build_pipeline._prepare_texture_correction_materials(job, target, tmp)

            self.assertEqual(mapping["dash_mat"], "dash_mat_beamxp_tc")
            material = json.loads(
                (target / "beamxp_texture_correction.materials.json").read_text(encoding="utf-8")
            )["dash_mat_beamxp_tc"]
            self.assertEqual(material["activeLayers"], 3)
            self.assertEqual(len(material["Stages"]), 3)
            self.assertIn("detailNormalMap", material["Stages"][1])
            self.assertIn("dash_nm.normal_rhd.dds", material["Stages"][1]["normalMap"])
            self.assertIn("dash_leather_o.data_rhd.dds", material["Stages"][1]["opacityMap"])
            self.assertIn("dash_carpet_o.data_rhd.dds", material["Stages"][2]["opacityMap"])

    def test_shared_base_texture_material_states_get_distinct_corrected_materials(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            job = tmp / "job"
            target = tmp / "vehicles/vivace"
            job.mkdir()
            for name in (
                "ardente_interior_b.color_rhd.dds",
                "ardente_interior_g.color_rhd.dds",
            ):
                _placeholder_dds(job / name)
            (job / "rhd_materials.json").write_text(
                json.dumps(
                    {
                        "materials": [
                            {
                                "aliases": [
                                    "ardente_interior",
                                    "ardente_interior-material",
                                    "ardente_interior_on",
                                ],
                                "maps": {
                                    "baseColorMap": "ardente_interior_b.color_rhd.dds",
                                    "emissiveMap": "ardente_interior_g.color_rhd.dds",
                                },
                                "outputMaps": [
                                    {
                                        "stageKey": "baseColorMap",
                                        "member": "vehicles/vivace/ardente/ardente_interior_b.color.png",
                                        "dds": "ardente_interior_b.color_rhd.dds",
                                    },
                                    {
                                        "stageKey": "emissiveMap",
                                        "member": "vehicles/vivace/ardente/ardente_interior_g.color.png",
                                        "dds": "ardente_interior_g.color_rhd.dds",
                                    },
                                ],
                                "sourceMaterials": [
                                    {
                                        "key": "ardente_interior",
                                        "aliases": ["ardente_interior"],
                                        "materialsMember": "vehicles/vivace/ardente/main.materials.json",
                                        "material": {
                                            "name": "ardente_interior",
                                            "mapTo": "ardente_interior",
                                            "class": "Material",
                                            "Stages": [
                                                {
                                                    "baseColorMap": "/vehicles/vivace/ardente/ardente_interior_b.color.png",
                                                },
                                            ],
                                        },
                                    },
                                    {
                                        "key": "ardente_interior_on",
                                        "aliases": ["ardente_interior_on"],
                                        "materialsMember": "vehicles/vivace/ardente/main.materials.json",
                                        "material": {
                                            "name": "ardente_interior_on",
                                            "mapTo": "ardente_interior_on",
                                            "class": "Material",
                                            "Stages": [
                                                {
                                                    "baseColorMap": "/vehicles/vivace/ardente/ardente_interior_b.color.png",
                                                    "emissiveMap": "/vehicles/vivace/ardente/ardente_interior_g.color.png",
                                                    "emissiveIntensityNits": 0.4,
                                                },
                                            ],
                                        },
                                    },
                                ],
                            }
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            mapping = build_pipeline._prepare_texture_correction_materials(job, target, tmp)
            materials = json.loads(
                (target / "beamxp_texture_correction.materials.json").read_text(encoding="utf-8")
            )

        self.assertEqual(mapping["ardente_interior"], "ardente_interior_beamxp_tc")
        self.assertEqual(mapping["ardente_interior_on"], "ardente_interior_on_beamxp_tc")
        self.assertIn("ardente_interior_beamxp_tc", materials)
        self.assertIn("ardente_interior_on_beamxp_tc", materials)
        self.assertNotIn("emissiveMap", materials["ardente_interior_beamxp_tc"]["Stages"][0])
        # The lit state reads the corrected emissive image. Which material's
        # name the copy carries is not part of the contract: one corrected
        # image is copied in once and every material naming it shares that
        # copy, so the prefix belongs to whichever minted it first.
        self.assertIn(
            "ardente_interior_g.color_rhd.dds",
            materials["ardente_interior_on_beamxp_tc"]["Stages"][0]["emissiveMap"],
        )

    def test_one_corrected_image_is_copied_in_once(self) -> None:
        # Every material naming a corrected image used to get its own copy of
        # it. Andronisk's V60 shipped 504.7 MB of corrected DDS holding 219.3
        # MB of distinct images: one fabric atlas copied once per skin, its
        # normal copied again beside it, and a switch base's off and on states
        # copied apart although they are one file.
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            job = tmp / "job"
            target = tmp / "vehicles/scintilla"
            job.mkdir()
            (job / "interior_b.color_rhd.dds").write_bytes(b"colour")
            (job / "interior_nm.normal_rhd.dds").write_bytes(b"normal")

            def entry(alias: str) -> dict:
                return {
                    "aliases": [alias],
                    "maps": {
                        "baseColorMap": "interior_b.color_rhd.dds",
                        "normalMap": "interior_nm.normal_rhd.dds",
                    },
                }

            (job / "rhd_materials.json").write_text(
                json.dumps(
                    {
                        "materials": [
                            entry("scintilla_interior"),
                            entry("scintilla_interior.skin_interior.luxe"),
                            entry("scintilla_interior.skin_interior.race"),
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            build_pipeline._prepare_texture_correction_materials(job, target, tmp)
            copied = sorted(path.name for path in target.glob("*.dds"))
            materials = json.loads(
                (target / "beamxp_texture_correction.materials.json").read_text(
                    encoding="utf-8"
                )
            )

        # Two distinct images in, two files out, however many materials name them.
        self.assertEqual(len(copied), 2)
        self.assertEqual(len(materials), 3)
        colour = {
            body["Stages"][0]["baseColorMap"] for body in materials.values()
        }
        normal = {body["Stages"][0]["normalMap"] for body in materials.values()}
        self.assertEqual(len(colour), 1)
        self.assertEqual(len(normal), 1)
        self.assertNotEqual(colour, normal)

    def test_switch_base_alias_maps_to_corrected_off_state_material(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            job = tmp / "job"
            target = tmp / "vehicles/lc500"
            job.mkdir()
            _placeholder_dds(job / "LEX_LC5_rhd.dds")
            (job / "rhd_materials.json").write_text(
                json.dumps(
                    {
                        "materials": [
                            {
                                "aliases": [
                                    "lc500_screen_off",
                                    "lc500_screen_off_off",
                                    "lc500_screen_off_on",
                                ],
                                "maps": {"baseColorMap": "LEX_LC5_rhd.dds"},
                                "outputMaps": [
                                    {
                                        "stageKey": "baseColorMap",
                                        "member": "vehicles/lc500/LEX_LC5.dds",
                                        "dds": "LEX_LC5_rhd.dds",
                                    }
                                ],
                                "sourceMaterials": [
                                    {
                                        "key": "lc500_screen_off_off",
                                        "aliases": ["lc500_screen_off_off"],
                                        "materialsMember": "vehicles/lc500/main.materials.json",
                                        "material": {
                                            "name": "lc500_screen_off_off",
                                            "mapTo": "lc500_screen_off_off",
                                            "class": "Material",
                                            "Stages": [
                                                {
                                                    "baseColorMap": "/vehicles/lc500/LEX_LC5.dds",
                                                },
                                            ],
                                        },
                                    },
                                    {
                                        "key": "lc500_screen_off_on",
                                        "aliases": ["lc500_screen_off_on"],
                                        "materialsMember": "vehicles/lc500/main.materials.json",
                                        "material": {
                                            "name": "lc500_screen_off_on",
                                            "mapTo": "lc500_screen_off_on",
                                            "class": "Material",
                                            "Stages": [
                                                {
                                                    "baseColorMap": "/vehicles/lc500/LEX_LC5.dds",
                                                    "emissiveMap": "/vehicles/lc500/LEX_LC5.dds",
                                                },
                                            ],
                                        },
                                    },
                                ],
                            }
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            mapping = build_pipeline._prepare_texture_correction_materials(job, target, tmp)

        self.assertEqual(mapping["lc500_screen_off"], "lc500_screen_off_off_beamxp_tc")
        self.assertEqual(mapping["lc500_screen_off_off"], "lc500_screen_off_off_beamxp_tc")
        self.assertEqual(mapping["lc500_screen_off_on"], "lc500_screen_off_on_beamxp_tc")

    def test_runtime_screen_on_state_does_not_claim_switch_base_alias(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            job = tmp / "job"
            target = tmp / "vehicles/lc500"
            job.mkdir()
            _placeholder_dds(job / "lc500_bootscreen_rhd.dds")
            (job / "rhd_materials.json").write_text(
                json.dumps(
                    {
                        "materials": [
                            {
                                "aliases": [
                                    "lc500_centralscreen",
                                    "lc500_centralscreen_on",
                                ],
                                "maps": {"baseColorMap": "lc500_bootscreen_rhd.dds"},
                                "outputMaps": [
                                    {
                                        "stageKey": "baseColorMap",
                                        "member": "vehicles/lc500/textures/lc500_bootscreen.png",
                                        "dds": "lc500_bootscreen_rhd.dds",
                                    }
                                ],
                                "sourceMaterials": [
                                    {
                                        "key": "lc500_centralscreen_on",
                                        "aliases": ["lc500_centralscreen_on"],
                                        "materialsMember": "vehicles/lc500/main.materials.json",
                                        "material": {
                                            "name": "lc500_centralscreen_on",
                                            "mapTo": "lc500_centralscreen_on",
                                            "class": "Material",
                                            "Stages": [
                                                {
                                                    "baseColorMap": "/vehicles/lc500/textures/lc500_bootscreen.png",
                                                },
                                            ],
                                        },
                                    },
                                ],
                            }
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            mapping = build_pipeline._prepare_texture_correction_materials(job, target, tmp)

        self.assertNotIn("lc500_centralscreen", mapping)
        self.assertEqual(
            mapping["lc500_centralscreen_on"],
            "lc500_centralscreen_on_beamxp_tc",
        )

    def test_repeated_alias_jobs_get_distinct_corrected_materials(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            target = tmp / "vehicles/scintilla"

            def write_job(name: str) -> Path:
                job = tmp / name
                job.mkdir()
                texture = f"{name}_b.color_rhd.dds"
                _placeholder_dds(job / texture)
                (job / "rhd_materials.json").write_text(
                    json.dumps(
                        {
                            "materials": [
                                {
                                    "aliases": ["dash_mat", "dash_mat-material"],
                                    "maps": {"baseColorMap": texture},
                                    "outputMaps": [
                                        {
                                            "stageKey": "baseColorMap",
                                            "member": "vehicles/car/dash_b.color.dds",
                                            "dds": texture,
                                        }
                                    ],
                                }
                            ]
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return job

            first = build_pipeline._prepare_texture_correction_materials(
                write_job("dash"), target, tmp
            )
            second = build_pipeline._prepare_texture_correction_materials(
                write_job("console"), target, tmp
            )

            self.assertEqual(first["dash_mat"], "dash_mat_beamxp_tc")
            self.assertEqual(second["dash_mat"], "dash_mat_beamxp_tc_2")
            materials = json.loads(
                (target / "beamxp_texture_correction.materials.json").read_text(encoding="utf-8")
            )
            self.assertIn("dash_b.color_rhd.dds", materials["dash_mat_beamxp_tc"]["Stages"][0]["baseColorMap"])
            self.assertIn("console_b.color_rhd.dds", materials["dash_mat_beamxp_tc_2"]["Stages"][0]["baseColorMap"])

    def test_a_mirrors_row_is_left_naming_the_mesh_it_always_named(self) -> None:
        """addMirror binds by mesh name, so a stale row reflects nothing.

        A mirror is rarely worth correcting, but its mesh can be swept along
        with the panel it shares an atlas with. While that split the mesh, the
        row had to be repointed at whichever piece kept the glass -- guesswork
        that gave up and shipped a dead reflection whenever no single piece
        painted with a mirror material. Grouping leaves the mesh named as it
        was, so there is nothing to repoint.
        """
        with tempfile.TemporaryDirectory() as raw:
            vehicle_dir = Path(raw) / "vehicles/scintilla"
            vehicle_dir.mkdir(parents=True)
            source = """{
"scintilla_interior_xp_rhd": {
    "flexbodies": [
        ["mesh", "[group]:"],
        ["scintilla_interior_mirror_xp_rhd", ["scintilla_interior"]],
    ],
    "mirrors": [
        ["mesh", "idRef:", "id1:", "id2:"],
        ["scintilla_interior_mirror_xp_rhd","rf1","rf1rr","f6l",{"refBaseTranslation":{"x":0.0,"y":0.0,"z":-0.09}}],
    ],
}
}"""
            (vehicle_dir / "car.jbeam").write_text(source, encoding="utf-8")

            build_pipeline._patch_texture_correction_jbeams(
                vehicle_dir,
                {"scintilla_interior_mirror_xp_rhd"},
            )
            updated = (vehicle_dir / "car.jbeam").read_text(encoding="utf-8")

        self.assertEqual(updated, source)

    @staticmethod
    def _skin_manifest_job(directory: Path) -> Path:
        """One corrected material and two skins over it, as scintilla ships them.

        ``scintilla_interior`` is the material the dashboard binds;
        ``scintilla_interior.skin_interior.luxe`` and ``.race`` are what the
        engine swaps in when the config selects that interior skin part.
        """
        job = directory / "job"
        job.mkdir(parents=True, exist_ok=True)
        for name in (
            "interior_b.color_rhd.dds",
            "interior_luxe_b.color_rhd.dds",
            "interior_race_b.color_rhd.dds",
        ):
            _placeholder_dds(job / name)

        def entry(key: str, source: str, output: str) -> dict:
            return {
                "aliases": ["scintilla_interior", key],
                "maps": {"baseColorMap": output},
                "outputMaps": [
                    {
                        "stageKey": "baseColorMap",
                        "member": f"vehicles/scintilla/{source}",
                        "dds": output,
                    }
                ],
                "sourceMaterials": [
                    {
                        "key": key,
                        "aliases": [key],
                        "materialsMember": "vehicles/scintilla/main.materials.json",
                        "material": {
                            "name": key,
                            "mapTo": key,
                            "class": "Material",
                            "Stages": [
                                {"baseColorMap": f"/vehicles/scintilla/{source}"}
                            ],
                            "version": 1.5,
                        },
                    }
                ],
            }

        (job / "rhd_materials.json").write_text(
            json.dumps(
                {
                    "materials": [
                        # Deliberately skin-first: a manifest lists variants in
                        # whatever order the exporter found them, and the base
                        # still has to be named before anything named after it.
                        entry(
                            "scintilla_interior.skin_interior.luxe",
                            "interior_luxe_b.color.dds",
                            "interior_luxe_b.color_rhd.dds",
                        ),
                        entry(
                            "scintilla_interior",
                            "interior_b.color.dds",
                            "interior_b.color_rhd.dds",
                        ),
                        entry(
                            "scintilla_interior.skin_interior.race",
                            "interior_race_b.color.dds",
                            "interior_race_b.color_rhd.dds",
                        ),
                    ]
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return job

    def test_a_corrected_skin_keeps_its_skin_suffix_on_the_end(self) -> None:
        """The engine composes <bound material>.<slotType>.<skinName>.

        Suffixing the whole alias instead -- ``...luxe_beamxp_tc`` -- names a
        material no config can ask for, because the mesh binds
        ``scintilla_interior_beamxp_tc`` and the engine appends the skin to
        that. Ten of the scintilla's sixteen trims wore the base interior.
        """
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            target = tmp / "vehicles/scintilla"
            target.mkdir(parents=True)
            mapping = build_pipeline._prepare_texture_correction_materials(
                self._skin_manifest_job(tmp), target, tmp
            )
            materials = json.loads(
                (target / "beamxp_texture_correction.materials.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(mapping["scintilla_interior"], "scintilla_interior_beamxp_tc")
        for skin in ("luxe", "race"):
            name = f"scintilla_interior_beamxp_tc.skin_interior.{skin}"
            self.assertEqual(mapping[f"scintilla_interior.skin_interior.{skin}"], name)
            self.assertIn(name, materials)
            # named for itself, or the engine cannot bind what it swapped in
            self.assertEqual(materials[name]["mapTo"], name)
            self.assertIn(
                f"interior_{skin}_b.color_rhd.dds",
                materials[name]["Stages"][0]["baseColorMap"],
            )
        self.assertNotIn("scintilla_interior.skin_interior.luxe_beamxp_tc", materials)

    def test_a_skin_follows_the_base_it_skins_when_one_alias_corrects_twice(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            target = tmp / "vehicles/scintilla"
            target.mkdir(parents=True)
            job = self._skin_manifest_job(tmp)
            build_pipeline._prepare_texture_correction_materials(job, target, tmp)
            second = build_pipeline._prepare_texture_correction_materials(job, target, tmp)

        # The second layout's skins belong to the second layout's base, not to
        # the first one's -- otherwise they overwrite a skin already correct.
        self.assertEqual(second["scintilla_interior"], "scintilla_interior_beamxp_tc_2")
        self.assertEqual(
            second["scintilla_interior.skin_interior.luxe"],
            "scintilla_interior_beamxp_tc_2.skin_interior.luxe",
        )

    def test_prune_keeps_a_skin_of_a_bound_material_and_drops_one_of_an_unbound(self) -> None:
        """A skin is reachable without ever being named by a mesh.

        Nothing binds ``..._beamxp_tc.skin_interior.luxe``; the engine composes
        it at runtime. Pruning on bound names alone deleted every corrected
        skin the exporter had just built -- 372 MB of it on the scintilla.
        """
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            vehicle_dir = tmp / "vehicles/scintilla"
            vehicle_dir.mkdir(parents=True)
            for name in ("base.dds", "luxe.dds", "other.dds", "other_luxe.dds"):
                _placeholder_dds(vehicle_dir / name)
            (vehicle_dir / "car.dae").write_text(
                '<x><instance_material symbol="interior_beamxp_tc" target="#a"/></x>',
                encoding="utf-8",
            )
            (vehicle_dir / "beamxp_texture_correction.materials.json").write_text(
                json.dumps(
                    {
                        "interior_beamxp_tc": {
                            "Stages": [{"baseColorMap": "/vehicles/scintilla/base.dds"}]
                        },
                        "interior_beamxp_tc.skin_interior.luxe": {
                            "Stages": [{"baseColorMap": "/vehicles/scintilla/luxe.dds"}]
                        },
                        "other_beamxp_tc": {
                            "Stages": [{"baseColorMap": "/vehicles/scintilla/other.dds"}]
                        },
                        "other_beamxp_tc.skin_interior.luxe": {
                            "Stages": [{"baseColorMap": "/vehicles/scintilla/other_luxe.dds"}]
                        },
                    }
                ),
                encoding="utf-8",
            )
            result = build_pipeline.prune_unused_texture_correction_assets(tmp, vehicle_dir)
            materials = json.loads(
                (vehicle_dir / "beamxp_texture_correction.materials.json").read_text(
                    encoding="utf-8"
                )
            )
            kept_skin_texture = (vehicle_dir / "luxe.dds").is_file()
            dropped_skin_texture = not (vehicle_dir / "other_luxe.dds").exists()

        self.assertEqual(
            result["removedMaterials"],
            ["other_beamxp_tc", "other_beamxp_tc.skin_interior.luxe"],
        )
        self.assertIn("interior_beamxp_tc.skin_interior.luxe", materials)
        self.assertTrue(kept_skin_texture)
        # an unbound base still takes its skins with it
        self.assertTrue(dropped_skin_texture)


class SwapMeshCorrectionOptInTests(unittest.TestCase):
    """Texture Fix governs whether a Swap Mesh row is corrected at all.

    Swap Mesh hands the object the opposite side's authored mesh, whose texture
    is already the one that side wants, so a ticked neighbour sharing its atlas
    must not drag it into a correction. The LC500 is the case: two interiors are
    ticked, both doors are Swap Mesh and unticked, and all four used to come out
    corrected -- the doors force-mirrored, in a second pass of their own.
    """

    LC500_MODES = {
        "lc500_interior": "mirror",
        "lc500_interior_facelift": "mirror",
        "lc500_door_L": "mirrorStructural",
        "lc500_door_R": "mirrorStructural",
    }
    LC500_SOURCES = {"lc500_door_L": "lc500_door_R", "lc500_door_R": "lc500_door_L"}

    def test_an_unticked_swap_mesh_is_not_dragged_in_by_a_ticked_neighbour(self) -> None:
        targets, forced = core.texture_correction_atlas_dependencies(
            self.LC500_MODES,
            self.LC500_SOURCES,
            {"lc500_interior", "lc500_interior_facelift"},
        )
        self.assertNotIn("lc500_door_L", targets)
        self.assertNotIn("lc500_door_R", targets)
        # Nothing forced means no deferred structural_mirror pass either.
        self.assertEqual(forced, set())

    def test_a_ticked_swap_mesh_still_force_mirrors_its_source(self) -> None:
        targets, forced = core.texture_correction_atlas_dependencies(
            self.LC500_MODES,
            self.LC500_SOURCES,
            {"lc500_interior", "lc500_door_L"},
        )
        # The correction is made on the donor the swap reskins from.
        self.assertEqual(targets["lc500_door_R"], {"lc500_door_L"})
        self.assertEqual(forced, {"lc500_door_R"})

    def test_mirror_still_follows_a_shared_atlas_without_being_ticked(self) -> None:
        # Mirror has no authored counterpart to fall back on, so it has to
        # follow the atlas whether or not its own column was ticked.
        targets, forced = core.texture_correction_atlas_dependencies(
            {"dashboard": "mirror", "intmirror": "mirror"},
            {},
            {"dashboard"},
        )
        self.assertEqual(targets["intmirror"], {"intmirror"})
        self.assertEqual(forced, set())

    def test_a_mode_that_moves_nothing_is_never_a_dependency(self) -> None:
        targets, forced = core.texture_correction_atlas_dependencies(
            {"sill": "translate", "badge": "replaceSource", "boot": "skip"},
            {},
            {"sill", "badge", "boot"},
        )
        self.assertEqual(targets, {})
        self.assertEqual(forced, set())


class WholeMeshMirrorCorrectionTests(unittest.TestCase):
    """A Mirror prop ships a whole-mesh reflection, so nothing about it is rigid.

    ``vehicleObj:addProp`` (lua/common/jbeam/sections/meshs.lua) spawns exactly
    the mesh the row names and places it from that mesh's own frame, so a prop
    keeps its whole-mesh copy rather than the grouped one a flexbody row is
    moved onto. The Andronisk's signal stalk is the case: the sweep called two
    of its triangles rigid, and their glyphs shipped unflipped under geometry
    the exporter reflected anyway.
    """

    def test_a_mirror_prop_forces_its_whole_domain_mirrored(self) -> None:
        forced = core.whole_mesh_mirror_correction_ids(
            {"signalstalk": "mirror", "dash": "mirror"},
            {},
            {"signalstalk"},
            {"dash"},
        )
        self.assertEqual(forced, {"signalstalk"})

    def test_a_mesh_bound_as_both_keeps_the_sweeps_answer(self) -> None:
        # Its flexbody row moves onto a corrected copy of its own, beside the
        # whole-mesh copy the prop goes on spawning, so the rigid regions still
        # have somewhere to ride.
        forced = core.whole_mesh_mirror_correction_ids(
            {"handle": "mirror"},
            {},
            {"handle"},
            {"handle"},
        )
        self.assertEqual(forced, set())

    def test_a_prop_that_only_moves_is_not_forced(self) -> None:
        forced = core.whole_mesh_mirror_correction_ids(
            {"stalk": "translate", "lever": "mirrorPosition", "knob": "skip"},
            {},
            {"stalk", "lever", "knob"},
            set(),
        )
        self.assertEqual(forced, set())

    def test_the_force_names_the_mesh_the_correction_is_made_on(self) -> None:
        forced = core.whole_mesh_mirror_correction_ids(
            {"stalk_L": "mirror"},
            {"stalk_L": "stalk_R"},
            {"stalk_L"},
            set(),
        )
        self.assertEqual(forced, {"stalk_R"})


class BakedPropCopyTests(unittest.TestCase):
    def test_copies_are_keyed_by_the_mesh_and_hand_they_were_minted_for(self) -> None:
        specs = [
            core.BakedMeshSpec(
                configured_mesh="signalstalk",
                source_mesh="signalstalk",
                output_mesh="signalstalk_xp_rhd__cross_250__dash__0000",
                target_hand=core.HAND_RHD,
                mode="mirror",
                placement_matrix=[],
                bake_transform_into_dae=False,
                is_prop=True,
            ),
            core.BakedMeshSpec(
                configured_mesh="signalstalk",
                source_mesh="signalstalk",
                output_mesh="signalstalk_xp_rhd__police__dash__0001",
                target_hand=core.HAND_RHD,
                mode="mirror",
                placement_matrix=[],
                bake_transform_into_dae=False,
                is_prop=True,
            ),
        ]
        self.assertEqual(
            build_pipeline.baked_mesh_copies_by_target(specs),
            {
                ("signalstalk", core.HAND_RHD): [
                    "signalstalk_xp_rhd__cross_250__dash__0000",
                    "signalstalk_xp_rhd__police__dash__0001",
                ]
            },
        )


class PartScopedCorrectedMaterialTests(unittest.TestCase):
    """One atlas corrected twice must bind a different material per mesh."""

    def _entry(self, texture: str, part_keys: list[str] | None) -> dict:
        entry = {
            "aliases": ["lc500_screens", "lc500_screens-material"],
            "maps": {"baseColorMap": texture},
            "outputMaps": [
                {
                    "stageKey": "baseColorMap",
                    "member": "vehicles/lc500/textures/screen.dds",
                    "dds": texture,
                }
            ],
        }
        if part_keys is not None:
            entry["partKeys"] = part_keys
        return entry

    def _prepare(self, entries: list[dict]):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            job = tmp / "job"
            job.mkdir()
            target = tmp / "vehicles/lc500"
            for entry in entries:
                _placeholder_dds(job / entry["maps"]["baseColorMap"])
            (job / "rhd_materials.json").write_text(
                json.dumps({"materials": entries}) + "\n", encoding="utf-8"
            )
            return build_pipeline._prepare_texture_correction_materials(job, target, tmp)

    def test_an_unscoped_manifest_still_reads_as_one_flat_map(self) -> None:
        materials = self._prepare([self._entry("screen_rhd.dds", None)])

        self.assertEqual(materials["lc500_screens"], "lc500_screens_beamxp_tc")
        # No scope means the answer is the same for every mesh, as before.
        self.assertEqual(materials.for_part("lc500_interior"), dict(materials))
        self.assertEqual(materials.for_part("anything_at_all"), dict(materials))

    def test_each_mesh_gets_the_correction_made_for_it(self) -> None:
        materials = self._prepare(
            [
                self._entry("screen_rhd.dds", ["lc500_interior"]),
                self._entry("screen_2_rhd.dds", ["lc500_interior_facelift"]),
            ]
        )

        self.assertEqual(
            materials.for_part("lc500_interior")["lc500_screens"],
            "lc500_screens_beamxp_tc",
        )
        self.assertEqual(
            materials.for_part("lc500_interior_facelift")["lc500_screens"],
            "lc500_screens_beamxp_tc_2",
        )

    def test_a_mesh_no_correction_names_keeps_the_shipped_texture(self) -> None:
        # The LC500's facelift: its own domain is entirely rigid, so nothing
        # was corrected for it. Retargeting it off the flat map would hand it
        # the base interior's flipped atlas instead of leaving it alone.
        materials = self._prepare([self._entry("screen_rhd.dds", ["lc500_interior"])])

        self.assertEqual(
            materials.for_part("lc500_interior")["lc500_screens"],
            "lc500_screens_beamxp_tc",
        )
        self.assertEqual(materials.for_part("lc500_interior_facelift"), {})

    def test_unscoped_aliases_reach_every_mesh_alongside_scoped_ones(self) -> None:
        trim = {
            "aliases": ["lc500_trim"],
            "maps": {"baseColorMap": "trim_rhd.dds"},
            "outputMaps": [
                {
                    "stageKey": "baseColorMap",
                    "member": "vehicles/lc500/textures/trim.dds",
                    "dds": "trim_rhd.dds",
                }
            ],
        }
        materials = self._prepare(
            [self._entry("screen_rhd.dds", ["lc500_interior"]), trim]
        )

        facelift = materials.for_part("lc500_interior_facelift")
        self.assertEqual(facelift, {"lc500_trim": "lc500_trim_beamxp_tc"})
        self.assertEqual(
            materials.for_part("lc500_interior"),
            {
                "lc500_trim": "lc500_trim_beamxp_tc",
                "lc500_screens": "lc500_screens_beamxp_tc",
            },
        )


class SharedCorrectedTextureTests(unittest.TestCase):
    """Staging folds repeats of one image, but only within the stage reading it."""

    def _stage(self, entry, files):
        tmp = Path(self._tmp)
        job = tmp / "job"
        job.mkdir(exist_ok=True)
        target = tmp / "vehicles/car"
        for name, payload in files.items():
            (job / name).write_bytes(payload)
        return build_pipeline._entry_corrected_texture_outputs(
            job, target, tmp, "mat", entry, self._shared
        )

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self._tmp = self._dir.name
        self._shared = {}
        self.addCleanup(self._dir.cleanup)

    def test_one_stage_folds_two_identical_sources_into_one_file(self):
        """etk800 ships its base and white normals as the same bytes."""
        _by_source, by_stage = self._stage(
            {
                "maps": {
                    "normalMap": "base_nm.normal_rhd.dds",
                    "detailNormalMap": "white_nm.normal_rhd.dds",
                }
            },
            {
                "base_nm.normal_rhd.dds": b"identical normal",
                "white_nm.normal_rhd.dds": b"identical normal",
            },
        )
        # Different stages, so both are staged even though the bytes match.
        self.assertNotEqual(by_stage["normalMap"], by_stage["detailNormalMap"])

        _by_source, second = self._stage(
            {"maps": {"normalMap": "other_nm.normal_rhd.dds"}},
            {"other_nm.normal_rhd.dds": b"identical normal"},
        )
        self.assertEqual(
            second["normalMap"],
            by_stage["normalMap"],
            "a second material's identical normal was staged again",
        )
        staged = sorted(p.name for p in (Path(self._tmp) / "vehicles/car").iterdir())
        self.assertEqual(len(staged), 2, staged)

    def test_two_stages_never_share_a_file_however_alike(self):
        """BeamNG spells roughness and opacity alike; the stage tells them apart."""
        _by_source, by_stage = self._stage(
            {
                "maps": {
                    "roughnessMap": "dash_r.data_rhd.dds",
                    "opacityMap": "dash_leather_o.data_rhd.dds",
                }
            },
            {
                "dash_r.data_rhd.dds": b"same bytes",
                "dash_leather_o.data_rhd.dds": b"same bytes",
            },
        )
        self.assertIn("dash_r.data_rhd.dds", by_stage["roughnessMap"])
        self.assertIn("dash_leather_o.data_rhd.dds", by_stage["opacityMap"])

    def test_maps_that_differ_are_each_staged(self):
        _by_source, by_stage = self._stage(
            {
                "maps": {
                    "baseColorMap": "a_b.color_rhd.dds",
                    "normalMap": "a_nm.normal_rhd.dds",
                }
            },
            {
                "a_b.color_rhd.dds": b"colour",
                "a_nm.normal_rhd.dds": b"normal",
            },
        )
        self.assertNotEqual(by_stage["baseColorMap"], by_stage["normalMap"])
        staged = sorted(p.name for p in (Path(self._tmp) / "vehicles/car").iterdir())
        self.assertEqual(len(staged), 2, staged)

    def test_one_output_file_named_twice_is_still_staged_once(self):
        """The original saving: several materials naming one corrected file."""
        _by_source, first = self._stage(
            {"maps": {"baseColorMap": "shared_b.color_rhd.dds"}},
            {"shared_b.color_rhd.dds": b"colour"},
        )
        _by_source, second = self._stage(
            {"maps": {"baseColorMap": "shared_b.color_rhd.dds"}}, {}
        )
        self.assertEqual(first["baseColorMap"], second["baseColorMap"])


if __name__ == "__main__":
    unittest.main()


class TextureQualityTierTests(unittest.TestCase):
    """The build's encoder tier is a user setting; the family is the image's."""

    def test_the_tier_picks_the_alpha_family_from_the_image(self) -> None:
        from mesh_segmentation_transform.mirror_texture_for_rhd import (
            resolve_bc7_profile,
        )

        for tier in ("basic", "fast", "veryfast"):
            self.assertEqual(resolve_bc7_profile(tier, False), tier)
            self.assertEqual(resolve_bc7_profile(tier, True), f"alpha_{tier}")

    def test_a_saved_alpha_profile_still_names_its_tier(self) -> None:
        from mesh_segmentation_transform.mirror_texture_for_rhd import (
            resolve_bc7_profile,
        )

        # Projects saved before the tier split stored the full profile name.
        self.assertEqual(resolve_bc7_profile("alpha_basic", False), "basic")
        self.assertEqual(resolve_bc7_profile("alpha_basic", True), "alpha_basic")

    def test_the_conversion_setting_is_clamped_to_offered_tiers(self) -> None:
        self.assertEqual(core.texture_quality_setting({}), core.DEFAULT_BC7_QUALITY)
        self.assertEqual(
            core.texture_quality_setting({"textureQuality": "fast"}), "fast"
        )
        self.assertEqual(
            core.texture_quality_setting({"textureQuality": "alpha_veryfast"}),
            "veryfast",
        )
        # ultrafast and slow are not offered; anything unknown falls back.
        for rejected in ("ultrafast", "slow", "", None, 7):
            self.assertEqual(
                core.texture_quality_setting({"textureQuality": rejected}),
                core.DEFAULT_BC7_QUALITY,
            )

    def test_every_offered_tier_has_a_label(self) -> None:
        self.assertEqual(
            sorted(core.TEXTURE_QUALITY_LABELS), sorted(core.BC7_QUALITY_TIERS)
        )
