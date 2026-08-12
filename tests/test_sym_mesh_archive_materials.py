from __future__ import annotations

import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET

import numpy as np

from mesh_segmentation_transform.beamxp_transform_sym_mesh_POC import (
    ArchiveMaterialPreviewLayer,
    ArchiveMaterialRecord,
    ArchiveTextureBinding,
    DaePart,
    GeometryInstance,
    LoadedDae,
    VehicleArchive,
    _blend_archive_preview_texture,
    archive_texture_choices_for_part,
    extract_archive_member,
    scan_vehicle_archive,
)


class SymMeshArchiveMaterialTests(unittest.TestCase):
    def test_runtime_only_material_aliases_are_indexed(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "runtime.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("vehicles/car/car.dae", "<COLLADA/>")
                archive.writestr(
                    "vehicles/car/main.materials.json",
                    """{
                      "car_screen_live": {
                        "name": "car_screen_live",
                        "mapTo": "car_screen_live",
                        "Stages": [{"colorMap": "@car_screen_canvas"}]
                      }
                    }""",
                )

            scanned = scan_vehicle_archive(archive_path, root / "scan")

        self.assertEqual(
            scanned.runtime_material_aliases,
            ("car_screen_canvas", "car_screen_live"),
        )
        self.assertEqual(scanned.materials, ())

    def test_materials_and_textures_resolve_from_dependency_archives(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_zip = root / "source.zip"
            common_zip = root / "common.zip"
            with zipfile.ZipFile(source_zip, "w") as archive:
                archive.writestr(
                    "vehicles/mod/mod.dae",
                    """<COLLADA>
                      <library_materials>
                        <material id="Generic_racing_interior-material" name="Generic_racing_interior"/>
                      </library_materials>
                      <library_visual_scenes>
                        <visual_scene>
                          <node id="mod_switches" name="mod_switches">
                            <instance_geometry url="#Mesh_001-mesh">
                              <bind_material>
                                <technique_common>
                                  <instance_material
                                    symbol="Generic_racing_interior-material"
                                    target="#Generic_racing_interior-material"/>
                                </technique_common>
                              </bind_material>
                            </instance_geometry>
                          </node>
                        </visual_scene>
                      </library_visual_scenes>
                    </COLLADA>""",
                )
            with zipfile.ZipFile(common_zip, "w") as archive:
                archive.writestr(
                    "vehicles/common/racinginterior/main.materials.json",
                    """{
                      "Generic_racing_interior": {
                        "name": "Generic_racing_interior",
                        "mapTo": "Generic_racing_interior",
                        "Stages": [
                          {
                            "baseColorMap": "/vehicles/common/racinginterior/generic_racing_interior_b.color.png"
                          }
                        ]
                      }
                    }""",
                )
                archive.writestr(
                    "vehicles/common/racinginterior/generic_racing_interior_b.color.dds",
                    b"dds bytes",
                )

            tree = ET.ElementTree(
                ET.fromstring(
                    zipfile.ZipFile(source_zip).read("vehicles/mod/mod.dae")
                )
            )
            loaded = LoadedDae(
                path=Path("vehicles/mod/mod.dae"),
                tree=tree,
                namespace="",
                unit_scale=1.0,
                parts=[],
                geometries={},
            )
            part = DaePart(
                key="mod_switches",
                label="mod_switches",
                node_id="mod_switches",
                node_name="mod_switches",
                matrix=np.eye(4),
                instances=(GeometryInstance("Mesh_001-mesh"),),
            )
            archive = scan_vehicle_archive(
                source_zip,
                root / "workspace",
                asset_archives=[common_zip],
            )

            choices = archive_texture_choices_for_part(archive, loaded, part)

            self.assertEqual(len(choices), 1)
            self.assertEqual(choices[0].material_key, "Generic_racing_interior")
            self.assertEqual(
                choices[0].texture_member,
                "vehicles/common/racinginterior/generic_racing_interior_b.color.dds",
            )
            extracted = extract_archive_member(archive, choices[0].texture_member)
            self.assertEqual(extracted.read_bytes(), b"dds bytes")

    def test_skin_material_variants_match_base_dae_material(self) -> None:
        tree = ET.ElementTree(
            ET.fromstring(
                """<COLLADA>
                  <library_materials>
                    <material id="etk800_interior-material" name="etk800_interior"/>
                  </library_materials>
                  <library_geometries>
                    <geometry id="Mesh_139-mesh"/>
                  </library_geometries>
                  <library_visual_scenes>
                    <visual_scene>
                      <node id="etk800_dash" name="etk800_dash">
                        <instance_geometry url="#Mesh_139-mesh">
                          <bind_material>
                            <technique_common>
                              <instance_material
                                symbol="etk800_interior-material"
                                target="#etk800_interior-material"/>
                            </technique_common>
                          </bind_material>
                        </instance_geometry>
                      </node>
                    </visual_scene>
                  </library_visual_scenes>
                </COLLADA>"""
            )
        )
        loaded = LoadedDae(
            path=Path("vehicles/etk800/etk800.dae"),
            tree=tree,
            namespace="",
            unit_scale=1.0,
            parts=[],
            geometries={},
        )
        part = DaePart(
            key="etk800_dash",
            label="etk800_dash",
            node_id="etk800_dash",
            node_name="etk800_dash",
            matrix=np.eye(4),
            instances=(GeometryInstance("Mesh_139-mesh"),),
        )
        texture_members = (
            "vehicles/etk800/etk800_interior_black_b.color.DDS",
            "vehicles/etk800/etk800_interior_beige_b.color.DDS",
            "vehicles/etk800/etk800_interior_brown_b.color.DDS",
        )
        archive = VehicleArchive(
            path=Path("etk800.zip"),
            members=texture_members,
            member_by_lower={member.lower(): member for member in texture_members},
            member_sizes={},
            dae_members=("vehicles/etk800/etk800.dae",),
            materials=(
                ArchiveMaterialRecord(
                    key="etk800_interior",
                    name="etk800_interior",
                    map_to="etk800_interior",
                    materials_member="vehicles/etk800/main.materials.json",
                    base_colour_reference="/vehicles/etk800/etk800_interior_black_b.color.png",
                ),
                ArchiveMaterialRecord(
                    key="etk800_interior.skin_interior.beige",
                    name="etk800_interior.skin_interior.beige",
                    map_to="etk800_interior.skin_interior.beige",
                    materials_member="vehicles/etk800/skin.materials.json",
                    base_colour_reference="/vehicles/etk800/etk800_interior_beige_b.color.png",
                    preview_layers=(
                        ArchiveMaterialPreviewLayer(
                            base_colour_reference="/vehicles/etk800/etk800_interior_beige_b.color.png",
                            base_colour_factor=(0.25, 0.25, 0.25, 1.0),
                            opacity_reference="/vehicles/etk800/etk800_interior_c.data.png",
                        ),
                    ),
                ),
                ArchiveMaterialRecord(
                    key="etk800_interior.skin_interior.brown",
                    name="etk800_interior.skin_interior.brown",
                    map_to="etk800_interior.skin_interior.brown",
                    materials_member="vehicles/etk800/skin.materials.json",
                    base_colour_reference="/vehicles/etk800/etk800_interior_brown_b.color.png",
                ),
            ),
            workspace=Path("."),
        )

        choices = archive_texture_choices_for_part(archive, loaded, part)

        self.assertEqual(
            [choice.material_key for choice in choices],
            [
                "etk800_interior",
                "etk800_interior.skin_interior.beige",
                "etk800_interior.skin_interior.brown",
            ],
        )
        self.assertEqual(len(choices[1].preview_layers), 1)

    def test_runtime_state_material_variants_match_base_dae_material(self) -> None:
        tree = ET.ElementTree(
            ET.fromstring(
                """<COLLADA>
                  <library_materials>
                    <material id="lc500_centralscreen-material" name="lc500_centralscreen"/>
                  </library_materials>
                  <library_visual_scenes>
                    <visual_scene>
                      <node id="lc500_shifter" name="lc500_shifter">
                        <instance_geometry url="#Mesh_001-mesh">
                          <bind_material>
                            <technique_common>
                              <instance_material
                                symbol="lc500_centralscreen-material"
                                target="#lc500_centralscreen-material"/>
                            </technique_common>
                          </bind_material>
                        </instance_geometry>
                      </node>
                    </visual_scene>
                  </library_visual_scenes>
                </COLLADA>"""
            )
        )
        loaded = LoadedDae(
            path=Path("vehicles/lc500/lc500.dae"),
            tree=tree,
            namespace="",
            unit_scale=1.0,
            parts=[],
            geometries={},
        )
        part = DaePart(
            key="lc500_shifter",
            label="lc500_shifter",
            node_id="lc500_shifter",
            node_name="lc500_shifter",
            matrix=np.eye(4),
            instances=(GeometryInstance("Mesh_001-mesh"),),
        )
        archive = VehicleArchive(
            path=Path("lc500.zip"),
            members=("vehicles/lc500/textures/lc500_bootscreen.png",),
            member_by_lower={
                "vehicles/lc500/textures/lc500_bootscreen.png": "vehicles/lc500/textures/lc500_bootscreen.png",
            },
            member_sizes={},
            dae_members=("vehicles/lc500/lc500.dae",),
            materials=(
                ArchiveMaterialRecord(
                    key="lc500_centralscreen_on",
                    name="lc500_centralscreen_on",
                    map_to="lc500_centralscreen_on",
                    materials_member="vehicles/lc500/main.materials.json",
                    base_colour_reference="/vehicles/lc500/textures/lc500_bootscreen.png",
                ),
            ),
            workspace=Path("."),
        )

        choices = archive_texture_choices_for_part(archive, loaded, part)

        self.assertEqual([choice.material_key for choice in choices], ["lc500_centralscreen_on"])

    def test_glowmap_switch_targets_match_unrelated_lit_material_name(self) -> None:
        tree = ET.ElementTree(
            ET.fromstring(
                """<COLLADA>
                  <library_materials>
                    <material id="lc500_intlowbeam-material" name="lc500_intlowbeam"/>
                  </library_materials>
                  <library_visual_scenes>
                    <visual_scene>
                      <node id="lc500_interior" name="lc500_interior">
                        <instance_geometry url="#Mesh_001-mesh">
                          <bind_material>
                            <technique_common>
                              <instance_material
                                symbol="lc500_intlowbeam-material"
                                target="#lc500_intlowbeam-material"/>
                            </technique_common>
                          </bind_material>
                        </instance_geometry>
                      </node>
                    </visual_scene>
                  </library_visual_scenes>
                </COLLADA>"""
            )
        )
        loaded = LoadedDae(
            path=Path("vehicles/lc500/lc500.dae"),
            tree=tree,
            namespace="",
            unit_scale=1.0,
            parts=[],
            geometries={},
        )
        part = DaePart(
            key="lc500_interior",
            label="lc500_interior",
            node_id="lc500_interior",
            node_name="lc500_interior",
            matrix=np.eye(4),
            instances=(GeometryInstance("Mesh_001-mesh"),),
        )
        archive = VehicleArchive(
            path=Path("lc500.zip"),
            members=("vehicles/lc500/textures/dashlights.png",),
            member_by_lower={
                "vehicles/lc500/textures/dashlights.png": "vehicles/lc500/textures/dashlights.png",
            },
            member_sizes={},
            dae_members=("vehicles/lc500/lc500.dae",),
            materials=(
                ArchiveMaterialRecord(
                    key="lc500_dashlights",
                    name="lc500_dashlights",
                    map_to="lc500_dashlights",
                    materials_member="vehicles/lc500/main.materials.json",
                    base_colour_reference="/vehicles/lc500/textures/dashlights.png",
                ),
            ),
            workspace=Path("."),
            material_switch_targets={
                "lc500_intlowbeam": ("invis", "lc500_dashlights"),
            },
        )

        choices = archive_texture_choices_for_part(archive, loaded, part)

        self.assertEqual([choice.material_key for choice in choices], ["lc500_dashlights"])

    def test_shared_glowmap_target_keeps_every_dae_material_binding(self) -> None:
        tree = ET.ElementTree(
            ET.fromstring(
                """<COLLADA>
                  <library_materials>
                    <material id="lc500_intsignal_L-material" name="lc500_intsignal_L"/>
                    <material id="lc500_intsignal_R-material" name="lc500_intsignal_R"/>
                  </library_materials>
                  <library_visual_scenes>
                    <visual_scene>
                      <node id="lc500_interior" name="lc500_interior">
                        <instance_geometry url="#Mesh_001-mesh">
                          <bind_material>
                            <technique_common>
                              <instance_material
                                symbol="lc500_intsignal_L-material"
                                target="#lc500_intsignal_L-material"/>
                              <instance_material
                                symbol="lc500_intsignal_R-material"
                                target="#lc500_intsignal_R-material"/>
                            </technique_common>
                          </bind_material>
                        </instance_geometry>
                      </node>
                    </visual_scene>
                  </library_visual_scenes>
                </COLLADA>"""
            )
        )
        loaded = LoadedDae(
            path=Path("vehicles/lc500/lc500.dae"),
            tree=tree,
            namespace="",
            unit_scale=1.0,
            parts=[],
            geometries={},
        )
        part = DaePart(
            key="lc500_interior",
            label="lc500_interior",
            node_id="lc500_interior",
            node_name="lc500_interior",
            matrix=np.eye(4),
            instances=(GeometryInstance("Mesh_001-mesh"),),
        )
        texture_member = "vehicles/lc500/textures/dashlights.png"
        archive = VehicleArchive(
            path=Path("lc500.zip"),
            members=(texture_member,),
            member_by_lower={texture_member: texture_member},
            member_sizes={},
            dae_members=("vehicles/lc500/lc500.dae",),
            materials=(
                ArchiveMaterialRecord(
                    key="lc500_dashlights",
                    name="lc500_dashlights",
                    map_to="lc500_dashlights",
                    materials_member="vehicles/lc500/main.materials.json",
                    base_colour_reference=f"/{texture_member}",
                ),
            ),
            workspace=Path("."),
            material_switch_targets={
                "lc500_intsignal_l": ("invis", "lc500_dashlights"),
                "lc500_intsignal_r": ("invis", "lc500_dashlights"),
            },
        )

        choices = archive_texture_choices_for_part(archive, loaded, part)

        self.assertEqual(
            [choice.dae_material for choice in choices],
            ["lc500_intsignal_L", "lc500_intsignal_R"],
        )
        self.assertEqual(
            [choice.material_key for choice in choices],
            ["lc500_dashlights", "lc500_dashlights"],
        )

    def test_emissive_only_material_is_indexed_as_texture_choice(self) -> None:
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            archive_path = workspace / "lc500.zip"
            with zipfile.ZipFile(archive_path, "w") as zf:
                zf.writestr(
                    "vehicles/lc500/main.materials.json",
                    """{
                      "lc500_gauges_needle_on": {
                        "name": "lc500_gauges_needle_on",
                        "mapTo": "lc500_gauges_needle_on",
                        "class": "Material",
                        "Stages": [
                          {
                            "emissiveMap": "/vehicles/lc500/textures/lc500_gauges_needle.png",
                            "opacityMap": "/vehicles/lc500/textures/lc500_gauges_needle.png"
                          }
                        ]
                      }
                    }""",
                )
                zf.writestr("vehicles/lc500/textures/lc500_gauges_needle.png", b"png")
                zf.writestr("vehicles/lc500/lc500.dae", "<COLLADA/>")
                zf.writestr(
                    "vehicles/lc500/lc500.jbeam",
                    """{
                      "lc500": {
                        "glowMap": {
                          "lc500_gauges_needle": {
                            "simpleFunction": "running",
                            "off": "invis",
                            "on": "lc500_gauges_needle_on"
                          },
                          "lc500_centralscreen": {
                            "simpleFunction": {"ignitionLevel": 0.5},
                            "off": "lc500_screens_off",
                            "on": "lc500_centralscreen_on",
                            "on_intense": "lc500_GPS"
                          }
                        }
                      }
                    }""",
                )

            archive = scan_vehicle_archive(archive_path, workspace / "scan")

        self.assertEqual(
            [record.key for record in archive.materials],
            ["lc500_gauges_needle_on"],
        )
        self.assertEqual(
            archive.material_switch_targets["lc500_gauges_needle"],
            ("invis", "lc500_gauges_needle_on"),
        )
        self.assertEqual(
            [
                (state.state, state.material)
                for state in archive.material_switch_states["lc500_gauges_needle"]
            ],
            [("off", "invis"), ("on", "lc500_gauges_needle_on")],
        )
        self.assertEqual(
            archive.material_switch_triggers["lc500_gauges_needle"],
            ("running",),
        )
        self.assertEqual(
            archive.material_switch_triggers["lc500_centralscreen"],
            ("ignitionLevel",),
        )

    def test_blender_duplicate_material_suffix_matches_base_material_once(self) -> None:
        tree = ET.ElementTree(
            ET.fromstring(
                """<COLLADA>
                  <library_materials>
                    <material id="lc500_paint_001-material" name="lc500_paint.001"/>
                  </library_materials>
                  <library_visual_scenes>
                    <visual_scene>
                      <node id="lc500_bumper_F_001" name="lc500_bumper_F.001">
                        <instance_geometry url="#bumper_f_high_003-mesh">
                          <bind_material>
                            <technique_common>
                              <instance_material
                                symbol="lc500_paint_001-material"
                                target="#lc500_paint_001-material"/>
                            </technique_common>
                          </bind_material>
                        </instance_geometry>
                      </node>
                    </visual_scene>
                  </library_visual_scenes>
                </COLLADA>"""
            )
        )
        loaded = LoadedDae(
            path=Path("vehicles/lc500/lc500.dae"),
            tree=tree,
            namespace="",
            unit_scale=1.0,
            parts=[],
            geometries={},
        )
        part = DaePart(
            key="lc500_bumper_F_001",
            label="lc500_bumper_F.001",
            node_id="lc500_bumper_F_001",
            node_name="lc500_bumper_F.001",
            matrix=np.eye(4),
            instances=(GeometryInstance("bumper_f_high_003-mesh"),),
        )
        archive = VehicleArchive(
            path=Path("lc500.zip"),
            members=("vehicles/lc500/textures/pbr/midsize_main_c.data.DDS",),
            member_by_lower={
                "vehicles/lc500/textures/pbr/midsize_main_c.data.dds": "vehicles/lc500/textures/pbr/midsize_main_c.data.DDS",
            },
            member_sizes={},
            dae_members=("vehicles/lc500/lc500.dae",),
            materials=(
                ArchiveMaterialRecord(
                    key="lc500_paint",
                    name="lc500_paint",
                    map_to="lc500_paint",
                    materials_member="vehicles/lc500/main.materials.json",
                    base_colour_reference="/vehicles/lc500/textures/pbr/midsize_main_c.data.DDS",
                ),
            ),
            workspace=Path("."),
        )

        choices = archive_texture_choices_for_part(archive, loaded, part)

        self.assertEqual([choice.material_key for choice in choices], ["lc500_paint"])

    def test_preview_texture_layers_are_baked_for_blender(self) -> None:
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is required for layered preview baking")

        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            base = workspace / "vehicles/bolide/bolide_interior_b.color.DDS"
            mask = workspace / "vehicles/bolide/bolide_interior_c.data.DDS"
            base.parent.mkdir(parents=True)
            Image.new("RGBA", (1, 1), (200, 160, 120, 255)).save(base)
            Image.new("L", (1, 1), 255).save(mask)

            archive = VehicleArchive(
                path=Path("bolide.zip"),
                members=(
                    "vehicles/bolide/bolide_interior_b.color.DDS",
                    "vehicles/bolide/bolide_interior_c.data.DDS",
                ),
                member_by_lower={
                    "vehicles/bolide/bolide_interior_b.color.dds": "vehicles/bolide/bolide_interior_b.color.DDS",
                    "vehicles/bolide/bolide_interior_c.data.dds": "vehicles/bolide/bolide_interior_c.data.DDS",
                },
                member_sizes={
                    "vehicles/bolide/bolide_interior_b.color.DDS": base.stat().st_size,
                    "vehicles/bolide/bolide_interior_c.data.DDS": mask.stat().st_size,
                },
                dae_members=("vehicles/bolide/bolide.dae",),
                materials=(),
                workspace=workspace,
            )
            binding = ArchiveTextureBinding(
                dae_material="bolide_interior",
                material_key="bolide_interior",
                materials_member="vehicles/bolide/main.materials.json",
                texture_reference="/vehicles/bolide/bolide_interior_b.color.png",
                texture_member="vehicles/bolide/bolide_interior_b.color.DDS",
                preview_layers=(
                    ArchiveMaterialPreviewLayer(
                        base_colour_reference="/vehicles/bolide/bolide_interior_b.color.png",
                        base_colour_factor=(0.25, 0.25, 0.25, 1.0),
                        opacity_reference="/vehicles/bolide/bolide_interior_c.data.png",
                    ),
                ),
            )

            output = _blend_archive_preview_texture(base, archive, binding)

            self.assertNotEqual(output, base)
            with Image.open(output) as image:
                self.assertEqual(image.convert("RGBA").getpixel((0, 0)), (50, 40, 30, 255))

    def test_shared_base_texture_variants_survive_when_preview_layers_differ(self) -> None:
        tree = ET.ElementTree(
            ET.fromstring(
                """<COLLADA>
                  <library_materials>
                    <material id="bolide_interior-material" name="bolide_interior"/>
                  </library_materials>
                  <library_visual_scenes>
                    <visual_scene>
                      <node id="bolide_dash" name="bolide_dash">
                        <instance_geometry url="#Mesh_030-mesh">
                          <bind_material>
                            <technique_common>
                              <instance_material
                                symbol="bolide_interior-material"
                                target="#bolide_interior-material"/>
                            </technique_common>
                          </bind_material>
                        </instance_geometry>
                      </node>
                    </visual_scene>
                  </library_visual_scenes>
                </COLLADA>"""
            )
        )
        loaded = LoadedDae(
            path=Path("vehicles/bolide/bolide.dae"),
            tree=tree,
            namespace="",
            unit_scale=1.0,
            parts=[],
            geometries={},
        )
        part = DaePart(
            key="bolide_dash",
            label="bolide_dash",
            node_id="bolide_dash",
            node_name="bolide_dash",
            matrix=np.eye(4),
            instances=(GeometryInstance("Mesh_030-mesh"),),
        )
        archive = VehicleArchive(
            path=Path("bolide.zip"),
            members=("vehicles/bolide/bolide_interior_b.color.DDS",),
            member_by_lower={
                "vehicles/bolide/bolide_interior_b.color.dds": "vehicles/bolide/bolide_interior_b.color.DDS",
            },
            member_sizes={},
            dae_members=("vehicles/bolide/bolide.dae",),
            materials=(
                ArchiveMaterialRecord(
                    key="bolide_interior",
                    name="bolide_interior",
                    map_to="bolide_interior",
                    materials_member="vehicles/bolide/main.materials.json",
                    base_colour_reference="/vehicles/bolide/bolide_interior_b.color.png",
                    preview_layers=(
                        ArchiveMaterialPreviewLayer(
                            base_colour_reference="/vehicles/bolide/bolide_interior_b.color.png",
                            base_colour_factor=(0.25, 0.25, 0.25, 1.0),
                            opacity_reference="/vehicles/bolide/bolide_interior_c.data.png",
                        ),
                    ),
                ),
                ArchiveMaterialRecord(
                    key="bolide_interior.skin_interior.cream",
                    name="bolide_interior.skin_interior.cream",
                    map_to="bolide_interior.skin_interior.cream",
                    materials_member="vehicles/bolide/skin.materials.json",
                    base_colour_reference="/vehicles/bolide/bolide_interior_b.color.png",
                    preview_layers=(
                        ArchiveMaterialPreviewLayer(
                            base_colour_reference="/vehicles/bolide/bolide_interior_b.color.png",
                            base_colour_factor=(0.8, 0.7, 0.6, 1.0),
                            opacity_reference="/vehicles/bolide/bolide_interior_c.data.png",
                        ),
                    ),
                ),
            ),
            workspace=Path("."),
        )

        choices = archive_texture_choices_for_part(archive, loaded, part)

        self.assertEqual(
            [choice.material_key for choice in choices],
            ["bolide_interior", "bolide_interior.skin_interior.cream"],
        )


if __name__ == "__main__":
    unittest.main()
