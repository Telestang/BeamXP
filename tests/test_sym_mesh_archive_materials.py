from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from mesh_segmentation_transform.beamxp_transform_sym_mesh_POC import (
    ArchiveMaterialRecord,
    ArchiveMaterialPreviewLayer,
    ArchiveTextureBinding,
    DaePart,
    GeometryInstance,
    LoadedDae,
    VehicleArchive,
    _blend_archive_preview_texture,
    archive_texture_choices_for_part,
)


class SymMeshArchiveMaterialTests(unittest.TestCase):
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
