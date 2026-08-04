from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from xml.etree import ElementTree as ET

from beamxp.core.dae import DaeObject
from beamxp.core.models import VehicleContext
from beamxp import hand_drive_core as core
from beamxp.hand_drive_parts import build_pipeline


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


class TextureCorrectionIntegrationTests(unittest.TestCase):
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

    def test_export_artifacts_isolates_marked_mesh_material_jobs(self) -> None:
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
                    report_path=tmp / "controls_rhd_preview.report.json",
                    dae_paths=(tmp / "scintilla_dash_controls_rhd.dae",),
                    textures=[object()],
                    seconds=1.25,
                ),
                SimpleNamespace(
                    report_path=tmp / "dashboard_rhd_preview.report.json",
                    dae_paths=(tmp / "scintilla_dashboard_rhd.dae",),
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
            self.assertEqual(export.call_count, 2)
            selected_parts = [call.args[2][0].key for call in export.call_args_list]
            self.assertEqual(
                selected_parts,
                ["scintilla_dash_controls", "scintilla_dashboard"],
            )
            self.assertTrue(all(call.args[4].detect_on_normal_map for call in export.call_args_list))
            self.assertEqual(len(report["jobs"]), 2)
            self.assertEqual(
                [job["meshes"] for job in report["jobs"]],
                [["scintilla_dash_controls"], ["scintilla_dashboard"]],
            )
            self.assertTrue(any("Texture correction:" in message for message in progress_messages))
            self.assertTrue(any("finished" in message for message in progress_messages))

    def test_jbeam_patch_replaces_every_flexbody_array(self) -> None:
        source = """{
          "part_a": {
            "flexbodies": [
              ["scintilla_dashboard_xp_rhd", ["dash"]]
            ]
          },
          "part_b": {
            "flexbodies": [
              ["scintilla_controls_xp_rhd", ["dash"]]
            ]
          }
        }"""

        updated = build_pipeline._replace_all_jbeam_array_regions(
            source,
            "flexbodies",
            lambda text: build_pipeline._expand_texture_correction_flexbody_array(
                text,
                {
                    "scintilla_dashboard_xp_rhd": [
                        "scintilla_dashboard__beamxp_mirrored_carrier",
                        "scintilla_dashboard__beamxp_rigid_001",
                    ],
                    "scintilla_controls_xp_rhd": [
                        "scintilla_controls__beamxp_mirrored_carrier",
                    ],
                },
            ),
        )

        self.assertNotIn('"scintilla_dashboard_xp_rhd"', updated)
        self.assertNotIn('"scintilla_controls_xp_rhd"', updated)
        self.assertIn('"scintilla_dashboard__beamxp_rigid_001"', updated)
        self.assertIn('"scintilla_controls__beamxp_mirrored_carrier"', updated)

    def test_append_texture_correction_dae_bakes_transform_and_material(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            target = tmp / "target.dae"
            source = tmp / "source.dae"
            target.write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema">
  <library_geometries/>
  <library_visual_scenes><visual_scene id="Scene"/></library_visual_scenes>
</COLLADA>
""",
                encoding="utf-8",
            )
            source.write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema">
  <library_effects>
    <effect id="dash_mat-effect"/>
  </library_effects>
  <library_materials>
    <material id="dash_mat-material" name="dash_mat">
      <instance_effect url="#dash_mat-effect"/>
    </material>
  </library_materials>
  <library_geometries>
    <geometry id="geom_001"><mesh>
      <source id="geom_001-positions"><float_array id="geom_001-positions-array" count="3">1 2 3</float_array></source>
      <source id="geom_001-normals"><float_array id="geom_001-normals-array" count="3">0 1 0</float_array></source>
      <triangles material="dash_mat-material" count="0"/>
    </mesh></geometry>
  </library_geometries>
  <library_visual_scenes>
    <visual_scene id="Scene">
      <node id="dash_split" name="dash_split">
        <matrix>1 0 0 2 0 1 0 3 0 0 1 4 0 0 0 1</matrix>
        <instance_geometry url="#geom_001">
          <bind_material><technique_common>
            <instance_material symbol="dash_mat-material" target="#dash_mat-material"/>
          </technique_common></bind_material>
        </instance_geometry>
      </node>
    </visual_scene>
  </library_visual_scenes>
</COLLADA>
""",
                encoding="utf-8",
            )

            appended = build_pipeline._append_texture_correction_dae(
                target,
                source,
                {"dash_split"},
                {"dash_mat": "dash_beamxp_tc"},
            )

            self.assertEqual(appended, ["dash_split"])
            root = ET.parse(target).getroot()
            ns = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
            positions = root.find(".//c:source[@id='geom_001-positions']/c:float_array", ns)
            normals = root.find(".//c:source[@id='geom_001-normals']/c:float_array", ns)
            node_matrix = root.find(".//c:node[@id='dash_split']/c:matrix", ns)
            triangle = root.find(".//c:triangles", ns)
            binding = root.find(".//c:instance_material", ns)
            material = root.find(".//c:material[@id='dash_beamxp_tc-material']", ns)
            effect = root.find(".//c:effect[@id='dash_mat-effect']", ns)

            self.assertEqual(positions.text, "3 5 7")
            self.assertEqual(normals.text, "0 1 0")
            self.assertEqual(node_matrix.text, "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1")
            self.assertEqual(triangle.get("material"), "dash_beamxp_tc")
            self.assertEqual(binding.get("symbol"), "dash_beamxp_tc")
            self.assertEqual(binding.get("target"), "#dash_beamxp_tc-material")
            self.assertIsNotNone(material)
            self.assertEqual(material.get("name"), "dash_beamxp_tc")
            self.assertIsNotNone(effect)

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
                (job / name).write_bytes(b"dds")
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

    def test_repeated_alias_jobs_get_distinct_corrected_materials(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            target = tmp / "vehicles/scintilla"

            def write_job(name: str) -> Path:
                job = tmp / name
                job.mkdir()
                texture = f"{name}_b.color_rhd.dds"
                (job / texture).write_bytes(b"dds")
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


if __name__ == "__main__":
    unittest.main()
