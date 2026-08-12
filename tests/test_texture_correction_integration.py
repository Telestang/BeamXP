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
            target_mesh = core.generated_mesh_name("lc500_interior", core.HAND_RHD)
            generated_jbeam = output / "handdrive_visual_conversion.jbeam"
            generated_jbeam.write_text(
                '{"lc500":{"slotType":"main","flexbodies":[["mesh","[group]:"],["'
                + target_mesh
                + '",["lc500_body"]]]},'
                '"lc500_body_xp_rhd":{"slotType":"lc500_body",'
                '"flexbodies":[["mesh","[group]:"],["'
                + target_mesh
                + '",["lc500_body"]]]},'
                '"lc500_interior_xp_rhd":{"slotType":"lc500_interior",'
                '"flexbodies":[["mesh","[group]:"],["'
                + target_mesh
                + '",["lc500_body"]]]}}',
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
                    "lc500_interior": {"materials": ["lc500_centralscreen"]}
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
                {"lc500_interior"},
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

        self.assertEqual(result["replacedRows"], 0)
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
                (vehicle_dir / name).write_bytes(b"dds")
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

        self.assertEqual(result["rowReplacements"], {})
        self.assertEqual(result["jbeamPatch"], {"files": [], "replacedRows": 0})
        self.assertEqual(result["daePatches"][0]["appendedNodes"], [])
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
                (job / name).write_bytes(b"dds")
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
        self.assertIn(
            "ardente_interior_on_beamxp_tc_ardente_interior_g.color_rhd.dds",
            materials["ardente_interior_on_beamxp_tc"]["Stages"][0]["emissiveMap"],
        )

    def test_switch_base_alias_maps_to_corrected_off_state_material(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            job = tmp / "job"
            target = tmp / "vehicles/lc500"
            job.mkdir()
            (job / "LEX_LC5_rhd.dds").write_bytes(b"dds")
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
            (job / "lc500_bootscreen_rhd.dds").write_bytes(b"dds")
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
