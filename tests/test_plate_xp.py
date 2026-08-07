from __future__ import annotations

import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from beamxp import hand_drive_core as core
from beamxp.plates import generator as plate_generator


class PlateXpTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.previous_data_dir = os.environ.get("BEAMXP_DATA_DIR")
        os.environ["BEAMXP_DATA_DIR"] = str(self.root / "data")

    def tearDown(self) -> None:
        if self.previous_data_dir is None:
            os.environ.pop("BEAMXP_DATA_DIR", None)
        else:
            os.environ["BEAMXP_DATA_DIR"] = self.previous_data_dir

    def _context(self, *, with_plate_parts: bool = False) -> core.VehicleContext:
        source = self.root / "test.zip"
        pc = {
            "mainPartName": "test",
            "parts": {
                "test_licenseplate_F": "plate_f_2",
                "test_licenseplate_R": "plate_r_wide",
            } if with_plate_parts else {},
            "licenseName": "STOCK",
        }
        with zipfile.ZipFile(source, "w") as archive:
            archive.writestr("vehicles/test/base.pc", json.dumps(pc))

        part_index: dict[str, tuple[str, str]] = {}
        if with_plate_parts:
            main = {
                "slotType": "main",
                "slots": [
                    ["type", "default", "description"],
                    ["test_licenseplate_F", "plate_f_2", "Front"],
                    ["test_licenseplate_R", "plate_r_wide", "Rear"],
                ],
            }

            def plate_body(part_id: str, slot: str, fmt: str, name: str, mesh: str | None = None) -> str:
                body = json.dumps({
                    "information": {"name": name},
                    "slotType": slot,
                    "licenseplateFormat": fmt,
                    "flexbodies": [
                        ["mesh", "[group]:", "nonFlexMaterials"],
                        [mesh or ("licenseplate" if fmt == "30-15" else "licenseplate-52-11"), [], []],
                    ],
                })
                return f'"{part_id}": {body}'

            part_index = {
                "test": (json.dumps(main), "test.jbeam"),
                "plate_f_2": (plate_body("plate_f_2", "test_licenseplate_F", "30-15", "Front US Plate"), "plates.jbeam"),
                "plate_f_wide": (plate_body("plate_f_wide", "test_licenseplate_F", "52-11", "Front EU Plate"), "plates.jbeam"),
                "plate_f_alt_wide": (
                    plate_body(
                        "plate_f_alt_wide",
                        "test_licenseplate_F_alt",
                        "52-11",
                        "Front EU Curved Plate",
                        "licenseplate-52-11-r0_5",
                    ),
                    "plates_alt.jbeam",
                ),
                "plate_r_2": (plate_body("plate_r_2", "test_licenseplate_R", "30-15", "Rear US Plate"), "plates.jbeam"),
                "plate_r_wide": (plate_body("plate_r_wide", "test_licenseplate_R", "52-11", "Rear EU Plate"), "plates.jbeam"),
            }

        return core.VehicleContext(
            source,
            "test",
            "vehicles/test",
            [],
            {"base": core.VariantInfo("base", "vehicles/test/base.pc", None, "Base")},
            {},
            {},
            {},
            {},
            self.root / "project",
            part_body_index=part_index,
        )

    def test_legacy_selected_trim_migrates_to_converted(self) -> None:
        context = self._context()
        conversion = core.merge_with_current_inventory(context, {
            "variants": {"base": {"selected": True, "sourceHandOverride": core.HAND_LHD}},
            "plate": {"enabled": False},
        })
        self.assertEqual(conversion["variants"]["base"]["build"], core.BUILD_CONVERTED)
        self.assertEqual(conversion["plate"]["mode"], plate_generator.PLATE_MODE_OFF)

    def test_detected_stock_hand_is_cached_across_context_sessions(self) -> None:
        context = self._context()
        conversion = core.base_conversion_config(context)
        with patch.object(core, "detect_hand_for_variant", return_value=core.HAND_LHD) as detect:
            hands = core.detect_hands_for_variants(context, conversion)
        self.assertEqual(hands, {"base": core.HAND_LHD})
        detect.assert_called_once_with(context, conversion, "base")
        self.assertTrue(core.variant_hands_cache_path(context).is_file())

        context.variant_hands_cache = {}
        with patch.object(core, "detect_hand_for_variant", side_effect=AssertionError("cache miss")):
            cached = core.detect_hands_for_variants(context, conversion)
        self.assertEqual(cached, {"base": core.HAND_LHD})

        conversion["parts"]["steering_ref"] = {"steeringRef": True}
        with patch.object(core, "detect_hand_for_variant", return_value=core.HAND_RHD) as detect_changed:
            changed = core.detect_hands_for_variants(context, conversion)
        self.assertEqual(changed, {"base": core.HAND_RHD})
        detect_changed.assert_called_once_with(context, conversion, "base")

    def test_live_set_reference_keeps_deleted_snapshot(self) -> None:
        config = plate_generator.default_plate_config()
        config["eu"]["pattern"] = "SET ##"
        plate_generator.save_plate_set({"id": "set-one", "name": "Set One", "config": config})
        conversion = {
            "plate": {"mode": "set", "setId": "set-one", "config": plate_generator.default_plate_config()},
            "variants": {"base": {"plate": {"mode": "general"}}},
        }
        resolved, set_id = plate_generator.effective_plate_selection(conversion, "base")
        self.assertEqual(set_id, "set-one")
        self.assertEqual(resolved["eu"]["pattern"], "SET ##")

        plate_generator.delete_plate_set("set-one")
        warnings: list[str] = []
        fallback, _set_id = plate_generator.effective_plate_selection(conversion, "base", warnings=warnings)
        self.assertEqual(fallback["eu"]["pattern"], "SET ##")
        self.assertTrue(warnings)

    def test_trim_custom_reference_is_live_and_keeps_a_snapshot(self) -> None:
        source_config = plate_generator.default_plate_config()
        source_config["eu"]["pattern"] = "SPORT ##"
        conversion = {
            "plate": plate_generator.default_plate_binding(),
            "variants": {
                "sport_RS_M": {
                    "plate": {
                        "mode": "custom",
                        "sourceConfig": "sport_RS_M",
                        "customDefined": True,
                        "config": source_config,
                    },
                },
                "sport_RS_DCT": {
                    "plate": {
                        "mode": "trim",
                        "sourceConfig": "sport_RS_M",
                        "config": plate_generator.default_plate_config(),
                    },
                },
            },
        }
        resolved, set_id = plate_generator.effective_plate_selection(conversion, "sport_RS_DCT")
        self.assertIsNone(set_id)
        self.assertEqual(resolved["eu"]["pattern"], "SPORT ##")

        conversion["variants"]["sport_RS_M"]["plate"]["customConfig"]["eu"]["pattern"] = "UPDATED ##"
        updated, _set_id = plate_generator.effective_plate_selection(conversion, "sport_RS_DCT")
        self.assertEqual(updated["eu"]["pattern"], "UPDATED ##")

        del conversion["variants"]["sport_RS_M"]
        warnings: list[str] = []
        fallback, _set_id = plate_generator.effective_plate_selection(
            conversion,
            "sport_RS_DCT",
            warnings=warnings,
        )
        self.assertEqual(fallback["eu"]["pattern"], "UPDATED ##")
        self.assertTrue(warnings)

    def test_inline_design_is_named_beamxp_custom(self) -> None:
        output = type("Design", (), {
            "design_json_rel": "vehicles/common/licenseplates/test/licensePlate.json",
            "bundled": False,
        })()
        body = json.loads(plate_generator._design_part_body(output, "EU", custom=True))
        self.assertEqual(body["information"]["name"], "BeamXP Custom")

    def test_both_expands_to_two_outputs_in_one_xp_package(self) -> None:
        context = self._context()
        conversion = core.base_conversion_config(context)
        settings = conversion["variants"]["base"]
        settings["sourceHandOverride"] = core.HAND_LHD
        core.set_variant_build_mode(settings, core.BUILD_BOTH)
        plans, skipped = core.selected_output_plans(context, conversion)
        self.assertFalse(skipped)
        self.assertEqual([plan["output"] for plan in plans], ["base_rhd", "base_plates"])
        self.assertEqual(core.package_name_for_context(context), "test_XP_conversion.zip")

    def test_rhd_trim_can_build_only_replacement_plates_without_an_lhd_output(self) -> None:
        context = self._context()
        conversion = core.base_conversion_config(context)
        settings = conversion["variants"]["base"]
        settings["sourceHandOverride"] = core.HAND_RHD
        core.set_variant_build_mode(settings, core.BUILD_ORIGINAL)
        plans, skipped = core.selected_output_plans(context, conversion)
        self.assertFalse(skipped)
        self.assertEqual(plans, [{
            "source": "base",
            "kind": core.BUILD_ORIGINAL,
            "targetHand": None,
            "output": "base_plates",
        }])

    def test_original_plate_build_changes_each_side_independently(self) -> None:
        context = self._context(with_plate_parts=True)
        front_values = plate_generator.plate_part_options_for_config(context, "base", "front")
        rear_values = plate_generator.plate_part_options_for_config(context, "base", "rear")
        self.assertEqual(front_values[0], "auto")
        self.assertIn("mesh:licenseplate-52-11", front_values)
        self.assertEqual(front_values[-1], "none")
        self.assertEqual(rear_values[0], "auto")
        self.assertIn("mesh:licenseplate", rear_values)
        self.assertEqual(rear_values[-1], "none")
        front_choices = plate_generator.plate_part_choices_for_config(context, "base", "front")
        self.assertEqual(front_choices[0].label, "US/JP (default)")
        conversion = core.base_conversion_config(context)
        settings = conversion["variants"]["base"]
        core.set_variant_build_mode(settings, core.BUILD_ORIGINAL)
        settings["frontPlate"] = "mesh:licenseplate-52-11"
        settings["rearPlate"] = plate_generator.PLATE_PART_NONE
        config = plate_generator.default_plate_config()
        config["enabled"] = True
        conversion["plate"] = {"mode": "custom", "setId": "", "config": config}

        preview_pc, _preview_aliases = plate_generator.preview_pc_with_plate_parts(context, conversion, "base")
        self.assertEqual(preview_pc["parts"]["test_licenseplate_F"], "plate_f_wide")
        self.assertEqual(preview_pc["parts"]["test_licenseplate_R"], "")

        result = core.build_batch(context, conversion, write_zip=True)
        generated_path = result.unpacked_dir / "vehicles/test/base_plates.pc"
        generated = json.loads(generated_path.read_text(encoding="utf-8"))
        self.assertEqual(generated["parts"]["test_licenseplate_F"], "plate_f_wide")
        self.assertEqual(generated["parts"]["test_licenseplate_R"], "")
        self.assertEqual(result.generated_configs, ["base_plates"])
        self.assertEqual(result.package_zip.name, "test_XP_conversion.zip")
        with zipfile.ZipFile(result.package_zip) as archive:
            self.assertTrue(all(info.compress_type == zipfile.ZIP_STORED for info in archive.infolist()))
        generated_parts = core.parse_beamng_json(
            (generated_path.parent / "jbeam/bhdc_licenseplates.jbeam").read_text(encoding="utf-8"),
            label="bhdc_licenseplates.jbeam",
        )
        custom_parts = [part for part in generated_parts.values() if isinstance(part, dict)]
        self.assertEqual(custom_parts[0]["information"]["name"], "BeamXP Custom")

    def test_plate_part_from_another_model_slot_is_cloned_into_the_trim_slot(self) -> None:
        context = self._context(with_plate_parts=True)
        conversion = core.base_conversion_config(context)
        settings = conversion["variants"]["base"]
        core.set_variant_build_mode(settings, core.BUILD_ORIGINAL)
        settings["frontPlate"] = "mesh:licenseplate-52-11-r0_5"

        preview_pc, preview_aliases = plate_generator.preview_pc_with_plate_parts(context, conversion, "base")
        preview_selected = preview_pc["parts"]["test_licenseplate_F"]
        self.assertTrue(preview_selected.startswith("bhdc_plate_plate_f_alt_wide_"))
        self.assertIn(preview_selected, preview_aliases)

        result = core.build_batch(context, conversion, write_zip=False)
        generated_path = result.unpacked_dir / "vehicles/test/base_plates.pc"
        generated = json.loads(generated_path.read_text(encoding="utf-8"))
        selected = generated["parts"]["test_licenseplate_F"]
        self.assertTrue(selected.startswith("bhdc_plate_plate_f_alt_wide_"))

        generated_parts = core.parse_beamng_json(
            (generated_path.parent / "jbeam/bhdc_licenseplates.jbeam").read_text(encoding="utf-8"),
            label="bhdc_licenseplates.jbeam",
        )
        cloned = generated_parts[selected]
        self.assertEqual(cloned["slotType"], "test_licenseplate_F")
        self.assertEqual(cloned["flexbodies"][1][0], "licenseplate-52-11-r0_5")

    def test_designs_always_define_rear_formats_even_when_rear_matches_front(self) -> None:
        config = plate_generator.default_plate_config()
        config["enabled"] = True
        self.assertFalse(plate_generator._rear_texture_differs(config))
        plate_generator.save_plate_set({"id": "same-both", "name": "Same Both", "config": config})

        result = plate_generator.export_all_plate_sets()
        with zipfile.ZipFile(result["zip"]) as archive:
            design_path = next(
                name for name in archive.namelist() if name.endswith("licensePlate.json")
            )
            design = json.loads(archive.read(design_path).decode("utf-8"))
        formats = design["data"]["format"]
        self.assertIn("bhdc-rear-wide", formats)
        self.assertIn("bhdc-rear-2-1", formats)
        self.assertIn("52-11", formats)
        self.assertIn("30-15", formats)

    def test_rear_part_is_cloned_even_when_the_design_rear_matches_the_front(self) -> None:
        context = self._context(with_plate_parts=True)
        conversion = core.base_conversion_config(context)
        settings = conversion["variants"]["base"]
        core.set_variant_build_mode(settings, core.BUILD_ORIGINAL)
        config = plate_generator.default_plate_config()
        config["enabled"] = True
        conversion["plate"] = {"mode": "custom", "setId": "", "config": config}

        result = core.build_batch(context, conversion, write_zip=False)
        generated_path = result.unpacked_dir / "vehicles/test/base_plates.pc"
        generated = json.loads(generated_path.read_text(encoding="utf-8"))
        rear_part = generated["parts"]["test_licenseplate_R"]
        self.assertTrue(rear_part.startswith("bhdc_rear_"), rear_part)
        self.assertEqual(result.plate_summary.get("rearPartsCloned"), 1)
        self.assertEqual(result.plate_summary.get("warnings"), [])
        generated_parts = core.parse_beamng_json(
            (generated_path.parent / "jbeam/bhdc_licenseplates.jbeam").read_text(encoding="utf-8"),
            label="bhdc_licenseplates.jbeam",
        )
        self.assertEqual(generated_parts[rear_part]["licenseplateFormat"], "bhdc-rear-wide")

    def test_cloned_rear_material_matches_the_plate_it_was_cloned_from(self) -> None:
        """Only the @licenseplate-* tags may differ from the source plate.

        A hand-authored stage made the rear diverge from the front: the stock
        plate material is metallic (metallicFactor 1 + metallicMap) and
        retroreflective (0.5), so omitting those rendered the rear flat and
        killed its headlight response.
        """
        source = {
            "name": "licenseplate-52-11",
            "mapTo": "licenseplate-52-11",
            "class": "Material",
            "Stages": [
                {
                    "baseColorMap": "@licenseplate-52-11",
                    "metallicFactor": 1,
                    "metallicMap": "@licenseplate-52-11-metallic",
                    "normalMap": "@licenseplate-52-11-normal",
                    "normalMapUseUV": 1,
                    "retroreflectivity": 0.5,
                    "roughnessMap": "@licenseplate-52-11-specular",
                },
                {}, {}, {},
            ],
            "dynamicCubemap": True,
            "version": 1.5,
        }
        entry = plate_generator._rear_material_entry("bhdc_rear_plate", "bhdc-rear-wide", source)

        self.assertEqual(entry["name"], "bhdc_rear_plate")
        self.assertEqual(entry["mapTo"], "bhdc_rear_plate")
        stage = entry["Stages"][0]
        # Inherited: the stock finish, which a hand-authored stage kept losing.
        self.assertEqual(stage["metallicFactor"], 1)
        self.assertEqual(stage["retroreflectivity"], 0.5)
        self.assertEqual(stage["baseColorMap"], "@licenseplate-bhdc-rear-wide")
        self.assertEqual(stage["roughnessMap"], "@licenseplate-bhdc-rear-wide-specular")
        self.assertNotIn("normalMapStrength", stage)
        self.assertIsNot(stage, source["Stages"][0])
        # ...but only the maps a custom format actually gets: see
        # _apply_custom_format_maps and the EuroPlates mod it follows.
        self.assertEqual(stage["metallicMap"], "@licenseplate-default-metallic")
        self.assertNotIn("normalMap", stage)
        self.assertNotIn("normalMapUseUV", stage)

    def test_clone_inherits_the_source_material_and_effect(self) -> None:
        """Rebuilding them as a bare lambert drops whatever the exporter wrote.

        The stock plate effect carries an emission colour and a reflectivity
        float; a synthetic stand-in has neither, so the clone would differ from
        the mesh it was copied from before any material JSON is consulted.
        """
        from xml.etree import ElementTree as ET

        ns = "http://www.collada.org/2005/11/COLLADASchema"
        doc = ET.fromstring(
            f'<COLLADA xmlns="{ns}">'
            '<library_effects><effect id="plate-effect"><profile_COMMON><technique sid="common">'
            '<lambert><emission><color sid="emission">0 0 0 1</color></emission>'
            '<diffuse><color sid="diffuse">0.735 0.735 0.735 1</color></diffuse>'
            '<reflectivity><float sid="specular">50</float></reflectivity>'
            "</lambert></technique></profile_COMMON></effect></library_effects>"
            '<library_materials><material id="plate-material" name="plate">'
            '<instance_effect url="#plate-effect"/></material></library_materials>'
            '<library_visual_scenes><visual_scene><node id="plate">'
            "<instance_geometry url=\"#g\"><bind_material><technique_common>"
            '<instance_material symbol="plate-material" target="#plate-material"/>'
            "</technique_common></bind_material></instance_geometry>"
            "</node></visual_scene></library_visual_scenes></COLLADA>"
        )
        node = doc.find(f".//{{{ns}}}node")
        mats = {m.get("id"): m for m in doc.iter(f"{{{ns}}}material")}
        effects = {e.get("id"): e for e in doc.iter(f"{{{ns}}}effect")}

        material, effect = plate_generator._inherited_material_pair(
            doc, node, "bhdc_rear_plate", mats, effects
        )
        self.assertEqual(material.get("id"), "bhdc_rear_plate-material")
        self.assertEqual(material.get("name"), "bhdc_rear_plate")
        self.assertEqual(
            material.find(f"{{{ns}}}instance_effect").get("url"), "#bhdc_rear_plate-effect"
        )
        self.assertEqual(effect.get("id"), "bhdc_rear_plate-effect")
        text = ET.tostring(effect, encoding="unicode")
        self.assertIn("emission", text)
        self.assertIn("reflectivity", text)
        self.assertIn("0.735", text)
        # the source document must not be mutated
        self.assertEqual(mats["plate-material"].get("id"), "plate-material")
        self.assertEqual(effects["plate-effect"].get("id"), "plate-effect")

    def test_a_vehicles_own_archive_is_never_treated_as_shared(self) -> None:
        """common_zip_candidates() leads with the vehicle's own archive.

        Taking that list at face value files every vehicle-owned plate mesh as
        shared and publishes it to vehicles/common under a name another
        conversion would overwrite.
        """
        context = self._context(with_plate_parts=True)
        self.assertFalse(plate_generator._is_shared_plate_zip(context, context.source_zip))

    def test_shared_rear_format_follows_the_mesh_not_the_part(self) -> None:
        """One material is written for all vehicles, so it cannot depend on a
        part; the cloned part must request the same format or it asks the game
        to render a tag its material never reads."""
        self.assertEqual(
            plate_generator._shared_rear_format("licenseplate-52-11"), "bhdc-rear-wide"
        )
        self.assertEqual(
            plate_generator._shared_rear_format("licenseplate-52-11-r0_5"), "bhdc-rear-wide"
        )
        self.assertEqual(plate_generator._shared_rear_format("licenseplate"), "bhdc-rear-2-1")

    def test_shared_clone_names_carry_no_machine_specific_digest(self) -> None:
        """A shared asset is written to the same virtual path by every BeamXP
        mod, so its names must not vary with the builder's archive layout."""
        context = self._context(with_plate_parts=True)
        obj = type("Obj", (), {"dae_path": "vehicles/common/empty.dae", "id": "licenseplate",
                               "dae_source_zip": None})()
        a = plate_generator._rear_clone_mesh_name(context, "licenseplate", obj, shared=True)
        self.assertEqual(a, "bhdc_rear_licenseplate")
        self.assertEqual(
            plate_generator._rear_clone_dae_name(Path("/one/common.zip"), "vehicles/common/empty.dae", shared=True),
            plate_generator._rear_clone_dae_name(Path("/other/common.zip"), "vehicles/common/empty.dae", shared=True),
        )
        # a vehicle-owned mesh still disambiguates by source archive
        self.assertNotEqual(
            plate_generator._rear_clone_mesh_name(context, "licenseplate", obj, shared=False), a
        )

    def test_rear_material_keeps_the_explicit_nulls_of_its_source(self) -> None:
        """Materials are read by the engine's C++ material manager, not the Lua
        reader beamxp.core.sjson ports - and that reader drops null. Every stock
        plate material spells out "metallicFactor": null on its unused stages,
        so dropping them is the one way a copied material can differ from its
        source that cannot be checked from Lua."""
        raw = json.dumps({
            "licenseplate-52-11": {
                "name": "licenseplate-52-11",
                "mapTo": "licenseplate-52-11",
                "class": "Material",
                "Stages": [
                    {"baseColorMap": "@licenseplate-52-11", "metallicFactor": 1},
                    {"metallicFactor": None, "normalMapUseUV": None, "retroreflectivity": None},
                    {"metallicFactor": None, "normalMapUseUV": None, "retroreflectivity": None},
                    {"metallicFactor": None, "normalMapUseUV": None, "retroreflectivity": None},
                ],
                "version": 1.5,
            }
        }, indent=2)
        # the tolerant reader is what silently ate them
        self.assertEqual(
            core.parse_beamng_json(raw, label="t")["licenseplate-52-11"]["Stages"][1], {}
        )

        source = plate_generator._read_material_file(raw, "t")["licenseplate-52-11"]
        self.assertEqual(
            source["Stages"][1], {"metallicFactor": None, "normalMapUseUV": None, "retroreflectivity": None}
        )
        entry = plate_generator._rear_material_entry("bhdc_rear_plate", "bhdc-rear-wide", source)
        for stage in entry["Stages"][1:]:
            self.assertEqual(
                stage, {"metallicFactor": None, "normalMapUseUV": None, "retroreflectivity": None}
            )
        self.assertIn('"metallicFactor": null', json.dumps(entry, indent=2))

        # a hand-edited file the strict reader chokes on still loads
        loose = '{"m": {"mapTo": "m", "Stages": [{"baseColorMap": "@licenseplate-52-11"},],},}'
        self.assertIn("m", plate_generator._read_material_file(loose, "t"))

    def test_rear_material_ignores_a_source_that_bakes_its_plate_texture(self) -> None:
        baked = {
            "name": "modplate",
            "mapTo": "modplate",
            "Stages": [{"baseColorMap": "vehicles/mod/plate_d.png"}, {}, {}, {}],
        }
        entry = plate_generator._rear_material_entry("bhdc_rear_plate", "bhdc-rear-wide", baked)
        # Adopting it would leave the rear unable to show the registration.
        self.assertEqual(entry["Stages"][0]["baseColorMap"], "@licenseplate-bhdc-rear-wide")

    def test_rear_material_without_a_source_uses_the_stock_plate_shape(self) -> None:
        entry = plate_generator._rear_material_entry("bhdc_rear_quad", "bhdc-rear-2-1", None)
        stage = entry["Stages"][0]
        self.assertEqual(stage["baseColorMap"], "@licenseplate-bhdc-rear-2-1")
        self.assertEqual(stage["metallicFactor"], 1)
        self.assertEqual(stage["metallicMap"], "@licenseplate-default-metallic")
        self.assertEqual(stage["retroreflectivity"], 0.5)
        self.assertEqual(stage["roughnessMap"], "@licenseplate-bhdc-rear-2-1-specular")
        # The module-level template must not be mutated by a previous call.
        self.assertEqual(
            plate_generator._STOCK_PLATE_MATERIAL["Stages"][0]["baseColorMap"],
            "@licenseplate-default",
        )

    def test_bundled_set_design_does_not_collide_with_the_library_part(self) -> None:
        """A vehicle build bundles its own copy of a library set.

        Both live on slot licenseplate_design_2_1 but point at different design
        folders, so sharing one part name makes jbeam/io.lua getPart() pick by
        folder search order (getAvailableParts logs "parts names are duplicate").
        """
        config = plate_generator.default_plate_config()
        config["enabled"] = True
        plate_generator.save_plate_set({"id": "uk-modern", "name": "UK Modern", "config": config})

        context = self._context(with_plate_parts=True)
        conversion = core.base_conversion_config(context)
        settings = conversion["variants"]["base"]
        core.set_variant_build_mode(settings, core.BUILD_ORIGINAL)
        conversion["plate"] = {"mode": "set", "setId": "uk-modern", "config": config}

        result = core.build_batch(context, conversion, write_zip=False)
        vehicle_dir = result.unpacked_dir / "vehicles" / context.vehicle_id
        bundled = core.parse_beamng_json(
            (vehicle_dir / "jbeam/bhdc_licenseplates.jbeam").read_text(encoding="utf-8"),
            label="bhdc_licenseplates.jbeam",
        )
        bundled_id = f"bhdc_{core.safe_id(context.vehicle_id)}_plateset_uk_modern"
        self.assertIn(bundled_id, bundled)
        self.assertNotIn("bhdc_plateset_uk_modern", bundled)

        generated = json.loads((vehicle_dir / "base_plates.pc").read_text(encoding="utf-8"))
        self.assertEqual(generated["parts"]["licenseplate_design_2_1"], bundled_id)

        library = plate_generator.export_all_plate_sets()
        with zipfile.ZipFile(library["zip"]) as archive:
            shared = core.parse_beamng_json(
                archive.read("vehicles/common/licenseplates/bhdc_plate_sets.jbeam").decode("utf-8"),
                label="bhdc_plate_sets.jbeam",
            )
        # The library keeps the stable name so existing selections survive.
        self.assertIn("bhdc_plateset_uk_modern", shared)
        self.assertFalse(set(shared) & set(bundled))
        self.assertNotEqual(
            bundled[bundled_id]["information"]["name"],
            shared["bhdc_plateset_uk_modern"]["information"]["name"],
        )

    def test_rear_formats_are_merged_into_the_stock_default_design(self) -> None:
        """core/licensePlateDesign.lua mergeLegacyFormats() reads this exact path.

        Without it setPlateText()'s per-format retry against the stock
        sktemplate has no bhdc-rear-* block, nothing binds
        @licenseplate-bhdc-rear-*, and the rear plate renders NO TEXTURE
        whenever a vanilla design is selected.
        """
        context = self._context(with_plate_parts=True)
        conversion = core.base_conversion_config(context)
        settings = conversion["variants"]["base"]
        core.set_variant_build_mode(settings, core.BUILD_ORIGINAL)
        config = plate_generator.default_plate_config()
        config["enabled"] = True
        conversion["plate"] = {"mode": "custom", "setId": "", "config": config}

        result = core.build_batch(context, conversion, write_zip=False)
        fallback = result.unpacked_dir / "vehicles/common/licenseplates/default/licensePlate-default.json"
        self.assertTrue(fallback.is_file())
        design = json.loads(fallback.read_text(encoding="utf-8"))
        formats = design["data"]["format"]
        # Both formats always, so the file is byte-identical in every BeamXP mod
        # and no conversion can shadow a format another conversion needs.
        self.assertEqual(sorted(formats), ["bhdc-rear-2-1", "bhdc-rear-wide"])
        # mergeLegacyFormats() only fills formats the sktemplate lacks, but a
        # stock key here would still be a rewrite of vanilla content.
        self.assertNotIn("52-11", formats)
        self.assertNotIn("30-15", formats)
        # The GE extension this replaced could never load: extNameToLuaPath()
        # maps '_' to '/', so "bhdc_rearPlates" resolved to bhdc/rearPlates.
        self.assertFalse((result.unpacked_dir / "lua").exists())
        self.assertFalse((result.unpacked_dir / "scripts").exists())

    def test_two_line_eu_design_carries_its_segment_ranges(self) -> None:
        """licensePlateDesign.translateLegacyFormat() turns each line's `limit`
        into format.segments; without it every segment is {0, 0} and skiaTemplate
        resolves both lines to an empty string."""
        font_path = plate_generator.resolve_font_path({"source": "default", "path": ""})
        font, metrics = plate_generator._plate_font_metrics(font_path)
        config = plate_generator.default_plate_config()
        # active_pattern() reads the family section, not a top-level "pattern"
        config["eu"]["pattern"] = "@@@## @@@"

        params = plate_generator._family_text_params(config, "30-15", metrics, font)
        self.assertEqual(params["layout"], "two-line")
        self.assertEqual([line["limit"] for line in params["lines"]], [[0, 5], [6, 9]])
        # buildPlateRoot() reads placement only from line.pos ([x, y, scale]);
        # the sibling x/y/scale keys alone leave both lines stacked dead centre
        # at full size, so pos must agree with them.
        for line in params["lines"]:
            self.assertEqual(line["pos"], [line["x"], line["y"], line["scale"]])
        top, bottom = params["lines"]
        self.assertLess(top["pos"][1], bottom["pos"][1])
        self.assertLess(top["pos"][2], 1.0)

    def test_emitted_scale_keeps_every_registration_inside_the_width_budget(self) -> None:
        """buildPlateRoot spans a line's node left 0 to right 0, so the game's
        `fit: "shrink"` only engages past the *full* plate width - it cannot see
        maxWidth. Unless the emitted font size already respects the budget, the
        preview is the only thing keeping text off the EU side band."""
        font_path = plate_generator.resolve_font_path({"source": "default", "path": ""})
        font, metrics = plate_generator._plate_font_metrics(font_path)
        width = 512
        spacing = 0

        def widest_line(pattern: str, band: str) -> tuple[float, float, float]:
            config = plate_generator.default_plate_config()
            config["eu"]["pattern"] = pattern
            config["eu"]["sideBand"] = band
            params = plate_generator._family_text_params(config, "30-15", metrics, font)
            worst = plate_generator._widest_registration(pattern, font)
            drawn = max(
                plate_generator._rendered_width(
                    worst[slice(*line["limit"])], font, metrics, spacing, params["scale"]
                )
                for line in params["lines"]
            )
            return drawn, params["lines"][0]["maxWidth"] * width, params["scale"]

        # Every roll of the pattern has to fit, not just the one the preview drew.
        for pattern in ("@@## @@@", "@@@## @@@", "@@@@## @@@@", "~~~~~~ ~~~~~~"):
            drawn, budget, _scale = widest_line(pattern, plate_generator.BAND_EU)
            self.assertLessEqual(round(drawn, 3), round(budget, 3), pattern)

        # A pattern that already fits keeps its natural size...
        short, _b, short_scale = widest_line("@@## @@@", plate_generator.BAND_EU)
        _d, _b2, unclamped = widest_line("@@## @@@", plate_generator.BAND_NONE)
        self.assertEqual(short_scale, unclamped)
        # ...and the wider no-band budget shrinks less than the banded one.
        _d, _b3, banded = widest_line("@@@## @@@", plate_generator.BAND_EU)
        _d, _b4, unbanded = widest_line("@@@## @@@", plate_generator.BAND_NONE)
        self.assertLess(banded, unbanded)

    def test_letter_spacing_comes_only_from_the_spacing_control(self) -> None:
        """buildPlateRoot gives a `lines` entry letterSpacing (xAdv or 0) + 2.

        Every other layout keeps textNode's 0, and the spacing setting is
        already baked into the atlas xadvance, so that constant would space
        two-line plates differently from every other format.
        """
        font_path = plate_generator.resolve_font_path({"source": "default", "path": ""})
        font, metrics = plate_generator._plate_font_metrics(font_path)
        for spacing in (-10, 0, 12, 30):
            config = plate_generator.default_plate_config()
            config["eu"]["pattern"] = "@@## @@@"
            config["eu"]["spacing"] = spacing
            params = plate_generator._family_text_params(config, "30-15", metrics, font)
            for line in params["lines"]:
                self.assertEqual(line["xAdv"] + 2, 0, f"spacing={spacing}")
            atlas = plate_generator.build_font_atlas(font_path, set("AB12"), spacing)
            advances = {
                entry["id"]: int(entry["xadvance"]) for entry in atlas.layout["chars"]["char"]
            }
            plain = plate_generator.build_font_atlas(font_path, set("AB12"), 0)
            base = {entry["id"]: int(entry["xadvance"]) for entry in plain.layout["chars"]["char"]}
            self.assertTrue(
                all(advances[k] - base[k] == spacing for k in advances),
                "the spacing control must be the one that moves the glyphs",
            )
        # 0-based, end-exclusive: text:sub(limit[1] + 1, limit[2]) in skiaTemplate
        registration = plate_generator.generate_registration("@@## @@@")
        self.assertEqual(registration[0:4], registration.split(" ")[0])
        self.assertEqual(registration[5:8], registration.split(" ")[1])

        wide = plate_generator._family_text_params(config, "52-11", metrics, font)
        self.assertNotIn("lines", wide)

    def test_eu_horizontal_text_offset_shifts_the_band_aware_centre(self) -> None:
        font_path = plate_generator.resolve_font_path({"source": "default", "path": ""})
        font, metrics = plate_generator._plate_font_metrics(font_path)

        def text_x(**eu_overrides):
            cfg = plate_generator.normalized_plate_config({"size": "EU", "eu": eu_overrides})
            return plate_generator._family_text_params(cfg, "52-11", metrics, font)["x"]

        self.assertEqual(text_x(sideBand="none"), 0.5)
        self.assertEqual(text_x(sideBand="none", textX=0.2), 0.7)
        # band compensation still applies underneath the user offset
        band_centre = 0.5 + plate_generator._EU_BAND_FRACTION / 2
        self.assertEqual(text_x(sideBand="eu", textX=-0.1), round(band_centre - 0.1, 4))
        # out-of-range values clamp instead of pushing text off the plate
        self.assertEqual(text_x(sideBand="none", textX=9), 0.9)

    def test_export_all_plate_sets_is_none_for_empty_library(self) -> None:
        self.assertIsNone(plate_generator.export_all_plate_sets())

    def test_export_all_plate_sets_reuses_unchanged_cached_zip(self) -> None:
        config = plate_generator.default_plate_config()
        config["enabled"] = True
        plate_generator.save_plate_set({"id": "set-one", "name": "Set One", "config": config})

        first = plate_generator.export_all_plate_sets()
        self.assertIsNotNone(first)
        assert first is not None
        self.assertTrue(Path(first["zip"]).is_file())

        with patch.object(plate_generator, "export_plate_sets", side_effect=AssertionError("unexpected rebuild")):
            second = plate_generator.export_all_plate_sets()

        self.assertEqual(second["zip"], first["zip"])
        self.assertTrue(second.get("cached"))
        self.assertEqual(second["designs"], 1)

    def test_install_refreshes_universal_plates_mod_with_all_library_sets(self) -> None:
        for set_id, name in (("set-one", "Set One"), ("set-two", "Set Two")):
            config = plate_generator.default_plate_config()
            config["enabled"] = True
            plate_generator.save_plate_set({"id": set_id, "name": name, "config": config})

        context = self._context(with_plate_parts=True)
        conversion = core.base_conversion_config(context)
        settings = conversion["variants"]["base"]
        core.set_variant_build_mode(settings, core.BUILD_ORIGINAL)
        settings["frontPlate"] = "mesh:licenseplate-52-11"

        mods_folder = self.root / "mods"
        result = core.build_batch(
            context,
            conversion,
            write_zip=True,
            install=True,
            mods_folder=mods_folder,
        )
        self.assertEqual(result.installed_plates_zip, mods_folder / "BeamXP_plates.zip")
        self.assertTrue(result.installed_plates_zip.is_file())
        self.assertEqual(result.package_zip, result.installed_zip)
        self.assertFalse((context.project_dir / "build" / "test_XP_conversion.zip").exists())
        self.assertEqual(result.plate_summary.get("libraryModDesigns"), 2)
        self.assertTrue(result.plate_summary.get("libraryModInstalled"))
        with zipfile.ZipFile(result.installed_plates_zip) as archive:
            jbeam = archive.read("vehicles/common/licenseplates/bhdc_plate_sets.jbeam").decode("utf-8")
        parts = core.parse_beamng_json(jbeam, label="bhdc_plate_sets.jbeam")
        self.assertIn("bhdc_plateset_set_one", parts)
        self.assertIn("bhdc_plateset_set_two", parts)

        second = core.build_batch(
            context,
            conversion,
            write_zip=True,
            install=True,
            mods_folder=mods_folder,
        )
        self.assertFalse(second.plate_summary.get("libraryModInstalled"))


class BackgroundImageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def _image(self, name: str, size: tuple[int, int], color: tuple[int, int, int]) -> str:
        from PIL import Image

        path = self.root / name
        Image.new("RGB", size, color).save(path)
        return str(path)

    def test_legacy_us_bg_image_migrates_to_front_background(self) -> None:
        cfg = plate_generator.normalized_plate_config({"size": "US", "us": {"bgImage": "C:/old.png"}})
        self.assertEqual(cfg["background"]["frontImage"], "C:/old.png")
        self.assertEqual(cfg["us"]["bgImage"], "")

    def test_absent_rear_image_keeps_the_solid_rear_colour(self) -> None:
        front = self._image("front.png", (64, 32), (200, 30, 30))
        cfg = plate_generator.normalized_plate_config({"background": {"frontImage": front}})
        self.assertIsNone(plate_generator._user_background(cfg, (100, 50), rear=True))
        # front is an image, rear is a colour, so rear formats are required
        self.assertTrue(plate_generator._rear_texture_differs(cfg))

    def test_rear_format_emission_follows_image_mismatches(self) -> None:
        front = self._image("front.png", (64, 32), (200, 30, 30))
        rear = self._image("rear.png", (64, 32), (30, 30, 200))
        differs = plate_generator._rear_texture_differs
        self.assertTrue(differs(plate_generator.normalized_plate_config(
            {"background": {"frontImage": front, "rearImage": rear}})))
        self.assertTrue(differs(plate_generator.normalized_plate_config(
            {"background": {"rearImage": rear}})))
        self.assertFalse(differs(plate_generator.normalized_plate_config(
            {"background": {"frontImage": front, "rearImage": front}})))

    def test_background_image_scales_to_cover_and_centre_crops(self) -> None:
        from PIL import Image

        # 200x100 source: left half red, right half blue. Fitted onto a
        # square canvas the width overflows, so the crop must keep the
        # horizontal middle - both colours still present at the seam.
        path = self.root / "wide.png"
        image = Image.new("RGB", (200, 100), (200, 30, 30))
        image.paste((30, 30, 200), (100, 0, 200, 100))
        image.save(path)
        cfg = plate_generator.normalized_plate_config({"background": {"frontImage": str(path)}})
        out = plate_generator._user_background(cfg, (100, 100))
        self.assertEqual(out.size, (100, 100))
        left = out.getpixel((10, 50))
        right = out.getpixel((90, 50))
        self.assertGreater(left[0], left[2], "left of centred crop should stay red")
        self.assertGreater(right[2], right[0], "right of centred crop should stay blue")

    def test_background_image_renders_for_every_family(self) -> None:
        front = self._image("front.png", (300, 80), (10, 180, 60))
        for family in plate_generator.PLATE_SIZES:
            cfg = plate_generator.normalized_plate_config({
                "size": family,
                "background": {"frontImage": front},
                "jp": {"region": "TOKYO", "classification": "300", "kana": "A"},
            })
            preview = plate_generator.render_plate_preview(cfg, "AB12 CDE")
            centre = preview.getpixel((preview.width // 2, int(preview.height * 0.9)))
            self.assertGreater(centre[1], 120, f"{family} background should show the image")


if __name__ == "__main__":
    unittest.main()
