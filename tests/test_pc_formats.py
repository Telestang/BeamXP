from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from beamxp import hand_drive_core as core


class PcFormatTests(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name)
        self.source_zip = self.directory / "acme.zip"
        self.pc_path = "vehicles/acme/manual.pc"

    def load(self, document: dict[str, object]) -> dict[str, object]:
        with zipfile.ZipFile(self.source_zip, "w") as archive:
            archive.writestr(self.pc_path, json.dumps(document))
        return core.load_pc(self.source_zip, self.pc_path)

    def test_legacy_choices_resolve_and_survive_export(self) -> None:
        choices = {"engine": "v6", "transmission": "manual", "shifter": "stick", "spoiler": "none"}
        self.load(choices)
        bodies = {
            "acme": {"slotType": "main", "slots": [
                ["type", "default", "description"],
                ["engine", "i4", "Engine"],
                ["shifter", "auto_lever", "Shifter"],
                ["spoiler", "wing", "Spoiler"],
            ]},
            "i4": {"slotType": "engine"},
            "v6": {"slotType": "engine", "slots": [
                ["type", "default", "description"],
                ["transmission", "automatic", "Transmission"],
            ]},
            "manual": {"slotType": "transmission"},
            "automatic": {"slotType": "transmission"},
            "stick": {"slotType": "shifter"},
            "auto_lever": {"slotType": "shifter"},
            "wing": {"slotType": "spoiler"},
        }
        index = core.build_part_body_index({"vehicles/acme/acme.jbeam": json.dumps(bodies)})
        variant = core.VariantInfo("manual", self.pc_path, None, "Manual")
        context = core.VehicleContext(
            source_zip=self.source_zip, vehicle_id="acme", vehicle_path="vehicles/acme",
            dae_paths=[], variants={"manual": variant}, objects={}, preview_by_id={},
            jbeam_texts={}, node_positions={}, project_dir=self.directory, part_body_index=index,
        )
        selected = core.selected_parts_for_config(context, "manual")
        self.assertEqual(selected["parts"], {"acme", "v6", "manual", "stick"})
        self.assertEqual(selected["missing_parts"], set())

        pc = core.load_context_pc(context, self.pc_path)
        output = self.directory / "output"
        output_name = core.write_converted_config(context, output, variant, "manual", core.HAND_RHD, pc)
        exported = core.parse_beamng_json(
            (output / f"{output_name}.pc").read_text(encoding="utf-8"), label="export",
        )
        self.assertEqual(exported["format"], 2)
        self.assertEqual(exported["parts"], choices)
        self.assertNotIn("shifter", exported)
        self.assertEqual(
            core.resolve_selected_parts(exported, {}, vehicle_id="acme", part_body_index=index)["parts"],
            selected["parts"],
        )
        self.assertNotIn("licenseName", core.load_context_pc(context, self.pc_path))

    def test_legacy_nested_values_stay_in_parts_as_in_the_game(self) -> None:
        # 2026-09-09: LX_M includes vars inside its legacy part map. The game's
        # wrapper does not promote that entry to config-level variable overrides.
        document = {"shifter": "stick", "vars": {"$headlightPitchComp": 0.4}}
        self.assertEqual(self.load(document), {"format": 2, "parts": document})

    def test_structured_configs_keep_metadata_and_empty_parts(self) -> None:
        documents = [
            {"format": 2, "parts": {"shifter": "stick"}, "vars": {"$x": 1}, "paints": []},
            {"format": 2, "mainPartName": "acme", "vars": {"$x": 1}},
            {"parts": {}},
            {"parts": {"shifter": "stick"}},
            {"partsTree": {"chosenPartName": "acme", "children": {}}},
        ]
        for document in documents:
            with self.subTest(document=document):
                self.assertEqual(self.load(document), document)


if __name__ == "__main__":
    unittest.main()
