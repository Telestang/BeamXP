from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from beamxp import hand_drive_core as core
from beamxp.core.beam_json import display_name_for, info_path_for_config
from beamxp.core.files import vehicle_catalog_entry_for_id, vehicle_ids_in_zip


class VehicleCatalogTests(unittest.TestCase):
    def _zip_path(self, files: dict[str, str]) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "vehicles.zip"
        with zipfile.ZipFile(path, "w") as zf:
            for name, text in files.items():
                zf.writestr(name, text)
        return path

    def test_selector_subgroups_split_catalog_vehicles(self) -> None:
        zip_path = self._zip_path(
            {
                "vehicles/vivace/vivace.dae": "",
                "vehicles/vivace/vivace.jbeam": "",
                "vehicles/vivace/vivace_110_M.pc": "{}",
                "vehicles/vivace/info_vivace_110_M.json": """
                {
                  "Configuration": "vehiclesData.vivace.vivace_110_M.Configuration",
                  "vehicleSelectorSubGroup": "vehiclesData.vivace.vehicleSelectorSubCluster"
                }
                """,
                "vehicles/vivace/ardente_190d_DCT.pc": "{}",
                "vehicles/vivace/info_ardente_190d_DCT.json": """
                {
                  "Configuration": "vehiclesData.vivace.ardente_190d_A.Configuration",
                  "vehicleSelectorSubGroup": "vehiclesData.vivace.ardente_190d_DCT.vehicleSelectorSubGroup"
                }
                """,
                "vehicles/vivace/ardente_230_M.pc": "{}",
                "vehicles/vivace/info_ardente_230_M.json": """
                {
                  "Configuration": "vehiclesData.vivace.ardente_230_M.Configuration",
                  "vehicleSelectorSubGroup": "vehiclesData.vivace.ardente_190d_DCT.vehicleSelectorSubGroup"
                }
                """,
            }
        )

        self.assertEqual(vehicle_ids_in_zip(zip_path), ["ardente", "vivace"])
        ardente = vehicle_catalog_entry_for_id(zip_path, "ardente")
        self.assertIsNotNone(ardente)
        self.assertEqual(ardente.source_vehicle_id, "vivace")
        self.assertEqual(ardente.config_names, ("ardente_190d_DCT", "ardente_230_M"))

        vivace = vehicle_catalog_entry_for_id(zip_path, "vivace")
        self.assertIsNotNone(vivace)
        self.assertEqual(vivace.source_vehicle_id, "vivace")
        self.assertEqual(vivace.config_names, ("vivace_110_M",))

    def test_legacy_vehicle_folder_is_still_a_single_catalog_entry(self) -> None:
        zip_path = self._zip_path(
            {
                "vehicles/acme/acme.dae": "",
                "vehicles/acme/acme.jbeam": "",
                "vehicles/acme/base.pc": "{}",
            }
        )

        self.assertEqual(vehicle_ids_in_zip(zip_path), ["acme"])
        entry = vehicle_catalog_entry_for_id(zip_path, "acme")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.source_vehicle_id, "acme")
        self.assertEqual(entry.config_names, ())

    def test_localization_keys_fall_back_to_readable_display_names(self) -> None:
        zip_path = self._zip_path(
            {
                "vehicles/vivace/vivace.dae": "",
                "vehicles/vivace/vivace.jbeam": "",
                "vehicles/vivace/ardente_190d_DCT.pc": "{}",
                "vehicles/vivace/info_ardente_190d_DCT.json": """
                {
                  "Configuration": "vehiclesData.vivace.ardente_190d_A.Configuration"
                }
                """,
            }
        )

        info_path = info_path_for_config(zip_path, "vivace", "ardente_190d_DCT")
        self.assertEqual(
            display_name_for(zip_path, info_path, "ardente_190d_DCT"),
            "Ardente 190d A",
        )

    def test_generated_output_info_does_not_keep_localization_keys(self) -> None:
        variant = core.VariantInfo(
            "ardente_carabinieri",
            "vehicles/vivace/ardente_carabinieri.pc",
            "vehicles/vivace/info_ardente_carabinieri.json",
            "Ardente Carabinieri",
        )
        info = {
            "Configuration": "vehiclesData.vivace.ardente_carabinieri.Configuration",
            "Description": "vehiclesData.vivace.ardente_carabinieri.Description",
        }

        self.assertEqual(core.generated_info_display_name(info, variant), "Ardente Carabinieri")
        self.assertEqual(core.generated_info_description(info), "")


if __name__ == "__main__":
    unittest.main()
