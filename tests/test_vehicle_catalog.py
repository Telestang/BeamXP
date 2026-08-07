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

    # The engine clusters selector tiles as
    #   config.model_key .. (config.useSubCluster and " " .. subCluster or "")
    # (lua/ge/extensions/ui/vehicleSelector/tileClustering.lua), so a vehicle is
    # one entry unless its model opts in via useSubCluster. vehicleSelectorSubGroup
    # is only a heading inside a tile ("Sedan", "Custom") and never splits.
    def _vivace_files(self, model_info: str) -> dict[str, str]:
        return {
            "vehicles/vivace/vivace.dae": "",
            "vehicles/vivace/vivace.jbeam": "",
            "vehicles/vivace/info.json": model_info,
            "vehicles/vivace/vivace_110_M.pc": "{}",
            "vehicles/vivace/info_vivace_110_M.json": """
            {
              "Configuration": "vehiclesData.vivace.vivace_110_M.Configuration",
              "vehicleSelectorSubCluster": "vehiclesData.vivace.vehicleSelectorSubCluster",
              "vehicleSelectorSubGroup": "vehiclesData.vivace.vehicleSelectorSubGroup"
            }
            """,
            "vehicles/vivace/ardente_190d_DCT.pc": "{}",
            "vehicles/vivace/info_ardente_190d_DCT.json": """
            {
              "Configuration": "vehiclesData.vivace.ardente_190d_A.Configuration",
              "vehicleSelectorSubCluster": "vehiclesData.vivace.ardente_190d_DCT.vehicleSelectorSubCluster",
              "vehicleSelectorSubGroup": "vehiclesData.vivace.ardente_190d_DCT.vehicleSelectorSubGroup"
            }
            """,
            "vehicles/vivace/ardente_230_M.pc": "{}",
            "vehicles/vivace/info_ardente_230_M.json": """
            {
              "Configuration": "vehiclesData.vivace.ardente_230_M.Configuration",
              "vehicleSelectorSubCluster": "vehiclesData.vivace.ardente_190d_DCT.vehicleSelectorSubCluster",
              "vehicleSelectorSubGroup": "vehiclesData.vivace.ardente_190d_DCT.vehicleSelectorSubGroup"
            }
            """,
        }

    def test_sub_cluster_opt_in_splits_catalog_vehicles(self) -> None:
        zip_path = self._zip_path(
            self._vivace_files('{"useSubCluster": true, "default_pc": "vivace_110_M"}')
        )

        self.assertEqual(vehicle_ids_in_zip(zip_path), ["vivace", "vivace_ardente_190d_dct"])
        # default_pc keeps its cluster on the plain vehicle id, so the primary
        # tile's project path never moves when BeamNG adds a sub-cluster.
        vivace = vehicle_catalog_entry_for_id(zip_path, "vivace")
        self.assertIsNotNone(vivace)
        self.assertEqual(vivace.source_vehicle_id, "vivace")
        self.assertEqual(vivace.config_names, ("vivace_110_M",))

        ardente = vehicle_catalog_entry_for_id(zip_path, "vivace_ardente_190d_dct")
        self.assertIsNotNone(ardente)
        self.assertEqual(ardente.source_vehicle_id, "vivace")
        self.assertEqual(ardente.config_names, ("ardente_190d_DCT", "ardente_230_M"))

    def test_sub_group_alone_does_not_split_catalog_vehicles(self) -> None:
        # Without useSubCluster the model is one tile, however many subgroups its
        # configs carry. BeamNG 0.39 added subgroups to 31 stock vehicles; reading
        # them as identity fragmented etk800 into four entries of 9/12/7/1 trims.
        zip_path = self._zip_path(self._vivace_files("{}"))

        self.assertEqual(vehicle_ids_in_zip(zip_path), ["vivace"])
        entry = vehicle_catalog_entry_for_id(zip_path, "vivace")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.source_vehicle_id, "vivace")
        # No filter: every trim in the folder stays reachable.
        self.assertEqual(entry.config_names, ())

    def test_config_only_sub_cluster_override_still_splits(self) -> None:
        # vehicle_catalog_entries_in_zip skips parsing config info json when the
        # model stays silent about useSubCluster, guarded by a raw-bytes check.
        # A config that opts in on its own must still be found by that guard.
        files = self._vivace_files("{}")
        files["vehicles/vivace/info_ardente_190d_DCT.json"] = """
        {
          "useSubCluster": true,
          "vehicleSelectorSubCluster": "vehiclesData.vivace.ardente_190d_DCT.vehicleSelectorSubCluster"
        }
        """
        zip_path = self._zip_path(files)

        self.assertEqual(
            vehicle_ids_in_zip(zip_path), ["vivace", "vivace_ardente_190d_dct"]
        )

    def test_display_name_and_preview_come_from_the_model(self) -> None:
        zip_path = self._zip_path(
            {
                "vehicles/acme/acme.dae": "",
                "vehicles/acme/acme.jbeam": "",
                "vehicles/acme/info.json": '{"Brand": "Acme", "Name": "Rocket", "Type": "Car"}',
                "vehicles/acme/default.jpg": "",
                "vehicles/acme/base.pc": "{}",
            }
        )

        entry = vehicle_catalog_entry_for_id(zip_path, "acme")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.display_name, "Acme Rocket")
        self.assertEqual(entry.vehicle_type, "Car")
        self.assertEqual(entry.preview_member, "vehicles/acme/default.jpg")

    def test_preview_falls_back_to_default_png_then_a_config_image(self) -> None:
        # Mods commonly ship default.png rather than stock's default.jpg.
        png = self._zip_path(
            {
                "vehicles/acme/acme.dae": "",
                "vehicles/acme/acme.jbeam": "",
                "vehicles/acme/info.json": '{"Name": "Rocket", "Type": "Car"}',
                "vehicles/acme/default.png": "",
                "vehicles/acme/base.pc": "{}",
            }
        )
        self.assertEqual(
            vehicle_catalog_entry_for_id(png, "acme").preview_member,
            "vehicles/acme/default.png",
        )

        config_only = self._zip_path(
            {
                "vehicles/acme/acme.dae": "",
                "vehicles/acme/acme.jbeam": "",
                "vehicles/acme/info.json": '{"Name": "Rocket", "Type": "Car"}',
                "vehicles/acme/base.png": "",
                "vehicles/acme/base.pc": "{}",
            }
        )
        self.assertEqual(
            vehicle_catalog_entry_for_id(config_only, "acme").preview_member,
            "vehicles/acme/base.png",
        )

    def test_shared_sub_group_key_from_another_vehicle_is_ignored(self) -> None:
        # Stock 0.39 tags 272 configs across barstow/bastion/bolide/etk800 with
        # vehiclesData.autobello.150_rally.vehicleSelectorSubGroup, which simply
        # translates to "Custom". It must not pull them into an autobello entry.
        zip_path = self._zip_path(
            {
                "vehicles/etk800/etk800.dae": "",
                "vehicles/etk800/etk800.jbeam": "",
                "vehicles/etk800/844_150_A.pc": "{}",
                "vehicles/etk800/info_844_150_A.json": """
                {"vehicleSelectorSubGroup": "vehiclesData.etk800.844_150_A.vehicleSelectorSubGroup"}
                """,
                "vehicles/etk800/844_police_A.pc": "{}",
                "vehicles/etk800/info_844_police_A.json": """
                {"vehicleSelectorSubGroup": "vehiclesData.autobello.150_rally.vehicleSelectorSubGroup"}
                """,
            }
        )

        self.assertEqual(vehicle_ids_in_zip(zip_path), ["etk800"])
        entry = vehicle_catalog_entry_for_id(zip_path, "etk800")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.config_names, ())

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

    # A config's authored Configuration is a locale key that the engine puts
    # through _tr before the selector shows it (core/vehicles.lua), so the
    # string in the locale table is the name the player reads. Prettifying the
    # key's own config segment is only a last resort: it can only ever produce
    # words that are already in the filename, which is how
    # ardente_S_410Q_M lost "Frisson".
    LOCALE_FILE = "locales/translations/en-US/acme.translation.json"

    def test_localization_keys_resolve_through_the_locale_table(self) -> None:
        zip_path = self._zip_path(
            {
                "vehicles/acme/acme.dae": "",
                "vehicles/acme/acme.jbeam": "",
                "vehicles/acme/sport_410Q_M.pc": "{}",
                "vehicles/acme/info_sport_410Q_M.json": """
                {
                  "Configuration": "vehiclesData.acme.sport_410Q_M.Configuration"
                }
                """,
                self.LOCALE_FILE: """
                {"vehiclesData.acme.sport_410Q_M.Configuration": "Sport 410Q Frisson (M)"}
                """,
            }
        )

        info_path = info_path_for_config(zip_path, "acme", "sport_410Q_M")
        self.assertEqual(
            display_name_for(zip_path, info_path, "sport_410Q_M"),
            "Sport 410Q Frisson (M)",
        )

    def test_an_unresolvable_key_still_falls_back_to_a_readable_name(self) -> None:
        zip_path = self._zip_path(
            {
                "vehicles/acme/acme.dae": "",
                "vehicles/acme/acme.jbeam": "",
                "vehicles/acme/sport_190d_DCT.pc": "{}",
                "vehicles/acme/info_sport_190d_DCT.json": """
                {
                  "Configuration": "vehiclesData.acme.sport_190d_A.Configuration"
                }
                """,
            }
        )

        info_path = info_path_for_config(zip_path, "acme", "sport_190d_DCT")
        self.assertEqual(
            display_name_for(zip_path, info_path, "sport_190d_DCT"),
            "Sport 190d A",
        )

    def test_generated_output_info_uses_the_translated_name(self) -> None:
        zip_path = self._zip_path(
            {
                "vehicles/acme/acme.dae": "",
                "vehicles/acme/acme.jbeam": "",
                self.LOCALE_FILE: """
                {
                  "vehiclesData.acme.police.Configuration": "Sport Carabinieri (M)",
                  "vehiclesData.acme.police.Description": "A patrol car."
                }
                """,
            }
        )
        variant = core.VariantInfo(
            "police",
            "vehicles/acme/police.pc",
            "vehicles/acme/info_police.json",
            "Police",
        )
        info = {
            "Configuration": "vehiclesData.acme.police.Configuration",
            "Description": "vehiclesData.acme.police.Description",
        }

        self.assertEqual(
            core.generated_info_display_name(info, variant, zip_path),
            "Sport Carabinieri (M)",
        )
        self.assertEqual(core.generated_info_description(info, zip_path), "A patrol car.")

    def test_generated_output_info_never_keeps_an_unresolved_key(self) -> None:
        variant = core.VariantInfo(
            "carabinieri",
            "vehicles/acme/carabinieri.pc",
            "vehicles/acme/info_carabinieri.json",
            "Acme Carabinieri",
        )
        info = {
            "Configuration": "vehiclesData.acme.carabinieri.Configuration",
            "Description": "vehiclesData.acme.carabinieri.Description",
        }

        self.assertEqual(core.generated_info_display_name(info, variant), "Carabinieri")
        self.assertEqual(core.generated_info_description(info), "")


if __name__ == "__main__":
    unittest.main()
