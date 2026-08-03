from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from beamxp.core.inventory import (
    read_preview_image_bytes,
    scan_vehicle_inventory,
)

PNG_1PX = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753"
    "de0000000c4944415408d763f8ffff3f0005fe02fea735c0000000000049454e44ae426082"
)


def vehicle_zip(folder: Path, name: str, vehicle_id: str, info: str, *, preview: bool = True) -> Path:
    path = folder / name
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"vehicles/{vehicle_id}/{vehicle_id}.dae", "")
        archive.writestr(f"vehicles/{vehicle_id}/{vehicle_id}.jbeam", "")
        archive.writestr(f"vehicles/{vehicle_id}/info.json", info)
        archive.writestr(f"vehicles/{vehicle_id}/base.pc", "{}")
        if preview:
            archive.writestr(f"vehicles/{vehicle_id}/default.png", PNG_1PX)
    return path


class VehicleInventoryTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.game = self.root / "game"
        self.mods = self.root / "mods"
        self.game.mkdir()
        (self.mods / "repo").mkdir(parents=True)

        vehicle_zip(self.game, "acme.zip", "acme", '{"Brand":"Acme","Name":"Rocket","Type":"Car"}')
        vehicle_zip(self.game, "hauler.zip", "hauler", '{"Brand":"Acme","Name":"Hauler","Type":"Truck"}')
        vehicle_zip(self.game, "cone.zip", "cone", '{"Name":"Cone","Type":"Prop"}')
        vehicle_zip(self.game, "digger.zip", "digger", '{"Name":"Digger","Type":"Heavy Machinery"}')
        # Mods live in subfolders as often as not.
        vehicle_zip(self.mods / "repo", "zippy.zip", "zippy", '{"Brand":"Zip","Name":"Zippy","Type":"Car"}')
        vehicle_zip(self.mods, "auto.zip", "auto", '{"Name":"Auto Export","Type":"Automation"}')

    def test_lists_only_cars_and_trucks_by_default(self) -> None:
        items = scan_vehicle_inventory(self.game, self.mods)
        self.assertEqual(
            [item.label() for item in items],
            ["Acme Hauler", "Acme Rocket", "Zip Zippy [mod]"],
        )

    def test_automation_is_opt_in(self) -> None:
        items = scan_vehicle_inventory(self.game, self.mods, include_automation=True)
        self.assertIn("Auto Export [mod]", [item.label() for item in items])

    def test_props_and_machinery_are_never_listed(self) -> None:
        labels = [
            item.label()
            for item in scan_vehicle_inventory(self.game, self.mods, include_automation=True)
        ]
        self.assertNotIn("Cone", labels)
        self.assertNotIn("Digger", labels)

    def test_a_mod_of_a_stock_vehicle_is_listed_alongside_it(self) -> None:
        vehicle_zip(self.mods, "acme_tuned.zip", "acme", '{"Brand":"Acme","Name":"Rocket","Type":"Car"}')
        items = scan_vehicle_inventory(self.game, self.mods)
        acme = [item for item in items if item.vehicle_id == "acme"]
        self.assertEqual([item.label() for item in acme], ["Acme Rocket", "Acme Rocket [mod]"])
        self.assertEqual([item.is_mod for item in acme], [False, True])

    def test_a_mod_without_model_info_inherits_the_stock_identity(self) -> None:
        # A bodykit/config mod ships trims for a stock vehicle but no model
        # info.json of its own, so its name, type and preview come from stock.
        path = self.mods / "acme_bodykit.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("vehicles/acme/acme_kit.dae", "")
            archive.writestr("vehicles/acme/acme_kit.jbeam", "")
            archive.writestr("vehicles/acme/kitted.pc", "{}")

        items = scan_vehicle_inventory(self.game, self.mods)
        kit = next(item for item in items if item.source_zip.name == "acme_bodykit.zip")
        self.assertEqual(kit.label(), "Acme Rocket [mod]")
        self.assertEqual(kit.vehicle_type, "Car")
        # The image lives in the stock zip, not the mod.
        self.assertEqual(kit.preview_zip.name, "acme.zip")
        self.assertEqual(read_preview_image_bytes(kit), PNG_1PX)

    def test_several_mods_of_one_vehicle_are_numbered(self) -> None:
        vehicle_zip(self.mods, "acme_a.zip", "acme", '{"Brand":"Acme","Name":"Rocket","Type":"Car"}')
        vehicle_zip(self.mods, "acme_b.zip", "acme", '{"Brand":"Acme","Name":"Rocket","Type":"Car"}')
        labels = [item.label() for item in scan_vehicle_inventory(self.game, self.mods)]
        self.assertIn("Acme Rocket", labels)
        self.assertIn("Acme Rocket [mod] #1", labels)
        self.assertIn("Acme Rocket [mod] #2", labels)
        self.assertNotIn("Acme Rocket [mod]", labels)

    def test_the_tools_own_output_is_never_offered_as_a_source(self) -> None:
        path = self.mods / "acme_XP_conversion.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("vehicles/acme/acme.dae", "")
            archive.writestr("vehicles/acme/acme.jbeam", "")
            archive.writestr("vehicles/acme/base_rhd.pc", "{}")
            archive.writestr("handedness_conversion/conversion.json", "{}")

        items = scan_vehicle_inventory(self.game, self.mods)
        self.assertEqual(
            [item.source_zip.name for item in items if item.vehicle_id == "acme"], ["acme.zip"]
        )

    def test_a_parts_only_overlay_is_not_listed(self) -> None:
        # A plate or livery pack ships meshes for a vehicle but no trim of its
        # own, so there is nothing to convert and it must not clutter the list.
        path = self.mods / "plates.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("vehicles/acme/plate.dae", "")
            archive.writestr("vehicles/acme/plate.jbeam", "")

        items = scan_vehicle_inventory(self.game, self.mods)
        self.assertNotIn("plates.zip", [item.source_zip.name for item in items])

    def test_two_stock_vehicles_sharing_a_name_are_told_apart(self) -> None:
        # Stock does this: midsize and pessima both resolve to "Ibishu Pessima".
        vehicle_zip(self.game, "rocket2.zip", "rocket_mk2", '{"Brand":"Acme","Name":"Rocket","Type":"Car"}')
        labels = [item.label() for item in scan_vehicle_inventory(self.game, self.mods)]
        self.assertIn("Acme Rocket (acme)", labels)
        self.assertIn("Acme Rocket (rocket_mk2)", labels)
        self.assertNotIn("Acme Rocket", labels)

    def test_every_label_is_distinct(self) -> None:
        vehicle_zip(self.game, "rocket2.zip", "rocket_mk2", '{"Brand":"Acme","Name":"Rocket","Type":"Car"}')
        vehicle_zip(self.mods, "acme_a.zip", "acme", '{"Brand":"Acme","Name":"Rocket","Type":"Car"}')
        vehicle_zip(self.mods, "acme_b.zip", "acme", '{"Brand":"Acme","Name":"Rocket","Type":"Car"}')
        labels = [item.label() for item in scan_vehicle_inventory(self.game, self.mods)]
        self.assertEqual(len(labels), len(set(labels)), labels)

    def test_the_common_parts_namespace_is_never_listed(self) -> None:
        vehicle_zip(self.mods, "shared.zip", "common", '{"Name":"Common","Type":"Car"}')
        items = scan_vehicle_inventory(self.game, self.mods)
        self.assertNotIn("common", [item.vehicle_id for item in items])

    def test_unreadable_zip_does_not_stop_the_scan(self) -> None:
        (self.game / "broken.zip").write_bytes(b"not a zip")
        items = scan_vehicle_inventory(self.game, self.mods)
        self.assertIn("Acme Rocket", [item.label() for item in items])

    def test_missing_folders_are_tolerated(self) -> None:
        self.assertEqual(scan_vehicle_inventory(self.root / "nope", None), [])
        items = scan_vehicle_inventory(self.game, self.root / "nope")
        self.assertEqual([item.label() for item in items], ["Acme Hauler", "Acme Rocket"])

    def test_preview_bytes_are_readable(self) -> None:
        items = scan_vehicle_inventory(self.game, self.mods)
        rocket = next(item for item in items if item.vehicle_id == "acme")
        self.assertEqual(read_preview_image_bytes(rocket), PNG_1PX)

    def test_missing_preview_returns_none(self) -> None:
        vehicle_zip(self.game, "bare.zip", "bare", '{"Name":"Bare","Type":"Car"}', preview=False)
        items = scan_vehicle_inventory(self.game, self.mods)
        bare = next(item for item in items if item.vehicle_id == "bare")
        self.assertIsNone(read_preview_image_bytes(bare))

    def test_display_name_falls_back_to_the_vehicle_id(self) -> None:
        # A mod with no model info.json at all still has to appear.
        path = self.game / "noinfo.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("vehicles/noinfo/noinfo.dae", "")
            archive.writestr("vehicles/noinfo/noinfo.jbeam", "")
            archive.writestr("vehicles/noinfo/base.pc", "{}")
        items = scan_vehicle_inventory(self.game, self.mods)
        # Type is unknown, so it is not a Car/Truck and stays out of the list.
        self.assertNotIn("noinfo", [item.vehicle_id for item in items])


if __name__ == "__main__":
    unittest.main()
