"""How the Model dropdown presents a vehicle, whichever route it took.

A folder-scanned vehicle carries its name and preview on its listing; a zip
opened through Load Zip has no listing at all, and used to show the bare
vehicle id and no picture. Both routes now read the same catalog entry, so
the thumbnail and the label are tested through the methods the widgets call.
"""

from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from beamxp.core.inventory import scan_vehicle_inventory
from beamxp.hand_drive_ui.vehicle_browser import VehicleBrowserMixin
from beamxp.hand_drive_ui.vehicle_workflow import VehicleWorkflowMixin

PNG_1PX = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753"
    "de0000000c4944415408d763f8ffff3f0005fe02fea735c0000000000049454e44ae426082"
)
JPG_1PX = bytes.fromhex("ffd8ffe000104a46494600010100000100010000ffd9")


def vehicle_zip(
    folder: Path,
    name: str,
    vehicle_id: str,
    info: str,
    *,
    preview: bytes | None = PNG_1PX,
) -> Path:
    path = folder / name
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"vehicles/{vehicle_id}/{vehicle_id}.dae", "")
        archive.writestr(f"vehicles/{vehicle_id}/{vehicle_id}.jbeam", "")
        if info:
            archive.writestr(f"vehicles/{vehicle_id}/info.json", info)
        archive.writestr(f"vehicles/{vehicle_id}/base.pc", "{}")
        if preview is not None:
            archive.writestr(f"vehicles/{vehicle_id}/default.png", preview)
    return path


class Harness(VehicleBrowserMixin, VehicleWorkflowMixin):
    """The dropdown's preview and label lookups, with no widgets behind them."""

    def __init__(
        self,
        listings: list[object],
        entries: dict[str, tuple[Path, str]],
        display_names: dict[str, str] | None = None,
    ) -> None:
        self.vehicle_listings = listings
        self.model_entries = entries
        self.vehicle_display_names = display_names or {}


class ModelPreviewThumbnailTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.game = self.root / "game"
        self.loose = self.root / "loose"
        self.game.mkdir()
        self.loose.mkdir()
        vehicle_zip(self.game, "acme.zip", "acme", '{"Brand":"Acme","Name":"Rocket","Type":"Car"}')

    def harness(self, entries: dict[str, tuple[Path, str]], *, scan: bool = True) -> Harness:
        listings = list(scan_vehicle_inventory(self.game, None)) if scan else []
        return Harness(listings, entries)

    def test_a_folder_scanned_vehicle_uses_its_listing(self) -> None:
        app = self.harness({"Acme Rocket": (self.game / "acme.zip", "acme")})
        self.assertEqual(app._preview_bytes_for_label("Acme Rocket"), PNG_1PX)

    def test_a_hand_opened_zip_reads_its_own_preview(self) -> None:
        # The case this whole path exists for: no listing, so the old lookup
        # gave up and the thumbnail read "no preview".
        path = vehicle_zip(self.loose, "hand.zip", "hand", '{"Name":"Hand","Type":"Car"}', preview=JPG_1PX)
        app = self.harness({"Hand [imported]": (path, "hand")})
        self.assertIsNone(app._listing_for_label("Hand [imported]"))
        self.assertEqual(app._preview_bytes_for_label("Hand [imported]"), JPG_1PX)

    def test_a_hand_opened_mod_borrows_the_stock_preview(self) -> None:
        # A mod extending a stock vehicle ships no preview of its own; the
        # folder scan lends it the stock image and so does this path.
        path = vehicle_zip(self.loose, "acmemod.zip", "acme", "", preview=None)
        app = self.harness({"Acme Rocket [imported]": (path, "acme")})
        self.assertEqual(app._preview_bytes_for_label("Acme Rocket [imported]"), PNG_1PX)

    def test_nothing_to_borrow_leaves_no_preview(self) -> None:
        path = vehicle_zip(self.loose, "bare.zip", "bare", '{"Name":"Bare","Type":"Car"}', preview=None)
        app = self.harness({"Bare [imported]": (path, "bare")}, scan=False)
        self.assertIsNone(app._preview_bytes_for_label("Bare [imported]"))

    def test_a_hand_opened_zip_is_labelled_by_its_display_name(self) -> None:
        # The folder scan would call this "Lexus LC"; opening the same zip by
        # hand used to fall back to the bare vehicle id.
        app = self.harness({}, scan=False)
        app.vehicle_display_names = {"lc500": "Lexus LC"}
        self.assertEqual(app._model_history_label("lc500", {}), "Lexus LC [imported]")

    def test_a_zip_with_no_model_info_still_falls_back_to_the_id(self) -> None:
        app = self.harness({}, scan=False)
        self.assertEqual(app._model_history_label("oldmod", {}), "oldmod [imported]")

    def test_a_clashing_label_is_still_numbered(self) -> None:
        # Same numbering a second mod of one vehicle gets: "... [mod] #2".
        app = self.harness({}, scan=False)
        app.vehicle_display_names = {"lc500": "Lexus LC"}
        taken = {"Lexus LC [imported]": object()}
        self.assertEqual(app._model_history_label("lc500", taken), "Lexus LC [imported] #2")

    def test_an_unknown_label_has_no_preview(self) -> None:
        app = self.harness({})
        self.assertIsNone(app._preview_bytes_for_label("nothing here"))

    def test_a_zip_that_has_gone_does_not_raise(self) -> None:
        # Recents keep entries for unplugged drives, and the dropdown previews
        # a highlighted row before the user commits to it.
        app = self.harness({"gone [imported]": (self.loose / "gone.zip", "gone")}, scan=False)
        self.assertIsNone(app._preview_bytes_for_label("gone [imported]"))


if __name__ == "__main__":
    unittest.main()
