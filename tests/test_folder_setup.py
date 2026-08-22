"""The launch-time folder prompt and the settings it reads and writes.

The dialog itself needs a display, so what is checked here is the logic it
turns on: which folders count as configured, when the prompt is owed, and
that both of its settings survive a restart.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from beamxp.hand_drive_parts import configuration
from beamxp.hand_drive_ui.folder_setup import DISMISSED_SETTING, FolderSetupMixin
from beamxp.hand_drive_ui.vehicle_browser import VehicleBrowserMixin


class FakeVar:
    def __init__(self, value: str = "") -> None:
        self._value = value

    def get(self) -> str:
        return self._value

    def set(self, value: str) -> None:
        self._value = value


class FakeApp(FolderSetupMixin, VehicleBrowserMixin):
    """Just enough of the app for the folder helpers, without a Tk window."""

    def __init__(self, settings: dict[str, object]) -> None:
        self.settings = settings
        self.mods_folder_var = FakeVar(str(settings.get("modsFolder") or ""))
        self.prompted = False

    def _open_folder_setup_dialog(self) -> None:
        self.prompted = True


class FolderConfigurationTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.game = self.root / "game"
        self.mods = self.root / "mods"
        self.game.mkdir()
        self.mods.mkdir()

    def app(self, **settings: object) -> FakeApp:
        return FakeApp(dict(settings))

    def test_a_folder_that_is_not_there_is_not_configured(self) -> None:
        # The mods setting defaults to BeamNG's install path whether or not the
        # game is installed, so a non-empty setting proves nothing.
        app = self.app(modsFolder=str(self.root / "nope"))
        self.assertFalse(app._folder_is_usable(app._mods_folder()))
        self.assertIn("mods", app._missing_folder_names())

    def test_an_existing_folder_is_configured(self) -> None:
        app = self.app(modsFolder=str(self.mods), gameVehiclesFolder=str(self.game))
        with patch("beamxp.hand_drive_ui.vehicle_browser.core.default_beamng_vehicles_dir", return_value=None):
            self.assertEqual(app._missing_folder_names(), [])

    def test_both_folders_are_reported_when_neither_is_set(self) -> None:
        app = self.app()
        with patch("beamxp.hand_drive_ui.vehicle_browser.core.default_beamng_vehicles_dir", return_value=None):
            self.assertEqual(app._missing_folder_names(), ["game vehicles", "mods"])

    def test_a_detected_game_folder_counts_as_configured(self) -> None:
        app = self.app(modsFolder=str(self.mods))
        with patch(
            "beamxp.hand_drive_ui.vehicle_browser.core.default_beamng_vehicles_dir",
            return_value=self.game,
        ):
            self.assertEqual(app._missing_folder_names(), [])
        self.assertEqual(app.settings["gameVehiclesFolder"], str(self.game))

    def test_a_remembered_game_folder_that_has_gone_falls_back_to_detection(self) -> None:
        app = self.app(gameVehiclesFolder=str(self.root / "moved"))
        with patch(
            "beamxp.hand_drive_ui.vehicle_browser.core.default_beamng_vehicles_dir",
            return_value=self.game,
        ):
            self.assertEqual(app._game_vehicles_folder(), self.game)

    def test_the_button_says_set_folder_when_the_path_is_gone(self) -> None:
        missing = self.root / "nope"
        self.assertEqual(
            VehicleBrowserMixin._folder_button_text("Mods", missing), "Mods: set folder..."
        )
        self.assertTrue(
            VehicleBrowserMixin._folder_button_text("Mods", self.mods).endswith(self.mods.name)
        )


class FolderPromptTests(FolderConfigurationTests):
    def test_the_prompt_opens_when_a_folder_is_missing(self) -> None:
        app = self.app(modsFolder=str(self.mods))
        with patch("beamxp.hand_drive_ui.vehicle_browser.core.default_beamng_vehicles_dir", return_value=None):
            app._maybe_prompt_for_folders()
        self.assertTrue(app.prompted)

    def test_the_prompt_stays_shut_when_both_folders_are_known(self) -> None:
        app = self.app(modsFolder=str(self.mods), gameVehiclesFolder=str(self.game))
        with patch("beamxp.hand_drive_ui.vehicle_browser.core.default_beamng_vehicles_dir", return_value=None):
            app._maybe_prompt_for_folders()
        self.assertFalse(app.prompted)

    def test_a_dismissed_prompt_does_not_come_back(self) -> None:
        app = self.app(**{DISMISSED_SETTING: True})
        with patch("beamxp.hand_drive_ui.vehicle_browser.core.default_beamng_vehicles_dir", return_value=None):
            app._maybe_prompt_for_folders()
        self.assertFalse(app.prompted)


class AppSettingsPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "hand_drive_tool_settings.json"
        patcher = patch.object(configuration, "APP_SETTINGS_PATH", self.path)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_chosen_game_folder_survives_a_restart(self) -> None:
        settings = configuration.load_app_settings()
        settings["gameVehiclesFolder"] = r"D:\Games\BeamNG\content\vehicles"
        configuration.save_app_settings(settings)
        self.assertEqual(
            configuration.load_app_settings()["gameVehiclesFolder"],
            r"D:\Games\BeamNG\content\vehicles",
        )

    def test_dismissing_the_prompt_survives_a_restart(self) -> None:
        settings = configuration.load_app_settings()
        self.assertFalse(settings[DISMISSED_SETTING])
        settings[DISMISSED_SETTING] = True
        configuration.save_app_settings(settings)
        self.assertTrue(configuration.load_app_settings()[DISMISSED_SETTING])

    def test_an_unreadable_settings_file_still_loads_defaults(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("{ not json", encoding="utf-8")
        settings = configuration.load_app_settings()
        self.assertEqual(settings["gameVehiclesFolder"], "")
        self.assertFalse(settings[DISMISSED_SETTING])

    def test_settings_written_before_these_keys_existed_still_load(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"modsFolder": r"C:\mods"}), encoding="utf-8")
        settings = configuration.load_app_settings()
        self.assertEqual(settings["modsFolder"], r"C:\mods")
        self.assertEqual(settings["gameVehiclesFolder"], "")
        self.assertFalse(settings[DISMISSED_SETTING])


if __name__ == "__main__":
    unittest.main()
