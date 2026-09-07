"""What a saved project writes down, and what it deliberately does not.

Two properties, both of which a project file has to keep:

*Skip is implicit.* A mesh nobody touched is not written, and a mesh nobody
touched loads back as Skip anyway, because the loader rebuilds the defaults
before it overlays the save. Format 1 spelt every mesh out; on a real vehicle
that was nine in ten of them, and a saved project came out ten to thirty times
larger than the decisions in it.

*Nothing in it names the user.* conversion.json carried the absolute path to
the source zip, and every plate design carried absolute paths to its font and
artwork. Both named a home directory, and the build embedded a copy of the
whole thing inside the mod zip it shipped. The designs now live in plates.json
beside it, and paths inside a BeamXP directory are written as tokens.

A format-1 save still loads, and loads to the same conversion it always did.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from beamxp import hand_drive_core as core
from beamxp.plates import generator as plate_generator


def obj(object_id: str, x: float = 0.0) -> core.DaeObject:
    return core.DaeObject(
        id=object_id,
        name=object_id,
        dae_path="vehicle.dae",
        x=x,
        y=0.0,
        z=0.0,
        geometry_ids=(),
    )


class ConversionFormatTests(unittest.TestCase):
    def setUp(self) -> None:
        self.project = Path(tempfile.mkdtemp(prefix="bxp_cfg_"))
        self.context = core.VehicleContext(
            source_zip=Path(r"C:\Users\someone\mods\veh.zip"),
            vehicle_id="veh",
            vehicle_path="vehicles/veh",
            dae_paths=["vehicles/veh/veh.dae"],
            variants={
                name: core.VariantInfo(
                    name=name, pc_path=f"{name}.pc", info_path=None, display_name=name
                )
                for name in ("base", "sport")
            },
            objects={
                "veh_steer": obj("veh_steer", 0.4),
                "veh_dash": obj("veh_dash"),
                "veh_seat_FL": obj("veh_seat_FL", 0.4),
                "veh_wheel_FL": obj("veh_wheel_FL", 0.8),
            },
            preview_by_id={},
            jbeam_texts={},
            node_positions={},
            project_dir=self.project,
        )

    def saved(self) -> dict:
        return json.loads((self.project / "conversion.json").read_text(encoding="utf-8"))

    def plates(self) -> dict:
        return json.loads((self.project / "plates.json").read_text(encoding="utf-8"))

    # ---- skip is implicit -------------------------------------------------

    def test_an_untouched_mesh_is_not_written(self) -> None:
        conversion, created = core.load_or_create_conversion(self.context)
        self.assertFalse(created)
        core.save_conversion(self.context, conversion)
        self.assertNotIn("veh_dash", self.saved()["parts"])
        self.assertNotIn("veh_wheel_FL", self.saved()["parts"])

    def test_a_configured_mesh_is_written_with_only_what_changed(self) -> None:
        conversion, _ = core.load_or_create_conversion(self.context)
        conversion["parts"]["veh_dash"]["mode"] = core.MODE_MIRROR
        core.save_conversion(self.context, conversion)
        self.assertEqual(self.saved()["parts"]["veh_dash"], {"mode": core.MODE_MIRROR})

    def test_an_untouched_mesh_still_loads_as_skip(self) -> None:
        conversion, _ = core.load_or_create_conversion(self.context)
        core.save_conversion(self.context, conversion)
        reloaded, loaded = core.load_or_create_conversion(self.context)
        self.assertTrue(loaded)
        self.assertEqual(reloaded["parts"]["veh_dash"]["mode"], core.MODE_SKIP)
        self.assertEqual(sorted(reloaded["parts"]), sorted(self.context.objects))

    def test_an_untouched_trim_is_not_written(self) -> None:
        conversion, _ = core.load_or_create_conversion(self.context)
        core.set_variant_build_mode(conversion["variants"]["sport"], "converted")
        core.save_conversion(self.context, conversion)
        written = self.saved()["variants"]
        self.assertEqual(sorted(written), ["sport"])
        # "selected" travels with "build": an older build reads the other one.
        self.assertEqual(written["sport"]["build"], "converted")
        self.assertTrue(written["sport"]["selected"])

    def test_a_round_trip_reproduces_the_conversion_exactly(self) -> None:
        conversion, _ = core.load_or_create_conversion(self.context)
        conversion["parts"]["veh_dash"]["mode"] = core.MODE_MIRROR
        conversion["parts"]["veh_dash"][core.PART_TEXTURE_CORRECTION_KEY] = True
        conversion["parts"]["veh_seat_FL"]["viewerVisible"] = False
        core.set_variant_build_mode(conversion["variants"]["base"], "converted")
        core.set_side_pair(conversion, "veh_seat_FL", "veh_seat_FR", kind="seat")
        core.save_conversion(self.context, conversion)

        first = json.dumps(conversion, sort_keys=True, default=str)
        reloaded, _ = core.load_or_create_conversion(self.context)
        core.save_conversion(self.context, reloaded)
        self.assertEqual(json.dumps(reloaded, sort_keys=True, default=str), first)

    # ---- nothing names the user ------------------------------------------

    def test_the_source_block_carries_no_path(self) -> None:
        conversion, _ = core.load_or_create_conversion(self.context)
        core.save_conversion(self.context, conversion)
        source = self.saved()["source"]
        self.assertEqual(sorted(source), ["fileName", "vehicleId"])
        self.assertEqual(source["fileName"], "veh.zip")
        self.assertNotIn("someone", json.dumps(self.saved()))

    def test_plate_designs_live_beside_the_conversion_not_inside_it(self) -> None:
        conversion, _ = core.load_or_create_conversion(self.context)
        conversion["plate"]["config"]["eu"]["pattern"] = "@@## @@@"
        core.save_conversion(self.context, conversion)
        self.assertNotIn("plate", self.saved())
        self.assertEqual(self.plates()["plate"]["config"]["eu"]["pattern"], "@@## @@@")
        reloaded, _ = core.load_or_create_conversion(self.context)
        self.assertEqual(reloaded["plate"]["config"]["eu"]["pattern"], "@@## @@@")

    def test_a_font_inside_the_app_folder_is_written_as_a_token(self) -> None:
        font = core.USER_DATA_DIR / "fonts" / "Mandatory.otf"
        conversion, _ = core.load_or_create_conversion(self.context)
        conversion["plate"]["config"]["font"] = {"source": "custom", "path": str(font)}
        core.save_conversion(self.context, conversion)
        self.assertEqual(
            self.plates()["plate"]["config"]["font"]["path"], "{fonts}/Mandatory.otf"
        )
        reloaded, _ = core.load_or_create_conversion(self.context)
        self.assertEqual(reloaded["plate"]["config"]["font"]["path"], str(font))

    def test_artwork_outside_the_app_folder_keeps_its_path(self) -> None:
        """There is no portable spelling for it, and the editor must reopen it.
        Keeping the designs out of conversion.json is what covers this case."""
        art = r"C:\Users\someone\Downloads\band.png"
        conversion, _ = core.load_or_create_conversion(self.context)
        conversion["plate"]["config"]["eu"]["bandFullImage"] = art
        core.save_conversion(self.context, conversion)
        self.assertEqual(self.plates()["plate"]["config"]["eu"]["bandFullImage"], art)
        self.assertNotIn("someone", json.dumps(self.saved()))

    # ---- format 1 still loads --------------------------------------------

    def test_a_dense_format_1_save_loads_unchanged(self) -> None:
        legacy = {
            "toolVersion": 2,
            "source": {
                "fileName": "veh.zip",
                "sourcePath": r"C:\Users\someone\mods\veh.zip",
                "vehicleId": "veh",
                "daeFiles": ["vehicles/veh/veh.dae"],
                "configs": ["base", "sport"],
            },
            "variants": {
                "base": {"selected": True, "build": "converted", "sourceHandOverride": "Auto"},
                "sport": {"selected": False, "build": "off", "sourceHandOverride": "Auto"},
            },
            "parts": {
                object_id: {
                    "mode": core.MODE_MIRROR if object_id == "veh_dash" else core.MODE_SKIP,
                    "mirrorSource": None,
                    "translateOffset": None,
                    # An old build's spelling of Texture Fix, which no longer
                    # exists and must not survive the load.
                    "textureFlip": False,
                    "steeringRef": object_id == "veh_steer",
                    "viewerVisible": True,
                    "viewerSolo": False,
                }
                for object_id in self.context.objects
            },
            "slotPairs": [],
            "sidePairs": [],
            "triggers": [],
            "plate": {"mode": "general", "config": {"eu": {"pattern": "@@## @@@"}}},
            "delta": {"manual": True, "magnitude": 0.75},
        }
        (self.project / "conversion.json").write_text(json.dumps(legacy), encoding="utf-8")

        conversion, loaded = core.load_or_create_conversion(self.context)
        self.assertTrue(loaded)
        self.assertEqual(conversion["parts"]["veh_dash"]["mode"], core.MODE_MIRROR)
        self.assertEqual(conversion["parts"]["veh_steer"]["mode"], core.MODE_SKIP)
        self.assertTrue(conversion["parts"]["veh_steer"]["steeringRef"])
        self.assertEqual(core.variant_build_mode(conversion["variants"]["base"]), "converted")
        self.assertEqual(conversion["plate"]["config"]["eu"]["pattern"], "@@## @@@")
        self.assertTrue(conversion["delta"]["manual"])
        self.assertNotIn("textureFlip", conversion["parts"]["veh_dash"])

    def test_re_saving_a_format_1_project_migrates_it(self) -> None:
        self.test_a_dense_format_1_save_loads_unchanged()
        conversion, _ = core.load_or_create_conversion(self.context)
        core.save_conversion(self.context, conversion)
        self.assertEqual(self.saved()["configFormat"], core.CONVERSION_FORMAT)
        self.assertNotIn("sourcePath", self.saved()["source"])
        self.assertEqual(self.plates()["plate"]["config"]["eu"]["pattern"], "@@## @@@")
        again, _ = core.load_or_create_conversion(self.context)
        self.assertEqual(again["parts"]["veh_dash"]["mode"], core.MODE_MIRROR)
        self.assertEqual(again["plate"]["config"]["eu"]["pattern"], "@@## @@@")

    def test_an_unreadable_plates_file_does_not_stop_the_conversion_opening(self) -> None:
        conversion, _ = core.load_or_create_conversion(self.context)
        core.save_conversion(self.context, conversion)
        (self.project / "plates.json").write_text("{ not json", encoding="utf-8")
        reloaded, loaded = core.load_or_create_conversion(self.context)
        self.assertTrue(loaded)
        self.assertIn("plate", reloaded)


class ShareablePathTests(unittest.TestCase):
    """The stricter policy for the copy the build ships inside the mod zip."""

    def test_a_path_under_a_beamxp_folder_becomes_a_token(self) -> None:
        self.assertEqual(
            core.shareable_path(str(core.USER_DATA_DIR / "fonts" / "Mandatory.otf")),
            "{fonts}/Mandatory.otf",
        )

    def test_a_project_path_is_named_relative_to_the_project(self) -> None:
        project = core.USER_DATA_DIR / "handedness_conversion_projects" / "veh"
        self.assertEqual(
            core.shareable_path(str(project / "build" / "a.dae"), project),
            "{project}/build/a.dae",
        )

    def test_any_other_absolute_path_is_cut_back_to_its_file_name(self) -> None:
        self.assertEqual(
            core.shareable_path(r"C:\Users\someone\mods\astra_modland.zip"),
            "astra_modland.zip",
        )
        # A UNC share, in both spellings. The separator split is done by hand
        # rather than through PurePath, which only understands a backslash on
        # Windows and returns nothing at all for the root of a share.
        self.assertEqual(core.shareable_path(r"\\nas\share\band.png"), "band.png")
        self.assertEqual(core.shareable_path("//nas/share/band.png"), "band.png")

    def test_a_jbeam_path_is_left_alone(self) -> None:
        """"/vehicles/common/" is a directory the game searches, not a place
        on this disk; cutting it to "common" would gut the diagnostics."""
        for text in ("/vehicles/common/steeringwheels.DAE", "vehicles/veh/veh.jbeam"):
            self.assertEqual(core.shareable_path(text), text)

    def test_things_that_are_not_paths_are_left_alone(self) -> None:
        for text in ("#ffffff", "@@## @@@", "etkc_doorpanel_L@@/etkc_body/etkc_door_L/", ""):
            self.assertEqual(core.shareable_path(text), text)


class PlateLibraryTests(unittest.TestCase):
    """A plate set is the most shared thing the tool makes: the library
    exports the lot as one zip, so its designs get the same treatment."""

    def setUp(self) -> None:
        self.data = Path(tempfile.mkdtemp(prefix="bxp_lib_"))
        self.previous = os.environ.get("BEAMXP_DATA_DIR")
        os.environ["BEAMXP_DATA_DIR"] = str(self.data)

    def tearDown(self) -> None:
        if self.previous is None:
            os.environ.pop("BEAMXP_DATA_DIR", None)
        else:
            os.environ["BEAMXP_DATA_DIR"] = self.previous

    def test_a_library_font_is_stored_as_a_token_and_reads_back_whole(self) -> None:
        font = self.data / "fonts" / "Mandatory.otf"
        plate_generator.save_plate_set({
            "id": "uk",
            "name": "UK",
            "config": {"font": {"source": "custom", "path": str(font)}},
        })
        raw = json.loads((self.data / "plates" / "uk.json").read_text(encoding="utf-8"))
        self.assertEqual(raw["config"]["font"]["path"], "{fonts}/Mandatory.otf")
        loaded = plate_generator.plate_set_by_id("uk")
        self.assertEqual(loaded["config"]["font"]["path"], str(font))

    def test_artwork_outside_the_app_folder_keeps_its_path(self) -> None:
        art = r"C:\Users\someone\Downloads\band.png"
        plate_generator.save_plate_set({
            "id": "uk",
            "name": "UK",
            "config": {"eu": {"bandFullImage": art}},
        })
        raw = json.loads((self.data / "plates" / "uk.json").read_text(encoding="utf-8"))
        self.assertEqual(raw["config"]["eu"]["bandFullImage"], art)

    def test_a_set_saved_by_an_older_build_still_reads(self) -> None:
        """No token in it, so nothing to expand -- it passes through whole."""
        font = self.data / "fonts" / "Kenteken.ttf"
        (self.data / "plates").mkdir(parents=True, exist_ok=True)
        (self.data / "plates" / "nl.json").write_text(
            json.dumps({
                "id": "nl",
                "name": "NL",
                "config": {"font": {"source": "custom", "path": str(font)}},
            }),
            encoding="utf-8",
        )
        loaded = plate_generator.plate_set_by_id("nl")
        self.assertEqual(loaded["config"]["font"]["path"], str(font))


if __name__ == "__main__":
    unittest.main()
