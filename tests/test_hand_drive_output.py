from __future__ import annotations

import json
import re
import shutil
import tempfile
import unittest
import zipfile
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from beamxp import hand_drive_core as core
from beamxp.hand_drive_parts import generation as generation_impl


def part(
    part_id: str,
    slot_type: str,
    display_name: str,
    slots: tuple[tuple[str, str], ...] = (),
) -> str:
    slot_rows = ""
    if slots:
        rows = ",\n".join(
            f'["{slot_id}", "{default_part}", "{slot_id}"]'
            for slot_id, default_part in slots
        )
        slot_rows = (
            ',\n"slots": [\n'
            '    ["type", "default", "description"],\n'
            f"    {rows}\n"
            "]"
        )
    return (
        f'"{part_id}": {{\n'
        f'"information": {{"name": "{display_name}"}},\n'
        f'"slotType": "{slot_type}"'
        f"{slot_rows}\n"
        "}"
    )


def context_with_parts(
    part_index: dict[str, tuple[str, str]],
    selected: dict[str, object],
) -> core.VehicleContext:
    context = core.VehicleContext(
        source_zip=Path("test.zip"),
        vehicle_id="acme",
        vehicle_path="vehicles/acme",
        dae_paths=[],
        variants={"trim": core.VariantInfo("trim", "trim.pc", None, "Trim")},
        objects={},
        preview_by_id={},
        jbeam_texts={},
        node_positions={},
        project_dir=Path("project"),
        part_body_index=part_index,
    )
    context.selected_parts_cache["trim"] = selected
    return context


def write_structural_doorpanel_output(
    context: core.VehicleContext,
    output_vehicle_dir: Path,
) -> None:
    source_zip = output_vehicle_dir.parent / "source.zip"
    source_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source_zip, "w"):
        pass
    context.source_zip = source_zip
    generation_impl.write_generated_jbeam_and_configs(
        context,
        output_vehicle_dir,
        {"parts": {"doorpanel_FL": {"mode": core.MODE_MIRROR_STRUCTURAL, "mirrorSource": "doorpanel_FR"}}},
        {"doorpanel_FL": core.MODE_MIRROR_STRUCTURAL},
        {"doorpanel_FL": "doorpanel_FR"},
        {},
        {"trim": core.HAND_RHD},
        {},
        set(),
        set(),
        set(),
        set(),
        set(),
        set(),
        [],
    )


class GeneratedPartIdentityTests(unittest.TestCase):
    def test_generated_part_identity_is_not_trim_specific(self) -> None:
        first = core.generated_variant_part_name("acme_dash", core.HAND_RHD, "base")
        second = core.generated_variant_part_name("acme_dash", core.HAND_RHD, "sport")
        self.assertEqual(first, "acme_dash_xp_rhd")
        self.assertEqual(second, first)


class GeneratedRootSlotDefaultTests(unittest.TestCase):
    def test_slots_defaults_can_point_at_generated_children(self) -> None:
        array_text = (
            "[\n"
            '    ["type", "default", "description"],\n'
            '    ["acme_steer", "acme_steer", "Steering Wheel"],\n'
            '    ["acme_radio", "acme_radio", "Radio"]\n'
            "]"
        )
        rewritten = core.rewrite_child_slot_defaults(
            array_text,
            {"acme_steer": "acme_steer_xp_rhd"},
            1,
        )
        self.assertIn('"acme_steer", "acme_steer_xp_rhd"', rewritten)
        self.assertIn('"acme_radio", "acme_radio"', rewritten)

    def test_slots2_defaults_can_point_at_generated_children(self) -> None:
        array_text = (
            "[\n"
            '    ["name", "allowTypes", "denyTypes", "default", "description"],\n'
            '    ["acme_pedals", ["acme_pedals"], [], "acme_pedals", "Pedals"],\n'
            '    ["soundscape_horn", ["soundscape_horn"], [], "horn", "Horn"]\n'
            "]"
        )
        rewritten = core.rewrite_child_slot_defaults(
            array_text,
            {"acme_pedals": "acme_pedals_xp_rhd"},
            3,
        )
        self.assertIn(
            '"acme_pedals", ["acme_pedals"], [], "acme_pedals_xp_rhd"',
            rewritten,
        )
        self.assertIn('"soundscape_horn", ["soundscape_horn"], [], "horn"', rewritten)

    def test_slots2_generated_defaults_get_handed_slot_names(self) -> None:
        array_text = (
            "[\n"
            '    ["name", "allowTypes", "denyTypes", "default", "description"],\n'
            '    ["acme_dash", ["acme_dash"], [], "acme_dash", "Dashboard"],\n'
            '    ["acme_seat", ["acme_seat"], [], "acme_seat", "Seat"]\n'
            "]"
        )
        rewritten = core.rewrite_child_slot_defaults(
            array_text,
            {"acme_dash": "acme_dash_xp_rhd"},
            3,
            "_xp_rhd",
        )
        self.assertIn(
            '"acme_dash_xp_rhd", ["acme_dash"], [], "acme_dash_xp_rhd"',
            rewritten,
        )
        self.assertIn('"acme_seat", ["acme_seat"], [], "acme_seat"', rewritten)


class GeneratedLightPatternTests(unittest.TestCase):
    def test_rhd_clone_forces_rhd_light_pattern(self) -> None:
        body = (
            '"acme_headlight": {\n'
            '  "slots2": [\n'
            '    ["lowbeam", ["led"], [], "led", "Low Beam", {"variables":{"$lightPattern":"LHD"}}],\n'
            '    ["lowbeam_alt", ["led"], [], "led", "Low Beam", {"variables":{"$lightPattern":"US"}}]\n'
            "  ]\n"
            "}"
        )
        rewritten = core.rewrite_light_pattern_for_target(body, core.HAND_RHD)
        self.assertEqual(rewritten.count('"$lightPattern":"RHD"'), 2)
        self.assertNotIn('"LHD"', rewritten)
        self.assertNotIn('"US"', rewritten)

    def test_lhd_clone_forces_lhd_light_pattern(self) -> None:
        body = '"acme_headlight": {"$lightPattern":"RHD"}'
        rewritten = core.rewrite_light_pattern_for_target(body, core.HAND_LHD)
        self.assertIn('"$lightPattern":"LHD"', rewritten)

    def test_light_slots_seed_generated_bridge_ancestors(self) -> None:
        part_index = {
            "car": (part("car", "main", "Car", (("chassis", "chassis"),)), "car.jbeam"),
            "chassis": (
                part("chassis", "chassis", "Chassis", (("fender_L", "fender_L"),)),
                "chassis.jbeam",
            ),
            "fender_L": (
                part(
                    "fender_L",
                    "fender_L",
                    "Left Fender",
                    (("headlight_L", "headlight_L"),),
                ),
                "fender.jbeam",
            ),
            "headlight_L": (
                '"headlight_L": {\n'
                '"information": {"name": "Left Headlight"},\n'
                '"slotType": "headlight_L",\n'
                '"slots2": [\n'
                '  ["name", "allowTypes", "denyTypes", "default", "description"],\n'
                '  ["headlight_L_lowbeam", ["bulb"], [], "bulb", "Low Beam", {"variables":{"$electric":"lowbeam","$lightPattern":"US"}}],\n'
                '  ["headlight_L_highbeam", ["bulb"], [], "bulb", "High Beam", {"variables":{"$electric":"highbeam","$lightPattern":"US"}}]\n'
                "]\n"
                "}",
                "headlight.jbeam",
            ),
        }
        selected = {
            "parts": {"car", "chassis", "fender_L", "headlight_L"},
            "main_part": "car",
            "part_instances": [
                {
                    "instance_id": "/car",
                    "part_id": "car",
                    "slot_id": "main",
                    "slot_path": "/",
                    "parent_instance_id": None,
                },
                {
                    "instance_id": "/chassis/chassis",
                    "part_id": "chassis",
                    "slot_id": "chassis",
                    "slot_path": "/chassis/",
                    "parent_instance_id": "/car",
                },
                {
                    "instance_id": "/chassis/fender_L/fender_L",
                    "part_id": "fender_L",
                    "slot_id": "fender_L",
                    "slot_path": "/chassis/fender_L/",
                    "parent_instance_id": "/chassis/chassis",
                },
                {
                    "instance_id": "/chassis/fender_L/headlight_L/headlight_L",
                    "part_id": "headlight_L",
                    "slot_id": "headlight_L",
                    "slot_path": "/chassis/fender_L/headlight_L/",
                    "parent_instance_id": "/chassis/fender_L/fender_L",
                },
            ],
            "selected_by_slot": {
                "main": "car",
                "chassis": "chassis",
                "fender_L": "fender_L",
                "headlight_L": "headlight_L",
            },
            "part_slot_options": {},
        }
        context = context_with_parts(part_index, selected)

        plan = generation_impl._generated_clone_plan(
            context,
            selected,
            core.HAND_RHD,
            "trim",
            {},
            {},
            set(),
        )

        self.assertEqual(
            plan,
            {
                "chassis": "chassis_xp_rhd",
                "fender_L": "fender_L_xp_rhd",
                "headlight_L": "headlight_L_xp_rhd",
            },
        )


# Two bulbs shaped like the real ones: a low beam whose cookie is chosen by
# $lightPattern, and a high beam that hard-codes the one symmetric high cookie
# the game ships. Which of the two a lamp slot holds is the whole question --
# the circuit name answers nothing, as wendover's driving lamps prove by
# running the reflector bulb off "highbeam".
PATTERN_BULB = (
    '"reflector_bulb": {\n'
    '"slotType": "reflector_bulb",\n'
    '"props": [\n'
    '  ["func", "mesh", "idRef:", "idX:", "idY:"],\n'
    '  ["$electric", "SPOTLIGHT", "$nodeRef", "$nodeX", "$nodeY",\n'
    '   {"cookieName":"$= $lightPattern == \'RHD\' and \'rhd.png\' or \'us.png\'"}]\n'
    "]\n"
    "}"
)
PLAIN_BULB = (
    '"high_bulb": {\n'
    '"slotType": "high_bulb",\n'
    '"props": [\n'
    '  ["func", "mesh", "idRef:", "idX:", "idY:"],\n'
    '  ["$electric", "SPOTLIGHT", "$nodeRef", "$nodeX", "$nodeY",\n'
    '   {"cookieName":"$= $cookieName ~= nil and $cookieName or \'high.png\'"}]\n'
    "]\n"
    "}"
)


def bulb_context(part_bodies: dict[str, str]) -> core.VehicleContext:
    return context_with_parts(
        {
            part_id: (body, f"{part_id}.jbeam")
            for part_id, body in part_bodies.items()
        },
        {"parts": set(), "main_part": "", "part_instances": []},
    )


class LightPatternBulbTests(unittest.TestCase):
    # A lamp part with no $lightPattern anywhere: the low beam mounts the
    # pattern-reading bulb, the high beam the symmetric one.
    LAMP_PART = (
        '"acme_headlight_R": {\n'
        '"slotType": "acme_headlight_R",\n'
        '"slots2": [\n'
        '  ["name", "allowTypes", "denyTypes", "default", "description"],\n'
        '  ["acme_low_bulb", ["reflector_bulb"], [], "reflector_bulb", "Low Beam",\n'
        '   {"coreSlot":true, "variables":{"$electric":"lowbeam_filament", "$nodeRef":"he4r"}}],\n'
        '  ["acme_high_bulb", ["high_bulb"], [], "high_bulb", "High Beam",\n'
        '   {"coreSlot":true, "variables":{"$electric":"highbeam_filament", "$nodeRef":"he4r"}}]\n'
        "]\n"
        "}"
    )

    def setUp(self) -> None:
        self.context = bulb_context(
            {
                "acme_headlight_R": self.LAMP_PART,
                "reflector_bulb": PATTERN_BULB,
                "high_bulb": PLAIN_BULB,
            }
        )
        self.bulbs = core.light_pattern_bulb_slot_types(self.context)

    def test_only_bulbs_that_read_the_pattern_are_indexed(self) -> None:
        self.assertEqual(self.bulbs, frozenset({"reflector_bulb"}))

    def test_unset_pattern_still_marks_the_part_handed(self) -> None:
        self.assertNotIn("$lightPattern", self.LAMP_PART)
        self.assertTrue(
            generation_impl._part_has_handed_light_slots(self.LAMP_PART, self.bulbs)
        )
        # Without the bulbs -- no vehicles/common on disk -- the old reading is
        # all there is, and it cannot see this part.
        self.assertFalse(
            generation_impl._part_has_handed_light_slots(self.LAMP_PART, frozenset())
        )

    def test_clone_inserts_the_pattern_only_where_the_bulb_reads_it(self) -> None:
        cloned = core.clone_part_for_target(
            self.LAMP_PART,
            "acme_headlight_R",
            core.HAND_RHD,
            None,
            {}, {}, {}, {}, {}, {},
            light_pattern_slot_types=self.bulbs,
        )
        self.assertEqual(cloned.count('"$lightPattern":"RHD"'), 1)
        low, high = cloned.split('"acme_high_bulb"')
        self.assertIn('"$lightPattern":"RHD"', low)
        self.assertNotIn("$lightPattern", high)
        core.parse_beamng_json("{" + cloned + "}", label="acme_headlight_R")

    def test_authored_pattern_is_retargeted_in_place(self) -> None:
        authored = self.LAMP_PART.replace(
            '"variables":{"$electric":"lowbeam_filament"',
            '"variables":{"$lightPattern":"US", "$electric":"lowbeam_filament"',
        )
        cloned = core.clone_part_for_target(
            authored,
            "acme_headlight_R",
            core.HAND_RHD,
            None,
            {}, {}, {}, {}, {}, {},
            light_pattern_slot_types=self.bulbs,
        )
        self.assertEqual(cloned.count("$lightPattern"), 1)
        self.assertIn('"$lightPattern":"RHD"', cloned)

    def test_a_mounting_part_is_not_mistaken_for_a_bulb(self) -> None:
        # acme_headlight_R names $lightPattern in its slots2 once converted, but
        # it sets the variable rather than reading it, so it must never be
        # indexed as a bulb -- that would make its parent slot look handed.
        context = bulb_context(
            {
                "acme_headlight_R": self.LAMP_PART.replace(
                    '"$electric":"lowbeam_filament"',
                    '"$lightPattern":"RHD", "$electric":"lowbeam_filament"',
                ),
                "reflector_bulb": PATTERN_BULB,
            }
        )
        self.assertEqual(
            core.light_pattern_bulb_slot_types(context),
            frozenset({"reflector_bulb"}),
        )

    def test_fog_lamp_with_a_plain_bulb_is_left_alone(self) -> None:
        foglight = (
            '"acme_foglight": {\n'
            '"slotType": "acme_foglight",\n'
            '"slots2": [\n'
            '  ["name", "allowTypes", "denyTypes", "default", "description"],\n'
            '  ["acme_fog_bulb", ["high_bulb"], [], "high_bulb", "Fog",\n'
            '   {"variables":{"$electric":"foglight_filament"}}]\n'
            "]\n"
            "}"
        )
        self.assertFalse(
            generation_impl._part_has_handed_light_slots(foglight, self.bulbs)
        )
        self.assertEqual(
            core.apply_light_pattern_to_bulb_slots(
                core.transform_helpers.extract_named_array(foglight, "slots2"),
                core.HAND_RHD,
                self.bulbs,
            ),
            core.transform_helpers.extract_named_array(foglight, "slots2"),
        )


class GeneratedCameraTests(unittest.TestCase):
    CAMERA_ARRAY = (
        "[\n"
        '  ["type", "x", "y", "z", "fov", "id1:", "id2:"],\n'
        '  ["dash", 0.34, 0.34, 1.07, 55, "f1ll", "f1r"]\n'
        "]"
    )

    def test_rhd_camera_clone_sets_rhd_flag(self) -> None:
        rewritten = core.rewrite_internal_cameras(
            self.CAMERA_ARRAY,
            {"f1ll": "f1rr", "f1r": "f1l"},
            core.HAND_RHD,
        )
        self.assertIn('"rightHandCamera":true', rewritten)
        self.assertIn('["dash", -0.34', rewritten)

    def test_lhd_camera_clone_sets_lhd_flag(self) -> None:
        rewritten = core.rewrite_internal_cameras(
            self.CAMERA_ARRAY,
            {"f1ll": "f1rr", "f1r": "f1l"},
            core.HAND_LHD,
        )
        self.assertIn('"rightHandCamera":false', rewritten)
        self.assertNotIn('"rightHandCamera":true', rewritten)

    def _seat_plan(self, seat_body: str) -> dict[str, str]:
        part_index = {
            "car": (part("car", "main", "Car", (("seat_FL", "seat_FL"),)), "car.jbeam"),
            "seat_FL": (seat_body, "interior.jbeam"),
        }
        selected = {
            "parts": {"car", "seat_FL"},
            "main_part": "car",
            "part_instances": [
                {
                    "instance_id": "/car",
                    "part_id": "car",
                    "slot_id": "main",
                    "slot_path": "/",
                    "parent_instance_id": None,
                },
                {
                    "instance_id": "/seat_FL/seat_FL",
                    "part_id": "seat_FL",
                    "slot_id": "seat_FL",
                    "slot_path": "/seat_FL/",
                    "parent_instance_id": "/car",
                },
            ],
            "selected_by_slot": {"main": "car", "seat_FL": "seat_FL"},
            "part_slot_options": {},
        }
        return generation_impl._generated_clone_plan(
            context_with_parts(part_index, selected),
            selected,
            core.HAND_RHD,
            "trim",
            {},
            {"f1ll": "f1rr", "f1r": "f1l"},
            set(),
        )

    def test_a_seat_owning_the_driver_camera_is_still_cloned(self) -> None:
        # etk800 declares its dash camera inside every driver-seat variant.
        # Seats otherwise cross the car by slot occupancy, but no slot swap
        # moves a camera -- and on a vehicle whose two seat slots are both
        # filled the swap never runs -- so the clone has to happen anyway.
        seat = (
            '"seat_FL": {\n'
            '"information": {"name": "Front Left Seat"},\n'
            '"slotType": "seat_FL",\n'
            f'"camerasInternal": {self.CAMERA_ARRAY}\n'
            "}"
        )
        self.assertEqual(self._seat_plan(seat), {"seat_FL": "seat_FL_xp_rhd"})

    def test_a_seat_without_a_camera_still_crosses_by_slot_occupancy(self) -> None:
        self.assertEqual(
            self._seat_plan(part("seat_FL", "seat_FL", "Front Left Seat")),
            {},
        )


class OutputPackageNameTests(unittest.TestCase):
    @staticmethod
    def _context(zip_stem: str, vehicle_id: str) -> core.VehicleContext:
        return core.VehicleContext(
            source_zip=Path(f"{zip_stem}.zip"),
            vehicle_id=vehicle_id,
            vehicle_path=f"vehicles/{vehicle_id}",
            dae_paths=[],
            variants={},
            objects={},
            preview_by_id={},
            jbeam_texts={},
            node_positions={},
            project_dir=Path("project"),
            part_body_index={},
        )

    def test_package_names_the_vehicle_when_the_zip_holds_several(self) -> None:
        # vivace.zip ships the vivace, the ardente and the tograc; converting
        # each in turn must not have them overwrite one another in the mods
        # folder.
        self.assertEqual(
            core.package_name_for_context(self._context("vivace", "vivace_ardente")),
            "vivace_vivace_ardente_XP_conversion.zip",
        )

    def test_package_stays_unrepeated_when_the_zip_is_the_vehicle(self) -> None:
        self.assertEqual(
            core.package_name_for_context(self._context("bx", "bx")),
            "bx_XP_conversion.zip",
        )


class ModManifestTests(unittest.TestCase):
    # core/modmanager.lua matches the lowercased in-zip path against this
    # before it will read a manifest at all, then upper-cases the capture as
    # the mod's id. A manifest anywhere else is silently never loaded, which is
    # what left the Unique ID and Author fields blank.
    MOD_INFO_RE = re.compile(r"^/?mod_info/([0-9a-zA-Z]*)/info\.json$")

    def _write(
        self,
        zip_stem: str,
        vehicle_id: str,
        *,
        generated_configs: tuple[str, ...] = (),
        config_sources: dict[str, str] | None = None,
    ) -> tuple[Path, dict[str, object]]:
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        self.root = root
        source = root / f"{zip_stem}.zip"
        with zipfile.ZipFile(source, "w") as archive:
            archive.writestr(f"vehicles/{vehicle_id}/{vehicle_id}.jbeam", "{}")
            archive.writestr(f"vehicles/{vehicle_id}/{vehicle_id}.dae", "")
            # No model default.jpg, so the tile falls through to the default
            # config's own image -- the case _tile_preview_member resolves.
            archive.writestr(
                f"vehicles/{vehicle_id}/info.json", json.dumps({"default_pc": "base"})
            )
            for config in ("base", "sport"):
                archive.writestr(f"vehicles/{vehicle_id}/{config}.pc", "{}")
                archive.writestr(f"vehicles/{vehicle_id}/{config}.jpg", f"{config} image")

        output_vehicle_dir = root / "vehicles" / vehicle_id
        output_vehicle_dir.mkdir(parents=True, exist_ok=True)
        for config in generated_configs:
            (output_vehicle_dir / f"{config}.jpg").write_text(f"{config} render")

        context = OutputPackageNameTests._context(zip_stem, vehicle_id)
        context = replace(context, source_zip=source)
        core.write_mod_info(
            root,
            context,
            output_vehicle_dir if generated_configs else None,
            generated_configs,
            config_sources,
        )
        manifest = next(root.rglob("info.json"))
        return manifest.relative_to(root), json.loads(manifest.read_text())

    def test_manifest_lands_where_the_mod_manager_reads_it(self) -> None:
        relative, info = self._write("vivace", "vivace_ardente")
        match = self.MOD_INFO_RE.match(relative.as_posix().lower())
        assert match is not None, relative
        self.assertEqual(match.group(1).upper(), info["tagid"])

    def test_unique_id_and_author_are_the_fields_the_panel_binds(self) -> None:
        _relative, info = self._write("vivace", "vivace_ardente")
        # info.html binds Unique ID to tagid and Author to username.
        self.assertEqual(info["tagid"], core.mod_id_for_context(
            OutputPackageNameTests._context("vivace", "vivace_ardente")
        ))
        self.assertEqual(info["username"], "Telestang - BeamXP")

    def test_the_id_is_reproducible_and_vehicle_specific(self) -> None:
        one = OutputPackageNameTests._context("vivace", "vivace_ardente")
        again = OutputPackageNameTests._context("vivace", "vivace_ardente")
        sibling = OutputPackageNameTests._context("vivace", "vivace_tograc")
        self.assertEqual(core.mod_id_for_context(one), core.mod_id_for_context(again))
        self.assertNotEqual(core.mod_id_for_context(one), core.mod_id_for_context(sibling))
        # Alphanumeric or the folder does not match at all.
        self.assertTrue(core.mod_id_for_context(one).isalnum())

    def test_the_thumbnail_is_the_converted_tile_image(self) -> None:
        # The selector's tile for this vehicle shows base.jpg, so the mod's
        # thumbnail is that same car after conversion.
        relative, info = self._write(
            "acme",
            "acme",
            generated_configs=("base_rhd", "sport_rhd"),
            config_sources={"base_rhd": "base", "sport_rhd": "sport"},
        )
        self.assertEqual(info["icon"], "preview.jpg")
        # modmanager.lua builds the image strip from attachments' thumb_filename.
        self.assertEqual(
            [entry["thumb_filename"] for entry in info["attachments"]],
            ["preview.jpg"],
        )
        thumbnail = self.root / relative.parent / str(info["icon"])
        self.assertEqual(thumbnail.read_text(), "base_rhd render")

    def test_the_thumbnail_falls_back_when_the_tile_config_was_not_converted(self) -> None:
        relative, info = self._write(
            "acme",
            "acme",
            generated_configs=("sport_rhd", "zzz_rhd"),
            config_sources={"sport_rhd": "sport", "zzz_rhd": "zzz"},
        )
        thumbnail = self.root / relative.parent / str(info["icon"])
        self.assertEqual(thumbnail.read_text(), "sport_rhd render")

    def test_a_build_with_no_previews_leaves_the_thumbnail_out(self) -> None:
        _relative, info = self._write("acme", "acme")
        self.assertNotIn("icon", info)
        self.assertNotIn("attachments", info)

    def test_the_body_lists_what_the_build_added(self) -> None:
        _relative, info = self._write(
            "acme",
            "acme",
            generated_configs=("base_rhd", "sport_rhd"),
            config_sources={"base_rhd": "base", "sport_rhd": "sport"},
        )
        # repository-details.html binds the description page to message;
        # text feeds only the mod manager's own fallback panel.
        for field in ("message", "text"):
            self.assertIn("[*]base_rhd", info[field])
            self.assertIn("[*]sport_rhd", info[field])
            self.assertIn("acme.zip", info[field])

    def test_the_body_shows_the_preview_by_its_in_game_path(self) -> None:
        # The repository page rebuilds icon as a beamng.com URL, so the only
        # way a local image reaches that page is an [img] in the body.
        _relative, info = self._write(
            "acme",
            "acme",
            generated_configs=("base_rhd",),
            config_sources={"base_rhd": "base"},
        )
        image = re.compile(r"\[img\]\s*?(\S*?(?=\[/img\]))\[/img\]", re.I)
        self.assertEqual(
            image.findall(info["message"]),
            [f"/mod_info/{info['tagid']}/preview.jpg"],
        )

    def test_last_update_is_the_epoch_seconds_the_page_multiplies(self) -> None:
        # repository.js does new Date(last_update * 1000); anything that is not
        # a number of seconds renders as null.
        _relative, info = self._write("acme", "acme")
        self.assertIsInstance(info["last_update"], int)
        rendered = datetime.fromtimestamp(info["last_update"], UTC)
        self.assertLess(abs((datetime.now(UTC) - rendered).total_seconds()), 600)


class GeneratedMirrorTests(unittest.TestCase):
    """The ``mirrors`` section, checked against BX's authored LHD/RHD pairs.

    BX ships both hands of the same interior, so its own two rows are the
    reference answer: the rear-view mirror keeps every field but flips the
    rotation that aims it at the driver.
    """

    INTERIOR = (
        "[\n"
        '  ["mesh", "idRef:", "id1:", "id2:"],\n'
        '  ["bx_mirror_int_lhd","rf1","rf1r","rf2",'
        '{"refBaseTranslation":{"x":0.00,"y":-0.08,"z":-0.12},'
        '"baseRotationGlobal":{"x":6,"y":0,"z":19}}],\n'
        "]"
    )
    WING_L = (
        "[\n"
        '  ["mesh", "idRef:", "id1:", "id2:"],\n'
        '  ["mirror_L","mi4l","mi3l","mi2l",'
        '{"refBaseTranslation":{"x":-0.085,"y":-0.034,"z":0.05},'
        '"baseRotationGlobal":{"x":0,"y":0,"z":-15}}],\n'
        "]"
    )
    WING_R_ROW = (
        '["mirror_R","mi4r","mi3r","mi2r",'
        '{"refBaseTranslation":{"x":0.085,"y":-0.034,"z":0.05},'
        '"baseRotationGlobal":{"x":0,"y":0,"z":22}}],'
    )

    def test_row_binds_the_mesh_the_converted_part_renders(self) -> None:
        # addMirror() looks the mesh up among the part's own meshes, so a row
        # left on the pre-conversion name reflects nothing at all.
        rewritten = core.rewrite_mirror_rows(
            self.INTERIOR,
            {"bx_mirror_int_lhd": "bx_mirror_int_lhd_xp_rhd"},
            {},
        )
        self.assertIn('["bx_mirror_int_lhd_xp_rhd","rf1"', rewritten)

    def test_a_mirrored_mesh_re_aims_at_the_new_driver_side(self) -> None:
        row = self.INTERIOR.splitlines()[2].strip()
        rewritten = core.rewrite_mirror_rows(
            self.INTERIOR,
            {"bx_mirror_int_lhd": "bx_mirror_int_lhd_xp_rhd"},
            {"bx_mirror_int_lhd": row},
        )
        # bx_mirror_int_rhd, authored: same offset, z negated.
        self.assertIn('"refBaseTranslation":{"x":0,"y":-0.08,"z":-0.12}', rewritten)
        self.assertIn('"baseRotationGlobal":{"x":6,"y":0,"z":-19}', rewritten)

    def test_a_swapped_wing_mirror_inherits_the_other_side_s_plane(self) -> None:
        # Swap Mesh renders the twin's glass reflected, so the plane comes from
        # the twin's row -- the left mirror keeps its left offset but takes the
        # aim the right mirror had, reflected.
        rewritten = core.rewrite_mirror_rows(
            self.WING_L,
            {"mirror_L": "mirror_L_xp_rhd"},
            {"mirror_L": self.WING_R_ROW},
        )
        self.assertIn('["mirror_L_xp_rhd","mi4l"', rewritten)
        self.assertIn('"refBaseTranslation":{"x":-0.085,"y":-0.034,"z":0.05}', rewritten)
        self.assertIn('"baseRotationGlobal":{"x":0,"y":0,"z":-22}', rewritten)

    def test_a_mesh_the_build_left_alone_keeps_its_authored_plane(self) -> None:
        rewritten = core.rewrite_mirror_rows(self.WING_L, {}, {})
        self.assertEqual(rewritten, self.WING_L)


class ReplaceSourceConfigOutputTests(unittest.TestCase):
    def test_selected_replace_source_part_updates_output_slot(self) -> None:
        part_index = {
            "car": (part("car", "main", "Car", (("mirror_L", "mirror_L_rhd"),)), "car.jbeam"),
            "mirror_L": (part("mirror_L", "mirror_L", "Mirror LHD"), "mirror.jbeam"),
            "mirror_L_rhd": (part("mirror_L_rhd", "mirror_L", "Mirror RHD"), "mirror.jbeam"),
        }
        selected = {
            "parts": {"car", "mirror_L_rhd"},
            "main_part": "car",
            "part_instances": [
                {
                    "part_id": "car",
                    "slot_id": "main",
                    "slot_path": "/",
                },
                {
                    "part_id": "mirror_L_rhd",
                    "slot_id": "mirror_L",
                    "slot_path": "/mirror_L/",
                },
            ],
            "selected_by_slot": {"main": "car", "mirror_L": "mirror_L_rhd"},
            "selected_by_path": {"/": "car", "/mirror_L/": "mirror_L_rhd"},
        }
        context = context_with_parts(part_index, selected)
        conversion = {
            "parts": {
                "mirror_L_rhd": {
                    "mode": core.MODE_REPLACE_SOURCE,
                    "mirrorSource": "mirror_L",
                }
            }
        }
        pc = {
            "parts": {
                "mirror_L": "mirror_L_rhd_xp_lhd",
                "mirror_L_xp_lhd": "mirror_L_rhd_xp_lhd",
            }
        }

        generation_impl.apply_replace_source_slot_updates(
            context,
            conversion,
            selected,
            pc,
            core.HAND_LHD,
        )

        self.assertEqual(pc["parts"]["mirror_L"], "mirror_L")
        self.assertEqual(pc["parts"]["/mirror_L/"], "mirror_L")
        self.assertEqual(pc["parts"]["mirror_L_xp_lhd"], "mirror_L")


class StructuralFlexbodyOutputTests(unittest.TestCase):
    def test_structural_leaf_swap_does_not_clone_physical_parent_door(self) -> None:
        car = part("car", "main", "Car", (("door_FL", "door_FL"),))
        door = part("door_FL", "door_FL", "Front Left Door", (("doorpanel_FL", "doorpanel_FL"),))
        panel = (
            '"doorpanel_FL": {\n'
            '"information": {"name": "Front Left Door Panel"},\n'
            '"slotType": "doorpanel_FL",\n'
            '"flexbodies": [\n'
            '    ["mesh", "[group]:"],\n'
            '    ["doorpanel_FL", ["door_FL"]]\n'
            "]\n"
            "}"
        )
        selected = {
            "parts": {"car", "door_FL", "doorpanel_FL"},
            "main_part": "car",
            "part_instances": [
                {
                    "instance_id": "/car",
                    "part_id": "car",
                    "slot_id": "main",
                    "slot_path": "/",
                },
                {
                    "instance_id": "/door_FL/door_FL",
                    "part_id": "door_FL",
                    "slot_id": "door_FL",
                    "slot_path": "/door_FL/",
                    "parent_instance_id": "/car",
                },
                {
                    "instance_id": "/door_FL/doorpanel_FL/doorpanel_FL",
                    "part_id": "doorpanel_FL",
                    "slot_id": "doorpanel_FL",
                    "slot_path": "/door_FL/doorpanel_FL/",
                    "parent_instance_id": "/door_FL/door_FL",
                },
            ],
            "selected_by_slot": {
                "main": "car",
                "door_FL": "door_FL",
                "doorpanel_FL": "doorpanel_FL",
            },
            "selected_by_path": {
                "/": "car",
                "/door_FL/": "door_FL",
                "/door_FL/doorpanel_FL/": "doorpanel_FL",
            },
        }
        context = context_with_parts(
            {
                "car": (car, "car.jbeam"),
                "door_FL": (door, "door.jbeam"),
                "doorpanel_FL": (panel, "panel.jbeam"),
            },
            selected,
        )
        context.objects = {
            "doorpanel_FL": core.DaeObject(
                "doorpanel_FL",
                "doorpanel_FL",
                "vehicles/acme/acme.dae",
                1.0,
                0.0,
                0.0,
                (),
            ),
            "doorpanel_FR": core.DaeObject(
                "doorpanel_FR",
                "doorpanel_FR",
                "vehicles/acme/acme.dae",
                -1.0,
                0.0,
                0.0,
                (),
            ),
        }
        context.pc_cache["trim.pc"] = {
            "parts": {
                "door_FL": "door_FL",
                "doorpanel_FL": "doorpanel_FL",
            }
        }

        with tempfile.TemporaryDirectory() as tmp:
            output_vehicle_dir = Path(tmp) / context.vehicle_path
            write_structural_doorpanel_output(context, output_vehicle_dir)
            generated_jbeam = (output_vehicle_dir / "jbeam" / "handdrive_visual_conversion.jbeam").read_text(
                encoding="utf-8"
            )
            generated_pc = core.read_json_file(output_vehicle_dir / "trim_rhd.pc")

        self.assertNotIn('"door_FL_xp_rhd"', generated_jbeam)
        self.assertIn('"doorpanel_FL_xp_rhd"', generated_jbeam)
        self.assertEqual(generated_pc["parts"]["door_FL"], "door_FL")
        self.assertEqual(generated_pc["parts"]["doorpanel_FL"], "doorpanel_FL_xp_rhd")


class GeneratedConfigMetadataTests(unittest.TestCase):
    def test_default_selector_flags_are_forced_off(self) -> None:
        data = {
            "default": True,
            "defaultConfig": True,
            "isDefault": True,
            "isDefaultConfig": True,
            "isDefaultForSubCluster": True,
            "default_pc": "base",
            "defaultPaintName1": "Blue",
        }

        generation_impl.clear_default_config_flags(data)

        self.assertFalse(data["default"])
        self.assertFalse(data["defaultConfig"])
        self.assertFalse(data["isDefault"])
        self.assertFalse(data["isDefaultConfig"])
        self.assertFalse(data["isDefaultForSubCluster"])
        self.assertNotIn("default_pc", data)
        self.assertEqual(data["defaultPaintName1"], "Blue")


class HandAuthoredGroupTests(unittest.TestCase):
    def setUp(self) -> None:
        # Model the vanilla pattern: dashboard roots share one slot, while
        # related child slots and parts carry their own LHD/RHD suffixes.
        self.part_index = {
            "car": (
                part("car", "main", "Car", (("dashboard", "dash_lhd"),)),
                "car.jbeam",
            ),
            "dash_lhd": (
                part(
                    "dash_lhd",
                    "dashboard",
                    "Left-Hand-Drive Dashboard",
                    (
                        ("steer", "steer"),
                        ("shifter_lhd", "shifter_A_lhd"),
                        ("gauges_lhd", "gauges_lhd"),
                    ),
                ),
                "dash_lhd.jbeam",
            ),
            "dash_rhd": (
                part(
                    "dash_rhd",
                    "dashboard",
                    "Right-Hand-Drive Dashboard",
                    (
                        ("steer", "steer"),
                        ("shifter_rhd", "shifter_A_rhd"),
                        ("gauges_rhd", "gauges_rhd"),
                    ),
                ),
                "dash_rhd.jbeam",
            ),
            "steer": (part("steer", "steer", "Steering Wheel"), "dash.jbeam"),
            "shifter_M_lhd": (
                part("shifter_M_lhd", "shifter_lhd", "Manual Shifter LHD"),
                "dash_lhd.jbeam",
            ),
            "shifter_M_rhd": (
                part("shifter_M_rhd", "shifter_rhd", "Manual Shifter RHD"),
                "dash_rhd.jbeam",
            ),
            "shifter_A_lhd": (
                part("shifter_A_lhd", "shifter_lhd", "Automatic Shifter LHD"),
                "dash_lhd.jbeam",
            ),
            "shifter_A_rhd": (
                part("shifter_A_rhd", "shifter_rhd", "Automatic Shifter RHD"),
                "dash_rhd.jbeam",
            ),
            "gauges_lhd": (
                part("gauges_lhd", "gauges_lhd", "Gauges LHD"),
                "dash_lhd.jbeam",
            ),
            "gauges_rhd": (
                part("gauges_rhd", "gauges_rhd", "Gauges RHD"),
                "dash_rhd.jbeam",
            ),
        }
        self.selected = {
            "parts": {"car", "dash_lhd", "steer", "shifter_M_lhd", "gauges_lhd"},
            "part_instances": [
                {
                    "instance_id": "/car",
                    "part_id": "car",
                    "slot_id": "main",
                    "slot_path": "/",
                    "parent_instance_id": None,
                    "inherited_options": (),
                },
                {
                    "instance_id": "/dashboard/dash_lhd",
                    "part_id": "dash_lhd",
                    "slot_id": "dashboard",
                    "slot_path": "/dashboard/",
                    "parent_instance_id": "/car",
                    "inherited_options": (),
                },
                {
                    "instance_id": "/dashboard/steer/steer",
                    "part_id": "steer",
                    "slot_id": "steer",
                    "slot_path": "/dashboard/steer/",
                    "parent_instance_id": "/dashboard/dash_lhd",
                    "inherited_options": (),
                },
                {
                    "instance_id": "/dashboard/shifter_lhd/shifter_M_lhd",
                    "part_id": "shifter_M_lhd",
                    "slot_id": "shifter_lhd",
                    "slot_path": "/dashboard/shifter_lhd/",
                    "parent_instance_id": "/dashboard/dash_lhd",
                    "inherited_options": (),
                },
                {
                    "instance_id": "/dashboard/gauges_lhd/gauges_lhd",
                    "part_id": "gauges_lhd",
                    "slot_id": "gauges_lhd",
                    "slot_path": "/dashboard/gauges_lhd/",
                    "parent_instance_id": "/dashboard/dash_lhd",
                    "inherited_options": (),
                },
            ],
            "selected_by_slot": {
                "main": "car",
                "dashboard": "dash_lhd",
                "steer": "steer",
                "shifter_lhd": "shifter_M_lhd",
                "gauges_lhd": "gauges_lhd",
            },
            "selected_by_path": {
                "/": "car",
                "/dashboard/": "dash_lhd",
                "/dashboard/steer/": "steer",
                "/dashboard/shifter_lhd/": "shifter_M_lhd",
                "/dashboard/gauges_lhd/": "gauges_lhd",
            },
            "part_slot_options": {},
        }

    def test_detects_authored_dashboard_and_maps_related_choices(self) -> None:
        context = context_with_parts(self.part_index, self.selected)
        group = core.find_hand_authored_opposite_group(
            context, "trim", core.HAND_LHD, core.HAND_RHD
        )
        self.assertIsNotNone(group)
        assert group is not None
        self.assertEqual(group["sourcePart"], "dash_lhd")
        self.assertEqual(group["targetPart"], "dash_rhd")
        updates = {
            selection["slotId"]: selection["partId"]
            for selection in group["selections"]
        }
        self.assertEqual(
            updates,
            {
                "dashboard": "dash_rhd",
                "steer": "steer",
                "shifter_rhd": "shifter_M_rhd",
                "gauges_rhd": "gauges_rhd",
            },
        )

    def test_unmarked_default_can_pair_with_explicit_rhd_part(self) -> None:
        part_index = dict(self.part_index)
        part_index["dash"] = (
            part(
                "dash",
                "dashboard",
                "Dashboard",
                (("steer", "steer"), ("shifter_lhd", "shifter_A_lhd")),
            ),
            "dash_lhd.jbeam",
        )
        selected = dict(self.selected)
        selected["parts"] = set(self.selected["parts"])
        selected["parts"].discard("dash_lhd")
        selected["parts"].add("dash")
        selected["part_instances"] = [
            dict(instance) for instance in self.selected["part_instances"]
        ]
        selected["part_instances"][1]["part_id"] = "dash"
        selected["selected_by_slot"] = dict(self.selected["selected_by_slot"])
        selected["selected_by_slot"]["dashboard"] = "dash"
        selected["selected_by_path"] = dict(self.selected["selected_by_path"])
        selected["selected_by_path"]["/dashboard/"] = "dash"
        context = context_with_parts(part_index, selected)
        group = core.find_hand_authored_opposite_group(
            context, "trim", core.HAND_LHD, core.HAND_RHD
        )
        self.assertIsNotNone(group)
        assert group is not None
        self.assertEqual(group["targetPart"], "dash_rhd")

    def test_leaf_pair_alone_is_not_treated_as_a_group(self) -> None:
        part_index = {
            "car": (part("car", "main", "Car", (("wheel", "wheel_lhd"),)), "a.jbeam"),
            "wheel_lhd": (part("wheel_lhd", "wheel", "LHD Wheel"), "a.jbeam"),
            "wheel_rhd": (part("wheel_rhd", "wheel", "RHD Wheel"), "a.jbeam"),
        }
        selected = {
            "part_instances": [
                {"part_id": "car", "slot_id": "main", "slot_path": "/"},
                {
                    "part_id": "wheel_lhd",
                    "slot_id": "wheel",
                    "slot_path": "/wheel/",
                },
            ],
            "selected_by_slot": {"main": "car", "wheel": "wheel_lhd"},
            "selected_by_path": {"/": "car", "/wheel/": "wheel_lhd"},
        }
        context = context_with_parts(part_index, selected)
        group = core.find_hand_authored_opposite_group(
            context, "trim", core.HAND_LHD, core.HAND_RHD
        )
        self.assertIsNone(group)

    def test_group_application_preserves_path_specific_keys(self) -> None:
        context = context_with_parts(self.part_index, self.selected)
        group = core.find_hand_authored_opposite_group(
            context, "trim", core.HAND_LHD, core.HAND_RHD
        )
        assert group is not None
        pc: dict[str, object] = {
            "parts": {
                "/dashboard/": "dash_lhd",
                "/dashboard/shifter_lhd/": "shifter_M_lhd",
            }
        }
        core.apply_hand_authored_group(pc, group)
        self.assertEqual(
            pc["parts"],
            {
                "/dashboard/": "dash_rhd",
                "/dashboard/shifter_rhd/": "shifter_M_rhd",
                "steer": "steer",
                "gauges_rhd": "gauges_rhd",
            },
        )

    def test_group_application_rewrites_child_slot_namespace(self) -> None:
        context = context_with_parts(self.part_index, self.selected)
        group = core.find_hand_authored_opposite_group(
            context, "trim", core.HAND_LHD, core.HAND_RHD
        )
        assert group is not None
        pc: dict[str, object] = {
            "parts": {
                "dashboard": "dash_lhd",
                "steer": "steer",
                "shifter_lhd": "shifter_M_lhd",
                "gauges_lhd": "gauges_lhd",
                "paint": "red",
            }
        }
        core.apply_hand_authored_group(pc, group)
        self.assertEqual(
            pc["parts"],
            {
                "dashboard": "dash_rhd",
                "steer": "steer",
                "shifter_rhd": "shifter_M_rhd",
                "gauges_rhd": "gauges_rhd",
                "paint": "red",
            },
        )

    def test_group_application_updates_generated_parent_slot_names(self) -> None:
        context = context_with_parts(self.part_index, self.selected)
        group = core.find_hand_authored_opposite_group(
            context, "trim", core.HAND_LHD, core.HAND_RHD
        )
        assert group is not None
        pc: dict[str, object] = {
            "parts": {
                "dashboard": "dash_rhd",
                "dashboard_xp_rhd": "dash_lhd_xp_rhd",
            }
        }
        generation_impl.apply_authored_group_suffixed_slot_updates(
            pc,
            group,
            core.HAND_RHD,
        )
        self.assertEqual(pc["parts"]["dashboard_xp_rhd"], "dash_rhd")


if __name__ == "__main__":
    unittest.main()
