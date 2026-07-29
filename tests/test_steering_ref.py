from __future__ import annotations

import unittest
from pathlib import Path

from beamxp import hand_drive_core as core


def make_context(vehicle_id: str, objects: dict[str, core.DaeObject]) -> core.VehicleContext:
    return core.VehicleContext(
        source_zip=Path("test.zip"),
        vehicle_id=vehicle_id,
        vehicle_path=f"vehicles/{vehicle_id}",
        dae_paths=[],
        variants={},
        objects=objects,
        preview_by_id={},
        jbeam_texts={},
        node_positions={},
        project_dir=Path("project"),
    )


def obj(object_id: str, x: float) -> core.DaeObject:
    return core.DaeObject(
        id=object_id,
        name=object_id,
        dae_path="vehicle.dae",
        x=x,
        y=0.0,
        z=0.0,
        geometry_ids=(),
    )


def steering_refs(parts: dict[str, object]) -> list[str]:
    return sorted(
        object_id
        for object_id, settings in parts.items()
        if isinstance(settings, dict) and settings.get("steeringRef")
    )


class SteeringRefTests(unittest.TestCase):
    def test_plain_steer_name_is_detected_as_default_ref(self) -> None:
        # etk800-style naming: no "wheel" token anywhere.
        self.assertTrue(core.is_default_steering_ref("etk800_steer", obj("etk800_steer", 0.4)))

    def test_centered_or_excluded_parts_are_not_default_refs(self) -> None:
        self.assertFalse(core.is_default_steering_ref("etk800_steer", obj("etk800_steer", 0.0)))
        self.assertFalse(
            core.is_default_steering_ref("etk800_steeringbox", obj("etk800_steeringbox", 0.4))
        )
        self.assertFalse(
            core.is_default_steering_ref("steering_column", obj("steering_column", 0.4))
        )

    def test_keep_single_prefers_vehicle_prefixed_wheel(self) -> None:
        # "steer_01a" sorts before "sunburst_steer", so a plain alphabetical
        # tie-break would pick the shared-library wheel over the vehicle's own.
        objects = {
            "steer_01a": obj("steer_01a", 0.4),
            "sunburst_steer": obj("sunburst_steer", 0.4),
        }
        context = make_context("sunburst", objects)
        parts = {
            "steer_01a": {"steeringRef": True},
            "sunburst_steer": {"steeringRef": True},
        }
        core.keep_single_steering_ref(context, parts)
        self.assertEqual(steering_refs(parts), ["sunburst_steer"])

    def test_likely_steering_ref_ids_rank_vehicle_wheel_first(self) -> None:
        objects = {
            "steer_01a": obj("steer_01a", 0.4),
            "sunburst_steer": obj("sunburst_steer", 0.4),
        }
        context = make_context("sunburst", objects)
        self.assertEqual(core.likely_steering_ref_ids(context)[0], "sunburst_steer")

    def test_merge_recovers_ref_from_save_without_one(self) -> None:
        objects = {
            "etk800_steer": obj("etk800_steer", 0.4),
            "steer_01a": obj("steer_01a", 0.4),
            "etk800_dash": obj("etk800_dash", 0.0),
        }
        context = make_context("etk800", objects)
        saved = core.base_conversion_config(context)
        for settings in saved["parts"].values():
            settings["steeringRef"] = False
        merged = core.merge_with_current_inventory(context, saved)
        self.assertEqual(steering_refs(merged["parts"]), ["etk800_steer"])

    def test_merge_keeps_user_chosen_ref(self) -> None:
        objects = {
            "etk800_steer": obj("etk800_steer", 0.4),
            "steer_01a": obj("steer_01a", 0.4),
        }
        context = make_context("etk800", objects)
        saved = core.base_conversion_config(context)
        for object_id, settings in saved["parts"].items():
            settings["steeringRef"] = object_id == "steer_01a"
        merged = core.merge_with_current_inventory(context, saved)
        self.assertEqual(steering_refs(merged["parts"]), ["steer_01a"])


if __name__ == "__main__":
    unittest.main()
