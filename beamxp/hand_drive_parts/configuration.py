"""Implementation slice extracted from ``beamng_hand_drive_core``.

Original source lines 3209-3520. Import the public orchestration module
``beamng_hand_drive_core`` rather than this implementation module directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from beamxp.core.constants import (
    ACTION_OPPOSITE,
    ACTION_SKIP,
    ACTION_TO_LHD,
    ACTION_TO_RHD,
    APP_DIR,
    APP_SETTINGS_PATH,
    BUILD_BOTH,
    BUILD_CHOICES,
    BUILD_CONVERTED,
    BUILD_OFF,
    BUILD_ORIGINAL,
    HAND_AUTO,
    HAND_CHOICES,
    HAND_LHD,
    HAND_RHD,
    HAND_UNKNOWN,
    MODE_CHOICES,
    MODE_MIRROR,
    MODE_MIRROR_STRUCTURAL,
    MODE_SKIP,
    MODE_TRANSLATE,
    NS,
    NUMBER_RE,
    PREVIEW_FAR_LIMIT,
    PROJECTS_DIR,
    SOURCE_ROOT_DIR,
    STEERING_NAME_EXCLUDES,
    THIS_DIR,
    TOOL_VERSION,
    USER_DATA_DIR,
    WORKSPACE_DIR,
    default_beamng_mods_dir,
    default_user_data_dir,
)
from beamxp.core.files import (
    beamng_game_common_zips,
    clean_dir,
    common_zip_candidates,
    direct_vehicle_files,
    fs_path,
    list_vehicle_files,
    load_jbeam_texts,
    make_zip,
    project_dir_for,
    read_json_file,
    safe_id,
    safe_project_segment,
    vehicle_ids_in_zip,
    vehicle_prefix,
    write_bytes_file,
    write_text_file,
    write_xml_tree,
    zip_member_path,
)
from beamxp.core.models import (
    BakedMeshSpec,
    BuildResult,
    MeshPlacement,
    ResolvedMeshPosition,
    SharedBakeContext,
    SlotDef,
    VariantInfo,
    VehicleContext,
)
from beamxp.plates import generator as plate_generator

def default_part_settings(context: VehicleContext) -> dict[str, dict[str, object]]:
    settings: dict[str, dict[str, object]] = {}
    for object_id, obj in sorted(context.objects.items()):
        settings[object_id] = {
            "mode": MODE_SKIP,
            "mirrorSource": None,
            "translateOffset": None,
            "steeringRef": is_default_steering_ref(object_id, obj),
            "viewerVisible": True,
            "viewerSolo": False,
        }
    # Vehicles often index several confident candidates (steering wheel
    # variants, columns, ...); auto-detect must flag only the best one.
    keep_single_steering_ref(context, settings)
    return settings


def default_variant_settings(context: VehicleContext) -> dict[str, dict[str, object]]:
    return {
        name: {
            "selected": False,
            "build": BUILD_OFF,
            "sourceHandOverride": HAND_AUTO,
            "frontPlate": plate_generator.PLATE_PART_AUTO,
            "rearPlate": plate_generator.PLATE_PART_AUTO,
        }
        for name in sorted(context.variants)
    }


def variant_build_mode(settings: object) -> str:
    """Return the output mode for one source trim.

    ``selected`` is retained as a compatibility mirror for pre-XP projects;
    old saves therefore migrate to Converted/Off without changing behaviour.
    """
    if not isinstance(settings, dict):
        return BUILD_OFF
    mode = str(settings.get("build") or "").lower()
    if mode in BUILD_CHOICES:
        return mode
    return BUILD_CONVERTED if settings.get("selected") else BUILD_OFF


def set_variant_build_mode(settings: dict[str, object], mode: str) -> None:
    normalized = mode if mode in BUILD_CHOICES else BUILD_OFF
    settings["build"] = normalized
    settings["selected"] = normalized != BUILD_OFF


def base_conversion_config(context: VehicleContext) -> dict[str, object]:
    return {
        "toolVersion": TOOL_VERSION,
        "source": {
            "fileName": context.source_zip.name,
            "sourcePath": str(context.source_zip),
            "vehicleId": context.vehicle_id,
            "daeFiles": context.dae_paths,
            "configs": sorted(context.variants),
        },
        "variants": default_variant_settings(context),
        "parts": default_part_settings(context),
        "plate": plate_generator.default_plate_binding(),
        "delta": {
            "manual": False,
            "magnitude": None,
            "steeringRefs": [
                object_id
                for object_id, part in default_part_settings(context).items()
                if part.get("steeringRef")
            ],
        },
    }


def conversion_path(context: VehicleContext) -> Path:
    return context.project_dir / "conversion.json"


def load_or_create_conversion(context: VehicleContext) -> tuple[dict[str, object], bool]:
    path = conversion_path(context)
    if path.exists():
        data = read_json_file(path)
        source = data.get("source", {})
        if not isinstance(source, dict) or source.get("vehicleId") in {None, context.vehicle_id}:
            return merge_with_current_inventory(context, data), True
    return base_conversion_config(context), False


def merge_with_current_inventory(context: VehicleContext, data: dict[str, object]) -> dict[str, object]:
    merged = base_conversion_config(context)
    old_variants = data.get("variants", {})
    if isinstance(old_variants, dict):
        for name, settings in old_variants.items():
            if name in merged["variants"] and isinstance(settings, dict):
                merged["variants"][name].update(
                    {
                        key: settings[key]
                        for key in (
                            "selected",
                            "build",
                            "sourceHandOverride",
                            "plate",
                            "frontPlate",
                            "rearPlate",
                        )
                        if key in settings
                    }
                )
                if "build" in settings:
                    migrated_build = variant_build_mode(settings)
                elif "selected" in settings:
                    migrated_build = BUILD_CONVERTED if settings.get("selected") else BUILD_OFF
                else:
                    migrated_build = variant_build_mode(merged["variants"][name])
                set_variant_build_mode(merged["variants"][name], migrated_build)
                merged["variants"][name]["plate"] = plate_generator.normalized_plate_binding(
                    merged["variants"][name].get("plate"), variant=True
                )

    if isinstance(data.get("plate"), dict):
        merged["plate"] = plate_generator.normalized_plate_binding(data["plate"])

    old_parts = data.get("parts", {})
    if isinstance(old_parts, dict):
        # The save's steering-ref choice wins over the auto-detected default:
        # clear the default flag(s) whenever the save carries a usable ref, so
        # the two can never combine into multiple refs.
        saved_has_ref = any(
            isinstance(settings, dict)
            and settings.get("steeringRef")
            and object_id in merged["parts"]
            for object_id, settings in old_parts.items()
        )
        if saved_has_ref:
            for settings in merged["parts"].values():
                settings["steeringRef"] = False
        for object_id, settings in old_parts.items():
            if object_id in merged["parts"] and isinstance(settings, dict):
                merged["parts"][object_id].update(
                    {
                        key: settings[key]
                        for key in (
                            "mode",
                            "mirrorSource",
                            "translateOffset",
                            "steeringRef",
                            "viewerVisible",
                            "viewerSolo",
                        )
                        if key in settings
                    }
                )
        # Old saves written before single-ref enforcement may carry several.
        keep_single_steering_ref(context, merged["parts"])
    # A save without any ref (older tool, different detection rules) must not
    # pin detection off forever: re-run it whenever the merge ends up empty.
    ensure_default_steering_ref(context, merged["parts"])

    old_delta = data.get("delta", {})
    if isinstance(old_delta, dict):
        merged["delta"].update(
            {
                key: old_delta[key]
                for key in ("manual", "magnitude")
                if key in old_delta
            }
        )
    return merged


def import_matching_conversion(
    context: VehicleContext,
    current: dict[str, object],
    imported: dict[str, object],
) -> tuple[dict[str, object], dict[str, int]]:
    out = merge_with_current_inventory(context, current)
    imported_variants = imported.get("variants", {})
    imported_parts = imported.get("parts", {})
    counts = {
        "variantImported": 0,
        "variantSkipped": 0,
        "partImported": 0,
        "partSkipped": 0,
    }

    if isinstance(imported_variants, dict):
        for name, settings in imported_variants.items():
            if name in out["variants"] and isinstance(settings, dict):
                out["variants"][name].update(
                    {
                        key: settings[key]
                        for key in (
                            "selected",
                            "build",
                            "sourceHandOverride",
                            "plate",
                            "frontPlate",
                            "rearPlate",
                        )
                        if key in settings
                    }
                )
                if "build" in settings:
                    imported_build = variant_build_mode(settings)
                elif "selected" in settings:
                    imported_build = BUILD_CONVERTED if settings.get("selected") else BUILD_OFF
                else:
                    imported_build = variant_build_mode(out["variants"][name])
                set_variant_build_mode(out["variants"][name], imported_build)
                out["variants"][name]["plate"] = plate_generator.normalized_plate_binding(
                    out["variants"][name].get("plate"), variant=True
                )
                counts["variantImported"] += 1
            else:
                counts["variantSkipped"] += 1

    if isinstance(imported_parts, dict):
        # Same single-ref rule as merge_with_current_inventory: an imported
        # steering ref replaces the current one instead of joining it.
        imported_has_ref = any(
            isinstance(settings, dict)
            and settings.get("steeringRef")
            and object_id in out["parts"]
            for object_id, settings in imported_parts.items()
        )
        if imported_has_ref:
            for settings in out["parts"].values():
                settings["steeringRef"] = False
        for object_id, settings in imported_parts.items():
            if object_id in out["parts"] and isinstance(settings, dict):
                out["parts"][object_id].update(
                    {
                        key: settings[key]
                        for key in (
                            "mode",
                            "mirrorSource",
                            "translateOffset",
                            "steeringRef",
                            "viewerVisible",
                            "viewerSolo",
                        )
                        if key in settings
                    }
                )
                counts["partImported"] += 1
            else:
                counts["partSkipped"] += 1
        keep_single_steering_ref(context, out["parts"])
    ensure_default_steering_ref(context, out["parts"])
    if isinstance(imported.get("plate"), dict):
        out["plate"] = plate_generator.normalized_plate_binding(imported["plate"])
    return out, counts


def save_conversion(context: VehicleContext, conversion: dict[str, object]) -> Path:
    context.project_dir.mkdir(parents=True, exist_ok=True)
    conversion["toolVersion"] = TOOL_VERSION
    conversion["plate"] = plate_generator.normalized_plate_binding(conversion.get("plate"))
    variants = conversion.get("variants", {})
    if isinstance(variants, dict):
        for settings in variants.values():
            if not isinstance(settings, dict):
                continue
            set_variant_build_mode(settings, variant_build_mode(settings))
            settings["plate"] = plate_generator.normalized_plate_binding(settings.get("plate"), variant=True)
            settings["frontPlate"] = plate_generator.normalized_plate_part_choice(settings.get("frontPlate"))
            settings["rearPlate"] = plate_generator.normalized_plate_part_choice(settings.get("rearPlate"))
    delta = conversion.setdefault("delta", {})
    if isinstance(delta, dict):
        delta["steeringRefs"] = selected_steering_refs(conversion)
    conversion["source"] = {
        "fileName": context.source_zip.name,
        "sourcePath": str(context.source_zip),
        "vehicleId": context.vehicle_id,
        "daeFiles": context.dae_paths,
        "configs": sorted(context.variants),
    }
    path = conversion_path(context)
    write_text_file(path, json.dumps(conversion, indent=2), encoding="utf-8")
    return path


def load_app_settings() -> dict[str, object]:
    default_mods = default_beamng_mods_dir()
    data: dict[str, object] = {}
    if APP_SETTINGS_PATH.exists():
        try:
            data = json.loads(APP_SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    return {
        "modsFolder": data.get("modsFolder") or str(default_mods),
        "blenderExecutable": data.get("blenderExecutable") or "",
        "lastVehicleZipPath": data.get("lastVehicleZipPath") or "",
        "lastVehicleId": data.get("lastVehicleId") or "",
        "lastVehicleZipFolder": data.get("lastVehicleZipFolder") or str(default_mods),
        "lastModsFolder": data.get("lastModsFolder") or str(default_mods),
        "lastBlenderFolder": data.get("lastBlenderFolder") or r"C:\Program Files",
        "previewOutputByVehicle": data.get("previewOutputByVehicle")
        if isinstance(data.get("previewOutputByVehicle"), dict)
        else {},
        "recentVehicles": data.get("recentVehicles")
        if isinstance(data.get("recentVehicles"), list)
        else [],
    }


def save_app_settings(settings: dict[str, object]) -> None:
    write_text_file(APP_SETTINGS_PATH, json.dumps(settings, indent=2), encoding="utf-8")

__all__ = ['default_part_settings', 'default_variant_settings', 'variant_build_mode', 'set_variant_build_mode', 'base_conversion_config', 'conversion_path', 'load_or_create_conversion', 'merge_with_current_inventory', 'import_matching_conversion', 'save_conversion', 'load_app_settings', 'save_app_settings']
