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
    PART_TEXTURE_CORRECTION_KEY,
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


def default_part_setting(object_id: str, obj: object | None = None) -> dict[str, object]:
    return {
        "mode": MODE_SKIP,
        "mirrorSource": None,
        "translateOffset": None,
        "steeringRef": is_default_steering_ref(object_id, obj) if obj is not None else False,
        "includeChildren": False,
        PART_TEXTURE_CORRECTION_KEY: False,
        "viewerVisible": True,
        "viewerSolo": False,
    }


def default_part_settings(context: VehicleContext) -> dict[str, dict[str, object]]:
    settings: dict[str, dict[str, object]] = {}
    for object_id, obj in sorted(context.objects.items()):
        settings[object_id] = default_part_setting(object_id, obj)
    # Vehicles often index several confident candidates (steering wheel
    # variants, columns, ...); auto-detect must flag only the best one.
    keep_single_steering_ref(context, settings)
    return settings


def normalized_slot_pairs(
    value: object,
    known_slots: set[str] | None = None,
) -> list[dict[str, object]]:
    """Clean a saved ``slotPairs`` list.

    Pairs are keyed by slot type because that is what the user sees and what
    survives a vehicle update; slot paths differ per trim. A slot may appear in
    only one pair -- pairing is a bijection between the two sides of the car,
    so a second pair naming an already-paired slot is dropped rather than
    silently overriding the first.
    """
    if not isinstance(value, (list, tuple)):
        return []
    pairs: list[dict[str, object]] = []
    claimed: set[str] = set()
    for entry in value:
        if not isinstance(entry, dict):
            continue
        slot_a = str(entry.get("a") or "")
        slot_b = str(entry.get("b") or "")
        if not slot_a or not slot_b or slot_a == slot_b:
            continue
        if known_slots is not None and not {slot_a, slot_b} <= known_slots:
            continue
        if slot_a in claimed or slot_b in claimed:
            continue
        claimed.update((slot_a, slot_b))
        # Ordered so the same pairing always serialises identically.
        first, second = sorted((slot_a, slot_b))
        pairs.append({"a": first, "b": second, "enabled": bool(entry.get("enabled", True))})
    pairs.sort(key=lambda pair: (str(pair["a"]), str(pair["b"])))
    return pairs


def normalized_side_pairs(
    value: object,
    *,
    known_parts: set[str] | None = None,
) -> list[dict[str, object]]:
    """Clean saved explicit side-pair records.

    These records are the user-facing intent: this left-side part is the
    equivalent of this right-side part. Authored RHD/LHD source substitution is
    handled by per-part Replace Source settings, not by this equivalence table.
    """
    if not isinstance(value, (list, tuple)):
        return []
    def base_part_id(ref: str) -> str:
        return ref.split("@@", 1)[0]

    pairs: list[dict[str, object]] = []
    claimed: set[tuple[str, str]] = set()
    for entry in value:
        if not isinstance(entry, dict):
            continue
        left = str(entry.get("left") or "")
        right = str(entry.get("right") or "")
        if not left or not right or left == right:
            continue
        if not bool(entry.get("enabled", True)):
            continue
        if known_parts is not None and not {base_part_id(left), base_part_id(right)} <= known_parts:
            continue
        key = tuple(sorted((left, right)))
        if key in claimed:
            continue
        claimed.add(key)
        kind = str(entry.get("kind") or "part").lower()
        if kind not in {"seat", "mirror", "door", "part"}:
            kind = "part"
        pair = {
            "left": left,
            "right": right,
            "kind": kind,
        }
        pairs.append(pair)
    pairs.sort(key=lambda pair: (str(pair["kind"]), str(pair["left"]), str(pair["right"])))
    return pairs


def set_side_pair(
    conversion: dict[str, object],
    left: str,
    right: str,
    *,
    kind: str = "part",
) -> None:
    """Add or replace one explicit side-pair record."""
    left = left.strip()
    right = right.strip()
    if not left or not right or left == right:
        return
    pairs = [
        pair
        for pair in normalized_side_pairs(conversion.get("sidePairs"))
        if left not in {pair["left"], pair["right"]} and right not in {pair["left"], pair["right"]}
    ]
    entry: dict[str, object] = {
        "left": left,
        "right": right,
        "kind": kind,
    }
    pairs.append(entry)
    conversion["sidePairs"] = normalized_side_pairs(pairs)


def clear_side_pairs(conversion: dict[str, object]) -> None:
    conversion["sidePairs"] = []


# A trigger box is placed by three jbeam nodes plus an offset in the frame
# they build, so what it ends up attached to is a question the automatic
# attribution can only guess at -- and on the scintilla it guesses nothing at
# all for six of sixteen boxes. These records let the user answer directly.
#
# The key is the box's authored position, not the part it is declared in.
# That is what makes the records trim-proof: the same switch declared in a
# road dash and a race dash is one row when both author it at the same place,
# and two rows -- "headlights #1", "headlights #2" -- when they do not. Either
# way the answer travels with the box rather than with a trim's part list.
TRIGGER_POSITION_PLACES = 3  # a millimetre; finer than any authored offset
TRIGGER_MODES = (MODE_SKIP, MODE_TRANSLATE, MODE_MIRROR)
# The trigger transforms that actually move the box, and so have a Move X to
# take. Skip is deliberately absent: it is the mode for leaving a box alone.
TRIGGER_OFFSET_MODES = frozenset({MODE_TRANSLATE, MODE_MIRROR})


def trigger_position_key(
    trigger_id: object,
    position: object,
) -> tuple[str, tuple[float, float, float]] | None:
    """(trigger id, position rounded to the millimetre), or None if unusable."""
    name = str(trigger_id or "")
    if not name or not isinstance(position, (list, tuple)) or len(position) != 3:
        return None
    try:
        rounded = tuple(round(float(value), TRIGGER_POSITION_PLACES) for value in position)
    except (TypeError, ValueError):
        return None
    return name, rounded


def trigger_offset_value(value: object) -> float | None:
    """A per-box Move X override, or None for "use the global delta".

    Signed, exactly like a part's ``translateOffset``: positive travels the way
    the trim converts, negative walks the box back the other way. Zero is not
    an override -- a box that moves nowhere is what Skip is for -- so it reads
    as absent rather than pinning the box at the origin of its own frame.
    """
    if value in (None, ""):
        return None
    try:
        offset = float(value)
    except (TypeError, ValueError):
        return None
    if offset != offset or offset in (float("inf"), float("-inf")) or offset == 0:
        return None
    return offset


def normalized_trigger_modes(value: object) -> list[dict[str, object]]:
    """Clean saved per-trigger transform records, last entry per key winning."""
    if not isinstance(value, (list, tuple)):
        return []
    by_key: dict[tuple[str, tuple[float, float, float]], dict[str, object]] = {}
    for entry in value:
        if not isinstance(entry, dict):
            continue
        key = trigger_position_key(entry.get("id"), entry.get("at"))
        if key is None:
            continue
        mode = str(entry.get("mode") or MODE_SKIP)
        if mode not in TRIGGER_MODES:
            continue
        record: dict[str, object] = {"id": key[0], "at": list(key[1]), "mode": mode}
        # A Move X means something on the modes that move the box -- the whole
        # distance for a Move, a nudge on top of the reflection for a Mirror --
        # and nothing on Skip, which is the one mode that leaves the box alone.
        # An offset left over from a row since set to Skip is dropped rather
        # than kept waiting to reappear.
        offset = (
            trigger_offset_value(entry.get("offset"))
            if mode in TRIGGER_OFFSET_MODES
            else None
        )
        if offset is not None:
            record["offset"] = offset
        by_key[key] = record
    return [by_key[key] for key in sorted(by_key)]


def trigger_mode_map(
    conversion: dict[str, object],
) -> dict[tuple[str, tuple[float, float, float]], str]:
    """(trigger id, authored position) -> the transform the user chose.

    An absent key has no answer from the user, which is what the resolver
    reads as "use the automatic attribution".
    """
    return {
        trigger_position_key(entry["id"], entry["at"]): str(entry["mode"])
        for entry in normalized_trigger_modes(conversion.get("triggers"))
    }


def trigger_offset_map(
    conversion: dict[str, object],
) -> dict[tuple[str, tuple[float, float, float]], float]:
    """(trigger id, authored position) -> the Move X the user typed for it.

    An absent key means the box travels the shared conversion delta.
    """
    offsets: dict[tuple[str, tuple[float, float, float]], float] = {}
    for entry in normalized_trigger_modes(conversion.get("triggers")):
        offset = trigger_offset_value(entry.get("offset"))
        key = trigger_position_key(entry["id"], entry["at"])
        if offset is not None and key is not None:
            offsets[key] = offset
    return offsets


def set_trigger_mode(
    conversion: dict[str, object],
    trigger_id: str,
    position: object,
    mode: str,
) -> None:
    """Record the transform for the box of this id at this position.

    A Move X already typed for this box survives a move between the modes that
    can carry one, and is dropped by ``normalized_trigger_modes`` on Skip.
    """
    key = trigger_position_key(trigger_id, position)
    if key is None or mode not in TRIGGER_MODES:
        return
    existing = {
        trigger_position_key(entry["id"], entry["at"]): entry
        for entry in normalized_trigger_modes(conversion.get("triggers"))
    }
    records = [entry for entry_key, entry in existing.items() if entry_key != key]
    record: dict[str, object] = {"id": key[0], "at": list(key[1]), "mode": mode}
    previous = existing.get(key)
    if previous is not None and previous.get("offset") is not None:
        record["offset"] = previous["offset"]
    records.append(record)
    conversion["triggers"] = normalized_trigger_modes(records)


def set_trigger_offset(
    conversion: dict[str, object],
    trigger_id: str,
    position: object,
    offset: object,
) -> None:
    """Set (or clear, with None/blank) one box's Move X override.

    A box with no transform record has nothing to override -- the Triggers
    table only offers the column on a row already set to Move -- so this does
    nothing rather than inventing a mode the user did not choose.
    """
    key = trigger_position_key(trigger_id, position)
    if key is None:
        return
    records = normalized_trigger_modes(conversion.get("triggers"))
    for entry in records:
        if trigger_position_key(entry["id"], entry["at"]) != key:
            continue
        cleaned = trigger_offset_value(offset)
        if cleaned is None:
            entry.pop("offset", None)
        else:
            entry["offset"] = cleaned
        conversion["triggers"] = normalized_trigger_modes(records)
        return


def clear_trigger_mode(conversion: dict[str, object], trigger_id: str, position: object) -> None:
    """Drop one record, handing that box back to the automatic attribution."""
    key = trigger_position_key(trigger_id, position)
    conversion["triggers"] = [
        entry
        for entry in normalized_trigger_modes(conversion.get("triggers"))
        if key is None or trigger_position_key(entry["id"], entry["at"]) != key
    ]


def clear_trigger_modes(conversion: dict[str, object]) -> None:
    conversion["triggers"] = []


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
        "slotPairs": [],
        "sidePairs": [],
        "triggers": [],
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
                            "includeChildren",
                            PART_TEXTURE_CORRECTION_KEY,
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

    merged["slotPairs"] = normalized_slot_pairs(data.get("slotPairs"))
    merged["sidePairs"] = normalized_side_pairs(data.get("sidePairs"))
    merged["triggers"] = normalized_trigger_modes(data.get("triggers"))

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
                            "includeChildren",
                            PART_TEXTURE_CORRECTION_KEY,
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
    imported_pairs = normalized_slot_pairs(imported.get("slotPairs"))
    if imported_pairs:
        out["slotPairs"] = imported_pairs
        counts["slotPairImported"] = len(imported_pairs)
    imported_side_pairs = normalized_side_pairs(imported.get("sidePairs"))
    if imported_side_pairs:
        out["sidePairs"] = imported_side_pairs
        counts["sidePairImported"] = len(imported_side_pairs)
    imported_triggers = normalized_trigger_modes(imported.get("triggers"))
    if imported_triggers:
        out["triggers"] = imported_triggers
        counts["triggerModeImported"] = len(imported_triggers)
    if isinstance(imported.get("plate"), dict):
        out["plate"] = plate_generator.normalized_plate_binding(imported["plate"])
    return out, counts


def save_conversion(context: VehicleContext, conversion: dict[str, object]) -> Path:
    context.project_dir.mkdir(parents=True, exist_ok=True)
    conversion["toolVersion"] = TOOL_VERSION
    conversion["plate"] = plate_generator.normalized_plate_binding(conversion.get("plate"))
    conversion["slotPairs"] = normalized_slot_pairs(conversion.get("slotPairs"))
    conversion["sidePairs"] = normalized_side_pairs(conversion.get("sidePairs"))
    conversion["triggers"] = normalized_trigger_modes(conversion.get("triggers"))
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

def active_slot_pairs(conversion: dict[str, object]) -> list[tuple[str, str]]:
    """Enabled slot pairings as (slot_a, slot_b) tuples."""
    return [
        (str(pair["a"]), str(pair["b"]))
        for pair in normalized_slot_pairs(conversion.get("slotPairs"))
        if pair.get("enabled")
    ]


def slot_pair_partner(conversion: dict[str, object], slot_type: str) -> str:
    """The slot ``slot_type`` is paired with, or "" when it is unpaired."""
    for slot_a, slot_b in active_slot_pairs(conversion):
        if slot_a == slot_type:
            return slot_b
        if slot_b == slot_type:
            return slot_a
    return ""


def set_slot_pair(conversion: dict[str, object], slot_type: str, partner: str) -> None:
    """Pair ``slot_type`` with ``partner``, or unpair it when partner is empty.

    Both slots are freed of any previous pairing first, so pairing always
    stays a bijection and the user never has to unpair the old partner by
    hand before choosing a new one.
    """
    pairs = [
        pair
        for pair in normalized_slot_pairs(conversion.get("slotPairs"))
        if slot_type not in {pair["a"], pair["b"]}
        and (not partner or partner not in {pair["a"], pair["b"]})
    ]
    if partner and partner != slot_type:
        first, second = sorted((slot_type, partner))
        pairs.append({"a": first, "b": second, "enabled": True})
    conversion["slotPairs"] = normalized_slot_pairs(pairs)


__all__ = ['normalized_slot_pairs', 'normalized_side_pairs', 'set_side_pair', 'clear_side_pairs', 'TRIGGER_MODES', 'TRIGGER_OFFSET_MODES', 'trigger_position_key', 'trigger_offset_value', 'normalized_trigger_modes', 'trigger_mode_map', 'trigger_offset_map', 'set_trigger_mode', 'set_trigger_offset', 'clear_trigger_mode', 'clear_trigger_modes', 'active_slot_pairs', 'slot_pair_partner', 'set_slot_pair', 'default_part_setting', 'default_part_settings', 'default_variant_settings', 'variant_build_mode', 'set_variant_build_mode', 'base_conversion_config', 'conversion_path', 'load_or_create_conversion', 'merge_with_current_inventory', 'import_matching_conversion', 'save_conversion', 'load_app_settings', 'save_app_settings']
