"""Implementation slice extracted from ``beamng_hand_drive_core``.

Original source lines 5810-6096. Import the public orchestration module
``beamng_hand_drive_core`` rather than this implementation module directly.
"""

from __future__ import annotations

import copy
import json
import shutil
from datetime import UTC, datetime
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
    MODE_MIRROR_POSITION,
    MODE_REPLACE_SOURCE,
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

def package_name_for_context(context: VehicleContext) -> str:
    return f"{context.source_zip.stem}_XP_conversion.zip"


def write_mod_info(root: Path, context: VehicleContext) -> None:
    mod_info = root / "mod_info"
    mod_info.mkdir(parents=True, exist_ok=True)
    source_name = conversion_source_name(context)
    info = {
        "name": f"{context.vehicle_id} BeamXP Conversion",
        "version": "0.2.1",
        "authors": source_name,
        "description": (
            f"Generated BeamXP handedness and/or plate configuration overlay for {context.vehicle_id}. "
            f"Depends on {context.source_zip.name}."
        ),
        "source": source_name,
    }
    write_text_file(mod_info / "info.json", json.dumps(info, indent=2), encoding="utf-8")


def selected_variant_targets(
    context: VehicleContext,
    conversion: dict[str, object],
) -> tuple[dict[str, str], dict[str, str]]:
    targets: dict[str, str] = {}
    skipped: dict[str, str] = {}
    variants = conversion.get("variants", {})
    if not isinstance(variants, dict):
        return targets, skipped
    for config_name, settings in variants.items():
        if config_name not in context.variants or not isinstance(settings, dict):
            continue
        if variant_build_mode(settings) not in {BUILD_CONVERTED, BUILD_BOTH}:
            continue
        source_hand = effective_source_hand(context, conversion, config_name)
        target = target_hand_for(source_hand, ACTION_OPPOSITE)
        if target is None:
            skipped[config_name] = f"No opposite target for source hand {source_hand}"
        else:
            targets[config_name] = target
    return targets, skipped


def selected_output_plans(
    context: VehicleContext,
    conversion: dict[str, object],
) -> tuple[list[dict[str, object]], dict[str, str]]:
    """Expand each trim row into zero, one, or two generated configs."""
    targets, skipped = selected_variant_targets(context, conversion)
    plans: list[dict[str, object]] = []
    variants = conversion.get("variants", {})
    if not isinstance(variants, dict):
        return plans, skipped
    for config_name, settings in sorted(variants.items()):
        if config_name not in context.variants or not isinstance(settings, dict):
            continue
        mode = variant_build_mode(settings)
        if mode in {BUILD_CONVERTED, BUILD_BOTH} and config_name in targets:
            target = targets[config_name]
            plans.append({
                "source": config_name,
                "kind": BUILD_CONVERTED,
                "targetHand": target,
                "output": variant_output_name(config_name, target),
            })
        if mode in {BUILD_ORIGINAL, BUILD_BOTH}:
            plans.append({
                "source": config_name,
                "kind": BUILD_ORIGINAL,
                "targetHand": None,
                "output": original_plate_output_name(config_name),
            })
    return plans, skipped


def split_authored_hand_drive_targets(
    context: VehicleContext,
    conversion: dict[str, object],
    variant_targets: dict[str, str],
) -> tuple[dict[str, str], dict[str, dict[str, object]]]:
    """Resolve authored LHD/RHD swaps, keeping every variant in the build.

    An authored swap covers its own subtree only. The rest of the trim -- seats,
    pedals, mirrors, anything hung off the body rather than the dashboard --
    still needs the generated mirroring pass, so the two run together on the
    same variant instead of one displacing the other.
    """
    generated: dict[str, str] = dict(variant_targets)
    authored: dict[str, dict[str, object]] = {}
    for config_name, target_hand in sorted(variant_targets.items()):
        source_hand = effective_source_hand(context, conversion, config_name)
        group = find_hand_authored_opposite_group(
            context,
            config_name,
            source_hand,
            target_hand,
        )
        if group is not None:
            authored[config_name] = group
    return generated, authored


def generated_mesh_scope(
    context: VehicleContext,
    selected_configs: Iterable[str],
    authored_groups: dict[str, dict[str, object]],
    slot_pair_plans: dict[str, dict[str, object]] | None = None,
) -> set[str]:
    """Meshes the generated pass may transform.

    Per trim, everything the trim uses minus the parts an authored swap or a
    slot pair already resolves -- those land as stock opposite-side parts, and
    mirroring their meshes on top would undo the swap.
    """
    slot_pair_plans = slot_pair_plans or {}
    scope: set[str] = set()
    for config_name in selected_configs:
        covered = authored_group_meshes(context, authored_groups.get(config_name))
        covered |= authored_group_meshes(context, slot_pair_plans.get(config_name))
        scope.update(used_meshes_for_config(context, config_name) - covered)
    return scope


def relocation_meshes(
    context: VehicleContext,
    slot_pair_plans: dict[str, dict[str, object]],
) -> set[str]:
    """Meshes belonging to parts that must be rebuilt on the other side.

    A relocated part has no stock counterpart to inherit geometry from, so its
    meshes always need a mirrored bake regardless of the mode the user left on
    them -- the part is crossing the car either way.
    """
    meshes: set[str] = set()
    for plan in slot_pair_plans.values():
        for relocation in slot_pair_plan_relocations(plan):
            found = part_body_for_context(context, str(relocation.get("partId") or ""))
            if found is not None:
                meshes.update(transform_helpers.extract_part_mesh_names(found[0]))
    return {mesh for mesh in meshes if mesh in context.objects}


def build_batch(
    context: VehicleContext,
    conversion: dict[str, object],
    *,
    write_zip: bool = True,
    install: bool = False,
    mods_folder: Path | None = None,
) -> BuildResult:
    output_plans, skipped = selected_output_plans(context, conversion)
    if not output_plans:
        raise RuntimeError("No trim outputs are selected")
    variant_targets, skipped = selected_variant_targets(context, conversion)
    generated_variant_targets, authored_groups = split_authored_hand_drive_targets(
        context, conversion, variant_targets
    )
    slot_pair_plans = slot_pair_plans_for_variants(
        context, conversion, sorted(variant_targets)
    )
    original_configs = [
        str(plan["source"])
        for plan in output_plans
        if plan["kind"] == BUILD_ORIGINAL
    ]
    no_op_originals = [
        config_name
        for config_name in original_configs
        if not plate_generator.variant_has_plate_changes(conversion, config_name, context)
    ]
    if no_op_originals:
        raise RuntimeError(
            "Plates Only output has no plate changes for: "
            + ", ".join(no_op_originals[:8])
            + ("..." if len(no_op_originals) > 8 else "")
            + ". Choose a plate design, a different physical plate, or None for at least one side."
        )

    object_modes: dict[str, str] = {}
    structural_sources: dict[str, str] = {}
    node_mirror_map: dict[str, str] = {}
    translated_prop_meshes: set[str] = set()
    mirrored_prop_meshes: set[str] = set()
    structural_prop_meshes: set[str] = set()
    mirror_position_prop_meshes: set[str] = set()
    mirror_position_flexbody_meshes: set[str] = set()
    translated_flexbody_meshes: set[str] = set()
    translate_magnitudes: dict[str, float] = {}
    texture_flip_ids: set[str] = set()
    if generated_variant_targets:
        object_modes = active_part_modes(conversion)
        # A vehicle whose every converted trim is fully covered by an authored
        # swap needs no transformed parts at all, so an empty mode set is only
        # an error when nothing authored is carrying the conversion.
        swap_driven = bool(authored_groups or slot_pair_plans)
        if not object_modes and not swap_driven:
            raise RuntimeError(
                "Converted outputs require at least one Move, Mirror Move, Mirror, "
                "Swap Mesh, Replace Source, or slot pair"
            )
        selected_configs = sorted(generated_variant_targets)
        flexbody_meshes, prop_meshes, _all_meshes = selected_mesh_roles(context, selected_configs)
        mesh_scope = generated_mesh_scope(
            context, selected_configs, authored_groups, slot_pair_plans
        )
        if mesh_scope:
            object_modes = {mesh: mode for mesh, mode in object_modes.items() if mesh in mesh_scope}
        for mesh in relocation_meshes(context, slot_pair_plans):
            object_modes[mesh] = MODE_MIRROR
        if not object_modes and not swap_driven:
            raise RuntimeError(
                "No Move, Mirror Move, Mirror, Swap Mesh, or Replace Source entries are used "
                "by the converted trims, and no slot pair applies to them"
            )
        texture_flip_ids = texture_flip_mesh_ids(context, object_modes)
        object_modes = fallback_structural_part_modes(
            context,
            conversion,
            object_modes,
            selected_configs=selected_configs,
        )
        structural_sources = structural_mirror_sources(context, conversion, object_modes)
        node_mirror_map = build_node_mirror_map(context.node_positions)
        translated_prop_meshes = {
            mesh for mesh, mode in object_modes.items() if mode == MODE_TRANSLATE and mesh in prop_meshes
        }
        mirrored_prop_meshes = {
            mesh for mesh, mode in object_modes.items() if mode == MODE_MIRROR and mesh in prop_meshes
        }
        mirror_position_prop_meshes = {
            mesh for mesh, mode in object_modes.items() if mode == MODE_MIRROR_POSITION and mesh in prop_meshes
        }
        structural_prop_meshes = {
            mesh
            for mesh, mode in object_modes.items()
            if mode in {MODE_MIRROR_STRUCTURAL, MODE_REPLACE_SOURCE} and mesh in prop_meshes
        }
        translated_flexbody_meshes = {
            mesh
            for mesh, mode in object_modes.items()
            if mode == MODE_TRANSLATE and mesh in flexbody_meshes and mesh not in translated_prop_meshes
        }
        mirror_position_flexbody_meshes = {
            mesh
            for mesh, mode in object_modes.items()
            if mode == MODE_MIRROR_POSITION and mesh in flexbody_meshes and mesh not in mirror_position_prop_meshes
        }
        translate_magnitudes = part_translate_magnitudes(context, conversion, object_modes)
        zero_translate = sorted(
            object_id
            for object_id, mode in object_modes.items()
            if mode == MODE_TRANSLATE and translate_magnitudes.get(object_id, 0.0) <= 0
        )
        if zero_translate:
            raise RuntimeError(
                "Delta X magnitude is zero for translated part(s): "
                + ", ".join(zero_translate[:8])
                + ("..." if len(zero_translate) > 8 else "")
                + ". Select a steering reference, enter a global manual delta, or set per-part offsets."
            )

    output_root = context.project_dir / "unpacked_output"
    build_dir = context.project_dir / "build"
    clean_dir(output_root)
    build_dir.mkdir(parents=True, exist_ok=True)
    output_vehicle_dir = output_root / context.vehicle_path

    baked_shared_specs: list[BakedMeshSpec] = []
    generated_configs: list[str] = []
    generated_daes: list[Path] = []
    if variant_targets:
        generated_configs.extend(write_generated_jbeam_and_configs(
            context,
            output_vehicle_dir,
            conversion,
            object_modes,
            structural_sources,
            node_mirror_map,
            variant_targets,
            translate_magnitudes,
            translated_prop_meshes,
            translated_flexbody_meshes,
            mirrored_prop_meshes,
            mirror_position_prop_meshes,
            mirror_position_flexbody_meshes,
            structural_prop_meshes,
            baked_shared_specs,
            authored_groups,
            slot_pair_plans,
        ))
    if generated_variant_targets and object_modes:
        generated_daes = generate_daes(
            context,
            output_root,
            output_vehicle_dir,
            object_modes,
            structural_sources,
            set(generated_variant_targets.values()),
            translate_magnitudes,
            translated_prop_meshes,
            translated_flexbody_meshes,
            context.jbeam_positioned_flexbodies,
            baked_shared_specs,
            texture_flip_ids,
        )
    generated_configs.extend(write_original_plate_configs(
        context,
        output_vehicle_dir,
        conversion,
        original_configs,
    ))
    generated_configs.sort()
    write_mod_info(output_root, context)
    # Licence plates are generated as a separate pass over the written output
    # so plate logic stays fully decoupled from the handedness transforms.
    try:
        plate_summary = plate_generator.apply_to_build(
            context,
            conversion,
            output_root,
            output_vehicle_dir,
            output_plans,
        )
    except plate_generator.PlateError as exc:
        raise RuntimeError(str(exc)) from exc
    embedded_dir = output_root / "handedness_conversion"
    embedded_dir.mkdir(parents=True, exist_ok=True)
    delta = conversion.setdefault("delta", {})
    if isinstance(delta, dict):
        delta["steeringRefs"] = selected_steering_refs(conversion)
    embedded = copy.deepcopy(conversion)
    embedded["builtAt"] = datetime.now(UTC).isoformat(timespec="seconds")
    embedded["build"] = {
        "generatedConfigs": generated_configs,
        "outputs": output_plans,
        "targetHands": variant_targets,
        "generatedTargetHands": generated_variant_targets,
        "authoredHandDriveGroups": authored_groups,
        "slotPairPlans": slot_pair_plans,
        "deltaMagnitude": delta_magnitude(context, conversion),
        "translateMagnitudes": translate_magnitudes,
        "mirroredPropMeshes": sorted(mirrored_prop_meshes),
        "textureFlipMeshes": sorted(texture_flip_ids),
        "structuralMirrorSources": structural_sources,
        "structuralPropMeshes": sorted(structural_prop_meshes),
        "bakedSharedMeshCount": len(baked_shared_specs),
        "cameraNodeMirrorCount": len(node_mirror_map),
        "plates": plate_summary,
    }
    write_text_file(embedded_dir / "conversion.json", json.dumps(embedded, indent=2), encoding="utf-8")

    package_zip = None
    installed_zip = None
    installed_plates_zip = None
    if write_zip:
        package_zip = build_dir / package_name_for_context(context)
        make_zip(output_root, package_zip)
    if install:
        if package_zip is None:
            raise RuntimeError("Install requires zip build")
        if mods_folder is None:
            raise RuntimeError("Install requested without a mods folder")
        mods_folder.mkdir(parents=True, exist_ok=True)
        installed_zip = mods_folder / package_zip.name
        shutil.copy2(package_zip, installed_zip)
        # Refresh the universal plates mod alongside the vehicle so every
        # library design stays selectable on any vehicle, not just the sets
        # bound to this build. A broken library set must not fail the build.
        try:
            plates_mod = plate_generator.export_all_plate_sets()
        except plate_generator.PlateError as exc:
            plate_summary.setdefault("warnings", []).append(f"plates library mod not refreshed: {exc}")
        else:
            if plates_mod is not None:
                plates_zip = Path(plates_mod["zip"])
                installed_plates_zip = mods_folder / plates_zip.name
                shutil.copy2(plates_zip, installed_plates_zip)
                plate_summary["libraryModDesigns"] = plates_mod["designs"]

    save_conversion(context, conversion)
    return BuildResult(
        unpacked_dir=output_root,
        package_zip=package_zip,
        installed_zip=installed_zip,
        generated_configs=generated_configs,
        generated_daes=generated_daes,
        skipped_variants=skipped,
        plate_summary=plate_summary,
        installed_plates_zip=installed_plates_zip,
    )

__all__ = ['generated_mesh_scope', 'relocation_meshes', 'package_name_for_context', 'write_mod_info', 'selected_variant_targets', 'selected_output_plans', 'split_authored_hand_drive_targets', 'build_batch']
