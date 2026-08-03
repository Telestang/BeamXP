"""Implementation slice extracted from ``beamng_hand_drive_core``.

Original source lines 4885-5809. Import the public orchestration module
``beamng_hand_drive_core`` rather than this implementation module directly.
"""

from __future__ import annotations

import copy
import json
import math
import re
import textwrap
import zipfile
from pathlib import Path
from collections.abc import Iterable
from xml.etree import ElementTree as ET
from beamxp import transform_helpers
from beamxp.core.beam_json import (
    add_missing_json_commas,
    display_name_from_localization_key,
    display_name_for,
    info_path_for_config,
    json_line_needs_comma,
    load_info,
    load_pc,
    parse_beamng_json,
    strip_json_comments,
    zip_json_by_name as _zip_json_by_name,
)
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
from beamxp.core.geometry import (
    PROP_VECTOR_RE,
    brg_rotation_matrix3,
    clamp_value,
    cross_product,
    euler_from_matrix3,
    euler_matrix3,
    euler_yzx_from_matrix3,
    identity_matrix,
    matrix3_from_axes,
    matrix3_from_matrix4,
    matrix4_flat,
    mirror_rotation_matrix_x,
    mirror_x_matrix4,
    multiply_matrix,
    multiply_matrix3,
    normalize_vector,
    prop_base_rotation_matrix3,
    prop_row_vector_objects,
    rotation_transpose_matrix3,
    rotation_transpose_matrix4,
    rotation_x_matrix,
    rotation_y_matrix,
    rotation_z_matrix,
    scale_matrix,
    sign_number,
    translation_matrix,
    vector_subtract,
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
from beamxp.hand_drive_parts.spatial_analysis import display_texture_flip_scope
from beamxp.plates import generator as plate_generator

def generate_daes(
    context: VehicleContext,
    output_root: Path,
    output_vehicle_dir: Path,
    object_modes: dict[str, str],
    structural_sources: dict[str, str],
    target_hands: set[str],
    translate_magnitudes: dict[str, float],
    translated_prop_meshes: set[str],
    translated_flexbody_meshes: set[str],
    jbeam_positioned_flexbodies: set[str],
    baked_shared_specs: list[BakedMeshSpec],
    texture_flip_ids: set[str] | None = None,
) -> list[Path]:
    texture_flip_ids = texture_flip_ids or set()
    # Per-mesh display material scope: which UV island(s) to reflect. A
    # dedicated screen mesh maps to all its symbols (whole-mesh flip); a shared
    # mesh (nav screen + cluster) maps to only the display islands.
    display_flip_scope = display_texture_flip_scope(context)
    generated: list[Path] = []
    objects_by_dae: dict[tuple[Path, str], list[tuple[str, str]]] = {}
    for object_id in object_modes:
        source_id = structural_sources.get(object_id, object_id)
        source_obj = context.objects.get(source_id)
        if source_obj is None or not source_obj.dae_path:
            continue
        dae_source_zip = source_obj.dae_source_zip or context.source_zip
        objects_by_dae.setdefault((dae_source_zip, source_obj.dae_path), []).append((object_id, source_id))

    baked_by_dae: dict[tuple[Path, str], list[BakedMeshSpec]] = {}
    for spec in baked_shared_specs:
        source_obj = context.objects.get(spec.source_mesh)
        if source_obj is None or not source_obj.dae_path:
            continue
        dae_source_zip = source_obj.dae_source_zip or context.source_zip
        baked_by_dae.setdefault((dae_source_zip, source_obj.dae_path), []).append(spec)

    dae_keys = sorted(
        set(objects_by_dae) | set(baked_by_dae),
        key=lambda item: (str(item[0]).lower(), item[1].lower()),
    )
    for dae_source_zip, dae_path in dae_keys:
        object_pairs = objects_by_dae.get((dae_source_zip, dae_path), [])
        baked_specs = baked_by_dae.get((dae_source_zip, dae_path), [])
        tree = parse_dae(dae_source_zip, dae_path)
        root = tree.getroot()
        library_geometries = root.find("c:library_geometries", NS)
        library_visual_scenes = root.find("c:library_visual_scenes", NS)
        if library_geometries is None or library_visual_scenes is None:
            continue
        geometries_by_id = {
            geom.get("id"): geom
            for geom in library_geometries.findall("c:geometry", NS)
            if geom.get("id")
        }
        selected_nodes: list[ET.Element] = []
        generated_geometry: dict[str, ET.Element] = {}

        for object_id, source_id in sorted(object_pairs):
            mode = object_modes[object_id]
            source_obj = context.objects.get(source_id)
            source_node_id = source_obj.id if source_obj is not None else source_id
            source_node = find_dae_node(root, source_node_id)
            if source_node is None:
                continue
            for target_hand in sorted(target_hands):
                suffix = suffix_for_hand(target_hand)
                signed_delta = signed_delta_for_target(
                    target_hand,
                    translate_magnitudes.get(object_id, 0.0),
                )
                new_name = generated_mesh_name(object_id, target_hand)
                new_node = copy.deepcopy(source_node)
                new_node.set("id", new_name)
                new_node.set("name", new_name)

                matrix_elem = new_node.find("c:matrix", NS)
                parsed_matrix = None
                translate_delta = None
                if matrix_elem is not None and matrix_elem.text:
                    parsed_matrix = transform_helpers.parse_matrix(matrix_elem.text)
                    if mode in {MODE_MIRROR, MODE_MIRROR_STRUCTURAL, MODE_REPLACE_SOURCE}:
                        matrix_elem.text = transform_helpers.format_matrix(transform_helpers.mirror_matrix_x(parsed_matrix))
                    elif mode == MODE_TRANSLATE:
                        if object_id in translated_prop_meshes or (
                            object_id in translated_flexbody_meshes
                            and object_id in jbeam_positioned_flexbodies
                        ):
                            pass
                        elif (
                            object_id in translated_flexbody_meshes
                            and object_id not in jbeam_positioned_flexbodies
                        ):
                            translate_delta = transform_helpers.local_delta_for_world_translation(
                                parsed_matrix,
                                (signed_delta, 0.0, 0.0),
                            )
                        else:
                            matrix_elem.text = transform_helpers.format_matrix(
                                transform_helpers.translate_matrix_x(parsed_matrix, signed_delta)
                            )

                for inst in new_node.findall(".//c:instance_geometry", NS):
                    url = inst.get("url", "")
                    if not url.startswith("#"):
                        continue
                    old_geom_id = url[1:]
                    old_geom = geometries_by_id.get(old_geom_id)
                    if old_geom is None:
                        continue
                    new_geom_id = safe_id(f"{old_geom_id}{suffix}_{object_id}")
                    if new_geom_id not in generated_geometry:
                        if mode in {MODE_MIRROR, MODE_MIRROR_STRUCTURAL, MODE_REPLACE_SOURCE}:
                            generated_geometry[new_geom_id] = transform_helpers.mirrored_geometry(
                                old_geom,
                                new_geom_id,
                                flip_texture=object_id in texture_flip_ids,
                                flip_materials=display_flip_scope.get(object_id),
                            )
                        elif (
                            mode == MODE_TRANSLATE
                            and object_id in translated_flexbody_meshes
                            and object_id not in jbeam_positioned_flexbodies
                        ):
                            if translate_delta is None:
                                raise RuntimeError(f"Missing translated geometry delta for {object_id}")
                            generated_geometry[new_geom_id] = transform_helpers.translated_geometry(
                                old_geom,
                                new_geom_id,
                                translate_delta,
                            )
                        else:
                            generated_geometry[new_geom_id] = transform_helpers.copied_geometry(old_geom, new_geom_id)
                    inst.set("url", f"#{new_geom_id}")
                    if inst.get("name"):
                        inst.set("name", new_name)

                selected_nodes.append(new_node)

        for spec in baked_specs:
            source_obj = context.objects.get(spec.source_mesh)
            if source_obj is None:
                continue
            source_node = find_dae_node(root, source_obj.id)
            if source_node is None:
                continue
            new_node = copy.deepcopy(source_node)
            new_node.set("id", spec.output_mesh)
            new_node.set("name", spec.output_mesh)

            matrix_elem = new_node.find("c:matrix", NS)
            if matrix_elem is not None and matrix_elem.text:
                source_node_matrix = transform_helpers.parse_matrix(matrix_elem.text)
                matrix_elem.text = transform_helpers.format_matrix(
                    baked_dae_matrix(source_node_matrix, spec, translate_magnitudes)
                )

            for inst in new_node.findall(".//c:instance_geometry", NS):
                url = inst.get("url", "")
                if not url.startswith("#"):
                    continue
                old_geom_id = url[1:]
                old_geom = geometries_by_id.get(old_geom_id)
                if old_geom is None:
                    continue
                new_geom_id = safe_id(f"{old_geom_id}_{spec.output_mesh}")
                if new_geom_id not in generated_geometry:
                    if spec.mode in {MODE_MIRROR, MODE_MIRROR_STRUCTURAL, MODE_REPLACE_SOURCE}:
                        generated_geometry[new_geom_id] = transform_helpers.mirrored_geometry(
                            old_geom,
                            new_geom_id,
                            flip_texture=spec.configured_mesh in texture_flip_ids,
                            flip_materials=display_flip_scope.get(spec.configured_mesh),
                        )
                    else:
                        generated_geometry[new_geom_id] = transform_helpers.copied_geometry(old_geom, new_geom_id)
                inst.set("url", f"#{new_geom_id}")
                if inst.get("name"):
                    inst.set("name", spec.output_mesh)

            selected_nodes.append(new_node)

        if not selected_nodes:
            continue
        for child in list(library_geometries):
            library_geometries.remove(child)
        for geom in generated_geometry.values():
            library_geometries.append(geom)
        for visual_scene in library_visual_scenes.findall("c:visual_scene", NS):
            for child in list(visual_scene):
                visual_scene.remove(child)
            for node in selected_nodes:
                visual_scene.append(node)

        target = generated_dae_output_path(output_root, output_vehicle_dir, context, dae_path)
        write_xml_tree(tree, target)
        generated.append(target)

    return generated


def variant_output_name(config_name: str, target_hand: str) -> str:
    suffix = "_rhd" if target_hand == HAND_RHD else "_lhd"
    if config_name.lower().endswith(suffix):
        return config_name
    return f"{config_name}{suffix}"


def original_plate_output_name(config_name: str) -> str:
    return config_name if config_name.lower().endswith("_plates") else f"{config_name}_plates"


def append_hand_label(name: object, target_hand: str) -> str:
    text = str(name or "").strip()
    if not text:
        return target_hand
    if re.search(rf"(?:\s|\(){re.escape(target_hand)}\)?$", text, re.IGNORECASE):
        return text
    return f"{text} {target_hand}"


def generated_info_display_name(info: dict[str, object], variant: VariantInfo) -> str:
    for key in ("Configuration", "Name", "name", "configuration"):
        value = info.get(key)
        if isinstance(value, str) and value.strip():
            return display_name_from_localization_key(value) or value.strip()
    return variant.display_name


def generated_info_description(info: dict[str, object]) -> str:
    for key in ("Description", "description"):
        value = info.get(key)
        if not isinstance(value, str):
            continue
        text = value.strip()
        if not text or display_name_from_localization_key(text):
            return ""
        return text
    return ""


def apply_hand_authored_group(
    pc: dict[str, object],
    group: dict[str, object],
) -> None:
    """Apply a resolved authored LHD/RHD subtree swap to a configuration."""
    main_part = group.get("mainPart")
    if isinstance(main_part, str) and main_part:
        pc["mainPartName"] = main_part

    parts = dict(pc.get("parts", {}))
    original_parts = dict(parts)
    clears = group.get("clears", ())
    if not isinstance(clears, (list, tuple)):
        clears = ()
    for clear in clears:
        if not isinstance(clear, dict):
            continue
        slot_id = str(clear.get("slotId") or "")
        slot_path = str(clear.get("slotPath") or "")
        if slot_path:
            parts.pop(slot_path, None)
        if slot_id:
            parts.pop(slot_id, None)

    removals: list[str] = []
    writes: list[tuple[str, str]] = []
    selections = group.get("selections", ())
    if isinstance(selections, (list, tuple)):
        for selection in selections:
            if not isinstance(selection, dict):
                continue
            part_id = str(selection.get("partId") or "")
            slot_id = str(selection.get("slotId") or "")
            slot_path = str(selection.get("slotPath") or "")
            source_slot_id = str(selection.get("sourceSlotId") or slot_id)
            source_slot_path = str(selection.get("sourceSlotPath") or slot_path)
            if not part_id or not slot_id:
                continue

            used_path = bool(source_slot_path and source_slot_path in original_parts)
            if source_slot_path and source_slot_path != slot_path:
                removals.append(source_slot_path)
            if source_slot_id and source_slot_id != slot_id:
                removals.append(source_slot_id)
            key = slot_path if used_path and slot_path else slot_id
            writes.append((key, part_id))

    # Every vacated key goes before every write. A two-way swap has each side
    # naming the other as its source, so interleaving would let the second
    # selection's vacate delete what the first selection just placed.
    for key in removals:
        parts.pop(key, None)
    for key, part_id in writes:
        parts[key] = part_id

    # Last, because a selection moving out of a slot pops that slot's key on
    # its way past. Dropping the key would let the engine re-apply the slot's
    # authored default -- for bx_seat_FL, the very seat that just moved across
    # -- so a slot a swap deliberately emptied has to say so explicitly.
    for clear in clears:
        if not isinstance(clear, dict) or not clear.get("setEmpty"):
            continue
        slot_id = str(clear.get("slotId") or "")
        if slot_id and slot_id not in parts:
            parts[slot_id] = ""
    pc["parts"] = parts


def write_converted_config(
    context: VehicleContext,
    output_vehicle_dir: Path,
    variant: VariantInfo,
    config_name: str,
    target_hand: str,
    pc: dict[str, object],
) -> str:
    output_config = variant_output_name(config_name, target_hand)
    pc["licenseName"] = append_hand_label(pc.get("licenseName") or context.vehicle_id, target_hand)
    clear_default_config_flags(pc)
    output_vehicle_dir.mkdir(parents=True, exist_ok=True)
    write_text_file(
        output_vehicle_dir / f"{output_config}.pc",
        json.dumps(pc, indent=2),
        encoding="utf-8",
    )

    info: dict[str, object] = {}
    if variant.info_path:
        try:
            info = load_info(context.source_zip, variant.info_path)
        except Exception:
            info = {}
    existing_name = generated_info_display_name(info, variant)
    existing_description = generated_info_description(info)
    converted_name = append_hand_label(existing_name, target_hand)
    info["Configuration"] = converted_name
    info["Name"] = converted_name
    info["Description"] = converted_description(existing_description, target_hand)
    info["Config Type"] = "Custom"
    info["Source"] = conversion_source_name(context)
    clear_default_config_flags(info)
    write_text_file(
        output_vehicle_dir / f"info_{output_config}.json",
        json.dumps(info, indent=2),
        encoding="utf-8",
    )
    write_mirrored_preview(context, output_vehicle_dir, config_name, output_config)
    return output_config


DEFAULT_CONFIG_FLAG_KEYS = (
    "default",
    "defaultConfig",
    "isDefault",
    "isDefaultConfig",
    "isDefaultForSubCluster",
)


def clear_default_config_flags(data: dict[str, object]) -> None:
    """Generated configs must never become the vehicle selector default."""
    for key in DEFAULT_CONFIG_FLAG_KEYS:
        if key in data:
            data[key] = False
    # This is model-level metadata, but strip it if a mod placed it on an info
    # file we are cloning into a generated config.
    data.pop("default_pc", None)


def _replace_source_part_for_config(
    context: VehicleContext,
    conversion: dict[str, object],
    part_id: str,
) -> str | None:
    parts = conversion.get("parts", {})
    if not isinstance(parts, dict):
        return None
    settings = parts.get(part_id)
    if not isinstance(settings, dict) or settings.get("mode") != MODE_REPLACE_SOURCE:
        return None
    source_id = str(settings.get("mirrorSource") or "")
    if not source_id or source_id == part_id:
        return None
    if part_body_for_context(context, source_id) is None:
        return None
    return source_id


def apply_replace_source_slot_updates(
    context: VehicleContext,
    conversion: dict[str, object],
    selected: dict[str, object],
    pc: dict[str, object],
    target_hand: str,
) -> None:
    """Select authored replacement parts directly in generated configs.

    Mesh-level Replace Source preview/build transforms still cover cases where
    the replacement is only a render source. When the selected JBeam part itself
    has an authored counterpart, the output config should use that counterpart
    instead of a generated mirror clone; otherwise parts like BX wing mirrors
    keep their old physical nodes and attach on the wrong side.
    """
    raw_parts = pc.get("parts", {})
    parts = dict(raw_parts) if isinstance(raw_parts, dict) else {}
    suffix = suffix_for_hand(target_hand)
    for instance in selected.get("part_instances", ()):
        if not isinstance(instance, dict):
            continue
        part_id = str(instance.get("part_id") or "")
        source_id = _replace_source_part_for_config(context, conversion, part_id)
        if source_id is None:
            continue
        slot_id = str(instance.get("slot_id") or "")
        slot_path = str(instance.get("slot_path") or "")
        if slot_id and slot_id != "main":
            parts[slot_id] = source_id
            parts[f"{slot_id}{suffix}"] = source_id
        if slot_path and slot_path != "/":
            parts[slot_path] = source_id
            parts.pop(f"{slot_path.rstrip('/')}{suffix}/", None)
    pc["parts"] = parts


def apply_authored_group_suffixed_slot_updates(
    pc: dict[str, object],
    authored_group: dict[str, object] | None,
    target_hand: str,
) -> None:
    if authored_group is None:
        return
    raw_parts = pc.get("parts", {})
    parts = dict(raw_parts) if isinstance(raw_parts, dict) else {}
    suffix = suffix_for_hand(target_hand)
    for selection in authored_group.get("selections", ()):
        if not isinstance(selection, dict):
            continue
        slot_id = str(selection.get("slotId") or "")
        part_id = str(selection.get("partId") or "")
        if slot_id and slot_id != "main" and part_id:
            parts[f"{slot_id}{suffix}"] = part_id
    pc["parts"] = parts


def _relocation_rewrite_context(context: VehicleContext) -> dict[str, object]:
    """The vehicle-wide name maps a relocation clone rewrites against.

    Built once per build: the node mirror map is an O(n^2) scan of every node
    in the vehicle, and every relocation in every trim wants the same one.
    """
    groups = vehicle_node_group_names(context)
    slots = {
        slot_def.slot_type
        for part_id in context.part_body_index
        for slot_def in extract_slot_defs(context.part_body_index[part_id][0])
    }
    return {
        "node_mirror_map": build_node_mirror_map(context.node_positions),
        "known_nodes": set(context.node_positions),
        "group_map": build_lateral_name_map(groups),
        "known_groups": groups,
        "slot_map": build_lateral_name_map(slots),
        "known_slots": slots,
    }


def relocated_part_name(source_part: str, target_hand: str, target_slot: str) -> str:
    """Name a relocation clone after where it is going.

    Unlike an ordinary generated part, one source part can produce two
    differently-slotted clones, so the target slot has to be part of the name.
    """
    return f"{source_part}{suffix_for_hand(target_hand)}__{target_slot}"


def _relocation_node_offset(
    context: VehicleContext,
    config_name: str,
    source_slot: str,
    target_slot: str,
) -> tuple[float, float, float]:
    """How far the target slot's own mounting misses the mirrored original.

    Zero for an ordinary mirrored pair. Non-zero when the two slots are not
    mirror images -- a co-driver seat mounted further back and higher -- in
    which case the difference is cancelled so the part lands where a driver
    sits on the other side rather than where the co-driver sat.
    """
    usage = slot_usage_for_configs(context, [config_name])
    source_usage = usage.get(source_slot)
    target_usage = usage.get(target_slot)
    if source_usage is None or target_usage is None:
        return (0.0, 0.0, 0.0)
    source_offset = slot_node_offset(source_usage, config_name)
    target_offset = slot_node_offset(target_usage, config_name)
    mirrored = (-source_offset[0], source_offset[1], source_offset[2])
    return tuple(mirrored[axis] - target_offset[axis] for axis in range(3))


def _relocation_clone_body(
    context: VehicleContext,
    config_name: str,
    target_hand: str,
    relocation: dict[str, object],
    new_part_id: str,
    object_modes: dict[str, str],
    node_mirror_map: dict[str, str],
    prop_node_positions: dict[str, tuple[float, float, float]],
    inherited_options: tuple[str, ...],
    baked_shared_specs: list[BakedMeshSpec],
    rewrite_context: dict[str, object],
) -> str | None:
    source_part_id = str(relocation.get("partId") or "")
    target_slot = str(relocation.get("slotId") or "")
    source_slot = str(relocation.get("sourceSlotId") or "")
    found = part_body_for_context(context, source_part_id)
    if found is None or not target_slot:
        return None

    part_body = found[0]
    suffix = suffix_for_hand(target_hand)
    meshes = sorted(transform_helpers.extract_part_mesh_names(part_body))
    mesh_map = {
        mesh: f"{mesh}{suffix}"
        for mesh in meshes
        if context.objects.get(mesh) is not None and context.objects[mesh].dae_path
    }
    # A relocated part always mirrors: it is moving to the other side of the
    # car, whatever mode its individual meshes were left on.
    row_transforms = {mesh: ("mirror", 0.0) for mesh in meshes}
    prop_globals = {
        mesh: mirrored_object_position(context, mesh, config_name)
        for mesh in meshes
        if mesh in context.objects
    }

    shared_bake = SharedBakeContext(
        context=context,
        config_name=config_name,
        target_hand=target_hand,
        source_part_id=source_part_id,
        object_modes=object_modes,
        structural_sources={},
        translate_magnitudes={},
        baked_specs=baked_shared_specs,
    )
    body = clone_part_for_target(
        part_body,
        source_part_id,
        target_hand,
        new_part_id,
        mesh_map,
        row_transforms,
        prop_globals,
        {},
        prop_node_positions,
        node_mirror_map,
        inherited_options,
        shared_bake,
        context.mesh_pivots,
    )
    return relocate_part_for_slot(
        body,
        SlotRelocation(
            source_slot=source_slot,
            target_slot=target_slot,
            node_offset=_relocation_node_offset(
                context, config_name, source_slot, target_slot
            ),
        ),
        rewrite_context["node_mirror_map"],
        rewrite_context["known_nodes"],
        rewrite_context["group_map"],
        rewrite_context["known_groups"],
        rewrite_context["slot_map"],
        rewrite_context["known_slots"],
    )


def _generated_clone_excluded(source_part_id: str, part_body: str) -> bool:
    tokens = [source_part_id.lower()]
    tokens.extend(slot_type.lower() for slot_type in transform_helpers.extract_part_slot_types(part_body))
    return any("seat" in token for token in tokens)


def _part_has_handed_light_slots(part_body: str) -> bool:
    has_beam_electric = re.search(
        r'"\$electric"\s*:\s*"(?:lowbeam|highbeam|lowhighbeam)',
        part_body,
    ) is not None
    has_light_pattern = re.search(
        r'"\$lightPattern"\s*:\s*"(?:LHD|RHD|US)"',
        part_body,
    ) is not None
    return has_beam_electric and has_light_pattern


def _part_needs_generated_clone(
    context: VehicleContext,
    source_part_id: str,
    part_body: str,
    object_modes: dict[str, str],
    node_mirror_map: dict[str, str],
) -> bool:
    part_meshes = transform_helpers.extract_part_mesh_names(part_body)
    return (
        any(mesh in object_modes for mesh in part_meshes)
        or part_has_transformable_internal_camera(part_body, node_mirror_map)
        or _part_has_handed_light_slots(part_body)
    )


def _generated_clone_plan(
    context: VehicleContext,
    selected: dict[str, object],
    target_hand: str,
    config_name: str,
    object_modes: dict[str, str],
    node_mirror_map: dict[str, str],
    authored_parts: set[str],
) -> dict[str, str]:
    """Generated source part -> output part id for one config.

    Seed the plan from parts with actual handed content: mesh transforms,
    internal cameras, or semantic rewrites such as ``$lightPattern``. Then add
    bridge ancestors from the resolved slot tree so selecting a generated root
    carries its handed child-slot namespace down to those leaves.
    """
    generated_parts: dict[str, str] = {}
    parts_with_child_slots: set[str] = set()
    part_children: dict[str, set[str]] = {}
    selected_part_ids = {str(part_id) for part_id in selected["parts"]}
    selected_instances = [
        instance
        for instance in selected.get("part_instances", ())
        if isinstance(instance, dict)
    ]
    part_by_instance = {
        str(instance.get("instance_id") or ""): str(instance.get("part_id") or "")
        for instance in selected_instances
        if instance.get("instance_id")
    }
    for instance in selected_instances:
        parent_id = str(instance.get("parent_instance_id") or "")
        parent_part = part_by_instance.get(parent_id)
        part_id = str(instance.get("part_id") or "")
        if parent_part and part_id:
            part_children.setdefault(parent_part, set()).add(part_id)

    pending_part_ids = set(selected_part_ids)
    inspected_part_ids: set[str] = set()
    while pending_part_ids:
        source_part_id = pending_part_ids.pop()
        if source_part_id in inspected_part_ids:
            continue
        inspected_part_ids.add(source_part_id)
        if source_part_id in authored_parts:
            continue
        found = part_body_for_context(context, source_part_id)
        if found is None:
            continue
        part_body, _filename = found
        if _generated_clone_excluded(source_part_id, part_body):
            continue
        slot_defs = extract_slot_defs(part_body)
        if slot_defs:
            parts_with_child_slots.add(source_part_id)
            for slot_def in slot_defs:
                default_part = slot_def.default_part
                if not default_part:
                    continue
                part_children.setdefault(source_part_id, set()).add(default_part)
                if default_part not in inspected_part_ids:
                    pending_part_ids.add(default_part)
        if _part_needs_generated_clone(
            context, source_part_id, part_body, object_modes, node_mirror_map
        ):
            generated_parts[source_part_id] = generated_variant_part_name(
                source_part_id, target_hand, config_name
            )

    changed = True
    while changed:
        changed = False
        for source_part_id in sorted(parts_with_child_slots):
            if source_part_id in generated_parts:
                continue
            if source_part_id == selected.get("main_part"):
                continue
            if any(
                child_part_id in generated_parts
                for child_part_id in part_children.get(source_part_id, ())
            ):
                generated_parts[source_part_id] = generated_variant_part_name(
                    source_part_id, target_hand, config_name
                )
                changed = True
    return generated_parts


def write_generated_jbeam_and_configs(
    context: VehicleContext,
    output_vehicle_dir: Path,
    conversion: dict[str, object],
    object_modes: dict[str, str],
    structural_sources: dict[str, str],
    node_mirror_map: dict[str, str],
    variant_targets: dict[str, str],
    translate_magnitudes: dict[str, float],
    translated_prop_meshes: set[str],
    translated_flexbody_meshes: set[str],
    mirrored_prop_meshes: set[str],
    mirror_position_prop_meshes: set[str],
    mirror_position_flexbody_meshes: set[str],
    structural_prop_meshes: set[str],
    baked_shared_specs: list[BakedMeshSpec],
    authored_groups: dict[str, dict[str, object]] | None = None,
    slot_pair_plans: dict[str, dict[str, object]] | None = None,
) -> list[str]:
    authored_groups = authored_groups or {}
    slot_pair_plans = slot_pair_plans or {}
    cloned_bodies: list[str] = []
    cloned_part_ids: set[str] = set()
    generated_configs: list[str] = []
    relocation_context = (
        _relocation_rewrite_context(context) if slot_pair_plans else None
    )

    for config_name, target_hand in sorted(variant_targets.items()):
        variant = context.variants[config_name]
        pc = load_pc(context.source_zip, variant.pc_path)
        authored_group = authored_groups.get(config_name)
        slot_plan = slot_pair_plans.get(config_name)
        # Parts an authored swap or a slot pair resolves are already correct
        # for the target hand; everything else in the trim still goes through
        # the generated mirroring below.
        authored_parts = authored_group_source_parts(authored_group)
        authored_parts |= authored_group_source_parts(slot_plan)

        selected = selected_parts_for_config(context, config_name)
        selected_node_positions = selected_node_positions_for_config(context, config_name)
        prop_node_positions = dict(context.node_positions)
        prop_node_positions.update(selected_node_positions)
        selected_by_slot = selected.get("selected_by_slot", {})
        part_slot_options = selected.get("part_slot_options", {})
        slot_updates: dict[str, str] = {}
        main_update: str | None = None
        suffix = suffix_for_hand(target_hand)
        selected_part_ids = {str(part_id) for part_id in selected["parts"]}
        generated_parts_for_source = _generated_clone_plan(
            context,
            selected,
            target_hand,
            config_name,
            object_modes,
            node_mirror_map,
            authored_parts,
        )

        clone_source_part_ids = selected_part_ids | set(generated_parts_for_source)
        for source_part_id in sorted(clone_source_part_ids):
            if str(source_part_id) in authored_parts:
                continue
            found = part_body_for_context(context, str(source_part_id))
            if found is None:
                continue
            part_body, _filename = found
            if _generated_clone_excluded(str(source_part_id), part_body):
                continue
            part_meshes = transform_helpers.extract_part_mesh_names(part_body)
            mesh_hits = sorted(mesh for mesh in part_meshes if mesh in object_modes)
            camera_hit = part_has_transformable_internal_camera(part_body, node_mirror_map)
            new_part_id = generated_parts_for_source.get(str(source_part_id))
            if not new_part_id:
                continue

            if str(source_part_id) == selected["main_part"]:
                main_update = new_part_id
            if str(source_part_id) in selected_part_ids:
                selected_slot_types = []
                if isinstance(selected_by_slot, dict):
                    selected_slot_types = [
                        str(slot_type)
                        for slot_type, part_id in selected_by_slot.items()
                        if slot_type != "main" and str(part_id) == str(source_part_id)
                    ]
                slot_types = selected_slot_types or transform_helpers.extract_part_slot_types(part_body)
                for slot_type in slot_types:
                    slot_updates[slot_type] = new_part_id
                    slot_updates[f"{slot_type}{suffix}"] = new_part_id

            if new_part_id in cloned_part_ids:
                continue
            cloned_part_ids.add(new_part_id)

            mesh_map = {}
            for mesh in mesh_hits:
                source_mesh = structural_sources.get(mesh, mesh)
                source_obj = context.objects.get(source_mesh)
                mesh_map[mesh] = f"{mesh}{suffix}" if source_obj is not None and source_obj.dae_path else mesh
            flexbody_row_transforms: dict[str, tuple[str, float]] = {}
            for mesh in mesh_hits:
                if object_modes.get(mesh) == MODE_TRANSLATE and mesh in translated_flexbody_meshes:
                    flexbody_row_transforms[mesh] = (
                        "translate",
                        signed_delta_for_target(target_hand, translate_magnitudes.get(mesh, 0.0)),
                    )
                elif object_modes.get(mesh) == MODE_MIRROR_POSITION and mesh in mirror_position_flexbody_meshes:
                    flexbody_row_transforms[mesh] = ("mirrorPosition", 0.0)
                elif object_modes.get(mesh) in {MODE_MIRROR, MODE_MIRROR_STRUCTURAL, MODE_REPLACE_SOURCE}:
                    # Structural rows must carry the mirror in the jbeam pos/rot
                    # like plain mirror rows: the engine drops the DAE node
                    # translation for flexbodies, so a side-swap baked into the
                    # copy's node matrix never reaches the screen.
                    flexbody_row_transforms[mesh] = ("mirror", 0.0)
            prop_row_transforms: dict[str, tuple[str, float]] = {}
            for mesh in mesh_hits:
                if object_modes.get(mesh) == MODE_TRANSLATE and mesh in translated_prop_meshes:
                    prop_row_transforms[mesh] = (
                        "translate",
                        signed_delta_for_target(target_hand, translate_magnitudes.get(mesh, 0.0)),
                    )
                elif object_modes.get(mesh) == MODE_MIRROR_POSITION and mesh in mirror_position_prop_meshes:
                    prop_row_transforms[mesh] = ("mirrorPosition", 0.0)
                elif object_modes.get(mesh) == MODE_MIRROR and mesh in mirrored_prop_meshes:
                    prop_row_transforms[mesh] = ("mirror", 0.0)
            # config_name: these positions are written into ONE trim's jbeam,
            # so they must be that trim's, not the cross-trim representative.
            prop_globals = {
                mesh: target_object_position(
                    context,
                    mesh,
                    signed_delta_for_target(target_hand, translate_magnitudes.get(mesh, 0.0)),
                    config_name,
                )
                for mesh in mesh_hits
                if mesh in translated_prop_meshes and object_modes.get(mesh) == MODE_TRANSLATE
            }
            prop_globals.update(
                {
                    mesh: mirrored_object_position(context, mesh, config_name)
                    for mesh in mesh_hits
                    if mesh in mirrored_prop_meshes and object_modes.get(mesh) == MODE_MIRROR
                }
            )
            prop_globals.update(
                {
                    mesh: mirrored_object_position(
                        context, structural_sources[mesh], config_name
                    )
                    for mesh in mesh_hits
                    if mesh in structural_prop_meshes
                    and object_modes.get(mesh) in {MODE_MIRROR_STRUCTURAL, MODE_REPLACE_SOURCE}
                    and mesh in structural_sources
                }
            )
            inherited_options = ()
            if isinstance(part_slot_options, dict):
                raw_options = part_slot_options.get(str(source_part_id), ())
                if isinstance(raw_options, (list, tuple)):
                    inherited_options = tuple(str(item) for item in raw_options if item)
            shared_bake = SharedBakeContext(
                context=context,
                config_name=config_name,
                target_hand=target_hand,
                source_part_id=str(source_part_id),
                object_modes=object_modes,
                structural_sources=structural_sources,
                translate_magnitudes=translate_magnitudes,
                baked_specs=baked_shared_specs,
            )
            cloned_bodies.append(
                clone_part_for_target(
                    part_body,
                    str(source_part_id),
                    target_hand,
                    new_part_id,
                    mesh_map,
                    flexbody_row_transforms,
                    prop_globals,
                    prop_row_transforms,
                    prop_node_positions,
                    node_mirror_map,
                    inherited_options,
                    shared_bake,
                    context.mesh_pivots,
                    generated_parts_for_source,
                )
            )

        for relocation in slot_pair_plan_relocations(slot_plan):
            source_part_id = str(relocation.get("partId") or "")
            target_slot = str(relocation.get("slotId") or "")
            new_part_id = relocated_part_name(source_part_id, target_hand, target_slot)
            slot_updates[target_slot] = new_part_id
            if new_part_id in cloned_part_ids or relocation_context is None:
                continue
            inherited_options = ()
            if isinstance(part_slot_options, dict):
                raw_options = part_slot_options.get(source_part_id, ())
                if isinstance(raw_options, (list, tuple)):
                    inherited_options = tuple(str(item) for item in raw_options if item)
            body = _relocation_clone_body(
                context,
                config_name,
                target_hand,
                relocation,
                new_part_id,
                object_modes,
                node_mirror_map,
                prop_node_positions,
                inherited_options,
                baked_shared_specs,
                relocation_context,
            )
            if body is not None:
                cloned_part_ids.add(new_part_id)
                cloned_bodies.append(body)

        # The authored swap moves whole slots around (bx_shifter_lhd ->
        # bx_shifter_rhd), so it lands first and the generated clones then
        # overwrite their own -- disjoint -- slots on top of the swapped tree.
        if authored_group is not None:
            apply_hand_authored_group(pc, authored_group)
        if slot_plan is not None:
            apply_hand_authored_group(pc, slot_plan)
        if main_update:
            pc["mainPartName"] = main_update
        parts = dict(pc.get("parts", {}))
        parts.update(slot_updates)
        pc["parts"] = parts
        apply_authored_group_suffixed_slot_updates(pc, authored_group, target_hand)
        apply_replace_source_slot_updates(context, conversion, selected, pc, target_hand)
        generated_configs.append(write_converted_config(
            context, output_vehicle_dir, variant, config_name, target_hand, pc
        ))

    if cloned_bodies:
        jbeam_dir = output_vehicle_dir / "jbeam"
        jbeam_dir.mkdir(parents=True, exist_ok=True)
        contents = textwrap.dedent(
            f"""\
            {{
            // Generated visual hand-drive conversion parts.
            // Source: {context.source_zip.name}
            {','.join(cloned_bodies)}
            }}
            """
        )
        write_text_file(jbeam_dir / "handdrive_visual_conversion.jbeam", contents, encoding="utf-8")
    return generated_configs


def write_original_plate_configs(
    context: VehicleContext,
    output_vehicle_dir: Path,
    conversion: dict[str, object],
    config_names: Iterable[str],
) -> list[str]:
    """Copy stock trims as new configs for the plates-only build path."""
    generated: list[str] = []
    output_vehicle_dir.mkdir(parents=True, exist_ok=True)
    for config_name in sorted(set(config_names)):
        variant = context.variants[config_name]
        pc = load_pc(context.source_zip, variant.pc_path)
        output_config = original_plate_output_name(config_name)
        write_text_file(output_vehicle_dir / f"{output_config}.pc", json.dumps(pc, indent=2), encoding="utf-8")

        info: dict[str, object] = {}
        if variant.info_path:
            try:
                info = load_info(context.source_zip, variant.info_path)
            except Exception:
                info = {}
        existing_name = generated_info_display_name(info, variant)
        plates_name = existing_name if existing_name.lower().endswith(" plates") else f"{existing_name} Plates"
        info["Configuration"] = plates_name
        info["Name"] = plates_name
        description = generated_info_description(info)
        info["Description"] = f"{description} - BeamXP plate configuration" if description else "BeamXP plate configuration"
        info["Config Type"] = "Custom"
        info["Source"] = conversion_source_name(context)
        write_text_file(output_vehicle_dir / f"info_{output_config}.json", json.dumps(info, indent=2), encoding="utf-8")
        write_stock_preview(context, output_vehicle_dir, config_name, output_config)
        generated.append(output_config)
    return generated


def variant_target_hand(
    context: VehicleContext,
    conversion: dict[str, object],
    config_name: str,
) -> str | None:
    variants = conversion.get("variants", {})
    settings = variants.get(config_name) if isinstance(variants, dict) else None
    if variant_build_mode(settings) not in {BUILD_CONVERTED, BUILD_BOTH}:
        return None
    return target_hand_for(effective_source_hand(context, conversion, config_name), ACTION_OPPOSITE)


def output_config_sources(
    context: VehicleContext,
    conversion: dict[str, object],
) -> dict[str, str]:
    plans, _skipped = selected_output_plans(context, conversion)
    return {str(plan["output"]): str(plan["source"]) for plan in plans}


def load_beamng_json_file(path: Path) -> dict[str, object]:
    return parse_beamng_json(path.read_text(encoding="utf-8", errors="replace"), label=str(path))


def prop_row_world_matrix(
    row: str,
    node_positions: dict[str, tuple[float, float, float]],
    pivot: tuple[float, float, float] | None,
    inherited_options: Iterable[str] = (),
    rotation_override: list[list[float]] | None = None,
    variables: dict[str, float] | None = None,
) -> list[list[float]] | None:
    """Affine map from DAE-world coordinates to vehicle space for a prop row
    at rest: W = T(anchor) * R * T(-pivot), per the engine model verified
    against in-game dumps.

    rotation_override supplies the resolved engine rest rotation (authored
    baseRotationGlobal or analytic engine model)."""
    anchor = prop_row_pivot_position(row, node_positions, pivot, inherited_options, variables)
    if anchor is None:
        return None
    rotation = rotation_override
    if rotation is None:
        rotation = prop_row_global_rotation_matrix(row, node_positions)
    matrix = matrix4_with_rotation_translation(rotation, anchor)
    t = pivot or (0.0, 0.0, 0.0)
    return multiply_matrix(matrix, translation_matrix((-t[0], -t[1], -t[2])))


def preview_node_names(obj: DaeObject) -> list[str]:
    node_names = [obj.id]
    for alias in (obj.name, obj.id.strip("_ ")):
        if alias and alias not in node_names:
            node_names.append(alias)
    return node_names


def extract_preview_dae(zip_path: Path, dae_path: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{Path(zip_path).stem}__{dae_path}")
    if not stem.lower().endswith(".dae"):
        stem += ".dae"
    target = cache_dir / stem
    try:
        if target.exists() and target.stat().st_mtime_ns >= Path(zip_path).stat().st_mtime_ns:
            return target
    except OSError:
        pass
    with zipfile.ZipFile(zip_path) as zf:
        data = zf.read(dae_path)
    target.write_bytes(data)
    return target


def output_vehicle_preview_payload(
    context: VehicleContext,
    conversion: dict[str, object],
    output_name: str,
    output_root: Path,
    generated_daes: Iterable[Path],
    run_dir: Path,
) -> dict[str, object]:
    """Blender preview payload for one generated output config.

    This follows the validated build artifacts instead of re-simulating the
    conversion directly from the source vehicle: it reads the selected output
    ``.pc``, resolves that part tree against source + generated JBeam, and
    instances the exact mesh names referenced by those final rows.
    """
    output_sources = output_config_sources(context, conversion)
    config_name = output_sources.get(output_name)
    if config_name is None:
        raise RuntimeError(f"Unknown generated output config {output_name!r}")

    output_vehicle_dir = output_root / context.vehicle_path
    pc_path = output_vehicle_dir / f"{output_name}.pc"
    if not pc_path.exists():
        raise RuntimeError(f"Generated output config does not exist: {pc_path}")

    output_jbeam_texts: dict[str, str] = {}
    if output_vehicle_dir.exists():
        for path in sorted(output_vehicle_dir.rglob("*.jbeam")):
            output_jbeam_texts[str(path.relative_to(output_vehicle_dir)).replace("\\", "/")] = path.read_text(
                encoding="utf-8",
                errors="replace",
            )

    combined_jbeam_texts = dict(context.jbeam_texts)
    combined_jbeam_texts.update({f"generated/{name}": text for name, text in output_jbeam_texts.items()})
    combined_part_index = dict(context.part_body_index)
    combined_part_index.update(build_part_body_index(output_jbeam_texts))

    output_objects: dict[str, DaeObject] = {}
    output_pivots: dict[str, tuple[float, float, float]] = {}
    for dae_path in sorted((Path(path) for path in generated_daes), key=lambda item: str(item).lower()):
        if not dae_path.exists():
            continue
        for alias, obj in list_dae_objects_for_path(dae_path).items():
            output_objects.setdefault(alias, obj)
            output_pivots.setdefault(alias, (obj.x, obj.y, obj.z))

    dae_index: dict[tuple[str, str, str], int] = {}
    dae_entries: list[dict[str, object]] = []
    instances: list[dict[str, object]] = []
    skipped: dict[str, str] = {}
    used_output_paths = {str(Path(path)) for path in generated_daes}
    cache_dir = context.project_dir / "blender_preview" / "dae_cache"

    def dae_ref(obj: DaeObject) -> int:
        if obj.dae_source_zip is None:
            path = Path(obj.dae_path)
            key = ("file", str(path), "")
            entry = {"path": str(path), "dae_path": str(path)}
        else:
            zip_path = obj.dae_source_zip or context.source_zip
            key = ("zip", str(zip_path), obj.dae_path)
            entry = {
                "zip": str(zip_path),
                "dae_path": obj.dae_path,
                "path": str(extract_preview_dae(zip_path, obj.dae_path, cache_dir)),
            }
        index = dae_index.get(key)
        if index is None:
            index = len(dae_entries)
            dae_index[key] = index
            dae_entries.append(entry)
        return index

    selected = resolve_selected_parts(
        load_beamng_json_file(pc_path),
        combined_jbeam_texts,
        vehicle_id=context.source_vehicle_id,
        part_body_index=combined_part_index,
    )
    selected_nodes = selected_node_positions_for_parts(
        selected,
        combined_jbeam_texts,
        combined_part_index,
    )
    node_positions = build_node_position_index(combined_jbeam_texts)
    node_positions.update(selected_nodes)
    output_populated_groups = node_groups_for_selection(
        selected,
        lambda part_id, section: (
            lambda found: transform_helpers.extract_named_array(found[0], section)
            if found is not None
            else None
        )(find_part_body(part_id, combined_jbeam_texts, combined_part_index)),
    )
    rotation_counts: dict[str, int] = {}

    for part_instance in selected_part_instances(selected):
        part_id = str(part_instance.get("part_id") or "")
        instance_id = str(part_instance.get("instance_id") or "")
        slot_path = str(part_instance.get("slot_path") or "/")
        found = find_part_body(part_id, combined_jbeam_texts, combined_part_index)
        if found is None:
            skipped.setdefault(instance_id or part_id, "part body not found")
            continue
        part_body, _filename = found
        opts = part_instance_options(part_instance)
        variables = part_instance_variable_scope(selected, part_instance)
        for kind, array_key in (("flex", "flexbodies"), ("prop", "props")):
            array_text = transform_helpers.extract_named_array(part_body, array_key)
            if not array_text:
                continue
            for raw_row in iter_active_top_level_rows(array_text):
                row = resolve_jbeam_row_strings(raw_row, variables)
                mesh = flexbody_row_mesh(row) if kind == "flex" else prop_row_mesh(row)
                if not mesh or mesh in ("SPOTLIGHT", "POINTLIGHT"):
                    continue
                obj = output_objects.get(mesh) or context.objects.get(mesh)
                if obj is None or not obj.dae_path:
                    skipped.setdefault(mesh, "no DAE geometry indexed")
                    continue
                if kind == "prop" and not prop_row_nodes_present(row, selected_nodes):
                    skipped.setdefault(mesh, "inactive row (prop nodes not in this config)")
                    continue
                if kind == "flex" and not flexbody_row_is_bound(row, output_populated_groups):
                    skipped.setdefault(mesh, "inactive row (node group empty in this config)")
                    continue
                rotation_source = None
                if kind == "flex":
                    world = flexbody_row_source_matrix(row, opts, variables)
                else:
                    pivot = output_pivots.get(mesh) or context.mesh_pivots.get(mesh)
                    rotation_override, rotation_source = prop_rest_rotation_override(row, node_positions)
                    rotation_counts[rotation_source] = rotation_counts.get(rotation_source, 0) + 1
                    world = prop_row_world_matrix(
                        row, node_positions, pivot, opts, rotation_override, variables
                    )
                if world is None:
                    skipped.setdefault(mesh, "placement unresolved (inactive row?)")
                    continue
                is_generated = obj.dae_source_zip is None and str(Path(obj.dae_path)) in used_output_paths
                instances.append(
                    {
                        "dae": dae_ref(obj),
                        "node": obj.id,
                        "node_names": preview_node_names(obj),
                        "mesh": mesh,
                        "part": part_id,
                        "part_instance": instance_id,
                        "slot_path": slot_path,
                        "kind": kind,
                        "mode": "output" if is_generated else MODE_SKIP,
                        "matrix": matrix4_flat(world),
                        "keep_node_translation": is_generated
                        or (kind == "flex" and flexbody_row_needs_node_translation(context, mesh)),
                        **({"rotation_source": rotation_source} if rotation_source else {}),
                    }
                )

    return {
        "preview_kind": "generated_output",
        "vehicle_id": context.vehicle_id,
        "config_name": config_name,
        "output_name": output_name,
        "target_hand": None
        if output_name == original_plate_output_name(config_name)
        else variant_target_hand(context, conversion, config_name),
        "output_root": str(output_root),
        "dae_files": dae_entries,
        "instances": instances,
        "skipped_meshes": skipped,
        "rotation_calibration": rotation_counts,
        "show_unchanged": True,
    }


def _changed_selected_slot_paths(
    source_selected: dict[str, object],
    target_selected: dict[str, object],
) -> set[str]:
    def selected_paths(selected_tree: dict[str, object]) -> dict[str, str]:
        selected_by_path = selected_tree.get("selected_by_path", {})
        if not isinstance(selected_by_path, dict):
            return {}
        return {
            str(path): str(part_id)
            for path, part_id in selected_by_path.items()
            if str(path) and str(part_id)
        }

    source_paths = selected_paths(source_selected)
    target_paths = selected_paths(target_selected)
    return {
        path
        for path, part_id in target_paths.items()
        if path != "/" and source_paths.get(path, "") != part_id
    }


def full_vehicle_preview_payload(
    context: VehicleContext,
    conversion: dict[str, object],
    config_name: str,
    run_dir: Path,
    extra_meshes: Iterable[str] = (),
) -> dict[str, object]:
    """Full-vehicle Blender preview of one config after conversion.

    Every flexbody/prop row of the resolved part tree gets a final world
    matrix computed with the same engine-verified functions the build uses;
    geometry is referenced from the ORIGINAL DAE files by node name, so the
    preview needs no build output and no generated meshes. Mirrored rows use
    negative-determinant matrices (fine for preview rendering).

    extra_meshes are object ids NOT used by this config that should still be
    included (the GUI passes selected-but-inactive parts so they can be shown
    temporarily); their instances carry \"extra\": True."""
    if config_name not in context.variants:
        raise RuntimeError(f"Unknown config {config_name!r}")
    target_hand = variant_target_hand(context, conversion, config_name)
    object_modes = active_part_modes(conversion)
    preview_pc, generated_plate_parts = plate_generator.preview_pc_with_plate_parts(
        context,
        conversion,
        config_name,
    )
    slot_plan = slot_pair_plans_for_variants(context, conversion, [config_name]).get(config_name)
    source_preview_pc = copy.deepcopy(preview_pc)
    plan_preview_pc = copy.deepcopy(preview_pc)
    if slot_plan is not None:
        apply_hand_authored_group(plan_preview_pc, slot_plan)
    if slot_plan is not None:
        apply_hand_authored_group(preview_pc, slot_plan)
    preview_part_index = dict(context.part_body_index)
    preview_part_index.update(
        {
            part_id: (body, "bhdc_preview_licenseplates.jbeam")
            for part_id, body in generated_plate_parts.items()
        }
    )
    selected = resolve_selected_parts(
        preview_pc,
        context.jbeam_texts,
        vehicle_id=context.source_vehicle_id,
        part_body_index=preview_part_index,
    )
    source_selected = resolve_selected_parts(
        source_preview_pc,
        context.jbeam_texts,
        vehicle_id=context.source_vehicle_id,
        part_body_index=preview_part_index,
    )
    plan_selected = resolve_selected_parts(
        plan_preview_pc,
        context.jbeam_texts,
        vehicle_id=context.source_vehicle_id,
        part_body_index=preview_part_index,
    )
    selected_nodes = selected_node_positions_for_parts(
        selected,
        context.jbeam_texts,
        preview_part_index,
    )
    source_selected_nodes = selected_node_positions_for_parts(
        source_selected,
        context.jbeam_texts,
        preview_part_index,
    )
    node_positions = dict(context.node_positions)
    node_positions.update(selected_nodes)
    source_node_positions = dict(context.node_positions)
    source_node_positions.update(source_selected_nodes)
    mirror = mirror_x_matrix4()
    convertible = {MODE_TRANSLATE, MODE_MIRROR_POSITION, MODE_MIRROR, MODE_REPLACE_SOURCE}
    source_meshes = structural_mirror_sources(context, conversion, object_modes)
    rotation_counts: dict[str, int] = {}

    dae_index: dict[tuple[str, str], int] = {}
    dae_entries: list[dict[str, object]] = []
    instances: list[dict[str, object]] = []
    skipped: dict[str, str] = {}

    def dae_ref(obj: DaeObject) -> int:
        zip_path = obj.dae_source_zip or context.source_zip
        key = (str(zip_path), obj.dae_path)
        index = dae_index.get(key)
        if index is None:
            index = len(dae_entries)
            dae_index[key] = index
            dae_entries.append({"zip": str(zip_path), "dae_path": obj.dae_path})
        return index

    def final_matrix(mesh: str, mode: str, world: list[list[float]]) -> list[list[float]]:
        if target_hand is None or mode not in convertible:
            return world
        if mode == MODE_TRANSLATE:
            delta = signed_delta_for_target(
                target_hand,
                part_translate_magnitude(context, conversion, mesh),
            )
            return multiply_matrix(translation_matrix((delta, 0.0, 0.0)), world)
        if mode == MODE_MIRROR_POSITION:
            return multiply_matrix(translation_matrix((-2.0 * world[0][3], 0.0, 0.0)), world)
        if mode == MODE_REPLACE_SOURCE:
            return world
        return multiply_matrix(mirror, world)

    def preview_part_array(part_id: str, array_key: str) -> str | None:
        body = generated_plate_parts.get(part_id)
        if body is not None:
            return transform_helpers.extract_named_array(body, array_key)
        return part_named_array_for_context(context, part_id, array_key)

    # A flexbody bound only to groups this trim leaves empty is ignored by the
    # engine, so it must not appear in the preview scene either -- the parts
    # table reads Active straight off the built scene.
    populated_groups = node_groups_for_selection(selected, preview_part_array)
    source_populated_groups = node_groups_for_selection(source_selected, preview_part_array)
    equivalent_changed_paths = _changed_selected_slot_paths(source_selected, plan_selected)

    def equivalent_path_changed(slot_path: str) -> bool:
        if not slot_path:
            return False
        return any(
            slot_path == path or slot_path.startswith(path)
            for path in equivalent_changed_paths
        )

    def part_meshes(part_id: str) -> set[str]:
        found = part_body_for_context(context, part_id)
        if found is None:
            return set()
        return set(transform_helpers.extract_part_mesh_names(found[0]))

    def replacement_root_part(slot_id: str, source_mesh: str) -> str:
        slot_def = SlotDef(slot_id, "", allow_types=(slot_id,))
        for candidate_part, (candidate_body, _filename) in context.part_body_index.items():
            if source_mesh not in transform_helpers.extract_part_mesh_names(candidate_body):
                continue
            if part_fits_slot(transform_helpers.extract_part_slot_types(candidate_body), slot_def):
                return candidate_part
        return ""

    hand_token_re = re.compile(
        r"(?<![a-z0-9])(?:rhd|lhd|right[-_ ]*hand(?:[-_ ]*drive)?|"
        r"left[-_ ]*hand(?:[-_ ]*drive)?|jdm|ukdm|uk)(?![a-z0-9])",
        re.IGNORECASE,
    )

    def handless_identity(value: str) -> str:
        stripped = hand_token_re.sub(" ", value.lower())
        return re.sub(r"[^a-z0-9]+", "", stripped)

    def part_info_name(part_body: str) -> str:
        info = transform_helpers.extract_keyed_object(part_body, "information")
        if not info:
            return ""
        match = re.search(r'"name"\s*:\s*"((?:[^"\\]|\\.)*)"', info)
        return match.group(1) if match is not None else ""

    def part_hand_neutral(part_id: str, part_body: str) -> bool:
        label = f"{part_id} {part_info_name(part_body)}"
        return handless_identity(label) == re.sub(r"[^a-z0-9]+", "", label.lower())

    def paired_target_slot(source_slot: SlotDef, target_slots: list[SlotDef]) -> SlotDef | None:
        ranked: list[tuple[tuple[int, int, int], SlotDef]] = []
        source_identity = handless_identity(source_slot.slot_type)
        source_default = handless_identity(source_slot.default_part)
        for target_slot in target_slots:
            exact = source_slot.slot_type == target_slot.slot_type
            identity_match = bool(source_identity and source_identity == handless_identity(target_slot.slot_type))
            default_match = bool(source_default and source_default == handless_identity(target_slot.default_part))
            if exact or identity_match or default_match:
                ranked.append(((int(exact), int(identity_match), int(default_match)), target_slot))
        if not ranked:
            return None
        ranked.sort(key=lambda item: tuple(-score for score in item[0]) + (item[1].slot_type,))
        best = ranked[0][0]
        return ranked[0][1] if sum(score == best for score, _slot in ranked) == 1 else None

    def selected_child_part(slot_type: str, slot_path: str) -> str:
        selected_by_path = selected.get("selected_by_path", {})
        if isinstance(selected_by_path, dict) and selected_by_path.get(slot_path):
            return str(selected_by_path[slot_path])
        selected_by_slot = selected.get("selected_by_slot", {})
        if isinstance(selected_by_slot, dict) and selected_by_slot.get(slot_type):
            return str(selected_by_slot[slot_type])
        return ""

    def child_counterpart_part(source_part: str, target_slot: SlotDef) -> str:
        source_found = part_body_for_context(context, source_part)
        if source_found is None:
            return ""
        source_body = source_found[0]
        source_part_identity = handless_identity(source_part)
        source_name_identity = handless_identity(part_info_name(source_body))
        ranked: list[tuple[tuple[int, int], str]] = []
        for candidate_part, (candidate_body, _filename) in context.part_body_index.items():
            if candidate_part == source_part:
                continue
            if not part_fits_slot(transform_helpers.extract_part_slot_types(candidate_body), target_slot):
                continue
            part_match = bool(source_part_identity and source_part_identity == handless_identity(candidate_part))
            name_match = bool(source_name_identity and source_name_identity == handless_identity(part_info_name(candidate_body)))
            if part_match or name_match:
                ranked.append(((int(part_match), int(name_match)), candidate_part))
        if not ranked:
            return ""
        ranked.sort(key=lambda item: tuple(-score for score in item[0]) + (item[1],))
        best = ranked[0][0]
        return ranked[0][1] if sum(score == best for score, _part in ranked) == 1 else ""

    def mesh_map_for_parts(source_part: str, target_part: str) -> dict[str, str]:
        source_found = part_body_for_context(context, source_part)
        target_found = part_body_for_context(context, target_part)
        if source_found is None or target_found is None:
            return {}
        source_meshes = sorted(transform_helpers.extract_part_mesh_names(source_found[0]))
        target_meshes = sorted(transform_helpers.extract_part_mesh_names(target_found[0]))
        fallback_source = source_meshes[0] if source_meshes else ""
        target_by_identity: dict[str, list[str]] = {}
        for target_mesh in target_meshes:
            target_by_identity.setdefault(handless_identity(target_mesh), []).append(target_mesh)
        out: dict[str, str] = {}
        for source_mesh in source_meshes:
            matches = target_by_identity.get(handless_identity(source_mesh), [])
            if len(matches) == 1:
                out[matches[0]] = source_mesh
        if fallback_source:
            for target_mesh in target_meshes:
                out.setdefault(target_mesh, fallback_source)
        return out

    def apply_replacement_child_choices(
        child_parts: dict[object, object],
        replacement_row_meshes: dict[str, str],
        source_part: str,
        target_part: str,
        source_path: str,
        target_path: str,
        visited: set[tuple[str, str, str, str]],
    ) -> None:
        visit = (source_part, target_part, source_path, target_path)
        if visit in visited:
            return
        visited.add(visit)
        source_found = part_body_for_context(context, source_part)
        target_found = part_body_for_context(context, target_part)
        if source_found is None or target_found is None:
            return
        replacement_row_meshes.update(mesh_map_for_parts(source_part, target_part))
        source_slots = extract_slot_defs(source_found[0])
        target_slots = extract_slot_defs(target_found[0])
        for source_slot in source_slots:
            source_child_path = f"{source_path}{source_slot.slot_type}/"
            source_choice = selected_child_part(source_slot.slot_type, source_child_path)
            child_parts.pop(source_slot.slot_type, None)
            child_parts.pop(source_child_path, None)
            if not source_choice:
                continue
            target_slot = paired_target_slot(source_slot, target_slots)
            if target_slot is None:
                continue
            target_child_path = f"{target_path}{target_slot.slot_type}/"
            target_choice = ""
            source_choice_found = part_body_for_context(context, source_choice)
            if source_choice_found is not None:
                source_choice_body = source_choice_found[0]
                if (
                    part_fits_slot(transform_helpers.extract_part_slot_types(source_choice_body), target_slot)
                    and part_hand_neutral(source_choice, source_choice_body)
                ):
                    target_choice = source_choice
            if not target_choice:
                target_choice = child_counterpart_part(source_choice, target_slot)
            if not target_choice:
                target_choice = target_slot.default_part
            if not target_choice:
                continue
            child_parts[target_slot.slot_type] = target_choice
            child_parts[target_child_path] = target_choice
            apply_replacement_child_choices(
                child_parts,
                replacement_row_meshes,
                source_choice,
                target_choice,
                source_child_path,
                target_child_path,
                visited,
            )

    def selected_instance_for_mesh(mesh: str) -> dict[str, object] | None:
        for instance in selected_part_instances(selected):
            part_id = str(instance.get("part_id") or "")
            if mesh in part_meshes(part_id):
                return dict(instance)
        return None

    child_skip_paths: set[str] = set()
    child_preview_groups: list[
        tuple[
            dict[str, object],
            dict[str, tuple[float, float, float]],
            dict[str, tuple[float, float, float]],
            set[str],
            str,
            str,
            dict[str, str],
        ]
    ] = []
    child_preview_roots: set[str] = set()
    parts_config = conversion.get("parts", {})
    if isinstance(parts_config, dict):
        for mesh, mode in object_modes.items():
            settings = parts_config.get(mesh)
            if mode != MODE_REPLACE_SOURCE or not isinstance(settings, dict) or not settings.get("includeChildren"):
                continue
            source_mesh = str(settings.get("mirrorSource") or "")
            root = selected_instance_for_mesh(mesh)
            if root is None:
                continue
            root_path = str(root.get("slot_path") or "")
            root_slot = str(root.get("slot_id") or "")
            target_part = replacement_root_part(root_slot, source_mesh)
            if not root_path or not root_slot or not target_part:
                continue
            if root_path in child_preview_roots:
                continue
            child_preview_roots.add(root_path)
            child_skip_paths.add(root_path)
            child_pc = copy.deepcopy(preview_pc)
            child_parts = child_pc.setdefault("parts", {})
            if not isinstance(child_parts, dict):
                child_parts = {}
                child_pc["parts"] = child_parts
            root_found = part_body_for_context(context, str(root.get("part_id") or ""))
            if root_found is not None:
                for slot_def in extract_slot_defs(root_found[0]):
                    child_parts.pop(slot_def.slot_type, None)
            for key in list(child_parts):
                if str(key) != root_path and str(key).startswith(root_path):
                    child_parts.pop(key, None)
            child_parts[root_slot] = target_part
            child_parts[root_path] = target_part
            replacement_row_meshes: dict[str, str] = {}
            apply_replacement_child_choices(
                child_parts,
                replacement_row_meshes,
                str(root.get("part_id") or ""),
                target_part,
                root_path,
                root_path,
                set(),
            )
            child_selected = resolve_selected_parts(
                child_pc,
                context.jbeam_texts,
                vehicle_id=context.source_vehicle_id,
                part_body_index=preview_part_index,
            )
            child_nodes = selected_node_positions_for_parts(
                child_selected,
                context.jbeam_texts,
                preview_part_index,
            )
            child_node_positions = dict(context.node_positions)
            child_node_positions.update(child_nodes)
            child_populated_groups = node_groups_for_selection(child_selected, preview_part_array)
            child_preview_groups.append(
                (
                    child_selected,
                    child_nodes,
                    child_node_positions,
                    child_populated_groups,
                    root_path,
                    mesh,
                    replacement_row_meshes,
                )
            )

    def append_selected_instances(
        selected_tree: dict[str, object],
        selected_nodes_for_tree: dict[str, tuple[float, float, float]],
        node_positions_for_tree: dict[str, tuple[float, float, float]],
        populated_groups_for_tree: set[str],
        *,
        skip_child_roots: set[str] | None = None,
        only_child_root: str = "",
        force_mode: str | None = None,
        row_meshes: dict[str, str] | None = None,
        row_parent_mesh: str = "",
    ) -> None:
        skip_child_roots = skip_child_roots or set()
        row_meshes = row_meshes or {}
        for part_instance in selected_part_instances(selected_tree):
            part_id = str(part_instance.get("part_id") or "")
            instance_id = str(part_instance.get("instance_id") or "")
            slot_path = str(part_instance.get("slot_path") or "/")
            if only_child_root:
                if not slot_path.startswith(only_child_root):
                    continue
            elif any(slot_path.startswith(root_path) for root_path in skip_child_roots):
                continue
            opts = part_instance_options(part_instance)
            variables = part_instance_variable_scope(selected_tree, part_instance)
            for kind, array_key in (("flex", "flexbodies"), ("prop", "props")):
                array_text = preview_part_array(part_id, array_key)
                if not array_text:
                    continue
                for raw_row in iter_active_top_level_rows(array_text):
                    row = resolve_jbeam_row_strings(raw_row, variables)
                    mesh = flexbody_row_mesh(row) if kind == "flex" else prop_row_mesh(row)
                    if not mesh or mesh in ("SPOTLIGHT", "POINTLIGHT"):
                        continue
                    mode = force_mode or object_modes.get(mesh, MODE_SKIP)
                    if force_mode is None and mode == MODE_SKIP and equivalent_path_changed(slot_path):
                        mode = "equivalent"
                    geometry_mesh = mesh if force_mode == MODE_REPLACE_SOURCE else source_meshes.get(mesh, mesh)
                    obj = context.objects.get(geometry_mesh)
                    if obj is None or not obj.dae_path:
                        skipped.setdefault(mesh, "no DAE geometry indexed")
                        continue
                    if kind == "prop" and not prop_row_nodes_present(row, selected_nodes_for_tree):
                        skipped.setdefault(mesh, "inactive row (prop nodes not in this config)")
                        continue
                    if kind == "flex" and not flexbody_row_is_bound(row, populated_groups_for_tree):
                        skipped.setdefault(mesh, "inactive row (node group empty in this config)")
                        continue
                    rotation_source = None
                    if kind == "flex":
                        world = flexbody_row_source_matrix(row, opts, variables)
                    else:
                        pivot = context.mesh_pivots.get(mesh)
                        rotation_override, rotation_source = prop_rest_rotation_override(row, node_positions_for_tree)
                        rotation_counts[rotation_source] = rotation_counts.get(rotation_source, 0) + 1
                        world = prop_row_world_matrix(
                            row, node_positions_for_tree, pivot, opts, rotation_override, variables
                        )
                    if world is None:
                        skipped.setdefault(mesh, "placement unresolved (inactive row?)")
                        continue
                    if math.hypot(world[0][3], world[1][3], world[2][3]) > PREVIEW_FAR_LIMIT:
                        skipped.setdefault(mesh, "placed far outside the vehicle (hidden by its jbeam)")
                        continue
                    instances.append(
                        {
                            "dae": dae_ref(obj),
                            "node": obj.id,
                            "node_names": preview_node_names(obj),
                            "mesh": mesh,
                            "instance_ref": mesh_instance_ref(mesh, slot_path),
                            **({"row_mesh": row_meshes[mesh]} if mesh in row_meshes else {}),
                            **({"row_parent_mesh": row_parent_mesh} if row_parent_mesh and mesh in row_meshes else {}),
                            "part": part_id,
                            "part_instance": instance_id,
                            "slot_path": slot_path,
                            "kind": kind,
                            "mode": mode if target_hand is not None else MODE_SKIP,
                            "matrix": matrix4_flat(final_matrix(mesh, mode, world)),
                            "stock_matrix": matrix4_flat(world),
                            "keep_node_translation": (
                                kind != "flex" or flexbody_row_needs_node_translation(context, geometry_mesh)
                            ),
                            **({"rotation_source": rotation_source} if rotation_source else {}),
                        }
                    )

    append_selected_instances(
        selected,
        selected_nodes,
        node_positions,
        populated_groups,
        skip_child_roots=child_skip_paths,
    )
    for child_selected, child_nodes, child_node_positions, child_populated_groups, root_path, root_mesh, row_meshes in child_preview_groups:
        append_selected_instances(
            child_selected,
            child_nodes,
            child_node_positions,
            child_populated_groups,
            only_child_root=root_path,
            force_mode=MODE_REPLACE_SOURCE,
            row_meshes=row_meshes,
            row_parent_mesh=root_mesh,
        )
    relocation_paths = {
        str(relocation.get("sourceSlotPath") or "")
        for relocation in slot_pair_plan_relocations(slot_plan)
        if isinstance(relocation, dict) and str(relocation.get("sourceSlotPath") or "")
    }
    for source_path in sorted(relocation_paths):
        append_selected_instances(
            source_selected,
            source_selected_nodes,
            source_node_positions,
            source_populated_groups,
            only_child_root=source_path,
            force_mode=MODE_MIRROR,
        )

    # Temporarily-shown parts that are NOT in this config's part tree: find
    # each mesh's flexbody/prop row in any part of the vehicle and place it
    # with the same helpers, falling back to its indexed position when the
    # row cannot be resolved outside its own config.
    extra_wanted = {str(mesh) for mesh in extra_meshes or ()}
    extra_wanted -= {str(inst["mesh"]) for inst in instances}
    extra_rows: dict[str, tuple[str, str, str]] = {}
    if extra_wanted:
        for extra_part_id in context.part_body_index:
            if len(extra_rows) == len(extra_wanted):
                break
            for kind, array_key in (("flex", "flexbodies"), ("prop", "props")):
                array_text = part_named_array_for_context(context, extra_part_id, array_key)
                if not array_text:
                    continue
                for row in iter_active_top_level_rows(array_text):
                    mesh = flexbody_row_mesh(row) if kind == "flex" else prop_row_mesh(row)
                    if mesh in extra_wanted and mesh not in extra_rows:
                        extra_rows[mesh] = (extra_part_id, kind, row)
    for mesh in sorted(extra_wanted):
        mode = object_modes.get(mesh, MODE_SKIP)
        geometry_mesh = source_meshes.get(mesh, mesh)
        obj = context.objects.get(geometry_mesh)
        if obj is None or not obj.dae_path:
            skipped.setdefault(mesh, "no DAE geometry indexed")
            continue
        extra_part_id, kind, row = extra_rows.get(mesh, ("", "flex", ""))
        world = None
        if row:
            if kind == "flex":
                world = flexbody_row_source_matrix(row)
            else:
                rotation_override, _source = prop_rest_rotation_override(row, node_positions)
                world = prop_row_world_matrix(
                    row, node_positions, context.mesh_pivots.get(mesh), (), rotation_override
                )
        if world is None:
            world = translation_matrix(context.mesh_pivots.get(mesh) or (obj.x, obj.y, obj.z))
        if math.hypot(world[0][3], world[1][3], world[2][3]) > PREVIEW_FAR_LIMIT:
            skipped.setdefault(mesh, "placed far outside the vehicle (hidden by its jbeam)")
            continue
        instances.append(
            {
                "dae": dae_ref(obj),
                "node": obj.id,
                "node_names": preview_node_names(obj),
                "mesh": mesh,
                "instance_ref": mesh,
                "part": extra_part_id,
                "kind": kind,
                "mode": mode if target_hand is not None else MODE_SKIP,
                "matrix": matrix4_flat(final_matrix(mesh, mode, world)),
                "stock_matrix": matrix4_flat(world),
                "keep_node_translation": (
                    kind != "flex" or flexbody_row_needs_node_translation(context, geometry_mesh)
                ),
                "extra": True,
            }
        )

    cache_dir = context.project_dir / "blender_preview" / "dae_cache"
    for entry in dae_entries:
        entry["path"] = str(extract_preview_dae(Path(str(entry["zip"])), str(entry["dae_path"]), cache_dir))

    output_name = config_name if target_hand is None else variant_output_name(config_name, target_hand)
    return {
        "vehicle_id": context.vehicle_id,
        "config_name": config_name,
        "output_name": output_name,
        "target_hand": target_hand,
        "dae_files": dae_entries,
        "instances": instances,
        "skipped_meshes": skipped,
        "rotation_calibration": rotation_counts,
    }

__all__ = ['generate_daes', 'variant_output_name', 'original_plate_output_name', 'append_hand_label', 'generated_info_display_name', 'generated_info_description', 'apply_hand_authored_group', 'relocated_part_name', 'write_converted_config', 'write_generated_jbeam_and_configs', 'write_original_plate_configs', 'variant_target_hand', 'output_config_sources', 'load_beamng_json_file', 'prop_row_world_matrix', 'preview_node_names', 'extract_preview_dae', 'output_vehicle_preview_payload', '_changed_selected_slot_paths', 'full_vehicle_preview_payload']
