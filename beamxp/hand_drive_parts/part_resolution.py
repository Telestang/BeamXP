"""Implementation slice extracted from ``beamng_hand_drive_core``.

Original source lines 3521-4068. Import the public orchestration module
``beamng_hand_drive_core`` rather than this implementation module directly.
"""

from __future__ import annotations

import re
from typing import Iterable
from beamxp import transform_helpers
from beamxp.core import mesh_resolution
from beamxp.core.beam_json import (
    add_missing_json_commas,
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

def find_part_body(
    part_id: str,
    jbeam_texts: dict[str, str],
    part_body_index: dict[str, tuple[str, str]] | None = None,
) -> tuple[str, str] | None:
    if part_body_index is not None:
        found = part_body_index.get(part_id)
        if found is not None:
            return found
    for name, text in jbeam_texts.items():
        body = transform_helpers.extract_keyed_object(text, part_id)
        if body is not None and '"slotType"' in body:
            return body, name
    return None


def part_body_for_context(context: VehicleContext, part_id: str) -> tuple[str, str] | None:
    return find_part_body(part_id, context.jbeam_texts, context.part_body_index)


def part_named_array_for_context(context: VehicleContext, part_id: str, array_key: str) -> str | None:
    cache_key = (part_id, array_key)
    if cache_key in context.part_array_cache:
        return context.part_array_cache[cache_key]
    found = part_body_for_context(context, part_id)
    if found is None:
        context.part_array_cache[cache_key] = None
        return None
    array_text = transform_helpers.extract_named_array(found[0], array_key)
    context.part_array_cache[cache_key] = array_text
    return array_text


def vehicle_namespace_main_part(
    vehicle_id: str,
    part_body_index: dict[str, tuple[str, str]] | None,
) -> str | None:
    """The part with slotType ``main`` declared in the vehicle's own namespace,
    mirroring jbeam ``io.getMainPartName``.

    BeamNG picks the root part by slot type, not by name: ``getMainPartName``
    returns ``partSlotMap[vehicleDir]['main'][1]``. The main part is usually
    named after the vehicle, but not always (mods especially), and many .pc
    files omit ``mainPartName`` entirely -- those must still find the right root
    instead of assuming a part literally named after the vehicle id exists.
    Scan is scoped to ``vehicles/<id>/`` so a common part never becomes the
    root, and sorted for a deterministic pick if a vehicle ships more than one.
    """
    if not part_body_index:
        return None
    prefix = f"vehicles/{vehicle_id}/"
    candidates = [
        part_id
        for part_id, (body, filename) in part_body_index.items()
        if filename.replace("\\", "/").startswith(prefix)
        and "main" in transform_helpers.extract_part_slot_types(body)
    ]
    return min(candidates) if candidates else None


def resolve_selected_parts(
    pc: dict[str, object],
    jbeam_texts: dict[str, str],
    *,
    vehicle_id: str,
    part_body_index: dict[str, tuple[str, str]] | None = None,
) -> dict[str, object]:
    explicit_parts = {
        str(slot_type): ("" if str(part_id) == "none" else str(part_id))
        for slot_type, part_id in dict(pc.get("parts", {})).items()
    }
    main_part = str(
        pc.get("mainPartName")
        or vehicle_namespace_main_part(vehicle_id, part_body_index)
        or vehicle_id
    )

    selected: set[str] = set()
    missing_parts: set[str] = set()
    parts_order: list[str] = []
    selected_by_slot: dict[str, str] = {"main": main_part}
    selected_by_path: dict[str, str] = {"/": main_part}
    part_slot_options: dict[str, tuple[str, ...]] = {main_part: tuple()}
    part_instances: list[dict[str, object]] = []
    cycles: list[dict[str, object]] = []
    instance_id_counts: dict[str, int] = {}

    queue: list[
        tuple[str, tuple[str, ...], str, str | None, str, tuple[str, ...]]
    ] = [(main_part, tuple(), "/", None, "main", tuple())]

    for slot_type, part_id in explicit_parts.items():
        if part_id and "/" not in slot_type:
            selected_by_slot[slot_type] = part_id

    def user_choice_for(slot_id: str, slot_path: str) -> str | None:
        if slot_id in explicit_parts:
            return explicit_parts[slot_id]
        if slot_path in explicit_parts:
            return explicit_parts[slot_path]
        return None

    while queue:
        part_id, inherited_options, part_path, parent_instance_id, slot_id, ancestry = queue.pop(0)
        if not part_id:
            continue
        if part_id in ancestry:
            cycles.append({"part_id": part_id, "slot_path": part_path, "ancestry": ancestry})
            continue

        base_instance_id = f"{part_path}{part_id}"
        count = instance_id_counts.get(base_instance_id, 0)
        instance_id_counts[base_instance_id] = count + 1
        instance_id = base_instance_id if count == 0 else f"{base_instance_id}#{count + 1}"

        found = find_part_body(part_id, jbeam_texts, part_body_index)
        source_file = found[1] if found is not None else None
        part_instances.append(
            {
                "instance_id": instance_id,
                "part_id": part_id,
                "slot_id": slot_id,
                "slot_path": part_path,
                "parent_instance_id": parent_instance_id,
                "inherited_options": inherited_options,
                "source_file": source_file,
            }
        )

        selected.add(part_id)
        if part_id not in parts_order:
            parts_order.append(part_id)
        part_slot_options.setdefault(part_id, inherited_options)

        if found is None:
            missing_parts.add(part_id)
            continue
        part_body, _filename = found
        next_ancestry = ancestry + (part_id,)

        for slot_def in extract_slot_defs(part_body):
            slot_path = part_path + slot_def.slot_type + "/"
            user_choice = user_choice_for(slot_def.slot_type, slot_path)
            if user_choice is not None:
                if user_choice == "":
                    continue
                chosen = user_choice
                picked = find_part_body(chosen, jbeam_texts, part_body_index)
                if picked is None or not part_fits_slot(
                    transform_helpers.extract_part_slot_types(picked[0]), slot_def
                ):
                    chosen = slot_def.default_part
            else:
                chosen = slot_def.default_part
            if not chosen:
                continue

            selected_by_slot[slot_def.slot_type] = chosen
            selected_by_path[slot_path] = chosen
            child_options = list(inherited_options)
            if slot_def.options:
                child_options.append(slot_def.options)
            child_options_tuple = tuple(child_options)
            part_slot_options.setdefault(chosen, child_options_tuple)
            queue.append(
                (
                    chosen,
                    child_options_tuple,
                    slot_path,
                    instance_id,
                    slot_def.slot_type,
                    next_ancestry,
                )
            )

    user_vars = pc.get("vars")
    user_variable_map = user_vars if isinstance(user_vars, dict) else {}
    part_variables = build_part_variable_scopes(
        parts_order, part_slot_options, jbeam_texts, part_body_index, user_variable_map
    )
    part_instance_variables = build_part_instance_variable_scopes(
        part_instances, jbeam_texts, part_body_index, user_variable_map
    )

    return {
        "main_part": main_part,
        "parts": selected,
        "parts_order": parts_order,
        "part_instances": part_instances,
        "selected_by_slot": selected_by_slot,
        "selected_by_path": selected_by_path,
        "part_slot_options": part_slot_options,
        "part_variables": part_variables,
        "part_instance_variables": part_instance_variables,
        "missing_parts": missing_parts,
        "cycles": cycles,
    }



def selected_parts_for_config(context: VehicleContext, config_name: str) -> dict[str, object]:
    cached = context.selected_parts_cache.get(config_name)
    if cached is not None:
        return cached
    variant = context.variants[config_name]
    pc = load_pc(context.source_zip, variant.pc_path)
    selected = resolve_selected_parts(
        pc,
        context.jbeam_texts,
        vehicle_id=context.vehicle_id,
        part_body_index=context.part_body_index,
    )
    context.selected_parts_cache[config_name] = selected
    return selected


def part_variable_scope(selected: dict[str, object], part_id: str) -> dict[str, float]:
    """The resolved variable values in force inside a selected part (empty when
    the part declares/inherits none), used to evaluate its $ expressions."""
    return mesh_resolution.part_variable_scope(selected, part_id)


_NODE_ROW_RE = re.compile(
    rf'^\s*\[\s*"(?P<id>(?:[^"\\]|\\.)*)"\s*,\s*'
    rf'(?P<x>{NUMBER_RE})\s*,\s*(?P<y>{NUMBER_RE})\s*,\s*(?P<z>{NUMBER_RE})'
)


# BEAMXP_PART_INSTANCE_FIX_V1: compatibility accessors for path-specific selected
# part occurrences.
def selected_part_instances(selected: dict[str, object]) -> list[dict[str, object]]:
    return mesh_resolution.selected_part_instances(selected)


def part_instance_options(instance: dict[str, object]) -> tuple[str, ...]:
    return mesh_resolution.part_instance_options(instance)


def part_instance_variable_scope(
    selected: dict[str, object],
    instance: dict[str, object],
) -> dict[str, object]:
    return mesh_resolution.part_instance_variable_scope(selected, instance)

def iter_node_rows(
    node_array: str,
    variables: dict[str, object] | None = None,
) -> Iterable[tuple[str, tuple[float, float, float], str]]:
    """Yield active node rows after resolving instance-specific expressions."""
    for raw_row in iter_active_top_level_rows(node_array):
        row = resolve_jbeam_row_strings(raw_row, variables)
        match = _NODE_ROW_RE.match(row)
        if match is None:
            continue
        node_id = match.group("id")
        if node_id in {"id", "type", "mesh", "func"}:
            continue
        position = (
            float(match.group("x")),
            float(match.group("y")),
            float(match.group("z")),
        )
        yield node_id, position, row



def selected_parts_in_merge_order(selected: dict[str, object]) -> list[str]:
    """Selected part ids in tree (parent-before-child) order, so a later part's
    node redefinition overrides an earlier one -- the order jbeam merges
    sections in. Falls back to a stable sort for older results (or caches) that
    predate parts_order."""
    return mesh_resolution.selected_parts_in_merge_order(selected)


def selected_node_positions_for_config(
    context: VehicleContext,
    config_name: str,
) -> dict[str, tuple[float, float, float]]:
    cached = context.selected_node_positions_cache.get(config_name)
    if cached is not None:
        return cached

    selected = selected_parts_for_config(context, config_name)
    nodes: dict[str, tuple[float, float, float]] = {}
    for instance in selected_part_instances(selected):
        part_id = str(instance.get("part_id") or "")
        node_array = part_named_array_for_context(context, part_id, "nodes")
        if not node_array:
            continue
        inherited_options = part_instance_options(instance)
        variables = part_instance_variable_scope(selected, instance)
        for node_id, position, row in iter_node_rows(node_array, variables):
            nodes[node_id] = pos_after_node_transforms(
                row, position, inherited_options, variables
            )

    context.selected_node_positions_cache[config_name] = nodes
    return nodes



def selected_node_positions_for_parts(
    selected: dict[str, object],
    jbeam_texts: dict[str, str],
    part_body_index: dict[str, tuple[str, str]] | None = None,
) -> dict[str, tuple[float, float, float]]:
    nodes: dict[str, tuple[float, float, float]] = {}
    for instance in selected_part_instances(selected):
        part_id = str(instance.get("part_id") or "")
        found = find_part_body(part_id, jbeam_texts, part_body_index)
        if found is None:
            continue
        part_body, _filename = found
        node_array = transform_helpers.extract_named_array(part_body, "nodes")
        if not node_array:
            continue
        inherited_options = part_instance_options(instance)
        variables = part_instance_variable_scope(selected, instance)
        for node_id, position, row in iter_node_rows(node_array, variables):
            nodes[node_id] = pos_after_node_transforms(
                row, position, inherited_options, variables
            )
    return nodes



def prop_row_mesh(row: str) -> str | None:
    strings = re.findall(r'"((?:[^"\\]|\\.)*)"', row)
    if len(strings) < 2:
        return None
    func, mesh = strings[:2]
    if func == "func" or mesh == "mesh":
        return None
    return mesh


def prop_row_nodes_present(row: str, node_positions: dict[str, tuple[float, float, float]]) -> bool:
    """Whether the row's idRef/idX/idY nodes all exist in node_positions.

    The engine only spawns a prop when its reference nodes exist in the
    assembled vehicle; pass the SELECTED parts' node positions to reproduce
    that (a global all-files node index would also resolve dormant rows,
    e.g. the manual handbrake mount in a sequential-shifter config)."""
    strings = re.findall(r'"((?:[^"\\]|\\.)*)"', row)
    if len(strings) < 5:
        return True
    return all(node_id in node_positions for node_id in strings[2:5])


def selected_prop_mesh_positions(
    context: VehicleContext,
    config_name: str,
    mesh_ids: set[str],
) -> dict[str, list[tuple[float, float, float]]]:
    selected = selected_parts_for_config(context, config_name)
    node_positions = selected_node_positions_for_config(context, config_name)
    positions: dict[str, list[tuple[float, float, float]]] = {}
    for instance in selected_part_instances(selected):
        part_id = str(instance.get("part_id") or "")
        props = part_named_array_for_context(context, part_id, "props")
        if not props:
            continue
        inherited_options = part_instance_options(instance)
        variables = part_instance_variable_scope(selected, instance)
        for raw_row in iter_active_top_level_rows(props):
            row = resolve_jbeam_row_strings(raw_row, variables)
            mesh = prop_row_mesh(row)
            if mesh not in mesh_ids:
                continue
            pivot = context.mesh_pivots.get(mesh)
            position = prop_row_pivot_position(
                row, node_positions, pivot, inherited_options, variables
            )
            if position is not None:
                positions.setdefault(mesh, []).append(position)
    return positions



def mesh_roles_for_config(
    context: VehicleContext,
    config_name: str,
) -> tuple[set[str], set[str], set[str]]:
    cached = context.mesh_roles_cache.get(config_name)
    if cached is not None:
        return cached

    flexbody_meshes: set[str] = set()
    prop_meshes: set[str] = set()
    all_meshes: set[str] = set()
    selected = selected_parts_for_config(context, config_name)
    for instance in selected_part_instances(selected):
        part_id = str(instance.get("part_id") or "")
        variables = part_instance_variable_scope(selected, instance)
        flexbodies = part_named_array_for_context(context, part_id, "flexbodies")
        if flexbodies:
            for raw_row in iter_active_top_level_rows(flexbodies):
                mesh = flexbody_row_mesh(resolve_jbeam_row_strings(raw_row, variables))
                if mesh:
                    flexbody_meshes.add(mesh)
                    all_meshes.add(mesh)
        props = part_named_array_for_context(context, part_id, "props")
        if props:
            for raw_row in iter_active_top_level_rows(props):
                mesh = prop_row_mesh(resolve_jbeam_row_strings(raw_row, variables))
                if mesh:
                    prop_meshes.add(mesh)
                    all_meshes.add(mesh)

    roles = (flexbody_meshes, prop_meshes, all_meshes)
    context.mesh_roles_cache[config_name] = roles
    return roles



def selected_mesh_roles(
    context: VehicleContext,
    selected_configs: list[str],
) -> tuple[set[str], set[str], set[str]]:
    flexbody_meshes: set[str] = set()
    prop_meshes: set[str] = set()
    all_meshes: set[str] = set()
    for config_name in selected_configs:
        config_flex, config_props, config_all = mesh_roles_for_config(context, config_name)
        flexbody_meshes.update(config_flex)
        prop_meshes.update(config_props)
        all_meshes.update(config_all)
    return flexbody_meshes, prop_meshes, all_meshes


def active_part_modes(conversion: dict[str, object]) -> dict[str, str]:
    parts = conversion.get("parts", {})
    modes: dict[str, str] = {}
    if not isinstance(parts, dict):
        return modes
    for object_id, settings in parts.items():
        if not isinstance(settings, dict):
            continue
        mode = str(settings.get("mode", MODE_SKIP))
        if mode in MODE_CHOICES and mode != MODE_SKIP:
            modes[str(object_id)] = mode
    return modes


def texture_flip_mesh_ids(
    context: VehicleContext,
    object_modes: dict[str, str],
) -> set[str]:
    """Aesthetic-mirror meshes carrying a beamNavigator screen island, whose
    texture must keep its left/right reading after the geometric mirror.

    Derived entirely from the vehicle's own data (nav_screen_mesh_scope reads
    the navigator controller and its glowMap) — there is no manual flag. A nav
    screen only reads backwards once its mesh is actually reflected, so this is
    gated on MODE_MIRROR: MODE_TRANSLATE keeps the texture as authored, and
    MODE_MIRROR_STRUCTURAL swaps in the opposite-side mesh, which already
    carries the correct mapping."""
    nav_scope = nav_screen_mesh_scope(context)
    if not nav_scope:
        return set()
    return {
        object_id
        for object_id, mode in object_modes.items()
        if mode == MODE_MIRROR and object_id in nav_scope
    }


def structural_mirror_source_for_settings(
    context: VehicleContext,
    object_id: str,
    settings: object,
) -> str | None:
    if not isinstance(settings, dict):
        return None
    source_id = str(settings.get("mirrorSource") or "")
    source_obj = context.objects.get(source_id)
    if source_id and source_id != object_id and source_obj is not None and source_obj.dae_path:
        return source_id
    return None


def structural_mirror_sources(
    context: VehicleContext,
    conversion: dict[str, object],
    object_modes: dict[str, str] | None = None,
) -> dict[str, str]:
    parts = conversion.get("parts", {})
    if not isinstance(parts, dict):
        return {}
    wanted = object_modes or active_part_modes(conversion)
    sources: dict[str, str] = {}
    for object_id, mode in wanted.items():
        if mode != MODE_MIRROR_STRUCTURAL:
            continue
        settings = parts.get(object_id, {})
        source_id = structural_mirror_source_for_settings(context, object_id, settings)
        if source_id is not None:
            sources[object_id] = source_id
    return sources


def fallback_structural_part_modes(
    context: VehicleContext,
    conversion: dict[str, object],
    object_modes: dict[str, str] | None = None,
    *,
    selected_configs: Iterable[str] = (),
) -> dict[str, str]:
    modes = dict(object_modes or active_part_modes(conversion))
    if not modes:
        return modes
    parts = conversion.get("parts", {})
    if not isinstance(parts, dict):
        return modes
    for object_id, mode in list(modes.items()):
        if mode != MODE_MIRROR_STRUCTURAL:
            continue
        source_id = structural_mirror_source_for_settings(context, object_id, parts.get(object_id, {}))
        if source_id is None:
            modes[object_id] = MODE_MIRROR
    return modes


def selected_steering_refs(conversion: dict[str, object]) -> list[str]:
    parts = conversion.get("parts", {})
    if not isinstance(parts, dict):
        return []
    return [
        str(object_id)
        for object_id, settings in parts.items()
        if isinstance(settings, dict) and settings.get("steeringRef")
    ]


def auto_delta_source_refs(context: VehicleContext, conversion: dict[str, object]) -> list[str]:
    """Steering-ref parts that actually contribute to the auto delta (indexed
    objects with a usable off-center X). Empty means the auto delta falls back
    to its default of 0."""
    return [
        object_id
        for object_id in selected_steering_refs(conversion)
        if object_id in context.objects and abs(context.objects[object_id].x) > 0.05
    ]


STEERING_PROP_STR_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')

__all__ = ['find_part_body', 'part_body_for_context', 'part_named_array_for_context', 'vehicle_namespace_main_part', 'resolve_selected_parts', 'selected_parts_for_config', 'part_variable_scope', '_NODE_ROW_RE', 'selected_part_instances', 'part_instance_options', 'part_instance_variable_scope', 'iter_node_rows', 'selected_parts_in_merge_order', 'selected_node_positions_for_config', 'selected_node_positions_for_parts', 'prop_row_mesh', 'prop_row_nodes_present', 'selected_prop_mesh_positions', 'mesh_roles_for_config', 'selected_mesh_roles', 'active_part_modes', 'texture_flip_mesh_ids', 'structural_mirror_source_for_settings', 'structural_mirror_sources', 'fallback_structural_part_modes', 'selected_steering_refs', 'auto_delta_source_refs', 'STEERING_PROP_STR_RE']
