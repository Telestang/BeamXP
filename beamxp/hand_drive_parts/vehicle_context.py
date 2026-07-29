"""Implementation slice extracted from ``beamng_hand_drive_core``.

Original source lines 2576-3208. Import the public orchestration module
``beamng_hand_drive_core`` rather than this implementation module directly.
"""

from __future__ import annotations

import re
from pathlib import Path
from beamxp import transform_helpers
from beamxp.core import cache as context_cache
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
from beamxp.core.cache import (
    CONTEXT_CACHE_VERSION,
    HAND_DETECTION_CACHE_VERSION,
    clear_parts_cache,
    clear_variant_hands_cache,
    context_cache_fingerprint,
    context_cache_path,
    context_fingerprint_hash,
    load_cached_part_ids,
    load_cached_vehicle_context,
    parts_cache_path,
    save_cached_part_ids,
    save_vehicle_context_cache,
    selection_cache_key,
    variant_hands_cache_fingerprint,
    variant_hands_cache_path,
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

def hand_from_text(text: str) -> str:
    lowered = text.lower()
    rhd_tokens = ("rhd", "right hand drive", "right-hand drive", "right hand-drive", "jdm", "uk")
    lhd_tokens = ("lhd", "left hand drive", "left-hand drive", "left hand-drive")
    if any(token in lowered for token in rhd_tokens):
        return HAND_RHD
    if any(token in lowered for token in lhd_tokens):
        return HAND_LHD
    return HAND_UNKNOWN


def variant_hand_detection_signature(conversion: dict[str, object]) -> tuple[str, ...]:
    """Inputs from a conversion that can change stock-hand detection."""
    return tuple(sorted(selected_steering_refs(conversion)))


def variant_hands_cache_key(conversion: dict[str, object]) -> str:
    return context_cache.variant_hands_cache_key(variant_hand_detection_signature(conversion))


def _normalized_cached_variant_hands(
    context: VehicleContext,
    value: object,
) -> dict[str, str]:
    return context_cache.normalized_cached_variant_hands(context, value)


def load_cached_variant_hands(
    context: VehicleContext,
    conversion: dict[str, object],
) -> dict[str, str] | None:
    """Load detected stock handedness for this steering-reference selection.

    Results are kept in memory and persisted across sessions. The source/common
    zip fingerprint and detection version prevent stale model data being reused.
    """
    return context_cache.load_cached_variant_hands_by_key(
        context,
        variant_hands_cache_key(conversion),
    )


def save_cached_variant_hands(
    context: VehicleContext,
    conversion: dict[str, object],
    hands: dict[str, str],
    max_entries: int = 8,
) -> None:
    context_cache.save_cached_variant_hands_by_key(
        context,
        variant_hands_cache_key(conversion),
        hands,
        max_entries=max_entries,
    )


def load_vehicle_context(
    source_zip: Path,
    vehicle_id: str | None = None,
    *,
    use_cache: bool = True,
) -> VehicleContext:
    source_zip = Path(source_zip)
    vehicle_ids = vehicle_ids_in_zip(source_zip)
    if not vehicle_ids:
        raise RuntimeError(f"No BeamNG vehicles with DAE/PC/JBeam files found in {source_zip}")
    selected_vehicle_id = vehicle_id or vehicle_ids[0]
    if selected_vehicle_id not in vehicle_ids:
        raise RuntimeError(f"Vehicle {selected_vehicle_id!r} not found in {source_zip}")

    if use_cache:
        cached = load_cached_vehicle_context(source_zip, selected_vehicle_id)
        if cached is not None:
            cached.loaded_from_cache = True
            return cached

    # Top-level DAEs first (they hold the main body and keep priority on any
    # duplicate mesh id via objects.setdefault), then DAEs in subdirectories.
    # Upfit bodies live under e.g. vehicles/us_semi/tanker/tanker.dae; without
    # the subdir DAEs those meshes (tanker, cargobox, dump, flatbed, ...) have no
    # geometry and vanish from the preview.
    dae_paths = direct_vehicle_files(source_zip, selected_vehicle_id, ".dae")
    for path in list_vehicle_files(source_zip, selected_vehicle_id, ".dae"):
        if path not in dae_paths:
            dae_paths.append(path)
    if not dae_paths:
        raise RuntimeError(f"No DAE files found for vehicles/{selected_vehicle_id}")

    objects: dict[str, DaeObject] = {}
    preview_by_id: dict[str, dict[str, object]] = {}
    for dae_path in dae_paths:
        # Parse once and feed both helpers; each used to re-parse the file.
        tree = parse_dae(source_zip, dae_path)
        for object_id, obj in dae_objects_from_tree(
            tree, dae_path, dae_source_zip=source_zip
        ).items():
            objects.setdefault(object_id, obj)
        for object_id, preview in preview_data_from_tree(tree).items():
            preview_by_id.setdefault(object_id, preview)

    variants: dict[str, VariantInfo] = {}
    for pc_path in direct_vehicle_files(source_zip, selected_vehicle_id, ".pc"):
        config_name = Path(pc_path).stem
        info_path = info_path_for_config(source_zip, selected_vehicle_id, config_name)
        display_name = display_name_for(source_zip, info_path, config_name)
        variants[config_name] = VariantInfo(
            name=config_name,
            pc_path=pc_path,
            info_path=info_path,
            display_name=display_name,
        )

    jbeam_texts, part_body_index, node_positions = load_resolver_inputs(
        source_zip, selected_vehicle_id
    )
    common_objects, common_previews, common_daes = load_common_dae_objects(
        source_zip,
        referenced_mesh_names(part_body_index),
        objects,
    )
    objects.update(common_objects)
    preview_by_id.update(common_previews)
    mesh_pivots = {
        object_id: (obj.x, obj.y, obj.z)
        for object_id, obj in objects.items()
        if obj.dae_path
    }
    prop_objects, prop_previews = collect_prop_only_objects(
        jbeam_texts,
        node_positions,
        objects,
        part_body_index,
    )
    objects.update(prop_objects)
    preview_by_id.update(prop_previews)
    # Snapshot authored centres before apply_resolved_mesh_positions rewrites
    # preview_by_id below -- resolving a bare flexbody row's position needs
    # the ORIGINAL geometry centre, not the representative-shifted one.
    mesh_authored_centers = {
        object_id: tuple(preview["center"])
        for object_id, preview in preview_by_id.items()
        if isinstance(preview, dict) and "center" in preview
    }
    # Only the positioned-mesh set is wanted here; the placements themselves
    # span parts that cannot coexist, so positions come from the per-config
    # resolution below instead.
    _placements, positioned_flexbodies = collect_flexbody_mesh_placements(
        objects, part_body_index, mesh_pivots
    )
    project_dir = project_dir_for(source_zip, selected_vehicle_id)

    context = VehicleContext(
        source_zip=source_zip,
        vehicle_id=selected_vehicle_id,
        vehicle_path=vehicle_prefix(selected_vehicle_id),
        dae_paths=dae_paths + [path for path in common_daes if path not in dae_paths],
        variants=variants,
        objects=objects,
        preview_by_id=preview_by_id,
        jbeam_texts=jbeam_texts,
        node_positions=node_positions,
        project_dir=project_dir,
        part_body_index=part_body_index,
        jbeam_positioned_flexbodies=positioned_flexbodies,
        mesh_pivots=mesh_pivots,
        mesh_authored_centers=mesh_authored_centers,
    )
    # Resolving positions needs a finished context (variants, part index,
    # pivots), so it runs here rather than inline above.
    representative, variant_dependent = representative_mesh_positions(context)
    apply_resolved_mesh_positions(
        context.objects,
        context.preview_by_id,
        representative,
        context.mesh_pivots,
        context.jbeam_positioned_flexbodies,
    )
    context.variant_dependent_meshes = variant_dependent
    save_vehicle_context_cache(context)
    context.loaded_from_cache = False
    return context


def median_value(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def steering_ref_score(object_id: str, obj: DaeObject) -> int:
    lowered = f"{object_id} {obj.name}".lower()
    compact = re.sub(r"[^a-z0-9]+", "", lowered)
    if "steer" not in compact:
        return 0

    score = 5
    if "wheel" in compact or "swheel" in compact:
        score += 25
    if abs(obj.x) > 0.05:
        score += 10
    if any(token in lowered for token in STEERING_NAME_EXCLUDES):
        score -= 25
    return score


def is_default_steering_ref(object_id: str, obj: DaeObject) -> bool:
    # 15 = "steer" in the name + off-center placement with no excluded token;
    # vehicles like the etk800 name their wheels plain "steer"/"steer_01a"
    # without a "wheel" token, so demanding the wheel bonus finds nothing.
    return abs(obj.x) > 0.05 and steering_ref_score(object_id, obj) >= 15


def vehicle_prefix_rank(context: VehicleContext, object_id: str) -> int:
    """Vehicle-named meshes (etk800_steer) outrank shared-library wheels
    (steer_01a, ...): the prefixed mesh is the vehicle's own default fitment
    while the rest are optional customisation parts."""
    return 0 if object_id.lower().startswith(f"{context.vehicle_id.lower()}_") else 1


def keep_single_steering_ref(context: VehicleContext, parts: dict[str, object]) -> None:
    """The tool works with exactly ONE steering reference (the GUI enforces
    this on click); when several part settings are flagged, keep the
    best-scoring one (same ordering as likely_steering_ref_ids) and clear
    the rest in place."""
    refs = [
        object_id
        for object_id, settings in parts.items()
        if isinstance(settings, dict) and settings.get("steeringRef")
    ]
    if len(refs) <= 1:
        return

    def rank(object_id: str) -> tuple[int, int, int, float, str]:
        obj = context.objects.get(object_id)
        if obj is None:
            return (1, 0, 0, 0.0, object_id)
        return (
            0,
            -steering_ref_score(object_id, obj),
            vehicle_prefix_rank(context, object_id),
            -abs(obj.x),
            object_id,
        )

    best = min(refs, key=rank)
    for object_id in refs:
        if object_id != best:
            parts[object_id]["steeringRef"] = False


def ensure_default_steering_ref(context: VehicleContext, parts: dict[str, object]) -> None:
    """Re-run steering-ref auto-detection when no part carries the flag, so a
    save written without one (older tool versions, cleared by hand) recovers
    the default on load instead of silencing detection forever."""
    for settings in parts.values():
        if isinstance(settings, dict) and settings.get("steeringRef"):
            return
    for object_id, settings in parts.items():
        if not isinstance(settings, dict):
            continue
        obj = context.objects.get(object_id)
        if obj is not None and is_default_steering_ref(object_id, obj):
            settings["steeringRef"] = True
    keep_single_steering_ref(context, parts)


def likely_steering_ref_ids(
    context: VehicleContext,
    used_meshes: set[str] | None = None,
) -> list[str]:
    candidates = used_meshes if used_meshes is not None else set(context.objects)
    scored: list[tuple[int, int, float, str]] = []
    for object_id in candidates:
        obj = context.objects.get(object_id)
        if obj is None:
            continue
        if abs(obj.x) <= 0.05:
            continue
        score = steering_ref_score(object_id, obj)
        if score >= 15:
            scored.append((score, vehicle_prefix_rank(context, object_id), abs(obj.x), object_id))
    scored.sort(key=lambda item: (-item[0], item[1], -item[2], item[3]))
    return [object_id for _score, _prefix, _abs_x, object_id in scored]


def estimated_vehicle_center_x(
    context: VehicleContext,
    used_meshes: set[str],
    steering_ids: set[str],
) -> float:
    object_xs = [
        context.objects[object_id].x
        for object_id in used_meshes
        if object_id in context.objects and object_id not in steering_ids
    ]
    object_center = median_value(object_xs)
    if object_center is not None and len(object_xs) >= 8:
        return object_center

    node_center = median_value([position[0] for position in context.node_positions.values()])
    if node_center is not None:
        return node_center

    return object_center if object_center is not None else 0.0


def hand_from_steering_positions(
    context: VehicleContext,
    steering_ids: list[str],
    used_meshes: set[str] | None = None,
) -> str:
    existing_ids = [object_id for object_id in steering_ids if object_id in context.objects]
    if not existing_ids:
        return HAND_UNKNOWN
    mesh_scope = used_meshes if used_meshes is not None else set(context.objects)
    center_x = estimated_vehicle_center_x(context, mesh_scope, set(existing_ids))
    offsets = [
        context.objects[object_id].x - center_x
        for object_id in existing_ids
        if abs(context.objects[object_id].x - center_x) > 0.01
    ]
    if not offsets:
        return HAND_UNKNOWN
    left_count = sum(1 for offset in offsets if offset > 0)
    right_count = sum(1 for offset in offsets if offset < 0)
    if left_count and not right_count:
        return HAND_LHD
    if right_count and not left_count:
        return HAND_RHD
    average_offset = sum(offsets) / len(offsets)
    if average_offset > 0.05 and left_count > right_count:
        return HAND_LHD
    if average_offset < -0.05 and right_count > left_count:
        return HAND_RHD
    return HAND_UNKNOWN


def hand_from_offsets(offsets: list[float]) -> str:
    offsets = [offset for offset in offsets if abs(offset) > 0.01]
    if not offsets:
        return HAND_UNKNOWN
    left_count = sum(1 for offset in offsets if offset > 0)
    right_count = sum(1 for offset in offsets if offset < 0)
    if left_count and not right_count:
        return HAND_LHD
    if right_count and not left_count:
        return HAND_RHD
    average_offset = sum(offsets) / len(offsets)
    if average_offset > 0.05 and left_count > right_count:
        return HAND_LHD
    if average_offset < -0.05 and right_count > left_count:
        return HAND_RHD
    return HAND_UNKNOWN


def stock_steering_ref_score(object_id: str, obj: DaeObject) -> int:
    lowered = f"{object_id} {obj.name}".lower()
    compact = re.sub(r"[^a-z0-9]+", "", lowered)
    if "steer" not in compact:
        return 0

    score = 10
    if "wheel" in compact or "swheel" in compact:
        score += 25
    if "airbag" in compact:
        score += 10
    stock_excludes = tuple(token for token in STEERING_NAME_EXCLUDES if token != "airbag") + ("boot",)
    if any(token in lowered for token in stock_excludes):
        score -= 25
    return score


def likely_stock_steering_ref_ids(
    context: VehicleContext,
    used_meshes: set[str],
) -> list[str]:
    scored: list[tuple[int, str]] = []
    for object_id in used_meshes:
        obj = context.objects.get(object_id)
        if obj is None:
            continue
        score = stock_steering_ref_score(object_id, obj)
        if score >= 10:
            scored.append((score, object_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [object_id for _score, object_id in scored]


def flexbody_row_needs_node_translation(context: VehicleContext, mesh: str) -> bool:
    """Whether mesh's own DAE node translation is real placement (add it) or a
    leftover export artefact the game ignores (drop it).

    The tell is whether ANY flexbody row placing this mesh authors its own
    "pos": if it does, the mesh is a reusable template the row positions
    (the D-Series gooseneck hitch is one part's pos:{y:0.325} away from
    another's pos:{y:-0.03}), and the node's translation is a second, real
    contribution that must be added on top -- dropping it put the hitch 3.85 m
    from the bed. A BARE row (mesh + material groups, no pos/rot/scale at all,
    e.g. etk800's manual shifter knob and boot) means the DAE vertices are
    already authored at their final position; the node still carries a
    translation, but it is a Blender-export artefact the game does not apply,
    and adding it moves the mesh across the vehicle (the shifter rendered on
    the passenger side, mirrored from the steering wheel). This is the exact
    signal jbeam_positioned_flexbodies already tracks for the same reason on
    the build side (generate_daes), just reused here for preview/detection."""
    return mesh in context.jbeam_positioned_flexbodies


def flexbody_mesh_reference_point(
    context: VehicleContext,
    mesh: str,
    obj: DaeObject,
) -> tuple[float, float, float]:
    """The point a flexbody row's own matrix should be applied to.

    KEEP: the node's authored translation (mesh_pivots) -- real placement.
    DROP: that translation backed out of the authored geometry centre, i.e.
    where the mesh sits once the node's redundant offset is removed. Falls
    back to the node translation if no authored centre was captured (e.g. a
    mesh with no geometry), which is at worst today's behaviour."""
    if flexbody_row_needs_node_translation(context, mesh):
        return context.mesh_pivots.get(mesh, (obj.x, obj.y, obj.z))
    center = context.mesh_authored_centers.get(mesh)
    pivot = context.mesh_pivots.get(mesh)
    if center is None or pivot is None:
        return context.mesh_pivots.get(mesh, (obj.x, obj.y, obj.z))
    return (center[0] - pivot[0], center[1] - pivot[1], center[2] - pivot[2])


def selected_flexbody_mesh_placements(
    context: VehicleContext,
    config_name: str,
    mesh_ids: set[str],
) -> dict[str, list[MeshPlacement]]:
    """Flexbody placements for every selected part occurrence in one config."""
    selected = selected_parts_for_config(context, config_name)
    placements: dict[str, list[MeshPlacement]] = {}
    for instance in selected_part_instances(selected):
        part_id = str(instance.get("part_id") or "")
        flexbodies = part_named_array_for_context(context, part_id, "flexbodies")
        if not flexbodies:
            continue
        inherited_options = part_instance_options(instance)
        variables = part_instance_variable_scope(selected, instance)
        for raw_row in iter_active_top_level_rows(flexbodies):
            row = resolve_jbeam_row_strings(raw_row, variables)
            mesh = flexbody_row_mesh(row)
            if mesh not in mesh_ids:
                continue
            obj = context.objects.get(mesh)
            if obj is None:
                continue
            pivot = flexbody_mesh_reference_point(context, mesh, obj)
            matrix = flexbody_row_source_matrix(row, inherited_options, variables)
            placements.setdefault(mesh, []).append(
                MeshPlacement(
                    position=transform_helpers.transform_point(matrix, pivot),
                    matrix=matrix,
                )
            )
    return placements



def selected_flexbody_mesh_positions(
    context: VehicleContext,
    config_name: str,
    mesh_ids: set[str],
) -> dict[str, list[tuple[float, float, float]]]:
    return {
        mesh: [placement.position for placement in placements]
        for mesh, placements in selected_flexbody_mesh_placements(
            context, config_name, mesh_ids
        ).items()
    }


def resolved_mesh_positions_for_config(
    context: VehicleContext,
    config_name: str,
) -> dict[str, ResolvedMeshPosition]:
    """Where each mesh of one trim actually sits.

    This is the honest answer the averaged DaeObject position cannot give: a
    mesh declared by several mutually exclusive parts (the D-Series gooseneck
    hitch sits in five, at two different offsets) resolves here to the offset
    of the part THIS trim selects."""
    cached = context.resolved_positions_cache.get(config_name)
    if cached is not None:
        return cached

    used = used_meshes_for_config(context, config_name)
    flex = selected_flexbody_mesh_placements(context, config_name, used)
    props = selected_prop_mesh_positions(context, config_name, used)

    resolved: dict[str, ResolvedMeshPosition] = {}
    for mesh in used:
        # jbeam hides a part it does not want by parking it kilometres away
        # (astrah stows a spare licence plate at y=-4.5e6). Those rows render
        # nothing, so averaging them in would drag the mesh off the vehicle --
        # the preview payload discards them on the same threshold.
        flex_placements = [
            placement
            for placement in flex.get(mesh, [])
            if not is_far_placement(placement.position)
        ]
        points = [placement.position for placement in flex_placements]
        points.extend(
            position for position in props.get(mesh, []) if not is_far_placement(position)
        )
        if not points:
            continue
        resolved[mesh] = ResolvedMeshPosition(
            position=average_position(points),
            matrices=tuple(
                tuple(tuple(row) for row in placement.matrix)
                for placement in flex_placements
            ),
        )
    context.resolved_positions_cache[config_name] = resolved
    return resolved


def preview_entries_for_config(
    context: VehicleContext,
    config_name: str,
) -> dict[str, dict[str, object]]:
    """preview_by_id shifted from the representative onto one trim.

    context.preview_by_id is baked once with the representative placement, so
    for a variant-dependent mesh its box sits where that mesh lands on some
    OTHER trim (the D-Series gooseneck hitch box is 0.61 m out on the long
    bed). Shifting by the difference between the two resolved positions is
    exact whenever the trims differ by translation, which is the case that
    produces a visible offset."""
    resolved = resolved_mesh_positions_for_config(context, config_name)
    entries: dict[str, dict[str, object]] = dict(context.preview_by_id)
    for mesh, entry in resolved.items():
        preview = context.preview_by_id.get(mesh)
        obj = context.objects.get(mesh)
        if preview is None or obj is None:
            continue
        delta = (
            entry.position[0] - obj.x,
            entry.position[1] - obj.y,
            entry.position[2] - obj.z,
        )
        if max(abs(value) for value in delta) < 1e-9:
            continue
        entries[mesh] = translate_preview_points(preview, delta)
    return entries


def representative_mesh_positions(
    context: VehicleContext,
) -> tuple[dict[str, ResolvedMeshPosition], set[str]]:
    """One position per mesh for callers with no trim in hand, plus the set of
    meshes for which that position is only a representative.

    The representative is the position the mesh holds in the MOST trims, ties
    broken by the alphabetically-first trim so it never depends on dict order.
    A mesh placed identically everywhere -- the overwhelming majority, and
    every steering reference measured so far -- resolves to exactly that
    position, which is what the old whole-index average produced too.

    Deliberately NOT the authored DAE pivot: shared-library meshes such as
    grp_steerwheel_hub are authored at the origin and positioned entirely by
    their jbeam row, so pivots would report x=0 and collapse the conversion
    delta computed from the steering reference."""
    grouped: dict[str, dict[tuple[float, ...], list[str]]] = {}
    entries: dict[str, dict[tuple[float, ...], ResolvedMeshPosition]] = {}
    for config_name in sorted(context.variants):
        for mesh, entry in resolved_mesh_positions_for_config(context, config_name).items():
            key = tuple(round(value, 6) for value in entry.position)
            grouped.setdefault(mesh, {}).setdefault(key, []).append(config_name)
            entries.setdefault(mesh, {}).setdefault(key, entry)

    representative: dict[str, ResolvedMeshPosition] = {}
    variant_dependent: set[str] = set()
    for mesh, groups in grouped.items():
        if len(groups) > 1:
            variant_dependent.add(mesh)
        winner = min(groups, key=lambda key: (-len(groups[key]), min(groups[key])))
        representative[mesh] = entries[mesh][winner]
    return representative, variant_dependent


def stock_steering_positions_for_config(
    context: VehicleContext,
    config_name: str,
    steering_ids: list[str],
) -> list[tuple[float, float, float]]:
    wanted = {object_id for object_id in steering_ids if object_id in context.objects}
    if not wanted:
        return []
    positions_by_mesh = selected_prop_mesh_positions(context, config_name, wanted)
    flex_positions = selected_flexbody_mesh_positions(context, config_name, wanted - set(positions_by_mesh))
    positions: list[tuple[float, float, float]] = []
    for mesh in steering_ids:
        positions.extend(positions_by_mesh.get(mesh, ()))
        positions.extend(flex_positions.get(mesh, ()))
    return positions


def selected_vehicle_center_x(context: VehicleContext, config_name: str, used_meshes: set[str]) -> float:
    selected_nodes = selected_node_positions_for_config(context, config_name)
    node_center = median_value([position[0] for position in selected_nodes.values()])
    if node_center is not None:
        return node_center
    return estimated_vehicle_center_x(context, used_meshes, set())


def hand_from_stock_steering_for_variant(
    context: VehicleContext,
    config_name: str,
    steering_ids: list[str],
    used_meshes: set[str],
) -> str:
    positions = stock_steering_positions_for_config(context, config_name, steering_ids)
    if not positions:
        return HAND_UNKNOWN
    center_x = selected_vehicle_center_x(context, config_name, used_meshes)
    return hand_from_offsets([position[0] - center_x for position in positions])


def used_meshes_for_config(context: VehicleContext, config_name: str) -> set[str]:
    return set(mesh_roles_for_config(context, config_name)[2])

__all__ = ['hand_from_text', 'variant_hand_detection_signature', 'variant_hands_cache_key', '_normalized_cached_variant_hands', 'load_cached_variant_hands', 'save_cached_variant_hands', 'load_vehicle_context', 'median_value', 'steering_ref_score', 'is_default_steering_ref', 'vehicle_prefix_rank', 'keep_single_steering_ref', 'ensure_default_steering_ref', 'likely_steering_ref_ids', 'estimated_vehicle_center_x', 'hand_from_steering_positions', 'hand_from_offsets', 'stock_steering_ref_score', 'likely_stock_steering_ref_ids', 'flexbody_row_needs_node_translation', 'flexbody_mesh_reference_point', 'selected_flexbody_mesh_placements', 'selected_flexbody_mesh_positions', 'resolved_mesh_positions_for_config', 'preview_entries_for_config', 'representative_mesh_positions', 'stock_steering_positions_for_config', 'selected_vehicle_center_x', 'hand_from_stock_steering_for_variant', 'used_meshes_for_config']
