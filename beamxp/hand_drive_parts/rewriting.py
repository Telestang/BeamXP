"""Implementation slice extracted from ``beamng_hand_drive_core``.

Original source lines 4256-4884. Import the public orchestration module
``beamng_hand_drive_core`` rather than this implementation module directly.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable
from beamxp import transform_helpers
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

def target_hand_for(
    source_hand: str,
    action: str,
) -> str | None:
    if action == ACTION_SKIP:
        return None
    if action == ACTION_OPPOSITE:
        if source_hand == HAND_LHD:
            return HAND_RHD
        if source_hand == HAND_RHD:
            return HAND_LHD
        return None
    if action == ACTION_TO_RHD:
        return None if source_hand == HAND_RHD else HAND_RHD
    if action == ACTION_TO_LHD:
        return None if source_hand == HAND_LHD else HAND_LHD
    return None


def suffix_for_hand(hand: str) -> str:
    return "_to_rhd" if hand == HAND_RHD else "_to_lhd"


def signed_delta_for_target(hand: str, magnitude: float) -> float:
    return -abs(magnitude) if hand == HAND_RHD else abs(magnitude)


def generated_mesh_name(source_mesh: str, target_hand: str) -> str:
    return f"{source_mesh}{suffix_for_hand(target_hand)}"


def generated_part_name(source_part: str, target_hand: str) -> str:
    return f"{source_part}{suffix_for_hand(target_hand)}"


def generated_variant_part_name(source_part: str, target_hand: str, config_name: str) -> str:
    return f"{generated_part_name(source_part, target_hand)}__{safe_id(config_name)}"


def generated_dae_output_path(
    output_root: Path,
    output_vehicle_dir: Path,
    context: VehicleContext,
    dae_path: str,
) -> Path:
    source_rel = zip_member_path(dae_path)
    vehicle_rel = zip_member_path(context.vehicle_path)
    try:
        local_rel = source_rel.relative_to(vehicle_rel)
        target_parent = output_vehicle_dir / local_rel.parent
        return target_parent / f"{local_rel.stem}_handdrive{local_rel.suffix}"
    except ValueError:
        pass

    flattened_source = source_rel.with_suffix("")
    if (
        len(source_rel.parts) >= 2
        and source_rel.parts[0].lower() == "vehicles"
        and source_rel.parts[1].lower() == "common"
    ):
        flattened_source = Path("common", *source_rel.parts[2:]).with_suffix("")
    target_stem = safe_id("_".join(flattened_source.parts))
    return output_vehicle_dir / f"{target_stem}_handdrive{source_rel.suffix}"


def source_object_position(
    context: VehicleContext,
    object_id: str,
    config_name: str | None = None,
) -> tuple[float, float, float]:
    """Where a mesh sits, in the given trim when one is known.

    The DaeObject coordinate is only a representative across trims (see
    representative_mesh_positions), so build callers pass their config: a mesh
    declared by mutually exclusive parts sits somewhere different in each, and
    writing the representative into one trim's jbeam would misplace it."""
    if config_name is not None:
        resolved = resolved_mesh_positions_for_config(context, config_name).get(object_id)
        if resolved is not None:
            return resolved.position
    obj = context.objects[object_id]
    return (obj.x, obj.y, obj.z)


def target_object_position(
    context: VehicleContext,
    object_id: str,
    signed_delta: float,
    config_name: str | None = None,
) -> tuple[float, float, float]:
    x, y, z = source_object_position(context, object_id, config_name)
    return (x + signed_delta, y, z)


def mirrored_object_position(
    context: VehicleContext,
    object_id: str,
    config_name: str | None = None,
) -> tuple[float, float, float]:
    x, y, z = source_object_position(context, object_id, config_name)
    return (-x, y, z)


def format_inline_vector(key: str, values: tuple[float, float, float]) -> str:
    x, y, z = values
    return (
        f'"{key}":{{"x":{transform_helpers.format_num(x)},'
        f'"y":{transform_helpers.format_num(y)},"z":{transform_helpers.format_num(z)}}}'
    )


def vector_pattern(key: str) -> re.Pattern[str]:
    return re.compile(
        rf'"{re.escape(key)}"\s*:\s*\{{\s*"x"\s*:\s*(?P<x>{NUMBER_RE})\s*,'
        rf'\s*"y"\s*:\s*(?P<y>{NUMBER_RE})\s*,\s*"z"\s*:\s*(?P<z>{NUMBER_RE})\s*\}}'
    )


def replace_inline_vector(row: str, key: str, values: tuple[float, float, float]) -> str:
    return vector_pattern(key).sub(format_inline_vector(key, values), row, count=1)


def insert_inline_vector_near_key(line: str, preferred_key: str, replacement: str) -> str | None:
    match = re.search(rf'"{re.escape(preferred_key)}"\s*:', line)
    if match is None:
        return None
    object_start = line.rfind("{", 0, match.start())
    if object_start < 0:
        return None
    try:
        object_end = transform_helpers.find_matching(line, object_start, "{", "}")
    except ValueError:
        return None
    object_close = object_end - 1
    return line[:object_close] + f",{replacement}" + line[object_close:]


def replace_or_append_inline_vector(
    line: str,
    key: str,
    values: tuple[float, float, float],
    *,
    preferred_key: str | None = None,
) -> str:
    existing = vector_pattern(key)
    if existing.search(line):
        return existing.sub(format_inline_vector(key, values), line, count=1)

    replacement = format_inline_vector(key, values)
    if preferred_key is not None:
        updated = insert_inline_vector_near_key(line, preferred_key, replacement)
        if updated is not None:
            return updated

    insert_at = line.rfind("]")
    if insert_at < 0:
        raise RuntimeError(f"Could not append {key} to prop row: {line}")
    return line[:insert_at] + f", {{{replacement}}}" + line[insert_at:]


def transform_flexbody_row(
    row: str,
    action: str,
    delta_x: float = 0.0,
    inherited_options: Iterable[str] = (),
) -> str:
    if action == "translate":
        pos = vector_from_row(row, "pos")
        if pos is None:
            return row
        source_pos = pos_after_node_transforms(row, pos, inherited_options)
        target_pos = (source_pos[0] + delta_x, source_pos[1], source_pos[2])
        return replace_inline_vector(row, "pos", pos_before_node_transforms(row, target_pos, inherited_options))

    if action == "mirror":
        out = row
        pos = vector_from_row(out, "pos")
        if pos is not None:
            source_pos = pos_after_node_transforms(out, pos, inherited_options)
            target_pos = (-source_pos[0], source_pos[1], source_pos[2])
        else:
            target_pos = None
        rot = vector_from_row(out, "rot")
        if rot is not None:
            target_rot = (rot[0], -rot[1], -rot[2])
            if target_pos is not None:
                jbeam_pos, jbeam_rot = pos_rot_before_node_transforms(
                    out,
                    target_pos,
                    target_rot,
                    inherited_options,
                )
                out = replace_inline_vector(out, "pos", jbeam_pos)
                out = replace_inline_vector(out, "rot", jbeam_rot)
            else:
                out = replace_inline_vector(out, "rot", target_rot)
        elif target_pos is not None:
            out = replace_inline_vector(out, "pos", pos_before_node_transforms(out, target_pos, inherited_options))
        return out

    return row


def flexbody_row_can_carry_transform(row: str, action: str) -> bool:
    if action == "translate":
        return vector_from_row(row, "pos") is not None
    if action == "mirror":
        return vector_from_row(row, "pos") is not None or vector_from_row(row, "rot") is not None
    return False


def rewrite_flexbody_meshes_with_transforms(
    array_text: str,
    mesh_map: dict[str, str],
    row_transforms: dict[str, tuple[str, float]],
    inherited_options: Iterable[str] = (),
    shared_bake: SharedBakeContext | None = None,
) -> str:
    spans: list[tuple[int, int, str]] = []
    idx = 1 if array_text.startswith("[") else 0
    while idx < len(array_text):
        if array_text[idx] == "[":
            end = transform_helpers.find_matching(array_text, idx, "[", "]")
            spans.append((idx, end, array_text[idx:end]))
            idx = end
            continue
        idx += 1

    if not spans:
        return rewrite_flexbody_meshes(array_text, mesh_map)

    out: list[str] = []
    cursor = 0
    for start, end, row in spans:
        out.append(array_text[cursor:start])
        mesh = flexbody_row_mesh(row)
        new_row = row
        baked_mesh = None
        row_transform: tuple[str, float] | None = row_transforms.get(mesh) if mesh else None
        bake_transform_into_dae = True
        if row_transform is not None:
            bake_transform_into_dae = not flexbody_row_can_carry_transform(row, row_transform[0])
        if mesh and mesh in mesh_map and shared_bake is not None:
            baked_mesh = add_baked_shared_mesh(
                shared_bake,
                mesh,
                flexbody_row_source_matrix(row, inherited_options),
                bake_transform_into_dae,
            )
        if row_transform is not None and (baked_mesh is None or not bake_transform_into_dae):
            action, delta_x = row_transform
            new_row = transform_flexbody_row(new_row, action, delta_x, inherited_options)
        if baked_mesh is not None:
            new_row = re.sub(
                rf'(\[\s*)"{re.escape(mesh)}"(?=\s*(?:,|\[|\{{))',
                rf'\1"{baked_mesh}"',
                new_row,
                count=1,
            )
        elif mesh in mesh_map:
            new_row = re.sub(
                rf'(\[\s*)"{re.escape(mesh)}"(?=\s*(?:,|\[|\{{))',
                rf'\1"{mesh_map[mesh]}"',
                new_row,
                count=1,
            )
        out.append(new_row)
        cursor = end
    out.append(array_text[cursor:])
    return "".join(out)


def replace_or_append_prop_translation_global(
    line: str,
    values: tuple[float, float, float],
) -> str:
    existing_translation_property = vector_pattern("baseTranslationGlobal")
    if existing_translation_property.search(line):
        return existing_translation_property.sub(format_inline_vector("baseTranslationGlobal", values), line, count=1)
    existing_translation_property = vector_pattern("baseTranslation")
    if existing_translation_property.search(line):
        return existing_translation_property.sub(format_inline_vector("baseTranslationGlobal", values), line, count=1)
    return replace_or_append_inline_vector(line, "baseTranslationGlobal", values)


def replace_or_append_prop_rotation_global(
    line: str,
    values: tuple[float, float, float],
) -> str:
    return replace_or_append_inline_vector(
        line,
        "baseRotationGlobal",
        values,
        preferred_key="baseTranslationGlobal",
    )


def rewrite_flexbody_meshes(array_text: str, mesh_map: dict[str, str]) -> str:
    return transform_helpers.rewrite_flexbody_meshes(array_text, mesh_map)


def rewrite_prop_meshes_with_globals(
    array_text: str,
    mesh_map: dict[str, str],
    prop_global_positions: dict[str, tuple[float, float, float]],
    prop_row_transforms: dict[str, tuple[str, float]],
    node_positions: dict[str, tuple[float, float, float]],
    inherited_options: Iterable[str] = (),
    shared_bake: SharedBakeContext | None = None,
    mesh_pivots: dict[str, tuple[float, float, float]] | None = None,
) -> str:
    out_lines: list[str] = []
    for line in array_text.splitlines(keepends=True):
        line_ending = ""
        content = line
        if content.endswith("\r\n"):
            content, line_ending = content[:-2], "\r\n"
        elif content.endswith("\n"):
            content, line_ending = content[:-1], "\n"

        matched_old_mesh: str | None = None
        baked_mesh: str | None = None
        for old_mesh, new_mesh in sorted(mesh_map.items(), key=lambda item: len(item[0]), reverse=True):
            pattern = rf'(\[\s*"((?:[^"\\]|\\.)*)"\s*(?:,\s*|\s+))"{re.escape(old_mesh)}"(?=\s*(?:,|"))'
            if re.search(pattern, content) is not None:
                matched_old_mesh = old_mesh
                if shared_bake is not None:
                    # Mirror bakes reflect across the frame's x-axis via
                    # D = R^T*S*R, so R must be the ENGINE's rest rotation:
                    # authored baseRotationGlobal, or the analytic engine
                    # model for rows without authored brg.
                    rest_rotation, _source = prop_rest_rotation_override(content, node_positions)
                    placement_matrix = prop_row_source_matrix(
                        content, node_positions, inherited_options, rest_rotation
                    )
                    if placement_matrix is not None:
                        baked_mesh = add_baked_shared_mesh(
                            shared_bake,
                            old_mesh,
                            placement_matrix,
                            False,
                            is_prop=True,
                        )
                replacement_mesh = baked_mesh or new_mesh
                content = re.sub(
                    pattern,
                    rf'\1"{replacement_mesh}"',
                    content,
                    count=1,
                )
                break

        row_position = None
        if matched_old_mesh in prop_row_transforms:
            pivot = (mesh_pivots or {}).get(matched_old_mesh)
            row_position = prop_row_pivot_position(content, node_positions, pivot, inherited_options)
        if matched_old_mesh in prop_row_transforms and row_position is not None:
            action, delta_x = prop_row_transforms[matched_old_mesh]
            if action == "translate":
                target_position = (row_position[0] + delta_x, row_position[1], row_position[2])
            elif action == "mirror":
                target_position = (-row_position[0], row_position[1], row_position[2])
            else:
                target_position = None
            if target_position is not None:
                jbeam_position = pos_before_node_transforms(content, target_position, inherited_options)
                content = replace_or_append_prop_translation_global(content, jbeam_position)
                if action == "mirror" and baked_mesh is None:
                    # Vehicle-local prop meshes carry the mirrored orientation via
                    # baseRotationGlobal. Baked shared copies instead have the frame-aligned
                    # reflection baked into the DAE (see baked_dae_matrix), so their rows
                    # must keep the original rotation fields untouched.
                    mirrored_rotation = mirrored_prop_global_rotation(content, node_positions)
                    if mirrored_rotation is not None:
                        _jbeam_position, jbeam_rotation = pos_rot_before_node_transforms(
                            content,
                            target_position,
                            mirrored_rotation,
                            inherited_options,
                        )
                        content = replace_or_append_prop_rotation_global(content, jbeam_rotation)
        elif matched_old_mesh in prop_row_transforms:
            pass
        elif matched_old_mesh in prop_global_positions:
            jbeam_position = pos_before_node_transforms(
                content,
                prop_global_positions[matched_old_mesh],
                inherited_options,
            )
            content = replace_or_append_prop_translation_global(
                content,
                jbeam_position,
            )
        out_lines.append(content + line_ending)
    return "".join(out_lines)


def swap_token_pair(value: str, left: str, right: str) -> str:
    left_marker = "\0LEFT_SIDE_TOKEN\0"
    right_marker = "\0RIGHT_SIDE_TOKEN\0"
    return value.replace(left, left_marker).replace(right, right_marker).replace(
        left_marker,
        right,
    ).replace(right_marker, left)


def mirror_lateral_node_id(value: str) -> str:
    token_pairs = (
        ("_FL", "_FR"),
        ("_FRONTLEFT", "_FRONTRIGHT"),
        ("_FrontLeft", "_FrontRight"),
        ("_frontLeft", "_frontRight"),
        ("_frontleft", "_frontright"),
        ("_RL", "_RR"),
        ("_REARLEFT", "_REARRIGHT"),
        ("_RearLeft", "_RearRight"),
        ("_rearLeft", "_rearRight"),
        ("_rearleft", "_rearright"),
        ("_LEFT", "_RIGHT"),
        ("_Left", "_Right"),
        ("_left", "_right"),
        ("_L", "_R"),
        ("_l", "_r"),
        ("-FL", "-FR"),
        ("-fl", "-fr"),
        ("-RL", "-RR"),
        ("-rl", "-rr"),
        ("-LEFT", "-RIGHT"),
        ("-Left", "-Right"),
        ("-left", "-right"),
        ("-L", "-R"),
        ("-l", "-r"),
        (".FL", ".FR"),
        (".fl", ".fr"),
        (".RL", ".RR"),
        (".rl", ".rr"),
        (".LEFT", ".RIGHT"),
        (".Left", ".Right"),
        (".left", ".right"),
        (".L", ".R"),
        (".l", ".r"),
    )
    for left, right in token_pairs:
        if left in value or right in value:
            return swap_token_pair(value, left, right)

    if value.endswith("ll"):
        return value[:-2] + "rr"
    if value.endswith("rr"):
        return value[:-2] + "ll"
    if value.endswith("l"):
        return value[:-1] + "r"
    if value.endswith("r"):
        return value[:-1] + "l"
    return value


def build_node_mirror_map(
    node_positions: dict[str, tuple[float, float, float]],
) -> dict[str, str]:
    mirror_map: dict[str, str] = {}
    items = list(node_positions.items())
    for node_id, (x, y, z) in items:
        if abs(x) < 1e-5:
            mirror_map[node_id] = node_id
            continue
        best: tuple[float, str] | None = None
        for candidate_id, (cx, cy, cz) in items:
            if candidate_id == node_id:
                continue
            same_side_penalty = 10.0 if x * cx > 0 and abs(x) > 0.02 and abs(cx) > 0.02 else 0.0
            score = same_side_penalty + abs(cx + x) * 4.0 + abs(cy - y) + abs(cz - z)
            if best is None or score < best[0]:
                best = (score, candidate_id)
        if best is not None and best[0] <= 0.18:
            mirror_map[node_id] = best[1]
    return mirror_map


def mirror_camera_reference(value: str, node_mirror_map: dict[str, str]) -> str:
    mapped = node_mirror_map.get(value)
    if mapped:
        return mapped
    return mirror_lateral_node_id(value)


def rewrite_internal_camera_line(
    content: str,
    node_mirror_map: dict[str, str],
) -> tuple[str, bool]:
    row_re = re.compile(
        rf'^(?P<prefix>\s*\[\s*"(?P<row_type>(?:[^"\\]|\\.)*)"\s*,\s*)'
        rf'(?P<x>{NUMBER_RE})'
        rf'(?P<rest>\s*,\s*{NUMBER_RE}\s*,\s*{NUMBER_RE}.*)$'
    )
    match = row_re.match(content)
    if match is None:
        return content, False

    x_value = -float(match.group("x"))
    if abs(x_value) < 1e-9:
        x_value = 0.0
    rest = match.group("rest")
    option_start = rest.find("{")
    if option_start >= 0:
        id_span = rest[:option_start]
        options = rest[option_start:]
    else:
        id_span = rest
        options = ""
    id_span = re.sub(
        r'"((?:[^"\\]|\\.)*)"',
        lambda item: f'"{mirror_camera_reference(item.group(1), node_mirror_map)}"',
        id_span,
    )
    return f"{match.group('prefix')}{transform_helpers.format_num(x_value)}{id_span}{options}", True


# The engine derives asymmetric first-person behavior (look-back direction,
# which side the head sticks out of the window) from the "driver"/"dash" row's
# rightHandCamera flag (lua/ge/extensions/core/cameraModes/driver.lua), so a
# mirror conversion must flip it alongside the x coordinate and node ids.
# Vanilla LHD/RHD pairs (bx, covet, miramar) differ by exactly these three edits.
CAMERA_HAND_FLAG_RE = re.compile(r'("rightHand(?:Camera|Door)"\s*:\s*)(true|false)')
# indent restricted to [ \t] so the match can't start on a masked comment line
# above the row and swallow its newline into the captured indent
CAMERA_DRIVER_ROW_RE = re.compile(r'^([ \t]*)\[[ \t]*"(?:dash|driver)"', re.MULTILINE)


def rewrite_internal_cameras(
    array_text: str,
    node_mirror_map: dict[str, str],
) -> str:
    out_lines: list[str] = []
    for line in array_text.splitlines(keepends=True):
        line_ending = ""
        content = line
        if content.endswith("\r\n"):
            content, line_ending = content[:-2], "\r\n"
        elif content.endswith("\n"):
            content, line_ending = content[:-1], "\n"
        rewritten, _changed = rewrite_internal_camera_line(content, node_mirror_map)
        out_lines.append(rewritten + line_ending)
    out = "".join(out_lines)
    masked = transform_helpers.mask_comments_preserve_offsets(out)
    flag_matches = list(CAMERA_HAND_FLAG_RE.finditer(masked))
    if flag_matches:
        for match in reversed(flag_matches):
            flipped = "false" if match.group(2) == "true" else "true"
            out = out[: match.start(2)] + flipped + out[match.end(2) :]
    else:
        driver_row = CAMERA_DRIVER_ROW_RE.search(masked)
        if driver_row is not None:
            indent = driver_row.group(1)
            newline = "\r\n" if "\r\n" in out else "\n"
            out = (
                out[: driver_row.start()]
                + f'{indent}{{"rightHandCamera":true}},{newline}'
                + out[driver_row.start() :]
            )
    return out


def part_has_transformable_internal_camera(
    part_body: str,
    node_mirror_map: dict[str, str],
) -> bool:
    cameras = transform_helpers.extract_named_array(part_body, "camerasInternal")
    if not cameras:
        return False
    return rewrite_internal_cameras(cameras, node_mirror_map) != cameras


def clone_part_for_target(
    part_body: str,
    source_part_id: str,
    target_hand: str,
    new_part_id: str | None,
    mesh_map: dict[str, str],
    flexbody_row_transforms: dict[str, tuple[str, float]],
    prop_global_positions: dict[str, tuple[float, float, float]],
    prop_row_transforms: dict[str, tuple[str, float]],
    node_positions: dict[str, tuple[float, float, float]],
    node_mirror_map: dict[str, str],
    inherited_options: Iterable[str] = (),
    shared_bake: SharedBakeContext | None = None,
    mesh_pivots: dict[str, tuple[float, float, float]] | None = None,
) -> str:
    new_part_id = new_part_id or generated_part_name(source_part_id, target_hand)
    out = transform_helpers.replace_first(part_body, f'"{source_part_id}"', f'"{new_part_id}"')
    out = transform_helpers.replace_array_region(
        out,
        "flexbodies",
        lambda text: rewrite_flexbody_meshes_with_transforms(
            text,
            mesh_map,
            flexbody_row_transforms,
            inherited_options,
            shared_bake,
        ),
    )
    out = transform_helpers.replace_array_region(
        out,
        "props",
        lambda text: rewrite_prop_meshes_with_globals(
            text,
            mesh_map,
            prop_global_positions,
            prop_row_transforms,
            node_positions,
            inherited_options,
            shared_bake,
            mesh_pivots,
        ),
    )
    out = transform_helpers.replace_array_region(
        out,
        "camerasInternal",
        lambda text: rewrite_internal_cameras(text, node_mirror_map),
    )
    out = re.sub(
        r'("name"\s*:\s*")([^"]*)(")',
        lambda match: f"{match.group(1)}{append_hand_label(match.group(2), target_hand)}{match.group(3)}",
        out,
        count=1,
    )
    return out

__all__ = ['target_hand_for', 'suffix_for_hand', 'signed_delta_for_target', 'generated_mesh_name', 'generated_part_name', 'generated_variant_part_name', 'generated_dae_output_path', 'source_object_position', 'target_object_position', 'mirrored_object_position', 'format_inline_vector', 'vector_pattern', 'replace_inline_vector', 'insert_inline_vector_near_key', 'replace_or_append_inline_vector', 'transform_flexbody_row', 'flexbody_row_can_carry_transform', 'rewrite_flexbody_meshes_with_transforms', 'replace_or_append_prop_translation_global', 'replace_or_append_prop_rotation_global', 'rewrite_flexbody_meshes', 'rewrite_prop_meshes_with_globals', 'swap_token_pair', 'mirror_lateral_node_id', 'build_node_mirror_map', 'mirror_camera_reference', 'rewrite_internal_camera_line', 'CAMERA_HAND_FLAG_RE', 'CAMERA_DRIVER_ROW_RE', 'rewrite_internal_cameras', 'part_has_transformable_internal_camera', 'clone_part_for_target']
