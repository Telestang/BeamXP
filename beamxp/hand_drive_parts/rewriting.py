"""Implementation slice extracted from ``beamng_hand_drive_core``.

Original source lines 4256-4884. Import the public orchestration module
``beamng_hand_drive_core`` rather than this implementation module directly.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from collections.abc import Iterable
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
    SlotRelocation,
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
    return "_xp_rhd" if hand == HAND_RHD else "_xp_lhd"


def signed_delta_for_target(hand: str, magnitude: float) -> float:
    return -abs(magnitude) if hand == HAND_RHD else abs(magnitude)


def generated_mesh_name(source_mesh: str, target_hand: str) -> str:
    return f"{source_mesh}{suffix_for_hand(target_hand)}"


def generated_part_name(source_part: str, target_hand: str) -> str:
    return f"{source_part}{suffix_for_hand(target_hand)}"


def generated_variant_part_name(source_part: str, target_hand: str, config_name: str) -> str:
    """Return the generated part shared by every trim using ``source_part``.

    ``config_name`` remains in the signature for compatibility with callers,
    but trim-specific IDs make BeamNG expose one near-identical picker entry
    per converted configuration.
    """
    return generated_part_name(source_part, target_hand)


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

    if action == "mirrorPosition":
        pos = vector_from_row(row, "pos")
        if pos is None:
            return row
        source_pos = pos_after_node_transforms(row, pos, inherited_options)
        target_pos = (-source_pos[0], source_pos[1], source_pos[2])
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
    if action == "mirrorPosition":
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
            elif action == "mirrorPosition":
                target_position = (-row_position[0], row_position[1], row_position[2])
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


# Arrays whose leading columns are node ids. Rewriting them by "any quoted
# value that is a known node id" is safe because the map only ever contains
# ids that really exist in this vehicle -- unlike a textual _L/_R guess, which
# would happily turn the material name bx_floor into bx_flool.
_NODE_REFERENCE_ARRAYS = (
    "beams",
    "triangles",
    "quads",
    "hydros",
    "ropes",
    "rails",
    "slidenodes",
    "torsionbars",
    "refNodes",
)

_QUOTED_STRING_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')


def build_lateral_name_map(names: Iterable[str]) -> dict[str, str]:
    """Pair up names that differ only by a left/right token.

    The pairing must be confirmed from both ends: ``mirror_lateral_node_id``
    falls back to swapping a trailing l/r, which turns the group ``bx_floor``
    into the non-existent ``bx_flool``. Requiring the swapped name to be a real
    name too keeps that guess from ever reaching the output.
    """
    known = set(names)
    return {
        name: swapped
        for name in known
        if (swapped := mirror_lateral_node_id(name)) != name and swapped in known
    }


def relocated_reference(value: str, mirror_map: dict[str, str], known: set[str]) -> str:
    """Mirror one node id or group name, or leave it alone.

    Falls back to the lateral name swap only when the swapped name is itself
    known, so an id with no twin stays put instead of becoming a dangling
    reference.
    """
    mapped = mirror_map.get(value)
    if mapped:
        return mapped
    swapped = mirror_lateral_node_id(value)
    if swapped != value and swapped in known:
        return swapped
    return value


def mirror_quoted_references(
    array_text: str,
    mirror_map: dict[str, str],
    known: set[str],
) -> str:
    return _QUOTED_STRING_RE.sub(
        lambda match: f'"{relocated_reference(match.group(1), mirror_map, known)}"',
        array_text,
    )


def mirror_node_rows(
    array_text: str,
    mirror_map: dict[str, str],
    known_nodes: set[str],
    group_map: dict[str, str],
    known_groups: set[str],
) -> str:
    """Rename node ids, negate posX, and swap group names in a nodes table."""
    out: list[str] = []
    cursor = 0
    idx = 1 if array_text.startswith("[") else 0
    while idx < len(array_text):
        if array_text[idx] != "[":
            idx += 1
            continue
        end = transform_helpers.find_matching(array_text, idx, "[", "]")
        row = array_text[idx:end]
        out.append(array_text[cursor:idx])
        out.append(_mirror_node_row(row, mirror_map, known_nodes, group_map, known_groups))
        cursor = end
        idx = end
    out.append(array_text[cursor:])
    # Group names also appear as bare {"group":...} directives between rows,
    # which carry to every following row and so must be swapped too.
    return _mirror_group_directives("".join(out), group_map, known_groups)


def _mirror_node_row(
    row: str,
    mirror_map: dict[str, str],
    known_nodes: set[str],
    group_map: dict[str, str],
    known_groups: set[str],
) -> str:
    values = split_top_level_values(row.lstrip("[").rstrip("]"))
    if len(values) < 4:
        return _mirror_group_directives(row, group_map, known_groups)
    name = values[0].strip()
    if not (name.startswith('"') and name.endswith('"')):
        return row
    node_id = name[1:-1]
    if node_id in {"id", "id:"}:
        return row

    new_id = relocated_reference(node_id, mirror_map, known_nodes)
    row = transform_helpers.replace_first(row, f'"{node_id}"', f'"{new_id}"')

    match = re.search(rf'"{re.escape(new_id)}"\s*,\s*({NUMBER_RE})', row)
    if match is not None:
        flipped = -float(match.group(1))
        if abs(flipped) < 1e-9:
            flipped = 0.0
        row = row[: match.start(1)] + transform_helpers.format_num(flipped) + row[match.end(1) :]
    return _mirror_group_directives(row, group_map, known_groups)


def _mirror_group_directives(
    text: str,
    group_map: dict[str, str],
    known_groups: set[str],
) -> str:
    def replace(match: re.Match[str]) -> str:
        body = match.group(2)
        swapped = _QUOTED_STRING_RE.sub(
            lambda inner: f'"{relocated_reference(inner.group(1), group_map, known_groups)}"',
            body,
        )
        return f"{match.group(1)}{swapped}"

    return re.sub(
        r'("group"\s*:\s*)((?:\[[^\]]*\])|(?:"(?:[^"\\]|\\.)*"))',
        replace,
        text,
    )


def mirror_flexbody_group_lists(
    array_text: str,
    group_map: dict[str, str],
    known_groups: set[str],
) -> str:
    """Swap the [group]: column of every flexbody row to the other side."""
    out: list[str] = []
    cursor = 0
    idx = 1 if array_text.startswith("[") else 0
    while idx < len(array_text):
        if array_text[idx] != "[":
            idx += 1
            continue
        end = transform_helpers.find_matching(array_text, idx, "[", "]")
        row = array_text[idx:end]
        out.append(array_text[cursor:idx])
        group_start = row.find("[", 1)
        if group_start == -1:
            out.append(row)
        else:
            try:
                group_end = transform_helpers.find_matching(row, group_start, "[", "]")
            except ValueError:
                out.append(row)
            else:
                groups = mirror_quoted_references(
                    row[group_start:group_end], group_map, known_groups
                )
                out.append(row[:group_start] + groups + row[group_end:])
        cursor = end
        idx = end
    out.append(array_text[cursor:])
    return "".join(out)


def relocate_slot_rows(
    array_text: str,
    slot_map: dict[str, str],
    known_slots: set[str],
) -> str:
    """Point a relocated part's child slots at their opposite-side names.

    The nodeMove x on those rows is negated with them: a race seat mounted at
    x=+0.35 by its parent slot has to be mounted at x=-0.35 once the part it
    hangs off has crossed the car.
    """
    text = mirror_quoted_references(array_text, slot_map, known_slots)
    return re.sub(
        r'("node(?:Move|Offset)\d*"\s*:\s*\{[^}]*?"x"\s*:\s*)(' + NUMBER_RE + r")",
        lambda match: f"{match.group(1)}{transform_helpers.format_num(-float(match.group(2)))}",
        text,
    )


def relocate_part_for_slot(
    part_body: str,
    relocation: SlotRelocation,
    node_mirror_map: dict[str, str],
    known_nodes: set[str],
    group_map: dict[str, str],
    known_groups: set[str],
    slot_map: dict[str, str],
    known_slots: set[str],
) -> str:
    """Rebuild a part to live in the paired slot on the other side.

    This is the fallback for a slot pair with no authored counterpart part.
    Unlike ``clone_part_for_target``, which only ever restyles a part's visual
    rows, this rewrites the part's own structure: its slotType, its node ids
    and x coordinates, every reference to those ids, its flexbody group
    bindings and its child slot names.
    """
    out = re.sub(
        r'("slotType"\s*:\s*")([^"]*)(")',
        lambda match: f"{match.group(1)}{relocation.target_slot}{match.group(3)}",
        part_body,
        count=1,
    )
    out = transform_helpers.replace_array_region(
        out,
        "nodes",
        lambda text: mirror_node_rows(
            text, node_mirror_map, known_nodes, group_map, known_groups
        ),
    )
    for key in _NODE_REFERENCE_ARRAYS:
        out = transform_helpers.replace_array_region(
            out,
            key,
            lambda text: mirror_quoted_references(text, node_mirror_map, known_nodes),
        )
    out = transform_helpers.replace_array_region(
        out,
        "flexbodies",
        lambda text: mirror_flexbody_group_lists(text, group_map, known_groups),
    )
    for key in ("slots", "slots2"):
        out = transform_helpers.replace_array_region(
            out,
            key,
            lambda text: relocate_slot_rows(text, slot_map, known_slots),
        )
    offset = relocation.node_offset
    if any(abs(value) > 1e-9 for value in offset):
        out = _inject_node_move(out, offset)
    return out


def _inject_node_move(part_body: str, offset: tuple[float, float, float]) -> str:
    """Nudge a relocated part back to the mirror of where it started.

    The target slot applies its own nodeMove to whatever lands in it. When the
    two paired slots are not mirror images -- a co-driver seat set further back
    and higher -- that would drag the part to the wrong place, so the
    difference is cancelled out here.

    A part that owns nodes is moved by a nodeMove directive so its physics
    follows. A part that owns none is purely a mesh hung off someone else's
    nodes, so its flexbody rows are offset instead.
    """
    directive = (
        '{"nodeMove":{'
        f'"x":{transform_helpers.format_num(offset[0])},'
        f'"y":{transform_helpers.format_num(offset[1])},'
        f'"z":{transform_helpers.format_num(offset[2])}'
        "}},"
    )

    def insert(text: str) -> str:
        idx = 1 if text.startswith("[") else 0
        while idx < len(text) and text[idx] != "[":
            idx += 1
        if idx >= len(text):
            return text
        # After the header row, so the directive applies to every node row.
        end = transform_helpers.find_matching(text, idx, "[", "]")
        return text[: end + 1] + "\n         " + directive + text[end + 1 :]

    if transform_helpers.extract_named_array(part_body, "nodes"):
        return transform_helpers.replace_array_region(part_body, "nodes", insert)
    return transform_helpers.replace_array_region(
        part_body,
        "flexbodies",
        lambda text: _offset_flexbody_rows(text, offset),
    )


def _offset_flexbody_rows(array_text: str, offset: tuple[float, float, float]) -> str:
    out: list[str] = []
    cursor = 0
    idx = 1 if array_text.startswith("[") else 0
    while idx < len(array_text):
        if array_text[idx] != "[":
            idx += 1
            continue
        end = transform_helpers.find_matching(array_text, idx, "[", "]")
        row = array_text[idx:end]
        out.append(array_text[cursor:idx])
        if flexbody_row_mesh(row):
            current = vector_from_row(row, "pos") or (0.0, 0.0, 0.0)
            moved = tuple(current[axis] + offset[axis] for axis in range(3))
            out.append(replace_or_append_inline_vector(row + "]", "pos", moved)[:-1])
        else:
            out.append(row)
        cursor = end
        idx = end
    out.append(array_text[cursor:])
    return "".join(out)


def rewrite_child_slot_defaults(
    array_text: str,
    child_part_map: dict[str, str],
    default_column: int,
    child_slot_suffix: str | None = None,
) -> str:
    """Point child slot defaults at generated counterparts when they exist.

    For slots2, callers may also suffix the slot id/name. BeamNG config
    selections are keyed by that name, so a handed subtree needs a handed slot
    name to keep stale selections from the opposite-hand tree from overriding
    the generated default.
    """
    if not child_part_map:
        return array_text

    out: list[str] = []
    cursor = 0
    idx = 1 if array_text.startswith("[") else 0
    while idx < len(array_text):
        if array_text[idx] != "[":
            idx += 1
            continue
        try:
            end = transform_helpers.find_matching(array_text, idx, "[", "]")
        except ValueError:
            idx += 1
            continue
        row = array_text[idx : end + 1]
        out.append(array_text[cursor:idx])
        out.append(
            _rewrite_slot_default_row(
                row,
                child_part_map,
                default_column,
                child_slot_suffix,
            )
        )
        cursor = end + 1
        idx = end + 1
    out.append(array_text[cursor:])
    return "".join(out)


def _rewrite_slot_default_row(
    row: str,
    child_part_map: dict[str, str],
    default_column: int,
    child_slot_suffix: str | None = None,
) -> str:
    columns: list[tuple[int, int]] = []
    start = 1 if row.startswith("[") else 0
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(row)):
        ch = row[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch in "[{":
            depth += 1
            continue
        if ch in "]}":
            if depth == 0 and ch == "]":
                columns.append((start, idx))
                break
            depth -= 1
            continue
        if ch == "," and depth == 0:
            columns.append((start, idx))
            start = idx + 1
    if default_column >= len(columns):
        return row

    col_start, col_end = columns[default_column]
    value = row[col_start:col_end]
    match = re.match(r'(\s*)"((?:[^"\\]|\\.)*)"(\s*)$', value)
    if match is None:
        return row
    try:
        part_id = json.loads(f'"{match.group(2)}"')
    except Exception:
        part_id = match.group(2)
    replacement = child_part_map.get(part_id)
    if not replacement:
        return row

    replacements = [
        (col_start, col_end, f'{match.group(1)}"{replacement}"{match.group(3)}')
    ]
    if child_slot_suffix and columns:
        slot_start, slot_end = columns[0]
        slot_value = row[slot_start:slot_end]
        slot_match = re.match(r'(\s*)"((?:[^"\\]|\\.)*)"(\s*)$', slot_value)
        if slot_match is not None:
            try:
                slot_id = json.loads(f'"{slot_match.group(2)}"')
            except Exception:
                slot_id = slot_match.group(2)
            if slot_id not in {"type", "name"} and not slot_id.endswith(child_slot_suffix):
                replacements.append(
                    (
                        slot_start,
                        slot_end,
                        f'{slot_match.group(1)}"{slot_id}{child_slot_suffix}"{slot_match.group(3)}',
                    )
                )

    for start, end, value in sorted(replacements, reverse=True):
        row = row[:start] + value + row[end:]
    return row


def rewrite_light_pattern_for_target(part_body: str, target_hand: str) -> str:
    pattern = "RHD" if target_hand == HAND_RHD else "LHD"
    return re.sub(
        r'("\$lightPattern"\s*:\s*")(?:LHD|RHD|US)(")',
        rf"\g<1>{pattern}\2",
        part_body,
    )


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
    child_part_map: dict[str, str] | None = None,
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
    out = transform_helpers.replace_array_region(
        out,
        "slots",
        lambda text: rewrite_child_slot_defaults(text, child_part_map or {}, 1),
    )
    out = transform_helpers.replace_array_region(
        out,
        "slots2",
        lambda text: rewrite_child_slot_defaults(
            text,
            child_part_map or {},
            3,
            suffix_for_hand(target_hand),
        ),
    )
    out = rewrite_light_pattern_for_target(out, target_hand)
    out = re.sub(
        r'("name"\s*:\s*")([^"]*)(")',
        lambda match: f"{match.group(1)}{append_hand_label(match.group(2), target_hand)}{match.group(3)}",
        out,
        count=1,
    )
    return out

__all__ = ['target_hand_for', 'suffix_for_hand', 'signed_delta_for_target', 'generated_mesh_name', 'generated_part_name', 'generated_variant_part_name', 'generated_dae_output_path', 'source_object_position', 'target_object_position', 'mirrored_object_position', 'format_inline_vector', 'vector_pattern', 'replace_inline_vector', 'insert_inline_vector_near_key', 'replace_or_append_inline_vector', 'transform_flexbody_row', 'flexbody_row_can_carry_transform', 'rewrite_flexbody_meshes_with_transforms', 'replace_or_append_prop_translation_global', 'replace_or_append_prop_rotation_global', 'rewrite_flexbody_meshes', 'rewrite_prop_meshes_with_globals', 'swap_token_pair', 'mirror_lateral_node_id', 'build_node_mirror_map', 'mirror_camera_reference', 'rewrite_internal_camera_line', 'CAMERA_HAND_FLAG_RE', 'CAMERA_DRIVER_ROW_RE', 'rewrite_internal_cameras', 'part_has_transformable_internal_camera', 'rewrite_child_slot_defaults', 'rewrite_light_pattern_for_target', 'clone_part_for_target', 'build_lateral_name_map', 'relocated_reference', 'mirror_quoted_references', 'mirror_node_rows', 'mirror_flexbody_group_lists', 'relocate_slot_rows', 'relocate_part_for_slot']
