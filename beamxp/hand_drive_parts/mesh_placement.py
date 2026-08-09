"""Implementation slice extracted from ``beamng_hand_drive_core``.

Original source lines 2071-2406. Import the public orchestration module
``beamng_hand_drive_core`` rather than this implementation module directly.
"""

from __future__ import annotations

import math
import re
from beamxp import transform_helpers
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

def is_shared_dae_object(context: VehicleContext, object_id: str) -> bool:
    obj = context.objects.get(object_id)
    if obj is None or not obj.dae_path:
        return False
    dae_path = obj.dae_path.replace("\\", "/").lower()
    return dae_path.startswith("vehicles/common/") or (
        obj.dae_source_zip is not None
        and obj.dae_source_zip.resolve(strict=False) != context.source_zip.resolve(strict=False)
    )


def baked_mesh_output_name(
    source_mesh: str,
    target_hand: str,
    config_name: str,
    source_part_id: str,
    index: int,
) -> str:
    return safe_id(
        f"{source_mesh}{suffix_for_hand(target_hand)}__{config_name}__{source_part_id}__{index:04d}"
    )


def add_baked_shared_mesh(
    bake_context: SharedBakeContext,
    mesh: str,
    placement_matrix: list[list[float]],
    bake_transform_into_dae: bool,
    is_prop: bool = False,
) -> str | None:
    source_mesh = bake_context.structural_sources.get(mesh, mesh)
    mode = bake_context.object_modes.get(mesh)
    if mode not in {MODE_TRANSLATE, MODE_MIRROR, MODE_MIRROR_STRUCTURAL}:
        return None
    # Mirrored props always need a per-row baked copy: their node-triad frame is
    # left-handed, so the mirrored orientation is a reflection that
    # baseRotationGlobal (an euler rotation) cannot express. Vehicle-local
    # meshes on other paths keep using the shared "_xp_rhd" copies.
    if not is_shared_dae_object(bake_context.context, source_mesh) and not (
        is_prop and mode == MODE_MIRROR
    ):
        return None
    source_obj = bake_context.context.objects.get(source_mesh)
    if source_obj is None or not source_obj.dae_path:
        return None
    output_mesh = baked_mesh_output_name(
        mesh,
        bake_context.target_hand,
        bake_context.config_name,
        bake_context.source_part_id,
        len(bake_context.baked_specs),
    )
    bake_context.baked_specs.append(
        BakedMeshSpec(
            configured_mesh=mesh,
            source_mesh=source_mesh,
            output_mesh=output_mesh,
            target_hand=bake_context.target_hand,
            mode=mode,
            placement_matrix=placement_matrix,
            bake_transform_into_dae=bake_transform_into_dae,
            is_prop=is_prop,
        )
    )
    return output_mesh


def baked_dae_matrix(
    source_node_matrix: list[list[float]],
    spec: BakedMeshSpec,
    translate_magnitudes: dict[str, float],
) -> list[list[float]]:
    mirror = mirror_x_matrix4()

    if not spec.bake_transform_into_dae:
        if spec.mode == MODE_TRANSLATE:
            # Position handled in jbeam (flexbody pos / prop baseTranslationGlobal);
            # the DAE copy stays identical to the source.
            return source_node_matrix
        if spec.is_prop:
            # Prop mirror: jbeam keeps the node-frame anchoring (baseTranslationGlobal moves
            # the anchor to the mirrored position) so the reflection must be baked into the
            # mesh in prop-model space. With world = T(A)*R*(M*g) and the mirrored anchor
            # T(S*A), we need M' = (R^T*S*R)*M*S paired with locally mirrored geometry S*g:
            # T(S*A)*R*M'*(S*g) = S*(T(A)*R*M*g).
            rotation = rotation_transpose_matrix4(spec.placement_matrix)
            reflection = multiply_matrix(
                multiply_matrix(rotation, mirror),
                matrix4_with_rotation_translation(matrix3_from_matrix4(spec.placement_matrix), (0.0, 0.0, 0.0)),
            )
            return multiply_matrix(multiply_matrix(reflection, source_node_matrix), mirror)
        # Flexbody mirror with the pos/rot mirrored in the jbeam row (P' = S*P*S):
        # the DAE copy must supply the world-mirrored mesh in DAE space, i.e.
        # node matrix S*M*S with locally mirrored geometry S*g, so that
        # P'*(S*M*S)*(S*g) = S*(P*M*g).
        return multiply_matrix(multiply_matrix(mirror, source_node_matrix), mirror)

    placement_inverse = inverse_affine_matrix(spec.placement_matrix)
    if spec.mode == MODE_TRANSLATE:
        delta = signed_delta_for_target(
            spec.target_hand,
            translate_magnitudes.get(spec.configured_mesh, 0.0),
        )
        return multiply_matrix(
            multiply_matrix(
                multiply_matrix(placement_inverse, translation_matrix((delta, 0.0, 0.0))),
                spec.placement_matrix,
            ),
            source_node_matrix,
        )
    # Mirror, baked. The reflection alone wants node matrix P^-1*S*P*M paired
    # with locally mirrored geometry S*g, giving world = S*(P*M*g). A Move X
    # nudge is a world-space slide laid on top of that, so T(d) goes between
    # the placement inverse and the reflection: P^-1*T(d)*S*P*M*S then renders
    # T(d)*S*(P*M*g). With no nudge typed d is 0 and this is the matrix that
    # every validated conversion already produces.
    nudge = translation_matrix(
        (signed_delta_for_target(spec.target_hand, translate_magnitudes.get(spec.configured_mesh, 0.0)), 0.0, 0.0)
    )
    return multiply_matrix(
        multiply_matrix(
            multiply_matrix(
                multiply_matrix(multiply_matrix(placement_inverse, nudge), mirror),
                spec.placement_matrix,
            ),
            source_node_matrix,
        ),
        mirror,
    )


def collect_prop_mesh_positions(
    node_positions: dict[str, tuple[float, float, float]],
    part_body_index: dict[str, tuple[str, str]],
    mesh_pivots: dict[str, tuple[float, float, float]] | None = None,
) -> dict[str, list[tuple[float, float, float]]]:
    positions: dict[str, list[tuple[float, float, float]]] = {}
    for part_body, _filename in part_body_index.values():
        props = transform_helpers.extract_named_array(part_body, "props")
        if not props:
            continue
        for row in iter_top_level_rows(props):
            strings = re.findall(r'"((?:[^"\\]|\\.)*)"', row)
            if len(strings) < 2:
                continue
            func, mesh = strings[:2]
            if func == "func" or mesh == "mesh":
                continue
            pivot = (mesh_pivots or {}).get(mesh)
            position = prop_row_pivot_position(row, node_positions, pivot)
            if position is not None:
                positions.setdefault(mesh, []).append(position)
    return positions


def collect_flexbody_mesh_placements(
    objects: dict[str, DaeObject],
    part_body_index: dict[str, tuple[str, str]],
    mesh_pivots: dict[str, tuple[float, float, float]] | None = None,
) -> tuple[dict[str, list[MeshPlacement]], set[str]]:
    """Every flexbody placement in the whole part index, ignoring trims.

    Placements from parts that can never coexist are all present here, so this
    must NOT be used to decide where a mesh sits -- see
    resolved_mesh_positions_for_config for that. It stays because the returned
    positioned-mesh set (meshes any jbeam row gives an explicit pos) is a
    whole-vehicle property the build relies on.

    Positions are measured from the authored pivot: passing objects whose
    coordinates have already been resolved would compound the placement onto
    an already-placed mesh."""
    placements: dict[str, list[MeshPlacement]] = {}
    positioned_meshes: set[str] = set()
    for part_body, _filename in part_body_index.values():
        flexbodies = transform_helpers.extract_named_array(part_body, "flexbodies")
        if not flexbodies:
            continue
        for row in iter_top_level_rows(flexbodies):
            mesh = flexbody_row_mesh(row)
            if not mesh or mesh not in objects:
                continue
            if (
                vector_from_row(row, "pos") is None
                and vector_from_row(row, "rot") is None
                and vector_from_row(row, "scale") is None
            ):
                continue
            obj = objects[mesh]
            pivot = (mesh_pivots or {}).get(mesh, (obj.x, obj.y, obj.z))
            matrix = flexbody_row_matrix(row)
            position = transform_helpers.transform_point(matrix, pivot)
            placements.setdefault(mesh, []).append(MeshPlacement(position=position, matrix=matrix))
            if vector_from_row(row, "pos") is not None:
                positioned_meshes.add(mesh)
    return placements, positioned_meshes


def is_far_placement(position: tuple[float, float, float]) -> bool:
    """Whether a row parks the mesh so far out it is really being hidden.

    Same threshold the preview payload uses to drop instances, so both agree
    on what counts as present in a configuration."""
    return math.hypot(position[0], position[1], position[2]) > PREVIEW_FAR_LIMIT


def average_position(positions: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    return (
        sum(position[0] for position in positions) / len(positions),
        sum(position[1] for position in positions) / len(positions),
        sum(position[2] for position in positions) / len(positions),
    )


def moved_object(obj: DaeObject, position: tuple[float, float, float]) -> DaeObject:
    return DaeObject(
        id=obj.id,
        name=obj.name,
        dae_path=obj.dae_path,
        x=position[0],
        y=position[1],
        z=position[2],
        geometry_ids=obj.geometry_ids,
        dae_source_zip=obj.dae_source_zip,
    )


def bounds_corners(
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    min_point, max_point = bounds
    return [
        (x, y, z)
        for x in (min_point[0], max_point[0])
        for y in (min_point[1], max_point[1])
        for z in (min_point[2], max_point[2])
    ]


def transform_preview_points(
    preview: dict[str, object],
    matrices: list[list[list[float]]],
) -> dict[str, object]:
    raw_points = list(preview.get("sample_points", []))
    if not raw_points and "center" in preview:
        raw_points = [preview["center"]]
    if "bounds" in preview:
        raw_points.extend(bounds_corners(preview["bounds"]))
    points = [
        transform_helpers.transform_point(matrix, point)
        for matrix in matrices
        for point in raw_points
    ]
    if not points:
        return preview
    bounds = transform_helpers.bounds_from_points(points)
    min_point, max_point = bounds
    return {
        **preview,
        "bounds": bounds,
        "center": (
            (min_point[0] + max_point[0]) / 2,
            (min_point[1] + max_point[1]) / 2,
            (min_point[2] + max_point[2]) / 2,
        ),
        "sample_points": transform_helpers.sample_points(points, 350),
    }


def translate_preview_points(
    preview: dict[str, object],
    delta: tuple[float, float, float],
) -> dict[str, object]:
    matrix = translation_matrix(delta)
    return transform_preview_points(preview, [matrix])


def apply_resolved_mesh_positions(
    objects: dict[str, DaeObject],
    preview_by_id: dict[str, dict[str, object]],
    resolved: dict[str, ResolvedMeshPosition],
    mesh_pivots: dict[str, tuple[float, float, float]] | None = None,
    jbeam_positioned_flexbodies: set[str] | None = None,
) -> None:
    """Move each mesh to its representative position (see
    representative_mesh_positions). Flexbody previews are transformed by the
    representative trim's row matrices so rotation/scale survive; prop previews
    are translated by the delta, matching how the engine places each kind.

    A flexbody mesh whose row authors no pos of its own additionally sheds the
    DAE node translation, because that is what the renderers do with it (see
    flexbody_row_needs_node_translation / mesh_preview.build_scene). Without
    this the preview boxes and the drawn geometry disagree by the node
    translation on exactly those meshes -- e.g. etk800's manual shifter."""
    positioned = jbeam_positioned_flexbodies or set()
    for mesh, entry in resolved.items():
        obj = objects.get(mesh)
        if obj is None:
            continue
        objects[mesh] = moved_object(obj, entry.position)

        preview = preview_by_id.get(mesh)
        if preview is None:
            continue
        if entry.matrices:
            preview = transform_preview_points(preview, list(entry.matrices))
            if mesh not in positioned:
                pivot = (mesh_pivots or {}).get(mesh)
                if pivot is not None and max(abs(value) for value in pivot) > 1e-9:
                    preview = translate_preview_points(
                        preview, (-pivot[0], -pivot[1], -pivot[2])
                    )
            preview_by_id[mesh] = preview
        else:
            preview_by_id[mesh] = translate_preview_points(
                preview,
                (
                    entry.position[0] - obj.x,
                    entry.position[1] - obj.y,
                    entry.position[2] - obj.z,
                ),
            )


def original_source_name(context: VehicleContext) -> str:
    source_posix = context.source_zip.as_posix().lower()
    if "/content/vehicles/" in source_posix:
        return "BeamNG - Official"

    mod_info = _zip_json_by_name(context.source_zip, "mod_info/info.json")
    for key in ("source", "Source", "name", "Name", "title", "Title"):
        value = mod_info.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    authors = mod_info.get("authors") or mod_info.get("Authors") or mod_info.get("author")
    if isinstance(authors, str) and authors.strip():
        return authors.strip()
    if isinstance(authors, list):
        joined = ", ".join(str(author).strip() for author in authors if str(author).strip())
        if joined:
            return joined

    return "Custom"

__all__ = ['is_shared_dae_object', 'baked_mesh_output_name', 'add_baked_shared_mesh', 'baked_dae_matrix', 'collect_prop_mesh_positions', 'collect_flexbody_mesh_placements', 'is_far_placement', 'average_position', 'moved_object', 'bounds_corners', 'transform_preview_points', 'translate_preview_points', 'apply_resolved_mesh_positions', 'original_source_name']
