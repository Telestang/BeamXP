"""Implementation slice extracted from ``beamng_hand_drive_core``.

Original source lines 153-584. Import the public orchestration module
``beamng_hand_drive_core`` rather than this implementation module directly.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from typing import Iterable
import numpy as np
from beamxp import transform_helpers
from beamxp.core import dae
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

# ---------------------------------------------------------------------------
# DAE loading/parsing now lives in beamng_dae.  Re-exported here so existing
# call sites keep working unchanged -- and so pickled context caches, which
# record the class as beamng_hand_drive_core.DaeObject, still unpickle.
# ---------------------------------------------------------------------------
DaeObject = dae.DaeObject
parse_dae = dae.parse_dae
dae_unit_scale = dae.dae_unit_scale
dae_objects_from_tree = dae.dae_objects_from_tree
dae_node_aliases = dae.dae_node_aliases
find_dae_node = dae.find_dae_node
list_dae_objects_for_file = dae.list_dae_objects_for_file
list_dae_objects_for_path = dae.list_dae_objects_for_path
common_dae_paths = dae.common_dae_paths
DAE_ALIAS_ATTR_RE = dae.DAE_ALIAS_ATTR_RE
dae_alias_candidates = dae.dae_alias_candidates
geometry_position_points = dae.geometry_position_points
preview_data_for_file = dae.preview_data_for_file
preview_data_from_tree = dae.preview_data_from_tree
surface_triangles_from_tree = dae.surface_triangles_from_tree


def load_common_dae_objects(
    source_zip: Path,
    wanted_meshes: set[str],
    existing_objects: dict[str, DaeObject],
) -> tuple[dict[str, DaeObject], dict[str, dict[str, object]], list[str]]:
    """Resolve meshes this vehicle references but does not ship.

    Thin wrapper that supplies the common-zip search path; the scan itself
    lives in :func:`beamng_dae.load_common_dae_objects`.
    """
    return dae.load_common_dae_objects(
        common_zip_candidates(source_zip),
        wanted_meshes,
        existing_objects,
    )


def referenced_mesh_names(part_body_index: dict[str, tuple[str, str]]) -> set[str]:
    meshes: set[str] = set()
    for part_body, _filename in part_body_index.values():
        meshes.update(transform_helpers.extract_part_mesh_names(part_body))
    return meshes


def dae_source_index(
    context: VehicleContext,
) -> tuple[
    dict[str, tuple[str, str]],
    dict[tuple[str, str], tuple[str, ...]],
    dict[tuple[str, str], tuple[Path, str]],
]:
    """Stable DAE source keys and file membership for every context object.

    Shared accessory packs make source paths significant, but resolving the
    same zip path inside every per-object/per-trim cache lookup is extremely
    expensive on Windows.  Build the immutable index once, resolving each
    distinct zip only once.
    """
    cached = getattr(context, "_dae_source_index", None)
    if cached is not None:
        return cached

    resolved_zips: dict[str, str] = {}
    keys: dict[str, tuple[str, str]] = {}
    members: dict[tuple[str, str], list[str]] = {}
    files: dict[tuple[str, str], tuple[Path, str]] = {}
    for object_id, obj in context.objects.items():
        source_zip = Path(obj.dae_source_zip or context.source_zip)
        raw_zip = str(source_zip)
        zip_key = resolved_zips.get(raw_zip)
        if zip_key is None:
            try:
                zip_key = str(source_zip.resolve(strict=False)).lower()
            except OSError:
                zip_key = raw_zip.lower()
            resolved_zips[raw_zip] = zip_key
        dae_path = obj.dae_path.replace("\\", "/")
        key = (zip_key, dae_path)
        keys[object_id] = key
        members.setdefault(key, []).append(object_id)
        files.setdefault(key, (source_zip, obj.dae_path))

    result = (
        keys,
        {key: tuple(object_ids) for key, object_ids in members.items()},
        files,
    )
    context._dae_source_index = result
    return result


def full_surface_triangles_for_ids(
    context: VehicleContext,
    ids: Iterable[str],
) -> dict[str, np.ndarray]:
    """DAE triangle surfaces aligned to the representative placed previews.

    Each source file is parsed once for surfaces, including shared accessory
    packs referenced through ``dae_source_zip``.  Missing or malformed
    primitives yield an empty surface so callers can safely retain the older
    point-shell result as their fallback.
    """
    surfaces = getattr(context, "_surface_triangles", None)
    if surfaces is None:
        surfaces = {}
        context._surface_triangles = surfaces
    authored_surfaces = getattr(context, "_authored_surface_triangles", None)
    if authored_surfaces is None:
        authored_surfaces = {}
        context._authored_surface_triangles = authored_surfaces
    parsed_files = getattr(context, "_surface_triangle_files", None)
    if parsed_files is None:
        parsed_files = set()
        context._surface_triangle_files = parsed_files

    requested = {str(object_id) for object_id in ids}
    source_keys, source_members, source_files = dae_source_index(context)

    by_file: dict[tuple[str, str], tuple[Path, str]] = {}
    for object_id in requested:
        if object_id in surfaces:
            continue
        obj = context.objects.get(object_id)
        if obj is None or not obj.dae_path:
            continue
        key = source_keys.get(object_id)
        if key is not None:
            by_file.setdefault(key, source_files[key])

    for key, (source_zip, dae_path) in by_file.items():
        if key in parsed_files:
            continue
        parsed_files.add(key)
        try:
            tree = parse_dae(source_zip, dae_path)
            file_surfaces = surface_triangles_from_tree(tree)
            authored_centers = getattr(context, "_authored_full_centers", {})
            needs_preview = any(
                object_id not in authored_centers
                for object_id in source_members.get(key, ())
            )
            file_preview = (
                preview_data_from_tree(tree, max_points_per_object=sys.maxsize)
                if needs_preview else {}
            )
        except Exception:
            file_surfaces = {}
            file_preview = {}

        clouds = getattr(context, "_full_clouds", None)
        if clouds is None:
            clouds = {}
            context._full_clouds = clouds
        authored_clouds = getattr(context, "_authored_full_clouds", None)
        if authored_clouds is None:
            authored_clouds = {}
            context._authored_full_clouds = authored_clouds
        authored_centers = getattr(context, "_authored_full_centers", None)
        if authored_centers is None:
            authored_centers = {}
            context._authored_full_centers = authored_centers
        full_files = getattr(context, "_full_cloud_files", None)
        if full_files is None:
            full_files = set()
            context._full_cloud_files = full_files
        full_files.add(key)

        for object_id in source_members.get(key, ()):
            triangles = file_surfaces.get(object_id)
            if triangles is None or len(triangles) == 0:
                surfaces.setdefault(object_id, np.empty((0, 3, 3), dtype=float))
            authored = file_preview.get(object_id)
            placed = context.preview_by_id.get(object_id)
            if authored is not None:
                points = np.asarray(authored.get("sample_points", ()), dtype=float)
                authored_center = np.asarray(authored.get("center"), dtype=float)
                if len(points):
                    authored_clouds[object_id] = points
                    if authored_center.shape == (3,):
                        authored_centers[object_id] = authored_center
                    if placed is not None:
                        placed_center = np.asarray(placed.get("center"), dtype=float)
                        if authored_center.shape == (3,) and placed_center.shape == (3,):
                            points = points + (placed_center - authored_center)
                    clouds[object_id] = points
            if triangles is None or len(triangles) == 0:
                continue
            authored_surfaces[object_id] = triangles
            authored_center = authored_centers.get(object_id)
            if authored_center is not None and placed is not None:
                authored_center = np.asarray(authored_center, dtype=float)
                placed_center = np.asarray(placed.get("center"), dtype=float)
                if authored_center.shape == (3,) and placed_center.shape == (3,):
                    triangles = triangles + (placed_center - authored_center)
            surfaces[object_id] = triangles

    return {
        object_id: surfaces.get(object_id, np.empty((0, 3, 3), dtype=float))
        for object_id in requested
    }


def full_vertex_clouds_for_ids(
    context: VehicleContext,
    ids: Iterable[str],
) -> dict[str, np.ndarray]:
    """Uncapped DAE vertex clouds, aligned to the placed preview centres.

    Preview clouds are deliberately capped for interactive work, but striding
    and truncating a vertex buffer can split exact mirror pairs.  Self-
    symmetry therefore needs every authored vertex.  Each source DAE is
    parsed at most once, including DAEs supplied by ``dae_source_zip``; any
    unreadable mesh falls back to its preview cloud so failure favours a
    benign extra mirror rather than a missed asymmetric part.

    JBeam-placed shared meshes are commonly authored at the origin.  Their
    full clouds are translated onto the representative preview bbox centre.
    Per-trim x translation is applied by the classifier, which has the trim's
    resolved preview in hand.
    """
    clouds = getattr(context, "_full_clouds", None)
    if clouds is None:
        clouds = {}
        context._full_clouds = clouds
    authored_clouds = getattr(context, "_authored_full_clouds", None)
    if authored_clouds is None:
        authored_clouds = {}
        context._authored_full_clouds = authored_clouds
    authored_centers = getattr(context, "_authored_full_centers", None)
    if authored_centers is None:
        authored_centers = {}
        context._authored_full_centers = authored_centers
    parsed_files = getattr(context, "_full_cloud_files", None)
    if parsed_files is None:
        parsed_files = set()
        context._full_cloud_files = parsed_files

    requested = {str(object_id) for object_id in ids}
    source_keys, source_members, source_files = dae_source_index(context)

    by_file: dict[tuple[str, str], tuple[Path, str]] = {}
    for object_id in requested:
        if object_id in clouds:
            continue
        obj = context.objects.get(object_id)
        if obj is None or not obj.dae_path:
            continue
        key = source_keys.get(object_id)
        if key is not None:
            by_file.setdefault(key, source_files[key])

    for key, (source_zip, dae_path) in by_file.items():
        if key in parsed_files:
            continue
        parsed_files.add(key)
        try:
            full_preview = preview_data_from_tree(
                parse_dae(source_zip, dae_path),
                max_points_per_object=sys.maxsize,
            )
        except Exception:
            full_preview = {}

        # Cache every context object backed by this DAE.  A later trim can ask
        # about a different mesh in the same file without forcing a reparse.
        for object_id in source_members.get(key, ()):
            authored = full_preview.get(object_id)
            placed = context.preview_by_id.get(object_id)
            if authored is None or placed is None:
                continue
            points = np.asarray(authored.get("sample_points", ()), dtype=float)
            if len(points) == 0:
                continue
            authored_clouds[object_id] = points
            authored_center = np.asarray(authored.get("center"), dtype=float)
            if authored_center.shape == (3,):
                authored_centers[object_id] = authored_center
            placed_center = np.asarray(placed.get("center"), dtype=float)
            if authored_center.shape == (3,) and placed_center.shape == (3,):
                points = points + (placed_center - authored_center)
            clouds[object_id] = points

    result: dict[str, np.ndarray] = {}
    for object_id in requested:
        points = clouds.get(object_id)
        if points is None:
            preview = context.preview_by_id.get(object_id, {})
            points = np.asarray(preview.get("sample_points", ()), dtype=float)
            clouds[object_id] = points
        result[object_id] = points
    return result


def vertex_cloud_for_resolved_placement(
    context: VehicleContext,
    object_id: str,
    resolved: ResolvedMeshPosition,
    max_points: int = 350,
) -> np.ndarray | None:
    """Rebuild one flexbody cloud from its authored DAE and trim matrices.

    Representative previews can contain a different instance count from the
    current trim.  Apply every real placement and return their sampled union;
    this avoids both phantom twins in single-seat trims and fictitious averaged
    positions in two-seat/four-wheel trims.
    """
    if not resolved.matrices:
        return None
    authored = getattr(context, "_authored_full_clouds", {}).get(object_id)
    if authored is None:
        full_vertex_clouds_for_ids(context, (object_id,))
        authored = getattr(context, "_authored_full_clouds", {}).get(object_id)
    if authored is None or len(authored) == 0:
        return None
    homogeneous = np.concatenate(
        [np.asarray(authored, dtype=float), np.ones((len(authored), 1), dtype=float)],
        axis=1,
    )
    chunks = [
        (homogeneous @ np.asarray(matrix, dtype=float).T)[:, :3]
        for matrix in resolved.matrices
    ]
    points = np.concatenate(chunks)
    if object_id not in context.jbeam_positioned_flexbodies:
        pivot = context.mesh_pivots.get(object_id)
        if pivot is not None and max(abs(value) for value in pivot) > 1e-9:
            points = points - np.asarray(pivot, dtype=float)
    if len(points) > max_points:
        stride = max(1, len(points) // max_points)
        points = points[::stride][:max_points]
    return points


def surface_triangles_for_resolved_placement(
    context: VehicleContext,
    object_id: str,
    resolved: ResolvedMeshPosition,
) -> np.ndarray | None:
    """Rebuild one object's authored surface at all of its trim matrices."""
    if not resolved.matrices:
        return None
    authored = getattr(context, "_authored_surface_triangles", {}).get(object_id)
    if authored is None:
        full_surface_triangles_for_ids(context, (object_id,))
        authored = getattr(context, "_authored_surface_triangles", {}).get(object_id)
    if authored is None or len(authored) == 0:
        return None
    flat = np.asarray(authored, dtype=float).reshape((-1, 3))
    homogeneous = np.concatenate(
        [flat, np.ones((len(flat), 1), dtype=float)], axis=1
    )
    triangles = np.concatenate([
        (homogeneous @ np.asarray(matrix, dtype=float).T)[:, :3].reshape((-1, 3, 3))
        for matrix in resolved.matrices
    ])
    if object_id not in context.jbeam_positioned_flexbodies:
        pivot = context.mesh_pivots.get(object_id)
        if pivot is not None and max(abs(value) for value in pivot) > 1e-9:
            triangles = triangles - np.asarray(pivot, dtype=float)
    return triangles


def prop_rest_rotation_override(
    row: str,
    node_positions: dict[str, tuple[float, float, float]] | None = None,
) -> tuple[list[list[float]] | None, str]:
    """Resolve a prop row's REST rotation and report the source.

    The engine renders props from NODE-LOCAL geometry: the DAE node's
    ROTATION is discarded at load (its translation becomes the pivot, scale
    is kept) and baseRotationGlobal - authored or engine-computed - IS the
    rest rotation applied to that local geometry. Renderers must therefore
    pair these rotations with derotated node transforms (see
    mesh_preview/blender backend). Ground truth: bx_steer's node carries
    Rx(~73.5deg) over flat-authored verts and dumps brg 71.6deg; the
    sunburst2 driveshaft node carries Rx(180) and authors brg -176.75 (the
    3.25deg difference is the deliberate driveline angle). Meshes with
    rotation-free nodes (grp_*, steeringwheels) are unaffected.

    Resolution order: authored baseRotationGlobal (the field is the rest
    rotation verbatim), then the analytic engine model. None means the caller
    falls back to prop_row_global_rotation_matrix, only approximate for rows
    without authored brg."""
    authored = vector_from_row(row, "baseRotationGlobal")
    if authored is not None:
        return (
            brg_rotation_matrix3([math.radians(float(v)) for v in authored]),
            "authored-brg",
        )
    if node_positions is not None:
        rotation = prop_engine_rest_rotation(row, node_positions)
        if rotation is not None:
            return rotation, "analytic-engine"
    return None, "analytic"


def prop_engine_rest_rotation(
    row: str,
    node_positions: dict[str, tuple[float, float, float]],
) -> list[list[float]] | None:
    """ENGINE-EXACT rest rotation for prop rows without authored brg.

    rest = F * B^T, where F is the RIGHT-handed triad frame
      x = norm(idX - idRef), y = norm((idY - idRef) x x), z = x x y
    and B is prop_base_rotation_matrix3 (the meshs.lua "-X -Z +Y" euler) -
    transposed because the engine composes in row-vector convention like
    every other euler here. Returns None on degenerate triads."""
    strings = re.findall(r'"((?:[^"\\]|\\.)*)"', row)
    if len(strings) < 5:
        return None
    ref_pos = node_positions.get(strings[2])
    x_pos = node_positions.get(strings[3])
    y_pos = node_positions.get(strings[4])
    if ref_pos is None or x_pos is None or y_pos is None:
        return None
    axis_x = normalize_vector(vector_subtract(x_pos, ref_pos))
    seed = normalize_vector(vector_subtract(y_pos, ref_pos))
    axis_y = normalize_vector(cross_product(seed, axis_x))
    if axis_x == (0.0, 0.0, 0.0) or axis_y == (0.0, 0.0, 0.0):
        return None
    axis_z = cross_product(axis_x, axis_y)
    frame = matrix3_from_axes(axis_x, axis_y, axis_z)
    vectors = prop_row_vector_objects(row)
    base = vectors[0] if vectors else (0.0, 0.0, 0.0)
    return multiply_matrix3(
        frame,
        rotation_transpose_matrix3(prop_base_rotation_matrix3(base)),
    )

__all__ = ['DaeObject', 'parse_dae', 'dae_unit_scale', 'dae_objects_from_tree', 'dae_node_aliases', 'find_dae_node', 'list_dae_objects_for_file', 'list_dae_objects_for_path', 'common_dae_paths', 'DAE_ALIAS_ATTR_RE', 'dae_alias_candidates', 'geometry_position_points', 'preview_data_for_file', 'preview_data_from_tree', 'surface_triangles_from_tree', 'load_common_dae_objects', 'referenced_mesh_names', 'dae_source_index', 'full_surface_triangles_for_ids', 'full_vertex_clouds_for_ids', 'vertex_cloud_for_resolved_placement', 'surface_triangles_for_resolved_placement', 'prop_rest_rotation_override', 'prop_engine_rest_rotation']
