"""Implementation slice extracted from ``beamng_hand_drive_core``.

Original source lines 6097-7267. Import the public orchestration module
``beamng_hand_drive_core`` rather than this implementation module directly.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import io
import math
import re
import zipfile
from dataclasses import dataclass
from collections.abc import Iterable
from xml.etree import ElementTree as ET
import numpy as np
from beamxp import spatial_visibility_backend
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

# ---------------------------------------------------------------------------
# Spatial geometry for Recommend Modes
#
# The classifier in beamng_hand_drive_tool.build_mode_recommendations reasons
# from the driver's viewpoint: an eye point parsed from the jbeam internal
# camera, a nearest-surface shell swept around it, and per-mesh evidence
# (visibility, backing, symmetry, twin geometry). The pure geometry lives
# here; the tool module only orchestrates.

SPATIAL_BIN_DEG = 6.0


@dataclass(frozen=True)
class DriverFrame:
    """The driver's viewpoint: everything the classifier measures is relative
    to this frame, so it transfers across cabins (saloon, truck, buggy)."""

    eye: tuple[float, float, float]
    forward: tuple[float, float, float]
    side: int  # +1 driver sits at +x of the centreline, -1 at -x
    center_x: float
    wheel_id: str | None
    wheel_center: tuple[float, float, float] | None
    source: str  # "camera+wheel" | "camera" | "wheel"


def iter_named_array_texts(text: str, key: str) -> list[str]:
    """Bracket-matched bodies of every '"key": [...]' array in a jbeam text.

    Raw-text variant of part_named_array_for_context for callers that have no
    part id in hand (the internal camera can sit in any part of any file)."""
    bodies: list[str] = []
    for match in re.finditer(rf'"{re.escape(key)}"\s*:\s*\[', text):
        depth = 0
        i = match.end() - 1
        start = i
        in_str = False
        escaped = False
        while i < len(text):
            ch = text[i]
            if in_str:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        bodies.append(text[start:i + 1])
    return bodies


def parse_internal_camera_positions(
    jbeam_texts: dict[str, str],
    node_positions: dict[str, tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    """Eye positions of every "dash"/"driver" camerasInternal row.

    Coordinates are literal numbers on every vanilla vehicle measured so far;
    a quoted value is resolved through node_positions component-wise as a
    fallback, and rows that resolve nothing are dropped rather than guessed."""
    rows: list[tuple[float, float, float]] = []
    for text in jbeam_texts.values():
        for array_text in iter_named_array_texts(text, "camerasInternal"):
            for row in iter_active_top_level_rows(array_text):
                values = split_top_level_values(row)
                if not values:
                    continue
                kind = quoted_string_value(values[0])
                if kind not in ("dash", "driver"):
                    continue
                position: list[float] = []
                ok = True
                for value in values[1:4]:
                    value = value.strip()
                    quoted = quoted_string_value(value)
                    if quoted is not None:
                        node = node_positions.get(quoted)
                        if node is None:
                            ok = False
                            break
                        position.append(float(node[len(position)]))
                        continue
                    try:
                        position.append(float(value))
                    except ValueError:
                        ok = False
                        break
                if ok and len(position) == 3:
                    rows.append((position[0], position[1], position[2]))
    return rows


def _camera_bearing_parts(
    context: VehicleContext,
) -> dict[str, list[tuple[float, float, float]]]:
    """{part_id: [(x, y, z), ...]} for every part that defines a driver/dash
    internal camera, scanned once per context and cached.

    Only a handful of a vehicle's thousands of parts carry a camera, so this
    lets camera_positions_for_config check just those instead of re-scanning
    every selected part on every config."""
    cached = getattr(context, "_camera_parts_cache", None)
    if cached is not None:
        return cached
    result: dict[str, list[tuple[float, float, float]]] = {}
    index = getattr(context, "part_body_index", None)
    for part_id in (list(index.keys()) if isinstance(index, dict) else []):
        body = part_body_for_context(context, part_id)
        if body is None or "camerasInternal" not in body[0]:
            continue
        cameras = parse_internal_camera_positions(
            {part_id: body[0]}, context.node_positions
        )
        if cameras:
            result[part_id] = cameras
    context._camera_parts_cache = result
    return result


def camera_positions_for_config(
    context: VehicleContext,
    config_name: str,
) -> list[tuple[float, float, float]]:
    """Driver/dash camera eyes for one config, with each camera-bearing part's
    nodeMove/nodeOffset slot transform applied.

    The flexbody pipeline already moves a part's meshes by its slot transform;
    the camera must get the same move or the eye lands in the unmoved frame. The
    us_semi cabover cab is nodeMove'd +1.94 m (conventional +2.05 m), so its raw
    camera at y=-1.2 would otherwise sit ~2 m behind the moved seats and dash."""
    camera_parts = _camera_bearing_parts(context)
    if not camera_parts:
        return []
    selected = selected_parts_for_config(context, config_name)
    selected_ids = {str(item) for item in selected.get("parts", set())}
    slot_options = selected.get("part_slot_options", {})
    if not isinstance(slot_options, dict):
        slot_options = {}
    rows: list[tuple[float, float, float]] = []
    for part_id, cameras in camera_parts.items():
        if part_id not in selected_ids:
            continue
        options = slot_options.get(part_id, ())
        ops = node_transform_ops(
            tuple(str(item) for item in options if item)
            if isinstance(options, (list, tuple))
            else ()
        )
        for x, y, z in cameras:
            dx, dy, dz = node_translation_offset(ops, sign_number(x))
            rows.append((x + dx, y + dy, z + dz))
    return rows


def _driver_frame_core(
    context: VehicleContext,
    object_ids: Iterable[str] | None,
) -> tuple[str | None, float, np.ndarray | None]:
    """Config-independent parts of the driver frame: the steering-ref wheel id,
    the lateral centre, and the wheel's raw sample points.

    These depend only on the context, never on the trim, so
    driver_frame_for_context caches this once per context (when the full object
    set is used) and reuses it for every trim -- only the eye and the
    eye-dependent rim/forward differ between trims."""
    candidates = list(object_ids) if object_ids is not None else list(context.objects)
    best: tuple[int, int, str] | None = None
    for object_id in candidates:
        obj = context.objects.get(object_id)
        if obj is None or object_id not in context.preview_by_id:
            continue
        if is_default_steering_ref(object_id, obj):
            rank = (
                steering_ref_score(object_id, obj),
                vehicle_prefix_rank(context, object_id),
            )
            if best is None or (rank[0], -rank[1]) > (best[0], -best[1]):
                best = (rank[0], rank[1], object_id)
    wheel_id = best[2] if best else None
    center_x = median_value([p[0] for p in context.node_positions.values()]) or 0.0
    wheel_points = (
        np.array(context.preview_by_id[wheel_id]["sample_points"], dtype=float)
        if wheel_id is not None
        else None
    )
    return wheel_id, center_x, wheel_points


def _assemble_driver_frame(
    eye: tuple[float, float, float] | None,
    wheel_id: str | None,
    center_x: float,
    wheel_points: np.ndarray | None,
) -> DriverFrame | None:
    """Build the DriverFrame from the eye plus the config-independent core.

    Eye-dependent only: the wheel rim filter (isolating the hub near/below the
    eye), the forward direction, and the side. A pure function of its inputs, so
    driver_frame_for_context memoises the result by eye."""
    wheel_center: tuple[float, float, float] | None = None
    if wheel_points is not None:
        points = wheel_points
        if eye is not None:
            # Wheel meshes may include the whole steering shaft; keep the rim
            # (near and below the eye) so the centre is the hub, not the rack.
            distances = np.linalg.norm(points - np.array(eye), axis=1)
            mask = (distances < 1.3) & (points[:, 2] < eye[2] + 0.25)
            if int(mask.sum()) >= 10:
                points = points[mask]
        wheel_center = tuple(float(v) for v in np.median(points, axis=0))

    if eye is None and wheel_center is None:
        return None
    if eye is None:
        eye = (wheel_center[0], wheel_center[1] + 0.60, wheel_center[2] + 0.35)
        source = "wheel"
        forward = (0.0, -1.0, 0.0)
    elif wheel_center is None:
        source = "camera"
        forward = (0.0, -1.0, 0.0)
    else:
        source = "camera+wheel"
        fxy = np.array([wheel_center[0] - eye[0], wheel_center[1] - eye[1]])
        norm = float(np.linalg.norm(fxy))
        forward = (fxy[0] / norm, fxy[1] / norm, 0.0) if norm > 0.05 else (0.0, -1.0, 0.0)
    side = 1 if (eye[0] - center_x) >= 0 else -1
    return DriverFrame(
        eye=eye,
        forward=(float(forward[0]), float(forward[1]), 0.0),
        side=side,
        center_x=float(center_x),
        wheel_id=wheel_id,
        wheel_center=wheel_center,
        source=source,
    )


def driver_frame_for_context(
    context: VehicleContext,
    object_ids: Iterable[str] | None = None,
    config_name: str | None = None,
) -> DriverFrame | None:
    """Anchor the classifier on the driver's eye.

    Camera rows give the eye; the steering wheel corroborates it and fixes the
    forward direction. Camera without wheel keeps BeamNG's -y-forward
    convention; wheel without camera estimates the eye behind/above the rim
    (degraded); neither means the spatial frame is untrustworthy and callers
    must emit nothing rather than guess.

    When config_name is given, camera eyes carry that config's per-part nodeMove
    (see camera_positions_for_config) so the eye stays with the moved cab;
    otherwise the raw camera rows across all parts are used (context-level).

    The config-independent core (steering-ref wheel, centre-x, wheel points) is
    computed once and cached on the context; assembled frames are memoised by
    eye. So the many trims of a vehicle that share a camera position -- e.g. a
    single-cab car's whole trim list -- reuse one frame instead of rescanning
    every object per trim. An explicit object_ids subset bypasses both caches."""
    cache_ok = object_ids is None

    core = getattr(context, "_driver_frame_core", None) if cache_ok else None
    if core is None:
        core = _driver_frame_core(context, object_ids)
        if cache_ok:
            context._driver_frame_core = core
    wheel_id, center_x, wheel_points = core

    if config_name is not None:
        camera_rows = camera_positions_for_config(context, config_name)
    else:
        camera_rows = parse_internal_camera_positions(context.jbeam_texts, context.node_positions)
    eye: tuple[float, float, float] | None = None
    if camera_rows:
        arr = np.array(camera_rows, dtype=float)
        eye = tuple(float(v) for v in np.median(arr, axis=0))

    frame_cache = None
    if cache_ok:
        frame_cache = getattr(context, "_driver_frame_by_eye", None)
        if frame_cache is None:
            frame_cache = {}
            context._driver_frame_by_eye = frame_cache
        if eye in frame_cache:
            return frame_cache[eye]

    result = _assemble_driver_frame(eye, wheel_id, center_x, wheel_points)
    if frame_cache is not None:
        frame_cache[eye] = result
    return result


def spatial_spherical_bins(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Equal-angle (elevation, azimuth) bin index and range per direction."""
    radii = np.maximum(np.linalg.norm(vectors, axis=1), 1e-9)
    elevation = np.arcsin(np.clip(vectors[:, 2] / radii, -1.0, 1.0))
    azimuth = np.arctan2(vectors[:, 1], vectors[:, 0])
    step = math.radians(SPATIAL_BIN_DEG)
    n_azimuth = int(2 * math.pi / step) + 1
    i_el = ((elevation + math.pi / 2) / step).astype(np.int64)
    i_az = ((azimuth + math.pi) / step).astype(np.int64)
    return i_el * n_azimuth + i_az, radii


def visibility_scan(
    points_by_id: dict[str, np.ndarray],
    eye: tuple[float, float, float],
    transparent_ids: set[str],
    forward: tuple[float, float, float] | None = None,
) -> dict[str, dict[str, float]]:
    """Nearest-surface shell around the eye plus per-mesh evidence.

    Per mesh:
      vf     -- fraction of points on/inside the 360-degree shell
      front_vf -- fraction both on/inside the shell and in the forward
                  180-degree hemisphere (equals vf when forward is omitted)
      backed -- fraction with ANY other mesh somewhere behind (not outermost)
      lined  -- fraction with another mesh CLOSE behind, 3-30 cm (a lining
                layer: door card over skin); own-mesh thickness never counts
      depth  -- mean distance points sit behind the shell (occlusion depth)
      front_backed -- backed fraction contributed only by forward points
      front_lined  -- lined fraction contributed only by forward points
      front_depth  -- mean occlusion depth of forward points only
      min_r  -- nearest point to the eye

    The shell is per-bin nearest with a range-scaled tolerance; deliberately
    NOT dilated -- on ~350-point clouds dilation over-occludes far worse than
    leaks under-occlude, and the vetoes downstream absorb the leaks."""
    ids: list[str] = []
    chunks: list[np.ndarray] = []
    owners: list[np.ndarray] = []
    for index, (object_id, points) in enumerate(points_by_id.items()):
        ids.append(object_id)
        chunks.append(points)
        owners.append(np.full(len(points), index))
    all_points = np.concatenate(chunks)
    owner = np.concatenate(owners)
    eye_vectors = all_points - np.array(eye)
    bins, radii = spatial_spherical_bins(eye_vectors)
    opaque = np.array([ids[o] not in transparent_ids for o in owner])
    if forward is None:
        forward_mask = np.ones(len(all_points), dtype=bool)
    else:
        forward_v = np.asarray(forward, dtype=float)
        norm = float(np.linalg.norm(forward_v))
        if norm > 1e-9:
            forward_v /= norm
        forward_mask = (eye_vectors @ forward_v) >= 0.0
    gpu_shell = spatial_visibility_backend.gpu_visibility_shell(
        eye_vectors, bins, radii, owner, opaque, forward
    )
    if gpu_shell is not None:
        visible, front_visible, backed, lined_flag, depth = gpu_shell
        stats: dict[str, dict[str, float]] = {}
        for index, object_id in enumerate(ids):
            mask = owner == index
            count = int(mask.sum())
            front_mask = mask & forward_mask
            stats[object_id] = {
                "n": count,
                "vf": float(visible[mask].sum()) / count,
                "front_vf": float(front_visible[mask].sum()) / count,
                "backed": float(backed[mask].sum()) / count,
                "lined": float(lined_flag[mask].sum()) / count,
                "depth": float(depth[mask].mean()),
                "front_backed": float(backed[front_mask].sum()) / count,
                "front_lined": float(lined_flag[front_mask].sum()) / count,
                "front_depth": (
                    float(depth[front_mask].mean())
                    if front_mask.any() else float("inf")
                ),
                "min_r": float(radii[mask].min()),
            }
        return stats
    n_bins = int(bins.max()) + 2
    field = np.full(n_bins, np.inf)
    np.minimum.at(field, bins[opaque], radii[opaque])

    far1 = np.full(n_bins, -np.inf)
    far1_owner = np.full(n_bins, -1)
    far2 = np.full(n_bins, -np.inf)
    far2_owner = np.full(n_bins, -1)
    bins_o = bins[opaque]
    radii_o = radii[opaque]
    owner_o = owner[opaque]
    order = np.lexsort((radii_o, bins_o))
    sorted_bins = bins_o[order]
    sorted_radii = radii_o[order]
    sorted_owner = owner_o[order]
    is_last = np.r_[sorted_bins[1:] != sorted_bins[:-1], True]
    far1[sorted_bins[is_last]] = sorted_radii[is_last]
    far1_owner[sorted_bins[is_last]] = sorted_owner[is_last]
    second_idx = np.flatnonzero(is_last) - 1
    second_idx = second_idx[second_idx >= 0]
    is_second = np.zeros(len(sorted_bins), dtype=bool)
    is_second[second_idx] = True
    if len(sorted_bins):
        is_second &= np.r_[sorted_bins[:-1] == sorted_bins[1:], False]
    far2[sorted_bins[is_second]] = sorted_radii[is_second]
    far2_owner[sorted_bins[is_second]] = sorted_owner[is_second]

    # close backing (lined): another mesh within (0.03, 0.30] behind, same bin
    lined_flag = np.zeros(len(all_points), dtype=bool)
    segment_start = np.flatnonzero(np.r_[True, sorted_bins[1:] != sorted_bins[:-1]])
    segment_end = np.r_[segment_start[1:], len(sorted_bins)]
    opaque_idx = np.flatnonzero(opaque)
    for seg_a, seg_b in zip(segment_start, segment_end):
        if seg_b - seg_a < 2:
            continue
        seg_radii = sorted_radii[seg_a:seg_b]
        seg_owner = sorted_owner[seg_a:seg_b]
        for i in range(seg_b - seg_a):
            r_i = seg_radii[i]
            lo = np.searchsorted(seg_radii, r_i + 0.03, side="right")
            hi = np.searchsorted(seg_radii, r_i + 0.30, side="right")
            if lo < hi and np.any(seg_owner[lo:hi] != seg_owner[i]):
                lined_flag[opaque_idx[order[seg_a + i]]] = True

    tolerance = 0.05 + 0.04 * radii
    visible = radii <= field[bins] + tolerance
    if forward is None:
        front_visible = visible
    else:
        front_visible = visible & forward_mask
    depth = np.maximum(0.0, radii - field[bins])
    behind1 = (far1[bins] > radii + 0.05) & (far1_owner[bins] != owner)
    behind2 = (far2[bins] > radii + 0.05) & (far2_owner[bins] != owner)
    backed = behind1 | behind2

    stats: dict[str, dict[str, float]] = {}
    for index, object_id in enumerate(ids):
        mask = owner == index
        count = int(mask.sum())
        front_mask = mask & forward_mask
        stats[object_id] = {
            "n": count,
            "vf": float(visible[mask].sum()) / count,
            "front_vf": float(front_visible[mask].sum()) / count,
            "backed": float(backed[mask].sum()) / count,
            "lined": float(lined_flag[mask].sum()) / count,
            "depth": float(depth[mask].mean()),
            "front_backed": float(backed[front_mask].sum()) / count,
            "front_lined": float(lined_flag[front_mask].sum()) / count,
            "front_depth": (
                float(depth[front_mask].mean())
                if front_mask.any() else float("inf")
            ),
            "min_r": float(radii[mask].min()),
        }
    return stats


def surface_visibility_stats(
    points: np.ndarray,
    eye: tuple[float, float, float],
    triangles_by_id: dict[str, np.ndarray] | np.ndarray,
    transparent_ids: set[str],
    forward: tuple[float, float, float] | None = None,
    endpoint_gap: float = 0.002,
) -> dict[str, float] | None:
    """Exact point visibility against filled mesh triangles.

    Rays run from the eye to each sampled point.  A point is hidden when any
    opaque triangle intersects the open segment, including an earlier surface
    of its own mesh.  Intersections within ``endpoint_gap`` of the sampled
    point are ignored so that the point's incident face does not hide itself.
    Callers may supply per-object arrays or one scene compiled once for the
    trim; the classifier uses the latter to avoid hundreds of tiny NumPy
    kernels per candidate.
    """
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return None
    origin = np.asarray(eye, dtype=float)
    directions = points - origin
    distances = np.linalg.norm(directions, axis=1)
    valid_rays = distances > 1e-9
    blocked = np.zeros(len(points), dtype=bool)
    has_surface = False

    # Moller-Trumbore, vectorised over all still-live rays and bounded chunks
    # of triangles.  Parameters use the unnormalised eye->point vector, so an
    # intersection lies on the segment precisely when 0 < t < 1.
    if isinstance(triangles_by_id, np.ndarray):
        surface_items = (("", triangles_by_id),)
    else:
        surface_items = triangles_by_id.items()
    for object_id, object_triangles in surface_items:
        if object_id in transparent_ids:
            continue
        triangles = np.asarray(object_triangles, dtype=float).reshape((-1, 3, 3))
        if len(triangles) == 0:
            continue
        has_surface = True
        for start in range(0, len(triangles), 1024):
            active = np.flatnonzero(valid_rays & ~blocked)
            if len(active) == 0:
                break
            tri = triangles[start : start + 1024]
            vertex0 = tri[:, 0]
            edge1 = tri[:, 1] - vertex0
            edge2 = tri[:, 2] - vertex0
            # Degenerate export faces cannot occlude anything and amplify
            # floating-point noise in the determinant.
            nondegenerate = np.linalg.norm(np.cross(edge1, edge2), axis=1) > 1e-12
            if not bool(nondegenerate.any()):
                continue
            vertex0 = vertex0[nondegenerate]
            edge1 = edge1[nondegenerate]
            edge2 = edge2[nondegenerate]
            ray_d = directions[active]
            pvec = np.cross(ray_d[:, None, :], edge2[None, :, :])
            det = np.einsum("tj,rtj->rt", edge1, pvec)
            determinant_ok = np.abs(det) > 1e-10
            inverse_det = np.zeros_like(det)
            np.divide(1.0, det, out=inverse_det, where=determinant_ok)
            tvec = origin[None, :] - vertex0
            u = np.einsum("tj,rtj->rt", tvec, pvec) * inverse_det
            qvec = np.cross(tvec, edge1)
            v = np.einsum("rj,tj->rt", ray_d, qvec) * inverse_det
            t_num = np.einsum("tj,tj->t", edge2, qvec)
            t = t_num[None, :] * inverse_det
            maximum_t = 1.0 - endpoint_gap / distances[active]
            hit = (
                determinant_ok
                & (u >= -1e-9)
                & (v >= -1e-9)
                & (u + v <= 1.0 + 1e-9)
                & (t > 1e-7)
                & (t < maximum_t[:, None])
            )
            blocked[active[np.any(hit, axis=1)]] = True
        if bool((valid_rays & ~blocked).sum()) == 0:
            break

    if not has_surface:
        return None
    visible = valid_rays & ~blocked
    if forward is None:
        front_visible = visible
    else:
        forward_v = np.asarray(forward, dtype=float)
        norm = float(np.linalg.norm(forward_v))
        if norm > 1e-9:
            forward_v /= norm
        front_visible = visible & ((directions @ forward_v) >= 0.0)
    return {
        "vf": float(visible.sum()) / len(points),
        "front_vf": float(front_visible.sum()) / len(points),
        "blocked": float(blocked.sum()) / len(points),
    }


def surface_visibility_stats_batch(
    points_by_id: dict[str, np.ndarray],
    eye: tuple[float, float, float],
    triangles: np.ndarray,
    forward: tuple[float, float, float] | None = None,
    endpoint_gap: float = 0.002,
    max_cpu_workers: int = 4,
) -> dict[str, dict[str, float] | None]:
    """Exact visibility for many meshes with one compiled triangle scene.

    The GPU path concatenates every candidate's rays into one compute dispatch
    and uploads the scene once.  It implements the same Moller-Trumbore/open-
    segment test as :func:`surface_visibility_stats`.  Machines without an
    OpenGL 4.3 compute context retain the previous bounded CPU thread pool.
    Set ``BEAMXP_SPATIAL_BACKEND=cpu`` to force that reference path.
    """
    usable = {
        object_id: np.asarray(points, dtype=float).reshape((-1, 3))
        for object_id, points in points_by_id.items()
        if len(points)
    }
    if not usable:
        return {}
    scene = np.asarray(triangles, dtype=float).reshape((-1, 3, 3))
    if not len(scene):
        return {object_id: None for object_id in usable}

    ids = list(usable)
    counts = [len(usable[object_id]) for object_id in ids]
    all_points = np.concatenate([usable[object_id] for object_id in ids])
    blocked = spatial_visibility_backend.gpu_blocked_rays(
        all_points, eye, scene, endpoint_gap
    )
    if blocked is None:
        workers = max(1, min(max_cpu_workers, len(ids)))
        if workers == 1:
            return {
                object_id: surface_visibility_stats(
                    usable[object_id], eye, scene, set(), forward, endpoint_gap
                )
                for object_id in ids
            }
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="spatial-rays"
        ) as executor:
            futures = {
                object_id: executor.submit(
                    surface_visibility_stats,
                    usable[object_id],
                    eye,
                    scene,
                    set(),
                    forward,
                    endpoint_gap,
                )
                for object_id in ids
            }
            return {
                object_id: futures[object_id].result()
                for object_id in ids
            }

    origin = np.asarray(eye, dtype=float)
    directions = all_points - origin
    distances = np.linalg.norm(directions, axis=1)
    valid = distances > 1e-9
    visible = valid & ~blocked
    if forward is None:
        front_visible = visible
    else:
        forward_v = np.asarray(forward, dtype=float)
        norm = float(np.linalg.norm(forward_v))
        if norm > 1e-9:
            forward_v /= norm
        front_visible = visible & ((directions @ forward_v) >= 0.0)

    results: dict[str, dict[str, float] | None] = {}
    offset = 0
    for object_id, count in zip(ids, counts):
        item = slice(offset, offset + count)
        results[object_id] = {
            "vf": float(visible[item].sum()) / count,
            "front_vf": float(front_visible[item].sum()) / count,
            "blocked": float(blocked[item].sum()) / count,
        }
        offset += count
    return results


def directional_verdict_backing(
    points_by_id: dict[str, np.ndarray],
    eye: tuple[float, float, float],
    foreground_ids: Iterable[str],
    verdict_by_id: dict[str, str],
    min_gap: float = 0.03,
) -> dict[str, dict[str, float]]:
    """Fractions of a mesh's sightline points backed by transformed classes.

    For each 6-degree direction bin, the farthest point belonging to a mesh
    in ``verdict_by_id`` supplies both its range and verdict class.  A
    foreground point is backed when that transformed geometry continues at
    least ``min_gap`` farther along the same eye ray.  Distances are otherwise
    unbounded: this is a line-of-sight relationship, not a contact test.
    """
    transformed = [
        object_id for object_id in verdict_by_id
        if object_id in points_by_id and len(points_by_id[object_id])
    ]
    foreground = [
        object_id for object_id in foreground_ids
        if object_id in points_by_id and len(points_by_id[object_id])
    ]
    result = {object_id: {} for object_id in foreground}
    if not transformed or not foreground:
        return result

    class_names = sorted({verdict_by_id[object_id] for object_id in transformed})
    class_index = {name: index for index, name in enumerate(class_names)}
    chunks = [points_by_id[object_id] for object_id in transformed]
    classes = np.concatenate([
        np.full(len(points_by_id[object_id]), class_index[verdict_by_id[object_id]])
        for object_id in transformed
    ])
    all_points = np.concatenate(chunks)
    bins, radii = spatial_spherical_bins(all_points - np.asarray(eye, dtype=float))
    order = np.lexsort((radii, bins))
    sorted_bins = bins[order]
    is_last = np.r_[sorted_bins[1:] != sorted_bins[:-1], True]
    winners = order[is_last]
    n_bins = int(bins.max()) + 2
    far_range = np.full(n_bins, -np.inf)
    far_class = np.full(n_bins, -1, dtype=np.int64)
    far_range[bins[winners]] = radii[winners]
    far_class[bins[winners]] = classes[winners]

    eye_v = np.asarray(eye, dtype=float)
    for object_id in foreground:
        points = points_by_id[object_id]
        point_bins, point_ranges = spatial_spherical_bins(points - eye_v)
        in_field = point_bins < n_bins
        backed = np.zeros(len(points), dtype=bool)
        backed[in_field] = far_range[point_bins[in_field]] > point_ranges[in_field] + min_gap
        count = len(points)
        for name, index in class_index.items():
            matched = backed & in_field
            matched[in_field] &= far_class[point_bins[in_field]] == index
            hits = int(matched.sum())
            if hits:
                result[object_id][name] = float(hits) / count
    return result


def floor_height_from_shell(
    points_by_id: dict[str, np.ndarray],
    eye: tuple[float, float, float],
    forward: tuple[float, float, float],
    transparent_ids: set[str],
) -> float | None:
    """Cabin floor height read off the shell in the forward-down (footwell)
    sector. Straight-down rays end on the seat cushion, so they are excluded."""
    chunks = [pts for oid, pts in points_by_id.items() if oid not in transparent_ids]
    if not chunks:
        return None
    all_points = np.concatenate(chunks)
    vectors = all_points - np.array(eye)
    radii = np.maximum(np.linalg.norm(vectors, axis=1), 1e-9)
    elevation = np.arcsin(np.clip(vectors[:, 2] / radii, -1.0, 1.0))
    horizontal = vectors[:, :2] / np.maximum(np.linalg.norm(vectors[:, :2], axis=1), 1e-9)[:, None]
    forwardness = horizontal @ np.array(forward[:2])
    footwell = (
        (elevation < math.radians(-35.0))
        & (elevation > math.radians(-75.0))
        & (forwardness > 0.5)
    )
    if int(footwell.sum()) < 20:
        return None
    bins, seg_radii = spatial_spherical_bins(vectors[footwell])
    heights = all_points[footwell][:, 2]
    n_bins = int(bins.max()) + 2
    nearest_field = np.full(n_bins, np.inf)
    np.minimum.at(nearest_field, bins, seg_radii)
    nearest = seg_radii <= nearest_field[bins] + 0.02
    return float(np.percentile(heights[nearest], 20))


def glass_beyond_fractions(
    points_by_id: dict[str, np.ndarray],
    eye: tuple[float, float, float],
    glass_ids: set[str],
    check_ids: Iterable[str],
) -> dict[str, float]:
    """Fraction of each mesh's points beyond a large planar glass pane within
    the pane's angular footprint: outside the glasshouse (wipers beyond the
    windscreen, the truck bed beyond the rear window). Small panes (gauge
    lenses) are not cabin boundaries and are ignored."""
    result = {object_id: 0.0 for object_id in check_ids}
    eye_v = np.array(eye)
    step = math.radians(SPATIAL_BIN_DEG)
    n_azimuth = int(2 * math.pi / step) + 1
    panes: list[tuple[np.ndarray, np.ndarray, set]] = []
    for glass_id in glass_ids:
        points = points_by_id.get(glass_id)
        if points is None or len(points) < 10:
            continue
        if float(np.linalg.norm(np.ptp(points, axis=0))) < 0.75:
            continue  # instrument lens, not a window
        centroid = points.mean(axis=0)
        centered = points - centroid
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        normal = vt[2]
        rms = float(np.sqrt(((centered @ normal) ** 2).mean()))
        if rms > 0.05:
            continue  # not planar enough to bound anything
        if (centroid - eye_v) @ normal < 0:
            normal = -normal  # orient away from the eye
        pane_bins, _ = spatial_spherical_bins(points - eye_v)
        footprint = set()
        for b in set(pane_bins.tolist()):
            for d_el in (-2, -1, 0, 1, 2):
                for d_az in (-2, -1, 0, 1, 2):
                    footprint.add(b + d_el * n_azimuth + d_az)
        panes.append((centroid, normal, footprint))
    if not panes:
        return result
    for object_id in result:
        points = points_by_id.get(object_id)
        if points is None or len(points) == 0:
            continue
        point_bins, _ = spatial_spherical_bins(points - eye_v)
        beyond = np.zeros(len(points), dtype=bool)
        for centroid, normal, footprint in panes:
            in_footprint = np.fromiter(
                (b in footprint for b in point_bins.tolist()), bool, len(points)
            )
            distance = (points - centroid) @ normal
            beyond |= in_footprint & (distance > 0.05)
        result[object_id] = float(beyond.mean())
    return result


def cloud_symmetry_residual(points: np.ndarray, center_x: float) -> float:
    """Normalised Chamfer residual of the cloud against its own x-reflection.

    Small means reflecting the mesh across the centreline is a visual no-op;
    large means the mesh is one-sided. Normalised by the bbox diagonal so a
    4 cm button and a 2 m dashboard are judged on the same scale."""
    reflected = points.copy()
    reflected[:, 0] = 2 * center_x - reflected[:, 0]
    squared = ((reflected[:, None, :] - points[None, :, :]) ** 2).sum(axis=2)
    nearest = np.sqrt(squared.min(axis=1))
    diagonal = float(np.linalg.norm(np.ptp(points, axis=0)))
    return float(np.median(nearest)) / max(diagonal, 0.05)


def reflected_orphan_stats(
    points: np.ndarray,
    center_x: float,
    exact_tol: float = 1e-4,
    coarse_tol: float = 0.02,
) -> tuple[int, float]:
    """Count vertices missing an x-reflected twin at two tolerances.

    ``orphans`` uses a 0.1 mm default tolerance to absorb DAE float dust while
    retaining a zero-threshold decision: only a mesh whose every vertex has a
    reflected partner is a symmetry no-op.  ``coarse_fraction`` is diagnostic
    evidence for confidence grading, never the skip decision.

    A spatial hash keeps each pass linear for ordinary mesh density.  Actual
    Euclidean distances are checked inside neighbouring cells, so quantisation
    cannot turn a near cell into a false match.
    """
    cloud = np.asarray(points, dtype=float)
    if len(cloud) == 0:
        return 0, 0.0
    if exact_tol <= 0.0 or coarse_tol <= 0.0:
        raise ValueError("symmetry tolerance must be positive")

    gpu_stats = spatial_visibility_backend.gpu_reflected_orphan_stats(
        cloud,
        float(center_x),
        float(exact_tol),
        float(coarse_tol),
    )
    if gpu_stats is not None:
        return gpu_stats

    reflected = cloud.copy()
    reflected[:, 0] = 2.0 * float(center_x) - reflected[:, 0]

    def orphan_count(tol: float) -> int:
        cells: dict[tuple[int, int, int], list[np.ndarray]] = {}
        indices = np.floor(cloud / tol).astype(np.int64)
        for index, cell in enumerate(indices):
            cells.setdefault((int(cell[0]), int(cell[1]), int(cell[2])), []).append(cloud[index])
        tol2 = tol * tol
        count = 0
        for query in reflected:
            base = np.floor(query / tol).astype(np.int64)
            matched = False
            for dx in (-1, 0, 1):
                if matched:
                    break
                for dy in (-1, 0, 1):
                    if matched:
                        break
                    for dz in (-1, 0, 1):
                        candidates = cells.get(
                            (int(base[0] + dx), int(base[1] + dy), int(base[2] + dz))
                        )
                        if candidates is None:
                            continue
                        if any(float(np.dot(point - query, point - query)) <= tol2 for point in candidates):
                            matched = True
                            break
            if not matched:
                count += 1
        return count

    exact_orphans = orphan_count(float(exact_tol))
    coarse_orphans = orphan_count(float(coarse_tol))
    return exact_orphans, float(coarse_orphans) / len(cloud)


def mirror_pair_distance(points_a: np.ndarray, points_b: np.ndarray, center_x: float) -> float:
    """Symmetric Chamfer distance in metres between reflect(A) and B."""
    reflected = points_a.copy()
    reflected[:, 0] = 2 * center_x - reflected[:, 0]

    def chamfer(u: np.ndarray, v: np.ndarray) -> float:
        squared = ((u[:, None, :] - v[None, :, :]) ** 2).sum(axis=2)
        return float(np.median(np.sqrt(squared.min(axis=1))))

    return (chamfer(reflected, points_b) + chamfer(points_b, reflected)) / 2.0


def mirror_pair_residual(points_a: np.ndarray, points_b: np.ndarray, center_x: float) -> float:
    """Symmetric normalised Chamfer distance between reflect(A) and B: how
    well B is A's geometric twin across the centreline."""
    distance = mirror_pair_distance(points_a, points_b, center_x)
    diagonal = float(np.linalg.norm(np.ptp(np.concatenate([points_a, points_b]), axis=0)))
    return distance / max(diagonal, 0.05)


def principal_extent_sds(points: np.ndarray) -> np.ndarray:
    """Ascending standard deviations along the cloud's principal axes
    (thickness, width, length): the planarity/elongation fingerprint."""
    centered = points - points.mean(axis=0)
    covariance = centered.T @ centered / max(len(points) - 1, 1)
    return np.sqrt(np.maximum(np.linalg.eigvalsh(covariance), 0.0))


def material_uses_glass_blending(value: dict[str, object]) -> bool:
    """Whether a translucent material has evidence of active alpha blending.

    BeamNG materials sometimes carry ``translucent: true`` while explicitly
    disabling blending and supplying no opacity map (for example crawler cage
    metal and hubcaps).  Those are not transparent panes and must not define
    the classifier's glasshouse.
    """
    if not bool(value.get("translucent")):
        return False
    blend_op = str(value.get("translucentBlendOp") or "").strip().lower()
    active_blend = blend_op not in {"", "none"}
    opacity_map = any(
        isinstance(stage, dict) and bool(stage.get("opacityMap"))
        for stage in (value.get("Stages") or [])
    )
    return active_blend or opacity_map


def material_flags_for_context(context: VehicleContext) -> dict[str, dict[str, bool]]:
    """Material name -> {"emissive", "glass"} from every *.materials.json in
    the vehicle zip and the game's common zips.

    emissive = any stage carries an emissiveMap (a lit display image).
    glass    = translucent with active alpha/opacity evidence and NOT emissive
               (a window; translucent+emissive is a screen). Cached on the
               context; missing zips degrade to {}."""
    cached = getattr(context, "_material_flags", None)
    if cached is not None:
        return cached
    flags: dict[str, dict[str, bool]] = {}
    zips = [context.source_zip] + common_zip_candidates(context.source_zip)
    for zip_path in zips:
        try:
            zf = zipfile.ZipFile(zip_path)
        except Exception:
            continue
        with zf:
            for name in zf.namelist():
                if not name.replace("\\", "/").endswith(".materials.json"):
                    continue
                try:
                    data = parse_beamng_json(
                        zf.read(name).decode("utf-8", errors="replace"), label=name
                    )
                except Exception:
                    continue
                for key, value in data.items():
                    if not isinstance(value, dict):
                        continue
                    material_name = str(value.get("mapTo") or value.get("name") or key)
                    stages = value.get("Stages") or []
                    emissive = any(
                        isinstance(stage, dict) and "emissiveMap" in stage for stage in stages
                    )
                    glass = material_uses_glass_blending(value) and not emissive
                    flags[material_name.lower()] = {"emissive": emissive, "glass": glass}
    context._material_flags = flags
    return flags


# A beamNavigator controller row: ["beamNavigator", {"screenMaterialName": "@mat", ...}].
_NAV_SCREEN_MATERIAL_RE = re.compile(
    r'"beamNavigator"\s*,\s*\{[^{}]*?"screenMaterialName"\s*:\s*"@?([^"]+)"'
)
# A quoted material value inside a glowMap state ("on_intense":"@mat").
_GLOW_STATE_VALUE_RE = re.compile(r':\s*"@?([^"]+)"')
# A glowMap base-material key ("etk800_gauges_screen":{...}).
_GLOW_BASE_KEY_RE = re.compile(r'"([A-Za-z0-9_]+)"\s*:[\s,]*\{')


def nav_screen_materials_for_context(context: VehicleContext) -> frozenset[str]:
    """DAE material symbols a beamNavigator controller drives as a live screen
    (the sat-nav / infotainment display).

    Two authoring patterns both resolve here, so a mirrored screen mesh can be
    texture-flipped from the vehicle data alone with no manual flag:
      * direct  -- the controller's screenMaterialName IS the mesh material
        (etk800: "@etk800_screen" and the mesh binds etk800_screen).
      * via glowMap -- the mesh binds a base material the same part's glowMap
        swaps to the screenMaterialName when lit (sunburst2: the mesh binds
        sunburst2_display_nav, whose glowMap on_intense is sunburst2_naviscreen_on
        == the screenMaterialName). The base is what the DAE actually binds, so
        it must be resolved back or the screen is never recognised.

    Only glow bases that reach THIS part's screen material count, so a dash
    part's unrelated gauge-cluster glowMap (etk800_gauges_screen) is not swept
    in. Names are lowercased to match mesh_material_symbols. Cached on the
    context; a vehicle with no navigator yields an empty set."""
    cached = getattr(context, "_nav_screen_materials", None)
    if cached is not None:
        return cached
    nav: set[str] = set()
    for part_body, _filename in context.part_body_index.values():
        controller = transform_helpers.extract_named_array(part_body, "controller")
        if not controller or "beamNavigator" not in controller:
            continue
        screen_mats = {
            match.group(1).strip().lower()
            for match in _NAV_SCREEN_MATERIAL_RE.finditer(controller)
        }
        if not screen_mats:
            continue
        nav |= screen_mats
        glow = transform_helpers.extract_keyed_object(part_body, "glowMap")
        if not glow:
            continue
        # Scan the glowMap body (past its own "glowMap":{ wrapper) so the outer
        # key is not mistaken for a base material.
        outer = glow.find("{")
        inner = glow[outer + 1:] if outer >= 0 else glow
        for key_match in _GLOW_BASE_KEY_RE.finditer(inner):
            base = key_match.group(1).strip().lower()
            brace = inner.find("{", key_match.start(), key_match.end())
            try:
                end = transform_helpers.find_matching(inner, brace, "{", "}")
            except ValueError:
                continue
            block = inner[brace:end]
            if any(
                value.group(1).strip().lower() in screen_mats
                for value in _GLOW_STATE_VALUE_RE.finditer(block)
            ):
                nav.add(base)
    result = frozenset(nav)
    context._nav_screen_materials = result
    return result


def nav_screen_mesh_scope(context: VehicleContext) -> dict[str, frozenset[str]]:
    """Mesh id -> the nav-screen material symbols it binds.

    A mesh appears only when it binds at least one beamNavigator screen
    material. The value is the subset of that mesh's material symbols to
    texture-flip; on a dedicated screen mesh (etk800_screen) it is every
    symbol, on a shared mesh (sunburst2_nav, which also carries the gauge
    cluster) it is only the screen island. Cached on the context."""
    cached = getattr(context, "_nav_screen_mesh_scope", None)
    if cached is not None:
        return cached
    nav_materials = nav_screen_materials_for_context(context)
    scope: dict[str, frozenset[str]] = {}
    if nav_materials:
        for object_id, symbols in mesh_material_symbols(context).items():
            matched = frozenset(sym for sym in symbols if sym in nav_materials)
            if matched:
                scope[object_id] = matched
    context._nav_screen_mesh_scope = scope
    return scope


def mesh_material_symbols(context: VehicleContext) -> dict[str, tuple[str, ...]]:
    """Mesh id -> material symbols bound in the DAE.

    New contexts carry them in preview_by_id (preview_data_from_tree); older
    cached contexts fall back to a one-off parse of the vehicle's own DAEs.
    Shared-library meshes without symbols simply get no material evidence."""
    cached = getattr(context, "_material_symbols", None)
    if cached is not None:
        return cached
    symbols: dict[str, tuple[str, ...]] = {}
    missing = False
    for object_id, item in context.preview_by_id.items():
        materials = item.get("materials")
        if materials is None:
            missing = True
        elif materials:
            symbols[object_id] = tuple(materials)
    if missing and not symbols:
        try:
            with zipfile.ZipFile(context.source_zip) as zf:
                for dae_path in context.dae_paths:
                    try:
                        tree = ET.parse(io.BytesIO(zf.read(dae_path)))
                    except Exception:
                        continue
                    for node in tree.getroot().findall(".//c:node", NS):
                        node_symbols = {
                            re.sub(r"-material$", "", symbol).lower()
                            for inst in node.findall(".//c:instance_material", NS)
                            for symbol in (inst.get("symbol") or inst.get("target", "").lstrip("#"),)
                            if symbol
                        }
                        if node_symbols:
                            for alias in dae_node_aliases(node):
                                symbols.setdefault(alias, tuple(sorted(node_symbols)))
        except Exception:
            pass
    context._material_symbols = symbols
    return symbols


def inert_material_alias_symbols(context: VehicleContext) -> frozenset[str]:
    """Material symbols whose definitions contain no functional properties.

    Some geometric twins bind separate L/R aliases even though both aliases
    are empty placeholders. Those aliases do not carry texture, blending,
    emissive, or other state and are safe for structural pairing. Missing or
    meaningful definitions remain conservative and are not returned.
    """
    cached = getattr(context, "_inert_material_alias_symbols", None)
    if cached is not None:
        return cached

    ignored_metadata = {
        "class", "mapTo", "name", "persistentId", "version", "Stages"
    }

    def has_value(value: object) -> bool:
        if value is None or value is False:
            return False
        if isinstance(value, (str, bytes, tuple, list, dict)):
            return bool(value)
        return True

    def is_inert(definition: dict[str, object]) -> bool:
        if any(
            has_value(value)
            for key, value in definition.items()
            if key not in ignored_metadata
        ):
            return False
        for stage in definition.get("Stages") or []:
            if isinstance(stage, dict) and any(
                has_value(value) for value in stage.values()
            ):
                return False
        return True

    states: dict[str, bool] = {}
    zips = [context.source_zip] + common_zip_candidates(context.source_zip)
    for zip_path in zips:
        try:
            zf = zipfile.ZipFile(zip_path)
        except Exception:
            continue
        with zf:
            for name in zf.namelist():
                if not name.replace("\\", "/").endswith(".materials.json"):
                    continue
                try:
                    data = parse_beamng_json(
                        zf.read(name).decode("utf-8", errors="replace"),
                        label=name,
                    )
                except Exception:
                    continue
                for key, value in data.items():
                    if not isinstance(value, dict):
                        continue
                    symbol = str(
                        value.get("mapTo") or value.get("name") or key
                    ).lower()
                    states[symbol] = states.get(symbol, True) and is_inert(value)

    result = frozenset(symbol for symbol, inert in states.items() if inert)
    context._inert_material_alias_symbols = result
    return result

__all__ = ['SPATIAL_BIN_DEG', 'DriverFrame', 'iter_named_array_texts', 'parse_internal_camera_positions', '_camera_bearing_parts', 'camera_positions_for_config', '_driver_frame_core', '_assemble_driver_frame', 'driver_frame_for_context', 'spatial_spherical_bins', 'visibility_scan', 'surface_visibility_stats', 'surface_visibility_stats_batch', 'directional_verdict_backing', 'floor_height_from_shell', 'glass_beyond_fractions', 'cloud_symmetry_residual', 'reflected_orphan_stats', 'mirror_pair_distance', 'mirror_pair_residual', 'principal_extent_sds', 'material_uses_glass_blending', 'material_flags_for_context', '_NAV_SCREEN_MATERIAL_RE', '_GLOW_STATE_VALUE_RE', '_GLOW_BASE_KEY_RE', 'nav_screen_materials_for_context', 'nav_screen_mesh_scope', 'mesh_material_symbols', 'inert_material_alias_symbols']
