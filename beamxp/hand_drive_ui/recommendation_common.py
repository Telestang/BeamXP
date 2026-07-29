from __future__ import annotations

from .shared import core

# Thresholds are metres or fractions, tuned against the hand-verified
# etk800 / pickup / sunburst2 baselines.
SPATIAL_PAIR_DISTANCE = 0.020
SPATIAL_PAIR_MIN_OFFSET = 0.05
SPATIAL_REACH_LIMIT = 1.35
SPATIAL_CONTACT_LIMIT = 0.0201
SPATIAL_VISIBLE_FRACTION = 0.28
SPATIAL_PASSENGER_VISIBLE_FRACTION = 0.08


def _driver_control_outboard_limit(below_eye: float) -> float:
    """Outboard reach of the control volume at a given height.

    Dashboard controls cluster close to the steering column, while foot
    controls legitimately spread farther towards the driver's door.  Blend
    between those two widths so the volume follows the sloping control area
    instead of treating the whole dashboard/footwell as a rectangular slab.
    """
    footwell_fraction = min(max((below_eye - 0.45) / 0.25, 0.0), 1.0)
    return 0.24 + 0.09 * footwell_fraction


def _is_enclosed_candidate(
    stats: dict[str, float],
    out80: float,
    half_width: float,
) -> bool:
    """Whether shell evidence places a mesh inside the occupied cabin."""
    ordinarily_inboard = out80 <= half_width - 0.02
    lined_at_boundary = (
        stats["front_vf"] >= 0.25
        and stats["front_backed"] >= 0.75
        and stats["front_lined"] >= 0.75
        and out80 <= half_width
        and stats["front_depth"] <= 0.35
    )
    return (
        stats["front_vf"] >= 0.08
        and stats["front_backed"] >= 0.45
        and stats["front_depth"] <= 0.45
        and (ordinarily_inboard or lined_at_boundary)
    )


def _unscoped_contact_is_cabin_furniture(
    points: object,
    frame: core.DriverFrame,
) -> bool:
    """Bound hidden contact inheritance to the occupant-sized cabin volume."""
    import numpy as np

    if points is None:
        return False
    cloud = np.asarray(points, dtype=float)
    if len(cloud) < 4:
        return False
    centroid = cloud.mean(axis=0)
    z70 = float(np.percentile(cloud[:, 2], 70))
    driver_eye = np.asarray(frame.eye, dtype=float)
    passenger_eye = driver_eye.copy()
    passenger_eye[0] = 2.0 * frame.center_x - driver_eye[0]
    driver_forward = np.asarray(frame.forward, dtype=float)
    passenger_forward = driver_forward.copy()
    passenger_forward[0] *= -1.0

    def inside_from(eye: np.ndarray, forward: np.ndarray) -> bool:
        ahead = float((centroid[:2] - eye[:2]) @ forward[:2])
        range80 = float(np.percentile(np.linalg.norm(cloud - eye, axis=1), 80))
        return (
            -0.60 <= ahead <= 1.00
            and eye[2] - 0.70 <= z70 <= eye[2] + 0.35
            and range80 <= 1.60
        )

    return inside_from(driver_eye, driver_forward) or inside_from(
        passenger_eye, passenger_forward
    )


def _spatial_entries_for_trim(
    context: core.VehicleContext,
    trim: str | None,
    available: set[str],
) -> tuple[list[str], dict[str, object]]:
    """Meshes present in one trim with their per-trim point clouds."""
    if trim is None:
        present = sorted(available)
        entries = context.preview_by_id
        resolved = {}
    else:
        present = sorted(core.used_meshes_for_config(context, trim) & available)
        entries = core.preview_entries_for_config(context, trim)
        resolved = core.resolved_mesh_positions_for_config(context, trim)
    import numpy as np

    arrays: dict[str, object] = {}
    for object_id in present:
        placement = resolved.get(object_id)
        if placement is not None and (
            len(placement.matrices) > 1
            or object_id in context.variant_dependent_meshes
        ):
            rebuilt = core.vertex_cloud_for_resolved_placement(context, object_id, placement)
            if rebuilt is not None:
                arrays[object_id] = rebuilt
                continue
        item = entries.get(object_id) or context.preview_by_id.get(object_id)
        if item is None:
            continue
        arrays[object_id] = np.array(item["sample_points"], dtype=float)
    return [o for o in present if o in arrays], arrays


def _spatial_surfaces_for_trim(
    context: core.VehicleContext,
    trim: str | None,
    present: list[str],
    entries_np: dict[str, object],
) -> dict[str, object]:
    """Filled DAE surfaces at the same per-trim placement as point clouds."""
    import numpy as np

    base = core.full_surface_triangles_for_ids(context, present)
    resolved = (
        core.resolved_mesh_positions_for_config(context, trim)
        if trim is not None else {}
    )
    surfaces: dict[str, object] = {}
    for object_id in present:
        placement = resolved.get(object_id)
        if placement is not None and (
            len(placement.matrices) > 1
            or object_id in context.variant_dependent_meshes
        ):
            rebuilt = core.surface_triangles_for_resolved_placement(
                context, object_id, placement
            )
            if rebuilt is not None and len(rebuilt):
                surfaces[object_id] = rebuilt
                continue
        triangles = base.get(object_id)
        points = entries_np.get(object_id)
        preview = context.preview_by_id.get(object_id)
        if triangles is None or len(triangles) == 0 or points is None or preview is None:
            continue
        placed_center = (np.min(points, axis=0) + np.max(points, axis=0)) / 2.0
        preview_center = np.asarray(preview.get("center"), dtype=float)
        if preview_center.shape == (3,):
            delta = placed_center - preview_center
            if float(np.max(np.abs(delta))) > 1e-9:
                triangles = triangles + delta
        surfaces[object_id] = triangles
    return surfaces


def _mesh_symmetry(
    context: core.VehicleContext,
    object_id: str,
    points: object,
    center_x: float,
) -> tuple[int, float]:
    """Full-vertex symmetry evidence at this trim's x placement."""
    import numpy as np

    full = core.full_vertex_clouds_for_ids(context, (object_id,)).get(object_id)
    if full is None or len(full) == 0:
        full = np.asarray(points, dtype=float)
    placed = np.asarray(points, dtype=float)
    preview = context.preview_by_id.get(object_id, {})
    base_center = preview.get("center")
    if len(placed):
        placed_center_x = float((placed[:, 0].min() + placed[:, 0].max()) / 2.0)
    else:
        placed_center_x = float(center_x)
    shift_x = placed_center_x - float(base_center[0]) if base_center is not None else 0.0
    key = (object_id, round(shift_x, 9), round(float(center_x), 9))
    cache = getattr(context, "_mesh_symmetry_cache", None)
    if cache is None:
        cache = {}
        context._mesh_symmetry_cache = cache
    if key not in cache:
        shifted = np.asarray(full, dtype=float).copy()
        shifted[:, 0] += shift_x
        cache[key] = core.reflected_orphan_stats(shifted, center_x)
    return cache[key]
