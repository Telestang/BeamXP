"""Synthetic vehicle contexts shared by the recommender and geometry tests.

`make_context` builds a context from point clouds -- what the geometry
helpers in `spatial_analysis.py` consume. `mesh_context` builds one from
mesh centres alone, which is all the name-and-placement recommender reads.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from beamxp import hand_drive_core as core

EYE = (0.40, 0.29, 1.20)
CAMERA_JBEAM = (
    '{"fixture_body": {"camerasInternal":[\n'
    '    ["type", "x", "y", "z", "fov", "id1:"],\n'
    '    ["dash", 0.40, 0.29, 1.20, 55, "f1"],\n'
    "]}}"
)


def box_cloud(center, size, n=200, seed=0):
    rng = np.random.default_rng(seed + int(abs(center[0]) * 1000) + n)
    half = np.array(size, dtype=float) / 2.0
    return np.array(center, dtype=float) + rng.uniform(-1.0, 1.0, size=(n, 3)) * half


def mirror_x(points):
    out = np.array(points, dtype=float).copy()
    out[:, 0] = -out[:, 0]
    return out


def sym_cloud(center, size, n=200, seed=0):
    """Exactly centreline-symmetric cloud (how symmetric meshes are modelled:
    mirrored vertices, not merely a symmetric envelope)."""
    half = box_cloud(center, size, n=max(n // 2, 20), seed=seed)
    return np.concatenate([half, mirror_x(half)])


def base_cabin() -> dict[str, np.ndarray]:
    """A minimal but occlusion-correct cabin around the eye at EYE.

    Door cards sit inboard of the door skins (the lining layers the spatial
    helpers depend on); the firewall backs the dash; seats flank the eye."""
    card_fl = box_cloud((0.73, -0.20, 0.68), (0.05, 1.00, 0.75), n=220, seed=1)
    skin_fl = box_cloud((0.82, -0.20, 0.80), (0.06, 1.20, 1.00), n=220, seed=2)
    # a seat is an L-shell (cushion + backrest), not a solid block: a filled
    # box would wrap phantom points around the eye and occlude the footwell
    seat_fl = np.concatenate([
        box_cloud((0.40, 0.15, 0.45), (0.50, 0.55, 0.20), n=120, seed=3),
        box_cloud((0.40, 0.44, 0.80), (0.50, 0.16, 0.85), n=120, seed=35),
    ])
    return {
        "veh_dash": sym_cloud((0.0, -0.60, 0.95), (1.56, 0.30, 0.50), n=260, seed=4),
        "veh_firewall": sym_cloud((0.0, -0.95, 0.70), (1.50, 0.06, 1.00), n=200, seed=5),
        "veh_floor": sym_cloud((0.0, 0.10, 0.20), (1.50, 2.00, 0.05), n=240, seed=6),
        "veh_headliner": sym_cloud((0.0, 0.50, 1.36), (1.30, 1.80, 0.04), n=200, seed=7),
        "veh_roof": sym_cloud((0.0, 0.50, 1.45), (1.40, 1.90, 0.04), n=200, seed=42),
        "veh_card_FL": card_fl,
        "veh_card_FR": mirror_x(card_fl),
        "veh_skin_FL": skin_fl,
        "veh_skin_FR": mirror_x(skin_fl),
        "veh_seat_FL": seat_fl,
        "veh_seat_FR": mirror_x(seat_fl),
        # named so the sanctioned steering score anchors it (etk800-style)
        "veh_steer": box_cloud((0.40, -0.30, 0.95), (0.36, 0.06, 0.36), n=200, seed=8),
    }


def make_context(
    meshes: dict[str, np.ndarray],
    *,
    camera: bool = True,
    trims: dict[str, list[str]] | None = None,
    materials: dict[str, tuple[str, ...]] | None = None,
    material_flags: dict[str, dict[str, bool]] | None = None,
) -> core.VehicleContext:
    objects = {}
    preview = {}
    for object_id, points in meshes.items():
        pts = np.asarray(points, dtype=float)
        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        center = tuple(float(v) for v in (lo + hi) / 2.0)
        objects[object_id] = core.DaeObject(
            id=object_id,
            name=object_id,
            dae_path="vehicle.dae",
            x=center[0],
            y=center[1],
            z=center[2],
            geometry_ids=(),
        )
        preview[object_id] = {
            "bounds": (tuple(float(v) for v in lo), tuple(float(v) for v in hi)),
            "center": center,
            "sample_points": [tuple(float(v) for v in p) for p in pts],
            "geometry_ids": (),
            "materials": tuple((materials or {}).get(object_id, ())),
        }
    variants = {}
    roles_cache = {}
    resolved_cache = {}
    for trim_name, ids in (trims or {}).items():
        variants[trim_name] = core.VariantInfo(
            name=trim_name, pc_path="", info_path=None, display_name=trim_name
        )
        roles_cache[trim_name] = (set(), set(), set(ids))
        resolved_cache[trim_name] = {}
    context = core.VehicleContext(
        source_zip=Path("test.zip"),
        vehicle_id="veh",
        vehicle_path="vehicles/veh",
        dae_paths=[],
        variants=variants,
        objects=objects,
        preview_by_id=preview,
        jbeam_texts={"vehicles/veh/veh.jbeam": CAMERA_JBEAM} if camera else {},
        node_positions={},
        project_dir=Path("project"),
    )
    context.mesh_roles_cache.update(roles_cache)
    context.resolved_positions_cache.update(resolved_cache)
    context._material_flags = dict(material_flags or {})
    return context


def mesh_context(
    centers: dict[str, tuple[float, float, float] | None],
    **kwargs,
) -> core.VehicleContext:
    """A context holding just names and placements.

    A centre of None models a mesh the preview never cached -- the case
    where a name is the only evidence the recommender has.
    """
    # An exact box about the centre, so the bounds the recommender reads back
    # are the centre the fixture asked for.
    corners = np.array([
        (sx, sy, sz)
        for sx in (-0.05, 0.05)
        for sy in (-0.05, 0.05)
        for sz in (-0.05, 0.05)
    ])
    meshes = {
        object_id: np.asarray(center or (0.0, 0.0, 0.0), dtype=float) + corners
        for object_id, center in centers.items()
    }
    context = make_context(meshes, **kwargs)
    for object_id, center in centers.items():
        if center is None:
            context.preview_by_id.pop(object_id, None)
            context.objects[object_id] = core.DaeObject(
                id=object_id,
                name=object_id,
                dae_path="vehicle.dae",
                x=0.0,
                y=0.0,
                z=0.0,
                geometry_ids=(),
            )
    return context
