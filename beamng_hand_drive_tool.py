from __future__ import annotations

import argparse
import json
import queue
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox
import tkinter as tk
from tkinter import ttk

import beamng_hand_drive_core as core
import plate_generator
from model_preview import ModelPreview
from plate_editor import PlateEditorDialog
from plate_library import PlateLibraryDialog

try:  # GPU mesh preview; the box viewer remains the fallback
    import mesh_preview
except Exception:
    mesh_preview = None


THIS_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
RESOURCE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
BLENDER_PREVIEW_SCRIPT = RESOURCE_DIR / "blender_preview_backend.py"
APP_ICON_NAME = "BeamXP_icon.ico"
BLENDER_CANDIDATES = (
    Path(r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"),
    Path(r"C:\Program Files\Blender Foundation\Blender 4.3\blender.exe"),
    Path(r"C:\Program Files\Blender Foundation\Blender 4.4\blender.exe"),
    Path(r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"),
)


MODEL_HISTORY_LIMIT = 12


def fmt_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def position_labels(
    position: tuple[float, float, float],
    variant_dependent: bool,
) -> tuple[str, str, str]:
    """x/y/z cells, marked when the part sits elsewhere on other trims.

    Without the marker the columns silently change meaning between rows: most
    parts have one position, but a part declared by mutually exclusive parts
    only has the one belonging to the trim on screen. All three cells carry
    the mark because it is the whole coordinate that is trim-specific --
    flagging x alone would imply x is the axis that moves, and usually it is
    not (the D-Series gooseneck hitch shifts along y)."""
    suffix = " *" if variant_dependent else ""
    return tuple(f"{fmt_float(value)}{suffix}" for value in position)


def yn_label(value: object) -> str:
    return "Y" if bool(value) else "N"


def mode_label(mode: str) -> str:
    return {
        core.MODE_SKIP: "Skip",
        core.MODE_MIRROR: "Mirror Aesthetic",
        core.MODE_MIRROR_STRUCTURAL: "Mirror Structural",
        core.MODE_TRANSLATE: "Translate",
    }.get(mode, "Skip")


MODE_CYCLE_VALUES = [core.MODE_SKIP, core.MODE_MIRROR, core.MODE_MIRROR_STRUCTURAL, core.MODE_TRANSLATE]
MODE_VALUES_BY_LABEL = {mode_label(mode): mode for mode in MODE_CYCLE_VALUES}
MODE_HOTKEYS = {
    "q": core.MODE_SKIP,
    "w": core.MODE_MIRROR,
    "e": core.MODE_MIRROR_STRUCTURAL,
    "r": core.MODE_TRANSLATE,
}

BUILD_LABELS = {
    core.BUILD_OFF: "Off",
    core.BUILD_CONVERTED: "Converted",
    core.BUILD_ORIGINAL: "Plates Only",
    core.BUILD_BOTH: "Both",
}

# How long (in milliseconds) a part may sit on Mirror Structural before the
# source-part prompt commits it. Tweak this value to change the timeout.
STRUCTURAL_PROMPT_DELAY_MS = 300


# ---------------------------------------------------------------------------
# Recommend Modes: eye-anchored spatial classifier
#
# Modes are decided from the vehicle's 3D geometry relative to the driver's
# eye (core.DriverFrame), not from part names. Per trim, a nearest-surface
# shell swept from the eye scopes interior CANDIDATES; corroborating evidence
# (backing/lining layers, the cabin envelope, glass planes) accepts or vetoes
# them; self-symmetry decides skip-vs-act; twin geometry decides structural
# pairs against the meshes actually present in that trim. The single
# sanctioned name usage is the steering-column hint at the resolution floor
# (top vs body are centimetres apart and slide with column length).
#
# Thresholds are metres or fractions, tuned against the hand-verified
# etk800 / pickup / sunburst2 baselines.

SPATIAL_PAIR_RESIDUAL = 0.10         # twin acceptance (normalised Chamfer)
SPATIAL_REACH_LIMIT = 1.35           # ergonomic: controls start within reach


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


def _classify_meshes_for_trim(
    context: core.VehicleContext,
    frame: core.DriverFrame,
    present: list[str],
    entries_np: dict[str, object],
    want: list[str],
    forced: frozenset[str] = frozenset(),
    hard_vetoed: set[str] | None = None,
    scoped: set[str] | None = None,
    surface_np: dict[str, object] | None = None,
) -> tuple[dict[str, tuple[str, str, str, dict]], set[str]]:
    """Intrinsic class per mesh from one trim's geometry.

    Returns (verdicts, vetoed): verdict is (class, reason, confidence, extra)
    with class in {"translate", "mirror", "pairable", "functional_skip",
    "none"}; vetoed lists meshes positively identified as exterior surfaces
    (they must never be offered as pairing twins)."""
    import numpy as np

    eye = np.array(frame.eye)
    f = np.array(frame.forward)
    cx0 = frame.center_x
    wheel = np.array(frame.wheel_center) if frame.wheel_center is not None else None

    material_symbols = core.mesh_material_symbols(context)
    material_flags = core.material_flags_for_context(context)

    def mesh_flag(object_id: str, flag: str, require_all: bool) -> bool:
        symbols = material_symbols.get(object_id)
        if not symbols:
            return False
        values = [bool(material_flags.get(symbol, {}).get(flag)) for symbol in symbols]
        return all(values) if require_all else any(values)

    glass_ids = {o for o in present if mesh_flag(o, "glass", require_all=True)}
    transparent = {
        o for o in present
        if context.objects.get(o) is not None
        and core.steering_ref_score(o, context.objects[o]) >= 15
    }
    driver_seat_ids: set[str] = set()
    # The eye camera sits inside the driver's seat volume.  Treat whichever
    # furniture actually surrounds that eye as transparent so its cushion and
    # backrest cannot hide rails/bases directly below it.  Geometry, not a
    # seat token, identifies the host; this also handles benches and mod seats.
    for object_id in present:
        seat_points = entries_np.get(object_id)
        if seat_points is None or len(seat_points) < 4:
            continue
        if float(np.linalg.norm(np.ptp(seat_points, axis=0))) < 0.50:
            continue
        under_eye = (
            (np.abs(seat_points[:, 0] - eye[0]) < 0.25)
            & (np.abs(seat_points[:, 1] - eye[1]) < 0.35)
            & (seat_points[:, 2] > eye[2] - 0.75)
            & (seat_points[:, 2] < eye[2] - 0.10)
        )
        if float(under_eye.mean()) >= 0.20:
            transparent.add(object_id)
            driver_seat_ids.add(object_id)

    # Rails/bases sit below the seat cushion and need conversion with it even
    # when the carpet or floor hides most of their eye rays.  This channel is
    # deliberately compact and directly under the detected driver seat, so a
    # longitudinal shaft or other underbody assembly cannot qualify.
    under_seat_candidates: set[str] = set()
    if driver_seat_ids:
        for object_id in present:
            points = entries_np.get(object_id)
            if points is None or len(points) < 4:
                continue
            diagonal = float(np.linalg.norm(np.ptp(points, axis=0)))
            if not 0.12 <= diagonal <= 0.70:
                continue
            centroid = points.mean(axis=0)
            if float((centroid[:2] - eye[:2]) @ f[:2]) <= 0.0:
                continue
            under_seat = (
                (np.abs(points[:, 0] - eye[0]) < 0.32)
                & (np.abs(points[:, 1] - eye[1]) < 0.42)
                & (points[:, 2] > eye[2] - 1.00)
                & (points[:, 2] < eye[2] - 0.45)
            )
            if float(under_seat.mean()) >= 0.20:
                under_seat_candidates.add(object_id)

    def compiled_surface(excluded: set[str]) -> object | None:
        chunks = [
            np.asarray(triangles, dtype=float).reshape((-1, 3, 3))
            for object_id, triangles in (surface_np or {}).items()
            if object_id not in excluded and len(triangles)
        ]
        return np.concatenate(chunks) if chunks else None

    # Concatenating once turns hundreds of tiny per-object NumPy kernels per
    # ray test into large, bounded batches.  The extra scene excludes glass
    # and is consulted only by the narrow exterior-fitment channel.
    surface_scene = compiled_surface(transparent)
    surface_scene_no_glass = compiled_surface(transparent | glass_ids)
    glass_chunks = [
        np.asarray(surface_np[object_id], dtype=float).reshape((-1, 3, 3))
        for object_id in glass_ids
        if surface_np and object_id in surface_np and len(surface_np[object_id])
    ]
    surface_scene_glass = np.concatenate(glass_chunks) if glass_chunks else None

    scan = core.visibility_scan(
        {o: entries_np[o] for o in present}, frame.eye, transparent, frame.forward
    )
    scan_no_glass = core.visibility_scan(
        {o: entries_np[o] for o in present if o not in glass_ids},
        frame.eye,
        transparent,
        frame.forward,
    )
    beyond = core.glass_beyond_fractions(entries_np, frame.eye, glass_ids, present)

    # cabin envelope from the lining: well-seen surfaces with structure behind
    lining = [
        o for o in present
        if scan[o]["vf"] >= 0.30 and scan[o]["backed"] >= 0.35 and scan[o]["n"] >= 40
        and o not in glass_ids and beyond.get(o, 0.0) < 0.30
        and float(entries_np[o].mean(axis=0)[2]) <= frame.eye[2] + 0.25
    ]
    if lining:
        lining_points = np.concatenate([entries_np[o] for o in lining])
        # half-width from closely-lined walls (card-over-skin layers); a mid
        # percentile keeps one exposed sheet in a gutted trim from dragging
        # the envelope out to the skin
        reaches = {
            o: float(np.percentile(np.abs(entries_np[o][:, 0] - cx0), 97)) for o in lining
        }
        wall_reaches = [v for o, v in reaches.items() if scan[o]["lined"] >= 0.40 and v >= 0.45]
        if wall_reaches:
            # p60: robust against the whole body shell or an exposed sheet
            # joining the wall list and dragging the envelope outward
            half_width = float(np.percentile(wall_reaches, 60))
        else:
            side_reaches = [v for v in reaches.values() if v >= 0.45] or list(reaches.values())
            half_width = float(np.percentile(side_reaches, 60))
        ceiling_z = float(np.percentile(lining_points[:, 2], 97))
        floor_z = float(np.percentile(lining_points[:, 2], 3))
        upper = lining_points[lining_points[:, 2] > frame.eye[2] - 0.75]
        if len(upper) >= 100:
            y_front = float(np.percentile(upper[:, 1], 2))
            y_rear = float(np.percentile(upper[:, 1], 98))
        else:
            y_front = float(np.percentile(lining_points[:, 1], 2))
            y_rear = float(np.percentile(lining_points[:, 1], 98))
    else:
        half_width, ceiling_z = 0.85, frame.eye[2] + 0.25
        floor_z = frame.eye[2] - 1.35
        y_front, y_rear = frame.eye[1] - 2.2, frame.eye[1] + 1.6
    shell_floor = core.floor_height_from_shell(entries_np, frame.eye, frame.forward, transparent)
    if shell_floor is not None:
        floor_z = max(floor_z, shell_floor)

    wheel_ahead = float((wheel - eye)[:2] @ f[:2]) if wheel is not None else 0.6
    wheel_dist = float(np.linalg.norm(wheel - eye)) if wheel is not None else 0.8
    wheel_x = float(wheel[0]) if wheel is not None else frame.eye[0]

    # Exact rays for different candidate meshes are independent.  Preserve
    # the stateful verdict/pair ordering below, but compute this expensive
    # broad-phase superset concurrently.  Four workers also composes cleanly
    # with the validator's three vehicle processes on a 12-thread machine.
    exact_by_id: dict[str, dict[str, float] | None] = {}
    exact_glass_by_id: dict[str, dict[str, float] | None] = {}
    if surface_scene is not None:
        exact_ids = []
        for object_id in want:
            if object_id in glass_ids:
                continue
            points = entries_np.get(object_id)
            if points is None or len(points) < 4:
                continue
            stats = scan[object_id]
            stats_ng = scan_no_glass.get(object_id, stats)
            centroid = points.mean(axis=0)
            extents = np.ptp(points, axis=0)
            diagonal = float(np.linalg.norm(extents))
            ahead = float((centroid[:2] - eye[:2]) @ f[:2])
            lat_signed = float(frame.side * (centroid[0] - wheel_x))
            below = frame.eye[2] - float(centroid[2])
            out80 = float(np.percentile(np.abs(points[:, 0] - cx0), 80))
            wall_lateral = (
                extents[0] < 0.22 and extents[1] > 0.45 and extents[2] > 0.45
            )
            in_cone = (
                0.20 <= ahead <= wheel_ahead + 1.0
                and -0.22 <= lat_signed <= 0.33
                and -0.10 <= below <= 1.35
                and not wall_lateral
            )
            enclosed = (
                stats["vf"] >= 0.08
                and stats["backed"] >= 0.45
                and out80 <= half_width - 0.02
                and stats["depth"] <= 0.45
            )
            fitment = (
                stats_ng["vf"] >= 0.12
                and diagonal <= 0.60
                and 0.15 <= ahead <= 1.5
                and abs(float(centroid[2]) - frame.eye[2]) <= 0.7
                and abs(float(centroid[0]) - cx0) >= half_width - 0.06
            )
            buried = stats["backed"] >= 0.75 and stats["depth"] <= 0.35
            cone = (
                in_cone
                and stats["depth"] <= 0.75
                and stats["min_r"] <= SPATIAL_REACH_LIMIT
                and (float(centroid[2]) >= floor_z - 0.10 or buried)
                and (
                    stats["vf"] >= 0.45
                    or stats["min_r"] <= wheel_dist + 0.45
                    or buried
                    or enclosed
                )
            )
            might_enter = (
                stats["front_vf"] >= 0.28
                or enclosed
                or fitment
                or cone
                or object_id in under_seat_candidates
                or object_id in forced
            )
            if might_enter:
                exact_ids.append(object_id)
        if exact_ids:
            with ThreadPoolExecutor(
                max_workers=min(4, len(exact_ids)),
                thread_name_prefix="spatial-rays",
            ) as executor:
                futures = {
                    object_id: executor.submit(
                        core.surface_visibility_stats,
                        entries_np[object_id],
                        frame.eye,
                        surface_scene,
                        set(),
                        frame.forward,
                    )
                    for object_id in exact_ids
                }
                glass_futures = {
                    object_id: executor.submit(
                        core.surface_visibility_stats,
                        entries_np[object_id],
                        frame.eye,
                        surface_scene_glass,
                        set(),
                        frame.forward,
                    )
                    for object_id in exact_ids
                    if surface_scene_glass is not None
                    and beyond.get(object_id, 0.0) >= 0.40
                }
                exact_by_id = {
                    object_id: future.result()
                    for object_id, future in futures.items()
                }
                exact_glass_by_id = {
                    object_id: future.result()
                    for object_id, future in glass_futures.items()
                }

    verdicts: dict[str, tuple[str, str, str, dict]] = {}
    vetoed: set[str] = set()
    for object_id in want:
        points = entries_np.get(object_id)
        obj = context.objects.get(object_id)
        if points is None or obj is None or len(points) < 4:
            continue
        centroid = points.mean(axis=0)
        stats = scan[object_id]
        near_eye = float(np.linalg.norm(centroid - eye)) <= 1.25 and centroid[2] >= frame.eye[2] - 0.8
        if (core.is_default_steering_ref(object_id, obj) or object_id == frame.wheel_id) and near_eye:
            # the wheel anchor's mesh may span the whole steering shaft, so it
            # is exempt from every other test
            verdicts[object_id] = ("translate", "steering wheel", "high", {})
            continue

        extents = np.ptp(points, axis=0)
        diagonal = float(np.linalg.norm(extents))
        ahead = float((centroid[:2] - eye[:2]) @ f[:2])
        lat_signed = float(frame.side * (centroid[0] - wheel_x))
        below = frame.eye[2] - float(centroid[2])
        out80 = float(np.percentile(np.abs(points[:, 0] - cx0), 80))
        stats_ng = scan_no_glass.get(object_id, stats)
        z70 = float(np.percentile(points[:, 2], 70))

        # oriented control cone: forward-and-down of the eye, laterally from
        # just inboard of the column out to the driver's door, no broad walls
        wall_lateral = extents[0] < 0.22 and extents[1] > 0.45 and extents[2] > 0.45
        in_cone = (
            0.20 <= ahead <= wheel_ahead + 1.0
            and -0.22 <= lat_signed <= 0.33
            and -0.10 <= below <= 1.35
            and not wall_lateral
        )

        if object_id in glass_ids:
            # a small pane in the cone is an instrument cover moving with the
            # cluster; real windows are wide or lateral and never convert
            if (diagonal <= 0.7 and 0.20 <= ahead <= wheel_ahead + 1.0
                    and -0.22 <= lat_signed <= 0.33 and -0.10 <= below <= 1.35
                    and stats["depth"] <= 0.30
                    and (stats["vf"] >= 0.30 or stats["min_r"] <= wheel_dist + 0.45)):
                verdicts[object_id] = ("translate", "instrument cover in the control cone", "med", {})
            continue

        # Scope channels are candidates, not absolutes.  The cheap point shell
        # is only a broad phase: when it says a mesh might enter, trace those
        # same sample rays against the filled DAE triangles.  This catches a
        # body/carpet face covering a part even when none of the face's sparse
        # vertices happens to share the point's 6-degree angular bin.
        def candidate_channels(
            candidate_stats: dict[str, float],
            candidate_stats_ng: dict[str, float],
        ) -> tuple[bool, bool, bool, bool]:
            visible = candidate_stats["front_vf"] >= 0.28
            enclosed = (
                candidate_stats["vf"] >= 0.08
                and candidate_stats["backed"] >= 0.45
                and out80 <= half_width - 0.02
                and candidate_stats["depth"] <= 0.45
            )
            fitment = (
                candidate_stats_ng["vf"] >= 0.12
                and diagonal <= 0.60
                and 0.15 <= ahead <= 1.5
                and abs(float(centroid[2]) - frame.eye[2]) <= 0.7
                and abs(float(centroid[0]) - cx0) >= half_width - 0.06
            )
            buried = (
                candidate_stats["backed"] >= 0.75
                and candidate_stats["depth"] <= 0.35
            )
            cone = (
                in_cone
                and candidate_stats["depth"] <= 0.75
                and candidate_stats["min_r"] <= SPATIAL_REACH_LIMIT
                and (float(centroid[2]) >= floor_z - 0.10 or buried)
                and (
                    candidate_stats["vf"] >= 0.45
                    or candidate_stats["min_r"] <= wheel_dist + 0.45
                    or buried
                    or enclosed
                )
            )
            return visible, enclosed, fitment, cone

        cand_visible, cand_enclosed, cand_fitment, cand_cone = candidate_channels(
            stats, stats_ng
        )
        point_cand_fitment = cand_fitment
        if surface_scene is not None and (
            cand_visible or cand_enclosed or cand_fitment or cand_cone
            or object_id in under_seat_candidates
            or object_id in forced
        ):
            exact = exact_by_id.get(object_id)
            if object_id not in exact_by_id:
                exact = core.surface_visibility_stats(
                    points, frame.eye, surface_scene, set(), frame.forward
                )
            if exact is not None:
                stats = dict(stats)
                stats.update({key: exact[key] for key in ("vf", "front_vf")})
                stats_ng = stats
                if point_cand_fitment and surface_scene_no_glass is not None:
                    exact_ng = core.surface_visibility_stats(
                        points,
                        frame.eye,
                        surface_scene_no_glass,
                        set(),
                        frame.forward,
                    )
                    if exact_ng is not None:
                        stats_ng = dict(stats)
                        stats_ng.update({
                            key: exact_ng[key] for key in ("vf", "front_vf")
                        })
                cand_visible, cand_enclosed, cand_fitment, cand_cone = candidate_channels(
                    stats, stats_ng
                )
        if not (cand_visible or cand_enclosed or cand_fitment or cand_cone
                or object_id in under_seat_candidates
                or object_id in forced):
            continue
        if scoped is not None:
            scoped.add(object_id)

        if not cand_fitment:
            beyond_fraction = beyond.get(object_id, 0.0)
            exact_glass = exact_glass_by_id.get(object_id)
            if exact_glass is not None:
                beyond_fraction = exact_glass["blocked"]
            if beyond_fraction >= 0.40:
                vetoed.add(object_id)
                if hard_vetoed is not None:
                    hard_vetoed.add(object_id)
                continue  # outside the glasshouse (wipers, hood, truck bed)
            if out80 > half_width + 0.04:
                vetoed.add(object_id)
                if hard_vetoed is not None:
                    hard_vetoed.add(object_id)
                continue  # protrudes past the cabin shell (door skin)
            if out80 > half_width - 0.02 and stats["lined"] < 0.35 and stats["backed"] < 0.35:
                vetoed.add(object_id)
                if hard_vetoed is not None:
                    hard_vetoed.add(object_id)
                continue  # at the shell with nothing behind: exposed skin
            if (out80 > half_width - 0.03 and z70 > frame.eye[2] + 0.05
                    and extents[0] < 0.40 and extents[1] > 0.6):
                vetoed.add(object_id)
                if hard_vetoed is not None:
                    hard_vetoed.add(object_id)
                continue  # shell wall rising past the beltline: door frame/skin
            if z70 > ceiling_z + 0.04:
                vetoed.add(object_id)
                if hard_vetoed is not None:
                    hard_vetoed.add(object_id)
                continue  # above the headliner (roof accessories)
            if (float(centroid[1]) < y_front - 0.12 and not cand_visible and not in_cone
                    and object_id not in forced):
                vetoed.add(object_id)
                continue  # ahead of the firewall
            if float(centroid[1]) > y_rear + 0.15:
                vetoed.add(object_id)
                if hard_vetoed is not None:
                    hard_vetoed.add(object_id)
                continue  # behind the cab
            if z70 < floor_z - 0.08 and not cand_cone:
                vetoed.add(object_id)
                if hard_vetoed is not None:
                    hard_vetoed.add(object_id)
                continue  # under the cabin floor

        if diagonal < 0.14 and stats["n"] < 40 and not in_cone:
            continue  # sub-resolution marker/dummy (engine light helpers)

        # Step 6, the resolution floor: the steering-column top (translate)
        # vs body/rack (mirror) are centimetres apart and slide with column
        # length -- the ONE sanctioned name hint, confined to the column axis
        lowered_name = f"{object_id} {obj.name}".lower()
        if (0.2 <= ahead and abs(float(centroid[0]) - wheel_x) <= 0.40
                and "column" in lowered_name and ahead > wheel_ahead - 0.1):
            if "top" in lowered_name:
                verdicts[object_id] = (
                    "translate", "steering column top (resolution floor: name hint)", "low", {})
            else:
                verdicts[object_id] = (
                    "mirror", "steering column body (resolution floor: name hint)", "low", {})
            continue

        if cand_cone:
            confidence = "high" if (abs(lat_signed) < 0.24 and ahead <= wheel_ahead + 0.75) else "med"
            verdicts[object_id] = ("translate", "in the driver control cone", confidence, {})
            continue

        emissive = mesh_flag(object_id, "emissive", require_all=False)
        sds = core.principal_extent_sds(points)
        planar = sds[0] / max(sds[1], 1e-6) < 0.35 or stats["n"] < 40
        display = emissive and planar and diagonal <= 0.9

        orphans, coarse_fraction = _mesh_symmetry(context, object_id, points, cx0)
        if (not cand_visible and abs(float(centroid[0]) - cx0) < 0.12
                and coarse_fraction < 0.15 and not cand_fitment and not cand_cone):
            continue  # barely-seen centred blob: mirroring it is a no-op
        if orphans == 0:
            xspan = float(np.ptp(points[:, 0]))
            z90 = float(np.percentile(points[:, 2], 90))
            fascia = (
                xspan >= max(1.05, 1.3 * half_width) and ahead >= 0.45
                and float(centroid[2]) <= frame.eye[2] + 0.05 and z90 >= frame.eye[2] - 0.62
                and extents[2] >= 0.28 and stats["vf"] >= 0.30
            )
            if display:
                verdicts[object_id] = ("mirror", "directional display", "med", {"flip": True})
            elif fascia:
                # Geometrically symmetric, but a fascia may carry directional
                # materials or generated detail, so preserve the established
                # dashboard transform.
                verdicts[object_id] = ("mirror", "dashboard fascia", "med", {})
            # else: symmetric about the centreline, reflection changes nothing
            continue

        confidence = "low" if coarse_fraction < 0.05 else (
            "med" if not cand_visible else "high")
        reason = ("exterior driver fitment" if cand_fitment and not cand_visible
                  else "one-sided interior part")
        if (out80 >= half_width - 0.05 and stats["front_vf"] < 0.50
                and not cand_fitment):
            confidence = "low"
            reason = "wall at the cabin shell (verify: possible exterior sheet)"
        verdicts[object_id] = ("pairable", reason, confidence, {"flip": display})
    return verdicts, vetoed


def _passenger_footwell_forced(
    frame: core.DriverFrame,
    present: list[str],
    entries_np: dict[str, object],
    modes: dict[str, tuple[str, str, str, dict]],
    hard_vetoed: set[str] | frozenset[str] = frozenset(),
) -> frozenset[str]:
    """Unclassified meshes in a 30-degree cone at the opposite footwell.

    The aim point is the reflected average of translated furniture below the
    wheel (pedals and their cluster).  This only grants scope admission; the
    ordinary veto, control-cone, symmetry, and pairing rules still decide the
    verdict.
    """
    import math
    import numpy as np

    if frame.wheel_center is None:
        return frozenset()
    eye = np.asarray(frame.eye, dtype=float)
    wheel_z = float(frame.wheel_center[2])
    translated_centroids = []
    for object_id in present:
        if object_id == frame.wheel_id or modes.get(object_id, ("none",))[0] != "translate":
            continue
        points = entries_np.get(object_id)
        if points is None or len(points) < 4:
            continue
        centroid = points.mean(axis=0)
        if centroid[2] < wheel_z - 0.10 and float(np.linalg.norm(centroid - eye)) <= 1.6:
            translated_centroids.append(centroid)
    if not translated_centroids:
        return frozenset()

    aim = np.mean(translated_centroids, axis=0)
    aim[0] = 2.0 * frame.center_x - aim[0]
    axis = aim - eye
    axis_length = float(np.linalg.norm(axis))
    if axis_length < 1e-6:
        return frozenset()
    axis /= axis_length
    min_cosine = math.cos(math.radians(30.0))
    forced: set[str] = set()
    for object_id in present:
        if object_id in hard_vetoed or modes.get(object_id, ("none",))[0] != "none":
            continue
        points = entries_np.get(object_id)
        if points is None or len(points) < 4:
            continue
        centroid = points.mean(axis=0)
        if centroid[2] >= wheel_z:
            continue  # a footwell cone never admits glazing/wipers above the wheel
        point_ranges = np.linalg.norm(points - eye, axis=1)
        if float(np.percentile(point_ranges, 80)) > 1.6:
            continue  # centroid-near body/exhaust meshes are not cabin furniture
        direction = centroid - eye
        distance = float(np.linalg.norm(direction))
        if 0.05 < distance <= 1.6 and float(direction @ axis) / distance >= min_cosine:
            forced.add(object_id)
    return frozenset(forced)


def _resolve_trim_pairs(
    context: core.VehicleContext,
    frame: core.DriverFrame,
    present: list[str],
    entries_np: dict[str, object],
    memo: dict[str, tuple[str, str, str, dict]],
    vetoed: set[str],
    pair_votes: dict[str, dict[str, int]],
) -> None:
    """Match pairable meshes to geometric twins among THIS trim's present set.

    Twins may also come from the latent pool: meshes the scan under-admitted
    (the passenger-side twin the eye barely sees) but never ones positively
    vetoed as exterior. Each twin is consumed once per trim, so mutually
    exclusive variants never compete for the same counterpart."""
    import numpy as np

    material_symbols = core.mesh_material_symbols(context)
    pairables = [o for o in present if memo.get(o, ("none",))[0] == "pairable"]
    latent = [
        o for o in present
        if memo.get(o, ("none",))[0] == "none" and o not in vetoed
        and o in entries_np and len(entries_np[o]) >= 4
    ]
    used: set[str] = set()
    cx0 = frame.center_x
    for object_id in sorted(
        pairables, key=lambda o: (-frame.side * float(entries_np[o][:, 0].mean()), o)
    ):
        if object_id in used:
            continue
        points_a = entries_np[object_id]
        centroid_a = points_a.mean(axis=0)
        if abs(float(centroid_a[0]) - cx0) < 0.08:
            continue  # centred: a one-sided fitment, nothing to pair with
        best: tuple[float, str] | None = None
        for twin_id in pairables + latent:
            if twin_id == object_id or twin_id in used:
                continue
            points_b = entries_np[twin_id]
            centroid_b = points_b.mean(axis=0)
            if abs((float(centroid_a[0]) - cx0) + (float(centroid_b[0]) - cx0)) > 0.14:
                continue
            if (abs(float(centroid_a[1]) - float(centroid_b[1])) > 0.35
                    or abs(float(centroid_a[2]) - float(centroid_b[2])) > 0.35):
                continue
            diag_a = float(np.linalg.norm(np.ptp(points_a, axis=0)))
            diag_b = float(np.linalg.norm(np.ptp(points_b, axis=0)))
            if max(diag_a, diag_b) / max(min(diag_a, diag_b), 1e-6) > 1.5:
                continue
            residual = core.mirror_pair_residual(points_a, points_b, cx0)
            if residual <= SPATIAL_PAIR_RESIDUAL and (best is None or residual < best[0]):
                best = (residual, twin_id)
        if best is not None:
            twin_id = best[1]
            used.add(object_id)
            used.add(twin_id)
            symbols_a = set(material_symbols.get(object_id, ()))
            symbols_b = set(material_symbols.get(twin_id, ()))
            # Directional-material twins bind mutually exclusive symbols.
            # Multi-material housings may legitimately share their body
            # material while using a side-specific auxiliary symbol; those
            # are still safe structural pairs.
            if symbols_a and symbols_b and symbols_a.isdisjoint(symbols_b):
                reason = (
                    "functionally sided: materials differ, needs build-side material rebind"
                )
                memo[object_id] = ("functional_skip", reason, "high", {})
                memo[twin_id] = ("functional_skip", reason, "high", {})
                pair_votes.pop(object_id, None)
                pair_votes.pop(twin_id, None)
                for votes in pair_votes.values():
                    votes.pop(object_id, None)
                    votes.pop(twin_id, None)
                continue
            if memo.get(twin_id, ("none",))[0] == "none":
                memo[twin_id] = ("pairable", "geometric twin across the centreline", "med", {})
            pair_votes.setdefault(object_id, {})[twin_id] = (
                pair_votes.get(object_id, {}).get(twin_id, 0) + 1)
            pair_votes.setdefault(twin_id, {})[object_id] = (
                pair_votes.get(twin_id, {}).get(object_id, 0) + 1)


def _inherit_mounted_parts(
    context: core.VehicleContext,
    frame: core.DriverFrame,
    present: list[str],
    entries_np: dict[str, object],
    memo: dict[str, tuple[str, str, str, dict]],
    vetoed: set[str],
    pair_votes: dict[str, dict[str, int]],
    scoped: set[str],
) -> None:
    """Assembly propagation: a small part mounted ON a mirrored surface
    mirrors with it.

    Individually, a hazard button or a handbrake lever is near-centred and
    symmetric -- reflection looks like a no-op at cloud resolution, so the
    per-mesh verdict is skip. But these are components of assemblies (button
    on dash, knob on lever, seals on fascia): when the surface a part touches
    is classified aesthetic Mirror, the part inherits Mirror at low
    confidence rather than staying behind. Two passes resolve chains
    (button -> console -> dash). Only skip/none verdicts are upgraded --
    translate/pair verdicts and vetoed exterior meshes are never touched --
    and glass panes never inherit."""
    import numpy as np

    material_symbols = core.mesh_material_symbols(context)
    material_flags = core.material_flags_for_context(context)

    def is_glass(object_id: str) -> bool:
        symbols = material_symbols.get(object_id)
        return bool(symbols) and all(
            material_flags.get(s, {}).get("glass") for s in symbols
        )

    eye = np.array(frame.eye)

    def is_mirror_host(object_id: str) -> bool:
        mode = memo.get(object_id, ("none",))[0]
        if mode == "mirror":
            return True
        # an unpaired pairable emits as aesthetic Mirror ("twin absent"), so
        # it anchors its satellites the same way; a paired host is structural
        # and its satellites pair on their own
        return mode == "pairable" and object_id not in pair_votes

    for _ in range(2):
        hosts = [
            o for o in present
            if is_mirror_host(o)
            and o in entries_np
            and float(np.linalg.norm(np.ptp(entries_np[o], axis=0))) >= 0.15
            and float(np.linalg.norm(entries_np[o].mean(axis=0) - eye)) <= 1.6
        ]
        if not hosts:
            break
        changed = False
        for object_id in present:
            if memo.get(object_id, ("none",))[0] != "none" or object_id in vetoed:
                continue
            points = entries_np.get(object_id)
            if points is None or len(points) < 4 or is_glass(object_id):
                continue
            diag = float(np.linalg.norm(np.ptp(points, axis=0)))
            if diag > 0.70:
                continue  # furniture-sized: judged on its own evidence
            if diag < 0.14 and len(points) < 40:
                continue  # sub-resolution marker/dummy (engine light helpers)
            if float(np.linalg.norm(points.mean(axis=0) - eye)) > 1.6:
                continue  # outside the cabin radius: not interior furniture
            for host in hosts:
                host_points = entries_np[host]
                gap2 = ((points[:, None, :] - host_points[None, :, :]) ** 2).sum(axis=2)
                if float(np.sqrt(gap2.min())) <= 0.03:
                    memo[object_id] = (
                        "mirror", f"mounted on {host}", "low", {})
                    changed = True
                    break
        if not changed:
            break

    # A genuinely floating scoped mesh can still be recognised by occlusion:
    # if its eye rays continue into any transformed cabin/mirror furniture,
    # the floater is cabin furniture too.  Contact inheritance above has
    # already consumed anything mounted within 3 cm.
    floaters: list[str] = []
    sightline_entries = dict(entries_np)
    forward = np.asarray(frame.forward, dtype=float)
    for object_id in present:
        if (object_id not in scoped or memo.get(object_id, ("none",))[0] != "none"
                or object_id in vetoed):
            continue
        points = entries_np.get(object_id)
        if points is None or len(points) < 4 or is_glass(object_id):
            continue
        diag = float(np.linalg.norm(np.ptp(points, axis=0)))
        if diag > 0.70 or (diag < 0.14 and len(points) < 40):
            continue
        if float(np.percentile(np.linalg.norm(points - eye, axis=1), 80)) > 1.6:
            continue
        # Sightline inheritance is driver-visible evidence, so use the same
        # forward 180-degree hemisphere as ordinary visible admission.  Keep
        # only the mesh points in front of the eye; a rear lamp must not
        # inherit merely because its backward rays terminate on bodywork.
        front_points = points[((points - eye) @ forward) >= 0.0]
        if not len(front_points):
            continue
        sightline_entries[object_id] = front_points
        floaters.append(object_id)
    if not floaters:
        return

    transformed = {
        object_id: memo[object_id][0]
        for object_id in present
        if object_id in entries_np
        and memo.get(object_id, ("none",))[0] in {"translate", "mirror", "pairable"}
        and float(np.percentile(
            np.linalg.norm(entries_np[object_id] - eye, axis=1), 80
        )) <= 1.6
    }
    backing = core.directional_verdict_backing(
        sightline_entries, frame.eye, floaters, transformed
    )
    for object_id in floaters:
        classes = backing.get(object_id, {})
        if not classes:
            continue
        behind_class = max(classes, key=lambda name: (classes[name], name))
        memo[object_id] = (
            "mirror", f"floating in front of {behind_class} geometry", "low", {}
        )


def build_mode_recommendations(
    context: core.VehicleContext,
    object_ids: list[str],
) -> list[dict[str, str]]:
    """Classify meshes for hand conversion from the driver's viewpoint.

    Batch model: the intrinsic class is a property of the mesh, not the trim,
    so each unique mesh is classified once (in the first trim that contains
    it) and memoised; later trims only re-solve low-confidence meshes whose
    position is trim-dependent, plus the inherently per-trim structural
    pairing. State is cached on the context so reopening the modal reuses
    every solved trim. No driver frame (no camera and no wheel) means no
    trustworthy spatial reasoning: the answer is no recommendations, never a
    name-based guess."""
    available = {o for o in object_ids if o in context.objects and o in context.preview_by_id}
    if not available:
        return []
    frame = core.driver_frame_for_context(context)
    if frame is None:
        return []

    state = getattr(context, "_spatial_recommendation_state", None)
    if state is None:
        state = {
            "memo": {}, "vetoed": set(), "hard_vetoed": set(),
            "scoped": set(), "pair_votes": {}, "trims_done": set(),
        }
        context._spatial_recommendation_state = state
    memo: dict[str, tuple[str, str, str, dict]] = state["memo"]
    vetoed: set[str] = state["vetoed"]
    hard_vetoed: set[str] = state.setdefault("hard_vetoed", set())
    scoped: set[str] = state.setdefault("scoped", set())
    pair_votes: dict[str, dict[str, int]] = state["pair_votes"]

    trims: list[str | None] = sorted(context.variants) if context.variants else [None]
    for trim in trims:
        present, entries_np = _spatial_entries_for_trim(context, trim, available)
        if not present:
            continue
        surface_np: dict[str, object] = {}
        todo = [
            o for o in present
            if o not in memo
            or (o in context.variant_dependent_meshes
                and (memo[o][0] == "none" or memo[o][2] == "low")
                and trim not in state["trims_done"])
        ]
        if todo:
            surface_np = _spatial_surfaces_for_trim(
                context, trim, present, entries_np
            )
            verdicts, newly_vetoed = _classify_meshes_for_trim(
                context, frame, present, entries_np, todo,
                hard_vetoed=hard_vetoed,
                scoped=scoped,
                surface_np=surface_np,
            )
            vetoed.update(newly_vetoed)
            for o in todo:
                verdict = verdicts.get(o, ("none", "", "med", {}))
                previous = memo.get(o)
                if previous is None or previous[0] == "none":
                    memo[o] = verdict
                elif previous[2] == "low" and verdict[0] != "none" and verdict[2] != "low":
                    memo[o] = verdict  # a trim resolved the borderline case
            forced = _passenger_footwell_forced(
                frame, present, entries_np, memo, hard_vetoed
            )
            forced_todo = [
                o for o in present
                if o in forced and memo.get(o, ("none",))[0] == "none"
            ]
            if forced_todo:
                if not surface_np:
                    surface_np = _spatial_surfaces_for_trim(
                        context, trim, present, entries_np
                    )
                forced_verdicts, forced_vetoed = _classify_meshes_for_trim(
                    context, frame, present, entries_np, forced_todo, forced,
                    hard_vetoed, scoped, surface_np,
                )
                vetoed.update(forced_vetoed)
                for o in forced_todo:
                    verdict = forced_verdicts.get(o, ("none", "", "med", {}))
                    if verdict[0] != "none":
                        memo[o] = verdict
                        vetoed.discard(o)
        if trim not in state["trims_done"]:
            _resolve_trim_pairs(
                context, frame, present, entries_np, memo, vetoed, pair_votes
            )
            _inherit_mounted_parts(
                context, frame, present, entries_np, memo, vetoed, pair_votes, scoped
            )
            state["trims_done"].add(trim)

    # Meshes no trim uses stay unclassified on purpose: the union of mutually
    # exclusive variants is not a cabin, and a part no config fits cannot be
    # converted anyway.

    recommendations: list[dict[str, str]] = []
    emitted_pairs: set[frozenset] = set()
    requested = set(object_ids)
    for object_id in sorted(requested & set(memo)):
        mode, reason, confidence, extra = memo[object_id]
        if mode in {"none", "functional_skip"}:
            continue
        if confidence == "low":
            reason = f"{reason} (low confidence)"
        if mode == "pairable":
            votes = pair_votes.get(object_id)
            twin = max(votes, key=lambda t: (votes[t], t)) if votes else None
            if twin is not None and twin in requested:
                key = frozenset((object_id, twin))
                if key in emitted_pairs:
                    continue
                emitted_pairs.add(key)
                # name the driver-side member so the modal reads naturally
                obj_a = context.objects.get(object_id)
                obj_b = context.objects.get(twin)
                first, second = object_id, twin
                if obj_a is not None and obj_b is not None:
                    if frame.side * obj_b.x > frame.side * obj_a.x:
                        first, second = twin, object_id
                recommendations.append({
                    "kind": "pair",
                    "object_id": first,
                    "source_id": second,
                    "mode": core.MODE_MIRROR_STRUCTURAL,
                    "reason": reason,
                    "confidence": confidence,
                })
            else:
                entry = {
                    "kind": "single",
                    "object_id": object_id,
                    "source_id": "",
                    "mode": core.MODE_MIRROR,
                    "reason": f"{reason}; twin absent in this trim",
                    "confidence": confidence,
                }
                if extra.get("flip"):
                    entry["textureFlip"] = True
                recommendations.append(entry)
        else:
            entry = {
                "kind": "single",
                "object_id": object_id,
                "source_id": "",
                "mode": core.MODE_TRANSLATE if mode == "translate" else core.MODE_MIRROR,
                "reason": reason,
                "confidence": confidence,
            }
            if mode == "mirror" and extra.get("flip"):
                entry["textureFlip"] = True
            recommendations.append(entry)

    mode_order = {
        core.MODE_TRANSLATE: 0,
        core.MODE_MIRROR: 1,
        core.MODE_MIRROR_STRUCTURAL: 2,
    }
    recommendations.sort(
        key=lambda item: (
            mode_order.get(item["mode"], 99),
            item["object_id"].lower(),
            item.get("source_id", "").lower(),
        )
    )
    return recommendations


def offset_label(value: object) -> str:
    if value in (None, ""):
        return ""
    try:
        return fmt_float(abs(float(value)))
    except (TypeError, ValueError):
        return ""


def offset_display(mode: str, value: object, *, manual_delta: bool) -> str:
    if mode != core.MODE_TRANSLATE:
        return "N/A"
    explicit = offset_label(value)
    if explicit:
        return explicit
    return "Manual" if manual_delta else "Auto"


def fliptex_display(mode: str, value: object) -> str:
    if mode != core.MODE_MIRROR:
        return "N/A"
    return yn_label(value)


def existing_initial_dir(path: object, fallback: Path) -> str:
    candidate = Path(str(path)) if path else fallback
    if candidate.is_file():
        candidate = candidate.parent
    if candidate.exists():
        return str(candidate)
    return str(fallback)


def app_icon_path() -> Path | None:
    for candidate in (RESOURCE_DIR / APP_ICON_NAME, THIS_DIR / APP_ICON_NAME):
        if candidate.exists():
            return candidate
    return None


class HandDriveToolApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("BeamXP - BeamNG Vehicle eXPort Services")
        self._set_app_icon()
        self.geometry("1480x840")
        self.minsize(480, 360)

        self.context: core.VehicleContext | None = None
        self.conversion: dict[str, object] = {}
        self.source_zip: Path | None = None
        self.vehicle_ids: list[str] = []
        # Model dropdown history: combo label -> (zip path, vehicle id)
        self.model_entries: dict[str, tuple[Path, str]] = {}
        self.model_load_busy = False
        self.settings = core.load_app_settings()
        self.worker_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker_running = False
        self.part_resolver = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rhd-parts")
        self.variant_detector = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rhd-variants")
        self.variant_detection_seq = 0
        self.variant_detection_running = False
        self.variant_detection_pending = False
        self.variant_detected_hands: dict[str, str] = {}
        self.variant_detection_complete = False
        self.part_refresh_after_id: str | None = None
        self.part_refresh_running = False
        self.part_refresh_pending = False
        self.part_refresh_pending_reset = False
        self.part_refresh_seq = 0
        self.resolved_part_ids: list[str] = []
        self.vehicle_load_seq = 0
        self.recommendation_seq = 0
        self.recommendation_modal: tk.Toplevel | None = None
        self.plate_editor_modal: PlateEditorDialog | None = None
        self.plate_library_modal: PlateLibraryDialog | None = None
        self.recommendation_tree: ttk.Treeview | None = None
        self.recommendation_rows: dict[str, dict[str, str]] = {}
        self.structural_prompt_after_id: str | None = None
        self.structural_prompt_part_id: str | None = None
        self.structural_prompt_previous_mode: str = core.MODE_SKIP
        self.structural_prompt_open = False
        # Per-table click-to-sort state: tree -> (column id or None, descending)
        self._tree_sort: dict[ttk.Treeview, tuple[str | None, bool]] = {}
        self._tree_heading_text: dict[ttk.Treeview, dict[str, str]] = {}
        # Treeview has no native cell editors.  Keep one temporary combobox
        # overlay at a time and manage its popdown/focus lifecycle explicitly.
        self._tree_combo_editor: ttk.Combobox | None = None
        self._tree_combo_focus_after_id: str | None = None
        self.part_filter_entry: ttk.Entry | None = None

        self.source_var = tk.StringVar(value="No source zip loaded")
        self.vehicle_var = tk.StringVar()
        self.project_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Ready")
        self.detail_var = tk.StringVar(value="")
        self.filter_var = tk.StringVar()
        self.auto_delta_var = tk.StringVar(value="")
        self.manual_delta_enabled = tk.BooleanVar(value=False)
        self.manual_delta_var = tk.StringVar(value="")
        self.plate_choice_var = tk.StringVar(value="Off")
        self.plate_choice_to_id: dict[str, str] = {}
        self.mods_folder_var = tk.StringVar(value=str(self.settings.get("modsFolder") or ""))
        self.blender_var = tk.StringVar(value=str(self.settings.get("blenderExecutable") or ""))
        self.preview_output_var = tk.StringVar(value="")
        self.preview_output_to_config: dict[str, str] = {}
        self.preview_output_to_output: dict[str, str] = {}
        # While the Config dropdown list is open, the highlighted (not
        # yet confirmed) entry hot-loads into the preview via this override.
        self.preview_output_hover: str | None = None
        self._preview_popdown_listbox: str | None = None
        self._preview_hover_after: str | None = None

        self.viewer: ModelPreview | None = None
        # Box-viewer preview data for the trim on screen; ModelPreview holds
        # this by reference, so it is mutated rather than replaced.
        self.box_preview_by_id: dict[str, dict[str, object]] = {}
        self._box_preview_config: str | None = None
        self.viewer_supports_scene = False
        self.mesh_scene_seq = 0
        self.mesh_scene_after: str | None = None
        self.mesh_scene_running = False
        self.mesh_scene_pending = False
        self.mesh_scene_hash: str | None = None
        self.mesh_scene_reset_pending = True
        self.current_part_ids: list[str] = []

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._configure_theme()
        self._build_ui()
        self.bind("<KeyPress-h>", self._toggle_selected_parts_visibility_shortcut)
        self.bind("<KeyPress-H>", self._toggle_selected_parts_visibility_shortcut)
        for hotkey, hotkey_mode in MODE_HOTKEYS.items():
            self.bind(f"<KeyPress-{hotkey}>", lambda event, m=hotkey_mode: self._set_selected_part_mode_shortcut(event, m))
            self.bind(f"<KeyPress-{hotkey.upper()}>", lambda event, m=hotkey_mode: self._set_selected_part_mode_shortcut(event, m))
        self.bind_all("<Button-1>", self._clear_part_filter_focus_on_click, add="+")
        self._rebuild_model_combo()
        self.after_idle(self._maximize_on_start)
        self.after(120, self._poll_worker_queue)

    def _on_close(self) -> None:
        self._close_tree_combo_editor()
        self._cancel_structural_prompt()
        self.part_resolver.shutdown(wait=False, cancel_futures=True)
        self.variant_detector.shutdown(wait=False, cancel_futures=True)
        self.destroy()

    def _set_app_icon(self) -> None:
        icon_path = app_icon_path()
        if icon_path is None:
            return
        try:
            if sys.platform == "win32":
                self.iconbitmap(default=str(icon_path))
            else:
                self.iconbitmap(str(icon_path))
        except tk.TclError:
            pass

    @staticmethod
    def _is_widget_or_child(widget: tk.Widget, parent: tk.Widget) -> bool:
        widget_path = str(widget)
        parent_path = str(parent)
        return widget_path == parent_path or widget_path.startswith(parent_path + ".")

    def _clear_part_filter_focus_on_click(self, event: tk.Event) -> None:
        filter_entry = self.part_filter_entry
        if filter_entry is None:
            return
        try:
            if self.focus_get() is not filter_entry:
                return
        except tk.TclError:
            return
        clicked = event.widget
        if clicked is not None and self._is_widget_or_child(clicked, filter_entry):
            return
        try:
            clicked.focus_set()
        except Exception:
            self.focus_set()

    def _part_display_name(self, object_id: str) -> str:
        if self.context is None:
            return object_id
        obj = self.context.objects.get(object_id)
        if obj is not None and not obj.dae_path and obj.name and obj.name != object_id:
            return f"{obj.name} [{object_id}]"
        prefix = f"{self.context.vehicle_id}_"
        if object_id.startswith(prefix):
            return object_id[len(prefix) :]
        return object_id

    def _configure_theme(self) -> None:
        self.ttk_style = ttk.Style(self)
        for theme in ("clam", "alt", "default"):
            if theme in self.ttk_style.theme_names():
                self.ttk_style.theme_use(theme)
                return

    def _maximize_on_start(self) -> None:
        try:
            self.state("zoomed")
        except tk.TclError:
            try:
                self.attributes("-zoomed", True)
            except tk.TclError:
                self.geometry(f"{self.winfo_screenwidth()}x{self.winfo_screenheight()}+0+0")

    def _current_monitor_work_area(self) -> tuple[int, int, int, int]:
        if sys.platform == "win32":
            try:
                import ctypes
                from ctypes import wintypes

                class RECT(ctypes.Structure):
                    _fields_ = (
                        ("left", wintypes.LONG),
                        ("top", wintypes.LONG),
                        ("right", wintypes.LONG),
                        ("bottom", wintypes.LONG),
                    )

                class MONITORINFO(ctypes.Structure):
                    _fields_ = (
                        ("cbSize", wintypes.DWORD),
                        ("rcMonitor", RECT),
                        ("rcWork", RECT),
                        ("dwFlags", wintypes.DWORD),
                    )

                monitor = ctypes.windll.user32.MonitorFromWindow(self.winfo_id(), 2)
                info = MONITORINFO()
                info.cbSize = ctypes.sizeof(MONITORINFO)
                if monitor and ctypes.windll.user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
                    work = info.rcWork
                    return work.left, work.top, work.right, work.bottom
            except Exception:
                pass
        left = self.winfo_vrootx()
        top = self.winfo_vrooty()
        return left, top, left + self.winfo_vrootwidth(), top + self.winfo_vrootheight()

    def _place_modal_on_app_monitor(self, modal: tk.Toplevel) -> None:
        self.update_idletasks()
        modal.update_idletasks()

        width = modal.winfo_width()
        height = modal.winfo_height()
        if width <= 1:
            width = modal.winfo_reqwidth()
        if height <= 1:
            height = modal.winfo_reqheight()

        parent_x = self.winfo_rootx()
        parent_y = self.winfo_rooty()
        parent_w = max(self.winfo_width(), 1)
        parent_h = max(self.winfo_height(), 1)
        x = parent_x + (parent_w - width) // 2
        y = parent_y + (parent_h - height) // 2

        work_left, work_top, work_right, work_bottom = self._current_monitor_work_area()
        x = min(max(x, work_left), max(work_left, work_right - width))
        y = min(max(y, work_top), max(work_top, work_bottom - height))
        modal.geometry(f"{width}x{height}+{x}+{y}")

    def _show_error(self, title: str, message: str, *, parent: tk.Widget | None = None) -> None:
        messagebox.showerror(title, message, parent=parent or self)

    def _ask_open_filename(self, **options) -> str:
        return filedialog.askopenfilename(parent=self, **options)

    def _ask_directory(self, **options) -> str:
        return filedialog.askdirectory(parent=self, **options)

    def _configure_tree_rows(self, tree: ttk.Treeview) -> None:
        tree.tag_configure("evenrow", background="#ffffff")
        tree.tag_configure("oddrow", background="#cccccc")

    def _row_tags(self, index: int) -> tuple[str, ...]:
        return ("oddrow",) if index % 2 else ("evenrow",)

    # ----- generic click-to-sort for all table views -----------------------

    def _tree_column_name(self, tree: ttk.Treeview, column_id: str) -> str | None:
        """Map a display column id ('#3') to its logical column name so click
        handlers stay correct no matter how many columns a table has. Returns
        None for the tree column ('#0') or on any mismatch."""
        if not column_id or column_id == "#0":
            return None
        try:
            index = int(column_id[1:]) - 1
        except ValueError:
            return None
        columns = tree["columns"]
        if 0 <= index < len(columns):
            return str(columns[index])
        return None

    def _register_tree_headings(self, tree: ttk.Treeview, headings: dict[str, str]) -> None:
        """Record each heading's plain label and wire its heading button to sort
        the table by that column. `headings` maps a column id ('#0' or a column
        name) to its display label."""
        self._tree_heading_text[tree] = dict(headings)
        for column in headings:
            tree.heading(column, command=lambda c=column, t=tree: self._sort_tree(t, c))

    def _sort_tree(self, tree: ttk.Treeview, column: str) -> None:
        self._close_tree_combo_editor()
        prev_column, prev_descending = self._tree_sort.get(tree, (None, False))
        descending = column == prev_column and not prev_descending
        self._tree_sort[tree] = (column, descending)
        self._apply_tree_sort(tree)

    def _scroll_tree(self, tree: ttk.Treeview, axis: str, *args: object) -> None:
        """Close a cell overlay before moving the rows beneath it."""
        self._close_tree_combo_editor()
        getattr(tree, axis)(*args)

    @staticmethod
    def _sort_key(value: object) -> tuple[int, object]:
        # Numeric-parseable cells sort numerically ahead of text cells, so
        # coordinate/offset columns order by value while Y/N and text columns
        # order alphabetically -- and float is never compared against str.
        text = str(value).strip()
        try:
            return (0, float(text))
        except ValueError:
            return (1, text.lower())

    def _apply_tree_sort(self, tree: ttk.Treeview) -> None:
        """Reorder the rows in place per the tree's current sort selection.
        Row iids are preserved (only their visual order changes) so selection,
        preview picking, and part/config identity mapping are unaffected."""
        entry = self._tree_sort.get(tree)
        if not entry or entry[0] is None:
            return
        column, descending = entry
        children = list(tree.get_children(""))
        if not children:
            return
        if column == "#0":
            cell = lambda iid: tree.item(iid, "text")
        else:
            cell = lambda iid: tree.set(iid, column)
        ordered = sorted(children, key=lambda iid: self._sort_key(cell(iid)), reverse=descending)
        for index, iid in enumerate(ordered):
            tree.move(iid, "", index)
            tree.item(iid, tags=self._row_tags(index))
        self._update_sort_indicators(tree)

    def _restore_tree_order(self, tree: ttk.Treeview, previous_order: list[str]) -> None:
        children = list(tree.get_children(""))
        if not children:
            return
        existing = set(children)
        seen: set[str] = set()
        ordered: list[str] = []
        for iid in previous_order:
            if iid in existing and iid not in seen:
                ordered.append(iid)
                seen.add(iid)
        ordered.extend(iid for iid in children if iid not in seen)
        for index, iid in enumerate(ordered):
            tree.move(iid, "", index)
            tree.item(iid, tags=self._row_tags(index))

    @staticmethod
    def _tree_body_click(tree: ttk.Treeview, event: tk.Event) -> bool:
        return tree.identify_region(event.x, event.y) in {"tree", "cell"}

    def _update_sort_indicators(self, tree: ttk.Treeview) -> None:
        base = self._tree_heading_text.get(tree)
        if not base:
            return
        entry = self._tree_sort.get(tree)
        sort_column, descending = entry if entry else (None, False)
        arrow = " ▼" if descending else " ▲"
        for column, label in base.items():
            tree.heading(column, text=label + arrow if column == sort_column else label)

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        top = ttk.Frame(self, padding=(10, 8, 10, 4))
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(2, weight=1)
        top.columnconfigure(5, weight=0)

        self.open_button = ttk.Button(top, text="Open Vehicle Zip", command=self._open_zip_dialog)
        self.open_button.grid(row=0, column=0, sticky="w")
        self.refresh_button = ttk.Button(
            top,
            text="Refresh",
            command=lambda: self._load_selected_vehicle(force_reload=True),
            state="disabled",
        )
        self.refresh_button.grid(row=0, column=1, sticky="w", padx=(6, 0))
        ttk.Label(top, textvariable=self.source_var).grid(row=0, column=2, sticky="ew", padx=(8, 16))
        ttk.Label(top, text="Model").grid(row=0, column=3, sticky="e")
        self.vehicle_combo = ttk.Combobox(top, textvariable=self.vehicle_var, state="disabled", width=22)
        self.vehicle_combo.grid(row=0, column=4, sticky="w", padx=(6, 12))
        self.vehicle_combo.bind("<<ComboboxSelected>>", lambda _event: self._on_model_selected())
        ttk.Button(top, text="Save Config", command=self._save_config).grid(row=0, column=5, sticky="e")
        ttk.Button(top, text="Import Config", command=self._import_config_dialog).grid(row=0, column=6, sticky="e", padx=(6, 0))

        ttk.Label(top, textvariable=self.project_var).grid(row=1, column=0, columnspan=7, sticky="ew", pady=(6, 0))

        # ttk.PanedWindow cannot change orientation after creation, so keep
        # one paned window per orientation and move the two panes between
        # them when the window aspect ratio flips (see _on_root_configure).
        # The pane frames are children of the toplevel so both paned windows
        # may manage them.
        self.main_paned_h = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.main_paned_v = ttk.PanedWindow(self, orient=tk.VERTICAL)
        self.main_orientation: str | None = None

        left = self.tables_pane = ttk.Frame(self)
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=0)
        left.rowconfigure(3, weight=1)

        right = self.preview_pane = ttk.Frame(self)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        self._build_variant_panel(left)
        self._build_part_panel(left)
        self._build_right_panel(right)

        self._apply_main_orientation("landscape")
        self.bind("<Configure>", self._on_root_configure, add="+")

        bottom = ttk.Frame(self, padding=(10, 4, 10, 8))
        bottom.grid(row=2, column=0, sticky="ew")
        bottom.columnconfigure(0, weight=1)
        ttk.Label(bottom, textvariable=self.detail_var).grid(row=0, column=0, sticky="w")
        ttk.Label(bottom, textvariable=self.status_var).grid(row=1, column=0, sticky="w", pady=(4, 0))

    def _on_root_configure(self, event: tk.Event) -> None:
        if event.widget is not self:
            return
        if event.width <= 1 or event.height <= 1:
            return
        self._apply_main_orientation("portrait" if event.height > event.width else "landscape")

    def _apply_main_orientation(self, mode: str) -> None:
        if mode == self.main_orientation:
            return
        self.main_orientation = mode
        for paned in (self.main_paned_h, self.main_paned_v):
            for pane in paned.panes():
                paned.forget(pane)
            paned.grid_remove()
        if mode == "landscape":
            paned = self.main_paned_h
            # Keep the tables at their requested width and give spare
            # horizontal space to the ModernGL preview. The sash remains
            # user-adjustable.
            paned.add(self.tables_pane, weight=0)
            paned.add(self.preview_pane, weight=1)
        else:
            paned = self.main_paned_v
            paned.add(self.tables_pane, weight=1)
            paned.add(self.preview_pane, weight=1)
        paned.grid(row=1, column=0, sticky="nsew", padx=10, pady=4)

    def _build_variant_panel(self, parent: ttk.Frame) -> None:
        header = ttk.Frame(parent)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        ttk.Label(header, text="Variants").pack(side="left")
        ttk.Button(header, text="Clear Builds", command=lambda: self._set_all_variants_selected(False)).pack(
            side="right"
        )
        ttk.Button(header, text="Convert All", command=lambda: self._set_all_variants_selected(True)).pack(
            side="right",
            padx=(0, 6),
        )

        frame = ttk.Frame(parent)
        frame.grid(row=1, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        columns = ("build", "config", "display", "stock_hand", "plate", "front_plate", "rear_plate")
        self.variant_tree = ttk.Treeview(frame, columns=columns, show="headings", height=8, selectmode="browse")
        headings = {
            "build": "Build",
            "config": "Config",
            "display": "Display Name",
            "stock_hand": "Stock drive side",
            "plate": "Plates",
            "front_plate": "Front plate",
            "rear_plate": "Rear plate",
        }
        widths = {
            "build": 88,
            "config": 130,
            "display": 260,
            "stock_hand": 110,
            "plate": 120,
            "front_plate": 94,
            "rear_plate": 94,
        }
        for col in columns:
            self.variant_tree.heading(
                col,
                text=headings[col],
                anchor="w",
            )
            self.variant_tree.column(
                col,
                width=widths[col],
                minwidth=48,
                stretch=col == "display",
                anchor="w",
            )
        self._register_tree_headings(self.variant_tree, headings)
        yscroll = ttk.Scrollbar(
            frame,
            orient=tk.VERTICAL,
            command=lambda *args: self._scroll_tree(self.variant_tree, "yview", *args),
        )
        xscroll = ttk.Scrollbar(
            frame,
            orient=tk.HORIZONTAL,
            command=lambda *args: self._scroll_tree(self.variant_tree, "xview", *args),
        )
        self.variant_tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.variant_tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        self._configure_tree_rows(self.variant_tree)
        self.variant_tree.bind("<Button-1>", self._variant_click)
        self.variant_tree.bind("<Double-1>", self._variant_double_click)
        self.variant_tree.bind("<MouseWheel>", lambda _event: self._close_tree_combo_editor(), add="+")
        self.variant_tree.bind("<Shift-MouseWheel>", lambda _event: self._close_tree_combo_editor(), add="+")
        self.variant_tree.bind("<Button-4>", lambda _event: self._close_tree_combo_editor(), add="+")
        self.variant_tree.bind("<Button-5>", lambda _event: self._close_tree_combo_editor(), add="+")
        self.variant_tree.bind("<Configure>", lambda _event: self._close_tree_combo_editor(), add="+")

    def _build_part_panel(self, parent: ttk.Frame) -> None:
        header = ttk.Frame(parent)
        header.grid(row=2, column=0, sticky="ew", pady=(10, 4))
        header.columnconfigure(1, weight=1)
        ttk.Label(header, text="Parts Used by Selected Variants").grid(row=0, column=0, sticky="w")
        self.part_filter_entry = ttk.Entry(header, textvariable=self.filter_var)
        self.part_filter_entry.grid(row=0, column=1, sticky="ew", padx=(8, 6))
        self.part_filter_entry.insert(0, "")
        self.recommend_button = ttk.Button(
            header,
            text="Recommend Modes",
            command=self._open_recommendations_modal,
            state="disabled",
        )
        self.recommend_button.grid(row=0, column=2, sticky="e")
        self.show_all_parts_button = ttk.Button(
            header,
            text="Show All",
            command=lambda: self._set_all_parts_visible(True),
            state="disabled",
        )
        self.show_all_parts_button.grid(row=0, column=3, sticky="e", padx=(6, 0))
        self.hide_all_parts_button = ttk.Button(
            header,
            text="Hide All",
            command=lambda: self._set_all_parts_visible(False),
            state="disabled",
        )
        self.hide_all_parts_button.grid(row=0, column=4, sticky="e", padx=(6, 0))
        self.filter_var.trace_add("write", lambda *_args: self._refresh_parts())

        frame = ttk.Frame(parent)
        frame.grid(row=3, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        columns = ("visible", "solo", "active", "mode", "offset", "fliptex", "steering", "x", "y", "z")
        self.part_tree = ttk.Treeview(frame, columns=columns, show=("tree", "headings"), selectmode="extended")
        self.part_tree.heading("#0", text="Part", anchor="w")
        self.part_tree.column("#0", width=250, minwidth=150, stretch=True, anchor="w")
        headings = {
            "mode": "Mode",
            "offset": "Offset X",
            "fliptex": "Flip Tex",
            "steering": "Steering Ref",
            "visible": "Visible",
            "solo": "Solo",
            "active": "Active",
            "x": "X",
            "y": "Y",
            "z": "Z",
        }
        widths = {
            "mode": 132,
            "offset": 82,
            "fliptex": 74,
            "steering": 96,
            "visible": 70,
            "solo": 60,
            "active": 64,
            "x": 82,
            "y": 82,
            "z": 82,
        }
        for col in columns:
            self.part_tree.heading(
                col,
                text=headings[col],
                anchor="w",
            )
            self.part_tree.column(
                col,
                width=widths[col],
                minwidth=50,
                stretch=False,
                anchor="center" if col in {"fliptex", "steering", "visible", "solo", "active"} else "w",
            )
        self._register_tree_headings(self.part_tree, {"#0": "Part", **headings})
        yscroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.part_tree.yview)
        xscroll = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=self.part_tree.xview)
        self.part_tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.part_tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        self._configure_tree_rows(self.part_tree)
        self.part_tree.bind("<<TreeviewSelect>>", lambda _event: self._part_selection_changed())
        self.part_tree.bind("<Button-1>", self._part_click)
        self.part_tree.bind("<Motion>", self._part_motion)
        self.part_tree.bind("<Leave>", self._part_leave)
        self.part_tree.bind("<Double-1>", self._part_double_click)
        self.part_tree.bind("<MouseWheel>", lambda _event: self._close_tree_combo_editor(), add="+")
        self.part_tree.bind("<Shift-MouseWheel>", lambda _event: self._close_tree_combo_editor(), add="+")
        self.part_tree.bind("<Button-4>", lambda _event: self._close_tree_combo_editor(), add="+")
        self.part_tree.bind("<Button-5>", lambda _event: self._close_tree_combo_editor(), add="+")
        self.part_tree.bind("<Configure>", lambda _event: self._close_tree_combo_editor(), add="+")

    def _build_right_panel(self, parent: ttk.Frame) -> None:
        parent.rowconfigure(0, weight=1)
        self.viewer_holder = ttk.Frame(parent)
        self.viewer_holder.grid(row=0, column=0, sticky="nsew")
        self.viewer_holder.columnconfigure(0, weight=1)
        self.viewer_holder.rowconfigure(0, weight=1)
        ttk.Label(self.viewer_holder, text="Load a vehicle zip to use the built-in part viewer").grid(row=0, column=0)

        controls = ttk.LabelFrame(parent, text="Build Settings", padding=8)
        controls.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        controls.columnconfigure(1, weight=1)

        ttk.Label(controls, text="Config").grid(row=0, column=0, sticky="w")
        self.preview_output_combo = ttk.Combobox(
            controls,
            textvariable=self.preview_output_var,
            state="disabled",
            width=28,
        )
        self.preview_output_combo.grid(row=0, column=1, columnspan=2, sticky="ew", padx=(8, 0))
        self.preview_output_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._preview_output_selected(),
        )
        self._wire_preview_output_popdown()

        ttk.Label(controls, text="Auto delta X").grid(row=1, column=0, sticky="w", pady=(6, 0))
        delta_row = ttk.Frame(controls)
        delta_row.grid(row=1, column=1, columnspan=2, sticky="ew", padx=(8, 0), pady=(6, 0))
        delta_row.columnconfigure(0, weight=1)
        ttk.Label(delta_row, textvariable=self.auto_delta_var).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(
            delta_row,
            text="Manual magnitude",
            variable=self.manual_delta_enabled,
            command=self._manual_delta_toggled,
        ).grid(row=0, column=1, sticky="e", padx=(12, 0))
        self.manual_delta_entry = ttk.Entry(delta_row, textvariable=self.manual_delta_var, width=12)
        self.manual_delta_entry.grid(row=0, column=2, sticky="e", padx=(6, 0))
        self.manual_delta_entry.bind("<FocusOut>", lambda _event: self._commit_delta_from_ui())
        self.manual_delta_entry.bind("<Return>", lambda _event: self._commit_delta_from_ui())

        ttk.Label(controls, text="Licence plates").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.plate_summary_var = tk.StringVar(value="Off")
        plate_row = ttk.Frame(controls)
        plate_row.grid(row=2, column=1, columnspan=2, sticky="ew", padx=(8, 0), pady=(6, 0))
        plate_row.columnconfigure(0, weight=1)
        self.plate_choice_combo = ttk.Combobox(
            plate_row,
            textvariable=self.plate_choice_var,
            state="readonly",
        )
        self.plate_choice_combo.grid(row=0, column=0, sticky="ew")
        self.plate_choice_combo.bind("<<ComboboxSelected>>", lambda _event: self._main_plate_choice_changed())
        self.plate_configure_button = ttk.Button(plate_row, text="Configure...", command=lambda: self._open_plate_editor(None))
        self.plate_configure_button.grid(row=0, column=1, sticky="e", padx=(6, 0))
        ttk.Button(plate_row, text="Library...", command=self._open_plate_library).grid(
            row=0, column=2, sticky="e", padx=(6, 0)
        )

        ttk.Label(controls, text="Mods folder").grid(row=3, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(controls, textvariable=self.mods_folder_var).grid(row=3, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))
        ttk.Button(controls, text="Browse", command=self._browse_mods_folder).grid(row=3, column=2, sticky="e", padx=(6, 0), pady=(6, 0))

        ttk.Label(controls, text="Blender exe").grid(row=4, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(controls, textvariable=self.blender_var).grid(row=4, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))
        ttk.Button(controls, text="Browse", command=self._browse_blender).grid(row=4, column=2, sticky="e", padx=(6, 0), pady=(6, 0))

        buttons = ttk.Frame(parent)
        buttons.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        buttons.columnconfigure((0, 1), weight=1)
        self.install_button = ttk.Button(buttons, text="Build + Install", command=lambda: self._start_build(install=True))
        self.install_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.blender_button = ttk.Button(buttons, text="Blender Preview", command=self._start_blender_preview)
        self.blender_button.grid(row=0, column=1, sticky="ew", padx=(4, 0))

    def _open_zip_dialog(self) -> None:
        initial = existing_initial_dir(self.settings.get("lastVehicleZipFolder"), core.WORKSPACE_DIR)
        path = self._ask_open_filename(
            title="Open BeamNG vehicle zip",
            initialdir=initial,
            filetypes=(("Zip files", "*.zip"), ("All files", "*.*")),
        )
        if path:
            self._load_source_zip(Path(path))

    # ----- Model dropdown history -----------------------------------------

    def _recent_vehicle_entries(self) -> list[tuple[Path, str]]:
        """Persisted (zip, vehicle id) history, newest first, malformed rows
        dropped. Missing zips are kept so the history survives an unplugged
        drive; they are handled when actually selected."""
        recent = self.settings.get("recentVehicles")
        if not isinstance(recent, list):
            return []
        entries: list[tuple[Path, str]] = []
        for item in recent:
            if not isinstance(item, dict):
                continue
            zip_str = str(item.get("zip") or "")
            vehicle_id = str(item.get("vehicleId") or "")
            if zip_str and vehicle_id:
                entries.append((Path(zip_str), vehicle_id))
        return entries

    def _record_recent_vehicle(self, source_zip: Path, vehicle_id: str) -> None:
        zip_str = str(source_zip)
        recent = self.settings.get("recentVehicles")
        if not isinstance(recent, list):
            recent = []
        deduped = [
            item
            for item in recent
            if isinstance(item, dict)
            and not (str(item.get("zip")) == zip_str and str(item.get("vehicleId")) == vehicle_id)
        ]
        deduped.insert(0, {"zip": zip_str, "vehicleId": vehicle_id})
        self.settings["recentVehicles"] = deduped[:MODEL_HISTORY_LIMIT]

    def _prune_recent_vehicle(self, source_zip: Path, vehicle_id: str) -> None:
        zip_str = str(source_zip)
        recent = self.settings.get("recentVehicles")
        if not isinstance(recent, list):
            return
        self.settings["recentVehicles"] = [
            item
            for item in recent
            if isinstance(item, dict)
            and not (str(item.get("zip")) == zip_str and str(item.get("vehicleId")) == vehicle_id)
        ]
        core.save_app_settings(self.settings)

    @staticmethod
    def _model_history_label(zip_path: Path, vehicle_id: str, taken: dict[str, object]) -> str:
        label = f"{vehicle_id}  ({zip_path.stem})"
        base = label
        suffix = 2
        while label in taken:
            label = f"{base} #{suffix}"
            suffix += 1
        return label

    def _rebuild_model_combo(self) -> None:
        """Rebuild the Model dropdown to hold the currently-open zip's vehicles
        plus recently-opened (zip, vehicle) combos, and remember which load each
        label maps to. Current-zip vehicles keep bare vehicle-id labels so the
        existing load path (which reads the id straight off the combo) is
        unchanged; cross-zip history entries are labelled with the zip stem."""
        entries: dict[str, tuple[Path, str]] = {}
        values: list[str] = []
        current_zip = str(self.source_zip) if self.source_zip is not None else None
        if self.source_zip is not None:
            for vid in self.vehicle_ids:
                if vid in entries:
                    continue
                entries[vid] = (self.source_zip, vid)
                values.append(vid)
        for zip_path, vid in self._recent_vehicle_entries():
            if current_zip is not None and str(zip_path) == current_zip and vid in self.vehicle_ids:
                continue  # already represented by the open zip's bare label
            label = self._model_history_label(zip_path, vid, entries)
            entries[label] = (zip_path, vid)
            values.append(label)
        self.model_entries = entries
        self.vehicle_combo.configure(values=values)
        self._update_model_combo_state()

    def _update_model_combo_state(self) -> None:
        count = len(self.vehicle_combo.cget("values"))
        if self.model_load_busy or count < 2:
            self.vehicle_combo.configure(state="disabled")
        else:
            self.vehicle_combo.configure(state="readonly")

    def _on_model_selected(self) -> None:
        label = self.vehicle_var.get()
        entry = self.model_entries.get(label)
        if entry is None:
            # Bare vehicle id from the open zip (older/direct path).
            self._load_selected_vehicle()
            return
        zip_path, vehicle_id = entry
        if self.source_zip is not None and str(zip_path) == str(self.source_zip):
            self.vehicle_var.set(vehicle_id)  # bare label for the load path
            self._load_selected_vehicle()
            return
        if not zip_path.exists():
            self._show_error(
                "Vehicle unavailable",
                f"This zip no longer exists and was removed from history:\n{zip_path}",
            )
            self._prune_recent_vehicle(zip_path, vehicle_id)
            # Restore the dropdown to the loaded vehicle and refresh the list.
            if self.context is not None:
                self.vehicle_var.set(self.context.vehicle_id)
            self._rebuild_model_combo()
            return
        self._load_source_zip(zip_path, vehicle_id)

    def _load_source_zip(self, source_zip: Path, vehicle_id: str | None = None) -> None:
        try:
            vehicle_ids = core.vehicle_ids_in_zip(source_zip)
            if not vehicle_ids:
                raise RuntimeError("No vehicles/<model>/ content with DAE/PC/JBeam files was found")
            self.source_zip = source_zip
            self.settings["lastVehicleZipFolder"] = str(source_zip.parent)
            core.save_app_settings(self.settings)
            self.vehicle_ids = vehicle_ids
            self.source_var.set(str(source_zip))
            selected_vehicle = vehicle_id if vehicle_id in vehicle_ids else vehicle_ids[0]
            self.vehicle_var.set(selected_vehicle)
            self._rebuild_model_combo()
            self._load_selected_vehicle()
        except Exception as exc:
            self._show_error("Open zip failed", str(exc))
            self.status_var.set("Open zip failed")

    def _load_selected_vehicle(self, *, force_reload: bool = False) -> None:
        if self.source_zip is None:
            return
        self._cancel_structural_prompt()
        vehicle_id = self.vehicle_var.get() or (self.vehicle_ids[0] if self.vehicle_ids else None)
        if not vehicle_id:
            return
        self.vehicle_load_seq += 1
        seq = self.vehicle_load_seq
        if force_reload:
            self.status_var.set(f"Re-scanning vehicles/{vehicle_id} (ignoring cache)...")
        else:
            self.status_var.set(f"Loading vehicles/{vehicle_id}...")
        self._set_load_busy(True)
        worker = threading.Thread(
            target=self._vehicle_load_worker,
            args=(self.source_zip, vehicle_id, force_reload, seq),
            daemon=True,
        )
        worker.start()

    def _vehicle_load_worker(
        self,
        source_zip: Path,
        vehicle_id: str,
        force_reload: bool,
        seq: int,
    ) -> None:
        try:
            context = core.load_vehicle_context(source_zip, vehicle_id, use_cache=not force_reload)
            if force_reload:
                core.clear_parts_cache(context)
                core.clear_variant_hands_cache(context)
            conversion, loaded = core.load_or_create_conversion(context)
            self.worker_queue.put(
                ("vehicle_load_success", (seq, source_zip, vehicle_id, context, conversion, loaded))
            )
        except Exception as exc:
            self.worker_queue.put(("vehicle_load_error", (seq, exc)))

    def _set_load_busy(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        self.open_button.configure(state=state)
        self.refresh_button.configure(state="disabled" if busy or self.context is None else "normal")
        self.recommend_button.configure(state="disabled" if busy or self.context is None else "normal")
        self.show_all_parts_button.configure(state="disabled" if busy or self.context is None else "normal")
        self.hide_all_parts_button.configure(state="disabled" if busy or self.context is None else "normal")
        self.model_load_busy = busy
        self._update_model_combo_state()
        self._set_busy(busy)

    def _handle_vehicle_load_success(self, payload: object) -> None:
        seq, source_zip, vehicle_id, context, conversion, loaded = payload
        if seq != self.vehicle_load_seq:
            return
        self.context = context
        self.conversion = conversion
        self.preview_output_var.set("")
        cached_hands = core.load_cached_variant_hands(context, conversion) or {}
        self.variant_detected_hands = cached_hands
        self.variant_detection_complete = all(name in cached_hands for name in context.variants)
        self.variant_detection_pending = False
        self.settings["lastVehicleZipPath"] = str(source_zip)
        self.settings["lastVehicleId"] = vehicle_id
        self._record_recent_vehicle(source_zip, vehicle_id)
        core.save_app_settings(self.settings)
        self.vehicle_var.set(vehicle_id)
        self._rebuild_model_combo()
        self.part_refresh_seq += 1
        self.resolved_part_ids = []
        self.current_part_ids = []
        self.mesh_scene_hash = None
        self.mesh_scene_reset_pending = True
        self._set_load_busy(False)
        self._sync_delta_to_ui()
        self._sync_plate_to_ui()
        self._replace_viewer()
        self._refresh_all(reset_view=True)
        if not self.variant_detection_complete:
            self._schedule_variant_detection()
        self._schedule_mesh_scene(immediate=True)
        loaded_text = "loaded exact project config" if loaded else "new project config"
        self.project_var.set(f"Project: {context.project_dir} ({loaded_text})")
        from_cache = " (from cache)" if getattr(context, "loaded_from_cache", False) else ""
        self.status_var.set(
            f"Loaded {context.vehicle_id}{from_cache}: {len(context.variants)} variant(s), "
            f"{len(context.objects)} DAE object(s)"
        )

    def _handle_vehicle_load_error(self, payload: object) -> None:
        seq, exc = payload
        if seq != self.vehicle_load_seq:
            return
        self._set_load_busy(False)
        self._show_error("Load vehicle failed", str(exc))
        self.status_var.set("Load vehicle failed")

    def _replace_viewer(self) -> None:
        if self.viewer is not None and self.viewer_supports_scene:
            try:
                self.viewer.destroy()  # releases the GL context
            except Exception:
                pass
        for child in self.viewer_holder.winfo_children():
            child.destroy()
        self.viewer = None
        self.viewer_supports_scene = False
        if self.context is None:
            return
        if mesh_preview is not None:
            try:
                self.viewer = mesh_preview.MeshPreview(self.viewer_holder)
                self.viewer_supports_scene = True
                self.viewer.on_pick = self._on_preview_pick
                self.viewer.set_message("building preview...")
            except Exception as exc:
                print(f"[preview] GPU mesh preview unavailable ({exc}); using box preview")
                self.viewer = None
        if self.viewer is None:
            # The box viewer reads this dict live, so it is refreshed in place
            # whenever the previewed trim changes (see _refresh_box_preview).
            self._refresh_box_preview()
            self.viewer = ModelPreview(self.viewer_holder, self.box_preview_by_id)
        self.viewer.grid(row=0, column=0, sticky="nsew")

    def _sync_delta_to_ui(self) -> None:
        delta = self.conversion.get("delta", {})
        if not isinstance(delta, dict):
            delta = {}
            self.conversion["delta"] = delta
        self.manual_delta_enabled.set(bool(delta.get("manual")))
        magnitude = delta.get("magnitude")
        self.manual_delta_var.set("" if magnitude in (None, "") else fmt_float(abs(float(magnitude))))
        self._manual_delta_toggled(refresh=False)

    def _sync_plate_to_ui(self) -> None:
        self.conversion["plate"] = plate_generator.normalized_plate_binding(self.conversion.get("plate"))
        self._refresh_plate_choices()
        self._refresh_plate_summary()

    def _refresh_plate_choices(self) -> None:
        if not hasattr(self, "plate_choice_combo"):
            return
        records = plate_generator.plate_set_records()
        custom_label = self._vehicle_custom_label()
        self.plate_choice_to_id = {"Off": "", custom_label: ""}
        values = ["Off", custom_label]
        for record in records:
            label = str(record["name"])
            if label in self.plate_choice_to_id:
                label = f"{label} ({record['id']})"
            values.append(label)
            self.plate_choice_to_id[label] = str(record["id"])
        self.plate_choice_combo.configure(values=values)
        binding = plate_generator.normalized_plate_binding(self.conversion.get("plate"))
        if binding["mode"] == plate_generator.PLATE_MODE_SET:
            set_id = str(binding.get("setId") or "")
            selected = next((label for label, value in self.plate_choice_to_id.items() if value == set_id), f"Missing set: {set_id}")
        elif binding["mode"] == plate_generator.PLATE_MODE_CUSTOM:
            selected = custom_label
        else:
            selected = "Off"
        self.plate_choice_var.set(selected)
        self.plate_configure_button.configure(state="disabled" if selected == "Off" else "normal")

    def _main_plate_choice_changed(self) -> None:
        label = self.plate_choice_var.get()
        custom_label = self._vehicle_custom_label()
        binding = plate_generator.normalized_plate_binding(self.conversion.get("plate"))
        if label == "Off":
            binding["mode"] = plate_generator.PLATE_MODE_OFF
            binding["setId"] = ""
        elif label == custom_label:
            binding["mode"] = plate_generator.PLATE_MODE_CUSTOM
            binding["setId"] = ""
            binding["customDefined"] = True
            binding["config"] = plate_generator.normalized_plate_config(binding.get("customConfig"))
        else:
            set_id = self.plate_choice_to_id.get(label, "")
            record = plate_generator.plate_set_by_id(set_id)
            if record is not None:
                binding["mode"] = plate_generator.PLATE_MODE_SET
                binding["setId"] = set_id
                binding["config"] = plate_generator.normalized_plate_config(record.get("config"))
        self.conversion["plate"] = binding
        self._refresh_plate_summary()
        self._refresh_plate_choices()
        self._refresh_variants()
        self._update_detail()
        self.status_var.set(f"Licence plates: {plate_generator.plate_summary_label(self.conversion)}")

    def _vehicle_custom_label(self) -> str:
        vehicle_id = self.context.vehicle_id if self.context is not None else "vehicle"
        return f"Custom ({vehicle_id})"

    def _vehicle_plate_label(self) -> str:
        binding = plate_generator.normalized_plate_binding(self.conversion.get("plate"))
        if binding["mode"] == plate_generator.PLATE_MODE_CUSTOM:
            return self._vehicle_custom_label()
        if binding["mode"] == plate_generator.PLATE_MODE_SET:
            set_id = str(binding.get("setId") or "")
            record = plate_generator.plate_set_by_id(set_id)
            return f"Set: {record['name']} (vehicle)" if record else f"Missing set: {set_id} (vehicle)"
        return "Off (vehicle)"

    def _refresh_all(self, *, reset_view: bool = False) -> None:
        self._refresh_variants()
        self._schedule_parts_refresh(reset_view=reset_view)
        self._refresh_delta_label()
        self._update_detail()

    def _refresh_variants(self) -> None:
        if self.context is None:
            return
        self._close_tree_combo_editor()
        keep = set(self.variant_tree.selection())
        previous_order = list(self.variant_tree.get_children(""))
        for item in self.variant_tree.get_children():
            self.variant_tree.delete(item)
        variants = self.conversion.setdefault("variants", {})
        row_index = 0
        for config_name, variant in sorted(self.context.variants.items()):
            settings = variants.setdefault(
                config_name,
                {
                    "selected": False,
                    "build": core.BUILD_OFF,
                    "sourceHandOverride": core.HAND_AUTO,
                    "frontPlate": plate_generator.PLATE_PART_AUTO,
                    "rearPlate": plate_generator.PLATE_PART_AUTO,
                },
            )
            if not isinstance(settings, dict):
                continue
            detected = self._detected_hand_for_ui(config_name)
            build_mode = core.variant_build_mode(settings)
            core.set_variant_build_mode(settings, build_mode)
            stock_hand = (
                self._variant_stock_hand_label(config_name, settings, detected)
                if build_mode in {core.BUILD_CONVERTED, core.BUILD_BOTH}
                else "—"
            )
            self.variant_tree.insert(
                "",
                "end",
                iid=config_name,
                tags=self._row_tags(row_index),
                values=(
                    BUILD_LABELS[build_mode],
                    config_name,
                    variant.display_name,
                    stock_hand,
                    self._variant_plate_label(config_name, settings),
                    plate_generator.plate_part_label_for_config(
                        self.context,
                        config_name,
                        "front",
                        settings.get("frontPlate"),
                    ),
                    plate_generator.plate_part_label_for_config(
                        self.context,
                        config_name,
                        "rear",
                        settings.get("rearPlate"),
                    ),
                ),
            )
            row_index += 1
        self._restore_tree_order(self.variant_tree, previous_order)
        visible_keep = [item for item in keep if self.variant_tree.exists(item)]
        if visible_keep:
            self.variant_tree.selection_set(visible_keep)
        self._refresh_plate_summary()
        self._refresh_preview_outputs()

    def _variant_plate_label(self, config_name: str, settings: dict[str, object]) -> str:
        mode = plate_generator.variant_plate_mode(settings)
        if mode == plate_generator.PLATE_MODE_CUSTOM:
            return f"Custom ({config_name})"
        if mode == plate_generator.PLATE_MODE_TRIM:
            binding = plate_generator.normalized_plate_binding(settings.get("plate"), variant=True)
            return f"Custom ({binding.get('sourceConfig') or 'missing'})"
        if mode == plate_generator.PLATE_MODE_OFF:
            return "Off"
        if mode == plate_generator.PLATE_MODE_SET:
            binding = plate_generator.normalized_plate_binding(settings.get("plate"), variant=True)
            set_id = str(binding.get("setId") or "")
            record = plate_generator.plate_set_by_id(set_id)
            return f"Set: {record['name']}" if record else f"Missing set: {set_id}"
        return self._vehicle_plate_label()

    def _refresh_plate_summary(self) -> None:
        if hasattr(self, "plate_summary_var"):
            self.plate_summary_var.set(plate_generator.plate_summary_label(self.conversion))

    def _detected_hand_for_ui(self, config_name: str) -> str:
        return self.variant_detected_hands.get(config_name, "..." if not self.variant_detection_complete else core.HAND_UNKNOWN)

    def _variant_stock_hand_label(
        self,
        config_name: str,
        settings: dict[str, object],
        detected: str | None = None,
    ) -> str:
        detected = detected if detected is not None else self._detected_hand_for_ui(config_name)
        override = str(settings.get("sourceHandOverride", core.HAND_AUTO))
        if override in {core.HAND_LHD, core.HAND_RHD, core.HAND_UNKNOWN} and override != detected:
            return override
        if detected == "...":
            return "Detecting..."
        return f"{detected} (default)"

    def _variant_stock_hand_choices(
        self,
        config_name: str,
        settings: dict[str, object],
    ) -> tuple[list[str], dict[str, str]]:
        detected = self._detected_hand_for_ui(config_name)
        default_label = "Detecting..." if detected == "..." else f"{detected} (default)"
        mapping = {default_label: core.HAND_AUTO}
        for hand in (core.HAND_LHD, core.HAND_RHD):
            if hand != detected:
                mapping[hand] = hand
        override = str(settings.get("sourceHandOverride", core.HAND_AUTO))
        if override == core.HAND_UNKNOWN and core.HAND_UNKNOWN not in mapping:
            mapping[core.HAND_UNKNOWN] = core.HAND_UNKNOWN
        return list(mapping), mapping

    def _variant_output_name_for_ui(
        self,
        config_name: str,
        settings: dict[str, object],
        detected: str | None = None,
    ) -> str:
        detected = detected if detected is not None else self._detected_hand_for_ui(config_name)
        mode = core.variant_build_mode(settings)
        if mode == core.BUILD_OFF:
            return "skip"
        override = str(settings.get("sourceHandOverride", core.HAND_AUTO))
        source = override if override != core.HAND_AUTO else detected
        outputs: list[str] = []
        if mode in {core.BUILD_CONVERTED, core.BUILD_BOTH}:
            if source == "...":
                outputs.append("detecting")
            else:
                target = core.target_hand_for(source, core.ACTION_OPPOSITE)
                outputs.append("skip" if target is None else core.variant_output_name(config_name, target))
        if mode in {core.BUILD_ORIGINAL, core.BUILD_BOTH}:
            outputs.append(core.original_plate_output_name(config_name))
        return ", ".join(outputs)

    @staticmethod
    def _preview_config_label(output_name: str) -> str:
        return re.sub(r"_(?:rhd|lhd)$", "", output_name, flags=re.IGNORECASE)

    def _output_config_sources_for_ui(self) -> tuple[dict[str, str], dict[str, str]]:
        if self.context is None:
            return {}, {}
        variants = self.conversion.get("variants", {})
        if not isinstance(variants, dict):
            return {}, {}
        choices: dict[str, str] = {}
        outputs: dict[str, str] = {}

        def add_choice(config_name: str, output_name: str) -> None:
            label = self._preview_config_label(config_name)
            suffix = 2
            base = label
            while label in choices and choices.get(label) != config_name:
                label = f"{base} {suffix}"
                suffix += 1
            choices[label] = config_name
            outputs[label] = output_name

        for config_name, settings in variants.items():
            if config_name not in self.context.variants or not isinstance(settings, dict):
                continue
            detected = self._detected_hand_for_ui(config_name)
            mode = core.variant_build_mode(settings)
            if mode == core.BUILD_OFF:
                continue
            output_name = ""
            if mode in {core.BUILD_CONVERTED, core.BUILD_BOTH}:
                override = str(settings.get("sourceHandOverride", core.HAND_AUTO))
                source = override if override != core.HAND_AUTO else detected
                target = core.target_hand_for(source, core.ACTION_OPPOSITE)
                if target is not None:
                    output_name = core.variant_output_name(config_name, target)
            if not output_name and mode in {core.BUILD_ORIGINAL, core.BUILD_BOTH}:
                output_name = core.original_plate_output_name(config_name)
            if output_name:
                add_choice(config_name, output_name)
        return choices, outputs

    def _refresh_preview_outputs(self) -> None:
        if self.context is None or not hasattr(self, "preview_output_combo"):
            self.preview_output_to_config = {}
            self.preview_output_to_output = {}
            self.preview_output_var.set("")
            return
        current = self.preview_output_var.get()
        choices, outputs = self._output_config_sources_for_ui()
        self.preview_output_to_config = choices
        self.preview_output_to_output = outputs
        values = sorted(choices)
        self.preview_output_combo.configure(values=values)
        if current in choices:
            selected = current
        else:
            selected = self._cached_preview_output(choices, outputs)
            tree_selection = self.variant_tree.selection()
            if tree_selection:
                config_name = tree_selection[0]
                settings = self.conversion.get("variants", {}).get(config_name, {})
                output = (
                    self._variant_output_name_for_ui(config_name, settings)
                    if isinstance(settings, dict)
                    else ""
                )
                if not selected:
                    actual_outputs = {item.strip() for item in output.split(",") if item.strip()}
                    selected = next(
                        (label for label, actual in outputs.items() if actual in actual_outputs),
                        "",
                    )
            if not selected and values:
                selected = values[0]
        self.preview_output_var.set(selected)
        if self.worker_running or not values:
            self.preview_output_combo.configure(state="disabled")
        else:
            self.preview_output_combo.configure(state="readonly")

    def _preview_output_cache_key(self) -> str | None:
        if self.context is None:
            return None
        source = str(self.context.source_zip.resolve(strict=False))
        return f"{source}|{self.context.vehicle_id}"

    def _cached_preview_output(
        self,
        choices: dict[str, str],
        outputs: dict[str, str],
    ) -> str:
        key = self._preview_output_cache_key()
        cache = self.settings.setdefault("previewOutputByVehicle", {})
        if not key or not isinstance(cache, dict):
            return ""
        entry = cache.get(key)
        if isinstance(entry, dict):
            output = str(entry.get("output") or "")
            if output in choices:
                return output
            for label, actual_output in outputs.items():
                if actual_output == output:
                    return label
            config = str(entry.get("config") or "")
            if config:
                for label, source_config in choices.items():
                    if source_config == config:
                        return label
        elif isinstance(entry, str):
            if entry in choices:
                return entry
            for label, actual_output in outputs.items():
                if actual_output == entry:
                    return label
        return ""

    def _remember_preview_output(self, label: str | None = None) -> None:
        key = self._preview_output_cache_key()
        if key is None:
            return
        display_label = (label if label is not None else self.preview_output_var.get()).strip()
        config = self.preview_output_to_config.get(display_label)
        output = self.preview_output_to_output.get(display_label)
        if not display_label or not output or not config:
            return
        cache = self.settings.setdefault("previewOutputByVehicle", {})
        if not isinstance(cache, dict):
            cache = {}
            self.settings["previewOutputByVehicle"] = cache
        cache[key] = {"output": output, "config": config}
        core.save_app_settings(self.settings)

    def _preview_output_selected(self) -> None:
        self.preview_output_hover = None
        self._remember_preview_output()
        self._schedule_mesh_scene(immediate=True)
        # The x/y/z columns and the box viewer both show the previewed trim's
        # positions, so they have to follow the Config dropdown.
        self._refresh_box_preview()
        self._refresh_parts()

    def _selected_preview_output_name(self) -> str:
        label = (self.preview_output_hover or self.preview_output_var.get()).strip()
        return self.preview_output_to_output.get(label, "")

    def _wire_preview_output_popdown(self) -> None:
        """Hot-load trims while scrolling the Config dropdown. The ttk
        combobox popdown listbox is a plain Tcl widget with no Python wrapper;
        watch it via its <Map> event and poll the highlighted entry while it
        stays open."""
        combo = self.preview_output_combo
        try:
            popdown = str(combo.tk.call("ttk::combobox::PopdownWindow", combo))
            listbox = f"{popdown}.f.l"
            if not int(combo.tk.call("winfo", "exists", listbox)):
                return
            start = combo.register(self._start_preview_hover_watch)
            combo.tk.call("bind", listbox, "<Map>", f"+{start}")
        except tk.TclError:
            return
        self._preview_popdown_listbox = listbox

    def _start_preview_hover_watch(self) -> None:
        if self._preview_hover_after is not None:
            try:
                self.after_cancel(self._preview_hover_after)
            except Exception:
                pass
            self._preview_hover_after = None
        self._preview_hover_poll()

    def _preview_hover_poll(self) -> None:
        self._preview_hover_after = None
        combo = self.preview_output_combo
        listbox = self._preview_popdown_listbox
        mapped = False
        label = None
        if listbox is not None:
            try:
                mapped = bool(int(combo.tk.call("winfo", "ismapped", listbox)))
                if mapped:
                    selection = combo.tk.call(listbox, "curselection")
                    if selection:
                        index = selection[0] if isinstance(selection, (tuple, list)) else selection
                        label = str(combo.tk.call(listbox, "get", index))
            except tk.TclError:
                mapped = False
        if not mapped:
            self._end_preview_hover_watch()
            return
        if (
            label
            and label in self.preview_output_to_config
            and label != (self.preview_output_hover or self.preview_output_var.get())
        ):
            self.preview_output_hover = label
            self._schedule_mesh_scene(immediate=True)
        self._preview_hover_after = self.after(90, self._preview_hover_poll)

    def _end_preview_hover_watch(self) -> None:
        if self.preview_output_hover is None:
            return
        self.preview_output_hover = None
        # Confirming fires <<ComboboxSelected>> with the same trim already
        # loaded (snapshot-guarded no-op); after a cancel this restores the
        # preview of the actual selection.
        self._schedule_mesh_scene(immediate=True)

    def _variant_detection_signature(self) -> tuple[str, ...]:
        return core.variant_hand_detection_signature(self.conversion)

    def _invalidate_variant_detection(self) -> None:
        self.variant_detected_hands = {}
        self.variant_detection_complete = False
        self.mesh_scene_hash = None

    def _schedule_variant_detection(self) -> None:
        if self.context is None:
            return
        if self.variant_detection_complete:
            return
        if self.variant_detection_running:
            self.variant_detection_pending = True
            return
        self._start_variant_detection()

    def _start_variant_detection(self) -> None:
        if self.context is None:
            return
        self.variant_detection_running = True
        self.variant_detection_pending = False
        self.variant_detection_seq += 1
        seq = self.variant_detection_seq
        context = self.context
        signature = self._variant_detection_signature()
        conversion_copy = json.loads(json.dumps(self.conversion, default=str))
        future = self.variant_detector.submit(self._variant_detection_worker, context, conversion_copy)
        future.add_done_callback(
            lambda completed, current_seq=seq, current_context=context, current_signature=signature: self.worker_queue.put(
                ("variant_hands_done", (current_seq, current_context, current_signature, completed))
            )
        )

    @staticmethod
    def _variant_detection_worker(
        context: core.VehicleContext,
        conversion: dict[str, object],
    ) -> dict[str, str]:
        return core.detect_hands_for_variants(context, conversion)

    def _handle_variant_hands_done(self, payload: object) -> None:
        seq, context, signature, completed = payload
        self.variant_detection_running = False
        should_apply = (
            seq == self.variant_detection_seq
            and context is self.context
            and signature == self._variant_detection_signature()
        )
        try:
            detected = completed.result()
        except Exception as exc:
            if should_apply:
                self.variant_detected_hands = {}
                self.variant_detection_complete = True
                self._refresh_variants()
                self.status_var.set(f"Trim handedness detection failed: {exc}")
            if self.variant_detection_pending:
                self._schedule_variant_detection()
            return
        if should_apply:
            self.variant_detected_hands = {
                config_name: hand
                for config_name, hand in detected.items()
                if hand in {core.HAND_LHD, core.HAND_RHD, core.HAND_UNKNOWN}
            }
            self.variant_detection_complete = True
            self.mesh_scene_hash = None
            self._refresh_variants()
            self._schedule_mesh_scene(immediate=True)
        if self.variant_detection_pending:
            self._schedule_variant_detection()

    def _table_position(self, object_id: str) -> tuple[tuple[float, float, float], bool]:
        """Where the part's geometry actually sits, and whether that varies by trim.

        These are the drawn mesh's centre, not its DAE pivot. The pivot is
        meaningless for meshes authored in vehicle space with an identity node
        matrix -- a whole column of engine parts read 0,0,0 while rendering
        correctly. The box preview data is already the placed geometry for the
        trim on screen, so it is the same number the viewer draws."""
        if self.context is None:
            return ((0.0, 0.0, 0.0), False)
        obj = self.context.objects.get(object_id)
        if obj is None:
            return ((0.0, 0.0, 0.0), False)
        varies = object_id in self.context.variant_dependent_meshes
        entry = self.box_preview_by_id.get(object_id) or self.context.preview_by_id.get(object_id)
        centre = (entry or {}).get("center")
        if centre is not None:
            return (tuple(float(value) for value in centre), varies)
        # No geometry (prop-only rows with no mesh in the DAE): fall back to
        # the resolved pivot rather than showing nothing.
        config = self._mesh_scene_config()
        if config is not None and config in self.context.variants:
            try:
                resolved = core.resolved_mesh_positions_for_config(self.context, config).get(object_id)
            except Exception:
                resolved = None
            if resolved is not None:
                return (resolved.position, varies)
        return ((obj.x, obj.y, obj.z), varies)

    def _refresh_box_preview(self, *, force: bool = False) -> None:
        """Re-point the placed-geometry data at the previewed trim.

        Feeds the fallback box viewer and the table's x/y/z columns (the GPU
        scene builds its own geometry per config). Updated in place because
        ModelPreview holds the dict by reference, and skipped when the trim has
        not changed since it costs a pass over every mesh."""
        if self.context is None:
            self.box_preview_by_id.clear()
            self._box_preview_config = None
            return
        config = self._mesh_scene_config()
        if not force and config == self._box_preview_config and self.box_preview_by_id:
            return
        self._box_preview_config = config
        self.box_preview_by_id.clear()
        if config is None or config not in self.context.variants:
            self.box_preview_by_id.update(self.context.preview_by_id)
            return
        try:
            self.box_preview_by_id.update(
                core.preview_entries_for_config(self.context, config)
            )
        except Exception:
            self.box_preview_by_id.update(self.context.preview_by_id)

    def _variant_position_note(self, object_id: str) -> str:
        """Where else this part's geometry sits, for the marked (*) rows.

        Geometry centres, matching the x/y/z columns -- quoting pivots here
        would contradict them."""
        if self.context is None:
            return ""
        baked = (self.context.preview_by_id.get(object_id) or {}).get("center")
        obj = self.context.objects.get(object_id)
        if baked is None or obj is None:
            return ""
        representative = (obj.x, obj.y, obj.z)
        by_position: dict[tuple[float, ...], list[str]] = {}
        for config_name in sorted(self.context.variants):
            try:
                resolved = core.resolved_mesh_positions_for_config(self.context, config_name)
            except Exception:
                continue
            entry = resolved.get(object_id)
            if entry is None:
                continue
            # Same shift preview_entries_for_config applies, for one mesh:
            # building the whole per-config mapping here would copy every
            # mesh once per trim.
            key = tuple(
                round(float(baked[i]) + entry.position[i] - representative[i], 4)
                for i in range(3)
            )
            by_position.setdefault(key, []).append(config_name)
        if len(by_position) < 2:
            return ""
        shown = sorted(by_position.items(), key=lambda item: -len(item[1]))[:3]
        parts = [f"x {key[0]:.4f} y {key[1]:.4f} ({len(cfgs)} trims)" for key, cfgs in shown]
        more = "" if len(by_position) <= 3 else f", +{len(by_position) - 3} more"
        return f" | * varies by trim: {'; '.join(parts)}{more}"

    def _refresh_parts(self, *, reset_view: bool = False) -> None:
        if self.context is None:
            return
        query = self.filter_var.get().strip().lower()
        # The x/y/z columns read placed geometry, so make sure it matches the
        # trim on screen before the rows are built.
        self._refresh_box_preview()
        keep = set(self.part_tree.selection())
        previous_order = list(self.part_tree.get_children(""))
        for item in self.part_tree.get_children():
            self.part_tree.delete(item)

        parts = self.conversion.setdefault("parts", {})
        ids = self.resolved_part_ids
        active_ids = self._preview_active_ids()
        displayed: list[str] = []
        row_index = 0
        for object_id in ids:
            obj = self.context.objects.get(object_id)
            if obj is None:
                continue
            settings = parts.setdefault(
                object_id,
                {
                    "mode": core.MODE_SKIP,
                    "mirrorSource": None,
                    "translateOffset": None,
                    "textureFlip": False,
                    "steeringRef": False,
                    "viewerVisible": True,
                    "viewerSolo": False,
                },
            )
            if not isinstance(settings, dict):
                continue
            mode = str(settings.get("mode", core.MODE_SKIP))
            display_name = self._part_display_name(object_id)
            if (
                query
                and query not in object_id.lower()
                and query not in display_name.lower()
                and query not in mode
            ):
                continue
            displayed.append(object_id)
            self.part_tree.insert(
                "",
                "end",
                iid=object_id,
                text=display_name,
                tags=self._row_tags(row_index),
                values=(
                    yn_label(settings.get("viewerVisible", True)),
                    yn_label(settings.get("viewerSolo")),
                    yn_label(object_id in active_ids),
                    mode_label(mode),
                    offset_display(
                        mode,
                        settings.get("translateOffset"),
                        manual_delta=self.manual_delta_enabled.get(),
                    ),
                    fliptex_display(mode, settings.get("textureFlip")),
                    yn_label(settings.get("steeringRef")),
                    *position_labels(*self._table_position(object_id)),
                ),
            )
            row_index += 1
        self.current_part_ids = displayed
        self._restore_tree_order(self.part_tree, previous_order)
        visible_keep = [item for item in keep if self.part_tree.exists(item)]
        if visible_keep:
            self.part_tree.selection_set(visible_keep)
        self._refresh_viewer(reset=reset_view)

    def _schedule_parts_refresh(self, *, reset_view: bool = False) -> None:
        self.part_refresh_pending_reset = self.part_refresh_pending_reset or reset_view
        if self.part_refresh_running:
            self.part_refresh_pending = True
            self.part_refresh_seq += 1
            return
        if self.part_refresh_after_id is not None:
            return
        self.part_refresh_after_id = self.after_idle(self._run_scheduled_parts_refresh)

    def _run_scheduled_parts_refresh(self) -> None:
        self.part_refresh_after_id = None
        reset_view = self.part_refresh_pending_reset
        self.part_refresh_pending = False
        self.part_refresh_pending_reset = False
        self._start_parts_refresh(reset_view=reset_view)

    def _start_parts_refresh(self, *, reset_view: bool = False) -> None:
        self.part_refresh_after_id = None
        if self.context is None:
            self.resolved_part_ids = []
            self._refresh_parts(reset_view=reset_view)
            return
        selected = tuple(self._selected_variant_names())
        self.part_refresh_seq += 1
        seq = self.part_refresh_seq
        if not selected:
            self.resolved_part_ids = []
            self._refresh_parts(reset_view=reset_view)
            self.status_var.set("No trims selected; 0 used part(s) displayed")
            return
        context = self.context
        cached_ids = core.load_cached_part_ids(context, selected)
        if cached_ids is not None:
            self.resolved_part_ids = [part_id for part_id in cached_ids if part_id in context.objects]
            self._refresh_parts(reset_view=reset_view)
            self._update_detail()
            self.status_var.set(f"{len(self.current_part_ids)} used part(s) displayed (parts cache)")
            return
        self.status_var.set(f"Resolving used parts for {len(selected)} trim(s)...")
        self.part_refresh_running = True
        future = self.part_resolver.submit(self._resolve_part_ids_worker, context, selected)
        future.add_done_callback(
            lambda completed, current_seq=seq, current_context=context, should_reset=reset_view, current_selected=selected: self.worker_queue.put(
                ("parts_success", (current_seq, current_context, should_reset, current_selected, completed))
            )
        )

    @staticmethod
    def _resolve_part_ids_worker(
        context: core.VehicleContext,
        selected: tuple[str, ...],
    ) -> list[str]:
        _flex, _props, all_meshes = core.selected_mesh_roles(context, list(selected))
        return sorted(mesh for mesh in all_meshes if mesh in context.objects)

    def _selected_variant_names(self) -> list[str]:
        if self.context is None:
            return []
        variants = self.conversion.get("variants", {})
        if not isinstance(variants, dict):
            return []
        return [
            name
            for name, settings in variants.items()
            if name in self.context.variants
            and isinstance(settings, dict)
            and core.variant_build_mode(settings) != core.BUILD_OFF
        ]

    def _preview_base_part_ids(self) -> list[str]:
        if self.context is None:
            return []
        return [
            object_id
            for object_id in (self.resolved_part_ids or self.current_part_ids)
            if object_id in self.context.objects
        ]

    def _resolved_visible_ids(self) -> set[str]:
        """The set of parts actually present in the active preview / final
        visible output for the current variant selection: Solo (if any part is
        soloed) or per-part Visible toggles, over the resolved used-part set.
        Table selection deliberately has no effect here -- Visible/Solo have the
        final say over what the preview and the converted output contain."""
        if self.context is None:
            return set()
        parts = self.conversion.get("parts", {})
        base_ids = self._preview_base_part_ids()
        solo_ids = {
            object_id
            for object_id in base_ids
            if isinstance(parts, dict)
            and isinstance(parts.get(object_id), dict)
            and parts[object_id].get("viewerSolo")
        }
        if solo_ids:
            return solo_ids
        return {
            object_id
            for object_id in base_ids
            if not isinstance(parts, dict)
            or not isinstance(parts.get(object_id), dict)
            or parts[object_id].get("viewerVisible", True)
        }

    def _refresh_viewer(self, *, reset: bool = False) -> None:
        if self.viewer is None:
            return
        visible_ids = self._resolved_visible_ids()
        # Selected inactive parts are temporarily injected into the GPU scene
        # (scene.extra); show them while they stay selected. Intersecting with
        # the live selection hides a stale extra instantly after deselection,
        # before the scene rebuild that drops it has landed.
        scene = getattr(self.viewer, "scene", None)
        visible_ids |= set(getattr(scene, "extra", ()) or ()) & set(self.part_tree.selection())
        dimmed_ids = visible_ids - set(self.current_part_ids)
        self.viewer.set_visible_ids(list(visible_ids), reset=reset)
        if hasattr(self.viewer, "set_dimmed_ids"):
            self.viewer.set_dimmed_ids(dimmed_ids)
        # Selection only drives the highlight outline (skipped for hidden parts
        # in the renderer); it never adds a part to the visible set above.
        self.viewer.set_selected_ids(set(self.part_tree.selection()))

    def _preview_active_ids(self) -> set[str]:
        """Object ids present on the trim currently shown in the moderngl
        preview -- i.e. the config chosen in the Config dropdown. This
        indicates which parts the converted trim actually uses; it is NOT
        affected by the viewer Visible/Solo toggles (those only filter what is
        drawn). Ground truth is the built scene's mesh groups (keyed by object
        id, already excluding inactive/geometry-less rows for this config)."""
        scene = getattr(self.viewer, "scene", None) if self.viewer is not None else None
        groups = getattr(scene, "groups", None)
        if groups:
            # Temporarily-shown inactive parts (scene.extra) are in the scene
            # but not part of the previewed trim; they are never Active.
            return set(groups.keys()) - set(getattr(scene, "extra", ()) or ())
        # No GPU scene yet (box-viewer fallback, or the preview is still
        # building): resolve the previewed config's meshes directly. Roles are
        # cached per config on the context, so this stays cheap.
        if self.context is None:
            return set()
        config = self._mesh_scene_config()
        if config is None:
            return set()
        try:
            _flex, _props, all_meshes = core.selected_mesh_roles(self.context, [config])
        except Exception:
            return set()
        return {mesh for mesh in all_meshes if mesh in self.context.objects}

    def _selected_extra_preview_ids(self) -> list[str]:
        """Selected table parts NOT used by the previewed config. These get
        temporarily injected into the GPU scene so selecting an inactive part
        still shows it; deselecting removes it again. Active parts are never
        in this list, so their behaviour is unchanged."""
        if self.context is None or not hasattr(self, "part_tree"):
            return []
        config = self._mesh_scene_config()
        if config is None:
            return []
        try:
            _flex, _props, all_meshes = core.selected_mesh_roles(self.context, [config])
        except Exception:
            return []
        return sorted(
            object_id
            for object_id in self.part_tree.selection()
            if object_id in self.context.objects and object_id not in all_meshes
        )

    def _refresh_active_cells(self) -> None:
        """Update the parts table Active (Y/N) column for every displayed row to
        reflect the trim currently shown in the moderngl preview."""
        if not hasattr(self, "part_tree") or self.part_tree is None:
            return
        active_ids = self._preview_active_ids()
        for object_id in self.part_tree.get_children():
            self.part_tree.set(object_id, "active", yn_label(object_id in active_ids))

    def _refresh_delta_label(self) -> None:
        if self.context is None:
            self.auto_delta_var.set("")
            return
        auto = core.auto_delta_magnitude(self.context, self.conversion)
        source_refs = core.auto_delta_source_refs(self.context, self.conversion)
        if source_refs:
            names = ", ".join(self._part_display_name(object_id) for object_id in source_refs)
            source = f"found using {names}"
        else:
            # No steering ref selected (or the selected one has no usable
            # off-center X), so the auto delta is just its default.
            source = "no steering ref found"
        self.auto_delta_var.set(f"{fmt_float(auto)} ({source})")

    def _update_detail(self) -> None:
        if self.context is None:
            self.detail_var.set("")
            return
        # Every conversion mutation (mode, translate offset, structural pairing,
        # steering ref, manual delta, variant hand override) funnels through here
        # as its final UI step, so this is where we keep the GPU preview live.
        # _schedule_mesh_scene is snapshot-guarded: pure selection/visibility
        # changes leave the fingerprint unchanged and cost only a cheap compare.
        self._schedule_mesh_scene()
        selected_parts = self.part_tree.selection()
        if selected_parts:
            object_id = selected_parts[0]
            obj = self.context.objects.get(object_id)
            settings = self.conversion.get("parts", {}).get(object_id, {})
            if obj:
                display_name = self._part_display_name(object_id)
                mode = str(settings.get("mode", core.MODE_SKIP)) if isinstance(settings, dict) else core.MODE_SKIP
                part_offset = (
                    offset_display(
                        mode,
                        settings.get("translateOffset") if isinstance(settings, dict) else None,
                        manual_delta=self.manual_delta_enabled.get(),
                    )
                    if mode == core.MODE_TRANSLATE
                    else "N/A"
                )
                flip_note = (
                    ", texture flip on"
                    if mode == core.MODE_MIRROR
                    and isinstance(settings, dict)
                    and settings.get("textureFlip")
                    else ""
                )
                position, varies = self._table_position(object_id)
                self.detail_var.set(
                    f"{display_name}: {mode_label(mode)}{flip_note}, "
                    f"full id {object_id}, x {fmt_float(position[0])}, offset {part_offset}, "
                    f"dae {obj.dae_path}{self._variant_position_note(object_id) if varies else ''}"
                )
                return
        active = len(core.active_part_modes(self.conversion))
        selected_variants = len(self._selected_variant_names())
        self.detail_var.set(
            f"{len(self.current_part_ids)} displayed part(s), {active} transformed part setting(s), "
            f"{selected_variants} selected variant(s)"
        )

    def _set_all_variants_selected(self, selected: bool) -> None:
        if self.context is None:
            return
        variants = self.conversion.setdefault("variants", {})
        for config_name in self.context.variants:
            settings = variants.setdefault(config_name, {})
            if isinstance(settings, dict):
                core.set_variant_build_mode(settings, core.BUILD_CONVERTED if selected else core.BUILD_OFF)
        self._refresh_variants()
        self._schedule_parts_refresh(reset_view=True)
        self._refresh_delta_label()
        self._update_detail()
        self.status_var.set(
            f"{'All trims selected' if selected else 'All trims cleared'}; updating used parts..."
        )

    def _toggle_variant_selected(self, config_name: str) -> None:
        variants = self.conversion.setdefault("variants", {})
        settings = variants.setdefault(config_name, {})
        if isinstance(settings, dict):
            mode = core.variant_build_mode(settings)
            core.set_variant_build_mode(
                settings,
                core.BUILD_OFF if mode != core.BUILD_OFF else core.BUILD_CONVERTED,
            )
        self._refresh_variants()
        self._schedule_parts_refresh(reset_view=True)
        self._refresh_delta_label()
        self._update_detail()
        state = BUILD_LABELS[core.variant_build_mode(settings)] if isinstance(settings, dict) else "Off"
        self.status_var.set(f"{config_name} {state}; updating used parts...")

    def _variant_click(self, event: tk.Event) -> None:
        self._close_tree_combo_editor()
        if not self._tree_body_click(self.variant_tree, event):
            return None
        item = self.variant_tree.identify_row(event.y)
        column = self.variant_tree.identify_column(event.x)
        if not item or self.context is None:
            return
        name = self._tree_column_name(self.variant_tree, column)
        if name == "build":
            settings = self.conversion.setdefault("variants", {}).setdefault(item, {})
            current = BUILD_LABELS[core.variant_build_mode(settings)] if isinstance(settings, dict) else BUILD_LABELS[core.BUILD_OFF]
            self._edit_tree_combo(
                self.variant_tree,
                item,
                column,
                list(BUILD_LABELS.values()),
                current,
                lambda value: self._set_variant_build_label(item, value),
            )
            return "break"
        if name == "stock_hand":
            settings = self.conversion.get("variants", {}).get(item, {})
            if core.variant_build_mode(settings) not in {core.BUILD_CONVERTED, core.BUILD_BOTH}:
                return "break"
            labels, mapping = self._variant_stock_hand_choices(item, settings)
            self._edit_tree_combo(
                self.variant_tree,
                item,
                column,
                labels,
                self._variant_stock_hand_label(item, settings),
                lambda value: self._set_variant_setting(item, "sourceHandOverride", mapping[value]),
            )
            return "break"
        if name == "plate":
            labels, mapping = self._variant_plate_choices(item)
            current_label = self._variant_plate_label(item, self.conversion.get("variants", {}).get(item, {}))
            self._edit_tree_combo(
                self.variant_tree,
                item,
                column,
                labels,
                current_label,
                lambda value: self._set_variant_plate_choice(item, mapping[value]),
            )
            return "break"
        if name in {"front_plate", "rear_plate"}:
            side = "front" if name == "front_plate" else "rear"
            key = "frontPlate" if side == "front" else "rearPlate"
            choices = plate_generator.plate_part_choices_for_config(self.context, item, side)
            labels = [choice.label for choice in choices]
            values_by_label = {choice.label: choice.value for choice in choices}
            current_value = self._get_variant_setting(item, key, plate_generator.PLATE_PART_AUTO)
            current_label = plate_generator.plate_part_label_for_config(
                self.context,
                item,
                side,
                current_value,
            )
            self._edit_tree_combo(
                self.variant_tree,
                item,
                column,
                labels,
                current_label,
                lambda value: self._set_variant_setting(item, key, values_by_label[value]),
            )
            return "break"
        if name is not None:
            # Clicking the descriptive cells keeps the old quick on/off action.
            self._toggle_variant_selected(item)
            return "break"
        return None

    def _variant_double_click(self, event: tk.Event) -> None:
        self._close_tree_combo_editor()
        if not self._tree_body_click(self.variant_tree, event):
            return None
        item = self.variant_tree.identify_row(event.y)
        column = self.variant_tree.identify_column(event.x)
        if not item:
            return
        name = self._tree_column_name(self.variant_tree, column)
        if name == "plate":
            self._open_plate_editor(item)
        elif name == "stock_hand":
            settings = self.conversion.get("variants", {}).get(item, {})
            if core.variant_build_mode(settings) not in {core.BUILD_CONVERTED, core.BUILD_BOTH}:
                return "break"
            labels, mapping = self._variant_stock_hand_choices(item, settings)
            self._edit_tree_combo(
                self.variant_tree,
                item,
                column,
                labels,
                self._variant_stock_hand_label(item, settings),
                lambda value: self._set_variant_setting(item, "sourceHandOverride", mapping[value]),
            )

    def _variant_plate_choices(self, config_name: str) -> tuple[list[str], dict[str, tuple[str, str, str]]]:
        mapping: dict[str, tuple[str, str, str]] = {
            self._vehicle_plate_label(): (plate_generator.PLATE_MODE_GENERAL, "", ""),
            f"Custom ({config_name})": (plate_generator.PLATE_MODE_CUSTOM, "", config_name),
        }
        variants = self.conversion.get("variants", {})
        if isinstance(variants, dict):
            for source_name, source_settings in sorted(variants.items()):
                if source_name == config_name or not isinstance(source_settings, dict):
                    continue
                source_binding = plate_generator.normalized_plate_binding(
                    source_settings.get("plate"), variant=True
                )
                if not source_binding.get("customDefined"):
                    continue
                mapping[f"Custom ({source_name})"] = (
                    plate_generator.PLATE_MODE_TRIM,
                    "",
                    str(source_name),
                )
        for record in plate_generator.plate_set_records():
            label = f"Set: {record['name']}"
            if label in mapping:
                label = f"{label} ({record['id']})"
            mapping[label] = (plate_generator.PLATE_MODE_SET, str(record["id"]), "")
        mapping["Off"] = (plate_generator.PLATE_MODE_OFF, "", "")
        return list(mapping), mapping

    def _set_variant_build_label(self, config_name: str, label: str) -> None:
        mode = next((key for key, value in BUILD_LABELS.items() if value == label), core.BUILD_OFF)
        variants = self.conversion.setdefault("variants", {})
        settings = variants.setdefault(config_name, {})
        if isinstance(settings, dict):
            core.set_variant_build_mode(settings, mode)
        self._refresh_variants()
        self._schedule_parts_refresh(reset_view=True)
        self._refresh_delta_label()
        self._update_detail()

    def _set_variant_plate_choice(self, config_name: str, choice: tuple[str, str, str]) -> None:
        mode, set_id, source_config = choice
        settings = self.conversion.setdefault("variants", {}).setdefault(config_name, {})
        if not isinstance(settings, dict):
            return
        binding = plate_generator.normalized_plate_binding(settings.get("plate"), variant=True)
        previous_mode = str(binding.get("mode"))
        if mode == plate_generator.PLATE_MODE_CUSTOM:
            if previous_mode != plate_generator.PLATE_MODE_CUSTOM and not binding.get("customDefined"):
                if previous_mode == plate_generator.PLATE_MODE_SET:
                    record = plate_generator.plate_set_by_id(str(binding.get("setId") or ""))
                    copy_config = record.get("config") if record is not None else binding.get("config")
                elif previous_mode == plate_generator.PLATE_MODE_TRIM:
                    referenced = str(binding.get("sourceConfig") or "")
                    referenced_settings = self.conversion.get("variants", {}).get(referenced, {})
                    referenced_binding = plate_generator.normalized_plate_binding(
                        referenced_settings.get("plate") if isinstance(referenced_settings, dict) else None,
                        variant=True,
                    )
                    copy_config = referenced_binding.get("customConfig")
                else:
                    general = plate_generator.normalized_plate_binding(self.conversion.get("plate"))
                    copy_config = (
                        general.get("customConfig")
                        if general.get("mode") == plate_generator.PLATE_MODE_CUSTOM
                        else general.get("config")
                    )
                binding["customConfig"] = plate_generator.normalized_plate_config(copy_config)
            binding["config"] = plate_generator.normalized_plate_config(binding.get("customConfig"))
        binding["mode"] = mode
        binding["setId"] = set_id
        binding["sourceConfig"] = source_config
        if mode == plate_generator.PLATE_MODE_CUSTOM:
            binding["customDefined"] = True
        elif mode == plate_generator.PLATE_MODE_TRIM:
            source_settings = self.conversion.get("variants", {}).get(source_config, {})
            source_binding = plate_generator.normalized_plate_binding(
                source_settings.get("plate") if isinstance(source_settings, dict) else None,
                variant=True,
            )
            binding["config"] = plate_generator.normalized_plate_config(source_binding.get("customConfig"))
        if mode == plate_generator.PLATE_MODE_SET:
            record = plate_generator.plate_set_by_id(set_id)
            if record is not None:
                binding["config"] = plate_generator.normalized_plate_config(record.get("config"))
        settings["plate"] = binding
        self._refresh_variants()
        self._update_detail()
        if mode == plate_generator.PLATE_MODE_CUSTOM:
            self._open_plate_editor(config_name)

    def _open_plate_library(self) -> None:
        if self.plate_library_modal is not None and self.plate_library_modal.winfo_exists():
            self.plate_library_modal.lift()
            return
        self.plate_library_modal = PlateLibraryDialog(self)

    def _open_plate_editor(self, variant_name: str | None, *, set_id: str | None = None) -> None:
        if self.context is None and set_id is None:
            self._show_error("No source", "Open a vehicle zip first.")
            return
        if self.plate_editor_modal is not None and self.plate_editor_modal.winfo_exists():
            self.plate_editor_modal.lift()
            return
        if set_id is None:
            if variant_name is None:
                binding = plate_generator.normalized_plate_binding(self.conversion.get("plate"))
            else:
                settings = self.conversion.get("variants", {}).get(variant_name, {})
                binding = plate_generator.normalized_plate_binding(
                    settings.get("plate") if isinstance(settings, dict) else None,
                    variant=True,
                )
                if binding.get("mode") == plate_generator.PLATE_MODE_TRIM:
                    source_config = str(binding.get("sourceConfig") or "")
                    source_settings = self.conversion.get("variants", {}).get(source_config, {})
                    if not isinstance(source_settings, dict):
                        self._show_error(
                            "Missing custom plate settings",
                            f"Custom plate source '{source_config}' is no longer available. Choose a new plate option for this trim.",
                        )
                        return
                    variant_name = source_config
                    binding = plate_generator.normalized_plate_binding(source_settings.get("plate"), variant=True)
                elif binding.get("mode") == plate_generator.PLATE_MODE_GENERAL:
                    binding = plate_generator.normalized_plate_binding(self.conversion.get("plate"))
                    variant_name = None
                    if binding.get("mode") == plate_generator.PLATE_MODE_OFF:
                        self._show_error(
                            "Plates are off",
                            "Choose a custom or library plate option before configuring it.",
                        )
                        return
            if binding.get("mode") == plate_generator.PLATE_MODE_SET:
                set_id = str(binding.get("setId") or "")
        if set_id is not None and plate_generator.plate_set_by_id(set_id) is None:
            self._show_error(
                "Missing plate set",
                f"Plate set '{set_id}' was deleted. The build can still use its saved snapshot; choose Custom to edit that snapshot.",
            )
            return
        self.plate_editor_modal = PlateEditorDialog(self, variant_name if set_id is None else None, set_id=set_id)

    def _plate_settings_applied(self) -> None:
        self._sync_plate_to_ui()
        self._refresh_variants()
        if self.plate_library_modal is not None and self.plate_library_modal.winfo_exists():
            self.plate_library_modal.refresh()
        self._update_detail()
        self.status_var.set(f"Licence plate settings updated ({plate_generator.plate_summary_label(self.conversion)})")

    def _part_option_label(self, object_id: str) -> str:
        display = self._part_display_name(object_id)
        if display == object_id:
            return object_id
        return f"{display} ({object_id})"

    def _name_pair_candidate(self, object_id: str, candidates: list[str]) -> str | None:
        lower_to_id = {candidate.lower(): candidate for candidate in candidates}
        pairs = (
            ("_FL", "_FR"),
            ("_FR", "_FL"),
            ("_RL", "_RR"),
            ("_RR", "_RL"),
            ("_left", "_right"),
            ("_right", "_left"),
            ("_driver", "_passenger"),
            ("_passenger", "_driver"),
            ("_L", "_R"),
            ("_R", "_L"),
            ("-L", "-R"),
            ("-R", "-L"),
            (".L", ".R"),
            (".R", ".L"),
        )
        lowered = object_id.lower()
        for old, new in pairs:
            old_lower = old.lower()
            if old_lower not in lowered:
                continue
            candidate_lower = lowered.replace(old_lower, new.lower(), 1)
            if candidate_lower in lower_to_id:
                return lower_to_id[candidate_lower]
        return None

    def _geometry_pair_candidate(self, object_id: str, candidates: list[str]) -> str | None:
        if self.context is None or object_id not in self.context.objects:
            return None
        obj = self.context.objects[object_id]
        best: tuple[float, str] | None = None
        for candidate in candidates:
            if candidate == object_id:
                continue
            other = self.context.objects.get(candidate)
            if other is None:
                continue
            if abs(obj.x) > 0.02 and abs(other.x) > 0.02 and obj.x * other.x > 0:
                continue
            score = (
                abs(obj.x + other.x) * 4.0
                + abs(obj.y - other.y)
                + abs(obj.z - other.z)
                + (0.0 if obj.dae_path == other.dae_path else 0.5)
            )
            if best is None or score < best[0]:
                best = (score, candidate)
        return best[1] if best is not None else None

    def _structural_candidate_ids(self, object_id: str) -> list[str]:
        if self.context is None:
            return []
        base_ids = self.resolved_part_ids or list(self.context.objects)
        seen: set[str] = set()
        out: list[str] = []
        for candidate in base_ids:
            if candidate == object_id or candidate not in self.context.objects or candidate in seen:
                continue
            out.append(candidate)
            seen.add(candidate)
        out.sort(key=lambda item: self._part_display_name(item).lower())
        return out

    def _suggest_structural_source(self, object_id: str, candidates: list[str]) -> str | None:
        return self._name_pair_candidate(object_id, candidates) or self._geometry_pair_candidate(
            object_id,
            candidates,
        )

    def _choose_structural_source(self, object_id: str) -> str | None:
        if self.context is None:
            return None
        candidates = self._structural_candidate_ids(object_id)
        existing = str(self._get_part_setting(object_id, "mirrorSource", "") or "")
        if existing and existing in self.context.objects and existing != object_id and existing not in candidates:
            candidates.append(existing)
        suggested = self._suggest_structural_source(object_id, candidates)
        if suggested and suggested not in candidates:
            candidates.append(suggested)
        if not candidates:
            self._show_error("Mirror Structural", "No other used mesh is available to mirror from.")
            return None

        value_by_label = {self._part_option_label(candidate): candidate for candidate in candidates}
        label_by_value = {value: label for label, value in value_by_label.items()}

        modal = tk.Toplevel(self)
        modal.title("Mirror Structural")
        modal.transient(self)
        modal.resizable(False, False)
        modal.columnconfigure(1, weight=1)

        ttk.Label(modal, text="Part").grid(row=0, column=0, sticky="w", padx=10, pady=(10, 4))
        ttk.Label(modal, text=self._part_option_label(object_id)).grid(
            row=0,
            column=1,
            sticky="ew",
            padx=(0, 10),
            pady=(10, 4),
        )
        ttk.Label(modal, text="Swap with").grid(row=1, column=0, sticky="w", padx=10, pady=4)
        source_var = tk.StringVar()
        initial = existing if existing in label_by_value else suggested
        if initial in label_by_value:
            source_var.set(label_by_value[initial])
        else:
            source_var.set(self._part_option_label(candidates[0]))
        combo = ttk.Combobox(
            modal,
            textvariable=source_var,
            values=list(value_by_label),
            state="readonly",
            width=72,
        )
        combo.grid(row=1, column=1, sticky="ew", padx=(0, 10), pady=4)

        suggestion_text = (
            f"Suggested: {self._part_option_label(suggested)}"
            if suggested
            else "Suggested: no obvious pair found"
        )
        ttk.Label(modal, text=suggestion_text).grid(
            row=2,
            column=0,
            columnspan=2,
            sticky="w",
            padx=10,
            pady=(0, 8),
        )

        result: dict[str, str | None] = {"source": None}

        def use_suggested() -> None:
            if suggested and suggested in label_by_value:
                source_var.set(label_by_value[suggested])

        def commit() -> None:
            selected = value_by_label.get(source_var.get())
            if not selected:
                self._show_error("Mirror Structural", "Select a source mesh to mirror from.", parent=modal)
                return
            if selected == object_id:
                self._show_error(
                    "Mirror Structural",
                    "A mesh cannot structurally mirror from itself.",
                    parent=modal,
                )
                return
            result["source"] = selected
            modal.destroy()

        def cancel() -> None:
            modal.destroy()

        buttons = ttk.Frame(modal)
        buttons.grid(row=3, column=0, columnspan=2, sticky="e", padx=10, pady=(0, 10))
        ttk.Button(buttons, text="Use Suggested", command=use_suggested).pack(side="left", padx=(0, 8))
        ttk.Button(buttons, text="OK", command=commit).pack(side="left", padx=(0, 8))
        ttk.Button(buttons, text="Cancel", command=cancel).pack(side="left")

        modal.bind("<Return>", lambda _event: commit())
        modal.bind("<Escape>", lambda _event: cancel())
        self._place_modal_on_app_monitor(modal)
        modal.grab_set()
        combo.focus_set()
        self.wait_window(modal)
        return result["source"]

    def _open_recommendations_modal(self) -> None:
        if self.context is None:
            self._show_error("No source", "Open a vehicle zip first.")
            return
        object_ids = list(self.resolved_part_ids or self.current_part_ids)
        if not object_ids:
            self._show_error(
                "No parts",
                "Select one or more variants and wait for the used-parts list to finish loading.",
            )
            return

        if self.recommendation_modal is not None and self.recommendation_modal.winfo_exists():
            self.recommendation_modal.lift()
            return

        self.recommendation_seq += 1
        seq = self.recommendation_seq
        self.recommendation_rows = {}

        modal = tk.Toplevel(self)
        self.recommendation_modal = modal
        modal.title("Recommended Part Modes")
        modal.transient(self)
        modal.geometry("1040x560")
        modal.minsize(820, 420)
        modal.columnconfigure(0, weight=1)
        modal.rowconfigure(2, weight=1)

        top = ttk.Frame(modal, padding=(10, 10, 10, 4))
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(3, weight=1)
        select_all_button = ttk.Button(top, text="Select All", command=lambda: self._set_all_recommendations(True))
        clear_all_button = ttk.Button(top, text="Clear All", command=lambda: self._set_all_recommendations(False))
        self.apply_recommendations_button = ttk.Button(
            top,
            text="Apply Selected",
            command=self._apply_selected_recommendations,
            state="disabled",
        )
        select_all_button.grid(row=0, column=0, sticky="w")
        clear_all_button.grid(row=0, column=1, sticky="w", padx=(6, 0))
        self.apply_recommendations_button.grid(row=0, column=2, sticky="w", padx=(12, 0))
        ttk.Button(top, text="Close", command=modal.destroy).grid(row=0, column=4, sticky="e")

        self.recommendation_status_var = tk.StringVar(value="Finding recommendations...")
        ttk.Label(modal, textvariable=self.recommendation_status_var, padding=(10, 0, 10, 4)).grid(
            row=1,
            column=0,
            sticky="ew",
        )

        frame = ttk.Frame(modal, padding=(10, 0, 10, 10))
        frame.grid(row=2, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        columns = ("apply", "mode", "part", "source", "current", "reason")
        tree = ttk.Treeview(frame, columns=columns, show="headings", selectmode="browse")
        self.recommendation_tree = tree
        headings = {
            "apply": "Selected",
            "mode": "Recommended",
            "part": "Part",
            "source": "Pair / Source",
            "current": "Current",
            "reason": "Reason",
        }
        widths = {
            "apply": 54,
            "mode": 132,
            "part": 290,
            "source": 250,
            "current": 190,
            "reason": 220,
        }
        for column in columns:
            tree.heading(
                column,
                text=headings[column],
                anchor="w",
            )
            tree.column(
                column,
                width=widths[column],
                minwidth=50,
                stretch=column in {"part", "reason"},
                anchor="center" if column == "apply" else "w",
            )
        self._register_tree_headings(tree, headings)
        yscroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        xscroll = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=tree.xview)
        tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        self._configure_tree_rows(tree)
        tree.bind("<Button-1>", self._recommendation_click)

        select_all_button.configure(state="disabled")
        clear_all_button.configure(state="disabled")
        self.recommendation_select_all_button = select_all_button
        self.recommendation_clear_all_button = clear_all_button

        def closed() -> None:
            self.recommendation_seq += 1
            self.recommendation_modal = None
            self._tree_sort.pop(tree, None)
            self._tree_heading_text.pop(tree, None)
            self.recommendation_tree = None
            self.recommendation_rows = {}
            modal.destroy()

        modal.protocol("WM_DELETE_WINDOW", closed)
        modal.bind("<Escape>", lambda _event: closed())
        self._place_modal_on_app_monitor(modal)

        worker = threading.Thread(
            target=self._recommendations_worker,
            args=(seq, self.context, object_ids),
            daemon=True,
        )
        worker.start()
        modal.grab_set()
        modal.focus_set()

    def _recommendations_worker(
        self,
        seq: int,
        context: core.VehicleContext,
        object_ids: list[str],
    ) -> None:
        try:
            recommendations = build_mode_recommendations(context, object_ids)
            self.worker_queue.put(("recommendations_success", (seq, context, recommendations)))
        except Exception as exc:
            self.worker_queue.put(("recommendations_error", (seq, exc)))

    def _handle_recommendations_success(self, payload: object) -> None:
        seq, context, recommendations = payload
        if seq != self.recommendation_seq or context is not self.context:
            return
        modal = self.recommendation_modal
        tree = self.recommendation_tree
        if modal is None or tree is None or not modal.winfo_exists():
            return
        for item in tree.get_children():
            tree.delete(item)
        self.recommendation_rows = {}
        for index, recommendation in enumerate(recommendations):
            row_id = f"rec_{index}"
            self.recommendation_rows[row_id] = recommendation
            object_id = recommendation["object_id"]
            source_id = recommendation.get("source_id", "")
            current = self._recommendation_current_label(recommendation)
            tree.insert(
                "",
                "end",
                iid=row_id,
                tags=self._row_tags(index),
                values=(
                    "Y",
                    mode_label(recommendation["mode"]),
                    self._part_option_label(object_id),
                    self._part_option_label(source_id) if source_id else "",
                    current,
                    recommendation.get("reason", ""),
                ),
            )
        count = len(recommendations)
        self.recommendation_status_var.set(
            f"{count} recommendation(s) found for {len(self.resolved_part_ids or self.current_part_ids)} used part(s)."
        )
        state = "normal" if count else "disabled"
        self.recommendation_select_all_button.configure(state=state)
        self.recommendation_clear_all_button.configure(state=state)
        self.apply_recommendations_button.configure(state=state)

    def _handle_recommendations_error(self, payload: object) -> None:
        seq, exc = payload
        if seq != self.recommendation_seq:
            return
        if self.recommendation_modal is not None and self.recommendation_modal.winfo_exists():
            self.recommendation_status_var.set("Recommendation scan failed.")
        self._show_error("Recommendations failed", str(exc))

    def _recommendation_current_label(self, recommendation: dict[str, str]) -> str:
        object_id = recommendation["object_id"]
        mode = str(self._get_part_setting(object_id, "mode", core.MODE_SKIP))
        source_id = recommendation.get("source_id", "")
        if not source_id:
            return mode_label(mode)
        source_mode = str(self._get_part_setting(source_id, "mode", core.MODE_SKIP))
        return f"{mode_label(mode)} / {mode_label(source_mode)}"

    def _recommendation_click(self, event: tk.Event) -> str | None:
        tree = self.recommendation_tree
        if tree is None:
            return None
        if not self._tree_body_click(tree, event):
            return None
        item = tree.identify_row(event.y)
        column = tree.identify_column(event.x)
        if not item or self._tree_column_name(tree, column) != "apply":
            return None
        current = str(tree.set(item, "apply"))
        tree.set(item, "apply", "N" if current == "Y" else "Y")
        return "break"

    def _set_all_recommendations(self, selected: bool) -> None:
        tree = self.recommendation_tree
        if tree is None:
            return
        value = "Y" if selected else "N"
        for item in tree.get_children():
            tree.set(item, "apply", value)

    def _apply_selected_recommendations(self) -> None:
        if self.context is None or self.recommendation_tree is None:
            return
        selected_rows = [
            self.recommendation_rows[item]
            for item in self.recommendation_tree.get_children()
            if self.recommendation_tree.set(item, "apply") == "Y"
        ]
        if not selected_rows:
            self._show_error("No recommendations selected", "Select at least one recommendation to apply.")
            return

        applied = 0
        for recommendation in selected_rows:
            mode = recommendation["mode"]
            object_id = recommendation["object_id"]
            source_id = recommendation.get("source_id", "")
            if mode == core.MODE_MIRROR_STRUCTURAL and source_id:
                self._apply_structural_pair(object_id, source_id)
                applied += 2
            else:
                self._apply_single_part_mode(
                    object_id,
                    mode,
                    texture_flip=recommendation.get("textureFlip"),
                )
                applied += 1

        self._refresh_parts()
        self._refresh_delta_label()
        self._update_detail()
        if self.recommendation_modal is not None and self.recommendation_modal.winfo_exists():
            self.recommendation_modal.destroy()
        self.recommendation_modal = None
        if self.recommendation_tree is not None:
            self._tree_sort.pop(self.recommendation_tree, None)
            self._tree_heading_text.pop(self.recommendation_tree, None)
        self.recommendation_tree = None
        self.recommendation_rows = {}
        self.status_var.set(f"Applied {len(selected_rows)} recommendation(s) to {applied} part setting(s)")

    def _apply_single_part_mode(self, object_id: str, mode: str, *, texture_flip: bool | None = None) -> None:
        settings = self._part_settings(object_id)
        if settings.get("mode") == core.MODE_MIRROR_STRUCTURAL:
            self._clear_structural_pair(object_id)
            settings = self._part_settings(object_id)
        settings["mode"] = mode
        settings["mirrorSource"] = None
        if texture_flip is not None:
            settings["textureFlip"] = bool(texture_flip)

    def _apply_structural_pair(self, object_id: str, source_id: str) -> None:
        self._clear_structural_pair(object_id)
        self._clear_structural_pair(source_id)
        settings = self._part_settings(object_id)
        source_settings = self._part_settings(source_id)
        settings["mode"] = core.MODE_MIRROR_STRUCTURAL
        settings["mirrorSource"] = source_id
        source_settings["mode"] = core.MODE_MIRROR_STRUCTURAL
        source_settings["mirrorSource"] = object_id

    def _part_click(self, event: tk.Event) -> None:
        self._close_tree_combo_editor()
        if not self._tree_body_click(self.part_tree, event):
            return None
        item = self.part_tree.identify_row(event.y)
        column = self.part_tree.identify_column(event.x)
        if not item:
            return None
        name = self._tree_column_name(self.part_tree, column)
        if name == "visible":
            self._toggle_part_bool(item, "viewerVisible", default=True)
            return "break"
        if name == "solo":
            self._toggle_part_bool(item, "viewerSolo")
            return "break"
        if name == "mode":
            self.part_tree.focus(item)
            self.part_tree.selection_set(item)
            current = mode_label(str(self._get_part_setting(item, "mode", core.MODE_SKIP)))
            self._edit_tree_combo(
                self.part_tree,
                item,
                column,
                [mode_label(mode) for mode in MODE_CYCLE_VALUES],
                current,
                lambda value: self._set_part_mode_from_label(item, value),
            )
            return "break"
        if name == "offset":
            if self._get_part_setting(item, "mode", core.MODE_SKIP) != core.MODE_TRANSLATE:
                self.status_var.set("Offset X only applies to Translate mode")
                return "break"
            self._edit_tree_entry(
                self.part_tree,
                item,
                column,
                offset_label(self._get_part_setting(item, "translateOffset", None)),
                lambda value: self._set_part_offset(item, value),
            )
            return "break"
        if name == "fliptex":
            mode = str(self._get_part_setting(item, "mode", core.MODE_SKIP))
            if mode != core.MODE_MIRROR:
                self.status_var.set("Flip Tex only applies to Mirror Aesthetic")
                return "break"
            self._toggle_part_bool(item, "textureFlip")
            flipped = bool(self._get_part_setting(item, "textureFlip", False))
            self.status_var.set(
                f"{self._part_display_name(item)}: texture flip "
                + ("on (un-mirrors the image, e.g. nav screens)" if flipped else "off")
            )
            return "break"
        if name == "steering":
            self._set_single_steering_ref(item)
            return "break"
        # "active" is read-only, and #0/coords fall through to default row select.
        return None

    def _part_motion(self, event: tk.Event) -> None:
        if self.structural_prompt_part_id is None or self.structural_prompt_open:
            return
        item = self.part_tree.identify_row(event.y)
        column = self.part_tree.identify_column(event.x)
        if item != self.structural_prompt_part_id or self._tree_column_name(self.part_tree, column) != "mode":
            self._trigger_structural_prompt()

    def _part_leave(self, _event: tk.Event) -> None:
        if self.structural_prompt_part_id is not None and not self.structural_prompt_open:
            self._trigger_structural_prompt()

    def _part_double_click(self, event: tk.Event) -> None:
        if not self._tree_body_click(self.part_tree, event):
            return None
        item = self.part_tree.identify_row(event.y)
        column = self.part_tree.identify_column(event.x)
        if not item:
            return
        name = self._tree_column_name(self.part_tree, column)
        if name == "mode":
            return "break"
        elif name == "offset":
            if self._get_part_setting(item, "mode", core.MODE_SKIP) != core.MODE_TRANSLATE:
                self.status_var.set("Offset X only applies to Translate mode")
                return
            self._edit_tree_entry(
                self.part_tree,
                item,
                column,
                offset_label(self._get_part_setting(item, "translateOffset", None)),
                lambda value: self._set_part_offset(item, value),
            )

    def _set_part_mode_from_label(self, object_id: str, label: str) -> None:
        mode = MODE_VALUES_BY_LABEL.get(label)
        if mode is None:
            return
        self._set_part_mode(object_id, mode)
        if mode != core.MODE_MIRROR_STRUCTURAL:
            # Mirror Structural sets its own "choose a source" status message.
            self.status_var.set(f"{self._part_display_name(object_id)}: {mode_label(mode)}")

    def _cancel_structural_prompt(self, object_id: str | None = None) -> None:
        if object_id is not None and self.structural_prompt_part_id != object_id:
            return
        if self.structural_prompt_after_id is not None:
            try:
                self.after_cancel(self.structural_prompt_after_id)
            except Exception:
                pass
        self.structural_prompt_after_id = None
        self.structural_prompt_part_id = None
        self.structural_prompt_previous_mode = core.MODE_SKIP

    def _schedule_structural_prompt(self, object_id: str, previous_mode: str) -> None:
        self._cancel_structural_prompt()
        self.structural_prompt_part_id = object_id
        self.structural_prompt_previous_mode = (
            previous_mode if previous_mode in MODE_CYCLE_VALUES else core.MODE_SKIP
        )
        self.structural_prompt_after_id = self.after(STRUCTURAL_PROMPT_DELAY_MS, self._trigger_structural_prompt)
        self.status_var.set(
            f"Mirror Structural selected for {self._part_display_name(object_id)}; choose a source to complete it"
        )

    def _trigger_structural_prompt(self) -> None:
        object_id = self.structural_prompt_part_id
        previous_mode = self.structural_prompt_previous_mode
        if object_id is None or self.structural_prompt_open:
            return
        self._cancel_structural_prompt(object_id)
        if self.context is None:
            return
        settings = self._part_settings(object_id)
        if settings.get("mode") != core.MODE_MIRROR_STRUCTURAL or settings.get("mirrorSource"):
            return

        self.structural_prompt_open = True
        try:
            source_id = self._choose_structural_source(object_id)
        finally:
            self.structural_prompt_open = False

        settings = self._part_settings(object_id)
        if settings.get("mode") != core.MODE_MIRROR_STRUCTURAL:
            return
        if source_id:
            self._set_structural_pair(object_id, source_id)
            return

        restore_mode = previous_mode if previous_mode != core.MODE_MIRROR_STRUCTURAL else core.MODE_SKIP
        settings["mode"] = restore_mode
        settings["mirrorSource"] = None
        self._refresh_parts()
        self._update_detail()
        self.status_var.set(
            f"Mirror Structural cancelled for {self._part_display_name(object_id)}; restored {mode_label(restore_mode)}"
        )

    def _edit_tree_combo(
        self,
        tree: ttk.Treeview,
        item: str,
        column: str,
        values: list[str],
        current: str,
        on_commit,
    ) -> None:
        self._close_tree_combo_editor()
        if not tree.exists(item):
            return
        tree.see(item)
        bbox = tree.bbox(item, column)
        if not bbox:
            return
        x, y, width, height = bbox
        combo = ttk.Combobox(
            tree,
            values=values,
            state="readonly",
            exportselection=False,
            height=min(max(len(values), 1), 15),
        )
        combo.set(current)
        combo.place(x=x, y=y, width=width, height=height)
        tree.focus(item)
        tree.selection_set(item)
        self._tree_combo_editor = combo
        combo.focus_set()

        def commit(_event=None) -> None:
            if self._tree_combo_editor is not combo:
                return "break"
            value = combo.get()
            self._close_tree_combo_editor()
            on_commit(value)
            return "break"

        def cancel(_event=None) -> str:
            if self._tree_combo_editor is combo:
                self._close_tree_combo_editor()
                tree.focus_set()
            return "break"

        def check_focus() -> None:
            self._tree_combo_focus_after_id = None
            if self._tree_combo_editor is not combo:
                return
            try:
                focus_path = str(self.tk.call("focus"))
                combo_path = str(combo)
                # The popdown list is a separate Tk window, but its Tcl path is
                # rooted below the combobox.  Keep checking until it unposts so
                # <<ComboboxSelected>> gets the first chance to commit.
                if focus_path.startswith(combo_path + ".") or combo.instate(("pressed",)):
                    self._tree_combo_focus_after_id = self.after(50, check_focus)
                    return
                # If the list was dismissed without a selection, ttk returns
                # focus to the combobox.  Treat that as cancellation; a real
                # selection has already emitted <<ComboboxSelected>> by now.
                if focus_path == combo_path:
                    self._close_tree_combo_editor()
                    return
            except tk.TclError:
                return
            self._close_tree_combo_editor()

        def focus_out(_event=None) -> None:
            if self._tree_combo_focus_after_id is not None:
                try:
                    self.after_cancel(self._tree_combo_focus_after_id)
                except tk.TclError:
                    pass
            self._tree_combo_focus_after_id = self.after(50, check_focus)

        def post_dropdown() -> None:
            if self._tree_combo_editor is not combo or not combo.winfo_exists():
                return
            combo.focus_set()
            # Drive the public mouse bindings instead of calling Tk's private
            # ttk::combobox::Post command.  This gives an in-cell DDL genuine
            # one-click behaviour while retaining native theme handling.
            arrow_x = max(1, combo.winfo_width() - 4)
            arrow_y = max(1, combo.winfo_height() // 2)
            combo.event_generate("<ButtonPress-1>", x=arrow_x, y=arrow_y)
            combo.event_generate("<ButtonRelease-1>", x=arrow_x, y=arrow_y)

        combo.bind("<<ComboboxSelected>>", commit)
        combo.bind("<Return>", commit)
        combo.bind("<KP_Enter>", commit)
        combo.bind("<Tab>", commit)
        combo.bind("<Escape>", cancel)
        combo.bind("<FocusOut>", focus_out)
        self.after_idle(post_dropdown)

    def _close_tree_combo_editor(self) -> None:
        if self._tree_combo_focus_after_id is not None:
            try:
                self.after_cancel(self._tree_combo_focus_after_id)
            except tk.TclError:
                pass
            self._tree_combo_focus_after_id = None
        combo = self._tree_combo_editor
        self._tree_combo_editor = None
        if combo is not None:
            try:
                combo.destroy()
            except tk.TclError:
                pass

    def _edit_tree_entry(
        self,
        tree: ttk.Treeview,
        item: str,
        column: str,
        current: str,
        on_commit,
    ) -> None:
        bbox = tree.bbox(item, column)
        if not bbox:
            return
        x, y, width, height = bbox
        entry = ttk.Entry(tree)
        entry.insert(0, current)
        entry.place(x=x, y=y, width=width, height=height)
        entry.focus_set()
        entry.selection_range(0, tk.END)

        committed = {"done": False}

        def commit(_event=None) -> None:
            if committed["done"]:
                return
            committed["done"] = True
            value = entry.get()
            entry.destroy()
            on_commit(value)

        entry.bind("<Return>", commit)
        entry.bind("<FocusOut>", commit)
        def cancel(_event=None) -> None:
            committed["done"] = True
            entry.destroy()

        entry.bind("<Escape>", cancel)

    def _get_variant_setting(self, config_name: str, key: str, default: object) -> object:
        variants = self.conversion.setdefault("variants", {})
        settings = variants.setdefault(config_name, {})
        if not isinstance(settings, dict):
            return default
        return settings.get(key, default)

    def _set_variant_setting(self, config_name: str, key: str, value: object) -> None:
        variants = self.conversion.setdefault("variants", {})
        settings = variants.setdefault(config_name, {})
        if isinstance(settings, dict):
            settings[key] = value
        self._refresh_variants()
        self._refresh_delta_label()
        self._update_detail()

    def _get_part_setting(self, object_id: str, key: str, default: object) -> object:
        parts = self.conversion.setdefault("parts", {})
        settings = parts.setdefault(object_id, {})
        if not isinstance(settings, dict):
            return default
        return settings.get(key, default)

    def _refresh_part_viewer_cells(self, object_ids: list[str] | tuple[str, ...] | set[str]) -> None:
        parts = self.conversion.get("parts", {})
        if not isinstance(parts, dict):
            return
        for object_id in object_ids:
            if not self.part_tree.exists(object_id):
                continue
            settings = parts.get(object_id)
            if not isinstance(settings, dict):
                settings = {}
            self.part_tree.set(object_id, "visible", yn_label(settings.get("viewerVisible", True)))
            self.part_tree.set(object_id, "solo", yn_label(settings.get("viewerSolo")))

    def _toggle_part_bool(self, object_id: str, key: str, *, default: bool = False) -> None:
        parts = self.conversion.setdefault("parts", {})
        settings = parts.setdefault(object_id, {})
        if isinstance(settings, dict):
            settings[key] = not bool(settings.get(key, default))
        if key in {"viewerVisible", "viewerSolo"}:
            self._refresh_part_viewer_cells([object_id])
            self._refresh_viewer()
            self._update_detail()
            return
        if key == "steeringRef":
            self._refresh_variants()
        self._refresh_parts()
        self._refresh_delta_label()
        self._update_detail()

    def _set_single_steering_ref(self, object_id: str) -> None:
        if self.context is None:
            return
        parts = self.conversion.setdefault("parts", {})
        was_selected = bool(self._get_part_setting(object_id, "steeringRef", False))
        for part_id in list(parts):
            settings = parts.get(part_id)
            if isinstance(settings, dict):
                settings["steeringRef"] = False
        settings = self._part_settings(object_id)
        settings["steeringRef"] = not was_selected
        self._invalidate_variant_detection()
        self._refresh_variants()
        self._schedule_variant_detection()
        self._refresh_parts()
        self._refresh_delta_label()
        self._update_detail()
        if settings["steeringRef"]:
            self.status_var.set(f"Steering reference set: {self._part_display_name(object_id)}")
        else:
            self.status_var.set("Steering reference cleared")

    def _set_all_parts_visible(self, visible: bool) -> None:
        if self.context is None:
            return
        object_ids = [
            object_id
            for object_id in self.current_part_ids
            if object_id in self.context.objects
        ]
        if not object_ids:
            if self.resolved_part_ids:
                self.status_var.set("No displayed parts match the current filter")
            else:
                self.status_var.set("No used parts are loaded yet")
            return
        for object_id in object_ids:
            settings = self._part_settings(object_id)
            settings["viewerVisible"] = visible
            settings["viewerSolo"] = False
        self._refresh_part_viewer_cells(object_ids)
        self._refresh_viewer()
        self._update_detail()
        state = "visible" if visible else "hidden"
        scope = "displayed" if self.filter_var.get().strip() else "used"
        self.status_var.set(f"Set {len(object_ids)} {scope} part(s) {state}; cleared solo flags")

    def _toggle_selected_parts_visibility_shortcut(self, event: tk.Event) -> str | None:
        focus = self.focus_get()
        if focus is not None and focus.winfo_class() in {
            "Entry",
            "TEntry",
            "Text",
            "Combobox",
            "TCombobox",
            "Spinbox",
            "TSpinbox",
        }:
            return None
        if self.context is None:
            return None
        selected = [
            object_id
            for object_id in self.part_tree.selection()
            if self.part_tree.exists(object_id) and object_id in self.context.objects
        ]
        if not selected:
            return None
        for object_id in selected:
            settings = self._part_settings(object_id)
            settings["viewerVisible"] = not bool(settings.get("viewerVisible", True))
        self._refresh_part_viewer_cells(selected)
        self._refresh_viewer()
        self._update_detail()
        if len(selected) == 1:
            object_id = selected[0]
            visible = bool(self._get_part_setting(object_id, "viewerVisible", True))
            self.status_var.set(
                f"{self._part_display_name(object_id)} {'visible' if visible else 'hidden'}"
            )
        else:
            self.status_var.set(f"Toggled visibility for {len(selected)} selected part(s)")
        return "break"

    def _set_selected_part_mode_shortcut(self, _event: tk.Event, mode: str) -> str | None:
        focus = self.focus_get()
        # Only typing targets swallow the hotkeys; buttons and other focusable
        # widgets don't react to letter keys, so mode setting stays live.
        if focus is not None and focus.winfo_class() in {
            "Entry",
            "TEntry",
            "Text",
            "Combobox",
            "TCombobox",
            "Spinbox",
            "TSpinbox",
        }:
            return None
        if self.context is None:
            return None
        targets = [
            object_id
            for object_id in self.part_tree.selection()
            if self.part_tree.exists(object_id) and object_id in self.context.objects
        ]
        if not targets:
            item = self.part_tree.focus()
            if item and self.part_tree.exists(item) and item in self.context.objects:
                targets = [item]
        if not targets:
            return None
        if mode == core.MODE_MIRROR_STRUCTURAL:
            # The source-pair prompt is a per-part modal; only sensible one at a time.
            if len(targets) != 1:
                self.status_var.set("Select a single part to set Mirror Structural (it needs a source pair)")
                return "break"
            self._set_part_mode(targets[0], mode)
            return "break"
        for object_id in targets:
            self._cancel_structural_prompt(object_id)
            self._apply_single_part_mode(object_id, mode)
        self._refresh_parts()
        self._refresh_delta_label()
        self._update_detail()
        if len(targets) == 1:
            self.status_var.set(f"{self._part_display_name(targets[0])}: {mode_label(mode)}")
        else:
            self.status_var.set(f"Set {mode_label(mode)} on {len(targets)} part(s)")
        return "break"

    def _part_settings(self, object_id: str) -> dict[str, object]:
        parts = self.conversion.setdefault("parts", {})
        settings = parts.setdefault(
            object_id,
            {
                "mode": core.MODE_SKIP,
                "mirrorSource": None,
                "translateOffset": None,
                "textureFlip": False,
                "steeringRef": False,
                "viewerVisible": True,
                "viewerSolo": False,
            },
        )
        if not isinstance(settings, dict):
            settings = {}
            parts[object_id] = settings
        return settings

    def _clear_structural_pair(self, object_id: str) -> None:
        settings = self._part_settings(object_id)
        source_id = str(settings.get("mirrorSource") or "")
        settings["mirrorSource"] = None
        if not source_id:
            return
        source_settings = self._part_settings(source_id)
        if (
            source_settings.get("mode") == core.MODE_MIRROR_STRUCTURAL
            and str(source_settings.get("mirrorSource") or "") == object_id
        ):
            source_settings["mode"] = core.MODE_SKIP
            source_settings["mirrorSource"] = None

    def _set_structural_pair(self, object_id: str, source_id: str) -> None:
        self._cancel_structural_prompt(object_id)
        self._cancel_structural_prompt(source_id)
        self._clear_structural_pair(object_id)
        self._clear_structural_pair(source_id)
        settings = self._part_settings(object_id)
        source_settings = self._part_settings(source_id)
        settings["mode"] = core.MODE_MIRROR_STRUCTURAL
        settings["mirrorSource"] = source_id
        source_settings["mode"] = core.MODE_MIRROR_STRUCTURAL
        source_settings["mirrorSource"] = object_id
        self._refresh_parts()
        self._update_detail()
        self.status_var.set(
            f"Structural mirror pair set: {self._part_display_name(object_id)} <-> "
            f"{self._part_display_name(source_id)}"
        )

    def _set_part_mode(self, object_id: str, mode: str) -> None:
        current_mode = str(self._get_part_setting(object_id, "mode", core.MODE_SKIP))
        if mode == core.MODE_MIRROR_STRUCTURAL:
            settings = self._part_settings(object_id)
            if current_mode == core.MODE_MIRROR_STRUCTURAL:
                self._clear_structural_pair(object_id)
                settings = self._part_settings(object_id)
            settings["mode"] = core.MODE_MIRROR_STRUCTURAL
            settings["mirrorSource"] = None
            self._refresh_parts()
            self._update_detail()
            self._schedule_structural_prompt(object_id, current_mode)
            return
        self._cancel_structural_prompt(object_id)
        settings = self._part_settings(object_id)
        if settings.get("mode") == core.MODE_MIRROR_STRUCTURAL:
            self._clear_structural_pair(object_id)
            settings = self._part_settings(object_id)
        settings["mode"] = mode
        settings["mirrorSource"] = None
        self._refresh_parts()
        self._update_detail()

    def _set_part_offset(self, object_id: str, value: str) -> None:
        parts = self.conversion.setdefault("parts", {})
        settings = parts.setdefault(object_id, {})
        if not isinstance(settings, dict):
            return
        cleaned = value.strip()
        if not cleaned:
            settings["translateOffset"] = None
        else:
            try:
                settings["translateOffset"] = abs(float(cleaned))
            except ValueError:
                self._show_error("Invalid offset", "Part offset must be blank or a number.")
                return
        self._refresh_parts()
        self._refresh_delta_label()
        self._update_detail()

    def _part_selection_changed(self) -> None:
        self._refresh_viewer()
        self._update_detail()

    def _on_preview_pick(self, object_id: object) -> None:
        """A part was clicked in the GPU preview. object_id is the picked mesh
        name (== a part_tree iid) or None for empty space. Setting the tree
        selection fires <<TreeviewSelect>> which refreshes the highlight+detail."""
        if self.context is None:
            return
        # A viewer click means the user is working with parts: pull keyboard
        # focus onto the table so the mode/visibility hotkeys apply directly.
        self.part_tree.focus_set()
        if not object_id:
            if self.part_tree.selection():
                self.part_tree.selection_set([])  # empty click -> deselect
            return
        object_id = str(object_id)
        # The clicked part is rendered but may be filtered out of the table;
        # clear the filter so its row exists and can be selected. The filter_var
        # write-trace rebuilds the table synchronously.
        if not self.part_tree.exists(object_id) and self.filter_var.get().strip():
            self.filter_var.set("")
        if self.part_tree.exists(object_id):
            self.part_tree.selection_set([object_id])
            self.part_tree.focus(object_id)
            self.part_tree.see(object_id)

    def _manual_delta_toggled(self, *, refresh: bool = True) -> None:
        state = "normal" if self.manual_delta_enabled.get() else "disabled"
        self.manual_delta_entry.configure(state=state)
        delta = self.conversion.setdefault("delta", {})
        if isinstance(delta, dict):
            delta["manual"] = bool(self.manual_delta_enabled.get())
        if refresh:
            self._commit_delta_from_ui()

    def _commit_delta_from_ui(self) -> None:
        delta = self.conversion.setdefault("delta", {})
        if isinstance(delta, dict):
            delta["manual"] = bool(self.manual_delta_enabled.get())
            if self.manual_delta_enabled.get():
                text = self.manual_delta_var.get().strip()
                try:
                    delta["magnitude"] = abs(float(text)) if text else 0.0
                except ValueError:
                    self._show_error("Invalid delta", "Manual delta magnitude must be a number.")
                    return
        self._refresh_delta_label()
        self._refresh_parts()
        self._update_detail()

    def _browse_mods_folder(self) -> None:
        initial = existing_initial_dir(
            self.settings.get("lastModsFolder") or self.mods_folder_var.get(),
            core.WORKSPACE_DIR,
        )
        path = self._ask_directory(title="Select BeamNG mods folder", initialdir=initial)
        if path:
            self.mods_folder_var.set(path)
            self.settings["lastModsFolder"] = path
            self._save_app_settings_from_ui()

    def _browse_blender(self) -> None:
        initial = existing_initial_dir(
            self.settings.get("lastBlenderFolder") or self.blender_var.get(),
            Path(r"C:\Program Files"),
        )
        path = self._ask_open_filename(
            title="Select blender.exe",
            initialdir=initial,
            filetypes=(("Executable", "*.exe"), ("All files", "*.*")),
        )
        if path:
            self.blender_var.set(path)
            self.settings["lastBlenderFolder"] = str(Path(path).parent)
            self._save_app_settings_from_ui()

    def _save_app_settings_from_ui(self) -> None:
        mods_folder = self.mods_folder_var.get().strip()
        blender_exe = self.blender_var.get().strip()
        self.settings["modsFolder"] = mods_folder
        self.settings["blenderExecutable"] = blender_exe
        if mods_folder:
            self.settings["lastModsFolder"] = mods_folder
        if blender_exe:
            self.settings["lastBlenderFolder"] = str(Path(blender_exe).parent)
        core.save_app_settings(self.settings)

    def _save_config(self) -> None:
        if self.context is None:
            return
        try:
            self._commit_delta_from_ui()
            path = core.save_conversion(self.context, self.conversion)
            self._save_app_settings_from_ui()
            self.status_var.set(f"Saved config: {path}")
        except Exception as exc:
            self._show_error("Save failed", str(exc))

    def _import_config_dialog(self) -> None:
        if self.context is None:
            return
        path = self._ask_open_filename(
            title="Import conversion config",
            initialdir=str(core.PROJECTS_DIR),
            filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
        )
        if not path:
            return
        try:
            imported = json.loads(Path(path).read_text(encoding="utf-8"))
            self.conversion, counts = core.import_matching_conversion(
                self.context,
                self.conversion,
                imported,
            )
            self._sync_delta_to_ui()
            self._invalidate_variant_detection()
            self._refresh_all(reset_view=True)
            self._schedule_variant_detection()
            self.status_var.set(
                "Imported matched settings: "
                f"{counts['variantImported']} variant(s), {counts['partImported']} part(s); "
                f"dropped {counts['variantSkipped']} variant(s), {counts['partSkipped']} part(s)"
            )
        except Exception as exc:
            self._show_error("Import failed", str(exc))

    def _set_busy(self, busy: bool) -> None:
        self.worker_running = busy
        state = "disabled" if busy else "normal"
        self.install_button.configure(state=state)
        self.blender_button.configure(state=state)
        if hasattr(self, "preview_output_combo"):
            if busy or not self.preview_output_to_config:
                self.preview_output_combo.configure(state="disabled")
            else:
                self.preview_output_combo.configure(state="readonly")

    def _start_build(self, *, install: bool) -> None:
        if self.context is None:
            self._show_error("No source", "Open a vehicle zip first.")
            return
        if install and not self.mods_folder_var.get().strip():
            self._show_error("No mods folder", "Set a BeamNG mods folder before installing.")
            return
        self._commit_delta_from_ui()
        self._save_app_settings_from_ui()
        self._set_busy(True)
        self.status_var.set("Building XP conversion zip...")
        worker = threading.Thread(target=self._build_worker, args=(install,), daemon=True)
        worker.start()

    def _build_worker(self, install: bool) -> None:
        assert self.context is not None
        try:
            result = core.build_batch(
                self.context,
                self.conversion,
                write_zip=True,
                install=install,
                mods_folder=Path(self.mods_folder_var.get()) if install else None,
            )
            self.worker_queue.put(("build_success", result))
        except Exception as exc:
            self.worker_queue.put(("error", exc))

    def _start_blender_preview(self) -> None:
        if self.context is None:
            self._show_error("No source", "Open a vehicle zip first.")
            return
        blender = self._resolve_blender()
        if blender is None:
            self._show_error("Blender not found", "Set the Blender executable path first.")
            return
        config_label = self.preview_output_var.get().strip()
        output_name = self.preview_output_to_output.get(config_label)
        if not output_name or config_label not in self.preview_output_to_config:
            self._show_error(
                "No config",
                "Select a buildable config in the Config dropdown.",
            )
            return
        self._commit_delta_from_ui()
        self._save_app_settings_from_ui()
        self._set_busy(True)
        self.status_var.set(f"Preparing Blender preview for {config_label}...")
        worker = threading.Thread(
            target=self._blender_preview_worker,
            args=(blender, output_name),
            daemon=True,
        )
        worker.start()

    def _resolve_blender(self) -> Path | None:
        configured = self.blender_var.get().strip()
        if configured and Path(configured).exists():
            return Path(configured)
        for candidate in BLENDER_CANDIDATES:
            if candidate.exists():
                self.blender_var.set(str(candidate))
                return candidate
        return None

    def _mesh_scene_config(self) -> str | None:
        # The dropdown's highlighted-but-unconfirmed entry wins while the
        # list is open, so trims hot-load as you scroll through them.
        label = (self.preview_output_hover or self.preview_output_var.get()).strip()
        config = self.preview_output_to_config.get(label)
        if config:
            return config
        selected = self._selected_variant_names()
        if selected:
            return selected[0]
        if self.context is not None and self.context.variants:
            return next(iter(self.context.variants))
        return None

    def _mesh_scene_snapshot(self) -> str | None:
        """Fingerprint of everything the 3D scene depends on. Viewer-only
        flags (visibility/solo) are excluded - those only filter the index
        buffer and never need a rebuild."""
        config = self._mesh_scene_config()
        if config is None:
            return None
        conversion = json.loads(json.dumps(self.conversion, default=str))
        parts = conversion.get("parts")
        if isinstance(parts, dict):
            for settings in parts.values():
                if isinstance(settings, dict):
                    settings.pop("viewerVisible", None)
                    settings.pop("viewerSolo", None)
        return json.dumps(
            {
                "config": config,
                "output": self._selected_preview_output_name(),
                "conversion": conversion,
                # Selected-but-inactive parts are injected into the scene, so
                # the scene must rebuild when that set changes (and only then;
                # selection moves between active parts leave it empty/equal).
                "extra": self._selected_extra_preview_ids(),
            },
            sort_keys=True,
        )

    def _schedule_mesh_scene(self, *, immediate: bool = False) -> None:
        if self.context is None or not self.viewer_supports_scene:
            return
        snapshot = self._mesh_scene_snapshot()
        if snapshot is None:
            return
        if snapshot == self.mesh_scene_hash:
            return
        if self.mesh_scene_running:
            self.mesh_scene_pending = True
            return
        if self.mesh_scene_after is not None:
            return
        if immediate:
            self._start_mesh_scene()
        else:
            self.mesh_scene_after = self.after_idle(self._start_mesh_scene)

    def _start_mesh_scene(self) -> None:
        self.mesh_scene_after = None
        if self.context is None or not self.viewer_supports_scene or self.viewer is None:
            return
        snapshot = self._mesh_scene_snapshot()
        config = self._mesh_scene_config()
        if snapshot is None or config is None:
            return
        if snapshot == self.mesh_scene_hash:
            return
        self.mesh_scene_hash = snapshot
        self.mesh_scene_seq += 1
        seq = self.mesh_scene_seq
        context = self.context
        conversion_copy = json.loads(json.dumps(self.conversion, default=str))
        # The in-app preview represents one source trim, independently of how
        # many build outputs were requested. Always prepare both transformed
        # and original-layout vertex buffers; the viewer checkbox switches
        # between them while the resolved replacement plates remain the same.
        settings = conversion_copy.get("variants", {}).get(config, {})
        if isinstance(settings, dict):
            core.set_variant_build_mode(settings, core.BUILD_CONVERTED)
        self.viewer.set_message(f"building preview: {config}...")
        self.mesh_scene_running = True
        extra_meshes = tuple(self._selected_extra_preview_ids())
        future = self.part_resolver.submit(
            self._mesh_scene_worker, context, conversion_copy, config, extra_meshes
        )
        future.add_done_callback(
            lambda completed, current_seq=seq, current_snapshot=snapshot: self.worker_queue.put(
                ("mesh_scene_done", (current_seq, current_snapshot, completed))
            )
        )

    @staticmethod
    def _mesh_scene_worker(
        context: core.VehicleContext,
        conversion: dict[str, object],
        config_name: str,
        extra_meshes: tuple[str, ...] = (),
    ):
        payload = core.full_vehicle_preview_payload(
            context,
            conversion,
            config_name,
            context.project_dir / "blender_preview",
            extra_meshes=extra_meshes,
        )
        cache_dir = context.project_dir / "blender_preview" / "dae_cache" / "mesh_cache"
        return mesh_preview.build_scene(payload, cache_dir)

    def _handle_mesh_scene_done(self, payload: object) -> None:
        seq, completed_snapshot, completed = payload
        self.mesh_scene_running = False
        should_apply = (
            seq == self.mesh_scene_seq
            and completed_snapshot == self._mesh_scene_snapshot()
            and self.viewer is not None
            and self.viewer_supports_scene
        )
        try:
            scene = completed.result()
        except Exception as exc:
            if should_apply and self.viewer is not None:
                self.viewer.set_message(f"preview failed: {exc}")
            self._schedule_pending_mesh_scene()
            return
        if not should_apply:
            self._schedule_pending_mesh_scene()
            return
        assert self.viewer is not None
        self.viewer.show_scene(scene, reset_view=self.mesh_scene_reset_pending)
        self.mesh_scene_reset_pending = False
        self._refresh_viewer()
        # The previewed trim (and thus its part set) may have changed; resync
        # the Active column to the scene now on screen.
        self._refresh_active_cells()
        self._schedule_pending_mesh_scene()

    def _schedule_pending_mesh_scene(self) -> None:
        if not self.mesh_scene_pending:
            return
        self.mesh_scene_pending = False
        self._schedule_mesh_scene(immediate=True)

    @staticmethod
    def _preview_needs_generated_output(
        context: core.VehicleContext,
        conversion: dict[str, object],
        config_name: str,
        output_name: str,
    ) -> bool:
        if output_name == core.original_plate_output_name(config_name):
            return True
        object_modes = core.active_part_modes(conversion)
        if not object_modes:
            return False
        _flex, _props, all_meshes = core.selected_mesh_roles(context, [config_name])
        return any(mesh in all_meshes for mesh in object_modes)

    def _blender_preview_worker(self, blender: Path, output_name: str) -> None:
        assert self.context is not None
        try:
            run_dir = self.context.project_dir / "blender_preview" / datetime.now().strftime("run_%Y%m%d_%H%M%S")
            run_dir.mkdir(parents=True, exist_ok=True)
            output_sources = core.output_config_sources(self.context, self.conversion)
            config_name = output_sources.get(output_name)
            if config_name is None:
                raise RuntimeError(f"Unknown generated config {output_name!r}")
            if self._preview_needs_generated_output(self.context, self.conversion, config_name, output_name):
                result = core.build_batch(
                    self.context,
                    self.conversion,
                    write_zip=False,
                    install=False,
                    mods_folder=None,
                )
                if output_name not in result.generated_configs:
                    raise RuntimeError(f"Output {output_name!r} was not generated by the current settings")
                payload = core.output_vehicle_preview_payload(
                    self.context,
                    self.conversion,
                    output_name,
                    result.unpacked_dir,
                    result.generated_daes,
                    run_dir,
                )
            else:
                payload = core.full_vehicle_preview_payload(
                    self.context,
                    self.conversion,
                    config_name,
                    run_dir,
                )
                payload["output_name"] = output_name
                payload["show_unchanged"] = True
            payload_path = run_dir / "blender_preview_payload.json"
            payload_path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
            subprocess.Popen(
                [
                    str(blender),
                    "--python",
                    str(BLENDER_PREVIEW_SCRIPT),
                    "--",
                    str(payload_path),
                ],
                cwd=str(THIS_DIR),
            )
            self.worker_queue.put(("preview_success", payload_path))
        except Exception as exc:
            self.worker_queue.put(("error", exc))

    def _poll_worker_queue(self) -> None:
        handled = False
        while True:
            try:
                kind, payload = self.worker_queue.get_nowait()
            except queue.Empty:
                break
            handled = True
            self._handle_worker_message(kind, payload)
        self.after(40 if handled else 80, self._poll_worker_queue)

    def _handle_worker_message(self, kind: str, payload: object) -> None:
        if kind == "parts_success":
            self._handle_parts_success(payload)
            return
        if kind == "vehicle_load_success":
            self._handle_vehicle_load_success(payload)
            return
        if kind == "vehicle_load_error":
            self._handle_vehicle_load_error(payload)
            return
        if kind == "recommendations_success":
            self._handle_recommendations_success(payload)
            return
        if kind == "recommendations_error":
            self._handle_recommendations_error(payload)
            return
        if kind == "mesh_scene_done":
            self._handle_mesh_scene_done(payload)
            return
        if kind == "variant_hands_done":
            self._handle_variant_hands_done(payload)
            return

        self._set_busy(False)
        if kind == "build_success":
            result: core.BuildResult = payload
            plate_note = ""
            plates = result.plate_summary or {}
            if plates.get("configsUpdated"):
                plate_note = f"; plates on {plates['configsUpdated']} config(s)"
                warnings = plates.get("warnings") or []
                if warnings:
                    plate_note += f" ({len(warnings)} plate warning(s), see conversion.json)"
            if result.installed_plates_zip:
                plate_note += f"; plate library mod refreshed ({plates.get('libraryModDesigns', 0)} design(s))"
            if result.installed_zip:
                self.status_var.set(
                    f"Built {result.package_zip} and installed {result.installed_zip}; "
                    f"{len(result.generated_configs)} config(s){plate_note}"
                )
            else:
                self.status_var.set(
                    f"Built {result.package_zip}; {len(result.generated_configs)} config(s){plate_note}"
                )
        elif kind == "preview_success":
            self.status_var.set(f"Blender preview launched: {payload}")
        else:
            self._show_error("Operation failed", str(payload))
            self.status_var.set("Operation failed")
        self._refresh_all()

    def _handle_parts_success(self, payload: object) -> None:
        seq, context, reset_view, selected, future = payload
        self.part_refresh_running = False
        should_apply = seq == self.part_refresh_seq and context is self.context
        try:
            result = future.result()
        except Exception as exc:
            if should_apply:
                self.resolved_part_ids = []
                self._refresh_parts(reset_view=reset_view)
                self.status_var.set(f"Part resolver failed: {exc}")
            self._schedule_pending_parts_refresh()
            return
        if not should_apply:
            self._schedule_pending_parts_refresh()
            return
        self.resolved_part_ids = result
        core.save_cached_part_ids(context, selected, self.resolved_part_ids)
        self._refresh_parts(reset_view=reset_view)
        self._update_detail()
        self.status_var.set(f"{len(self.current_part_ids)} used part(s) displayed")
        self._schedule_pending_parts_refresh()

    def _schedule_pending_parts_refresh(self) -> None:
        if not self.part_refresh_pending:
            return
        reset_view = self.part_refresh_pending_reset
        self.part_refresh_pending = False
        self.part_refresh_pending_reset = False
        self._schedule_parts_refresh(reset_view=reset_view)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generic BeamNG hand-drive visual conversion tool")
    parser.add_argument("--source", help="Vehicle source zip to open")
    parser.add_argument("--vehicle", help="Vehicle model folder under vehicles/")
    parser.add_argument("--validate", action="store_true", help="Print detected inventory and exit")
    return parser.parse_args()


def validate_source(source: Path, vehicle: str | None) -> None:
    context = core.load_vehicle_context(source, vehicle)
    conversion, loaded = core.load_or_create_conversion(context)
    print(f"Source: {context.source_zip}")
    print(f"Vehicle: {context.vehicle_id}")
    print(f"Project: {context.project_dir}")
    print(f"Project config loaded: {loaded}")
    print(f"DAE files: {len(context.dae_paths)}")
    print(f"Variants: {len(context.variants)}")
    print(f"DAE objects: {len(context.objects)}")
    print(f"Auto delta magnitude: {fmt_float(core.auto_delta_magnitude(context, conversion))}")


def main() -> None:
    args = parse_args()
    if args.validate:
        if not args.source:
            raise SystemExit("--validate requires --source")
        validate_source(Path(args.source), args.vehicle)
        return

    app = HandDriveToolApp()
    if args.source:
        app.after(50, lambda: app._load_source_zip(Path(args.source), args.vehicle))
    else:
        last_source = str(app.settings.get("lastVehicleZipPath") or "")
        last_vehicle = str(app.settings.get("lastVehicleId") or "")
        if last_source and Path(last_source).exists():
            app.after(
                50,
                lambda: app._load_source_zip(
                    Path(last_source),
                    last_vehicle or None,
                ),
            )
    app.mainloop()


if __name__ == "__main__":
    main()
