from __future__ import annotations

from .recommendation_common import (
    SPATIAL_PAIR_MIN_OFFSET,
    SPATIAL_PASSENGER_VISIBLE_FRACTION,
    SPATIAL_REACH_LIMIT,
    SPATIAL_VISIBLE_FRACTION,
    _driver_control_outboard_limit,
    _is_enclosed_candidate,
    _mesh_symmetry,
)
from .shared import core


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
    passenger_eye = eye.copy()
    passenger_eye[0] = 2.0 * cx0 - eye[0]
    passenger_f = f.copy()
    passenger_f[0] *= -1.0
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
    base_transparent = {
        o for o in present
        if context.objects.get(o) is not None
        and core.steering_ref_score(o, context.objects[o]) >= 15
    }
    transparent = set(base_transparent)
    passenger_transparent = set(base_transparent)
    driver_seat_ids: set[str] = set()
    # Each eye camera may sit inside its seat volume. Treat the furniture
    # surrounding that eye as transparent only to that eye, so the seat cannot
    # hide nearby cabin parts. Geometry, not a seat token, identifies the host;
    # this also handles benches and mod seats.
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
        under_passenger_eye = (
            (np.abs(seat_points[:, 0] - passenger_eye[0]) < 0.25)
            & (np.abs(seat_points[:, 1] - passenger_eye[1]) < 0.35)
            & (seat_points[:, 2] > passenger_eye[2] - 0.75)
            & (seat_points[:, 2] < passenger_eye[2] - 0.10)
        )
        if float(under_passenger_eye.mean()) >= 0.20:
            passenger_transparent.add(object_id)

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
    passenger_surface_scene = compiled_surface(passenger_transparent)
    surface_scene_no_glass = compiled_surface(transparent | glass_ids)
    passenger_surface_scene_no_glass = compiled_surface(
        passenger_transparent | glass_ids
    )
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
    passenger_scan = core.visibility_scan(
        {o: entries_np[o] for o in present},
        passenger_eye,
        passenger_transparent,
        passenger_f,
    )
    passenger_scan_no_glass = core.visibility_scan(
        {o: entries_np[o] for o in present if o not in glass_ids},
        passenger_eye,
        passenger_transparent,
        passenger_f,
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
    # the stateful verdict/pair ordering below, but batch this expensive
    # broad-phase superset into one GPU scene upload/dispatch.  The core
    # helper retains the previous bounded CPU thread pool as its fallback.
    exact_by_id: dict[str, dict[str, float] | None] = {}
    passenger_exact_by_id: dict[str, dict[str, float] | None] = {}
    exact_no_glass_by_id: dict[str, dict[str, float] | None] = {}
    passenger_exact_no_glass_by_id: dict[str, dict[str, float] | None] = {}
    exact_glass_by_id: dict[str, dict[str, float] | None] = {}
    if surface_scene is not None or passenger_surface_scene is not None:
        exact_ids = []
        passenger_exact_ids = []
        fitment_ids = []
        passenger_fitment_ids = []
        for object_id in want:
            points = entries_np.get(object_id)
            if points is None or len(points) < 4:
                continue
            stats = scan[object_id]
            stats_ng = scan_no_glass.get(object_id, stats)
            passenger_stats = passenger_scan[object_id]
            passenger_stats_ng = passenger_scan_no_glass.get(
                object_id, passenger_stats
            )
            centroid = points.mean(axis=0)
            extents = np.ptp(points, axis=0)
            diagonal = float(np.linalg.norm(extents))
            ahead = float((centroid[:2] - eye[:2]) @ f[:2])
            passenger_ahead = float(
                (centroid[:2] - passenger_eye[:2]) @ passenger_f[:2]
            )
            lat_signed = float(frame.side * (centroid[0] - wheel_x))
            below = frame.eye[2] - float(centroid[2])
            out80 = float(np.percentile(np.abs(points[:, 0] - cx0), 80))
            wall_lateral = (
                extents[0] < 0.22 and extents[1] > 0.45 and extents[2] > 0.45
            )
            in_cone = (
                0.20 <= ahead <= wheel_ahead + 1.0
                and -0.22 <= lat_signed <= _driver_control_outboard_limit(below)
                and -0.10 <= below <= 1.35
                and not wall_lateral
            )
            enclosed = _is_enclosed_candidate(stats, out80, half_width)
            fitment = (
                stats_ng["vf"] >= 0.12
                and diagonal <= 0.60
                and 0.15 <= ahead <= 1.5
                and abs(float(centroid[2]) - frame.eye[2]) <= 0.7
                and abs(float(centroid[0]) - cx0) >= half_width - 0.06
            )
            passenger_fitment = (
                passenger_stats_ng["vf"] >= 0.12
                and diagonal <= 0.60
                and 0.15 <= passenger_ahead <= 1.5
                and abs(float(centroid[2]) - passenger_eye[2]) <= 0.7
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
            passenger_visible = (
                passenger_stats["front_vf"]
                >= SPATIAL_PASSENGER_VISIBLE_FRACTION
            )
            might_enter = (
                stats["front_vf"] >= SPATIAL_VISIBLE_FRACTION
                or passenger_visible
                or enclosed
                or fitment
                or passenger_fitment
                or cone
                or object_id in under_seat_candidates
                or object_id in forced
            )
            if might_enter:
                exact_ids.append(object_id)
                if passenger_visible or passenger_fitment:
                    passenger_exact_ids.append(object_id)
                if fitment:
                    fitment_ids.append(object_id)
                if passenger_fitment:
                    passenger_fitment_ids.append(object_id)
        if exact_ids and surface_scene is not None:
            exact_by_id = core.surface_visibility_stats_batch(
                {object_id: entries_np[object_id] for object_id in exact_ids},
                frame.eye,
                surface_scene,
                frame.forward,
            )
        if passenger_exact_ids and passenger_surface_scene is not None:
            passenger_exact_by_id = core.surface_visibility_stats_batch(
                {
                    object_id: entries_np[object_id]
                    for object_id in passenger_exact_ids
                },
                passenger_eye,
                passenger_surface_scene,
                passenger_f,
            )
        if (
            passenger_fitment_ids
            and passenger_surface_scene_no_glass is not None
        ):
            passenger_exact_no_glass_by_id = core.surface_visibility_stats_batch(
                {
                    object_id: entries_np[object_id]
                    for object_id in passenger_fitment_ids
                },
                passenger_eye,
                passenger_surface_scene_no_glass,
                passenger_f,
            )
        if exact_ids:
            if fitment_ids and surface_scene_no_glass is not None:
                exact_no_glass_by_id = core.surface_visibility_stats_batch(
                    {
                        object_id: entries_np[object_id]
                        for object_id in fitment_ids
                    },
                    frame.eye,
                    surface_scene_no_glass,
                    frame.forward,
                )
            glass_ids_to_scan = [
                object_id for object_id in exact_ids
                if surface_scene_glass is not None
                and beyond.get(object_id, 0.0) >= 0.40
            ]
            if glass_ids_to_scan:
                exact_glass_by_id = core.surface_visibility_stats_batch(
                    {
                        object_id: entries_np[object_id]
                        for object_id in glass_ids_to_scan
                    },
                    frame.eye,
                    surface_scene_glass,
                    frame.forward,
                )

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
            verdicts[object_id] = (
                "translate", "steering wheel", "high",
                {"detection": "steering-wheel anchor"},
            )
            continue

        extents = np.ptp(points, axis=0)
        diagonal = float(np.linalg.norm(extents))
        ahead = float((centroid[:2] - eye[:2]) @ f[:2])
        passenger_ahead = float(
            (centroid[:2] - passenger_eye[:2]) @ passenger_f[:2]
        )
        lat_signed = float(frame.side * (centroid[0] - wheel_x))
        below = frame.eye[2] - float(centroid[2])
        out80 = float(np.percentile(np.abs(points[:, 0] - cx0), 80))
        stats_ng = scan_no_glass.get(object_id, stats)
        passenger_stats = passenger_scan[object_id]
        passenger_stats_ng = passenger_scan_no_glass.get(
            object_id, passenger_stats
        )
        z70 = float(np.percentile(points[:, 2], 70))

        # oriented control cone: forward-and-down of the eye, laterally from
        # just inboard of the column out to the driver's door, no broad walls
        wall_lateral = extents[0] < 0.22 and extents[1] > 0.45 and extents[2] > 0.45
        in_cone = (
            0.20 <= ahead <= wheel_ahead + 1.0
            and -0.22 <= lat_signed <= _driver_control_outboard_limit(below)
            and -0.10 <= below <= 1.35
            and not wall_lateral
        )

        # Scope channels are candidates, not absolutes.  The cheap point shell
        # is only a broad phase: when it says a mesh might enter, trace those
        # same sample rays against the filled DAE triangles.  This catches a
        # body/carpet face covering a part even when none of the face's sparse
        # vertices happens to share the point's 6-degree angular bin.
        def candidate_channels(
            candidate_stats: dict[str, float],
            candidate_stats_ng: dict[str, float],
            candidate_passenger_stats: dict[str, float],
            candidate_passenger_stats_ng: dict[str, float],
            *,
            candidate_out80: float = out80,
            candidate_half_width: float = half_width,
            candidate_diagonal: float = diagonal,
            candidate_ahead: float = ahead,
            candidate_centroid: np.ndarray = centroid,
            candidate_passenger_ahead: float = passenger_ahead,
            candidate_in_cone: bool = in_cone,
        ) -> tuple[bool, bool, bool, bool, bool, bool]:
            visible = (
                candidate_stats["front_vf"] >= SPATIAL_VISIBLE_FRACTION
            )
            passenger_visible = (
                candidate_passenger_stats["front_vf"]
                >= SPATIAL_PASSENGER_VISIBLE_FRACTION
            )
            enclosed = _is_enclosed_candidate(
                candidate_stats,
                candidate_out80,
                candidate_half_width,
            )
            fitment = (
                candidate_stats_ng["vf"] >= 0.12
                and candidate_diagonal <= 0.60
                and 0.15 <= candidate_ahead <= 1.5
                and abs(float(candidate_centroid[2]) - frame.eye[2]) <= 0.7
                and abs(float(candidate_centroid[0]) - cx0) >= candidate_half_width - 0.06
            )
            passenger_fitment = (
                candidate_passenger_stats_ng["vf"] >= 0.12
                and candidate_diagonal <= 0.60
                and 0.15 <= candidate_passenger_ahead <= 1.5
                and abs(float(candidate_centroid[2]) - passenger_eye[2]) <= 0.7
                and abs(float(candidate_centroid[0]) - cx0) >= candidate_half_width - 0.06
            )
            buried = (
                candidate_stats["backed"] >= 0.75
                and candidate_stats["depth"] <= 0.35
            )
            cone = (
                candidate_in_cone
                and candidate_stats["depth"] <= 0.75
                and candidate_stats["min_r"] <= SPATIAL_REACH_LIMIT
                and (float(candidate_centroid[2]) >= floor_z - 0.10 or buried)
                and (
                    candidate_stats["vf"] >= 0.45
                    or candidate_stats["min_r"] <= wheel_dist + 0.45
                    or buried
                    or enclosed
                )
            )
            return (
                visible,
                passenger_visible,
                enclosed,
                fitment,
                passenger_fitment,
                cone,
            )

        (
            cand_visible,
            cand_passenger_visible,
            cand_enclosed,
            cand_fitment,
            cand_passenger_fitment,
            cand_cone,
        ) = candidate_channels(
            stats, stats_ng, passenger_stats, passenger_stats_ng
        )
        point_cand_fitment = cand_fitment
        if (surface_scene is not None or passenger_surface_scene is not None) and (
            cand_visible or cand_passenger_visible or cand_enclosed
            or cand_fitment or cand_passenger_fitment or cand_cone
            or object_id in under_seat_candidates
            or object_id in forced
        ):
            if surface_scene is not None:
                exact = exact_by_id.get(object_id)
                if object_id not in exact_by_id:
                    exact = core.surface_visibility_stats_batch(
                        {object_id: points},
                        frame.eye,
                        surface_scene,
                        frame.forward,
                    ).get(object_id)
                if exact is not None:
                    stats = dict(stats)
                    stats.update({key: exact[key] for key in ("vf", "front_vf")})
                    stats_ng = stats
                    if point_cand_fitment and surface_scene_no_glass is not None:
                        exact_ng = exact_no_glass_by_id.get(object_id)
                        if object_id not in exact_no_glass_by_id:
                            exact_ng = core.surface_visibility_stats_batch(
                                {object_id: points},
                                frame.eye,
                                surface_scene_no_glass,
                                frame.forward,
                            ).get(object_id)
                        if exact_ng is not None:
                            stats_ng = dict(stats)
                            stats_ng.update({
                                key: exact_ng[key] for key in ("vf", "front_vf")
                            })
            if passenger_surface_scene is not None and (
                cand_passenger_visible or cand_passenger_fitment
            ):
                passenger_exact = passenger_exact_by_id.get(object_id)
                if object_id not in passenger_exact_by_id:
                    passenger_exact = core.surface_visibility_stats_batch(
                        {object_id: points},
                        passenger_eye,
                        passenger_surface_scene,
                        passenger_f,
                    ).get(object_id)
                if passenger_exact is not None:
                    passenger_stats = dict(passenger_stats)
                    passenger_stats.update({
                        key: passenger_exact[key] for key in ("vf", "front_vf")
                    })
                    passenger_stats_ng = passenger_stats
                    if (
                        cand_passenger_fitment
                        and passenger_surface_scene_no_glass is not None
                    ):
                        passenger_exact_ng = passenger_exact_no_glass_by_id.get(
                            object_id
                        )
                        if object_id not in passenger_exact_no_glass_by_id:
                            passenger_exact_ng = core.surface_visibility_stats_batch(
                                {object_id: points},
                                passenger_eye,
                                passenger_surface_scene_no_glass,
                                passenger_f,
                            ).get(object_id)
                        if passenger_exact_ng is not None:
                            passenger_stats_ng = dict(passenger_stats)
                            passenger_stats_ng.update({
                                key: passenger_exact_ng[key]
                                for key in ("vf", "front_vf")
                            })
            (
                cand_visible,
                cand_passenger_visible,
                cand_enclosed,
                cand_fitment,
                cand_passenger_fitment,
                cand_cone,
            ) = candidate_channels(
                stats, stats_ng, passenger_stats, passenger_stats_ng
            )
        if not (
            cand_visible or cand_passenger_visible or cand_enclosed
            or cand_fitment or cand_passenger_fitment or cand_cone
            or object_id in under_seat_candidates
        ):
            continue
        if scoped is not None:
            scoped.add(object_id)

        detection_channels = []
        if cand_visible:
            detection_channels.append("forward visibility")
        if cand_passenger_visible:
            detection_channels.append("passenger forward visibility")
        if cand_enclosed:
            detection_channels.append("cabin enclosure shell")
        if cand_fitment:
            detection_channels.append("exterior driver fitment")
        if cand_passenger_fitment:
            detection_channels.append("exterior passenger fitment")
        if cand_cone:
            detection_channels.append("driver control cone")
        if object_id in under_seat_candidates:
            detection_channels.append("under-seat geometry")
        if object_id in forced:
            detection_channels.append("passenger-footwell forced candidate")
        detection = ", ".join(detection_channels)

        if not (cand_fitment or cand_passenger_fitment):
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
                    "translate", "steering column top (resolution floor: name hint)", "low",
                    {"detection": f"{detection}, steering-column name fallback"})
            else:
                verdicts[object_id] = (
                    "mirror", "steering column body (resolution floor: name hint)", "low",
                    {"detection": f"{detection}, steering-column name fallback"})
            continue

        if cand_cone:
            confidence = "high" if (abs(lat_signed) < 0.24 and ahead <= wheel_ahead + 0.75) else "med"
            verdicts[object_id] = (
                "translate", "in the driver control cone", confidence,
                {"detection": detection},
            )
            continue

        emissive = mesh_flag(object_id, "emissive", require_all=False)
        sds = core.principal_extent_sds(points)
        planar = sds[0] / max(sds[1], 1e-6) < 0.35 or stats["n"] < 40
        display = emissive and planar and diagonal <= 0.9

        orphans, coarse_fraction = _mesh_symmetry(context, object_id, points, cx0)
        if orphans == 0:
            xspan = float(np.ptp(points[:, 0]))
            z90 = float(np.percentile(points[:, 2], 90))
            fascia = (
                xspan >= max(1.05, 1.3 * half_width) and ahead >= 0.45
                and float(centroid[2]) <= frame.eye[2] - 0.18
                and z90 >= frame.eye[2] - 0.62
                and extents[2] >= 0.28 and stats["vf"] >= 0.30
            )
            if display:
                verdicts[object_id] = (
                    "mirror", "directional display", "med",
                    {"flip": True, "detection": detection},
                )
            elif fascia:
                # Geometrically symmetric, but a fascia may carry directional
                # materials or generated detail, so preserve the established
                # dashboard transform.
                verdicts[object_id] = (
                    "mirror", "dashboard fascia", "med",
                    {"detection": detection},
                )
            else:
                verdicts[object_id] = (
                    "none", "perfectly symmetric", "high",
                    {"detection": f"{detection}, exact self-symmetry"},
                )
            # else: symmetric about the centreline, reflection changes nothing
            continue

        visible_from_either_eye = cand_visible or cand_passenger_visible
        confidence = "low" if coarse_fraction < 0.05 else (
            "med" if not visible_from_either_eye else "high")
        if cand_fitment and not visible_from_either_eye:
            reason = "exterior driver fitment"
        elif cand_passenger_fitment and not visible_from_either_eye:
            reason = "exterior passenger fitment"
        else:
            reason = "one-sided interior part"
        if (out80 >= half_width - 0.05
                and max(stats["front_vf"], passenger_stats["front_vf"]) < 0.50
                and not (cand_fitment or cand_passenger_fitment)):
            confidence = "low"
            reason = "wall at the cabin shell (verify: possible exterior sheet)"
        lateral_center = float(
            (np.min(points[:, 0]) + np.max(points[:, 0])) / 2.0
        )
        mode = (
            "pairable"
            if abs(lateral_center - cx0) >= SPATIAL_PAIR_MIN_OFFSET
            else "mirror"
        )
        verdicts[object_id] = (
            mode, reason, confidence,
            {"flip": display, "detection": detection},
        )
    return verdicts, vetoed
