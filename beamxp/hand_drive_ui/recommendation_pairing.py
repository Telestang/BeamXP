from __future__ import annotations

from .recommendation_common import (
    SPATIAL_CONTACT_LIMIT,
    SPATIAL_PAIR_DISTANCE,
    SPATIAL_PAIR_MIN_OFFSET,
    _unscoped_contact_is_cabin_furniture,
)
from .shared import core


def _passenger_footwell_forced(
    frame: core.DriverFrame,
    present: list[str],
    entries_np: dict[str, object],
    modes: dict[str, tuple[str, str, str, dict]],
    hard_vetoed: set[str] | frozenset[str] = frozenset(),
) -> frozenset[str]:
    """Unclassified meshes to surface-rescan in the opposite footwell.

    The aim point is the reflected average of translated furniture below the
    wheel (pedals and their cluster).  Cone membership requests the exact
    filled-surface scan even when the sparse point shell reports no exposure;
    it does not itself grant admission or a verdict.
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
    inert_material_aliases = core.inert_material_alias_symbols(context)
    pairables = [o for o in present if memo.get(o, ("none",))[0] == "pairable"]
    latent = [
        o for o in present
        if memo.get(o, ("none",))[0] == "none" and o not in vetoed
        and o in entries_np and len(entries_np[o]) >= 4
    ]
    cx0 = frame.center_x
    lateral_centers = {
        object_id: float(
            (np.min(entries_np[object_id][:, 0]) + np.max(entries_np[object_id][:, 0]))
            / 2.0
        )
        for object_id in present
        if object_id in entries_np and len(entries_np[object_id])
    }
    pairable_set = set(pairables)
    pool = sorted(pairable_set | set(latent))
    candidates: list[tuple[float, float, str, str]] = []
    for index, object_id in enumerate(pool):
        points_a = entries_np[object_id]
        centroid_a = points_a.mean(axis=0)
        center_a_x = lateral_centers[object_id]
        offset_a = center_a_x - cx0
        if abs(offset_a) < SPATIAL_PAIR_MIN_OFFSET:
            continue  # centred asymmetric mesh: aesthetic Mirror, never a pair
        diag_a = float(np.linalg.norm(np.ptp(points_a, axis=0)))
        for twin_id in pool[index + 1:]:
            if object_id not in pairable_set and twin_id not in pairable_set:
                continue  # two under-admitted meshes cannot promote each other
            points_b = entries_np[twin_id]
            centroid_b = points_b.mean(axis=0)
            center_b_x = lateral_centers[twin_id]
            offset_b = center_b_x - cx0
            if abs(offset_b) < SPATIAL_PAIR_MIN_OFFSET or offset_a * offset_b >= 0.0:
                continue
            if abs(offset_a + offset_b) > 0.14:
                continue
            if (abs(float(centroid_a[1]) - float(centroid_b[1])) > 0.35
                    or abs(float(centroid_a[2]) - float(centroid_b[2])) > 0.35):
                continue
            diag_b = float(np.linalg.norm(np.ptp(points_b, axis=0)))
            if max(diag_a, diag_b) / max(min(diag_a, diag_b), 1e-6) > 1.5:
                continue
            distance = core.mirror_pair_distance(points_a, points_b, cx0)
            if distance > SPATIAL_PAIR_DISTANCE:
                continue
            combined_diagonal = float(np.linalg.norm(np.ptp(
                np.concatenate((points_a, points_b)), axis=0
            )))
            residual = distance / max(combined_diagonal, 0.05)
            candidates.append((residual, distance, object_id, twin_id))

    # Consider the complete candidate graph before consuming either endpoint.
    # This makes exact/strong reflected twins win over an earlier approximate
    # match while retaining the latter when no stronger pairing exists.
    used: set[str] = set()
    for _residual, _distance, object_id, twin_id in sorted(candidates):
        if object_id in used or twin_id in used:
            continue
        used.add(object_id)
        used.add(twin_id)
        symbols_a = set(material_symbols.get(object_id, ()))
        symbols_b = set(material_symbols.get(twin_id, ()))
        # Directional-material twins bind mutually exclusive symbols.
        # Multi-material housings may legitimately share their body material
        # while using a side-specific auxiliary symbol; those are still safe
        # structural pairs.
        distinct_symbols = symbols_a | symbols_b
        if (
            symbols_a
            and symbols_b
            and symbols_a.isdisjoint(symbols_b)
            and not distinct_symbols.issubset(inert_material_aliases)
        ):
            reason = (
                "functionally sided: materials differ, needs build-side material rebind"
            )
            memo[object_id] = (
                "functional_skip", reason, "high",
                {"detection": "structural twin, material-symbol veto"},
            )
            memo[twin_id] = (
                "functional_skip", reason, "high",
                {"detection": "structural twin, material-symbol veto"},
            )
            pair_votes.pop(object_id, None)
            pair_votes.pop(twin_id, None)
            for votes in pair_votes.values():
                votes.pop(object_id, None)
                votes.pop(twin_id, None)
            continue
        if memo.get(twin_id, ("none",))[0] == "none":
            memo[twin_id] = (
                "pairable", "geometric twin across the centreline", "med",
                {"detection": "structural geometric twin"},
            )
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
    and transparent panes use the same rules as every other mesh."""
    import numpy as np

    eye = np.array(frame.eye)
    passenger_eye = eye.copy()
    passenger_eye[0] = 2.0 * frame.center_x - eye[0]

    def centre_within_cabin_radius(points: object) -> bool:
        centre = points.mean(axis=0)
        return min(
            float(np.linalg.norm(centre - eye)),
            float(np.linalg.norm(centre - passenger_eye)),
        ) <= 1.6

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
            and centre_within_cabin_radius(entries_np[o])
        ]
        if not hosts:
            break
        changed = False
        for object_id in present:
            if (
                (
                    object_id not in scoped
                    and not _unscoped_contact_is_cabin_furniture(
                        entries_np.get(object_id),
                        frame,
                    )
                )
                or memo.get(object_id, ("none",))[0] != "none"
                or object_id in vetoed
            ):
                continue
            points = entries_np.get(object_id)
            if points is None or len(points) < 4:
                continue
            diag = float(np.linalg.norm(np.ptp(points, axis=0)))
            if diag > 0.70:
                continue  # furniture-sized: judged on its own evidence
            if diag < 0.14 and len(points) < 40:
                continue  # sub-resolution marker/dummy (engine light helpers)
            if not centre_within_cabin_radius(points):
                continue  # outside the cabin radius: not interior furniture
            for host in hosts:
                host_points = entries_np[host]
                gap2 = ((points[:, None, :] - host_points[None, :, :]) ** 2).sum(axis=2)
                if float(np.sqrt(gap2.min())) <= SPATIAL_CONTACT_LIMIT:
                    lateral_center = float(
                        (np.min(points[:, 0]) + np.max(points[:, 0])) / 2.0
                    )
                    inherited_mode = (
                        "pairable"
                        if abs(lateral_center - frame.center_x)
                        >= SPATIAL_PAIR_MIN_OFFSET
                        else "mirror"
                    )
                    memo[object_id] = (
                        inherited_mode, f"mounted on {host}", "low",
                        {"detection": f"contact mount within 2 cm of {host}"})
                    changed = True
                    break
        if not changed:
            break

    # A genuinely floating scoped mesh can still be recognised by occlusion:
    # if its eye rays continue into any transformed cabin/mirror furniture,
    # the floater is cabin furniture too. Contact inheritance above has
    # already consumed anything mounted within 2 cm.
    floaters: list[str] = []
    sightline_entries = dict(entries_np)
    forward = np.asarray(frame.forward, dtype=float)
    for object_id in present:
        if (object_id not in scoped or memo.get(object_id, ("none",))[0] != "none"
                or object_id in vetoed):
            continue
        points = entries_np.get(object_id)
        if points is None or len(points) < 4:
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
            "mirror", f"floating in front of {behind_class} geometry", "low",
            {"detection": f"forward sightline backed by {behind_class} geometry"},
        )
