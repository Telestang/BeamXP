"""2026-09-10: Pedal assembly moves and surface-mounted trigger recommendations."""
from __future__ import annotations

import numpy as np

from .recommendation_common import (
    mesh_center,
    recommendation_matches,
    recommendation_text,
)
from .recommendation_geometry import geometry_for_ids, surface_distance
from .shared import core

PEDAL_FUNCS = frozenset({"brake", "clutch", "throttle"})

# 2026-09-11: True applies twice the pedal-channel mean to every complete
# assembly. Set False to restore the selective steering-distance checks.
APPLY_PEDAL_AVERAGE_TO_ALL = True


def driver_footrest(context, evidence, mesh, available, center_x, side):
    """2026-09-10: Fixed footrests have no channel; require pedal proximity too."""
    text = recommendation_text(context, mesh)
    if not recommendation_matches(text, (r"footrest|footstep|dead_?pedal",)):
        return False
    center = mesh_center(context, mesh)
    if center is None or (center[0] - center_x) * side < .15:
        return False
    for pedal in available:
        if not evidence.funcs(pedal) & PEDAL_FUNCS:
            continue
        at = mesh_center(context, pedal)
        if at is not None and abs(at[0] - center[0]) <= .25 and abs(at[1] - center[1]) <= .25 and abs(at[2] - center[2]) <= .25:
            return True
    return False


def steering_mount_meshes(context, evidence, reference):
    """Controls bound to a dedicated group consisting of the wheel's three anchors."""
    frames = evidence.prop_frames.get(reference, frozenset())
    if not frames:
        return frozenset()
    key = (id(evidence), reference)
    cached = getattr(context, "_recommendation_steering_mounts", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    anchors = {node for frame in frames for node in frame}
    texts, parsed, groups = {}, {}, set()
    for part in evidence.selected_part_ids:
        body = context.part_body_index.get(part, ("", ""))[0]
        if '"nodes"' not in body:
            continue
        text = core.part_named_array_for_context(context, part, "nodes") or "[]"
        texts[part] = text
        if not any(f'"{node}"' in text for node in anchors):
            continue
        parsed[part] = list(core.iter_jbeam_table_rows(text))
        for row in parsed[part]:
            if row.get("id") in anchors:
                groups.update(core.jbeam_group_names(row.get("group")))
    members = {group: set() for group in groups}
    for part, text in texts.items():
        if part not in parsed:
            if not any(f'"{group}"' in text for group in groups):
                continue
            parsed[part] = list(core.iter_jbeam_table_rows(text))
        for row in parsed[part]:
            for group in core.jbeam_group_names(row.get("group")):
                if group in members and isinstance(row.get("id"), str):
                    members[group].add(row["id"])
    # 2026-09-10: meshs.lua/processProps consumes the reference triad;
    # processFlexbodies consumes the resolved group nodes. An exact group
    # match is a dedicated mount, unlike the broad dashboard/body groups.
    dedicated = {group for group, nodes in members.items()
                 if frozenset(nodes) in {frozenset(frame) for frame in frames}}
    result = set()
    if dedicated:
        result.update(mesh for mesh, attached in evidence.prop_frames.items()
                      if attached and attached <= frames)
        for part in evidence.selected_part_ids:
            text = core.part_named_array_for_context(context, part, "flexbodies") or "[]"
            if not any(f'"{group}"' in text for group in dedicated):
                continue
            for row in core.iter_jbeam_table_rows(text):
                bound = set(core.jbeam_group_names(row.get("[group]:")))
                if bound and bound <= dedicated and isinstance(row.get("mesh"), str):
                    result.add(row["mesh"])
    context._recommendation_steering_mounts = (key, frozenset(result))
    return frozenset(result)


def pedal_offsets(context, evidence, moving, center_x, side, steering_delta=None):
    """Average control centres equally within nearby stock/shared assemblies."""
    pedals = {}
    for mesh in sorted(moving):
        channels = evidence.funcs(mesh) & PEDAL_FUNCS
        center = mesh_center(context, mesh)
        if channels and center is not None:
            pedals[mesh] = (channels, center)
    groups = []
    for mesh, (channels, center) in pedals.items():
        obj = context.objects[mesh]
        # 2026-09-10: Source identity separates optional race pedalboxes from
        # stock pedals; y/z locality separates assemblies sharing one DAE.
        source = (str(obj.dae_source_zip or context.source_zip), obj.dae_path)
        group = next((g for g in groups if g[0] == source and
                      abs(g[1][1] - center[1]) < .15 and abs(g[1][2] - center[2]) < .15), None)
        if group is None:
            group = [source, center, []]
            groups.append(group)
        group[2].append(mesh)
    result = {}
    for _, _, meshes in groups:
        by_channel = {}
        for mesh in meshes:
            for channel in pedals[mesh][0]:
                by_channel.setdefault(channel, []).append(pedals[mesh][1][0])
        if not {"brake", "throttle"} <= by_channel.keys():
            continue
        centers = [sum(values) / len(values) for values in by_channel.values()]
        offset = round(2 * (sum(centers) / len(centers) - center_x) * side, 3)
        # 2026-09-11: Selective mode keeps the earlier target-based filter:
        # recenter narrower two-pedal stations or outboard assemblies, while
        # other assemblies inherit the existing Move distance.
        if not APPLY_PEDAL_AVERAGE_TO_ALL and steering_delta is not None:
            narrower_pair = len(by_channel) == 2 and offset < steering_delta - .10
            outboard = offset > steering_delta + .06
            if not (narrower_pair or outboard):
                continue
        result.update(dict.fromkeys(meshes, offset))
    return result


def trigger_hosts(context, available, boxes):
    """Find a nearby filled surface, with a small control preferred over its fascia."""
    # 2026-09-10: Attribution depends on geometry and box placement, not on
    # the current conversion settings; reopening the modal reuses it.
    key = (id(context.preview_by_id), tuple(sorted(available)), tuple(
        tuple(box.get("centre") or box.get("at") or ()) for box in boxes))
    cached = getattr(context, "_recommendation_trigger_hosts", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    candidates = {}
    for index, box in enumerate(boxes):
        point = box.get("centre") or box.get("at")
        if point is None or len(point) != 3:
            continue
        point = np.asarray(point, dtype=float)
        near = []
        for mesh in available:
            # 2026-09-10: A representative placement cannot identify a host
            # consistently when selected trims put it in different places.
            if mesh in context.variant_dependent_meshes:
                continue
            bounds = context.preview_by_id.get(mesh, {}).get("bounds")
            if bounds is None:
                continue
            low, high = np.asarray(bounds, dtype=float)
            gap = np.maximum(np.maximum(low - point, point - high), 0)
            if np.linalg.norm(gap) <= .04:
                near.append((mesh, float(np.linalg.norm(high - low))))
        candidates[index] = near
    ids = {mesh for near in candidates.values() for mesh, _ in near}
    if not ids:
        return {}
    surfaces = geometry_for_ids(context, ids)
    result = {}
    for index, near in candidates.items():
        box = boxes[index]
        point = box.get("centre") or box.get("at")
        ranked = []
        for mesh, extent in near:
            distance = surface_distance(point, surfaces.get(mesh, {}).get("triangles", ()))
            if distance <= .04:
                ranked.append((distance, extent, mesh))
        if not ranked:
            continue
        closest = min(r[0] for r in ranked)
        # 2026-09-10: A button and the panel it sits on can be millimetres
        # apart. Prefer the control only within 6 mm of the nearest surface.
        shortlist = sorted((extent, distance, mesh) for distance, extent, mesh in ranked
                           if distance <= closest + .006)
        result[index] = shortlist[0][2]
    context._recommendation_trigger_hosts = (key, result)
    return result
