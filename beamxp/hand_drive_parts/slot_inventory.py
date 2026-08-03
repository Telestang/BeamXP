"""The tool's view of a vehicle's slots.

JBeam has no slot registry: a slot exists only as a row inside whichever part
declares it, and its identity across trims is its name. Everything else in
BeamXP is keyed by DAE mesh object id, which is the right key for "how should
this mesh be transformed" and the wrong key for "which side of the car does
this part live on". This module supplies the missing view so slot pairing can
be expressed, previewed and built.
"""

from __future__ import annotations

from collections.abc import Iterable

from beamxp import transform_helpers
from beamxp.core.models import (
    SlotDef,
    SlotUsage,
    VehicleContext,
)


def slot_usage_for_configs(
    context: VehicleContext,
    config_names: Iterable[str],
) -> dict[str, SlotUsage]:
    """Every slot reachable from the given trims, keyed by slot type.

    A slot that exists in the part tree but is empty in every trim is still
    reported: an empty slot on one side is exactly what a lone driver's seat
    needs to move into, so hiding it would hide the interesting half of most
    pairs.
    """
    part_by_config: dict[str, dict[str, str]] = {}
    paths_by_config: dict[str, dict[str, str]] = {}
    options_by_config: dict[str, dict[str, tuple[str, ...]]] = {}
    parent_parts: dict[str, list[str]] = {}

    configs = [name for name in config_names if name in context.variants]
    for config_name in configs:
        selected = selected_parts_for_config(context, config_name)
        selected_by_path = selected.get("selected_by_path", {})
        if not isinstance(selected_by_path, dict):
            selected_by_path = {}

        for instance in selected_part_instances(selected):
            part_id = str(instance.get("part_id") or "")
            found = part_body_for_context(context, part_id)
            if found is None:
                continue
            slot_path = str(instance.get("slot_path") or "")
            inherited = part_instance_options(instance)
            for slot_def in extract_slot_defs(found[0]):
                slot_type = slot_def.slot_type
                child_path = f"{slot_path}{slot_type}/"
                chosen = str(selected_by_path.get(child_path) or "")

                part_by_config.setdefault(slot_type, {})[config_name] = chosen
                paths_by_config.setdefault(slot_type, {})[config_name] = child_path
                child_options = tuple(inherited)
                if slot_def.options:
                    child_options = child_options + (slot_def.options,)
                options_by_config.setdefault(slot_type, {})[config_name] = child_options
                owners = parent_parts.setdefault(slot_type, [])
                if part_id not in owners:
                    owners.append(part_id)

    return {
        slot_type: SlotUsage(
            slot_type=slot_type,
            part_by_config=dict(part_by_config[slot_type]),
            paths_by_config=dict(paths_by_config.get(slot_type, {})),
            parent_parts=tuple(parent_parts.get(slot_type, ())),
            options_by_config=dict(options_by_config.get(slot_type, {})),
        )
        for slot_type in sorted(part_by_config)
    }


def slot_def_for_usage(
    context: VehicleContext,
    usage: SlotUsage,
) -> SlotDef | None:
    """The declaring slot row, taken from the first parent part that has one."""
    for parent_id in usage.parent_parts:
        found = part_body_for_context(context, parent_id)
        if found is None:
            continue
        for slot_def in extract_slot_defs(found[0]):
            if slot_def.slot_type == usage.slot_type:
                return slot_def
    return None


def slot_node_offset(
    usage: SlotUsage,
    config_name: str,
) -> tuple[float, float, float]:
    """The slot's authored node displacement for one trim.

    This is the only place the tool reads nodeMove/nodeOffset as a value rather
    than folding it into placement maths, and it is what makes an asymmetric
    pair tractable: a co-driver slot mounted further back and higher differs
    from the driver's slot by exactly this vector.
    """
    options = usage.options_by_config.get(config_name, ())
    ops = node_transform_ops(options)
    if not ops:
        return (0.0, 0.0, 0.0)
    # pos_x_sign 1: nodeOffset mirrors with the node's own side, and a slot has
    # no single side. Callers comparing two slots only use the difference, and
    # both are read on the same convention.
    return node_translation_offset(ops, 1)


def slot_anchor_position(
    context: VehicleContext,
    config_name: str,
    usage: SlotUsage,
) -> tuple[float, float, float] | None:
    """Where the slot's contents sit in this trim, or None when unknowable.

    Prefers the resolved position of the meshes the occupying part actually
    renders, because that is measured rather than declared. Falls back to the
    slot's own node displacement for an empty slot, which is the case that
    matters: an empty seat slot still has to report which side it is on.
    """
    part_id = str(usage.part_by_config.get(config_name) or "")
    if part_id:
        found = part_body_for_context(context, part_id)
        if found is not None:
            meshes = transform_helpers.extract_part_mesh_names(found[0])
            resolved = resolved_mesh_positions_for_config(context, config_name)
            points = [
                resolved[mesh].position for mesh in meshes if mesh in resolved
            ]
            if points:
                return average_position(points)

    offset = slot_node_offset(usage, config_name)
    if any(abs(value) > 1e-6 for value in offset):
        return offset
    return None


def slot_anchor_positions(
    context: VehicleContext,
    usage: SlotUsage,
    config_names: Iterable[str],
) -> dict[str, tuple[float, float, float]]:
    positions: dict[str, tuple[float, float, float]] = {}
    for config_name in config_names:
        position = slot_anchor_position(context, config_name, usage)
        if position is not None:
            positions[config_name] = position
    return positions


def representative_slot_anchor(
    context: VehicleContext,
    usage: SlotUsage,
    config_names: Iterable[str],
) -> tuple[float, float, float] | None:
    """One position for the slot across trims, for display and pair scoring."""
    positions = list(slot_anchor_positions(context, usage, config_names).values())
    if not positions:
        return None
    return average_position(positions)


__all__ = [
    'slot_usage_for_configs',
    'slot_def_for_usage',
    'slot_node_offset',
    'slot_anchor_position',
    'slot_anchor_positions',
    'representative_slot_anchor',
]
