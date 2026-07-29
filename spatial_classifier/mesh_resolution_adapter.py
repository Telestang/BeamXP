"""Bridge the classifier child repo to BeamXP's authoritative mesh resolver.

The classifier keeps its large standalone geometry implementation, but the
question "which JBeam parts/meshes are fitted in this trim?" must come from the
parent BeamXP checkout.  Duplicating that resolver here is what allowed orphaned
.pc entries such as Miramar's dormant ``racing_seat_FR`` to leak into scoring.

This module imports ``beamxp.hand_drive_core`` from the parent checkout, then
exposes the small resolver/placement API the child repo needs.
"""

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterable

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SHARED_CORE_MODULE = "beamxp.hand_drive_core"
SHARED_CORE_PATH = ROOT / "beamxp" / "hand_drive_core.py"
_PRIVATE_MODULE_NAME = "_beamxp_classifier_shared_core"


def _load_shared_core() -> ModuleType | None:
    existing = sys.modules.get(_PRIVATE_MODULE_NAME)
    if isinstance(existing, ModuleType):
        return existing
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if not SHARED_CORE_PATH.is_file():
        return None
    try:
        module = importlib.import_module(SHARED_CORE_MODULE)
    except Exception:
        sys.modules.pop(_PRIVATE_MODULE_NAME, None)
        raise
    sys.modules[_PRIVATE_MODULE_NAME] = module
    return module


shared_core = _load_shared_core()


def require_shared_core() -> ModuleType:
    if shared_core is None:
        raise RuntimeError(
            "The classifier must live directly inside a BeamXP checkout. "
            f"Could not import the parent resolver {SHARED_CORE_MODULE} "
            f"from {SHARED_CORE_PATH}."
        )
    required = (
        "resolve_selected_parts",
        "selected_parts_for_config",
        "selected_node_positions_for_config",
        "mesh_roles_for_config",
        "used_meshes_for_config",
        "resolved_mesh_positions_for_config",
        "preview_entries_for_config",
        "object_number_property",
        "node_transform_ops",
        "node_translation_offset",
        "node_transform_matrix",
        "vector_from_row",
        "pos_after_node_transforms",
    )
    missing = [name for name in required if not hasattr(shared_core, name)]
    if missing:
        raise RuntimeError(
            "The parent BeamXP checkout is too old for this classifier: "
            f"missing {', '.join(missing)} in {SHARED_CORE_MODULE}."
        )
    return shared_core


def resolve_selected_parts(
    pc: dict[str, object],
    jbeam_texts: dict[str, str],
    *,
    vehicle_id: str,
    part_body_index: dict[str, tuple[str, str]] | None = None,
) -> dict[str, object]:
    core = require_shared_core()
    return core.resolve_selected_parts(
        pc,
        jbeam_texts,
        vehicle_id=vehicle_id,
        part_body_index=part_body_index,
    )


def selected_parts_for_config(context, config_name: str) -> dict[str, object]:
    return require_shared_core().selected_parts_for_config(context, config_name)


def selected_node_positions_for_config(
    context,
    config_name: str,
) -> dict[str, tuple[float, float, float]]:
    return require_shared_core().selected_node_positions_for_config(
        context,
        config_name,
    )


def selected_prop_mesh_positions(
    context,
    config_name: str,
    mesh_ids: set[str],
):
    core = require_shared_core()
    function = getattr(core, "selected_prop_mesh_positions", None)
    if function is None:
        raise RuntimeError("Parent BeamXP core has no selected_prop_mesh_positions")
    return function(context, config_name, mesh_ids)


def selected_flexbody_mesh_placements(
    context,
    config_name: str,
    mesh_ids: set[str],
):
    core = require_shared_core()
    function = getattr(core, "selected_flexbody_mesh_placements", None)
    if function is None:
        raise RuntimeError(
            "Parent BeamXP core has no selected_flexbody_mesh_placements"
        )
    return function(context, config_name, mesh_ids)


def mesh_roles_for_config(context, config_name: str):
    return require_shared_core().mesh_roles_for_config(context, config_name)


def used_meshes_for_config(context, config_name: str) -> set[str]:
    return set(require_shared_core().used_meshes_for_config(context, config_name))


def resolved_mesh_positions_for_config(context, config_name: str):
    return require_shared_core().resolved_mesh_positions_for_config(
        context,
        config_name,
    )


def preview_entries_for_config(context, config_name: str):
    return require_shared_core().preview_entries_for_config(context, config_name)


def selected_parts_in_merge_order(selected: dict[str, object]) -> list[str]:
    core = require_shared_core()
    function = getattr(core, "selected_parts_in_merge_order", None)
    if function is not None:
        return list(function(selected))
    order = selected.get("parts_order")
    parts = {str(item) for item in selected.get("parts", set())}
    if isinstance(order, list):
        result: list[str] = []
        for value in order:
            part_id = str(value)
            if part_id in parts and part_id not in result:
                result.append(part_id)
        result.extend(sorted(parts - set(result)))
        return result
    return sorted(parts)


def selected_part_instances(selected: dict[str, object]) -> list[dict[str, object]]:
    core = require_shared_core()
    function = getattr(core, "selected_part_instances", None)
    if function is not None:
        return [dict(item) for item in function(selected)]

    options_by_part = selected.get("part_slot_options", {})
    result: list[dict[str, object]] = []
    for index, part_id in enumerate(selected_parts_in_merge_order(selected)):
        raw_options = (
            options_by_part.get(part_id, ())
            if isinstance(options_by_part, dict)
            else ()
        )
        options = (
            tuple(str(item) for item in raw_options if item)
            if isinstance(raw_options, (list, tuple))
            else ()
        )
        result.append(
            {
                "instance_id": f"legacy:{index}:{part_id}",
                "part_id": part_id,
                "slot_path": "/",
                "inherited_options": options,
            }
        )
    return result


def part_instance_options(instance: dict[str, object]) -> tuple[str, ...]:
    core = require_shared_core()
    function = getattr(core, "part_instance_options", None)
    if function is not None:
        return tuple(function(instance))
    raw = instance.get("inherited_options", ())
    return (
        tuple(str(item) for item in raw if item)
        if isinstance(raw, (list, tuple))
        else ()
    )


def part_variable_scope(
    selected: dict[str, object],
    part_id: str,
) -> dict[str, object]:
    core = require_shared_core()
    function = getattr(core, "part_variable_scope", None)
    if function is not None:
        return dict(function(selected, part_id))
    scopes = selected.get("part_variables")
    if isinstance(scopes, dict) and isinstance(scopes.get(part_id), dict):
        return dict(scopes[part_id])
    return {}


def part_instance_variable_scope(
    selected: dict[str, object],
    instance: dict[str, object],
) -> dict[str, object]:
    core = require_shared_core()
    function = getattr(core, "part_instance_variable_scope", None)
    if function is not None:
        return dict(function(selected, instance))
    instance_id = str(instance.get("instance_id") or "")
    scopes = selected.get("part_instance_variables")
    if isinstance(scopes, dict) and isinstance(scopes.get(instance_id), dict):
        return dict(scopes[instance_id])
    return part_variable_scope(selected, str(instance.get("part_id") or ""))


def iter_node_rows(
    node_array: str,
    variables: dict[str, object] | None = None,
):
    core = require_shared_core()
    function = getattr(core, "iter_node_rows")
    try:
        parameter_count = len(inspect.signature(function).parameters)
    except (TypeError, ValueError):
        parameter_count = 1
    if parameter_count >= 2:
        return function(node_array, variables)
    return function(node_array)


def resolve_jbeam_row_strings(
    row: str,
    variables: dict[str, object] | None = None,
) -> str:
    core = require_shared_core()
    function = getattr(core, "resolve_jbeam_row_strings", None)
    if function is None:
        return row
    return str(function(row, variables))


def resolve_jbeam_value(
    value: str,
    variables: dict[str, object] | None = None,
) -> object:
    core = require_shared_core()
    function = getattr(core, "resolve_jbeam_value", None)
    if function is None:
        variables = variables or {}
        return variables.get(value) if value.startswith("$") else value
    return function(value, variables)



def object_number_property(
    object_text: str,
    key: str,
    variables: dict[str, object] | None = None,
) -> float | None:
    """Parent-core numeric property reader, including typed JBeam variables."""
    return require_shared_core().object_number_property(
        object_text,
        key,
        variables,
    )


def node_transform_ops(
    texts: Iterable[str],
    variables: dict[str, object] | None = None,
):
    """Parent-core slot/node transform merge semantics."""
    return require_shared_core().node_transform_ops(texts, variables)


def node_translation_offset(ops, pos_x_sign: int) -> tuple[float, float, float]:
    return require_shared_core().node_translation_offset(ops, pos_x_sign)


def node_transform_matrix(ops, pos_x: float):
    return require_shared_core().node_transform_matrix(ops, pos_x)


def vector_from_row(
    row: str,
    key: str,
    variables: dict[str, object] | None = None,
) -> tuple[float, float, float] | None:
    return require_shared_core().vector_from_row(row, key, variables)


def pos_after_node_transforms(
    row: str,
    position: tuple[float, float, float],
    inherited_options: Iterable[str] = (),
    variables: dict[str, object] | None = None,
) -> tuple[float, float, float]:
    return require_shared_core().pos_after_node_transforms(
        row,
        position,
        inherited_options,
        variables,
    )

def _resolved_needs_rebuild(context, object_id: str, placement) -> bool:
    matrices = getattr(placement, "matrices", ())
    variant_dependent = getattr(context, "variant_dependent_meshes", set())
    return len(matrices) > 1 or object_id in variant_dependent


def spatial_entries_for_trim(
    context,
    trim: str | None,
    available: Iterable[str],
) -> tuple[list[str], dict[str, np.ndarray]]:
    """Return only meshes selected by the authoritative BeamXP slot tree.

    This is the scorer's admission boundary.  A mesh mentioned by a .pc but not
    reached by the selected slot tree is absent even when its geometry exists in
    ``context.objects``/``preview_by_id``.
    """
    core = require_shared_core()
    available_set = set(available)
    if trim is None:
        present = sorted(available_set)
        entries = context.preview_by_id
        resolved = {}
    else:
        present = sorted(
            set(core.used_meshes_for_config(context, trim)) & available_set
        )
        entries = core.preview_entries_for_config(context, trim)
        resolved = core.resolved_mesh_positions_for_config(context, trim)

    arrays: dict[str, np.ndarray] = {}
    for object_id in present:
        placement = resolved.get(object_id)
        if placement is not None and _resolved_needs_rebuild(
            context,
            object_id,
            placement,
        ):
            rebuild = getattr(core, "vertex_cloud_for_resolved_placement", None)
            rebuilt = (
                rebuild(context, object_id, placement)
                if rebuild is not None
                else None
            )
            if rebuilt is not None:
                arrays[object_id] = np.asarray(rebuilt, dtype=float)
                continue
        item = entries.get(object_id) or context.preview_by_id.get(object_id)
        if item is None:
            continue
        arrays[object_id] = np.asarray(item["sample_points"], dtype=float)
    return [object_id for object_id in present if object_id in arrays], arrays


def spatial_surfaces_for_trim(
    context,
    trim: str | None,
    present: Iterable[str],
    entries_np: dict[str, object],
) -> dict[str, np.ndarray]:
    """Filled DAE surfaces at the same authoritative per-trim placement."""
    core = require_shared_core()
    present_list = list(present)
    base = core.full_surface_triangles_for_ids(context, present_list)
    resolved = (
        core.resolved_mesh_positions_for_config(context, trim)
        if trim is not None
        else {}
    )
    surfaces: dict[str, np.ndarray] = {}
    for object_id in present_list:
        placement = resolved.get(object_id)
        if placement is not None and _resolved_needs_rebuild(
            context,
            object_id,
            placement,
        ):
            rebuild = getattr(
                core,
                "surface_triangles_for_resolved_placement",
                None,
            )
            rebuilt = (
                rebuild(context, object_id, placement)
                if rebuild is not None
                else None
            )
            if rebuilt is not None and len(rebuilt):
                surfaces[object_id] = np.asarray(rebuilt, dtype=float)
                continue

        triangles = base.get(object_id)
        points = entries_np.get(object_id)
        preview = context.preview_by_id.get(object_id)
        if (
            triangles is None
            or len(triangles) == 0
            or points is None
            or preview is None
        ):
            continue
        triangles_np = np.asarray(triangles, dtype=float)
        points_np = np.asarray(points, dtype=float)
        placed_center = (
            np.min(points_np, axis=0) + np.max(points_np, axis=0)
        ) / 2.0
        preview_center = np.asarray(preview.get("center"), dtype=float)
        if preview_center.shape == (3,):
            delta = placed_center - preview_center
            if float(np.max(np.abs(delta))) > 1e-9:
                triangles_np = triangles_np + delta
        surfaces[object_id] = triangles_np
    return surfaces
