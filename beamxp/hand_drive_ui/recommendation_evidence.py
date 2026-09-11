"""What the game itself says about a mesh, gathered once per vehicle.

Recommend Transforms used to reason from mesh names alone. Names are the last
thing BeamNG looks at, though: the engine decides what a mesh *does* from
sections it actually consumes, and those are cheap to read.

Three of them carry almost all the signal:

``props``
    A prop row's first column is its ``func`` -- the electrics channel the
    engine drives it from (``common/jbeam/sections/meshs.lua``, ``processProps``,
    reads ``prop.func``). ``steering``, ``brake``, ``throttle``, ``clutch``,
    ``wheelspeed`` name driver controls and instruments; ``parkingbrake`` and
    ``sequentialLeverY`` name console furniture. That is the distinction the
    conversion turns on, stated by the author and consumed by the engine.

``mirrors``
    A row here binds a real reflector (``sections/mirror.lua``). It marks the
    wing-mirror assembly without guessing at the dozen spellings of "mirror
    signal glass stalk".

``slotType``
    The slot a part fills is what the slot tree selects it by
    (``slotSystem.fillSlots_rec``), so it names the part's job: ``*_mirror_L``,
    ``*_doorpanel_FL``, ``*_pedals``, ``*_gauges``, ``*_dash``.

2026-09-07: scanning every part body in the vehicle for mesh names costs about
1.3 s (etkc, 2709 parts) because each one is comment-masked. Scoped to the
parts the selected trims actually resolve -- 146 to 576 of them -- it is 37 to
145 ms, so the scan takes the trim list and stays inside the modal's own
worker. The props and ``mirrors`` reads are ~40 ms over the whole vehicle and
are not scoped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from beamxp import transform_helpers
from beamxp.core import sjson

from .shared import core

# Cached as a plain attribute rather than a VehicleContext field on purpose:
# the context cache fingerprint covers the field names, so adding one would
# invalidate every built context on disk.
_EVIDENCE_ATTR = "_recommendation_evidence"

_SLOT_TYPE_RE = re.compile(r'"slotType"\s*:')


@dataclass(frozen=True)
class VehicleEvidence:
    """Per-mesh facts read off the jbeam the engine reads."""

    prop_funcs: dict[str, frozenset[str]] = field(default_factory=dict)
    slot_types: dict[str, frozenset[str]] = field(default_factory=dict)
    part_ids: dict[str, frozenset[str]] = field(default_factory=dict)
    meshes_by_part: dict[str, frozenset[str]] = field(default_factory=dict)
    mirror_meshes: frozenset[str] = frozenset()
    prop_frames: dict[str, frozenset[tuple[str, str, str]]] = field(default_factory=dict)
    selected_part_ids: tuple[str, ...] = ()
    material_funcs: dict[str, frozenset[str]] = field(default_factory=dict)

    def funcs(self, object_id: str) -> frozenset[str]:
        return self.prop_funcs.get(object_id, frozenset())

    def slots(self, object_id: str) -> frozenset[str]:
        return self.slot_types.get(object_id, frozenset())

    def siblings(self, object_id: str) -> frozenset[str]:
        """Every mesh declared by a part that also declares this one."""
        out: set[str] = set()
        for part_id in self.part_ids.get(object_id, ()):
            out.update(self.meshes_by_part.get(part_id, ()))
        out.discard(object_id)
        return frozenset(out)


def _prop_funcs_for_part(context: core.VehicleContext, part_id: str) -> list[tuple[str, str]]:
    """(func, mesh) for every prop row this part declares."""
    array_text = core.part_named_array_for_context(context, part_id, "props")
    if not array_text:
        return []
    rows: list[tuple[str, str]] = []
    for row in core.iter_jbeam_table_rows(array_text):
        func, mesh = row.get("func"), row.get("mesh")
        if isinstance(func, str) and isinstance(mesh, str) and mesh:
            rows.append((func, mesh))
    return rows


def _relevant_part_ids(
    context: core.VehicleContext,
    config_names: tuple[str, ...],
) -> list[str]:
    """The parts the selected trims resolve to, or every part as a fallback.

    ``selected_parts_for_config`` is memoised on the context and the parts
    table has already paid for it by the time the modal opens, so this is a
    dictionary read in practice. With no trims named -- a caller that has none
    to give -- the whole vehicle is scanned instead, which is correct but slow.
    """
    part_ids: set[str] = set()
    for config_name in config_names:
        try:
            selected = core.selected_parts_for_config(context, config_name)
        except Exception:
            continue
        part_ids.update(str(part_id) for part_id in selected.get("parts", ()))
    if part_ids:
        return sorted(part_ids)
    return sorted(context.part_body_index)


def vehicle_evidence(
    context: core.VehicleContext,
    config_names: tuple[str, ...] = (),
) -> VehicleEvidence:
    """The evidence for this vehicle and trim selection, built once.

    Keyed on the trim tuple as well as the part index, because a wider trim
    selection resolves more parts and so knows about more meshes.
    """
    part_index = context.part_body_index
    cached = getattr(context, _EVIDENCE_ATTR, None)
    key = (id(part_index), len(part_index), tuple(sorted(config_names)))
    if isinstance(cached, tuple) and cached[0] == key:
        return cached[1]

    prop_funcs: dict[str, set[str]] = {}
    slot_types: dict[str, set[str]] = {}
    part_ids: dict[str, set[str]] = {}
    meshes_by_part: dict[str, frozenset[str]] = {}
    prop_frames: dict[str, set[tuple[str, str, str]]] = {}
    material_funcs: dict[str, set[str]] = {}
    selected_part_ids = _relevant_part_ids(context, config_names)

    for part_id in selected_part_ids:
        found = part_index.get(part_id)
        if found is None:
            continue
        body = found[0]
        # 2026-09-10: materials.lua/process matches each glowMap base to its
        # bound meshes, then evaluates simpleFunction/advancedFunction.triggers.
        # These are actual indicator channels, independent of material spelling.
        if '"glowMap"' in body:
            glow = transform_helpers.extract_keyed_object(body, "glowMap")
            try:
                entries = sjson.decode(glow[glow.index("{"):]) if glow else {}
            except (ValueError, TypeError):
                entries = {}
            if isinstance(entries, dict):
                for material, entry in entries.items():
                    if not isinstance(entry, dict):
                        continue
                    simple = entry.get("simpleFunction")
                    funcs = {simple} if isinstance(simple, str) else set(simple) if isinstance(simple, dict) else set()
                    advanced = entry.get("advancedFunction")
                    if isinstance(advanced, dict) and isinstance(advanced.get("triggers"), list):
                        funcs.update(value for value in advanced["triggers"] if isinstance(value, str))
                    material_funcs.setdefault(str(material).lower(), set()).update(funcs)
        meshes = core.part_mesh_names_for_context(context, part_id)
        if not meshes:
            continue
        meshes_by_part[part_id] = frozenset(meshes)
        declared = (
            transform_helpers.extract_part_slot_types(body)
            if _SLOT_TYPE_RE.search(body)
            else []
        )
        for mesh in meshes:
            part_ids.setdefault(mesh, set()).add(part_id)
            if declared:
                slot_types.setdefault(mesh, set()).update(declared)
        if '"props"' in body:
            # 2026-09-10: Table rows support compact/multiline arrays and
            # modifier rows, matching tableSchema rather than line layout.
            props = core.part_named_array_for_context(context, part_id, "props")
            for row in core.iter_jbeam_table_rows(props or "[]"):
                func, mesh = row.get("func"), row.get("mesh")
                if not isinstance(func, str) or not isinstance(mesh, str) or not mesh:
                    continue
                prop_funcs.setdefault(mesh, set()).add(func)
                frame = tuple(row.get(key) for key in ("idRef:", "idX:", "idY:"))
                if all(isinstance(node, str) and node for node in frame):
                    prop_frames.setdefault(mesh, set()).add(frame)

    evidence = VehicleEvidence(
        prop_funcs={mesh: frozenset(funcs) for mesh, funcs in prop_funcs.items()},
        slot_types={mesh: frozenset(slots) for mesh, slots in slot_types.items()},
        part_ids={mesh: frozenset(parts) for mesh, parts in part_ids.items()},
        meshes_by_part=meshes_by_part,
        # Vehicle-wide on purpose: a wing mirror's reflector row can be
        # authored in a part a narrow trim selection never reaches, and the
        # read is ~13 ms.
        mirror_meshes=frozenset(core.authored_mirror_rows(context)),
        prop_frames={mesh: frozenset(frames) for mesh, frames in prop_frames.items()},
        selected_part_ids=tuple(selected_part_ids),
        material_funcs={material: frozenset(funcs) for material, funcs in material_funcs.items()},
    )
    setattr(context, _EVIDENCE_ATTR, (key, evidence))
    return evidence
