"""Implementation slice extracted from ``beamng_hand_drive_core``.

Original source lines 3521-4068. Import the public orchestration module
``beamng_hand_drive_core`` rather than this implementation module directly.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from beamxp import transform_helpers
from beamxp.core import mesh_resolution
from beamxp.core import sjson
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
    MODE_REPLACE_SOURCE,
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
from beamxp.hand_drive_parts.spatial_analysis import display_texture_flip_scope

def find_part_body(
    part_id: str,
    jbeam_texts: dict[str, str],
    part_body_index: dict[str, tuple[str, str]] | None = None,
) -> tuple[str, str] | None:
    if part_body_index is not None:
        found = part_body_index.get(part_id)
        if found is not None:
            return found
    for name, text in jbeam_texts.items():
        body = transform_helpers.extract_keyed_object(text, part_id)
        if body is not None and '"slotType"' in body:
            return body, name
    return None


def part_body_for_context(context: VehicleContext, part_id: str) -> tuple[str, str] | None:
    return find_part_body(part_id, context.jbeam_texts, context.part_body_index)


def part_named_array_for_context(context: VehicleContext, part_id: str, array_key: str) -> str | None:
    cache_key = (part_id, array_key)
    if cache_key in context.part_array_cache:
        return context.part_array_cache[cache_key]
    found = part_body_for_context(context, part_id)
    if found is None:
        context.part_array_cache[cache_key] = None
        return None
    array_text = transform_helpers.extract_named_array(found[0], array_key)
    context.part_array_cache[cache_key] = array_text
    return array_text


def vehicle_namespace_main_part(
    vehicle_id: str,
    part_body_index: dict[str, tuple[str, str]] | None,
) -> str | None:
    """The part with slotType ``main`` declared in the vehicle's own namespace,
    mirroring jbeam ``io.getMainPartName``.

    BeamNG picks the root part by slot type, not by name: ``getMainPartName``
    returns ``partSlotMap[vehicleDir]['main'][1]``. The main part is usually
    named after the vehicle, but not always (mods especially), and many .pc
    files omit ``mainPartName`` entirely -- those must still find the right root
    instead of assuming a part literally named after the vehicle id exists.
    Scan is scoped to ``vehicles/<id>/`` so a common part never becomes the
    root, and sorted for a deterministic pick if a vehicle ships more than one.
    """
    if not part_body_index:
        return None
    prefix = f"vehicles/{vehicle_id}/"
    candidates = [
        part_id
        for part_id, (body, filename) in part_body_index.items()
        if filename.replace("\\", "/").startswith(prefix)
        and "main" in transform_helpers.extract_part_slot_types(body)
    ]
    return min(candidates) if candidates else None


def resolve_selected_parts(
    pc: dict[str, object],
    jbeam_texts: dict[str, str],
    *,
    vehicle_id: str,
    part_body_index: dict[str, tuple[str, str]] | None = None,
) -> dict[str, object]:
    explicit_parts = {
        str(slot_type): ("" if str(part_id) == "none" else str(part_id))
        for slot_type, part_id in dict(pc.get("parts", {})).items()
    }
    main_part = str(
        pc.get("mainPartName")
        or vehicle_namespace_main_part(vehicle_id, part_body_index)
        or vehicle_id
    )

    selected: set[str] = set()
    missing_parts: set[str] = set()
    parts_order: list[str] = []
    selected_by_slot: dict[str, str] = {"main": main_part}
    selected_by_path: dict[str, str] = {"/": main_part}
    part_slot_options: dict[str, tuple[str, ...]] = {main_part: ()}
    part_instances: list[dict[str, object]] = []
    cycles: list[dict[str, object]] = []
    instance_id_counts: dict[str, int] = {}

    queue: list[
        tuple[str, tuple[str, ...], str, str | None, str, tuple[str, ...]]
    ] = [(main_part, (), "/", None, "main", ())]

    for slot_type, part_id in explicit_parts.items():
        if part_id and "/" not in slot_type:
            selected_by_slot[slot_type] = part_id

    def user_choice_for(slot_id: str, slot_path: str) -> str | None:
        if slot_id in explicit_parts:
            return explicit_parts[slot_id]
        if slot_path in explicit_parts:
            return explicit_parts[slot_path]
        return None

    while queue:
        part_id, inherited_options, part_path, parent_instance_id, slot_id, ancestry = queue.pop(0)
        if not part_id:
            continue
        if part_id in ancestry:
            cycles.append({"part_id": part_id, "slot_path": part_path, "ancestry": ancestry})
            continue

        base_instance_id = f"{part_path}{part_id}"
        count = instance_id_counts.get(base_instance_id, 0)
        instance_id_counts[base_instance_id] = count + 1
        instance_id = base_instance_id if count == 0 else f"{base_instance_id}#{count + 1}"

        found = find_part_body(part_id, jbeam_texts, part_body_index)
        source_file = found[1] if found is not None else None
        part_instances.append(
            {
                "instance_id": instance_id,
                "part_id": part_id,
                "slot_id": slot_id,
                "slot_path": part_path,
                "parent_instance_id": parent_instance_id,
                "inherited_options": inherited_options,
                "source_file": source_file,
            }
        )

        selected.add(part_id)
        if part_id not in parts_order:
            parts_order.append(part_id)
        part_slot_options.setdefault(part_id, inherited_options)

        if found is None:
            missing_parts.add(part_id)
            continue
        part_body, _filename = found
        next_ancestry = ancestry + (part_id,)

        for slot_def in extract_slot_defs(part_body):
            slot_path = part_path + slot_def.slot_type + "/"
            user_choice = user_choice_for(slot_def.slot_type, slot_path)
            if user_choice is not None:
                if user_choice == "":
                    continue
                chosen = user_choice
                picked = find_part_body(chosen, jbeam_texts, part_body_index)
                if picked is None or not part_fits_slot(
                    transform_helpers.extract_part_slot_types(picked[0]), slot_def
                ):
                    chosen = slot_def.default_part
            else:
                chosen = slot_def.default_part
            if not chosen:
                continue

            selected_by_slot[slot_def.slot_type] = chosen
            selected_by_path[slot_path] = chosen
            child_options = list(inherited_options)
            if slot_def.options:
                child_options.append(slot_def.options)
            child_options_tuple = tuple(child_options)
            part_slot_options.setdefault(chosen, child_options_tuple)
            queue.append(
                (
                    chosen,
                    child_options_tuple,
                    slot_path,
                    instance_id,
                    slot_def.slot_type,
                    next_ancestry,
                )
            )

    user_vars = pc.get("vars")
    user_variable_map = user_vars if isinstance(user_vars, dict) else {}
    part_variables = build_part_variable_scopes(
        parts_order, part_slot_options, jbeam_texts, part_body_index, user_variable_map
    )
    part_instance_variables = build_part_instance_variable_scopes(
        part_instances, jbeam_texts, part_body_index, user_variable_map
    )

    return {
        "main_part": main_part,
        "parts": selected,
        "parts_order": parts_order,
        "part_instances": part_instances,
        "selected_by_slot": selected_by_slot,
        "selected_by_path": selected_by_path,
        "part_slot_options": part_slot_options,
        "part_variables": part_variables,
        "part_instance_variables": part_instance_variables,
        "missing_parts": missing_parts,
        "cycles": cycles,
    }



def selected_parts_for_config(context: VehicleContext, config_name: str) -> dict[str, object]:
    cached = context.selected_parts_cache.get(config_name)
    if cached is not None:
        return cached
    variant = context.variants[config_name]
    pc = load_pc(context.source_zip, variant.pc_path)
    selected = resolve_selected_parts(
        pc,
        context.jbeam_texts,
        vehicle_id=context.source_vehicle_id,
        part_body_index=context.part_body_index,
    )
    context.selected_parts_cache[config_name] = selected
    return selected


_PART_INFORMATION_NAME_RE = re.compile(
    r'"name"\s*:\s*"((?:[^"\\]|\\.)*)"',
    re.IGNORECASE,
)
_RHD_PART_TOKEN_RE = re.compile(
    r"(?<![a-z0-9])(?:rhd|right[-_ ]*hand(?:[-_ ]*drive)?|jdm|ukdm|uk)(?![a-z0-9])",
    re.IGNORECASE,
)
_LHD_PART_TOKEN_RE = re.compile(
    r"(?<![a-z0-9])(?:lhd|left[-_ ]*hand(?:[-_ ]*drive)?)(?![a-z0-9])",
    re.IGNORECASE,
)
_HAND_PART_TOKEN_RE = re.compile(
    r"(?<![a-z0-9])(?:rhd|lhd|right[-_ ]*hand(?:[-_ ]*drive)?|"
    r"left[-_ ]*hand(?:[-_ ]*drive)?|jdm|ukdm|uk)(?![a-z0-9])",
    re.IGNORECASE,
)


def _part_information_name(part_body: str) -> str:
    information = transform_helpers.extract_keyed_object(part_body, "information")
    if not information:
        return ""
    match = _PART_INFORMATION_NAME_RE.search(information)
    return match.group(1) if match is not None else ""


def _part_hand_label(part_id: str, part_body: str) -> str:
    return f"{part_id} {_part_information_name(part_body)}".strip()


def _part_hand_hint(part_id: str, part_body: str) -> str:
    label = _part_hand_label(part_id, part_body)
    if _RHD_PART_TOKEN_RE.search(label):
        return HAND_RHD
    if _LHD_PART_TOKEN_RE.search(label):
        return HAND_LHD
    return HAND_UNKNOWN


_SIDE_PART_TOKEN_RE = re.compile(
    r"(?<![a-z0-9])(?:"
    r"fl|fr|rl|rr|l|r|"
    r"left|right|frontleft|frontright|rearleft|rearright|"
    r"driver|passenger|codriver"
    r")(?![a-z0-9])",
    re.IGNORECASE,
)


def _handless_identity(value: str) -> str:
    stripped = _HAND_PART_TOKEN_RE.sub(" ", value.lower())
    return re.sub(r"[^a-z0-9]+", "", stripped)


def _sideless_identity(value: str) -> str:
    """Identity with left/right markers removed.

    The counterpart of a slot pair differs from its source only by side, so
    stripping the side token is what lets ``bx_race_seat_FL`` recognise
    ``bx_race_seat_FR`` as the same part on the other side of the car while
    still telling it apart from the plain ``bx_seat_FR``.
    """
    stripped = _SIDE_PART_TOKEN_RE.sub(" ", value.lower())
    return re.sub(r"[^a-z0-9]+", "", stripped)


def _part_identity_score(
    source_part_id: str,
    source_body: str,
    candidate_id: str,
    candidate_body: str,
    identity: Callable[[str], str] = _handless_identity,
) -> tuple[int, int]:
    source_id = identity(source_part_id)
    candidate_part_id = identity(candidate_id)
    id_match = bool(source_id and source_id == candidate_part_id)

    source_name = identity(_part_information_name(source_body))
    candidate_name = identity(_part_information_name(candidate_body))
    name_match = bool(source_name and source_name == candidate_name)
    return int(id_match), int(name_match)


def _hand_authored_counterpart(
    context: VehicleContext,
    source_part_id: str,
    target_slot: SlotDef,
    source_hand: str,
    target_hand: str,
) -> str | None:
    found = part_body_for_context(context, source_part_id)
    if found is None:
        return None
    source_body = found[0]
    source_hint = _part_hand_hint(source_part_id, source_body)
    if source_hint not in {source_hand, HAND_UNKNOWN}:
        return None

    ranked: list[tuple[tuple[int, int, int, int], str]] = []
    for candidate_id, (candidate_body, _candidate_file) in context.part_body_index.items():
        if candidate_id == source_part_id:
            continue
        candidate_slot_types = transform_helpers.extract_part_slot_types(candidate_body)
        if not part_fits_slot(candidate_slot_types, target_slot):
            continue

        candidate_hint = _part_hand_hint(candidate_id, candidate_body)
        if candidate_hint not in {target_hand, HAND_UNKNOWN}:
            continue
        if source_hint == HAND_UNKNOWN and candidate_hint == HAND_UNKNOWN:
            continue

        identity_score = _part_identity_score(
            source_part_id,
            source_body,
            candidate_id,
            candidate_body,
            _handless_identity,
        )
        if identity_score == (0, 0):
            continue
        score = (
            identity_score[0],
            identity_score[1],
            int(candidate_hint == target_hand),
            int(source_hint == source_hand),
        )
        ranked.append((score, candidate_id))

    if not ranked:
        return None
    ranked.sort(key=lambda item: tuple(-value for value in item[0]) + (item[1],))
    best_score = ranked[0][0]
    if sum(score == best_score for score, _candidate_id in ranked) != 1:
        return None
    return ranked[0][1]


def _sideless_counterpart(
    context: VehicleContext,
    source_part_id: str,
    target_slot: SlotDef,
) -> str | None:
    """The same part authored for the other side, fitting ``target_slot``.

    Slot fitment already pins the side -- ``bx_race_seat_FR`` declares
    ``slotType: bx_seat_FR`` and simply cannot go anywhere else -- so unlike
    the LHD/RHD search this needs no side hint of its own, only a unique
    identity match among the parts that fit.
    """
    found = part_body_for_context(context, source_part_id)
    if found is None:
        return None
    source_body = found[0]

    ranked: list[tuple[tuple[int, int], str]] = []
    for candidate_id, (candidate_body, _candidate_file) in context.part_body_index.items():
        if candidate_id == source_part_id:
            continue
        candidate_slot_types = transform_helpers.extract_part_slot_types(candidate_body)
        if not part_fits_slot(candidate_slot_types, target_slot):
            continue
        identity_score = _part_identity_score(
            source_part_id,
            source_body,
            candidate_id,
            candidate_body,
            _sideless_identity,
        )
        if identity_score == (0, 0):
            continue
        ranked.append((identity_score, candidate_id))

    if not ranked:
        return None
    ranked.sort(key=lambda item: tuple(-value for value in item[0]) + (item[1],))
    best_score = ranked[0][0]
    if sum(score == best_score for score, _candidate_id in ranked) != 1:
        return None
    return ranked[0][1]


def _paired_target_slot(
    source_slot: SlotDef,
    target_slots: Iterable[SlotDef],
    identity: Callable[[str], str] = _handless_identity,
) -> SlotDef | None:
    source_identity = identity(source_slot.slot_type)
    source_default = identity(source_slot.default_part)
    ranked: list[tuple[tuple[int, int, int], SlotDef]] = []
    for target_slot in target_slots:
        target_identity = identity(target_slot.slot_type)
        target_default = identity(target_slot.default_part)
        exact = source_slot.slot_type == target_slot.slot_type
        identity_match = bool(source_identity and source_identity == target_identity)
        default_match = bool(source_default and source_default == target_default)
        if not exact and not identity_match and not default_match:
            continue
        ranked.append(
            ((int(exact), int(identity_match), int(default_match)), target_slot)
        )

    if not ranked:
        return None
    ranked.sort(
        key=lambda item: tuple(-value for value in item[0]) + (item[1].slot_type,)
    )
    best_score = ranked[0][0]
    if sum(score == best_score for score, _slot in ranked) != 1:
        return None
    return ranked[0][1]


def _selected_child_part(
    selected: dict[str, object],
    slot_type: str,
    slot_path: str,
) -> str:
    selected_by_path = selected.get("selected_by_path", {})
    if isinstance(selected_by_path, dict):
        value = selected_by_path.get(slot_path)
        if value:
            return str(value)
    selected_by_slot = selected.get("selected_by_slot", {})
    if isinstance(selected_by_slot, dict):
        value = selected_by_slot.get(slot_type)
        if value:
            return str(value)
    return ""


@dataclass(frozen=True)
class _PairStrategy:
    """How one flavour of opposite-side pairing recognises its counterparts.

    Two flavours exist. The LHD/RHD flavour pairs an authored dashboard subtree
    with its opposite-hand twin and is driven by hand tokens in part names. The
    side flavour pairs two slots the user nominated and is driven by slot
    fitment plus left/right tokens. The subtree walk below is identical for
    both, so it takes the differences as callables rather than branching.
    """

    identity: Callable[[str], str]
    counterpart: Callable[[VehicleContext, str, SlotDef], "str | None"]
    carries_across: Callable[[str, str], bool]


def _hand_pair_strategy(source_hand: str, target_hand: str) -> _PairStrategy:
    return _PairStrategy(
        identity=_handless_identity,
        counterpart=lambda context, part_id, slot: _hand_authored_counterpart(
            context, part_id, slot, source_hand, target_hand
        ),
        # A part with no hand of its own, or one already authored for the
        # target hand, is reused verbatim in the swapped subtree.
        carries_across=lambda part_id, body: _part_hand_hint(part_id, body) != source_hand,
    )


def _side_pair_strategy() -> _PairStrategy:
    return _PairStrategy(
        identity=_sideless_identity,
        counterpart=_sideless_counterpart,
        # Only a part with no side marker at all carries across untouched; a
        # sided one needs its counterpart or the swap has moved nothing.
        carries_across=lambda part_id, body: _sideless_identity(part_id)
        == re.sub(r"[^a-z0-9]+", "", part_id.lower()),
    )


def _map_paired_children(
    context: VehicleContext,
    selected: dict[str, object],
    source_part_id: str,
    target_part_id: str,
    source_path: str,
    target_path: str,
    strategy: _PairStrategy,
    selections: list[dict[str, object]],
    clears: list[dict[str, str]],
    covered: set[str],
    visited: set[tuple[str, str, str, str]],
) -> None:
    visit = (source_part_id, target_part_id, source_path, target_path)
    if visit in visited:
        return
    visited.add(visit)

    source_found = part_body_for_context(context, source_part_id)
    target_found = part_body_for_context(context, target_part_id)
    if source_found is None or target_found is None:
        return
    source_slots = extract_slot_defs(source_found[0])
    target_slots = extract_slot_defs(target_found[0])

    for source_slot in source_slots:
        source_child_path = f"{source_path}{source_slot.slot_type}/"
        source_choice = _selected_child_part(
            selected,
            source_slot.slot_type,
            source_child_path,
        )
        if not source_choice:
            continue

        # Whatever the swap does with this choice -- replace it, drop it, or
        # carry it across unchanged -- the authored subtree owns it, so the
        # generated mirroring pass must leave it alone.
        covered.add(source_choice)

        target_slot = _paired_target_slot(source_slot, target_slots, strategy.identity)
        if target_slot is None:
            clears.append(
                {"slotId": source_slot.slot_type, "slotPath": source_child_path}
            )
            continue

        target_child_path = f"{target_path}{target_slot.slot_type}/"
        target_choice = ""
        source_choice_found = part_body_for_context(context, source_choice)
        if source_choice_found is not None:
            source_choice_body = source_choice_found[0]
            source_choice_types = transform_helpers.extract_part_slot_types(
                source_choice_body
            )
            if part_fits_slot(
                source_choice_types, target_slot
            ) and strategy.carries_across(source_choice, source_choice_body):
                target_choice = source_choice

        if not target_choice:
            target_choice = strategy.counterpart(context, source_choice, target_slot) or ""

        if source_slot.slot_type != target_slot.slot_type:
            clears.append(
                {"slotId": source_slot.slot_type, "slotPath": source_child_path}
            )
        if not target_choice:
            continue

        selections.append(
            {
                "sourceSlotId": source_slot.slot_type,
                "sourceSlotPath": source_child_path,
                "slotId": target_slot.slot_type,
                "slotPath": target_child_path,
                "partId": target_choice,
            }
        )
        _map_paired_children(
            context,
            selected,
            source_choice,
            target_choice,
            source_child_path,
            target_child_path,
            strategy,
            selections,
            clears,
            covered,
            visited,
        )


def find_hand_authored_opposite_group(
    context: VehicleContext,
    config_name: str,
    source_hand: str,
    target_hand: str,
) -> dict[str, object] | None:
    """Find a selected authored LHD/RHD subtree and its opposite root.

    The root parts must fit the same slot and have matching identities after
    hand labels are removed. A leaf pair such as two steering wheels is not a
    group: either root must own child slots. Child selections are translated to
    the opposite root's slot namespace so trim-specific choices such as manual
    shifters survive the swap.
    """
    if source_hand not in {HAND_LHD, HAND_RHD}:
        return None
    if target_hand not in {HAND_LHD, HAND_RHD} or target_hand == source_hand:
        return None

    selected = selected_parts_for_config(context, config_name)
    instances = selected_part_instances(selected)
    roots: list[tuple[tuple[int, int, int, int], dict[str, object], str]] = []

    for order, instance in enumerate(instances):
        source_part_id = str(instance.get("part_id") or "")
        slot_id = str(instance.get("slot_id") or "")
        slot_path = str(instance.get("slot_path") or "")
        if not source_part_id or not slot_id or not slot_path:
            continue
        target_slot = SlotDef(slot_id, "", allow_types=(slot_id,))
        counterpart = _hand_authored_counterpart(
            context,
            source_part_id,
            target_slot,
            source_hand,
            target_hand,
        )
        if counterpart is None:
            continue

        source_found = part_body_for_context(context, source_part_id)
        target_found = part_body_for_context(context, counterpart)
        if source_found is None or target_found is None:
            continue
        source_slots = extract_slot_defs(source_found[0])
        target_slots = extract_slot_defs(target_found[0])
        if not source_slots and not target_slots:
            continue

        depth = len([segment for segment in slot_path.split("/") if segment])
        descendant_count = sum(
            1
            for other in instances
            if other is not instance
            and str(other.get("slot_path") or "").startswith(slot_path)
        )
        label = _part_hand_label(source_part_id, source_found[0]).lower()
        group_keyword = int(
            any(
                token in label
                for token in ("dash", "cockpit", "interior", "cabin", "chassis", "body")
            )
        )
        roots.append(
            ((depth, -descendant_count, -group_keyword, order), instance, counterpart)
        )

    if not roots:
        return None
    roots.sort(key=lambda item: item[0])
    _score, root, target_root = roots[0]
    root_slot = str(root["slot_id"])
    root_path = str(root["slot_path"])

    selections: list[dict[str, object]] = []
    clears: list[dict[str, str]] = []
    covered: set[str] = {str(root["part_id"])}
    if root_slot != "main":
        selections.append(
            {
                "sourceSlotId": root_slot,
                "sourceSlotPath": root_path,
                "slotId": root_slot,
                "slotPath": root_path,
                "partId": target_root,
            }
        )
    _map_paired_children(
        context,
        selected,
        str(root["part_id"]),
        target_root,
        root_path,
        root_path,
        _hand_pair_strategy(source_hand, target_hand),
        selections,
        clears,
        covered,
        set(),
    )

    return {
        "rootSlot": root_slot,
        "rootPath": root_path,
        "sourcePart": str(root["part_id"]),
        "targetPart": target_root,
        "mainPart": target_root if root_slot == "main" else None,
        "selections": selections,
        "clears": clears,
        # Source parts the authored swap already resolves. The generated
        # mirroring pass skips these and still runs for everything else in the
        # trim, so seats, pedals and mirrors outside the authored subtree are
        # not silently left on the stock side.
        "sourceParts": sorted(covered),
    }

def _slot_pair_move(
    context: VehicleContext,
    selected: dict[str, object],
    source_slot: str,
    source_path: str,
    source_part: str,
    target_slot_def: SlotDef,
    target_slot: str,
    target_path: str,
    selections: list[dict[str, object]],
    relocations: list[dict[str, object]],
    clears: list[dict[str, str]],
    covered: set[str],
) -> bool:
    """Send one slot's occupant to the paired slot. True when a part landed."""
    strategy = _side_pair_strategy()
    covered.add(source_part)

    target_choice = ""
    found = part_body_for_context(context, source_part)
    if found is not None and part_fits_slot(
        transform_helpers.extract_part_slot_types(found[0]), target_slot_def
    ) and strategy.carries_across(source_part, found[0]):
        # A side-neutral part needs no counterpart: it simply moves.
        target_choice = source_part
    if not target_choice:
        target_choice = _sideless_counterpart(context, source_part, target_slot_def) or ""

    if not target_choice:
        # Nothing authored for the other side, so the part itself has to be
        # rebuilt there. The build emits a relocation clone; the .pc still
        # points the target slot at it, hence no selection here.
        relocations.append(
            {
                "sourceSlotId": source_slot,
                "sourceSlotPath": source_path,
                "slotId": target_slot,
                "slotPath": target_path,
                "partId": source_part,
            }
        )
        return True

    selections.append(
        {
            "sourceSlotId": source_slot,
            "sourceSlotPath": source_path,
            "slotId": target_slot,
            "slotPath": target_path,
            "partId": target_choice,
        }
    )
    _map_paired_children(
        context,
        selected,
        source_part,
        target_choice,
        source_path,
        target_path,
        strategy,
        selections,
        clears,
        covered,
        set(),
    )
    return True


def resolve_slot_pair_plan(
    context: VehicleContext,
    config_name: str,
    slot_pairs: Iterable[tuple[str, str]],
) -> dict[str, object] | None:
    """Turn the user's slot pairings into .pc edits for one trim.

    Each pair is resolved independently and the results are merged, so a trim
    can carry a seat swap, a pedal swap and a mirror swap at once -- unlike the
    authored LHD/RHD group, which only ever applies its single best root.

    Three outcomes per pair, in the order they are preferred:
      * both sides already hold the same part -- nothing to do;
      * the other side has an authored counterpart part -- a pure .pc swap,
        which keeps stock geometry and stock physics;
      * no counterpart exists -- a relocation clone, recorded for the build.
    """
    selected = selected_parts_for_config(context, config_name)
    usage = slot_usage_for_configs(context, [config_name])

    selections: list[dict[str, object]] = []
    relocations: list[dict[str, object]] = []
    clears: list[dict[str, str]] = []
    covered: set[str] = set()

    for slot_a, slot_b in slot_pairs:
        usage_a = usage.get(slot_a)
        usage_b = usage.get(slot_b)
        if usage_a is None or usage_b is None:
            continue
        part_a = str(usage_a.part_by_config.get(config_name) or "")
        part_b = str(usage_b.part_by_config.get(config_name) or "")
        if not part_a and not part_b:
            continue
        if part_a and part_b and _sideless_identity(part_a) == _sideless_identity(part_b):
            # Both sides already carry the same part; a swap would be a no-op
            # and would only churn the .pc.
            continue

        path_a = str(usage_a.paths_by_config.get(config_name) or f"/{slot_a}/")
        path_b = str(usage_b.paths_by_config.get(config_name) or f"/{slot_b}/")
        def_a = slot_def_for_usage(context, usage_a) or SlotDef(slot_a, "", allow_types=(slot_a,))
        def_b = slot_def_for_usage(context, usage_b) or SlotDef(slot_b, "", allow_types=(slot_b,))

        filled: set[str] = set()
        if part_a and _slot_pair_move(
            context, selected, slot_a, path_a, part_a,
            def_b, slot_b, path_b, selections, relocations, clears, covered,
        ):
            filled.add(slot_b)
        if part_b and _slot_pair_move(
            context, selected, slot_b, path_b, part_b,
            def_a, slot_a, path_a, selections, relocations, clears, covered,
        ):
            filled.add(slot_a)

        # A slot nothing moved into must be emptied explicitly. Dropping the
        # .pc key would let the slot fall back to its authored default, which
        # for bx_seat_FL is the very seat that just moved to the other side.
        for slot_type, path, part_id in (
            (slot_a, path_a, part_a),
            (slot_b, path_b, part_b),
        ):
            if part_id and slot_type not in filled:
                clears.append(
                    {"slotId": slot_type, "slotPath": path, "setEmpty": "1"}
                )

    if not selections and not relocations and not clears:
        return None
    return {
        "selections": selections,
        "relocations": relocations,
        "clears": clears,
        "sourceParts": sorted(covered),
    }


def slot_pair_plans_for_variants(
    context: VehicleContext,
    conversion: dict[str, object],
    config_names: Iterable[str],
) -> dict[str, dict[str, object]]:
    pairs = active_slot_pairs(conversion)
    if not pairs:
        return {}
    plans: dict[str, dict[str, object]] = {}
    for config_name in config_names:
        plan = resolve_slot_pair_plan(context, config_name, pairs)
        if plan is not None:
            plans[config_name] = plan
    return plans


def slot_pair_plan_relocations(plan: dict[str, object] | None) -> list[dict[str, object]]:
    if not plan:
        return []
    relocations = plan.get("relocations", ())
    if not isinstance(relocations, (list, tuple)):
        return []
    return [entry for entry in relocations if isinstance(entry, dict)]


def authored_group_source_parts(group: dict[str, object] | None) -> set[str]:
    """Source part ids an authored LHD/RHD swap already resolves."""
    if not group:
        return set()
    raw = group.get("sourceParts", ())
    if not isinstance(raw, (list, tuple, set, frozenset)):
        return set()
    return {str(part_id) for part_id in raw if part_id}


def authored_group_meshes(
    context: VehicleContext,
    group: dict[str, object] | None,
) -> set[str]:
    """Meshes owned by an authored swap's source subtree.

    Excluded from the generated mirroring scope so the build does not bake a
    mirrored DAE for a dashboard that the stock opposite-hand part replaces
    outright.
    """
    meshes: set[str] = set()
    for part_id in authored_group_source_parts(group):
        found = part_body_for_context(context, part_id)
        if found is None:
            continue
        meshes.update(transform_helpers.extract_part_mesh_names(found[0]))
    return meshes


def part_variable_scope(selected: dict[str, object], part_id: str) -> dict[str, float]:
    """The resolved variable values in force inside a selected part (empty when
    the part declares/inherits none), used to evaluate its $ expressions."""
    return mesh_resolution.part_variable_scope(selected, part_id)


_NODE_ROW_RE = re.compile(
    rf'^\s*\[\s*"(?P<id>(?:[^"\\]|\\.)*)"\s*,\s*'
    rf'(?P<x>{NUMBER_RE})\s*,\s*(?P<y>{NUMBER_RE})\s*,\s*(?P<z>{NUMBER_RE})'
)


# BEAMXP_PART_INSTANCE_FIX_V1: compatibility accessors for path-specific selected
# part occurrences.
def selected_part_instances(selected: dict[str, object]) -> list[dict[str, object]]:
    return mesh_resolution.selected_part_instances(selected)


def part_instance_options(instance: dict[str, object]) -> tuple[str, ...]:
    return mesh_resolution.part_instance_options(instance)


def part_instance_variable_scope(
    selected: dict[str, object],
    instance: dict[str, object],
) -> dict[str, object]:
    return mesh_resolution.part_instance_variable_scope(selected, instance)

def iter_node_rows(
    node_array: str,
    variables: dict[str, object] | None = None,
) -> Iterable[tuple[str, tuple[float, float, float], str]]:
    """Yield active node rows after resolving instance-specific expressions."""
    for raw_row in iter_active_top_level_rows(node_array):
        row = resolve_jbeam_row_strings(raw_row, variables)
        match = _NODE_ROW_RE.match(row)
        if match is None:
            continue
        node_id = match.group("id")
        if node_id in {"id", "type", "mesh", "func"}:
            continue
        position = (
            float(match.group("x")),
            float(match.group("y")),
            float(match.group("z")),
        )
        yield node_id, position, row


# Columns of a wheel row whose value names a group the engine puts on the
# hub/hubcap/tire nodes it generates at spawn (wheels.lua:
# ``hubcapOptions.group = hubcapOptions.hubcapGroup or hubcapOptions.group``).
# Those nodes exist in no ``nodes`` section, so brake discs, calipers and
# hubcaps look unbound unless these are counted.
WHEEL_GROUP_KEYS = ("group", "hubGroup", "hubcapGroup", "hubcapBreakGroup")
WHEEL_SECTIONS = ("pressureWheels", "hubWheels", "wheels")


def jbeam_group_names(value: object) -> tuple[str, ...]:
    """Group names a ``group``-style cell refers to.

    Per BeamNG's Nodes documentation, "when the argument is a string, it
    specifies the exact name of the group the node belongs to" and "when the
    argument is a table, the node belongs to all the specified groups at once".
    ``{"group":""}`` names nothing, which is how the documented reset works.
    """
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value if str(item))
    if isinstance(value, str):
        return (value,) if value else ()
    return ()


def iter_jbeam_table_rows(array_text: str) -> Iterable[dict[str, object]]:
    """Yield each row of a jbeam table section as a resolved key/value dict.

    Mirrors tableSchema.lua: the first row is the header, a dict row merges into
    the options carried forward to every later row (``tableMerge(
    ctx.localOptions, ...)``), and a trailing dict on a list row is that row's
    inline options.

    Decoded with the SJSON reader rather than iter_active_top_level_rows, which
    collects only ``[...]`` rows and walks straight through ``{...}`` -- it
    would hand back a modifier's inner array as though it were a data row.
    """
    try:
        rows = sjson.decode(array_text)
    except Exception:
        return
    if not isinstance(rows, list) or not rows:
        return
    header = rows[0]
    if not isinstance(header, list):
        return
    columns = [str(name) for name in header]
    options: dict[str, object] = {}
    for row in rows[1:]:
        if isinstance(row, dict):
            options.update(row)
            continue
        if not isinstance(row, list):
            continue
        merged: dict[str, object] = dict(options)
        for index, value in enumerate(row):
            if isinstance(value, dict):
                merged.update(value)  # inline options column
            elif index < len(columns):
                merged[columns[index]] = value
        yield merged


def node_group_names(nodes_array: str) -> set[str]:
    """Groups owning at least one real node row in this ``nodes`` section."""
    found: set[str] = set()
    for row in iter_jbeam_table_rows(nodes_array):
        node_id = row.get("id")
        if not isinstance(node_id, str) or node_id in {"id", "type", "mesh", "func"}:
            continue
        found.update(jbeam_group_names(row.get("group")))
    return found


def wheel_group_names(wheels_array: str) -> set[str]:
    """Groups the engine assigns to the nodes it generates for each wheel."""
    found: set[str] = set()
    for row in iter_jbeam_table_rows(wheels_array):
        name = row.get("name")
        if not isinstance(name, str) or name == "name":
            continue
        for key in WHEEL_GROUP_KEYS:
            found.update(jbeam_group_names(row.get(key)))
    return found


def flexbody_row_groups(row: str) -> tuple[str, ...] | None:
    """The groups a flexbody row binds to, or None when it cannot be read.

    The flexbodies header is ``["mesh", "[group]:"]``, so the binding is the
    row's second column. None means "undetermined": callers must not filter.
    """
    try:
        parsed = sjson.decode(row.strip())
    except Exception:
        return None
    if not isinstance(parsed, list) or len(parsed) < 2:
        return None
    value = parsed[1]
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value if str(item))
    if isinstance(value, str):
        return (value,) if value else ()
    return None


def node_groups_for_selection(selected: dict[str, object], part_array) -> set[str]:
    """Node groups a resolved part selection fills.

    ``part_array(part_id, section)`` returns that section's text, letting a
    caller with its own part index (the preview injects generated plate parts)
    use the same rule.
    """
    groups: set[str] = set()
    for instance in selected_part_instances(selected):
        part_id = str(instance.get("part_id") or "")
        nodes = part_array(part_id, "nodes")
        if nodes:
            groups.update(node_group_names(nodes))
        for section in WHEEL_SECTIONS:
            wheels = part_array(part_id, section)
            if wheels:
                groups.update(wheel_group_names(wheels))
    return groups


def flexbody_row_is_bound(row: str, populated_groups: set[str]) -> bool:
    """Whether the engine would give this flexbody any nodes to deform with.

    meshs.lua skips a flexbody whose ``_group_nodes`` is empty, so an unbound
    row contributes no mesh at all. A row whose binding cannot be read is
    treated as bound, so an unreadable row never silently hides geometry.
    """
    groups = flexbody_row_groups(row)
    if not groups:
        return True
    return bool(set(groups) & populated_groups)


def vehicle_node_group_names(context: VehicleContext) -> set[str]:
    """Every node group name declared anywhere in the vehicle.

    Deliberately not per-trim: relocating a part into an empty slot needs to
    recognise the group name on the side that is currently unoccupied, and
    populated_node_groups would by definition not report it.
    """
    groups: set[str] = set()
    for part_id in context.part_body_index:
        nodes = part_named_array_for_context(context, part_id, "nodes")
        if nodes:
            groups.update(node_group_names(nodes))
        for section in WHEEL_SECTIONS:
            wheels = part_named_array_for_context(context, part_id, section)
            if wheels:
                groups.update(wheel_group_names(wheels))
    return groups


def populated_node_groups(context: VehicleContext, config_name: str) -> set[str]:
    """Every node group this trim actually fills with nodes.

    links.lua builds a flexbody's ``_group_nodes`` from the rows of the
    assembled ``nodes`` table whose group cell matches, and meshs.lua skips the
    flexbody entirely when that comes out empty -- so a group missing here means
    the mesh never spawns.
    """
    cached = context.node_groups_cache.get(config_name)
    if cached is not None:
        return cached
    selected = selected_parts_for_config(context, config_name)
    groups = node_groups_for_selection(
        selected,
        lambda part_id, section: part_named_array_for_context(context, part_id, section),
    )
    context.node_groups_cache[config_name] = groups
    return groups


def selected_parts_in_merge_order(selected: dict[str, object]) -> list[str]:
    """Selected part ids in tree (parent-before-child) order, so a later part's
    node redefinition overrides an earlier one -- the order jbeam merges
    sections in. Falls back to a stable sort for older results (or caches) that
    predate parts_order."""
    return mesh_resolution.selected_parts_in_merge_order(selected)


def selected_node_positions_for_config(
    context: VehicleContext,
    config_name: str,
) -> dict[str, tuple[float, float, float]]:
    cached = context.selected_node_positions_cache.get(config_name)
    if cached is not None:
        return cached

    selected = selected_parts_for_config(context, config_name)
    nodes: dict[str, tuple[float, float, float]] = {}
    for instance in selected_part_instances(selected):
        part_id = str(instance.get("part_id") or "")
        node_array = part_named_array_for_context(context, part_id, "nodes")
        if not node_array:
            continue
        inherited_options = part_instance_options(instance)
        variables = part_instance_variable_scope(selected, instance)
        for node_id, position, row in iter_node_rows(node_array, variables):
            nodes[node_id] = pos_after_node_transforms(
                row, position, inherited_options, variables
            )

    context.selected_node_positions_cache[config_name] = nodes
    return nodes



def selected_node_positions_for_parts(
    selected: dict[str, object],
    jbeam_texts: dict[str, str],
    part_body_index: dict[str, tuple[str, str]] | None = None,
) -> dict[str, tuple[float, float, float]]:
    nodes: dict[str, tuple[float, float, float]] = {}
    for instance in selected_part_instances(selected):
        part_id = str(instance.get("part_id") or "")
        found = find_part_body(part_id, jbeam_texts, part_body_index)
        if found is None:
            continue
        part_body, _filename = found
        node_array = transform_helpers.extract_named_array(part_body, "nodes")
        if not node_array:
            continue
        inherited_options = part_instance_options(instance)
        variables = part_instance_variable_scope(selected, instance)
        for node_id, position, row in iter_node_rows(node_array, variables):
            nodes[node_id] = pos_after_node_transforms(
                row, position, inherited_options, variables
            )
    return nodes



def prop_row_mesh(row: str) -> str | None:
    strings = re.findall(r'"((?:[^"\\]|\\.)*)"', row)
    if len(strings) < 2:
        return None
    func, mesh = strings[:2]
    if func == "func" or mesh == "mesh":
        return None
    return mesh


def prop_row_nodes_present(row: str, node_positions: dict[str, tuple[float, float, float]]) -> bool:
    """Whether the row's idRef/idX/idY nodes all exist in node_positions.

    The engine only spawns a prop when its reference nodes exist in the
    assembled vehicle; pass the SELECTED parts' node positions to reproduce
    that (a global all-files node index would also resolve dormant rows,
    e.g. the manual handbrake mount in a sequential-shifter config)."""
    strings = re.findall(r'"((?:[^"\\]|\\.)*)"', row)
    if len(strings) < 5:
        return True
    return all(node_id in node_positions for node_id in strings[2:5])


def selected_prop_mesh_positions(
    context: VehicleContext,
    config_name: str,
    mesh_ids: set[str],
) -> dict[str, list[tuple[float, float, float]]]:
    selected = selected_parts_for_config(context, config_name)
    node_positions = selected_node_positions_for_config(context, config_name)
    positions: dict[str, list[tuple[float, float, float]]] = {}
    for instance in selected_part_instances(selected):
        part_id = str(instance.get("part_id") or "")
        props = part_named_array_for_context(context, part_id, "props")
        if not props:
            continue
        inherited_options = part_instance_options(instance)
        variables = part_instance_variable_scope(selected, instance)
        for raw_row in iter_active_top_level_rows(props):
            row = resolve_jbeam_row_strings(raw_row, variables)
            mesh = prop_row_mesh(row)
            if mesh not in mesh_ids:
                continue
            pivot = context.mesh_pivots.get(mesh)
            position = prop_row_pivot_position(
                row, node_positions, pivot, inherited_options, variables
            )
            if position is not None:
                positions.setdefault(mesh, []).append(position)
    return positions



def mesh_roles_for_config(
    context: VehicleContext,
    config_name: str,
) -> tuple[set[str], set[str], set[str]]:
    cached = context.mesh_roles_cache.get(config_name)
    if cached is not None:
        return cached

    flexbody_meshes: set[str] = set()
    prop_meshes: set[str] = set()
    all_meshes: set[str] = set()
    selected = selected_parts_for_config(context, config_name)
    # A flexbody bound only to groups this trim leaves empty is dropped by the
    # engine, so the mesh never spawns. Stock relies on it: the BX race trims
    # delete the door panel that owns bx_doorpanel_L2/R2, which would otherwise
    # leave its window controls floating, and a pickup with no windshield would
    # keep its interior mirror.
    populated = populated_node_groups(context, config_name)
    for instance in selected_part_instances(selected):
        part_id = str(instance.get("part_id") or "")
        variables = part_instance_variable_scope(selected, instance)
        flexbodies = part_named_array_for_context(context, part_id, "flexbodies")
        if flexbodies:
            for raw_row in iter_active_top_level_rows(flexbodies):
                row = resolve_jbeam_row_strings(raw_row, variables)
                mesh = flexbody_row_mesh(row)
                if not mesh:
                    continue
                if not flexbody_row_is_bound(row, populated):
                    continue
                flexbody_meshes.add(mesh)
                all_meshes.add(mesh)
        props = part_named_array_for_context(context, part_id, "props")
        if props:
            for raw_row in iter_active_top_level_rows(props):
                mesh = prop_row_mesh(resolve_jbeam_row_strings(raw_row, variables))
                if mesh:
                    prop_meshes.add(mesh)
                    all_meshes.add(mesh)

    roles = (flexbody_meshes, prop_meshes, all_meshes)
    context.mesh_roles_cache[config_name] = roles
    return roles



def selected_mesh_roles(
    context: VehicleContext,
    selected_configs: list[str],
) -> tuple[set[str], set[str], set[str]]:
    flexbody_meshes: set[str] = set()
    prop_meshes: set[str] = set()
    all_meshes: set[str] = set()
    for config_name in selected_configs:
        config_flex, config_props, config_all = mesh_roles_for_config(context, config_name)
        flexbody_meshes.update(config_flex)
        prop_meshes.update(config_props)
        all_meshes.update(config_all)
    return flexbody_meshes, prop_meshes, all_meshes


def active_part_modes(conversion: dict[str, object]) -> dict[str, str]:
    parts = conversion.get("parts", {})
    modes: dict[str, str] = {}
    if not isinstance(parts, dict):
        return modes
    for object_id, settings in parts.items():
        if not isinstance(settings, dict):
            continue
        mode = str(settings.get("mode", MODE_SKIP))
        if mode in MODE_CHOICES and mode != MODE_SKIP:
            modes[str(object_id)] = mode
    return modes


def texture_flip_mesh_ids(
    context: VehicleContext,
    object_modes: dict[str, str],
) -> set[str]:
    """Mirrored meshes carrying display-screen UV islands, whose textures must
    keep their left/right reading after the geometric mirror.

    Derived entirely from the vehicle's own data: navigator controllers,
    glowMaps and emissive screen-like materials. A screen only reads backwards
    once its mesh is actually reflected, so translate mode keeps the texture as
    authored."""
    display_scope = display_texture_flip_scope(context)
    if not display_scope:
        return set()
    return {
        object_id
        for object_id, mode in object_modes.items()
        if mode in {MODE_MIRROR, MODE_MIRROR_STRUCTURAL, MODE_REPLACE_SOURCE} and object_id in display_scope
    }


def structural_mirror_source_for_settings(
    context: VehicleContext,
    object_id: str,
    settings: object,
) -> str | None:
    if not isinstance(settings, dict):
        return None
    source_id = str(settings.get("mirrorSource") or "")
    source_obj = context.objects.get(source_id)
    if source_id and source_id != object_id and source_obj is not None and source_obj.dae_path:
        return source_id
    return None


def structural_mirror_sources(
    context: VehicleContext,
    conversion: dict[str, object],
    object_modes: dict[str, str] | None = None,
) -> dict[str, str]:
    parts = conversion.get("parts", {})
    if not isinstance(parts, dict):
        return {}
    wanted = object_modes or active_part_modes(conversion)
    sources: dict[str, str] = {}
    for object_id, mode in wanted.items():
        if mode not in {MODE_MIRROR_STRUCTURAL, MODE_REPLACE_SOURCE}:
            continue
        settings = parts.get(object_id, {})
        source_id = structural_mirror_source_for_settings(context, object_id, settings)
        if source_id is not None:
            sources[object_id] = source_id
    return sources


def fallback_structural_part_modes(
    context: VehicleContext,
    conversion: dict[str, object],
    object_modes: dict[str, str] | None = None,
    *,
    selected_configs: Iterable[str] = (),
) -> dict[str, str]:
    modes = dict(object_modes or active_part_modes(conversion))
    if not modes:
        return modes
    parts = conversion.get("parts", {})
    if not isinstance(parts, dict):
        return modes
    for object_id, mode in list(modes.items()):
        if mode not in {MODE_MIRROR_STRUCTURAL, MODE_REPLACE_SOURCE}:
            continue
        source_id = structural_mirror_source_for_settings(context, object_id, parts.get(object_id, {}))
        if source_id is None:
            modes[object_id] = MODE_MIRROR
    return modes


def selected_steering_refs(conversion: dict[str, object]) -> list[str]:
    parts = conversion.get("parts", {})
    if not isinstance(parts, dict):
        return []
    return [
        str(object_id)
        for object_id, settings in parts.items()
        if isinstance(settings, dict) and settings.get("steeringRef")
    ]


def auto_delta_source_refs(context: VehicleContext, conversion: dict[str, object]) -> list[str]:
    """Steering-ref parts that actually contribute to the auto delta (indexed
    objects with a usable off-center X). Empty means the auto delta falls back
    to its default of 0."""
    return [
        object_id
        for object_id in selected_steering_refs(conversion)
        if object_id in context.objects and abs(context.objects[object_id].x) > 0.05
    ]


STEERING_PROP_STR_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')

__all__ = ['find_part_body', 'part_body_for_context', 'part_named_array_for_context', 'vehicle_namespace_main_part', 'resolve_selected_parts', 'selected_parts_for_config', 'find_hand_authored_opposite_group', 'resolve_slot_pair_plan', 'slot_pair_plans_for_variants', 'slot_pair_plan_relocations', 'authored_group_source_parts', 'authored_group_meshes', 'part_variable_scope', '_NODE_ROW_RE', 'selected_part_instances', 'part_instance_options', 'part_instance_variable_scope', 'iter_node_rows', 'jbeam_group_names', 'iter_jbeam_table_rows', 'node_group_names', 'vehicle_node_group_names', 'wheel_group_names', 'flexbody_row_groups', 'populated_node_groups', 'node_groups_for_selection', 'flexbody_row_is_bound', 'selected_parts_in_merge_order', 'selected_node_positions_for_config', 'selected_node_positions_for_parts', 'prop_row_mesh', 'prop_row_nodes_present', 'selected_prop_mesh_positions', 'mesh_roles_for_config', 'selected_mesh_roles', 'active_part_modes', 'texture_flip_mesh_ids', 'structural_mirror_source_for_settings', 'structural_mirror_sources', 'fallback_structural_part_modes', 'selected_steering_refs', 'auto_delta_source_refs', 'STEERING_PROP_STR_RE']
