"""Implementation slice extracted from ``beamng_hand_drive_core``.

Original source lines 4069-4255. Import the public orchestration module
``beamng_hand_drive_core`` rather than this implementation module directly.
"""

from __future__ import annotations

from beamxp import transform_helpers
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
    MODE_MIRROR_POSITION,
    MODE_MIRROR_STRUCTURAL,
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

def steering_column_axis_offsets(context: VehicleContext) -> list[float]:
    """|x| of each steering column's rotation centre -- the idRef node of a
    ``func:"steering"`` prop row, the point the wheel animation spins around.

    A delta fallback only. It is NOT the primary signal because the rotation
    centre is only as trustworthy as the mod authored it: the sheik_yaris
    places its wheel on the right (x=-0.33) but leaves int_strw on the left
    (x=+0.37), so trusting the node there would mirror the wheel the wrong
    way. Where the wheel's own geometry gives a usable offset that wins; this
    covers the case where it does not, e.g. a wheel mesh authored at the
    origin and placed entirely by its prop row."""
    offsets: list[float] = []
    seen: set[str] = set()
    for body, _filename in context.part_body_index.values():
        props = transform_helpers.extract_named_array(body, "props")
        if not props or '"steering"' not in props:
            continue
        for row in iter_active_top_level_rows(props):
            strings = STEERING_PROP_STR_RE.findall(row)
            if len(strings) < 5 or strings[0] != "steering":
                continue
            idref = strings[2]
            if idref in seen:
                continue
            seen.add(idref)
            node = context.node_positions.get(idref)
            if node is not None and abs(node[0]) > 0.05:
                offsets.append(abs(node[0]))
    return offsets


def auto_delta_magnitude(context: VehicleContext, conversion: dict[str, object]) -> float:
    offsets = [
        abs(context.objects[object_id].x)
        for object_id in auto_delta_source_refs(context, conversion)
    ]
    offset = median_value(offsets)
    if offset is None:
        # No steering wheel geometry far enough off-centre to measure; fall
        # back to the steering column's rotation centre. Never overrides usable
        # wheel geometry, so validated conversions are unaffected.
        offset = median_value(steering_column_axis_offsets(context))
    if offset is None:
        return 0.0
    return offset * 2.0


def delta_magnitude(context: VehicleContext, conversion: dict[str, object]) -> float:
    delta = conversion.get("delta", {})
    if isinstance(delta, dict) and delta.get("manual"):
        try:
            return abs(float(delta.get("magnitude") or 0.0))
        except (TypeError, ValueError):
            return 0.0
    return auto_delta_magnitude(context, conversion)


# Modes whose Move X is a nudge laid on top of the transform rather than the
# transform itself. A Mirror already knows where it is going -- the reflection
# of where it started -- so the column corrects that landing point, which is
# what an off-centre mod needs; with none typed the nudge is zero and the
# reflection is exactly what it always was. Swap Mesh and Replace Source are
# deliberately absent: those adopt another mesh whole, pivot included.
NUDGE_MODES = frozenset({MODE_MIRROR, MODE_MIRROR_POSITION})


def part_translate_magnitude(
    context: VehicleContext,
    conversion: dict[str, object],
    object_id: str,
    mode: str = MODE_TRANSLATE,
) -> float:
    """This part's Move X, measured along the conversion direction.

    Signed, not absolute: positive is whichever way this trim's conversion
    already runs (toward the target hand's side), so a negative offset walks
    the part back the other way -- useful when a mesh sits off-centre and the
    shared delta overshoots it. ``signed_delta_for_target`` turns this into the
    world-space X delta, and it is the target hand there that decides which way
    "positive" points, so the same saved number does the right thing on an
    LHD->RHD trim and on an RHD->LHD one.

    What an unset column falls back to is the whole difference between the two
    kinds of mode. A Move is nothing but its delta, so it inherits the shared
    conversion delta. A Mirror carries the column as a correction on top of a
    reflection that already stands on its own, so it falls back to zero and
    leaves that reflection untouched.

    A mode that takes no Move X at all answers 0.0 outright. Swap Mesh and
    Replace Source adopt another mesh whole, pivot included, so there is
    nothing here to correct -- and a value left over from a spell on Move must
    not leak into their placement now that the callers ask about every mode
    rather than only the moving ones.

    The global delta stays unsigned; only the per-part override carries a sign.
    """
    if mode != MODE_TRANSLATE and mode not in NUDGE_MODES:
        return 0.0
    parts = conversion.get("parts", {})
    settings = parts.get(object_id, {}) if isinstance(parts, dict) else {}
    if isinstance(settings, dict):
        raw = settings.get("translateOffset")
        if raw not in (None, ""):
            try:
                return float(raw)
            except (TypeError, ValueError):
                return 0.0
    return 0.0 if mode in NUDGE_MODES else delta_magnitude(context, conversion)


def part_translate_magnitudes(
    context: VehicleContext,
    conversion: dict[str, object],
    object_modes: dict[str, str],
) -> dict[str, float]:
    """Each convertible mesh's Move X, keyed by mesh.

    Covers the nudge modes as well as Move, so callers hold one map of "how far
    this mesh slides along x". Every reader is already mode-guarded, and the
    fallbacks differ per mode (see ``part_translate_magnitude``), so a Mirror
    that was never given a Move X contributes a harmless 0.0.
    """
    return {
        object_id: part_translate_magnitude(context, conversion, object_id, mode)
        for object_id, mode in object_modes.items()
        if mode == MODE_TRANSLATE or mode in NUDGE_MODES
    }


def detect_hand_for_variant(
    context: VehicleContext,
    conversion: dict[str, object],
    config_name: str,
) -> str:
    used_meshes = used_meshes_for_config(context, config_name)
    explicit_used_refs = [
        object_id
        for object_id in selected_steering_refs(conversion)
        if object_id in context.objects and object_id in used_meshes
    ]
    stock_hand = hand_from_stock_steering_for_variant(
        context,
        config_name,
        explicit_used_refs,
        used_meshes,
    )
    if stock_hand != HAND_UNKNOWN:
        return stock_hand

    stock_hand = hand_from_stock_steering_for_variant(
        context,
        config_name,
        likely_stock_steering_ref_ids(context, used_meshes),
        used_meshes,
    )
    if stock_hand != HAND_UNKNOWN:
        return stock_hand

    variant = context.variants[config_name]
    metadata_hand = hand_from_text(f"{variant.name} {variant.display_name}")
    if metadata_hand != HAND_UNKNOWN:
        return metadata_hand

    explicit_refs = [
        object_id
        for object_id in selected_steering_refs(conversion)
        if object_id in context.objects
    ]
    global_hand = hand_from_steering_positions(context, explicit_refs)
    if global_hand != HAND_UNKNOWN:
        return global_hand

    global_hand = hand_from_steering_positions(context, likely_steering_ref_ids(context))
    if global_hand != HAND_UNKNOWN:
        return global_hand

    steering_ids = [
        object_id
        for object_id in explicit_refs
        if object_id in context.objects and object_id in used_meshes
    ]
    if not steering_ids:
        steering_ids = likely_steering_ref_ids(context, used_meshes)
    return hand_from_steering_positions(context, steering_ids, used_meshes)


def detect_hands_for_variants(
    context: VehicleContext,
    conversion: dict[str, object],
) -> dict[str, str]:
    """Return all stock-hand detections, filling only cache misses."""
    hands = load_cached_variant_hands(context, conversion) or {}
    changed = False
    for config_name in sorted(context.variants):
        if config_name in hands:
            continue
        hands[config_name] = detect_hand_for_variant(context, conversion, config_name)
        changed = True
    if changed:
        save_cached_variant_hands(context, conversion, hands)
    return hands


def cached_hand_for_variant(
    context: VehicleContext,
    conversion: dict[str, object],
    config_name: str,
) -> str:
    hands = load_cached_variant_hands(context, conversion) or {}
    hand = hands.get(config_name)
    if hand is not None:
        return hand
    hand = detect_hand_for_variant(context, conversion, config_name)
    hands[config_name] = hand
    save_cached_variant_hands(context, conversion, hands)
    return hand


def effective_source_hand(
    context: VehicleContext,
    conversion: dict[str, object],
    config_name: str,
) -> str:
    variant_settings = dict(conversion.get("variants", {}).get(config_name, {}))
    override = str(variant_settings.get("sourceHandOverride", HAND_AUTO))
    if override in {HAND_LHD, HAND_RHD, HAND_UNKNOWN}:
        return override
    return cached_hand_for_variant(context, conversion, config_name)

__all__ = ['NUDGE_MODES', 'steering_column_axis_offsets', 'auto_delta_magnitude', 'delta_magnitude', 'part_translate_magnitude', 'part_translate_magnitudes', 'detect_hand_for_variant', 'detect_hands_for_variants', 'cached_hand_for_variant', 'effective_source_hand']
