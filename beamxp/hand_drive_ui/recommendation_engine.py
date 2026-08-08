from __future__ import annotations

from .recommendation_common import (
    HANDED_PATTERNS,
    LOW_CONFIDENCE_PATTERNS,
    MIRROR_PATTERNS,
    TRANSLATE_EXCLUDE_PATTERNS,
    TRANSLATE_PATTERNS,
    UNPAIRED_MIRROR_EXCLUDE_PATTERNS,
    _is_seat_mesh_id,
    _side_pair_kind_for_mesh,
    mesh_center,
    recommendation_matches,
    recommendation_text,
)
from .recommendation_pairing import resolve_side_twin
from .shared import core


def _driver_side(context: core.VehicleContext, steering_ids: set[str], center_x: float) -> int:
    """+1 when the wheel sits left of the centreline, -1 when it sits right.

    Only used to decide which member of a pair the modal names first, so a
    vehicle with no locatable wheel simply keeps the left-hand convention.
    """
    offsets = []
    for object_id in steering_ids:
        center = mesh_center(context, object_id)
        if center is not None:
            offsets.append(center[0] - center_x)
    offsets = [offset for offset in offsets if abs(offset) > 0.01]
    if not offsets:
        return 1
    return 1 if sum(offsets) >= 0.0 else -1


def build_mode_recommendations(
    context: core.VehicleContext,
    object_ids: list[str],
) -> list[dict[str, str]]:
    """Recommend a conversion mode for each mesh from its name and placement.

    Handed families go first, because a door card or a wing mirror that has
    an opposite-side twin converts by swapping sides rather than by
    transforming anything: whichever rule would otherwise claim it, the swap
    wins. Seats are handed too, but their slot accepts either side's part, so
    they convert as an Equivalent Parts row and their meshes stay untouched.
    What is left is judged on the job its name describes -- driver controls
    and instruments move with the driver, asymmetric cabin furniture mirrors,
    and anything the tables do not recognise is left alone rather than
    guessed at.
    """
    available = sorted(
        object_id for object_id in set(object_ids) if object_id in context.objects
    )
    if not available:
        return []

    text_by_id = {
        object_id: recommendation_text(context, object_id) for object_id in available
    }
    steering_ids = {
        object_id
        for object_id in available
        if core.steering_ref_score(object_id, context.objects[object_id]) >= 15
    }
    center_x = core.estimated_vehicle_center_x(context, set(available), steering_ids)
    side = _driver_side(context, steering_ids, center_x)

    recommendations: list[dict[str, str]] = []
    claimed: set[str] = set()

    def confidence_for(*texts: str) -> str:
        return (
            "low"
            if any(recommendation_matches(text, LOW_CONFIDENCE_PATTERNS) for text in texts)
            else "med"
        )

    for object_id in available:
        if object_id in claimed:
            continue
        text = text_by_id[object_id]
        if not recommendation_matches(text, HANDED_PATTERNS):
            continue
        twin_id = resolve_side_twin(
            context,
            object_id,
            [candidate for candidate in available if candidate not in claimed],
            center_x,
        )
        if twin_id is not None:
            claimed.add(object_id)
            claimed.add(twin_id)
            # Name the driver-side member first so the modal reads naturally.
            first, second = object_id, twin_id
            center_a = mesh_center(context, object_id)
            center_b = mesh_center(context, twin_id)
            if (
                center_a is not None
                and center_b is not None
                and side * (center_b[0] - center_x) > side * (center_a[0] - center_x)
            ):
                first, second = twin_id, object_id
            seats = _is_seat_mesh_id(first) or _is_seat_mesh_id(second)
            recommendations.append({
                "kind": "equivalent" if seats else "pair",
                "object_id": first,
                "source_id": second,
                "mode": core.MODE_SKIP if seats else core.MODE_MIRROR_STRUCTURAL,
                "reason": (
                    "left/right name pair; seats swap as equivalent parts"
                    if seats
                    else "left/right name pair"
                ),
                "confidence": confidence_for(text_by_id[first], text_by_id[second]),
                # Only the seat converts by slot occupancy, so only the seat
                # earns an Equivalent Parts row. A door card or a wing mirror
                # is slot-locked -- its part fits nothing but its own side --
                # so the cross-swap IS the conversion, and a second row saying
                # the same thing in the other mechanism can only fight it.
                "equivalent": seats,
                "pair_kind": _side_pair_kind_for_mesh(first, second) if seats else "",
            })
            continue
        if recommendation_matches(text, UNPAIRED_MIRROR_EXCLUDE_PATTERNS):
            continue  # a cabin-spanning bench has no side to swap with
        if recommendation_matches(text, TRANSLATE_PATTERNS):
            continue  # instrument named for a handed family: judged below
        # One-sided seat or mirror hardware -- a racing seat base, a single
        # wing mirror -- has no counterpart to swap with, so it mirrors.
        claimed.add(object_id)
        recommendations.append({
            "kind": "single",
            "object_id": object_id,
            "source_id": "",
            "mode": core.MODE_MIRROR,
            "reason": "one-sided seat/mirror part; no opposite-side counterpart",
            "confidence": confidence_for(text),
            "equivalent": False,
            "pair_kind": "",
        })

    for object_id in available:
        if object_id in claimed:
            continue
        text = text_by_id[object_id]
        if object_id in steering_ids:
            mode, reason = core.MODE_TRANSLATE, "steering wheel"
        elif recommendation_matches(text, TRANSLATE_PATTERNS) and not recommendation_matches(
            text,
            TRANSLATE_EXCLUDE_PATTERNS,
        ):
            mode, reason = core.MODE_TRANSLATE, "driver control or instrument name"
        elif recommendation_matches(text, MIRROR_PATTERNS):
            mode, reason = core.MODE_MIRROR, "asymmetric interior name"
        else:
            continue
        recommendations.append({
            "kind": "single",
            "object_id": object_id,
            "source_id": "",
            "mode": mode,
            "reason": reason,
            "confidence": confidence_for(text),
            "equivalent": False,
            "pair_kind": "",
        })

    mode_order = {
        core.MODE_TRANSLATE: 0,
        core.MODE_MIRROR: 1,
        core.MODE_MIRROR_STRUCTURAL: 2,
        core.MODE_SKIP: 3,
    }
    recommendations.sort(
        key=lambda item: (
            mode_order.get(item["mode"], 99),
            item["object_id"].lower(),
            item.get("source_id", "").lower(),
        )
    )
    return recommendations
