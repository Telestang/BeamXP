"""Matching a mesh to its opposite-side twin by name, checked by placement."""

from __future__ import annotations

import re

from .recommendation_common import (
    PAIR_MIN_OFFSET,
    SIDE_TOKEN_PAIRS,
    mesh_center,
)
from .shared import core

_HAND_TOKEN_RE = re.compile(r"[lr]hd")


def _name_pair_candidate(object_id: str, candidates: list[str] | set[str]) -> str | None:
    """The candidate whose name is this one with its side token flipped.

    "lhd"/"rhd" are handedness tokens, not side tokens: the "_l" inside
    "_lhd" must not pair bx_mirror_int_lhd with bx_mirror_int_rhd, which are
    mutually exclusive builds of the same part rather than two halves of one.
    Side tokens elsewhere in such a name (bx_mirror_L_rhd) still pair.
    """
    lowered = object_id.lower()
    lower_to_id = {str(candidate).lower(): candidate for candidate in candidates}
    hand_spans = [match.span() for match in _HAND_TOKEN_RE.finditer(lowered)]
    for old, new in SIDE_TOKEN_PAIRS:
        start = 0
        while True:
            index = lowered.find(old, start)
            if index < 0:
                break
            if any(index < end and begin < index + len(old) for begin, end in hand_spans):
                start = index + 1
                continue
            candidate = lowered[:index] + new + lowered[index + len(old):]
            if candidate in lower_to_id:
                return lower_to_id[candidate]
            break
    return None


def _straddles_centreline(
    context: core.VehicleContext,
    object_id: str,
    twin_id: str,
    center_x: float,
) -> bool:
    """Whether two named twins really do sit on opposite sides.

    A mesh with no cached geometry gets the benefit of the doubt: the name
    is then the only evidence there is, and refusing to pair would drop the
    part from the recommendations entirely.
    """
    center_a = mesh_center(context, object_id)
    center_b = mesh_center(context, twin_id)
    if center_a is None or center_b is None:
        return True
    offset_a = center_a[0] - center_x
    offset_b = center_b[0] - center_x
    if min(abs(offset_a), abs(offset_b)) < PAIR_MIN_OFFSET:
        return False
    return offset_a * offset_b < 0.0


def resolve_side_twin(
    context: core.VehicleContext,
    object_id: str,
    candidates: list[str] | set[str],
    center_x: float,
) -> str | None:
    """This mesh's opposite-side twin, or None when nothing answers for it."""
    twin_id = _name_pair_candidate(object_id, candidates)
    if twin_id is None or twin_id == object_id:
        return None
    if not _straddles_centreline(context, object_id, twin_id, center_x):
        return None
    return twin_id
