"""Which opposite-side pairs convert by swapping sides, and with what.

Finding a mesh's twin is easy -- flip the side token in its name and check
the two really do straddle the centreline. Almost every vehicle has dozens
that do: headlights, wheel arches, door glass, quarter panels. Those are true
mirror pairs and swapping them changes nothing.

The ones that matter are the pairs authored *handed*: the front door card,
whose window switches and armrest are placed for the driver, and the wing
mirror, whose casing and glass are aimed from the driver's eye. Both are
identified from the slot tree rather than from names, because the slotType is
what BeamNG selects the part by and the names are spelt a dozen ways
(mirrorstalk, mirror_signal_glass, doorpanel, door_panel, door_L).

The gate, measured over nine hand-configured vehicles: 37 of 37 hand-made
swaps are named by it, out of 451 straddling twins in the same mesh sets.
"""

from __future__ import annotations

import re

from .recommendation_common import (
    _DOOR_CARD_NAME_RE,
    DOOR_CARD_SLOT_RE,
    DOOR_EXCLUDE_SLOT_RE,
    DOOR_SLOT_RE,
    FRONT_SIDE_SLOT_RE,
    MIRROR_SLOT_RE,
    PAIR_MIN_OFFSET,
    REAR_SIDE_SLOT_RE,
    mesh_center,
    slot_family,
)
from .shared import core

_HAND_TOKEN_RE = re.compile(r"[lr]hd")

# The side tokens a mesh name uses. Kept local to the pairing because the
# order matters here: a longer token is tried before the single letter it
# contains, so "_fl" is never read as "_l".
_SIDE_TOKEN_PAIRS = (
    ("_fl", "_fr"),
    ("_fr", "_fl"),
    ("_rl", "_rr"),
    ("_rr", "_rl"),
    ("_frontleft", "_frontright"),
    ("_frontright", "_frontleft"),
    ("_rearleft", "_rearright"),
    ("_rearright", "_rearleft"),
    ("_left", "_right"),
    ("_right", "_left"),
    ("_driver", "_passenger"),
    ("_passenger", "_driver"),
    ("_l", "_r"),
    ("_r", "_l"),
    ("-l", "-r"),
    ("-r", "-l"),
    (".l", ".r"),
    (".r", ".l"),
)


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
    for old, new in _SIDE_TOKEN_PAIRS:
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


def _slot_is_wing_mirror(family: str) -> bool:
    return bool(MIRROR_SLOT_RE.search(family)) and bool(FRONT_SIDE_SLOT_RE.search(family))


def _slot_is_door_card(family: str) -> bool:
    if REAR_SIDE_SLOT_RE.search(family) or DOOR_EXCLUDE_SLOT_RE.search(family):
        return False
    return bool(DOOR_CARD_SLOT_RE.search(family)) and bool(FRONT_SIDE_SLOT_RE.search(family))


def _slot_is_front_door(family: str) -> bool:
    if REAR_SIDE_SLOT_RE.search(family) or DOOR_EXCLUDE_SLOT_RE.search(family):
        return False
    return bool(DOOR_SLOT_RE.search(family)) and bool(FRONT_SIDE_SLOT_RE.search(family))


def handed_mesh_ids(
    context: core.VehicleContext,
    evidence,
    scope: set[str],
) -> frozenset[str]:
    """The meshes that hold handed cabin furniture.

    Wing mirrors first: any mesh a ``*_mirror_L/R`` slot's part declares, plus
    any mesh the ``mirrors`` section binds. Both readings are needed -- a mod
    can hang the reflector row on the door part -- and together they cover the
    casing, the stalk, the repeater and the glass without knowing what any of
    them are called.

    Then the front door card. Where the vehicle authors a dedicated door-card
    slot (``etkc_doorpanel_L``, ``scintilla_doorpanel_L``) that slot *is* the
    card and the door slot beside it holds the skin. Where it does not
    (``r8_door_L_hardtop``, ``lc500_door_L``) the door slot's own meshes are
    the card, because there is nothing else for them to be -- unless one of
    them is *named* for the card, as the Volvo's ``v60_andronisk_doorpanel_FL``
    is, in which case the skin beside it stays out. Deciding this per vehicle
    rather than per mesh keeps ``etkc_door_L`` -- a skin -- out while letting
    ``r8_door_L`` -- a card -- in.

    2026-09-07: this reads each mesh's own slots and never widens to the other
    meshes its part declares. Widening looks harmless and is not: the R8's
    light parts and its wing-mirror parts share one ``SPOTLIGHT`` cone mesh,
    so a mesh->part->mesh hop made every headlight and tail light in the
    vehicle a handed candidate.
    """
    vehicle_id = context.vehicle_id
    mirror_meshes: set[str] = set()
    card_slot_meshes: set[str] = set()
    door_slot_meshes: set[str] = set()

    for mesh in scope:
        families = [
            slot_family(slot_type, vehicle_id)
            for slot_type in evidence.slot_types.get(mesh, frozenset())
        ]
        if mesh in evidence.mirror_meshes or any(
            _slot_is_wing_mirror(family) for family in families
        ):
            mirror_meshes.add(mesh)
        if any(_slot_is_door_card(family) for family in families):
            card_slot_meshes.add(mesh)
        elif any(_slot_is_front_door(family) for family in families):
            door_slot_meshes.add(mesh)

    if card_slot_meshes:
        # A vehicle with a real door-card slot has already told us which mesh
        # is the card; taking the door slot as well would re-admit the skin.
        doors = card_slot_meshes
    else:
        named = {mesh for mesh in door_slot_meshes if _DOOR_CARD_NAME_RE.search(mesh.lower())}
        doors = named or door_slot_meshes
    return frozenset(mirror_meshes | doors)


def handed_pair_family(
    context: core.VehicleContext,
    evidence,
    object_id: str,
    twin_id: str,
) -> tuple[str, str]:
    """(family, reason) for a handed pair: what it is and how we know.

    The family is the Equivalent Parts kind the pair belongs to, and it comes
    from the slot tree rather than the mesh names. That matters for the door
    card: the R8 and the Lexus call theirs "r8_door_L" and "lc500_door_L",
    which no name table would read as a card, and the Texture Fix pass needs
    to know it is one.
    """
    vehicle_id = context.vehicle_id
    families = [
        slot_family(slot_type, vehicle_id)
        for mesh in (object_id, twin_id)
        for slot_type in evidence.slot_types.get(mesh, frozenset())
    ]
    if object_id in evidence.mirror_meshes or twin_id in evidence.mirror_meshes:
        return "mirror", "wing mirror: bound by a mirrors row"
    if any(_slot_is_wing_mirror(family) for family in families):
        return "mirror", "wing mirror assembly slot"
    if any(_slot_is_door_card(family) for family in families):
        return "door", "front door card slot"
    if any(_slot_is_front_door(family) for family in families):
        return "door", "front door slot; no separate card"
    return "part", "handed cabin part with an opposite-side twin"
