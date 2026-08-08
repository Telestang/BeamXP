"""Name and placement rules behind Recommend Transforms.

The recommender used to reason spatially: it rebuilt each trim's point
clouds, cast rays from the driver's eye to decide what an occupant could
actually see, and scored self-symmetry vertex by vertex. That cost seconds
per vehicle and still disagreed with the hand-verified baselines often
enough to need reviewing anyway.

Mesh names carry the same intent far more cheaply, because BeamNG's authors
encode a part's job in its name -- and the handed families we care about
(door cards, wing mirrors, seats) are named for their side. Placement is
used only to confirm what a name claims: an L/R pair has to straddle the
centreline. Centres come from the preview bounds already cached for the 3D
view, so a full pass over a vehicle is a dictionary walk.
"""

from __future__ import annotations

import re

from .shared import core

# A named twin must sit at least this far off the centreline before the
# pairing is believed: a centred mesh has no opposite side to swap with.
PAIR_MIN_OFFSET = 0.05

# The side tokens a mesh name uses to say which side it is on. "lhd"/"rhd"
# are deliberately absent -- they mark mutually exclusive builds of the whole
# cabin, not two halves of one, and are masked out below.
SIDE_TOKEN_PAIRS = (
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

TRANSLATE_PATTERNS = (
    r"digidash|digital_?dash|cluster|instrument",
    r"gauge|gauges|needle|speedo|tacho|tachometer",
    r"(?:gas|brake|clutch|throttle).*pedal|pedal.*(?:gas|brake|clutch|throttle)",
    r"pedalbox|pedal_box|padalbox",
    r"steer(?:ing)?_?wheel|steerwheel|(?:^|_)steer_[0-9]",
    r"paddle|signal_?stalk|wiper_?stalk",
    r"shift_?light",
    # Only the column TOP moves with the wheel (ignition and stalk details
    # face the driver); column bodies and racks stay in the mirror pool.
    r"steering_?column\w*top",
)

TRANSLATE_EXCLUDE_PATTERNS = (
    # A pedalbox's own footplate moves with the pedals; standalone
    # footplates and stands are cabin furniture and stay in the mirror pool.
    r"(?<!box_)footplate|(?:^|_)stand(?:_|$)|stand_plate",
)

# Headliners and sunvisors deliberately do NOT appear here: they span the
# cabin symmetrically on essentially every vehicle, so mirroring them only
# generates mesh copies with no visual change.
MIRROR_PATTERNS = (
    r"dash|dashboard|console",
    r"parking_?brake|park_?brake|pbrake|hand_?brake|(?:^|_)hb_",
    r"shifter|shift_?knob|(?:^|_)grp_shift",
    r"radio|laptop|interior",
    r"steering_?column|(?:^|_)column(?:_|$)",
    r"intmirror|grp_mirror|hazard|dash_key|(?:^|_)key(?:_|$)",
    r"extinguisher|footplate|(?:^|_)stand(?:_|$)|stand_plate|cable",
    # A display panel mirrors like any other fascia part. The texture flip
    # that keeps it readable is derived at build time from the
    # beamNavigator controller, not recommended here. "windscreen" must not
    # match; "gauges_screen" is claimed by the translate patterns first.
    r"(?<!wind)screen",
)

# The families that convert by swapping sides rather than by transforming a
# mesh in place. Vanilla bx ships all three as authored LHD/RHD pairs.
#
# The mirror rule is deliberately just "anything mirror related", because a
# wing mirror is never one mesh: the casing comes with a stalk, a turn-signal
# repeater, and on some vehicles a separate lens over it, and every vehicle
# spells those differently -- mirrorstalk, mirrorsignal, mirror_signal,
# mirrorsignalglass, sidemirror_mount. Enumerating the spellings only ever
# misses one, and the pairing does the real work anyway: a name with no
# opposite-side twin falls through to a plain Mirror below, which is the right
# answer for the centred interior mirrors this also catches.
HANDED_PATTERNS = (
    r"door_?panel|door_?card",
    r"mirror",
    r"(?:^|_)(?:race_?)?seats?(?:_|$)|racing_?seat",
)

# Plural "seats" meshes are cabin-spanning benches (etk800_seats_R = rear
# bench, where R means rear, not right): symmetric, nothing to mirror.
UNPAIRED_MIRROR_EXCLUDE_PATTERNS = (
    r"seats(?:_|$)",
)

# Names whose verdict rides on a single token the mesh resolution cannot
# confirm, so the row is offered as a suggestion rather than a finding.
LOW_CONFIDENCE_PATTERNS = (
    r"steering_?column|(?:^|_)column(?:_|$)",
)

# Each handed family converts by a different mechanism: a seat rides in a
# slot the opposite part also fits, while a door card is locked to its side
# and can only cross-swap its mesh.
_SEAT_NAME_RE = re.compile(r"\b(seat|bucket|racingseat)\b")
_SEAT_NAME_TOKENS = ("seat_", "_seat", "racingseat", "seatbase", "seat_base")
_DOOR_CARD_NAME_RE = re.compile(r"door[^a-z0-9]{0,2}(panel|card)")


def _is_seat_mesh_id(object_id: str) -> bool:
    text = object_id.lower()
    if _SEAT_NAME_RE.search(text):
        return True
    return any(token in text for token in _SEAT_NAME_TOKENS)


def _is_door_card_mesh_id(object_id: str) -> bool:
    """A door card is named for its lining panel, not the door skin."""
    return bool(_DOOR_CARD_NAME_RE.search(object_id.lower()))


def _side_pair_kind_for_mesh(*names: str) -> str:
    """The Equivalent Parts family an equivalence row belongs to."""
    text = " ".join(name.lower() for name in names if name)
    if _is_seat_mesh_id(text):
        return "seat"
    if _is_door_card_mesh_id(text):
        return "door"
    if "mirror" in text:
        return "mirror"
    return "part"


def recommendation_text(context: core.VehicleContext, object_id: str) -> str:
    """The name evidence for one mesh: its id plus its DAE node name."""
    values = [object_id]
    obj = context.objects.get(object_id)
    if obj is not None and obj.name and obj.name != object_id:
        values.append(obj.name)
    return re.sub(r"[^a-z0-9]+", "_", " ".join(values).lower())


def recommendation_matches(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text) is not None for pattern in patterns)


def mesh_center(context: core.VehicleContext, object_id: str) -> tuple[float, float, float] | None:
    """Where a mesh sits, from the bounds already cached for the preview.

    A DaeObject's own x/y/z is the node transform, which is the origin for
    well over half of a real vehicle's meshes (the placement lives in the
    flexbody rows instead). The preview centre is the geometry's, so it is
    the only cheap position worth gating a pairing on.
    """
    preview = context.preview_by_id.get(object_id)
    if preview is not None:
        center = preview.get("center")
        if center is not None and len(center) == 3:
            return (float(center[0]), float(center[1]), float(center[2]))
    obj = context.objects.get(object_id)
    if obj is None:
        return None
    if obj.x == 0.0 and obj.y == 0.0 and obj.z == 0.0:
        # The exact origin means no placement was recorded on the node, not
        # that the mesh straddles the centreline. Saying "unknown" keeps the
        # name pairing; saying "centred" would silently veto it.
        return None
    return (float(obj.x), float(obj.y), float(obj.z))
