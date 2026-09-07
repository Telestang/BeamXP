"""What each piece of evidence means, behind Recommend Transforms.

The recommender's ladder is in ``recommendation_engine``; this module holds
the tables it reads and the small geometric helpers it shares with
``recommendation_pairing``.

The ordering of evidence is deliberate and matches the repo's standing rule:
a deterministic field the engine consumes beats a name every time.

1.  A prop's ``func`` (``sections/meshs.lua``). The author states which
    electrics channel drives the mesh, so ``steering``/``brake``/``throttle``
    is the driver's station and ``parkingbrake``/``sequentialLeverY`` is the
    centre console. Measured over nine hand-configured vehicles: ``steering``
    41/42 Move, the pedal funcs 73/73 Move, ``parkingbrake`` 14/14 Mirror,
    ``sequentialLeverY`` 9/9 Mirror.
2.  A ``mirrors`` row (``sections/mirror.lua``) and the slotType a part fills
    (``slotSystem.fillSlots_rec``). Between them these name the wing-mirror
    assembly and the front door card without enumerating spellings.
3.  Where the mesh sits, from the bounds already cached for the preview.
4.  Names, last, and only to separate families inside one cabin slot -- a
    wiper from the windscreen it shares a slot with.

2026-09-07: measured against the nine vehicles listed in
``tests/test_recommendation_engine.py``.
"""

from __future__ import annotations

import re

from .shared import core

# A named twin must sit at least this far off the centreline before the
# pairing is believed: a centred mesh has no opposite side to swap with.
PAIR_MIN_OFFSET = 0.05

# How far a Move's own mirrored landing point may differ from the vehicle's
# shared steering delta before it is worth proposing as a Move X override.
# Below this the two answers put the mesh in the same place.
OFFSET_SIGNIFICANCE = 0.06

# The band, measured from the centreline, in which a cabin mesh counts as
# standing at the driver's station rather than on the centre console or out
# at the body side. Gauges, head-up displays and pedal boxes land in it;
# dashboards, consoles and handbrakes sit inboard of it; door furniture and
# bonnet-release levers sit outboard.
DRIVER_BAND = (0.15, 0.56)

# How far behind the wheel the passenger compartment reaches, for deciding
# whether a box named for a body panel is on the panel or is its interior
# release. Long enough for a rear bench, short of any boot lid.
CABIN_REAR_REACH = 1.5

# The side tokens a mesh name uses to say which side it is on. "lhd"/"rhd"
# are deliberately absent -- they mark mutually exclusive builds of the whole
# cabin, not two halves of one, and are masked out in the pairing.
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

# ---------------------------------------------------------------------------
# props func -> what the mesh is for
# ---------------------------------------------------------------------------

# Channels that only a driver's control or instrument is wired to. The mesh
# moves with the driver, so it converts by Move.
#
# The gauge channels are here because a needle is driven by the value it
# reads: a vehicle can add its own (boost, oilpressure, airspeed) and the
# prefix match below catches those without a new entry.
#
# Lamp channels are deliberately absent -- "highbeam", "lowbeam",
# "brakelights", "signal_L/R". They read like dashboard tell-tales and are
# nothing of the sort: the R8's shared SPOTLIGHT cone carries all eight of
# them, is declared by every light part in the vehicle AND by both wing
# mirrors, and sits on the centreline. A tell-tale that really is on the
# binnacle is claimed by its gauge slot instead.
DRIVER_FUNCS = frozenset({
    "steering",
    "brake",
    "throttle",
    "clutch",
    # Both column stalks -- but see STALK_FUNC_PARTNERS below: a stalk
    # modelled without its opposite number reflects rather than moves.
    "wipers",
    "turnsignal",
    "ignitionLevel",
    "dualAxisX",
    "dualAxisY",
    "wheelspeed",
    "rpm",
    "fuel",
    "oiltemp",
    "watertemp",
    "coolanttemp",
    "oilpressure",
    "boost",
    "airspeed",
    "altitude",
    "odometer",
    "gearIndex",
})

# Channels wired to something that lives on the centre console between the
# seats. It stays where it is relative to the tunnel, so it converts by
# Mirror rather than by Move.
CENTRE_FUNCS = frozenset({
    "parkingbrake",
    "sequentialLeverY",
    "sequentialLeverX",
    "gearModeIndex",
})

# Prefixes that identify a gauge channel a vehicle invented for itself.
_GAUGE_FUNC_PREFIXES = ("needle", "gauge", "temp", "press", "level", "volt")

# The two column stalks, each named by the other. They are a pair straddling
# the column, and how one converts depends on whether the other is modelled.
#
# With both of them there, the column carries both and their arrangement is
# preserved, so they Move. With only one, the other is painted into the dash
# texture -- and the dash is reflected, which has already carried the painted
# stalk across the column. The modelled stalk has to reflect too or the two
# end up on the same side of the column.
#
# 2026-09-07: this is what separates the Audi R8 and the Volvo V60, whose
# indicator stalks are hand-set to Mirror and which model no wiper stalk,
# from the Kia EV3, ardente, etkc, etk800 and sunburst2, which model both and
# are hand-set to Move. 7 of 7, off a field the engine consumes.
STALK_FUNC_PARTNERS = {
    "turnsignal": "wipers",
    "wipers": "turnsignal",
}


def lone_column_stalk_funcs(funcs_in_scope: set[str]) -> frozenset[str]:
    """Stalk channels this vehicle models without their opposite number."""
    return frozenset(
        func
        for func, partner in STALK_FUNC_PARTNERS.items()
        if func in funcs_in_scope and partner not in funcs_in_scope
    )


def prop_func_role(funcs: frozenset[str]) -> str:
    """"driver", "centre" or "" for a mesh's set of prop funcs.

    A mesh can carry more than one row (a needle authored once per gauge
    skin); the driver reading wins, because a mesh wired to a driver channel
    at all is part of the driver's station.
    """
    if not funcs:
        return ""
    if funcs & DRIVER_FUNCS:
        return "driver"
    if funcs & CENTRE_FUNCS:
        return "centre"
    for func in funcs:
        lowered = func.lower()
        if any(lowered.startswith(prefix) for prefix in _GAUGE_FUNC_PREFIXES):
            return "driver"
    return ""


# A part declaring more meshes than this is a trim's whole fit-out, not one
# assembly: a Volvo dash part carries the dashboard, the head lining, the
# interior mirror, the navigation screen and the pedals together. Borrowing a
# prop's reading across a part that size says nothing about any of them.
ASSEMBLY_MESH_LIMIT = 10

# How far apart two meshes one part declares may sit laterally and still be
# treated as one object when working out where a Move should land. A gear
# knob and its boot overlap; a gear knob and the clutch pedal declared beside
# it are half a metre apart and travel independently.
ASSEMBLY_GAP = 0.10

# A gear selector authored closer than this to the centreline goes nowhere
# whichever transform it gets, and a Move X of zero reads as an instruction
# without being one. The etkc's sits at 0.034 and the etk800's at 0.030, so
# this has to be small; the sunburst2's is on the centreline exactly.
CENTRED_SELECTOR_TOLERANCE = 0.015

# How much narrower than the widest mesh on the steering channel another one
# may be and still count as a rim rather than a shaft. A vehicle's optional
# rims differ by millimetres; a shaft is a tenth of a rim's width.
RIM_WIDTH_SHARE = 0.9


def assembly_func_role(evidence, object_id: str) -> str:
    """The prop reading for the assembly this mesh belongs to.

    A handbrake's lever is a ``parkingbrake`` prop while its body, boots and
    line beside it are plain flexbodies; a race shifter's knob is a
    ``sequentialLeverY`` prop while its stick and housing are not. Reading the
    mesh alone answers for the prop and shrugs at the rest of the assembly,
    which is how ``grp_hb_body`` and ``grp_shifter_m_stick`` end up
    disagreeing with the levers they are bolted to.

    Only the *centre* reading is borrowed from a sibling, and only from a
    small part. An assembly anchored to the tunnel anchors everything bolted
    to it, so the reading travels; a driver control does not make its
    neighbours into driver controls, and letting it travel put the Volvo's
    dashboard, head lining and interior mirror on Move because the pedals
    happened to share their part. 2026-09-07: that mistake cost 52 rows.
    """
    own = prop_func_role(evidence.funcs(object_id))
    if own:
        return own
    for part_id in evidence.part_ids.get(object_id, ()):
        meshes = evidence.meshes_by_part.get(part_id, ())
        if len(meshes) > ASSEMBLY_MESH_LIMIT:
            continue
        for sibling in meshes:
            if sibling != object_id and prop_func_role(evidence.funcs(sibling)) == "centre":
                return "centre"
    return ""


# ---------------------------------------------------------------------------
# slotType -> what the part is for
# ---------------------------------------------------------------------------

# Slots that hold the driver's station. Everything a part in one of these
# declares moves with the driver: the wheel and its column top, the pedal box
# and its linkages, the instrument binnacle, the shift light.
#
# Matched on the slotType with the vehicle prefix stripped, so "etkc_pedals",
# "vivace_pedalbox_race" and "scintilla_pedals" all answer to "pedal".
DRIVER_SLOT_TOKENS = (
    "steer",
    "pedal",
    "gauge",
    "shiftlight",
    "instrument",
    "digidash",
    "hud",
)

# Slots that hold cabin furniture: fixed to the tunnel or spanning the cabin,
# asymmetric, and converted by reflecting them.
#
# Deliberately short. "roof", "rollcage", "carpet", "headliner" and "antenna"
# were tried and removed: the meshes that want them (etkc_intcarpet,
# sunburst2_headliner_sedan) already sit in a dash or interior slot, while the
# slots named for them hold roof racks, light bars and cage tubes that nothing
# should touch.
CABIN_SLOT_TOKENS = (
    "dash",
    "interior",
    "console",
    "handbrake",
    "parkingbrake",
    "shifter",
    "shiftknob",
    "laptop",
    "column",
    "windshield",
    "windscreen",
    "ambient",
    "radar",
    "extinguisher",
)

# The slot a wing-mirror assembly fills, and the slot a front door card fills.
# Both are matched with a side token so a rear door's card is left alone.
MIRROR_SLOT_RE = re.compile(r"(?:^|_)mirror(?:_|$)")
DOOR_CARD_SLOT_RE = re.compile(r"door_?(?:panel|card)")
DOOR_SLOT_RE = re.compile(r"(?:^|_)door(?:_|$)")
# Slots whose contents are glazing or exterior trim on a door, never the card.
DOOR_EXCLUDE_SLOT_RE = re.compile(r"glass|window|trim|cladding|flare|handle|sill")

# Side tokens as they appear in a slotType, split front from rear. Only the
# front doors are handed: a rear door card holds no driver controls, and
# etk800_doorpanel_RL/RR are authored as a true mirror pair.
FRONT_SIDE_SLOT_RE = re.compile(r"(?:^|_)(?:fl|fr|l|r|left|right)(?:_|$)")
REAR_SIDE_SLOT_RE = re.compile(r"(?:^|_)(?:rl|rr|rearleft|rearright)(?:_|$)")


def slot_family(slot_type: str, vehicle_id: str) -> str:
    """One slotType with its vehicle prefix stripped, lowercased."""
    lowered = str(slot_type or "").lower()
    prefix = f"{str(vehicle_id or '').lower()}_"
    if prefix != "_" and lowered.startswith(prefix):
        lowered = lowered[len(prefix):]
    return lowered


def slot_role(slot_types: frozenset[str], vehicle_id: str) -> str:
    """"driver", "cabin" or "" for the slots a mesh is declared in.

    A mesh is often declared by several parts, and the reading has to be a
    majority of them rather than any one. ``etkc_body`` is declared by the
    body, both inner-fender variants and the front swaybar, and one of those
    parts fills the windscreen slot -- enough for an "any" reading to call the
    whole shell cabin furniture. The shared ``SPOTLIGHT`` cone is worse: it
    belongs to every light part in the vehicle and to the wing mirrors, so its
    slots say everything and therefore nothing.

    Where a mesh has one declaring part -- which is the ordinary case -- the
    majority and the "any" reading are the same thing.
    """
    families = [slot_family(slot_type, vehicle_id) for slot_type in slot_types]
    if not families:
        return ""
    threshold = len(families) / 2.0
    driver = 0
    cabin = 0
    for family in families:
        # Cabin first: a family can answer to both tables and mean the cabin
        # reading. "steering_column" holds the "steer" token but is the column
        # shroud, which stays with the bulkhead.
        if any(token in family for token in CABIN_SLOT_TOKENS):
            cabin += 1
        elif any(token in family for token in DRIVER_SLOT_TOKENS):
            driver += 1
    if driver >= threshold and driver >= cabin:
        return "driver"
    if cabin >= threshold:
        return "cabin"
    return ""


def is_rear_side_slot(slot_types: frozenset[str], vehicle_id: str) -> bool:
    return any(
        REAR_SIDE_SLOT_RE.search(slot_family(slot_type, vehicle_id))
        for slot_type in slot_types
    )


# ---------------------------------------------------------------------------
# names, used only to separate families inside one slot
# ---------------------------------------------------------------------------

# A driver's instrument or control whose slot does not say so -- a digital
# dash and its screen live in the dash slot alongside the dashboard itself.
DRIVER_NAME_PATTERNS = (
    r"digidash|digital_?dash|(?:^|_)cluster|instrument_?panel",
    r"gauge|needle|speedo|tacho|tachometer|decals_gauges|fueltemp",
    r"(?:gas|brake|clutch|throttle)_?pedal|pedal_?(?:gas|brake|clutch|throttle)",
    r"pedalbox|pedal_?box|padalbox",
    r"steer(?:ing)?_?wheel|steerwheel|steershaft|steerwheel_hub|(?:^|_)steer_[0-9]",
    r"paddle|signal_?stalk|wiper_?stalk|turn_?stalk",
    r"shift_?light|(?:^|_)hud(?:_|$)|head_?up",
    r"dash_?key|ignition_?key",
    # Only the column TOP moves with the wheel: the shaft below it runs to
    # the rack and mirrors with the bulkhead.
    r"column\w*_?top|column_support|column1",
)

# Cabin furniture whose slot does not say so, or which shares a slot with
# something inert.
CABIN_NAME_PATTERNS = (
    r"dash|dashboard|console|fascia",
    r"parking_?brake|park_?brake|pbrake|hand_?brake|(?:^|_)hb_",
    r"radio|laptop|(?:^|_)nav(?:_|$)|navi|sat_?nav",
    r"wiper(?!_?stalk)",
    # The column shroud below the wheel stays with the bulkhead; only the
    # top, matched in the driver table above, rides round with the wheel.
    r"steering_?column|(?:^|_)column(?:_|$)",
    r"intmirror|inter(?:ior)?_?mirror|grp_mirror|rear_?view",
    r"hazard|(?:^|_)key(?:_|$)|switchpanel|(?:^|_)buttons?(?:_|$)",
    r"extinguisher|hood_?release|bonnet_?release",
    r"carpet|headliner|head_?lining",
    # A display panel mirrors like any other fascia part. The texture flip
    # that keeps it readable is derived at build time from the beamNavigator
    # controller, not recommended here. "windscreen" must not match.
    r"(?<!wind)screen",
)

# Names that mean glazing, structure or bodywork: a mesh in a cabin slot that
# matches one of these is left alone however its slot reads.
# A bare "body" token is deliberately absent: "grp_hb_body" is a handbrake
# and "sunburst2_console_body" is a console. A whole shell is kept out by its
# slot instead -- a *_body slot is not a cabin slot -- which is also why the
# recommender never proposes mirroring one (2026-09-07: asked for and
# declined; the five vehicles that need it need it for five different reasons
# and nothing cheap separates them from the four that do not).
INERT_NAME_PATTERNS = (
    r"glass|window(?!net)|windshield|windscreen|glazing",
    r"(?:^|_)roof(?:_|$)|roof_panel|pillar|sunroof|sunvisor",
    r"underbody|chassis|frame|quarter|fender|bumper|cladding",
    # Steering and pedal hardware fills the same slots as the controls it
    # drives, but lives under the floor or in front of the bulkhead.
    r"(?:^|_)rack(?:_|$)|steeringrack|tierod|tie_rod|swaybar|subframe",
    r"steering_?box|steeringboot|steering_?boot|knuckle|hydro",
    # A rear wiper is on the tailgate, which the conversion does not touch.
    r"rear_?wiper|wipers?_?r(?:_|$)",
    r"seat(?:_|$)|seats|bench|belt|harness",
    r"antenna|aerial",
)

# A wiper blade is cabin furniture parked on the driver's side of the screen,
# but the slot it sits in is the windscreen's on most vehicles and the body's
# on some, so the name has to carry it. "wiperstalk" is claimed by the driver
# table first: the stalk rides round with the column.
# The rear wiper is kept out by INERT_NAME_PATTERNS, not here.
WIPER_NAME_PATTERNS = (r"wiper(?!_?stalk)",)

# Slots holding the gear selector. When nothing in the assembly is wired to a
# centre-console channel, the selector is on the console beside the driver and
# travels with them; when something is -- a sequential lever, a gate -- the
# assembly is anchored to the tunnel and reflects instead.
SHIFTER_SLOT_TOKENS = ("shifter", "shiftknob", "gearlever")

# ...but not a race one. A slot named for a race fit-out holds a sequential
# gate or an H-pattern unit bolted to the tunnel behind the handbrake, and it
# stays with the tunnel. This is the slot's own name, which is what BeamNG
# selects the part by, so it is a step above reading the mesh's.
RACE_SLOT_TOKENS = ("race", "racing", "rally", "drift")

# Checked after the console-selector rule, not with the rest of the cabin
# furniture: a gear selector's own name says "shifter" whichever kind it is,
# so the slot has to be asked first. A dashboard whose part also fills a
# shifter slot -- etkc_dash does -- is caught by "dash" before either.
SHIFTER_NAME_PATTERNS = (
    r"shifter|shift_?knob|(?:^|_)grp_shift|gear_?lever|gearstick",
)

# Names whose verdict rides on a token the evidence cannot confirm, so the
# row is offered unticked rather than applied by Select All.
LOW_CONFIDENCE_PATTERNS = (
    r"column|carpet|headliner|rollcage|(?:^|_)stand(?:_|$)|cable",
    r"footplate|footrest|floor",
)

# A mirrored mesh whose texture carries marks a reader would notice running
# backwards: switch legends, gauge faces, badges, gear patterns, warning
# labels. Texture Fix only ever applies to a reflecting mode, so this table
# only has to separate the lettered cabin parts from the plain ones.
TEXTURE_GLYPH_PATTERNS = (
    r"dash|dashboard|console|fascia|switchpanel",
    r"(?:^|_)buttons?(?:_|$)|(?:^|_)button_",
    r"(?<!wind)screen|display|(?:^|_)nav(?:_|$)|navi|gps|hud",
    r"shift_?knob|shifter$|shifter_a(?:_|$)|knob_a|gear_?knob",
    r"door_?panel|door_?card|doorpanel",
    r"extinguisher|laptop(?!_mount)|latch|placard|label|badge",
    r"signal_?stalk|taxistatus",
    # "interior" as a whole trim mesh carries the door and dash lettering with
    # it; "interior_mirror" is a reflector and carries nothing.
    r"interior$|interior_facelift",
    r"gauge|instrument|cluster",
)

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


def cabin_volume(
    context: core.VehicleContext,
    steering_ids: set[str],
) -> tuple[float, float] | None:
    """The (min y, min z) a mesh must clear to count as being in the cabin.

    A steer slot holds the wheel AND the rack, tie rods and steering boots
    that run to the front axle, and a pedal slot holds the pedal box and the
    master cylinder behind the bulkhead. Slot type alone therefore claims a
    lot of running gear, all of it in front of and below the driver.

    The wheel is the anchor: it is the one mesh whose position the whole
    conversion is measured from. The pedals reach about 0.6 m forward of it
    and its column drops about 0.6 m below, so a mesh clearing both by a
    margin is in the cabin and one that does not is under the bonnet.
    Returns None when no wheel can be located, which leaves the gate open.
    """
    forward: list[float] = []
    lows: list[float] = []
    for object_id in steering_ids:
        center = mesh_center(context, object_id)
        if center is not None:
            forward.append(center[1])
            lows.append(center[2])
    if not forward:
        return None
    return (min(forward) - 0.75, min(lows) - 0.75)


def in_cabin(
    context: core.VehicleContext,
    object_id: str,
    volume: tuple[float, float] | None,
) -> bool:
    if volume is None:
        return True
    center = mesh_center(context, object_id)
    if center is None:
        return True
    return center[1] >= volume[0] and center[2] >= volume[1]


def mesh_bounds_x(context: core.VehicleContext, object_id: str) -> tuple[float, float] | None:
    """The mesh's lateral extent, for grouping a Move's landing point."""
    preview = context.preview_by_id.get(object_id)
    if preview is None:
        return None
    bounds = preview.get("bounds")
    if not bounds or len(bounds) != 2:
        return None
    try:
        return (float(bounds[0][0]), float(bounds[1][0]))
    except (TypeError, ValueError, IndexError):
        return None
