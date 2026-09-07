"""Recommend Transforms: one pass over a vehicle's used meshes.

The ladder, in order. Each rung is decided by the strongest evidence
available at it, and a mesh leaves the ladder as soon as one claims it.

1.  Handed pairs. A mesh in the wing-mirror assembly or the front door card
    that has an opposite-side twin converts by swapping the two meshes, and
    earns an Equivalent Parts row alongside so the slot selection follows.
2.  Front seats. They convert by slot occupancy alone -- the opposite seat
    already fits -- so they earn the Equivalent Parts row and no mesh
    transform at all.
3.  Driver-bound props. A prop wired to a driver electrics channel is part of
    the driver's station and moves with it -- unless it is a column stalk
    whose opposite number is not modelled, in which case the missing one is
    painted into the dash and this one reflects with it.
4.  Driver-station slots. Everything a part in a steer/pedal/gauge slot
    declares moves too, whether or not it is a prop.
5.  Cabin furniture. What is left in a cabin slot is judged on where it sits:
    at the driver's station it moves, anywhere else it reflects.
6.  Everything else is left alone.

Then four passes over what the ladder decided, answering the settings the
modal used to leave to the parts table: a Move X for the rows whose own
landing point is not the shared steering delta, Texture Fix on the reflecting
rows whose surfaces carry legible marks, the steering reference, and each
trigger box's transform.

Whole-shell meshes (``*_body``, ``*_interior``) are never recommended. Five
of the nine baselines below mirror theirs and four do not, and the reason
differs every time -- an asymmetric interior-mirror base on one, a bonnet
release painted into the texture on another, a footwell that does not meet
the dash on a third. Nothing cheap separates them, and mirroring a whole
shell by mistake is expensive to undo.

2026-09-07: measured against nine hand-configured vehicles (etkc, etk800,
sunburst2, scintilla, vivace ardente, Kia EV3, Audi R8, Lexus LC, Volvo V60).
Mesh modes 98.0% against 95.0% for the name/placement tables this replaces --
Swap Mesh 97% recall at 100% precision where it was 87/82, Move 98/98 where it
was 89/95, Mirror 89/80 where it was 76/64. Trigger boxes 95%, steering
reference 8 of 9, Equivalent Parts 48 of 50. It runs in about 0.2 s on a cold
evidence cache and 15 ms on a warm one, against 10 ms before.
"""

from __future__ import annotations

from .recommendation_common import (
    ASSEMBLY_GAP,
    ASSEMBLY_MESH_LIMIT,
    CABIN_NAME_PATTERNS,
    CABIN_REAR_REACH,
    CENTRED_SELECTOR_TOLERANCE,
    DRIVER_BAND,
    DRIVER_NAME_PATTERNS,
    INERT_NAME_PATTERNS,
    LOW_CONFIDENCE_PATTERNS,
    OFFSET_SIGNIFICANCE,
    RACE_SLOT_TOKENS,
    RIM_WIDTH_SHARE,
    SHIFTER_NAME_PATTERNS,
    SHIFTER_SLOT_TOKENS,
    TEXTURE_GLYPH_PATTERNS,
    WIPER_NAME_PATTERNS,
    _is_seat_mesh_id,
    assembly_func_role,
    cabin_volume,
    in_cabin,
    is_rear_side_slot,
    lone_column_stalk_funcs,
    mesh_bounds_x,
    mesh_center,
    recommendation_matches,
    recommendation_text,
    slot_family,
    slot_role,
)
from .recommendation_evidence import vehicle_evidence
from .recommendation_pairing import (
    handed_mesh_ids,
    handed_pair_family,
    resolve_side_twin,
)
from .shared import core


def _driver_side(context: core.VehicleContext, steering_ids: set[str], center_x: float) -> int:
    """+1 when the wheel sits left of the centreline, -1 when it sits right.

    Decides which member of a pair the modal names first, and which way a
    Move X override points.
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


def _steering_meshes(
    context: core.VehicleContext,
    evidence,
    available: list[str],
) -> set[str]:
    """Meshes the engine drives from the ``steering`` channel.

    The name score stays as the fallback for a vehicle whose wheel is a plain
    flexbody, but a props row saying ``steering`` is what the engine reads, so
    it wins where it exists.
    """
    wired = {
        object_id
        for object_id in available
        if "steering" in evidence.funcs(object_id)
    }
    if wired:
        return wired
    return {
        object_id
        for object_id in available
        if core.steering_ref_score(object_id, context.objects[object_id]) >= 15
    }


def _is_shared_library_mesh(context: core.VehicleContext, object_id: str) -> bool:
    """Whether this mesh is authored in BeamNG's shared vehicle library.

    ``vehicles/common`` is where the game keeps the geometry any vehicle can
    fit -- the generic steering wheels, the racing seats, the pedal boxes --
    and it is the directory the loader falls back to when the vehicle's own
    has no match (``jbeamIO``: {vehicleSpecificDir, '/vehicles/common/'}). A
    mesh that comes from there is an option somebody can fit, not the fitment
    the vehicle ships with.

    Read off the DAE, not the part: a vehicle declares its own parts for the
    shared wheels it offers (``ardente_steer_01a`` sits in the vivace's own
    jbeam), so the part file says nothing, while the mesh inside it still
    comes from ``vehicles/common/steeringwheels.DAE``. The mesh name says
    nothing either -- a mod's vehicle id and its mesh prefix routinely
    disagree, the ardente's being "vivace_ardente" against
    "ardente_steeringwheel".
    """
    obj = context.objects.get(object_id)
    if obj is None or not obj.dae_path:
        return False
    return "/vehicles/common/" in f"/{str(obj.dae_path).replace(chr(92), '/').lower()}"


def _steering_reference(
    context: core.VehicleContext,
    steering_ids: set[str],
) -> str:
    """The one steering mesh the conversion should measure its delta from.

    Every mesh in the column is wired to ``steering`` -- the shaft, the
    joints, the paddles, the hub -- and the delta wants the rim, because that
    is the part whose lateral offset IS the driver's offset. The rim is the
    widest of them by a long way: on the ardente the wheel spans 0.36 m and
    the shaft 0.03 m.

    Width alone is not enough to pick between rims, though, because a vehicle
    ships several: the Kia's GT-Line wheel is 4 mm wider than its base one and
    the ardente's two are identical. So everything within a tenth of the
    widest counts as a rim, and among those the base fitment wins -- the
    vehicle's own part over one from the shared library, then the plainest
    name, because a trim option is the base wheel's name with something added
    ("..._gtline", "..._buttons").
    """
    rims: list[tuple[float, str]] = []
    for object_id in steering_ids:
        bounds = mesh_bounds_x(context, object_id)
        if bounds is not None:
            rims.append((abs(bounds[1] - bounds[0]), object_id))
    if not rims:
        return ""
    widest = max(width for width, _ in rims)
    if widest <= 0.0:
        return ""
    candidates = [
        object_id for width, object_id in rims if width >= widest * RIM_WIDTH_SHARE
    ]
    return min(
        candidates,
        key=lambda object_id: (
            _is_shared_library_mesh(context, object_id),
            len(object_id),
            object_id,
        ),
    )


def _mode_for_cabin_mesh(
    context: core.VehicleContext,
    evidence,
    object_id: str,
    text: str,
    center_x: float,
    side: int,
    lone_stalks: frozenset[str],
) -> tuple[str, str]:
    """Move or Mirror for a mesh in a cabin slot, and why.

    Everything here shares one question: does this move with the driver, or
    does it stay where it is relative to the tunnel and reflect? A prop's func
    answers it outright where there is one. A name answers it for the
    instruments and the console furniture that share a dash slot. What is left
    is answered by where the mesh sits -- the driver's station is a band off
    the centreline, and a mesh inboard of it is console furniture while one
    outboard of it is on the body side.
    """
    if evidence.funcs(object_id) & lone_stalks:
        return core.MODE_MIRROR, "column stalk with no opposite stalk modelled"
    role = assembly_func_role(evidence, object_id)
    if role == "driver":
        return core.MODE_TRANSLATE, "assembly wired to a driver control"
    if role == "centre":
        return core.MODE_MIRROR, "assembly wired to a centre-console control"
    if recommendation_matches(text, DRIVER_NAME_PATTERNS):
        return core.MODE_TRANSLATE, "driver control or instrument"
    if recommendation_matches(text, CABIN_NAME_PATTERNS):
        return core.MODE_MIRROR, "asymmetric cabin furniture"
    if _is_console_selector(context, evidence, object_id, center_x):
        # Nothing in the assembly is wired to the tunnel, so this is the
        # console selector beside the driver rather than a race gate bolted
        # to it, and it travels with them.
        return core.MODE_TRANSLATE, "console gear selector"
    if recommendation_matches(text, SHIFTER_NAME_PATTERNS):
        return core.MODE_MIRROR, "gear selector fixed to the tunnel"
    center = mesh_center(context, object_id)
    if center is not None:
        offset = (center[0] - center_x) * side
        if DRIVER_BAND[0] <= offset <= DRIVER_BAND[1]:
            return core.MODE_TRANSLATE, "sits at the driver's station"
    return core.MODE_MIRROR, "cabin part off the driver's station"


def _at_driver_station(
    context: core.VehicleContext,
    object_id: str,
    center_x: float,
    side: int,
) -> bool:
    """Whether a mesh really stands where the driver sits.

    A steer slot holds the rack, the tie rods and the steering boots as well
    as the wheel; a pedal slot holds the master cylinder as well as the
    pedals. All of that running gear sits on the centreline or straddles it,
    while every driver control the nine baselines move sits clear of the
    centreline on the driver's side -- the closest is the scintilla's
    throttle pedal at 0.153.
    """
    center = mesh_center(context, object_id)
    if center is None:
        return False
    return (center[0] - center_x) * side >= DRIVER_BAND[0]


def _is_console_selector(
    context: core.VehicleContext,
    evidence,
    object_id: str,
    center_x: float,
) -> bool:
    families = [
        slot_family(slot_type, context.vehicle_id)
        for slot_type in evidence.slots(object_id)
    ]
    if families and all(
        any(token in family for token in RACE_SLOT_TOKENS) for family in families
    ):
        return False  # every slot it fills is a race fit-out
    center = mesh_center(context, object_id)
    if center is None or abs(center[0] - center_x) < CENTRED_SELECTOR_TOLERANCE:
        # A selector authored on the centreline goes nowhere either way, so
        # let it fall through to the reflection rather than propose a Move X
        # of zero.
        return False
    return any(
        token in family for family in families for token in SHIFTER_SLOT_TOKENS
    )


def _move_offset(
    context: core.VehicleContext,
    evidence,
    object_id: str,
    moving: set[str],
    center_x: float,
    side: int,
    delta: float,
) -> float | None:
    """The Move X this mesh wants, or None to inherit the shared delta.

    The shared delta is how far the *wheel* travels, and everything at the
    driver's station travels with it. A mesh off that axis has its own
    answer: landing it at its mirrored position means travelling twice its
    own offset from the centreline.

    Most of the time the two are close enough that the shared delta is the
    right answer, and saying so is worth more than being exactly right --
    every Move X written here is a number the user then has to check. So an
    override is only proposed when the shared delta is materially wrong for
    this mesh, which is one of two things:

    * it would carry the mesh straight past the centreline and out the far
      side, which is what happens to anything already sitting on the
      passenger's side -- a dead pedal, a footwell extinguisher;
    * it overshoots the mirrored position by more than half the delta, which
      is what happens to a console gear selector: the etkc's sits 0.034 off
      the centreline and wants 0.068, while the shared delta is 0.820.

    Measured per assembly, not per mesh: the meshes one part declares are one
    object and have to travel together. Against the nine baselines this
    proposes 0.067 for the etkc shifter (hand-set 0.068), 0.060 for the
    etk800's (0.061), 0.145 for the Lexus's (0.145), and the two hand-set
    footwell offsets on the passenger side; it proposes nothing for the
    pedals, gauges and columns the baselines leave on the shared delta.
    """
    own = mesh_bounds_x(context, object_id)
    if own is None:
        return None
    # Grow the group out from this mesh through the moving siblings its part
    # declares, taking only the ones it is laterally next to. Co-declaration
    # alone is not enough: the etkc's manual shifter part declares the clutch
    # pedal and the dashboard too, and stretching the group to the pedal put
    # the gear knob's landing point half a metre out.
    low, high = own
    for _pass in range(len(evidence.part_ids.get(object_id, ())) + 2):
        grew = False
        for part_id in evidence.part_ids.get(object_id, ()):
            meshes = evidence.meshes_by_part.get(part_id, ())
            if len(meshes) > ASSEMBLY_MESH_LIMIT:
                continue
            for sibling in meshes:
                if sibling == object_id or sibling not in moving:
                    continue
                bounds = mesh_bounds_x(context, sibling)
                if bounds is None:
                    continue
                if bounds[0] > high + ASSEMBLY_GAP or bounds[1] < low - ASSEMBLY_GAP:
                    continue
                if bounds[0] < low or bounds[1] > high:
                    low = min(low, bounds[0])
                    high = max(high, bounds[1])
                    grew = True
        if not grew:
            break
    low -= center_x
    high -= center_x
    # low + high is twice the assembly's centre, signed in world space; the
    # side flips it into the conversion's own direction.
    signed = (low + high) * side
    if signed * delta <= 0.0:
        return round(signed, 3)  # the shared delta points the wrong way
    if abs(signed - delta) > max(OFFSET_SIGNIFICANCE, abs(delta) * 0.5):
        return round(signed, 3)
    return None


def build_mode_recommendations(
    context: core.VehicleContext,
    object_ids: list[str],
    config_names: tuple[str, ...] = (),
    trigger_boxes: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    """Recommend a conversion for each used mesh, trigger box and setting."""
    available = sorted(
        object_id for object_id in set(object_ids) if object_id in context.objects
    )
    if not available:
        return []

    evidence = vehicle_evidence(context, tuple(config_names))
    scope = set(available)
    text_by_id = {
        object_id: recommendation_text(context, object_id) for object_id in available
    }
    steering_ids = _steering_meshes(context, evidence, available)
    center_x = core.estimated_vehicle_center_x(context, scope, steering_ids)
    side = _driver_side(context, steering_ids, center_x)
    reference = _steering_reference(context, steering_ids)
    delta = _delta_for_reference(context, reference, center_x)
    handed = handed_mesh_ids(context, evidence, scope)
    volume = cabin_volume(context, steering_ids)
    # Whether each column stalk has its opposite number modelled beside it,
    # taken once. Read over the meshes these trims actually use, not the whole
    # vehicle: the R8 authors a wiper stalk in its spyder body and none in its
    # hardtop, so which trims are selected really does change the answer, and
    # it should -- the stalk beside it is either there in the cabin or it is
    # not.
    lone_stalks = lone_column_stalk_funcs(
        {func for object_id in scope for func in evidence.funcs(object_id)}
    )

    recommendations: list[dict[str, object]] = []
    claimed: set[str] = set()

    def confidence_for(*texts: str) -> str:
        return (
            "low"
            if any(recommendation_matches(text, LOW_CONFIDENCE_PATTERNS) for text in texts)
            else "med"
        )

    # 1. Handed pairs: swap the meshes and record the parts as equivalents.
    for object_id in available:
        if object_id in claimed:
            continue
        if object_id not in handed:
            continue
        if is_rear_side_slot(evidence.slots(object_id), context.vehicle_id):
            continue
        twin_id = resolve_side_twin(
            context,
            object_id,
            [candidate for candidate in available if candidate not in claimed],
            center_x,
        )
        if twin_id is None:
            continue
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
        family, reason = handed_pair_family(context, evidence, first, second)
        recommendations.append({
            "kind": "pair",
            "object_id": first,
            "source_id": second,
            "mode": core.MODE_MIRROR_STRUCTURAL,
            "reason": reason,
            "confidence": "high",
            "equivalent": True,
            "pair_kind": family,
            "marks": "door card lettering would read backwards" if family == "door" else "",
        })

    # 2. Front seats: the opposite seat already fits the slot, so the swap is
    #    an Equivalent Parts row and the meshes themselves stay untouched.
    for object_id in available:
        if object_id in claimed or not _is_seat_mesh_id(object_id):
            continue
        if is_rear_side_slot(evidence.slots(object_id), context.vehicle_id):
            continue
        twin_id = resolve_side_twin(
            context,
            object_id,
            [candidate for candidate in available if candidate not in claimed],
            center_x,
        )
        if twin_id is None:
            continue
        claimed.add(object_id)
        claimed.add(twin_id)
        first, second = object_id, twin_id
        center_a = mesh_center(context, object_id)
        center_b = mesh_center(context, twin_id)
        if (
            center_a is not None
            and center_b is not None
            and side * (center_b[0] - center_x) > side * (center_a[0] - center_x)
        ):
            first, second = twin_id, object_id
        recommendations.append({
            "kind": "equivalent",
            "object_id": first,
            "source_id": second,
            "mode": core.MODE_SKIP,
            "reason": "front seat pair; swaps by slot occupancy",
            "confidence": "high",
            "equivalent": True,
            "pair_kind": "seat",
        })

    # 3-5. Single meshes.
    for object_id in available:
        if object_id in claimed:
            continue
        text = text_by_id[object_id]
        role = assembly_func_role(evidence, object_id)
        slot = slot_role(evidence.slots(object_id), context.vehicle_id)
        inert = recommendation_matches(text, INERT_NAME_PATTERNS)
        cabin = in_cabin(context, object_id, volume)
        # A mesh with an opposite-side twin is half of a pair. Either the pair
        # is handed, in which case rung 1 has already swapped it, or it is a
        # true mirror pair -- a kick plate, a rear door card, a tie rod -- and
        # reflecting one half on its own only moves it onto its own twin.
        twinned = resolve_side_twin(context, object_id, available, center_x) is not None
        marks = ""
        if evidence.funcs(object_id) & lone_stalks and cabin and not twinned:
            mode, reason = core.MODE_MIRROR, "column stalk with no opposite stalk modelled"
            marks = "stalk symbols would read backwards"
        elif role == "driver" and cabin and not inert and not twinned:
            # The cabin gate matters here as much as at the slot rung below:
            # a steering column is one prop chain from the rim to the rack,
            # every link of it wired to "steering", and only the part of it
            # the driver sits behind travels with them.
            mode, reason = core.MODE_TRANSLATE, "prop wired to a driver control"
        elif slot == "driver" and cabin and not inert and not twinned:
            if not _at_driver_station(context, object_id, center_x, side):
                continue  # steering rack, tie rods, pedal-box mounts: hardware
            mode, reason = core.MODE_TRANSLATE, "part fills a driver-station slot"
        elif role == "centre" and not twinned:
            mode, reason = core.MODE_MIRROR, "assembly wired to a centre-console control"
        elif object_id in evidence.mirror_meshes and not twinned:
            # A reflector row with no opposite-side twin is the interior
            # rear-view mirror. Its stalk and base are cut for the driver's
            # eye, so it reflects with the rest of the cabin.
            mode, reason = core.MODE_MIRROR, "interior mirror: a mirrors row with no twin"
        elif recommendation_matches(text, WIPER_NAME_PATTERNS) and not inert and not twinned:
            mode, reason = core.MODE_MIRROR, "wiper parked on the driver's side"
        elif slot == "cabin" and cabin and not inert:
            mode, reason = _mode_for_cabin_mesh(
                context, evidence, object_id, text, center_x, side, lone_stalks
            )
            if mode == core.MODE_MIRROR and twinned:
                continue
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
            "marks": marks,
        })

    _add_move_offsets(context, evidence, recommendations, center_x, side, delta)
    _add_texture_rows(recommendations, text_by_id)
    _add_steering_row(reference, recommendations)
    if trigger_boxes:
        wheel_y = min(
            (c[1] for c in (mesh_center(context, m) for m in steering_ids) if c is not None),
            default=None,
        )
        _add_trigger_rows(recommendations, trigger_boxes, volume, wheel_y)

    kind_order = {"pair": 0, "equivalent": 1, "single": 2, "texture": 3, "steering": 4, "trigger": 5}
    mode_order = {
        core.MODE_TRANSLATE: 0,
        core.MODE_MIRROR: 1,
        core.MODE_MIRROR_STRUCTURAL: 2,
        core.MODE_SKIP: 3,
    }
    recommendations.sort(
        key=lambda item: (
            kind_order.get(str(item.get("kind")), 9),
            mode_order.get(str(item.get("mode")), 99),
            str(item.get("object_id", "")).lower(),
            str(item.get("source_id", "")).lower(),
        )
    )
    return recommendations


def _add_move_offsets(
    context: core.VehicleContext,
    evidence,
    recommendations: list[dict[str, object]],
    center_x: float,
    side: int,
    delta: float,
) -> None:
    """Attach a Move X to the Move rows whose own landing point is not the delta."""
    if delta <= 0.0:
        return
    moving = {
        str(item["object_id"])
        for item in recommendations
        if item.get("mode") == core.MODE_TRANSLATE
    }
    for item in recommendations:
        if item.get("mode") != core.MODE_TRANSLATE:
            continue
        offset = _move_offset(
            context, evidence, str(item["object_id"]), moving, center_x, side, delta
        )
        if offset is None:
            continue
        item["offset"] = offset
        item["reason"] = f"{item['reason']}; lands at its mirrored position"


def _add_texture_rows(
    recommendations: list[dict[str, object]],
    text_by_id: dict[str, str],
) -> None:
    """Turn on Texture Fix where a reflected mesh carries legible marks.

    Only a reflecting mode reverses a texture, so a Move never wants this --
    the parts table refuses it there for the same reason. Within the
    reflecting rows the question is whether anyone would read what is on the
    surface: switch legends, gauge faces, a gear pattern, a warning label. A
    carpet, a headliner or a shifter housing reflects just as happily with
    its texture untouched.
    """
    reflecting = {core.MODE_MIRROR, core.MODE_MIRROR_STRUCTURAL}
    seen: set[str] = set()
    rows: list[dict[str, object]] = []
    for item in recommendations:
        if item.get("mode") not in reflecting:
            continue
        # Two families qualify whatever they are called, because the ladder
        # has already established what they are and the name table has not:
        # a door card, whose window switches are legible marks by definition,
        # and a column stalk, which carries the wiper and indicator symbols.
        # Both are routinely spelt in ways no name table would catch -- the
        # R8 and the Lexus write "r8_door_L" and "lc500_door_L" for the card,
        # and the R8 writes "r8_signal" for the stalk.
        marked = str(item.get("marks") or "")
        for key in ("object_id", "source_id"):
            object_id = str(item.get(key) or "")
            if not object_id or object_id in seen:
                continue
            text = text_by_id.get(object_id, object_id.lower())
            if not marked and not recommendation_matches(text, TEXTURE_GLYPH_PATTERNS):
                continue
            seen.add(object_id)
            rows.append({
                "kind": "texture",
                "object_id": object_id,
                "source_id": "",
                "mode": item["mode"],
                "reason": marked or "reflected surface carries legible marks",
                "confidence": "med",
                "equivalent": False,
                "pair_kind": "",
                "texture": True,
            })
    recommendations.extend(rows)


def _delta_for_reference(
    context: core.VehicleContext,
    reference: str,
    center_x: float,
) -> float:
    """How far the conversion will move the driver's station.

    The same number ``auto_delta_magnitude`` produces once the steering
    reference is set -- twice the wheel's offset from the centreline -- but
    measured from the reference this pass just chose rather than from the
    conversion, which has not been told about it yet. Asking the conversion
    here would read whatever reference a previous session left, or nothing at
    all on a fresh project, and every Move X below is judged against it.
    """
    center = mesh_center(context, reference) if reference else None
    if center is None:
        return 0.0
    return abs(center[0] - center_x) * 2.0


def _add_steering_row(
    reference: str,
    recommendations: list[dict[str, object]],
) -> None:
    if not reference:
        return
    recommendations.append({
        "kind": "steering",
        "object_id": reference,
        "source_id": "",
        "mode": core.MODE_TRANSLATE,
        "reason": "widest mesh on the steering channel: the wheel rim",
        "confidence": "high",
        "equivalent": False,
        "pair_kind": "",
        "steering": True,
    })


def _add_trigger_rows(
    recommendations: list[dict[str, object]],
    trigger_boxes: list[dict[str, object]],
    volume: tuple[float, float] | None,
    wheel_y: float | None,
) -> None:
    """A transform for each trigger box, from what it is attached to.

    A trigger is a clickable volume, so it ends up wherever the surface it
    sits on ends up. Four kinds, in order:

    * A door's boxes stay put whatever else happens. The door itself is not
      converted, so its exterior handle and its interior release both stay on
      it -- and a twinned pair (door_L and door_R both authored) already has a
      box on each side.
    * Head-lining furniture -- sun visors, sunroof, ceiling and interior
      lamps, a rear table -- spans the cabin symmetrically and stays.
    * A box named for a body panel is on that panel when it *is* out there,
      and is the interior release for it when it is inside the cabin. The
      Lexus authors both under the name "trunk": one on the boot lid at
      y=2.45 and one by the driver's knee at y=-0.15. Position separates them;
      the name cannot.
    * What is left is a cabin control, and reflects with the fascia it is set
      into. Every one of them, including the release levers down by the sill
      at x=0.7 and the switches in the middle of the dash.

    No box is recommended for Move. 2026-09-07: five of the 163 baseline boxes
    are hand-set to Move -- two ESC buttons, two headlight switches and one
    ignition -- and nothing separates them from the 56 hand-set to Mirror. A
    lateral test was tried: the headlight switches sit at x=0.55 and 0.57, but
    so do the bonnet and boot releases at x=0.65 to 0.75, which all reflect.
    Reflecting everything scores 155/163; adding the test scored 142.
    """
    for box in trigger_boxes:
        if box.get("twinned"):
            continue
        at = box.get("at") or box.get("centre")
        if not at or len(at) != 3:
            continue
        lowered = str(box.get("trigger") or "").lower()
        if recommendation_matches(lowered, _TRIGGER_FIXED_PATTERNS):
            continue
        inside = _trigger_in_cabin(at, volume, wheel_y)
        if recommendation_matches(lowered, _TRIGGER_PANEL_PATTERNS) and not inside:
            continue
        recommendations.append({
            "kind": "trigger",
            "object_id": str(box.get("label") or box.get("trigger") or ""),
            "source_id": "",
            "mode": core.MODE_MIRROR,
            "reason": "cabin control; follows the fascia",
            "confidence": "med",
            "equivalent": False,
            "pair_kind": "",
            "trigger_key": box.get("key"),
        })


def _trigger_in_cabin(
    at,
    volume: tuple[float, float] | None,
    wheel_y: float | None,
) -> bool:
    """Whether a trigger box sits inside the passenger compartment.

    The front edge is the cabin volume the meshes already use; the back edge
    is a car's length of rear seat behind the wheel, which is enough to hold
    a rear bench and short of any boot lid.
    """
    if volume is None or wheel_y is None:
        return True
    return volume[0] <= float(at[1]) <= wheel_y + CABIN_REAR_REACH


# Boxes that stay where they are however the cabin converts. A door is not
# converted, so nothing on it moves -- interior release included. Head-lining
# furniture spans the cabin symmetrically.
_TRIGGER_FIXED_PATTERNS = (
    r"^door",
    r"^sunroof|sunvisor|^visor",
    r"cellinglight|ceilinglight|int_lights|domelight|^table",
)

# Boxes named for a body panel. On the panel when they are out at it; the
# interior release for it when they are inside the cabin.
_TRIGGER_PANEL_PATTERNS = (
    r"^hood|^bonnet",
    r"^trunk|^tailgate|^boot",
    r"latch|catch|^chargingcap|^fuelcap|^gascap",
)
