from __future__ import annotations

import unittest
from pathlib import Path

from beamxp import hand_drive_core as core
from beamxp.core.models import SlotRelocation


def part(
    part_id: str,
    slot_type: str,
    display_name: str,
    slots: tuple[tuple[str, str], ...] = (),
    extra: str = "",
) -> str:
    slot_rows = ""
    if slots:
        rows = ",\n".join(
            f'["{slot_id}", "{default_part}", "{slot_id}"]'
            for slot_id, default_part in slots
        )
        slot_rows = (
            ',\n"slots": [\n'
            '    ["type", "default", "description"],\n'
            f"    {rows}\n"
            "]"
        )
    return (
        f'"{part_id}": {{\n'
        f'"information": {{"name": "{display_name}"}},\n'
        f'"slotType": "{slot_type}"'
        f"{slot_rows}"
        f"{extra}\n"
        "}"
    )


def context_with_parts(
    part_index: dict[str, tuple[str, str]],
    selected_by_config: dict[str, dict[str, object]],
) -> core.VehicleContext:
    context = core.VehicleContext(
        source_zip=Path("test.zip"),
        vehicle_id="acme",
        vehicle_path="vehicles/acme",
        dae_paths=[],
        variants={
            name: core.VariantInfo(name, f"{name}.pc", None, name)
            for name in selected_by_config
        },
        objects={},
        preview_by_id={},
        jbeam_texts={},
        node_positions={},
        project_dir=Path("project"),
        part_body_index=part_index,
    )
    for name, selected in selected_by_config.items():
        context.selected_parts_cache[name] = selected
    return context


def selection(instances: tuple[tuple[str, str, str], ...]) -> dict[str, object]:
    """Build a resolved-parts dict from (part_id, slot_id, slot_path) triples."""
    return {
        "part_instances": [
            {
                "instance_id": f"{slot_path}{part_id}",
                "part_id": part_id,
                "slot_id": slot_id,
                "slot_path": slot_path,
                "inherited_options": (),
            }
            for part_id, slot_id, slot_path in instances
        ],
        "selected_by_slot": {slot_id: part_id for part_id, slot_id, _ in instances},
        "selected_by_path": {
            f"{slot_path}": part_id for part_id, _, slot_path in instances
        },
        "part_slot_options": {},
    }


# A cabin with a left and a right seat slot, each with its own authored part,
# plus race buckets that fit only their own side -- the shape that makes a
# mesh-level mode useless and a slot pair necessary.
CABIN_PARTS = {
    "car": (
        part("car", "main", "Car", (("seat_FL", "seat_FL"), ("seat_FR", "seat_FR"))),
        "car.jbeam",
    ),
    "seat_FL": (part("seat_FL", "seat_FL", "Front Left Seat"), "seats.jbeam"),
    "seat_FR": (part("seat_FR", "seat_FR", "Front Right Seat"), "seats.jbeam"),
    "race_seat_FL": (
        part("race_seat_FL", "seat_FL", "Front Left Race Seat", (("skin_FL", "skin_FL_red"),)),
        "seats.jbeam",
    ),
    "race_seat_FR": (
        part("race_seat_FR", "seat_FR", "Front Right Race Seat", (("skin_FR", "skin_FR_red"),)),
        "seats.jbeam",
    ),
    "skin_FL_red": (part("skin_FL_red", "skin_FL", "Red Skin"), "seats.jbeam"),
    "skin_FR_red": (part("skin_FR_red", "skin_FR", "Red Skin"), "seats.jbeam"),
}


# The etkc's drift trim: a race seat sits in a race-seat base on the left, and
# the right seat position holds an ordinary seat that declares no race seat
# slot at all. The mirrored slot an equivalent row wants does not exist in this
# trim, and inventing one used to delete the seat from both sides.
ONE_SIDED_PARTS = {
    "car": (
        part("car", "main", "Car", (("seat_FL", "seat_FL"), ("seat_FR", "seat_FR"))),
        "car.jbeam",
    ),
    "race_base_FL": (
        part("race_base_FL", "seat_FL", "Race Base", (("race_seat_FL", "race_seat_FL"),)),
        "seats.jbeam",
    ),
    "plain_seat_FR": (part("plain_seat_FR", "seat_FR", "Seat"), "seats.jbeam"),
    "race_seat_FL": (part("race_seat_FL", "race_seat_FL", "Race Seat L"), "seats.jbeam"),
    # Authored by the vehicle, but nothing in this trim opens a slot for it.
    "race_seat_FR": (part("race_seat_FR", "race_seat_FR", "Race Seat R"), "seats.jbeam"),
}


# The etkc's drift trim: a race seat sits in a race-seat base on the left, and
# the passenger seat slot on the right is empty. The mirrored slot an
# equivalent row names does not exist yet -- it is declared by the right race
# base that would go into that empty slot.
EMPTY_OPPOSITE_PARTS = {
    "car": (
        part("car", "main", "Car", (("seat_FL", "seat_FL"), ("seat_FR", "seat_FR"))),
        "car.jbeam",
    ),
    "race_base_FL": (
        part("race_base_FL", "seat_FL", "Race Base L", (("race_seat_FL", "race_seat_FL"),)),
        "seats.jbeam",
    ),
    "race_base_FR": (
        part("race_base_FR", "seat_FR", "Race Base R", (("race_seat_FR", "race_seat_FR"),)),
        "seats.jbeam",
    ),
    # Also fits the empty slot, but declares no race seat slot, so it is never
    # the part that opens the path.
    "plain_seat_FR": (part("plain_seat_FR", "seat_FR", "Seat R"), "seats.jbeam"),
    "race_seat_FL": (part("race_seat_FL", "race_seat_FL", "Race Seat L"), "seats.jbeam"),
    "race_seat_FR": (part("race_seat_FR", "race_seat_FR", "Race Seat R"), "seats.jbeam"),
}


# The row the user draws in Equivalent Parts: it names the seat inside the
# base, and carries the full path of the counterpart on the other side.
CHILD_ROW = {
    "left": "race_seat_FL@@/seat_FL/race_seat_FL/",
    "right": "race_seat_FR@@/seat_FR/race_seat_FR/",
    "kind": "seat",
}


# The hopper: the frame declares two body slots and each trim fills one of
# them, so the same seat is reached through `body` on one trim and through
# `body_crawler` on another. An Equivalent Parts row is captured from whichever
# trim was on screen, so its path names one of those routes -- and the table is
# vehicle-level, so it has to apply on both.
TWO_BODY_PARTS = {
    "car": (
        part("car", "main", "Car", (("body", "body"), ("body_crawler", ""))),
        "car.jbeam",
    ),
    "body": (
        part("body", "body", "Body", (("seat_FL", "seat_FL"), ("seat_FR", "seat_FR"))),
        "body.jbeam",
    ),
    "body_crawler": (
        part(
            "body_crawler",
            "body_crawler",
            "Crawler Body",
            (("seat_FL", "seat_FL"), ("seat_FR", "seat_FR")),
        ),
        "body.jbeam",
    ),
    "race_base_FL": (
        part("race_base_FL", "seat_FL", "Race Base L", (("race_seat_FL", "race_seat_FL"),)),
        "seats.jbeam",
    ),
    "race_base_FR": (
        part("race_base_FR", "seat_FR", "Race Base R", (("race_seat_FR", "race_seat_FR"),)),
        "seats.jbeam",
    ),
    "race_seat_FL": (part("race_seat_FL", "race_seat_FL", "Race Seat L"), "seats.jbeam"),
    "race_seat_FR": (part("race_seat_FR", "race_seat_FR", "Race Seat R"), "seats.jbeam"),
}


def flexbodies(*meshes: str) -> str:
    rows = ",\n".join(f'["{mesh}", ["body"]]' for mesh in meshes)
    return (
        ',\n"flexbodies": [\n'
        '    ["mesh", "[group]:"],\n'
        f"    {rows}\n"
        "]"
    )


# The same cabin with mesh ids that are nothing like their part ids, plus a
# sport seat that reuses the base seat's mesh -- the shape the Equivalent Parts
# table actually records, because its rows come from the Parts Used mesh list.
MESH_CABIN_PARTS = {
    "car": (
        part("car", "main", "Car", (("seat_FL", "seat_FL"), ("seat_FR", "seat_FR"))),
        "car.jbeam",
    ),
    "seat_FL": (
        part("seat_FL", "seat_FL", "Front Left Seat", extra=flexbodies("seat_base_L")),
        "seats.jbeam",
    ),
    "seat_FR": (
        part("seat_FR", "seat_FR", "Front Right Seat", extra=flexbodies("seat_base_R")),
        "seats.jbeam",
    ),
    "sport_seat_FL": (
        part("sport_seat_FL", "seat_FL", "Sport Seat", extra=flexbodies("seat_base_L", "sport_bolster_L")),
        "seats.jbeam",
    ),
    "sport_seat_FR": (
        part("sport_seat_FR", "seat_FR", "Sport Seat", extra=flexbodies("seat_base_R", "sport_bolster_R")),
        "seats.jbeam",
    ),
    "race_seat_FL": (
        part("race_seat_FL", "seat_FL", "Race Seat", extra=flexbodies("seat_shell_L")),
        "seats.jbeam",
    ),
    "race_seat_FR": (
        part("race_seat_FR", "seat_FR", "Race Seat", extra=flexbodies("seat_shell_R")),
        "seats.jbeam",
    ),
}


# A door assembly that declares its own card's flexbody, with the card also
# sold as a part in a child slot -- etk800's shape, and the one that makes a
# mesh-named row reachable from two very different parts.
DOOR_CARD_PARTS = {
    "car": (
        part("car", "main", "Car", (("door_FL", "door_FL"), ("door_FR", "door_FR"))),
        "car.jbeam",
    ),
    "door_FL": (
        part(
            "door_FL",
            "door_FL",
            "Front Left Door",
            (("doorpanel_FL", "doorpanel_FL"),),
            extra=flexbodies("door_skin_FL", "doorpanel_FL"),
        ),
        "doors.jbeam",
    ),
    "door_FR": (
        part(
            "door_FR",
            "door_FR",
            "Front Right Door",
            (("doorpanel_FR", "doorpanel_FR"),),
            extra=flexbodies("door_skin_FR", "doorpanel_FR"),
        ),
        "doors.jbeam",
    ),
    "doorpanel_FL": (
        part("doorpanel_FL", "doorpanel_FL", "Front Left Door Card",
             extra=flexbodies("doorpanel_FL")),
        "doors.jbeam",
    ),
    "doorpanel_FR": (
        part("doorpanel_FR", "doorpanel_FR", "Front Right Door Card",
             extra=flexbodies("doorpanel_FR")),
        "doors.jbeam",
    ),
}


class SlotPairPlanTests(unittest.TestCase):
    def _context(self, instances: tuple[tuple[str, str, str], ...]) -> core.VehicleContext:
        return context_with_parts(CABIN_PARTS, {"trim": selection(instances)})

    def _plan(self, context: core.VehicleContext) -> dict[str, object] | None:
        return core.resolve_slot_pair_plan(context, "trim", [("seat_FL", "seat_FR")])

    def test_matching_seats_are_left_alone(self) -> None:
        context = self._context((
            ("car", "main", "/"),
            ("seat_FL", "seat_FL", "/seat_FL/"),
            ("seat_FR", "seat_FR", "/seat_FR/"),
        ))
        self.assertIsNone(self._plan(context))

    def test_lone_seat_moves_to_the_paired_slot(self) -> None:
        context = self._context((
            ("car", "main", "/"),
            ("seat_FL", "seat_FL", "/seat_FL/"),
        ))
        plan = self._plan(context)
        assert plan is not None
        self.assertEqual(
            [(entry["slotId"], entry["partId"]) for entry in plan["selections"]],
            [("seat_FR", "seat_FR")],
        )
        # The vacated slot must be written empty, not dropped: dropping it
        # would let the engine re-apply the slot's authored default.
        self.assertEqual(
            [(entry["slotId"], entry.get("setEmpty")) for entry in plan["clears"]],
            [("seat_FL", "1")],
        )

    def test_two_occupied_seats_are_exchanged(self) -> None:
        context = self._context((
            ("car", "main", "/"),
            ("seat_FL", "seat_FL", "/seat_FL/"),
            ("race_seat_FR", "seat_FR", "/seat_FR/"),
            ("skin_FR_red", "skin_FR", "/seat_FR/skin_FR/"),
        ))
        plan = self._plan(context)
        assert plan is not None
        updates = {entry["slotId"]: entry["partId"] for entry in plan["selections"]}
        self.assertEqual(updates["seat_FR"], "seat_FR")
        self.assertEqual(updates["seat_FL"], "race_seat_FL")
        # The trim-specific child choice follows the bucket across the car.
        self.assertEqual(updates["skin_FL"], "skin_FL_red")

    def test_exchange_survives_application_to_a_config(self) -> None:
        context = self._context((
            ("car", "main", "/"),
            ("seat_FL", "seat_FL", "/seat_FL/"),
            ("race_seat_FR", "seat_FR", "/seat_FR/"),
            ("skin_FR_red", "skin_FR", "/seat_FR/skin_FR/"),
        ))
        plan = self._plan(context)
        assert plan is not None
        pc: dict[str, object] = {
            "parts": {
                "seat_FL": "seat_FL",
                "seat_FR": "race_seat_FR",
                "skin_FR": "skin_FR_red",
            }
        }
        core.apply_hand_authored_group(pc, plan)
        # Both sides land: a two-way swap must not let one direction's vacate
        # delete what the other direction just wrote.
        self.assertEqual(pc["parts"]["seat_FL"], "race_seat_FL")
        self.assertEqual(pc["parts"]["seat_FR"], "seat_FR")
        self.assertEqual(pc["parts"]["skin_FL"], "skin_FL_red")
        self.assertNotIn("skin_FR", pc["parts"])

    def test_seat_without_a_counterpart_becomes_a_relocation(self) -> None:
        parts = dict(CABIN_PARTS)
        # A co-driver bucket with nothing authored for the other side.
        parts["rally_seat_FL"] = (
            part("rally_seat_FL", "seat_FL", "Rally Co-driver Seat"),
            "seats.jbeam",
        )
        context = context_with_parts(
            parts,
            {"trim": selection((
                ("car", "main", "/"),
                ("rally_seat_FL", "seat_FL", "/seat_FL/"),
            ))},
        )
        plan = core.resolve_slot_pair_plan(context, "trim", [("seat_FL", "seat_FR")])
        assert plan is not None
        self.assertEqual(plan["selections"], [])
        self.assertEqual(
            [(entry["slotId"], entry["partId"]) for entry in plan["relocations"]],
            [("seat_FR", "rally_seat_FL")],
        )

    def test_paired_parts_are_hidden_from_the_generated_mirroring_pass(self) -> None:
        context = self._context((
            ("car", "main", "/"),
            ("seat_FL", "seat_FL", "/seat_FL/"),
        ))
        plan = self._plan(context)
        assert plan is not None
        self.assertIn("seat_FL", core.authored_group_source_parts(plan))


class EquivalentPartPlanTests(unittest.TestCase):
    def _context(self, instances: tuple[tuple[str, str, str], ...]) -> core.VehicleContext:
        return context_with_parts(CABIN_PARTS, {"trim": selection(instances)})

    def test_equivalent_part_moves_to_authored_counterpart(self) -> None:
        context = self._context((
            ("car", "main", "/"),
            ("seat_FL", "seat_FL", "/seat_FL/"),
        ))
        plan = core.resolve_side_pair_plan(
            context,
            "trim",
            [{"left": "seat_FL", "right": "seat_FR", "kind": "seat"}],
        )
        assert plan is not None
        self.assertEqual(
            [(entry["slotId"], entry["partId"]) for entry in plan["selections"]],
            [("seat_FR", "seat_FR")],
        )
        self.assertEqual(
            [(entry["slotId"], entry.get("setEmpty")) for entry in plan["clears"]],
            [("seat_FL", "1")],
        )

    def test_equivalent_part_exchanges_mismatched_seats(self) -> None:
        context = self._context((
            ("car", "main", "/"),
            ("seat_FL", "seat_FL", "/seat_FL/"),
            ("race_seat_FR", "seat_FR", "/seat_FR/"),
            ("skin_FR_red", "skin_FR", "/seat_FR/skin_FR/"),
        ))
        plan = core.resolve_side_pair_plan(
            context,
            "trim",
            [
                {"left": "seat_FL", "right": "seat_FR", "kind": "seat"},
                {"left": "race_seat_FL", "right": "race_seat_FR", "kind": "seat"},
            ],
        )
        assert plan is not None
        updates = {entry["slotId"]: entry["partId"] for entry in plan["selections"]}
        self.assertEqual(updates["seat_FR"], "seat_FR")
        self.assertEqual(updates["seat_FL"], "race_seat_FL")
        self.assertEqual(updates["skin_FL"], "skin_FL_red")

    def test_equivalent_part_without_counterpart_becomes_relocation(self) -> None:
        parts = dict(CABIN_PARTS)
        parts["rally_seat_FL"] = (
            part("rally_seat_FL", "seat_FL", "Rally Co-driver Seat"),
            "seats.jbeam",
        )
        context = context_with_parts(
            parts,
            {"trim": selection((
                ("car", "main", "/"),
                ("rally_seat_FL", "seat_FL", "/seat_FL/"),
            ))},
        )
        plan = core.resolve_side_pair_plan(
            context,
            "trim",
            [{"left": "rally_seat_FL", "right": "rally_seat_FR", "kind": "seat"}],
        )
        assert plan is not None
        self.assertEqual(plan["selections"], [])
        self.assertEqual(
            [(entry["slotId"], entry["partId"]) for entry in plan["relocations"]],
            [("seat_FR", "rally_seat_FL")],
        )

    def _one_sided_context(self) -> core.VehicleContext:
        return context_with_parts(
            ONE_SIDED_PARTS,
            {"trim": selection((
                ("car", "main", "/"),
                ("race_base_FL", "seat_FL", "/seat_FL/"),
                ("race_seat_FL", "race_seat_FL", "/seat_FL/race_seat_FL/"),
                ("plain_seat_FR", "seat_FR", "/seat_FR/"),
            ))},
        )

    def test_a_mirrored_slot_the_trim_lacks_never_becomes_a_selection(self) -> None:
        # The slot def stood in for a slot that is not there, and every
        # counterpart fits a def that allows exactly its own type -- so the
        # plan wrote a part into /seat_FL/race_seat_FR/, a path nothing
        # declares. Resolution dropped it and kept the clear, which is how the
        # seat disappeared from the preview and the build.
        plan = core.resolve_side_pair_plan(
            self._one_sided_context(),
            "trim",
            [{"left": "race_seat_FL", "right": "race_seat_FR", "kind": "seat"}],
        )
        selections = plan["selections"] if plan else []
        self.assertEqual(
            [entry["slotPath"] for entry in selections],
            [],
            "a slot no part in the trim declares cannot be selected into",
        )

    def test_the_source_part_is_never_cleared_without_a_replacement(self) -> None:
        # The destructive half: whatever the row decides, it must not empty the
        # slot the seat is in unless something takes its place.
        plan = core.resolve_side_pair_plan(
            self._one_sided_context(),
            "trim",
            [{"left": "race_seat_FL", "right": "race_seat_FR", "kind": "seat"}],
        )
        if plan is None:
            return
        emptied = {
            entry["slotId"] for entry in plan["clears"] if entry.get("setEmpty")
        }
        filled = {entry["slotId"] for entry in plan["selections"]}
        filled |= {entry["sourceSlotId"] for entry in plan["relocations"]}
        moved_into = {entry["slotId"] for entry in plan["relocations"]}
        self.assertEqual(
            emptied - filled - moved_into,
            set(),
            "a slot was emptied with nothing put anywhere in its place",
        )

    def test_a_part_row_still_relocates_when_the_slot_is_missing(self) -> None:
        # The documented fallback: no slot to fit into means clone the source
        # across, exactly as when a slot exists but no part fits it.
        plan = core.resolve_side_pair_plan(
            self._one_sided_context(),
            "trim",
            [{"left": "race_seat_FL", "right": "race_seat_FR", "kind": "seat"}],
        )
        assert plan is not None
        self.assertEqual(
            [(entry["slotId"], entry["partId"]) for entry in plan["relocations"]],
            [("race_seat_FR", "race_seat_FL")],
        )

    def test_an_empty_mirrored_slot_is_still_fitted_normally(self) -> None:
        # The case the existence check must not break: the slot is declared by
        # a part the trim fits, it just has nothing in it.
        context = self._context((
            ("car", "main", "/"),
            ("seat_FL", "seat_FL", "/seat_FL/"),
        ))
        plan = core.resolve_side_pair_plan(
            context,
            "trim",
            [{"left": "seat_FL", "right": "seat_FR", "kind": "seat"}],
        )
        assert plan is not None
        self.assertEqual(
            [(entry["slotId"], entry["partId"]) for entry in plan["selections"]],
            [("seat_FR", "seat_FR")],
        )

    def _empty_opposite_context(self) -> core.VehicleContext:
        return context_with_parts(
            EMPTY_OPPOSITE_PARTS,
            {"trim": selection((
                ("car", "main", "/"),
                ("race_base_FL", "seat_FL", "/seat_FL/"),
                ("race_seat_FL", "race_seat_FL", "/seat_FL/race_seat_FL/"),
            ))},
        )

    def test_a_row_naming_a_slot_the_trim_can_open_rises_to_the_assembly(self) -> None:
        # The row asks for a seat inside a base that is not fitted yet, so the
        # swap happens at the base: fit the right base into the empty slot,
        # and the seat follows it as a paired child.
        plan = core.resolve_side_pair_plan(
            self._empty_opposite_context(), "trim", [dict(CHILD_ROW)]
        )
        assert plan is not None
        self.assertEqual(
            [(entry["slotId"], entry["partId"]) for entry in plan["selections"]],
            [("seat_FR", "race_base_FR"), ("race_seat_FR", "race_seat_FR")],
        )

    def test_the_lifted_swap_empties_the_side_it_came_from(self) -> None:
        plan = core.resolve_side_pair_plan(
            self._empty_opposite_context(), "trim", [dict(CHILD_ROW)]
        )
        assert plan is not None
        self.assertIn(
            ("seat_FL", "1"),
            [(entry["slotId"], entry.get("setEmpty")) for entry in plan["clears"]],
        )

    def test_the_lift_matches_pairing_the_assemblies_by_hand(self) -> None:
        # The whole justification: naming the child is a longer way of saying
        # what pairing the two bases says, so both must plan the same swap.
        context = self._empty_opposite_context()
        by_child = core.resolve_side_pair_plan(context, "trim", [dict(CHILD_ROW)])
        by_base = core.resolve_side_pair_plan(
            context,
            "trim",
            [{"left": "race_base_FL", "right": "race_base_FR", "kind": "seat"}],
        )
        assert by_child is not None and by_base is not None
        self.assertEqual(by_child["selections"], by_base["selections"])
        self.assertEqual(by_child["clears"], by_base["clears"])

    def test_an_occupied_opposite_slot_is_never_taken_over(self) -> None:
        # Opening a slot is only safe while it is empty: a part already fitted
        # there was chosen, by the trim or by the user, and is not ours to
        # discard on the strength of a row about someone else's child.
        context = context_with_parts(
            EMPTY_OPPOSITE_PARTS,
            {"trim": selection((
                ("car", "main", "/"),
                ("race_base_FL", "seat_FL", "/seat_FL/"),
                ("race_seat_FL", "race_seat_FL", "/seat_FL/race_seat_FL/"),
                ("plain_seat_FR", "seat_FR", "/seat_FR/"),
            ))},
        )
        plan = core.resolve_side_pair_plan(context, "trim", [dict(CHILD_ROW)])
        selections = plan["selections"] if plan else []
        self.assertNotIn("seat_FR", [entry["slotId"] for entry in selections])

    def test_a_row_applies_to_a_trim_that_reaches_the_part_another_way(self) -> None:
        # The row was drawn on the crawler trim, so it names the crawler's
        # route. The road trim reaches the same seat through its own body, and
        # the table is vehicle-level -- holding the authoring trim's ancestry
        # against it left the hopper's single-seat drag trim unconverted, its
        # seat still on the left with the wheel already on the right.
        context = context_with_parts(TWO_BODY_PARTS, {"trim": selection((
            ("car", "main", "/"),
            ("body", "body", "/body/"),
            ("race_base_FL", "seat_FL", "/body/seat_FL/"),
            ("race_seat_FL", "race_seat_FL", "/body/seat_FL/race_seat_FL/"),
        ))})
        plan = core.resolve_side_pair_plan(
            context,
            "trim",
            [{
                "left": "race_seat_FL@@/body_crawler/seat_FL/race_seat_FL/",
                "right": "race_seat_FR@@/body_crawler/seat_FR/race_seat_FR/",
                "kind": "seat",
            }],
        )
        assert plan is not None
        self.assertEqual(
            [(entry["slotId"], entry["partId"]) for entry in plan["selections"]],
            [("seat_FR", "race_base_FR"), ("race_seat_FR", "race_seat_FR")],
        )
        self.assertIn(
            ("seat_FL", "1"),
            [(entry["slotId"], entry.get("setEmpty")) for entry in plan["clears"]],
        )

    def test_a_row_with_no_path_lifts_to_the_mirrored_place(self) -> None:
        # A bare name carries no path, but the row is still a mirror
        # statement: the place it asks for is the source's own path reflected,
        # so it opens the same slot the fully qualified row does. The hopper's
        # single-seat drag trim depends on it -- its racing-seat row was drawn
        # against a trim that has both sides, so the counterpart is bare.
        context = self._empty_opposite_context()
        bare = core.resolve_side_pair_plan(
            context,
            "trim",
            [{"left": "race_seat_FL", "right": "race_seat_FR", "kind": "seat"}],
        )
        qualified = core.resolve_side_pair_plan(context, "trim", [dict(CHILD_ROW)])
        assert bare is not None and qualified is not None
        self.assertEqual(bare["selections"], qualified["selections"])
        self.assertEqual(bare["clears"], qualified["clears"])

    def test_equivalent_row_naming_a_mesh_reaches_the_part_that_carries_it(self) -> None:
        # The table is filled from the Parts Used rows, which are mesh
        # instances, so a row routinely names a mesh whose id is nothing like
        # the part's -- bx pairs `racing_seat_FL`, carried by `race_seat_FL`.
        context = context_with_parts(MESH_CABIN_PARTS, {"trim": selection((
            ("car", "main", "/"),
            ("race_seat_FL", "seat_FL", "/seat_FL/"),
        ))})
        plan = core.resolve_side_pair_plan(
            context,
            "trim",
            [{
                "left": "seat_shell_L@@/seat_FL/",
                "right": "seat_shell_R@@/seat_FR/",
                "kind": "seat",
            }],
        )
        assert plan is not None
        self.assertEqual(
            [(entry["slotId"], entry["partId"]) for entry in plan["selections"]],
            [("seat_FR", "race_seat_FR")],
        )
        self.assertEqual(
            [(entry["slotId"], entry.get("setEmpty")) for entry in plan["clears"]],
            [("seat_FL", "1")],
        )

    def test_a_row_that_hands_nothing_across_leaves_the_meshes_alone(self) -> None:
        # Both sides already hold their own counterpart, so the row has nothing
        # to swap on this trim. Reporting the parts as covered would drop their
        # meshes from the generated pass and cancel the Swap Mesh set on them --
        # the ardente's wing mirrors, which never change slots.
        context = self._context((
            ("car", "main", "/"),
            ("seat_FL", "seat_FL", "/seat_FL/"),
            ("seat_FR", "seat_FR", "/seat_FR/"),
        ))
        plan = core.resolve_side_pair_plan(
            context,
            "trim",
            [{"left": "seat_FL", "right": "seat_FR", "kind": "seat"}],
        )
        self.assertIsNone(plan)
        self.assertEqual(core.authored_group_meshes(context, plan), set())

    def test_a_door_card_row_never_reaches_the_door_that_carries_it(self) -> None:
        # The door declares the card's flexbody, so the card's row matches the
        # door as readily as the card -- and the door does not fit the opposite
        # door's card slot, so it used to come out as a whole-door relocation.
        # The preview then drew every door mesh twice, the second copy mirrored
        # over the Swap Mesh the row was about.
        context = context_with_parts(DOOR_CARD_PARTS, {"trim": selection((
            ("car", "main", "/"),
            ("door_FL", "door_FL", "/door_FL/"),
            ("doorpanel_FL", "doorpanel_FL", "/door_FL/doorpanel_FL/"),
            ("door_FR", "door_FR", "/door_FR/"),
            ("doorpanel_FR", "doorpanel_FR", "/door_FR/doorpanel_FR/"),
        ))})
        plan = core.resolve_side_pair_plan(
            context,
            "trim",
            [{"left": "doorpanel_FL", "right": "doorpanel_FR", "kind": "door"}],
        )
        self.assertIsNone(plan)
        self.assertEqual(core.authored_group_meshes(context, plan), set())

    def test_a_card_row_still_reaches_the_door_when_no_card_part_exists(self) -> None:
        # With no dedicated card part in the trim, the door IS the part that
        # answers for the mesh -- and the answer is the opposite door, which
        # already occupies its slot, so the row hands nothing across.
        context = context_with_parts(DOOR_CARD_PARTS, {"trim": selection((
            ("car", "main", "/"),
            ("door_FL", "door_FL", "/door_FL/"),
            ("door_FR", "door_FR", "/door_FR/"),
        ))})
        plan = core.resolve_side_pair_plan(
            context,
            "trim",
            [{"left": "doorpanel_FL", "right": "doorpanel_FR", "kind": "door"}],
        )
        self.assertIsNone(plan)

    def test_a_part_row_wins_over_a_mesh_row_the_part_reuses(self) -> None:
        # The sport seat reuses the base seat's mesh, so the base seat's mesh
        # row must not capture it and swap it for the base part on the far side.
        context = context_with_parts(MESH_CABIN_PARTS, {"trim": selection((
            ("car", "main", "/"),
            ("sport_seat_FL", "seat_FL", "/seat_FL/"),
        ))})
        plan = core.resolve_side_pair_plan(
            context,
            "trim",
            [
                {"left": "seat_base_L", "right": "seat_base_R", "kind": "seat"},
                {"left": "sport_seat_FL", "right": "sport_seat_FR", "kind": "seat"},
            ],
        )
        assert plan is not None
        self.assertEqual(
            [(entry["slotId"], entry["partId"]) for entry in plan["selections"]],
            [("seat_FR", "sport_seat_FR")],
        )

    def test_equivalent_preview_color_paths_only_include_written_changes(self) -> None:
        source = {
            "selected_by_path": {
                "/": "car",
                "/seat_FL/": "seat_FL",
                "/seat_FR/": "race_seat_FR",
            }
        }
        target = {
            "selected_by_path": {
                "/": "car",
                "/seat_FL/": "seat_FL",
                "/seat_FR/": "seat_FR",
                "/seat_FR/race_seat_FR/": "race_seat_FR",
            }
        }

        self.assertEqual(
            core._changed_selected_slot_paths(source, target),
            {"/seat_FR/", "/seat_FR/race_seat_FR/"},
        )


class SlotPairSettingsTests(unittest.TestCase):
    def test_pairing_is_a_bijection(self) -> None:
        conversion: dict[str, object] = {"slotPairs": []}
        core.set_slot_pair(conversion, "seat_FL", "seat_FR")
        self.assertEqual(core.slot_pair_partner(conversion, "seat_FR"), "seat_FL")
        # Re-pairing one side must free its old partner rather than leave a
        # slot claimed by two pairs.
        core.set_slot_pair(conversion, "seat_FL", "mirror_R")
        self.assertEqual(core.slot_pair_partner(conversion, "seat_FR"), "")
        self.assertEqual(core.slot_pair_partner(conversion, "mirror_R"), "seat_FL")
        core.set_slot_pair(conversion, "seat_FL", "")
        self.assertEqual(core.active_slot_pairs(conversion), [])

    def test_self_pairs_and_duplicates_are_rejected(self) -> None:
        pairs = core.normalized_slot_pairs([
            {"a": "seat_FL", "b": "seat_FL"},
            {"a": "seat_FR", "b": "seat_FL"},
            {"a": "seat_FL", "b": "mirror_R"},
            {"a": "", "b": "x"},
        ])
        self.assertEqual(pairs, [{"a": "seat_FL", "b": "seat_FR", "enabled": True}])

    def test_swap_mesh_settings_are_preserved(self) -> None:
        parts = {
            "left": {"mode": core.MODE_MIRROR_STRUCTURAL, "mirrorSource": "right"},
            "other": {"mode": core.MODE_TRANSLATE},
        }
        conversion = {"parts": parts}
        self.assertEqual(core.active_part_modes(conversion)["left"], core.MODE_MIRROR_STRUCTURAL)
        self.assertEqual(parts["left"], {"mode": core.MODE_MIRROR_STRUCTURAL, "mirrorSource": "right"})
        self.assertEqual(parts["other"], {"mode": core.MODE_TRANSLATE})

    def test_swap_mesh_is_offerable(self) -> None:
        self.assertIn(core.MODE_MIRROR_STRUCTURAL, core.MODE_CHOICES)


class RelocationRewriteTests(unittest.TestCase):
    BODY = (
        '"seat_FL": {\n'
        '"information": {"name": "Front Left Seat"},\n'
        '"slotType": "seat_FL",\n'
        '"slots2": [\n'
        '    ["name", "allowTypes", "denyTypes", "default", "description"],\n'
        '    ["skin_FL", ["skin_FL"], [], "skin_FL_red", "Skin", {"nodeMove":{"x":0.35,"y":0.1,"z":0.2}}]\n'
        "],\n"
        '"flexbodies": [\n'
        '    ["mesh", "[group]:"],\n'
        '    ["seat_mesh", ["floor", "seat_FL"]]\n'
        "],\n"
        '"nodes": [\n'
        '    ["id", "posX", "posY", "posZ"],\n'
        '    {"group":"seat_FL"},\n'
        '    ["sf1l", 0.35, 0.1, 0.5]\n'
        "],\n"
        '"beams": [\n'
        '    ["id1:", "id2:"],\n'
        '    ["sf1l", "f2l"]\n'
        "]\n"
        "}"
    )

    def _relocate(self, offset=(0.0, 0.0, 0.0)) -> str:
        groups = {"floor", "seat_FL", "seat_FR"}
        nodes = {"sf1l", "sf1r", "f2l", "f2r"}
        return core.relocate_part_for_slot(
            self.BODY,
            SlotRelocation("seat_FL", "seat_FR", offset),
            {"sf1l": "sf1r", "f2l": "f2r"},
            nodes,
            core.build_lateral_name_map(groups),
            groups,
            core.build_lateral_name_map({"skin_FL", "skin_FR"}),
            {"skin_FL", "skin_FR"},
        )

    def test_relocation_rewrites_slot_nodes_groups_and_child_slots(self) -> None:
        out = self._relocate()
        self.assertIn('"slotType": "seat_FR"', out)
        self.assertIn('["sf1r", -0.35, 0.1, 0.5]', out)
        self.assertIn('["sf1r", "f2r"]', out)
        self.assertIn('["floor", "seat_FR"]', out)
        self.assertIn('"skin_FR"', out)
        # The child slot's mounting crosses the car with the part.
        self.assertIn('"x":-0.35', out)
        self.assertIn('{"group":"seat_FR"}', out)

    def test_relocation_offset_is_injected_as_a_node_move(self) -> None:
        out = self._relocate((0.0, 0.25, 0.12))
        self.assertIn('"nodeMove"', out.split('"nodes"')[1])

    def test_lateral_name_map_refuses_unconfirmed_guesses(self) -> None:
        # mirror_lateral_node_id turns the trailing r of "floor" into an l;
        # the map must only keep swaps whose result is a real name.
        mapping = core.build_lateral_name_map({"floor", "seat_FL", "seat_FR"})
        self.assertNotIn("floor", mapping)
        self.assertEqual(mapping["seat_FL"], "seat_FR")


if __name__ == "__main__":
    unittest.main()
