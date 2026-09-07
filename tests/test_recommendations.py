"""Recommend Transforms: what the game's own jbeam earns each mesh.

The recommender reads three things the engine reads -- a prop's ``func``, the
``mirrors`` section, and the slotType a part fills -- plus the mesh centres
already cached for the preview. Names come last and only separate families
inside one cabin slot. Fixtures are therefore part bodies plus positions, and
`test_recommending_never_reads_geometry` guards that no spatial work creeps
back in.

2026-09-07: rewritten with the recommender. The suite it replaces described
the name/placement tables that used to decide everything, and those tables --
`HANDED_PATTERNS`, `TRANSLATE_PATTERNS` and the rest -- no longer exist.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from beamxp import hand_drive_core as core
from beamxp import hand_drive_tool as tool
from tests.cabin_fixtures import cabin_parts, mesh_context, part_body


def named_cabin() -> dict[str, tuple[float, float, float]]:
    """A left-hand-drive cabin named the way BeamNG names one."""
    return {
        "veh_dash": (0.0, -0.60, 0.95),
        "veh_floor": (0.0, 0.10, 0.20),
        "veh_headliner": (0.0, 0.50, 1.36),
        "veh_windscreen": (0.0, -0.78, 1.15),
        "veh_doorpanel_FL": (0.73, -0.20, 0.68),
        "veh_doorpanel_FR": (-0.73, -0.20, 0.68),
        "veh_door_FL": (0.82, -0.20, 0.80),
        "veh_door_FR": (-0.82, -0.20, 0.80),
        "veh_seat_FL": (0.40, 0.15, 0.55),
        "veh_seat_FR": (-0.40, 0.15, 0.55),
        "veh_mirror_L": (0.90, -0.42, 1.00),
        "veh_mirror_R": (-0.90, -0.42, 1.00),
        "veh_steer": (0.40, -0.30, 0.95),
    }


def recommend(
    centers: dict[str, tuple[float, float, float] | None],
    parts: dict[str, str] | None = None,
    **kwargs,
) -> dict[str, dict]:
    context = mesh_context(centers, parts=parts or cabin_parts(), **kwargs)
    return {
        row["object_id"]: row
        for row in tool.build_mode_recommendations(context, list(centers))
        if row.get("kind") != "texture"
    }


def all_rows(
    centers: dict[str, tuple[float, float, float] | None],
    parts: dict[str, str] | None = None,
    trigger_boxes: list[dict] | None = None,
) -> list[dict]:
    context = mesh_context(centers, parts=parts or cabin_parts())
    return tool.build_mode_recommendations(context, list(centers), (), trigger_boxes)


class PropFuncTests(unittest.TestCase):
    """The func column is what the engine reads, so it is what we read."""

    def test_a_steering_prop_moves_whatever_it_is_called(self) -> None:
        centers = named_cabin()
        centers["oddly_named_rim"] = (0.40, -0.30, 0.95)
        parts = cabin_parts()
        parts["veh_wheel"] = part_body(
            "veh_wheel_slot", props=(("steering", "oddly_named_rim"),)
        )
        recs = recommend(centers, parts)
        self.assertEqual(recs["oddly_named_rim"]["mode"], core.MODE_TRANSLATE)
        self.assertEqual(
            recs["oddly_named_rim"]["reason"], "prop wired to a driver control"
        )

    def test_pedal_and_gauge_channels_move(self) -> None:
        centers = named_cabin()
        centers.update({
            "veh_pedal_a": (0.30, -0.80, 0.28),
            "veh_pedal_b": (0.20, -0.80, 0.28),
            "veh_needle_x": (0.40, -0.52, 0.94),
        })
        parts = cabin_parts()
        parts["veh_pedals"] = part_body(
            "veh_pedals",
            props=(("brake", "veh_pedal_a"), ("throttle", "veh_pedal_b")),
        )
        parts["veh_gauges"] = part_body(
            "veh_gauges", props=(("wheelspeed", "veh_needle_x"),)
        )
        recs = recommend(centers, parts)
        for mesh in ("veh_pedal_a", "veh_pedal_b", "veh_needle_x"):
            self.assertEqual(recs[mesh]["mode"], core.MODE_TRANSLATE, mesh)

    def test_a_parking_brake_channel_reflects_instead(self) -> None:
        centers = named_cabin()
        centers["veh_lever"] = (0.02, -0.20, 0.65)
        parts = cabin_parts()
        parts["veh_handbrake"] = part_body(
            "veh_handbrake", props=(("parkingbrake", "veh_lever"),)
        )
        recs = recommend(centers, parts)
        self.assertEqual(recs["veh_lever"]["mode"], core.MODE_MIRROR)

    def test_an_assembly_follows_the_lever_it_is_bolted_to(self) -> None:
        """A handbrake's body and boots carry no func of their own."""
        centers = named_cabin()
        centers.update({
            "veh_hb_lever": (0.02, -0.20, 0.65),
            "veh_hb_body": (0.02, -0.24, 0.58),
            "veh_hb_boot": (0.02, -0.22, 0.60),
        })
        parts = cabin_parts()
        parts["veh_handbrake"] = part_body(
            "veh_handbrake",
            flexbodies=("veh_hb_body", "veh_hb_boot"),
            props=(("parkingbrake", "veh_hb_lever"),),
        )
        recs = recommend(centers, parts)
        for mesh in ("veh_hb_body", "veh_hb_boot"):
            self.assertEqual(recs[mesh]["mode"], core.MODE_MIRROR, mesh)
            self.assertEqual(
                recs[mesh]["reason"], "assembly wired to a centre-console control"
            )

    def _stalk_cabin(self, wiper: bool) -> dict:
        centers = named_cabin()
        centers["veh_signalstalk"] = (0.50, -0.46, 0.90)
        props = [("turnsignal", "veh_signalstalk")]
        if wiper:
            centers["veh_wiperstalk"] = (-0.50, -0.46, 0.90)
            props.append(("wipers", "veh_wiperstalk"))
        parts = cabin_parts()
        parts["veh_stalks"] = part_body("veh_dash", props=tuple(props))
        return recommend(centers, parts)

    def test_a_stalk_moves_when_its_opposite_number_is_modelled(self) -> None:
        recs = self._stalk_cabin(wiper=True)
        for mesh in ("veh_signalstalk", "veh_wiperstalk"):
            self.assertEqual(recs[mesh]["mode"], core.MODE_TRANSLATE, mesh)

    def test_a_lone_stalk_reflects_instead(self) -> None:
        """With no wiper stalk modelled it is painted into the dash, and the
        dash is reflected -- so the indicator stalk has to reflect too or the
        two end up on the same side of the column."""
        row = self._stalk_cabin(wiper=False)["veh_signalstalk"]
        self.assertEqual(row["mode"], core.MODE_MIRROR)
        self.assertEqual(row["reason"], "column stalk with no opposite stalk modelled")

    def test_a_lamp_channel_is_not_a_driver_control(self) -> None:
        """A shared light cone carries highbeam and signal funcs and belongs
        to every light part in the vehicle; it is not the driver's."""
        centers = named_cabin()
        centers["SPOTLIGHT"] = (0.0, -0.26, 0.67)
        parts = cabin_parts()
        parts["veh_headlight_L"] = part_body(
            "veh_headlight_L", props=(("highbeam", "SPOTLIGHT"),)
        )
        recs = recommend(centers, parts)
        self.assertNotIn("SPOTLIGHT", recs)


class DriverStationTests(unittest.TestCase):
    def test_a_driver_station_slot_moves_its_meshes(self) -> None:
        centers = named_cabin()
        centers["veh_binnacle"] = (0.40, -0.52, 0.94)
        parts = cabin_parts()
        parts["veh_gauges"] = part_body("veh_gauges", flexbodies=("veh_binnacle",))
        recs = recommend(centers, parts)
        self.assertEqual(recs["veh_binnacle"]["mode"], core.MODE_TRANSLATE)
        self.assertEqual(
            recs["veh_binnacle"]["reason"], "part fills a driver-station slot"
        )

    def test_steering_hardware_in_the_same_slot_is_left_alone(self) -> None:
        """A steer slot holds the rack and the tie rods as well as the rim."""
        centers = named_cabin()
        centers.update({
            "veh_steeringrack": (0.0, -1.30, 0.30),
            "veh_tierod_L": (0.55, -1.30, 0.30),
            "veh_tierod_R": (-0.55, -1.30, 0.30),
        })
        parts = cabin_parts()
        parts["veh_steering"] = part_body(
            "veh_steering",
            flexbodies=("veh_steeringrack", "veh_tierod_L", "veh_tierod_R"),
        )
        recs = recommend(centers, parts)
        for mesh in ("veh_steeringrack", "veh_tierod_L", "veh_tierod_R"):
            self.assertNotIn(mesh, recs, mesh)

    def test_a_move_row_off_the_wheel_axis_carries_its_own_move_x(self) -> None:
        centers = named_cabin()
        centers["veh_shifter_knob"] = (0.04, -0.20, 0.66)
        parts = cabin_parts()
        parts["veh_shifter"] = part_body("veh_shifter", flexbodies=("veh_shifter_knob",))
        recs = recommend(centers, parts)
        row = recs["veh_shifter_knob"]
        self.assertEqual(row["mode"], core.MODE_TRANSLATE)
        # Twice its own offset from the centreline: where a reflection would
        # have put it, without reflecting the mesh.
        self.assertAlmostEqual(row["offset"], 0.08, places=2)

    def test_a_move_row_on_the_wheel_axis_inherits_the_shared_delta(self) -> None:
        centers = named_cabin()
        centers["veh_gaspedal"] = (0.38, -0.80, 0.28)
        parts = cabin_parts()
        parts["veh_pedals"] = part_body("veh_pedals", flexbodies=("veh_gaspedal",))
        self.assertIsNone(recommend(centers, parts)["veh_gaspedal"].get("offset"))


class CabinFurnitureTests(unittest.TestCase):
    def test_a_dash_slot_mesh_on_the_centreline_reflects(self) -> None:
        recs = recommend(named_cabin())
        self.assertEqual(recs["veh_dash"]["mode"], core.MODE_MIRROR)

    def test_glazing_and_structure_in_a_cabin_slot_are_left_alone(self) -> None:
        recs = recommend(named_cabin())
        self.assertNotIn("veh_windscreen", recs)

    def test_a_mesh_in_no_cabin_slot_is_left_alone(self) -> None:
        centers = named_cabin()
        centers["veh_widget"] = (0.20, -0.30, 0.70)
        parts = cabin_parts()
        parts["veh_misc"] = part_body("veh_misc", flexbodies=("veh_widget",))
        self.assertNotIn("veh_widget", recommend(centers, parts))

    def test_an_interior_mirror_row_with_no_twin_reflects(self) -> None:
        centers = named_cabin()
        centers["veh_intmirror"] = (0.0, -0.55, 1.30)
        parts = cabin_parts()
        parts["veh_roof"] = part_body(
            "veh_roof", flexbodies=("veh_intmirror",), mirrors=("veh_intmirror",)
        )
        recs = recommend(centers, parts)
        self.assertEqual(recs["veh_intmirror"]["mode"], core.MODE_MIRROR)
        self.assertEqual(
            recs["veh_intmirror"]["reason"], "interior mirror: a mirrors row with no twin"
        )


class HandedFamilyTests(unittest.TestCase):
    def test_door_cards_and_wing_mirrors_swap_meshes(self) -> None:
        recs = recommend(named_cabin())
        card = recs["veh_doorpanel_FL"]
        self.assertEqual(card["mode"], core.MODE_MIRROR_STRUCTURAL)
        self.assertEqual(card["source_id"], "veh_doorpanel_FR")
        self.assertEqual(card["pair_kind"], "door")
        wing = recs["veh_mirror_L"]
        self.assertEqual(wing["mode"], core.MODE_MIRROR_STRUCTURAL)
        self.assertEqual(wing["pair_kind"], "mirror")

    def test_a_door_skin_beside_a_card_slot_is_not_swapped(self) -> None:
        self.assertNotIn("veh_door_FL", recommend(named_cabin()))

    def test_a_door_slot_with_no_card_slot_swaps_its_own_mesh(self) -> None:
        """The R8 and the Lexus name the card "..._door_L" and author no
        separate door-card slot; the door slot's mesh is then the card."""
        centers = named_cabin()
        for mesh in ("veh_doorpanel_FL", "veh_doorpanel_FR"):
            centers.pop(mesh)
        parts = cabin_parts()
        for part_id in ("veh_doorpanel_FL", "veh_doorpanel_FR"):
            parts.pop(part_id)
        recs = recommend(centers, parts)
        self.assertEqual(recs["veh_door_FL"]["mode"], core.MODE_MIRROR_STRUCTURAL)
        self.assertEqual(recs["veh_door_FL"]["source_id"], "veh_door_FR")

    def test_a_rear_door_card_is_not_handed(self) -> None:
        centers = named_cabin()
        centers.update({
            "veh_doorpanel_RL": (0.73, 0.80, 0.68),
            "veh_doorpanel_RR": (-0.73, 0.80, 0.68),
        })
        parts = cabin_parts()
        parts["veh_doorpanel_RL"] = part_body(
            "veh_doorpanel_RL", flexbodies=("veh_doorpanel_RL",)
        )
        parts["veh_doorpanel_RR"] = part_body(
            "veh_doorpanel_RR", flexbodies=("veh_doorpanel_RR",)
        )
        recs = recommend(centers, parts)
        self.assertNotIn("veh_doorpanel_RL", recs)
        self.assertNotIn("veh_doorpanel_RR", recs)

    def test_seats_swap_as_equivalent_parts_never_as_meshes(self) -> None:
        recs = recommend(named_cabin())
        seat = recs["veh_seat_FL"]
        self.assertEqual(seat["mode"], core.MODE_SKIP)
        self.assertTrue(seat["equivalent"])
        self.assertEqual(seat["pair_kind"], "seat")
        self.assertEqual(seat["source_id"], "veh_seat_FR")

    def test_every_mesh_swap_also_earns_an_equivalent_parts_row(self) -> None:
        recs = recommend(named_cabin())
        for mesh in ("veh_doorpanel_FL", "veh_mirror_L"):
            self.assertTrue(recs[mesh]["equivalent"], mesh)

    def test_the_driver_side_member_is_named_first(self) -> None:
        recs = recommend(named_cabin())
        self.assertEqual(recs["veh_mirror_L"]["source_id"], "veh_mirror_R")

    def test_a_symmetric_pair_outside_the_handed_families_never_swaps(self) -> None:
        centers = named_cabin()
        centers.update({
            "veh_headlight_L": (0.60, -1.95, 0.60),
            "veh_headlight_R": (-0.60, -1.95, 0.60),
        })
        parts = cabin_parts()
        parts["veh_headlight_L"] = part_body(
            "veh_headlight_L", flexbodies=("veh_headlight_L",)
        )
        parts["veh_headlight_R"] = part_body(
            "veh_headlight_R", flexbodies=("veh_headlight_R",)
        )
        recs = recommend(centers, parts)
        self.assertNotIn("veh_headlight_L", recs)

    def test_mutually_exclusive_hand_variants_never_pair(self) -> None:
        centers = named_cabin()
        centers.update({
            "veh_mirror_int_lhd": (0.10, -0.50, 1.25),
            "veh_mirror_int_rhd": (-0.10, -0.50, 1.25),
        })
        parts = cabin_parts()
        parts["veh_intmirror"] = part_body(
            "veh_mirror_L", flexbodies=("veh_mirror_int_lhd", "veh_mirror_int_rhd")
        )
        recs = recommend(centers, parts)
        self.assertNotEqual(
            recs.get("veh_mirror_int_lhd", {}).get("source_id"), "veh_mirror_int_rhd"
        )


class PlacementCheckTests(unittest.TestCase):
    def test_a_same_side_namesake_is_not_a_twin(self) -> None:
        centers = named_cabin()
        centers["veh_mirror_R"] = (0.88, -0.42, 1.00)  # both on the left
        recs = recommend(centers)
        self.assertNotEqual(
            recs["veh_mirror_L"]["mode"], core.MODE_MIRROR_STRUCTURAL
        )

    def test_a_centred_namesake_is_not_a_twin(self) -> None:
        centers = named_cabin()
        centers["veh_mirror_L"] = (0.01, -0.42, 1.00)
        centers["veh_mirror_R"] = (-0.01, -0.42, 1.00)
        recs = recommend(centers)
        self.assertNotEqual(recs.get("veh_mirror_L", {}).get("mode"), core.MODE_MIRROR_STRUCTURAL)

    def test_an_uncached_mesh_pairs_on_its_name_alone(self) -> None:
        centers = named_cabin()
        centers["veh_mirror_L"] = None
        centers["veh_mirror_R"] = None
        recs = recommend(centers)
        self.assertEqual(recs["veh_mirror_L"]["mode"], core.MODE_MIRROR_STRUCTURAL)


class OtherSettingTests(unittest.TestCase):
    def test_the_widest_mesh_on_the_steering_channel_is_the_reference(self) -> None:
        centers = named_cabin()
        centers["veh_steershaft"] = (0.40, -0.90, 0.55)
        parts = cabin_parts()
        parts["veh_steer"] = part_body(
            "veh_steer",
            props=(("steering", "veh_steer"), ("steering", "veh_steershaft")),
        )
        rows = [row for row in all_rows(centers, parts) if row["kind"] == "steering"]
        self.assertEqual(len(rows), 1)
        # Both fixtures are the same size, so the vehicle's plainer name wins.
        self.assertEqual(rows[0]["object_id"], "veh_steer")

    def test_texture_fix_is_only_offered_on_a_reflecting_row(self) -> None:
        rows = all_rows(named_cabin())
        modes = {}
        for row in rows:
            if row["kind"] == "texture":
                continue
            for key in ("object_id", "source_id"):
                if row.get(key):
                    modes[row[key]] = row["mode"]
        for row in rows:
            if row["kind"] != "texture":
                continue
            self.assertIn(
                modes.get(row["object_id"]),
                {core.MODE_MIRROR, core.MODE_MIRROR_STRUCTURAL},
                row["object_id"],
            )

    def test_a_door_card_always_earns_texture_fix(self) -> None:
        rows = all_rows(named_cabin())
        textured = {row["object_id"] for row in rows if row["kind"] == "texture"}
        self.assertIn("veh_doorpanel_FL", textured)
        self.assertIn("veh_doorpanel_FR", textured)

    def test_a_door_box_stays_put_and_a_dash_box_reflects(self) -> None:
        boxes = [
            {"trigger": "door_FL_int", "at": (0.73, -0.20, 0.90), "key": ("door_FL_int", (0.73, -0.20, 0.90)), "label": "door_FL_int"},
            {"trigger": "hazard", "at": (0.02, -0.45, 0.80), "key": ("hazard", (0.02, -0.45, 0.80)), "label": "hazard"},
        ]
        rows = {
            row["object_id"]: row
            for row in all_rows(named_cabin(), trigger_boxes=boxes)
            if row["kind"] == "trigger"
        }
        self.assertNotIn("door_FL_int", rows)
        self.assertEqual(rows["hazard"]["mode"], core.MODE_MIRROR)

    def test_a_boot_release_reflects_only_when_it_is_inside_the_cabin(self) -> None:
        """The Lexus authors both under the name "trunk"."""
        boxes = [
            {"trigger": "trunk", "at": (0.64, -0.15, 0.78), "key": ("trunk", (0.64, -0.15, 0.78)), "label": "cabin"},
            {"trigger": "trunk", "at": (-0.38, 2.45, 0.85), "key": ("trunk", (-0.38, 2.45, 0.85)), "label": "lid"},
        ]
        rows = {
            row["object_id"]: row
            for row in all_rows(named_cabin(), trigger_boxes=boxes)
            if row["kind"] == "trigger"
        }
        self.assertEqual(rows["cabin"]["mode"], core.MODE_MIRROR)
        self.assertNotIn("lid", rows)


class CostTests(unittest.TestCase):
    def test_recommending_never_reads_geometry(self) -> None:
        """Placement comes from the preview bounds already cached for the 3D
        view. Rebuilding point clouds is what the retired spatial classifier
        cost 35 s a vehicle doing."""
        context = mesh_context(named_cabin(), parts=cabin_parts())
        guarded = (
            "selected_prop_mesh_positions",
            "selected_node_positions_for_config",
            "mesh_roles_for_config",
        )
        present = [name for name in guarded if hasattr(core, name)]
        self.assertTrue(present, "nothing left to guard against")

        def refuse(name):
            return lambda *args, **kwargs: self.fail(f"{name} was called")

        with patch.multiple(core, **{name: refuse(name) for name in present}):
            tool.build_mode_recommendations(context, list(named_cabin()))

    def test_a_vehicle_with_no_jbeam_still_yields_recommendations(self) -> None:
        """A context whose parts were never indexed falls back to the name
        score for the wheel and simply finds less; it must not raise."""
        context = mesh_context(named_cabin(), camera=False, parts={})
        rows = tool.build_mode_recommendations(context, list(named_cabin()))
        self.assertTrue(any(row["mode"] == core.MODE_TRANSLATE for row in rows))


if __name__ == "__main__":
    unittest.main()
