"""Recommend Transforms: what each mesh name and placement earns.

The recommender reads names and the mesh centres already cached for the
preview -- nothing else. Fixtures are therefore just names with positions;
`test_recommending_never_reads_geometry` guards that no spatial work creeps
back in.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from beamxp import hand_drive_core as core
from beamxp import hand_drive_tool as tool
from tests.cabin_fixtures import mesh_context


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


def recommend(centers: dict[str, tuple[float, float, float] | None]) -> dict[str, dict]:
    context = mesh_context(centers)
    return {
        row["object_id"]: row
        for row in tool.build_mode_recommendations(context, list(centers))
    }


class DriverControlTests(unittest.TestCase):
    def test_steering_wheel_translates_from_its_anchor_score(self) -> None:
        recs = recommend(named_cabin())
        self.assertEqual(recs["veh_steer"]["mode"], core.MODE_TRANSLATE)
        self.assertEqual(recs["veh_steer"]["reason"], "steering wheel")

    def test_instrument_and_pedal_names_translate(self) -> None:
        centers = named_cabin()
        centers.update({
            "veh_gauges": (0.40, -0.52, 0.94),
            "veh_needle_speedo": (0.40, -0.52, 0.94),
            "veh_gaspedal": (0.40, -0.80, 0.28),
            "veh_wiperstalk": (0.34, -0.44, 0.92),
            "veh_paddles": (0.40, -0.40, 0.92),
        })
        recs = recommend(centers)
        for object_id in (
            "veh_gauges", "veh_needle_speedo", "veh_gaspedal",
            "veh_wiperstalk", "veh_paddles",
        ):
            self.assertEqual(recs[object_id]["mode"], core.MODE_TRANSLATE, object_id)
            self.assertFalse(recs[object_id]["equivalent"], object_id)

    def test_pedalbox_footplate_translates_but_a_standalone_one_mirrors(self) -> None:
        centers = named_cabin()
        centers["veh_pedalbox_footplate"] = (0.40, -0.80, 0.35)
        centers["veh_race_footplate"] = (-0.40, -0.80, 0.35)
        recs = recommend(centers)
        self.assertEqual(recs["veh_pedalbox_footplate"]["mode"], core.MODE_TRANSLATE)
        self.assertEqual(recs["veh_race_footplate"]["mode"], core.MODE_MIRROR)

    def test_cluster_screen_translates_and_the_windscreen_is_left_alone(self) -> None:
        centers = named_cabin()
        centers["veh_gauges_screen"] = (0.40, -0.52, 0.94)
        centers["veh_screen"] = (0.02, -0.47, 0.95)
        recs = recommend(centers)
        self.assertEqual(recs["veh_gauges_screen"]["mode"], core.MODE_TRANSLATE)
        self.assertEqual(recs["veh_screen"]["mode"], core.MODE_MIRROR)
        self.assertNotIn("textureFlip", recs["veh_screen"])
        self.assertNotIn("veh_windscreen", recs)


class CabinFurnitureTests(unittest.TestCase):
    def test_dashboard_console_and_shifter_mirror(self) -> None:
        centers = named_cabin()
        centers["veh_console"] = (0.0, 0.10, 0.50)
        centers["veh_shifter_knob"] = (0.10, -0.35, 0.62)
        recs = recommend(centers)
        for object_id in ("veh_dash", "veh_console", "veh_shifter_knob"):
            self.assertEqual(recs[object_id]["mode"], core.MODE_MIRROR, object_id)
            self.assertEqual(recs[object_id]["reason"], "asymmetric interior name")

    def test_symmetric_cabin_spanning_names_get_nothing(self) -> None:
        centers = named_cabin()
        centers["veh_sunvisor"] = (0.0, -0.11, 1.30)
        recs = recommend(centers)
        for object_id in (
            "veh_headliner", "veh_floor", "veh_sunvisor", "veh_windscreen",
        ):
            self.assertNotIn(object_id, recs)

    def test_unrecognised_names_are_left_alone_rather_than_guessed_at(self) -> None:
        centers = named_cabin()
        centers["veh_exhaust_R"] = (-0.30, 1.80, 0.25)
        centers["veh_driveshaft"] = (0.0, 0.40, 0.15)
        recs = recommend(centers)
        self.assertNotIn("veh_exhaust_R", recs)
        self.assertNotIn("veh_driveshaft", recs)
        self.assertNotIn("veh_door_FL", recs)  # the skin, not its card

    def test_steering_column_split_is_offered_at_low_confidence(self) -> None:
        centers = named_cabin()
        centers["veh_steering_column_top"] = (0.40, -0.62, 0.72)
        centers["veh_steering_column"] = (0.37, -0.80, 0.48)
        recs = recommend(centers)
        self.assertEqual(recs["veh_steering_column_top"]["mode"], core.MODE_TRANSLATE)
        self.assertEqual(recs["veh_steering_column"]["mode"], core.MODE_MIRROR)
        for object_id in ("veh_steering_column_top", "veh_steering_column"):
            self.assertEqual(recs[object_id]["confidence"], "low", object_id)


class HandedFamilyTests(unittest.TestCase):
    def test_door_cards_and_wing_mirrors_swap_meshes(self) -> None:
        # The cross-swap IS the conversion for these two: their parts are
        # slot-locked, so an Equivalent Parts row alongside it would be a
        # second mechanism aimed at the same meshes -- and the row is written
        # against the mesh, which the whole door also declares.
        recs = recommend(named_cabin())
        card = recs["veh_doorpanel_FL"]
        self.assertEqual(card["kind"], "pair")
        self.assertEqual(card["mode"], core.MODE_MIRROR_STRUCTURAL)
        self.assertEqual(card["source_id"], "veh_doorpanel_FR")
        self.assertEqual(card["pair_kind"], "")
        self.assertFalse(card["equivalent"])

        wing = recs["veh_mirror_L"]
        self.assertEqual(wing["mode"], core.MODE_MIRROR_STRUCTURAL)
        self.assertEqual(wing["source_id"], "veh_mirror_R")
        self.assertEqual(wing["pair_kind"], "")
        self.assertFalse(wing["equivalent"])

    def test_rear_door_cards_pair_as_readily_as_front_ones(self) -> None:
        centers = named_cabin()
        centers["veh_doorpanel_RL"] = (0.73, 0.77, 0.64)
        centers["veh_doorpanel_RR"] = (-0.73, 0.77, 0.64)
        recs = recommend(centers)
        self.assertEqual(recs["veh_doorpanel_RL"]["source_id"], "veh_doorpanel_RR")
        self.assertEqual(recs["veh_doorpanel_FL"]["source_id"], "veh_doorpanel_FR")

    def test_seats_swap_as_equivalent_parts_never_as_meshes(self) -> None:
        seat = recommend(named_cabin())["veh_seat_FL"]
        self.assertEqual(seat["kind"], "equivalent")
        self.assertEqual(seat["mode"], core.MODE_SKIP)
        self.assertEqual(seat["source_id"], "veh_seat_FR")
        self.assertEqual(seat["pair_kind"], "seat")
        self.assertTrue(seat["equivalent"])

    def test_the_driver_side_member_is_named_first(self) -> None:
        recs = recommend(named_cabin())
        self.assertIn("veh_doorpanel_FL", recs)
        self.assertNotIn("veh_doorpanel_FR", recs)

        # The same cabin with the wheel on the right: the driver's card is
        # now the one named _FR, because +x is the left of the vehicle.
        right_hand = named_cabin()
        right_hand["veh_steer"] = (-0.40, -0.30, 0.95)
        rhd_recs = {
            row["object_id"]: row
            for row in tool.build_mode_recommendations(
                mesh_context(right_hand), list(right_hand)
            )
        }
        self.assertIn("veh_doorpanel_FR", rhd_recs)
        self.assertNotIn("veh_doorpanel_FL", rhd_recs)

    def test_a_one_sided_seat_base_mirrors_instead(self) -> None:
        centers = named_cabin()
        centers["racingseat_base"] = (0.42, 0.15, 0.30)
        recs = recommend(centers)
        self.assertEqual(recs["racingseat_base"]["mode"], core.MODE_MIRROR)
        self.assertEqual(recs["racingseat_base"]["source_id"], "")
        self.assertFalse(recs["racingseat_base"]["equivalent"])

    def test_a_rear_bench_is_not_a_handed_part(self) -> None:
        # veh_seats_R: R means rear, not right, and a bench spans the cabin
        centers = named_cabin()
        centers["veh_seats_R"] = (0.0, 1.05, 0.75)
        self.assertNotIn("veh_seats_R", recommend(centers))

    def test_mutually_exclusive_hand_variants_never_pair(self) -> None:
        # bx ships whole cabins per hand: _lhd/_rhd are alternatives, not
        # two halves of one part, so the "_l" inside "_lhd" must not match.
        centers = named_cabin()
        centers["veh_intmirror_lhd"] = (0.12, -0.30, 1.24)
        centers["veh_intmirror_rhd"] = (-0.12, -0.30, 1.24)
        recs = recommend(centers)
        for object_id in ("veh_intmirror_lhd", "veh_intmirror_rhd"):
            self.assertEqual(recs[object_id]["mode"], core.MODE_MIRROR, object_id)
            self.assertEqual(recs[object_id]["source_id"], "", object_id)

    def test_a_side_token_beside_a_hand_token_still_pairs(self) -> None:
        centers = named_cabin()
        centers["veh_mirror_L_rhd"] = (0.90, -0.42, 1.00)
        centers["veh_mirror_R_rhd"] = (-0.90, -0.42, 1.00)
        recs = recommend(centers)
        pair = recs.get("veh_mirror_L_rhd") or recs.get("veh_mirror_R_rhd")
        self.assertEqual(
            {pair["object_id"], pair["source_id"]},
            {"veh_mirror_L_rhd", "veh_mirror_R_rhd"},
        )


class PlacementCheckTests(unittest.TestCase):
    def test_a_same_side_namesake_is_not_a_twin(self) -> None:
        centers = named_cabin()
        centers["veh_doorpanel_RL"] = (0.73, 0.77, 0.64)
        centers["veh_doorpanel_RR"] = (0.71, 0.77, 0.64)  # both on the left
        recs = recommend(centers)
        for object_id in ("veh_doorpanel_RL", "veh_doorpanel_RR"):
            self.assertEqual(recs[object_id]["mode"], core.MODE_MIRROR, object_id)
            self.assertEqual(recs[object_id]["source_id"], "", object_id)

    def test_a_centred_namesake_is_not_a_twin(self) -> None:
        centers = named_cabin()
        centers["veh_doorpanel_RL"] = (0.01, 0.77, 0.64)
        centers["veh_doorpanel_RR"] = (-0.01, 0.77, 0.64)
        recs = recommend(centers)
        self.assertEqual(recs["veh_doorpanel_RL"]["source_id"], "")

    def test_an_uncached_mesh_pairs_on_its_name_alone(self) -> None:
        # Over half a real vehicle's DAE nodes sit at the origin, so a mesh
        # the preview never cached must not lose its pairing to the check.
        centers = named_cabin()
        centers["veh_doorpanel_RL"] = None
        centers["veh_doorpanel_RR"] = None
        recs = recommend(centers)
        pair = recs.get("veh_doorpanel_RL") or recs.get("veh_doorpanel_RR")
        self.assertEqual(pair["mode"], core.MODE_MIRROR_STRUCTURAL)
        self.assertEqual(
            {pair["object_id"], pair["source_id"]},
            {"veh_doorpanel_RL", "veh_doorpanel_RR"},
        )


class CostTests(unittest.TestCase):
    def test_recommending_never_reads_geometry(self) -> None:
        """No point clouds, no ray casts, no driver frame: names and centres."""
        centers = named_cabin()
        context = mesh_context(centers)

        def forbidden(*_args, **_kwargs):
            raise AssertionError("the recommender must not do spatial work")

        with patch.multiple(
            core,
            visibility_scan=forbidden,
            surface_visibility_stats_batch=forbidden,
            reflected_orphan_stats=forbidden,
            full_vertex_clouds_for_ids=forbidden,
            full_surface_triangles_for_ids=forbidden,
            driver_frame_for_context=forbidden,
        ):
            recommendations = tool.build_mode_recommendations(context, list(centers))

        self.assertTrue(recommendations)

    def test_no_camera_and_no_wheel_still_yields_recommendations(self) -> None:
        centers = {
            object_id: center
            for object_id, center in named_cabin().items()
            if object_id != "veh_steer"
        }
        context = mesh_context(centers, camera=False)
        recommendations = tool.build_mode_recommendations(context, list(centers))
        self.assertTrue(any(row["object_id"] == "veh_dash" for row in recommendations))


if __name__ == "__main__":
    unittest.main()
