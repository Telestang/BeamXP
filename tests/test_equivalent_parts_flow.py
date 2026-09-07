"""2026-09-07: Equivalent Parts rows are made by New and only by New.

The table used to be edited in place: New dropped a blank row and each of its
two cells armed a part pick when clicked. That made a click on the table mean
two things at once -- select this row for Remove, or start editing it -- and it
left rows in the table that named only one side. A pair is now assembled off
the table (pick a part, answer the Swap Mesh modal with its counterpart) and
lands only once both sides are known.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from beamxp.hand_drive_ui.recommendations_ui import RecommendationsUIMixin
from beamxp.hand_drive_ui.slots_workflow import SlotsWorkflowMixin


def click(x: int = 10, y: int = 20) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y)


class RowClickTests(unittest.TestCase):
    def make_app(self) -> Mock:
        app = Mock()
        app.side_pair_pick_target = None
        app._tree_body_click.return_value = True
        return app

    def test_clicking_a_row_selects_it_and_starts_no_pick(self):
        app = self.make_app()
        app.slot_tree.identify_row.return_value = "pair_0"

        self.assertEqual(SlotsWorkflowMixin._slot_click(app, click()), "break")

        app.slot_tree.selection_set.assert_called_once_with(["pair_0"])
        app.slot_tree.focus.assert_called_once_with("pair_0")
        self.assertIsNone(app.side_pair_pick_target)

    def test_clicking_below_the_last_row_selects_nothing(self):
        app = self.make_app()
        app.slot_tree.identify_row.return_value = ""

        self.assertIsNone(SlotsWorkflowMixin._slot_click(app, click()))

        app.slot_tree.selection_set.assert_not_called()


class NewButtonTests(unittest.TestCase):
    def test_new_arms_a_pick_and_adds_no_row(self):
        app = Mock(context=object())
        app.side_pair_pick_target = None

        SlotsWorkflowMixin._start_new_side_pair(app)

        self.assertEqual(app.side_pair_pick_target, {"mode": "new"})
        app._refresh_slots.assert_not_called()
        app.status_var.set.assert_called_once()

    def test_new_without_a_vehicle_says_so(self):
        app = Mock(context=None)
        app.side_pair_pick_target = None

        SlotsWorkflowMixin._start_new_side_pair(app)

        app._show_error.assert_called_once()
        self.assertIsNone(app.side_pair_pick_target)


class PartPickTests(unittest.TestCase):
    def make_app(self) -> Mock:
        app = Mock()
        app.side_pair_pick_target = {"mode": "new"}
        app._clear_side_pair_pick_target = lambda: setattr(app, "side_pair_pick_target", None)
        app.part_tree.exists.return_value = True
        app._part_row_side_ref.return_value = "etk_seat@@seat_FL"
        app._choose_and_commit_side_pair.return_value = True
        return app

    def test_a_picked_part_disarms_the_pick_and_opens_the_modal_after_the_click(self):
        app = self.make_app()

        self.assertTrue(SlotsWorkflowMixin._commit_side_pair_part_pick_from_row(app, "row_1"))

        # Not during the click: a grab taken inside a Treeview button press
        # leaves the tree believing the button is still down.
        app._choose_and_commit_side_pair.assert_not_called()
        app.after.call_args[0][1]()
        app._choose_and_commit_side_pair.assert_called_once_with("etk_seat@@seat_FL")
        self.assertIsNone(app.side_pair_pick_target)

    def test_a_part_click_with_no_pick_armed_is_left_alone(self):
        app = self.make_app()
        app.side_pair_pick_target = None

        self.assertFalse(SlotsWorkflowMixin._commit_side_pair_part_pick_from_row(app, "row_1"))

        app._choose_and_commit_side_pair.assert_not_called()

    def test_a_row_the_table_no_longer_holds_cancels(self):
        app = self.make_app()
        app.part_tree.exists.return_value = False

        self.assertFalse(SlotsWorkflowMixin._commit_side_pair_part_pick_from_row(app, "row_1"))

        app._choose_and_commit_side_pair.assert_not_called()
        self.assertIsNone(app.side_pair_pick_target)


class SuggestionTests(unittest.TestCase):
    """What the modal offers before the user touches the dropdown."""

    def make_app(self) -> Mock:
        app = Mock(context=None)
        app._name_pair_candidate = lambda object_id, candidates: (
            RecommendationsUIMixin._name_pair_candidate(app, object_id, candidates)
        )
        app._mirrored_position_ref = SlotsWorkflowMixin._mirrored_position_ref
        return app

    def suggest(self, app, subject, candidates) -> str:
        return SlotsWorkflowMixin._suggest_side_pair_counterpart(app, subject, candidates)

    def test_a_name_pair_answers_first(self):
        app = self.make_app()
        subject = {"ref": "seat_FL@@a", "mesh_id": "seat_FL", "position": (0.4, 0.2, 0.5)}
        candidates = [
            {"ref": "seat_RR@@b", "mesh_id": "seat_RR", "position": (-0.4, -0.6, 0.5)},
            {"ref": "seat_FR@@c", "mesh_id": "seat_FR", "position": (-0.4, 0.2, 0.5)},
        ]

        self.assertEqual(self.suggest(app, subject, candidates), "seat_FR@@c")

    def test_position_separates_two_instances_of_one_mesh(self):
        """The case no name can answer: one mesh fitted once per side."""
        app = self.make_app()
        subject = {"ref": "doorcard@@fl", "mesh_id": "doorcard", "position": (0.75, 0.3, 0.7)}
        candidates = [
            {"ref": "doorcard@@rl", "mesh_id": "doorcard", "position": (0.75, -0.7, 0.7)},
            {"ref": "doorcard@@fr", "mesh_id": "doorcard", "position": (-0.75, 0.3, 0.7)},
        ]

        self.assertEqual(self.suggest(app, subject, candidates), "doorcard@@fr")

    def test_nothing_on_the_far_side_suggests_nothing(self):
        app = self.make_app()
        subject = {"ref": "doorcard@@fl", "mesh_id": "doorcard", "position": (0.75, 0.3, 0.7)}
        candidates = [
            {"ref": "doorcard@@rl", "mesh_id": "doorcard", "position": (0.75, -0.7, 0.7)},
        ]

        self.assertEqual(self.suggest(app, subject, candidates), "")

    def test_a_named_pair_with_several_instances_takes_the_mirrored_one(self):
        app = self.make_app()
        subject = {"ref": "seat_FL@@a", "mesh_id": "seat_FL", "position": (0.4, 0.2, 0.5)}
        candidates = [
            {"ref": "seat_FR@@rear", "mesh_id": "seat_FR", "position": (-0.4, -0.9, 0.5)},
            {"ref": "seat_FR@@front", "mesh_id": "seat_FR", "position": (-0.4, 0.2, 0.5)},
        ]

        self.assertEqual(self.suggest(app, subject, candidates), "seat_FR@@front")


class ModalCommitTests(unittest.TestCase):
    def make_app(self, answer: str | None) -> Mock:
        app = Mock()
        app._side_pair_table_context.return_value = {
            "source_candidates": [
                {"ref": "seat_FL@@a", "mesh_id": "seat_FL", "label": "Seat FL", "position": (0.4, 0.2, 0.5)},
                {"ref": "seat_FR@@b", "mesh_id": "seat_FR", "label": "Seat FR", "position": (-0.4, 0.2, 0.5)},
            ]
        }
        app._side_pair_source_option_maps.return_value = ({"Seat FR": "seat_FR@@b"}, {})
        app._side_pair_partner_ref.return_value = ""
        app._suggest_side_pair_counterpart.return_value = "seat_FR@@b"
        app._side_pair_ref_part_id = SlotsWorkflowMixin._side_pair_ref_part_id.__get__(app)
        app._default_side_pair_kind.return_value = "Seat"
        app._choose_counterpart_modal.return_value = answer
        app._commit_side_pair.return_value = True
        return app

    def test_the_answer_becomes_a_row(self):
        app = self.make_app("seat_FR@@b")

        self.assertTrue(SlotsWorkflowMixin._choose_and_commit_side_pair(app, "seat_FL@@a"))

        pair, kwargs = app._commit_side_pair.call_args
        self.assertEqual(pair[0], {"left": "seat_FL@@a", "right": "seat_FR@@b", "kind": "seat"})
        self.assertEqual(kwargs["action"], "Added")

    def test_the_clicked_part_is_never_offered_as_its_own_counterpart(self):
        app = self.make_app("seat_FR@@b")

        SlotsWorkflowMixin._choose_and_commit_side_pair(app, "seat_FL@@a")

        offered = app._side_pair_source_option_maps.call_args[0][0]
        self.assertEqual([candidate["ref"] for candidate in offered], ["seat_FR@@b"])
        self.assertEqual(app._choose_counterpart_modal.call_args[1]["subject_value"], "seat_FL@@a")

    def test_cancelling_the_modal_writes_nothing(self):
        app = self.make_app(None)

        self.assertFalse(SlotsWorkflowMixin._choose_and_commit_side_pair(app, "seat_FL@@a"))

        app._commit_side_pair.assert_not_called()

    def test_repairing_a_part_that_already_has_a_row_reads_as_an_update(self):
        app = self.make_app("seat_FR@@b")
        app._side_pair_partner_ref.return_value = "seat_RR@@c"

        SlotsWorkflowMixin._choose_and_commit_side_pair(app, "seat_FL@@a")

        self.assertEqual(app._commit_side_pair.call_args[1]["action"], "Updated")
        self.assertEqual(app._choose_counterpart_modal.call_args[1]["initial"], "seat_RR@@c")

    def test_a_vehicle_with_one_usable_part_says_so_instead_of_opening_a_modal(self):
        app = self.make_app("seat_FR@@b")
        app._side_pair_table_context.return_value = {
            "source_candidates": [
                {"ref": "seat_FL@@a", "mesh_id": "seat_FL", "label": "Seat FL", "position": (0.4, 0.2, 0.5)},
            ]
        }

        self.assertFalse(SlotsWorkflowMixin._choose_and_commit_side_pair(app, "seat_FL@@a"))

        app._show_error.assert_called_once()
        app._choose_counterpart_modal.assert_not_called()


if __name__ == "__main__":
    unittest.main()
