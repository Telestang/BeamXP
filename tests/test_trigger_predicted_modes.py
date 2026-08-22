"""The Triggers table states the transform a box will get, chosen or not.

An unanswered row used to read "(auto: Move)", and a box the attribution
ladder claimed no owner for read "(auto: unattributed)" -- a transform that
does not exist, named after the reason for it rather than the outcome. Rows
now read as the plain transform the build will apply, and a box with no owner
reads Skip, which is what the build does with it.
"""

from __future__ import annotations

import unittest

from beamxp import hand_drive_core as core
from beamxp.hand_drive_ui.triggers_workflow import (
    TRIGGER_MODE_LABELS,
    TriggersWorkflowMixin,
)


class Harness(TriggersWorkflowMixin):
    """The label and prediction helpers, with no table behind them."""


def box(ref: str = "node_a", *, twinned: bool = False) -> dict[str, object]:
    return {"ref": ref, "twinned": twinned, "key": ("hood_release", (0.0, 0.0, 0.0))}


def row(mode: object, auto_action: str) -> dict[str, object]:
    return {**box(), "mode": mode, "auto_action": auto_action}


class PredictedActionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = Harness()

    def test_an_owned_box_takes_its_owners_transform(self) -> None:
        transforms = {"node_a": (core.MODE_MIRROR, "some_mesh")}
        self.assertEqual(self.app._auto_action(box(), transforms), core.MODE_MIRROR)

    def test_a_box_with_no_owner_is_predicted_to_skip(self) -> None:
        # Previously an empty string, surfaced as "unattributed". The build
        # finds no transforming owner and leaves the box where it was, which
        # is Skip by any other name.
        self.assertEqual(self.app._auto_action(box(), {}), core.MODE_SKIP)

    def test_a_twinned_pair_is_predicted_to_skip(self) -> None:
        # Decided before attribution, so the owner must not get a vote.
        transforms = {"node_a": (core.MODE_MIRROR, "some_mesh")}
        self.assertEqual(self.app._auto_action(box(twinned=True), transforms), core.MODE_SKIP)


class TransformCellTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = Harness()

    def test_a_predicted_row_reads_as_a_plain_transform(self) -> None:
        self.assertEqual(self.app._trigger_mode_label(row(None, core.MODE_TRANSLATE)), "Move")
        self.assertEqual(self.app._trigger_mode_label(row(None, core.MODE_MIRROR)), "Mirror")

    def test_no_row_announces_itself_as_a_guess(self) -> None:
        for action in (core.MODE_SKIP, core.MODE_TRANSLATE, core.MODE_MIRROR):
            label = self.app._trigger_mode_label(row(None, action))
            self.assertNotIn("auto", label.lower())
            self.assertNotIn("unattributed", label.lower())

    def test_an_unowned_row_reads_skip(self) -> None:
        self.assertEqual(self.app._trigger_mode_label(row(None, core.MODE_SKIP)), "Skip")

    def test_a_predicted_row_is_indistinguishable_from_a_chosen_one(self) -> None:
        # The whole point: the table states the transform, not its provenance.
        chosen = self.app._trigger_mode_label(row(core.MODE_MIRROR, core.MODE_SKIP))
        predicted = self.app._trigger_mode_label(row(None, core.MODE_MIRROR))
        self.assertEqual(chosen, predicted)

    def test_a_chosen_transform_wins_over_the_prediction(self) -> None:
        self.assertEqual(self.app._trigger_mode_label(row(core.MODE_SKIP, core.MODE_MIRROR)), "Skip")

    def test_a_predicted_mirror_move_keeps_its_own_name(self) -> None:
        # The ladder can return this even though the dropdown does not offer
        # it, so it must not be flattened into one of the three.
        self.assertEqual(
            self.app._trigger_mode_label(row(None, core.MODE_MIRROR_POSITION)), "Mirror Move"
        )


class EffectiveModeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = Harness()

    def test_the_prediction_stands_in_for_an_unanswered_row(self) -> None:
        self.assertEqual(
            self.app._effective_trigger_mode(row(None, core.MODE_TRANSLATE)), core.MODE_TRANSLATE
        )

    def test_an_answered_row_uses_the_answer(self) -> None:
        self.assertEqual(
            self.app._effective_trigger_mode(row(core.MODE_SKIP, core.MODE_MIRROR)), core.MODE_SKIP
        )

    def test_a_row_with_nothing_at_all_falls_back_to_skip(self) -> None:
        self.assertEqual(self.app._effective_trigger_mode({"mode": None}), core.MODE_SKIP)

    def test_the_dropdown_opens_on_what_the_cell_shows(self) -> None:
        # What _trigger_click preselects; a predicted Mirror must not open the
        # list on a blank selection.
        for action in (core.MODE_SKIP, core.MODE_TRANSLATE, core.MODE_MIRROR):
            effective = self.app._effective_trigger_mode(row(None, action))
            self.assertEqual(
                TRIGGER_MODE_LABELS.get(effective), self.app._trigger_mode_label(row(None, action))
            )


if __name__ == "__main__":
    unittest.main()
