"""A cloned parent renames child slots; the config has to answer the new name.

``rewrite_child_slot_defaults`` suffixes a child slot whenever it also clones
the part that slot defaults to, so a cloned door asks about
``etkc_doorpanel_L_xp_rhd`` where the stock door asked about
``etkc_doorpanel_L``. The generated config still answered under the old name,
so the renamed slot went unanswered and the engine filled it from its default
part -- which is how the etkc's drift trim, with both door card slots
deliberately empty, came out of a build wearing door cards.
"""

from __future__ import annotations

import unittest

# The implementation modules are not importable standalone -- they use names
# the hand_drive_core facade injects at import time (here, extract_slot_defs).
from beamxp import hand_drive_core as _core  # noqa: F401
from beamxp.hand_drive_parts.generation import (
    carry_answers_to_renamed_slots,
    renamed_child_slots,
)

SUFFIX = "_xp_rhd"

# Shaped like the real generated door: the card slot renamed and defaulting to
# the cloned card, the glass slot left alone because its part was not cloned.
CLONED_DOOR = """{
"etkc_door_L_xp_rhd": {
    "information": {"name": "Left Door"},
    "slotType": "etkc_door_L",
    "slots2": [
        ["type", "default", "description"],
        ["etkc_doorpanel_L_xp_rhd", ["etkc_doorpanel_L"], [], "etkc_doorpanel_L_xp_rhd", "Left Door Panel"],
        ["etkc_doorglass_L", ["etkc_doorglass_L"], [], "etkc_doorglass_L", "Glass"]
    ]
}
}"""


class RenamedChildSlotTests(unittest.TestCase):
    def test_a_renamed_child_slot_is_found(self) -> None:
        self.assertEqual(
            renamed_child_slots([CLONED_DOOR], SUFFIX), {"etkc_doorpanel_L_xp_rhd"}
        )

    def test_an_untouched_child_slot_is_not_reported(self) -> None:
        self.assertNotIn("etkc_doorglass_L", renamed_child_slots([CLONED_DOOR], SUFFIX))

    def test_the_other_hand_is_not_picked_up(self) -> None:
        self.assertEqual(renamed_child_slots([CLONED_DOOR], "_xp_lhd"), set())


class CarriedAnswerTests(unittest.TestCase):
    def carry(self, parts, slot_updates=None):
        return carry_answers_to_renamed_slots(
            parts, [CLONED_DOOR], SUFFIX, slot_updates or {}
        )

    def test_an_emptied_slot_stays_empty(self) -> None:
        # The reported bug. Without the carry the renamed slot goes unanswered
        # and its default -- the cloned door card -- fills it.
        carried = self.carry({"etkc_doorpanel_L": ""})
        self.assertEqual(carried["etkc_doorpanel_L_xp_rhd"], "")

    def test_a_chosen_part_survives_the_rename(self) -> None:
        # Not only about empty slots: a trim that fitted the race card lost
        # that choice too, and got the base card the slot defaults to.
        carried = self.carry({"etkc_doorpanel_L": "etkc_doorpanel_L_race"})
        self.assertEqual(carried["etkc_doorpanel_L_xp_rhd"], "etkc_doorpanel_L_race")

    def test_a_cloned_part_keeps_the_clone_answer(self) -> None:
        # slot_updates already answers a slot whose own part was cloned, and
        # the caller applies it after this, so the carry must stand aside.
        updates = {"etkc_doorpanel_L_xp_rhd": "etkc_doorpanel_L_xp_rhd"}
        carried = self.carry({"etkc_doorpanel_L": ""}, updates)
        carried.update(updates)
        self.assertEqual(carried["etkc_doorpanel_L_xp_rhd"], "etkc_doorpanel_L_xp_rhd")

    def test_slots_that_were_not_renamed_gain_nothing(self) -> None:
        # The glass slot keeps its name, so nothing is written for it: the
        # config must not fill up with keys naming slots that do not exist.
        carried = self.carry({"etkc_doorglass_L": "etkc_doorglass_L_lightweight"})
        self.assertEqual(carried, {"etkc_doorglass_L": "etkc_doorglass_L_lightweight"})

    def test_a_slot_the_config_never_mentioned_is_left_alone(self) -> None:
        # Nothing to carry: the stock default for the renamed slot is then the
        # right answer, exactly as it was before the clone.
        self.assertEqual(self.carry({}), {})

    def test_the_original_answer_is_left_in_place(self) -> None:
        # The old key still names a real slot on any part that was not cloned,
        # so carrying must add, never move.
        carried = self.carry({"etkc_doorpanel_L": ""})
        self.assertEqual(carried["etkc_doorpanel_L"], "")

    def test_an_existing_renamed_answer_is_not_overwritten(self) -> None:
        carried = self.carry(
            {"etkc_doorpanel_L": "", "etkc_doorpanel_L_xp_rhd": "etkc_doorpanel_L_race"}
        )
        self.assertEqual(carried["etkc_doorpanel_L_xp_rhd"], "etkc_doorpanel_L_race")


if __name__ == "__main__":
    unittest.main()
