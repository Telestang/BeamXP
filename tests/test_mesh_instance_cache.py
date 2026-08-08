"""Resolving a trim's mesh instances happens once per trim, not once per ask.

Each resolve re-parses that trim's whole part tree. The worker walks every
trim to build the instance numbering, and the UI thread then wanted the same
walk again to label two Equivalent Parts rows -- six seconds of it. The worker
gets a *copy* of the conversion, so the two only meet if the cache is keyed on
what the resolution depends on rather than on the dict it was handed.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from beamxp.core.models import MeshTransformInstance
from beamxp.hand_drive_ui.parts_workflow import PartsWorkflowMixin


def instance(mesh: str) -> MeshTransformInstance:
    return MeshTransformInstance(
        instance_id=f"{mesh}@@/slot/",
        mesh_id=mesh,
        part_id="slot",
        slot_id="slot",
        slot_path="/slot/",
        position=(0.0, 0.0, 0.0),
        count_for_mesh=1,
        ordinal_for_mesh=1,
    )


def context() -> SimpleNamespace:
    return SimpleNamespace(variants={"a": object()}, objects={"mesh": object()})


class MeshInstanceCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        PartsWorkflowMixin._clear_mesh_instance_cache()
        self.addCleanup(PartsWorkflowMixin._clear_mesh_instance_cache)
        self.context = context()
        self.resolves = 0

        def resolve(_context, _conversion, _config):
            self.resolves += 1
            return [instance("mesh")]

        patcher = patch.object(
            PartsWorkflowMixin, "_resolve_mesh_transform_instances", staticmethod(resolve)
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def ask(self, conversion, config: str = "a"):
        return PartsWorkflowMixin._mesh_transform_instances_for_config_data(
            self.context, conversion, config
        )

    def test_the_same_trim_is_resolved_once(self) -> None:
        first = self.ask({})
        second = self.ask({})
        self.assertEqual(self.resolves, 1)
        self.assertEqual(first, second)

    def test_a_copy_of_the_conversion_reads_back_what_the_worker_resolved(self) -> None:
        """The worker's copy and the live dict must land on the same entry."""
        live = {"sidePairs": [{"left": "a", "right": "b"}], "parts": {"mesh": {}}}
        worker_copy = {"sidePairs": [{"left": "a", "right": "b"}], "parts": {"mesh": {}}}
        self.assertIsNot(live, worker_copy)
        self.ask(worker_copy)
        self.ask(live)
        self.assertEqual(self.resolves, 1)

    def test_a_transform_edit_does_not_throw_the_resolution_away(self) -> None:
        # Modes and offsets never move a part between slots.
        self.ask({"parts": {"mesh": {"mode": "skip"}}})
        self.ask({"parts": {"mesh": {"mode": "mirror"}}})
        self.assertEqual(self.resolves, 1)

    def test_a_pairing_edit_does(self) -> None:
        self.ask({})
        self.ask({"sidePairs": [{"left": "a", "right": "b"}]})
        self.assertEqual(self.resolves, 2)

    def test_each_trim_gets_its_own_entry(self) -> None:
        self.context.variants["b"] = object()
        self.ask({}, "a")
        self.ask({}, "b")
        self.ask({}, "a")
        self.assertEqual(self.resolves, 2)

    def test_the_cache_stays_bounded(self) -> None:
        for index in range(PartsWorkflowMixin.MESH_INSTANCE_CACHE_LIMIT + 20):
            self.ask({}, f"config_{index}")
        self.assertLessEqual(
            len(PartsWorkflowMixin._mesh_instance_cache),
            PartsWorkflowMixin.MESH_INSTANCE_CACHE_LIMIT,
        )

    def test_a_freed_context_cannot_have_its_address_reused_under_an_entry(self) -> None:
        """The entry holds its context, so a later one never inherits its key."""
        first_id = id(self.context)
        self.ask({})
        self.context = context()
        for _ in range(200):  # plenty of chances to land on the freed address
            if id(self.context) == first_id:
                self.fail("the cached context was collected and its address reused")
            self.context = context()
        self.ask({})
        self.assertEqual(self.resolves, 2)


class NumberForRefTests(unittest.TestCase):
    """The ref -> ordinal lookup that replaced a walk of every trim.

    The numbering pass records each ref's number keys in trim order; the
    lookup takes the first that has been given an ordinal, which is the trim
    the walk would have stopped at.
    """

    def setUp(self) -> None:
        self.app = SimpleNamespace(
            context=SimpleNamespace(variants={"a": object()}),
            mesh_instance_numbering_cache={
                "seat": {"slot:seat_L|x:-": 1, "slot:seat_R|x:+": 2},
            },
            mesh_instance_keys_cache={
                "seat@@/seat_L/": ["slot:seat_L|x:-"],
                "seat@@/seat_R/": ["slot:seat_R|x:+"],
                # a ref whose first trim gives a place that was never numbered
                "seat@@/seat_odd/": ["slot:seat_odd|x:0", "slot:seat_R|x:+"],
                "seat@@/nowhere/": ["slot:gone|x:0"],
            },
        )
        self.app._vehicle_mesh_instance_numbering = (
            lambda: self.app.mesh_instance_numbering_cache
        )

    def number(self, ref: str):
        return PartsWorkflowMixin._mesh_instance_number_for_ref(self.app, ref)

    def test_a_ref_gets_its_place_and_the_count(self) -> None:
        self.assertEqual(self.number("seat@@/seat_L/"), (1, 2))
        self.assertEqual(self.number("seat@@/seat_R/"), (2, 2))

    def test_an_unnumbered_place_falls_through_to_the_next_trim(self) -> None:
        self.assertEqual(self.number("seat@@/seat_odd/"), (2, 2))

    def test_a_ref_with_no_numbered_place_has_no_number(self) -> None:
        self.assertIsNone(self.number("seat@@/nowhere/"))

    def test_a_mesh_that_appears_once_is_never_numbered(self) -> None:
        self.assertIsNone(self.number("wheel@@/front_L/"))

    def test_a_ref_the_numbering_never_saw_has_no_number(self) -> None:
        self.assertIsNone(self.number("seat@@/unseen/"))


if __name__ == "__main__":
    unittest.main()
