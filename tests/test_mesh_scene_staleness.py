"""A preview build that never reaches the screen must not be recorded as shown.

``mesh_scene_hash`` is written when a build is dispatched, so a second edit
arriving mid-build does not queue an identical rebuild. If that build's result
is then dropped, the hash was left claiming the screen shows a state it never
showed -- and because the snapshot covers the part selection as well as the
conversion, the window can return to that state and be refused a rebuild for
good. That is how an Equivalent Parts row set at the wrong moment stayed out
of the preview until the app was restarted.
"""

from __future__ import annotations

import unittest

from beamxp.hand_drive_ui.build_and_preview import BuildAndPreviewMixin


class FakeFuture:
    def __init__(self, scene: object = "scene", error: Exception | None = None) -> None:
        self._scene = scene
        self._error = error

    def result(self) -> object:
        if self._error is not None:
            raise self._error
        return self._scene


class FakeViewer:
    supports_scene = True

    def __init__(self) -> None:
        self.shown: list[object] = []
        self.messages: list[str] = []

    def show_scene(self, scene, **_options) -> None:
        self.shown.append(scene)

    def set_message(self, message: str) -> None:
        self.messages.append(message)


class Harness(BuildAndPreviewMixin):
    """The scene scheduler over a stubbed dispatch, so builds are countable."""

    def __init__(self, snapshot: str = "A") -> None:
        self.context = object()
        self.viewer = FakeViewer()
        self.viewer_supports_scene = True
        self.snapshot = snapshot
        self.mesh_scene_hash = None
        self.mesh_scene_seq = 0
        self.mesh_scene_running = False
        self.mesh_scene_pending = False
        self.mesh_scene_after = None
        self.mesh_scene_reset_pending = False
        self.dispatched: list[str] = []

    # The snapshot is the thing under test; make it settable.
    def _mesh_scene_snapshot(self) -> str:
        return self.snapshot

    def _mesh_scene_config(self) -> str:
        return "trim"

    def _start_mesh_scene(self) -> None:
        """Stands in for the real dispatch, keeping its bookkeeping exactly."""
        self.mesh_scene_after = None
        snapshot = self._mesh_scene_snapshot()
        if snapshot == self.mesh_scene_hash:
            return
        self.mesh_scene_hash = snapshot
        self.mesh_scene_seq += 1
        self.mesh_scene_running = True
        self.dispatched.append(snapshot)

    def _refresh_preview_dependent_part_cells(self) -> None: ...
    def _refresh_viewer(self, *, reset: bool = False) -> None: ...

    def complete(self, snapshot: str, *, error: Exception | None = None) -> None:
        self._handle_mesh_scene_done(
            (self.mesh_scene_seq, snapshot, FakeFuture(error=error))
        )


class MeshSceneStalenessTests(unittest.TestCase):
    def test_an_applied_build_is_recorded_and_not_repeated(self) -> None:
        app = Harness("A")
        app._schedule_mesh_scene(immediate=True)
        app.complete("A")
        self.assertEqual(app.dispatched, ["A"])
        self.assertEqual(app.viewer.shown, ["scene"])
        app._schedule_mesh_scene(immediate=True)
        self.assertEqual(app.dispatched, ["A"])  # nothing changed, no rebuild

    def test_a_state_the_screen_never_showed_is_rebuilt_when_asked_again(self) -> None:
        # The reported bug, in sequence: a build for B is dispatched, the
        # window moves to C mid-build without a request landing, B comes back
        # and is dropped -- and then the window returns to B. Before the fix
        # the hash still read B here, so the rebuild was skipped as a no-op
        # and the preview kept showing whatever was on screen before.
        app = Harness("B")
        app._schedule_mesh_scene(immediate=True)
        self.assertEqual(app.dispatched, ["B"])
        app.snapshot = "C"
        app.complete("B")
        self.assertEqual(app.viewer.shown, [], "a superseded build must not be shown")

        app.snapshot = "B"
        app._schedule_mesh_scene(immediate=True)
        # Whatever is in flight, the window must converge on B: run the queue
        # out and check the screen ends up showing a build made for B.
        for _ in range(4):
            if not app.mesh_scene_running:
                break
            app.complete(app.dispatched[-1])
        self.assertEqual(app.dispatched[-1], "B")
        self.assertEqual(app.viewer.shown, ["scene"])
        self.assertEqual(app.mesh_scene_hash, "B")

    def test_a_dropped_build_immediately_rebuilds_for_what_is_wanted_now(self) -> None:
        app = Harness("B")
        app._schedule_mesh_scene(immediate=True)
        app.snapshot = "C"
        app.complete("B")
        self.assertEqual(app.dispatched, ["B", "C"])

    def test_a_failed_build_can_be_retried(self) -> None:
        # A hash left behind by a build that raised would refuse the retry.
        app = Harness("A")
        app._schedule_mesh_scene(immediate=True)
        app.complete("A", error=RuntimeError("boom"))
        self.assertEqual(app.viewer.messages[-1], "preview failed: boom")
        app._schedule_mesh_scene(immediate=True)
        self.assertEqual(app.dispatched, ["A", "A"])

    def test_an_edit_arriving_mid_build_still_rebuilds_afterwards(self) -> None:
        # The path that already worked, which the fix must not disturb.
        app = Harness("A")
        app._schedule_mesh_scene(immediate=True)
        app.snapshot = "B"
        app._schedule_mesh_scene()  # running, so this only marks it pending
        self.assertTrue(app.mesh_scene_pending)
        app.complete("A")
        self.assertEqual(app.dispatched, ["A", "B"])

    def test_the_box_viewer_fallback_does_not_spin(self) -> None:
        # With no scene-capable viewer nothing can be applied, so a discard
        # must not turn into an endless redispatch.
        app = Harness("A")
        app._schedule_mesh_scene(immediate=True)
        app.viewer_supports_scene = False
        app.complete("A")
        self.assertEqual(app.dispatched, ["A"])
        self.assertIsNone(app.mesh_scene_hash)


if __name__ == "__main__":
    unittest.main()
