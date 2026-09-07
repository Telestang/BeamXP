"""How the busy bar moves while a build runs.

It used to move back and forth. Four separate reasons, all of them fixed
here and each with a test below:

* every one of the eight stages set the bar's maximum to its own step count
  and its value to its own position, so the bar filled and snapped back to
  empty eight times over a build;
* inside the texture stage two phases counted different things -- textures
  for the UV masks, groups for the corrections -- and drove the bar in turn;
* a stage that reported no counts flipped the bar to indeterminate, and every
  message during it called ``start`` again and reset the block to the left
  edge, which twitched rather than swept;
* the corrections were reported when the consuming loop reached them rather
  than when they finished, so twenty-five of twenty-eight arrived in the same
  instant and the bar crossed a tenth of its width between two frames.

Timings quoted here were measured off recorded lc500 builds, not guessed.
"""

from __future__ import annotations

import unittest
from typing import ClassVar

from beamxp.hand_drive_ui.build_and_preview import (
    PROGRESS_RESOLUTION,
    BuildAndPreviewMixin,
)
from beamxp.hand_drive_ui.worker_handlers import STAGE_WEIGHTS, WorkerHandlersMixin


class FakeBar:
    def __init__(self) -> None:
        self.mode = "indeterminate"
        self.maximum = 100
        self.value = 0
        self.starts = 0
        self.stops = 0
        self.shown = True

    def configure(self, **kwargs) -> None:
        self.mode = kwargs.get("mode", self.mode)
        self.maximum = kwargs.get("maximum", self.maximum)
        if "value" in kwargs:
            self.value = kwargs["value"]

    def start(self, _interval: int = 50) -> None:
        self.starts += 1

    def stop(self) -> None:
        self.stops += 1

    def grid(self) -> None:
        self.shown = True

    def grid_remove(self) -> None:
        self.shown = False

    @property
    def fraction(self) -> float:
        return self.value / self.maximum if self.maximum else 0.0


class FakeApp(BuildAndPreviewMixin, WorkerHandlersMixin):
    """The two mixins that own the bar, with Tk's timer loop stood in for."""

    worker_running = False
    preview_output_to_config: ClassVar[dict[str, str]] = {}

    def __init__(self) -> None:
        self.busy_progress = FakeBar()
        self.timers: dict[str, object] = {}
        self.cancelled: list[str] = []
        self._handle = 0
        self.status_var = type("Var", (), {"set": lambda self, value: None})()

    def after(self, _ms, callback):
        self._handle += 1
        handle = f"after{self._handle}"
        self.timers[handle] = callback
        return handle

    def after_cancel(self, handle) -> None:
        self.cancelled.append(handle)
        self.timers.pop(handle, None)

    def _refresh_vehicle_control_state(self) -> None:
        pass

    # -- helpers --
    def tick(self, times: int = 1) -> None:
        """Run the pending easing callback, as Tk's timer loop would."""
        for _ in range(times):
            if not self.timers:
                return
            handle = next(iter(self.timers))
            callback = self.timers.pop(handle)
            callback()

    def settle(self, limit: int = 400) -> None:
        for _ in range(limit):
            if not self.timers:
                return
            self.tick()

    def progress(self, stage: int, phase: str, current=None, total=None) -> None:
        self._handle_build_progress({
            "stage": stage,
            "stage_total": len(STAGE_WEIGHTS),
            "phase": phase,
            "message": phase,
            "current": current,
            "total": total,
        })


class ProgressDirectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = FakeApp()
        self.app._set_busy(True)

    def test_the_bar_never_goes_backwards_across_stages(self) -> None:
        seen = []
        for stage in range(1, len(STAGE_WEIGHTS) + 1):
            for current in (0, 1):
                self.app.progress(stage, f"phase{stage}", current=current, total=1)
                self.app.settle()
                seen.append(self.app.busy_progress.fraction)
        self.assertEqual(seen, sorted(seen))
        self.assertAlmostEqual(seen[-1], 1.0, places=3)

    def test_a_stage_reporting_nothing_leaves_the_bar_where_it_was(self) -> None:
        self.app.progress(4, "generate_daes", current=1, total=1)
        self.app.settle()
        before = self.app.busy_progress.fraction
        self.app.progress(5, "texture_correction")
        self.app.settle()
        self.assertEqual(self.app.busy_progress.fraction, before)
        self.assertEqual(self.app.busy_progress.mode, "determinate")

    def test_the_two_texture_phases_hand_over_instead_of_fighting(self) -> None:
        """Masks count 34 textures, corrections count 28 groups. Given the
        whole stage each would run it end to end and the next would rewind."""
        self.app.progress(5, "build_domain_masks", current=34, total=34)
        self.app.settle()
        after_masks = self.app.busy_progress.fraction
        self.app.progress(5, "correct_textures", current=1, total=28)
        self.app.settle()
        self.assertGreaterEqual(self.app.busy_progress.fraction, after_masks)
        self.app.progress(5, "correct_textures", current=28, total=28)
        self.app.settle()
        self.assertGreater(self.app.busy_progress.fraction, after_masks)


class StageWeightTests(unittest.TestCase):
    def test_texture_correction_owns_most_of_the_bar(self) -> None:
        """It is 86% of a measured build; eight equal stages gave it an
        eighth, and the bar barely moved for most of a run."""
        app = FakeApp()
        app._set_busy(True)
        app.progress(4, "generate_daes", current=1, total=1)
        app.settle()
        before = app.busy_progress.fraction
        app.progress(5, "correct_textures", current=28, total=28)
        app.settle()
        self.assertGreater(app.busy_progress.fraction - before, 0.5)

    def test_an_unexpected_stage_count_falls_back_to_equal_shares(self) -> None:
        app = FakeApp()
        app._set_busy(True)
        app._handle_build_progress({
            "stage": 1, "stage_total": 4, "phase": "x", "current": 1, "total": 1,
        })
        app.settle()
        self.assertAlmostEqual(app.busy_progress.fraction, 0.25, places=3)


class EasingTests(unittest.TestCase):
    def test_a_large_jump_is_walked_rather_than_taken_in_one_frame(self) -> None:
        app = FakeApp()
        app._set_busy(True)
        app.progress(5, "correct_textures", current=28, total=28)
        first = app.busy_progress.fraction
        app.tick()
        second = app.busy_progress.fraction
        self.assertLess(second, 0.5)
        self.assertGreater(second, first)
        app.settle()
        self.assertGreater(app.busy_progress.fraction, 0.5)

    def test_the_walk_stops_when_it_arrives(self) -> None:
        app = FakeApp()
        app._set_busy(True)
        app.progress(2, "prepare_output", current=1, total=1)
        app.settle()
        self.assertEqual(app.timers, {})

    def test_going_idle_cancels_the_walk(self) -> None:
        app = FakeApp()
        app._set_busy(True)
        app.progress(5, "correct_textures", current=28, total=28)
        self.assertTrue(app.timers)
        app._set_busy(False)
        self.assertEqual(app.timers, {})
        self.assertFalse(app.busy_progress.shown)


class SweepTests(unittest.TestCase):
    def test_the_sweep_is_started_once_however_often_it_is_asked_for(self) -> None:
        """Restarting it, and resetting the value to zero, dragged the block
        back to the left edge on every message."""
        app = FakeApp()
        app._set_busy(True)
        self.assertEqual(app.busy_progress.starts, 1)
        for _ in range(5):
            app._sweep_progress()
        self.assertEqual(app.busy_progress.starts, 1)

    def test_a_job_with_no_stages_keeps_sweeping(self) -> None:
        """A vehicle load and a Blender preview count nothing."""
        app = FakeApp()
        app._set_busy(True)
        app._handle_build_progress({"stage": 0, "stage_total": 0, "message": "..."})
        self.assertEqual(app.busy_progress.mode, "indeterminate")
        self.assertEqual(app.busy_progress.starts, 1)

    def test_the_first_counted_stage_takes_the_bar_off_the_sweep(self) -> None:
        app = FakeApp()
        app._set_busy(True)
        app.progress(1, "build_plan", current=0, total=1)
        app.settle()
        self.assertEqual(app.busy_progress.mode, "determinate")
        self.assertEqual(app.busy_progress.maximum, PROGRESS_RESOLUTION)


if __name__ == "__main__":
    unittest.main()
