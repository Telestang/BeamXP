"""The export's own timing ledger.

The bug worth pinning is the one this hit on its first run: the in-process
correction path returned a snapshot of the ledger it had just written into, and
the parent absorbed it back, once per task.  Ten tasks later a 117-second
export was reporting 1,210 seconds of work.
"""

from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mesh_segmentation_transform import mirror_texture_for_rhd as rhd
from mesh_segmentation_transform.export_timings import TimingLedger


class LedgerTests(unittest.TestCase):
    def test_a_span_records_seconds_and_a_call(self) -> None:
        ledger = TimingLedger()
        with ledger.span("encode dds"):
            time.sleep(0.01)
        self.assertEqual(ledger.calls["encode dds"], 1)
        self.assertGreater(ledger.seconds["encode dds"], 0.0)

    def test_a_raising_span_still_records(self) -> None:
        """A failed job's cost is part of where the export's time went."""
        ledger = TimingLedger()
        with self.assertRaises(ValueError), ledger.span("encode dds"):
            raise ValueError("boom")
        self.assertEqual(ledger.calls["encode dds"], 1)

    def test_absorbing_nothing_is_a_no_op(self) -> None:
        ledger = TimingLedger()
        with ledger.span("encode dds"):
            pass
        before = dict(ledger.seconds)
        ledger.absorb(None)
        self.assertEqual(dict(ledger.seconds), before)

    def test_a_snapshot_is_a_copy_not_a_view(self) -> None:
        """A worker's snapshot must not keep growing after it is taken."""
        ledger = TimingLedger()
        with ledger.span("encode dds"):
            pass
        seconds, calls = ledger.snapshot()
        with ledger.span("encode dds"):
            pass
        self.assertEqual(calls["encode dds"], 1)
        self.assertEqual(ledger.calls["encode dds"], 2)
        self.assertLess(seconds["encode dds"], ledger.seconds["encode dds"])

    def test_absorbing_a_worker_adds_its_costs(self) -> None:
        parent, worker = TimingLedger(), TimingLedger()
        with parent.span("encode dds"):
            pass
        with worker.span("encode dds"):
            pass
        parent.absorb(worker.snapshot())
        self.assertEqual(parent.calls["encode dds"], 2)

    def test_clear_empties_the_ledger(self) -> None:
        ledger = TimingLedger()
        with ledger.span("encode dds"):
            pass
        ledger.clear()
        self.assertFalse(ledger.seconds)
        self.assertFalse(ledger.calls)

    def test_concurrency_is_recorded_so_a_sum_can_be_read_correctly(self) -> None:
        """Above one, the components sum to CPU time, not to wall."""
        ledger = TimingLedger()

        def work() -> None:
            with ledger.span("encode dds"):
                time.sleep(0.05)

        threads = [threading.Thread(target=work) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertGreater(ledger.peak_concurrency, 1)

    def test_a_serial_ledger_never_exceeds_its_wall(self) -> None:
        """The invariant the self-absorption bug broke."""
        ledger = TimingLedger()
        started = time.perf_counter()
        for _ in range(5):
            with ledger.span("encode dds"):
                time.sleep(0.005)
        wall = time.perf_counter() - started
        self.assertEqual(ledger.peak_concurrency, 1)
        self.assertLessEqual(sum(ledger.seconds.values()), wall + 1e-6)


class SummaryTests(unittest.TestCase):
    def test_an_empty_ledger_reports_nothing(self) -> None:
        self.assertEqual(TimingLedger().summary(1.0), [])

    def test_components_are_grouped_and_shares_sum_to_one(self) -> None:
        ledger = TimingLedger()
        ledger.seconds.update({"detect (glyph_lab)": 6.0, "encode dds": 2.0,
                               "mesh seams": 2.0})
        ledger.calls.update({"detect (glyph_lab)": 1, "encode dds": 1,
                             "mesh seams": 1})
        lines = "\n".join(ledger.summary(10.0))
        self.assertIn("detection", lines)
        self.assertIn("60.0%", lines)
        self.assertIn("texture encoding", lines)
        self.assertIn("mesh geometry", lines)

    def test_an_unknown_label_still_reports(self) -> None:
        ledger = TimingLedger()
        ledger.seconds["something new"] = 1.0
        ledger.calls["something new"] = 1
        self.assertIn("something new", "\n".join(ledger.summary(2.0)))

    def test_the_unmeasured_line_appears_only_for_a_serial_export(self) -> None:
        """Across workers the components exceed wall, so the gap is meaningless."""
        ledger = TimingLedger()
        ledger.seconds["encode dds"] = 1.0
        ledger.calls["encode dds"] = 1
        self.assertIn("unmeasured", "\n".join(ledger.summary(10.0)))
        self.assertNotIn("unmeasured", "\n".join(ledger.summary(10.0, workers=4)))

    def test_a_pool_says_how_many_workers_the_total_is_spread_over(self) -> None:
        ledger = TimingLedger()
        ledger.seconds["encode dds"] = 8.0
        ledger.calls["encode dds"] = 4
        self.assertIn("over 4 worker(s)", "\n".join(ledger.summary(3.0, workers=4)))


class InProcessPathTests(unittest.TestCase):
    def test_an_in_process_outcome_carries_no_snapshot(self) -> None:
        """It wrote into the parent's own ledger; handing it back doubles it."""
        outcome = rhd.TextureCorrectionTaskOutcome(None, [], "failed")
        self.assertIsNone(outcome.timings)


if __name__ == "__main__":
    unittest.main()
