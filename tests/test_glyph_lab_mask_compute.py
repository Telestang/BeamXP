"""2026-09-07: exact mask boundaries, asynchronous ordering and CPU fallback."""

import unittest
from concurrent.futures import Future
from threading import Event
from unittest.mock import patch

import numpy as np

from glyph_lab.detect import sparse_field_with_mesh_cuts
from glyph_lab.mask_compute import (
    MaskService,
    compute_backend,
    pipeline_candidate,
    prepared_items,
    sparse_result,
    submit_sparse_field,
)


class SchedulingTests(unittest.TestCase):
    def test_cpu_mode_never_submits_gpu_work(self):
        mask = np.ones((512, 512), dtype=bool)
        with patch("glyph_lab.mask_compute.MaskService") as service:
            self.assertIsNone(submit_sparse_field(mask, mask, None, 11, 2, "cpu"))
        service.assert_not_called()

    def test_auto_keeps_small_and_unsupported_work_on_cpu(self):
        mask = np.ones((32, 32), dtype=bool)
        with patch("glyph_lab.mask_compute.MaskService") as service:
            for kernel, count, backend in [(11, 2, "auto"), (33, 2, "gpu"), (5, -1, "gpu")]:
                self.assertIsNone(submit_sparse_field(mask, mask, None, kernel, count, backend))
            labels = np.full(mask.shape, 1 << 30, dtype=np.int64)
            self.assertIsNone(submit_sparse_field(mask, mask, labels, 5, 2, "gpu"))
        service.assert_not_called()

    def test_small_texture_pass_stays_cpu_until_enough_work_can_overlap(self):
        crop = (1, (slice(0, 512), slice(0, 512)))
        self.assertFalse(pipeline_candidate([crop] * 3, 11, "auto"))
        self.assertTrue(pipeline_candidate([crop] * 4, 11, "auto"))
        self.assertFalse(pipeline_candidate([crop] * 4, 11, "cpu"))
        self.assertTrue(pipeline_candidate([crop], 11, "gpu"))

    def test_prepares_next_item_while_consumer_works_and_preserves_order(self):
        second_started = Event()
        consumed = []

        def prepare(index):
            if index == 1:
                second_started.set()
            return index

        with prepared_items(range(4), prepare, True) as ready:
            for value in ready:
                if value == 0:
                    self.assertTrue(second_started.wait(3))
                consumed.append(value)
        self.assertEqual(consumed, [0, 1, 2, 3])

    def test_preparation_error_propagates(self):
        def prepare(index):
            if index == 1:
                raise ValueError("failed CPU preparation")
            return index

        with (self.assertRaisesRegex(ValueError, "failed CPU preparation"),
              prepared_items(range(3), prepare, True) as ready):
            list(ready)

    def test_device_failure_uses_cpu_and_is_not_retried_per_crop(self):
        service = MaskService()
        self.addCleanup(service.close)
        mask = np.ones((16, 16), dtype=bool)
        expected = sparse_field_with_mesh_cuts(mask, mask, None, 5, 2)
        with patch("glyph_lab.mask_compute._GpuMasks", side_effect=RuntimeError("no device")) as create:
            with self.assertLogs("glyph_lab.mask_compute", level="WARNING"):
                failed = service.submit(mask, mask, None, 5, 2)
                failed.result()
                result = sparse_result(
                    failed, lambda: expected,
                )
            self.assertIs(result, expected)
            self.assertIsNone(service.submit(mask, mask, None, 5, 2).result())
        self.assertEqual(create.call_count, 1)

    def test_cpu_claims_a_queued_job_without_duplicate_execution(self):
        queued = Future()
        answer = object()
        self.assertIs(sparse_result(queued, lambda: answer), answer)
        self.assertTrue(queued.cancelled())
        self.assertFalse(queued.set_running_or_notify_cancel())

    def test_started_gpu_job_retains_ownership(self):
        started = Future()
        self.assertTrue(started.set_running_or_notify_cancel())
        answer = object()
        started.set_result(answer)
        self.assertIs(sparse_result(started, lambda: self.fail("duplicated GPU work")), answer)

    def test_cpu_help_leaves_heavy_jobs_to_gpu(self):
        heavy = Future()
        heavy.glyph_cpu_work = 100_000_000
        answer = object()
        heavy.set_result(answer)
        with patch.object(heavy, "cancel") as cancel:
            self.assertIs(sparse_result(heavy, lambda: self.fail("claimed heavy job")), answer)
        cancel.assert_not_called()

    def test_runtime_override_is_validated(self):
        with patch.dict("os.environ", {"BEAMXP_GLYPH_COMPUTE": "cpu"}):
            self.assertEqual(compute_backend(), "cpu")
            self.assertEqual(compute_backend("gpu"), "gpu")
        with self.assertRaises(ValueError):
            compute_backend("typo")


class ExactGpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = MaskService()
        mask = np.ones((1, 1), dtype=bool)
        if cls.service.submit(mask, mask, None, 1, 1).result() is None:
            cls.service.close()
            raise unittest.SkipTest("OpenGL compute unavailable")

    @classmethod
    def tearDownClass(cls):
        cls.service.close()

    def test_fields_match_with_edges_holes_mesh_cuts_and_even_windows(self):
        rng = np.random.default_rng(907)
        for shape in [(1, 1), (1, 37), (37, 1), (7, 13), (33, 65), (257, 321)]:
            rows, columns = np.indices(shape)
            for density in [0.0, 0.15, 1.0]:
                lit = rng.random(shape) < density
                sample = rng.random(shape) < 0.8
                # 2026-09-07: adjacent pieces and zero/negative labels expose
                # accidental cross-seam counting, including an empty field.
                labels = ((columns // 3 + rows // 5) % 4 - 1).astype(np.int32)
                for mesh in [None, labels, np.zeros(shape, dtype=np.int32)]:
                    for kernel in [1, 5, 10, 11, 31]:
                        expected = sparse_field_with_mesh_cuts(lit, sample, mesh, kernel, 3)
                        actual = self.service.submit(lit, sample, mesh, kernel, 3).result()
                        self.assertIsNotNone(actual)
                        np.testing.assert_array_equal(actual[0], expected[0])
                        np.testing.assert_array_equal(actual[1], expected[1])

    def test_noncontiguous_inputs_and_all_clear_limit(self):
        lit = np.ones((67, 97), dtype=bool)[::2, ::2]
        sample = np.ones_like(lit)
        for count in [0, 1, 100_000]:
            actual = self.service.submit(lit, sample, None, 11, count).result()
            expected = sparse_field_with_mesh_cuts(lit, sample, None, 11, count)
            self.assertIsNotNone(actual)
            np.testing.assert_array_equal(actual[0], expected[0])


if __name__ == "__main__":
    unittest.main()
