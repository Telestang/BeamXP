"""2026-09-07: exact normal-map arithmetic and independent GPU ownership."""

import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Event, get_ident
from unittest.mock import patch

import numpy as np

from glyph_lab.mask_compute import MaskService, normal_strength
from glyph_lab.normal_map import edge_steepness


def reference_strength(normal):
    xy = normal[:, :, :2].astype(np.float32) / 127.5 - 1.0
    z = np.sqrt(np.clip(1.0 - np.square(xy).sum(axis=2), 1e-6, 1.0))
    result = np.zeros(normal.shape[:2], np.float32)
    for component in (xy[:, :, 0], xy[:, :, 1], z):
        dx = np.diff(component, axis=1)
        dx *= dx
        result[:, 1:] += dx
        result[:, :-1] += dx
        dy = np.diff(component, axis=0)
        dy *= dy
        result[1:, :] += dy
        result[:-1, :] += dy
    return result


class NormalSchedulingTests(unittest.TestCase):
    def test_cpu_small_and_non_byte_maps_do_not_create_gpu_workers(self):
        normal = np.full((16, 16, 3), 128, np.uint8)
        with patch("glyph_lab.mask_compute.MaskService") as service:
            self.assertIsNone(normal_strength(normal, "cpu"))
            self.assertIsNone(normal_strength(normal, "auto"))
            self.assertIsNone(normal_strength(normal.astype(np.float32), "gpu"))
        service.assert_not_called()

    def test_two_workers_overlap_and_release_on_their_own_threads(self):
        started = [Event(), Event()]
        release = Event()
        owners, closed = [], []

        class Backend:
            def __init__(self):
                self.owner = get_ident()
                owners.append(self.owner)

            def normal_strength(self, normal):
                started[int(normal[0, 0, 0])].set()
                if not release.wait(5):
                    raise TimeoutError("second GPU worker never started")
                return normal

            def close(self):
                closed.append((self.owner, get_ident()))

        with patch("glyph_lab.mask_compute._GpuMasks", Backend):
            service = MaskService(workers=2)
            try:
                futures = [service.submit_normal(np.full((1, 1, 3), n, np.uint8)) for n in range(2)]
                self.assertTrue(all(event.wait(3) for event in started))
                self.assertEqual(len(set(owners)), 2)
                release.set()
                for n, future in enumerate(futures):
                    self.assertEqual(int(future.result()[0, 0, 0]), n)
                self.assertEqual(service.completed, 2)
                self.assertEqual(service.texels, 2)
            finally:
                release.set()
                service.close()
        self.assertEqual(service._pending, [0, 0])
        self.assertEqual(len(closed), 2)
        self.assertTrue(all(owner == closer for owner, closer in closed))

    def test_worker_count_is_bounded(self):
        for count in [0, 3, 100]:
            with self.assertRaises(ValueError):
                MaskService(workers=count)


class NormalGpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = MaskService(workers=2)
        if cls.service.submit_normal(np.zeros((1, 1, 3), np.uint8)).result() is None:
            cls.service.close()
            raise unittest.SkipTest("OpenGL compute unavailable")

    @classmethod
    def tearDownClass(cls):
        cls.service.close()

    def test_raw_float_bits_match_all_byte_pairs_and_random_borders(self):
        rows, columns = np.indices((256, 256), dtype=np.uint8)
        complete = np.stack((columns, rows, columns), axis=2)
        rng = np.random.default_rng(909)
        inputs = [complete, complete[::-1, ::2], np.full((67, 35, 4), 128, np.uint8)]
        inputs.extend(rng.integers(0, 256, (*shape, 3), np.uint8)
                      for shape in [(1, 1), (1, 67), (67, 1), (19, 35), (513, 511)])
        futures = [self.service.submit_normal(normal) for normal in inputs]
        for normal, future in zip(inputs, futures):
            actual = future.result()
            self.assertIsNotNone(actual)
            np.testing.assert_array_equal(actual.view(np.uint32), reference_strength(normal).view(np.uint32))

    def test_final_edges_match_with_sparse_empty_and_full_domains(self):
        rng = np.random.default_rng(910)
        normal = rng.integers(0, 256, (257, 511, 4), np.uint8)
        with patch("glyph_lab.mask_compute._SERVICE", self.service):
            for domains in [np.zeros(normal.shape[:2], np.int32),
                            np.ones(normal.shape[:2], np.int32),
                            rng.integers(0, 4, normal.shape[:2], np.int32)]:
                expected = edge_steepness(normal, domains, backend="cpu")
                actual = edge_steepness(normal, domains, backend="gpu")
                np.testing.assert_array_equal(actual, expected)

    def test_failed_driver_arithmetic_check_falls_back_and_disables_all_workers(self):
        normal = np.full((35, 67, 3), 128, np.uint8)
        domains = np.ones(normal.shape[:2], np.int32)
        expected = edge_steepness(normal, domains, backend="cpu")
        service = MaskService(workers=2)
        self.addCleanup(service.close)
        with (patch("glyph_lab.mask_compute._SERVICE", service),
              patch("glyph_lab.normal_compute.NormalGradients.strength",
                    side_effect=lambda normal, *_: np.ones(normal.shape[:2], np.float32)) as broken):
            with self.assertLogs("glyph_lab.mask_compute", level="WARNING"):
                actual = edge_steepness(normal, domains, backend="gpu")
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(edge_steepness(normal, domains, backend="gpu"), expected)
            self.assertTrue(service._failed)
            self.assertEqual(broken.call_count, 1)

    def test_mixed_masks_and_normal_jobs_share_reused_buffers_safely(self):
        from glyph_lab.detect import sparse_field_with_mesh_cuts

        rng = np.random.default_rng(911)
        normal = rng.integers(0, 256, (257, 513, 3), np.uint8)
        lit = rng.random((71, 53)) > 0.7
        sample = np.ones_like(lit)

        def run(index):
            if index % 2:
                actual = self.service.submit_normal(normal).result()
                np.testing.assert_array_equal(actual, reference_strength(normal))
            else:
                actual = self.service.submit(lit, sample, None, 11, 3).result()
                expected = sparse_field_with_mesh_cuts(lit, sample, None, 11, 3)
                np.testing.assert_array_equal(actual[0], expected[0])

        with ThreadPoolExecutor(max_workers=4) as callers:
            list(callers.map(run, range(12)))


if __name__ == "__main__":
    unittest.main()
