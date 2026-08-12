"""Tests for the production colour/normal cold-path coordinator."""

from __future__ import annotations

import threading
import unittest
from unittest.mock import patch

import numpy as np

from mesh_segmentation_transform.annotate_texture_regions import (
    LocalContrastDetection,
    MserConfig,
    run_detection,
)
from mesh_segmentation_transform.mirror_texture_for_rhd import (
    RhdTextureConfig,
    _build_detection_views,
)
from mesh_segmentation_transform.production_texture_detection import (
    NormalGpuEdgeData,
    ProductionDetectionSession,
    prepare_normal_gpu_edge_data,
    run_colour_and_relief_jobs,
)
from mesh_segmentation_transform.relief_from_normals import ReliefConfig


class ProductionTextureDetectionTests(unittest.TestCase):
    def test_session_deduplicates_only_exact_layer_inputs(self) -> None:
        session: ProductionDetectionSession[str] = ProductionDetectionSession()
        image = np.zeros((4, 5, 3), dtype=np.uint8)
        mirror = np.ones((4, 5), dtype=bool)
        config = MserConfig()
        first = session.key(image, mirror, mirror, config, policy="production")
        duplicate = session.key(image.copy(), mirror.copy(), mirror.copy(), config, policy="production")
        changed = image.copy()
        changed[1, 2, 0] = 1
        different = session.key(changed, mirror, mirror, config, policy="production")

        session.put(first, "detected")

        self.assertEqual(session.get(duplicate), "detected")
        self.assertIsNone(session.get(different))
        self.assertEqual((session.hits, session.misses), (1, 1))

    def test_normal_gpu_edge_is_prepared_once_for_both_detectors(self) -> None:
        normal = np.full((4, 5, 3), 128, dtype=np.uint8)
        relief = np.full((4, 5, 3), 64, dtype=np.uint8)
        response = np.full((4, 5), 17.0, dtype=np.float32)
        with (
            patch(
                "mesh_segmentation_transform.production_texture_detection.render_relief",
                return_value=relief,
            ) as render,
            patch(
                "mesh_segmentation_transform.production_texture_detection.compute_edge_response",
                return_value=response,
            ) as edge,
        ):
            result = prepare_normal_gpu_edge_data(
                normal, ReliefConfig(), "laplacian", 3, 1.0
            )

        render.assert_called_once_with(normal, ReliefConfig())
        edge.assert_called_once()
        grey, operator, kernel, blur = edge.call_args.args
        self.assertEqual(grey.shape, (4, 5))
        self.assertTrue(grey.flags.c_contiguous)
        self.assertEqual((operator, kernel, blur), ("laplacian", 3, 1.0))
        self.assertIs(result.relief_bgr, relief)
        self.assertIs(result.edge_response, response)

    def test_session_reuses_exact_normal_edge_data_across_layers(self) -> None:
        session: ProductionDetectionSession[str] = ProductionDetectionSession()
        normal = np.full((4, 5, 3), 128, dtype=np.uint8)
        prepared = NormalGpuEdgeData(
            relief_bgr=np.zeros((4, 5, 3), dtype=np.uint8),
            edge_response=np.ones((4, 5), dtype=np.float32),
            render_seconds=0.1,
            edge_seconds=0.2,
        )
        with patch(
            "mesh_segmentation_transform.production_texture_detection.prepare_normal_gpu_edge_data",
            return_value=prepared,
        ) as prepare:
            first, first_reused = session.normal_edge_data(normal, ReliefConfig())
            second, second_reused = session.normal_edge_data(
                normal.copy(), ReliefConfig()
            )

        prepare.assert_called_once()
        self.assertIs(first, prepared)
        self.assertIs(second, prepared)
        self.assertFalse(first_reused)
        self.assertTrue(second_reused)
        self.assertEqual((session.normal_edge_hits, session.normal_edge_misses), (1, 1))

    def test_normal_edge_response_tracks_a_detection_crop(self) -> None:
        bgr = np.zeros((12, 16, 3), dtype=np.uint8)
        mirror = np.zeros((12, 16), dtype=bool)
        mirror[3:8, 5:11] = True
        response = np.arange(12 * 16, dtype=np.float32).reshape(12, 16)
        config = RhdTextureConfig(
            crop_detection_to_domain=True,
            detection_crop_padding_px=1,
            detect_island_tiles_individually=False,
            collage_detection_islands=False,
        )
        views = _build_detection_views(bgr, mirror, mirror, config, response)

        self.assertEqual(len(views), 1)
        self.assertEqual(views[0].mode, "crop")
        np.testing.assert_array_equal(views[0].relief_bridge_response, response[2:9, 4:12])

    def test_colour_gpu_front_end_keeps_the_supplied_normal_edge_data(self) -> None:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        domain = np.ones((8, 8), dtype=bool)
        normal_edges = np.full((8, 8), 123.0, dtype=np.float32)
        contrast = LocalContrastDetection(
            boxes=np.empty((0, 4), dtype=np.int32),
            response=np.zeros((8, 8), dtype=np.float32),
            threshold=0.0,
        )
        with patch(
            "mesh_segmentation_transform.annotate_texture_regions.detect_local_contrast_gpu",
            return_value=contrast,
        ):
            run = run_detection(
                image,
                domain,
                MserConfig(box_source="contrast_gpu"),
                initial_relief_bridge_response=normal_edges,
            )

        # Entry state 1 is the state produced by the raw local-contrast step.
        self.assertIs(run.entry_states[1].relief_bridge_response, normal_edges)

    def test_colour_only_keeps_a_single_job_path(self) -> None:
        result = run_colour_and_relief_jobs(lambda: "colour")

        self.assertEqual(result.colour.value, "colour")
        self.assertIsNone(result.relief)
        self.assertGreaterEqual(result.colour.seconds, 0.0)
        self.assertGreaterEqual(result.wall_seconds, result.colour.seconds)

    def test_independent_jobs_start_together(self) -> None:
        """Normal preparation can overlap colour work before GPU serialization."""
        ready = threading.Barrier(2)

        def job(value: str) -> str:
            ready.wait(timeout=1.0)
            return value

        result = run_colour_and_relief_jobs(
            lambda: job("colour"), lambda: job("relief"),
        )

        self.assertEqual(result.colour.value, "colour")
        self.assertIsNotNone(result.relief)
        assert result.relief is not None
        self.assertEqual(result.relief.value, "relief")
        self.assertGreaterEqual(result.wall_seconds, 0.0)
