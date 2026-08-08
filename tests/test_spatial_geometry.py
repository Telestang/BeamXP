"""Regressions for the geometry helpers in `hand_drive_parts/spatial_analysis`.

Recommend Transforms no longer reasons spatially, but these helpers -- filled
surface visibility, the CPU/GPU kernel equivalence behind them, reflected
symmetry and the driver frame -- remain in the library for the preview, the
build and the standalone classifier sandbox, so they keep their tests.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from beamxp import hand_drive_core as core
from beamxp import spatial_visibility_backend
from tests.cabin_fixtures import (
    CAMERA_JBEAM,
    EYE,
    base_cabin,
    box_cloud,
    make_context,
    mirror_x,
)


class DriverFrameTests(unittest.TestCase):
    def test_frame_from_camera_and_wheel(self) -> None:
        context = make_context(base_cabin())
        frame = core.driver_frame_for_context(context)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.source, "camera+wheel")
        self.assertAlmostEqual(frame.eye[0], 0.40, places=2)
        self.assertAlmostEqual(frame.eye[2], 1.20, places=2)
        self.assertEqual(frame.side, 1)
        # the wheel sits ahead of the eye: forward is -y
        self.assertLess(frame.forward[1], -0.9)
        self.assertEqual(frame.wheel_id, "veh_steer")


class SurfaceVisibilityTests(unittest.TestCase):
    def test_filled_triangle_occludes_ray_through_its_interior(self) -> None:
        points = np.array(((0.0, 0.0, 2.0), (2.0, 0.0, 2.0)))
        cover = np.array((
            ((-1.0, -1.0, 1.0), (1.0, -1.0, 1.0), (0.0, 1.0, 1.0)),
        ))
        stats = core.surface_visibility_stats(
            points,
            (0.0, 0.0, 0.0),
            {"cover": cover},
            set(),
            (0.0, 0.0, 1.0),
        )
        self.assertIsNotNone(stats)
        self.assertEqual(stats["vf"], 0.5)
        self.assertEqual(stats["front_vf"], 0.5)

    def test_batched_cpu_surface_visibility_matches_scalar_reference(self) -> None:
        points_by_id = {
            "first": np.array(((0.0, 0.0, 2.0), (2.0, 0.0, 2.0))),
            "second": np.array(((0.2, 0.0, 3.0), (-2.0, 0.0, 2.0))),
        }
        cover = np.array((
            ((-1.0, -1.0, 1.0), (1.0, -1.0, 1.0), (0.0, 1.0, 1.0)),
        ))
        expected = {
            object_id: core.surface_visibility_stats(
                points,
                (0.0, 0.0, 0.0),
                cover,
                set(),
                (0.0, 0.0, 1.0),
            )
            for object_id, points in points_by_id.items()
        }
        with patch.dict(
            "os.environ", {spatial_visibility_backend.SPATIAL_BACKEND_ENV: "cpu"}
        ):
            actual = core.surface_visibility_stats_batch(
                points_by_id,
                (0.0, 0.0, 0.0),
                cover,
                (0.0, 0.0, 1.0),
            )
        self.assertEqual(actual, expected)

    def test_gpu_surface_visibility_matches_scalar_reference_when_available(
        self,
    ) -> None:
        points_by_id = {
            "first": np.array(((0.0, 0.0, 2.0), (2.0, 0.0, 2.0))),
            "second": np.array(((0.2, 0.0, 3.0), (-2.0, 0.0, 2.0))),
        }
        cover = np.array((
            ((-1.0, -1.0, 1.0), (1.0, -1.0, 1.0), (0.0, 1.0, 1.0)),
        ))
        expected = {
            object_id: core.surface_visibility_stats(
                points,
                (0.0, 0.0, 0.0),
                cover,
                set(),
                (0.0, 0.0, 1.0),
            )
            for object_id, points in points_by_id.items()
        }
        spatial_visibility_backend.reset_thread_backend()
        try:
            with patch.dict(
                "os.environ",
                {spatial_visibility_backend.SPATIAL_BACKEND_ENV: "gpu"},
            ):
                if spatial_visibility_backend.gpu_renderer() is None:
                    self.skipTest("OpenGL 4.3 compute context is unavailable")
                actual = core.surface_visibility_stats_batch(
                    points_by_id,
                    (0.0, 0.0, 0.0),
                    cover,
                    (0.0, 0.0, 1.0),
                )
        finally:
            spatial_visibility_backend.reset_thread_backend()
        self.assertEqual(actual, expected)

    def test_visible_admission_is_limited_to_forward_hemisphere(self) -> None:
        front = box_cloud((0.40, -0.50, 0.90), (0.10, 0.10, 0.10), n=80, seed=56)
        rear = box_cloud((0.40, 0.90, 0.90), (0.10, 0.10, 0.10), n=80, seed=57)
        stats = core.visibility_scan(
            {"front": front, "rear": rear},
            EYE,
            set(),
            (0.0, -1.0, 0.0),
        )
        self.assertGreater(stats["front"]["front_vf"], 0.80)
        self.assertEqual(stats["rear"]["front_vf"], 0.0)
        self.assertEqual(stats["rear"]["front_backed"], 0.0)
        self.assertTrue(np.isinf(stats["rear"]["front_depth"]))
        # The raw shell remains spherical for other diagnostics, but
        # forward admission consumes only its forward evidence.
        self.assertGreater(stats["rear"]["vf"], 0.70)

    def test_gpu_visibility_shell_matches_cpu_reference_when_available(self) -> None:
        meshes = base_cabin()
        with patch.dict(
            "os.environ", {spatial_visibility_backend.SPATIAL_BACKEND_ENV: "cpu"}
        ):
            expected = core.visibility_scan(
                meshes, EYE, {"veh_steer"}, (0.0, -1.0, 0.0)
            )

        spatial_visibility_backend.reset_thread_backend()
        try:
            with patch.dict(
                "os.environ",
                {spatial_visibility_backend.SPATIAL_BACKEND_ENV: "gpu"},
            ):
                if spatial_visibility_backend.gpu_renderer() is None:
                    self.skipTest("OpenGL 4.3 compute context is unavailable")
                actual = core.visibility_scan(
                    meshes, EYE, {"veh_steer"}, (0.0, -1.0, 0.0)
                )
        finally:
            spatial_visibility_backend.reset_thread_backend()
        self.assertEqual(actual, expected)


class MaterialTests(unittest.TestCase):
    def test_translucent_material_needs_active_alpha_evidence_for_glass(self) -> None:
        opaque_cage = {
            "translucent": True,
            "translucentBlendOp": "None",
            "Stages": [{"baseColorMap": "cage_d.color.png"}],
        }
        real_glass = {
            "translucent": True,
            "translucentBlendOp": "PreMulAlpha",
            "Stages": [{"opacityMap": "glass_o.data.png"}],
        }
        self.assertFalse(core.material_uses_glass_blending(opaque_cage))
        self.assertTrue(core.material_uses_glass_blending(real_glass))


class GeometryHelperTests(unittest.TestCase):
    def test_reflected_orphans_require_every_exact_twin(self) -> None:
        half = box_cloud((0.35, 0.0, 0.0), (0.2, 0.4, 0.3), n=80, seed=39)
        symmetric = np.concatenate([half, mirror_x(half)])
        orphans, coarse = core.reflected_orphan_stats(symmetric, 0.0)
        self.assertEqual(orphans, 0)
        self.assertEqual(coarse, 0.0)

        asymmetric = symmetric.copy()
        asymmetric[0, 1] += 0.05
        orphans, coarse = core.reflected_orphan_stats(asymmetric, 0.0)
        self.assertEqual(orphans, 2)
        self.assertGreater(coarse, 0.0)

    def test_gpu_reflected_orphans_match_cpu_reference_when_available(self) -> None:
        half = box_cloud((0.35, 0.0, 0.0), (0.2, 0.4, 0.3), n=80, seed=61)
        cloud = np.concatenate([half, mirror_x(half)])
        cloud[0, 1] += 0.05
        with patch.dict(
            "os.environ", {spatial_visibility_backend.SPATIAL_BACKEND_ENV: "cpu"}
        ):
            expected = core.reflected_orphan_stats(cloud, 0.0)

        spatial_visibility_backend.reset_thread_backend()
        try:
            with patch.dict(
                "os.environ",
                {spatial_visibility_backend.SPATIAL_BACKEND_ENV: "gpu"},
            ):
                if spatial_visibility_backend.gpu_renderer() is None:
                    self.skipTest("OpenGL 4.3 compute context is unavailable")
                actual = core.reflected_orphan_stats(cloud, 0.0)
        finally:
            spatial_visibility_backend.reset_thread_backend()
        self.assertEqual(actual, expected)

    def test_symmetry_residual_separates_centred_from_one_sided(self) -> None:
        centred = box_cloud((0.0, 0.0, 0.0), (1.0, 0.5, 0.5), n=300, seed=30)
        offset = box_cloud((0.4, 0.0, 0.0), (0.3, 0.5, 0.5), n=300, seed=31)
        self.assertLess(core.cloud_symmetry_residual(centred, 0.0), 0.045)
        self.assertGreater(core.cloud_symmetry_residual(offset, 0.0), 0.09)

    def test_pair_residual_accepts_reflections_and_rejects_strangers(self) -> None:
        left = box_cloud((0.5, 0.1, 0.4), (0.3, 0.4, 0.3), n=250, seed=32)
        right = mirror_x(left)
        stranger = box_cloud((-0.5, 0.4, 0.6), (0.5, 0.2, 0.2), n=250, seed=33)
        self.assertLess(core.mirror_pair_residual(left, right, 0.0), 0.10)
        self.assertGreater(core.mirror_pair_residual(left, stranger, 0.0), 0.10)

    def test_camera_row_parsing(self) -> None:
        rows = core.parse_internal_camera_positions(
            {"veh.jbeam": CAMERA_JBEAM}, {}
        )
        self.assertEqual(rows, [(0.40, 0.29, 1.20)])


if __name__ == "__main__":
    unittest.main()
