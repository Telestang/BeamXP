from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import beamng_hand_drive_core as core
import beamng_hand_drive_tool as tool
import spatial_visibility_backend


# ---------------------------------------------------------------------------
# Geometry fixtures
#
# The classifier reasons from point clouds around a driver eye, so every
# fixture is a small synthetic cabin: real coordinates, real occlusion, a real
# camerasInternal row. Nothing here depends on part names except the two
# sanctioned uses (the steering-wheel anchor score and the steering-column
# resolution-floor hint), which have their own dedicated tests.

EYE = (0.40, 0.29, 1.20)
CAMERA_JBEAM = (
    '{"fixture_body": {"camerasInternal":[\n'
    '    ["type", "x", "y", "z", "fov", "id1:"],\n'
    '    ["dash", 0.40, 0.29, 1.20, 55, "f1"],\n'
    "]}}"
)


def box_cloud(center, size, n=200, seed=0):
    rng = np.random.default_rng(seed + int(abs(center[0]) * 1000) + n)
    half = np.array(size, dtype=float) / 2.0
    return np.array(center, dtype=float) + rng.uniform(-1.0, 1.0, size=(n, 3)) * half


def mirror_x(points):
    out = np.array(points, dtype=float).copy()
    out[:, 0] = -out[:, 0]
    return out


def sym_cloud(center, size, n=200, seed=0):
    """Exactly centreline-symmetric cloud (how symmetric meshes are modelled:
    mirrored vertices, not merely a symmetric envelope)."""
    half = box_cloud(center, size, n=max(n // 2, 20), seed=seed)
    return np.concatenate([half, mirror_x(half)])


def base_cabin() -> dict[str, np.ndarray]:
    """A minimal but occlusion-correct cabin around the eye at EYE.

    Door cards sit inboard of the door skins (the lining layers the spatial
    scope depends on); the firewall backs the dash; seats flank the eye."""
    card_fl = box_cloud((0.73, -0.20, 0.68), (0.05, 1.00, 0.75), n=220, seed=1)
    skin_fl = box_cloud((0.82, -0.20, 0.80), (0.06, 1.20, 1.00), n=220, seed=2)
    # a seat is an L-shell (cushion + backrest), not a solid block: a filled
    # box would wrap phantom points around the eye and occlude the footwell
    seat_fl = np.concatenate([
        box_cloud((0.40, 0.15, 0.45), (0.50, 0.55, 0.20), n=120, seed=3),
        box_cloud((0.40, 0.44, 0.80), (0.50, 0.16, 0.85), n=120, seed=35),
    ])
    return {
        "veh_dash": sym_cloud((0.0, -0.60, 0.95), (1.56, 0.30, 0.50), n=260, seed=4),
        "veh_firewall": sym_cloud((0.0, -0.95, 0.70), (1.50, 0.06, 1.00), n=200, seed=5),
        "veh_floor": sym_cloud((0.0, 0.10, 0.20), (1.50, 2.00, 0.05), n=240, seed=6),
        "veh_headliner": sym_cloud((0.0, 0.50, 1.36), (1.30, 1.80, 0.04), n=200, seed=7),
        "veh_roof": sym_cloud((0.0, 0.50, 1.45), (1.40, 1.90, 0.04), n=200, seed=42),
        "veh_card_FL": card_fl,
        "veh_card_FR": mirror_x(card_fl),
        "veh_skin_FL": skin_fl,
        "veh_skin_FR": mirror_x(skin_fl),
        "veh_seat_FL": seat_fl,
        "veh_seat_FR": mirror_x(seat_fl),
        # named so the sanctioned steering score anchors it (etk800-style)
        "veh_steer": box_cloud((0.40, -0.30, 0.95), (0.36, 0.06, 0.36), n=200, seed=8),
    }


def make_context(
    meshes: dict[str, np.ndarray],
    *,
    camera: bool = True,
    trims: dict[str, list[str]] | None = None,
    materials: dict[str, tuple[str, ...]] | None = None,
    material_flags: dict[str, dict[str, bool]] | None = None,
) -> core.VehicleContext:
    objects = {}
    preview = {}
    for object_id, points in meshes.items():
        pts = np.asarray(points, dtype=float)
        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        center = tuple(float(v) for v in (lo + hi) / 2.0)
        objects[object_id] = core.DaeObject(
            id=object_id,
            name=object_id,
            dae_path="vehicle.dae",
            x=center[0],
            y=center[1],
            z=center[2],
            geometry_ids=(),
        )
        preview[object_id] = {
            "bounds": (tuple(float(v) for v in lo), tuple(float(v) for v in hi)),
            "center": center,
            "sample_points": [tuple(float(v) for v in p) for p in pts],
            "geometry_ids": (),
            "materials": tuple((materials or {}).get(object_id, ())),
        }
    variants = {}
    roles_cache = {}
    resolved_cache = {}
    for trim_name, ids in (trims or {}).items():
        variants[trim_name] = core.VariantInfo(
            name=trim_name, pc_path="", info_path=None, display_name=trim_name
        )
        roles_cache[trim_name] = (set(), set(), set(ids))
        resolved_cache[trim_name] = {}
    context = core.VehicleContext(
        source_zip=Path("test.zip"),
        vehicle_id="veh",
        vehicle_path="vehicles/veh",
        dae_paths=[],
        variants=variants,
        objects=objects,
        preview_by_id=preview,
        jbeam_texts={"vehicles/veh/veh.jbeam": CAMERA_JBEAM} if camera else {},
        node_positions={},
        project_dir=Path("project"),
    )
    context.mesh_roles_cache.update(roles_cache)
    context.resolved_positions_cache.update(resolved_cache)
    context._material_flags = dict(material_flags or {})
    return context


def recommend(context: core.VehicleContext, object_ids=None) -> dict[str, dict[str, object]]:
    ids = list(object_ids) if object_ids is not None else list(context.objects)
    return {r["object_id"]: r for r in tool.build_mode_recommendations(context, ids)}


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

    def test_no_camera_and_no_wheel_yields_no_recommendations(self) -> None:
        meshes = base_cabin()
        meshes["veh_wheelish"] = meshes.pop("veh_steer")  # no steering token
        context = make_context(meshes, camera=False)
        self.assertIsNone(core.driver_frame_for_context(context))
        self.assertEqual(tool.build_mode_recommendations(context, list(context.objects)), [])


class ScopeTests(unittest.TestCase):
    def test_strongly_lined_boundary_mesh_is_enclosed(self) -> None:
        stats = {
            "vf": 0.267,
            "front_vf": 0.267,
            "backed": 1.0,
            "lined": 0.95,
            "depth": 0.21,
        }
        self.assertTrue(tool._is_enclosed_candidate(stats, 0.792, 0.803))
        stats["lined"] = 0.30
        self.assertFalse(tool._is_enclosed_candidate(stats, 0.792, 0.803))

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
        # The enclosure shell remains spherical; only driver-visible
        # admission is field-of-view limited.
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

    def test_rearward_exposed_mesh_does_not_enter_from_visibility(self) -> None:
        meshes = base_cabin()
        meshes["rear_exposed"] = box_cloud(
            (0.28, 0.75, 0.42), (0.30, 0.40, 0.15), n=110, seed=58
        )
        context = make_context(meshes)
        frame = core.driver_frame_for_context(context)
        present, arrays = tool._spatial_entries_for_trim(context, None, set(meshes))
        stats = core.visibility_scan(arrays, frame.eye, set(), frame.forward)

        # This fuel-tank-like fixture is exposed in the spherical shell but
        # entirely behind the eye, so exposure alone must not admit it.
        self.assertGreater(stats["rear_exposed"]["vf"], 0.28)
        self.assertEqual(stats["rear_exposed"]["front_vf"], 0.0)
        self.assertNotIn("rear_exposed", recommend(context))

    def test_door_card_pairs_and_exterior_skin_gets_nothing(self) -> None:
        recs = recommend(make_context(base_cabin()))
        self.assertIn("veh_card_FL", recs)
        self.assertEqual(recs["veh_card_FL"]["mode"], core.MODE_MIRROR_STRUCTURAL)
        self.assertEqual(recs["veh_card_FL"]["source_id"], "veh_card_FR")
        self.assertNotIn("veh_skin_FL", recs)
        self.assertNotIn("veh_skin_FR", recs)

    def test_stripped_trim_rejects_exposed_skin_and_keeps_lone_card(self) -> None:
        # the crux case: the passenger card is stripped, exposing the skin.
        # The skin must NOT be promoted to interior; the lone driver card
        # falls back to aesthetic mirror because its twin is absent.
        meshes = base_cabin()
        del meshes["veh_card_FR"]
        recs = recommend(make_context(meshes))
        self.assertNotIn("veh_skin_FR", recs)
        self.assertNotIn("veh_skin_FL", recs)
        self.assertIn("veh_card_FL", recs)
        self.assertEqual(recs["veh_card_FL"]["mode"], core.MODE_MIRROR)
        self.assertEqual(recs["veh_card_FL"]["source_id"], "")
        self.assertIn("twin absent", recs["veh_card_FL"]["reason"])

    def test_headliner_floor_and_sunvisor_are_symmetric_no_ops(self) -> None:
        meshes = base_cabin()
        meshes["veh_sunvisor"] = sym_cloud((0.0, -0.11, 1.30), (0.95, 0.16, 0.04), n=160, seed=9)
        recs = recommend(make_context(meshes))
        self.assertNotIn("veh_headliner", recs)
        self.assertNotIn("veh_floor", recs)
        self.assertNotIn("veh_sunvisor", recs)
        self.assertNotIn("veh_roof", recs)

    def test_dashboard_fascia_mirrors_despite_symmetry(self) -> None:
        recs = recommend(make_context(base_cabin()))
        self.assertIn("veh_dash", recs)
        self.assertEqual(recs["veh_dash"]["mode"], core.MODE_MIRROR)
        self.assertIn("fascia", recs["veh_dash"]["reason"])


class ControlConeTests(unittest.TestCase):
    def test_steering_wheel_translates(self) -> None:
        recs = recommend(make_context(base_cabin()))
        self.assertEqual(recs["veh_steer"]["mode"], core.MODE_TRANSLATE)
        self.assertEqual(recs["veh_steer"]["reason"], "steering wheel")

    def test_gauge_cluster_translates_without_a_gauge_name(self) -> None:
        meshes = base_cabin()
        meshes["veh_podA"] = box_cloud((0.40, -0.52, 0.95), (0.36, 0.08, 0.18), n=90, seed=10)
        recs = recommend(make_context(meshes))
        self.assertEqual(recs["veh_podA"]["mode"], core.MODE_TRANSLATE)

    def test_console_shifter_mirrors_but_column_shifter_translates(self) -> None:
        meshes = base_cabin()
        meshes["veh_lever_console"] = box_cloud((0.10, -0.35, 0.62), (0.05, 0.08, 0.20), n=80, seed=11)
        meshes["veh_lever_high"] = box_cloud((0.28, -0.42, 1.02), (0.08, 0.10, 0.06), n=80, seed=12)
        recs = recommend(make_context(meshes))
        self.assertEqual(recs["veh_lever_high"]["mode"], core.MODE_TRANSLATE)
        self.assertEqual(recs["veh_lever_console"]["mode"], core.MODE_MIRROR)
        self.assertEqual(recs["veh_lever_console"]["source_id"], "")

    def test_foot_parking_brake_translates_despite_brake_position_by_door(self) -> None:
        meshes = base_cabin()
        meshes["veh_footbrake"] = box_cloud((0.66, -0.75, 0.58), (0.08, 0.22, 0.20), n=90, seed=13)
        recs = recommend(make_context(meshes))
        self.assertEqual(recs["veh_footbrake"]["mode"], core.MODE_TRANSLATE)

    def test_pedals_translate_and_mirror_named_part_in_cone_translates(self) -> None:
        meshes = base_cabin()
        meshes["veh_pedalbox"] = box_cloud((0.40, -0.82, 0.55), (0.22, 0.20, 0.26), n=110, seed=14)
        # the old lexical case: a "mirror" token must not force Mirror when
        # the cloud sits squarely in the control cone (shift light on the pod)
        meshes["shiftlight_multi_led_mirror"] = box_cloud(
            (0.40, -0.50, 1.06), (0.10, 0.06, 0.05), n=60, seed=15)
        recs = recommend(make_context(meshes))
        self.assertEqual(recs["veh_pedalbox"]["mode"], core.MODE_TRANSLATE)
        self.assertEqual(recs["shiftlight_multi_led_mirror"]["mode"], core.MODE_TRANSLATE)

    def test_pedalbox_footplate_translates_but_passenger_footplate_mirrors(self) -> None:
        meshes = base_cabin()
        meshes["grp_padalbox_footplate"] = box_cloud((0.40, -0.80, 0.35), (0.24, 0.24, 0.08), n=90, seed=16)
        meshes["race_footplate"] = box_cloud((-0.40, -0.80, 0.35), (0.24, 0.24, 0.08), n=90, seed=17)
        recs = recommend(make_context(meshes))
        self.assertEqual(recs["grp_padalbox_footplate"]["mode"], core.MODE_TRANSLATE)
        self.assertEqual(recs["race_footplate"]["mode"], core.MODE_MIRROR)

    def test_passenger_cone_rescans_but_cannot_see_through_a_surface(self) -> None:
        meshes = base_cabin()
        meshes["driver_pedal_cluster"] = box_cloud(
            (0.40, -0.82, 0.45), (0.22, 0.20, 0.20), n=110, seed=51
        )
        plate = box_cloud(
            (-0.40, -0.75, 0.40), (0.24, 0.24, 0.08), n=90, seed=52
        )
        meshes["passenger_blind_plate"] = plate
        # Sparse blocker points at 60% range make the broad point shell report
        # no exposure. The passenger cone must request an exact surface scan,
        # then obey that scan instead of granting x-ray admission.
        eye = np.asarray(EYE, dtype=float)
        meshes["passenger_ray_blocker"] = eye + 0.60 * (plate - eye)
        context = make_context(meshes)
        frame = core.driver_frame_for_context(context)
        present, arrays = tool._spatial_entries_for_trim(context, None, set(meshes))
        first, _vetoed = tool._classify_meshes_for_trim(
            context, frame, present, arrays, present
        )
        self.assertNotIn("passenger_blind_plate", first)
        self.assertEqual(first["driver_pedal_cluster"][0], "translate")
        forced = tool._passenger_footwell_forced(
            frame, present, arrays, first
        )
        self.assertIn("passenger_blind_plate", forced)

        harmless_surface = np.array((
            ((-1.0, -1.0, -10.0), (1.0, -1.0, -10.0), (0.0, 1.0, -10.0)),
        ))
        visible, _ = tool._classify_meshes_for_trim(
            context,
            frame,
            present,
            arrays,
            ["passenger_blind_plate"],
            forced,
            surface_np={"harmless": harmless_surface},
        )
        self.assertEqual(visible["passenger_blind_plate"][0], "pairable")

        cover_triangles = []
        for point in plate:
            direction = point - eye
            direction /= np.linalg.norm(direction)
            reference = np.array((0.0, 0.0, 1.0))
            if abs(float(direction @ reference)) > 0.9:
                reference = np.array((0.0, 1.0, 0.0))
            across = np.cross(direction, reference)
            across /= np.linalg.norm(across)
            upward = np.cross(direction, across)
            centre = eye + 0.60 * (point - eye)
            radius = 0.004
            cover_triangles.append((
                centre + radius * across,
                centre - radius * across + radius * upward,
                centre - radius * across - radius * upward,
            ))
        hidden, _ = tool._classify_meshes_for_trim(
            context,
            frame,
            present,
            arrays,
            ["passenger_blind_plate"],
            forced,
            surface_np={"opaque_cover": np.asarray(cover_triangles)},
        )
        self.assertNotIn("passenger_blind_plate", hidden)


class SymmetryAndPairingTests(unittest.TestCase):
    def test_large_centred_near_symmetry_still_mirrors_exact_orphans(self) -> None:
        meshes = base_cabin()
        meshes["near_symmetric_large"] = sym_cloud(
            (0.0, -0.10, 0.80), (1.20, 1.20, 0.80), n=300, seed=70
        ) + np.array((0.002, 0.0, 0.0))
        context = make_context(meshes)

        recs = recommend(context)
        orphans, coarse = tool._mesh_symmetry(
            context,
            "near_symmetric_large",
            meshes["near_symmetric_large"],
            0.0,
        )

        self.assertGreater(orphans, 0)
        self.assertEqual(coarse, 0.0)
        self.assertEqual(recs["near_symmetric_large"]["mode"], core.MODE_MIRROR)

    def test_centred_asymmetric_mesh_is_aesthetic_not_pairable(self) -> None:
        meshes = base_cabin()
        centred = box_cloud(
            (0.0, -0.48, 0.72), (0.40, 0.10, 0.18), n=120, seed=61
        )
        # Uneven tessellation pulls the point centroid well beyond 5 cm even
        # though the geometric bbox remains centred on the vehicle.
        dense_side = centred[centred[:, 0] > 0.05]
        centred = np.concatenate([centred, dense_side, dense_side, dense_side])
        meshes["centred_asymmetric"] = centred
        context = make_context(meshes)

        recs = recommend(context)

        bbox_center_x = (centred[:, 0].min() + centred[:, 0].max()) / 2.0
        self.assertGreater(abs(float(centred[:, 0].mean())), 0.05)
        self.assertLess(abs(float(bbox_center_x)), 0.05)
        self.assertEqual(recs["centred_asymmetric"]["mode"], core.MODE_MIRROR)
        self.assertEqual(
            context._spatial_recommendation_state["memo"]["centred_asymmetric"][0],
            "mirror",
        )

    def test_pairing_rejects_a_centred_latent_twin(self) -> None:
        meshes = {
            "offcentre": box_cloud(
                (0.06, -0.48, 0.72), (0.08, 0.10, 0.18), n=80, seed=62
            ),
            "centred_latent": box_cloud(
                (0.0, -0.48, 0.72), (0.08, 0.10, 0.18), n=80, seed=63
            ),
        }
        context = make_context(meshes)
        frame = core.DriverFrame(
            eye=EYE,
            center_x=0.0,
            side=1,
            forward=(0.0, -1.0, 0.0),
            wheel_id=None,
            wheel_center=None,
            source="fixture",
        )
        entries = {key: np.asarray(value) for key, value in meshes.items()}
        memo = {
            "offcentre": ("pairable", "fixture", "high", {}),
            "centred_latent": ("none", "", "med", {}),
        }
        votes: dict[str, dict[str, int]] = {}

        with patch.object(core, "mirror_pair_residual", return_value=0.0):
            tool._resolve_trim_pairs(
                context,
                frame,
                list(meshes),
                entries,
                memo,
                set(),
                votes,
            )

        self.assertNotIn("offcentre", votes)
        self.assertEqual(memo["centred_latent"][0], "none")

    def test_front_bench_skips_but_buckets_pair(self) -> None:
        bench_meshes = base_cabin()
        del bench_meshes["veh_seat_FL"]
        del bench_meshes["veh_seat_FR"]
        bench_meshes["veh_bench"] = np.concatenate([
            sym_cloud((0.0, 0.15, 0.45), (1.30, 0.55, 0.20), n=140, seed=18),
            sym_cloud((0.0, 0.44, 0.80), (1.30, 0.16, 0.85), n=140, seed=36),
        ])
        bench_recs = recommend(make_context(bench_meshes))
        self.assertNotIn("veh_bench", bench_recs)

        bucket_recs = recommend(make_context(base_cabin()))
        self.assertEqual(bucket_recs["veh_seat_FL"]["mode"], core.MODE_MIRROR_STRUCTURAL)
        self.assertEqual(bucket_recs["veh_seat_FL"]["source_id"], "veh_seat_FR")

    def test_rear_bench_with_R_suffix_is_judged_by_geometry_not_name(self) -> None:
        # etk800_seats_R: R means rear, not right -- the symmetric cloud
        # decides, the suffix is irrelevant
        meshes = base_cabin()
        meshes["veh_seats_R"] = sym_cloud((0.0, 1.05, 0.75), (1.30, 0.70, 0.80), n=240, seed=19)
        recs = recommend(make_context(meshes))
        self.assertNotIn("veh_seats_R", recs)

    def test_one_sided_underseat_hardware_mirrors(self) -> None:
        meshes = base_cabin()
        meshes["racingseat_base"] = box_cloud((0.42, 0.15, 0.30), (0.30, 0.30, 0.08), n=110, seed=20)
        recs = recommend(make_context(meshes))
        self.assertEqual(recs["racingseat_base"]["mode"], core.MODE_MIRROR)
        self.assertEqual(recs["racingseat_base"]["source_id"], "")

    def test_driver_seat_is_transparent_to_paired_and_lone_bases(self) -> None:
        meshes = base_cabin()
        driver_base = box_cloud(
            (0.40, 0.15, 0.30), (0.30, 0.30, 0.08), n=110, seed=44
        )
        meshes["fixture_base_FL"] = driver_base
        meshes["fixture_base_FR"] = mirror_x(driver_base)
        cabin_ids = [
            object_id for object_id in base_cabin()
            if object_id not in {"veh_seat_FR"}
        ]

        paired = make_context(
            meshes,
            trims={"two_seat": list(meshes)},
        )
        paired_recs = recommend(paired)
        pair = paired_recs.get("fixture_base_FL") or paired_recs.get("fixture_base_FR")
        self.assertIsNotNone(pair)
        self.assertEqual(pair["mode"], core.MODE_MIRROR_STRUCTURAL)

        single = make_context(
            meshes,
            trims={"single_seat": cabin_ids + ["fixture_base_FL"]},
        )
        single_recs = recommend(single)
        self.assertEqual(single_recs["fixture_base_FL"]["mode"], core.MODE_MIRROR)
        self.assertIn("twin absent", single_recs["fixture_base_FL"]["reason"])

    def test_variant_dependent_none_is_retried_at_later_placement(self) -> None:
        meshes = base_cabin()
        meshes["moving_fixture"] = sym_cloud(
            (0.0, -0.45, 0.82), (0.28, 0.12, 0.18), n=100, seed=60
        )
        ids = list(meshes)
        context = make_context(
            meshes,
            trims={"a_centred": ids, "b_offset": ids},
        )
        context.variant_dependent_meshes.add("moving_fixture")
        context.resolved_positions_cache["a_centred"]["moving_fixture"] = (
            core.ResolvedMeshPosition((0.0, -0.45, 0.82))
        )
        context.resolved_positions_cache["b_offset"]["moving_fixture"] = (
            core.ResolvedMeshPosition((-0.40, -0.45, 0.82))
        )

        recs = recommend(context)
        self.assertEqual(recs["moving_fixture"]["mode"], core.MODE_MIRROR)

    def test_mutually_exclusive_variants_do_not_pair_but_cotrim_twins_do(self) -> None:
        # recast of the lhd/rhd-token case: two interior-mirror variants are
        # geometric reflections of each other, but they never coexist in a
        # trim, so they must NOT become a structural pair; each lone variant
        # is still one-sided hardware worth mirroring.
        cabin = base_cabin()
        lhd = box_cloud((0.12, -0.30, 1.24), (0.16, 0.08, 0.08), n=80, seed=21)
        meshes = dict(cabin)
        meshes["veh_intmirror_lhd"] = lhd
        meshes["veh_intmirror_rhd"] = mirror_x(lhd)
        cabin_ids = list(cabin)
        split = make_context(
            meshes,
            trims={
                "A": cabin_ids + ["veh_intmirror_lhd"],
                "B": cabin_ids + ["veh_intmirror_rhd"],
            },
        )
        recs = recommend(split)
        self.assertEqual(recs["veh_intmirror_lhd"]["mode"], core.MODE_MIRROR)
        self.assertEqual(recs["veh_intmirror_lhd"]["source_id"], "")
        self.assertEqual(recs["veh_intmirror_rhd"]["mode"], core.MODE_MIRROR)

        # the same two meshes in ONE trim are genuine twins and must pair
        cotrim = make_context(
            meshes,
            trims={"A": cabin_ids + ["veh_intmirror_lhd", "veh_intmirror_rhd"]},
        )
        recs = recommend(cotrim)
        pair = recs.get("veh_intmirror_lhd") or recs.get("veh_intmirror_rhd")
        self.assertIsNotNone(pair)
        self.assertEqual(pair["mode"], core.MODE_MIRROR_STRUCTURAL)

    def test_wing_mirrors_and_door_glass_pair_by_geometry(self) -> None:
        meshes = base_cabin()
        glass_fl = box_cloud((0.76, -0.05, 1.10), (0.03, 0.75, 0.32), n=120, seed=22)
        wing_l = box_cloud((0.90, -0.42, 1.00), (0.14, 0.14, 0.12), n=110, seed=23)
        meshes["veh_doorglass_FL"] = glass_fl
        meshes["veh_doorglass_FR"] = mirror_x(glass_fl)
        meshes["veh_wing_L"] = wing_l
        meshes["veh_wing_R"] = mirror_x(wing_l)
        context = make_context(
            meshes,
            materials={
                "veh_doorglass_FL": ("veh_glass",),
                "veh_doorglass_FR": ("veh_glass",),
            },
            material_flags={"veh_glass": {"glass": True, "emissive": False}},
        )
        recs = recommend(context)
        glass_pair = recs.get("veh_doorglass_FL") or recs.get("veh_doorglass_FR")
        self.assertIsNotNone(glass_pair)
        self.assertEqual(glass_pair["mode"], core.MODE_MIRROR_STRUCTURAL)
        self.assertEqual(
            {glass_pair["object_id"], glass_pair["source_id"]},
            {"veh_doorglass_FL", "veh_doorglass_FR"},
        )
        pair = recs.get("veh_wing_L") or recs.get("veh_wing_R")
        self.assertIsNotNone(pair)
        self.assertEqual(pair["mode"], core.MODE_MIRROR_STRUCTURAL)
        self.assertEqual(
            {pair["object_id"], pair["source_id"]}, {"veh_wing_L", "veh_wing_R"}
        )


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

    def test_geometric_twins_with_different_materials_are_protected_skips(self) -> None:
        meshes = base_cabin()
        left = box_cloud((0.90, -0.42, 1.00), (0.14, 0.14, 0.12), n=110, seed=55)
        meshes["functional_L"] = left
        meshes["functional_R"] = mirror_x(left)
        context = make_context(
            meshes,
            materials={
                "functional_L": ("signal_l",),
                "functional_R": ("signal_r",),
            },
        )
        recs = recommend(context)
        self.assertNotIn("functional_L", recs)
        self.assertNotIn("functional_R", recs)
        memo = context._spatial_recommendation_state["memo"]
        for object_id in ("functional_L", "functional_R"):
            self.assertEqual(memo[object_id][0], "functional_skip")
            self.assertIn("needs build-side material rebind", memo[object_id][1])

    def test_directional_display_mirrors_with_texture_flip(self) -> None:
        meshes = base_cabin()
        meshes["veh_screen"] = box_cloud((0.02, -0.47, 0.95), (0.20, 0.02, 0.08), n=30, seed=24)
        context = make_context(
            meshes,
            materials={"veh_screen": ("veh_screen_mat",)},
            material_flags={"veh_screen_mat": {"glass": False, "emissive": True}},
        )
        recs = recommend(context)
        self.assertEqual(recs["veh_screen"]["mode"], core.MODE_MIRROR)
        self.assertTrue(recs["veh_screen"].get("textureFlip"))

    def test_cluster_screen_in_cone_translates_and_windscreen_skips(self) -> None:
        meshes = base_cabin()
        meshes["veh_gauges_screen"] = box_cloud((0.40, -0.53, 0.95), (0.20, 0.02, 0.10), n=30, seed=25)
        meshes["veh_windscreen"] = sym_cloud(
            (0.0, -0.78, 1.15), (1.45, 0.06, 0.40), n=140, seed=26
        )
        context = make_context(
            meshes,
            materials={
                "veh_gauges_screen": ("veh_screen_mat",),
                "veh_windscreen": ("veh_glass",),
            },
            material_flags={
                "veh_screen_mat": {"glass": False, "emissive": True},
                "veh_glass": {"glass": True, "emissive": False},
            },
        )
        recs = recommend(context)
        self.assertEqual(recs["veh_gauges_screen"]["mode"], core.MODE_TRANSLATE)
        self.assertNotIn("textureFlip", recs["veh_gauges_screen"])
        self.assertNotIn("veh_windscreen", recs)


class SightlineInheritanceTests(unittest.TestCase):
    def test_contact_inheritance_allows_submillimetre_threshold_dust(self) -> None:
        meshes = base_cabin()
        host = np.array((
            (0.30, -0.60, 0.70), (0.50, -0.60, 0.70),
            (0.30, -0.40, 0.70), (0.50, -0.40, 0.70),
        ))
        satellite = host + np.array((0.0, 0.0, 0.03002))
        meshes["mirror_host"] = host
        meshes["mounted_satellite"] = satellite
        context = make_context(meshes)
        frame = core.driver_frame_for_context(context)
        present, arrays = tool._spatial_entries_for_trim(context, None, set(meshes))
        memo = {object_id: ("none", "", "med", {}) for object_id in present}
        memo["mirror_host"] = ("mirror", "fixture", "high", {})

        tool._inherit_mounted_parts(
            context,
            frame,
            present,
            arrays,
            memo,
            set(),
            {},
            {"mounted_satellite"},
        )

        self.assertEqual(memo["mounted_satellite"][0], "mirror")
        self.assertIn("mounted on mirror_host", memo["mounted_satellite"][1])

    def test_unscoped_contact_outside_cabin_volume_cannot_inherit(self) -> None:
        meshes = base_cabin()
        host = np.array((
            (0.30, -1.10, 0.70), (0.50, -1.10, 0.70),
            (0.30, -1.00, 0.70), (0.50, -1.00, 0.70),
        ))
        meshes["mirror_host"] = host
        meshes["outside_scope"] = host + np.array((0.0, 0.0, 0.01))
        context = make_context(meshes)
        frame = core.driver_frame_for_context(context)
        present, arrays = tool._spatial_entries_for_trim(context, None, set(meshes))
        memo = {object_id: ("none", "", "med", {}) for object_id in present}
        memo["mirror_host"] = ("mirror", "fixture", "high", {})

        tool._inherit_mounted_parts(
            context, frame, present, arrays, memo, set(), {}, set()
        )

        self.assertEqual(memo["outside_scope"][0], "none")

    def test_unscoped_contact_inside_cabin_volume_can_inherit(self) -> None:
        meshes = base_cabin()
        host = np.array((
            (0.30, -0.60, 0.70), (0.50, -0.60, 0.70),
            (0.30, -0.40, 0.70), (0.50, -0.40, 0.70),
        ))
        meshes["mirror_host"] = host
        meshes["hidden_button"] = host + np.array((0.0, 0.0, 0.01))
        context = make_context(meshes)
        frame = core.driver_frame_for_context(context)
        present, arrays = tool._spatial_entries_for_trim(context, None, set(meshes))
        memo = {object_id: ("none", "", "med", {}) for object_id in present}
        memo["mirror_host"] = ("mirror", "fixture", "high", {})

        tool._inherit_mounted_parts(
            context, frame, present, arrays, memo, set(), {}, set()
        )

        self.assertEqual(memo["hidden_button"][0], "mirror")

    def test_floating_scoped_mesh_in_front_of_translate_inherits_mirror(self) -> None:
        meshes = base_cabin()
        host = box_cloud((0.40, -0.62, 0.78), (0.30, 0.16, 0.24), n=100, seed=53)
        eye = np.asarray(EYE, dtype=float)
        floater = eye + 0.55 * (host - eye)
        dummy = eye + 0.40 * (
            box_cloud((0.40, -0.62, 0.78), (0.05, 0.04, 0.04), n=20, seed=54) - eye
        )
        meshes["translated_host"] = host
        meshes["floating_pole"] = floater
        meshes["SPOTLIGHT"] = dummy
        context = make_context(meshes)
        frame = core.driver_frame_for_context(context)
        present, arrays = tool._spatial_entries_for_trim(context, None, set(meshes))

        backing = core.directional_verdict_backing(
            arrays,
            frame.eye,
            ["floating_pole"],
            {"translated_host": "translate"},
        )
        self.assertGreater(backing["floating_pole"]["translate"], 0.0)

        memo = {
            object_id: ("none", "", "med", {}) for object_id in present
        }
        memo["translated_host"] = ("translate", "fixture", "high", {})
        tool._inherit_mounted_parts(
            context,
            frame,
            present,
            arrays,
            memo,
            set(),
            {},
            {"floating_pole", "SPOTLIGHT"},
        )
        self.assertEqual(memo["floating_pole"][0], "mirror")
        self.assertIn("translate geometry", memo["floating_pole"][1])
        self.assertEqual(memo["SPOTLIGHT"][0], "none")

    def test_rearward_floater_does_not_inherit_from_sightline(self) -> None:
        meshes = base_cabin()
        host = box_cloud((0.40, 1.05, 0.78), (0.30, 0.16, 0.24), n=100, seed=59)
        eye = np.asarray(EYE, dtype=float)
        floater = eye + 0.55 * (host - eye)
        meshes["rear_transformed_host"] = host
        meshes["rear_floating_lamp"] = floater
        context = make_context(meshes)
        frame = core.driver_frame_for_context(context)
        present, arrays = tool._spatial_entries_for_trim(context, None, set(meshes))
        memo = {
            object_id: ("none", "", "med", {}) for object_id in present
        }
        memo["rear_transformed_host"] = ("mirror", "fixture", "high", {})

        tool._inherit_mounted_parts(
            context,
            frame,
            present,
            arrays,
            memo,
            set(),
            {},
            {"rear_floating_lamp"},
        )

        self.assertEqual(memo["rear_floating_lamp"][0], "none")


class ResolutionFloorTests(unittest.TestCase):
    def test_column_top_translates_but_column_body_and_rack_mirror(self) -> None:
        # finer than spatial resolution: the split rides on the one
        # sanctioned name hint and must come back low-confidence
        meshes = base_cabin()
        meshes["veh_steering_column_top"] = box_cloud(
            (0.40, -0.62, 0.72), (0.04, 0.45, 0.05), n=70, seed=27)
        meshes["veh_steering_column"] = box_cloud(
            (0.37, -0.80, 0.48), (0.05, 0.40, 0.06), n=70, seed=28)
        meshes["veh_steering_column_race_rack"] = box_cloud(
            (0.35, -0.76, 0.45), (0.05, 0.30, 0.05), n=70, seed=29)
        recs = recommend(make_context(meshes))
        self.assertEqual(recs["veh_steering_column_top"]["mode"], core.MODE_TRANSLATE)
        self.assertEqual(recs["veh_steering_column"]["mode"], core.MODE_MIRROR)
        self.assertEqual(recs["veh_steering_column_race_rack"]["mode"], core.MODE_MIRROR)
        for object_id in (
            "veh_steering_column_top",
            "veh_steering_column",
            "veh_steering_column_race_rack",
        ):
            self.assertEqual(recs[object_id].get("confidence"), "low")


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
