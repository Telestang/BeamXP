"""Session persistence for the tuning harness.

Only the parts that do not need a display: the settings file's schema, and the
rule that the box source follows the detection source rather than being chosen.
The widget behaviour these support -- each source holding its own values, and
sections hiding when they cannot reach the current image -- needs a live Tk
window and is exercised by hand.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest import mock

import numpy as np

from mesh_segmentation_transform.annotate_texture_tuning_app import (
    BOX_SOURCE_FOR_SOURCE,
    CHOICE_PARAMETERS,
    COLOUR_GLYPH_NAMESPACE,
    DETECTION_SOURCES,
    RELIEF_GLYPH_NAMESPACE,
    HIDDEN_PARAMETERS,
    PARAMETER_SECTIONS,
    PIPELINES,
    SOURCE_COLOUR,
    SOURCE_COLOUR_CONTRAST,
    SOURCE_COLOUR_CONTRAST_GPU,
    SOURCE_COLOUR_RELIEF_CONTRAST,
    SOURCE_COLOUR_MSER,
    SOURCE_RELIEF,
    SOURCE_RELIEF_GPU,
    SOURCE_RELIEF_CONTRAST,
    load_session,
    namespaced_parameters,
    pipeline_for_source,
    run_detection_by_uv_island,
    save_session,
)
from mesh_segmentation_transform import extract_uv_island_paths as uv_paths
from mesh_segmentation_transform.annotate_texture_regions import (
    DEFAULT_CONFIG,
    DEFAULT_RELIEF_DETECTION_CONFIG,
    DetectionRun,
    DetectionStage,
    MserConfig,
    PARAMETER_STEP,
    run_detection,
)
from mesh_segmentation_transform.relief_from_normals import DEFAULT_RELIEF_CONFIG


class SessionFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp(prefix="beamxp_session_test_"))
        self.path = self.directory / "session.json"
        patcher = mock.patch(
            "mesh_segmentation_transform.annotate_texture_tuning_app."
            "session_settings_path",
            return_value=self.path,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_round_trip(self) -> None:
        save_session(
            {
                "vehicle": "C:/vehicles/vivace.zip",
                "dae_member": "vehicles/vivace/ardente/ardente.dae",
                "part_filter": "dashboard",
            }
        )
        self.assertEqual(
            load_session(),
            {
                "vehicle": "C:/vehicles/vivace.zip",
                "dae_member": "vehicles/vivace/ardente/ardente.dae",
                "part_filter": "dashboard",
            },
        )

    def test_per_pipeline_parameters_survive_under_stable_ids(self) -> None:
        save_session(
            {
                "vehicle": "C:/vehicles/vivace.zip",
                "pipelines": {
                    "colour_foreground": {"mser:delta": "12"},
                    "relief_edge": {"mser:delta": "7", "relief:mode": "shaded"},
                },
            }
        )
        pipelines = load_session()["pipelines"]
        self.assertEqual(pipelines["colour_foreground"]["mser:delta"], "12")
        self.assertEqual(pipelines["relief_edge"]["mser:delta"], "7")
        self.assertEqual(pipelines["relief_edge"]["relief:mode"], "shaded")

    def test_shared_parameters_round_trip_outside_the_pipelines(self) -> None:
        """UV island symmetry belongs to the texture, not to a detection path."""
        save_session(
            {
                "vehicle": "C:/vehicles/vivace.zip",
                "pipelines": {"colour_foreground": {"mser:delta": "12"}},
                "shared": {"uv:min_uv_island_symmetry": "0.9"},
            }
        )

        session = load_session()

        self.assertEqual(session["shared"], {"uv:min_uv_island_symmetry": "0.9"})
        self.assertEqual(session["pipelines"]["colour_foreground"], {"mser:delta": "12"})

    def test_a_shared_parameter_is_never_written_into_a_pipeline(self) -> None:
        """Otherwise the per-path copy outlives the shared one and wins later."""
        save_session(
            {
                "vehicle": "C:/vehicles/vivace.zip",
                "pipelines": {
                    "colour_foreground": {
                        "mser:delta": "12",
                        "uv:min_uv_island_symmetry": "0.5",
                    }
                },
                "shared": {"uv:min_uv_island_symmetry": "0.9"},
            }
        )

        payload = json.loads(self.path.read_text(encoding="utf-8"))

        self.assertEqual(payload["pipelines"]["colour_foreground"], {"mser:delta": "12"})
        self.assertEqual(payload["shared"], {"uv:min_uv_island_symmetry": "0.9"})

    def test_a_shared_parameter_left_in_an_old_session_is_hoisted(self) -> None:
        """Sessions written before the split kept it per pipeline."""
        self.path.write_text(
            json.dumps(
                {
                    "pipelines": {
                        "colour_foreground": {
                            "mser:delta": "12",
                            "uv:min_uv_island_symmetry": "0.9",
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

        session = load_session()

        self.assertEqual(session["shared"]["uv:min_uv_island_symmetry"], "0.9")

    def test_a_session_with_nothing_shared_grows_no_empty_object(self) -> None:
        save_session({"vehicle": "C:/vehicles/vivace.zip"})
        self.assertNotIn("shared", load_session())

    def test_legacy_display_label_parameters_are_loaded_for_migration(self) -> None:
        self.path.write_text(
            json.dumps({"parameters": {SOURCE_RELIEF: {"mser:delta": "7"}}}),
            encoding="utf-8",
        )
        self.assertEqual(load_session()["pipelines"][SOURCE_RELIEF]["mser:delta"], "7")

    def test_unknown_keys_are_dropped(self) -> None:
        save_session({"vehicle": "C:/x.zip", "something_else": "no"})
        self.assertNotIn("something_else", load_session())

    def test_a_missing_file_is_not_an_error(self) -> None:
        self.assertEqual(load_session(), {})

    def test_a_corrupt_file_is_not_an_error(self) -> None:
        # A settings file is a convenience; a stale or truncated one must never
        # stop the harness opening.
        self.path.write_text("{not json at all", encoding="utf-8")
        self.assertEqual(load_session(), {})

    def test_a_non_object_file_is_not_an_error(self) -> None:
        self.path.write_text("[1, 2, 3]", encoding="utf-8")
        self.assertEqual(load_session(), {})

    def test_saving_is_atomic_enough_to_leave_valid_json(self) -> None:
        save_session({"vehicle": "C:/x.zip"})
        json.loads(self.path.read_text(encoding="utf-8"))
        self.assertFalse(self.path.with_name(self.path.name + ".tmp").exists())

    def test_an_unwritable_location_is_survived(self) -> None:
        with mock.patch(
            "mesh_segmentation_transform.annotate_texture_tuning_app."
            "session_settings_path",
            return_value=Path("/nonexistent-root/deep/session.json"),
        ):
            save_session({"vehicle": "C:/x.zip"})  # must not raise


class BoxSourceRuleTests(unittest.TestCase):
    def test_each_detect_source_has_one_named_pipeline_class_and_stable_id(self) -> None:
        self.assertEqual(len(PIPELINES), len(DETECTION_SOURCES))
        self.assertEqual(len({pipeline.pipeline_id for pipeline in PIPELINES}), len(PIPELINES))
        self.assertTrue(all(type(pipeline) is not type(PIPELINES[0]) for pipeline in PIPELINES[1:]))

    def test_each_detection_source_maps_to_one_front_end(self) -> None:
        self.assertEqual(BOX_SOURCE_FOR_SOURCE[SOURCE_COLOUR], "foreground")
        self.assertEqual(BOX_SOURCE_FOR_SOURCE[SOURCE_COLOUR_MSER], "mser")
        self.assertEqual(BOX_SOURCE_FOR_SOURCE[SOURCE_RELIEF], "edge")
        self.assertEqual(BOX_SOURCE_FOR_SOURCE[SOURCE_RELIEF_GPU], "edge_gpu")
        self.assertEqual(BOX_SOURCE_FOR_SOURCE[SOURCE_COLOUR_RELIEF_CONTRAST], "contrast_gpu")
        self.assertEqual(BOX_SOURCE_FOR_SOURCE[SOURCE_RELIEF_CONTRAST], "contrast")

    def test_colour_relief_pipeline_uses_gpu_for_both_front_end_responses(self) -> None:
        """The hybrid route must not quietly retain either CPU front end."""
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        normal = np.full((8, 8, 3), 128, dtype=np.uint8)
        response = np.ones((8, 8), dtype=np.float32)
        pipeline = pipeline_for_source(SOURCE_COLOUR_RELIEF_CONTRAST)
        with mock.patch(
            "mesh_segmentation_transform.annotate_texture_tuning_app."
            "slope_relief_edge_response_gpu",
            return_value=response,
        ) as gpu:
            detected, barrier = pipeline.prepare_inputs(
                SimpleNamespace(image=image, normal_rgb=normal),
                DEFAULT_RELIEF_CONFIG,
            )
        self.assertEqual(pipeline.box_source, "contrast_gpu")
        self.assertIs(detected, image)
        self.assertIs(barrier, response)
        gpu.assert_called_once_with(normal, DEFAULT_RELIEF_CONFIG)

    def test_the_box_source_is_never_offered_as_a_choice(self) -> None:
        self.assertNotIn("box_source", CHOICE_PARAMETERS)
        listed = {n for _t, names, _m in PARAMETER_SECTIONS for n in names}
        self.assertNotIn("box_source", listed)
        # Hidden as well as unlisted, or the "Other" catch-all shows it again.
        self.assertIn("box_source", HIDDEN_PARAMETERS)


class SourceDefaultTests(unittest.TestCase):
    def test_colour_defaults_match_the_promoted_cached_hybrid_parameters(self) -> None:
        self.assertEqual(DEFAULT_CONFIG.contrast_kernel_px, 7)
        self.assertEqual(DEFAULT_CONFIG.contrast_close_px, 0)
        self.assertEqual(DEFAULT_CONFIG.contrast_min_component_px, 24)
        self.assertEqual(DEFAULT_CONFIG.contrast_merge_gap_px, 0)
        self.assertTrue(DEFAULT_CONFIG.enable_feature_extension_filter)
        self.assertEqual(DEFAULT_CONFIG.feature_extension_context_px, 12)
        self.assertEqual(DEFAULT_CONFIG.feature_extension_reference_extent_px, 25)
        self.assertEqual(DEFAULT_CONFIG.feature_extension_grace_px, 4)
        self.assertEqual(DEFAULT_CONFIG.feature_extension_grace_max_fraction, 0.20)
        self.assertEqual(DEFAULT_CONFIG.feature_extension_soft_fringe_ratio, 0.75)
        # Promoted 2026-08-16 from the colour+relief-edge session, along with
        # min_region_uv_coverage below.
        self.assertEqual(DEFAULT_CONFIG.feature_extension_min_ratio, 0.03)
        self.assertEqual(DEFAULT_CONFIG.min_region_uv_coverage, 0.98)

    def test_colour_and_normal_detection_defaults_are_separate(self) -> None:
        self.assertEqual(DEFAULT_CONFIG.box_source, "foreground")
        self.assertEqual(DEFAULT_RELIEF_DETECTION_CONFIG.box_source, "edge")
        self.assertEqual(DEFAULT_RELIEF_DETECTION_CONFIG.edge_operator, "laplacian")
        self.assertTrue(DEFAULT_RELIEF_DETECTION_CONFIG.enable_rotated_bounds_filter)
        self.assertTrue(DEFAULT_RELIEF_DETECTION_CONFIG.enable_symmetry_rotation)
        self.assertEqual(DEFAULT_RELIEF_DETECTION_CONFIG.min_box_width_px, 8)
        self.assertEqual(DEFAULT_RELIEF_DETECTION_CONFIG.min_box_height_px, 8)
        self.assertEqual(DEFAULT_RELIEF_DETECTION_CONFIG.merge_distance_px, 21)
        self.assertEqual(DEFAULT_RELIEF_DETECTION_CONFIG.min_region_relief, 5.0)
        # Measured on the GPU edge response and shared with the CPU relief
        # paths: one preset serves every relief front end, so these hold for
        # "Relief (normal map)" and its GPU twin alike.
        self.assertEqual(DEFAULT_RELIEF_DETECTION_CONFIG.region_flatness_percentile, 60.0)
        self.assertEqual(DEFAULT_RELIEF_DETECTION_CONFIG.min_rotation_symmetry, 0.7)
        self.assertEqual(DEFAULT_RELIEF_DETECTION_CONFIG.min_rotated_elongation, 1.0)
        self.assertEqual(DEFAULT_RELIEF_DETECTION_CONFIG.ring_smoothness_width_px, 1)
        self.assertEqual(DEFAULT_RELIEF_DETECTION_CONFIG.ring_smoothness_percentile, 20.0)
        self.assertTrue(DEFAULT_RELIEF_DETECTION_CONFIG.enable_relief_glyph_filter)
        self.assertTrue(DEFAULT_RELIEF_DETECTION_CONFIG.enable_relief_text_filter)
        self.assertFalse(DEFAULT_RELIEF_DETECTION_CONFIG.enable_feature_extension_filter)
        self.assertEqual(DEFAULT_RELIEF_DETECTION_CONFIG.final_max_aspect, 20.0)

    def test_normal_relief_render_defaults_match_the_cached_normal_source(self) -> None:
        self.assertEqual(DEFAULT_RELIEF_CONFIG.mode, "slope")
        self.assertTrue(DEFAULT_RELIEF_CONFIG.invert)

    def test_partial_saved_source_values_do_not_inherit_the_other_source(self) -> None:
        from mesh_segmentation_transform.annotate_texture_tuning_app import TuningApp

        app = object.__new__(TuningApp)
        app.parameter_vars = {
            "delta": object(),
            "edge_operator": object(),
            "enable_blob_shape_filter": object(),
            "min_blob_region_area_px": object(),
        }
        app.symmetry_parameter_vars = {}
        app.relief_parameter_vars = {}
        app.rhd_parameter_vars = {}
        app.mode_parameters = {
            SOURCE_COLOUR: {
                "mser:enable_blob_shape_filter": True,
                "mser:min_blob_region_area_px": "99",
            },
            SOURCE_RELIEF: {
                "mser:delta": "7",
            },
        }

        values = app._parameters_for_source(SOURCE_RELIEF)
        config = app._mser_config_from_values(values, SOURCE_RELIEF)

        self.assertEqual(config.delta, 7)
        self.assertEqual(config.edge_operator, "laplacian")
        self.assertEqual(
            config.enable_blob_shape_filter,
            DEFAULT_RELIEF_DETECTION_CONFIG.enable_blob_shape_filter,
        )
        self.assertEqual(
            config.min_blob_region_area_px,
            DEFAULT_RELIEF_DETECTION_CONFIG.min_blob_region_area_px,
        )

    def test_the_colour_glyph_paths_read_one_shared_parameter_namespace(self) -> None:
        """Local contrast CPU, GPU and relief-edge grouping are one tuning."""
        from mesh_segmentation_transform.annotate_texture_tuning_app import TuningApp

        app = object.__new__(TuningApp)
        app.parameter_vars = {"contrast_kernel_px": object()}
        app.symmetry_parameter_vars = {}
        app.relief_parameter_vars = {}
        app.rhd_parameter_vars = {}
        app.mode_parameters = {COLOUR_GLYPH_NAMESPACE: {"mser:contrast_kernel_px": "9"}}

        for source in (
            SOURCE_COLOUR_CONTRAST, SOURCE_COLOUR_CONTRAST_GPU, SOURCE_COLOUR_RELIEF_CONTRAST,
        ):
            values = app._parameters_for_source(source)
            self.assertEqual(values["mser:contrast_kernel_px"], "9", source)

        # A path outside the namespace is unaffected, or the shared object would
        # be a second set of global defaults rather than one path's tuning.
        self.assertEqual(
            app._parameters_for_source(SOURCE_COLOUR)["mser:contrast_kernel_px"],
            DEFAULT_CONFIG.contrast_kernel_px,
        )

    def test_the_normal_map_paths_read_one_shared_parameter_namespace(self) -> None:
        """CPU and GPU relief edge are one edge detector over one render."""
        from mesh_segmentation_transform.annotate_texture_tuning_app import TuningApp

        app = object.__new__(TuningApp)
        app.parameter_vars = {"min_rotation_symmetry": object()}
        app.symmetry_parameter_vars = {}
        app.relief_parameter_vars = {}
        app.rhd_parameter_vars = {}
        app.mode_parameters = {RELIEF_GLYPH_NAMESPACE: {"mser:min_rotation_symmetry": "0.55"}}

        for source in (SOURCE_RELIEF, SOURCE_RELIEF_GPU):
            values = app._parameters_for_source(source)
            self.assertEqual(values["mser:min_rotation_symmetry"], "0.55", source)

        # The slope path renders the same relief but reaches the local-contrast
        # front end, so its detector tuning stays its own.
        self.assertEqual(
            app._parameters_for_source(SOURCE_RELIEF_CONTRAST)["mser:min_rotation_symmetry"],
            DEFAULT_RELIEF_DETECTION_CONFIG.min_rotation_symmetry,
        )

    def test_every_shared_path_saves_into_its_namespace(self) -> None:
        shared = {
            SOURCE_COLOUR_CONTRAST: COLOUR_GLYPH_NAMESPACE,
            SOURCE_COLOUR_CONTRAST_GPU: COLOUR_GLYPH_NAMESPACE,
            SOURCE_COLOUR_RELIEF_CONTRAST: COLOUR_GLYPH_NAMESPACE,
            SOURCE_RELIEF: RELIEF_GLYPH_NAMESPACE,
            SOURCE_RELIEF_GPU: RELIEF_GLYPH_NAMESPACE,
        }
        for source, namespace in shared.items():
            self.assertEqual(
                pipeline_for_source(source).parameter_namespace, namespace, source,
            )
        # Everything else keeps its own object under its own id.  The slope
        # relief path subclasses the CPU edge one, so this also guards against
        # it inheriting a namespace it should not be in.
        for pipeline in PIPELINES:
            if pipeline.label not in shared:
                self.assertEqual(pipeline.parameter_namespace, pipeline.pipeline_id)


class ParameterNamespaceLoadTests(unittest.TestCase):
    """How saved objects collapse onto the namespace each path reads."""

    def test_a_shared_namespace_keeps_its_own_saved_object(self) -> None:
        grouped = namespaced_parameters(
            {
                "colour_local_contrast_gpu": {"mser:contrast_kernel_px": "5"},
                COLOUR_GLYPH_NAMESPACE: {"mser:contrast_kernel_px": "7"},
            }
        )

        self.assertEqual(grouped[COLOUR_GLYPH_NAMESPACE]["mser:contrast_kernel_px"], "7")

    def test_a_stale_member_object_never_wins_whatever_the_key_order(self) -> None:
        """A path not opened since the namespace was shared must not overwrite it."""
        grouped = namespaced_parameters(
            {
                COLOUR_GLYPH_NAMESPACE: {"mser:contrast_kernel_px": "7"},
                "colour_local_contrast_cpu": {"mser:contrast_kernel_px": "5"},
                "colour_local_contrast_gpu": {"mser:contrast_kernel_px": "3"},
            }
        )

        self.assertEqual(grouped[COLOUR_GLYPH_NAMESPACE]["mser:contrast_kernel_px"], "7")
        self.assertNotIn("colour_local_contrast_cpu", grouped)
        self.assertNotIn("colour_local_contrast_gpu", grouped)

    def test_a_member_object_seeds_a_namespace_nothing_has_written(self) -> None:
        grouped = namespaced_parameters(
            {"colour_local_contrast_gpu": {"mser:contrast_kernel_px": "5"}}
        )

        self.assertEqual(grouped[COLOUR_GLYPH_NAMESPACE]["mser:contrast_kernel_px"], "5")

    def test_unshared_paths_keep_their_own_objects(self) -> None:
        grouped = namespaced_parameters(
            {
                "colour_foreground": {"mser:delta": "12"},
                "colour_mser": {"mser:delta": "7"},
                "relief_local_contrast_cpu": {"mser:delta": "9"},
            }
        )

        self.assertEqual(grouped["colour_foreground"]["mser:delta"], "12")
        self.assertEqual(grouped["colour_mser"]["mser:delta"], "7")
        self.assertEqual(grouped["relief_local_contrast_cpu"]["mser:delta"], "9")

    def test_the_cpu_edge_object_seeds_the_relief_namespace(self) -> None:
        """The GPU path's object is the namespace; the CPU one only seeds it."""
        self.assertEqual(
            namespaced_parameters({"relief_edge": {"mser:delta": "7"}}),
            {RELIEF_GLYPH_NAMESPACE: {"mser:delta": "7"}},
        )
        grouped = namespaced_parameters(
            {
                "relief_edge": {"mser:delta": "7"},
                RELIEF_GLYPH_NAMESPACE: {"mser:delta": "12"},
            }
        )
        self.assertEqual(grouped, {RELIEF_GLYPH_NAMESPACE: {"mser:delta": "12"}})

    def test_unknown_and_malformed_keys_are_dropped(self) -> None:
        self.assertEqual(namespaced_parameters({"gone_pipeline": {"mser:delta": "1"}}), {})
        self.assertEqual(namespaced_parameters({"colour_foreground": "not a dict"}), {})
        self.assertEqual(namespaced_parameters(None), {})


class SharedParameterTests(unittest.TestCase):
    """UV island symmetry is one value for every detection path at once.

    It is read off the UV mask, no detector touches it, and a threshold that
    describes an island describes it whichever image is being searched.  Held
    per pipeline it silently reverted whenever the source changed.
    """

    def _app(self):
        from mesh_segmentation_transform.annotate_texture_tuning_app import TuningApp

        app = object.__new__(TuningApp)
        app.parameter_vars = {"delta": FakeVariable("5")}
        app.symmetry_parameter_vars = {"min_uv_island_symmetry": FakeVariable("0.98")}
        app.relief_parameter_vars = {}
        app.rhd_parameter_vars = {}
        app.mode_parameters = {}
        app.session = {}
        app.active_source = SOURCE_COLOUR
        return app

    def test_the_shared_store_is_left_out_of_the_per_pipeline_capture(self) -> None:
        app = self._app()
        app.symmetry_parameter_vars["min_uv_island_symmetry"].set("0.9")

        captured = app._capture_parameters()

        self.assertIn("mser:delta", captured)
        self.assertNotIn("uv:min_uv_island_symmetry", captured)
        self.assertEqual(
            app._capture_shared_parameters(), {"uv:min_uv_island_symmetry": "0.9"}
        )

    def test_switching_source_does_not_disturb_the_shared_widgets(self) -> None:
        app = self._app()
        app.symmetry_parameter_vars["min_uv_island_symmetry"].set("0.9")

        app._apply_parameters({
            "mser:delta": "7",
            "uv:min_uv_island_symmetry": "0.5",
        })

        self.assertEqual(app.parameter_vars["delta"].get(), "7")
        self.assertEqual(
            app.symmetry_parameter_vars["min_uv_island_symmetry"].get(), "0.9"
        )

    def test_the_remembered_shared_value_is_restored_at_startup(self) -> None:
        app = self._app()
        app.session = {"shared": {"uv:min_uv_island_symmetry": "0.87"}}

        app._apply_shared_parameters()

        self.assertEqual(
            app.symmetry_parameter_vars["min_uv_island_symmetry"].get(), "0.87"
        )


class FakeVariable:
    """Stands in for tk.Variable, which needs a live interpreter."""

    def __init__(self, value: object) -> None:
        self._value = value

    def get(self) -> object:
        return self._value

    def set(self, value: object) -> None:
        self._value = value


class ParameterSectionTests(unittest.TestCase):
    def test_sections_follow_the_pipeline_order(self) -> None:
        # Walking down the panel should walk down the stage tabs.
        listed = [n for _t, names, _m in PARAMETER_SECTIONS for n in names]
        steps = [PARAMETER_STEP[n] for n in listed if n in PARAMETER_STEP]
        self.assertEqual(steps, sorted(steps))

    def test_no_parameter_appears_twice(self) -> None:
        listed = [n for _t, names, _m in PARAMETER_SECTIONS for n in names]
        self.assertEqual(sorted(listed), sorted(set(listed)))

    def test_every_parameter_is_placed_or_deliberately_hidden(self) -> None:
        from dataclasses import fields

        listed = {n for _t, names, _m in PARAMETER_SECTIONS for n in names}
        missing = [
            field.name
            for field in fields(MserConfig)
            if field.name not in listed and field.name not in HIDDEN_PARAMETERS
        ]
        self.assertEqual(missing, [])

    def test_the_front_ends_are_scoped_to_their_own_source(self) -> None:
        by_title = {title: mode for title, _names, mode in PARAMETER_SECTIONS}
        self.assertEqual(by_title["1. Foreground-mask detector"], SOURCE_COLOUR)
        self.assertEqual(by_title["1. MSER detector (comparison)"], SOURCE_COLOUR_MSER)
        self.assertEqual(
            by_title["1. Edge detector"], (SOURCE_RELIEF, SOURCE_RELIEF_GPU),
        )
        self.assertEqual(
            by_title["1. Local-contrast detector"],
            (
                SOURCE_COLOUR_CONTRAST,
                SOURCE_COLOUR_CONTRAST_GPU,
                SOURCE_COLOUR_RELIEF_CONTRAST,
                SOURCE_RELIEF_CONTRAST,
            ),
        )
        self.assertEqual(BOX_SOURCE_FOR_SOURCE[SOURCE_COLOUR_CONTRAST], "contrast")
        self.assertEqual(BOX_SOURCE_FOR_SOURCE[SOURCE_COLOUR_CONTRAST_GPU], "contrast_gpu")


    def test_shared_stages_apply_to_both_sources(self) -> None:
        by_title = {title: mode for title, _names, mode in PARAMETER_SECTIONS}
        for title in (
            "2. Stroke width",
            "3. Box filtering",
            "5. Initial grouping",
        ):
            self.assertIsNone(by_title[title], title)
        self.assertNotIn("4. Overlap grouping", by_title)


class PerIslandHarnessTests(unittest.TestCase):
    def test_tight_island_raster_matches_the_full_atlas_exactly(self) -> None:
        triangles = [
            [(0.10, 0.10), (0.55, 0.10), (0.10, 0.60)],
            [(0.55, 0.10), (0.55, 0.60), (0.10, 0.60)],
        ]
        full, _ = uv_paths._rasterise_triangles(triangles, 64, 64)
        crop = uv_paths._rasterise_triangles_crop(triangles, 64, 64)
        rebuilt = np.zeros((64, 64), dtype=bool)
        assert crop is not None
        height, width = crop.mask.shape
        rebuilt[crop.y:crop.y + height, crop.x:crop.x + width] = crop.mask

        self.assertTrue(np.array_equal(rebuilt, full > 0))

    def test_exact_tile_range_matches_the_previous_neighbourhood_raster(self) -> None:
        triangles = [
            [(0.10, 0.10), (0.55, 0.10), (0.10, 0.60)],
            [(-0.20, 0.25), (0.20, 0.25), (0.20, 0.75)],
            [(0.85, 0.85), (1.15, 0.85), (0.85, 1.15)],
        ]
        reference = np.zeros((64, 64), dtype=np.uint8)
        for triangle in triangles:
            min_u, max_u = min(u for u, _v in triangle), max(u for u, _v in triangle)
            min_v, max_v = min(v for _u, v in triangle), max(v for _u, v in triangle)
            for shift_u in range(int(np.floor(min_u)) - 1, int(np.floor(max_u)) + 2):
                for shift_v in range(int(np.floor(min_v)) - 1, int(np.floor(max_v)) + 2):
                    clipped = uv_paths._clip_to_unit_tile(
                        [(u - shift_u, v - shift_v) for u, v in triangle]
                    )
                    if len(clipped) < 3:
                        continue
                    points = uv_paths._uv_polygon_to_pixels(clipped, 64, 64)
                    if uv_paths.cv2.contourArea(points) > 0.01:
                        uv_paths.cv2.fillPoly(reference, [points], 255, lineType=uv_paths.cv2.LINE_8)

        exact, _filled = uv_paths._rasterise_triangles(triangles, 64, 64)
        self.assertTrue(np.array_equal(exact, reference))

    def test_complete_detection_pipeline_runs_in_each_island_crop(self) -> None:
        image = np.zeros((10, 20, 3), dtype=np.uint8)
        first = np.zeros((10, 20), dtype=bool)
        second = np.zeros((10, 20), dtype=bool)
        first[2:6, 1:5] = True
        second[2:6, 12:18] = True

        def fake_run(crop, mask, config, **_kwargs):
            stage = DetectionStage(
                key="boxes", title="Boxes", kept=((0, 0, crop.shape[1], crop.shape[0]),)
            )
            return DetectionRun(config=config, stages=[stage], entry_states=[])

        with mock.patch(
            "mesh_segmentation_transform.annotate_texture_tuning_app.run_detection",
            side_effect=fake_run,
        ) as detect:
            run = run_detection_by_uv_island(
                image, (first, second), MserConfig(box_source="mser")
            )

        self.assertEqual(detect.call_count, 2)
        self.assertEqual(detect.call_args_list[0].args[0].shape[:2], (4, 4))
        self.assertEqual(detect.call_args_list[1].args[0].shape[:2], (4, 6))
        # These are fitted boxes from the two distinct runs, not their union.
        self.assertEqual(run.stages[0].kept, ((1, 2, 4, 4), (12, 2, 6, 4)))

    def test_near_duplicate_uv_consumers_share_one_detection_pipeline(self) -> None:
        image = np.zeros((32, 48, 3), dtype=np.uint8)
        first = uv_paths.UvIslandCrop(4, 6, np.ones((20, 20), dtype=bool))
        second = uv_paths.UvIslandCrop(5, 6, np.ones((20, 20), dtype=bool))

        def fake_run(crop, mask, config, **_kwargs):
            stage = DetectionStage(
                key="boxes", title="Boxes", kept=((0, 0, crop.shape[1], crop.shape[0]),)
            )
            return DetectionRun(config=config, stages=[stage], entry_states=[])

        with mock.patch(
            "mesh_segmentation_transform.annotate_texture_tuning_app.run_detection",
            side_effect=fake_run,
        ) as detect:
            run = run_detection_by_uv_island(
                image, (first, second), MserConfig(box_source="mser")
            )

        self.assertEqual(detect.call_count, 1)
        self.assertEqual(detect.call_args.args[0].shape[:2], (20, 21))
        self.assertIn("coalesced from 2 topological UV islands", run.stages[0].detail)

    def test_overlap_stage_collapses_intersections_across_detection_domains(self) -> None:
        image = np.zeros((16, 40, 3), dtype=np.uint8)
        first = np.zeros((16, 40), dtype=bool)
        second = np.zeros((16, 40), dtype=bool)
        first[2:12, 0:10] = True
        second[2:12, 20:30] = True
        config = MserConfig(box_source="mser")
        runs = [
            DetectionRun(
                config=config,
                stages=[DetectionStage(
                    key="overlap_box_group",
                    title="Overlap grouping",
                    kept=((0, 0, 25, 10),),
                )],
                entry_states=[],
            ),
            DetectionRun(
                config=config,
                stages=[DetectionStage(
                    key="overlap_box_group",
                    title="Overlap grouping",
                    kept=((0, 0, 10, 10),),
                )],
                entry_states=[],
            ),
        ]

        with mock.patch(
            "mesh_segmentation_transform.annotate_texture_tuning_app.run_detection",
            side_effect=runs,
        ):
            run = run_detection_by_uv_island(image, (first, second), config)

        self.assertEqual(run.stages[0].kept, ((0, 2, 30, 10),))
        self.assertEqual(run.stages[0].adjusted, 1)

    def test_precomputed_foreground_boxes_still_resume_late_filters(self) -> None:
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        mask = np.ones((16, 16), dtype=bool)
        boxes = np.asarray([(2, 2, 8, 6)], dtype=np.int32)
        first = run_detection(image, mask, MserConfig(), initial_boxes=boxes)
        changed = MserConfig(final_region_padding_px=5)
        resumed = run_detection(
            image, mask, changed, previous=first, initial_boxes=boxes
        )

        self.assertEqual(resumed.resumed_from, PARAMETER_STEP["final_region_padding_px"])


if __name__ == "__main__":
    unittest.main()
