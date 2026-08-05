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
from pathlib import Path
from unittest import mock

from mesh_segmentation_transform.annotate_texture_tuning_app import (
    BOX_SOURCE_FOR_SOURCE,
    CHOICE_PARAMETERS,
    HIDDEN_PARAMETERS,
    PARAMETER_SECTIONS,
    SOURCE_COLOUR,
    SOURCE_RELIEF,
    load_session,
    save_session,
)
from mesh_segmentation_transform.annotate_texture_regions import (
    DEFAULT_CONFIG,
    DEFAULT_RELIEF_DETECTION_CONFIG,
    MserConfig,
    PARAMETER_STEP,
    SHAPE_ROTATED,
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

    def test_per_source_parameters_survive(self) -> None:
        save_session(
            {
                "vehicle": "C:/vehicles/vivace.zip",
                "parameters": {
                    SOURCE_COLOUR: {"mser:delta": "12"},
                    SOURCE_RELIEF: {"mser:delta": "7", "relief:mode": "shaded"},
                },
            }
        )
        parameters = load_session()["parameters"]
        self.assertEqual(parameters[SOURCE_COLOUR]["mser:delta"], "12")
        self.assertEqual(parameters[SOURCE_RELIEF]["mser:delta"], "7")
        self.assertEqual(parameters[SOURCE_RELIEF]["relief:mode"], "shaded")

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
    def test_each_detection_source_maps_to_one_front_end(self) -> None:
        # MSER finds regions of uniform intensity, which is what print is;
        # relief has none, only the edges where the surface steps.
        self.assertEqual(BOX_SOURCE_FOR_SOURCE[SOURCE_COLOUR], "mser")
        self.assertEqual(BOX_SOURCE_FOR_SOURCE[SOURCE_RELIEF], "edge")

    def test_the_box_source_is_never_offered_as_a_choice(self) -> None:
        self.assertNotIn("box_source", CHOICE_PARAMETERS)
        listed = {n for _t, names, _m in PARAMETER_SECTIONS for n in names}
        self.assertNotIn("box_source", listed)
        # Hidden as well as unlisted, or the "Other" catch-all shows it again.
        self.assertIn("box_source", HIDDEN_PARAMETERS)


class SourceDefaultTests(unittest.TestCase):
    def test_colour_and_normal_detection_defaults_are_separate(self) -> None:
        self.assertEqual(DEFAULT_CONFIG.box_source, "mser")
        self.assertEqual(DEFAULT_RELIEF_DETECTION_CONFIG.box_source, "edge")
        self.assertEqual(DEFAULT_RELIEF_DETECTION_CONFIG.edge_operator, "laplacian")
        self.assertTrue(DEFAULT_RELIEF_DETECTION_CONFIG.enable_rotated_bounds_filter)
        self.assertEqual(DEFAULT_RELIEF_DETECTION_CONFIG.bounds_shape, SHAPE_ROTATED)
        self.assertTrue(DEFAULT_RELIEF_DETECTION_CONFIG.enable_edge_aligned_rotation)
        self.assertEqual(DEFAULT_RELIEF_DETECTION_CONFIG.rotation_edge_search_px, 50)
        self.assertEqual(DEFAULT_RELIEF_DETECTION_CONFIG.min_region_relief, 30.0)
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

    def test_the_two_front_ends_are_scoped_to_their_own_source(self) -> None:
        by_title = {title: mode for title, _names, mode in PARAMETER_SECTIONS}
        self.assertEqual(by_title["1. MSER detector"], SOURCE_COLOUR)
        self.assertEqual(by_title["1. Edge detector"], SOURCE_RELIEF)

    def test_shared_stages_apply_to_both_sources(self) -> None:
        by_title = {title: mode for title, _names, mode in PARAMETER_SECTIONS}
        for title in (
            "2. Stroke width",
            "3. Box filtering",
            "4. Overlap grouping",
            "5. Initial grouping",
        ):
            self.assertIsNone(by_title[title], title)


if __name__ == "__main__":
    unittest.main()
