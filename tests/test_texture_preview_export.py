from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image

from mesh_segmentation_transform import mirror_texture_for_rhd as rhd
from mesh_segmentation_transform import beamxp_transform_sym_mesh_POC as sym_mesh
from mesh_segmentation_transform.beamxp_transform_sym_mesh_POC import (
    ArchiveMaterialRecord,
    ArchiveMaterialSwitchState,
    ArchiveTextureBinding,
    DaePart,
    GeometryInstance,
    SourceFaceRef,
)


def part(key: str) -> DaePart:
    return DaePart(
        key=key,
        label=key,
        node_id=key,
        node_name=key,
        matrix=np.eye(4),
        instances=(GeometryInstance(f"{key}-mesh"),),
    )


class RuntimeDisplayUvFlipTests(unittest.TestCase):
    def test_selected_runtime_material_resolves_to_dae_bound_aliases(self) -> None:
        archive = SimpleNamespace(
            runtime_material_aliases=("lc500_gps", "lc500_gauges_screen"),
            material_switch_states={
                "lc500_centralscreen": (
                    ArchiveMaterialSwitchState("on_intense", "lc500_gps"),
                )
            },
        )
        with patch.object(
            rhd,
            "material_aliases_for_parts",
            return_value={"lc500_centralscreen", "lc500_gauges_screen"},
        ):
            aliases = rhd.runtime_display_dae_aliases_for_parts(
                archive, SimpleNamespace(), []
            )

        self.assertEqual(aliases, {"lc500_centralscreen", "lc500_gauges_screen"})

    def test_production_opacity_layers_use_direct_foreground_detection(self) -> None:
        colour = rhd._production_layer_detection_config(
            rhd.DEFAULT_CONFIG, True, ("baseColorMap", "emissiveMap")
        )
        opacity = rhd._production_layer_detection_config(
            rhd.DEFAULT_CONFIG, True, ("opacityMap",)
        )
        authoritative = rhd._production_layer_detection_config(
            rhd.DEFAULT_CONFIG,
            True,
            ("baseColorMap",),
            authoritative_opacity_mask=True,
        )

        self.assertEqual(colour.box_source, "contrast_gpu")
        self.assertTrue(colour.enable_feature_extension_filter)
        self.assertEqual(colour.feature_extension_context_px, 12)
        self.assertEqual(colour.feature_extension_min_ratio, 0.25)
        self.assertEqual(opacity.box_source, "opacity_mask")
        self.assertEqual(authoritative.box_source, "opacity_mask")
        self.assertFalse(opacity.enable_region_domain_filter)
        self.assertFalse(authoritative.enable_region_domain_filter)
        self.assertFalse(opacity.enable_feature_extension_filter)
        self.assertFalse(authoritative.enable_feature_extension_filter)

    def test_runtime_display_uv_flip_is_material_scoped(self) -> None:
        namespace = "http://www.collada.org/2005/11/COLLADASchema"
        geometry = ET.fromstring(
            f"""<geometry xmlns="{namespace}" id="screen">
              <mesh>
                <source id="screen-uv">
                  <float_array id="screen-uv-array" count="8">
                    0.1 0.0 0.4 0.0 0.7 0.0 0.9 0.0
                  </float_array>
                  <technique_common>
                    <accessor source="#screen-uv-array" count="4" stride="2">
                      <param name="S" type="float"/>
                      <param name="T" type="float"/>
                    </accessor>
                  </technique_common>
                </source>
                <triangles material="lc500_centralscreen-material" count="1">
                  <input semantic="TEXCOORD" source="#screen-uv" offset="0"/>
                  <p>0 1 0</p>
                </triangles>
                <triangles material="lc500_trim-material" count="1">
                  <input semantic="TEXCOORD" source="#screen-uv" offset="0"/>
                  <p>2 3 2</p>
                </triangles>
              </mesh>
            </geometry>"""
        )

        result = sym_mesh._flip_material_texcoord_islands(
            geometry,
            namespace,
            {"lc500_centralscreen"},
        )

        values = [
            float(value)
            for value in geometry.find(
                f"{{{namespace}}}mesh/{{{namespace}}}source/"
                f"{{{namespace}}}float_array"
            ).text.split()
        ]
        self.assertEqual(values[0::2], [0.4, 0.1, 0.7, 0.9])
        self.assertEqual(result["matched_materials"], ["lc500_centralscreen"])
        self.assertEqual(result["modified_texcoords"], 2)



class SurfaceFlipAxisTests(unittest.TestCase):
    def test_forced_mirror_domain_ignores_symmetric_candidate_labels(self) -> None:
        selected = part("door_source")
        uv = np.asarray(((0.1, 0.1), (0.2, 0.1), (0.1, 0.2)), dtype=float)
        topology = SimpleNamespace(
            vertices=np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
            triangles=np.asarray(((0, 1, 2),)),
            source_faces=(SourceFaceRef(0, "door-mesh", 0, 0),),
        )
        sweep = SimpleNamespace(
            topology=topology,
            candidates=(SimpleNamespace(candidate_id=7, faces=(0,)),),
        )
        with (
            patch.object(
                rhd,
                "uv_triangles_by_source",
                return_value={(0, 0, 0): uv},
            ),
            patch.object(rhd, "sweep_part", return_value=sweep),
        ):
            mirrored, rigid, _surfaces = rhd.split_mirrored_and_rigid(
                SimpleNamespace(),
                selected,
                {"door-material"},
                rhd.RhdTextureConfig(),
            )
            forced, forced_rigid, forced_surfaces = rhd.split_mirrored_and_rigid(
                SimpleNamespace(),
                selected,
                {"door-material"},
                rhd.RhdTextureConfig(),
                force_mirrored=True,
            )

        self.assertEqual(mirrored, [])
        self.assertEqual(len(rigid), 1)
        self.assertEqual(len(forced), 1)
        self.assertEqual(forced_rigid, [])
        self.assertEqual(len(forced_surfaces), 1)

    def test_side_surface_horizontal_flip_when_image_vertical_plane_matches_z(self) -> None:
        uv = np.asarray([[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]])
        xyz = np.asarray(
            [
                [
                    (0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0),
                ]
            ]
        )

        axes = rhd.surface_flip_axes(uv, xyz, width=100, height=100)

        self.assertEqual(axes.tolist(), [rhd.AXIS_HORIZONTAL])

    def test_side_surface_vertical_flip_when_image_horizontal_plane_matches_z(self) -> None:
        uv = np.asarray([[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]])
        xyz = np.asarray(
            [
                [
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0),
                    (1.0, 0.0, 0.0),
                ]
            ]
        )

        axes = rhd.surface_flip_axes(uv, xyz, width=100, height=100)

        self.assertEqual(axes.tolist(), [rhd.AXIS_VERTICAL])

    def test_horizontal_surface_horizontal_flip_when_image_vertical_plane_is_yz(self) -> None:
        uv = np.asarray([[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]])
        xyz = np.asarray(
            [
                [
                    (0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                ]
            ]
        )

        axes = rhd.surface_flip_axes(uv, xyz, width=100, height=100)

        self.assertEqual(axes.tolist(), [rhd.AXIS_HORIZONTAL])

    def test_horizontal_surface_vertical_flip_when_image_horizontal_plane_is_yz(self) -> None:
        uv = np.asarray([[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]])
        xyz = np.asarray(
            [
                [
                    (0.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (1.0, 0.0, 0.0),
                ]
            ]
        )

        axes = rhd.surface_flip_axes(uv, xyz, width=100, height=100)

        self.assertEqual(axes.tolist(), [rhd.AXIS_VERTICAL])

    def test_skew_delta_flags_texture_reflection_that_is_not_a_flat_flip(self) -> None:
        uv = (np.asarray([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]),)
        xyz = (
            np.asarray(
                [
                    (0.0, 0.0, 0.0),
                    (100.0, 0.0, 0.0),
                    (-50.0, -100.0, 0.0),
                ]
            ),
        )
        config = replace(rhd.DEFAULT_RHD_CONFIG, skewed_region_min_delta=0.01)

        delta = rhd.skew_delta_for_region(
            uv, xyz, (20, 20, 30, 30), "horizontal", 100, 100, config
        )

        self.assertIsNotNone(delta)
        self.assertGreater(delta, config.skewed_region_min_delta)

    def test_skew_delta_ignores_axis_aligned_texture_reflection(self) -> None:
        uv = (np.asarray([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]),)
        xyz = (
            np.asarray(
                [
                    (0.0, 0.0, 0.0),
                    (100.0, 0.0, 0.0),
                    (0.0, -100.0, 0.0),
                ]
            ),
        )

        delta = rhd.skew_delta_for_region(
            uv, xyz, (20, 20, 30, 30), "horizontal", 100, 100,
            rhd.DEFAULT_RHD_CONFIG,
        )

        self.assertIsNone(delta)


class CroppedDetectionTests(unittest.TestCase):
    def test_cropped_detection_reports_full_atlas_coordinates(self) -> None:
        bgr = np.zeros((100, 120, 3), dtype=np.uint8)
        mirror = np.zeros((100, 120), dtype=bool)
        mirror[40:60, 50:80] = True
        domain = mirror.copy()
        fake_stage = SimpleNamespace(
            kept=[(10, 8, 4, 5)],
            rotations=[((10.0, 8.0), (14.0, 8.0), (14.0, 13.0), (10.0, 13.0))],
        )
        fake_run = SimpleNamespace(stages=[fake_stage])
        config = rhd.RhdTextureConfig(
            enable_containment_filter=False,
            enable_blob_filter=False,
            detection_crop_padding_px=5,
        )

        with patch.object(rhd, "run_detection", return_value=fake_run) as run:
            result = rhd.detect_flip_regions(
                bgr,
                mirror,
                domain,
                config,
                replace(rhd.DEFAULT_CONFIG, enable_edge_aligned_rotation=True),
                log=lambda *_a: None,
            )

        cropped_image = run.call_args.args[0]
        self.assertEqual(cropped_image.shape[:2], (30, 40))
        self.assertEqual(result.regions, [(55, 43, 4, 5)])
        self.assertEqual(
            result.rotations,
            [((55.0, 43.0), (59.0, 43.0), (59.0, 48.0), (55.0, 48.0))],
        )

    def test_collaged_detection_reports_full_atlas_coordinates(self) -> None:
        bgr = np.zeros((8, 40, 3), dtype=np.uint8)
        mirror = np.zeros((8, 40), dtype=bool)
        mirror[0:4, 0:4] = True
        mirror[0:4, 20:24] = True
        domain = mirror.copy()
        fake_stage = SimpleNamespace(
            kept=[(6, 0, 4, 4)],
            rotations=[((6.0, 0.0), (10.0, 0.0), (10.0, 4.0), (6.0, 4.0))],
        )
        fake_run = SimpleNamespace(stages=[fake_stage])
        config = rhd.RhdTextureConfig(
            enable_containment_filter=False,
            enable_blob_filter=False,
            detection_crop_padding_px=0,
            detect_island_tiles_individually=False,
            detection_collage_gutter_px=2,
        )

        with patch.object(rhd, "run_detection", return_value=fake_run) as run:
            result = rhd.detect_flip_regions(
                bgr,
                mirror,
                domain,
                config,
                replace(rhd.DEFAULT_CONFIG, enable_edge_aligned_rotation=True),
                log=lambda *_a: None,
            )

        collage = run.call_args.args[0]
        self.assertEqual(collage.shape[:2], (4, 10))
        self.assertEqual(result.regions, [(20, 0, 4, 4)])
        self.assertEqual(
            result.rotations,
            [((20.0, 0.0), (24.0, 0.0), (24.0, 4.0), (20.0, 4.0))],
        )

    def test_individual_tile_detection_reports_full_atlas_coordinates(self) -> None:
        bgr = np.zeros((8, 40, 3), dtype=np.uint8)
        mirror = np.zeros((8, 40), dtype=bool)
        mirror[0:4, 0:4] = True
        mirror[0:4, 20:24] = True
        domain = mirror.copy()
        runs = [
            SimpleNamespace(stages=[SimpleNamespace(kept=[], rotations=[])]),
            SimpleNamespace(
                stages=[
                    SimpleNamespace(
                        kept=[(0, 0, 4, 4)],
                        rotations=[
                            (
                                (0.0, 0.0),
                                (4.0, 0.0),
                                (4.0, 4.0),
                                (0.0, 4.0),
                            )
                        ],
                    )
                ]
            ),
        ]
        config = rhd.RhdTextureConfig(
            enable_containment_filter=False,
            enable_blob_filter=False,
            detection_crop_padding_px=0,
            detection_collage_gutter_px=2,
        )

        with patch.object(rhd, "run_detection", side_effect=runs) as run:
            result = rhd.detect_flip_regions(
                bgr,
                mirror,
                domain,
                config,
                replace(rhd.DEFAULT_CONFIG, enable_edge_aligned_rotation=True),
                log=lambda *_a: None,
            )

        self.assertEqual(run.call_count, 2)
        self.assertEqual(result.regions, [(20, 0, 4, 4)])
        self.assertEqual(
            result.rotations,
            [((20.0, 0.0), (24.0, 0.0), (24.0, 4.0), (20.0, 4.0))],
        )

    def test_neighbouring_islands_share_one_detection_tile(self) -> None:
        bgr = np.zeros((8, 40, 3), dtype=np.uint8)
        mirror = np.zeros((8, 40), dtype=bool)
        mirror[0:4, 0:4] = True
        mirror[0:4, 6:10] = True
        mirror[0:4, 30:34] = True
        domain = mirror.copy()
        runs = [
            SimpleNamespace(
                stages=[
                    SimpleNamespace(
                        kept=[(6, 0, 4, 4)],
                        rotations=[
                            (
                                (6.0, 0.0),
                                (10.0, 0.0),
                                (10.0, 4.0),
                                (6.0, 4.0),
                            )
                        ],
                    )
                ]
            ),
            SimpleNamespace(stages=[SimpleNamespace(kept=[], rotations=[])]),
        ]
        config = rhd.RhdTextureConfig(
            enable_containment_filter=False,
            enable_blob_filter=False,
            detection_crop_padding_px=0,
            detection_tile_group_gap_px=4,
            detection_tile_group_max_area_growth=1.5,
        )

        with patch.object(rhd, "run_detection", side_effect=runs) as run:
            result = rhd.detect_flip_regions(
                bgr,
                mirror,
                domain,
                config,
                replace(rhd.DEFAULT_CONFIG, enable_edge_aligned_rotation=True),
                log=lambda *_a: None,
            )

        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args_list[0].args[0].shape[:2], (4, 10))
        self.assertEqual(run.call_args_list[1].args[0].shape[:2], (4, 4))
        self.assertEqual(result.regions, [(6, 0, 4, 4)])

    def test_foreground_detection_never_groups_touching_uv_islands(self) -> None:
        bgr = np.zeros((8, 12, 3), dtype=np.uint8)
        first = np.zeros((8, 12), dtype=bool)
        second = np.zeros((8, 12), dtype=bool)
        # The raster masks touch, but they are separate topological islands.
        first[0:4, 0:4] = True
        second[0:4, 4:8] = True
        runs = [
            rhd.RegionDetection(source="first", detected=1, regions=[(0, 0, 4, 4)], rotations=[None]),
            rhd.RegionDetection(source="second", detected=1, regions=[(4, 0, 4, 4)], rotations=[None]),
        ]
        config = rhd.RhdTextureConfig(
            enable_containment_filter=False,
            enable_blob_filter=False,
            detection_crop_padding_px=0,
        )

        with patch.object(rhd, "detect_flip_regions", side_effect=runs) as detect:
            result = rhd.detect_flip_regions_by_uv_island(
                bgr,
                (first, second),
                config,
                rhd.DEFAULT_CONFIG,
                log=lambda *_a: None,
            )

        self.assertEqual(detect.call_count, 2)
        self.assertTrue(np.array_equal(detect.call_args_list[0].args[1], first))
        self.assertTrue(np.array_equal(detect.call_args_list[1].args[1], second))
        self.assertEqual(result.regions, [(0, 0, 4, 4), (4, 0, 4, 4)])

    def test_near_duplicate_uv_consumers_are_coalesced_before_detection(self) -> None:
        first = np.zeros((24, 32), dtype=bool)
        second = np.zeros((24, 32), dtype=bool)
        first[2:22, 4:24] = True
        second[2:22, 5:25] = True

        merged = rhd._coalesce_overlapping_uv_consumers((first, second))

        self.assertEqual(len(merged), 1)
        consumer, island_bits = merged[0]
        self.assertIsInstance(consumer, rhd.UvIslandCrop)
        assert isinstance(consumer, rhd.UvIslandCrop)
        self.assertEqual(consumer.bounds, (4, 2, 25, 22))
        self.assertTrue(consumer.mask.all())
        # Which chart owns each pixel survives the union, so grouping can still
        # refuse to reach from one into the other.
        self.assertIsNotNone(island_bits)
        assert island_bits is not None
        self.assertEqual(island_bits.shape, consumer.mask.shape)
        self.assertEqual(int(island_bits[0, 0]), 0b01)
        self.assertEqual(int(island_bits[0, -1]), 0b10)
        self.assertEqual(int(island_bits[0, 10]), 0b11)

    def test_a_lone_uv_consumer_records_no_chart_membership(self) -> None:
        only = np.zeros((16, 16), dtype=bool)
        only[2:10, 2:10] = True

        merged = rhd._coalesce_overlapping_uv_consumers((only,))

        self.assertEqual(len(merged), 1)
        consumer, island_bits = merged[0]
        self.assertIs(consumer, only)
        # Nothing was coalesced, so no join could cross anything.
        self.assertIsNone(island_bits)

    def test_topological_islands_do_not_use_raster_connectivity(self) -> None:
        # The almost-coincident boundary rasterises as a shared edge, but the
        # UV vertices are not shared, so these remain independent islands.
        triangles = (
            np.asarray([(0.0, 0.0), (0.4999999, 0.0), (0.0, 1.0)]),
            np.asarray([(0.5000001, 0.0), (1.0, 0.0), (1.0, 1.0)]),
        )
        mirror = rhd.rasterise_uv_triangles(list(triangles), 8, 8)
        islands = rhd.mirrored_uv_island_masks(triangles, mirror)

        self.assertEqual(len(islands), 2)

    def test_a_shared_uv_corner_needs_the_meshes_to_meet_there_too(self) -> None:
        # Two charts packed so they share their boundary UV vertices exactly.
        # Nothing in the atlas separates them; only the surface does.
        triangles = (
            np.asarray([(0.1, 0.1), (0.5, 0.1), (0.1, 0.5)]),
            np.asarray([(0.5, 0.1), (0.9, 0.5), (0.1, 0.5)]),
        )
        mirror = rhd.rasterise_uv_triangles(list(triangles), 32, 32)

        joined = (
            np.asarray([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]),
            np.asarray([(1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]),
        )
        apart = (
            joined[0],
            joined[1] + np.asarray([0.0, 0.0, 0.25]),
        )

        self.assertEqual(len(rhd.mirrored_uv_island_masks(triangles, mirror)), 1)
        self.assertEqual(
            len(rhd.mirrored_uv_island_masks(triangles, mirror, joined)), 1
        )
        self.assertEqual(
            len(rhd.mirrored_uv_island_masks(triangles, mirror, apart)), 2
        )

    def test_surface_corners_only_split_where_the_mesh_is_really_apart(self) -> None:
        # A 10 um offset is transform noise on a welded vertex, not a boundary.
        triangles = (
            np.asarray([(0.1, 0.1), (0.5, 0.1), (0.1, 0.5)]),
            np.asarray([(0.5, 0.1), (0.9, 0.5), (0.1, 0.5)]),
        )
        mirror = rhd.rasterise_uv_triangles(list(triangles), 32, 32)
        surfaces = (
            np.asarray([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]),
            np.asarray([(1.0, 0.0, 1e-6), (1.0, 1.0, 0.0), (0.0, 1.0, 1e-6)]),
        )

        self.assertEqual(
            len(rhd.mirrored_uv_island_masks(triangles, mirror, surfaces)), 1
        )

    def test_mismatched_surface_corners_fall_back_to_the_uv_grouping(self) -> None:
        triangles = (
            np.asarray([(0.1, 0.1), (0.5, 0.1), (0.1, 0.5)]),
            np.asarray([(0.5, 0.1), (0.9, 0.5), (0.1, 0.5)]),
        )
        mirror = rhd.rasterise_uv_triangles(list(triangles), 32, 32)

        self.assertEqual(
            len(
                rhd.mirrored_uv_island_masks(
                    triangles,
                    mirror,
                    (np.asarray([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]),),
                )
            ),
            1,
        )

    def test_compact_uv_island_crops_reconstruct_full_masks_exactly(self) -> None:
        triangles = (
            np.asarray([(0.1, 0.1), (0.4, 0.1), (0.1, 0.4)]),
            np.asarray([(0.6, 0.6), (0.9, 0.6), (0.9, 0.9)]),
        )
        mirror = rhd.rasterise_uv_triangles(list(triangles), 32, 24)
        full = rhd.mirrored_uv_island_masks(triangles, mirror)
        crops = rhd.mirrored_uv_island_crops(triangles, mirror, padding_px=3)

        rebuilt = []
        for crop in crops:
            mask = np.zeros_like(mirror)
            x0, y0, x1, y1 = crop.bounds
            mask[y0:y1, x0:x1] = crop.mask
            rebuilt.append(mask)

        self.assertEqual(len(rebuilt), len(full))
        for actual, expected in zip(rebuilt, full):
            self.assertTrue(np.array_equal(actual, expected))

    def test_cropped_uv_raster_matches_full_atlas_for_wrapped_triangles(self) -> None:
        triangles = [
            np.asarray([(0.10, 0.15), (0.55, 0.20), (0.20, 0.70)]),
            np.asarray([(0.85, 0.25), (1.15, 0.30), (1.05, 0.80)]),
            np.asarray([(-0.10, 0.65), (0.15, 0.55), (0.05, 1.10)]),
        ]
        mirror = np.ones((47, 61), dtype=bool)
        mirror[8:14, 20:34] = False
        expected = rhd.rasterise_uv_triangles(triangles, 61, 47) & mirror

        compact = rhd.rasterise_uv_triangles_crop(triangles, mirror, padding_px=4)

        self.assertIsNotNone(compact)
        assert compact is not None
        actual = np.zeros_like(mirror)
        x0, y0, x1, y1 = compact.bounds
        actual[y0:y1, x0:x1] = compact.mask
        self.assertTrue(np.array_equal(actual, expected))

    def test_gpu_contrast_islands_are_dispatched_as_one_batch(self) -> None:
        bgr = np.zeros((12, 32, 3), dtype=np.uint8)
        first = rhd.UvIslandCrop((2, 2, 8, 8), np.ones((6, 6), dtype=bool))
        second = rhd.UvIslandCrop((20, 2, 26, 8), np.ones((6, 6), dtype=bool))
        detections = [
            rhd.LocalContrastDetection(
                np.empty((0, 4), dtype=np.int32),
                np.zeros((6, 6), dtype=np.float32),
                4.0,
            ),
            rhd.LocalContrastDetection(
                np.asarray([(1, 1, 2, 2)], dtype=np.int32),
                np.zeros((6, 6), dtype=np.float32),
                4.0,
            ),
        ]
        config = rhd.RhdTextureConfig(
            enable_containment_filter=False,
            enable_blob_filter=False,
            detection_crop_padding_px=0,
        )
        detector = replace(rhd.DEFAULT_CONFIG, box_source="contrast_gpu")

        def fake_run(_image, _mask, _config, **kwargs):
            kept = [tuple(int(value) for value in box) for box in kwargs["initial_boxes"]]
            return SimpleNamespace(
                stages=[SimpleNamespace(kept=kept, rotations=[None] * len(kept))]
            )

        with (
            patch.object(
                rhd, "detect_local_contrast_gpu_batch", return_value=detections
            ) as batch,
            patch.object(rhd, "run_detection", side_effect=fake_run) as run,
        ):
            result = rhd.detect_flip_regions_by_uv_island(
                bgr,
                (first, second),
                config,
                detector,
                log=lambda *_a: None,
            )

        self.assertEqual(batch.call_count, 1)
        self.assertEqual(len(batch.call_args.args[0]), 2)
        self.assertEqual(run.call_count, 2)
        responses = {
            id(call.kwargs["initial_contrast_response"])
            for call in run.call_args_list
        }
        self.assertEqual(responses, {id(item.response) for item in detections})
        self.assertEqual(result.regions, [(21, 3, 2, 2)])


class MultiPartPreviewTests(unittest.TestCase):
    def test_manifest_reassembles_independently_corrected_material_layers(self) -> None:
        source = {
            "key": "lc500_screen_on",
            "aliases": ["lc500_screen_on"],
            "materialsMember": "vehicles/lc500/main.materials.json",
            "material": {"Stages": [{}]},
        }

        def result(member: str, stage_key: str, kind: str) -> rhd.RhdTextureResult:
            stem = Path(member).stem
            png = Path(f"{stem}_rhd.png")
            return rhd.RhdTextureResult(
                texture_member=member,
                size=(4, 4),
                parts_analysed=1,
                mirrored_triangles=1,
                rigid_triangles=0,
                mirror_coverage=1.0,
                rigid_coverage=0.0,
                conflict_coverage=0.0,
                glyph_regions=1,
                mirrored_glyph_regions=1,
                material_aliases=("lc500_screen", "lc500_screen_on"),
                png_path=png,
                dds_path=Path(f"{stem}_rhd.dds"),
                report={
                    "layer_bindings": [
                        {
                            "dae_material": "lc500_screen",
                            "material_key": "lc500_screen_on",
                            "materials_member": source["materialsMember"],
                            "stage_key": stage_key,
                            "kind": kind,
                        }
                    ],
                    "source_materials": [source],
                    "switch_base_aliases": ["lc500_screen"],
                    "outputs": {"preview": None},
                },
            )

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            rhd.write_blender_preview(
                output,
                [
                    result("vehicles/lc500/screen.png", "baseColorMap", "colour"),
                    result("vehicles/lc500/screen_glow.png", "emissiveMap", "scalar"),
                ],
                log=lambda *_a: None,
            )
            manifest = json.loads((output / "rhd_materials.json").read_text())

        self.assertEqual(len(manifest["materials"]), 1)
        entry = manifest["materials"][0]
        self.assertEqual(set(entry["maps"]), {"baseColorMap", "emissiveMap"})
        self.assertEqual(len(entry["outputMaps"]), 2)

    def test_same_texture_material_states_are_kept_as_distinct_bindings(self) -> None:
        dash = part("dash")
        off = ArchiveTextureBinding(
            dae_material="ardente_interior",
            material_key="ardente_interior",
            materials_member="vehicles/vivace/ardente/main.materials.json",
            texture_reference="/vehicles/vivace/ardente/ardente_interior_b.color.png",
            texture_member="vehicles/vivace/ardente/ardente_interior_b.color.png",
        )
        on = ArchiveTextureBinding(
            dae_material="ardente_interior",
            material_key="ardente_interior_on",
            materials_member="vehicles/vivace/ardente/main.materials.json",
            texture_reference="/vehicles/vivace/ardente/ardente_interior_b.color.png",
            texture_member="vehicles/vivace/ardente/ardente_interior_b.color.png",
        )

        with patch.object(
            rhd,
            "archive_texture_choices_for_part",
            return_value=(off, on),
        ):
            grouped = rhd.texture_bindings_for_parts(
                SimpleNamespace(),
                SimpleNamespace(),
                [dash],
            )

        bindings = grouped["vehicles/vivace/ardente/ardente_interior_b.color.png"]
        self.assertEqual([binding.material_key for _part, binding in bindings], [
            "ardente_interior",
            "ardente_interior_on",
        ])

    def test_same_texture_target_keeps_distinct_dae_source_materials(self) -> None:
        dash = part("dash")
        member = "vehicles/lc500/textures/dashlights.png"
        left = ArchiveTextureBinding(
            dae_material="lc500_intsignal_L",
            material_key="lc500_dashlights",
            materials_member="vehicles/lc500/main.materials.json",
            texture_reference=f"/{member}",
            texture_member=member,
        )
        right = ArchiveTextureBinding(
            dae_material="lc500_intsignal_R",
            material_key="lc500_dashlights",
            materials_member="vehicles/lc500/main.materials.json",
            texture_reference=f"/{member}",
            texture_member=member,
        )

        with patch.object(
            rhd,
            "archive_texture_choices_for_part",
            return_value=(left, right),
        ):
            grouped = rhd.texture_bindings_for_parts(
                SimpleNamespace(),
                SimpleNamespace(),
                [dash],
            )

        bindings = grouped[member]
        self.assertEqual(
            [binding.dae_material for _part, binding in bindings],
            ["lc500_intsignal_L", "lc500_intsignal_R"],
        )

    def test_every_archive_backed_material_layer_becomes_its_own_binding(self) -> None:
        dash = part("dash")
        materials_member = "vehicles/lc500/main.materials.json"
        base_member = "vehicles/lc500/textures/screen.png"
        glow_member = "vehicles/lc500/textures/screen_glow.png"
        normal_member = "vehicles/lc500/textures/screen_n.png"
        record = ArchiveMaterialRecord(
            key="lc500_screen_on",
            name="lc500_screen_on",
            map_to="",
            materials_member=materials_member,
            base_colour_reference=f"/{base_member}",
            source_material={
                "Stages": [
                    {
                        "baseColorMap": f"/{base_member}",
                        "emissiveMap": f"/{glow_member}",
                        "normalMap": f"/{normal_member}",
                    },
                    {
                        # A duplicate reference is one physical job but retains
                        # both material slots for final wiring.
                        "baseColorMap": f"/{glow_member}",
                    },
                ]
            },
        )
        binding = ArchiveTextureBinding(
            dae_material="lc500_screen",
            material_key=record.key,
            materials_member=materials_member,
            texture_reference=f"/{base_member}",
            texture_member=base_member,
        )
        members = (base_member, glow_member, normal_member)
        archive = SimpleNamespace(
            materials=(record,),
            members=members,
            member_by_lower={member.lower(): member for member in members},
            member_archive_indices={},
        )

        with patch.object(
            rhd,
            "archive_texture_choices_for_part",
            return_value=(binding,),
        ):
            grouped = rhd.texture_bindings_for_parts(archive, SimpleNamespace(), [dash])

        self.assertEqual(set(grouped), set(members))
        self.assertEqual(
            [layer.stage_key for _part, layer in grouped[glow_member]],
            ["emissiveMap", "baseColorMap"],
        )
        self.assertEqual(grouped[normal_member][0][1].kind, "normal")

    def test_layer_resolves_the_normal_from_its_own_material_stage(self) -> None:
        dash = part("dash")
        materials_member = "vehicles/lc500/main.materials.json"
        base_member = "vehicles/lc500/textures/screen.png"
        normal_member = "vehicles/lc500/textures/screen_n.png"
        other_normal_member = "vehicles/lc500/textures/trim_n.png"
        record = ArchiveMaterialRecord(
            key="lc500_screen",
            name="lc500_screen",
            map_to="",
            materials_member=materials_member,
            base_colour_reference=f"/{base_member}",
            source_material={
                "Stages": [
                    {
                        "baseColorMap": f"/{base_member}",
                        "normalMap": f"/{normal_member}",
                    },
                    {
                        "baseColorMap": "/vehicles/lc500/textures/trim.png",
                        "normalMap": f"/{other_normal_member}",
                    },
                ]
            },
        )
        binding = rhd.MaterialTextureLayerBinding(
            dae_material="lc500_screen",
            material_key=record.key,
            materials_member=materials_member,
            texture_reference=f"/{base_member}",
            texture_member=base_member,
            stage_key="baseColorMap",
            kind="colour",
        )
        members = (
            base_member,
            normal_member,
            "vehicles/lc500/textures/trim.png",
            other_normal_member,
        )
        archive = SimpleNamespace(
            materials=(record,),
            members=members,
            member_by_lower={member.lower(): member for member in members},
            member_archive_indices={},
        )

        normals = rhd.normal_maps_for_layer_bindings(
            archive, [(dash, binding)], base_member
        )

        self.assertEqual(
            normals,
            (rhd.CompanionMap(normal_member, "normalMap", "normal"),),
        )

    def test_opacity_map_is_authoritative_only_when_every_binding_is_masked(self) -> None:
        dash = part("dash")
        materials_member = "vehicles/lc500/main.materials.json"
        base_member = "vehicles/lc500/textures/intemis.dds"
        mask_member = "vehicles/lc500/textures/intemis_opmap.png"
        masked = ArchiveMaterialRecord(
            key="lc500_intemis_on",
            name="lc500_intemis_on",
            map_to="",
            materials_member=materials_member,
            base_colour_reference=f"/{base_member}",
            source_material={
                "Stages": [{
                    "baseColorMap": f"/{base_member}",
                    "opacityMap": f"/{mask_member}",
                }]
            },
        )
        unmasked = ArchiveMaterialRecord(
            key="lc500_trim",
            name="lc500_trim",
            map_to="",
            materials_member=materials_member,
            base_colour_reference=f"/{base_member}",
            source_material={"Stages": [{"baseColorMap": f"/{base_member}"}]},
        )
        members = (base_member, mask_member)
        archive = SimpleNamespace(
            materials=(masked,),
            members=members,
            member_by_lower={member.lower(): member for member in members},
            member_archive_indices={},
        )

        def binding(material_key: str) -> rhd.MaterialTextureLayerBinding:
            return rhd.MaterialTextureLayerBinding(
                dae_material=material_key,
                material_key=material_key,
                materials_member=materials_member,
                texture_reference=f"/{base_member}",
                texture_member=base_member,
                stage_key="baseColorMap",
                kind="colour",
            )

        self.assertEqual(
            rhd.authoritative_visibility_masks_for_layer_bindings(
                archive, [(dash, binding(masked.key))], base_member
            ),
            (rhd.CompanionMap(mask_member, "opacityMap", "scalar"),),
        )

        archive.materials = (masked, unmasked)
        self.assertEqual(
            rhd.authoritative_visibility_masks_for_layer_bindings(
                archive,
                [(dash, binding(masked.key)), (dash, binding(unmasked.key))],
                base_member,
            ),
            (),
        )

    def test_masked_layer_uses_only_opacity_detection_regions(self) -> None:
        dash = part("dash")
        materials_member = "vehicles/lc500/main.materials.json"
        base_member = "vehicles/lc500/textures/intemis.dds"
        mask_member = "vehicles/lc500/textures/intemis_opmap.png"
        record = ArchiveMaterialRecord(
            key="lc500_intemis_on",
            name="lc500_intemis_on",
            map_to="",
            materials_member=materials_member,
            base_colour_reference=f"/{base_member}",
            source_material={
                "Stages": [{
                    "baseColorMap": f"/{base_member}",
                    "opacityMap": f"/{mask_member}",
                }]
            },
        )
        binding = rhd.MaterialTextureLayerBinding(
            dae_material="lc500_intemissive",
            material_key=record.key,
            materials_member=materials_member,
            texture_reference=f"/{base_member}",
            texture_member=base_member,
            stage_key="baseColorMap",
            kind="colour",
        )
        archive = SimpleNamespace(
            materials=(record,),
            members=(base_member, mask_member),
            member_by_lower={
                base_member.lower(): base_member,
                mask_member.lower(): mask_member,
            },
            member_archive_indices={},
            material_switch_targets={},
        )

        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            base_path = workspace / "intemis.dds"
            mask_path = workspace / "intemis_opmap.png"
            Image.new("RGBA", (8, 8), (255, 0, 255, 255)).save(
                base_path, format="PNG"
            )
            Image.new("RGBA", (8, 8), (0, 0, 0, 255)).save(mask_path)
            mirror = np.zeros((8, 8), dtype=bool)
            mirror[:, :4] = True
            masks = rhd.DomainMasks(
                mirror=mirror,
                rigid=~mirror,
                conflict_coverage=0.0,
                mirrored_triangles=1,
                rigid_triangles=1,
                parts_analysed=1,
            )
            planned_regions: list[tuple[int, int, int, int]] = []

            def extracted(_archive, member):
                return mask_path if member == mask_member else base_path

            def detect(
                _image,
                _islands,
                _config,
                detector_config,
                source,
                _log,
                relief_bridge_response=None,
            ):
                self.assertTrue(str(source).startswith("opacityMap:"))
                self.assertEqual(detector_config.box_source, "opacity_mask")
                self.assertIsNone(relief_bridge_response)
                return rhd.RegionDetection(
                    source=str(source),
                    detected=1,
                    regions=[(0, 1, 4, 3)],
                    rotations=[None],
                )

            def plan(_mirror, regions, _config, _axis_map, _components=None):
                planned_regions.extend(regions)
                return [], np.zeros((8, 8), dtype=bool)

            with (
                patch.object(
                    rhd,
                    "scoped_parts_using_material",
                    return_value=[(dash, binding)],
                ),
                patch.object(
                    rhd,
                    "material_symbols_for_binding",
                    return_value=("lc500_intemissive-material",),
                ),
                patch.object(rhd, "extract_archive_member", side_effect=extracted),
                patch.object(
                    rhd,
                    "detect_flip_regions_by_uv_island",
                    side_effect=detect,
                ),
                patch.object(rhd, "plan_island_flips", side_effect=plan),
                patch.object(
                    rhd,
                    "skew_delta_for_region",
                    side_effect=AssertionError(
                        "authoritative opacity regions must bypass inferred UV skew"
                    ),
                ),
                patch.object(
                    rhd,
                    "normal_maps_for_layer_bindings",
                    side_effect=AssertionError("masked layers must not inspect normals"),
                ),
            ):
                result = rhd.build_rhd_texture(
                    archive,
                    SimpleNamespace(),
                    base_member,
                    workspace,
                    config=rhd.RhdTextureConfig(
                        detect_on_normal_map=True,
                        write_debug_overlays=False,
                    ),
                    part_scope=[dash],
                    masks=masks,
                    log=lambda *_a: None,
                )

        self.assertEqual(planned_regions, [(0, 1, 4, 3)])
        self.assertEqual(result.report["detection_authority"], "opacity_mask")
        self.assertFalse(result.report["skewed_region_filter_applied"])
        self.assertEqual(
            [(step["x"], step["y"], step["w"], step["h"])
             for step in result.report["in_place_flips"]],
            [(0, 1, 4, 3)],
        )
        self.assertEqual(
            result.report["in_place_flips"][0]["stencil"],
            rhd.STENCIL_REGION,
        )
        self.assertEqual(result.report["authoritative_mask_sources"], [mask_member])
        self.assertEqual(
            result.report["detection"][0]["pipeline"],
            "authoritative_opacity_mask",
        )

    def _capture_production_detector(
        self, *, with_normal: bool
    ) -> tuple[dict[str, object], rhd.RhdTextureResult]:
        dash = part("dash")
        base_member = "vehicles/lc500/textures/screen.png"
        normal_member = "vehicles/lc500/textures/screen_n.png"
        binding = rhd.MaterialTextureLayerBinding(
            dae_material="lc500_screen",
            material_key="lc500_screen",
            materials_member="vehicles/lc500/main.materials.json",
            texture_reference=f"/{base_member}",
            texture_member=base_member,
            stage_key="baseColorMap",
            kind="colour",
        )
        captured: dict[str, object] = {}
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            base = workspace / "screen.png"
            normal = workspace / "screen_n.png"
            Image.new("RGBA", (4, 4), (0, 0, 0, 255)).save(base)
            Image.new("RGB", (4, 4), (128, 128, 255)).save(normal)
            mirror = np.zeros((4, 4), dtype=bool)
            mirror[:, :2] = True
            rigid = ~mirror
            masks = rhd.DomainMasks(
                mirror=mirror,
                rigid=rigid,
                conflict_coverage=0.0,
                mirrored_triangles=1,
                rigid_triangles=1,
                parts_analysed=1,
            )
            response = np.full((4, 4), 23.0, dtype=np.float32)
            edge_data = SimpleNamespace(
                edge_response=response,
                render_seconds=0.1,
                edge_seconds=0.2,
            )
            session = rhd.ProductionDetectionSession()

            def detect(
                _image,
                _islands,
                _config,
                detector,
                source,
                _log,
                relief_bridge_response=None,
            ):
                captured["box_source"] = detector.box_source
                captured["source"] = source
                captured["relief_bridge_response"] = relief_bridge_response
                return rhd.RegionDetection(source=str(source), detected=0)

            def extracted(_archive, member):
                return normal if member == normal_member else base

            normal_maps = (
                (rhd.CompanionMap(normal_member, "normalMap", "normal"),)
                if with_normal
                else ()
            )
            with (
                patch.object(rhd, "scoped_parts_using_material", return_value=[(dash, binding)]),
                patch.object(rhd, "material_symbols_for_binding", return_value=("lc500_screen-material",)),
                patch.object(rhd, "extract_archive_member", side_effect=extracted),
                patch.object(rhd, "build_domain_masks", return_value=masks),
                patch.object(rhd, "normal_maps_for_layer_bindings", return_value=normal_maps),
                patch.object(session, "normal_edge_data", return_value=(edge_data, False)),
                patch.object(rhd, "detect_flip_regions_by_uv_island", side_effect=detect),
                patch.object(rhd, "plan_island_flips", return_value=([], np.zeros((4, 4), dtype=bool))),
            ):
                result = rhd.build_rhd_texture(
                    SimpleNamespace(materials=[]),
                    SimpleNamespace(),
                    base_member,
                    workspace,
                    config=rhd.RhdTextureConfig(
                        detect_on_normal_map=True,
                        write_debug_overlays=False,
                    ),
                    part_scope=[dash],
                    masks=masks,
                    detection_session=session,
                    log=lambda *_a: None,
                )
        return captured, result

    def test_production_uses_colour_glyphs_with_cached_relief_edges(self) -> None:
        captured, result = self._capture_production_detector(with_normal=True)

        self.assertEqual(captured["box_source"], "contrast_gpu")
        self.assertIsNotNone(captured["relief_bridge_response"])
        self.assertEqual(
            result.report["detection"][0]["pipeline"],
            "colour_glyphs_relief_edge_grouping",
        )

    def test_production_falls_back_to_gpu_colour_without_a_normal(self) -> None:
        captured, result = self._capture_production_detector(with_normal=False)

        self.assertEqual(captured["box_source"], "contrast_gpu")
        self.assertIsNone(captured["relief_bridge_response"])
        self.assertEqual(
            result.report["detection"][0]["pipeline"],
            "colour_local_contrast_gpu",
        )

    def test_normal_layer_uses_island_scoped_gpu_relief_detection(self) -> None:
        dash = part("dash")
        normal_member = "vehicles/vivace/ardente/ardente_interior_nm.normal.png"
        binding = rhd.MaterialTextureLayerBinding(
            dae_material="ardente_interior",
            material_key="ardente_interior",
            materials_member="vehicles/vivace/ardente/main.materials.json",
            texture_reference=f"/{normal_member}",
            texture_member=normal_member,
            stage_key="normalMap",
            kind="normal",
        )
        captured: dict[str, object] = {}
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            normal = workspace / "ardente_interior_nm.normal.png"
            Image.new("RGB", (4, 4), (128, 128, 255)).save(normal)
            mirror = np.zeros((4, 4), dtype=bool)
            mirror[:, :2] = True
            masks = rhd.DomainMasks(
                mirror=mirror,
                rigid=~mirror,
                conflict_coverage=0.0,
                mirrored_triangles=1,
                rigid_triangles=1,
                parts_analysed=1,
            )
            relief = np.full((4, 4, 3), 71, dtype=np.uint8)
            response = np.full((4, 4), 29.0, dtype=np.float32)
            edge_data = SimpleNamespace(
                relief_bgr=relief,
                edge_response=response,
                render_seconds=0.1,
                edge_seconds=0.2,
            )
            session = rhd.ProductionDetectionSession()

            def detect(
                image,
                islands,
                _config,
                detector,
                source,
                _log,
                relief_bridge_response=None,
            ):
                captured["image"] = image
                captured["box_source"] = detector.box_source
                captured["detector"] = detector
                captured["source"] = source
                captured["relief_bridge_response"] = relief_bridge_response
                captured["islands"] = islands
                return rhd.RegionDetection(source=str(source), detected=0)

            with (
                patch.object(
                    rhd,
                    "scoped_parts_using_material",
                    return_value=[(dash, binding)],
                ),
                patch.object(
                    rhd,
                    "material_symbols_for_binding",
                    return_value=("ardente_interior-material",),
                ),
                patch.object(rhd, "extract_archive_member", return_value=normal),
                patch.object(
                    rhd,
                    "normal_maps_for_layer_bindings",
                    side_effect=AssertionError(
                        "a physical normal layer must use its own pixels"
                    ),
                ),
                patch.object(session, "normal_edge_data", return_value=(edge_data, False)),
                patch.object(
                    rhd,
                    "detect_flip_regions_by_uv_island",
                    side_effect=detect,
                ),
                patch.object(
                    rhd,
                    "detect_flip_regions",
                    side_effect=AssertionError(
                        "normal relief must match the tuning harness's UV scope"
                    ),
                ),
                patch.object(
                    rhd,
                    "plan_island_flips",
                    return_value=([], np.zeros((4, 4), dtype=bool)),
                ),
            ):
                result = rhd.build_rhd_texture(
                    SimpleNamespace(materials=[]),
                    SimpleNamespace(),
                    normal_member,
                    workspace,
                    config=rhd.RhdTextureConfig(write_debug_overlays=False),
                    part_scope=[dash],
                    masks=masks,
                    detection_session=session,
                    log=lambda *_a: None,
                )

        self.assertIs(captured["image"], relief)
        self.assertEqual(captured["box_source"], "edge_gpu")
        self.assertEqual(captured["source"], "relief")
        self.assertIs(captured["relief_bridge_response"], response)
        self.assertGreater(len(captured["islands"]), 0)
        self.assertEqual(
            captured["detector"],
            replace(rhd.DEFAULT_RELIEF_DETECTION_CONFIG, box_source="edge_gpu"),
        )
        self.assertEqual(result.report["detection"][0]["source"], "relief")
        self.assertEqual(
            result.report["detection"][0]["pipeline"],
            "relief_edge_gpu",
        )
        self.assertFalse(result.report["normal_region_detail_gate"])

    def test_companion_maps_do_not_replay_the_colour_plan(self) -> None:
        dash = part("dash")
        binding = ArchiveTextureBinding(
            dae_material="ardente_interior",
            material_key="ardente_interior_on",
            materials_member="vehicles/vivace/ardente/main.materials.json",
            texture_reference="/vehicles/vivace/ardente/ardente_interior_b.color.png",
            texture_member="vehicles/vivace/ardente/ardente_interior_b.color.png",
        )

        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            base = workspace / "ardente_interior_b.color.png"
            Image.new("RGBA", (4, 4), (0, 0, 0, 255)).save(base)
            mask = np.ones((4, 4), dtype=bool)
            masks = rhd.DomainMasks(
                mirror=mask,
                rigid=np.zeros((4, 4), dtype=bool),
                conflict_coverage=0.0,
                mirrored_triangles=1,
                rigid_triangles=0,
                parts_analysed=1,
            )
            companion_result = rhd.CompanionResult(
                member="vehicles/vivace/ardente/ardente_interior_g.color.png",
                stage_key="emissiveMap",
                kind="scalar",
                codec="rgba",
                texels_moved=16,
                png_path=workspace / "ardente_interior_g.color_rhd.png",
                dds_path=workspace / "ardente_interior_g.color_rhd.dds",
            )

            with (
                patch.object(rhd, "scoped_parts_using_material", return_value=[(dash, binding)]),
                patch.object(rhd, "material_symbols_for_binding", return_value=("ardente_interior-material",)),
                patch.object(rhd, "extract_archive_member", return_value=base),
                patch.object(rhd, "build_domain_masks", return_value=masks),
                patch.object(
                    rhd,
                    "companion_maps_for_binding",
                    return_value=(
                        rhd.CompanionMap(
                            member="vehicles/vivace/ardente/ardente_interior_g.color.png",
                            stage_key="emissiveMap",
                            kind="scalar",
                        ),
                    ),
                ),
                patch.object(
                    rhd,
                    "detect_flip_regions",
                    return_value=rhd.RegionDetection(
                        source="colour",
                        detected=1,
                        regions=[(0, 0, 4, 4)],
                        rotations=[None],
                    ),
                ),
                patch.object(
                    rhd,
                    "plan_island_flips",
                    return_value=(
                        [
                            rhd.IslandFlip(
                                label=1,
                                bounds=(0, 0, 4, 4),
                                area_px=16,
                                axis="horizontal",
                                horizontal_similarity=1.0,
                                vertical_similarity=1.0,
                                glyph_count=1,
                            )
                        ],
                        mask,
                    ),
                ),
                patch.object(rhd, "rebuild_companion_map", return_value=companion_result) as rebuild,
            ):
                result = rhd.build_rhd_texture(
                    SimpleNamespace(materials=[]),
                    SimpleNamespace(),
                    binding.texture_member,
                    workspace,
                    config=rhd.RhdTextureConfig(
                        rebuild_companion_maps=True,
                        detect_on_normal_map=False,
                    ),
                    part_scope=[dash],
                    log=lambda *_a: None,
                )

        rebuild.assert_not_called()
        self.assertEqual(result.companions, [])

    def test_emissive_companion_regions_do_not_join_base_flip_plan(self) -> None:
        dash = part("dash")
        binding = ArchiveTextureBinding(
            dae_material="lc500_door_buttons",
            material_key="lc500_door_buttons_on",
            materials_member="vehicles/lc500/main.materials.json",
            texture_reference="/vehicles/lc500/textures/buttons.dds",
            texture_member="vehicles/lc500/textures/buttons.dds",
        )

        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            base = workspace / "buttons.dds"
            glow = workspace / "buttons_glow.png"
            Image.new("RGBA", (4, 4), (0, 0, 0, 255)).save(base, format="PNG")
            Image.new("RGBA", (4, 4), (255, 0, 0, 255)).save(glow)
            mask = np.ones((4, 4), dtype=bool)
            masks = rhd.DomainMasks(
                mirror=mask,
                rigid=np.zeros((4, 4), dtype=bool),
                conflict_coverage=0.0,
                mirrored_triangles=1,
                rigid_triangles=0,
                parts_analysed=1,
            )
            planned_regions: list[tuple[int, int, int, int]] = []

            def detection(
                _image,
                _mirror,
                _domain,
                _config,
                _mser_config,
                source,
                _log,
            ) -> rhd.RegionDetection:
                if str(source).startswith("emissiveMap:"):
                    return rhd.RegionDetection(
                        source=str(source),
                        detected=1,
                        regions=[(0, 0, 4, 4)],
                        rotations=[None],
                    )
                return rhd.RegionDetection(source=str(source), detected=0)

            def plan(
                _mirror,
                regions,
                _config,
                _axis_map,
                _components=None,
            ):
                planned_regions.extend(regions)
                return (
                    [
                        rhd.IslandFlip(
                            label=1,
                            bounds=(0, 0, 4, 4),
                            area_px=16,
                            axis="horizontal",
                            horizontal_similarity=1.0,
                            vertical_similarity=1.0,
                            glyph_count=1,
                        )
                    ],
                    mask,
                )

            with (
                patch.object(rhd, "scoped_parts_using_material", return_value=[(dash, binding)]),
                patch.object(rhd, "material_symbols_for_binding", return_value=("lc500_door_buttons-material",)),
                patch.object(rhd, "extract_archive_member", side_effect=[base, glow]),
                patch.object(rhd, "build_domain_masks", return_value=masks),
                patch.object(
                    rhd,
                    "companion_maps_for_binding",
                    return_value=(
                        rhd.CompanionMap(
                            member="vehicles/lc500/textures/buttons_glow.png",
                            stage_key="emissiveMap",
                            kind="scalar",
                        ),
                    ),
                ),
                patch.object(rhd, "detect_flip_regions", side_effect=detection),
                patch.object(rhd, "plan_island_flips", side_effect=plan),
                patch.object(rhd, "rebuild_companion_map", return_value=None),
            ):
                result = rhd.build_rhd_texture(
                    SimpleNamespace(materials=[]),
                    SimpleNamespace(),
                    binding.texture_member,
                    workspace,
                    config=rhd.RhdTextureConfig(
                        rebuild_companion_maps=True,
                        detect_on_normal_map=False,
                    ),
                    part_scope=[dash],
                    log=lambda *_a: None,
                )

        self.assertEqual(planned_regions, [(0, 0, 4, 4)])
        self.assertEqual(result.report["companion_regions_added"], 0)

    def test_switch_group_state_regions_do_not_join_base_flip_plan(self) -> None:
        dash = part("dash")
        binding = ArchiveTextureBinding(
            dae_material="lc500_screen_off",
            material_key="lc500_screen_off_off",
            materials_member="vehicles/lc500/main.materials.json",
            texture_reference="/vehicles/lc500/LEX_LC5.dds",
            texture_member="vehicles/lc500/LEX_LC5.dds",
        )
        on_record = ArchiveMaterialRecord(
            key="lc500_screen_off_on",
            name="lc500_screen_off_on",
            map_to="",
            materials_member="vehicles/lc500/main.materials.json",
            base_colour_reference="/vehicles/lc500/textures/screen_on.png",
            source_material={
                "Stages": [
                    {
                        "baseColorMap": "/vehicles/lc500/textures/screen_on.png",
                        "emissiveMap": "/vehicles/lc500/textures/screen_on.png",
                    }
                ]
            },
        )

        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            base = workspace / "LEX_LC5.dds"
            on = workspace / "screen_on.png"
            Image.new("RGBA", (8, 8), (0, 0, 0, 255)).save(base, format="PNG")
            Image.new("RGBA", (4, 4), (255, 0, 0, 255)).save(on)
            mask = np.ones((8, 8), dtype=bool)
            masks = rhd.DomainMasks(
                mirror=mask,
                rigid=np.zeros((8, 8), dtype=bool),
                conflict_coverage=0.0,
                mirrored_triangles=1,
                rigid_triangles=0,
                parts_analysed=1,
            )
            archive = SimpleNamespace(
                materials=(on_record,),
                material_switch_targets={
                    "lc500_screen_off": ("lc500_screen_off_off", "lc500_screen_off_on")
                },
                material_switch_states={
                    "lc500_screen_off": (
                        ArchiveMaterialSwitchState("off", "lc500_screen_off_off"),
                        ArchiveMaterialSwitchState("on", "lc500_screen_off_on"),
                    )
                },
                member_by_lower={
                    "vehicles/lc500/lex_lc5.dds": "vehicles/lc500/LEX_LC5.dds",
                    "vehicles/lc500/textures/screen_on.png": (
                        "vehicles/lc500/textures/screen_on.png"
                    ),
                },
                members=(
                    "vehicles/lc500/LEX_LC5.dds",
                    "vehicles/lc500/textures/screen_on.png",
                ),
                member_archive_indices={},
            )
            planned_regions: list[tuple[int, int, int, int]] = []

            def detection(
                _image,
                _mirror,
                _domain,
                _config,
                _mser_config,
                source,
                _log,
            ) -> rhd.RegionDetection:
                if str(source).startswith("state:"):
                    return rhd.RegionDetection(
                        source=str(source),
                        detected=1,
                        regions=[(1, 1, 2, 2)],
                        rotations=[None],
                    )
                return rhd.RegionDetection(source=str(source), detected=0)

            def plan(
                _mirror,
                regions,
                _config,
                _axis_map,
                _components=None,
            ):
                planned_regions.extend(regions)
                return (
                    [
                        rhd.IslandFlip(
                            label=1,
                            bounds=(2, 2, 4, 4),
                            area_px=16,
                            axis="horizontal",
                            horizontal_similarity=1.0,
                            vertical_similarity=1.0,
                            glyph_count=1,
                        )
                    ],
                    mask,
                )

            def member_path(_archive, member):
                return {
                    "vehicles/lc500/LEX_LC5.dds": base,
                    "vehicles/lc500/textures/screen_on.png": on,
                }[member]

            with (
                patch.object(rhd, "scoped_parts_using_material", return_value=[(dash, binding)]),
                patch.object(rhd, "material_symbols_for_binding", return_value=("lc500_screen_off-material",)),
                patch.object(rhd, "extract_archive_member", side_effect=member_path),
                patch.object(rhd, "build_domain_masks", return_value=masks),
                patch.object(rhd, "companion_maps_for_binding", return_value=()),
                patch.object(rhd, "detect_flip_regions", side_effect=detection),
                patch.object(rhd, "plan_island_flips", side_effect=plan),
            ):
                result = rhd.build_rhd_texture(
                    archive,
                    SimpleNamespace(),
                    binding.texture_member,
                    workspace,
                    config=rhd.RhdTextureConfig(
                        rebuild_companion_maps=True,
                        detect_on_normal_map=False,
                    ),
                    part_scope=[dash],
                    log=lambda *_a: None,
                )

        self.assertEqual(planned_regions, [(0, 0, 8, 8)])
        self.assertEqual(result.report["state_group_regions_added"], 0)
        self.assertFalse(
            any(
                str(entry["source"]).startswith("state:")
                for entry in result.report["detection"]
            )
        )

    def test_switch_group_detection_does_not_merge_shared_off_states(self) -> None:
        dash = part("dash")
        binding = ArchiveTextureBinding(
            dae_material="lc500_centralscreen",
            material_key="lc500_screens_off",
            materials_member="vehicles/lc500/main.materials.json",
            texture_reference="/vehicles/lc500/textures/lc500_bootscreen.png",
            texture_member="vehicles/lc500/textures/lc500_bootscreen.png",
        )
        own_on = ArchiveMaterialRecord(
            key="lc500_centralscreen_on",
            name="lc500_centralscreen_on",
            map_to="",
            materials_member="vehicles/lc500/main.materials.json",
            base_colour_reference="/vehicles/lc500/textures/lc500_bootscreen_on.png",
            source_material={
                "Stages": [
                    {"baseColorMap": "/vehicles/lc500/textures/lc500_bootscreen_on.png"}
                ]
            },
        )
        other_on = ArchiveMaterialRecord(
            key="lc500_screens_on",
            name="lc500_screens_on",
            map_to="",
            materials_member="vehicles/lc500/main.materials.json",
            base_colour_reference="/vehicles/lc500/textures/screen.dds",
            source_material={
                "Stages": [
                    {"baseColorMap": "/vehicles/lc500/textures/screen.dds"}
                ]
            },
        )
        archive = SimpleNamespace(
            materials=(own_on, other_on),
            material_switch_targets={
                "lc500_centralscreen": (
                    "lc500_screens_off",
                    "lc500_centralscreen_on",
                    "lc500_gps",
                ),
                "lc500_screens": ("lc500_screens_off", "lc500_screens_on"),
            },
            material_switch_states={
                "lc500_centralscreen": (
                    ArchiveMaterialSwitchState("off", "lc500_screens_off"),
                    ArchiveMaterialSwitchState("on", "lc500_centralscreen_on"),
                    ArchiveMaterialSwitchState("on_intense", "lc500_gps"),
                ),
                "lc500_screens": (
                    ArchiveMaterialSwitchState("off", "lc500_screens_off"),
                    ArchiveMaterialSwitchState("on", "lc500_screens_on"),
                ),
            },
            material_switch_triggers={
                "lc500_centralscreen": ("ignitionLevel",),
                "lc500_screens": ("running",),
            },
            member_by_lower={
                "vehicles/lc500/textures/lc500_bootscreen.png": (
                    "vehicles/lc500/textures/lc500_bootscreen.png"
                ),
                "vehicles/lc500/textures/lc500_bootscreen_on.png": (
                    "vehicles/lc500/textures/lc500_bootscreen_on.png"
                ),
                "vehicles/lc500/textures/screen.dds": (
                    "vehicles/lc500/textures/screen.dds"
                ),
            },
            members=(
                "vehicles/lc500/textures/lc500_bootscreen.png",
                "vehicles/lc500/textures/lc500_bootscreen_on.png",
                "vehicles/lc500/textures/screen.dds",
            ),
            member_archive_indices={},
        )

        maps = rhd.switch_group_detection_maps_for_candidates(
            archive,
            [(dash, binding)],
            binding.texture_member,
        )

        self.assertEqual(
            [(item.material_key, item.switch_state, item.member) for item in maps],
            [
                (
                    "lc500_centralscreen_on",
                    "on",
                    "vehicles/lc500/textures/lc500_bootscreen_on.png",
                )
            ],
        )

    def test_switch_group_detection_borrows_ignition_on_sibling_states(self) -> None:
        dash = part("dash")
        binding = ArchiveTextureBinding(
            dae_material="lc500_centralscreen",
            material_key="lc500_screens_off",
            materials_member="vehicles/lc500/main.materials.json",
            texture_reference="/vehicles/lc500/textures/lc500_bootscreen.png",
            texture_member="vehicles/lc500/textures/lc500_bootscreen.png",
        )
        own_on = ArchiveMaterialRecord(
            key="lc500_centralscreen_on",
            name="lc500_centralscreen_on",
            map_to="",
            materials_member="vehicles/lc500/main.materials.json",
            base_colour_reference="/vehicles/lc500/textures/lc500_bootscreen_on.png",
            source_material={
                "Stages": [
                    {"baseColorMap": "/vehicles/lc500/textures/lc500_bootscreen_on.png"}
                ]
            },
        )
        sibling_on = ArchiveMaterialRecord(
            key="lc500_screens_on",
            name="lc500_screens_on",
            map_to="",
            materials_member="vehicles/lc500/main.materials.json",
            base_colour_reference="/vehicles/lc500/textures/screen.dds",
            source_material={
                "Stages": [
                    {"baseColorMap": "/vehicles/lc500/textures/screen.dds"}
                ]
            },
        )
        running_on = ArchiveMaterialRecord(
            key="lc500_gauges_needle_on",
            name="lc500_gauges_needle_on",
            map_to="",
            materials_member="vehicles/lc500/main.materials.json",
            base_colour_reference="/vehicles/lc500/textures/needle.png",
            source_material={
                "Stages": [
                    {"emissiveMap": "/vehicles/lc500/textures/needle.png"}
                ]
            },
        )
        archive = SimpleNamespace(
            materials=(own_on, sibling_on, running_on),
            material_switch_targets={
                "lc500_centralscreen": (
                    "lc500_screens_off",
                    "lc500_centralscreen_on",
                    "lc500_gps",
                ),
                "lc500_screens": ("lc500_screens_off", "lc500_screens_on"),
                "lc500_gauges_needle": ("invis", "lc500_gauges_needle_on"),
            },
            material_switch_states={
                "lc500_centralscreen": (
                    ArchiveMaterialSwitchState("off", "lc500_screens_off"),
                    ArchiveMaterialSwitchState("on", "lc500_centralscreen_on"),
                    ArchiveMaterialSwitchState("on_intense", "lc500_gps"),
                ),
                "lc500_screens": (
                    ArchiveMaterialSwitchState("off", "lc500_screens_off"),
                    ArchiveMaterialSwitchState("on", "lc500_screens_on"),
                    ArchiveMaterialSwitchState("on_intense", "lc500_screens_on"),
                ),
                "lc500_gauges_needle": (
                    ArchiveMaterialSwitchState("off", "invis"),
                    ArchiveMaterialSwitchState("on", "lc500_gauges_needle_on"),
                ),
            },
            material_switch_triggers={
                "lc500_centralscreen": ("ignitionLevel",),
                "lc500_screens": ("ignitionLevel",),
                "lc500_gauges_needle": ("running",),
            },
            member_by_lower={
                "vehicles/lc500/textures/lc500_bootscreen.png": (
                    "vehicles/lc500/textures/lc500_bootscreen.png"
                ),
                "vehicles/lc500/textures/lc500_bootscreen_on.png": (
                    "vehicles/lc500/textures/lc500_bootscreen_on.png"
                ),
                "vehicles/lc500/textures/screen.dds": (
                    "vehicles/lc500/textures/screen.dds"
                ),
                "vehicles/lc500/textures/needle.png": (
                    "vehicles/lc500/textures/needle.png"
                ),
            },
            members=(
                "vehicles/lc500/textures/lc500_bootscreen.png",
                "vehicles/lc500/textures/lc500_bootscreen_on.png",
                "vehicles/lc500/textures/screen.dds",
                "vehicles/lc500/textures/needle.png",
            ),
            member_archive_indices={},
        )

        maps = rhd.switch_group_detection_maps_for_candidates(
            archive,
            [(dash, binding)],
            binding.texture_member,
        )

        self.assertEqual(
            [(item.material_key, item.switch_state, item.member) for item in maps],
            [
                (
                    "lc500_screens_on",
                    "on_intense",
                    "vehicles/lc500/textures/screen.dds",
                ),
                (
                    "lc500_centralscreen_on",
                    "on",
                    "vehicles/lc500/textures/lc500_bootscreen_on.png",
                ),
            ],
        )

    def test_shared_atlas_is_rebuilt_once_for_selected_parts(self) -> None:
        dash = part("dash")
        console = part("console")
        binding = ArchiveTextureBinding(
            dae_material="interior",
            material_key="interior",
            materials_member="vehicles/car/main.materials.json",
            texture_reference="/vehicles/car/interior.dds",
            texture_member="vehicles/car/interior.dds",
        )

        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            source = workspace / "interior.dds"
            Image.new("RGBA", (4, 4), (0, 0, 0, 255)).save(source, format="PNG")
            mask = np.ones((4, 4), dtype=bool)
            masks = rhd.DomainMasks(
                mirror=mask,
                rigid=np.zeros((4, 4), dtype=bool),
                conflict_coverage=0.0,
                mirrored_triangles=2,
                rigid_triangles=0,
                parts_analysed=2,
            )
            texture_result = rhd.RhdTextureResult(
                texture_member=binding.texture_member,
                size=(4, 4),
                parts_analysed=2,
                mirrored_triangles=2,
                rigid_triangles=0,
                mirror_coverage=1.0,
                rigid_coverage=0.0,
                conflict_coverage=0.0,
                glyph_regions=0,
                mirrored_glyph_regions=0,
                material_aliases=("interior",),
                png_path=workspace / "interior_rhd.png",
                report={
                    "texture": binding.texture_member,
                    "selected_parts": [{"key": "dash"}, {"key": "console"}],
                },
            )
            export_info = {
                "carrier": {
                    "node_id": "generated_carrier",
                    "geometry_ids": ["generated_carrier_mesh"],
                    "triangle_count": 2,
                },
                "rigid_symmetric_nodes": [],
            }
            events: list[dict[str, object]] = []

            with (
                patch.object(
                    rhd,
                    "texture_bindings_for_parts",
                    return_value={binding.texture_member: [(dash, binding), (console, binding)]},
                ),
                patch.object(rhd, "material_symbols_for_binding", return_value=("interior-material",)),
                patch.object(rhd, "extract_archive_member", return_value=source),
                patch.object(
                    rhd, "build_domain_masks", return_value=masks,
                ) as build_masks,
                patch.object(rhd, "build_rhd_texture", return_value=texture_result) as build_texture,
                patch.object(rhd, "write_blender_preview", return_value=None),
                patch.object(rhd, "sweep_part", return_value=object()),
                patch.object(
                    rhd, "export_transformed_part_dae", return_value=export_info
                ) as export_dae,
            ):
                preview = rhd.export_parts_preview(
                    SimpleNamespace(),
                    SimpleNamespace(),
                    [dash, console],
                    workspace,
                    bake=False,
                    log=lambda *_a: None,
                    progress=events.append,
                )
            self.assertIsNotNone(preview.report_path)
            report = json.loads(preview.report_path.read_text(encoding="utf-8"))

        # Each mesh is measured for its own domain, but the two come out equal,
        # so one correction serves both rather than competing per-part copies.
        self.assertEqual(build_masks.call_count, 2)
        build_texture.assert_called_once()
        self.assertEqual(build_texture.call_args.kwargs["part_scope"], [dash, console])
        self.assertEqual(build_texture.call_args.kwargs["part_group_index"], 0)
        self.assertIs(build_texture.call_args.kwargs["masks"], masks)
        self.assertEqual(export_dae.call_count, 2)
        self.assertEqual(len(preview.dae_paths), 2)
        self.assertEqual(len(report["selected_parts"]), 2)
        self.assertEqual(len(report["texture_jobs"]), 1)
        self.assertEqual(len(report["dae_exports"]), 2)
        self.assertEqual(
            report["texture_detection_inventory"]["candidate_files"],
            [binding.texture_member],
        )
        self.assertEqual(
            report["texture_detection_inventory"]["detection_attempted_files"],
            [binding.texture_member],
        )
        self.assertEqual(
            report["dae_exports"][0]["generated_flexbody_rows"][0]["node_id"],
            "generated_carrier",
        )
        phases = [event["phase"] for event in events]
        self.assertIn("preview_export", phases)
        self.assertIn("resolve_materials", phases)
        self.assertIn("build_domain_masks", phases)
        self.assertIn("export_part_dae", phases)
        self.assertIn("write_preview_report", phases)
        reported_phases = [entry["phase"] for entry in report["phase_timings"]]
        self.assertIn("resolve_materials", reported_phases)
        self.assertIn("build_domain_masks", reported_phases)
        self.assertIn("export_part_dae", reported_phases)
        self.assertIn("write_preview_report", reported_phases)

    def _preview_with_export_results(self, results: list[object]):
        """Run a two-part export whose per-part DAE writes give `results`."""
        dash = part("dash")
        console = part("console")
        binding = ArchiveTextureBinding(
            dae_material="interior",
            material_key="interior",
            materials_member="vehicles/car/main.materials.json",
            texture_reference="/vehicles/car/interior.dds",
            texture_member="vehicles/car/interior.dds",
        )
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            source = workspace / "interior.dds"
            Image.new("RGBA", (4, 4), (0, 0, 0, 255)).save(source, format="PNG")
            mask = np.ones((4, 4), dtype=bool)
            masks = rhd.DomainMasks(
                mirror=mask,
                rigid=np.zeros((4, 4), dtype=bool),
                conflict_coverage=0.0,
                mirrored_triangles=2,
                rigid_triangles=0,
                parts_analysed=2,
            )
            texture_result = rhd.RhdTextureResult(
                texture_member=binding.texture_member,
                size=(4, 4),
                parts_analysed=2,
                mirrored_triangles=2,
                rigid_triangles=0,
                mirror_coverage=1.0,
                rigid_coverage=0.0,
                conflict_coverage=0.0,
                glyph_regions=0,
                mirrored_glyph_regions=0,
                material_aliases=("interior",),
                png_path=workspace / "interior_rhd.png",
                report={"texture": binding.texture_member, "selected_parts": []},
            )
            with (
                patch.object(
                    rhd,
                    "texture_bindings_for_parts",
                    return_value={binding.texture_member: [(dash, binding), (console, binding)]},
                ),
                patch.object(rhd, "material_symbols_for_binding", return_value=("interior-material",)),
                patch.object(rhd, "extract_archive_member", return_value=source),
                patch.object(rhd, "build_domain_masks", return_value=masks),
                patch.object(rhd, "build_rhd_texture", return_value=texture_result),
                patch.object(rhd, "write_blender_preview", return_value=None),
                patch.object(rhd, "sweep_part", return_value=object()),
                patch.object(rhd, "export_transformed_part_dae", side_effect=results),
            ):
                return rhd.export_parts_preview(
                    SimpleNamespace(),
                    SimpleNamespace(),
                    [dash, console],
                    workspace,
                    bake=False,
                    log=lambda *_a: None,
                )

    def test_a_part_with_no_bindable_atlas_does_not_sink_the_others(self) -> None:
        # Scintilla's race console carries only scintilla_main_carbon, a detail
        # material with no base-colour atlas, so it cannot be wired. It used to
        # abort the whole per-DAE export, costing the nine other marked meshes
        # in scintilla.dae their correction with no error raised anywhere.
        export_info = {
            "carrier": {
                "node_id": "generated_carrier",
                "geometry_ids": ["generated_carrier_mesh"],
                "triangle_count": 2,
            },
            "rigid_symmetric_nodes": [],
        }
        preview = self._preview_with_export_results(
            [ValueError("archive texture aliases did not match carbon"), export_info]
        )

        self.assertEqual(len(preview.dae_paths), 1)
        self.assertEqual(len(preview.failed_parts), 1)
        self.assertEqual(preview.failed_parts[0]["source_part"]["key"], "dash")
        self.assertIn("carbon", preview.failed_parts[0]["error"])
        self.assertEqual(
            [entry["source_part"]["key"] for entry in preview.report["dae_exports"]],
            ["console"],
        )

    def test_an_export_where_every_part_fails_is_still_an_error(self) -> None:
        with self.assertRaises(ValueError) as caught:
            self._preview_with_export_results(
                [ValueError("no atlas"), ValueError("no atlas")]
            )
        self.assertIn("No part could be exported", str(caught.exception))


class UnchangedStageIsNotEncodedTests(unittest.TestCase):
    """A stage whose plan is empty must not be re-encoded to reproduce itself."""

    def _build(self, detected_regions):
        dash = part("dash")
        member = "vehicles/lc500/textures/screen.png"
        binding = rhd.MaterialTextureLayerBinding(
            dae_material="lc500_screen",
            material_key="lc500_screen",
            materials_member="vehicles/lc500/main.materials.json",
            texture_reference=f"/{member}",
            texture_member=member,
            stage_key="baseColorMap",
            kind="colour",
        )
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            source = workspace / "screen.png"
            Image.new("RGBA", (4, 4), (0, 0, 0, 255)).save(source)
            mirror = np.zeros((4, 4), dtype=bool)
            mirror[:, :2] = True
            masks = rhd.DomainMasks(
                mirror=mirror,
                rigid=~mirror,
                conflict_coverage=0.0,
                mirrored_triangles=2,
                rigid_triangles=2,
                parts_analysed=1,
            )

            def detect(*_args, **_kwargs):
                return rhd.RegionDetection(
                    source="colour",
                    detected=len(detected_regions),
                    regions=list(detected_regions),
                    rotations=[None] * len(detected_regions),
                )

            def plan(_mirror, regions, _config, _axis_map, _components=None):
                # Whatever was detected, nothing survives into a flip step.
                return [], np.zeros((4, 4), dtype=bool)

            with (
                patch.object(
                    rhd, "scoped_parts_using_material", return_value=[(dash, binding)]
                ),
                patch.object(
                    rhd, "material_symbols_for_binding",
                    return_value=("lc500_screen-material",),
                ),
                patch.object(rhd, "extract_archive_member", return_value=source),
                patch.object(
                    rhd, "detect_flip_regions_by_uv_island", side_effect=detect
                ),
                patch.object(rhd, "plan_island_flips", side_effect=plan),
                patch.object(
                    rhd, "normal_maps_for_layer_bindings", return_value=(),
                ),
            ):
                result = rhd.build_rhd_texture(
                    SimpleNamespace(materials=[]),
                    SimpleNamespace(),
                    member,
                    workspace,
                    config=rhd.RhdTextureConfig(write_debug_overlays=False),
                    part_scope=[dash],
                    masks=masks,
                    log=lambda *_a: None,
                )
            written = sorted(
                p.name for p in workspace.iterdir() if p.name.startswith("screen_rhd")
            )
        return result, written

    def test_an_empty_plan_writes_no_image(self) -> None:
        result, written = self._build([])

        self.assertEqual(result.island_flips, [])
        self.assertEqual(result.in_place_flips, [])
        self.assertIsNone(result.png_path)
        self.assertIsNone(result.dds_path)
        # The plan is still recorded -- it is the evidence nothing was needed.
        self.assertEqual(written, ["screen_rhd.plan.json"])
        self.assertIsNone(result.report["outputs"]["png"])
        self.assertIsNone(result.report["outputs"]["dds"])

    def test_a_skipped_stage_reaches_no_manifest_entry(self) -> None:
        result, _written = self._build([])

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            rhd.write_blender_preview(output, [result], log=lambda *_a: None)
            # No corrected file means no corrected material: the mesh keeps the
            # texture it shipped with.
            self.assertFalse((output / "rhd_materials.json").exists())

    def test_a_planned_flip_still_encodes(self) -> None:
        dash = part("dash")
        member = "vehicles/lc500/textures/screen.png"
        binding = rhd.MaterialTextureLayerBinding(
            dae_material="lc500_screen",
            material_key="lc500_screen",
            materials_member="vehicles/lc500/main.materials.json",
            texture_reference=f"/{member}",
            texture_member=member,
            stage_key="baseColorMap",
            kind="colour",
        )
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            source = workspace / "screen.png"
            Image.new("RGBA", (8, 8), (0, 0, 0, 255)).save(source)
            mirror = np.ones((8, 8), dtype=bool)
            masks = rhd.DomainMasks(
                mirror=mirror,
                rigid=np.zeros((8, 8), dtype=bool),
                conflict_coverage=0.0,
                mirrored_triangles=2,
                rigid_triangles=0,
                parts_analysed=1,
            )

            def detect(*_args, **_kwargs):
                return rhd.RegionDetection(
                    source="colour", detected=1,
                    regions=[(0, 0, 4, 4)], rotations=[None],
                )

            with (
                patch.object(
                    rhd, "scoped_parts_using_material", return_value=[(dash, binding)]
                ),
                patch.object(
                    rhd, "material_symbols_for_binding",
                    return_value=("lc500_screen-material",),
                ),
                patch.object(rhd, "extract_archive_member", return_value=source),
                patch.object(
                    rhd, "detect_flip_regions_by_uv_island", side_effect=detect
                ),
                patch.object(rhd, "normal_maps_for_layer_bindings", return_value=()),
            ):
                result = rhd.build_rhd_texture(
                    SimpleNamespace(materials=[]),
                    SimpleNamespace(),
                    member,
                    workspace,
                    config=rhd.RhdTextureConfig(write_debug_overlays=False),
                    part_scope=[dash],
                    masks=masks,
                    log=lambda *_a: None,
                )
            names = {p.name for p in workspace.iterdir()}

        self.assertTrue(result.island_flips or result.in_place_flips)
        self.assertIsNotNone(result.png_path)
        self.assertIn("screen_rhd.png", names)


class PreviewPngTests(unittest.TestCase):
    """The inspection PNG is scratch; the non-power-of-two PNG is the asset."""

    def _build(self, size, *, write_preview_png):
        dash = part("dash")
        member = "vehicles/lc500/textures/screen.png"
        binding = rhd.MaterialTextureLayerBinding(
            dae_material="lc500_screen",
            material_key="lc500_screen",
            materials_member="vehicles/lc500/main.materials.json",
            texture_reference=f"/{member}",
            texture_member=member,
            stage_key="baseColorMap",
            kind="colour",
        )
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            source = workspace / "screen.png"
            Image.new("RGBA", size, (0, 0, 0, 255)).save(source)
            mirror = np.ones((size[1], size[0]), dtype=bool)
            masks = rhd.DomainMasks(
                mirror=mirror,
                rigid=np.zeros((size[1], size[0]), dtype=bool),
                conflict_coverage=0.0,
                mirrored_triangles=2,
                rigid_triangles=0,
                parts_analysed=1,
            )

            def detect(*_args, **_kwargs):
                return rhd.RegionDetection(
                    source="colour", detected=1,
                    regions=[(0, 0, 4, 4)], rotations=[None],
                )

            with (
                patch.object(
                    rhd, "scoped_parts_using_material", return_value=[(dash, binding)]
                ),
                patch.object(
                    rhd, "material_symbols_for_binding",
                    return_value=("lc500_screen-material",),
                ),
                patch.object(rhd, "extract_archive_member", return_value=source),
                patch.object(
                    rhd, "detect_flip_regions_by_uv_island", side_effect=detect
                ),
                patch.object(rhd, "normal_maps_for_layer_bindings", return_value=()),
            ):
                result = rhd.build_rhd_texture(
                    SimpleNamespace(materials=[]),
                    SimpleNamespace(),
                    member,
                    workspace,
                    config=rhd.RhdTextureConfig(
                        write_debug_overlays=False,
                        write_preview_png=write_preview_png,
                    ),
                    part_scope=[dash],
                    masks=masks,
                    log=lambda *_a: None,
                )
            names = {p.name for p in workspace.iterdir()}
            sizes = {
                p.name: p.stat().st_size
                for p in workspace.iterdir()
                if p.suffix == ".png"
            }
        return result, names, sizes

    def test_a_shipped_dds_needs_no_inspection_png(self) -> None:
        result, names, _sizes = self._build((64, 64), write_preview_png=False)

        self.assertIsNotNone(result.dds_path)
        self.assertIsNone(result.png_path)
        self.assertIn("screen_rhd.dds", names)
        self.assertNotIn("screen_rhd.png", names)

    def test_the_inspection_png_is_kept_when_asked_for(self) -> None:
        result, names, _sizes = self._build((64, 64), write_preview_png=True)

        self.assertIsNotNone(result.dds_path)
        self.assertIsNotNone(result.png_path)
        self.assertIn("screen_rhd.png", names)

    def test_a_non_power_of_two_ships_its_png_even_unasked(self) -> None:
        # Block compression has no such size, so here the PNG is the asset.
        result, names, _sizes = self._build((48, 64), write_preview_png=False)

        self.assertIsNone(result.dds_path)
        self.assertIsNotNone(result.png_path)
        self.assertIn("screen_rhd.png", names)
        self.assertNotIn("screen_rhd.dds", names)

    def test_a_shipped_png_is_deflated_and_a_scratch_one_is_not(self) -> None:
        # Same pixels either way, so any size difference is the compression.
        _shipped, _n1, shipped_sizes = self._build((48, 64), write_preview_png=False)
        self.assertEqual(rhd.SHIPPED_PNG_COMPRESS_LEVEL, 3)
        with tempfile.TemporaryDirectory() as temporary:
            reference = Path(temporary) / "raw.png"
            Image.new("RGBA", (48, 64), (0, 0, 0, 255)).save(
                reference, compress_level=0
            )
            raw_bytes = reference.stat().st_size
        self.assertLess(shipped_sizes["screen_rhd.png"], raw_bytes)


class PerMeshCorrectionJobTests(unittest.TestCase):
    """A correction is scoped to one mesh's use of one material."""

    def _part(self, key: str):
        return SimpleNamespace(key=key, node_id=key, node_name=key, label=key)

    def _binding(self, material: str, stage_key: str = "baseColorMap"):
        return SimpleNamespace(
            dae_material=material,
            material_key=f"{material}_on",
            materials_member="vehicles/lc500/main.materials.json",
            stage_key=stage_key,
            kind="colour",
        )

    def _jobs(self, entries):
        with patch.object(
            rhd,
            "material_symbols_for_binding",
            side_effect=lambda _l, b: (f"{b.dae_material}-material",),
        ):
            return rhd.correction_jobs_for_texture(SimpleNamespace(), entries)

    def test_one_mesh_on_one_material_is_one_job(self) -> None:
        dash = self._part("dash")
        jobs = self._jobs([(dash, self._binding("interior"))])

        self.assertEqual(len(jobs), 1)
        self.assertIs(jobs[0].part, dash)
        self.assertEqual(jobs[0].material, "interior")
        self.assertEqual(jobs[0].symbols, frozenset({"interior-material"}))

    def test_two_meshes_on_one_material_stay_separate(self) -> None:
        # Pooling these is what let the LC500 facelift's rigid domain erase the
        # base interior's mirrored one.
        binding = self._binding("lc500_screens")
        jobs = self._jobs(
            [
                (self._part("lc500_interior"), binding),
                (self._part("lc500_interior_facelift"), binding),
            ]
        )

        self.assertEqual(
            [(job.part.key, job.material) for job in jobs],
            [("lc500_interior", "lc500_screens"),
             ("lc500_interior_facelift", "lc500_screens")],
        )

    def test_two_materials_on_one_mesh_stay_separate(self) -> None:
        # screen.dds carries the 8.9% HVAC strip under lc500_screens and a
        # full-atlas quad under lc500_centralscreen. Unioning their domains
        # made the strip flip about the atlas centre instead of itself.
        interior = self._part("lc500_interior")
        jobs = self._jobs(
            [
                (interior, self._binding("lc500_screens")),
                (interior, self._binding("lc500_centralscreen")),
            ]
        )

        self.assertEqual(
            [job.material for job in jobs],
            ["lc500_screens", "lc500_centralscreen"],
        )
        self.assertEqual(jobs[0].symbols, frozenset({"lc500_screens-material"}))
        self.assertEqual(
            jobs[1].symbols, frozenset({"lc500_centralscreen-material"})
        )

    def test_stage_keys_of_one_material_share_a_job(self) -> None:
        # A base colour and its emissive are the same UV layout.
        interior = self._part("lc500_interior")
        jobs = self._jobs(
            [
                (interior, self._binding("lc500_screens", "baseColorMap")),
                (interior, self._binding("lc500_screens", "emissiveMap")),
            ]
        )

        self.assertEqual(len(jobs), 1)


class PerMeshCorrectionOutputTests(unittest.TestCase):
    """Each job builds its own texture, scoped to its own mesh and material."""

    def _preview(self, entries, mirrored_by_part, layouts=None):
        parts: list[SimpleNamespace] = []
        for part, _binding in entries:
            if not any(part is seen for seen in parts):
                parts.append(part)
        calls: list[dict] = []
        layouts = layouts or {}
        # Meshes get distinct domains unless the test says they share a layout,
        # so an identical mask means a real shared unwrap, not a stub artefact.
        seen_layouts: list[object] = []

        def masks(_loaded, scope, symbols, *_a, **_k):
            mirror = np.zeros((4, 4), dtype=bool)
            if mirrored_by_part.get(scope[0].key, True):
                layout = layouts.get(
                    scope[0].key, (scope[0].key, frozenset(symbols))
                )
                if layout not in seen_layouts:
                    seen_layouts.append(layout)
                mirror[seen_layouts.index(layout):] = True
            return rhd.DomainMasks(
                mirror=mirror, rigid=np.zeros((4, 4), dtype=bool),
                conflict_coverage=0.0, mirrored_triangles=1,
                rigid_triangles=0, parts_analysed=1,
            )

        def build(*_args, **kwargs):
            calls.append(kwargs)
            return rhd.RhdTextureResult(
                texture_member="vehicles/lc500/textures/screen.dds",
                size=(4, 4), parts_analysed=1, mirrored_triangles=1,
                rigid_triangles=0, mirror_coverage=1.0, rigid_coverage=0.0,
                conflict_coverage=0.0, glyph_regions=0, mirrored_glyph_regions=0,
            )

        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            source = workspace / "screen.dds"
            Image.new("RGBA", (4, 4)).save(source, format="PNG")
            with (
                patch.object(
                    rhd, "texture_bindings_for_parts",
                    return_value={"vehicles/lc500/textures/screen.dds": entries},
                ),
                patch.object(
                    rhd, "material_symbols_for_binding",
                    side_effect=lambda _l, b: (f"{b.dae_material}-material",),
                ),
                patch.object(rhd, "extract_archive_member", return_value=source),
                patch.object(rhd, "build_domain_masks", side_effect=masks),
                patch.object(rhd, "build_rhd_texture", side_effect=build),
                patch.object(rhd, "write_blender_preview", return_value=None),
                patch.object(rhd, "sweep_part", return_value=object()),
                patch.object(
                    rhd, "export_transformed_part_dae",
                    return_value={"rigid_symmetric_nodes": []},
                ),
            ):
                rhd.export_parts_preview(
                    SimpleNamespace(), SimpleNamespace(), parts, workspace,
                    bake=False, log=lambda *_a: None,
                )
        return calls

    def _part(self, key: str):
        return SimpleNamespace(
            key=key, node_id=key, node_name=key, label=key, instances=[],
        )

    def _binding(self, material: str):
        return SimpleNamespace(
            dae_material=material, material_key=f"{material}_on",
            materials_member="vehicles/lc500/main.materials.json",
            stage_key="baseColorMap", kind="colour",
        )

    def test_a_single_job_keeps_the_names_it_has_always_had(self) -> None:
        dash = self._part("dash")
        calls = self._preview([(dash, self._binding("interior"))], {})

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["part_group_index"], 0)
        self.assertEqual(calls[0]["part_scope"], [dash])
        self.assertEqual(calls[0]["material_scope"], ("interior",))

    def test_each_mesh_and_material_builds_its_own_texture(self) -> None:
        interior = self._part("lc500_interior")
        facelift = self._part("lc500_interior_facelift")
        calls = self._preview(
            [
                (interior, self._binding("lc500_screens")),
                (interior, self._binding("lc500_centralscreen")),
                (facelift, self._binding("lc500_screens")),
            ],
            {},
        )

        self.assertEqual(
            [(c["part_scope"][0].key, c["material_scope"][0]) for c in calls],
            [
                ("lc500_interior", "lc500_screens"),
                ("lc500_interior", "lc500_centralscreen"),
                ("lc500_interior_facelift", "lc500_screens"),
            ],
        )
        self.assertEqual([c["part_group_index"] for c in calls], [1, 2, 3])

    def test_a_job_with_nothing_mirrored_is_skipped_not_the_texture(self) -> None:
        interior = self._part("lc500_interior")
        facelift = self._part("lc500_interior_facelift")
        calls = self._preview(
            [
                (interior, self._binding("lc500_screens")),
                (facelift, self._binding("lc500_screens")),
            ],
            {"lc500_interior_facelift": False},
        )

        self.assertEqual([c["part_scope"][0].key for c in calls], ["lc500_interior"])

    def test_meshes_sharing_a_uv_layout_share_one_correction(self) -> None:
        # A left and right panel on one unwrap would correct to the same image
        # twice, so the equal domains merge into a single build.
        left = self._part("door_panel_L")
        right = self._part("door_panel_R")
        calls = self._preview(
            [
                (left, self._binding("interior")),
                (right, self._binding("interior")),
            ],
            {},
            layouts={"door_panel_L": "shared", "door_panel_R": "shared"},
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["part_scope"], [left, right])
        self.assertEqual(calls[0]["part_group_index"], 0)


class SplitTextureManifestTests(unittest.TestCase):
    """The manifest has to say which meshes each corrected copy is for."""

    def _result(self, group_index: int, scope: list[str]) -> rhd.RhdTextureResult:
        stem = "screen" if group_index < 2 else f"screen_{group_index}"
        report = {
            "layer_bindings": [
                {
                    "dae_material": "lc500_screens",
                    "material_key": "lc500_screens_on",
                    "materials_member": "vehicles/lc500/main.materials.json",
                    "stage_key": "baseColorMap",
                    "kind": "colour",
                }
            ],
            "source_materials": [
                {
                    "key": "lc500_screens_on",
                    "aliases": ["lc500_screens_on"],
                    "materialsMember": "vehicles/lc500/main.materials.json",
                    "material": {"Stages": [{}]},
                }
            ],
            "switch_base_aliases": [],
            "outputs": {"preview": None},
        }
        if group_index:
            report["part_group"] = group_index
            report["part_scope"] = scope
            report["material_scope"] = ["lc500_screens"]
        return rhd.RhdTextureResult(
            texture_member="vehicles/lc500/textures/screen.dds",
            size=(4, 4), parts_analysed=1, mirrored_triangles=1,
            rigid_triangles=0, mirror_coverage=1.0, rigid_coverage=0.0,
            conflict_coverage=0.0, glyph_regions=1, mirrored_glyph_regions=1,
            material_aliases=("lc500_screens", "lc500_screens_on"),
            png_path=Path(f"{stem}_rhd.png"),
            dds_path=Path(f"{stem}_rhd.dds"),
            report=report,
        )

    def _manifest(self, results: list[rhd.RhdTextureResult]) -> dict:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            rhd.write_blender_preview(output, results, log=lambda *_a: None)
            return json.loads((output / "rhd_materials.json").read_text())

    def test_an_unscoped_correction_names_no_parts(self) -> None:
        materials = self._manifest([self._result(0, [])])["materials"]

        self.assertEqual(len(materials), 1)
        self.assertNotIn("partKeys", materials[0])

    def _tell_tale(self, index: int, dae_material: str) -> rhd.RhdTextureResult:
        """One of several DAE materials sharing a single materials-JSON entry."""
        stem = "buttons" if index < 2 else f"buttons_{index}"
        return rhd.RhdTextureResult(
            texture_member="vehicles/lc500/faceliftbuttons.png",
            size=(4, 4), parts_analysed=1, mirrored_triangles=1,
            rigid_triangles=0, mirror_coverage=1.0, rigid_coverage=0.0,
            conflict_coverage=0.0, glyph_regions=1, mirrored_glyph_regions=1,
            material_aliases=(dae_material, "lc500_dashlights"),
            png_path=Path(f"{stem}_rhd.png"),
            dds_path=Path(f"{stem}_rhd.dds"),
            report={
                "layer_bindings": [
                    {
                        "dae_material": dae_material,
                        "material_key": "lc500_dashlights",
                        "materials_member": "vehicles/lc500/main.materials.json",
                        "stage_key": "opacityMap",
                        "kind": "scalar",
                    }
                ],
                "source_materials": [
                    {
                        "key": "lc500_dashlights",
                        "aliases": [
                            "lc500_intsignal_L",
                            "lc500_intsignal_R",
                            "lc500_inthighbeam",
                        ],
                        "materialsMember": "vehicles/lc500/main.materials.json",
                        "material": {"Stages": [{}]},
                    }
                ],
                "switch_base_aliases": [],
                "outputs": {"preview": None},
                "part_group": index,
                "part_scope": ["lc500_interior"],
                "material_scope": [dae_material],
            },
        )

    def test_materials_sharing_one_json_entry_keep_their_own_textures(self) -> None:
        # The LC500's turn-signal and high-beam tell-tales are three DAE
        # materials over one lc500_dashlights entry, each on its own island.
        # Keyed only on that entry they folded together and two of the three
        # corrected files were orphaned.
        materials = self._manifest(
            [
                self._tell_tale(1, "lc500_intsignal_L"),
                self._tell_tale(2, "lc500_intsignal_R"),
                self._tell_tale(3, "lc500_inthighbeam"),
            ]
        )["materials"]

        self.assertEqual(len(materials), 3)
        self.assertEqual(
            [entry["maps"]["opacityMap"] for entry in materials],
            ["buttons_rhd.png", "buttons_2_rhd.png", "buttons_3_rhd.png"],
        )
        # A scoped correction must not claim its siblings' aliases, or they
        # would all retarget onto whichever was minted first.
        self.assertEqual(
            [entry["aliases"][0] for entry in materials],
            ["lc500_intsignal_L", "lc500_intsignal_R", "lc500_inthighbeam"],
        )
        for entry in materials:
            self.assertNotIn(
                "lc500_inthighbeam",
                entry["aliases"][1:],
            )

    def test_two_corrections_of_one_texture_stay_separate_entries(self) -> None:
        materials = self._manifest(
            [self._result(1, ["interior"]), self._result(2, ["facelift"])]
        )["materials"]

        # Same alias, same source material, different meshes and different
        # files: folding these together is what left the facelift mirrored.
        self.assertEqual(len(materials), 2)
        self.assertEqual(
            [entry["partKeys"] for entry in materials],
            [["interior"], ["facelift"]],
        )
        self.assertEqual(
            [entry["maps"]["baseColorMap"] for entry in materials],
            ["screen_rhd.png", "screen_2_rhd.png"],
        )


class MaterialAliasOrderTests(unittest.TestCase):
    """Aliases must come out in the same order whatever the hash seed.

    They are assembled from a set of COLLADA symbols, and Python randomises
    string hashing per process. A pooled correction pass and a serial one over
    the V60 agreed on all 59 plans except the order of this one field, because
    the pool's workers hashed differently from the parent.
    """

    def _binding(self, material: str, key: str):
        return SimpleNamespace(dae_material=material, material_key=key)

    def test_the_binding_leads_and_the_symbols_follow_in_order(self) -> None:
        candidates = [(SimpleNamespace(key="dash"), self._binding("buttons", "buttons_off"))]
        symbols = {"buttons-material", "buttons_off-material", "aaa-material"}

        self.assertEqual(
            rhd.material_aliases_for_candidates(candidates, symbols),
            (
                "buttons",
                "buttons_off",
                "aaa-material",
                "buttons-material",
                "buttons_off-material",
            ),
        )

    def test_the_order_does_not_follow_the_sets_iteration(self) -> None:
        # Sets of these strings iterate differently per process; building the
        # same aliases from equal sets constructed differently must not.
        candidates = [(SimpleNamespace(key="dash"), self._binding("m", "m_off"))]
        forward = {f"sym{index}-material" for index in range(12)}
        backward = {f"sym{index}-material" for index in reversed(range(12))}
        self.assertEqual(
            rhd.material_aliases_for_candidates(candidates, forward),
            rhd.material_aliases_for_candidates(candidates, backward),
        )

    def test_a_name_is_kept_at_its_first_appearance(self) -> None:
        candidates = [
            (SimpleNamespace(key="a"), self._binding("m", "m_off")),
            (SimpleNamespace(key="b"), self._binding("m", "m_on")),
        ]
        aliases = rhd.material_aliases_for_candidates(candidates, {"m-material"})
        self.assertEqual(aliases, ("m", "m_off", "m-material", "m_on"))
        self.assertEqual(len(aliases), len(set(aliases)))

    def test_an_empty_name_is_dropped(self) -> None:
        candidates = [(SimpleNamespace(key="a"), self._binding("m", ""))]
        self.assertEqual(
            rhd.material_aliases_for_candidates(candidates, {"m-material"}),
            ("m", "m-material"),
        )

if __name__ == "__main__":
    unittest.main()
