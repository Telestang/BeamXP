from __future__ import annotations

import tempfile
import unittest
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from mesh_segmentation_transform.beamxp_transform_sym_mesh_POC import (
    ArchiveTextureBinding,
    DaePart,
    GeometryInstance,
)
from mesh_segmentation_transform import mirror_texture_for_rhd as rhd


def part(key: str) -> DaePart:
    return DaePart(
        key=key,
        label=key,
        node_id=key,
        node_name=key,
        matrix=np.eye(4),
        instances=(GeometryInstance(f"{key}-mesh"),),
    )


class SurfaceFlipAxisTests(unittest.TestCase):
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


class MultiPartPreviewTests(unittest.TestCase):
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
                patch.object(rhd, "build_domain_masks", return_value=masks) as build_masks,
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

        build_masks.assert_called_once()
        build_texture.assert_called_once()
        self.assertEqual(build_texture.call_args.kwargs["part_scope"], [dash, console])
        self.assertIs(build_texture.call_args.kwargs["masks"], masks)
        self.assertEqual(export_dae.call_count, 2)
        self.assertEqual(len(preview.dae_paths), 2)
        self.assertEqual(len(report["selected_parts"]), 2)
        self.assertEqual(len(report["texture_jobs"]), 1)
        self.assertEqual(len(report["dae_exports"]), 2)
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


if __name__ == "__main__":
    unittest.main()
