"""Detecting every layer of a material at the material's finest resolution.

The defect these pin: the detector measures every distance in absolute texels,
so a half-resolution companion map has the same parameters cover twice the
surface.  On the ardente that welded three switch glyphs -- 36 to 49 texels
apart on the 4096 colour map, 18 to 24 on the 2048 AO map -- into one blob,
which the shared-layer ledger then scaled up and unioned with the colour
layer's correctly separated glyphs.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mesh_segmentation_transform import mirror_texture_for_rhd as rhd
from mesh_segmentation_transform.glyph_lab_detection import (
    DetectedMark,
    scale_marks_to_native,
    upscale_domain_crops,
    upscale_for_detection,
)


def task(
    member: str,
    material: str,
    scope: tuple[str, ...] = ("a",),
    *,
    detect: bool = True,
):
    parts = tuple(
        type("P", (), {"key": name})() for name in scope
    )
    return rhd.TextureCorrectionTask(
        member=member, material=material, part_scope=parts,
        masks=None, part_group_index=0, detect=detect,
    )


class UpscaleTests(unittest.TestCase):
    def test_a_factor_of_one_returns_the_image_unchanged(self) -> None:
        image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        self.assertIs(upscale_for_detection(image, 1), image)

    def test_samples_are_repeated_not_interpolated(self) -> None:
        """Interpolation would manufacture contrast the map does not contain."""
        image = np.array([[0, 255]], dtype=np.uint8)
        grown = upscale_for_detection(image, 2)
        self.assertEqual(grown.shape, (2, 4))
        self.assertEqual(sorted(np.unique(grown).tolist()), [0, 255])
        self.assertTrue((grown[:, :2] == 0).all())
        self.assertTrue((grown[:, 2:] == 255).all())

    def test_domain_crops_move_with_their_masks(self) -> None:
        crops = [(3, 5, np.ones((2, 4), dtype=bool))]
        grown = upscale_domain_crops(crops, 2)
        x, y, mask = grown[0]
        self.assertEqual((x, y), (6, 10))
        self.assertEqual(mask.shape, (4, 8))

    def test_two_marks_a_gap_apart_stay_apart_when_upscaled(self) -> None:
        """The property the whole change rests on, stated directly."""
        native = np.zeros((1, 40), dtype=np.uint8)
        native[0, 5] = 255
        native[0, 25] = 255          # 20 texels apart natively
        grown = upscale_for_detection(native, 2)
        lit = np.flatnonzero(grown[0])
        self.assertEqual(int(lit.max() - lit.min()), 41)   # ~40 apart at 2x


class ScaleBackTests(unittest.TestCase):
    def test_a_mark_returns_to_its_native_frame(self) -> None:
        mask = np.ones((8, 12), dtype=bool)
        marks = [DetectedMark((20, 16, 12, 8), mask, None)]
        native = scale_marks_to_native(marks, 2)[0]
        self.assertEqual(native.bounds, (10, 8, 6, 4))
        self.assertEqual(native.mask.shape, (4, 6))

    def test_a_rotation_scales_with_the_bounds(self) -> None:
        corners = ((20.0, 16.0), (32.0, 16.0), (32.0, 24.0), (20.0, 24.0))
        marks = [DetectedMark((20, 16, 12, 8), np.ones((8, 12), dtype=bool), corners)]
        native = scale_marks_to_native(marks, 2)[0]
        self.assertEqual(native.rotation[0], (10.0, 8.0))
        self.assertEqual(native.rotation[2], (16.0, 12.0))

    def test_a_stroke_thinner_than_the_factor_keeps_its_rectangle(self) -> None:
        """An empty stencil would silently decline the flip; the box would not."""
        mask = np.zeros((8, 12), dtype=bool)
        mask[1, 1] = True
        marks = [DetectedMark((20, 16, 12, 8), mask, None)]
        native = scale_marks_to_native(marks, 4)[0]
        self.assertTrue(native.mask.any())

    def test_a_factor_of_one_is_a_pass_through(self) -> None:
        marks = [DetectedMark((1, 2, 3, 4), np.ones((4, 3), dtype=bool), None)]
        self.assertIs(scale_marks_to_native(marks, 1), marks)


class GroupResolutionTests(unittest.TestCase):
    def test_every_layer_of_a_material_takes_the_finest_size(self) -> None:
        tasks = [
            task("colour.dds", "interior"),
            task("ao.data.dds", "interior"),
            task("r.data.dds", "interior"),
        ]
        sizes = {
            "colour.dds": (4096, 4096),
            "ao.data.dds": (2048, 2048),
            "r.data.dds": (2048, 2048),
        }
        rhd.assign_group_detection_size(tasks, sizes.get)
        for entry in tasks:
            self.assertEqual(entry.detection_size, (4096, 4096))

    def test_a_different_material_keeps_its_own_reference(self) -> None:
        """The ledger never crosses materials, so neither may the resolution."""
        tasks = [
            task("colour.dds", "interior"),
            task("badge.dds", "badges"),
        ]
        sizes = {"colour.dds": (4096, 4096), "badge.dds": (512, 512)}
        rhd.assign_group_detection_size(tasks, sizes.get)
        self.assertEqual(tasks[0].detection_size, (4096, 4096))
        self.assertEqual(tasks[1].detection_size, (512, 512))

    def test_a_different_part_scope_is_a_different_group(self) -> None:
        tasks = [
            task("colour.dds", "interior", ("dash",)),
            task("ao.data.dds", "interior", ("door",)),
        ]
        sizes = {"colour.dds": (4096, 4096), "ao.data.dds": (1024, 1024)}
        rhd.assign_group_detection_size(tasks, sizes.get)
        self.assertEqual(tasks[0].detection_size, (4096, 4096))
        self.assertEqual(tasks[1].detection_size, (1024, 1024))

    def test_an_unreadable_member_does_not_stop_the_others(self) -> None:
        tasks = [task("colour.dds", "interior"), task("broken.dds", "interior")]
        sizes = {"colour.dds": (4096, 4096), "broken.dds": None}
        rhd.assign_group_detection_size(tasks, sizes.get)
        self.assertEqual(tasks[0].detection_size, (4096, 4096))

    def test_a_correction_only_layer_does_not_raise_detection_resolution(self) -> None:
        tasks = [
            task("colour.dds", "interior"),
            task("roughness.dds", "interior", detect=False),
        ]
        sizes = {"colour.dds": (4096, 4096), "roughness.dds": (8192, 8192)}
        rhd.assign_group_detection_size(tasks, sizes.get)
        self.assertEqual(tasks[0].detection_size, (4096, 4096))

    def test_only_detection_sources_enter_the_shared_evidence_prepass(self) -> None:
        source = task("colour.dds", "interior")
        correction_only = task("roughness.dds", "interior", detect=False)
        evidence_members: list[str] = []
        corrected_members: list[str] = []

        def record(entry, *_args):
            evidence_members.append(entry.member)

        def correct(entry, *_args):
            corrected_members.append(entry.member)
            return rhd.TextureCorrectionTaskOutcome(None, [], None)

        with (
            patch.object(
                rhd, "_record_texture_task_layer_evidence", side_effect=record
            ),
            patch.object(rhd, "_run_texture_correction_task", side_effect=correct),
        ):
            outcomes = rhd._run_texture_correction_task_group(
                (source, correction_only), Path("."), {}
            )

        self.assertEqual(evidence_members, ["colour.dds"])
        self.assertEqual(corrected_members, ["colour.dds", "roughness.dds"])
        self.assertEqual(len(outcomes), 2)


class TextureStagePolicyTests(unittest.TestCase):
    def test_direct_detection_sources_are_the_visual_layer_family(self) -> None:
        for stage_key in (
            "clearCoatBottomNormalMap",
            "baseColorMap",
            "colorMap",
            "diffuseMap",
            "opacityMap",
            "emissiveMap",
            "overlayMap",
        ):
            self.assertIn(stage_key, rhd.MATERIAL_TEXTURE_STAGE_KINDS)
            self.assertTrue(rhd.stage_keys_allow_detection((stage_key,)))

    def test_response_layers_are_correction_only(self) -> None:
        for stage_key in (
            "ambientOcclusionMap",
            "roughnessMap",
            "metallicMap",
            "specularMap",
            "specularityMap",
            "clearCoatMap",
            "clearCoatRoughnessMap",
            "reflectivityMap",
            "colorPaletteMap",
            "lightMap",
        ):
            self.assertIn(stage_key, rhd.MATERIAL_TEXTURE_STAGE_KINDS)
            self.assertFalse(rhd.stage_keys_allow_detection((stage_key,)))

    def test_normal_and_ao_maps_are_reenabled_after_the_diagnostic_build(self) -> None:
        self.assertTrue(rhd.stage_keys_allow_detection(("normalMap",)))
        self.assertFalse(rhd.stage_keys_allow_detection(("ambientOcclusionMap",)))
        for stage_key in ("normalMap", "ambientOcclusionMap"):
            self.assertIn(stage_key, rhd.MATERIAL_TEXTURE_STAGE_KINDS)
            self.assertTrue(rhd.stage_keys_allow_correction((stage_key,)))
        self.assertEqual(
            rhd.MATERIAL_TEXTURE_STAGE_KINDS["ambientOcclusionMap"],
            "occlusion",
        )

    def test_annotation_legacy_and_tiled_detail_layers_are_not_jobs(self) -> None:
        for stage_key in (
            "annotationMap",
            "baseTex",
            "bumpMap",
            "bumpTex",
            "detailTex",
            "overlayTex",
            "detailMap",
            "detailNormalMap",
            "ambientOcclusionDetailMap",
            "roughnessDetailMap",
            "metallicDetailMap",
            "opacityDetailMap",
            "macroMap",
        ):
            self.assertNotIn(stage_key, rhd.MATERIAL_TEXTURE_STAGE_KINDS)


class DetectionScalingTests(unittest.TestCase):
    """The factor is adopted only where it is exact, matching rescale_plan."""

    @staticmethod
    def factor_used(native, reference):
        seen = {}
        import mesh_segmentation_transform.mirror_texture_for_rhd as module

        def spy(name, bgr, crops, kind, **kwargs):
            seen["width"] = bgr.shape[1]
            return []

        original = module.detect_layer_marks
        module.detect_layer_marks = spy
        try:
            module.detect_layer_regions_with_glyph_lab(
                np.zeros((native[1], native[0], 3), dtype=np.uint8),
                np.ones((native[1], native[0]), dtype=bool),
                np.ones((native[1], native[0]), dtype=bool),
                "colour", "colour", (),
                rhd.DEFAULT_RHD_CONFIG,
                log=lambda *_a: None,
                detection_size=reference,
            )
        finally:
            module.detect_layer_marks = original
        return seen.get("width", native[0]) // native[0]

    def test_an_exact_double_is_adopted(self) -> None:
        self.assertEqual(self.factor_used((64, 64), (128, 128)), 2)

    def test_a_non_integer_ratio_is_declined(self) -> None:
        self.assertEqual(self.factor_used((64, 64), (96, 96)), 1)

    def test_a_mismatched_aspect_is_declined(self) -> None:
        self.assertEqual(self.factor_used((64, 64), (128, 192)), 1)

    def test_a_layer_already_at_the_reference_is_untouched(self) -> None:
        self.assertEqual(self.factor_used((64, 64), (64, 64)), 1)

    def test_a_layer_larger_than_the_reference_is_never_downscaled(self) -> None:
        self.assertEqual(self.factor_used((128, 128), (64, 64)), 1)


if __name__ == "__main__":
    unittest.main()
