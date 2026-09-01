"""Optional source-UV geometry sidecars for Glyph Lab corpora."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from glyph_lab.corpus import (
    MANIFEST_NAME,
    UvDomainGeometry,
    load_corpus,
    write_texture,
)
from glyph_lab.uv_geometry import (
    DEFAULT_SIMPLE_UV_GEOMETRY_CONFIG,
    fit_domain_seam_parallelogram,
    fit_seam_parallelogram,
    fit_source_seam_parallelogram,
    seam_region_axes,
    seam_region_axis_candidates,
    seam_region_parallelograms,
    source_seam_axis_candidates,
    source_seam_axis_measurements,
    source_seam_guides,
    wrapped_axis_copies,
)


class GeometrySidecarTests(unittest.TestCase):
    def test_an_unwrapped_axis_returns_to_its_visible_atlas_copy(self) -> None:
        copies = wrapped_axis_copies(
            ((7_287.5, 100.0), (7_287.5, 300.0)), (4_096, 4_096),
        )

        self.assertEqual(copies, (((3_191.5, 100.0), (3_191.5, 300.0)),))

    def test_mesh_seams_supply_a_sheared_parallelogram_frame(self) -> None:
        image = Image.new("1", (100, 80))
        corners = ((20, 20), (70, 25), (60, 60), (10, 55))
        ImageDraw.Draw(image).line([*corners, corners[0]], fill=1)

        fitted = fit_seam_parallelogram(np.asarray(image, dtype=bool))

        self.assertIsNotNone(fitted)
        np.testing.assert_allclose(sorted(fitted), sorted(corners), atol=0.01)

    def test_mesh_seams_reject_a_non_parallel_quadrilateral(self) -> None:
        image = Image.new("1", (100, 80))
        corners = ((20, 20), (70, 20), (30, 60), (10, 60))
        ImageDraw.Draw(image).line([*corners, corners[0]], fill=1)

        self.assertIsNone(fit_seam_parallelogram(np.asarray(image, dtype=bool)))

    def test_mesh_seams_reject_an_axis_aligned_parallelogram(self) -> None:
        image = Image.new("1", (100, 80))
        corners = ((20, 20), (70, 20), (70, 60), (20, 60))
        ImageDraw.Draw(image).line([*corners, corners[0]], fill=1)

        self.assertIsNone(fit_seam_parallelogram(np.asarray(image, dtype=bool)))

    def test_source_seams_reject_an_axis_aligned_parallelogram(self) -> None:
        corners = ((0.2, 0.2), (0.7, 0.2), (0.7, 0.6), (0.2, 0.6))
        segments = tuple(
            (first, second)
            for first, second in zip(corners, (*corners[1:], corners[0]), strict=True)
        )

        self.assertIsNone(fit_source_seam_parallelogram(segments, (100, 100)))

    def test_domain_fit_uses_nonzero_even_seam_extents(self) -> None:
        """Extent magnitudes are data, not a bit mask for the domain label."""
        outline = Image.new("1", (100, 80))
        corners = ((20, 20), (70, 25), (60, 60), (10, 55))
        ImageDraw.Draw(outline).line([*corners, corners[0]], fill=1)
        domains = np.zeros((80, 100), dtype=np.uint16)
        domains[10:70, 5:75] = 1
        extents = np.asarray(outline, dtype=np.uint16) * 36

        fitted = fit_domain_seam_parallelogram(domains, extents, 1)

        self.assertIsNotNone(fitted)
        np.testing.assert_allclose(sorted(fitted), sorted(corners), atol=0.01)

    def test_disconnected_seam_regions_never_share_a_frame(self) -> None:
        image = Image.new("1", (140, 80))
        left = ((10, 15), (50, 20), (45, 50), (5, 45))
        right = ((85, 20), (125, 25), (120, 55), (80, 50))
        draw = ImageDraw.Draw(image)
        draw.line([*left, left[0]], fill=1)
        draw.line([*right, right[0]], fill=1)
        domains = np.ones((80, 140), dtype=np.uint16)

        frames = seam_region_parallelograms(domains, np.asarray(image, dtype=bool))

        self.assertEqual(len(frames), 2)
        self.assertEqual(
            {tuple(sorted(corners)) for _label, corners in frames},
            {tuple(sorted(left)), tuple(sorted(right))},
        )

    def test_a_symmetry_guide_uses_the_seam_region_coordinates(self) -> None:
        seam = np.zeros((32, 40), dtype=bool)
        seam[4:28, 6] = seam[4:28, 33] = True
        seam[4, 6:34] = seam[27, 6:34] = True

        axes = seam_region_axes(seam, (100, 200))

        self.assertTrue(axes)
        self.assertTrue(
            all(100 <= point[0] < 140 and 200 <= point[1] < 232
                for axis in axes for point in axis),
            f"axis escaped its seam region: {axes}",
        )

    def test_a_closed_seam_frame_scores_its_enclosed_area_on_both_axes(self) -> None:
        seam = np.zeros((32, 40), dtype=bool)
        seam[4:28, 6] = seam[4:28, 33] = True
        seam[4, 6:34] = seam[27, 6:34] = True

        candidates = seam_region_axis_candidates(seam, (0, 0))

        self.assertEqual(len(candidates), 2)
        self.assertTrue(all(score == 1.0 for score, _axis in candidates))

    def test_a_wrapped_source_frame_is_scored_before_tile_clipping(self) -> None:
        corners = ((-0.03, 0.2), (0.08, 0.2), (0.08, 0.3), (-0.03, 0.3))
        segments = tuple(
            (first, second)
            for first, second in zip(corners, (*corners[1:], corners[0]), strict=True)
        )

        candidates = source_seam_axis_candidates(segments, (1024, 1024))

        self.assertEqual(len(candidates), 2)
        self.assertTrue(all(score == 1.0 for score, _axis in candidates))

    def test_source_frame_rejects_extra_boundary_that_a_hull_would_hide(self) -> None:
        corners = ((0.2, 0.2), (0.8, 0.25), (0.75, 0.8), (0.15, 0.75))
        boundary = tuple(
            (first, second)
            for first, second in zip(corners, (*corners[1:], corners[0]), strict=True)
        ) + (((0.3, 0.5), (0.7, 0.5)),)

        loose = source_seam_guides(boundary, (100, 100), replace(
            DEFAULT_SIMPLE_UV_GEOMETRY_CONFIG,
            seam_symmetry_min_score=1.1,
            parallelogram_min_boundary_coverage=0.80,
        ))
        strict = source_seam_guides(
            boundary, (100, 100), replace(
                DEFAULT_SIMPLE_UV_GEOMETRY_CONFIG,
                seam_symmetry_min_score=1.1,
                parallelogram_min_boundary_coverage=0.95,
            ),
        )

        self.assertTrue(any(kind.startswith("parallelogram") for kind, _data, _score in loose))
        self.assertEqual(strict, ())

    def test_symmetry_can_limit_unmirrored_area_in_texels(self) -> None:
        corners = ((0.2, 0.2), (0.8, 0.2), (0.8, 0.6),
                   (0.5, 0.6), (0.5, 0.8), (0.2, 0.8))
        boundary = tuple(
            (first, second)
            for first, second in zip(corners, (*corners[1:], corners[0]), strict=True)
        )
        measurements = source_seam_axis_measurements(boundary, (100, 100))
        mismatch = min(round(area * (1.0 - score))
                       for area, score, _axis in measurements)
        base = replace(
            DEFAULT_SIMPLE_UV_GEOMETRY_CONFIG,
            seam_symmetry_min_score=0.8,
        )

        rejected = source_seam_guides(boundary, (100, 100), replace(
            base, seam_symmetry_max_unmirrored_area_px=mismatch - 1,
        ))
        accepted = source_seam_guides(boundary, (100, 100), replace(
            base, seam_symmetry_max_unmirrored_area_px=mismatch,
        ))

        self.assertFalse(any(kind == "symmetric" for kind, _data, _score in rejected))
        self.assertTrue(any(kind == "symmetric" for kind, _data, _score in accepted))

    def test_geometry_round_trips_without_becoming_a_detector_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            colour = np.zeros((16, 16, 3), dtype=np.uint8)
            domains = np.zeros((16, 16), dtype=np.uint16)
            domains[2:14, 2:14] = 1
            geometry = (
                UvDomainGeometry(0, "parallelogram", segments=(
                    ((0.0, 8.0), (2.0, 8.0)), ((13.0, 8.0), (15.0, 8.0)),
                )),
                UvDomainGeometry(
                    1, "symmetric", axis=((8.0, 2.0), (8.0, 13.0)),
                    symmetry_score=0.98,
                ),
                UvDomainGeometry(1, "parallelogram", corners=(
                    (2.0, 2.0), (13.0, 2.0), (13.0, 13.0), (2.0, 13.0),
                )),
                UvDomainGeometry(1, "parallelogram", segments=(
                    ((0.0, 8.0), (2.0, 8.0)), ((13.0, 8.0), (15.0, 8.0)),
                )),
                UvDomainGeometry(1, "seam", source_segments=(
                    ((0.0, 0.0), (1.0, 0.0)), ((1.0, 0.0), (1.0, 1.0)),
                )),
            )
            record = write_texture(directory, "atlas", colour, domains, None,
                                   uv_geometry=geometry)
            (directory / MANIFEST_NAME).write_text(json.dumps({"textures": [record]}))

            loaded = load_corpus(directory)[0]
            self.assertEqual(loaded.uv_geometry, geometry)

            (directory / "atlas.geometry.json").write_text("not json")
            self.assertEqual(load_corpus(directory)[0].uv_geometry, ())

    def test_geometry_context_round_trips_independently_of_raw_seams(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            colour = np.zeros((8, 8, 3), dtype=np.uint8)
            domains = np.ones((8, 8), dtype=np.uint16)
            raw_seams = np.zeros((8, 8), dtype=np.uint16)
            raw_seams[0, :] = 7
            geometry_seams = np.zeros((8, 8), dtype=bool)
            geometry_seams[2, 2:6] = True
            record = write_texture(
                directory, "atlas", colour, domains, None, raw_seams,
                geometry_seams=geometry_seams,
            )
            (directory / MANIFEST_NAME).write_text(json.dumps({"textures": [record]}))

            loaded = load_corpus(directory)[0]
            np.testing.assert_array_equal(loaded.seam_extents, raw_seams)
            np.testing.assert_array_equal(loaded.geometry_seams, geometry_seams)

            (directory / "atlas.geometry.json").write_text(json.dumps([
                {"label": 1, "kind": "symmetric", "axis": [[8], [8]]},
            ]))
            self.assertEqual(load_corpus(directory)[0].uv_geometry, ())

    def test_a_pre_geometry_corpus_stays_valid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            colour = np.zeros((8, 8, 3), dtype=np.uint8)
            domains = np.ones((8, 8), dtype=np.uint16)
            record = write_texture(directory, "legacy", colour, domains, None)
            (directory / MANIFEST_NAME).write_text(json.dumps({"textures": [record]}))
            self.assertEqual(load_corpus(directory)[0].uv_geometry, ())
