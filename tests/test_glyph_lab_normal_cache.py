"""Sharing the normal-map derivations across one material's layers.

A cache that returns a *different* answer from the uncached path is worse than
no cache: it makes a build depend on which layer happened to run first.  These
pin the equivalence, and the key that keeps two genuinely different inputs
apart.
"""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import glyph_lab.detect as glyph_detect
from glyph_lab.normal_map import absence_of_slopes, edge_steepness
from mesh_segmentation_transform.glyph_lab_detection import (
    DEFAULT_GLYPH_LAB_CONFIG,
    detect_layer_marks,
    edge_steepness_for,
    flat_ground_config_key,
    flat_ground_for,
    normal_evidence_inputs,
)
from mesh_segmentation_transform.production_texture_detection import (
    ProductionDetectionSession,
)


def normal_map(size: int = 64) -> np.ndarray:
    """A flat surface with a raised square and a slope-free plateau inside it."""
    normal = np.full((size, size, 3), (128, 128, 255), dtype=np.uint8)
    normal[16:48, 16:48, 0] = 200
    normal[22:42, 22:42, 0] = 128
    normal[30:34, 8:56, 1] = 210
    return normal


def crops(size: int = 64) -> list[tuple[int, int, np.ndarray]]:
    return [(0, 0, np.ones((size, size), dtype=bool))]


def colour_map(size: int = 64) -> np.ndarray:
    colour = np.zeros((size, size, 3), dtype=np.uint8)
    colour[20:28, 12:52] = 240
    return colour


class EquivalenceTests(unittest.TestCase):
    def test_the_cached_steepness_matches_the_detectors_own(self) -> None:
        normal = normal_map()
        domains, evidence = normal_evidence_inputs(normal, crops(), "normal", None)
        self.assertTrue(np.array_equal(
            edge_steepness_for(evidence, domains),
            edge_steepness(evidence, domains),
        ))

    def test_the_cached_flat_ground_matches_the_detectors_own(self) -> None:
        """Built through a probe texture, so it cannot drift from the real one."""
        from glyph_lab.source import texture_from_layer

        normal = normal_map()
        colour = colour_map()
        texture = texture_from_layer("t", colour, crops(), normal_rgb=normal)
        config = DEFAULT_GLYPH_LAB_CONFIG
        unaided = absence_of_slopes(
            texture,
            boundary_skip_px=config.normal.boundary_skip_px,
            slope_threshold=config.normal.slope_threshold,
            absence_kernel_px=config.normal.absence_kernel_px,
            max_slope_texels=config.normal.max_slope_texels,
        )
        domains, evidence = normal_evidence_inputs(colour, crops(), "colour", normal)
        cached = flat_ground_for(evidence, domains, config)
        self.assertIsNotNone(unaided)
        self.assertTrue(np.array_equal(cached, unaided))

    def test_a_layer_detects_the_same_marks_with_and_without_the_cache(self) -> None:
        """The end-to-end claim: supplying the evidence changes no answer."""
        normal, colour = normal_map(), colour_map()
        domains, evidence = normal_evidence_inputs(colour, crops(), "colour", normal)
        flat = flat_ground_for(evidence, domains, DEFAULT_GLYPH_LAB_CONFIG)

        unaided = detect_layer_marks("t", colour, crops(), "colour", normal_rgb=normal)
        supplied = detect_layer_marks(
            "t", colour, crops(), "colour", normal_rgb=normal, flat_ground=flat,
        )
        self.assertEqual(
            [mark.bounds for mark in unaided], [mark.bounds for mark in supplied]
        )
        for first, second in zip(unaided, supplied):
            self.assertTrue(np.array_equal(first.mask, second.mask))

    def test_a_normal_layer_detects_the_same_with_a_supplied_steepness(self) -> None:
        normal = normal_map()
        domains, evidence = normal_evidence_inputs(normal, crops(), "normal", None)
        steep = edge_steepness_for(evidence, domains)

        unaided = detect_layer_marks("t", normal, crops(), "normal")
        supplied = detect_layer_marks("t", normal, crops(), "normal", steepness=steep)
        self.assertEqual(
            [mark.bounds for mark in unaided], [mark.bounds for mark in supplied]
        )
        for first, second in zip(unaided, supplied):
            self.assertTrue(np.array_equal(first.mask, second.mask))


class SessionCacheTests(unittest.TestCase):
    def test_an_unchanged_uv_domain_reuses_its_detection_across_changed_atlas_pixels(self) -> None:
        """2026-09-05: skin differences outside a chart must not rerun it."""
        session = ProductionDetectionSession()
        first = colour_map()
        changed_elsewhere = first.copy()
        changed_elsewhere[:, 40:] = 180
        changed_inside = changed_elsewhere.copy()
        changed_inside[20:28, 12:20] = 20
        domain = [(0, 0, np.ones((64, 32), dtype=bool))]
        config = replace(
            DEFAULT_GLYPH_LAB_CONFIG,
            contrast=replace(
                DEFAULT_GLYPH_LAB_CONFIG.contrast,
                enable_repeat_texture_filter=False,
            ),
        )
        real = glyph_detect.regions_in_domain
        with patch.object(glyph_detect, "regions_in_domain", wraps=real) as run:
            baseline = detect_layer_marks(
                "first", first, domain, "colour", config=config,
                domain_cache=session,
            )
            reused = detect_layer_marks(
                "second", changed_elsewhere, domain, "colour", config=config,
                domain_cache=session,
            )
            changed = detect_layer_marks(
                "third", changed_inside, domain, "colour", config=config,
                domain_cache=session,
            )
            uncached_changed = detect_layer_marks(
                "third-uncached", changed_inside, domain, "colour", config=config,
            )

        self.assertEqual(run.call_count, 3)
        self.assertEqual(session.glyph_domain_hits, 1)
        self.assertEqual(session.glyph_domain_misses, 2)
        self.assertEqual(
            [mark.bounds for mark in baseline], [mark.bounds for mark in reused]
        )
        self.assertEqual(
            [mark.bounds for mark in changed],
            [mark.bounds for mark in uncached_changed],
        )

    def test_a_changed_repeat_filter_context_does_not_reuse_a_domain(self) -> None:
        """The bounded atlas-wide repeat search is a real detector input."""
        session = ProductionDetectionSession()
        first = np.zeros((512, 512, 3), dtype=np.uint8)
        changed_context = first.copy()
        # The domain ends at x=64; this lies outside it but within the 192 px
        # maximum repeat-search radius and must therefore miss the cache.
        changed_context[32:96, 128:192] = 255
        domain = [(0, 0, np.ones((64, 64), dtype=bool))]
        real = glyph_detect.regions_in_domain
        with patch.object(glyph_detect, "regions_in_domain", wraps=real) as run:
            detect_layer_marks(
                "first", first, domain, "colour", domain_cache=session,
            )
            detect_layer_marks(
                "changed-context", changed_context, domain, "colour",
                domain_cache=session,
            )

        self.assertEqual(run.call_count, 2)
        self.assertEqual(session.glyph_domain_hits, 0)
        self.assertEqual(session.glyph_domain_misses, 2)

    def test_one_derivation_serves_every_layer_of_a_material(self) -> None:
        session = ProductionDetectionSession()
        normal = normal_map()
        domains = np.ones((64, 64), dtype=np.uint16)
        calls = []

        def produce():
            calls.append(1)
            return edge_steepness(normal, domains)

        first, reused_first = session.normal_steepness(normal, domains, produce)
        second, reused_second = session.normal_steepness(normal, domains, produce)
        self.assertEqual(len(calls), 1)
        self.assertFalse(reused_first)
        self.assertTrue(reused_second)
        self.assertIs(first, second)
        self.assertEqual(session.normal_evidence_hits, 1)
        self.assertEqual(session.normal_evidence_misses, 1)

    def test_different_domains_do_not_share_an_entry(self) -> None:
        """The percentile is taken inside the domains, so they are an input."""
        session = ProductionDetectionSession()
        normal = normal_map()
        wide = np.ones((64, 64), dtype=np.uint16)
        narrow = np.zeros((64, 64), dtype=np.uint16)
        narrow[10:20, 10:20] = 1
        calls = []

        def produce_for(domains):
            def produce():
                calls.append(1)
                return edge_steepness(normal, domains)
            return produce

        session.normal_steepness(normal, wide, produce_for(wide))
        session.normal_steepness(normal, narrow, produce_for(narrow))
        self.assertEqual(len(calls), 2)

    def test_a_changed_flat_ground_parameter_misses(self) -> None:
        from dataclasses import replace

        session = ProductionDetectionSession()
        normal = normal_map()
        domains = np.ones((64, 64), dtype=np.uint16)
        config = DEFAULT_GLYPH_LAB_CONFIG
        other = replace(
            config,
            normal=replace(
                config.normal,
                slope_threshold=config.normal.slope_threshold + 5,
            ),
        )
        self.assertNotEqual(
            flat_ground_config_key(config), flat_ground_config_key(other)
        )
        calls = []

        def produce():
            calls.append(1)
            return np.zeros((64, 64), dtype=bool)

        session.normal_flat_ground(
            normal, domains, flat_ground_config_key(config), produce,
        )
        session.normal_flat_ground(
            normal, domains, flat_ground_config_key(other), produce,
        )
        self.assertEqual(len(calls), 2)

    def test_an_unrelated_normal_parameter_still_hits(self) -> None:
        """The key holds only what ``absence_of_slopes`` actually reads."""
        from dataclasses import replace

        config = DEFAULT_GLYPH_LAB_CONFIG
        unrelated = replace(
            config,
            normal=replace(config.normal, frame_max_holes=config.normal.frame_max_holes + 1),
        )
        self.assertEqual(
            flat_ground_config_key(config), flat_ground_config_key(unrelated)
        )

    def test_the_seam_cache_serves_sibling_layers(self) -> None:
        session = ProductionDetectionSession()
        calls = []

        def produce():
            calls.append(1)
            return np.zeros((8, 8), dtype=np.uint16)

        identity = (1234, ("a-material",), 8, 8)
        session.mesh_seam_extents(identity, produce)
        _second, reused = session.mesh_seam_extents(identity, produce)
        self.assertEqual(len(calls), 1)
        self.assertTrue(reused)
        self.assertEqual(session.mesh_seam_hits, 1)

    def test_a_material_with_no_seams_caches_the_absence(self) -> None:
        """None is an answer, and re-walking the DAE to rediscover it is not."""
        session = ProductionDetectionSession()
        calls = []

        def produce():
            calls.append(1)

        identity = (1234, ("a-material",), 8, 8)
        self.assertIsNone(session.mesh_seam_extents(identity, produce)[0])
        self.assertIsNone(session.mesh_seam_extents(identity, produce)[0])
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
