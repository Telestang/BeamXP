from __future__ import annotations

from dataclasses import replace
import types
import unittest
from xml.etree import ElementTree as ET

import cv2
import numpy as np

from beamxp import hand_drive_core as core
from beamxp import transform_helpers as th
from mesh_segmentation_transform.annotate_texture_regions import (
    DEFAULT_UV_ISLAND_SYMMETRY_CONFIG,
    STEP_INDEX,
    MserConfig,
    detect_foreground_boxes,
    detect_local_contrast,
    detect_local_contrast_boxes,
    detect_local_contrast_gpu,
    relief_glyph_measures,
    relief_text_measures,
    run_detection,
)
from mesh_segmentation_transform import mirror_texture_for_rhd as rhd


def build_geometry(uv_values: str) -> ET.Element:
    xml = f"""
    <geometry xmlns="{th.NS['c']}" id="Mesh_077-mesh" name="screen">
      <mesh>
        <source id="Mesh_077-mesh-positions">
          <float_array id="Mesh_077-mesh-positions-array" count="12">
            -0.5 0 0  0.5 0 0  0.5 0 1  -0.5 0 1
          </float_array>
          <technique_common>
            <accessor source="#Mesh_077-mesh-positions-array" count="4" stride="3">
              <param name="X" type="float"/>
              <param name="Y" type="float"/>
              <param name="Z" type="float"/>
            </accessor>
          </technique_common>
        </source>
        <source id="Mesh_077-mesh-map-0">
          <float_array id="Mesh_077-mesh-map-0-array" count="8">{uv_values}</float_array>
          <technique_common>
            <accessor source="#Mesh_077-mesh-map-0-array" count="4" stride="2">
              <param name="S" type="float"/>
              <param name="T" type="float"/>
            </accessor>
          </technique_common>
        </source>
        <vertices id="Mesh_077-mesh-vertices">
          <input semantic="POSITION" source="#Mesh_077-mesh-positions"/>
        </vertices>
        <triangles material="screen-material" count="2">
          <input semantic="VERTEX" source="#Mesh_077-mesh-vertices" offset="0"/>
          <input semantic="TEXCOORD" source="#Mesh_077-mesh-map-0" offset="1" set="0"/>
          <p>0 0 1 1 2 2 0 0 2 2 3 3</p>
        </triangles>
      </mesh>
    </geometry>
    """
    return ET.fromstring(xml)


def uv_pairs(geometry: ET.Element) -> tuple[list[float], list[float]]:
    mesh = geometry.find("c:mesh", th.NS)
    for source in mesh.findall("c:source", th.NS):
        accessor = source.find(".//c:accessor", th.NS)
        params = [p.get("name") for p in accessor.findall("c:param", th.NS)]
        if params[:2] == ["S", "T"]:
            values = [float(v) for v in source.find("c:float_array", th.NS).text.split()]
            return values[0::2], values[1::2]
    raise AssertionError("No TEXCOORD source found")


class TextureFlipTests(unittest.TestCase):
    def test_full_mirrored_texture_region_claims_dedicated_uv_domain(self) -> None:
        mirror = np.ones((349, 922), dtype=bool)
        mirror[0, :60] = False
        rigid = np.zeros_like(mirror)

        region = rhd.full_mirrored_texture_region(mirror, rigid, 0.0)

        self.assertEqual(region, (0, 0, 922, 349))

    def test_full_mirrored_texture_region_rejects_shared_atlas(self) -> None:
        mirror = np.zeros((100, 100), dtype=bool)
        mirror[:, :60] = True
        rigid = np.zeros_like(mirror)
        rigid[:, 60:] = True

        self.assertIsNone(rhd.full_mirrored_texture_region(mirror, rigid, 0.0))

    def test_mirrored_geometry_leaves_uvs_alone_by_default(self) -> None:
        geometry = build_geometry("0.1 0.2  0.9 0.2  0.9 0.8  0.1 0.8")
        out = th.mirrored_geometry(geometry, "new")
        s, t = uv_pairs(out)
        self.assertEqual(s, [0.1, 0.9, 0.9, 0.1])
        self.assertEqual(t, [0.2, 0.2, 0.8, 0.8])

    def test_flip_texture_reflects_s_within_bounds(self) -> None:
        geometry = build_geometry("0.1 0.2  0.9 0.2  0.9 0.8  0.1 0.8")
        out = th.mirrored_geometry(geometry, "new", flip_texture=True)
        s, t = uv_pairs(out)
        for got, expected in zip(s, [0.9, 0.1, 0.1, 0.9]):
            self.assertAlmostEqual(got, expected)
        self.assertEqual(t, [0.2, 0.2, 0.8, 0.8])

    def test_flip_texture_preserves_offset_footprint(self) -> None:
        # UVs that live in a sub-region of an atlas (and outside 0..1, as the
        # stock etk800 screen does) must keep sampling the same region.
        geometry = build_geometry("-0.995 0.0  -0.4276 0.0  -0.4276 1.0  -0.995 1.0")
        out = th.mirrored_geometry(geometry, "new", flip_texture=True)
        s, _t = uv_pairs(out)
        self.assertAlmostEqual(min(s), -0.995)
        self.assertAlmostEqual(max(s), -0.4276)
        for got, expected in zip(s, [-0.4276, -0.995, -0.995, -0.4276]):
            self.assertAlmostEqual(got, expected)

    def test_flip_texture_still_mirrors_positions(self) -> None:
        geometry = build_geometry("0.0 0.0  1.0 0.0  1.0 1.0  0.0 1.0")
        out = th.mirrored_geometry(geometry, "new", flip_texture=True)
        mesh = out.find("c:mesh", th.NS)
        positions = next(
            source
            for source in mesh.findall("c:source", th.NS)
            if "positions" in source.get("id")
        )
        values = [float(v) for v in positions.find("c:float_array", th.NS).text.split()]
        self.assertEqual(values[0::3], [0.5, -0.5, -0.5, 0.5])

    def test_texture_flip_mesh_ids_is_display_screens_in_mirror_modes(self) -> None:
        # Derived from display detection (no manual flag) and gated to modes
        # that reflect geometry. Translate never reflects the geometry.
        context = types.SimpleNamespace(
            _display_texture_flip_scope={
                "screen": frozenset({"screen"}),
                "gauge": frozenset({"gauge_screen"}),
                "panel": frozenset({"panel_screen"}),
            }
        )
        modes = {
            "screen": core.MODE_MIRROR,     # nav + mirror -> flipped
            "dash": core.MODE_MIRROR,       # mirror but not a nav screen
            "gauge": core.MODE_TRANSLATE,   # nav but translated, not reflected
            "panel": core.MODE_SKIP,        # display but nothing reflects it
        }
        self.assertEqual(core.texture_flip_mesh_ids(context, modes), {"screen"})


class IslandMirrorLineTests(unittest.TestCase):
    """Which line pass 2 turns a whole island over about.

    The ETK K-series console controller is the case these describe: a rounded
    pad whose rasterised silhouette carries a small spur on one side.  The spur
    moves the bounding box without moving the pad, so a reflection about the
    box scored it 0.970 -- and turned over about the box it would land every
    texel three columns off the partner it was meant to exchange with.
    """

    def spurred_island(self) -> tuple[np.ndarray, np.ndarray]:
        """A symmetric pad whose box is dragged three texels to the left."""
        mirror = np.zeros((120, 240), dtype=bool)
        mirror[10:110, 20:220] = True
        mirror[59:61, 17:20] = True
        axis_map = np.full(mirror.shape, rhd.AXIS_HORIZONTAL, dtype=np.uint8)
        return mirror, axis_map

    def test_the_exporter_uses_the_harness_s_symmetry_threshold(self) -> None:
        # One question, one number: an island the tuning app draws as
        # symmetric is one the exporter has to be willing to turn over.
        self.assertEqual(
            rhd.DEFAULT_RHD_CONFIG.min_island_symmetry,
            DEFAULT_UV_ISLAND_SYMMETRY_CONFIG.min_uv_island_symmetry,
        )

    def test_the_flip_is_planned_about_the_island_s_own_line(self) -> None:
        mirror, axis_map = self.spurred_island()

        flips, flipped = rhd.plan_island_flips(
            mirror, [(40, 30, 20, 12)], rhd.DEFAULT_RHD_CONFIG, axis_map
        )

        self.assertEqual(len(flips), 1)
        x, _y, w, _h = flips[0].bounds
        # The pad spans 20..219, so its own line is 119.5; the box runs 17..219
        # and centres on 118.  It is the pad's line the rectangle must reflect
        # about, because it is the pad the flip has to land back on.
        self.assertEqual((x + x + w - 1) / 2.0, 119.5)
        self.assertEqual(flips[0].axis_shift, 3)
        self.assertTrue(bool(flipped[30:42, 40:60].all()))

    def test_the_island_scores_better_on_its_own_line_than_on_its_box(self) -> None:
        mirror, axis_map = self.spurred_island()
        component = mirror[10:110, 17:220]

        flips, _flipped = rhd.plan_island_flips(
            mirror, [(40, 30, 20, 12)], rhd.DEFAULT_RHD_CONFIG, axis_map
        )

        self.assertGreater(
            flips[0].horizontal_similarity,
            rhd._reflection_similarity(component, np.fliplr(component)),
        )

    def test_an_island_already_centred_on_its_line_is_left_where_it_is(self) -> None:
        # Nothing to recover, so nothing moves: the search must not trade a
        # correct rectangle for an equal-scoring one a texel away.
        mirror = np.zeros((120, 240), dtype=bool)
        mirror[10:110, 20:220] = True
        axis_map = np.full(mirror.shape, rhd.AXIS_HORIZONTAL, dtype=np.uint8)

        flips, _flipped = rhd.plan_island_flips(
            mirror, [(40, 30, 20, 12)], rhd.DEFAULT_RHD_CONFIG, axis_map
        )

        self.assertEqual(flips[0].bounds, (20, 10, 200, 100))
        self.assertEqual(flips[0].axis_shift, 0)

    def test_a_genuinely_lopsided_island_is_still_refused(self) -> None:
        # No line within reach reflects this onto itself, so none is taken and
        # the glyphs inside fall to pass 3 to be handled one at a time.
        mirror = np.zeros((120, 260), dtype=bool)
        mirror[10:110, 20:220] = True
        mirror[10:60, 220:236] = True
        axis_map = np.full(mirror.shape, rhd.AXIS_HORIZONTAL, dtype=np.uint8)

        flips, _flipped = rhd.plan_island_flips(
            mirror, [(40, 30, 20, 12)], rhd.DEFAULT_RHD_CONFIG, axis_map
        )

        self.assertEqual(flips, [])

    def test_a_line_that_would_leave_the_atlas_is_not_taken(self) -> None:
        # The rectangle grows to move its centre, and a rectangle clipped by
        # the atlas edge is re-centred by the flip, so it must not be planned.
        mirror = np.zeros((120, 203), dtype=bool)
        mirror[10:110, 0:200] = True
        mirror[59:61, 200:203] = True
        axis_map = np.full(mirror.shape, rhd.AXIS_HORIZONTAL, dtype=np.uint8)

        flips, _flipped = rhd.plan_island_flips(
            mirror, [(90, 30, 20, 12)], rhd.DEFAULT_RHD_CONFIG, axis_map
        )

        self.assertEqual(flips, [])


class IslandLabellingTests(unittest.TestCase):
    """What pass 2 counts as one island.

    An atlas packs its charts hard against each other -- the ETK K-series
    console has four separate 155x25 strips stacked with no gutter at all --
    so pixel adjacency answers a question about the image while pass 2 is
    asking one about the mesh.
    """

    def chart(
        self,
        u0: float,
        v0: float,
        u1: float,
        v1: float,
        x: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """One quad's two triangles in UV, placed at world ``x``."""
        uv = np.asarray(
            [[(u0, v0), (u1, v0), (u1, v1)], [(u0, v0), (u1, v1), (u0, v1)]],
            dtype=float,
        )
        xyz = np.asarray(
            [
                [(x, 0.0, 0.0), (x + 1.0, 0.0, 0.0), (x + 1.0, 1.0, 0.0)],
                [(x, 0.0, 0.0), (x + 1.0, 1.0, 0.0), (x, 1.0, 0.0)],
            ],
            dtype=float,
        )
        return uv, xyz

    def stacked_charts(self, shared_world: bool = False) -> rhd.DomainMasks:
        """Two charts touching along a UV edge, on unrelated geometry."""
        lower = self.chart(0.25, 0.25, 0.75, 0.5, 0.0)
        upper = self.chart(0.25, 0.5, 0.75, 0.75, 0.0 if shared_world else 10.0)
        mirror = np.zeros((64, 64), dtype=bool)
        mirror[16:48, 16:48] = True
        return rhd.DomainMasks(
            mirror=mirror,
            rigid=~mirror,
            conflict_coverage=0.0,
            mirrored_triangles=4,
            rigid_triangles=0,
            parts_analysed=1,
            mirrored_uv=(lower[0][0], lower[0][1], upper[0][0], upper[0][1]),
            mirrored_xyz=(lower[1][0], lower[1][1], upper[1][0], upper[1][1]),
        )

    def test_charts_touching_along_an_edge_stay_apart(self) -> None:
        masks = self.stacked_charts()

        count, labels, _stats = rhd.domain_island_labels(masks)

        fused = cv2.connectedComponentsWithStats(
            masks.mirror.astype(np.uint8), connectivity=8
        )[0]
        self.assertEqual(fused - 1, 1)  # pixel adjacency sees one blob
        self.assertEqual(count - 1, 2)  # the mesh has two charts
        self.assertNotEqual(labels[24, 32], labels[40, 32])

    def test_two_meshes_on_one_atlas_region_are_one_island(self) -> None:
        # Not two competing labels: they are the same texels, and whichever
        # was painted second would otherwise be left a partial island.
        masks = self.stacked_charts()
        masks.mirrored_uv = masks.mirrored_uv + masks.mirrored_uv[:2]
        masks.mirrored_xyz = masks.mirrored_xyz + masks.mirrored_xyz[:2]

        count, _labels, _stats = rhd.domain_island_labels(masks)

        self.assertEqual(count - 1, 2)

    def test_every_labelled_texel_belongs_to_exactly_one_island(self) -> None:
        masks = self.stacked_charts()

        _count, labels, stats = rhd.domain_island_labels(masks)

        for label in range(1, int(labels.max()) + 1):
            claimed = labels == label
            self.assertEqual(int(claimed.sum()), int(stats[label, 4]))
            self.assertEqual(int(np.nonzero(claimed.any(axis=0))[0].min()),
                             int(stats[label, 0]))

    def test_the_labels_are_cached_on_the_masks(self) -> None:
        masks = self.stacked_charts()

        first = rhd.domain_island_labels(masks)[1]

        self.assertIs(rhd.domain_island_labels(masks)[1], first)


class ForegroundMaskDetectorTests(unittest.TestCase):
    def test_detects_contrasting_ui_component_only_inside_domain(self) -> None:
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        # A live-display-like bright glyph and an unrelated atlas mark.
        image[18:31, 18:42] = (255, 210, 35)
        image[42:55, 42:60] = (255, 255, 255)
        domain = np.zeros((64, 64), dtype=bool)
        domain[10:38, 10:50] = True
        boxes = detect_foreground_boxes(
            image,
            domain,
            MserConfig(
                foreground_edge_threshold=9999,
                foreground_open_px=0,
                foreground_close_px=0,
                foreground_min_component_px=4,
            ),
        )
        self.assertEqual(len(boxes), 1)
        x, y, w, h = (int(value) for value in boxes[0])
        self.assertLessEqual(x, 18)
        self.assertLessEqual(y, 18)
        self.assertGreaterEqual(x + w, 42)
        self.assertGreaterEqual(y + h, 31)

    def test_refines_solid_component_to_internal_detail_regardless_of_polarity(self) -> None:
        """A switch fill locates its own contrasting glyph, not its whole face."""
        config = MserConfig(
            foreground_edge_threshold=9999,
            foreground_open_px=0,
            foreground_close_px=0,
            foreground_min_component_px=4,
            foreground_detail_min_component_px=4,
            foreground_detail_min_parent_px=40,
            foreground_detail_inset_px=1,
            foreground_merge_gap_px=1,
            foreground_refine_internal_details=True,
        )
        for panel, glyph in ((80, 255), (230, 0)):
            image = np.zeros((80, 100, 3), dtype=np.uint8)
            image[20:60, 20:80] = panel
            # Deliberately use a non-white second case: this proves the
            # refinement is a local contrast hierarchy, not bright-ink logic.
            image[33:47, 43:57] = glyph
            boxes = detect_foreground_boxes(image, np.ones((80, 100), dtype=bool), config)
            self.assertTrue(len(boxes), boxes)
            self.assertTrue(
                any(
                    x <= 43 and y <= 33 and x + w >= 57 and y + h >= 47
                    and w < 40 and h < 30
                    for x, y, w, h in boxes
                ),
                boxes,
            )
            self.assertFalse(
                any(x <= 20 and y <= 20 and x + w >= 80 and y + h >= 60 for x, y, w, h in boxes),
                boxes,
            )


class ReliefGlyphStructureTests(unittest.TestCase):
    def test_aligned_fragments_pass_but_a_single_seam_does_not(self) -> None:
        config = MserConfig(relief_glyph_min_component_px=4)
        line = np.zeros((32, 80), dtype=bool)
        for x in (8, 30, 52):
            line[12:19, x:x + 7] = True
        coverage, components, dominant, scatter, outline_scatter = relief_glyph_measures(
            line, (0, 0, 80, 32), config,
        )
        self.assertLess(coverage, config.relief_glyph_min_compact_edge_coverage)
        self.assertEqual(components, 3)
        self.assertLessEqual(scatter, config.relief_glyph_max_line_scatter)
        self.assertLessEqual(dominant, config.relief_glyph_max_dominant_component_fraction)
        self.assertGreaterEqual(outline_scatter, 0.0)

        seam = np.zeros((32, 80), dtype=bool)
        seam[14:18, 4:76] = True
        _coverage, components, dominant, scatter, outline_scatter = relief_glyph_measures(
            seam, (0, 0, 80, 32), config,
        )
        self.assertEqual(components, 1)
        self.assertGreater(dominant, config.relief_glyph_max_dominant_component_fraction)
        self.assertEqual(scatter, float("inf"))
        self.assertLess(outline_scatter, config.relief_glyph_min_outline_scatter)

    def test_sparse_two_dimensional_outline_is_not_treated_as_a_seam(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            DetectionState,
            _step_relief_glyph_structure,
            _step_relief_text,
        )

        mask = np.zeros((40, 80), dtype=bool)
        for x, y in ((5, 5), (61, 5), (5, 23), (61, 23)):
            mask[y:y + 12, x:x + 12] = True
        config = MserConfig(
            enable_relief_glyph_filter=True,
            relief_glyph_min_component_px=4,
        )
        image = np.zeros((40, 80, 3), dtype=np.uint8)
        group = (0, 0, 80, 40)

        state, stage = _step_relief_glyph_structure(
            image,
            np.ones(mask.shape, dtype=bool),
            config,
            DetectionState(
                np.empty((0, 4), dtype=np.int32),
                [group],
                relief_bridge_response=mask.astype(np.float32),
                relief_edge_mask=mask,
            ),
        )

        self.assertEqual(state.groups, [group])
        self.assertEqual(stage.rejected, ())
        self.assertIn("1 outlined", stage.detail)

        text_state, text_stage = _step_relief_text(
            image,
            np.ones(mask.shape, dtype=bool),
            replace(config, enable_relief_text_filter=True),
            state,
        )
        self.assertEqual(text_state.groups, [group])
        self.assertEqual(text_stage.rejected, ())
        self.assertIn("1 outlined logo retained", text_stage.detail)


class ReliefTextTests(unittest.TestCase):
    def test_projection_recognises_connected_or_fragmented_letters_not_a_seam(self) -> None:
        config = MserConfig(
            relief_text_response_percentile=80.0,
            relief_text_projection_min_coverage=0.10,
            relief_text_min_runs=4,
        )
        # Five vertical response bands share their cross-axis support.  They
        # mimic letters which are connected by a shallow embossed baseline:
        # connected components would see one mark, projection still sees five.
        letters = np.ones((20, 80), dtype=np.float32)
        for x in (4, 19, 34, 49, 64):
            letters[:, x:x + 7] = 20.0
        runs, substantial, horizontal, _coverage = relief_text_measures(
            letters, np.ones(letters.shape, dtype=bool), (0, 0, 80, 20), config,
        )
        self.assertTrue(horizontal)
        self.assertEqual(runs, 5)
        self.assertEqual(substantial, 5)

        seam = np.ones((20, 80), dtype=np.float32)
        seam[:, 5:75] = 20.0
        runs, substantial, horizontal, _coverage = relief_text_measures(
            seam, np.ones(seam.shape, dtype=bool), (0, 0, 80, 20), config,
        )
        self.assertTrue(horizontal)
        self.assertEqual(runs, 1)
        self.assertEqual(substantial, 1)

    def test_filter_keeps_rotated_outline_for_accepted_relief_text(self) -> None:
        from mesh_segmentation_transform.annotate_texture_regions import (
            DetectionState,
            _step_relief_text,
        )

        response = np.ones((24, 96), dtype=np.float32)
        for x in (5, 22, 39, 56, 73):
            response[:, x:x + 8] = 20.0
        outline = ((4.0, 3.0), (91.0, 5.0), (90.0, 20.0), (3.0, 18.0))
        state = DetectionState(
            np.empty((0, 4), dtype=np.int32),
            [(0, 0, 96, 24)],
            rotations=[outline],
            relief_bridge_response=response,
        )
        config = MserConfig(
            enable_relief_text_filter=True,
            relief_text_response_percentile=80.0,
            relief_text_projection_min_coverage=0.10,
            relief_text_min_runs=4,
        )

        result, stage = _step_relief_text(
            np.zeros((24, 96, 3), dtype=np.uint8),
            np.ones((24, 96), dtype=bool), config, state,
        )

        self.assertEqual(result.groups, [(0, 0, 96, 24)])
        self.assertEqual(result.rotations, [outline])
        self.assertEqual(stage.rotations, (outline,))


class LocalContrastDetectorTests(unittest.TestCase):
    def test_detects_internal_detail_at_both_contrast_polarities(self) -> None:
        """The local response is not a white-ink or dark-ink special case."""
        config = MserConfig(
            box_source="contrast",
            contrast_kernel_px=9,
            contrast_min_response=18,
            contrast_percentile=80,
            contrast_close_px=0,
            contrast_merge_gap_px=1,
            contrast_min_component_px=4,
        )
        for panel, glyph in ((80, 255), (230, 0)):
            image = np.zeros((80, 100, 3), dtype=np.uint8)
            image[20:60, 20:80] = panel
            image[33:47, 43:57] = glyph
            boxes = detect_local_contrast_boxes(
                image, np.ones((80, 100), dtype=bool), config,
            )
            self.assertTrue(
                any(
                    x <= 43 and y <= 33 and x + w >= 57 and y + h >= 47
                    for x, y, w, h in boxes
                ), boxes,
            )

    def test_percentile_is_scoped_to_the_uv_island(self) -> None:
        image = np.zeros((80, 100, 3), dtype=np.uint8)
        image[22:58, 22:78] = 100
        image[35:45, 45:55] = 180
        # Intense unrelated atlas artwork must not raise this island's cut.
        image[0:12, 0:12] = 255
        domain = np.zeros((80, 100), dtype=bool)
        domain[20:60, 20:80] = True
        boxes = detect_local_contrast_boxes(
            image, domain,
            MserConfig(
                box_source="contrast", contrast_kernel_px=9,
                contrast_min_response=15, contrast_percentile=70,
                contrast_close_px=0, contrast_merge_gap_px=1,
                contrast_min_component_px=4,
            ),
        )
        self.assertTrue(
            any(x <= 45 and y <= 35 and x + w >= 55 and y + h >= 45 for x, y, w, h in boxes),
            boxes,
        )

    def test_cached_high_contrast_bridge_refuses_proximity_grouping(self) -> None:
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        boxes = np.asarray(((2, 6, 4, 4), (16, 6, 4, 4)), dtype=np.int32)
        config = MserConfig(
            box_source="contrast", merge_distance_px=20,
            enable_box_feature_filter=False,
            enable_contrast_continuity_grouping=True,
            contrast_bridge_max_high_coverage=0.0,
        )
        barrier = np.zeros((20, 30), dtype=np.float32)
        barrier[6:10, 6:16] = 10.0
        blocked = run_detection(
            image, np.ones((20, 30), dtype=bool), config,
            initial_boxes=boxes, initial_contrast_response=barrier,
            initial_contrast_threshold=5.0,
        )
        self.assertEqual(len(blocked.stages[STEP_INDEX["grouped"]].kept), 2)

        clear = run_detection(
            image, np.ones((20, 30), dtype=bool), config,
            initial_boxes=boxes,
            initial_contrast_response=np.zeros_like(barrier),
            initial_contrast_threshold=5.0,
        )
        self.assertEqual(len(clear.stages[STEP_INDEX["grouped"]].kept), 1)

    def test_bridge_uses_the_islands_selected_contrast_cut(self) -> None:
        """Mild backing variation must not split one multi-line label.

        The cached field is deliberately reused rather than recomputed during
        grouping.  Its absolute floor is useful for real dividers, but a
        response below this island's candidate threshold is ordinary backing,
        not proof of a boundary between adjacent text fragments.
        """
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        boxes = np.asarray(((2, 6, 4, 4), (16, 6, 4, 4)), dtype=np.int32)
        config = MserConfig(
            box_source="contrast", merge_distance_px=20,
            enable_box_feature_filter=False,
            enable_contrast_continuity_grouping=True,
            contrast_bridge_min_response=8.0,
            contrast_bridge_max_high_coverage=0.0,
        )
        backing = np.zeros((20, 30), dtype=np.float32)
        backing[6:10, 6:16] = 10.0
        grouped = run_detection(
            image, np.ones((20, 30), dtype=bool), config,
            initial_boxes=boxes, initial_contrast_response=backing,
            initial_contrast_threshold=20.0,
        )
        self.assertEqual(len(grouped.stages[STEP_INDEX["grouped"]].kept), 1)

        divider = backing.copy()
        divider[6:10, 6:16] = 25.0
        split = run_detection(
            image, np.ones((20, 30), dtype=bool), config,
            initial_boxes=boxes, initial_contrast_response=divider,
            initial_contrast_threshold=20.0,
        )
        self.assertEqual(len(split.stages[STEP_INDEX["grouped"]].kept), 2)

    def test_disabling_box_filter_keeps_boxes_below_its_minimum_size(self) -> None:
        image = np.zeros((12, 12, 3), dtype=np.uint8)
        tiny = np.asarray(((4, 4, 1, 1),), dtype=np.int32)
        config = MserConfig(
            min_box_width_px=5,
            min_box_height_px=5,
            enable_box_feature_filter=False,
        )

        run = run_detection(
            image, np.ones((12, 12), dtype=bool), config, initial_boxes=tiny,
        )

        self.assertEqual(run.stages[STEP_INDEX["box_filter"]].kept, ((4, 4, 1, 1),))
        self.assertEqual(run.stages[STEP_INDEX["box_filter"]].rejected, ())

    def test_local_contrast_collapses_nested_boxes_before_proximity_grouping(self) -> None:
        image = np.zeros((24, 24, 3), dtype=np.uint8)
        nested = np.asarray(((3, 3, 16, 16), (7, 7, 4, 4)), dtype=np.int32)
        config = MserConfig(
            box_source="contrast",
            enable_box_feature_filter=False,
            enable_contrast_continuity_grouping=False,
        )

        run = run_detection(
            image, np.ones((24, 24), dtype=bool), config, initial_boxes=nested,
        )

        self.assertEqual(len(run.stages[STEP_INDEX["overlap_box_group"]].kept), 1)  # overlap grouping
        self.assertEqual(len(run.stages[STEP_INDEX["grouped"]].kept), 1)  # initial grouping
        self.assertEqual(len(run.stages[STEP_INDEX["overlap_group"]].kept), 1)  # Domain recovery is terminal

    def test_local_contrast_collapses_partial_overlap_before_proximity_grouping(self) -> None:
        image = np.zeros((24, 24, 3), dtype=np.uint8)
        overlapping = np.asarray(((2, 3, 10, 10), (8, 6, 10, 10)), dtype=np.int32)
        config = MserConfig(
            box_source="contrast",
            enable_box_feature_filter=False,
            enable_contrast_continuity_grouping=False,
        )

        run = run_detection(
            image, np.ones((24, 24), dtype=bool), config,
            initial_boxes=overlapping,
        )

        self.assertEqual(len(run.stages[STEP_INDEX["overlap_box_group"]].kept), 1)
        # Initial grouping receives the one overlap-connected candidate and
        # does not need to reinterpret its geometry as proximity.
        self.assertEqual(len(run.stages[STEP_INDEX["grouped"]].kept), 1)
        self.assertEqual(len(run.stages[STEP_INDEX["overlap_group"]].kept), 1)

    def test_cached_relief_edge_refuses_colour_box_proximity_grouping(self) -> None:
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        boxes = np.asarray(((2, 6, 4, 4), (16, 6, 4, 4)), dtype=np.int32)
        config = MserConfig(
            box_source="contrast", merge_distance_px=20,
            enable_box_feature_filter=False,
            enable_contrast_continuity_grouping=False,
            enable_relief_edge_bridge_grouping=True,
            relief_bridge_min_response=100.0,
            relief_bridge_min_cross_axis_coverage=0.7,
        )
        barrier = np.zeros((20, 30), dtype=np.float32)
        barrier[6:10, 6:16] = 200.0
        blocked = run_detection(
            image, np.ones((20, 30), dtype=bool), config,
            initial_boxes=boxes, initial_relief_bridge_response=barrier,
        )
        self.assertEqual(len(blocked.stages[STEP_INDEX["grouped"]].kept), 2)

        clear = run_detection(
            image, np.ones((20, 30), dtype=bool), config,
            initial_boxes=boxes,
            initial_relief_bridge_response=np.zeros_like(barrier),
        )
        self.assertEqual(len(clear.stages[STEP_INDEX["grouped"]].kept), 1)

    def test_relief_bridge_ignores_adjacent_glyph_perimeters(self) -> None:
        """Top/bottom glyph outlines inside a narrow gap are not a divider."""
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        boxes = np.asarray(((2, 6, 4, 4), (16, 6, 4, 4)), dtype=np.int32)
        config = MserConfig(
            box_source="contrast", merge_distance_px=20,
            enable_box_feature_filter=False,
            enable_contrast_continuity_grouping=False,
            enable_relief_edge_bridge_grouping=True,
            relief_bridge_min_response=100.0,
            relief_bridge_min_cross_axis_coverage=0.7,
        )
        perimeters = np.zeros((20, 30), dtype=np.float32)
        perimeters[6, 6:16] = 200.0
        perimeters[9, 6:16] = 200.0
        grouped = run_detection(
            image, np.ones((20, 30), dtype=bool), config,
            initial_boxes=boxes, initial_relief_bridge_response=perimeters,
        )
        self.assertEqual(len(grouped.stages[STEP_INDEX["grouped"]].kept), 1)

    def test_gpu_response_matches_cpu_when_compute_is_available(self) -> None:
        image = np.zeros((64, 80, 3), dtype=np.uint8)
        image[12:54, 10:70] = 55
        image[25:39, 32:48] = 230
        domain = np.zeros((64, 80), dtype=bool)
        domain[10:56, 8:72] = True
        config = MserConfig(
            contrast_kernel_px=7, contrast_percentile=80,
            contrast_close_px=0, contrast_merge_gap_px=1,
        )
        try:
            from mesh_segmentation_transform.texture_local_contrast_gpu import (
                LocalContrastGpuUnavailable,
            )
            gpu = detect_local_contrast_gpu(image, domain, config)
        except LocalContrastGpuUnavailable:
            self.skipTest("OpenGL 4.3 compute unavailable")
        cpu = detect_local_contrast(image, domain, config)
        np.testing.assert_allclose(
            gpu.response[domain], cpu.response[domain], rtol=1e-5, atol=2e-4,
        )
        self.assertEqual(gpu.boxes.tolist(), cpu.boxes.tolist())


class DisplayScreenDetectionTests(unittest.TestCase):
    """The beamNavigator screen material resolves to what the DAE binds, both
    directly (etk800) and through the same part's glowMap (sunburst2)."""

    def _context(self, part_bodies: dict[str, str], symbols: dict[str, tuple[str, ...]]):
        context = types.SimpleNamespace(
            part_body_index={name: (body, f"{name}.jbeam") for name, body in part_bodies.items()},
            preview_by_id={
                object_id: {"materials": mats} for object_id, mats in symbols.items()
            },
        )
        return context

    def test_direct_screen_material_match(self) -> None:
        # etk800: screenMaterialName IS the bound material.
        body = """
        "etk800_dash": { "slotType": "etk800_dash",
          "controller": [ ["fileName"],
            ["beamNavigator", {"screenMaterialName": "@etk800_screen", "name": "etk800_navi"}] ]
        }
        """
        context = self._context({"etk800_dash": body}, {"etk800_screen": ("etk800_screen",)})
        self.assertEqual(core.nav_screen_materials_for_context(context), frozenset({"etk800_screen"}))
        self.assertEqual(
            core.nav_screen_mesh_scope(context),
            {"etk800_screen": frozenset({"etk800_screen"})},
        )

    def test_glowmap_resolves_runtime_base_material(self) -> None:
        # sunburst2: the mesh binds sunburst2_display_nav; the glowMap swaps it
        # to the screenMaterialName when lit, so the base must be resolved back.
        body = """
        "sunburst2_nav": { "slotType": "sunburst2_radio",
          "controller": [ ["fileName"],
            ["beamNavigator", {"screenMaterialName": "@sunburst2_naviscreen_on", "name": "sunburst2_navi"}] ],
          "glowMap": {
            "sunburst2_display_nav": {"simpleFunction": {"ignitionLevel": 0.5},
               "off": "screen_off", "on": "sunburst2_naviscreen_accessory",
               "on_intense": "sunburst2_naviscreen_on"}
          }
        }
        """
        context = self._context(
            {"sunburst2_nav": body},
            {"sunburst2_nav": ("sunburst2_gauges", "sunburst2_display_nav")},
        )
        materials = core.nav_screen_materials_for_context(context)
        self.assertIn("sunburst2_display_nav", materials)
        # Only the screen island is scoped; the gauge cluster material is not.
        self.assertEqual(
            core.nav_screen_mesh_scope(context),
            {"sunburst2_nav": frozenset({"sunburst2_display_nav"})},
        )

    def test_lexus_runtime_gps_flips_the_bound_switch_base_not_a_bitmap(self) -> None:
        # The live navigator is @lc500_GPS, which has no archive image to
        # rewrite.  The DAE binds lc500_centralscreen, so the glowMap base is
        # the material whose UV island must be reflected when the mesh mirrors.
        body = """
        "lc500_dash": { "slotType": "lc500_dash",
          "controller": [ ["fileName"],
            ["beamNavigator", {"screenMaterialName": "@lc500_GPS", "name": "lc500_navi"}] ],
          "glowMap": {
            "lc500_centralscreen": {"simpleFunction": {"ignitionLevel": 0.1},
              "off": "lc500_screens_off", "on": "lc500_centralscreen_on",
              "on_intense": "@lc500_GPS"},
            "lc500_screenoverlay": {"off": "lc500_screens_off", "on": "lc500_screenoverlay_on"}
          }
        }
        """
        context = self._context(
            {"lc500_dash": body},
            {"lc500_screen_mesh": ("lc500_centralscreen", "lc500_screenoverlay")},
        )
        context._material_flags = {}
        self.assertEqual(
            core.display_texture_flip_scope(context),
            {"lc500_screen_mesh": frozenset({"lc500_centralscreen"})},
        )
        self.assertEqual(
            core.texture_flip_mesh_ids(
                context, {"lc500_screen_mesh": core.MODE_MIRROR}
            ),
            {"lc500_screen_mesh"},
        )

    def test_grouped_nav_and_gauge_screen_materials_share_flip_scope(self) -> None:
        body = """
        "ardente_dash": { "slotType": "ardente_dash",
          "controller": [ ["fileName"],
            ["beamNavigator", {"screenMaterialName": "@ardente_gps_screen", "name": "ardente_navi"}] ],
          "glowMap": {
            "ardente_gauges_screen": {"simpleFunction": {"ignitionLevel": 0.5},
              "off": "screen_off", "on": "ardente_gauges_screen_accessory",
              "on_intense": "ardente_gauges_screen_accessory"},
            "ardente_gps_screen": {"simpleFunction": {"ignitionLevel": 0.5},
              "off": "screen_off", "on": "ardente_naviscreen_accessory",
              "on_intense": "ardente_naviscreen_accessory"}
          }
        }
        """
        context = self._context(
            {"ardente_dash": body},
            {"ardente_screens": ("ardente_gauges_screen", "ardente_gps_screen")},
        )
        context._material_flags = {
            "ardente_gauges_screen": {"emissive": True, "glass": False},
            "ardente_gps_screen": {"emissive": True, "glass": False},
        }
        self.assertEqual(
            core.display_texture_flip_scope(context),
            {"ardente_screens": frozenset({"ardente_gauges_screen", "ardente_gps_screen"})},
        )

    def test_no_navigator_yields_empty(self) -> None:
        body = '"plain_dash": { "slotType": "plain_dash", "flexbodies": [["mesh"]] }'
        context = self._context({"plain_dash": body}, {"plain_dash": ("dash_mat",)})
        context._material_flags = {}
        self.assertEqual(core.nav_screen_materials_for_context(context), frozenset())
        self.assertEqual(core.nav_screen_mesh_scope(context), {})
        self.assertEqual(core.display_texture_flip_scope(context), {})


class ScopedTextureFlipTests(unittest.TestCase):
    """A nav screen sharing a mesh (and one texcoord source) with a cluster,
    like sunburst2_nav: only the screen's UV island may be reflected."""

    def build_shared_geometry(self) -> ET.Element:
        # Two islands in one map source: cluster in U[0,1), nav screen in U[1,2).
        xml = f"""
        <geometry xmlns="{th.NS['c']}" id="nav-mesh" name="nav">
          <mesh>
            <source id="nav-mesh-positions">
              <float_array id="nav-mesh-positions-array" count="24">
                -0.5 0 0  0.5 0 0  0.5 0 1  -0.5 0 1
                -0.5 0 0  0.5 0 0  0.5 0 1  -0.5 0 1
              </float_array>
              <technique_common>
                <accessor source="#nav-mesh-positions-array" count="8" stride="3">
                  <param name="X" type="float"/>
                  <param name="Y" type="float"/>
                  <param name="Z" type="float"/>
                </accessor>
              </technique_common>
            </source>
            <source id="nav-mesh-map-0">
              <float_array id="nav-mesh-map-0-array" count="16">
                0.1 0.2  0.9 0.2  0.9 0.8  0.1 0.8
                1.1 0.2  1.9 0.2  1.9 0.8  1.1 0.8
              </float_array>
              <technique_common>
                <accessor source="#nav-mesh-map-0-array" count="8" stride="2">
                  <param name="S" type="float"/>
                  <param name="T" type="float"/>
                </accessor>
              </technique_common>
            </source>
            <vertices id="nav-mesh-vertices">
              <input semantic="POSITION" source="#nav-mesh-positions"/>
            </vertices>
            <triangles material="cluster-material" count="2">
              <input semantic="VERTEX" source="#nav-mesh-vertices" offset="0"/>
              <input semantic="TEXCOORD" source="#nav-mesh-map-0" offset="1" set="0"/>
              <p>0 0 1 1 2 2  0 0 2 2 3 3</p>
            </triangles>
            <triangles material="display_nav-material" count="2">
              <input semantic="VERTEX" source="#nav-mesh-vertices" offset="0"/>
              <input semantic="TEXCOORD" source="#nav-mesh-map-0" offset="1" set="0"/>
              <p>4 4 5 5 6 6  4 4 6 6 7 7</p>
            </triangles>
          </mesh>
        </geometry>
        """
        return ET.fromstring(xml)

    def build_multi_target_geometry(self) -> ET.Element:
        # Three islands share one TEXCOORD source.  The first target is split
        # across two material primitives that reuse its corner entries, which
        # proves shared consumers are reflected once rather than once per row.
        xml = f"""
        <geometry xmlns="{th.NS['c']}" id="atlas-mesh" name="atlas">
          <mesh>
            <source id="atlas-mesh-positions">
              <float_array id="atlas-mesh-positions-array" count="36">
                0 0 0  1 0 0  1 0 1  0 0 1
                0 0 0  1 0 0  1 0 1  0 0 1
                0 0 0  1 0 0  1 0 1  0 0 1
              </float_array>
              <technique_common>
                <accessor source="#atlas-mesh-positions-array" count="12" stride="3">
                  <param name="X" type="float"/>
                  <param name="Y" type="float"/>
                  <param name="Z" type="float"/>
                </accessor>
              </technique_common>
            </source>
            <source id="atlas-mesh-map-0">
              <float_array id="atlas-mesh-map-0-array" count="24">
                0.1 0.2  0.9 0.2  0.9 0.8  0.1 0.8
                1.1 0.2  1.9 0.2  1.9 0.8  1.1 0.8
                2.1 0.2  2.9 0.2  2.9 0.8  2.1 0.8
              </float_array>
              <technique_common>
                <accessor source="#atlas-mesh-map-0-array" count="12" stride="2">
                  <param name="S" type="float"/>
                  <param name="T" type="float"/>
                </accessor>
              </technique_common>
            </source>
            <vertices id="atlas-mesh-vertices">
              <input semantic="POSITION" source="#atlas-mesh-positions"/>
            </vertices>
            <triangles material="display_a-material" count="1">
              <input semantic="VERTEX" source="#atlas-mesh-vertices" offset="0"/>
              <input semantic="TEXCOORD" source="#atlas-mesh-map-0" offset="1" set="0"/>
              <p>0 0 1 1 2 2</p>
            </triangles>
            <triangles material="display_a_detail-material" count="1">
              <input semantic="VERTEX" source="#atlas-mesh-vertices" offset="0"/>
              <input semantic="TEXCOORD" source="#atlas-mesh-map-0" offset="1" set="0"/>
              <p>0 0 2 2 3 3</p>
            </triangles>
            <triangles material="display_b-material" count="2">
              <input semantic="VERTEX" source="#atlas-mesh-vertices" offset="0"/>
              <input semantic="TEXCOORD" source="#atlas-mesh-map-0" offset="1" set="0"/>
              <p>4 4 5 5 6 6  4 4 6 6 7 7</p>
            </triangles>
            <triangles material="protected_trim-material" count="2">
              <input semantic="VERTEX" source="#atlas-mesh-vertices" offset="0"/>
              <input semantic="TEXCOORD" source="#atlas-mesh-map-0" offset="1" set="0"/>
              <p>8 8 9 9 10 10  8 8 10 10 11 11</p>
            </triangles>
          </mesh>
        </geometry>
        """
        return ET.fromstring(xml)

    def test_scoped_flip_reflects_only_nav_island(self) -> None:
        geometry = self.build_shared_geometry()
        out = th.mirrored_geometry(
            geometry, "new", flip_texture=True, flip_materials={"display_nav"}
        )
        s, t = uv_pairs(out)
        # Cluster island (indices 0-3) is untouched.
        self.assertEqual(s[0:4], [0.1, 0.9, 0.9, 0.1])
        # Nav island (indices 4-7) reflects within its own tile: u' = 3.0 - u.
        for got, expected in zip(s[4:8], [1.9, 1.1, 1.1, 1.9]):
            self.assertAlmostEqual(got, expected)
        # T (vertical) is never touched.
        self.assertEqual(t, [0.2, 0.2, 0.8, 0.8, 0.2, 0.2, 0.8, 0.8])

    def test_scoped_flip_keeps_nav_island_in_its_own_tile(self) -> None:
        geometry = self.build_shared_geometry()
        out = th.mirrored_geometry(
            geometry, "new", flip_texture=True, flip_materials={"display_nav"}
        )
        s, _t = uv_pairs(out)
        # The reflection stays within U[1,2); it must not land on the cluster.
        self.assertGreaterEqual(min(s[4:8]), 1.0)
        self.assertLessEqual(max(s[4:8]), 2.0)

    def test_each_targeted_uv_component_uses_its_own_pivot(self) -> None:
        geometry = self.build_multi_target_geometry()
        out = th.mirrored_geometry(
            geometry,
            "new",
            flip_texture=True,
            flip_materials={"display_a", "display_a_detail", "display_b"},
        )
        s, t = uv_pairs(out)

        # The two material consumers sharing tile A are one component, so its
        # entries are reflected exactly once around 0.5.
        self.assertEqual(s[0:4], [0.9, 0.1, 0.1, 0.9])
        # Disconnected tile B gets its own 1.5 pivot; it cannot be translated
        # into tile A by a combined source-wide min/max.
        self.assertEqual(s[4:8], [1.9, 1.1, 1.1, 1.9])
        # A third, non-targeted material sharing the source remains authored.
        self.assertEqual(s[8:12], [2.1, 2.9, 2.9, 2.1])
        self.assertEqual(t, [0.2, 0.2, 0.8, 0.8] * 3)

    def build_per_corner_screen_geometry(self) -> ET.Element:
        # The ETK nav screen: two quads side by side in one contiguous UV
        # rectangle, with a private texcoord entry per corner.  No two
        # triangles share an index, so index identity alone sees four islands.
        xml = f"""
        <geometry xmlns="{th.NS['c']}" id="screen-mesh" name="screen">
          <mesh>
            <source id="screen-mesh-positions">
              <float_array id="screen-mesh-positions-array" count="12">
                0 0 0  1 0 0  1 0 1  0 0 1
              </float_array>
              <technique_common>
                <accessor source="#screen-mesh-positions-array" count="4" stride="3">
                  <param name="X" type="float"/>
                  <param name="Y" type="float"/>
                  <param name="Z" type="float"/>
                </accessor>
              </technique_common>
            </source>
            <source id="screen-mesh-map-0">
              <float_array id="screen-mesh-map-0-array" count="24">
                0.0 0.0  0.6 0.0  0.6 1.0
                0.0 0.0  0.6 1.0  0.0 1.0
                0.6 0.0  1.0 0.0  1.0 1.0
                0.6 0.0  1.0 1.0  0.6 1.0
              </float_array>
              <technique_common>
                <accessor source="#screen-mesh-map-0-array" count="12" stride="2">
                  <param name="S" type="float"/>
                  <param name="T" type="float"/>
                </accessor>
              </technique_common>
            </source>
            <vertices id="screen-mesh-vertices">
              <input semantic="POSITION" source="#screen-mesh-positions"/>
            </vertices>
            <triangles material="display_screen-material" count="4">
              <input semantic="VERTEX" source="#screen-mesh-vertices" offset="0"/>
              <input semantic="TEXCOORD" source="#screen-mesh-map-0" offset="1" set="0"/>
              <p>0 0 1 1 2 2  0 3 2 4 3 5  1 6 0 7 3 8  1 9 3 10 2 11</p>
            </triangles>
          </mesh>
        </geometry>
        """
        return ET.fromstring(xml)

    def test_per_corner_texcoords_flip_as_one_screen(self) -> None:
        geometry = self.build_per_corner_screen_geometry()
        out = th.mirrored_geometry(
            geometry, "new", flip_texture=True, flip_materials={"display_screen"}
        )
        s, t = uv_pairs(out)

        # One reflection about the screen's own 0..1 span, not one per quad:
        # the narrow right-hand panel has to end up on the left.
        for got, expected in zip(
            s,
            [1.0, 0.4, 0.4, 1.0, 0.4, 1.0, 0.4, 0.0, 0.0, 0.4, 0.0, 0.4],
        ):
            self.assertAlmostEqual(got, expected)
        self.assertEqual(
            t,
            [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
        )

    def test_target_component_touching_protected_entry_is_not_torn(self) -> None:
        geometry = self.build_multi_target_geometry()
        mesh = geometry.find("c:mesh", th.NS)
        protected = next(
            primitive
            for primitive in mesh.findall("c:triangles", th.NS)
            if primitive.get("material") == "protected_trim-material"
        )
        # The protected consumer reuses entry 2 at its material boundary.
        protected.find("c:p", th.NS).text = (
            "8 2 9 9 10 10  8 2 10 10 11 11"
        )

        out = th.mirrored_geometry(
            geometry,
            "new",
            flip_texture=True,
            flip_materials={"display_a", "display_a_detail", "display_b"},
        )
        s, _t = uv_pairs(out)

        # All of tile A stays fixed, including target-only entries 0, 1 and 3;
        # flipping just those while preserving shared entry 2 would distort it.
        self.assertEqual(s[0:4], [0.1, 0.9, 0.9, 0.1])
        # The independent targeted component is still safe to reflect.
        self.assertEqual(s[4:8], [1.9, 1.1, 1.1, 1.9])
        self.assertEqual(s[8:12], [2.1, 2.9, 2.9, 2.1])


if __name__ == "__main__":
    unittest.main()


class OutlinePartnerShareTests(unittest.TestCase):
    """A fitted outline is only a frame to mirror in while it is on the atlas.

    The lexlc500 shifter gate is the case these are drawn from: its outline
    reached 22.5 texels past the right edge of a 512 atlas, so a ninth of it
    had no partner to exchange with, the flip reached only part of the legend
    and tore the R.  Nothing measured it -- the authoritative-mask path
    asserted a share of 1.0 rather than taking one -- so it shipped.
    """

    # The outline the LC500 shifter gate was actually flipped about.
    OVERHANGING = (
        (459.8, 286.3), (425.5, 406.4), (499.2, 461.7), (533.5, 341.6),
    )

    def test_outline_inside_the_atlas_keeps_every_partner(self) -> None:
        corners = ((40.0, 40.0), (40.0, 140.0), (110.0, 140.0), (110.0, 40.0))
        self.assertAlmostEqual(
            rhd.outline_partner_share(corners, "short", (256, 256)), 1.0
        )

    def test_overhanging_outline_loses_the_share_that_ran_off(self) -> None:
        share = rhd.outline_partner_share(self.OVERHANGING, "short", (512, 512))
        self.assertLess(share, rhd.RhdTextureConfig().min_region_exchangeable)
        self.assertAlmostEqual(share, 0.890, places=2)

    def test_the_same_outline_on_a_larger_atlas_is_whole(self) -> None:
        # Nothing about the shape is wrong; it is the edge it meets.
        self.assertAlmostEqual(
            rhd.outline_partner_share(self.OVERHANGING, "short", (1024, 1024)),
            1.0,
        )

    def test_a_flat_flip_of_the_region_box_is_always_exchangeable(self) -> None:
        # Why dropping the outline is a fallback and not another failure.
        stencil = np.ones((512, 512), dtype=bool)
        self.assertAlmostEqual(
            rhd.exchangeable_share(stencil, (449, 324, 61, 100), "horizontal"),
            1.0,
        )

    def test_derotated_region_is_flipped_flat_rather_than_left_alone(self) -> None:
        # The point of the fallback: the legend still ends up mirrored, in
        # place, and whole.  Leaving it unflipped would ship it backwards.
        image = np.zeros((512, 512, 3), np.uint8)
        image[330:360, 455:470] = 255          # a mark inside the region box
        stencil = np.ones((512, 512), dtype=bool)

        flat = image.copy()
        moved = rhd.apply_masked_flip(flat, stencil, (449, 324, 61, 100), "horizontal")
        self.assertEqual(moved, 61 * 100)
        self.assertFalse(np.array_equal(flat, image))
        # bounded by the box: nothing outside it was touched
        outside = np.ones(image.shape[:2], bool)
        outside[324:424, 449:510] = False
        self.assertTrue(np.array_equal(flat[outside], image[outside]))
        # and it is a true mirror of the box, so no texel was left behind
        box_before = image[324:424, 449:510]
        box_after = flat[324:424, 449:510]
        self.assertTrue(np.array_equal(box_after, box_before[:, ::-1]))
