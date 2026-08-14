#!/usr/bin/env python3
"""Annotate text and symbol regions on BeamNG dashboard/interior textures.

Uses OpenCV's Maximally Stable Extremal Regions (MSER) detector to locate
high-contrast glyphs, numerals, and pictographic symbols painted on the
flat-colour backgrounds typical of BeamNG vehicle gauge faces, warning
indicators, and interior label textures.

Detected regions are grouped by spatial proximity and annotated with a
2 px composite border: a green 1 px inner rectangle on the detected bounds
and a red 1 px outer rectangle immediately around it.  The diagnostic PNG
can be visually inspected before horizontal-flip correction is applied to
the source texture.

Dependencies (cross-platform, Windows + Linux):
    numpy
    opencv-python   (cv2)
    Pillow          (PIL)      DDS decoding

Usage:
    python annotate_texture_regions.py scintilla_gauges.png
    python annotate_texture_regions.py scintilla_interior_b.color.DDS -o annotated.png

Tune detection behaviour by editing DEFAULT_CONFIG near the top of this file.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as _PILImage

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


SHAPE_HULL = "hull"
SHAPE_ROTATED = "rotated"
BOUNDS_SHAPES = (SHAPE_HULL, SHAPE_ROTATED)


@dataclass(frozen=True, slots=True)
class MserConfig:
    """Tuneable MSER and grouping parameters.

    Defaults target BeamNG dashboard and interior labels on flat or
    near-flat colour backgrounds.
    """

    # Which front-end produces the raw boxes.  Everything after it -- box
    # filtering, grouping, domain recovery, size -- is shared.
    #
    # "mser" finds maximally stable extremal regions, which is right for print:
    # a painted glyph is a patch of one colour against another.
    #
    # "edge" is for relief.  A moulded glyph is not a uniform region at all --
    # the trim inside the stroke is the same material, at the same height, as
    # the trim outside it.  What exists is the pair of edges where the surface
    # steps up and back down, so MSER has nothing to be stable about and finds
    # nothing.  A kernel gradient keys on exactly the thing that is there, and
    # costs one convolution instead of a region search.
    # "foreground" is the cheaper default for authored UI/emissive atlases.
    # It removes the dominant flat backing, then labels the remaining connected
    # foreground.  Unlike MSER it does not require a glyph to contain a stable
    # intensity plateau, which is important for anti-aliased screen artwork.
    # "opacity_mask" reads an authored visibility mask literally: non-black
    # texels are content.  It must not infer a dominant backing because a glyph
    # can occupy nearly all of one small UV island.
    box_source:    str   = "foreground"  # "foreground", "opacity_mask", "contrast", "contrast_gpu", "mser", "edge", or "edge_gpu"

    # MSER detector
    delta:         int   = 10  # default: 10
    min_area:      int   = 10  # default: 30
    max_area:      int   = 10000  # default: 1_024
    max_variation: float = 1.0  # default: 0.25
    min_diversity: float = 0.1  # default: 0.2

    # Edge detector.  Read by both the CPU ``edge`` and ModernGL ``edge_gpu``
    # front ends; everything after raw component fitting is shared.
    edge_operator:              str   = "scharr"  # default: "scharr" ("scharr", "sobel", "laplacian")
    edge_kernel_px:             int   = 3  # default: 3 (sobel/laplacian aperture; scharr is always 3)
    edge_blur_sigma:            float = 1.0  # default: 1.0 (pre-smooth, in texture pixels)
    # Threshold against a local average of the gradient rather than a global
    # level.  This is the parameter that decides whether shallow marks are
    # found at all.  On the ardente the trim seam beside "ARDENTE" answers
    # tens of times louder than the lettering, so any single threshold set
    # high enough to exclude the seam also excludes the word -- measured, a
    # 97th-percentile global cut found the word and 12,000 other things, and a
    # 99th found neither.  Compared with its own neighbourhood the word is
    # locally prominent and the flat trim around it is not.
    # Zero falls back to the global percentile below.
    edge_local_window_px:       int   = 64  # default: 64
    edge_local_k:               float = 2.2  # default: 2.2 (multiple of the local mean)
    # Used when edge_local_window_px is zero.
    edge_threshold_percentile:  float = 99.0  # default: 99.0
    edge_threshold_floor:       float = 2.0  # default: 2.0 (absolute gradient floor)
    # A stroke arrives as two parallel edges with untouched material between
    # them.  Closing by about the stroke width fuses them into one mark, which
    # is what the grouping stage downstream expects to be handed.
    edge_close_px:              int   = 5  # default: 5
    edge_dilate_px:             int   = 1  # default: 1
    edge_min_component_px:      int   = 12  # default: 12

    # Foreground-mask detector.  The dominant colour is measured only inside
    # the UV domain, so empty atlas pixels cannot become the background model.
    # A pixel is foreground when it is sufficiently distant from that dominant
    # backing, sufficiently colourful/bright compared with it, or lies on a
    # strong local edge.  The components are deliberately broad; the shared
    # filters and domain checks downstream decide what is safe to flip.
    foreground_background_bins:       int   = 16
    foreground_background_distance:   float = 30.0
    foreground_value_contrast:        float = 20.0
    foreground_min_saturation:        float = 28.0
    foreground_edge_threshold:        float = 35.0
    foreground_open_px:               int   = 1
    foreground_close_px:              int   = 3
    foreground_min_component_px:      int   = 10
    foreground_merge_gap_px:          int   = 6
    foreground_max_coverage:          float = 0.85
    # A foreground component can be a whole button or trim insert because its
    # fill differs from the atlas backing.  Re-run the same dominant-colour
    # idea *inside* solid components: that fill then becomes the local
    # background and independently contrasting content is fitted as children.
    # This is polarity-neutral (dark-on-light and light-on-dark both work),
    # unlike a white-ink threshold.
    foreground_refine_internal_details: bool = False
    foreground_detail_inset_px:        int   = 2
    foreground_detail_min_parent_px:   int   = 100
    foreground_detail_min_coverage:    float = 0.15
    foreground_detail_min_component_px:int   = 5
    foreground_detail_replace_area_ratio: float = 4.0
    foreground_detail_max_depth:        int   = 2
    # Keep the hierarchy a cheap panel/switch refinement, not a second
    # texture-segmentation pass over carpet, seams or large atlas islands.
    foreground_detail_max_parent_px:   int   = 10000
    foreground_detail_max_children:    int   = 8

    # Authoritative opacity/visibility-mask detector.  A small non-zero floor
    # retains anti-aliased glyph edges while ignoring near-black compression
    # noise.  Component sizing and joining deliberately share the foreground
    # controls above so downstream grouping behaves identically.
    opacity_mask_threshold:             int   = 8

    # Local-contrast detector.  A masked box filter calculates the colour
    # average at every pixel, using only texels from the current UV island.
    # The response is the distance from that local mean.  Its absolute floor
    # rejects compression/grain noise; its percentile is measured per island
    # so one noisy or bright atlas area cannot set another island's threshold.
    contrast_kernel_px:                int   = 5
    contrast_min_response:             float = 20.0
    contrast_percentile:               float = 70.0
    contrast_open_px:                  int   = 0
    contrast_close_px:                 int   = 2
    contrast_min_component_px:         int   = 6
    contrast_merge_gap_px:             int   = 5
    contrast_max_coverage:             float = 0.85
    # Only affects local-contrast runs.  A proximity join is refused if its
    # shared-axis corridor crosses too much response which the front end had
    # already classified as high contrast.  This keeps separate button faces
    # apart without remeasuring colour during grouping.
    enable_contrast_continuity_grouping: bool = True
    contrast_bridge_min_response:      float = 8.0
    contrast_bridge_max_high_coverage: float = 0.05
    # A normal map can contribute without becoming a second glyph detector:
    # reject a colour-box proximity join when its corridor crosses a sharp,
    # cached relief edge (for example a button or panel boundary).  Unlike the
    # colour bridge, relief is directional: a glyph's own top/bottom outline
    # may touch a narrow gap, but a real separator must span most of the
    # cross-axis between the two boxes.
    enable_relief_edge_bridge_grouping: bool = True
    relief_bridge_min_response:        float = 150.0
    relief_bridge_min_cross_axis_coverage: float = 0.70

    # Stroke width (Epshtein et al.).  Off by default, so the colour pipeline
    # is exactly what it was.  This is the test that separates lettering from
    # everything else a relief map is full of: a letter is drawn with a pen of
    # one thickness, while a panel seam, a bevel or a run of stitching is not,
    # however strong an edge it makes.  Needs a height-like image -- see
    # stroke_width_transform.
    enable_stroke_width_filter: bool  = False  # default: False
    swt_polarity:               str   = "raised"  # default: "raised" ("raised", "engraved", "both")
    swt_gradient_tolerance_degrees: float = 45.0  # default: 45.0 (how opposed the facing edge must be)
    swt_min_stroke_px:          int   = 2  # default: 2
    swt_max_stroke_px:          int   = 40  # default: 40 (also caps the ray march)
    swt_median_px:              int   = 3  # default: 3 (0 or 1 disables the smoothing pass)
    # Reject above this standard-deviation-over-mean of the widths in a box.
    swt_max_width_variation:    float = 0.5  # default: 0.5
    # And reject a box too little of which carries any stroke at all.
    swt_min_coverage:           float = 0.05  # default: 0.05

    # Ring smoothness.  Judges the material around a region rather than the
    # region itself: a moulded mark is put on plain trim, while the grilles,
    # weaves and carpet that keep being mistaken for one continue past their
    # own bounds in every direction.  Off by default, so the colour pipeline
    # is exactly what it was.
    # Rotated bounds.  Fits the smallest rotated rectangle to the feature in
    # each region and judges the region by its true proportions rather than by
    # an axis-aligned box that a tilted mark barely fills.  Runs before the
    # flatness test, whose percentile is diluted by exactly that background.
    # Off by default.
    enable_rotated_bounds_filter: bool = False  # default: False
    rotated_bounds_min_points:     int = 12  # default: 12 (else nothing is fitted)
    # Rotated area over axis-aligned area.  A mark sitting square in its box
    # approaches 1; a thin diagonal streak collapses towards 0.
    min_rotated_fill:            float = 0.25  # default: 0.25
    # True long-over-short, which for anything tilted the axis-aligned box
    # gets wrong.
    max_rotated_aspect:          float = 14.0  # default: 14.0
    # Which boundary describes the feature.  The hull hugs about a fifth
    # tighter than the rectangle, so it is the better answer to "where is this
    # mark"; the rectangle is kept because it is the thing an angle and a true
    # aspect ratio come from, and because a hull of thirty points is a busier
    # outline to read at a glance.
    bounds_shape:                  str = SHAPE_HULL  # default: "hull"
    # Optional UV-edge direction for adopting a rotated outline.  The closest
    # contiguous UV contour is authoritative: glyph pixels determine only the
    # outline extent and its distance from that contour, never its angle.
    enable_edge_aligned_rotation: bool = False  # default: False
    rotation_edge_min_gap_px:     float = 2.0  # default: 2.0 (touching the edge is not proximity)
    rotation_edge_search_px:      int = 24  # default: 24 (max distance to UV edge)
    rotation_edge_band_px:        int = 6  # default: 6 (edge samples near closest approach)
    rotation_edge_min_points:     int = 8  # default: 8 (else no edge angle is fitted)
    max_rotation_edge_angle_degrees: float = 12.0  # default: 12.0 (max local UV-tangent instability)
    max_opposite_rotation_edge_fraction: float = 0.5  # default: 0.5 (reject UV-bounded strips)
    # How much of the enclosing shape has to be feature before the outline is
    # worth adopting.  An absolute bar, not a comparison against the
    # axis-aligned box: comparing two shapes around one feature, the ratio of
    # their tightness is only the ratio of their areas, so a relative test says
    # nothing the areas did not already say.  Below this the boundary is
    # wrapping mostly empty space -- scattered speckle rather than a mark --
    # and drawing it would overstate the fit.  Measured on the ardente,
    # ARDENTE's hull is 0.41 and AIRBAG's 0.66 against a median of 0.40.
    # The region is kept either way; this decides only which shape describes it.
    min_feature_tightness:       float = 0.20  # default: 0.20
    # And how elongated the feature has to be before its orientation means
    # anything.  A rotated boundary claims a direction, and for a near-square
    # or round mark there is no direction to claim: the fit settles wherever
    # the noise puts it.  Measured on the ardente, both horn icons come out at
    # aspect 1.5 tilted 8 degrees -- a wobble, not an angle -- while ARDENTE
    # reads 9.1 and AIRBAG 5.3 about axes that are unmistakably real.  Below
    # this the region keeps its axis-aligned box, which claims nothing.
    min_rotated_elongation:      float = 2.5  # default: 2.5

    # Text lines.  The only filter here that asks what a mark *is* rather than
    # whether it is one.  A horn or seat pictogram passes every other test --
    # it is a genuine mark on plain trim -- and is still not the AIRBAG text on
    # the fascia.  What makes text text is several similar marks in a row.
    # Off by default.
    enable_text_line_filter:      bool = False  # default: False
    # Components smaller than this are speckle, not characters.
    text_min_component_px:         int = 20  # default: 20
    # Measured on the ardente, AIRBAG has 6 components and ARDENTE 14, while
    # every horn and seat icon has one.  Four leaves room for a short word
    # without admitting a two-part symbol.
    text_min_characters:           int = 4  # default: 4
    # How far the component centres may stray from one straight line.  AIRBAG
    # scatters 0.019 and ARDENTE 0.072; the non-text regions with enough
    # components to measure scatter 0.18 to 0.56.
    max_baseline_scatter:        float = 0.15  # default: 0.15

    # Final relief-text classifier.  Unlike ``Text lines`` this does not rely
    # on connected components: shallow letters can touch in the edge mask.
    # It thresholds the already-rendered relief region at a local response
    # percentile and counts repeated occupied bands along its long axis.
    enable_relief_text_filter: bool = False
    relief_text_response_percentile: float = 80.0
    relief_text_projection_min_coverage: float = 0.10
    relief_text_min_runs: int = 4
    relief_text_min_band_scale: float = 0.20

    # Region flatness.  The other half of the pair: the ring test rejects a
    # region because of what surrounds it, this one because there is nothing
    # in it.  A bolster roll or a fascia curve raises an edge without carrying
    # a mark, and to the ring test that is indistinguishable from lettering on
    # plain trim.  Off by default.
    enable_region_flatness_filter: bool = False  # default: False
    # A high percentile, not a mean: a small glyph in a generous box is mostly
    # background, and what matters is whether anything stands up at all.
    region_flatness_percentile:  float = 90.0  # default: 90.0
    region_flatness_min_domain_px: int = 48  # default: 48 (else nothing is concluded)
    # As a multiple of the atlas's typical gradient.  Reject below this.
    #
    # 10 rather than the 2 or 3 that sounds cautious, because anything the edge
    # front-end returns has edges in it by construction: measured on the
    # ardente, the flattest of 68 detections still read 3.78, so a floor of 3
    # can never fire.  At 10 it takes 21 regions to 13 while ARDENTE reads 45
    # and AIRBAG 64, so both keep four to six times the margin.  Lower it to 6
    # for a gentler cut.
    #
    # It does nothing at all on the colour path -- painted glyphs are
    # high-contrast too, and no threshold up to 10 removed a single one of 113
    # colour detections.  This is a relief-side filter.
    min_region_relief:           float = 10.0  # default: 10.0

    # Relief glyph structure.  A normal-map glyph is either a compact, dense
    # edge mark (an icon or the AIRBAG emboss), or several independent edge
    # fragments lying on one baseline (a word such as ARDENTE).  A trim seam
    # generally has one dominant component and fails both tests.  This is
    # deliberately an edge-only semantic filter, separate from generic colour
    # blob/text heuristics.
    enable_relief_glyph_filter: bool = False
    relief_glyph_min_component_px: int = 8
    relief_glyph_min_line_components: int = 3
    relief_glyph_max_line_scatter: float = 0.15
    relief_glyph_max_dominant_component_fraction: float = 0.5
    relief_glyph_min_compact_edge_coverage: float = 0.30
    # Outlined badges and script marks can be sparse and deliberately curved,
    # so neither the dense-compact nor straight-baseline branch describes them.
    # Their edge pixels still occupy two dimensions, unlike a trim seam.
    relief_glyph_min_outline_components: int = 3
    relief_glyph_min_outline_edge_coverage: float = 0.10
    relief_glyph_max_outline_dominant_component_fraction: float = 0.65
    relief_glyph_min_outline_scatter: float = 0.20

    enable_ring_smoothness_filter: bool = False  # default: False
    ring_smoothness_width_px:      int = 2  # default: 2 (how thick a ring)
    # Held off the region so the mark's own outer edge is not counted as
    # surrounding busyness; without it every real mark measures rough.
    ring_smoothness_margin_px:     int = 3  # default: 3
    ring_smoothness_percentile:  float = 1.0  # default: 1.0 (ring's own level)
    ring_smoothness_min_domain_px: int = 48  # default: 48 (else nothing is concluded)
    # As a multiple of the atlas's typical gradient, so it travels between
    # vehicles.  Measured on the ardente interior: AIRBAG's ring reads 1.01 and
    # ARDENTE's 1.99, against a median of 10.2 over all detections and 63-102
    # for the grilles.  3 sits on a plateau -- raising it from 2.5 costs four
    # regions and raising it again costs none -- so it buys ARDENTE a margin
    # for nothing.
    max_ring_roughness:          float = 3.0  # default: 3.0

    # Late colour-region feature filters.  These run after ring smoothness,
    # using a background colour estimated from an offset ring around the region.
    # They are deliberately off by default while the thresholds are tuned.
    region_feature_ring_margin_px:     int = 3  # default: 3
    region_feature_ring_width_px:      int = 8  # default: 8
    region_feature_colour_tolerance:   int = 16  # default: 16
    region_feature_variance_scale:     float = 2.0  # default: 2.0
    region_feature_min_domain_px:      int = 24  # default: 24
    region_feature_min_px:             int = 12  # default: 12

    # A filled pill, rectangle or dot has feature/hull fill close to one; text
    # and symbols leave more empty hull space.  Reject blobs, keep the rest.
    enable_blob_shape_filter:          bool = True  # default: True
    min_blob_region_area_px:           int = 512  # default: 512
    max_blob_hull_fill:                float = 0.95  # default: 0.95
    min_blob_internal_colour_variation: float = 48.0  # default: 48.0

    # If the same magic-wand feature continues outside the detected bounds, the
    # box is probably a slice of a larger motif/material run rather than a mark.
    enable_feature_extension_filter:   bool = False  # default: False
    feature_extension_context_px:      int = 12  # default: 12
    feature_extension_min_ratio:       float = 0.25  # default: 0.25 (outside / inside)

    # Post-detection filtering.  MSER's own max_area already caps a region far
    # below any sane fraction of the atlas, so no image-fraction test is applied
    # to boxes; measured over eight 4k dashboards it rejected nothing.
    enable_aspect_ratio_filter:       bool  = True  # default: True
    min_aspect:                       float = 0.1  # default: 0.1
    max_aspect:                       float = 10.0  # default: 10.0

    # Early box filters, applied to raw MSER boxes before grouping.  The test
    # that survives is the feature-blob size: an MSER box is homogeneous by
    # definition, so it is judged together with a ring of context, and what
    # separates a glyph stroke from blank trim is whether the non-background
    # pixels form one whole run or break into speckle.
    enable_box_feature_filter:   bool  = True  # default: True
    min_box_width_px:            int   = 3  # default: 3
    min_box_height_px:           int   = 3  # default: 3
    box_feature_colour_tolerance: int  = 16  # default: 16 (magic-wand sensitivity)
    box_feature_min_domain_px:   int   = 10  # default: 10
    box_feature_context_px:      int   = 3  # default: 3 (an MSER box alone is always flat)
    min_box_uv_coverage:         float = 0.8  # default: 0.75 (share of the box inside one UV island)
    box_min_feature_px:          int   = 24  # default: 24 (largest non-background blob)
    # Repeating-pattern rejection, applied to assembled groups.  A group covers
    # a whole panel, so the statistic sees several periods; the same test on a
    # single MSER box separated nothing.  Measured on the dash, perforated
    # grilles score 0.93-0.95 and the chevron weave 0.61-0.63, while every glyph
    # stays at or below 0.29.
    enable_pattern_group_filter: bool  = True  # default: True
    max_pattern_autocorrelation: float = 0.45  # default: 0.45 (reject above)
    pattern_window_scale:        float = 1.0  # default: 1.0 (the group itself)
    pattern_min_window_px:       int   = 48  # default: 48
    pattern_min_period_px:       int   = 4  # default: 4
    pattern_max_period_px:       int   = 160  # default: 160

    # Grouping
    # Boxes are expanded by this many pixels on every side before the
    # minimum union-area test.  This replaces the old additive
    # merge_distance_px + group_dilate_px pair.
    #
    # Candidate joins first collapse every pair with positive overlap, which
    # keeps nested MSER fragments attached to their containing symbol.  Remaining
    # joins are attempted closest-first, but only along rows or columns.  A
    # diagonal group has no useful near-symmetry once flipped, so boxes separated
    # on both axes are left as separate regions.  When the domain filter is
    # active, a merge that would make the accumulated group fail the loose
    # rectangular UV coverage test is refused, leaving the already-valid smaller
    # groups alive.
    merge_distance_px:              int  = 150  # default: 150
    min_group_union_region_px:      int  = 1  # default: 1 (any overlap merges)
    # Proximity grouping is directional.  Boxes must both overlap on the
    # shared axis and have their centre lines close on that axis; overlap alone
    # lets a thin seam attached to an edge of a tall panel masquerade as text
    # on the same row.  Expressed as a fraction of the smaller box's shared
    # dimension so it stays scale independent across texture resolutions.
    group_axis_center_tolerance:    float = 0.50
    enable_island_bounded_grouping: bool = True  # default: True

    # Round regions: inscribe a circle in a squarish group and keep it when the
    # corners it drops are one solid colour or contain no UV domain at all.
    enable_circular_groups:            bool  = True  # default: True
    circular_group_min_squareness:     float = 0.80  # default: 0.80 (min side / max side)
    circular_group_colour_tolerance:   int   = 24  # default: 24
    circular_group_max_corner_content: float = 0.05  # default: 0.05 of the corner area

    # UV island mask used when the caller does not pass one in
    uv_island_mask_path:                   str   = "mesh_segmentation_transform/segmentation_outputs/scintilla_interior_b.color.full_uv_filled_mask.png"  # black UV islands on white background
    # Final size and shape
    enable_final_size_filter:              bool  = True  # default: True
    final_min_width_px:                    int   = 4  # default: 4
    final_min_height_px:                   int   = 4  # default: 4
    final_min_area_px:                     int   = 50  # default: 50
    enable_final_aspect_filter:            bool  = True  # default: True
    final_max_aspect:                      float = 12.0  # default: 12.0 (long side / short)

    # Two-pass shaped-domain recovery.  Grouping uses min_box_uv_coverage as a
    # loose rectangular gate so a circular candidate can form before its shape
    # is known.  This pass then applies the stricter shaped-domain test by
    # splitting any failed group into exact source boxes, removing invalid boxes
    # and regrouping survivors, with one half-distance retry.
    enable_region_domain_filter:           bool  = True  # default: True
    min_region_uv_coverage:                float = 1.0  # default: 0.98

    # Terminal output shaping.  Applied only after every detection/filter stage,
    # so padding cannot cause a region to fail an earlier size, colour, or domain
    # test.  Zero leaves the detected bounds unchanged.
    final_region_padding_px:               int   = 2  # default: 2

    # Annotation drawing
    green_colour:    tuple[int, int, int] = (0, 255, 0)  # default: (0, 255, 0)
    red_colour:      tuple[int, int, int] = (0, 0, 255)  # default: (0, 0, 255)
    green_thickness: int                  = 1  # default: 1
    red_thickness:   int                  = 1  # default: 1


# ---------------------------------------------------------------------------
# Active tuning preset
# ---------------------------------------------------------------------------
#
# Edit the MserConfig field values above while tuning the colour path.
# Relief/normal-map detection uses a separate preset: moulded marks need an
# edge-heavy pipeline and rotated boxes, while printed marks should keep the
# colour-oriented defaults.

DEFAULT_COLOUR_CONFIG = replace(
    MserConfig(),
    contrast_kernel_px=7,
    contrast_close_px=0,
    contrast_min_component_px=24,
    contrast_merge_gap_px=0,
    enable_feature_extension_filter=True,
)
DEFAULT_RELIEF_DETECTION_CONFIG = replace(
    DEFAULT_COLOUR_CONFIG,
    box_source="edge",
    delta=12,
    edge_operator="laplacian",
    swt_polarity="both",
    enable_rotated_bounds_filter=True,
    min_rotated_fill=0.01,
    bounds_shape=SHAPE_ROTATED,
    enable_edge_aligned_rotation=True,
    rotation_edge_search_px=50,
    min_feature_tightness=0.01,
    enable_region_flatness_filter=True,
    region_flatness_percentile=95.0,
    min_region_relief=5.0,
    enable_relief_glyph_filter=True,
    enable_relief_text_filter=True,
    enable_feature_extension_filter=False,
    enable_ring_smoothness_filter=True,
    ring_smoothness_margin_px=4,
    ring_smoothness_percentile=30.0,
    min_box_width_px=8,
    min_box_height_px=8,
    box_feature_min_domain_px=8,
    box_min_feature_px=15,
    merge_distance_px=21,
    final_max_aspect=20.0,
)
DEFAULT_CONFIG = DEFAULT_COLOUR_CONFIG


@dataclass(frozen=True, slots=True)
class UvIslandSymmetryConfig:
    """Independent UV-island reflection-symmetry annotation settings.

    These settings are not read by the MSER, grouping, or clean-up pipeline.
    They only control a blue contour overlay derived directly from the UV mask.
    """

    enable_uv_island_symmetry: bool = True
    min_uv_island_symmetry: float = 0.98
    blue_colour: tuple[int, int, int] = (255, 0, 0)  # BGR
    blue_thickness: int = 1


DEFAULT_UV_ISLAND_SYMMETRY_CONFIG = UvIslandSymmetryConfig()


@dataclass(frozen=True, slots=True)
class UvIslandSymmetryMatch:
    """One connected UV island accepted by either reflection-axis test."""

    label: int
    bounds: tuple[int, int, int, int]
    area_px: int
    x_similarity: float
    y_similarity: float
    matched_axes: tuple[str, ...]
    contours: tuple[tuple[tuple[int, int], ...], ...]


@dataclass(frozen=True, slots=True)
class TextureRegionAnnotationOutputs:
    """External files written by the annotation pipeline."""

    annotated_image: Path
    summary_json: Path | None = None


@dataclass(frozen=True, slots=True)
class DetectionStage:
    """One step of the pipeline, with what it kept and what it removed.

    ``rejected`` holds the boxes this step alone discarded, so a tuning UI can
    show why a region disappeared without re-running earlier steps.
    """

    key: str
    title: str
    kept: tuple[tuple[int, int, int, int], ...]
    rejected: tuple[tuple[int, int, int, int], ...] = ()
    detail: str = ""
    adjusted: int = 0  # regions this stage moved, expanded, or absorbed
    # Radius per kept region, or None where the region stays rectangular.
    circles: tuple[int | None, ...] = ()
    # Corner points of the rotated rectangle fitted to each kept region, where
    # one was fitted.  Carried so the viewer can outline what was actually
    # measured rather than the axis-aligned box it happened to sit in -- on a
    # tilted mark those are very different shapes, and a filter judged on one
    # while the other is drawn cannot be tuned by eye.
    rotations: tuple[tuple[tuple[float, float], ...] | None, ...] = ()


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------


def load_image(path: Path) -> np.ndarray:
    """Load a texture as a BGR uint8 ndarray.

    PNG, JPG, BMP, TIFF and other formats supported by the local OpenCV build
    are read directly; DDS goes through Pillow.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Texture not found: {path}")

    if path.suffix.lower() == ".dds":
        with _PILImage.open(path) as pil_image:
            rgba = pil_image.convert("RGBA")
        return cv2.cvtColor(np.asarray(rgba, dtype=np.uint8), cv2.COLOR_RGBA2BGR)

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(
            f"OpenCV could not decode {path.name}.  "
            f"Use PNG, JPG, BMP, TIFF, DDS, or another format supported by "
            f"your OpenCV build."
        )
    return image


def load_uv_island_mask(path: Path, image_shape: tuple[int, int]) -> np.ndarray:
    """Load black-island-on-white UV mask as a boolean island mask."""
    mask_image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask_image is None:
        raise FileNotFoundError(f"UV island mask not found or unreadable: {path}")

    height, width = image_shape[:2]
    if mask_image.shape[:2] != (height, width):
        mask_image = cv2.resize(
            mask_image,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
    return mask_image < 128


def _reflection_similarity(
    component: np.ndarray,
    reflected: np.ndarray,
) -> float:
    """Return intersection-over-union agreement with a reflected island mask."""
    union = component | reflected
    union_area = int(union.sum())
    if union_area == 0:
        return 1.0
    intersection = component & reflected
    return float(intersection.sum()) / float(union_area)


def analyse_uv_island_symmetry(
    uv_mask: np.ndarray | None,
    config: UvIslandSymmetryConfig | None = None,
) -> tuple[UvIslandSymmetryMatch, ...]:
    """Find connected UV islands symmetric about either local bounding-box axis.

    X-axis symmetry compares the island with a top/bottom reflection.  Y-axis
    symmetry compares it with a left/right reflection.  Similarity is measured
    as mask intersection-over-union, so 1.0 is exact and lower values tolerate
    an increasing fraction of asymmetric rasterised area.
    """
    config = config or DEFAULT_UV_ISLAND_SYMMETRY_CONFIG
    if (
        not config.enable_uv_island_symmetry
        or uv_mask is None
        or not bool(uv_mask.any())
    ):
        return ()

    threshold = min(max(float(config.min_uv_island_symmetry), 0.0), 1.0)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        uv_mask.astype(np.uint8), connectivity=8
    )
    matches: list[UvIslandSymmetryMatch] = []

    for label in range(1, count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        component = labels[y : y + h, x : x + w] == label

        x_similarity = _reflection_similarity(component, np.flipud(component))
        y_similarity = _reflection_similarity(component, np.fliplr(component))
        axes = tuple(
            axis
            for axis, score in (("x", x_similarity), ("y", y_similarity))
            if score >= threshold
        )
        if not axes:
            continue

        local_contours, _hierarchy = cv2.findContours(
            component.astype(np.uint8),
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        contours: list[tuple[tuple[int, int], ...]] = []
        for contour in local_contours:
            points = contour.reshape(-1, 2)
            if points.size == 0:
                continue
            contours.append(
                tuple((int(px) + x, int(py) + y) for px, py in points)
            )

        matches.append(
            UvIslandSymmetryMatch(
                label=label,
                bounds=(x, y, w, h),
                area_px=area,
                x_similarity=x_similarity,
                y_similarity=y_similarity,
                matched_axes=axes,
                contours=tuple(contours),
            )
        )

    return tuple(matches)


def draw_uv_island_symmetry_annotations(
    image: np.ndarray,
    matches: tuple[UvIslandSymmetryMatch, ...],
    config: UvIslandSymmetryConfig | None = None,
) -> np.ndarray:
    """Draw accepted UV-island borders blue without changing detection state."""
    config = config or DEFAULT_UV_ISLAND_SYMMETRY_CONFIG
    if not config.enable_uv_island_symmetry or not matches:
        return image

    annotated = image.copy()
    thickness = max(int(config.blue_thickness), 1)
    for match in matches:
        for contour in match.contours:
            if not contour:
                continue
            points = np.asarray(contour, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(
                annotated,
                [points],
                True,
                config.blue_colour,
                thickness,
                cv2.LINE_8,
            )
    return annotated


def resolve_config_path(path_text: str) -> Path:
    """Resolve config paths relative to cwd first, then repo root."""
    path = Path(path_text)
    if path.is_absolute() or path.is_file():
        return path

    script_root = Path(__file__).resolve().parent
    repo_root = script_root.parent
    repo_relative = repo_root / path
    if repo_relative.is_file():
        return repo_relative

    script_relative = script_root / path
    if script_relative.is_file():
        return script_relative

    return repo_relative


def config_with_uv_mask_path(
    config: MserConfig,
    uv_island_mask_path: Path | str | None,
) -> MserConfig:
    """Return config with an explicit external UV mask path when provided."""
    if uv_island_mask_path is None:
        return config
    return replace(config, uv_island_mask_path=str(uv_island_mask_path))


# ---------------------------------------------------------------------------
# MSER detection and filtering
# ---------------------------------------------------------------------------


def detect_mser_boxes(
    grey: np.ndarray,
    config: MserConfig,
) -> np.ndarray:
    """Return filtered axis-aligned bounding boxes from MSER detection.

    Each row is ``[x, y, w, h]`` in pixel coordinates.  Only the aspect-ratio
    test is applied here.  Size is left to ``config.min_area``/``max_area``,
    which MSER enforces on the region itself rather than on its bounding box,
    and to the minimum box dimensions in the next stage.
    """
    mser = cv2.MSER_create(
        config.delta,
        config.min_area,
        config.max_area,
        config.max_variation,
        config.min_diversity,
    )
    regions, _ = mser.detectRegions(grey)

    if len(regions) == 0:
        return np.empty((0, 4), dtype=np.int32)

    boxes: list[tuple[int, int, int, int]] = []
    for region in regions:
        x, y, w, h = cv2.boundingRect(region)
        aspect = max(w, h) / max(min(w, h), 1)
        if config.enable_aspect_ratio_filter and (
            aspect < config.min_aspect or aspect > config.max_aspect
        ):
            continue
        boxes.append((x, y, w, h))

    if not boxes:
        return np.empty((0, 4), dtype=np.int32)

    return np.asarray(boxes, dtype=np.int32)


def edge_response(grey: np.ndarray, config: MserConfig) -> np.ndarray:
    """Gradient magnitude of a greyscale image, as a float32 map.

    Scharr by default: at a 3x3 aperture it is markedly more rotation-accurate
    than Sobel, which matters here because moulded lettering has strokes at
    every angle and a detector that answers differently to a vertical and a
    diagonal stroke will find some letters and lose others.
    """
    if config.box_source == "edge_gpu":
        return edge_response_gpu(grey, config)
    source = grey.astype(np.float32)
    if config.edge_blur_sigma > 0:
        source = cv2.GaussianBlur(source, (0, 0), config.edge_blur_sigma)

    if config.edge_operator == "laplacian":
        aperture = max(int(config.edge_kernel_px) | 1, 1)
        return np.abs(cv2.Laplacian(source, cv2.CV_32F, ksize=aperture))
    if config.edge_operator == "sobel":
        aperture = max(int(config.edge_kernel_px) | 1, 1)
        dx = cv2.Sobel(source, cv2.CV_32F, 1, 0, ksize=aperture)
        dy = cv2.Sobel(source, cv2.CV_32F, 0, 1, ksize=aperture)
    else:
        dx = cv2.Scharr(source, cv2.CV_32F, 1, 0)
        dy = cv2.Scharr(source, cv2.CV_32F, 0, 1)
    return cv2.magnitude(dx, dy)


def edge_response_gpu(grey: np.ndarray, config: MserConfig) -> np.ndarray:
    """Run the edge/Laplacian front end on the shared ModernGL context.

    ``edge_gpu`` is an explicit detector mode, so this function never hides a
    GPU failure behind a CPU result.  The production build and tuning harness
    call the same service in ``texture_local_contrast_gpu``.
    """
    from mesh_segmentation_transform.texture_local_contrast_gpu import (
        compute_edge_response,
    )
    return compute_edge_response(
        np.ascontiguousarray(grey), config.edge_operator,
        config.edge_kernel_px, config.edge_blur_sigma,
    )


def edge_mask(
    grey: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Threshold the gradient into an edge mask, and return it with the response.

    Shared by the edge box source and the stroke-width transform, so both agree
    on what an edge is; two different answers to that would make the stroke
    widths describe edges the boxes were never built from.
    """
    response = edge_response(grey, config)
    return edge_mask_from_response(response, uv_mask, config)


def edge_mask_from_response(
    response: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Threshold an already-computed edge response inside one UV domain."""
    domain = uv_mask if uv_mask is not None else np.ones(response.shape[:2], bool)
    inside = response[domain]
    if inside.size == 0:
        return np.zeros(response.shape[:2], dtype=bool), response

    if config.edge_local_window_px > 0:
        window = max(int(config.edge_local_window_px), 3)
        # Averaged over the domain only, so material beside a void is compared
        # with material rather than with the empty half of its neighbourhood.
        weight = domain.astype(np.float32)
        total = cv2.blur(response * weight, (window, window))
        share = cv2.blur(weight, (window, window))
        local = total / np.maximum(share, 1e-6)
        threshold = np.maximum(
            local * float(config.edge_local_k), float(config.edge_threshold_floor)
        )
    else:
        percentile = float(
            np.percentile(inside, min(max(config.edge_threshold_percentile, 0.0), 100.0))
        )
        threshold = np.float32(max(percentile, float(config.edge_threshold_floor)))
    return (response >= threshold) & domain, response


def detect_edge_boxes_with_response(
    grey: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return edge boxes and the response used to construct them.

    Closing runs before components are labelled so a stroke's two edges become
    one blob; without it every letter arrives as a pair of thin rings and the
    grouping stage has twice as much to join.  Keeping the response lets the
    grouping stage distinguish a blank gap between glyphs from a sharp relief
    boundary, without computing the Laplacian a second time.
    """
    mask, response = edge_mask(grey, uv_mask, config)
    return detect_edge_boxes_from_response(mask, response, config)


def detect_edge_boxes_from_response(
    mask: np.ndarray,
    response: np.ndarray,
    config: MserConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit edge components from a thresholded response already in memory."""
    edges = mask.astype(np.uint8)
    if not bool(edges.any()):
        return np.empty((0, 4), dtype=np.int32), response

    if config.edge_close_px > 0:
        size = max(int(config.edge_close_px), 1)
        edges = cv2.morphologyEx(
            edges, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
        )
    if config.edge_dilate_px > 0:
        size = max(int(config.edge_dilate_px) * 2 + 1, 1)
        edges = cv2.dilate(
            edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        )

    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        edges, connectivity=8
    )
    boxes: list[tuple[int, int, int, int]] = []
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) < max(config.edge_min_component_px, 1):
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        aspect = max(w, h) / max(min(w, h), 1)
        if config.enable_aspect_ratio_filter and (
            aspect < config.min_aspect or aspect > config.max_aspect
        ):
            continue
        boxes.append((x, y, w, h))
    if not boxes:
        return np.empty((0, 4), dtype=np.int32), response
    return np.asarray(boxes, dtype=np.int32), response


def _merge_foreground_boxes(
    boxes: list[tuple[int, int, int, int]],
    gap: int,
    charts: list[int | None] | None = None,
) -> list[tuple[int, int, int, int]]:
    """Merge nearby foreground components without requiring MSER fragments.

    Screen labels often arrive as one component per letter or icon stroke.  A
    small Chebyshev gap joins those into useful rectangular correction regions
    before the normal grouping/filtering pipeline refines them further.

    ``charts`` names the UV chart each component belongs to, and a join across
    two charts is refused on the same grounds grouping refuses one: charts are
    coalesced into a single detection consumer only because they can share
    atlas texels, which is not a reason to let a mark on one chart absorb a
    mark on another.  Proximity in the atlas says nothing about proximity on
    the car -- the LC500's door otherwise welds its mirror-select icons, both
    padlocks and the whole ``AUTO L R`` legend into one box at a gap of six
    texels, and six separate legends stop existing as candidates before any
    chart-aware stage gets to see them.
    """
    merged = list(boxes)
    owners = list(charts) if charts is not None else [None] * len(merged)
    gap = max(int(gap), 0)
    changed = True
    while changed:
        changed = False
        for left, first in enumerate(merged):
            ax, ay, aw, ah = first
            ax1, ay1 = ax + aw, ay + ah
            for right in range(left + 1, len(merged)):
                bx, by, bw, bh = merged[right]
                bx1, by1 = bx + bw, by + bh
                dx = max(bx - ax1, ax - bx1, 0)
                dy = max(by - ay1, ay - by1, 0)
                if max(dx, dy) > gap:
                    continue
                # A component on no recorded chart says nothing either way.
                if (
                    owners[left] is not None
                    and owners[right] is not None
                    and owners[left] != owners[right]
                ):
                    continue
                x0, y0 = min(ax, bx), min(ay, by)
                x1, y1 = max(ax1, bx1), max(ay1, by1)
                merged[left] = (x0, y0, x1 - x0, y1 - y0)
                if owners[left] is None:
                    owners[left] = owners[right]
                merged.pop(right)
                owners.pop(right)
                changed = True
                break
            if changed:
                break
    return merged


def _refine_foreground_internal_details(
    bgr: np.ndarray,
    domain: np.ndarray,
    foreground: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    config: MserConfig,
) -> list[tuple[int, int, int, int]]:
    """Replace solid, broad foreground blobs with their local detail.

    The first foreground pass correctly says that a grey switch insert differs
    from the black texture backing, but that is normally not the content which
    needs mirroring.  For each sufficiently solid box, sample its foreground
    pixels as a *new* local background, then retain only pixels which differ
    from that local mode.  It is deliberately a colour-distance/value test,
    not a bright-pixel rule, so black glyphs on light switches work just as
    well as white glyphs on dark ones.

    The outer inset suppresses the button boundary, whose contrast is useful
    for finding the parent but should not by itself become child detail.
    """
    if not config.foreground_refine_internal_details:
        return boxes

    bins = max(int(config.foreground_background_bins), 2)
    inset = max(int(config.foreground_detail_inset_px), 0)
    min_parent = max(int(config.foreground_detail_min_parent_px), 1)
    min_child = max(int(config.foreground_detail_min_component_px), 1)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    max_depth = max(int(config.foreground_detail_max_depth), 1)
    max_parent = max(int(config.foreground_detail_max_parent_px), min_parent)
    max_children = max(int(config.foreground_detail_max_children), 1)

    def local_children(
        x: int, y: int, width: int, height: int, support: np.ndarray,
    ) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
        """Find one locally-contrasting layer inside an exact parent mask."""
        parent_area = width * height
        if (
            parent_area < min_parent
            or parent_area > max_parent
            or width <= inset * 2
            or height <= inset * 2
        ):
            return []
        support_count = int(support.sum())
        if (
            support_count < min_child
            or support_count / max(parent_area, 1)
            < float(config.foreground_detail_min_coverage)
        ):
            return []

        # A dark glyph in a light switch is an enclosed hole in the parent
        # support.  Include only such holes, never exterior atlas pixels.
        padded_inverse = (~np.pad(support, 1, constant_values=False)).astype(np.uint8)
        cv2.floodFill(padded_inverse, None, (0, 0), 0)
        detail_domain = support | padded_inverse[1:-1, 1:-1].astype(bool)
        detail_domain &= domain[y:y + height, x:x + width]
        if inset:
            detail_domain[:inset, :] = False
            detail_domain[height - inset:, :] = False
            detail_domain[:, :inset] = False
            detail_domain[:, width - inset:] = False

        crop = bgr[y:y + height, x:x + width]
        quantized = ((crop.astype(np.uint16) * bins) // 256).astype(np.uint8)
        samples = quantized[support]
        packed = (
            samples[:, 0].astype(np.int32) * bins * bins
            + samples[:, 1].astype(np.int32) * bins
            + samples[:, 2].astype(np.int32)
        )
        mode = int(np.bincount(packed, minlength=bins ** 3).argmax())
        local_background = np.array(
            [mode // (bins * bins), (mode // bins) % bins, mode % bins],
            dtype=np.float32,
        )
        local_background = (local_background + 0.5) * (256.0 / bins)
        distance = np.linalg.norm(crop.astype(np.float32) - local_background, axis=2)
        local_value_delta = np.abs(
            hsv[y:y + height, x:x + width, 2].astype(np.float32)
            - float(local_background.max())
        )
        local_saturation = hsv[y:y + height, x:x + width, 1].astype(np.float32)
        detail = (
            (distance >= float(config.foreground_background_distance))
            | (local_value_delta >= float(config.foreground_value_contrast))
            | ((local_saturation >= float(config.foreground_min_saturation))
               & (distance >= float(config.foreground_background_distance) * 0.5))
        ) & detail_domain
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            detail.astype(np.uint8), 8,
        )
        children: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
        for label in range(1, count):
            child_x, child_y, child_w, child_h, child_area = (
                int(value) for value in stats[label]
            )
            if child_area >= min_child and child_w > 0 and child_h > 0:
                children.append((
                    (x + child_x, y + child_y, child_w, child_h),
                    detail[child_y:child_y + child_h, child_x:child_x + child_w],
                ))
        return children if len(children) <= max_children else []

    def refine_tree(
        x: int, y: int, width: int, height: int, support: np.ndarray, depth: int,
    ) -> list[tuple[int, int, int, int]]:
        children = local_children(x, y, width, height, support)
        if not children:
            return [(x, y, width, height)]
        leaves = [
            leaf
            for child_box, child_support in children
            for leaf in (
                refine_tree(*child_box, child_support, depth + 1)
                if depth < max_depth else [child_box]
            )
        ]
        leaf_box_area = sum(leaf_width * leaf_height for _, _, leaf_width, leaf_height in leaves)
        return (
            leaves
            if leaf_box_area
            and width * height >= leaf_box_area
            * float(config.foreground_detail_replace_area_ratio)
            else [(x, y, width, height)]
        )

    refined = [
        leaf
        for x, y, width, height in boxes
        for leaf in refine_tree(
            x, y, width, height,
            foreground[y:y + height, x:x + width].astype(bool)
            & domain[y:y + height, x:x + width],
            1,
        )
    ]
    return _merge_foreground_boxes(refined, config.foreground_merge_gap_px)


def detect_foreground_boxes(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig,
) -> np.ndarray:
    """Return boxes from a cheap dominant-background foreground mask.

    This front-end is intentionally independent of MSER.  It works well on
    base, emissive, opacity, overlay and glow-state images where useful marks
    are defined by contrast against a mostly flat panel rather than by stable
    extremal regions.  If an RGBA image is supplied its alpha is folded into
    the mask; normal BGR callers continue to work unchanged.
    """
    if image.ndim != 3 or image.shape[2] < 3:
        return np.empty((0, 4), dtype=np.int32)
    bgr = image[:, :, :3]
    height, width = bgr.shape[:2]
    domain = (
        uv_mask.astype(bool)
        if uv_mask is not None and uv_mask.shape[:2] == (height, width)
        else np.ones((height, width), dtype=bool)
    )
    if not bool(domain.any()):
        return np.empty((0, 4), dtype=np.int32)

    # Quantisation makes the mode resilient to compression noise and gentle
    # gradients while retaining the authored panel colour.
    bins = max(int(config.foreground_background_bins), 2)
    quantized = ((bgr.astype(np.uint16) * bins) // 256).astype(np.uint8)
    samples = quantized[domain]
    packed = (
        samples[:, 0].astype(np.int32) * bins * bins
        + samples[:, 1].astype(np.int32) * bins
        + samples[:, 2].astype(np.int32)
    )
    mode = int(np.bincount(packed, minlength=bins ** 3).argmax())
    background = np.array(
        [mode // (bins * bins), (mode // bins) % bins, mode % bins],
        dtype=np.float32,
    )
    background = (background + 0.5) * (256.0 / bins)
    pixels = bgr.astype(np.float32)
    distance = np.linalg.norm(pixels - background, axis=2)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    background_value = float(background.max())
    value_delta = np.abs(hsv[:, :, 2].astype(np.float32) - background_value)
    saturation = hsv[:, :, 1].astype(np.float32)
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gradient = edge_response(grey, config)

    mask = (
        (distance >= float(config.foreground_background_distance))
        | (value_delta >= float(config.foreground_value_contrast))
        | ((saturation >= float(config.foreground_min_saturation))
           & (distance >= float(config.foreground_background_distance) * 0.5))
        | (gradient >= float(config.foreground_edge_threshold))
    ) & domain
    if image.shape[2] >= 4:
        mask &= image[:, :, 3] > 0
    work = mask.astype(np.uint8)
    if config.foreground_open_px > 0:
        size = max(int(config.foreground_open_px) * 2 + 1, 1)
        work = cv2.morphologyEx(
            work, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
        )
    if config.foreground_close_px > 0:
        size = max(int(config.foreground_close_px) * 2 + 1, 1)
        work = cv2.morphologyEx(
            work, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
        )
    work &= domain.astype(np.uint8)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(work, 8)
    boxes: list[tuple[int, int, int, int]] = []
    min_area = max(int(config.foreground_min_component_px), 1)
    for label in range(1, count):
        x, y, w, h, area = (int(value) for value in stats[label])
        if area < min_area or w <= 0 or h <= 0:
            continue
        coverage = area / max(w * h, 1)
        # A solid icon is a valid foreground component.  Reject only a nearly
        # solid component that also consumes most of the selected UV domain --
        # that is the backing panel escaping the background model, not a mark.
        if (
            coverage > float(config.foreground_max_coverage)
            and area / max(int(domain.sum()), 1) > float(config.foreground_max_coverage)
        ):
            continue
        boxes.append((x, y, w, h))
    boxes = _merge_foreground_boxes(boxes, config.foreground_merge_gap_px)
    boxes = _refine_foreground_internal_details(
        bgr, domain, work, boxes, config,
    )
    if not boxes:
        return np.empty((0, 4), dtype=np.int32)
    return np.asarray(boxes, dtype=np.int32)


def _component_chart_owners(
    labels: np.ndarray,
    count: int,
    island_bits: np.ndarray | None,
) -> list[int | None] | None:
    """Which UV chart owns each connected component, by texel count.

    A component is not asked which charts it touches but which one it belongs
    to.  Two charts abutting in the atlas both rasterise their shared boundary
    row, so a mark can overlap its neighbour by a few texels while plainly
    belonging to one chart -- the LC500 door's padlock puts 449 texels on its
    own chart and 3 on the icon chart above it.  Reading that as shared
    ownership would let the two weld together anyway.
    """
    if island_bits is None or count <= 1:
        return None
    present = int(np.bitwise_or.reduce(island_bits.ravel()))
    if not present:
        return None
    best = np.zeros(count, dtype=np.int64)
    owner = np.full(count, -1, dtype=np.int64)
    for bit in range(int(present).bit_length()):
        if not (present >> bit) & 1:
            continue
        selected = ((island_bits >> np.uint64(bit)) & np.uint64(1)).astype(bool)
        counts = np.bincount(labels[selected].ravel(), minlength=count)[:count]
        better = counts > best
        owner[better] = bit
        best[better] = counts[better]
    return [int(value) if value >= 0 else None for value in owner]


def detect_opacity_mask_boxes(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig,
    island_bits: np.ndarray | None = None,
) -> np.ndarray:
    """Return components declared visible by an authored opacity mask.

    Unlike dominant-background detection, this preserves a glyph which fills
    most or all of its UV island.  Production invokes it per topological UV
    island -- except where several charts were coalesced into one consumer for
    sharing texels, and then clipping to ``uv_mask`` no longer separates them.
    ``island_bits`` carries which chart owns each texel, so the merge can stop
    at a chart boundary rather than weld neighbours that are only adjacent in
    the atlas.
    """
    if image.ndim != 3 or image.shape[2] < 3:
        return np.empty((0, 4), dtype=np.int32)
    height, width = image.shape[:2]
    domain = (
        uv_mask.astype(bool)
        if uv_mask is not None and uv_mask.shape[:2] == (height, width)
        else np.ones((height, width), dtype=bool)
    )
    if not bool(domain.any()):
        return np.empty((0, 4), dtype=np.int32)

    intensity = np.max(image[:, :, :3], axis=2)
    work = (
        (intensity > max(int(config.opacity_mask_threshold), 0)) & domain
    ).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(work, 8)
    min_area = max(int(config.foreground_min_component_px), 1)
    owners = _component_chart_owners(labels, count, island_bits)
    kept = [
        label
        for label in range(1, count)
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area
        and int(stats[label, cv2.CC_STAT_WIDTH]) > 0
        and int(stats[label, cv2.CC_STAT_HEIGHT]) > 0
    ]
    boxes = [
        (int(stats[label, cv2.CC_STAT_LEFT]),
         int(stats[label, cv2.CC_STAT_TOP]),
         int(stats[label, cv2.CC_STAT_WIDTH]),
         int(stats[label, cv2.CC_STAT_HEIGHT]))
        for label in kept
    ]
    boxes = _merge_foreground_boxes(
        boxes,
        config.foreground_merge_gap_px,
        [owners[label] for label in kept] if owners is not None else None,
    )
    if not boxes:
        return np.empty((0, 4), dtype=np.int32)
    return np.asarray(boxes, dtype=np.int32)


def detect_edge_boxes(
    grey: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig,
) -> np.ndarray:
    """Return bounding boxes of connected edge structure."""
    return detect_edge_boxes_with_response(grey, uv_mask, config)[0]


@dataclass(frozen=True, slots=True)
class LocalContrastDetection:
    """Reusable local-contrast data for detection and later grouping."""

    boxes: np.ndarray
    response: np.ndarray
    threshold: float


def local_contrast_detection_from_response(
    response: np.ndarray,
    domain: np.ndarray,
    config: MserConfig,
) -> LocalContrastDetection:
    """Apply the shared percentile/mask/component stages to one response."""
    if response.shape != domain.shape or not bool(domain.any()):
        return LocalContrastDetection(
            np.empty((0, 4), dtype=np.int32),
            np.asarray(response, dtype=np.float32), 0.0,
        )
    island_response = response[domain]
    percentile = min(max(float(config.contrast_percentile), 0.0), 100.0)
    threshold = max(
        float(config.contrast_min_response),
        float(np.percentile(island_response, percentile)),
    )
    work = ((response >= threshold) & domain).astype(np.uint8)
    if config.contrast_open_px > 0:
        size = max(int(config.contrast_open_px) * 2 + 1, 1)
        work = cv2.morphologyEx(
            work, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
        )
    if config.contrast_close_px > 0:
        size = max(int(config.contrast_close_px) * 2 + 1, 1)
        work = cv2.morphologyEx(
            work, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
        )
    work &= domain.astype(np.uint8)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(work, 8)
    min_area = max(int(config.contrast_min_component_px), 1)
    boxes: list[tuple[int, int, int, int]] = []
    domain_area = max(int(domain.sum()), 1)
    for label in range(1, count):
        x, y, box_width, box_height, area = (int(value) for value in stats[label])
        if area < min_area or box_width <= 0 or box_height <= 0:
            continue
        coverage = area / max(box_width * box_height, 1)
        if (
            coverage > float(config.contrast_max_coverage)
            and area / domain_area > float(config.contrast_max_coverage)
        ):
            continue
        boxes.append((x, y, box_width, box_height))
    boxes = _merge_foreground_boxes(boxes, config.contrast_merge_gap_px)
    return LocalContrastDetection(
        np.asarray(boxes, dtype=np.int32)
        if boxes else np.empty((0, 4), dtype=np.int32),
        response.astype(np.float32, copy=False),
        threshold,
    )


def detect_local_contrast(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig,
) -> LocalContrastDetection:
    """Return boxes whose pixels are unusually contrasty for this UV island.

    ``boxFilter`` provides the kernel measurement in vectorised OpenCV code.
    Dividing filtered colours by filtered domain weights is important: the
    kernel cannot borrow arbitrary neighbouring atlas texels across a UV-island
    boundary.  The response has no bright/dark polarity, so glyphs, cut-outs,
    coloured marks, and embossed-looking colour details share one front end.
    """
    if image.ndim != 3 or image.shape[2] < 3:
        return LocalContrastDetection(
            np.empty((0, 4), dtype=np.int32), np.empty((0, 0), dtype=np.float32), 0.0,
        )
    bgr = image[:, :, :3]
    height, width = bgr.shape[:2]
    domain = (
        uv_mask.astype(bool)
        if uv_mask is not None and uv_mask.shape[:2] == (height, width)
        else np.ones((height, width), dtype=bool)
    )
    if image.shape[2] >= 4:
        domain &= image[:, :, 3] > 0
    if not bool(domain.any()):
        return LocalContrastDetection(
            np.empty((0, 4), dtype=np.int32), np.zeros((height, width), dtype=np.float32), 0.0,
        )

    kernel = max(int(config.contrast_kernel_px), 1)
    kernel += 1 - kernel % 2  # a centred kernel has no directional bias
    weights = cv2.boxFilter(
        domain.astype(np.float32), cv2.CV_32F, (kernel, kernel), normalize=False,
        borderType=cv2.BORDER_REPLICATE,
    )
    pixels = bgr.astype(np.float32)
    weighted = pixels * domain[:, :, None].astype(np.float32)
    local_sum = cv2.boxFilter(
        weighted, cv2.CV_32F, (kernel, kernel), normalize=False,
        borderType=cv2.BORDER_REPLICATE,
    )
    local_mean = local_sum / np.maximum(weights[:, :, None], 1.0)
    response = np.linalg.norm(pixels - local_mean, axis=2)
    return local_contrast_detection_from_response(response, domain, config)


def detect_local_contrast_gpu(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig,
) -> LocalContrastDetection:
    """Run the local-contrast kernel on GPU; no CPU detector fallback exists."""
    if image.ndim != 3 or image.shape[2] < 3:
        return LocalContrastDetection(
            np.empty((0, 4), dtype=np.int32), np.empty((0, 0), dtype=np.float32), 0.0,
        )
    bgr = image[:, :, :3]
    height, width = bgr.shape[:2]
    domain = (
        uv_mask.astype(bool)
        if uv_mask is not None and uv_mask.shape[:2] == (height, width)
        else np.ones((height, width), dtype=bool)
    )
    if image.shape[2] >= 4:
        domain &= image[:, :, 3] > 0
    if not bool(domain.any()):
        return local_contrast_detection_from_response(
            np.zeros((height, width), dtype=np.float32), domain, config,
        )
    from mesh_segmentation_transform.texture_local_contrast_gpu import (
        compute_local_contrast_response,
    )
    response = compute_local_contrast_response(bgr, domain, config.contrast_kernel_px)
    return local_contrast_detection_from_response(response, domain, config)


def detect_local_contrast_gpu_batch(
    entries: list[tuple[np.ndarray, np.ndarray]],
    config: MserConfig,
) -> list[LocalContrastDetection]:
    """GPU response for independent UV crops, packed before upload/dispatch."""
    from mesh_segmentation_transform.texture_local_contrast_gpu import (
        compute_local_contrast_responses,
    )
    images = [image[:, :, :3] for image, _mask in entries]
    domains = [mask.astype(bool) for _image, mask in entries]
    responses = compute_local_contrast_responses(images, domains, config.contrast_kernel_px)
    return [
        local_contrast_detection_from_response(response, domain, config)
        for response, domain in zip(responses, domains)
    ]


def prewarm_local_contrast_gpu() -> None:
    """Schedule context/shader creation before a GPU detector run is requested."""
    from mesh_segmentation_transform.texture_local_contrast_gpu import prewarm_gpu
    prewarm_gpu()


def prewarm_edge_gpu() -> None:
    """Prewarm the same ModernGL context used by GPU local contrast."""
    from mesh_segmentation_transform.texture_local_contrast_gpu import prewarm_gpu
    prewarm_gpu()


def local_contrast_gpu_warm_state() -> str:
    """Expose GPU context warm-up state to the tuning timing report."""
    from mesh_segmentation_transform.texture_local_contrast_gpu import gpu_warm_state
    return gpu_warm_state()


def edge_gpu_warm_state() -> str:
    """Expose the shared edge GPU context warm-up state."""
    from mesh_segmentation_transform.texture_local_contrast_gpu import gpu_warm_state
    return gpu_warm_state()


def detect_local_contrast_boxes(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig,
) -> np.ndarray:
    """Return only the raw boxes for callers that do not need join evidence."""
    return detect_local_contrast(image, uv_mask, config).boxes


@dataclass(frozen=True, slots=True)
class FeatureShape:
    """The feature inside a region, and the shapes that enclose it."""

    area_px: float  # feature texels
    hull: tuple[tuple[float, float], ...]
    hull_area: float
    rectangle: tuple[tuple[float, float], tuple[float, float], float]
    rectangle_area: float

    def tightness(self, shape: str) -> float:
        """Share of the enclosing shape that is actually feature.

        This is the measure that says how closely a boundary hugs what it
        surrounds, and it is not the same question as how much smaller that
        boundary is than the axis-aligned box.  Comparing two shapes around one
        feature, the ratio of their tightness is just the ratio of their areas,
        so "improved on the box" carries no information the areas did not.  As
        an absolute bar it does: a boundary enclosing mostly nothing describes
        scattered speckle, whatever shape it is.

        Slightly over 1 is normal and not an error -- the hull is a polygon
        through pixel centres while the feature is counted in whole pixels.
        """
        area = self.hull_area if shape == SHAPE_HULL else self.rectangle_area
        return self.area_px / max(area, 1e-6)

    def outline(self, shape: str) -> tuple[tuple[float, float], ...]:
        if shape == SHAPE_HULL:
            return self.hull
        return tuple(
            (float(px), float(py)) for px, py in cv2.boxPoints(self.rectangle)
        )


@dataclass(frozen=True, slots=True)
class RegionMagicFeature:
    """Magic-wand feature mask extracted around one detected region."""

    bounds: tuple[int, int, int, int]
    inner: tuple[int, int, int, int]
    feature: np.ndarray
    tolerance: int


@dataclass(frozen=True, slots=True)
class RegionFeatureHull:
    """Convex hull of a magic-wand feature, in absolute texture coordinates."""

    points: tuple[tuple[int, int], ...]
    feature_area_px: int
    hull_area_px: float
    colour_variation: float


@dataclass(frozen=True, slots=True)
class FeatureExtensionMeasure:
    """Connected feature continuation measured around a detected region."""

    feature_area_px: int
    extension_area_px: int

    @property
    def extension_ratio(self) -> float:
        return float(self.extension_area_px) / max(float(self.feature_area_px), 1.0)


@dataclass(frozen=True, slots=True)
class EdgeAlignment:
    """A feature rectangle whose side is parallel to a nearby UV edge."""

    outline: tuple[tuple[float, float], ...]
    edge_angle_degrees: float
    feature_angle_degrees: float
    angle_delta_degrees: float
    edge_stability_degrees: float
    distance_px: float
    edge_points: int


def feature_shape(
    mask: np.ndarray,
    bounds: tuple[int, int, int, int],
    config: MserConfig,
) -> FeatureShape | None:
    """Fit a convex hull and its minimum-area rectangle to a region's feature.

    The hull is computed first and the rectangle derived from it, which is both
    exact and nearly free: rotating calipers over the hull's handful of points
    gives a rectangle identical to the one fitted to every feature texel --
    measured across 60 regions, the areas never differed at all -- and the hull
    is where the work was anyway.

    The hull is the tighter of the two boundaries by about a fifth, so it is
    the better description of where a mark actually is.  It is not the cheaper
    one, which is worth saying because it sounds as though it should be:
    ``minAreaRect`` over the raw points is a single optimised call at 5.5 ms
    across those regions, while taking the hull and measuring its area costs
    5.9 ms in two.  Doing both comes to 6.5 ms, against 316 ms for the gradient
    threshold the mask came from, so the choice is about which shape describes
    the feature and not about cost.
    """
    x, y, w, h = bounds
    height, width = mask.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, width), min(y + h, height)
    if x1 <= x0 or y1 <= y0:
        return None
    points = cv2.findNonZero(mask[y0:y1, x0:x1].astype(np.uint8))
    if points is None or len(points) < max(config.rotated_bounds_min_points, 3):
        return None

    hull = cv2.convexHull(points)
    rectangle = cv2.minAreaRect(hull)
    (cx, cy), size, angle = rectangle
    return FeatureShape(
        area_px=float(len(points)),
        hull=tuple(
            (float(px) + x0, float(py) + y0) for px, py in hull.reshape(-1, 2)
        ),
        hull_area=float(cv2.contourArea(hull)),
        rectangle=((cx + x0, cy + y0), size, angle),
        rectangle_area=float(size[0] * size[1]),
    )


def _normalised_axis_angle_degrees(angle: float) -> float:
    """Map a line direction to [0, 180), because opposite edges are parallel."""
    return float(angle % 180.0)


def _parallel_angle_delta_degrees(first: float, second: float) -> float:
    """Smallest angle between two unoriented line directions."""
    delta = abs(
        _normalised_axis_angle_degrees(first)
        - _normalised_axis_angle_degrees(second)
    )
    return float(min(delta, 180.0 - delta))


def _rectangle_long_axis_angle_degrees(
    rectangle: tuple[tuple[float, float], tuple[float, float], float],
) -> float:
    """Return the direction of a min-area rectangle's long side."""
    _centre, (rw, rh), angle = rectangle
    if rh > rw:
        angle += 90.0
    return _normalised_axis_angle_degrees(angle)


def _feature_points(
    mask: np.ndarray,
    bounds: tuple[int, int, int, int],
    config: MserConfig,
) -> np.ndarray | None:
    """Feature texel centres inside one clamped region, in absolute pixels."""
    x, y, w, h = bounds
    height, width = mask.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, width), min(y + h, height)
    if x1 <= x0 or y1 <= y0:
        return None
    points = cv2.findNonZero(mask[y0:y1, x0:x1].astype(np.uint8))
    if points is None or len(points) < max(config.rotated_bounds_min_points, 3):
        return None
    absolute = points.reshape(-1, 2).astype(np.float32)
    absolute[:, 0] += x0
    absolute[:, 1] += y0
    return absolute


def _oriented_rectangle_outline(
    points: np.ndarray,
    angle_degrees: float,
) -> tuple[tuple[float, float], ...]:
    """Axis-aligned bounds in a rotated basis, returned as image-space corners."""
    radians = math.radians(angle_degrees)
    along = np.asarray((math.cos(radians), math.sin(radians)), dtype=np.float32)
    across = np.asarray((-math.sin(radians), math.cos(radians)), dtype=np.float32)
    projected_along = points @ along
    projected_across = points @ across
    min_along, max_along = float(projected_along.min()), float(projected_along.max())
    min_across, max_across = float(projected_across.min()), float(projected_across.max())
    corners = (
        along * min_along + across * min_across,
        along * max_along + across * min_across,
        along * max_along + across * max_across,
        along * min_along + across * max_across,
    )
    return tuple((float(point[0]), float(point[1])) for point in corners)


def edge_aligned_feature_outline(
    mask: np.ndarray,
    uv_mask: np.ndarray | None,
    bounds: tuple[int, int, int, int],
    shape: FeatureShape,
    config: MserConfig,
    uv_boundary: np.ndarray | None = None,
    uv_contours: tuple[np.ndarray, ...] | None = None,
) -> EdgeAlignment | None:
    """Return a rotated feature outline whose side follows a nearby UV edge.

    Proximity is measured from the feature pixels, not from the box centre.  A
    long word beside a trim edge can be many pixels from that edge at its
    centre while one long side is still genuinely close.  The UV-edge angle is
    fitted from a short contiguous arc around the single closest contour point.
    The glyph does not contribute an angle, so curved lettering and asymmetric
    logos cannot tilt a rectangle away from the surface direction.
    """
    if uv_mask is None:
        return None
    feature_points = _feature_points(mask, bounds, config)
    if feature_points is None:
        return None

    search = max(int(config.rotation_edge_search_px), 0)
    band = max(int(config.rotation_edge_band_px), 0)
    x, y, w, h = expanded_box(bounds, search)
    clamped = clamp_group((x, y, w, h), uv_mask.shape)
    if clamped is None:
        return None
    sx, sy, sw, sh = clamped

    if uv_boundary is None:
        domain = uv_mask.astype(np.uint8)
        uv_boundary = cv2.morphologyEx(
            domain,
            cv2.MORPH_GRADIENT,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        ).astype(bool)
    boundary = uv_boundary
    boundary_crop = boundary[sy : sy + sh, sx : sx + sw]
    if not bool(boundary_crop.any()):
        return None

    feature_crop = np.zeros((sh, sw), dtype=bool)
    local_x = np.rint(feature_points[:, 0] - sx).astype(np.int32)
    local_y = np.rint(feature_points[:, 1] - sy).astype(np.int32)
    inside = (local_x >= 0) & (local_x < sw) & (local_y >= 0) & (local_y < sh)
    if not bool(inside.any()):
        return None
    feature_crop[local_y[inside], local_x[inside]] = True

    distance_to_feature = cv2.distanceTransform(
        (~feature_crop).astype(np.uint8), cv2.DIST_L2, 3
    )
    if uv_contours is None:
        found, _hierarchy = cv2.findContours(
            uv_mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE,
        )
        uv_contours = tuple(found)

    nearest: tuple[float, np.ndarray, int] | None = None
    for contour in uv_contours:
        points = contour.reshape(-1, 2)
        if len(points) < 2:
            continue
        inside_search = (
            (points[:, 0] >= sx)
            & (points[:, 0] < sx + sw)
            & (points[:, 1] >= sy)
            & (points[:, 1] < sy + sh)
        )
        indices = np.flatnonzero(inside_search)
        if not len(indices):
            continue
        local = points[indices] - np.asarray((sx, sy), dtype=np.int32)
        distances = distance_to_feature[local[:, 1], local[:, 0]]
        local_index = int(np.argmin(distances))
        candidate = (float(distances[local_index]), points, int(indices[local_index]))
        if nearest is None or candidate[0] < nearest[0]:
            nearest = candidate
    if nearest is None:
        return None
    closest, contour_points, closest_index = nearest
    minimum_gap = max(float(config.rotation_edge_min_gap_px), 0.0)
    if closest < minimum_gap:
        return None
    if closest > search:
        return None

    minimum = max(int(config.rotation_edge_min_points), 2)
    half_arc = max(minimum // 2, band * 2, 2)

    def contour_arc(half_width: int) -> np.ndarray:
        if len(contour_points) <= half_width * 2 + 1:
            return contour_points.astype(np.float32)
        offsets = np.arange(-half_width, half_width + 1, dtype=np.int32)
        return contour_points[(closest_index + offsets) % len(contour_points)].astype(
            np.float32
        )

    tangent_points = contour_arc(half_arc)
    if len(tangent_points) < minimum:
        return None
    vx, vy, _px, _py = cv2.fitLine(
        tangent_points, cv2.DIST_L2, 0, 0.01, 0.01
    ).reshape(4)
    edge_angle = _normalised_axis_angle_degrees(
        math.degrees(math.atan2(float(vy), float(vx)))
    )

    wider_points = contour_arc(half_arc * 2)
    wide_vx, wide_vy, _wide_px, _wide_py = cv2.fitLine(
        wider_points, cv2.DIST_L2, 0, 0.01, 0.01
    ).reshape(4)
    wider_angle = _normalised_axis_angle_degrees(
        math.degrees(math.atan2(float(wide_vy), float(wide_vx)))
    )
    edge_stability = _parallel_angle_delta_degrees(edge_angle, wider_angle)
    if edge_stability > float(config.max_rotation_edge_angle_degrees):
        return None

    # Keep the broader nearby-boundary set only for the two-sided-strip guard;
    # it no longer participates in the angle fit.
    nearby_edge = boundary_crop & (distance_to_feature <= closest + band)
    support_points = cv2.findNonZero(nearby_edge.astype(np.uint8))
    if support_points is None or len(support_points) < minimum:
        return None
    feature_angle = _rectangle_long_axis_angle_degrees(shape.rectangle)
    delta = _parallel_angle_delta_degrees(edge_angle, feature_angle)

    radians = math.radians(edge_angle)
    along = np.asarray((math.cos(radians), math.sin(radians)), dtype=np.float32)
    across = np.asarray((-math.sin(radians), math.cos(radians)), dtype=np.float32)
    feature_along = feature_points @ along
    feature_across = feature_points @ across
    min_along, max_along = float(feature_along.min()), float(feature_along.max())
    min_across, max_across = float(feature_across.min()), float(feature_across.max())

    edge_absolute = support_points.reshape(-1, 2).astype(np.float32)
    edge_absolute[:, 0] += sx
    edge_absolute[:, 1] += sy
    edge_along = edge_absolute @ along
    edge_across = edge_absolute @ across
    overlaps_side = (
        (edge_along >= min_along - band)
        & (edge_along <= max_along + band)
    )
    min_side_distances = np.abs(edge_across - min_across)
    max_side_distances = np.abs(edge_across - max_across)
    eligible_edge = overlaps_side & (
        np.minimum(min_side_distances, max_side_distances) >= minimum_gap
    ) & (
        np.minimum(min_side_distances, max_side_distances) <= search
    )
    min_side_candidates = (
        eligible_edge
        & (min_side_distances <= max_side_distances)
    )
    max_side_candidates = (
        eligible_edge
        & (max_side_distances < min_side_distances)
    )
    min_support = int(min_side_candidates.sum())
    max_support = int(max_side_candidates.sum())
    primary_support = max(min_support, max_support)
    opposite_support = min(min_support, max_support)
    if primary_support < minimum:
        return None
    opposite_fraction = opposite_support / max(float(primary_support), 1.0)
    if opposite_fraction > float(config.max_opposite_rotation_edge_fraction):
        return None
    side_candidates = (
        min_side_candidates if min_support >= max_support else max_side_candidates
    )
    side_distances = (
        min_side_distances if min_support >= max_support else max_side_distances
    )

    outline = (
        (along * min_along + across * min_across),
        (along * max_along + across * min_across),
        (along * max_along + across * max_across),
        (along * min_along + across * max_across),
    )

    return EdgeAlignment(
        outline=tuple(
            (float(point[0]), float(point[1]))
            for point in outline
        ),
        edge_angle_degrees=edge_angle,
        feature_angle_degrees=feature_angle,
        angle_delta_degrees=delta,
        edge_stability_degrees=edge_stability,
        distance_px=float(side_distances[side_candidates].min()),
        edge_points=int(side_candidates.sum()),
    )


def rotated_feature_bounds(
    mask: np.ndarray,
    bounds: tuple[int, int, int, int],
    config: MserConfig,
) -> tuple[tuple[float, float], tuple[float, float], float] | None:
    """Smallest rotated rectangle around the feature inside a region.

    Rotating calipers over the convex hull, which is not the cost: measured on
    a 4096 atlas the median region takes 18 microseconds and all of them
    together 8 ms, against 79 ms for one gradient pass.  What costs is walking
    the crop for its set pixels, so it scales with the region's area rather
    than with how complicated its shape is.

    Worth having because an axis-aligned box measures a tilted mark badly: the
    box of a diagonal streak is mostly background, which flatters its aspect
    ratio and dilutes anything measured over its interior.  The rotated
    rectangle gives the mark's true proportions and how much of its own box it
    really occupies.

    ``None`` when there are too few feature pixels to fit anything to.
    """
    x, y, w, h = bounds
    height, width = mask.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, width), min(y + h, height)
    if x1 <= x0 or y1 <= y0:
        return None
    points = cv2.findNonZero(mask[y0:y1, x0:x1].astype(np.uint8))
    if points is None or len(points) < max(config.rotated_bounds_min_points, 3):
        return None
    (cx, cy), (rw, rh), angle = cv2.minAreaRect(points)
    return (cx + x0, cy + y0), (rw, rh), angle


def rotated_bounds_measures(
    rectangle: tuple[tuple[float, float], tuple[float, float], float],
    bounds: tuple[int, int, int, int],
) -> tuple[float, float]:
    """Return (fill, aspect) for a rotated rectangle against its region.

    ``fill`` is the rotated rectangle's area over the axis-aligned box's, so a
    mark that sits square in its box approaches 1 while a thin diagonal streak
    -- a seam fragment, a bevel edge -- collapses towards 0.  ``aspect`` is the
    true long-over-short, which for anything tilted the axis-aligned box gets
    wrong.
    """
    _centre, (rw, rh), _angle = rectangle
    _x, _y, w, h = bounds
    fill = (rw * rh) / max(float(w * h), 1.0)
    aspect = max(rw, rh) / max(min(rw, rh), 1.0)
    return fill, aspect


def text_line_measures(
    mask: np.ndarray,
    bounds: tuple[int, int, int, int],
    config: MserConfig,
) -> tuple[int, float] | None:
    """How many separate marks a region holds, and how well they line up.

    This is the question the other filters do not answer.  They separate a mark
    from the material around it; none of them separates a word from a symbol,
    and a horn or a seat pictogram passes every one of them because it is a
    real mark on plain trim -- it simply is not text.

    What makes text text is that it is several similar marks in a row.  So the
    region's feature is split into components and two things measured: how many
    there are, and how close their centres lie to a single straight line, as
    the smaller singular value of the centred centres over the larger.  Zero is
    a perfect line.

    Measured on the ardente: AIRBAG gives 6 components scattering 0.019 and
    ARDENTE 14 scattering 0.072, while every horn and seat icon gives one
    component and the non-text regions that do have several scatter 0.18 to
    0.56.

    Returns ``None`` when there are too few components to say anything -- two
    points lie on a perfect line by definition, so scatter is meaningless
    below three and would wave every two-part icon through.
    """
    x, y, w, h = bounds
    height, width = mask.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, width), min(y + h, height)
    if x1 <= x0 or y1 <= y0:
        return None

    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask[y0:y1, x0:x1].astype(np.uint8), connectivity=8
    )
    centres = np.asarray(
        [
            centroids[index]
            for index in range(1, count)
            if stats[index, cv2.CC_STAT_AREA] >= max(config.text_min_component_px, 1)
        ],
        dtype=float,
    )
    if len(centres) < 3:
        return (len(centres), float("inf")) if len(centres) else None

    centred = centres - centres.mean(axis=0)
    singular = np.linalg.svd(centred, compute_uv=False)
    scatter = float(singular[1] / max(singular[0], 1e-6))
    return len(centres), scatter


def region_flatness(
    response: np.ndarray,
    uv_mask: np.ndarray | None,
    bounds: tuple[int, int, int, int],
    config: MserConfig,
    reference: float,
) -> float | None:
    """How much relief a region actually contains.

    The companion to ``ring_roughness``, and it catches the opposite mistake.
    The ring test asks whether a region is isolated, which rejects a patch of
    grille because the grille goes on around it; it has nothing to say about a
    region that is isolated and also empty.  Those arrive in numbers -- the
    soft roll of a bolster, the curve of a fascia, anywhere the surface bends
    enough to raise an edge but carries no mark -- and to the ring test they
    look exactly like lettering on plain trim.

    Read as a high percentile rather than a mean, because a small glyph in a
    generous box is mostly background: what matters is whether anything in
    there stands up, not how much of the box it fills.

    Expressed against the atlas's typical gradient, so it is a multiple and
    travels between vehicles.  ``None`` when too little of the region is inside
    the UV domain to judge.
    """
    x, y, w, h = bounds
    height, width = response.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, width), min(y + h, height)
    if x1 <= x0 or y1 <= y0:
        return None

    window = response[y0:y1, x0:x1]
    if uv_mask is not None:
        inside = uv_mask[y0:y1, x0:x1]
        if int(inside.sum()) < max(config.region_flatness_min_domain_px, 1):
            return None
        values = window[inside]
    else:
        if window.size < max(config.region_flatness_min_domain_px, 1):
            return None
        values = window.ravel()

    level = float(
        np.percentile(values, min(max(config.region_flatness_percentile, 0.0), 100.0))
    )
    return level / max(reference, 1e-6)


def ring_roughness(
    response: np.ndarray,
    uv_mask: np.ndarray | None,
    bounds: tuple[int, int, int, int],
    config: MserConfig,
    reference: float,
    feature_hull: RegionFeatureHull | None = None,
) -> float | None:
    """How busy the material immediately around a region is.

    A moulded mark is put on trim that is otherwise plain: lettering on a
    fascia, a symbol on a switch face.  The things that keep being mistaken for
    one are not -- a perforated speaker grille, a woven insert, carpet -- and
    what separates them is not the region at all but what surrounds it.  Inside
    its own bounds a patch of grille looks much like a glyph: strong, closed,
    stroke-like edges.  One ring further out, the grille carries on and the
    fascia does not.

    Measured as the ring's typical gradient over the atlas's typical gradient,
    so it is a multiple rather than a level and does not need retuning per
    vehicle.  Roughly 1 is ordinary material; a grille runs several times that.

    A gap is left between the region and the ring, or the mark's own outer edge
    is counted as surrounding busyness and every real mark looks rough.

    ``None`` when too little of the ring is inside the UV domain to judge --
    a region at an island edge is measuring the void, not the material.
    """
    ring_result = feature_hull_ring(
        response.shape, feature_hull, config
    ) if feature_hull is not None else None
    if ring_result is None:
        ring_result = rectangular_region_ring(response.shape, bounds, config)
    if ring_result is None:
        return None
    (ox0, oy0, ox1, oy1), ring = ring_result

    if uv_mask is not None:
        ring &= uv_mask[oy0:oy1, ox0:ox1]
    if int(ring.sum()) < max(config.ring_smoothness_min_domain_px, 1):
        return None

    values = response[oy0:oy1, ox0:ox1][ring]
    busy = float(
        np.percentile(values, min(max(config.ring_smoothness_percentile, 0.0), 100.0))
    )
    return busy / max(reference, 1e-6)


def rectangular_region_ring(
    image_shape: tuple[int, ...],
    bounds: tuple[int, int, int, int],
    config: MserConfig,
) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
    """Return the legacy rectangular offset ring for a region."""
    x, y, w, h = bounds
    height, width = image_shape[:2]
    margin = max(int(config.ring_smoothness_margin_px), 0)
    band = max(int(config.ring_smoothness_width_px), 1)
    outer = margin + band

    ox0, oy0 = max(x - outer, 0), max(y - outer, 0)
    ox1, oy1 = min(x + w + outer, width), min(y + h + outer, height)
    if ox1 <= ox0 or oy1 <= oy0:
        return None

    ring = np.ones((oy1 - oy0, ox1 - ox0), dtype=bool)
    ring[
        max(y - margin, 0) - oy0 : min(y + h + margin, height) - oy0,
        max(x - margin, 0) - ox0 : min(x + w + margin, width) - ox0,
    ] = False
    return (ox0, oy0, ox1, oy1), ring


def feature_hull_ring(
    image_shape: tuple[int, ...],
    hull: RegionFeatureHull | None,
    config: MserConfig,
) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
    """Return an offset ring around a feature's convex hull."""
    if hull is None or len(hull.points) < 3:
        return None
    height, width = image_shape[:2]
    margin = max(int(config.ring_smoothness_margin_px), 0)
    band = max(int(config.ring_smoothness_width_px), 1)
    outer = margin + band
    points = np.asarray(hull.points, dtype=np.int32)
    x0 = max(int(points[:, 0].min()) - outer, 0)
    y0 = max(int(points[:, 1].min()) - outer, 0)
    x1 = min(int(points[:, 0].max()) + outer + 1, width)
    y1 = min(int(points[:, 1].max()) + outer + 1, height)
    if x1 <= x0 or y1 <= y0:
        return None

    local = points - np.asarray((x0, y0), dtype=np.int32)
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.fillConvexPoly(mask, local, 255)

    outer_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (outer * 2 + 1, outer * 2 + 1)
    )
    outer_mask = cv2.dilate(mask, outer_kernel) > 0
    if margin:
        inner_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (margin * 2 + 1, margin * 2 + 1)
        )
        inner_mask = cv2.dilate(mask, inner_kernel) > 0
    else:
        inner_mask = mask > 0
    return (x0, y0, x1, y1), outer_mask & ~inner_mask


def stroke_width_transform(
    grey: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig,
) -> np.ndarray:
    """Per-texel stroke width, after Epshtein et al.

    From every edge texel a ray is cast along the gradient.  If it meets
    another edge texel whose gradient is roughly opposite, the two are the
    facing sides of one stroke and the distance between them is its width;
    every texel on the ray is labelled with it.  What makes this worth having
    is not the width itself but its *consistency*: a letter is drawn with a
    pen of one thickness, and a panel seam, a bevel or a run of stitching is
    not, however strong an edge it makes.  That is the discriminator the
    gradient alone never had.

    Expects a height-like image, where a raised stroke is a band brighter than
    the material either side of it.  On a shaded render the middle of a stroke
    sits back at the flat level, so a stroke shows as four edges rather than
    two and the ray pairing means nothing.  Set the relief mode to height or
    tophat before turning this on.

    Returns a float32 map, zero where no valid stroke was found.

    Marched a step at a time across all rays at once rather than ray by ray:
    on a 4k atlas there are hundreds of thousands of edge texels, and the
    per-ray loop the paper describes is far too slow in Python.  The paper's
    second pass, which resets each ray to its own median, is approximated by a
    small median filter over the map; that costs exact per-ray widths, which
    only matter for grouping letters, and this is used for statistics per box.
    """
    mask, response = edge_mask(grey, uv_mask, config)
    height, width = grey.shape[:2]
    widths = np.zeros((height, width), dtype=np.float32)
    if not bool(mask.any()):
        return widths

    source = grey.astype(np.float32)
    if config.edge_blur_sigma > 0:
        source = cv2.GaussianBlur(source, (0, 0), config.edge_blur_sigma)
    dx = cv2.Scharr(source, cv2.CV_32F, 1, 0)
    dy = cv2.Scharr(source, cv2.CV_32F, 0, 1)
    magnitude = np.maximum(np.hypot(dx, dy), 1e-6)
    gx, gy = dx / magnitude, dy / magnitude

    rows, columns = np.nonzero(mask)
    if rows.size == 0:
        return widths
    # A raised stroke is brighter than its surround, so the gradient at both of
    # its edges points inwards and a ray along +g reaches the far side.  An
    # engraved one is the same picture upside down.
    directions = []
    if config.swt_polarity in ("raised", "both"):
        directions.append(1.0)
    if config.swt_polarity in ("engraved", "both"):
        directions.append(-1.0)
    tolerance = float(np.cos(np.radians(config.swt_gradient_tolerance_degrees)))
    maximum = max(int(config.swt_max_stroke_px), 1)
    minimum = max(int(config.swt_min_stroke_px), 1)

    for sign in directions:
        start_x = columns.astype(np.float32)
        start_y = rows.astype(np.float32)
        ray_x = gx[rows, columns] * sign
        ray_y = gy[rows, columns] * sign
        found = np.zeros(rows.size, dtype=np.float32)
        live = np.ones(rows.size, dtype=bool)
        for step in range(1, maximum + 1):
            if not bool(live.any()):
                break
            px = np.rint(start_x + ray_x * step).astype(np.int32)
            py = np.rint(start_y + ray_y * step).astype(np.int32)
            inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
            live &= inside
            if not bool(live.any()):
                break
            hit = np.zeros(rows.size, dtype=bool)
            index = np.nonzero(live)[0]
            hit[index] = mask[py[index], px[index]]
            candidate = np.nonzero(live & hit)[0]
            if candidate.size:
                # Opposing means the far edge faces back the way we came.
                opposite = (
                    ray_x[candidate] * gx[py[candidate], px[candidate]]
                    + ray_y[candidate] * gy[py[candidate], px[candidate]]
                ) <= -tolerance
                settled = candidate[opposite]
                if settled.size:
                    found[settled] = step
                    live[settled] = False
                # An edge that faces the wrong way is not the other side of a
                # stroke; the ray carries on through it.

        valid = np.nonzero((found >= minimum) & (found <= maximum))[0]
        if valid.size == 0:
            continue
        # Paint the width along each accepted ray, keeping the narrowest claim
        # where rays cross, as the paper does.
        for step in range(0, maximum + 1):
            active = valid[found[valid] >= step]
            if active.size == 0:
                break
            px = np.clip(
                np.rint(start_x[active] + ray_x[active] * step).astype(np.int32),
                0, width - 1,
            )
            py = np.clip(
                np.rint(start_y[active] + ray_y[active] * step).astype(np.int32),
                0, height - 1,
            )
            current = widths[py, px]
            claim = found[active]
            narrower = (current == 0) | (claim < current)
            widths[py[narrower], px[narrower]] = claim[narrower]

    if config.swt_median_px > 1:
        size = int(config.swt_median_px) | 1
        smoothed = cv2.medianBlur(widths, size)
        widths = np.where(widths > 0, smoothed, widths)
    return widths


def stroke_width_stats(
    widths: np.ndarray,
    bounds: tuple[int, int, int, int],
) -> tuple[float, float, float]:
    """Return (median width, variation, coverage) for one rectangle.

    Variation is the standard deviation over the mean, which is the number the
    text/not-text decision actually rests on: a stroke drawn with one pen is
    consistent whatever its absolute width, so a ratio travels between fonts,
    resolutions and vehicles where an absolute tolerance would not.
    """
    x, y, w, h = bounds
    height, width = widths.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, width), min(y + h, height)
    if x1 <= x0 or y1 <= y0:
        return 0.0, float("inf"), 0.0
    window = widths[y0:y1, x0:x1]
    marked = window[window > 0]
    area = max((x1 - x0) * (y1 - y0), 1)
    if marked.size == 0:
        return 0.0, float("inf"), 0.0
    mean = float(marked.mean())
    variation = float(marked.std() / mean) if mean > 1e-6 else float("inf")
    return float(np.median(marked)), variation, marked.size / area


def context_box(
    box: tuple[int, int, int, int],
    ring: int,
) -> tuple[int, int, int, int]:
    """Grow a box by a fixed ring of pixels on every side.

    Deliberately additive rather than a multiple of the box: scaling a long or
    tall box reaches far enough to swallow whole neighbouring regions, so a box
    sitting in flat black beside a light panel measured as half-and-half and
    never registered as flat.  A small ring stays within the region the box is
    actually in, while still bringing in the background around a glyph stroke.
    """
    x, y, w, h = box
    if ring <= 0:
        return box
    return (x - ring, y - ring, w + ring * 2, h + ring * 2)


def dominant_island_component(
    domain: np.ndarray,
    inner: tuple[int, int, int, int],
) -> np.ndarray:
    """Keep only the UV island the box itself sits on.

    Islands can be separated by a gap of two or three pixels, so a context ring
    hops straight into the neighbouring island and mixes in its colour: a box on
    a flat grey panel next to a flat black one measured as two-toned and escaped
    the flat filter.  ``inner`` is the box within the window, used to pick which
    connected component of the domain to keep.
    """
    count, labels = cv2.connectedComponents(domain.astype(np.uint8), connectivity=8)
    if count <= 2:  # background plus at most one island
        return domain
    x, y, w, h = inner
    within = labels[y : y + h, x : x + w]
    within = within[within > 0]
    if within.size == 0:
        return domain
    return labels == int(np.bincount(within).argmax())


def channel_colour_mask(
    crop: np.ndarray,
    domain_mask: np.ndarray,
    colour: np.ndarray,
    tolerance: int,
) -> np.ndarray:
    """Return domain pixels within ``tolerance`` of a colour on every channel.

    Per-channel rather than Euclidean, to stay consistent with the quantised
    bins ``estimate_background_colour`` uses to pick that colour: a bin of width
    t spans up to t*sqrt(3) in Euclidean distance, so an Euclidean test with the
    same number splits colours the binning already called identical.  Uniform
    dark trim carrying +/-20 of compression dither reads as 75% flat under the
    Euclidean test and 100% under this one.
    """
    delta = np.abs(crop.astype(np.int16) - colour.astype(np.int16))
    return (delta.max(axis=2) <= max(tolerance, 0)) & domain_mask


def box_uv_coverage(
    uv_mask: np.ndarray | None,
    box: tuple[int, int, int, int],
    image_shape: tuple[int, ...],
    domain: UvDomainIndex | None = None,
) -> float:
    """Return the share of a box covered by its single largest UV island."""
    if uv_mask is None:
        return 1.0
    clamped = clamp_group(box, image_shape)
    if clamped is None:
        return 0.0
    x, y, w, h = clamped
    _bx, _by, bw, bh = box
    area = max(bw * bh, 1)
    if domain is not None:
        labels = domain.island_labels
    else:
        _count, labels = cv2.connectedComponents(
            uv_mask.astype(np.uint8), connectivity=8
        )
    within = labels[y : y + h, x : x + w]
    within = within[within > 0]
    if within.size == 0:
        return 0.0
    return float(np.bincount(within).max()) / float(area)


def box_feature_blob_px(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    box: tuple[int, int, int, int],
    config: MserConfig,
) -> int | None:
    """Return the area of a box's largest connected non-background run.

    Speckle noise on blank trim breaks into tiny fragments, while a glyph
    stroke stays whole, so this one number separates the two.  The share of the
    box that is a single colour was measured alongside it for a long time and
    dropped: across eight 4k dashboards it rejected between zero and six boxes,
    none of which survived to affect a final region.

    Uses the magic wand's colour model: the dominant colour is estimated inside
    the UV domain, and anything further than ``box_feature_colour_tolerance``
    from it on any channel counts as feature.  ``None`` means the box has too
    little UV domain to judge.

    The test covers the box plus ``box_feature_context_px``, because an MSER
    region is homogeneous by definition -- measured on the bare box, a glyph
    stroke has no background to stand out against.  With a ring of context, the
    stroke's surroundings come in and the stroke becomes the feature.
    """
    inner = box
    box = context_box(box, config.box_feature_context_px)
    clamped = clamp_group(box, image.shape)
    if clamped is None:
        return None
    x, y, w, h = clamped
    crop = image[y : y + h, x : x + w]
    domain = (
        uv_mask[y : y + h, x : x + w]
        if uv_mask is not None
        else np.ones((h, w), dtype=bool)
    )
    if uv_mask is not None:
        domain = dominant_island_component(
            domain,
            (
                max(inner[0] - x, 0),
                max(inner[1] - y, 0),
                max(min(inner[2], w), 1),
                max(min(inner[3], h), 1),
            ),
        )
    if int(domain.sum()) < max(config.box_feature_min_domain_px, 1):
        return None

    tolerance = max(config.box_feature_colour_tolerance, 0)
    dominant = estimate_background_colour(crop, domain, max(tolerance, 1))
    if dominant is None:
        return None

    feature = domain & ~channel_colour_mask(crop, domain, dominant, tolerance)
    if not bool(feature.any()):
        return 0
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        feature.astype(np.uint8), connectivity=8
    )
    return max(
        (int(stats[label, cv2.CC_STAT_AREA]) for label in range(1, count)),
        default=0,
    )


def _local_inner_rect(
    outer: tuple[int, int, int, int],
    inner: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    ox, oy, ow, oh = outer
    x, y, w, h = inner
    x0 = max(x - ox, 0)
    y0 = max(y - oy, 0)
    x1 = min(x + w - ox, ow)
    y1 = min(y + h - oy, oh)
    return (x0, y0, max(x1 - x0, 0), max(y1 - y0, 0))


def _region_background_ring(
    image_shape: tuple[int, ...],
    group: tuple[int, int, int, int],
    config: MserConfig,
) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
    x, y, w, h = group
    margin = max(int(config.region_feature_ring_margin_px), 0)
    band = max(int(config.region_feature_ring_width_px), 1)
    outer = margin + band
    clamped = clamp_group(context_box(group, outer), image_shape)
    if clamped is None:
        return None
    ox, oy, ow, oh = clamped
    ring = np.ones((oh, ow), dtype=bool)
    ring[
        max(y - margin, 0) - oy : min(y + h + margin, image_shape[0]) - oy,
        max(x - margin, 0) - ox : min(x + w + margin, image_shape[1]) - ox,
    ] = False
    return clamped, ring


def region_magic_feature(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    group: tuple[int, int, int, int],
    config: MserConfig,
    *,
    context_px: int = 0,
) -> RegionMagicFeature | None:
    """Extract non-background pixels using a background ring around a region."""
    ring_result = _region_background_ring(image.shape, group, config)
    if ring_result is None:
        return None
    ring_bounds, ring = ring_result
    rx, ry, rw, rh = ring_bounds
    ring_crop = image[ry : ry + rh, rx : rx + rw]
    ring_domain = (
        uv_mask[ry : ry + rh, rx : rx + rw]
        if uv_mask is not None
        else np.ones((rh, rw), dtype=bool)
    )
    if uv_mask is not None:
        ring_domain = dominant_island_component(
            ring_domain, _local_inner_rect(ring_bounds, group)
        )
    ring &= ring_domain
    if int(ring.sum()) < max(config.region_feature_min_domain_px, 1):
        return None

    base_tolerance = max(int(config.region_feature_colour_tolerance), 0)
    background = estimate_background_colour(ring_crop, ring, max(base_tolerance, 1))
    if background is None:
        return None
    ring_pixels = ring_crop[ring].astype(np.float32)
    variance_tolerance = int(
        math.ceil(float(ring_pixels.std(axis=0).max()) * config.region_feature_variance_scale)
    )
    tolerance = max(base_tolerance, variance_tolerance)

    feature_bounds = clamp_group(
        context_box(group, max(int(context_px), 0)), image.shape
    )
    if feature_bounds is None:
        return None
    fx, fy, fw, fh = feature_bounds
    crop = image[fy : fy + fh, fx : fx + fw]
    domain = (
        uv_mask[fy : fy + fh, fx : fx + fw]
        if uv_mask is not None
        else np.ones((fh, fw), dtype=bool)
    )
    if uv_mask is not None:
        domain = dominant_island_component(
            domain, _local_inner_rect(feature_bounds, group)
        )
    feature = domain & ~channel_colour_mask(crop, domain, background, tolerance)
    return RegionMagicFeature(
        bounds=feature_bounds,
        inner=_local_inner_rect(feature_bounds, group),
        feature=feature,
        tolerance=tolerance,
    )


def blob_hull_fill(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    group: tuple[int, int, int, int],
    config: MserConfig,
) -> float | None:
    """Return feature area over convex-hull area for a detected region."""
    measures = blob_shape_measures(image, uv_mask, group, config)
    return measures[0] if measures is not None else None


def region_feature_hull(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    group: tuple[int, int, int, int],
    config: MserConfig,
) -> RegionFeatureHull | None:
    """Return the convex hull and colour spread of a region's extracted feature."""
    extracted = region_magic_feature(image, uv_mask, group, config)
    if extracted is None:
        return None
    x, y, w, h = extracted.inner
    if w <= 0 or h <= 0:
        return None
    feature = extracted.feature[y : y + h, x : x + w]
    points = cv2.findNonZero(feature.astype(np.uint8))
    if points is None or len(points) < max(config.region_feature_min_px, 3):
        return None
    hull = cv2.convexHull(points)
    feature_pixels = image[
        extracted.bounds[1] + y : extracted.bounds[1] + y + h,
        extracted.bounds[0] + x : extracted.bounds[0] + x + w,
    ][feature]
    if feature_pixels.size == 0:
        return None
    pixels = feature_pixels.astype(np.float32)
    lo = np.percentile(pixels, 5.0, axis=0)
    hi = np.percentile(pixels, 95.0, axis=0)
    offset = np.asarray(
        (extracted.bounds[0] + x, extracted.bounds[1] + y),
        dtype=np.int32,
    )
    absolute = hull.reshape(-1, 2) + offset
    return RegionFeatureHull(
        points=tuple((int(px), int(py)) for px, py in absolute),
        feature_area_px=int(len(points)),
        hull_area_px=float(cv2.contourArea(hull)),
        colour_variation=float((hi - lo).max()),
    )


def blob_shape_measures(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    group: tuple[int, int, int, int],
    config: MserConfig,
) -> tuple[float, float] | None:
    """Return hull fill and internal colour variation for a blob candidate."""
    hull = region_feature_hull(image, uv_mask, group, config)
    if hull is None:
        return None
    return (
        float(hull.feature_area_px) / max(hull.hull_area_px, 1e-6),
        hull.colour_variation,
    )


def feature_extension_measure(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    group: tuple[int, int, int, int],
    config: MserConfig,
) -> FeatureExtensionMeasure | None:
    """Return connected feature area inside and outside the detected region."""
    extracted = region_magic_feature(
        image,
        uv_mask,
        group,
        config,
        context_px=config.feature_extension_context_px,
    )
    if extracted is None:
        return None
    x, y, w, h = extracted.inner
    if w <= 0 or h <= 0:
        return None
    inside = np.zeros(extracted.feature.shape, dtype=bool)
    inside[y : y + h, x : x + w] = True
    feature_area_px = int((extracted.feature & inside).sum())
    if feature_area_px < max(config.region_feature_min_px, 1):
        return None

    count, labels = cv2.connectedComponents(
        extracted.feature.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return FeatureExtensionMeasure(feature_area_px, 0)
    inner_labels = labels[inside & extracted.feature]
    inner_labels = inner_labels[inner_labels > 0]
    if inner_labels.size == 0:
        return None
    connected = np.isin(labels, np.unique(inner_labels))
    return FeatureExtensionMeasure(
        feature_area_px,
        int((connected & extracted.feature & ~inside).sum()),
    )


def feature_extension_px(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    group: tuple[int, int, int, int],
    config: MserConfig,
) -> int | None:
    """Return connected feature pixels that continue outside the region."""
    measure = feature_extension_measure(image, uv_mask, group, config)
    return measure.extension_area_px if measure is not None else None


def filter_boxes_by_feature(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    boxes: np.ndarray,
    config: MserConfig,
    domain: UvDomainIndex | None = None,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Drop undersized, featureless, or out-of-domain MSER boxes.

    The pre-MSER fill makes every island silhouette a hard edge, so MSER finds a
    few regions straddling the boundary.  Those carry no geometry over most of
    their area and would otherwise drag groups out across the gaps.
    """
    if not config.enable_box_feature_filter:
        # This checkbox owns the whole Box filtering stage.  Keeping minimum
        # dimensions active while it was off made a supposedly disabled stage
        # still reject small relief candidates.
        return np.asarray(boxes, dtype=np.int32).reshape(-1, 4).copy(), []
    if len(boxes) == 0:
        return boxes, []

    kept: list[tuple[int, int, int, int]] = []
    rejected: list[tuple[int, int, int, int]] = []
    for raw in boxes:
        box = (int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3]))
        if (
            box[2] < max(config.min_box_width_px, 1)
            or box[3] < max(config.min_box_height_px, 1)
        ):
            rejected.append(box)
            continue
        if (
            box_uv_coverage(uv_mask, box, image.shape, domain)
            < config.min_box_uv_coverage
        ):
            rejected.append(box)
            continue
        feature_px = box_feature_blob_px(image, uv_mask, box, config)
        if feature_px is None or feature_px < config.box_min_feature_px:
            rejected.append(box)
            continue
        kept.append(box)
    return (
        np.asarray(kept, dtype=np.int32) if kept else np.empty((0, 4), dtype=np.int32),
        rejected,
    )


def repeating_pattern_score(window: np.ndarray, config: MserConfig) -> float:
    """Return how strongly a patch revives its own correlation at a shift.

    Normalised autocorrelation through the FFT (Wiener-Khinchin), reduced to a
    profile over lag radius.  Non-repeating content decays away from zero lag
    and never recovers, so the statistic is the largest rise back above the
    running minimum: a genuine second peak, not just a high correlation.  The
    raw peak is useless here because it mostly measures smoothness -- blank trim
    scores higher than a woven panel.

    The measure only separates anything once it sees several periods, so it is
    applied to assembled groups and not to single MSER boxes.  Even then it is
    narrow: across eight 4k dashboards it changed the outcome on one.
    """
    patch = window.astype(np.float32)
    patch -= float(patch.mean())
    if float(np.abs(patch).max()) < 1e-3:
        return 0.0  # featureless: the flat-colour filter owns this case

    height, width = patch.shape
    max_lag = min(
        max(config.pattern_max_period_px, 1),
        (height - 1) // 2,
        (width - 1) // 2,
    )
    min_lag = max(config.pattern_min_period_px, 2)
    if max_lag < min_lag:
        return 0.0

    # Taper the edges so the patch border does not fake a periodic signal.
    patch = patch * np.hanning(height)[:, None] * np.hanning(width)[None, :]
    spectrum = np.fft.rfft2(patch, s=(height * 2, width * 2))
    correlation = np.fft.irfft2(np.abs(spectrum) ** 2, s=(height * 2, width * 2))
    peak = float(correlation[0, 0])
    if peak <= 0.0:
        return 0.0
    correlation /= peak

    # Non-negative row lags cover the plane by symmetry; keep both column signs
    # so diagonal weaves are caught whichever way they lean.
    rows = correlation[: max_lag + 1]
    region = np.concatenate([rows[:, : max_lag + 1], rows[:, -max_lag:]], axis=1)
    lag_y = np.arange(max_lag + 1, dtype=np.float32)[:, None]
    lag_x = np.concatenate(
        [
            np.arange(max_lag + 1, dtype=np.float32),
            np.arange(-max_lag, 0, dtype=np.float32),
        ]
    )[None, :]
    radius = np.sqrt(lag_y**2 + lag_x**2)

    best = 0.0
    running_minimum = 1.0
    for lag in range(1, max_lag + 1):
        ring = (radius >= lag - 0.5) & (radius < lag + 0.5)
        if not bool(ring.any()):
            continue
        value = float(region[ring].max())
        if lag >= min_lag:
            best = max(best, value - running_minimum)
        running_minimum = min(running_minimum, value)
    return best


def pattern_window_for_box(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    config: MserConfig,
) -> np.ndarray | None:
    """Return a square grey neighbourhood centred on a box.

    The side is the box's longer edge, so a wide, short group is scored over a
    tall square that reaches well past it.  That is deliberate for a weave,
    which continues beyond whatever slice of it MSER happened to group, but it
    does dilute the statistic for a long strip of text.
    """
    x, y, w, h = box
    target = max(
        int(round(max(w, h) * max(config.pattern_window_scale, 1.0))),
        max(config.pattern_min_window_px, 4),
    )
    centre_x = x + w // 2
    centre_y = y + h // 2
    half = target // 2
    clamped = clamp_group(
        (centre_x - half, centre_y - half, half * 2, half * 2), image.shape
    )
    if clamped is None:
        return None
    cx, cy, cw, ch = clamped
    if cw < 8 or ch < 8:
        return None
    return cv2.cvtColor(image[cy : cy + ch, cx : cx + cw], cv2.COLOR_BGR2GRAY)


def circle_uv_coverage(
    uv_mask: np.ndarray | None,
    centre: tuple[float, float],
    radius: float,
    domain: UvDomainIndex | None = None,
) -> float:
    """Return the share of a disc covered by its single largest UV island."""
    if uv_mask is None:
        return 1.0
    height, width = uv_mask.shape[:2]
    centre_x, centre_y = centre
    x0 = max(int(np.floor(centre_x - radius)), 0)
    y0 = max(int(np.floor(centre_y - radius)), 0)
    x1 = min(int(np.ceil(centre_x + radius)), width)
    y1 = min(int(np.ceil(centre_y + radius)), height)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    columns = np.arange(x0, x1, dtype=np.float32)[None, :] + 0.5
    rows = np.arange(y0, y1, dtype=np.float32)[:, None] + 0.5
    disc = ((columns - centre_x) ** 2 + (rows - centre_y) ** 2) <= radius**2
    area = int(disc.sum())
    if area == 0:
        return 0.0
    if domain is not None:
        labels = domain.island_labels
    else:
        _count, labels = cv2.connectedComponents(
            uv_mask.astype(np.uint8), connectivity=8
        )
    within = labels[y0:y1, x0:x1][disc]
    within = within[within > 0]
    if within.size == 0:
        return 0.0
    return float(np.bincount(within).max()) / float(area)


def inscribed_circle_radius(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    group: tuple[int, int, int, int],
    config: MserConfig,
) -> int | None:
    """Return an inscribed-circle radius when the corners it drops are empty.

    A round gauge or button fills a square box, but the four corners outside its
    inscribed circle carry nothing: either one locally solid colour, or no UV
    domain at all.  The corner colour does not need to match the dominant colour
    elsewhere in the group.  When that holds the circle is the honest region and
    the corners are dead area; when anything else lives there the square is kept.

    Domain recovery may use this circle in place of the rectangle.  It must
    therefore be a true inscribed circle: a padded circle can extend beyond its
    squarish rectangle and create the very UV-domain failure it is meant to
    avoid.
    """
    if not config.enable_circular_groups:
        return None

    x, y, w, h = group
    if w <= 0 or h <= 0:
        return None
    if min(w, h) / max(w, h) < config.circular_group_min_squareness:
        return None

    clamped = clamp_group(group, image.shape)
    if clamped is None:
        return None
    cx, cy, cw, ch = clamped
    crop = image[cy : cy + ch, cx : cx + cw]
    domain = (
        uv_mask[cy : cy + ch, cx : cx + cw]
        if uv_mask is not None
        else np.ones((ch, cw), dtype=bool)
    )

    radius = min(w, h) / 2.0
    # A circle is only worth having if it is the tighter region.
    if math.pi * radius**2 >= w * h:
        return None
    if not _circle_keeps_uv_coverage(uv_mask, group, radius, config):
        return None
    centre_x = (x + w / 2.0) - cx
    centre_y = (y + h / 2.0) - cy
    columns = np.arange(cw, dtype=np.float32)[None, :] + 0.5
    rows = np.arange(ch, dtype=np.float32)[:, None] + 0.5
    outside_circle = ((columns - centre_x) ** 2 + (rows - centre_y) ** 2) > radius**2

    # Corners with no UV domain are already dead; only judge domain pixels.
    corners = outside_circle & domain
    corner_area = int(corners.sum())
    if corner_area == 0:
        # Later coverage and output code consumes an integer radius.  Rounding
        # an odd 31 px box's 15.5 px inscribe up to 16 makes the represented
        # disc 32 px wide, so floor is required to keep it inside the box.
        return int(math.floor(radius))

    # Judge the discarded ring against its own dominant colour, not against the
    # primary background colour of the complete group.  This permits a uniformly
    # coloured bezel/ring even when the gauge face uses a different background.
    ring_colour = estimate_background_colour(
        crop, corners, config.circular_group_colour_tolerance
    )
    if ring_colour is None:
        return None
    matching = channel_colour_mask(
        crop, corners, ring_colour, config.circular_group_colour_tolerance
    )
    stray = float((corners & ~matching).sum()) / float(corner_area)
    if stray > config.circular_group_max_corner_content:
        return None
    return int(math.floor(radius))


def cached_inscribed_circle_radius(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    group: tuple[int, int, int, int],
    config: MserConfig,
    cache: dict[tuple[MserConfig, tuple[int, int, int, int]], int | None] | None,
) -> int | None:
    """Memoise the pure circle test for one detection run.

    Domain recovery revisits the same assembled bounds while it tries strict
    rebuilds.  The circle decision includes colour and UV work, so it is much
    more expensive than the dictionary lookup; the cached value is exact for a
    fixed image, UV mask and immutable configuration.
    """
    if cache is None:
        return inscribed_circle_radius(image, uv_mask, group, config)
    key = (config, group)
    if key not in cache:
        cache[key] = inscribed_circle_radius(image, uv_mask, group, config)
    return cache[key]


def _circle_keeps_uv_coverage(
    uv_mask: np.ndarray | None,
    group: tuple[int, int, int, int],
    radius: float,
    config: MserConfig,
    domain: UvDomainIndex | None = None,
) -> bool:
    """Hold the circle to the same UV-domain rule that gates group merges.

    Padding grows the circle past the box edges, so the region can reach outside
    the domain even though the rectangle it replaces did not.
    """
    x, y, w, h = group
    return (
        circle_uv_coverage(uv_mask, (x + w / 2.0, y + h / 2.0), radius, domain)
        >= config.min_box_uv_coverage
    )


def filter_groups_by_pattern(
    image: np.ndarray,
    groups: list[tuple[int, int, int, int]],
    config: MserConfig,
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    """Re-test assembled groups for repeating material.

    A group spans a whole panel rather than one glyph stroke, so the statistic
    finally has several periods to work with -- the same test on a single MSER
    box could not separate anything.  See ``pattern_window_for_box`` for the
    window the score is actually measured over.
    """
    if not config.enable_pattern_group_filter or not groups:
        return groups, []

    kept: list[tuple[int, int, int, int]] = []
    rejected: list[tuple[int, int, int, int]] = []
    for group in groups:
        window = pattern_window_for_box(image, group, config)
        if window is None:
            kept.append(group)
            continue
        score = repeating_pattern_score(window, config)
        if score > config.max_pattern_autocorrelation:
            rejected.append(group)
            continue
        kept.append(group)
    return kept, rejected


def expanded_box(
    box: tuple[int, int, int, int],
    distance: int,
) -> tuple[int, int, int, int]:
    """Return a box expanded by distance on every side."""
    x, y, w, h = box
    return (x - distance, y - distance, w + distance * 2, h + distance * 2)


def intersection_area(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> int:
    """Return pixel area of two rectangle intersection."""
    ax0, ay0, aw, ah = first
    bx0, by0, bw, bh = second
    ax1 = ax0 + aw
    ay1 = ay0 + ah
    bx1 = bx0 + bw
    by1 = by0 + bh
    width = max(0, min(ax1, bx1) - max(ax0, bx0))
    height = max(0, min(ay1, by1) - max(ay0, by0))
    return width * height


def grid_cells_for_box(
    box: tuple[int, int, int, int],
    cell_size: int,
) -> list[tuple[int, int]]:
    """Return flat grid cell ids spanned by a box."""
    x, y, w, h = box
    cell_size = max(cell_size, 1)
    x0 = x // cell_size
    y0 = y // cell_size
    x1 = (x + w - 1) // cell_size
    y1 = (y + h - 1) // cell_size
    return [(cx, cy) for cy in range(y0, y1 + 1) for cx in range(x0, x1 + 1)]


@dataclass(frozen=True, slots=True)
class GroupCandidate:
    """One proximity group together with the exact boxes that formed it."""

    bounds: tuple[int, int, int, int]
    members: tuple[tuple[int, int, int, int], ...]
    units: tuple[tuple[int, int, int, int], ...] = ()
    unit_members: tuple[tuple[tuple[int, int, int, int], ...], ...] = ()

    def recovery_units(self) -> tuple[tuple[int, int, int, int], ...]:
        """Return the grouping level Domain recovery should split back to."""
        return self.units or self.members

    def source_members_for_unit(
        self,
        index: int,
    ) -> tuple[tuple[int, int, int, int], ...]:
        """Return the raw boxes absorbed into one overlap recovery unit."""
        if index < len(self.unit_members):
            return self.unit_members[index]
        return ()


@dataclass(frozen=True, slots=True)
class UvDomainIndex:
    """Whole-mask derivatives shared by every grouping call in one run.

    Domain recovery regroups the members of each failed candidate separately,
    so without this the connected-component pass over the entire atlas ran once
    per failure -- work that does not depend on the boxes being regrouped.  The
    labels are read-only and safe to share between calls and between resumed
    runs of the same texture.
    """

    island_labels: np.ndarray


def build_uv_domain_index(uv_mask: np.ndarray | None) -> UvDomainIndex | None:
    """Label the UV islands once, for reuse by every grouping call."""
    if uv_mask is None:
        return None
    _count, labels = cv2.connectedComponents(
        uv_mask.astype(np.uint8), connectivity=8
    )
    return UvDomainIndex(island_labels=labels)


def box_island_bits(bits: np.ndarray, box: tuple[int, int, int, int]) -> int:
    """Union of the UV charts a box stands on, as a bit set."""
    x, y, w, h = box
    height, width = bits.shape[:2]
    x0, y0 = max(int(x), 0), max(int(y), 0)
    x1, y1 = min(int(x) + int(w), width), min(int(y) + int(h), height)
    if x1 <= x0 or y1 <= y0:
        return 0
    return int(np.bitwise_or.reduce(bits[y0:y1, x0:x1].ravel()))


def union_region_group_candidates(
    boxes: np.ndarray,
    image: np.ndarray,
    config: MserConfig,
    uv_mask: np.ndarray | None = None,
    domain: UvDomainIndex | None = None,
    *,
    raster_tolerance: bool = True,
    defer_domain_validation: bool = False,
    circle_cache: dict[tuple[MserConfig, tuple[int, int, int, int]], int | None] | None = None,
    island_bits: np.ndarray | None = None,
) -> list[GroupCandidate]:
    """Group boxes by overlap/proximity and retain exact membership."""
    candidates = [
        GroupCandidate(
            bounds=tuple(int(value) for value in box),
            members=(tuple(int(value) for value in box),),
        )
        for box in boxes
    ]
    return _merge_region_group_candidates(
        candidates,
        image,
        config,
        uv_mask,
        domain,
        include_overlap=True,
        include_proximity=True,
        raster_tolerance=raster_tolerance,
        defer_domain_validation=defer_domain_validation,
        circle_cache=circle_cache,
        island_bits=island_bits,
    )


def strict_region_recovery_candidates(
    units: list[tuple[int, int, int, int]],
    image: np.ndarray,
    config: MserConfig,
    uv_mask: np.ndarray | None = None,
    domain: UvDomainIndex | None = None,
    *,
    include_overlap: bool = False,
    circle_cache: dict[tuple[MserConfig, tuple[int, int, int, int]], int | None] | None = None,
    island_bits: np.ndarray | None = None,
) -> list[GroupCandidate]:
    """Regroup valid recovery units without ever growing outside the domain."""
    candidates = [
        GroupCandidate(bounds=unit, members=(unit,))
        for unit in units
    ]
    return _merge_region_group_candidates(
        candidates,
        image,
        config,
        uv_mask,
        domain,
        include_overlap=include_overlap,
        include_proximity=True,
        enforce_region_coverage=True,
        raster_tolerance=False,
        defer_domain_validation=True,
        circle_cache=circle_cache,
        island_bits=island_bits,
    )


def overlap_region_group_candidates(
    boxes: np.ndarray,
    image: np.ndarray,
    config: MserConfig,
    uv_mask: np.ndarray | None = None,
    domain: UvDomainIndex | None = None,
    *,
    defer_domain_validation: bool = False,
) -> list[GroupCandidate]:
    """Collapse every positive-area box overlap before distance grouping."""
    candidates = [
        GroupCandidate(
            bounds=tuple(int(value) for value in box),
            members=(tuple(int(value) for value in box),),
        )
        for box in boxes
    ]
    return _merge_region_group_candidates(
        candidates,
        image,
        config,
        uv_mask,
        domain,
        include_overlap=True,
        include_proximity=False,
        defer_domain_validation=defer_domain_validation,
    )


def _merge_region_group_candidates(
    seed_candidates: list[GroupCandidate],
    image: np.ndarray,
    config: MserConfig,
    uv_mask: np.ndarray | None,
    domain: UvDomainIndex | None,
    *,
    include_overlap: bool,
    include_proximity: bool,
    enforce_region_coverage: bool = False,
    raster_tolerance: bool = True,
    contrast_response: np.ndarray | None = None,
    contrast_threshold: float | None = None,
    relief_bridge_response: np.ndarray | None = None,
    island_bits: np.ndarray | None = None,
    preserve_nested_pairs: bool = False,
    preserve_overlapping_pairs: bool = False,
    circle_cache: dict[tuple[MserConfig, tuple[int, int, int, int]], int | None] | None = None,
    defer_domain_validation: bool = False,
) -> list[GroupCandidate]:
    """Merge candidate bounds while preserving each candidate's source boxes.

    Candidate joins are attempted in two distinct phases: every positive-area
    overlap merges first, then touching and progressively more distant pairs
    are considered only when proximity grouping is enabled.  When the UV-domain filter is active, a join whose
    accumulated rectangular bounds would fail the loose box-domain test is
    refused while the already-formed smaller groups are kept.  The stricter
    shaped-domain test waits for Domain recovery, after circular groups have had
    a chance to form.

    ``domain`` is the shared island labelling; it is built on demand when the
    caller has none.
    """
    if not seed_candidates:
        return []

    height, width = image.shape[:2]
    distance = max(0, config.merge_distance_px)
    min_union_area = max(1, config.min_group_union_region_px)
    box_tuples = [candidate.bounds for candidate in seed_candidates]
    # Box bounds are half-open raster intervals.  Give every non-zero merge
    # radius one pixel of symmetric tolerance so two boxes that land exactly
    # on the integer expansion boundary still share one texel.  Otherwise a
    # one-pixel extraction/rasterisation difference splits a row or column
    # even though it is within the configured grouping distance in practice.
    # Domain recovery is intentionally stricter: its job is to rebuild a
    # domain-failed candidate without taking a rounding tolerance that could
    # create another invalid union.  Ordinary Initial grouping gets the
    # tolerance; strict recovery keeps its exact-domain contract.
    expanded_distance = distance + (1 if distance and raster_tolerance else 0)
    expanded_boxes = [expanded_box(box, expanded_distance) for box in box_tuples]
    cell_size = max(distance + int(np.sqrt(min_union_area)), 16)
    parent = list(range(len(box_tuples)))
    bounds = [(x, y, x + w, y + h) for x, y, w, h in box_tuples]

    island = [0] * len(box_tuples)
    if uv_mask is not None and config.enable_island_bounded_grouping:
        if domain is None:
            domain = build_uv_domain_index(uv_mask)
        labels = domain.island_labels  # type: ignore[union-attr]
        for index, (bx, by, bw, bh) in enumerate(box_tuples):
            clamped = clamp_group((bx, by, bw, bh), uv_mask.shape)
            if clamped is None:
                continue
            cx, cy, cw, ch = clamped
            within = labels[cy : cy + ch, cx : cx + cw]
            within = within[within > 0]
            if within.size:
                island[index] = int(np.bincount(within).argmax())

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def merged_bounds(root: int, other_root: int) -> tuple[int, int, int, int]:
        first = bounds[root]
        second = bounds[other_root]
        return (
            min(first[0], second[0]),
            min(first[1], second[1]),
            max(first[2], second[2]),
            max(first[3], second[3]),
        )

    def merged_pre_domain_ok(rect: tuple[int, int, int, int]) -> bool:
        # Build the complete overlap/proximity candidate first.  A rectangular
        # bounds can cross a concave UV void while its inscribed circle fits;
        # when neither geometry fits, Domain recovery needs the complete group
        # provenance to undo only the joins that made it invalid.  The strict
        # recovery pass is the sole place where a prospective merge is gated
        # by final shape coverage.
        if defer_domain_validation:
            return True
        if (
            uv_mask is None
            or not config.enable_region_domain_filter
            or config.min_box_uv_coverage <= 0.0
        ):
            return True
        x0, y0, x1, y1 = rect
        group = (x0, y0, x1 - x0, y1 - y0)
        return (
            box_uv_coverage(uv_mask, group, image.shape, domain)
            >= config.min_box_uv_coverage
        )

    def merged_strict_region_ok(rect: tuple[int, int, int, int]) -> bool:
        if not enforce_region_coverage or uv_mask is None:
            return True
        x0, y0, x1, y1 = rect
        _radius, coverage = _region_shape_and_coverage(
            image, uv_mask, (x0, y0, x1 - x0, y1 - y0), config, domain,
            circle_cache,
        )
        return coverage >= config.min_region_uv_coverage

    def merge(root: int, other_root: int, rect: tuple[int, int, int, int]) -> None:
        """Attach one resolved root to another, growing the surviving bounds."""
        parent[other_root] = root
        bounds[root] = rect

    def root_box(root: int) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = bounds[root]
        return (x0, y0, x1 - x0, y1 - y0)

    def pair_priority(
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
        left: int,
        right: int,
    ) -> tuple[float, int, int, int, int]:
        ax, ay, aw, ah = first
        bx, by, bw, bh = second
        ax1, ay1 = ax + aw, ay + ah
        bx1, by1 = bx + bw, by + bh
        gap_x = max(bx - ax1, ax - bx1, 0)
        gap_y = max(by - ay1, ay - by1, 0)
        gap = math.hypot(gap_x, gap_y)
        overlap = intersection_area(first, second)
        union_area = (max(ax1, bx1) - min(ax, bx)) * (max(ay1, by1) - min(ay, by))
        return (gap, -overlap, union_area, min(left, right), max(left, right))

    def overlap_priority(
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
        left: int,
        right: int,
    ) -> tuple[float, int, int, int, int]:
        ax, ay, aw, ah = first
        bx, by, bw, bh = second
        ax1, ay1 = ax + aw, ay + ah
        bx1, by1 = bx + bw, by + bh
        overlap = intersection_area(first, second)
        smaller_area = max(min(aw * ah, bw * bh), 1)
        ratio = float(overlap) / float(smaller_area)
        union_area = (max(ax1, bx1) - min(ax, bx)) * (max(ay1, by1) - min(ay, by))
        return (-ratio, -overlap, union_area, min(left, right), max(left, right))

    def axis_aligned_grouping_link(
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
    ) -> bool:
        """Only let grouping grow along well-aligned rows or columns."""
        ax, ay, aw, ah = first
        bx, by, bw, bh = second
        ax1, ay1 = ax + aw, ay + ah
        bx1, by1 = bx + bw, by + bh
        overlap_x = max(min(ax1, bx1) - max(ax, bx), 0)
        overlap_y = max(min(ay1, by1) - max(ay, by), 0)
        separated_x = overlap_x <= 0
        separated_y = overlap_y <= 0
        # Overlap collapse is intentionally its own earlier stage.  Initial
        # grouping is solely for a *gap* between components on one axis.  In
        # particular, an enormous panel candidate must not absorb a smaller
        # glyph/seam sitting inside it merely because their rectangles overlap
        # in both axes.
        if separated_x == separated_y:
            return False
        min_axis_overlap = 0.5
        centre_tolerance = max(float(config.group_axis_center_tolerance), 0.0)
        if separated_x:
            # Horizontal join: same visual row, not merely a thin feature
            # touching the top or bottom edge of a much taller box.
            return (
                overlap_y >= min(ah, bh) * min_axis_overlap
                and abs((ay + ay1) - (by + by1)) * 0.5
                <= min(ah, bh) * centre_tolerance
            )
        if separated_y:
            # Vertical join: same visual column by the equivalent test.
            return (
                overlap_x >= min(aw, bw) * min_axis_overlap
                and abs((ax + ax1) - (bx + bx1)) * 0.5
                <= min(aw, bw) * centre_tolerance
            )
        return True

    def strictly_nested(
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
    ) -> bool:
        """Whether one box wholly contains the other without being identical."""
        ax, ay, aw, ah = first
        bx, by, bw, bh = second
        ax1, ay1 = ax + aw, ay + ah
        bx1, by1 = bx + bw, by + bh
        first_contains_second = ax <= bx and ay <= by and ax1 >= bx1 and ay1 >= by1
        second_contains_first = bx <= ax and by <= ay and bx1 >= ax1 and by1 >= ay1
        return (first_contains_second or second_contains_first) and first != second

    def nested_with_raster_tolerance(
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
    ) -> bool:
        """Treat a one-texel detector edge disagreement as containment."""
        if not raster_tolerance:
            return False
        ax, ay, aw, ah = first
        bx, by, bw, bh = second
        ax1, ay1 = ax + aw, ay + ah
        bx1, by1 = bx + bw, by + bh
        first_contains_second = (
            ax - 1 <= bx and ay - 1 <= by and ax1 + 1 >= bx1 and ay1 + 1 >= by1
        )
        second_contains_first = (
            bx - 1 <= ax and by - 1 <= ay and bx1 + 1 >= ax1 and by1 + 1 >= ay1
        )
        return first_contains_second or second_contains_first

    def bridge_is_clear(
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
        response: np.ndarray | None,
        enabled: bool,
        minimum_response: float,
        max_high_coverage: float,
        min_cross_axis_coverage: float | None = None,
    ) -> bool:
        """Whether one cached response field permits this proximity join."""
        if (
            not enabled
            or response is None
        ):
            return True
        ax, ay, aw, ah = first
        bx, by, bw, bh = second
        ax1, ay1 = ax + aw, ay + ah
        bx1, by1 = bx + bw, by + bh
        # Only test the true gap, restricted to the shared axis.  Overlaps
        # have no separating material and are handled by the separate overlap
        # stage, so they stay eligible regardless of this evidence.
        horizontal_join = ax1 <= bx or bx1 <= ax
        if horizontal_join:
            x0, x1 = (ax1, bx) if ax1 <= bx else (bx1, ax)
            y0, y1 = max(ay, by), min(ay1, by1)
        elif ay1 <= by or by1 <= ay:
            y0, y1 = (ay1, by) if ay1 <= by else (by1, ay)
            x0, x1 = max(ax, bx), min(ax1, bx1)
        else:
            return True
        if x1 <= x0 or y1 <= y0:
            return True
        bridge = response[
            max(y0, 0):min(y1, response.shape[0]),
            max(x0, 0):min(x1, response.shape[1]),
        ]
        if bridge.size == 0:
            return True
        # ``minimum_response`` is an absolute safety floor.  For colour,
        # ``contrast_threshold`` additionally carries the per-island cut that
        # produced the glyph candidates.  A few mildly contrasting texels in
        # a tiny gap (the red field behind ENGINE / start / stop, for example)
        # are not a separator.  They must not defeat a join unless they are as
        # strong as the island's own selected glyph response.
        bridge_threshold = max(float(minimum_response), 0.0)
        high = bridge >= bridge_threshold
        if min_cross_axis_coverage is not None:
            # A horizontal proximity join is blocked only by a near-vertical
            # relief feature that crosses the shared height; similarly a
            # vertical join needs a near-horizontal feature crossing its shared
            # width.  This rejects an actual panel/button seam without treating
            # the adjacent glyphs' own perimeter edges as a separator.
            spans = high.mean(axis=0) if horizontal_join else high.mean(axis=1)
            maximum_span = float(spans.max()) if spans.size else 0.0
            return maximum_span < float(min_cross_axis_coverage)
        high_coverage = float(high.mean())
        return high_coverage <= float(max_high_coverage)

    island_bit_cache: dict[tuple[int, int, int, int], int] = {}

    def islands_share(
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
    ) -> bool:
        """Whether two boxes stand on a common UV chart.

        Charts are coalesced into one detection consumer only because they can
        share atlas texels, and a shared texel has exactly one answer.  That is
        not a reason to let a mark on one chart pull in a mark on another: they
        are different surfaces that merely landed near each other in the atlas.
        Crossing a chart boundary therefore vetoes the join.
        """
        if island_bits is None:
            return True
        left = island_bit_cache.get(first)
        if left is None:
            left = box_island_bits(island_bits, first)
            island_bit_cache[first] = left
        right = island_bit_cache.get(second)
        if right is None:
            right = box_island_bits(island_bits, second)
            island_bit_cache[second] = right
        if not left or not right:
            # A box standing on no recorded chart says nothing either way.
            return True
        return bool(left & right)

    def bridge_is_continuous(
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
    ) -> bool:
        """Accept only corridors clear of both colour and relief barriers."""
        return (
            islands_share(first, second)
            and (
            bridge_is_clear(
                first, second,
                contrast_response,
                config.enable_contrast_continuity_grouping and contrast_threshold is not None,
                max(config.contrast_bridge_min_response, contrast_threshold or 0.0),
                config.contrast_bridge_max_high_coverage,
            )
            and bridge_is_clear(
                first, second,
                relief_bridge_response,
                config.enable_relief_edge_bridge_grouping,
                config.relief_bridge_min_response,
                0.0,
                config.relief_bridge_min_cross_axis_coverage,
            )
        ))

    # Neighbour search over a uniform grid whose cell is the merge distance, so
    # two expanded boxes that overlap always share at least one cell.  Pairs are
    # collected first and then replayed nearest-first; that prevents a distant
    # transitive chain from growing a group before closer local detail has had a
    # chance to settle.
    grid: dict[tuple[int, int], list[int]] = {}
    overlap_edges: list[tuple[tuple[float, int, int, int, int], int, int]] = []
    merge_edges: list[tuple[tuple[float, int, int, int, int], int, int]] = []
    for index, expanded in enumerate(expanded_boxes):
        checked: set[int] = set()
        cells = grid_cells_for_box(expanded, cell_size)
        for cell in cells:
            for other in grid.get(cell, ()):
                if other in checked:
                    continue
                checked.add(other)
                if include_overlap:
                    overlap = intersection_area(box_tuples[index], box_tuples[other])
                    if (
                        overlap > 0
                        or nested_with_raster_tolerance(
                            box_tuples[index], box_tuples[other]
                        )
                    ):
                        priority = overlap_priority(
                            box_tuples[index], box_tuples[other], index, other
                        )
                        overlap_edges.append((priority, index, other))
                if include_proximity:
                    if island[index] != island[other]:
                        continue
                    if preserve_overlapping_pairs and intersection_area(
                        box_tuples[index], box_tuples[other]
                    ) > 0:
                        continue
                    if preserve_nested_pairs and strictly_nested(
                        box_tuples[index], box_tuples[other]
                    ):
                        continue
                    if not axis_aligned_grouping_link(
                        box_tuples[index], box_tuples[other]
                    ):
                        continue
                    if not bridge_is_continuous(
                        box_tuples[index], box_tuples[other]
                    ):
                        continue
                    if intersection_area(expanded, expanded_boxes[other]) >= min_union_area:
                        priority = pair_priority(
                            box_tuples[index], box_tuples[other], index, other
                        )
                        merge_edges.append((priority, index, other))
        for cell in cells:
            grid.setdefault(cell, []).append(index)

    for _priority, index, other in sorted(overlap_edges):
        root = find(index)
        other_root = find(other)
        if root == other_root:
            continue
        rect = merged_bounds(root, other_root)
        if not merged_pre_domain_ok(rect) or not merged_strict_region_ok(rect):
            continue
        merge(root, other_root, rect)

    # A rectangular union can intersect a third candidate even when neither
    # source rectangle did.  Re-evaluate the surviving bounds until the stage
    # reaches its stated invariant: no mergeable positive-area overlaps remain.
    if include_overlap:
        while True:
            roots = sorted({find(index) for index in range(len(box_tuples))})
            root_overlap_edges: list[
                tuple[tuple[float, int, int, int, int], int, int]
            ] = []
            for left_pos, root in enumerate(roots):
                first = root_box(root)
                for other_root in roots[left_pos + 1:]:
                    second = root_box(other_root)
                    if (
                        intersection_area(first, second) <= 0
                        and not nested_with_raster_tolerance(first, second)
                    ):
                        continue
                    root_overlap_edges.append(
                        (
                            overlap_priority(first, second, root, other_root),
                            root,
                            other_root,
                        )
                    )

            merged_any = False
            for _priority, root, other_root in sorted(root_overlap_edges):
                root = find(root)
                other_root = find(other_root)
                if root == other_root:
                    continue
                rect = merged_bounds(root, other_root)
                if not merged_pre_domain_ok(rect) or not merged_strict_region_ok(rect):
                    continue
                merge(root, other_root, rect)
                merged_any = True
            if not merged_any:
                break

    for _priority, index, other in sorted(merge_edges):
        root = find(index)
        other_root = find(other)
        if root == other_root or island[root] != island[other_root]:
            continue
        rect = merged_bounds(root, other_root)
        if not merged_pre_domain_ok(rect) or not merged_strict_region_ok(rect):
            continue
        merge(root, other_root, rect)

    if include_proximity:
        while True:
            roots = sorted({find(index) for index in range(len(box_tuples))})
            group_edges: list[
                tuple[tuple[float, int, int, int, int], int, int]
            ] = []
            for left_pos, root in enumerate(roots):
                first = root_box(root)
                first_expanded = expanded_box(first, expanded_distance)
                for other_root in roots[left_pos + 1:]:
                    if island[root] != island[other_root]:
                        continue
                    second = root_box(other_root)
                    if preserve_overlapping_pairs and intersection_area(first, second) > 0:
                        continue
                    if preserve_nested_pairs and strictly_nested(first, second):
                        continue
                    if not axis_aligned_grouping_link(first, second):
                        continue
                    if not bridge_is_continuous(first, second):
                        continue
                    if (
                        intersection_area(
                            first_expanded, expanded_box(second, expanded_distance)
                        )
                        < min_union_area
                    ):
                        continue
                    group_edges.append((pair_priority(first, second, root, other_root), root, other_root))

            merged_any = False
            for _priority, root, other_root in sorted(group_edges):
                root = find(root)
                other_root = find(other_root)
                if root == other_root or island[root] != island[other_root]:
                    continue
                rect = merged_bounds(root, other_root)
                if not merged_pre_domain_ok(rect) or not merged_strict_region_ok(rect):
                    continue
                merge(root, other_root, rect)
                merged_any = True
            if not merged_any:
                break

    members: dict[int, list[int]] = {}
    for index in range(len(box_tuples)):
        members.setdefault(find(index), []).append(index)

    candidates: list[GroupCandidate] = []
    for root, indices in members.items():
        rect = bounds[root]
        x0 = max(rect[0], 0)
        y0 = max(rect[1], 0)
        x1 = min(rect[2], width)
        y1 = min(rect[3], height)
        if x1 <= x0 or y1 <= y0:
            continue
        candidate_bounds = (x0, y0, x1 - x0, y1 - y0)
        recovery_member_groups: tuple[
            tuple[tuple[int, int, int, int], ...], ...
        ] = ()
        if include_overlap and not include_proximity:
            recovery_units = (candidate_bounds,)
            recovery_member_groups = (
                tuple(
                    member
                    for index in indices
                    for member in seed_candidates[index].members
                ),
            )
        elif any(seed_candidates[index].units for index in indices):
            recovery_units = tuple(seed_candidates[index].bounds for index in indices)
            recovery_member_groups = tuple(
                seed_candidates[index].members for index in indices
            )
        else:
            recovery_units = ()
        candidates.append(
            GroupCandidate(
                bounds=candidate_bounds,
                members=tuple(
                    member
                    for index in indices
                    for member in seed_candidates[index].members
                ),
                units=recovery_units,
                unit_members=recovery_member_groups,
            )
        )

    candidates.sort(key=lambda candidate: (candidate.bounds[1], candidate.bounds[0]))
    return candidates


# ---------------------------------------------------------------------------
# Geometry and colour helpers
# ---------------------------------------------------------------------------


def clamp_group(
    group: tuple[int, int, int, int],
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    """Clamp a rectangle to image bounds and discard empty results."""
    x, y, w, h = group
    image_height, image_width = image_shape[:2]
    x0 = max(x, 0)
    y0 = max(y, 0)
    x1 = min(x + w, image_width)
    y1 = min(y + h, image_height)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


def dominant_quantized_bin(quantized: np.ndarray, levels: int) -> np.ndarray:
    """Return the most populated quantized BGR bin, given quantized pixels."""
    if levels**3 > 1_000_000:
        bins, counts = np.unique(quantized.reshape(-1, 3), axis=0, return_counts=True)
        return bins[int(np.argmax(counts))]

    codes = (quantized[:, 0] * levels + quantized[:, 1]) * levels + quantized[:, 2]
    dominant_code = int(np.bincount(codes).argmax())
    b = dominant_code // (levels * levels)
    remainder = dominant_code % (levels * levels)
    return np.asarray((b, remainder // levels, remainder % levels), dtype=np.int32)


def estimate_background_colour(
    crop: np.ndarray,
    domain_mask: np.ndarray,
    colour_thresh: int,
) -> np.ndarray | None:
    """Estimate the dominant background BGR colour inside the current domain.

    Quantises once and reuses that array for both the bin histogram and the
    membership test; this runs per MSER box, so it is the pipeline's hottest
    piece of arithmetic.
    """
    domain_pixels = crop[domain_mask]
    if len(domain_pixels) == 0:
        return None

    quant_step = max(colour_thresh, 1)
    quantized = (domain_pixels // quant_step).astype(np.int32)
    levels = (255 // quant_step) + 1
    dominant_matches = np.all(
        quantized == dominant_quantized_bin(quantized, levels), axis=1
    )
    if not bool(dominant_matches.any()):
        return None

    return np.median(domain_pixels[dominant_matches], axis=0).astype(np.int16)


def filter_groups_by_final_size(
    groups: list[tuple[int, int, int, int]],
    config: MserConfig,
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    """Drop remaining groups too small, or too elongated, to be a detection."""
    if not config.enable_final_size_filter and not config.enable_final_aspect_filter:
        return groups, []

    kept: list[tuple[int, int, int, int]] = []
    rejected: list[tuple[int, int, int, int]] = []
    for group in groups:
        _x, _y, w, h = group
        if config.enable_final_size_filter and (
            w < config.final_min_width_px
            or h < config.final_min_height_px
            or w * h < config.final_min_area_px
        ):
            rejected.append(group)
            continue
        if (
            config.enable_final_aspect_filter
            and max(w, h) / max(min(w, h), 1) > config.final_max_aspect
        ):
            rejected.append(group)
            continue
        kept.append(group)

    return kept, rejected


# ---------------------------------------------------------------------------
# Annotation drawing
# ---------------------------------------------------------------------------


def draw_annotations(
    image: np.ndarray,
    groups: list[tuple[int, int, int, int]],
    config: MserConfig,
) -> np.ndarray:
    """Draw a 2 px composite border on a copy.

    The detected region is marked with a green 1 px inner edge and an
    immediately adjacent red 1 px outer edge.
    """
    annotated = image.copy()
    height, width = annotated.shape[:2]

    for x, y, w, h in groups:
        gx0 = max(x, 0)
        gy0 = max(y, 0)
        gx1 = min(x + w - 1, width - 1)
        gy1 = min(y + h - 1, height - 1)
        if gx1 < gx0 or gy1 < gy0:
            continue

        rx0 = max(gx0 - 1, 0)
        ry0 = max(gy0 - 1, 0)
        rx1 = min(gx1 + 1, width - 1)
        ry1 = min(gy1 + 1, height - 1)

        cv2.rectangle(
            annotated,
            (rx0, ry0),
            (rx1, ry1),
            config.red_colour,
            config.red_thickness,
            cv2.LINE_8,
        )
        cv2.rectangle(
            annotated,
            (gx0, gy0),
            (gx1, gy1),
            config.green_colour,
            config.green_thickness,
            cv2.LINE_8,
        )

    return annotated


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DetectionState:
    """Everything one pipeline step hands to the next."""

    boxes: np.ndarray
    groups: list[tuple[int, int, int, int]]
    candidates: list[GroupCandidate] = field(default_factory=list)
    # Built once by the grouping step and read by domain recovery.  Shared by
    # reference: the labelling is read-only and depends only on the UV mask,
    # which never changes within a run or across a resumed one.
    domain: UvDomainIndex | None = None
    # One entry per group, once the rotated-bounds step has run: the corner
    # points of the rectangle fitted to that group's feature, or None where
    # none was fitted or adopted.  Carried in the state rather than rebuilt per
    # stage so the outline survives every later filter -- refitting after each
    # one would cost a gradient pass apiece and could quietly disagree with
    # what the earlier stage decided on.
    rotations: list[tuple[tuple[float, float], ...] | None] = field(
        default_factory=list
    )
    feature_hulls: list[RegionFeatureHull | None] = field(default_factory=list)
    # Local-contrast runs compute this once in the front end.  It is retained
    # through the early grouping stages so a join can inspect its bridge
    # without doing another blur/percentile pass.
    contrast_response: np.ndarray | None = None
    contrast_threshold: float | None = None
    relief_bridge_response: np.ndarray | None = None
    # Which member UV chart owns each pixel, one bit per chart, when this view
    # is a coalesced consumer of several.  Grouping refuses a join whose two
    # sides share no chart: coalescing exists because charts can share texels,
    # not as a licence to merge marks that merely sit near each other.
    island_bits: np.ndarray | None = None
    # The edge front end and the relief-aware filters all need the same
    # thresholded (but deliberately not component-closed) edge pixels.  Keep
    # that per-island mask beside its response so later filters do not repeat
    # percentile/local-threshold and morphology work merely to inspect it.
    relief_edge_mask: np.ndarray | None = None
    # Shared by reference between successive stages in one run.  It is a
    # performance cache only: its keys include the immutable configuration and
    # it never changes a circle decision.
    circle_radii: dict[tuple[MserConfig, tuple[int, int, int, int]], int | None] = field(
        default_factory=dict
    )

    def copy(self) -> "DetectionState":
        return DetectionState(
            boxes=self.boxes.copy(),
            groups=list(self.groups),
            candidates=list(self.candidates),
            domain=self.domain,
            rotations=list(self.rotations),
            feature_hulls=list(self.feature_hulls),
            contrast_response=self.contrast_response,
            contrast_threshold=self.contrast_threshold,
            relief_bridge_response=self.relief_bridge_response,
            island_bits=self.island_bits,
            relief_edge_mask=self.relief_edge_mask,
            circle_radii=self.circle_radii,
        )


def _step_mser(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig,
    state: DetectionState,
) -> tuple[DetectionState, DetectionStage]:
    """Produce the raw boxes, by whichever front-end the config selects."""
    grey = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    # A colour pass may be supplied a normal-map edge response by the
    # production bundle.  Keep it alive through the contrast front-end for
    # relief-aware grouping instead of replacing it with ``None``.
    edge_bridge_response = state.relief_bridge_response
    edge_mask_cache: np.ndarray | None = None
    if config.box_source in {"edge", "edge_gpu"}:
        # Per-island GPU runs receive one atlas response sliced by the caller.
        # That is both cheaper than 500 context dispatches and avoids making a
        # crop edge look like a relief edge.  A direct/final-build run creates
        # the shared-context response here when no slice was supplied.
        if edge_bridge_response is None:
            edge_bridge_response = edge_response(grey, config)
        edge_mask_cache, _response = edge_mask_from_response(
            edge_bridge_response, uv_mask, config,
        )
        boxes, edge_bridge_response = detect_edge_boxes_from_response(
            edge_mask_cache, edge_bridge_response, config,
        )
        detail = (
            f"{config.edge_operator} gradient, threshold at the "
            f"{config.edge_threshold_percentile:g}th percentile inside the UV "
            f"domain (floor {config.edge_threshold_floor:g}), closed by "
            f"{config.edge_close_px} px"
        )
        title = "Edge GPU boxes" if config.box_source == "edge_gpu" else "Edge boxes"
    elif config.box_source == "foreground":
        boxes = detect_foreground_boxes(image, uv_mask, config)
        detail = (
            "dominant-background foreground mask, "
            f"distance {config.foreground_background_distance:g}, "
            f"closed by {config.foreground_close_px} px"
        )
        title = "Foreground-mask boxes"
    elif config.box_source == "opacity_mask":
        boxes = detect_opacity_mask_boxes(
            image, uv_mask, config, state.island_bits
        )
        detail = (
            "authored opacity-mask foreground, "
            f"threshold {config.opacity_mask_threshold}"
            + (
                "; merges stop at a UV chart boundary"
                if state.island_bits is not None
                else ""
            )
        )
        title = "Opacity-mask boxes"
    elif config.box_source in {"contrast", "contrast_gpu"}:
        contrast = (
            detect_local_contrast_gpu(image, uv_mask, config)
            if config.box_source == "contrast_gpu"
            else detect_local_contrast(image, uv_mask, config)
        )
        boxes = contrast.boxes
        detail = (
            f"masked {config.contrast_kernel_px}px local-colour contrast, "
            f"max({config.contrast_min_response:g}, "
            f"{config.contrast_percentile:g}th island percentile), closed by "
            f"{config.contrast_close_px} px"
        )
        title = "Local-contrast GPU boxes" if config.box_source == "contrast_gpu" else "Local-contrast CPU boxes"
    else:
        boxes = detect_mser_boxes(grey, config)
        detail = f"delta={config.delta}, area {config.min_area}-{config.max_area} px"
        title = "MSER boxes"
    return DetectionState(
        boxes, [],
        contrast_response=(contrast.response if config.box_source in {"contrast", "contrast_gpu"} else None),
        contrast_threshold=(contrast.threshold if config.box_source in {"contrast", "contrast_gpu"} else None),
        # Relief-edge grouping needs the same barrier evidence as the colour
        # plus normal-map path.  The Laplacian was just computed above, so
        # retain it rather than re-running a normal-map pass per UV island.
        relief_bridge_response=edge_bridge_response,
        island_bits=state.island_bits,
        relief_edge_mask=edge_mask_cache,
    ), DetectionStage(
        key="mser",
        title=title,
        kept=tuple(tuple(int(v) for v in box) for box in boxes),  # type: ignore[misc]
        detail=detail,
    )


def _step_stroke_width(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig,
    state: DetectionState,
) -> tuple[DetectionState, DetectionStage]:
    """Keep only boxes whose strokes are of one consistent thickness."""
    if not config.enable_stroke_width_filter or len(state.boxes) == 0:
        return state, DetectionStage(
            key="stroke_width",
            title="Stroke width",
            kept=tuple(tuple(int(v) for v in box) for box in state.boxes),  # type: ignore[misc]
            detail=(
                "disabled"
                if not config.enable_stroke_width_filter
                else "no boxes to test"
            ),
        )

    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    widths = stroke_width_transform(grey, uv_mask, config)
    kept: list[tuple[int, int, int, int]] = []
    rejected: list[tuple[int, int, int, int]] = []
    for raw in state.boxes:
        box = (int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3]))
        _median, variation, coverage = stroke_width_stats(widths, box)
        if coverage < config.swt_min_coverage:
            rejected.append(box)
        elif variation > config.swt_max_width_variation:
            rejected.append(box)
        else:
            kept.append(box)
    boxes = (
        np.asarray(kept, dtype=np.int32) if kept else np.empty((0, 4), dtype=np.int32)
    )
    marked = float((widths > 0).mean())
    return DetectionState(
        boxes, [],
        contrast_response=state.contrast_response,
        contrast_threshold=state.contrast_threshold,
        relief_bridge_response=state.relief_bridge_response,
        island_bits=state.island_bits,
        relief_edge_mask=state.relief_edge_mask,
    ), DetectionStage(
        key="stroke_width",
        title="Stroke width",
        kept=tuple(kept),
        rejected=tuple(rejected),
        detail=(
            f"{config.swt_polarity} strokes of {config.swt_min_stroke_px}-"
            f"{config.swt_max_stroke_px} px, facing edges within "
            f"{config.swt_gradient_tolerance_degrees:g} degrees; keep variation "
            f"<= {config.swt_max_width_variation:g} and coverage >= "
            f"{config.swt_min_coverage:.0%}; {marked:.2%} of the atlas carries a stroke"
        ),
    )


def _step_box_filter(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig,
    state: DetectionState,
) -> tuple[DetectionState, DetectionStage]:
    domain = build_uv_domain_index(uv_mask)
    boxes, rejected = filter_boxes_by_feature(
        image, uv_mask, state.boxes, config, domain
    )
    return DetectionState(
        boxes, [], domain=domain,
        contrast_response=state.contrast_response,
        contrast_threshold=state.contrast_threshold,
        relief_bridge_response=state.relief_bridge_response,
        island_bits=state.island_bits,
        relief_edge_mask=state.relief_edge_mask,
    ), DetectionStage(
        key="box_filter",
        title="Box filtering",
        kept=tuple(tuple(int(v) for v in box) for box in boxes),  # type: ignore[misc]
        rejected=tuple(rejected),
        detail=(
            f"minimum {max(config.min_box_width_px, 1)}x"
            f"{max(config.min_box_height_px, 1)} px; "
            + (
                f"need >= {config.min_box_uv_coverage:.0%} of the box inside one UV "
                f"island and a non-background blob of >= {config.box_min_feature_px} "
                f"px within {config.box_feature_colour_tolerance} of one colour"
                if config.enable_box_feature_filter
                else "feature and UV-domain checks disabled"
            )
        ),
    )


def _step_overlap_box_group(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig,
    state: DetectionState,
) -> tuple[DetectionState, DetectionStage]:
    """Collapse all intersecting boxes before proximity grouping."""
    domain = state.domain or build_uv_domain_index(uv_mask)
    candidates = overlap_region_group_candidates(
        state.boxes, image, config, uv_mask, domain,
        defer_domain_validation=True,
    )
    groups = [candidate.bounds for candidate in candidates]
    absorbed = sum(max(len(candidate.members) - 1, 0) for candidate in candidates)
    return DetectionState(
        state.boxes, groups, candidates, domain,
        contrast_response=state.contrast_response,
        contrast_threshold=state.contrast_threshold,
        relief_bridge_response=state.relief_bridge_response,
        island_bits=state.island_bits,
        relief_edge_mask=state.relief_edge_mask,
    ), DetectionStage(
        key="overlap_box_group",
        title="Overlap grouping",
        kept=tuple(groups),
        adjusted=absorbed,
        detail=(
            "every positive-area overlap, including one-texel raster "
            "containment, is collapsed first; "
            f"{absorbed} nested/overlapping box"
            f"{'es' if absorbed != 1 else ''} absorbed"
        ),
    )


def _step_grouped(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig,
    state: DetectionState,
) -> tuple[DetectionState, DetectionStage]:
    """Create nearest-first groups and retain their exact members."""
    domain = state.domain or build_uv_domain_index(uv_mask)
    seed_candidates = state.candidates or [
        GroupCandidate(
            bounds=tuple(int(value) for value in box),
            members=(tuple(int(value) for value in box),),
        )
        for box in state.boxes
    ]
    candidates = _merge_region_group_candidates(
        seed_candidates,
        image,
        config,
        uv_mask,
        domain,
        include_overlap=False,
        include_proximity=True,
        contrast_response=state.contrast_response,
        contrast_threshold=state.contrast_threshold,
        relief_bridge_response=state.relief_bridge_response,
        island_bits=state.island_bits,
        defer_domain_validation=True,
    )
    groups = [candidate.bounds for candidate in candidates]
    return DetectionState(
        state.boxes, groups, candidates, domain,
        contrast_response=state.contrast_response,
        contrast_threshold=state.contrast_threshold,
        relief_bridge_response=state.relief_bridge_response,
        island_bits=state.island_bits,
        relief_edge_mask=state.relief_edge_mask,
    ), DetectionStage(
        key="grouped",
        title="Initial grouping",
        kept=tuple(groups),
        detail=(
            f"boxes expanded by {config.merge_distance_px} px per side; "
            "nearest horizontal/vertical joins with >= 50% shared-axis overlap "
            f"and centres within {config.group_axis_center_tolerance:g}x the smaller shared axis "
            "are accepted only across colour- and relief-clear bridges, then rechecked as groups grow; diagonal and weak-overlap "
            "joins are left as smaller groups. Domain recovery validates completed groups and splits invalid ones."
        ),
    )


def _step_pattern_group(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig,
    state: DetectionState,
) -> tuple[DetectionState, DetectionStage]:
    groups, rejected = filter_groups_by_pattern(image, state.groups, config)
    return DetectionState(
        state.boxes, groups, relief_bridge_response=state.relief_bridge_response,
        island_bits=state.island_bits,
        relief_edge_mask=state.relief_edge_mask,
    ), DetectionStage(
        key="pattern_group",
        title="Repeating pattern (groups)",
        kept=tuple(groups),
        rejected=tuple(rejected),
        detail=(
            f"reject above {config.max_pattern_autocorrelation} autocorrelation "
            f"at lags {config.pattern_min_period_px}-{config.pattern_max_period_px} px"
        ),
    )


def _rotations_for_subset(
    rotations: list[tuple[tuple[float, float], ...] | None],
    original: list[tuple[int, int, int, int]],
    kept: list[tuple[int, int, int, int]],
) -> list[tuple[tuple[float, float], ...] | None]:
    """Realign carried outlines to an order-preserving subset of the regions.

    Matched by walking both lists forward rather than by looking each region
    up, because two regions can have identical bounds and a lookup would give
    them the same outline.
    """
    if not rotations:
        return []
    aligned: list[tuple[tuple[float, float], ...] | None] = []
    index = 0
    for group in kept:
        while index < len(original) and original[index] != group:
            index += 1
        aligned.append(rotations[index] if index < len(rotations) else None)
        index += 1
    return aligned


def _step_rotated_bounds(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig,
    state: DetectionState,
) -> tuple[DetectionState, DetectionStage]:
    """Judge each region by the true shape of the feature inside it."""
    if not config.enable_rotated_bounds_filter or not state.groups:
        return state, DetectionStage(
            key="rotated_bounds",
            title="Rotated bounds",
            kept=tuple(state.groups),
            detail=(
                "disabled"
                if not config.enable_rotated_bounds_filter
                else "no regions to test"
            ),
        )

    mask = state.relief_edge_mask
    if mask is None:
        if state.relief_bridge_response is not None:
            mask, _response = edge_mask_from_response(
                state.relief_bridge_response, uv_mask, config,
            )
        else:
            grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mask, _response = edge_mask(grey, uv_mask, config)
    if mask is None:  # keeps static typing honest; the branches always fill it.
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask, _response = edge_mask(grey, uv_mask, config)
    uv_boundary = None
    uv_contours: tuple[np.ndarray, ...] | None = None
    if config.enable_edge_aligned_rotation and uv_mask is not None:
        uv_boundary = cv2.morphologyEx(
            uv_mask.astype(np.uint8),
            cv2.MORPH_GRADIENT,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        ).astype(bool)
        found_contours, _hierarchy = cv2.findContours(
            uv_mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE,
        )
        uv_contours = tuple(found_contours)

    kept: list[tuple[int, int, int, int]] = []
    rejected: list[tuple[int, int, int, int]] = []
    rotations: list[tuple[tuple[float, float], ...] | None] = []
    unfitted = 0
    unadopted = 0
    edge_unadopted = 0
    edge_adopted = 0
    for group in state.groups:
        if cached_inscribed_circle_radius(
            image, uv_mask, group, config, state.circle_radii,
        ) is not None:
            kept.append(group)
            rotations.append(None)
            continue
        shape = feature_shape(mask, group, config)
        if shape is None:
            # Nothing was fitted, so nothing is concluded.
            unfitted += 1
            kept.append(group)
            rotations.append(None)
            continue
        fill, aspect = rotated_bounds_measures(shape.rectangle, group)
        if fill < config.min_rotated_fill or aspect > config.max_rotated_aspect:
            rejected.append(group)
            continue
        kept.append(group)
        # Whether to adopt the outline is a separate question from whether to
        # keep the region: a shape wrapping mostly empty space describes the
        # feature badly, but that is not evidence the region holds no mark.
        if (
            shape.tightness(config.bounds_shape) < config.min_feature_tightness
            or aspect < config.min_rotated_elongation
        ):
            unadopted += 1
            rotations.append(None)
            continue
        if config.enable_edge_aligned_rotation:
            alignment = edge_aligned_feature_outline(
                mask, uv_mask, group, shape, config, uv_boundary, uv_contours
            )
            if alignment is None:
                edge_unadopted += 1
                rotations.append(None)
                continue
            edge_adopted += 1
            rotations.append(alignment.outline)
            continue
        # Points from OpenCV rather than re-derived, so what is drawn is the
        # shape that was measured, to the pixel.
        rotations.append(shape.outline(config.bounds_shape))

    adopted_shape = (
        "edge-aligned rectangle"
        if config.enable_edge_aligned_rotation
        else f"{config.bounds_shape} outline"
    )
    return DetectionState(
        state.boxes, kept, rotations=rotations,
        relief_bridge_response=state.relief_bridge_response,
        island_bits=state.island_bits,
        relief_edge_mask=mask,
    ), DetectionStage(
        key="rotated_bounds",
        title="Rotated bounds",
        kept=tuple(kept),
        rejected=tuple(rejected),
        rotations=tuple(rotations),
        detail=(
            f"keep fill >= {config.min_rotated_fill:g} of the axis-aligned box "
            f"and true aspect <= {config.max_rotated_aspect:g}; adopt the "
            f"{adopted_shape} only where it is >= "
            f"{config.min_feature_tightness:.0%} feature and elongated "
            f">= {config.min_rotated_elongation:g}"
            + (
                f"; edge-aligned rotation adopted {edge_adopted} and declined "
                f"{edge_unadopted} closer than {config.rotation_edge_min_gap_px:g} px, "
                f"outside {config.rotation_edge_search_px} px, or "
                f"with a local UV tangent changing by more than "
                f"{config.max_rotation_edge_angle_degrees:g} degrees across fit scales; "
                f"opposite side support must be <= "
                f"{config.max_opposite_rotation_edge_fraction:.0%}"
                if config.enable_edge_aligned_rotation
                else ""
            )
            + (f"; {unfitted} had too few feature pixels to fit" if unfitted else "")
            + (f"; {unadopted} kept their axis-aligned box" if unadopted else "")
        ),
    )


def _step_region_flatness(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig,
    state: DetectionState,
) -> tuple[DetectionState, DetectionStage]:
    """Drop regions with no relief in them at all."""
    if not config.enable_region_flatness_filter or not state.groups:
        return state, DetectionStage(
            key="region_flatness",
            title="Region flatness",
            kept=tuple(state.groups),
            rotations=tuple(state.rotations),
            detail=(
                "disabled"
                if not config.enable_region_flatness_filter
                else "no regions to test"
            ),
        )

    response = state.relief_bridge_response
    if response is None:
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        response = edge_response(grey, config)
    domain = uv_mask if uv_mask is not None else np.ones(response.shape[:2], bool)
    inside = response[domain]
    reference = float(np.median(inside)) if inside.size else 0.0

    kept: list[tuple[int, int, int, int]] = []
    rejected: list[tuple[int, int, int, int]] = []
    rotations: list[tuple[tuple[float, float], ...] | None] = []
    unjudged = 0
    for index, group in enumerate(state.groups):
        outline = state.rotations[index] if index < len(state.rotations) else None
        relief = region_flatness(response, uv_mask, group, config, reference)
        if relief is None:
            unjudged += 1
            kept.append(group)
            rotations.append(outline)
        elif relief < config.min_region_relief:
            rejected.append(group)
        else:
            kept.append(group)
            rotations.append(outline)

    return DetectionState(
        state.boxes, kept, rotations=rotations,
        relief_bridge_response=state.relief_bridge_response,
        island_bits=state.island_bits,
        relief_edge_mask=state.relief_edge_mask,
    ), DetectionStage(
        key="region_flatness",
        rotations=tuple(rotations),
        title="Region flatness",
        kept=tuple(kept),
        rejected=tuple(rejected),
        detail=(
            f"reject below {config.min_region_relief:g}x the atlas's median "
            f"gradient at the {config.region_flatness_percentile:g}th percentile "
            "inside the region"
            + (f"; {unjudged} had too little domain to judge" if unjudged else "")
        ),
    )


def relief_glyph_measures(
    mask: np.ndarray,
    bounds: tuple[int, int, int, int],
    config: MserConfig,
) -> tuple[float, int, float, float, float] | None:
    """Measure compact-mark or aligned-fragment evidence within one region."""
    x, y, w, h = bounds
    height, width = mask.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, width), min(y + h, height)
    if x1 <= x0 or y1 <= y0:
        return None
    window = mask[y0:y1, x0:x1].astype(np.uint8)
    if not bool(window.any()):
        return None
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        window, connectivity=8,
    )
    valid = np.flatnonzero(
        stats[1:, cv2.CC_STAT_AREA] >= max(config.relief_glyph_min_component_px, 1)
    ) + 1
    coverage = float(window.mean())
    if not len(valid):
        return coverage, 0, 1.0, float("inf"), 0.0
    areas = stats[valid, cv2.CC_STAT_AREA].astype(np.float64)
    dominant = float(areas.max() / max(areas.sum(), 1.0))
    points = centroids[valid]
    if len(points) < 3:
        scatter = float("inf")
    else:
        singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
        scatter = float(singular[1] / max(singular[0], 1e-6))
    valid_labels = np.zeros(count, dtype=bool)
    valid_labels[valid] = True
    edge_y, edge_x = np.nonzero(valid_labels[_labels])
    if len(edge_x) < 3:
        outline_scatter = 0.0
    else:
        edge_points = np.column_stack((edge_x, edge_y)).astype(np.float64)
        singular = np.linalg.svd(
            edge_points - edge_points.mean(axis=0), compute_uv=False,
        )
        outline_scatter = float(singular[1] / max(singular[0], 1e-6))
    return coverage, len(valid), dominant, scatter, outline_scatter


def has_relief_outline_structure(
    measures: tuple[float, int, float, float, float] | None,
    config: MserConfig,
) -> bool:
    """Return whether sparse edge evidence forms a two-dimensional logo."""
    if measures is None:
        return False
    coverage, components, dominant, _line_scatter, outline_scatter = measures
    return (
        coverage >= float(config.relief_glyph_min_outline_edge_coverage)
        and components >= max(int(config.relief_glyph_min_outline_components), 1)
        and dominant
        <= float(config.relief_glyph_max_outline_dominant_component_fraction)
        and outline_scatter >= float(config.relief_glyph_min_outline_scatter)
    )


def _step_relief_glyph_structure(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig,
    state: DetectionState,
) -> tuple[DetectionState, DetectionStage]:
    """Reject broad relief/seam regions that do not have glyph structure."""
    if not config.enable_relief_glyph_filter or not state.groups:
        return state, DetectionStage(
            key="relief_glyph_structure",
            title="Relief glyph structure",
            kept=tuple(state.groups),
            rotations=tuple(state.rotations),
            detail=("disabled" if not config.enable_relief_glyph_filter else "no regions to test"),
        )
    mask = state.relief_edge_mask
    if mask is None:
        if state.relief_bridge_response is not None:
            mask, _response = edge_mask_from_response(
                state.relief_bridge_response, uv_mask, config,
            )
        else:
            grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mask, _response = edge_mask(grey, uv_mask, config)
    if mask is None:  # keeps static typing honest; the branches always fill it.
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask, _response = edge_mask(grey, uv_mask, config)
    kept: list[tuple[int, int, int, int]] = []
    rejected: list[tuple[int, int, int, int]] = []
    rotations: list[tuple[tuple[float, float], ...] | None] = []
    compact_kept = 0
    line_kept = 0
    outline_kept = 0
    for index, group in enumerate(state.groups):
        measures = relief_glyph_measures(mask, group, config)
        outline = state.rotations[index] if index < len(state.rotations) else None
        if measures is None:
            rejected.append(group)
            continue
        coverage, components, dominant, scatter, outline_scatter = measures
        compact = coverage >= float(config.relief_glyph_min_compact_edge_coverage)
        line = (
            components >= max(int(config.relief_glyph_min_line_components), 3)
            and dominant <= float(config.relief_glyph_max_dominant_component_fraction)
            and scatter <= float(config.relief_glyph_max_line_scatter)
        )
        outlined = has_relief_outline_structure(measures, config)
        if not compact and not line and not outlined:
            rejected.append(group)
            continue
        compact_kept += int(compact)
        line_kept += int(line and not compact)
        outline_kept += int(outlined and not compact and not line)
        kept.append(group)
        rotations.append(outline)
    return DetectionState(
        state.boxes, kept, rotations=rotations,
        relief_bridge_response=state.relief_bridge_response,
        island_bits=state.island_bits,
        relief_edge_mask=mask,
    ), DetectionStage(
        key="relief_glyph_structure",
        title="Relief glyph structure",
        kept=tuple(kept),
        rejected=tuple(rejected),
        rotations=tuple(rotations),
        detail=(
            f"keep compact edge coverage >= {config.relief_glyph_min_compact_edge_coverage:.0%} "
            f"or >= {config.relief_glyph_min_line_components} aligned components "
            f"(scatter <= {config.relief_glyph_max_line_scatter:g}, dominant <= "
            f"{config.relief_glyph_max_dominant_component_fraction:.0%}); "
            f"or outlined edge scatter >= {config.relief_glyph_min_outline_scatter:g} "
            f"with >= {config.relief_glyph_min_outline_edge_coverage:.0%} coverage; "
            f"{compact_kept} compact, {line_kept} aligned, {outline_kept} outlined"
        ),
    )


def _step_ring_smoothness(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig,
    state: DetectionState,
) -> tuple[DetectionState, DetectionStage]:
    """Keep only regions whose surroundings are plain material."""
    if not config.enable_ring_smoothness_filter or not state.groups:
        return state, DetectionStage(
            key="ring_smoothness",
            title="Ring smoothness",
            kept=tuple(state.groups),
            rotations=tuple(state.rotations),
            detail=(
                "disabled"
                if not config.enable_ring_smoothness_filter
                else "no regions to test"
            ),
        )

    response = state.relief_bridge_response
    if response is None:
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        response = edge_response(grey, config)
    domain = uv_mask if uv_mask is not None else np.ones(response.shape[:2], bool)
    inside = response[domain]
    reference = float(np.median(inside)) if inside.size else 0.0

    kept: list[tuple[int, int, int, int]] = []
    rejected: list[tuple[int, int, int, int]] = []
    rotations: list[tuple[tuple[float, float], ...] | None] = []
    unjudged = 0
    for index, group in enumerate(state.groups):
        outline = state.rotations[index] if index < len(state.rotations) else None
        hull = (
            state.feature_hulls[index]
            if index < len(state.feature_hulls)
            else None
        )
        roughness = ring_roughness(response, uv_mask, group, config, reference, hull)
        if roughness is None:
            # Nothing was measured, so nothing is concluded.
            unjudged += 1
            kept.append(group)
            rotations.append(outline)
        elif roughness > config.max_ring_roughness:
            rejected.append(group)
        else:
            kept.append(group)
            rotations.append(outline)

    return DetectionState(
        state.boxes, kept, rotations=rotations,
        relief_bridge_response=state.relief_bridge_response,
        island_bits=state.island_bits,
        relief_edge_mask=state.relief_edge_mask,
    ), DetectionStage(
        key="ring_smoothness",
        rotations=tuple(rotations),
        title="Ring smoothness",
        kept=tuple(kept),
        rejected=tuple(rejected),
        detail=(
            f"reject above {config.max_ring_roughness:g}x the atlas's median "
            f"gradient, measured over a {config.ring_smoothness_width_px} px ring "
            f"held {config.ring_smoothness_margin_px} px off the feature hull "
            "when available, otherwise the region"
            + (f"; {unjudged} had too little domain to judge" if unjudged else "")
        ),
    )


def _step_blob_shape(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig,
    state: DetectionState,
) -> tuple[DetectionState, DetectionStage]:
    """Reject solid blob-like regions with little distinctive shape."""
    if (
        not config.enable_blob_shape_filter
        and not config.enable_ring_smoothness_filter
    ) or not state.groups:
        return state, DetectionStage(
            key="blob_shape",
            title="Blob shape",
            kept=tuple(state.groups),
            rotations=tuple(state.rotations),
            detail=(
                "disabled"
                if not config.enable_blob_shape_filter
                else "no regions to test"
            ),
        )

    kept: list[tuple[int, int, int, int]] = []
    rejected: list[tuple[int, int, int, int]] = []
    rotations: list[tuple[tuple[float, float], ...] | None] = []
    hulls: list[RegionFeatureHull | None] = []
    unjudged = 0
    skipped_small = 0
    detailed = 0
    for index, group in enumerate(state.groups):
        outline = state.rotations[index] if index < len(state.rotations) else None
        area = max(group[2] * group[3], 0)
        hull = region_feature_hull(image, uv_mask, group, config)
        if not config.enable_blob_shape_filter:
            kept.append(group)
            rotations.append(outline)
            hulls.append(hull)
            continue
        if area < max(config.min_blob_region_area_px, 0):
            skipped_small += 1
            kept.append(group)
            rotations.append(outline)
            hulls.append(hull)
            continue
        if hull is None:
            unjudged += 1
            kept.append(group)
            rotations.append(outline)
            hulls.append(None)
            continue
        fill = float(hull.feature_area_px) / max(hull.hull_area_px, 1e-6)
        colour_variation = hull.colour_variation
        if (
            fill >= config.max_blob_hull_fill
            and colour_variation < config.min_blob_internal_colour_variation
        ):
            rejected.append(group)
        elif fill >= config.max_blob_hull_fill:
            detailed += 1
            kept.append(group)
            rotations.append(outline)
            hulls.append(hull)
        else:
            kept.append(group)
            rotations.append(outline)
            hulls.append(hull)

    detail = (
        "disabled; feature hulls cached for ring smoothness"
        if not config.enable_blob_shape_filter
        else (
            f"for regions >= {config.min_blob_region_area_px} px^2, reject "
            f"feature/hull fill >= {config.max_blob_hull_fill:.0%} only when "
            f"internal colour range < {config.min_blob_internal_colour_variation:g}; "
            f"background from a {config.region_feature_ring_width_px} px ring "
            f"held {config.region_feature_ring_margin_px} px off the region"
            + (f"; {detailed} blob-like region{'s' if detailed != 1 else ''} kept for internal detail" if detailed else "")
            + (f"; {skipped_small} smaller region{'s' if skipped_small != 1 else ''} skipped" if skipped_small else "")
            + (f"; {unjudged} had too little feature/ring domain to judge" if unjudged else "")
        )
    )

    return DetectionState(
        state.boxes, kept, rotations=rotations, feature_hulls=hulls,
        relief_bridge_response=state.relief_bridge_response,
        island_bits=state.island_bits,
        relief_edge_mask=state.relief_edge_mask,
    ), DetectionStage(
        key="blob_shape",
        title="Blob shape",
        kept=tuple(kept),
        rejected=tuple(rejected),
        rotations=tuple(rotations),
        detail=detail,
    )


def _step_feature_extension(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig,
    state: DetectionState,
) -> tuple[DetectionState, DetectionStage]:
    """Reject regions that are only a slice of a larger connected feature."""
    if not config.enable_feature_extension_filter or not state.groups:
        return state, DetectionStage(
            key="feature_extension",
            title="Feature extension",
            kept=tuple(state.groups),
            rotations=tuple(state.rotations),
            detail=(
                "disabled"
                if not config.enable_feature_extension_filter
                else "no regions to test"
            ),
        )

    kept: list[tuple[int, int, int, int]] = []
    rejected: list[tuple[int, int, int, int]] = []
    rotations: list[tuple[tuple[float, float], ...] | None] = []
    unjudged = 0
    threshold = max(float(config.feature_extension_min_ratio), 0.0)
    for index, group in enumerate(state.groups):
        outline = state.rotations[index] if index < len(state.rotations) else None
        measure = feature_extension_measure(image, uv_mask, group, config)
        if measure is None:
            unjudged += 1
            kept.append(group)
            rotations.append(outline)
        elif measure.extension_ratio >= threshold:
            rejected.append(group)
        else:
            kept.append(group)
            rotations.append(outline)

    return DetectionState(
        state.boxes, kept, rotations=rotations,
        relief_bridge_response=state.relief_bridge_response,
        island_bits=state.island_bits,
        relief_edge_mask=state.relief_edge_mask,
    ), DetectionStage(
        key="feature_extension",
        title="Feature extension",
        kept=tuple(kept),
        rejected=tuple(rejected),
        rotations=tuple(rotations),
        detail=(
            "reject connected feature continuation where outside/inside "
            f">= {threshold:.0%}, searched {config.feature_extension_context_px} px past it"
            + (f"; {unjudged} had too little feature/ring domain to judge" if unjudged else "")
        ),
    )


def _step_text_line(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig,
    state: DetectionState,
) -> tuple[DetectionState, DetectionStage]:
    """Keep only regions holding several marks in a row."""
    if not config.enable_text_line_filter or not state.groups:
        return state, DetectionStage(
            key="text_line",
            title="Text lines",
            kept=tuple(state.groups),
            rotations=tuple(state.rotations),
            detail=(
                "disabled"
                if not config.enable_text_line_filter
                else "no regions to test"
            ),
        )

    mask = state.relief_edge_mask
    if mask is None:
        if state.relief_bridge_response is not None:
            mask, _response = edge_mask_from_response(
                state.relief_bridge_response, uv_mask, config,
            )
        else:
            grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mask, _response = edge_mask(grey, uv_mask, config)
    if mask is None:  # keeps static typing honest; the branches always fill it.
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask, _response = edge_mask(grey, uv_mask, config)

    kept: list[tuple[int, int, int, int]] = []
    rejected: list[tuple[int, int, int, int]] = []
    rotations: list[tuple[tuple[float, float], ...] | None] = []
    for index, group in enumerate(state.groups):
        outline = state.rotations[index] if index < len(state.rotations) else None
        measures = text_line_measures(mask, group, config)
        if measures is None:
            rejected.append(group)
            continue
        characters, scatter = measures
        if (
            characters < config.text_min_characters
            or scatter > config.max_baseline_scatter
        ):
            rejected.append(group)
            continue
        kept.append(group)
        rotations.append(outline)

    return DetectionState(
        state.boxes, kept, rotations=rotations,
        relief_bridge_response=state.relief_bridge_response,
        island_bits=state.island_bits,
        relief_edge_mask=mask,
    ), DetectionStage(
        key="text_line",
        title="Text lines",
        kept=tuple(kept),
        rejected=tuple(rejected),
        rotations=tuple(rotations),
        detail=(
            f"keep >= {config.text_min_characters} components of "
            f">= {config.text_min_component_px} px whose centres scatter "
            f"<= {config.max_baseline_scatter:g} from a straight line"
        ),
    )


def _step_size(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig,
    state: DetectionState,
) -> tuple[DetectionState, DetectionStage]:
    groups, rejected = filter_groups_by_final_size(state.groups, config)
    rotations = _rotations_for_subset(state.rotations, state.groups, groups)
    return DetectionState(
        state.boxes, groups, rotations=rotations,
        relief_bridge_response=state.relief_bridge_response,
        island_bits=state.island_bits,
        relief_edge_mask=state.relief_edge_mask,
    ), DetectionStage(
        key="size",
        title="Final size",
        kept=tuple(groups),
        rejected=tuple(rejected),
        rotations=tuple(rotations),
        detail=(
            f"min {config.final_min_width_px}x{config.final_min_height_px} px, "
            f"min area {config.final_min_area_px} px^2, "
            f"aspect <= {config.final_max_aspect}"
        ),
    )


def _region_shape_and_coverage(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    group: tuple[int, int, int, int],
    config: MserConfig,
    domain: UvDomainIndex | None = None,
    circle_cache: dict[tuple[MserConfig, tuple[int, int, int, int]], int | None] | None = None,
) -> tuple[int | None, float]:
    """Infer a region's circle, then measure that circle or its rectangle."""
    radius = cached_inscribed_circle_radius(
        image, uv_mask, group, config, circle_cache,
    )
    if radius is not None:
        x, y, w, h = group
        return radius, circle_uv_coverage(
            uv_mask,
            (x + w / 2.0, y + h / 2.0),
            float(radius),
            domain,
        )
    return None, box_uv_coverage(uv_mask, group, image.shape, domain)


def _step_region_domain(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig,
    state: DetectionState,
) -> tuple[DetectionState, DetectionStage]:
    """Recover failed groups with a full-distance pass and a half-distance retry.

    Every initial group first gets its circle decision and shaped-domain test.
    A passing group remains intact.  A failing group is split back into its
    recovery units: after overlap grouping, those are the overlap-group boxes.
    A unit failing the same shaped-domain test is split once more into the raw
    boxes absorbed by overlap grouping.  Valid descendants are then regrouped
    while enforcing the shaped-domain threshold on every growth step.  Older
    raw-member recovery still uses the full-distance rebuild plus one
    half-distance retry before final rejection.
    """
    candidates = state.candidates or [
        GroupCandidate(bounds=group, members=(group,)) for group in state.groups
    ]
    domain = state.domain or build_uv_domain_index(uv_mask)
    if not config.enable_region_domain_filter or uv_mask is None:
        return state, DetectionStage(
            key="region_domain",
            title="Domain recovery",
            kept=tuple(state.groups),
            detail="disabled" if uv_mask is not None else "no UV mask",
        )

    kept_candidates: list[GroupCandidate] = []
    rejected: list[tuple[int, int, int, int]] = []
    failed_initial = 0
    removed_units = 0
    overlap_members_recovered = 0
    half_distance_attempts = 0
    half_distance_rescued = 0
    strict_recovery_groups = 0
    final_rejected = 0
    half_merge_distance = max(config.merge_distance_px // 2, 0)
    half_distance_config = replace(
        config,
        merge_distance_px=half_merge_distance,
    )

    for candidate in candidates:
        _radius, initial_coverage = _region_shape_and_coverage(
            image, uv_mask, candidate.bounds, config, domain, state.circle_radii,
        )
        if initial_coverage >= config.min_region_uv_coverage:
            kept_candidates.append(candidate)
            continue

        failed_initial += 1
        rejected.append(candidate.bounds)

        valid_units: list[tuple[int, int, int, int]] = []
        valid_unit_set: set[tuple[int, int, int, int]] = set()
        candidate_overlap_members_recovered = 0
        for unit_index, unit in enumerate(candidate.recovery_units()):
            _unit_radius, unit_coverage = _region_shape_and_coverage(
                image, uv_mask, unit, config, domain, state.circle_radii,
            )
            if unit_coverage < config.min_region_uv_coverage:
                rejected.append(unit)
                removed_units += 1
                for member in candidate.source_members_for_unit(unit_index):
                    if member == unit:
                        continue
                    _member_radius, member_coverage = _region_shape_and_coverage(
                        image, uv_mask, member, config, domain, state.circle_radii,
                    )
                    if member_coverage < config.min_region_uv_coverage:
                        rejected.append(member)
                        removed_units += 1
                        continue
                    if member not in valid_unit_set:
                        valid_units.append(member)
                        valid_unit_set.add(member)
                        overlap_members_recovered += 1
                        candidate_overlap_members_recovered += 1
                continue
            if unit not in valid_unit_set:
                valid_units.append(unit)
                valid_unit_set.add(unit)

        if not valid_units:
            continue

        if candidate.units:
            recovered = strict_region_recovery_candidates(
                valid_units,
                image,
                config,
                uv_mask,
                domain,
                include_overlap=candidate_overlap_members_recovered > 0,
                circle_cache=state.circle_radii,
                island_bits=state.island_bits,
            )
            kept_candidates.extend(recovered)
            strict_recovery_groups += len(recovered)
            continue

        rebuilt = union_region_group_candidates(
            np.asarray(valid_units, dtype=np.int32),
            image,
            config,
            uv_mask,
            domain,
            raster_tolerance=False,
            defer_domain_validation=True,
            circle_cache=state.circle_radii,
            island_bits=state.island_bits,
        )
        for rebuilt_candidate in rebuilt:
            _rebuilt_radius, rebuilt_coverage = _region_shape_and_coverage(
                image, uv_mask, rebuilt_candidate.bounds, config, domain,
                state.circle_radii,
            )
            if rebuilt_coverage >= config.min_region_uv_coverage:
                kept_candidates.append(rebuilt_candidate)
                continue

            # The full-distance rebuild is still too broad.  Do not discard all
            # of its individually valid members: reduce only this recovery pass
            # to half the normal merge distance and try once more.
            rejected.append(rebuilt_candidate.bounds)
            half_distance_attempts += 1
            retry_candidates = union_region_group_candidates(
                np.asarray(rebuilt_candidate.recovery_units(), dtype=np.int32),
                image,
                half_distance_config,
                uv_mask,
                domain,
                raster_tolerance=False,
                defer_domain_validation=True,
                circle_cache=state.circle_radii,
                island_bits=state.island_bits,
            )
            rescued_here = 0
            for retry_candidate in retry_candidates:
                _retry_radius, retry_coverage = _region_shape_and_coverage(
                    image, uv_mask, retry_candidate.bounds, config, domain,
                    state.circle_radii,
                )
                if retry_coverage < config.min_region_uv_coverage:
                    rejected.append(retry_candidate.bounds)
                    final_rejected += 1
                    continue
                kept_candidates.append(retry_candidate)
                rescued_here += 1
            half_distance_rescued += rescued_here

    kept_candidates.sort(key=lambda candidate: (candidate.bounds[1], candidate.bounds[0]))
    kept = [candidate.bounds for candidate in kept_candidates]
    return DetectionState(
        state.boxes, kept, kept_candidates, domain,
        relief_bridge_response=state.relief_bridge_response,
        island_bits=state.island_bits,
        relief_edge_mask=state.relief_edge_mask,
    ), DetectionStage(
        key="region_domain",
        title="Domain recovery",
        kept=tuple(kept),
        rejected=tuple(rejected),
        detail=(
            f"first shaped test >= {config.min_region_uv_coverage:.0%}; "
            f"{failed_initial} initial group{'s' if failed_initial != 1 else ''} split, "
            f"{removed_units} invalid recovery unit{'s' if removed_units != 1 else ''} removed; "
            f"{overlap_members_recovered} overlap member"
            f"{'s' if overlap_members_recovered != 1 else ''} recovered; "
            f"{strict_recovery_groups} strict recovery group"
            f"{'s' if strict_recovery_groups != 1 else ''} kept; "
            f"{half_distance_attempts} broad rebuild"
            f"{'s' if half_distance_attempts != 1 else ''} retried at "
            f"{half_merge_distance} px, {half_distance_rescued} retry group"
            f"{'s' if half_distance_rescued != 1 else ''} rescued, "
            f"{final_rejected} final group{'s' if final_rejected != 1 else ''} rejected"
        ),
    )



def _shaped_regions_overlap(
    first: tuple[int, int, int, int],
    first_radius: int | None,
    second: tuple[int, int, int, int],
    second_radius: int | None,
) -> bool:
    """Return whether two final shaped regions overlap with positive area.

    Circular groups are tested as discs rather than by their square source
    bounds.  Tangential contact alone is not overlap: the shapes must share
    actual interior area.
    """
    ax, ay, aw, ah = first
    bx, by, bw, bh = second

    if first_radius is None and second_radius is None:
        return (
            max(ax, bx) < min(ax + aw, bx + bw)
            and max(ay, by) < min(ay + ah, by + bh)
        )

    if first_radius is not None and second_radius is not None:
        first_centre_x = ax + aw / 2.0
        first_centre_y = ay + ah / 2.0
        second_centre_x = bx + bw / 2.0
        second_centre_y = by + bh / 2.0
        distance_sq = (
            (first_centre_x - second_centre_x) ** 2
            + (first_centre_y - second_centre_y) ** 2
        )
        return distance_sq < float(first_radius + second_radius) ** 2

    # Keep the circle in the first slot so the circle/rectangle calculation is
    # written once.
    if first_radius is None:
        return _shaped_regions_overlap(second, second_radius, first, first_radius)

    centre_x = ax + aw / 2.0
    centre_y = ay + ah / 2.0
    nearest_x = min(max(centre_x, bx), bx + bw)
    nearest_y = min(max(centre_y, by), by + bh)
    distance_sq = (centre_x - nearest_x) ** 2 + (centre_y - nearest_y) ** 2
    return distance_sq < float(first_radius) ** 2


def _shaped_region_bounds(
    group: tuple[int, int, int, int],
    radius: int | None,
    image_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    """Return image-clamped bounds enclosing the rectangle or inferred disc."""
    x, y, w, h = group
    if radius is None:
        return clamp_group(group, image_shape) or group

    centre_x = x + w / 2.0
    centre_y = y + h / 2.0
    x0 = int(math.floor(centre_x - radius))
    y0 = int(math.floor(centre_y - radius))
    x1 = int(math.ceil(centre_x + radius))
    y1 = int(math.ceil(centre_y + radius))
    shape = (x0, y0, x1 - x0, y1 - y0)
    return clamp_group(shape, image_shape) or group


def _shaped_region_uv_coverage(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    group: tuple[int, int, int, int],
    radius: int | None,
    domain: UvDomainIndex | None = None,
) -> float:
    """Return UV-domain coverage for one already-shaped region."""
    if uv_mask is None:
        return 1.0
    if radius is not None:
        x, y, w, h = group
        return circle_uv_coverage(
            uv_mask,
            (x + w / 2.0, y + h / 2.0),
            float(radius),
            domain,
        )
    return box_uv_coverage(uv_mask, group, image.shape, domain)


def _step_overlap_group(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig,
    state: DetectionState,
) -> tuple[DetectionState, DetectionStage]:
    """Force a compact control cluster into one already-validated circle.

    This deliberately is *not* a second general proximity pass: Domain
    recovery remains terminal for ordinary rectangles.  It only reconnects a
    circular control whose independent cardinal glyphs cannot share a row or
    column, or whose vertical/horizontal arms overlap as a cross (a D-pad is
    the canonical case).  The enclosing rectangle must itself earn an
    inscribed circle and satisfy that circle's UV-domain rule, so this cannot
    recreate a broad panel group.
    """
    groups = list(state.groups)
    domain = state.domain or build_uv_domain_index(uv_mask)

    if not config.enable_circular_groups or len(groups) < 2:
        return DetectionState(
            state.boxes, groups, domain=domain,
            relief_bridge_response=state.relief_bridge_response,
        island_bits=state.island_bits,
            relief_edge_mask=state.relief_edge_mask,
            circle_radii=state.circle_radii,
        ), DetectionStage(
            key="overlap_group",
            title="Post-circle forced merge",
            kept=tuple(groups),
            detail=(
                "circular groups disabled"
                if not config.enable_circular_groups
                else "fewer than two recovered groups; nothing to circle-merge"
            ),
        )

    distance = max(int(config.merge_distance_px), 0)

    def centre_and_extent(
        group: tuple[int, int, int, int],
    ) -> tuple[float, float, float]:
        x, y, width, height = group
        return (
            x + width / 2.0,
            y + height / 2.0,
            math.hypot(width, height) / 2.0,
        )

    centres = [centre_and_extent(group) for group in groups]

    def box_gap(
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
    ) -> int:
        ax, ay, aw, ah = first
        bx, by, bw, bh = second
        return max(
            bx - (ax + aw), ax - (bx + bw),
            by - (ay + ah), ay - (by + bh), 0,
        )

    def bounds_for(indices: tuple[int, ...]) -> tuple[int, int, int, int]:
        x0 = min(groups[index][0] for index in indices)
        y0 = min(groups[index][1] for index in indices)
        x1 = max(groups[index][0] + groups[index][2] for index in indices)
        y1 = max(groups[index][1] + groups[index][3] for index in indices)
        return x0, y0, x1 - x0, y1 - y0

    def cardinal_spread(
        indices: tuple[int, ...], bounds: tuple[int, int, int, int],
    ) -> int:
        """Count occupied quadrants around the proposed circular control."""
        x, y, width, height = bounds
        centre_x = x + width / 2.0
        centre_y = y + height / 2.0
        quadrants: set[int] = set()
        for index in indices:
            point_x, point_y, _extent = centres[index]
            angle = math.atan2(point_y - centre_y, point_x - centre_x)
            quadrants.add(int(math.floor((angle + math.pi) / (math.pi / 2.0))) % 4)
        return len(quadrants)

    # A pair of opposing cardinal marks reveals the control's centre and
    # radius.  Collect every recovered group whose *centre* fits that trial
    # disc, rather than taking one broad proximity component: a nearby label
    # cannot poison a valid D-pad cluster merely through a short chain.
    candidate_sets: set[tuple[int, ...]] = set()
    for left in range(len(groups)):
        left_x, left_y, left_extent = centres[left]
        for right in range(left + 1, len(groups)):
            if box_gap(groups[left], groups[right]) > distance:
                continue
            right_x, right_y, right_extent = centres[right]
            centre_x = (left_x + right_x) / 2.0
            centre_y = (left_y + right_y) / 2.0
            trial_radius = (
                math.hypot(left_x - right_x, left_y - right_y) / 2.0
                + max(left_extent, right_extent)
                + 2.0  # tolerate half-open raster bounds at the disc edge
            )
            indices = tuple(
                index
                for index, (point_x, point_y, _extent) in enumerate(centres)
                if math.hypot(point_x - centre_x, point_y - centre_y)
                <= trial_radius
            )
            if len(indices) >= 3:
                candidate_sets.add(indices)
            # A vertical and horizontal D-pad arm overlap at the centre.  The
            # pair has no useful centre-line direction, but its *combined*
            # square can still be a valid circle.  Keep this narrow exception
            # separate from generic two-mark proximity, which remains banned.
            if intersection_area(groups[left], groups[right]) > 0:
                candidate_sets.add((left, right))

    accepted: list[tuple[tuple[int, ...], tuple[int, int, int, int]]] = []
    for indices in candidate_sets:
        bounds = bounds_for(indices)
        width, height = bounds[2], bounds[3]
        if width <= 0 or height <= 0:
            continue
        if min(width, height) / max(width, height) < config.circular_group_min_squareness:
            continue
        if len(indices) == 2:
            if intersection_area(groups[indices[0]], groups[indices[1]]) <= 0:
                continue
        elif cardinal_spread(indices, bounds) < 3:
            continue
        radius = cached_inscribed_circle_radius(
            image, uv_mask, bounds, config, state.circle_radii,
        )
        if radius is None:
            continue
        coverage = circle_uv_coverage(
            uv_mask,
            (bounds[0] + bounds[2] / 2.0, bounds[1] + bounds[3] / 2.0),
            float(radius), domain,
        )
        if coverage < config.min_region_uv_coverage:
            continue
        accepted.append((indices, bounds))

    # Choose the most complete valid control first.  Its members are consumed,
    # which keeps two nearby circular controls independent.
    accepted.sort(key=lambda item: (-len(item[0]), item[1][1], item[1][0]))
    consumed: set[int] = set()
    merged: list[tuple[int, int, int, int]] = []
    absorbed = 0
    for indices, bounds in accepted:
        if any(index in consumed for index in indices):
            continue
        consumed.update(indices)
        merged.append(bounds)
        absorbed += len(indices) - 1
    merged.extend(group for index, group in enumerate(groups) if index not in consumed)
    merged.sort(key=lambda group: (group[1], group[0]))

    return DetectionState(
        state.boxes, merged, domain=domain,
        relief_bridge_response=state.relief_bridge_response,
        island_bits=state.island_bits,
        relief_edge_mask=state.relief_edge_mask,
        circle_radii=state.circle_radii,
    ), DetectionStage(
        key="overlap_group",
        title="Post-circle forced merge",
        kept=tuple(merged),
        detail=(
            f"forced {absorbed} cardinal fragment{'s' if absorbed != 1 else ''} "
            "into UV-valid inscribed circles; ordinary rectangle proximity is not retried"
        ),
        adjusted=absorbed,
    )

def _step_final_padding(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig,
    state: DetectionState,
) -> tuple[DetectionState, DetectionStage]:
    """Pad final regions without feeding the larger bounds back into detection.

    Rectangles are expanded independently to the image edges.  A circular
    region is expanded symmetrically so its centre remains unchanged; when an
    image edge is too close, its usable padding is reduced accordingly.
    """
    requested = max(int(config.final_region_padding_px), 0)
    padded: list[tuple[int, int, int, int]] = []
    circles: list[int | None] = []
    adjusted = 0
    image_height, image_width = image.shape[:2]

    for group in state.groups:
        radius = cached_inscribed_circle_radius(
            image, uv_mask, group, config, state.circle_radii,
        )
        x, y, w, h = group

        if radius is not None:
            centre_x = x + w / 2.0
            centre_y = y + h / 2.0
            # Keep the inferred circle centre fixed.  Existing circle padding can
            # already extend beyond the rectangular group, so bound the extra
            # padding against the circle itself rather than just the group box.
            edge_room = min(
                centre_x - radius,
                centre_y - radius,
                image_width - (centre_x + radius),
                image_height - (centre_y + radius),
            )
            usable = min(requested, max(int(math.floor(edge_room)), 0))
            result = (
                max(x - usable, 0),
                max(y - usable, 0),
                min(w + usable * 2, image_width),
                min(h + usable * 2, image_height),
            )
            circles.append(radius + usable)
        else:
            result = clamp_group(
                (x - requested, y - requested, w + requested * 2, h + requested * 2),
                image.shape,
            )
            if result is None:
                # A valid incoming region cannot normally reach this branch, but
                # retaining it is safer than silently deleting a final detection.
                result = group
            circles.append(None)

        if result != group:
            adjusted += 1
        padded.append(result)

    detail = (
        f"expanded by {requested} px on each side, clamped to the image"
        if requested
        else "0 px; final bounds unchanged"
    )
    # Outlines pass through unpadded: they describe the feature, not the box
    # that was grown around it, and growing one would misreport the fit.
    return DetectionState(
        state.boxes, padded, rotations=list(state.rotations),
        relief_bridge_response=state.relief_bridge_response,
        island_bits=state.island_bits,
        relief_edge_mask=state.relief_edge_mask,
    ), DetectionStage(
        key="final_padding",
        title="Final padding",
        kept=tuple(padded),
        detail=detail,
        adjusted=adjusted,
        circles=tuple(circles),
        rotations=tuple(state.rotations),
    )


def relief_text_measures(
    response: np.ndarray,
    uv_mask: np.ndarray | None,
    bounds: tuple[int, int, int, int],
    config: MserConfig,
) -> tuple[int, int, bool, float] | None:
    """Count repeated high-relief bands along a candidate's long axis.

    Connected-component counting cannot see individual shallow letters once
    their edges touch.  A projection at a locally chosen response percentile
    can: a word produces several occupied bands along its baseline, whereas a
    seam stays occupied as one continuous band.
    """
    x, y, w, h = bounds
    height, width = response.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, width), min(y + h, height)
    if x1 <= x0 or y1 <= y0:
        return None
    window = response[y0:y1, x0:x1]
    domain = (
        uv_mask[y0:y1, x0:x1].astype(bool)
        if uv_mask is not None else np.ones(window.shape, dtype=bool)
    )
    values = window[domain]
    if not values.size:
        return None
    threshold = float(np.percentile(
        values, min(max(config.relief_text_response_percentile, 0.0), 100.0),
    ))
    feature = (window >= threshold) & domain
    horizontal = (x1 - x0) >= (y1 - y0)
    if horizontal:
        support = feature.sum(axis=0)
        available = domain.sum(axis=0)
    else:
        support = feature.sum(axis=1)
        available = domain.sum(axis=1)
    occupied = support >= np.maximum(
        1, np.ceil(available * float(config.relief_text_projection_min_coverage))
    )
    transitions = np.diff(np.r_[False, occupied, False].astype(np.int8))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    runs = int(len(starts))
    # Long seams and serrations can both create many occupied projection runs.
    # A letter-scale run must also span a material fraction of the cross-axis:
    # AIRBAG has several such bands despite being one connected edge component,
    # whereas an incidental seam has at most one and fine teeth have none.
    cross_axis = (y1 - y0) if horizontal else (x1 - x0)
    minimum_band = max(int(np.ceil(cross_axis * config.relief_text_min_band_scale)), 1)
    substantial = int(sum((end - start) >= minimum_band for start, end in zip(starts, ends)))
    return runs, substantial, horizontal, float(feature.mean())


def _step_relief_text(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig,
    state: DetectionState,
) -> tuple[DetectionState, DetectionStage]:
    """Keep final relief regions whose edge projection reads as text."""
    if not config.enable_relief_text_filter or not state.groups:
        return state, DetectionStage(
            key="relief_text",
            title="Relief text",
            kept=tuple(state.groups),
            rotations=tuple(state.rotations),
            detail=("disabled" if not config.enable_relief_text_filter else "no regions to test"),
        )
    # The edge front end already made this response.  Keeping it through the
    # pipeline makes this terminal classifier an inspection, not another full
    # Laplacian pass.  The fallback preserves the helper's standalone use.
    response = state.relief_bridge_response
    if response is None:
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        response = edge_response(grey, config)
    kept: list[tuple[int, int, int, int]] = []
    rejected: list[tuple[int, int, int, int]] = []
    rotations: list[tuple[tuple[float, float], ...] | None] = []
    outlined_kept = 0
    mask = state.relief_edge_mask
    if mask is None:
        mask, _response = edge_mask_from_response(response, uv_mask, config)
    for index, group in enumerate(state.groups):
        outline = state.rotations[index] if index < len(state.rotations) else None
        measures = relief_text_measures(response, uv_mask, group, config)
        text = (
            measures is not None
            and measures[0] >= max(int(config.relief_text_min_runs), 1)
            and measures[1] >= max(int(config.relief_text_min_runs), 1)
        )
        outlined = has_relief_outline_structure(
            relief_glyph_measures(mask, group, config), config,
        )
        if text or outlined:
            kept.append(group)
            rotations.append(outline)
            outlined_kept += int(outlined and not text)
        else:
            rejected.append(group)
    return DetectionState(
        state.boxes, kept, rotations=rotations,
        relief_bridge_response=state.relief_bridge_response,
        island_bits=state.island_bits,
        relief_edge_mask=state.relief_edge_mask,
    ), DetectionStage(
        key="relief_text",
        title="Relief text",
        kept=tuple(kept),
        rejected=tuple(rejected),
        rotations=tuple(rotations),
        detail=(
            f"keep >= {config.relief_text_min_runs} repeated bands, each with "
            f">= {config.relief_text_min_band_scale:.0%} of the cross-axis, at the "
            f"{config.relief_text_response_percentile:g}th response percentile, "
            "or a previously admissible two-dimensional relief outline; "
            f"{outlined_kept} outlined logo"
            f"{'s' if outlined_kept != 1 else ''} retained"
        ),
    )


# Stages holding assembled regions rather than raw MSER boxes; only these are
# worth shaping, and only these are few enough for it to be cheap.  Overlap
# collapse deliberately stays rectangular: it is only a housekeeping step and
# Initial grouping is the first stage at which a circle can describe the group
# the user will actually tune.
GROUP_STAGE_KEYS = frozenset(
    {
        "grouped",
        "region_domain",
        "overlap_group",
        "pattern_group",
        "rotated_bounds",
        "blob_shape",
        "ring_smoothness",
        "size",
        "feature_extension",
    }
)

PIPELINE_STEPS = (
    _step_mser,
    _step_stroke_width,
    _step_box_filter,
    _step_overlap_box_group,
    _step_grouped,
    _step_region_domain,
    _step_overlap_group,
    _step_pattern_group,
    _step_rotated_bounds,
    _step_region_flatness,
    _step_relief_glyph_structure,
    _step_blob_shape,
    _step_ring_smoothness,
    _step_feature_extension,
    _step_text_line,
    _step_size,
    _step_final_padding,
    _step_relief_text,
)

# Named so the table below cannot drift when a step is inserted.  It was ints
# before; adding stroke width in the middle would have shifted every later
# entry by one, and a parameter left pointing at the wrong step resumes from
# the wrong place and shows stale boxes while it is being tuned.
STEP_INDEX = {
    name: index
    for index, name in enumerate(
        (
            "boxes",
            "stroke_width",
            "box_filter",
            "overlap_box_group",
            "grouped",
            "region_domain",
            "overlap_group",
            "pattern_group",
            "rotated_bounds",
            "region_flatness",
            "relief_glyph_structure",
            "blob_shape",
            "ring_smoothness",
            "feature_extension",
            "text_line",
            "size",
            "final_padding",
            "relief_text",
        )
    )
}
assert len(STEP_INDEX) == len(PIPELINE_STEPS)

# Which step first reads each parameter.  A parameter read by more than one step
# maps to the earliest.  Anything missing here forces a full re-run, so a new
# MserConfig field is safe by default -- it just will not resume.
PARAMETER_STEP = {
    "box_source": STEP_INDEX["boxes"],
    "delta": STEP_INDEX["boxes"],
    "min_area": STEP_INDEX["boxes"],
    "max_area": STEP_INDEX["boxes"],
    "max_variation": STEP_INDEX["boxes"],
    "min_diversity": STEP_INDEX["boxes"],
    "edge_operator": STEP_INDEX["boxes"],
    "edge_kernel_px": STEP_INDEX["boxes"],
    "edge_blur_sigma": STEP_INDEX["boxes"],
    "edge_local_window_px": STEP_INDEX["boxes"],
    "edge_local_k": STEP_INDEX["boxes"],
    "edge_threshold_percentile": STEP_INDEX["boxes"],
    "edge_threshold_floor": STEP_INDEX["boxes"],
    "edge_close_px": STEP_INDEX["boxes"],
    "edge_dilate_px": STEP_INDEX["boxes"],
    "edge_min_component_px": STEP_INDEX["boxes"],
    "foreground_background_bins": STEP_INDEX["boxes"],
    "foreground_background_distance": STEP_INDEX["boxes"],
    "foreground_value_contrast": STEP_INDEX["boxes"],
    "foreground_min_saturation": STEP_INDEX["boxes"],
    "foreground_edge_threshold": STEP_INDEX["boxes"],
    "foreground_open_px": STEP_INDEX["boxes"],
    "foreground_close_px": STEP_INDEX["boxes"],
    "foreground_min_component_px": STEP_INDEX["boxes"],
    "foreground_merge_gap_px": STEP_INDEX["boxes"],
    "foreground_max_coverage": STEP_INDEX["boxes"],
    "foreground_refine_internal_details": STEP_INDEX["boxes"],
    "foreground_detail_inset_px": STEP_INDEX["boxes"],
    "foreground_detail_min_parent_px": STEP_INDEX["boxes"],
    "foreground_detail_min_coverage": STEP_INDEX["boxes"],
    "foreground_detail_min_component_px": STEP_INDEX["boxes"],
    "foreground_detail_replace_area_ratio": STEP_INDEX["boxes"],
    "foreground_detail_max_depth": STEP_INDEX["boxes"],
    "foreground_detail_max_parent_px": STEP_INDEX["boxes"],
    "foreground_detail_max_children": STEP_INDEX["boxes"],
    "opacity_mask_threshold": STEP_INDEX["boxes"],
    "contrast_kernel_px": STEP_INDEX["boxes"],
    "contrast_min_response": STEP_INDEX["boxes"],
    "contrast_percentile": STEP_INDEX["boxes"],
    "contrast_open_px": STEP_INDEX["boxes"],
    "contrast_close_px": STEP_INDEX["boxes"],
    "contrast_min_component_px": STEP_INDEX["boxes"],
    "contrast_merge_gap_px": STEP_INDEX["boxes"],
    "contrast_max_coverage": STEP_INDEX["boxes"],
    "enable_contrast_continuity_grouping": STEP_INDEX["grouped"],
    "contrast_bridge_min_response": STEP_INDEX["grouped"],
    "contrast_bridge_max_high_coverage": STEP_INDEX["grouped"],
    "enable_relief_edge_bridge_grouping": STEP_INDEX["grouped"],
    "relief_bridge_min_response": STEP_INDEX["grouped"],
    "relief_bridge_min_cross_axis_coverage": STEP_INDEX["grouped"],
    "enable_stroke_width_filter": STEP_INDEX["stroke_width"],
    "swt_polarity": STEP_INDEX["stroke_width"],
    "swt_gradient_tolerance_degrees": STEP_INDEX["stroke_width"],
    "swt_min_stroke_px": STEP_INDEX["stroke_width"],
    "swt_max_stroke_px": STEP_INDEX["stroke_width"],
    "swt_median_px": STEP_INDEX["stroke_width"],
    "swt_max_width_variation": STEP_INDEX["stroke_width"],
    "swt_min_coverage": STEP_INDEX["stroke_width"],
    "enable_aspect_ratio_filter": STEP_INDEX["boxes"],
    "min_aspect": STEP_INDEX["boxes"],
    "max_aspect": STEP_INDEX["boxes"],
    "enable_box_feature_filter": STEP_INDEX["box_filter"],
    "min_box_width_px": STEP_INDEX["box_filter"],
    "min_box_height_px": STEP_INDEX["box_filter"],
    "box_feature_colour_tolerance": STEP_INDEX["box_filter"],
    "box_feature_min_domain_px": STEP_INDEX["box_filter"],
    "box_feature_context_px": STEP_INDEX["box_filter"],
    "min_box_uv_coverage": STEP_INDEX["box_filter"],
    "box_min_feature_px": STEP_INDEX["box_filter"],
    "group_axis_center_tolerance": STEP_INDEX["grouped"],
    "merge_distance_px": STEP_INDEX["grouped"],
    "min_group_union_region_px": STEP_INDEX["grouped"],
    "enable_island_bounded_grouping": STEP_INDEX["grouped"],
    "enable_circular_groups": STEP_INDEX["grouped"],
    "circular_group_min_squareness": STEP_INDEX["grouped"],
    "circular_group_colour_tolerance": STEP_INDEX["grouped"],
    "circular_group_max_corner_content": STEP_INDEX["grouped"],
    "enable_region_domain_filter": STEP_INDEX["grouped"],
    "min_region_uv_coverage": STEP_INDEX["region_domain"],
    "enable_pattern_group_filter": STEP_INDEX["pattern_group"],
    "max_pattern_autocorrelation": STEP_INDEX["pattern_group"],
    "pattern_window_scale": STEP_INDEX["pattern_group"],
    "pattern_min_window_px": STEP_INDEX["pattern_group"],
    "pattern_min_period_px": STEP_INDEX["pattern_group"],
    "pattern_max_period_px": STEP_INDEX["pattern_group"],
    "enable_rotated_bounds_filter": STEP_INDEX["rotated_bounds"],
    "rotated_bounds_min_points": STEP_INDEX["rotated_bounds"],
    "min_rotated_fill": STEP_INDEX["rotated_bounds"],
    "max_rotated_aspect": STEP_INDEX["rotated_bounds"],
    "min_feature_tightness": STEP_INDEX["rotated_bounds"],
    "min_rotated_elongation": STEP_INDEX["rotated_bounds"],
    "bounds_shape": STEP_INDEX["rotated_bounds"],
    "enable_edge_aligned_rotation": STEP_INDEX["rotated_bounds"],
    "rotation_edge_min_gap_px": STEP_INDEX["rotated_bounds"],
    "rotation_edge_search_px": STEP_INDEX["rotated_bounds"],
    "rotation_edge_band_px": STEP_INDEX["rotated_bounds"],
    "rotation_edge_min_points": STEP_INDEX["rotated_bounds"],
    "max_rotation_edge_angle_degrees": STEP_INDEX["rotated_bounds"],
    "max_opposite_rotation_edge_fraction": STEP_INDEX["rotated_bounds"],
    "enable_region_flatness_filter": STEP_INDEX["region_flatness"],
    "region_flatness_percentile": STEP_INDEX["region_flatness"],
    "region_flatness_min_domain_px": STEP_INDEX["region_flatness"],
    "min_region_relief": STEP_INDEX["region_flatness"],
    "enable_relief_glyph_filter": STEP_INDEX["relief_glyph_structure"],
    "relief_glyph_min_component_px": STEP_INDEX["relief_glyph_structure"],
    "relief_glyph_min_line_components": STEP_INDEX["relief_glyph_structure"],
    "relief_glyph_max_line_scatter": STEP_INDEX["relief_glyph_structure"],
    "relief_glyph_max_dominant_component_fraction": STEP_INDEX["relief_glyph_structure"],
    "relief_glyph_min_compact_edge_coverage": STEP_INDEX["relief_glyph_structure"],
    "relief_glyph_min_outline_components": STEP_INDEX["relief_glyph_structure"],
    "relief_glyph_min_outline_edge_coverage": STEP_INDEX["relief_glyph_structure"],
    "relief_glyph_max_outline_dominant_component_fraction": STEP_INDEX["relief_glyph_structure"],
    "relief_glyph_min_outline_scatter": STEP_INDEX["relief_glyph_structure"],
    "enable_ring_smoothness_filter": STEP_INDEX["ring_smoothness"],
    "ring_smoothness_width_px": STEP_INDEX["ring_smoothness"],
    "ring_smoothness_margin_px": STEP_INDEX["ring_smoothness"],
    "ring_smoothness_percentile": STEP_INDEX["ring_smoothness"],
    "ring_smoothness_min_domain_px": STEP_INDEX["ring_smoothness"],
    "max_ring_roughness": STEP_INDEX["ring_smoothness"],
    "region_feature_ring_margin_px": STEP_INDEX["blob_shape"],
    "region_feature_ring_width_px": STEP_INDEX["blob_shape"],
    "region_feature_colour_tolerance": STEP_INDEX["blob_shape"],
    "region_feature_variance_scale": STEP_INDEX["blob_shape"],
    "region_feature_min_domain_px": STEP_INDEX["blob_shape"],
    "region_feature_min_px": STEP_INDEX["blob_shape"],
    "enable_blob_shape_filter": STEP_INDEX["blob_shape"],
    "min_blob_region_area_px": STEP_INDEX["blob_shape"],
    "max_blob_hull_fill": STEP_INDEX["blob_shape"],
    "min_blob_internal_colour_variation": STEP_INDEX["blob_shape"],
    "enable_feature_extension_filter": STEP_INDEX["feature_extension"],
    "feature_extension_context_px": STEP_INDEX["feature_extension"],
    "feature_extension_min_ratio": STEP_INDEX["feature_extension"],
    "enable_text_line_filter": STEP_INDEX["text_line"],
    "text_min_component_px": STEP_INDEX["text_line"],
    "text_min_characters": STEP_INDEX["text_line"],
    "max_baseline_scatter": STEP_INDEX["text_line"],
    "enable_final_size_filter": STEP_INDEX["size"],
    "final_min_width_px": STEP_INDEX["size"],
    "final_min_height_px": STEP_INDEX["size"],
    "final_min_area_px": STEP_INDEX["size"],
    "enable_final_aspect_filter": STEP_INDEX["size"],
    "final_max_aspect": STEP_INDEX["size"],
    "final_region_padding_px": STEP_INDEX["final_padding"],
    "enable_relief_text_filter": STEP_INDEX["relief_text"],
    "relief_text_response_percentile": STEP_INDEX["relief_text"],
    "relief_text_projection_min_coverage": STEP_INDEX["relief_text"],
    "relief_text_min_runs": STEP_INDEX["relief_text"],
    "relief_text_min_band_scale": STEP_INDEX["relief_text"],
    "uv_island_mask_path": len(PIPELINE_STEPS),
    "green_colour": len(PIPELINE_STEPS),
    "red_colour": len(PIPELINE_STEPS),
    "green_thickness": len(PIPELINE_STEPS),
    "red_thickness": len(PIPELINE_STEPS),
}


@dataclass(slots=True)
class DetectionRun:
    """A completed pipeline run, resumable by a later run."""

    config: MserConfig
    stages: list[DetectionStage]
    entry_states: list[DetectionState]  # state each step started from
    resumed_from: int = 0


def first_changed_step(previous: MserConfig, current: MserConfig) -> int:
    """Return the earliest pipeline step affected by a config change."""
    earliest = len(PIPELINE_STEPS)
    for field in fields(MserConfig):
        if getattr(previous, field.name) == getattr(current, field.name):
            continue
        earliest = min(earliest, PARAMETER_STEP.get(field.name, 0))
    return earliest


def run_detection(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig | None = None,
    previous: DetectionRun | None = None,
    initial_boxes: np.ndarray | None = None,
    initial_contrast_response: np.ndarray | None = None,
    initial_contrast_threshold: float | None = None,
    initial_relief_bridge_response: np.ndarray | None = None,
    initial_island_bits: np.ndarray | None = None,
    initial_relief_edge_mask: np.ndarray | None = None,
) -> DetectionRun:
    """Run the pipeline, resuming from the first step a config change affects.

    Steps before that are reused verbatim from ``previous``, so tuning a late
    filter does not re-run MSER over the whole atlas.
    """
    config = config or DEFAULT_CONFIG
    if initial_boxes is not None:
        # The tuning harness preflights foreground components per UV island.
        # Reusing those raw boxes avoids repeating HSV/gradient/morphology work
        # before the unchanged downstream grouping and fitting stages.
        boxes = np.asarray(initial_boxes, dtype=np.int32).reshape(-1, 4)
        if config.box_source == "foreground":
            title = "Foreground-mask boxes"
            detail = "precomputed dominant-background foreground mask"
        elif config.box_source == "opacity_mask":
            title = "Opacity-mask boxes"
            detail = "precomputed authored opacity-mask foreground"
        elif config.box_source in {"contrast", "contrast_gpu"}:
            title = "Local-contrast GPU boxes" if config.box_source == "contrast_gpu" else "Local-contrast CPU boxes"
            detail = "precomputed masked local-contrast boxes and bridge response"
        elif config.box_source in {"edge", "edge_gpu"}:
            title = "Edge GPU boxes" if config.box_source == "edge_gpu" else "Edge boxes"
            detail = "precomputed edge boxes"
        else:
            title = "MSER boxes"
            detail = "precomputed MSER boxes"
        raw_stage = DetectionStage(
            key="mser", title=title,
            kept=tuple(tuple(int(value) for value in box) for box in boxes),
            detail=detail,
        )
        if previous is not None and len(previous.entry_states) == len(PIPELINE_STEPS):
            start = first_changed_step(previous.config, config)
            if start >= len(PIPELINE_STEPS):
                return DetectionRun(
                    config=config,
                    stages=list(previous.stages),
                    entry_states=previous.entry_states,
                    resumed_from=start,
                )
            if start > 0:
                stages = list(previous.stages[:start])
                entry_states = list(previous.entry_states[:start])
                state = previous.entry_states[start].copy()
            else:
                stages = [raw_stage]
                entry_states = [DetectionState(np.empty((0, 4), dtype=np.int32), [])]
                state = DetectionState(
                    boxes.copy(), [],
                    contrast_response=initial_contrast_response,
                    contrast_threshold=initial_contrast_threshold,
                    relief_bridge_response=initial_relief_bridge_response,
                    island_bits=initial_island_bits,
                    relief_edge_mask=initial_relief_edge_mask,
                )
                start = 1
        else:
            stages = [raw_stage]
            entry_states = [DetectionState(np.empty((0, 4), dtype=np.int32), [])]
            state = DetectionState(
                boxes.copy(), [],
                contrast_response=initial_contrast_response,
                contrast_threshold=initial_contrast_threshold,
                relief_bridge_response=initial_relief_bridge_response,
                island_bits=initial_island_bits,
                relief_edge_mask=initial_relief_edge_mask,
            )
            start = 1
    else:
        start = 0
    if initial_boxes is None and previous is not None and len(previous.entry_states) == len(PIPELINE_STEPS):
        start = first_changed_step(previous.config, config)
        if start >= len(PIPELINE_STEPS):
            # Nothing the pipeline reads changed, so the previous run still
            # stands; re-running would only reproduce it.
            return DetectionRun(
                config=config,
                stages=list(previous.stages),
                entry_states=previous.entry_states,
                resumed_from=start,
            )

    if initial_boxes is None:
        stages = list(previous.stages[:start]) if previous is not None and start else []
        entry_states = list(previous.entry_states[:start]) if previous is not None and start else []
        state = (
            previous.entry_states[start].copy()
            if previous is not None and start
            else DetectionState(
                np.empty((0, 4), dtype=np.int32), [],
                relief_bridge_response=initial_relief_bridge_response,
                island_bits=initial_island_bits,
                relief_edge_mask=initial_relief_edge_mask,
            )
        )

    for index in range(start, len(PIPELINE_STEPS)):
        entry_states.append(state.copy())
        state, stage = PIPELINE_STEPS[index](image, uv_mask, config, state)
        # Most stages only change boxes/groups.  Preserve the immutable
        # per-island edge evidence automatically, rather than relying on every
        # filter constructor to remember to thread it through.
        if state.relief_edge_mask is None:
            state.relief_edge_mask = entry_states[-1].relief_edge_mask
        if not state.circle_radii:
            state.circle_radii = entry_states[-1].circle_radii
        if stage.key in GROUP_STAGE_KEYS and config.enable_circular_groups:
            # Shape is a property of the region's pixels, so it is derived per
            # stage rather than carried: a stage that moves a box reshapes it.
            stage = replace(
                stage,
                circles=tuple(
                    cached_inscribed_circle_radius(
                        image, uv_mask, group, config, state.circle_radii,
                    )
                    for group in stage.kept
                ),
            )
        stages.append(stage)

    return DetectionRun(
        config=config,
        stages=stages,
        entry_states=entry_states,
        resumed_from=start,
    )


def detect_texture_region_stages(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig | None = None,
) -> list[DetectionStage]:
    """Run detection and return every intermediate stage."""
    return run_detection(image, uv_mask, config).stages


def detect_texture_regions(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig | None = None,
) -> dict[str, object]:
    """Run texture-region detection on an already loaded BGR image."""
    stages = detect_texture_region_stages(image, uv_mask, config)
    by_key = {stage.key: stage for stage in stages}
    # Relief text is a terminal semantic filter, after padding.  Keep the
    # original final-padding endpoint for every other detector so their circle
    # metadata retains its established meaning.
    active_config = config or DEFAULT_CONFIG
    final = (
        by_key["relief_text"]
        if active_config.enable_relief_text_filter
        else by_key["final_padding"]
    )

    return {
        "image_size": f"{image.shape[1]}x{image.shape[0]}",
        "mser_boxes": len(by_key["mser"].kept),
        "boxes_rejected": len(by_key["box_filter"].rejected),
        "candidate_groups": len(by_key["grouped"].kept),
        "pattern_groups_rejected": len(by_key["pattern_group"].rejected),
        "final_size_rejected": len(by_key["size"].rejected),
        "domain_recovery_rejected": len(by_key["region_domain"].rejected),
        "region_domain_rejected": len(by_key["region_domain"].rejected),
        "overlap_regions_absorbed": by_key["overlap_group"].adjusted,
        "final_padding_px": max(int((config or DEFAULT_CONFIG).final_region_padding_px), 0),
        "grouped_regions": len(final.kept),
        "groups": [
            {"x": x, "y": y, "w": w, "h": h, "radius": radius}
            for (x, y, w, h), radius in zip(
                final.kept, final.circles or (None,) * len(final.kept)
            )
        ],
    }


def write_annotation_outputs(
    image: np.ndarray,
    summary: dict[str, object],
    outputs: TextureRegionAnnotationOutputs,
    config: MserConfig,
    symmetry_matches: tuple[UvIslandSymmetryMatch, ...] = (),
    symmetry_config: UvIslandSymmetryConfig | None = None,
) -> None:
    """Write requested external annotation artifacts."""
    annotated = draw_annotations(
        image,
        [
            (
                int(group["x"]),
                int(group["y"]),
                int(group["w"]),
                int(group["h"]),
            )
            for group in summary["groups"]
        ],
        config,
    )
    # This overlay is intentionally drawn after the red/green annotations and
    # is not represented in DetectionState or PIPELINE_STEPS.
    annotated = draw_uv_island_symmetry_annotations(
        annotated, symmetry_matches, symmetry_config
    )

    outputs.annotated_image.parent.mkdir(parents=True, exist_ok=True)
    try:
        wrote_output = cv2.imwrite(str(outputs.annotated_image), annotated)
    except cv2.error as exc:
        raise ValueError(
            f"OpenCV could not write output image: {outputs.annotated_image}"
        ) from exc
    if not wrote_output:
        raise ValueError(f"OpenCV could not write output image: {outputs.annotated_image}")

    if outputs.summary_json is not None:
        outputs.summary_json.parent.mkdir(parents=True, exist_ok=True)
        outputs.summary_json.write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )


def annotate_texture_regions(
    texture_path: Path,
    outputs: TextureRegionAnnotationOutputs,
    uv_island_mask_path: Path | str | None = None,
    config: MserConfig | None = None,
    symmetry_config: UvIslandSymmetryConfig | None = None,
) -> dict[str, object]:
    """Detect texture regions and write caller-specified output files."""
    config = config_with_uv_mask_path(config or DEFAULT_CONFIG, uv_island_mask_path)
    image = load_image(texture_path)

    uv_mask = None
    if config.uv_island_mask_path:
        uv_mask = load_uv_island_mask(
            resolve_config_path(config.uv_island_mask_path),
            image.shape,
        )

    summary = detect_texture_regions(image, uv_mask, config)
    symmetry_config = symmetry_config or DEFAULT_UV_ISLAND_SYMMETRY_CONFIG
    symmetry_matches = analyse_uv_island_symmetry(uv_mask, symmetry_config)
    summary.update(
        {
            "input": str(texture_path),
            "output": str(outputs.annotated_image),
            "uv_island_mask": str(uv_island_mask_path or config.uv_island_mask_path),
            "symmetric_uv_islands": len(symmetry_matches),
            "uv_island_symmetry_threshold": min(
                max(float(symmetry_config.min_uv_island_symmetry), 0.0), 1.0
            ),
            "uv_island_symmetry": [
                {
                    "x": match.bounds[0],
                    "y": match.bounds[1],
                    "w": match.bounds[2],
                    "h": match.bounds[3],
                    "area_px": match.area_px,
                    "x_similarity": match.x_similarity,
                    "y_similarity": match.y_similarity,
                    "matched_axes": list(match.matched_axes),
                }
                for match in symmetry_matches
            ],
        }
    )
    if outputs.summary_json is not None:
        summary["summary_json"] = str(outputs.summary_json)

    write_annotation_outputs(
        image,
        summary,
        outputs,
        config,
        symmetry_matches,
        symmetry_config,
    )
    return summary


def annotate_texture(
    input_path: Path,
    output_path: Path,
    config: MserConfig | None = None,
) -> dict[str, object]:
    """Backward-compatible wrapper for the original CLI-style call."""
    config = config or DEFAULT_CONFIG
    return annotate_texture_regions(
        input_path,
        TextureRegionAnnotationOutputs(annotated_image=output_path),
        config.uv_island_mask_path,
        config,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Detect and annotate text/symbol regions on a BeamNG dashboard "
            "or interior texture using OpenCV MSER.  Outputs a PNG with "
            "green inner and red outer 2 px composite border rectangles around each "
            "detected group."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        type=Path,
        help=(
            "Path to the source texture "
            "(PNG, JPG, BMP, TIFF, DDS, or OpenCV-supported format)."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output annotated PNG path.  "
            "Defaults to <input_stem>_annotated.png beside the input."
        ),
    )
    parser.add_argument(
        "--uv-mask",
        type=Path,
        default=None,
        help="UV island mask path. Defaults to DEFAULT_CONFIG.uv_island_mask_path.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional JSON summary output path.",
    )
    parser.add_argument(
        "--min-box-width",
        type=int,
        default=None,
        metavar="PX",
        help="Reject raw MSER boxes narrower than this before grouping.",
    )
    parser.add_argument(
        "--min-box-height",
        type=int,
        default=None,
        metavar="PX",
        help="Reject raw MSER boxes shorter than this before grouping.",
    )
    parser.add_argument(
        "--final-padding",
        type=int,
        default=None,
        metavar="PX",
        help="Padding added to every final detected region, in pixels.",
    )
    parser.add_argument(
        "--uv-symmetry-threshold",
        type=float,
        default=None,
        metavar="FRACTION",
        help=(
            "Minimum reflected-mask similarity for a UV island to receive a "
            "blue outline; either X- or Y-axis symmetry may qualify."
        ),
    )
    parser.add_argument(
        "--no-uv-symmetry",
        action="store_true",
        help="Disable the independent blue UV-island symmetry overlay.",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    input_path: Path = args.input
    if not input_path.is_file():
        parser.error(f"Input file not found: {input_path}")

    output_path: Path = args.output or input_path.with_name(
        f"{input_path.stem}_annotated.png"
    )
    if input_path.resolve() == output_path.resolve():
        parser.error("Output path must be different from the input texture path.")

    overrides: dict[str, int] = {}
    if args.min_box_width is not None:
        overrides["min_box_width_px"] = max(args.min_box_width, 1)
    if args.min_box_height is not None:
        overrides["min_box_height_px"] = max(args.min_box_height, 1)
    if args.final_padding is not None:
        overrides["final_region_padding_px"] = max(args.final_padding, 0)
    config = replace(DEFAULT_CONFIG, **overrides) if overrides else DEFAULT_CONFIG

    symmetry_overrides: dict[str, object] = {}
    if args.uv_symmetry_threshold is not None:
        symmetry_overrides["min_uv_island_symmetry"] = min(
            max(float(args.uv_symmetry_threshold), 0.0), 1.0
        )
    if args.no_uv_symmetry:
        symmetry_overrides["enable_uv_island_symmetry"] = False
    symmetry_config = (
        replace(DEFAULT_UV_ISLAND_SYMMETRY_CONFIG, **symmetry_overrides)
        if symmetry_overrides
        else DEFAULT_UV_ISLAND_SYMMETRY_CONFIG
    )

    started = time.perf_counter()

    try:
        summary = annotate_texture_regions(
            input_path,
            TextureRegionAnnotationOutputs(
                annotated_image=output_path,
                summary_json=args.summary_json,
            ),
            args.uv_mask,
            config,
            symmetry_config,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    elapsed_ms = (time.perf_counter() - started) * 1000.0

    print(
        f"{summary['image_size']}  "
        f"{summary['mser_boxes']} MSER boxes → "
        f"{summary['grouped_regions']} annotated groups; "
        f"{summary['symmetric_uv_islands']} symmetric UV islands"
    )
    print(
        f"{summary['boxes_rejected']} boxes rejected, "
        f"{summary['candidate_groups']} candidate groups, "
        f"{summary['pattern_groups_rejected']} rejected as repeating pattern, "
        f"{summary['final_size_rejected']} rejected by final size, "
        f"{summary['region_domain_rejected']} rejected as outside the UV domain, "
        f"{summary['overlap_regions_absorbed']} overlapping regions absorbed"
    )
    print(f"Wrote {summary['output']}")
    print(f"elapsed: {elapsed_ms:.1f} ms")

    for index, group in enumerate(summary["groups"], start=1):
        shape = f"circle r={group['radius']}" if group.get("radius") else "rect"
        print(
            f"  [{index:3d}]  x={group['x']:5d}  y={group['y']:5d}  "
            f"w={group['w']:4d}  h={group['h']:4d}  {shape}"
        )


if __name__ == "__main__":
    main()
