"""The local-contrast front end, producing regions that carry their texels.

Deliberately numpy and scipy only.  cv2 is on its way out of this repo, and a
new package is the wrong place to add call sites to it; a masked box filter is
two ``uniform_filter`` passes and a divide, and labelling is one ``ndimage.label``.

Detection runs per UV domain, not over the atlas, for the same reason the first
pipeline does: the response is a pixel's distance from the mean of its
neighbourhood, and a neighbourhood must not reach across a UV seam into a
material the mark has nothing to do with.

The order matters and is the point of the rebuild: threshold, then throw out
texels that were never marks, and only then group what is left.  The first
pipeline grouped first and argued afterwards, which is a much weaker position
-- a component welded together by noise cannot be taken apart by any later
stage.

2026-08-28: eight stages, ending on the marks.  The nine that followed -- the
seed grown back through the threshold, hysteresis, a denoise pass, texel
filters for UV rims and mesh seams, a second grouping, and a region stage that
judged each mark on its shape -- were the algorithm this rebuild replaces, and
they went once the front end had earned it.  The history has them.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from glyph_lab.corpus import CorpusTexture
from glyph_lab.material_transitions import (
    classify_material_transition_candidates,
    clear_of_material_transitions,
    material_transition_candidates,
    material_transition_features,
)
from glyph_lab.probe import RegionObservation, RegionProbe
from glyph_lab.regions import NEIGHBOURHOOD, Region, closed_body

# The last stage `stop_after_stage` can name, which is the last stage there
# is: the pipeline ends on the marks and answers with them.
LAST_STOP = 14

# Every point in the pipeline worth looking at, in the order it happens.  The
# viewer lists these and draws whichever is picked in place of the response, so
# a mark that arrives whole and leaves in pieces can be watched losing its
# shape rather than inferred about from the two ends.
#
# 2026-08-27: added because four rounds of argument about a run of seat
# stitching were all conducted on the output alone.  The answer each time was
# somewhere in the middle, and each time it took a one-off script to find it.
# 2026-08-27: named, not indexed.  Stages are being inserted one at a time as
# each is argued out, and every insertion used to renumber twenty call sites
# and half the tests -- which is exactly the kind of edit that quietly moves a
# stage's contents into its neighbour's name.  The number stays in the label
# because the picker is a flat list and reads better ordered.
AT_THRESHOLD = "1 over threshold"
SPARSE = "2 the sparse field"
UNCROWDED = "3 clear of a crowd"
SIZED = "4 past the size floor"
UNREPEATED = "5 clear of a repeat"
UNRIMMED = "6 clear of the rim"
UNTRANSITIONED = "7 clear material transitions"
OFF_BOUNDARY = "8 clear of the boundary"
SEEDED = "9 structure seed"
DENSE_SEED = "10 seed over density"
PATTERNED = "11 pattern candidates"
TRANSFORMED = "12 past the transform"
BRIDGED = "13 bridged across the gaps"
MARKED = "14 grouped into marks"

STAGES: tuple[str, ...] = (
    AT_THRESHOLD,
    SPARSE,
    UNCROWDED,
    SIZED,
    UNREPEATED,
    UNRIMMED,
    UNTRANSITIONED,
    OFF_BOUNDARY,
    SEEDED,
    DENSE_SEED,
    PATTERNED,
    TRANSFORMED,
    BRIDGED,
    MARKED,
)


# Which stage each parameter belongs to, by stage number.  0 means it is not
# part of any one stage -- `stop_after_stage` itself is the only such field.
#
# 2026-08-27: added so the panel can hide what a truncated pipeline does not
# read.  A box that still takes a value for a stage that no longer runs is
# worse than no box: it invites tuning a number that cannot do anything, which
# is a whole afternoon of wondering why nothing changes.
#
# A stage is where a field is *first* read.
STAGE_OF: dict[str, int] = {
    "stop_after_stage": 0,
    # 1 over threshold
    "boundary_skip_px": 1,
    "kernel_px": 1,
    "blur_px": 1,
    "min_response": 1,
    "percentile": 1,
    "local_floor_multiple": 1,
    "local_floor_px": 1,
    # 2 the sparse field
    "sparse_kernel_px": 2,
    "max_sparse_texels": 2,
    # ...and the paths drawn round the holes it leaves.
    "min_void_texels": 2,
    "outline_px": 2,
    # 3 clear of a crowd
    "crowd_radius_px": 3,
    "max_neighbours": 3,
    "crowd_large_region_area_px": 3,
    # 4 past the size floor
    "min_width_px": 4,
    "min_height_px": 4,
    "min_area_px": 4,
    # 5 clear of a repeat
    "enable_repeat_texture_filter": 5,
    "repeat_texture_decimation": 5,
    "repeat_texture_highpass_px": 5,
    "repeat_texture_tile_px": 5,
    "repeat_texture_search_scale": 5,
    "repeat_texture_min_search_px": 5,
    "repeat_texture_max_search_px": 5,
    "repeat_texture_match_threshold": 5,
    "repeat_texture_min_tile_contrast": 5,
    "repeat_texture_min_region_span_px": 5,
    "repeat_texture_min_region_area_px": 5,
    "repeat_texture_min_repeats": 5,
    "repeat_texture_context_repeats": 5,
    "repeat_texture_context_directions": 5,
    "repeat_texture_context_margin_px": 5,
    "repeat_texture_fallback_tile_px": 5,
    "repeat_texture_fallback_max_fill": 5,
    "repeat_texture_fallback_min_energy": 5,
    "repeat_texture_fallback_min_overlap": 5,
    "repeat_texture_fallback_min_dip": 5,
    # 6 clear of the rim
    "min_rim_margin_px": 6,
    "max_rim_margin_px": 6,
    "rim_margin_share": 6,
    "min_rim_contact": 6,
    "rim_artifact_max_span_px": 6,
    "rim_artifact_min_perimeter_contact": 6,
    "rim_artifact_large_max_fill": 6,
    # 7 clear material transitions
    "transition_min_span_px": 7,
    "transition_max_fill": 7,
    "transition_close_gap_px": 7,
    "transition_min_enclosed_texels": 7,
    "transition_min_body_void_share": 7,
    "transition_max_interior_detail": 7,
    "transition_min_long_aspect": 7,
    "transition_max_thin_trace_span_px": 7,
    "transition_min_thin_trace_aspect": 7,
    "transition_max_flat_loop_texels": 7,
    "transition_context_gap_px": 7,
    # 8 clear of the boundary
    "min_boundary_reach_px": 8,
    # 9 structure seed
    "structure_px": 9,
    "structure_inner_px": 9,
    "min_structure": 9,
    # 10 seed over density
    "seed_density_px": 10,
    "min_seed_density": 10,
    # 11 pattern candidates
    "pattern_gap_px": 11,
    "min_pattern_texels": 11,
    "min_pattern_fill": 11,
    "min_pattern_area": 11,
    "min_pattern_coverage": 11,
    # 12 past the transform
    "repeat_skip_px": 12,
    "repeat_longest_px": 12,
    "min_repeat_energy": 12,
    "min_repeat_overlap": 12,
    "min_repeat_dip": 12,
    # 13 bridged across the gaps
    "bridge_clear_px": 13,
    "bridge_void_px": 13,
    "min_bridge_void": 13,
    "bridge_span_px": 13,
    "min_bridge_response": 13,
    "max_bridge_share": 13,
    # 14 grouped into marks
    "mark_gap_px": 14,
    "min_stroke_texels": 14,
}


# Where in ``STAGES`` each overlay first means anything.  Before its stage the
# thing simply does not exist yet, so the viewer draws it blank rather than
# reaching for the finished article.  Showing a stage's texels under the
# *final* outlines was the first version of this and it is worse than showing
# nothing: it invites reading a late decision back onto an early picture.
STAGE_FIRST: dict[str, int] = {
    # 2026-08-28: the sparse field is the first stage with a region in it.  Its
    # holes *are* the regions -- there is nothing to group, because the field
    # has already drawn each mark as one closed shape -- so a path can be drawn
    # from the moment it exists.
    "outlines": STAGES.index(SPARSE),
}


@dataclass(frozen=True, slots=True)
class ContrastConfig:
    """What the front end reads.  One knob per decision, and no more."""

    # 2026-08-27: where the pipeline ends, as a stage number.  0 runs the lot,
    # and so does 7, which is the last stage there is.
    #
    # 2026-08-28: a debugging tool now rather than a way of holding the old
    # tail back -- there is no tail.  Set it to an earlier stage and that
    # stage's texels become the result: only the marks stage and the candidate
    # stage before it have grouped anything, so stopping before either leaves a
    # mask and no regions, which is the honest answer for a stage that has not
    # decided anything yet.
    #
    # The panel hides the parameters of every stage past the stop, so it also
    # narrows the window down to the step being worked on.
    stop_after_stage: int = 7

    # 2026-08-27: how far in from an island's rim a texel has to be before its
    # colour is treated as a measurement of the surface.
    #
    # A texel on the boundary is not a sample of the material.  The UV
    # rasteriser writes it wherever a triangle covers *any* of it, so its value
    # is a blend of the surface and whatever the atlas holds outside -- the
    # padding bleed, or nothing at all.  `masked_local_mean` already refuses to
    # average the emptiness beyond the island, which is a different and lesser
    # thing: it keeps the mean from being dragged to black, but the rim texel
    # itself still goes into every neighbourhood that reaches it, and every one
    # of those means comes out wrong.  So every UV island has a false step
    # drawn one texel inside its own edge, in the exact place the response is
    # least able to tell a mark from a seam.
    #
    # Excluded from the output as well as from the sample, which is the only
    # coherent way to do it.  Taking the rim out of the sample alone makes it
    # fire *harder*: at blur_px 1 its own value is the near mean, so a
    # contaminated centre would be compared against a newly clean surround.
    # Nothing is lost by refusing to answer there, because there was never an
    # answer.
    #
    # 0 goes back to sampling the whole domain.
    #
    # 0 goes back to sampling the whole domain.
    #
    # 2026-08-29: 1, and only 1.  This is what the *rasteriser* writes wrong,
    # which is one texel, and nothing more.  The step an island edge reads as
    # carries further in than that, but widening this to cover it is the wrong
    # instrument: it trims the sample, so a mark that straddles the margin is
    # clipped rather than judged.  That is `min_rim_margin_px`, which refuses a
    # region only when the whole of it lies inside the margin.
    boundary_skip_px: int = 1
    # 2026-08-28: the ground the marks stand on, read off the threshold alone.
    #
    # A texel is painted when its `sparse_kernel_px` window holds no more than
    # `max_sparse_texels` of what fired.  What that draws is the blank part of
    # the surface -- and a mark is what puts a hole in it.  The complement is
    # not a filter and nothing downstream reads it; the stage is a marker, like
    # the pattern candidates further on, and `lit` goes past it untouched.
    #
    # It is here because it says something no filter over the texels does.  On
    # the etk800 faceplate every digit and word punches a clean island out of
    # the field, gaps between letters included, because a window sitting in a
    # gap still sees the strokes either side -- so the grouping a mark needs is
    # already drawn before any grouping stage runs.  A grained panel puts no
    # hole in it at all.
    #
    # The window is a disc, not a square -- see `disc_counts` for why, and note
    # that a disc of 9 holds 49 texels where the square around it holds 81, so
    # the same count reads more generously than it did.
    #
    # The two knobs pull against each other and both matter.  The window sets
    # what counts as near: at 5 almost everything is blank, at 15 a speckled
    # panel closes up entirely and a real badge on it stops reading as a hole.
    # The count is what erodes noise: at 1 a grain field is one solid mass and
    # its sea cannot be told from a large glyph by size, and raising it eats
    # the field from the inside while a drawn stroke -- solid, so far above any
    # small count -- is untouched.
    #
    # 9 and 1 to start, which is the plain reading of "blank".  Both are here
    # to be turned rather than trusted.  0 on either paints nothing.
    sparse_kernel_px: int = 9
    max_sparse_texels: int = 1
    # 2026-08-28: and how small a hole may be and still be worth keeping.
    #
    # Width and height are the hole's bounding box; the area is the hole
    # itself, not its box, so a diagonal stroke is not credited with the
    # emptiness either side of it.  All three at 0 keep every hole.
    #
    # There are a great many holes and almost all of them are nothing: at disc
    # 11 the corpus leaves 1,020 to 3,445 per texture with a median of 3 to 9
    # texels, against legends and badges in the hundreds and up.  The floor is
    # the whole of the sorting, which is why the three of them are their own
    # stage rather than a clause in the one above.
    #
    # 4, 4 and 36 to start, and deliberately loose -- this is the stage the
    # rebuild has just reached, so the numbers are here to be turned rather
    # than trusted.
    min_width_px: int = 4
    min_height_px: int = 4
    min_area_px: int = 36
    # 2026-08-29: and how much company a mark may keep.
    #
    # A region is refused when more than `max_neighbours` holes have their
    # centre within `crowd_radius_px` of its own.  What it is for is quilting,
    # perforation and weave: the sparse field paints the material out, but
    # every gap in the quilting is a hole and every hole is a region, so a
    # padded panel arrives as hundreds of them.  Size cannot help -- each gap
    # is a perfectly ordinary size, and the ardente panel that came in at 463
    # regions was mostly one such field.
    #
    # Company is counted over *every* hole the field left, not over the ones
    # that cleared the size floor.  Counting only the survivors was the first
    # version and it does not finish the job: the floor has already thinned the
    # very field being judged, so what is left of a quilted panel is a sparse
    # scatter that looks like honest company.  Measured, it left 187 of
    # ardente's regions standing with the panel still visibly speckled; over
    # all the holes it leaves 105 and the panel goes entire.
    #
    # 2026-08-29: solidity was the first idea and it is wrong.  It reads a
    # region against its own filled silhouette, on the guess that the material
    # arrives as one lace web; it does not, it arrives as hundreds of solid
    # little blobs, and the measure flags one region on ardente which is not
    # even the right one.  What is distinctive is not the shape of any one of
    # them but that there are hundreds.
    #
    # 120 and 40, swept at 8, 20, 40 and 80 against the whole-hole count.  The
    # bar is what decides how much is lost elsewhere rather than how much of
    # the panel is caught -- the panel is refused at every setting swept.  At 8
    # etk800 loses its 5.0 badge and two faceplate icons; at 40 it keeps all
    # 123 of its regions; at 80 ardente's speckle field starts surviving again.
    #
    # Distances are in atlas coordinates, which the region bounds already
    # carry.  Company is gathered per UV domain like every other stage, so a
    # crowd on one island is not read against a mark on another that happens
    # to lie beside it in the atlas.
    #
    # What this does not catch is a repeating material that arrives as a
    # *single* region -- scintilla's chevron tread is one connected thing, so
    # it has no neighbours to be crowded by.  That is `max_repeat_dip` below.
    #
    # `crowd_radius_px` at 0 switches the stage off.  `max_neighbours` at 0
    # does not -- it means a mark may keep no company at all, which is a
    # setting worth having and not the same statement.
    crowd_radius_px: int = 120
    max_neighbours: int = 40
    # 2026-08-29: birdshot can surround a real, substantial mark.  The crowd
    # rule is evidence about the small fragments, not a reason to discard the
    # large region they happen to border.  Area is the fired region rather
    # than its bounding box, so a long thin stroke is not over-credited.  The
    # normal hole builder memoises this count; 0 turns the exception off.
    crowd_large_region_area_px: int = 1024
    # 2026-08-29: classify material from the atlas, not from a region's own
    # silhouette.  A small high-passed tile must reappear several times in a
    # bounded neighbourhood.  A label such as ON/OFF can contain similar
    # strokes, but it does not have four copies of the same tile; a chevron,
    # grille, stitch run or carbon weave does.  The index is built once per
    # atlas and each probe is cached, which keeps the stage out of the hot path.
    enable_repeat_texture_filter: bool = True
    repeat_texture_decimation: int = 2
    repeat_texture_highpass_px: int = 12
    repeat_texture_tile_px: int = 48
    repeat_texture_search_scale: float = 6.0
    repeat_texture_min_search_px: int = 256
    repeat_texture_max_search_px: int = 384
    repeat_texture_match_threshold: float = 0.7
    repeat_texture_min_tile_contrast: float = 2.0
    # 2026-08-29: A thin stroke can recur within one glyph (the ON/OFF false
    # positive), and a small solid hole is not enough material to make the
    # image search worth paying for.  Both use the region's already-computed
    # extent, so they decline it before allocating a correlation response.
    # The 1,024-texel floor retains the smallest presently recognised corpus
    # repeat (ardente's 1,857-texel vertical grille).
    repeat_texture_min_region_span_px: int = 16
    repeat_texture_min_region_area_px: int = 1024
    repeat_texture_min_repeats: int = 4
    repeat_texture_context_repeats: int = 12
    repeat_texture_context_directions: int = 2
    repeat_texture_context_margin_px: int = 24
    # 2026-08-30: a compact repeated insert can be too coarse for the normal
    # 48px material tile.  Retry only masks that are themselves strongly
    # periodic, then confirm them at this smaller tile.  This keeps keypad
    # texture out while the repeated strokes in SEEK and ENGINE remain glyphs.
    # Set the tile to 0 to disable the fallback.
    repeat_texture_fallback_tile_px: int = 32
    repeat_texture_fallback_max_fill: float = 0.75
    repeat_texture_fallback_min_energy: float = 0.07
    repeat_texture_fallback_min_overlap: float = 0.40
    repeat_texture_fallback_min_dip: float = 0.35
    # 2026-08-29: and which regions are the island's own edge rather than
    # something drawn on it.
    #
    # An island edge is a step in the atlas whatever the surface does, and the
    # response to it carries several texels inward -- so a band forms just
    # inside the outline of every chart, and once the sparse field has run that
    # band is a hole and the hole is a region.  Edge stitching arrives the same
    # way: a run of small marks following the rim.
    #
    # Two things have to be true before one is refused, and it takes both.
    #
    # First, all of it lies inside the margin.  Touching the rim is not held
    # against anything and only never leaving it is -- a band hugs the edge
    # along its whole length, while a mark that runs up against one has a body
    # carrying on inward.  Trimming the *sample* by this margin instead was
    # tried and is wrong: it clips a mark that straddles the margin rather than
    # judging it, and `boundary_skip_px` is left at the one texel the
    # rasteriser actually costs.
    #
    # Second, enough of it runs *along* the rim.  `min_rim_contact` is how
    # much of the rim a region follows, over its own length, so a band lying
    # against an edge reads about 1 and a mark that merely touches one reads
    # near 0.  This is what separates a ring following a boundary from a
    # legend that happens to sit near one: measured over the corpus, the edge
    # bands read 0.41 to 0.74 -- the ducting on etk800, the pleat tops and the
    # edge stitching on scintilla -- while what is drawn reads 0.26 and below,
    # and 0.00 for anything not touching an edge at all.  0.40 sits in the gap
    # and takes 6, 6, 1 and 3 more regions across the four textures.
    #
    # Over the region's length rather than its own outline, which was the
    # first version and let a wiggle dilute the reading -- see `rim_contact`.
    # The readings above 1 all belong to regions that *fill* a round island --
    # a dial readout at 3.05, two honeycomb pedal faces at 1.05 and 0.41 --
    # and none of them ever reach this test, because filling an island makes
    # them deep and the margin keeps them first.
    #
    # The margin slides with the island, because the islands are not one size.
    # It is a share of the island's own size -- the mean of its bounding box's
    # two sides -- held between a floor and a ceiling.  The floor applies
    # everywhere; the ceiling stops a large panel condemning anything within
    # so many texels of its border.
    #
    # 3, 16 and 0.08.  The share halved when the size measure changed from the
    # island's reach, which is a half-thickness, to the mean of its bounds,
    # which is roughly twice that -- the same margin, read off a box that costs
    # nothing instead of a distance transform that cost the whole stage.
    #
    # `min_rim_margin_px` at 0 with no share switches the stage off, and
    # `min_rim_contact` at 0 drops the second condition so the margin decides
    # alone.
    min_rim_margin_px: int = 3
    max_rim_margin_px: int = 16
    rim_margin_share: float = 0.08
    min_rim_contact: float = 0.40
    # 2026-08-29: an edge artefact has a bounded minor axis.  That includes a
    # continuous band and the separate, low-fill pieces of edge stitching.
    # Short pieces are only removed as a locally straight run along the rim;
    # otherwise compact controls near a small chart's edge would be mistaken
    # for stitching.  A wide control/ring is not a rim artefact just because
    # one side is near a chart boundary.  Zero restores the un-gated
    # margin/contact test.
    rim_artifact_max_span_px: int = 32
    # 2026-08-29: a short candidate needs this fraction of its own perimeter
    # on the UV rim to be an artifact by itself.  The rim probe separates the
    # edge stitching at 0.32--0.51 from compact control glyphs below 0.20;
    # the minor-axis gate above protects large rings that also have high rim
    # contact.  Lower contacts still need a locally straight stitch train.
    rim_artifact_min_perimeter_contact: float = 0.30
    # 2026-08-29: a broad, mostly-empty edge trace is a panel stitch or a
    # cutout's border, not a compact control.  This only overrides the span
    # guard when enough of the trace is on the rim as well; 0 disables it.
    rim_artifact_large_max_fill: float = 0.15
    # 2026-08-29: broad sparse regions are candidates only.  A material seam
    # may be open, nearly closed or fully closed, so the decision combines
    # geometric closure with the visual detail inside a true enclosed loop.
    transition_min_span_px: int = 64
    transition_max_fill: float = 0.30
    transition_close_gap_px: int = 6
    transition_min_enclosed_texels: int = 512
    transition_min_body_void_share: float = 0.03
    transition_max_interior_detail: float = 6.0
    transition_min_long_aspect: float = 4.5
    # 2026-08-30: tiny, dense dashes on a material boundary are below the
    # broad transition gate, yet too short for the normal long-trace rule.
    # These bounds isolate the ETK pad seam without admitting the 11px-wide
    # scintilla legend that is a real glyph.
    transition_max_thin_trace_span_px: int = 10
    transition_min_thin_trace_aspect: float = 3.5
    transition_max_flat_loop_texels: int = 6_000
    transition_context_gap_px: int = 3
    # 2026-08-27: how far a response that touches an island's edge has to
    # reach inward before it is believed.
    #
    # `boundary_skip_px` above stops the rim texel from being *read*; this is
    # about what is left once it has been.  An island edge is a step in the
    # atlas whatever the surface does, and the one-sided neighbourhood beside
    # it is a poor estimate of anything, so the threshold lights a ribbon
    # following the outline of every chart.  Those ribbons are not marks and
    # nothing downstream can tell, because locally they look exactly like the
    # stroke of a glyph.
    #
    # What tells them apart is not where they start but where they go.  A
    # boundary artefact hugs the edge along its whole length; a glyph that
    # happens to run up against an edge has a body that carries on inward.  So
    # the test is per component, on the deepest it ever gets: touching the edge
    # is not held against anything, and only never leaving it is.
    #
    # This replaces the `uv_edge_margin_*` band, which asked the other and
    # cruder question -- how close is this to an edge -- and so had to choose
    # between leaving the ribbons and clipping every mark near a chart border.
    #
    # 0 disables the test.
    min_boundary_reach_px: int = 4
    # 2026-08-27: 3, not 5.  Composed with a blur this spans blur_px + 2, so
    # the pair is what sets the real footprint -- at blur_px 3 a kernel of 3
    # already reaches 5 texels.
    kernel_px: int = 3
    # 2026-08-27: the inner scale of the band-pass.  1 subtracts the local mean
    # from the raw texel, which is the plain high-pass this started as; above
    # that the texel is replaced by its own small mean first, and the response
    # becomes the difference of two blurs.
    #
    # Folding a blur into the contrast kernel is legitimate -- convolution is
    # associative, so blur (*) (identity - mean) is a single kernel, and that
    # kernel is a difference of two blurs, which is why this stays two
    # separable box filters rather than a dense one.
    #
    # It matters that it happens *before* the norm across RGB, the one
    # non-linear step here.  Blurring the response afterwards only spreads
    # noise around; blurring the values beforehand lets grain cancel against
    # itself, because its excursions are signed and a drawn edge's are not.
    blur_px: int = 1
    # The response floor, and the per-domain percentile it is a floor *for*.
    # The floor is what stops a flat domain reporting its own noise.
    #
    # 2026-08-27: 10, which is low enough that on most domains it stops binding
    # and the percentile alone decides.  Higher loses faint marks -- the
    # Ginetta script fires at 23, being white on light grey -- and the noise it
    # lets in is handled downstream by `min_region_standout` rather than by
    # refusing to look.
    min_response: float = 10.0
    percentile: float = 70.0
    # 2026-08-27: the threshold as a multiple of the local noise floor -- the
    # mean response over `local_floor_px` -- rather than a percentile of the
    # whole domain.  0 falls back to the percentile.
    #
    # This is the answer to a problem a blur cannot solve.  How much contrast
    # is worth noticing is not a property of the atlas, or even of a domain: a
    # clear but faint glyph on a blank plate needs a low bar, and a strong one
    # on flecked plastic needs a high one, and both turn up on the same
    # texture.  A fixed band-pass answers it once and gets one of them wrong --
    # measured, blur 3 erases the Ginetta script entirely.  Asking each texel
    # to beat its *own* surroundings answers it everywhere at once.
    #
    # Measured over the corpus crops, going from the per-domain percentile to
    # 2.5x the local floor: etk800 grain 33.2% lit -> 4.7%, ardente confetti
    # 37.7% -> 10.5%, and the script -- faint, on a blank plate, the case that
    # blurring destroys -- 8.6% -> 8.5%, which is to say untouched.
    local_floor_multiple: float = 2.5
    local_floor_px: int = 31
    # 2026-08-27: is there a *step* here, or only fluctuation?
    #
    # Derived from fifty 7x7 crops sampled at random from fired texels across
    # the corpus and hand-labelled.  Every reason for calling a crop a glyph
    # reduced to one thing: the patch splits into two locally-flat levels.
    # Every reason for calling it noise was the absence of that -- grain
    # fluctuates about a single level, and weave, perforation and fleck have
    # steps in every direction so the patch cannot be split in two at all.
    #
    # Measured as the deviation over `structure_px` against the *average*
    # deviation over `structure_inner_px` inside it: a step makes the wide
    # window vary while each side stays flat, so the ratio rises; noise moves
    # both together and it sits at 1.  On the labelled crops, glyphs read a
    # median 3.05 and grain and pattern 1.11.
    #
    # The inner window must stay small enough to fit wholly inside one side of
    # a step: at 7/5 the measure drops to 1.73 against 1.04, worse than 5/3.
    #
    # What this cannot do is tell a mark from a panel edge -- trim, louvres, a
    # grille -- which read 2.61, squarely among the glyphs.  That is not a
    # tuning failure: at seven texels a legend edge and a panel edge are the
    # same object, and what separates them is extent, which only the region
    # stage knows.  Measures aimed at it here failed as expected: centre minus
    # surround gives 0.45 against 0.32, and the Otsu minority fraction runs
    # backwards, 0.39 against 0.43.  0 disables the gate.
    #
    # 2026-08-28: seeded and grown, not gated -- see `structure_seed`.  This
    # was the intent from the start, but the growing was done by the hysteresis
    # and the reclaim further down the pipeline, and both went with the stages
    # that were deleted on 2026-08-28.  What was left was the hard per-texel
    # test the note here had always warned against, and it showed: ardente's
    # play-button triangle fell from 268 texels to 14, and a row of legend text
    # from 2,366 to 1,163.
    #
    # 2026-08-28: 1.5 swept again with the growing in place, because the two
    # interact -- a speck that scrapes over the bar now returns the whole piece
    # it sits in, which makes this a sharper dial than it was, not a softer
    # one.  Reading the play triangle against ardente's confetti field, of
    # which 18,094 texels reach this stage: at 1.5 the triangle keeps 124 and
    # the field 518, at 1.4 144 against 4,261, at 1.3 203 against 11,677, and
    # at 1.2 225 against 16,023 -- the field entire, for 100 texels of glyph.
    # scintilla's dial digits are the one real temptation, 1,076 rising to
    # 1,328 by 1.3, and they are not worth two thirds of a grain field.
    structure_px: int = 7
    structure_inner_px: int = 3
    min_structure: float = 1.5
    # 2026-08-27: how crowded a seed texel's neighbourhood has to be, measured
    # over the seed itself, before the seed is believed.
    #
    # This is the one thing the structure test cannot say.  It reads a 7-texel
    # window, so it answers "is there a step here" and nothing about whether
    # the step goes anywhere -- and grain, weave and fleck all throw the odd
    # window that happens to split cleanly in two.  Those chance passes arrive
    # scattered, one here and one four texels away; a real edge seeds along its
    # whole length, so every seed on it has more seed either side.
    #
    # This is NOT the density measured for `denoise_seed`, which came out
    # backwards and is recorded there as such.  That one asked about the
    # *threshold* mask, where grain fires everywhere and is locally dense.  This
    # asks about the seed, after the structure gate has already thinned grain to
    # specks -- the same question, put one stage later, separates the other way.
    #
    # Counted over in-domain texels only, so a seed near a UV seam is not
    # punished for the emptiness outside the island.  The centre counts itself,
    # so a wholly isolated seed scores 1/k^2 -- 0.0044 at the window below.
    #
    # 15, measured against 5, 9 and 21.  Held to the level that costs 2% of the
    # seed lying inside an accepted region, a wider window always takes at
    # least as much of the rest: on etk800 4.8% -> 6.5% -> 10.2% -> 11.1% at 5,
    # 9, 15 and 21, and on ardente 7.5% -> 8.2% -> 9.5% -> 9.6%.  21 buys
    # almost nothing over 15 and costs a wider blur.
    #
    # 2026-08-27, OFF by default, and the corpus is why.  The measure is real
    # -- on scintilla_interior the seed inside accepted regions runs a median
    # 0.346 against 0.099 outside it, and a cut at 2% of the mark takes 59% of
    # the rest -- but on the other three there is no such gap: etk800 reads
    # 0.235 inside against 0.395 *outside*, because its grain panels seed
    # densely, and scintilla_badges 0.259 against 0.284.  Swept end to end at
    # window 15, region counts drift the wrong way (etk800 83 -> 84 -> 85 -> 89
    # at 0, 0.02, 0.04, 0.07) as long refused runs break into pieces that
    # each pass on their own, and scintilla_badges loses a 42-texel legend at
    # 0.04 outright.  At 0.25 a 6,531-texel badge shatters into eight.
    #
    # Kept and switched off because the gate itself is sound -- what it removes
    # is genuinely isolated seed -- and because the thing it was built for is
    # not visible in a region count.  0.02 at window 15 is the safe end: it
    # holds every region on three textures and takes 87 to 2,398 seed texels.
    #
    # 0 disables the test.
    seed_density_px: int = 15
    min_seed_density: float = 0.0
    # 2026-08-27: how far apart two seed texels can be and still be the same
    # mark.
    #
    # 2, measured against 0, 1, 3, 4 and 6.  At 0 a glyph does not arrive as
    # one thing at all -- a letter is a ring of two-texel fragments and a
    # button's bezel comes in as five arcs, so etk800 gives 4,691 components of
    # which the median is 3 texels.  At 2 the words close up: AUTO, MAX, VOL
    # and TUNE each become a single group, and each ring becomes one ring.
    # Past 2 it runs away -- scintilla's largest group goes 5,742 -> 23,041
    # texels between gaps 1 and 2 as whole panels weld together.
    mark_gap_px: int = 2
    #
    # 2026-08-27: no size test here, deliberately.  There was one -- 36 texels,
    # measured on the seed rather than on the dilation -- and it did most of
    # the work of removing grain: it took etk800 from 4,691 raw components to
    # 116 regions.  It came out because a size test is a judgement about
    # whether a group is worth correcting, and this stage is only about what a
    # group *is*.  Sizing here settles that question before anything has
    # measured the group's shape, its ground, or its company, and a mark thrown
    # out at this point cannot be argued back in later.
    #
    # `min_texels` at stage 11 is the size test the old tail applies.  Whatever
    # replaces it belongs after the stages that have yet to be built, not here.
    # 2026-08-27: and the largest single unbroken piece it has to contain.
    #
    # This is what tells a legend from a perforated panel, and nothing simpler
    # does.  Both arrive as dust: measured at gap 2 on etk800, the AUTO legend
    # is 56 fragments averaging 3.2 texels and the MODE row is 85 averaging
    # 3.4, which is the same texture of data as scintilla's perforation field
    # at 2,945 fragments averaging 1.3.  Two obvious tests were tried against
    # that and both failed on measurement: mean fragment size puts the legends
    # at 2.1-4.0 against the perforation's 1.3, overlapping, and mean seed
    # density puts the legends at 0.070-0.126 against 0.062, overlapping
    # outright.
    #
    # What does separate them is that something drawn has at least one solid
    # stroke somewhere in it, and a field of identical dots has none however
    # many dots there are.  Measured over the corpus at gap 2, the largest
    # fragment in a group: the perforation fields top out at 5, 6 and 7 texels,
    # while the smallest real thing found -- a scintilla digit -- reaches 29.
    # scintilla_badges, which is nothing but clean glyphs, has a minimum of 76.
    #
    # 8, and the gap is narrower than those numbers suggest.  It drops 13
    # groups and 7,672 texels on scintilla_interior, which is the perforation,
    # for 124 texels on etk800, 75 on ardente and nothing at all on badges.
    #
    # But 8 is also the *minimum* that works: the perforation's longest run is
    # exactly 7, so 7 leaves 5,896 of its texels standing.  And the etk800 cost
    # is not noise -- it is two legends, one of them the word MAX, whose
    # longest run is 4.  There is no setting that keeps MAX and drops the
    # perforation, because MAX is the more broken of the two.  Left at 8 with
    # the loss stated rather than tuned around: an 87-texel legend that arrives
    # as 42 fragments of two texels is not data anything downstream can use.
    #
    # 0 disables it.
    min_stroke_texels: int = 8
    # 2026-08-28: which groups are worth putting a transform over.
    #
    # A large, moderately dense area, and that is the whole of it.  What makes
    # something a repeating material is that it repeats, which is a question
    # for a Fourier transform and not for a screen in front of one; the job
    # here is only to say where the transform should look, generously enough
    # that nothing periodic is missed before it gets there.
    #
    # 2026-08-28: two earlier attempts to decide it here are recorded because
    # both looked right and neither was.  A tile grid measuring how evenly the
    # seed fills a window did separate -- scintilla's chevron, hatch and
    # perforation at 0.13 to 0.19 against 0.26 and up for everything drawn --
    # but only at one scale, and it removed half of scintilla's owl badge,
    # which is a smooth ring that fills a window as evenly as a hatch does.
    # Per group, without tiling, neither measure separates at all: the chevron
    # reads 0.121 fill and 4.26 spread, sitting among the etk800 legend row
    # (0.159, 4.58) and the AUTO ring (0.125, 4.00), while the dial bar --
    # trim, not pattern -- reads 10.54 spread, out among the hatch.
    #
    # Nothing is removed here.  The stage marks.
    #
    # 2026-08-28: its own grouping distance, not `mark_gap_px`.  That one is
    # tuned to join the strokes of a letter and is two texels; a repeating
    # material's elements stand much further apart than that, so at 2 a pedal
    # pad's dot grid arrives as 69 separate groups of forty texels and never
    # looks like an area at all.  At 6 each pad is one group -- measured, the
    # two scintilla pads come in at 1,526 and 997 texels -- and opening it
    # further changes almost nothing, since by then everything that belongs
    # together has already met.
    pattern_gap_px: int = 6
    # 500, from the smaller of those two pads at 997.  Generous on purpose: an
    # area that turns out not to repeat costs one transform that says no.
    # Across the corpus it offers 19, 19, 2 and 21 groups.
    min_pattern_texels: int = 500
    # Moderately dense: the group's seed over the part of its bounding
    # rectangle that lies inside the UV island.  A floor, not a discriminator
    # -- it is here so a single sprawling wisp does not qualify.  Measured
    # against the whole rectangle the four patterns in the corpus run 0.029 to
    # 0.144; against the island intersection they can only read higher, since
    # that is the same count over a smaller area.
    min_pattern_fill: float = 0.02
    # 2026-08-28: and how much surface the group actually spans.
    #
    # `min_pattern_texels` counts what fired, which is a poor proxy for how
    # much material a group covers: an edge response traces outlines, so a
    # sprawling hatch and a compact badge can hold the same number of texels.
    # Lowering it far enough to catch a faint hatch therefore drags in a crowd
    # of small marks, and the transform confirms some of them -- a 38x31 blob
    # repeats about as convincingly as anything else its size.
    #
    # So two floors on the area instead, both taken over the part of the
    # group's bounding rectangle that lies inside the UV island: how big it is,
    # and what share of the island it is.
    #
    # Coverage is the one that does the work, and it works because of what a
    # repeating material *is*: the surface of a panel, wall to wall.  A glyph
    # sits on a panel and takes a corner of it.  Measured at gap 8 and 250
    # texels, every genuine pattern in the corpus covers at least 0.387 of its
    # island -- the hatch, lowest of them -- and most cover 0.85 to 1.00, while
    # the ten groups the transform confirmed that should not have been cover
    # 0.008 to 0.109.
    #
    # The area floor guards the case coverage cannot see: a small island that a
    # small mark happens to fill entirely reads 1.00 and means nothing.  The
    # smallest genuine pattern spans 3,311 texels of surface, so 3,000.
    min_pattern_area: int = 3000
    min_pattern_coverage: float = 0.30
    # 2026-08-28: and here is the transform the stage above exists to feed.
    #
    # Each candidate group is judged whole, on two readings of its own
    # spectrum, and it has to clear both.
    #
    # `min_repeat_energy` is how much of the group's spectral energy sits in
    # its sixteen strongest bins, counting only bins that are off both axes.
    # A repeating lattice puts nearly everything into a few peaks.  The
    # off-axis part is what stops a straight edge scoring: a bar's transform is
    # a ridge along one axis through the origin, and by peak height alone the
    # etk800 dial bar was the single most "periodic" thing in the corpus at
    # 2,971 times the mean bin.  Off the axes it reads 0.001.
    #
    # `min_repeat_overlap` is the best that the group shifts onto itself --
    # its autocorrelation at the strongest non-zero lag, over the value at zero
    # lag, with the shrinking overlap divided back out so a long shift is not
    # penalised for having less to agree over.  Computed from the same
    # transform, since an autocorrelation is the inverse transform of a power
    # spectrum.
    #
    # Neither separates on its own and they fail on different things, which is
    # why both.  Measured over the corpus: the owl badge is two mirrored halves
    # and shifts onto itself at 0.794, higher than any pattern but the chevron
    # -- and its off-axis energy is 0.013.  The D-pad ring's energy is 0.117,
    # up among the patterns -- and it shifts onto itself at 0.201.
    #
    # 0.07 and 0.40 take all seven repeating materials found in the corpus:
    # the hatch (0.379/0.805), the perforation (0.270/0.654), a paddle shifter
    # (0.195/0.730), the chevron tread (0.130/0.908), both pedal faces
    # (0.113/0.582 and 0.110/0.526) and ardente's honeycomb grille
    # (0.108/0.403).  They leave the owl badge, the Cinetta script, the ETK
    # roundel, the dial bar and the D-pad ring.
    #
    # 2026-08-28: a run of seat stitching (0.178/0.629) and a trim strip
    # (0.141/0.709) cleared both of these, and the cycle floor below now takes
    # them: they read 3.6 and 4.5 cycles.
    #
    # 2026-08-28: the transform was also tried on the *image* under each group
    # rather than on the detections in it, on the reasoning that the picture
    # holds far more of the repeat than a sparse mask of fired texels does.  It
    # is not better here, and it is worse in a way worth recording: the
    # readings all rise, but they rise for the drawn marks too, because the
    # material a badge sits on has a weave of its own that has nothing to do
    # with whether the badge repeats.  On the same sixteen landmarks the image
    # takes the Cinetta script (0.161/0.527/10.4 against the pedal face's
    # 0.165/0.871/10.8) while the mask leaves it at 0.041 energy.  Asking about
    # the detections is asking the question that was meant.
    #
    # 2026-08-28: and the autocorrelation has to dip before it peaks.
    #
    # `min_repeat_dip` is the best self-overlap minus the lowest point on the
    # way out to it, walking the ray from the origin to that lag.  A repeating
    # material's autocorrelation oscillates: it falls away from the origin,
    # troughs between one cell and the next, and comes back up at the lattice
    # vector.  A single edge's only decays -- shift it a little along itself
    # and it still overlaps, shift it more and it overlaps less, and there is
    # never a rise.  So the trough is the whole of the difference, and the
    # height of the peak is none of it: an edge overlaps itself 0.67 to 0.97,
    # which is why `min_repeat_overlap` cannot do this on its own.
    #
    # It needs no axis exclusion, which is what recommends it over the two
    # measures it replaced.  Both of those read the spectrum: first where the
    # peak sat in cycles across the crop, then how far it stood above the
    # excluded DC core.  Each separated on the landmarks it was fitted to and
    # then failed on the next case.  The cycle count failed because a coarse
    # pattern in a small crop repeats few times by definition -- ardente's
    # honeycomb pedal faces are five or six cells across and peak at 5.7, 6.3
    # and 3.6 cycles, and 3.6 is exactly what etk800's steering-wheel panels
    # read.  The rise failed more quietly: it separated, but only just, and
    # widening `pattern_gap_px` to 12 walked a wheel panel from 1.97 up to 2.4
    # and through a bar of 2.2, taking every legend on the panel with it.
    #
    # Measured over every landmark that clears the energy bar: the trim strip
    # and the D-pad ring 0.00, the wheel panels 0.07, a column of dial digits
    # 0.11, a run of stitching 0.29 -- against the honeycomb grille at 0.40, a
    # pedal face at 0.53, three more at 0.59 to 0.72, scintilla's chevron at
    # 0.87 and the hatch at 1.01.  0.35 sits in the gap.
    #
    # Unlike the rise it does not move with the grouping: at candidate gaps of
    # 6, 8, 12 and 16 the corpus confirms nothing at all on etk800 and the same
    # 8,362 texels on ardente.
    #
    # What it does not do is rescue an axis-aligned stripe -- see
    # `min_repeat_energy`.  Pure autocorrelation cannot replace that test
    # either: the owl badge is two mirrored halves, so it dips 0.85 on its way
    # to overlapping one eye onto the other, right among the patterns, and only
    # the spectral energy refuses it at 0.012.
    #
    # Any of the three at 0 switches the transform off and the stage passes
    # everything.
    min_repeat_energy: float = 0.07
    min_repeat_overlap: float = 0.40
    min_repeat_dip: float = 0.35
    # Lags shorter than this are the mark overlapping itself rather than
    # repeating: at zero lag everything scores 1, and it falls away over the
    # width of a stroke.
    repeat_skip_px: int = 6
    # Longest side before the crop is reduced.  A repeating material's period
    # is many texels, so halving the resolution costs nothing and it keeps the
    # transform on a 1,011 x 110 hatch the same size as one on a 97 x 167 pedal.
    repeat_longest_px: int = 512
    # 2026-08-28: where bridging is allowed to happen at all -- read off the
    # *voids* in the threshold mask rather than off anything that fired.
    #
    # A drawn mark sits against blank ground.  A material that is merely noisy
    # has no blank ground anywhere: its threshold response is spread evenly, so
    # however much of it there is, none of the surface is ever clear.  So the
    # reading is the share of a neighbourhood lying at least `bridge_clear_px`
    # from anything over the threshold, and bridging is allowed only where that
    # share clears `min_bridge_void`.
    #
    # Measured at ten sites, share of an 81-texel window five texels clear:
    # the four grain sites read 0.013, 0.025, 0.026 and 0.061, and everything
    # drawn reads 0.148 (the badge), 0.228 (a slot outline), 0.314 (a broken
    # diagonal), 0.461 (the play triangle) and 0.593 (the START/STOP legend).
    # 0.10 sits in a gap of 2.4x.
    #
    # Every reading of where the threshold *is* fails, and fails on the same
    # case.  Density at windows of 5, 9 and 21 reads 0.06-0.12 on grain against
    # 0.04-0.15 on marks, overlapping outright -- and it orders them backwards
    # at island scale, where grain is the densest thing on the atlas (0.08) and
    # the stitching the sparsest (0.02).  Clumping -- the mean size of the
    # threshold component a texel sits in -- gives 3.2-5.4 on grain against
    # 22.5-29.5 on the triangle, the slot and the legend, and then 3.0 on the
    # scintilla badge.  The density of threshold runs of nine texels or more
    # gives 0.006-0.017 against 0.046-0.100, and 0.000 on the badge.  The badge
    # is a dashed outline: at the scale of what fired it is indistinguishable
    # from grain, and only the clear ground around it says otherwise.
    #
    # 5 and 81, both swept.  A clearance of 3 overlaps (a slot outline reads
    # 0.306 against grain at 0.359) and 8 reads 0.000 on the badge, level with
    # grain.  A 41-texel window is too small for the badge (0.083 against grain
    # at 0.114) and 121 reads much as 81 does.  Reading the void over the whole
    # UV island rather than a window does not work at all: the corpus domains
    # are coalesced and run to half a million texels, so a slot outline in a
    # large island reads 0.016 against grain at 0.070.
    #
    # Two cheaper readings that need no clearance were tried and are thinner:
    # the mean distance to anything lit separates 2.42 -> 2.81, and the largest
    # clearance in the window 7.3 -> 9.8, against this one's 0.061 -> 0.148.
    #
    # 2026-08-28: this replaced a gate on *survival* -- the seed over the
    # threshold mask, how much of what fired was believed -- which separated
    # 0.26 -> 0.44 and worked, but reads the pipeline's own later decisions and
    # so is a poor backstop for them.  On the same landmarks the void gate is
    # slightly better where it counts: the play triangle repairs to 357 texels
    # against 323, the badge to 398 against 388, with the grain sites
    # identical.
    #
    # What neither catches is a corrugated surface -- scintilla's pleated trim
    # reads 0.347 here, squarely among the marks, because ribs standing well
    # apart leave plenty of clear ground between them.  It is a repeating
    # material and belongs to the transform at stage 6, not to this gate.
    #
    # `min_bridge_void` at 0 switches the gate off.
    bridge_clear_px: int = 5
    bridge_void_px: int = 81
    min_bridge_void: float = 0.10
    # 2026-08-28: how long a crossing from one piece of a mark to another may
    # be before the two stop being the same mark, and how much of the threshold
    # the crossing has to hold along the way.
    #
    # This is the answer to a group that arrives in pieces.  A drawn mark does
    # not stop being drawn where the response dips under the bar -- a corner of
    # ardente's play triangle, a run of stitching, the dashes of a panel
    # outline -- and `mark_gap_px` cannot rescue it, because that one distance
    # has to serve every mark on the atlas at once.  Measured, opening it is
    # the wrong instrument: gap 2 -> 4 takes scintilla from 101 marks to 58,
    # and the same widening welds etk800's radio faceplate trim into a group
    # that already spans the whole panel.
    #
    # So the distance stays where it is and the picture decides each join
    # instead.  What is asked is the one thing that separates a broken stroke
    # from two neighbours: a stroke that broke leaves a trace under the bar,
    # and blank ground does not.  Measured over the two dashboards, the
    # response as a fraction of the local threshold: on the seed itself a
    # median 1.54-1.71, within 8 texels of it p90 0.42-0.49 and p99 0.93-0.95,
    # and more than 16 texels out p99 0.00-0.28.  There is a signal beside a
    # mark and nothing at all away from one.
    #
    # `bridge_span_px` is the whole crossing, piece to piece, not how far the
    # trace may reach from either end -- and that distinction is the difference
    # between the rule working and not.  Bounding the reach instead admits
    # every spur the trace throws off, because a spur that touches the mark
    # anywhere rides in on the run that made a genuine join: measured, the slot
    # outlines on scintilla's hatched trim grew a radial fringe of 400 to 470
    # texels each, and two concentric rings of a dial welded.  Cutting dead
    # ends back does not fix it either -- a spur two texels wide has no end to
    # cut.  Asking for the length of the crossing *through* each texel refuses
    # all of it by construction: the same slots gain 12 to 46 texels, where
    # their outline really had broken, and the rings stay apart.
    #
    # The crossing is estimated rather than walked: a texel's distance to the
    # nearest piece, plus the way on from there to the ridge where that piece's
    # ground gives way to another's, plus that ridge texel's own distance to a
    # piece.  Three distance transforms and no per-pair work.  It over-states
    # the true crossing, so what it admits is a subset of what a walk would.
    #
    # 16 and 0.40 -- and both are this generous only because
    # `min_bridge_void` above decides *where* bridging may happen at all.
    # Before that gate existed these were 10 and 0.60 and could not be opened:
    # every relaxation was spent on grain rather than on marks.  With the place
    # settled first the same relaxation reaches the marks instead, which is the
    # whole argument for gating the place rather than the bridge.
    #
    # Swept at spans 10/16/20 and floors 0.60/0.50/0.40/0.30 with the gate on.
    # The play triangle goes 63 texels -> 323, scintilla's badge outline
    # 167 -> 388, the START/STOP legend 316 -> 431, a slot outline 361 -> 463.
    # Every one of the eight grain sites stays at exactly the size it was
    # before bridging, and the largest mark on each texture holds at 8,767,
    # 2,672, 6,400 and 4,294 texels across the entire sweep.
    #
    # It breaks at floor 0.30 with span 20 and not before: a grain site on
    # ardente goes 15 texels -> 122 there.  0.40/16 is the last setting inside
    # the shelf, not the middle of it, so move it down rather than up.
    #
    # What the reach buys past a repair is fill: at 16 a crossing will run
    # between a mark's own parts -- the bands of the play triangle, the
    # counters of a letter -- rather than only across a break.  That is
    # deliberate and was chosen; it is also what the span controls, so a
    # narrower one is the dial if a mark should stay hollow.
    #
    # Either at 0 switches the stage off.
    bridge_span_px: int = 16
    min_bridge_response: float = 0.40
    # 2026-08-28: and how much of a mark a bridge is allowed to be.  A bridge
    # is a repair, not a construction.
    #
    # Measured as the run's texels over the detected texels of the pieces it
    # joins.  What this refuses is a grain field: on ardente's mottled panels
    # the seed arrives as a scatter of specks a few texels each, the trace
    # between them is as raised as the specks are, and the crossings are
    # short -- so every test above passes and the bridge knits the scatter
    # into a constellation.  Watched at stage 7 it is unmistakable: the field
    # is saturated at the threshold, thinned to a dozen specks by the
    # structure gate, and the bridge puts half of them back.
    #
    # Nothing about the bridge itself tells them apart.  Measured over the
    # runs at fifteen grain sites and nine real ones, the run's own structure
    # reading is 1.12-1.36 for grain against 1.00-1.68 for the real ones --
    # the play triangle's own repair reads 1.09, *below* every grain site --
    # and its response, its thinness, and the density of the trace around it
    # all overlap outright.  What separates is only the size of the repair
    # against the thing repaired: grain runs 0.39-2.86 with a median of 1.13,
    # the real ones 0.08-0.56 -- a slot outline 0.08, a stitch 0.10, the
    # triangle 0.38, the badge outline 0.55.
    #
    # 2026-08-28: OFF by default, and superseded rather than wrong.  This was
    # the instrument before the void gate existed, and at 0.60 it took
    # six of the eight grain sites; the gate takes all eight, and leaves the
    # marks fuller while doing it.  What is left for the cap is one mark in the
    # whole corpus -- two specks on etk800 joined into a 31-texel blob -- which
    # is not worth what it costs.
    #
    # And it costs a great deal, because it works against leniency.  A lower
    # floor offers more trace, which makes every run larger, which makes the
    # cap refuse it: measured with the cap at 0.60, dropping the floor from
    # 0.60 to 0.50 *loses* the scintilla badge repair outright, 270 texels
    # back to 167.  With the cap off the same drop takes it to 314.  The two
    # cannot both be turned up, so the one that separates better wins.
    #
    # Kept because the measure is real and one line brings it back.  0.60 is
    # the value to use if it is ever wanted: swept at 0.25, 0.40 and 0.80,
    # 0.40 loses the badge outline entirely and 0.80 lets a grain site in.
    #
    # 2026-08-28: reusing the texels the *threshold* stage found and later
    # stages dropped -- rather than the sub-threshold trace -- was tried here
    # first, and it is worse on both counts.  It misses the repairs, because
    # the evidence a broken stroke leaves is under the bar by definition:
    # stage 1 never fired in the break, so the substrate is empty exactly
    # where the repair belongs.  The play triangle falls from 203 texels to
    # 150, the scintilla badge outline from 272 to 191, the START/STOP legend
    # from 362 to 327.
    #
    # And it lets *more* grain through, which is the part worth writing down
    # because it runs the other way to intuition.  A sparse substrate makes a
    # bad bridge look cheap.  Measured at two grain sites, per run: the trace
    # is dense and unbroken across a grain field, so one run sweeps up the
    # whole area and costs a median 16 and 14 texels against the 16 and 17 it
    # joins -- 1.00 and 0.82, refused.  The threshold's leavings are isolated
    # dots, so the same joins arrive as runs of 9 and 6 texels -- 0.33 and
    # 0.40, admitted.  The cap only works because the substrate is generous
    # enough for a bridge that is doing something stupid to look expensive.
    # On the badge the same numbers run the other way: one 96-texel trace run
    # against 176 texels of outline is 0.55 and gets in, where the leavings
    # give three 7-texel runs that repair nothing.
    #
    # 0 switches the cap off and any crossing is allowed.
    max_bridge_share: float = 0.0
    # How big a void has to be before it gets an interior path of its own.
    # A dial bezel encloses one; the speckle of unfired texels inside a
    # thick digit does not, and outlining every hole would bury the glyph.
    min_void_texels: int = 36
    # Thickness of the painted path, in texels.  One erosion per texel.
    outline_px: int = 1

@dataclass(frozen=True, slots=True)
class LabConfig:
    """Everything one run of the lab is parameterised by."""

    contrast: ContrastConfig = field(default_factory=ContrastConfig)


@dataclass(frozen=True, slots=True)
class Detection:
    """One texture's result: the marks, and the picture they summarise."""

    regions: list[Region]
    fired: np.ndarray       # texels that survived to the marks stage
    outlines: np.ndarray    # every mark's painted path, atlas-wide
    # The blank ground the marks stand on, atlas-wide.  Always built, unlike
    # the captured stages: it is one box filter, and it is the layer the viewer
    # draws over whatever else is on screen rather than in place of it.
    sparse: np.ndarray
    # Every stage of the pipeline, atlas wide, when the viewer asked for them.
    # Packed to bits: a 4096 atlas is 16.8 MB a stage as bool and 2.1 MB packed.
    stages: dict[str, np.ndarray] = field(default_factory=dict)
    # 2026-08-29: each stage's *own* region paths and count, alongside its
    # texels.  Without these the viewer drew the finished outlines over every
    # stage's texels, so stepping back through the stages showed where a mark's
    # texels went but never which stage stopped calling it a region -- which is
    # the one question the stage picker exists to answer.
    stage_outlines: dict[str, np.ndarray] = field(default_factory=dict)
    stage_counts: dict[str, int] = field(default_factory=dict)
    # Seconds spent reaching each stage, summed over the texture's domains.
    # Always collected -- it is two `perf_counter` calls a stage -- because the
    # cost of a step is part of judging it, and having to switch capture on to
    # see it would mean never looking.  Excludes the capture's own work.
    stage_times: dict[str, float] = field(default_factory=dict)

    @property
    def kept_texels(self) -> int:
        return int(np.count_nonzero(self.fired))

    def stage_cost(self) -> list[tuple[str, float, float]]:
        """Each timed stage with its seconds and its share, in pipeline order."""
        total = sum(self.stage_times.values()) or 1.0
        return [
            (name, self.stage_times[name], self.stage_times[name] / total)
            for name in STAGES if name in self.stage_times
        ]

    def stage_mask(self, name: str) -> np.ndarray | None:
        """One captured stage, unpacked, or None if it was never captured."""
        return self._unpack(self.stages.get(name))

    def stage_outline(self, name: str) -> np.ndarray | None:
        """One captured stage's region paths, or None where it drew none."""
        return self._unpack(self.stage_outlines.get(name))

    def _unpack(self, packed: np.ndarray | None) -> np.ndarray | None:
        if packed is None:
            return None
        return np.unpackbits(packed, count=self.fired.size).astype(bool).reshape(
            self.fired.shape
        )


def disc_counts(lit: np.ndarray, diameter: int) -> np.ndarray:
    """How many lit texels lie within a disc of ``diameter`` of each texel.

    2026-08-28: a disc, not a square.  A square window reaches sqrt(2) further
    on the diagonal than on the axes, so "is there anything near me" is asked
    over a different distance depending on which way the neighbour lies -- and
    the answer is a picture of blank ground, where that bias shows up as
    corners.  A disc asks the same question in every direction.

    Built as one prefix sum along x and two slices per row of the disc.  A disc
    is not separable, but each of its rows is a span, and a span is a
    difference of two running totals -- so the whole thing is ``diameter``
    subtractions of whole slices, with no filter and no float arithmetic.

    2026-08-29: it counts a boolean mask, so it counts in integers.  The
    version before this ran ``diameter`` separable box filters over float32 and
    scaled each back up, which is the same answer through a multiply, a divide
    and a cast.  Measured on a 4096 atlas at diameter 11, that and the mirrored
    rows below together take it from 0.84 s to 0.39 s -- bit for bit identical,
    and still checked against a brute-force convolution with the disc footprint
    at diameters 5, 9, 15 and 21.

    Note that a disc holds far fewer texels than the square around it -- 49
    against 81 at diameter 9 -- so the same count is a more generous test.
    """
    radius = max(int(diameter), 1) // 2
    rows, columns = lit.shape
    # The running total along each row, padded by the radius on both sides so
    # a span that hangs off the end reads as the row's own end rather than
    # wrapping or needing a clamp per texel.
    running = np.zeros((rows, columns + 2 * radius + 1), dtype=np.int32)
    np.cumsum(
        lit, axis=1, dtype=np.int32,
        out=running[:, radius + 1:radius + 1 + columns],
    )
    running[:, radius + 1 + columns:] = running[
        :, radius + columns:radius + 1 + columns
    ]
    total = np.zeros((rows, columns), dtype=np.int32)
    # 2026-08-29: the disc's rows are mirrored about its middle, so a row above
    # and the row below it are the same span -- computed once and added at both
    # shifts.  Measured on a 4096 atlas at diameter 11: 0.39 s against 0.47 s.
    for offset in range(radius + 1):
        half = math.isqrt(radius * radius - offset * offset)
        start = radius - half
        stop = radius + half + 1
        span = running[:, stop:stop + columns] - running[:, start:start + columns]
        if offset == 0:
            total += span
            continue
        total[:-offset] += span[offset:]
        total[offset:] += span[:-offset]
    return total


def sparse_field(
    lit: np.ndarray, sample: np.ndarray, kernel: int, count: int
) -> np.ndarray:
    """Where a disc of ``kernel`` across holds at most ``count`` of what fired.

    The blank part of the surface.  See `ContrastConfig.sparse_kernel_px`.

    Counted over the raw threshold rather than as a masked mean: a disc hanging
    off the edge of an island genuinely holds fewer lit texels, and that edge
    *is* blank ground as far as this reading is concerned.  Only the painting
    is held to the island.
    """
    if kernel <= 0 or count < 0:
        return np.zeros(lit.shape, dtype=bool)
    width = max(int(kernel), 1)
    width += 1 - width % 2  # centred, so the disc has a texel at its middle
    return (disc_counts(lit, width) <= int(count)) & sample


def island_size(domain: np.ndarray) -> float:
    """How big an island is, near enough: the mean side of its bounds.

    2026-08-29: the bounding box, not a distance transform.  The margin only
    wants a scale, and a domain already arrives cropped to its own bounds, so
    this is free where the transform was the entire cost of the stage.

    The mean of the two sides rather than either one: the shorter side alone
    calls a long panel small, and the longer alone calls a thin strip large.
    It over-reads a diagonal strip, whose box is wide in both axes while the
    strip itself is thin -- the price of measuring the box and not the shape.
    """
    height, width = domain.shape
    return (float(height) + float(width)) / 2.0


def boundary_margin(size: float, floor: int, ceiling: int, share: float) -> int:
    """How far this island's margin reaches in.  See `min_rim_margin_px`.

    A share of the island's own size, held between a floor and a ceiling, and
    whole because it is applied as that many erosions.
    """
    low = max(int(floor), 0)
    high = max(int(ceiling), low)
    return int(min(max(float(share) * float(size), low), high))


def rim_contact(
    mask: np.ndarray, rim: np.ndarray, length: int | None = None,
) -> float:
    """How much of the island's rim a region runs along, over its own length.

    ``length`` is the region's longer side.  A band lying along an edge reads
    about 1; a mark that merely
    touches one reads near 0.  It can exceed 1 for a region whose whole
    perimeter is on the rim -- a disc filling a round island -- which is left
    alone rather than capped, because such a region is deep and the margin test
    has already kept it.  See `ContrastConfig.min_rim_contact`.

    2026-08-29: over the length, not over the region's own outline.  Dividing
    by the outline lets a wiggle dilute the reading: the scalloped bands along
    etk800's ducts touch the rim for 86, 88 and 89 texels while their outlines
    run to 263, 394 and 424, so they read 0.21 to 0.33 and slipped under a bar
    of 0.35 that the straight bands did not.  Over the length the same three
    read 0.68, 0.52 and 0.43.
    """
    if not mask.any():
        return 0.0
    grown = ndimage.binary_dilation(mask, structure=NEIGHBOURHOOD)
    return float((rim & grown).sum()) / float(
        max(max(mask.shape) if length is None else int(length), 1)
    )


def rim_perimeter_contact(mask: np.ndarray, rim: np.ndarray) -> float:
    """The fraction of a region's own perimeter touching the sampled rim.

    A compact control can sit close to a UV boundary, but only one small part
    of its outline reaches it.  An edge band or a stitch lying on that edge
    spends a substantial part of its perimeter there.  This complements
    `rim_contact`: long runs retain its length-normalised score, while compact
    pieces use this relative score before they can form a stitch train.
    """
    if not mask.any():
        return 0.0
    perimeter = mask & ~ndimage.binary_erosion(
        mask, structure=NEIGHBOURHOOD, border_value=0
    )
    touching = rim & ndimage.binary_dilation(mask, structure=NEIGHBOURHOOD)
    return float(touching.sum()) / float(max(int(perimeter.sum()), 1))


def sampling_mask(domain: np.ndarray, skip: int) -> np.ndarray:
    """The part of a domain whose colour is a measurement of the surface.

    The domain eroded by ``skip``, so the partial-coverage texels the UV
    rasteriser leaves around every island are neither read nor answered for.
    See `ContrastConfig.boundary_skip_px`.

    Eroded with the array border counted as outside, which is right here and
    would not be in general: a domain arrives cropped to its own bounding box,
    so a texel against the edge of that crop is on the island's rim by
    construction.

    A domain thin enough to erode away entirely keeps nothing, and that is the
    honest answer rather than an edge case -- a domain that vanishes under one
    erosion is two texels across, which is partial coverage all the way
    through, with no surface in it to measure.  Measured at skip 1 over the
    corpus: 2 domains of 831, one of 13 texels on scintilla_interior and one of
    176 on ardente.  The 176 is a long two-wide ribbon, not a panel.
    """
    depth = max(int(skip), 0)
    if depth == 0:
        return domain
    return ndimage.binary_erosion(
        domain, structure=NEIGHBOURHOOD, iterations=depth, border_value=0
    )


def pattern_candidates(
    seed: np.ndarray, sample: np.ndarray, gap: int, config: ContrastConfig
) -> np.ndarray:
    """Groups big enough, dense enough and wide enough to be worth transforming.

    Grouped by ``gap``, which is the candidate stage's own distance and wider
    than the one the marks stage uses -- see `ContrastConfig.pattern_gap_px`.

    Returns the candidate texels, whole groups at a time.  Empty when switched
    off.  See `ContrastConfig.min_pattern_texels`.
    """
    min_texels = int(config.min_pattern_texels)
    if min_texels <= 0 or not seed.any():
        return np.zeros(seed.shape, dtype=bool)
    gap = max(int(gap), 0)
    grown = (
        ndimage.binary_dilation(seed, structure=NEIGHBOURHOOD, iterations=gap)
        if gap else seed
    )
    labels, count = ndimage.label(grown, structure=NEIGHBOURHOOD)
    if count == 0:
        return np.zeros(seed.shape, dtype=bool)
    texels = np.bincount(labels[seed], minlength=count + 1)
    keep = texels >= min_texels
    island = int(np.count_nonzero(sample))
    if config.min_pattern_fill > 0 or config.min_pattern_area > 0 or (
        config.min_pattern_coverage > 0
    ):
        # 2026-08-28: density over the part of the group's rectangle that is
        # actually surface, not over the whole rectangle.
        #
        # A rectangle drawn round a group on a chart that is not itself
        # rectangular hangs off the island on at least one side, and every
        # texel of that overhang is somewhere nothing could ever have fired.
        # Counting it as area makes a perfectly ordinary group look sparse for
        # no reason but where it was packed.
        #
        # Bounds from the group's own texels rather than from `find_objects`,
        # which returns the bounds of the dilation -- `gap` wider on every side
        # than anything that was detected.
        for index, slices in enumerate(ndimage.find_objects(labels), start=1):
            if slices is None or not keep[index]:
                continue
            rows, columns = slices
            shape = (labels[rows, columns] == index) & seed[rows, columns]
            down = np.flatnonzero(shape.any(axis=1))
            across = np.flatnonzero(shape.any(axis=0))
            if not down.size:
                keep[index] = False
                continue
            box = np.s_[
                rows.start + down[0]:rows.start + down[-1] + 1,
                columns.start + across[0]:columns.start + across[-1] + 1,
            ]
            area = int(np.count_nonzero(sample[box]))
            keep[index] = (
                area >= max(config.min_pattern_area, 1)
                and texels[index] >= config.min_pattern_fill * area
                and area >= config.min_pattern_coverage * max(island, 1)
            )
    keep[0] = False
    return keep[labels] & seed


def repeat_scores(
    mask: np.ndarray, skip: int = 6, longest: int = 512
) -> tuple[float, float, float]:
    """Three readings of one group's spectrum.

    Peak energy, self-overlap, and how far the autocorrelation dips before it
    reaches that overlap.

    See `ContrastConfig.min_repeat_energy` for what each one is and what they
    read at each landmark.  Zeros for anything too small to transform.

    Reduced to ``longest`` on its longest side first, by OR-ing whole blocks
    together, so the cost does not run away with the group's size.
    """
    step = max(1, int(np.ceil(max(mask.shape) / max(int(longest), 16))))
    # 2026-08-28: never reduce the short side below what can be transformed.
    # A 23 x 622 strip of stripes was cut to 11 x 311 by its long side and then
    # refused for being under 16 wide, so the reading came back three zeros and
    # the strip read as the least periodic thing in the corpus.
    step = min(step, max(min(mask.shape) // 16, 1))
    if step > 1:
        down = mask.shape[0] // step * step
        across = mask.shape[1] // step * step
        if down < step or across < step:
            return 0.0, 0.0, 0.0
        mask = mask[:down, :across].reshape(
            down // step, step, across // step, step
        ).any(axis=(1, 3))
        skip = max(int(skip) // step, 2)
    height, width = mask.shape
    if min(height, width) < 16:
        return 0.0, 0.0, 0.0
    values = mask.astype(np.float32)
    values -= values.mean()

    # A Hann window in both axes.  A rectangular crop leaks a cross along the
    # axes that is about the crop's edges and not about what is inside it.
    windowed = values * (
        np.hanning(height)[:, None] * np.hanning(width)[None, :]
    )
    power = np.abs(np.fft.rfft2(windowed)) ** 2
    rows = np.fft.fftfreq(height)[:, None] * height
    columns = np.fft.rfftfreq(width)[None, :] * width
    # Away from DC, which describes the group's own outline rather than
    # anything inside it...
    band = (rows ** 2 + columns ** 2) >= 9.0  # away from DC
    # ...and off both axes, so a straight edge cannot score.
    plane = band & (np.abs(rows) >= 2.0) & (np.abs(columns) >= 2.0)
    total = float(power[band].sum())
    energy = (
        float(np.sort(power[plane])[::-1][:16].sum() / total)
        if total > 0 and plane.any() else 0.0
    )

    # The autocorrelation, from the same transform.  Padded to twice the crop
    # so the correlation is linear rather than wrapping round the edges.
    padded = (2 * height, 2 * width)
    spectrum = np.fft.rfft2(values, s=padded)
    whole = np.fft.fftshift(np.fft.irfft2(np.abs(spectrum) ** 2, s=padded))
    lags = whole[
        height - height // 2:height + height // 2,
        width - width // 2:width + width // 2,
    ]
    down, across = np.array(lags.shape) // 2
    middle = float(lags[down, across])
    if middle <= 0:
        return energy, 0.0, 0.0
    ys = np.arange(lags.shape[0])[:, None] - down
    xs = np.arange(lags.shape[1])[None, :] - across
    far = (ys ** 2 + xs ** 2) >= max(int(skip), 1) ** 2
    if not far.any():
        return energy, 0.0, 0.0
    # Divided by how much of the crop still overlaps at each lag, so a long
    # shift is not penalised for having less to agree over.
    share = np.maximum(
        (height - np.abs(ys)) * (width - np.abs(xs)), 1
    ).astype(np.float32) / float(height * width)
    curve = np.where(far, lags / share, -np.inf) / middle
    at = np.unravel_index(np.argmax(curve), curve.shape)
    overlap = float(curve[at])

    # ...and how far it dips on the way there.  See
    # `ContrastConfig.min_repeat_dip`.
    step_down, step_across = at[0] - down, at[1] - across
    reach = float(np.hypot(step_down, step_across))
    if reach < 2:
        return energy, overlap, 0.0
    whole_curve = (lags / share) / middle
    # Sampled from just clear of the origin rather than from `skip`, which is
    # there to keep the peak search off the self-overlap plateau -- a pattern
    # whose period is near it would otherwise leave no room for a trough.
    along = np.linspace(0.0, 1.0, max(4 * int(reach), 32))
    trough = min(
        float(whole_curve[
            round(down + step_down * t), round(across + step_across * t)
        ])
        for t in along if np.hypot(step_down * t, step_across * t) >= 2.0
    )
    return energy, overlap, overlap - trough


def confirmed_repeats(
    candidates: np.ndarray, gap: int, config: ContrastConfig
) -> np.ndarray:
    """Of the candidate groups, the ones the transform says do repeat.

    Whole groups, in or out: the question is what the run of texels *is*, and
    half a hatch is still a hatch.
    """
    if (
        config.min_repeat_energy <= 0
        or config.min_repeat_overlap <= 0
        or config.min_repeat_dip <= 0
    ):
        return np.zeros(candidates.shape, dtype=bool)
    if not candidates.any():
        return np.zeros(candidates.shape, dtype=bool)
    gap = max(int(gap), 0)
    grown = (
        ndimage.binary_dilation(
            candidates, structure=NEIGHBOURHOOD, iterations=gap
        ) if gap else candidates
    )
    labels, count = ndimage.label(grown, structure=NEIGHBOURHOOD)
    if count == 0:
        return np.zeros(candidates.shape, dtype=bool)
    keep = np.zeros(count + 1, dtype=bool)
    for index, slices in enumerate(ndimage.find_objects(labels), start=1):
        if slices is None:
            continue
        crop = (labels[slices] == index) & candidates[slices]
        if not crop.any():
            continue
        energy, overlap, dip = repeat_scores(
            crop, config.repeat_skip_px, config.repeat_longest_px
        )
        keep[index] = (
            energy >= config.min_repeat_energy
            and overlap >= config.min_repeat_overlap
            and dip >= config.min_repeat_dip
        )
    keep[0] = False
    return keep[labels] & candidates


def local_share(
    mask: np.ndarray, sample: np.ndarray, window: int
) -> np.ndarray:
    """What share of each texel's neighbourhood is set, in-domain texels only.

    The same masked-mean shape as `seed_density` and for the same reason: a
    window hanging off the edge of an island would read as empty and every
    texel near a seam would be misjudged.  Separable, so the width is free.
    """
    width = max(int(window), 1)
    width += 1 - width % 2
    inside = ndimage.uniform_filter(
        sample.astype(np.float32), size=width, mode="constant"
    )
    part = ndimage.uniform_filter(
        (mask & sample).astype(np.float32), size=width, mode="constant"
    )
    return part / np.maximum(inside, 1e-6)


def gap_bridges(
    seed: np.ndarray, ratio: np.ndarray, sample: np.ndarray,
    span: int, floor: float, gap: int, share: float = 0.0,
    void: float = 0.0, clear: int = 0, window: int = 0,
    threshold: np.ndarray | None = None,
) -> np.ndarray:
    """The under-bar texels lying on a short crossing from one piece to another.

    ``ratio`` is the response over the threshold it was held to, so 1 is the
    bar and everything here is below it.  See `ContrastConfig.bridge_span_px`
    for what this is for, what the readings are, and why the test is the length
    of the crossing rather than the reach of the trace, and
    `max_bridge_share` for how big a repair may be against what it repairs.

    Three distance transforms and two rank filters, none of them per pair, and
    the share is two `bincount`s over the runs.

    2026-08-28: the pieces are the seed closed at ``gap`` -- the same distance
    the marks stage groups at -- and the trace is taken from outside that
    closing, never from beside a stroke.  Admitting the skirt a stroke's own
    falloff leaves adds 20-45% to every mark on the corpus, which is not
    completing anything; it is drawing each stroke wider.
    """
    if span <= 0 or floor <= 0 or not seed.any():
        return np.zeros_like(seed)
    gap = max(int(gap), 0)
    close = (
        ndimage.binary_dilation(seed, structure=NEIGHBOURHOOD, iterations=gap)
        if gap else seed
    )
    bridge = (ratio >= floor) & sample & ~close
    if void > 0 and clear > 0 and window > 0 and threshold is not None:
        # Only where the surface has clear ground on it.  See
        # `ContrastConfig.min_bridge_void`.
        far = ndimage.distance_transform_edt(~(threshold & sample))
        bridge &= local_share(far >= clear, sample, window) >= void
    if not bridge.any():
        return bridge
    pieces, count = ndimage.label(close, structure=NEIGHBOURHOOD)
    if count < 2:
        return np.zeros_like(seed)
    # Which piece owns the ground each texel stands on, and how far off it is.
    near, index = ndimage.distance_transform_edt(~close, return_indices=True)
    first = pieces[tuple(index)]
    # The ridge is where one piece's ground gives way to another's, so the way
    # on to a *different* piece runs through it: out to the ridge, and then as
    # far again as the ridge texel's own nearest piece.  That over-states the
    # true crossing -- the two legs need not meet at a right angle -- so the
    # span admits a subset of what walking every path would.
    ridge = (ndimage.maximum_filter(first, size=3)
             != ndimage.minimum_filter(first, size=3))
    if not ridge.any():
        return np.zeros_like(seed)
    reach, at = ndimage.distance_transform_edt(~ridge, return_indices=True)
    bridge &= (near + reach + near[tuple(at)]) <= float(span)
    if not bridge.any():
        return bridge
    # ...and the run a texel belongs to has to arrive at two pieces, not come
    # back to the one it left.  Grouped at ``gap`` so that a run and the pieces
    # it lands on are read at exactly the distance the marks stage will use.
    runs, many = ndimage.label(
        ndimage.binary_dilation(bridge, structure=NEIGHBOURHOOD, iterations=gap)
        if gap else bridge,
        structure=NEIGHBOURHOOD,
    )
    met = close & (runs > 0)
    touched = np.unique(np.stack([runs[met], pieces[met]]), axis=1)
    keep = np.bincount(touched[0], minlength=many + 1) >= 2
    if share > 0:
        # What the run costs, against the detected texels of everything it
        # lands on.  Both are counted over the whole run rather than per pair,
        # so a run reaching three pieces is judged as the one repair it is.
        piece_texels = np.bincount((pieces * seed).ravel(), minlength=count + 1)
        piece_texels[0] = 0
        repaired = np.zeros(many + 1, dtype=np.int64)
        np.add.at(repaired, touched[0], piece_texels[touched[1]])
        cost = np.bincount(runs[bridge], minlength=many + 1)
        keep &= cost <= float(share) * np.maximum(repaired, 1)
    keep[0] = False
    return keep[runs] & bridge


def marks_from_seed(
    seed: np.ndarray, gap: int, min_stroke: int, drawn: np.ndarray | None = None
) -> np.ndarray:
    """Group the seed into marks, and drop the groups that are only dust.

    One test, per group and whole: does the group hold a single unbroken
    fragment big enough to have been drawn.  See
    `ContrastConfig.min_stroke_texels` for what it is for and what was tried
    before it, and `mark_gap_px` for why there is no size test alongside it.

    ``drawn`` is the part of ``seed`` that was actually detected, where the
    stage before this one has bridged the gaps and the rest is under-bar
    evidence.  It defaults to all of it.

    2026-08-28: the stroke is measured on what was drawn and never on a bridge.
    A bridge says two pieces belong together; it does not say either of them
    was drawn, and counting it as stroke lets a bridged pair of specks clear a
    bar neither could -- measured, that *raises* the mark count above the
    unbridged pipeline on three of the four corpus textures (237 -> 242,
    13 -> 19, 225 -> 253) as dust arrives as new marks.  Read on the seed alone
    the same settings give 225, 13 and 204.

    Everything is a labelled `bincount`; there is no loop over the groups.
    ``np.maximum.at`` over the fragments is the only scattered write, and it
    runs over the fragment count rather than the texel count.
    """
    if min_stroke <= 0:
        return seed
    gap = max(int(gap), 0)
    grown = (
        ndimage.binary_dilation(seed, structure=NEIGHBOURHOOD, iterations=gap)
        if gap else seed
    )
    labels, count = ndimage.label(grown, structure=NEIGHBOURHOOD)
    if count == 0:
        return seed
    # The unbroken pieces of the *seed*, before any dilation -- the dilation is
    # what joined them, so measuring strokes on it would measure the gap rather
    # than the mark.
    drawn = seed if drawn is None else (drawn & seed)
    pieces, many = ndimage.label(drawn, structure=NEIGHBOURHOOD)
    piece_texels = np.bincount(pieces.ravel(), minlength=many + 1)
    # Which group each piece landed in.  Every texel of a piece is in the same
    # group, so one pass of plain assignment settles it.
    owner = np.zeros(many + 1, dtype=np.int64)
    owner[pieces[drawn]] = labels[drawn]
    longest = np.zeros(count + 1, dtype=np.int64)
    np.maximum.at(longest, owner[1:], piece_texels[1:])
    keep = longest >= int(min_stroke)
    keep[0] = False
    return keep[labels] & seed


def regions_from_holes(
    blank: np.ndarray, sample: np.ndarray, label: int, offset: tuple[int, int],
) -> list[Region]:
    """Every hole in the blank field, as a region, above the size floor.

    2026-08-28: the inverse of `sparse_field`, and the whole of the region
    stage.  Where the old front end thresholded, screened, transformed,
    bridged and then grouped, this reads one thing: the ground the threshold
    left blank, and the shapes punched out of it.

    A hole arrives whole and needs no grouping.  A window sitting in the gap
    between two letters still sees the strokes either side, so the gap never
    goes blank and the word is one hole -- which is the bridging problem
    answered before it is asked.

    The path is drawn from the filled hole, so an annulus -- a bezel, a ring --
    keeps the blank ground it encloses and gets its inner rim drawn, exactly as
    `mark_outline` does for a mark's own voids.  The region's mask stays the
    hole itself.
    """
    holes = sample & ~blank
    if not holes.any():
        return []
    labels, count = ndimage.label(holes, structure=NEIGHBOURHOOD)
    if count == 0:
        return []
    x_offset, y_offset = offset
    regions: list[Region] = []
    for index, slices in enumerate(ndimage.find_objects(labels), start=1):
        if slices is None:
            continue
        rows, columns = slices
        # `find_objects` is already tight round a labelled component, so the
        # box is the hole's own -- unlike the marks stage, where it was the
        # bounds of a dilation and had to be pulled back in.
        width = int(columns.stop - columns.start)
        height = int(rows.stop - rows.start)
        piece = labels[rows, columns] == index
        texels = int(piece.sum())
        regions.append(
            Region(
                bounds=(int(columns.start) + x_offset,
                        int(rows.start) + y_offset, width, height),
                mask=piece, domain=label, _cache={"texels": texels},
            )
        )
    return regions


def past_the_size_floor(
    regions: list[Region], min_width: int, min_height: int, min_area: int
) -> list[Region]:
    """The holes worth keeping.  See `ContrastConfig.min_width_px`.

    Width and height are the hole's bounding box; the area is the hole itself
    and not its box, so a diagonal stroke is not credited with the emptiness
    either side of it.
    """
    return [
        region for region in regions
        if region.bounds[2] >= int(min_width)
        and region.bounds[3] >= int(min_height)
        and region.texels >= int(min_area)
    ]


def paint_holes(
    regions: list[Region], shape: tuple[int, int], offset: tuple[int, int],
    config: ContrastConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """The paths round a set of holes, and the texels they cover.

    Kept apart from the labelling so that a stage which only *refuses* regions
    can redraw without going back to the field -- a region already owns its
    mask and its bounds, which is all a path needs.
    """
    x_offset, y_offset = offset
    outline = np.zeros(shape, dtype=bool)
    kept = np.zeros(shape, dtype=bool)
    for region in regions:
        x, y, width, height = region.bounds
        rows = slice(y - y_offset, y - y_offset + height)
        columns = slice(x - x_offset, x - x_offset + width)
        piece = region.mask
        kept[rows, columns] |= piece
        outline[rows, columns] |= mark_outline(
            piece, ndimage.binary_fill_holes(piece),
            config.min_void_texels, config.outline_px,
        )
    return outline, kept


@dataclass(slots=True)
class RimCandidate:
    """One shallow region, with the evidence used by the rim classifier."""

    region: Region
    margin_px: int
    minor_span_px: int
    major_span_px: int
    fill: float
    perimeter_texels: int
    rim_touch_texels: int
    length_contact: float
    perimeter_contact: float
    anchor_x: float | None
    anchor_y: float | None
    width_eligible: bool
    contact_eligible: bool
    perimeter_eligible: bool
    large_low_fill_eligible: bool
    neighbour_count: int = 0
    neighbour_span_px: float = 0.0
    neighbour_aspect: float = 0.0
    discarded: bool = False


def rim_artifact_candidates(
    regions: list[Region], sample: np.ndarray, offset: tuple[int, int],
    floor: int, ceiling: int, share: float, contact: float = 0.0,
    max_band_span: int = 0, min_perimeter_contact: float = 0.30,
    large_max_fill: float = 0.15,
) -> list[RimCandidate]:
    """Measure every shallow region that could be a rim artefact.

    2026-08-29: this is intentionally separate from the decision.  The probe
    can show the same evidence over every candidate, including candidates the
    current decision keeps, without duplicating or instrumenting the detector.
    """
    if (floor <= 0 and share <= 0) or not regions:
        return []
    margin = boundary_margin(island_size(sample), floor, ceiling, share)
    if margin <= 0:
        return []
    interior = ndimage.binary_erosion(
        sample, structure=NEIGHBOURHOOD, iterations=margin, border_value=0
    )
    rim = sample & ~ndimage.binary_erosion(
        sample, structure=NEIGHBOURHOOD, border_value=0
    )
    max_span = max(int(max_band_span), 0)
    long_run = max(max_span * 3, 48)
    x_offset, y_offset = offset
    candidates: list[RimCandidate] = []
    for region in regions:
        x, y, width, height = region.bounds
        rows = slice(y - y_offset, y - y_offset + height)
        columns = slice(x - x_offset, x - x_offset + width)
        if (interior[rows, columns] & region.mask).any():
            continue
        row0 = max(rows.start - 1, 0)
        row1 = min(rows.stop + 1, rim.shape[0])
        column0 = max(columns.start - 1, 0)
        column1 = min(columns.stop + 1, rim.shape[1])
        contact_mask = np.zeros((row1 - row0, column1 - column0), dtype=bool)
        contact_mask[
            rows.start - row0:rows.stop - row0,
            columns.start - column0:columns.stop - column0,
        ] = region.mask
        rim_piece = rim[row0:row1, column0:column1]
        touches = ndimage.binary_dilation(
            contact_mask, structure=NEIGHBOURHOOD
        ) & rim_piece
        perimeter = contact_mask & ~ndimage.binary_erosion(
            contact_mask, structure=NEIGHBOURHOOD, border_value=0
        )
        rows_on_rim, columns_on_rim = np.nonzero(touches)
        minor = min(width, height)
        major = max(width, height)
        length = rim_contact(contact_mask, rim_piece, major)
        perimeter_share = rim_perimeter_contact(contact_mask, rim_piece)
        score = length if major >= long_run else perimeter_share
        width_eligible = max_span == 0 or minor <= max_span
        perimeter_eligible = (
            float(min_perimeter_contact) > 0.0
            and perimeter_share >= float(min_perimeter_contact)
        )
        candidates.append(RimCandidate(
            region=region, margin_px=margin, minor_span_px=minor,
            major_span_px=major, fill=region.fill,
            perimeter_texels=int(perimeter.sum()),
            rim_touch_texels=int(touches.sum()), length_contact=length,
            perimeter_contact=perimeter_share,
            anchor_x=(float(columns_on_rim.mean() + column0 + x_offset)
                      if rows_on_rim.size else None),
            anchor_y=(float(rows_on_rim.mean() + row0 + y_offset)
                      if rows_on_rim.size else None),
            width_eligible=width_eligible,
            contact_eligible=(contact <= 0 or score >= float(contact)),
            perimeter_eligible=perimeter_eligible,
            large_low_fill_eligible=(
                not width_eligible and float(large_max_fill) > 0.0
                and region.fill <= float(large_max_fill)
                and perimeter_eligible
            ),
        ))
    return candidates


def classify_rim_candidates(
    candidates: list[RimCandidate], contact: float, max_band_span: int,
    min_perimeter_contact: float = 0.30, large_max_fill: float = 0.15,
) -> None:
    """Set each candidate's decision and local stitch-train measurements."""
    max_span = max(int(max_band_span), 0)
    if not candidates:
        return
    for candidate in candidates:
        candidate.discarded = candidate.width_eligible and contact <= 0
    if contact <= 0:
        return
    eligible = [
        candidate for candidate in candidates
        if (candidate.width_eligible or candidate.large_low_fill_eligible)
        and candidate.contact_eligible
        and candidate.anchor_x is not None and candidate.anchor_y is not None
    ]
    if not eligible:
        return
    anchors = np.array([
        (candidate.anchor_x, candidate.anchor_y) for candidate in eligible
    ], dtype=np.float64)
    long_run = max(max_span * 3, 48)
    stitch_radius = max(max_span * 3, 48)
    for index, candidate in enumerate(eligible):
        if candidate.major_span_px >= long_run:
            candidate.discarded = True
            continue
        if candidate.perimeter_eligible:
            candidate.discarded = True
            continue
        neighbours = np.flatnonzero(
            np.square(anchors - anchors[index]).sum(axis=1) <= stitch_radius ** 2
        )
        candidate.neighbour_count = int(neighbours.size)
        if neighbours.size < 3:
            continue
        centred = anchors[neighbours] - anchors[neighbours].mean(axis=0)
        _left, values, right = np.linalg.svd(centred, full_matrices=False)
        if values.size < 2 or values[1] <= 1e-6:
            candidate.neighbour_span_px = float(values[0]) if values.size else 0.0
            candidate.neighbour_aspect = float("inf")
        else:
            candidate.neighbour_span_px = float(np.ptp(centred @ right[0]))
            candidate.neighbour_aspect = float(values[0] / values[1])
        candidate.discarded = (
            candidate.neighbour_span_px >= max_span
            and candidate.neighbour_aspect >= 2.5
        )


def clear_of_the_rim(
    regions: list[Region], sample: np.ndarray, offset: tuple[int, int],
    floor: int, ceiling: int, share: float, contact: float = 0.0,
    max_band_span: int = 0, min_perimeter_contact: float = 0.30,
    large_max_fill: float = 0.15,
    measurements: list[RimCandidate] | None = None,
) -> list[Region]:
    """Drop edge bands and stitch trains, preserving compact rim-adjacent marks.

    2026-08-29: the sampling boundary, rather than the raw UV domain, is the
    edge the detector answers at.  Measuring the raw one-pixel rim misses a
    response that begins just inside `boundary_skip_px`.
    """
    measured = measurements if measurements is not None else rim_artifact_candidates(
        regions, sample, offset, floor, ceiling, share, contact, max_band_span,
        min_perimeter_contact, large_max_fill,
    )
    classify_rim_candidates(
        measured, contact, max_band_span, min_perimeter_contact, large_max_fill,
    )
    discarded = {id(candidate.region) for candidate in measured if candidate.discarded}
    return [region for region in regions if id(region) not in discarded]


def rim_candidate_features(candidate: RimCandidate) -> dict[str, bool | int | float | None]:
    """The scalar facts a generic late-stage probe records for the rim rule."""
    x, y, width, height = candidate.region.bounds
    return {
        "x": x, "y": y, "width_px": width, "height_px": height,
        "texels": candidate.region.texels, "margin_px": candidate.margin_px,
        "minor_span_px": candidate.minor_span_px,
        "major_span_px": candidate.major_span_px, "fill": candidate.fill,
        "perimeter_texels": candidate.perimeter_texels,
        "rim_touch_texels": candidate.rim_touch_texels,
        "length_contact": candidate.length_contact,
        "perimeter_contact": candidate.perimeter_contact,
        "anchor_x": candidate.anchor_x, "anchor_y": candidate.anchor_y,
        "width_eligible": candidate.width_eligible,
        "contact_eligible": candidate.contact_eligible,
        "perimeter_eligible": candidate.perimeter_eligible,
        "large_low_fill_eligible": candidate.large_low_fill_eligible,
        "neighbour_count": candidate.neighbour_count,
        "neighbour_span_px": candidate.neighbour_span_px,
        "neighbour_aspect": candidate.neighbour_aspect,
    }


@dataclass(slots=True)
class RepeatTextureIndex:
    """One atlas-wide detail field and the recurrence answers already read."""

    detail: np.ndarray
    decimation: int
    matches: dict[tuple[int, int, int, int, int, int, int, int], int] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class RepeatSearch:
    """The search-side work shared by every probe around one region."""

    local_norm: np.ndarray
    spectrum: np.ndarray
    response_shape: tuple[int, int]
    transform_shape: tuple[int, int]


def build_repeat_texture_index(
    colour: np.ndarray, config: ContrastConfig,
) -> RepeatTextureIndex:
    """Build the decimated, high-passed atlas shared by stage 5."""
    step = max(int(config.repeat_texture_decimation), 1)
    values = colour[:, :, :3].astype(np.float32) @ np.array(
        [0.299, 0.587, 0.114], dtype=np.float32
    )
    if step > 1:
        height = values.shape[0] // step * step
        width = values.shape[1] // step * step
        values = values[:height, :width].reshape(
            height // step, step, width // step, step
        ).mean(axis=(1, 3))
    radius = max(int(config.repeat_texture_highpass_px) // (2 * step), 1)
    size = radius * 2 + 1
    baseline = ndimage.uniform_filter(
        ndimage.uniform_filter(values, size=size, mode="nearest"),
        size=size, mode="nearest",
    )
    return RepeatTextureIndex(
        detail=(values - baseline).astype(np.float32), decimation=step
    )


def _repeat_window_totals(values: np.ndarray, height: int, width: int) -> np.ndarray:
    integral = np.pad(values, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return (
        integral[height:, width:]
        - integral[:-height, width:]
        - integral[height:, :-width]
        + integral[:-height, :-width]
    )


def _prepare_repeat_search(search: np.ndarray, template_shape: tuple[int, int]) -> RepeatSearch | None:
    """Do the invariant half of normalised correlation once per search window."""
    tile_height, tile_width = template_shape
    height, width = search.shape
    if height < tile_height or width < tile_width:
        return None
    search = search.astype(np.float64)
    area = tile_height * tile_width
    totals = _repeat_window_totals(search, tile_height, tile_width)
    squares = _repeat_window_totals(np.square(search), tile_height, tile_width)
    variance = np.maximum(squares - np.square(totals) / area, 0.0)
    transform_shape = (height + tile_height - 1, width + tile_width - 1)
    return RepeatSearch(
        local_norm=np.sqrt(variance),
        spectrum=np.fft.rfft2(search, transform_shape),
        response_shape=(height - tile_height + 1, width - tile_width + 1),
        transform_shape=transform_shape,
    )


def _normalised_repeat_match(
    prepared: RepeatSearch | None, template: np.ndarray, min_energy: float = 0.5,
) -> np.ndarray:
    """Normalised correlation after the search-side work is already prepared."""
    if prepared is None:
        return np.empty((0, 0), dtype=np.float32)
    template = template.astype(np.float64)
    template = template - template.mean()
    template_norm = float(np.sqrt(np.square(template).sum()))
    if template_norm < 1e-8:
        return np.zeros(prepared.response_shape)
    correlation = np.fft.irfft2(
        prepared.spectrum * np.conj(np.fft.rfft2(template, prepared.transform_shape)),
        prepared.transform_shape,
    )[:prepared.response_shape[0], :prepared.response_shape[1]]
    denominator = prepared.local_norm * template_norm
    with np.errstate(divide="ignore", invalid="ignore"):
        response = np.where(
            (prepared.local_norm >= template_norm * min_energy) & (denominator > 1e-8),
            correlation / denominator,
            0.0,
        )
    return np.clip(response, -1.0, 1.0).astype(np.float32)


def normalised_repeat_match(
    search: np.ndarray, template: np.ndarray, min_energy: float = 0.5,
) -> np.ndarray:
    """Normalised cross-correlation, excluding locally flat matches."""
    return _normalised_repeat_match(
        _prepare_repeat_search(search, template.shape), template, min_energy
    )


def count_repeat_matches(
    response: np.ndarray, threshold: float, separation: int, limit: int,
) -> int:
    """Count distinct correlation peaks, stopping when the verdict is known."""
    if response.size == 0:
        return 0
    work = response.copy()
    found = 0
    for _ in range(max(limit, 1)):
        flat = int(np.argmax(work))
        row, column = divmod(flat, work.shape[1])
        if float(work[row, column]) < threshold:
            break
        found += 1
        work[
            max(row - separation, 0):row + separation + 1,
            max(column - separation, 0):column + separation + 1,
        ] = -1.0
    return found


def _repeat_probe(
    index: RepeatTextureIndex, centre: tuple[int, int],
    bounds: tuple[int, int, int, int], required: int,
    prepared: RepeatSearch | None, config: ContrastConfig,
) -> int:
    """Count only enough separate matches to settle this probe's verdict."""
    required = max(int(required), 0)
    step = index.decimation
    tile = max(int(config.repeat_texture_tile_px) // step, 4)
    half = tile // 2
    centre_x, centre_y = centre[0] // step, centre[1] // step
    x0, y0, x1, y1 = (value // step for value in bounds)
    key = (centre_x, centre_y, x0, y0, x1 - x0, y1 - y0, tile, required)
    if key in index.matches:
        return index.matches[key]
    top, left = centre_y - half, centre_x - half
    detail = index.detail
    if top < 0 or left < 0 or top + tile > detail.shape[0] or left + tile > detail.shape[1]:
        index.matches[key] = 0
        return 0
    template = detail[top:top + tile, left:left + tile]
    if float(template.std()) < config.repeat_texture_min_tile_contrast:
        index.matches[key] = 0
        return 0
    if required == 0:
        index.matches[key] = 0
        return 0
    found = count_repeat_matches(
        _normalised_repeat_match(prepared, template),
        config.repeat_texture_match_threshold,
        max(half, 2),
        required,
    )
    index.matches[key] = found
    return found


def repeat_texture_evidence(
    index: RepeatTextureIndex, bounds: tuple[int, int, int, int],
    config: ContrastConfig,
) -> tuple[int, int]:
    """Return recurrence evidence, capped once either verdict is known."""
    x, y, width, height = bounds
    centre = (x + width // 2, y + height // 2)
    atlas_height, atlas_width = (size * index.decimation for size in index.detail.shape)
    span = max(
        round(max(width, height) * max(config.repeat_texture_search_scale, 1)),
        max(config.repeat_texture_min_search_px, 16),
    )
    span = min(span, max(config.repeat_texture_max_search_px, 16))
    half = span // 2
    search = (
        max(centre[0] - half, 0), max(centre[1] - half, 0),
        min(centre[0] + half, atlas_width), min(centre[1] + half, atlas_height),
    )
    step = index.decimation
    tile = max(int(config.repeat_texture_tile_px) // step, 4)
    prepared = _prepare_repeat_search(
        index.detail[search[1] // step:search[3] // step,
                     search[0] // step:search[2] // step],
        (tile, tile),
    )
    own = _repeat_probe(
        index, centre, search, config.repeat_texture_min_repeats, prepared, config
    )
    if own >= config.repeat_texture_min_repeats:
        return own, 0
    needed = int(config.repeat_texture_context_directions)
    if needed <= 0:
        return own, 0
    margin = max(config.repeat_texture_context_margin_px, 1)
    reach_x = width // 2 + config.repeat_texture_tile_px // 2 + margin
    reach_y = height // 2 + config.repeat_texture_tile_px // 2 + margin
    directions = 0
    # 2026-08-29: Two matching directions already decide the default verdict;
    # running the opposite two still builds full correlation maps but cannot
    # change it.  Keep the configured count rather than baking in that two.
    for dx, dy in ((reach_x, 0), (-reach_x, 0), (0, reach_y), (0, -reach_y)):
        if _repeat_probe(
            index, (centre[0] + dx, centre[1] + dy), search,
            config.repeat_texture_context_repeats, prepared, config,
        ) >= config.repeat_texture_context_repeats:
            directions += 1
            if directions >= needed:
                break
    return own, directions


def clear_of_a_repeat(
    regions: list[Region], index: RepeatTextureIndex | None, config: ContrastConfig,
) -> list[Region]:
    """Drop a region only when its image tile or surrounding material recurs."""
    if not config.enable_repeat_texture_filter or index is None or not regions:
        return regions
    fallback_tile = int(config.repeat_texture_fallback_tile_px)
    fallback = (
        replace(config, repeat_texture_tile_px=fallback_tile)
        if fallback_tile > 0 and fallback_tile != config.repeat_texture_tile_px
        else None
    )
    kept: list[Region] = []
    for region in regions:
        if (
            min(region.width, region.height)
            < config.repeat_texture_min_region_span_px
            or region.texels < config.repeat_texture_min_region_area_px
        ):
            kept.append(region)
            continue
        own, directions = repeat_texture_evidence(index, region.bounds, config)
        if (
            own < config.repeat_texture_min_repeats
            and directions < config.repeat_texture_context_directions
            and (fallback is None or not confirmed_small_repeat(region, index, fallback))
        ):
            kept.append(region)
    return kept


def confirmed_small_repeat(
    region: Region, index: RepeatTextureIndex, config: ContrastConfig,
) -> bool:
    """Confirm a compact periodic mask using a smaller image tile."""
    if region.fill > config.repeat_texture_fallback_max_fill:
        return False
    energy, overlap, dip = repeat_scores(
        region.mask, config.repeat_skip_px, config.repeat_longest_px
    )
    if (
        energy < config.repeat_texture_fallback_min_energy
        or overlap < config.repeat_texture_fallback_min_overlap
        or dip < config.repeat_texture_fallback_min_dip
    ):
        return False
    own, directions = repeat_texture_evidence(index, region.bounds, config)
    return (
        own >= config.repeat_texture_min_repeats
        or directions >= config.repeat_texture_context_directions
    )


def region_centres(regions: list[Region]) -> np.ndarray:
    """Each region's middle, as (x, y) floats.  Empty comes back (0, 2)."""
    if not regions:
        return np.zeros((0, 2), dtype=np.float64)
    return np.array(
        [(x + width / 2.0, y + height / 2.0)
         for x, y, width, height in (region.bounds for region in regions)],
        dtype=np.float64,
    )


def clear_of_a_crowd(
    regions: list[Region], company: list[Region], radius: int, limit: int,
    large_area: int = 0,
) -> list[Region]:
    """Drop the regions that are one of many in a small neighbourhood.

    ``company`` is what counts as being nearby, which is every hole the field
    left rather than only the ones that cleared the size floor -- see
    `ContrastConfig.crowd_radius_px`.

    2026-08-29: a k-d tree over the centres, not a disc filter over the domain.
    The filter was a 241-wide convolution run once per UV domain to count a few
    hundred points, and on a texture with 426 domains that came to 19.3 s of a
    35.4 s corpus run -- 55% of the pipeline, for the cheapest question in it.
    The tree is milliseconds and gives the same answer.

    Distances are in atlas coordinates, which the bounds already carry, so
    nothing has to be mapped back into the domain first.
    """
    if radius <= 0 or limit < 0 or not regions or not company:
        return regions
    # 2026-08-29: Do not even query a large candidate.  It is retained by
    # policy and excluding it from the query set avoids making the birdshot
    # exception cost more than the rule it protects against.
    floor = max(int(large_area), 0)
    judged = [
        index for index, region in enumerate(regions)
        if floor == 0 or region.texels < floor
    ]
    if not judged:
        return regions
    tree = cKDTree(region_centres(company))
    near = tree.query_ball_point(
        region_centres([regions[index] for index in judged]),
        float(radius), return_length=True,
    )
    # A region counts itself when it is part of its own company.
    same = regions is company
    refused = {
        index for index, count in zip(judged, near)
        if int(count) - (1 if same else 0) > int(limit)
    }
    return [
        region for index, region in enumerate(regions) if index not in refused
    ]


def regions_from_marks(
    kept: np.ndarray, gap: int, label: int, offset: tuple[int, int],
    config: ContrastConfig,
) -> tuple[list[Region], np.ndarray]:
    """One region per grouped mark, with the path drawn round it.

    2026-08-27: the marks stage is the end of the pipeline while the front end
    is being rebuilt, so its groups are the answer and have to arrive as
    regions rather than as a mask -- a green outline round a constellation of
    texels is the thing being judged, and a count of them is how a change gets
    read at a glance.

    Only called when the run stops here.  When the whole pipeline runs, the
    region stage downstream makes its own and these would be thrown away.

    The mark is relabelled rather than threaded through from
    `marks_from_seed`: it is a far smaller mask by this point, and keeping the
    two functions independent is worth one labelling.
    """
    outline = np.zeros(kept.shape, dtype=bool)
    if not kept.any():
        return [], outline
    gap = max(int(gap), 0)
    grown = (
        ndimage.binary_dilation(kept, structure=NEIGHBOURHOOD, iterations=gap)
        if gap else kept
    )
    labels, count = ndimage.label(grown, structure=NEIGHBOURHOOD)
    if count == 0:
        return [], outline
    x_offset, y_offset = offset
    regions: list[Region] = []
    for index, slices in enumerate(ndimage.find_objects(labels), start=1):
        if slices is None:
            continue
        rows, columns = slices
        shape = (labels[rows, columns] == index) & kept[rows, columns]
        if not shape.any():
            continue
        # Tighten to the mark: `find_objects` gives the bounds of the dilation,
        # which is `gap` wider on every side than anything that was detected.
        mark_rows = np.flatnonzero(shape.any(axis=1))
        mark_columns = np.flatnonzero(shape.any(axis=0))
        tight = np.s_[
            mark_rows[0]:mark_rows[-1] + 1,
            mark_columns[0]:mark_columns[-1] + 1,
        ]
        piece = shape[tight]
        outline[rows, columns][tight] |= mark_outline(
            piece, cluster_body(piece, gap),
            config.min_void_texels, config.outline_px,
        )
        regions.append(
            Region(
                bounds=(
                    int(columns.start) + int(mark_columns[0]) + x_offset,
                    int(rows.start) + int(mark_rows[0]) + y_offset,
                    int(mark_columns[-1] - mark_columns[0]) + 1,
                    int(mark_rows[-1] - mark_rows[0]) + 1,
                ),
                mask=piece, domain=label,
                _cache={"texels": int(piece.sum())},
            )
        )
    return regions, outline


def paint_mark_regions(
    regions: list[Region], shape: tuple[int, int], offset: tuple[int, int],
    gap: int, config: ContrastConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Redraw final mark regions after a region-level decision removed some."""
    x_offset, y_offset = offset
    outline = np.zeros(shape, dtype=bool)
    kept = np.zeros(shape, dtype=bool)
    for region in regions:
        x, y, width, height = region.bounds
        rows = slice(y - y_offset, y - y_offset + height)
        columns = slice(x - x_offset, x - x_offset + width)
        kept[rows, columns] |= region.mask
        outline[rows, columns] |= mark_outline(
            region.mask, cluster_body(region.mask, gap),
            config.min_void_texels, config.outline_px,
        )
    return outline, kept


def clear_of_boundary(
    lit: np.ndarray, sample: np.ndarray, reach: int
) -> np.ndarray:
    """Drop responses that touch the island's edge and never leave it.

    Per connected component, and whole: the question is what a run of texels
    *is*, and half a boundary ribbon is still a boundary ribbon.  A component
    that never touches the edge is not examined at all.

    Both questions are asked the same way -- does the component meet a ring of
    the island -- so each is one erosion, one `bincount` and no per-component
    work at all.

    Depth is Chebyshev, being a count of 3x3 erosions, rather than the
    Euclidean distance transform this started as.  Chebyshev distance never
    exceeds Euclidean, so the erosion is the stricter of the two and can only
    remove more.  Measured over the corpus at reach 4 that is 4, 93, 0 and 26
    texels of a ~16,000-texel removal -- against 2.2 s of a 33 s run, which is
    what the transform cost.
    """
    if reach <= 0 or not lit.any():
        return lit
    labels, count = ndimage.label(lit, structure=NEIGHBOURHOOD)
    if count == 0:
        return lit
    # The outermost ring the detector answers for, and the part of the island
    # that lies `reach` texels in.  A component is refused when it meets the
    # first and not the second.  `binary_erosion` counts from 1 -- the surviving
    # texels after k passes are those at depth k + 1 -- so `reach` texels in is
    # `reach - 1` passes.
    rim = sample & ~ndimage.binary_erosion(
        sample, structure=NEIGHBOURHOOD, border_value=0
    )
    deep = sample if reach <= 1 else ndimage.binary_erosion(
        sample, structure=NEIGHBOURHOOD, iterations=int(reach) - 1,
        border_value=0,
    )
    touching = np.bincount(labels[rim], minlength=count + 1) > 0
    reaching = np.bincount(labels[deep], minlength=count + 1) > 0
    keep = np.ones(count + 1, dtype=bool)
    keep[1:] = ~touching[1:] | reaching[1:]
    return keep[labels] & lit


def masked_local_mean(
    values: np.ndarray, domain: np.ndarray, kernel_px: int
) -> np.ndarray:
    """Mean of each pixel's neighbourhood, counting only in-domain texels.

    Dividing filtered values by filtered domain weights is the whole point: a
    kernel that borrowed the zeros outside the domain would read every texel
    near a seam as high contrast, and every seam would come back a glyph.  It
    does not make the estimate at the edge *good* -- the sample there is
    one-sided -- only unbiased by the emptiness outside.
    """
    weights = domain.astype(np.float32)
    return weighted_local_mean(values * weights[:, :, None], weights, kernel_px)


def weighted_local_mean(
    weighted: np.ndarray, weights: np.ndarray, kernel_px: int
) -> np.ndarray:
    """The same, for a caller that has already multiplied by the weights.

    Two means of the same values at two kernel sizes is what
    `contrast_response` wants, and the weighting is identical for both -- so it
    is done once and handed in.  Measured, that multiply over a three-channel
    4k crop was 0.7 s of the corpus run on its own.
    """
    kernel = max(int(kernel_px), 1)
    kernel += 1 - kernel % 2  # centred, so the response has no directional bias
    # 2026-08-28: a kernel of 1 is the texel itself, and the arithmetic proves
    # it exactly -- the filters are no-ops, so the result is values*w/max(w,eps),
    # which is `values` where w is 1 and 0 where it is 0.  Worth catching
    # because `blur_px` 1 is the default and the general path spent 1.6 s of
    # the corpus run computing an identity.
    if kernel == 1:
        return weighted
    # One call over all three channels rather than one per channel: scipy
    # filters only the axes whose size is above 1, in axis order, so a size of
    # (k, k, 1) is the same sequence of 1-D passes the per-channel version ran
    # -- checked bit for bit over the corpus.
    inside = ndimage.uniform_filter(weights, size=kernel, mode="constant", cval=0.0)
    totals = ndimage.uniform_filter(
        weighted, size=(kernel, kernel, 1), mode="constant", cval=0.0,
    )
    return totals / np.maximum(inside, 1e-6)[:, :, None]


def contrast_response(
    colour: np.ndarray, domain: np.ndarray, config: ContrastConfig,
    sample: np.ndarray | None = None,
) -> np.ndarray:
    """How far each texel sits from the colour of its own neighbourhood.

    A band-pass: the difference between a narrow mean and a wide one.  At
    ``blur_px`` 1 the narrow mean is the texel itself and this is the plain
    high-pass it began as.
    """
    values = colour.astype(np.float32)
    # Derived here when nobody hands one over, so a caller holding only a
    # domain still gets the right answer.  `regions_in_domain` passes its own,
    # because it needs the same mask six times over: measured, the erosion is
    # 0.83 s across the corpus's 831 domains against a 29 s run, so deriving it
    # per call rather than per domain is the difference between 3% and more.
    if sample is None:
        sample = sampling_mask(domain, config.boundary_skip_px)
    weights = sample.astype(np.float32)
    weighted = values * weights[:, :, None]
    near = weighted_local_mean(weighted, weights, config.blur_px)
    far = weighted_local_mean(weighted, weights, config.kernel_px)
    response = np.linalg.norm(near - far, axis=2)
    # 2026-08-28: multiply rather than `np.where(sample, response, 0.0)`.  The
    # response is a norm, so it is finite and non-negative everywhere, and
    # r * 1.0 and r * 0.0 are exactly r and 0.0 -- verified against the whole
    # corpus, stage by stage.  The `where` allocated a second full array and
    # then `astype` copied it again.
    return np.multiply(response, weights, out=response)


def response_threshold(
    response: np.ndarray, domain: np.ndarray, config: ContrastConfig
) -> float:
    if not domain.any():
        return float("inf")
    percentile = min(max(float(config.percentile), 0.0), 100.0)
    return max(
        float(config.min_response),
        float(np.percentile(response[domain], percentile)),
    )


# Rec. 601 luma, as float64: the cast to float32 happens after the sum, so the
# weights stay double and the arithmetic matches what three scaled channels did.
GREY_WEIGHTS = np.array([0.299, 0.587, 0.114])


def structure_map(
    colour: np.ndarray, domain: np.ndarray, outer: int, inner: int
) -> np.ndarray:
    """How much of a window's variation is a step rather than fluctuation.

    The wide window's deviation over the mean of the narrow windows inside it.
    A step makes the wide one vary while each side stays flat, so the ratio
    rises; noise moves both together and it sits near 1.  See
    `ContrastConfig.min_structure` for the crops this came from.

    Masked by the domain for the same reason everything else here is: a window
    that borrowed the emptiness outside would read every seam as a step.

    2026-08-28: the *mean* of the narrow deviations, not the smallest of them
    -- tried, worse, do not repeat.  The smallest is the more faithful reading
    of the premise, since a stroke thinner than the narrow window has flat
    ground either side but no narrow window that sits inside the stroke, and it
    shows on exactly those: with the bar set so ardente's confetti still passes
    3%, the play triangle goes from 22% of its texels to 90% and a row of
    legend text from 50% to 94%.  It is also an order statistic over the 49
    windows in reach, and weave finds a flat one by chance -- scintilla's
    interior went from 87,913 seed texels to 143,628, the extra arriving as
    34,185 blobs of under eight texels, and etk800 came out of the density
    stage *lower* than before at 56,327 against 66,225.  Half the mean and half
    the smallest splits the difference in both directions and settles neither.
    """
    # 2026-08-28: one matrix product rather than three scaled channels summed.
    # An inner dimension of three is a plain dot, so it accumulates in the same
    # order and gives the same float64 before the cast -- checked against the
    # corpus, and against random uint8 crops.  About 16% quicker, and it makes
    # one temporary instead of five.
    grey = (colour @ GREY_WEIGHTS).astype(np.float32)
    squared = grey * grey
    weight = domain.astype(np.float32)

    # 2026-08-28: the domain's own filter, kept per size.
    #
    # It does not depend on what is being averaged, and five of the seven means
    # below want one of two sizes -- so the first version filtered the weights
    # three times at `outer` and twice at `inner` to get the same two arrays.
    # Measured over the corpus that was 3 of the 10 filter passes this function
    # ran, and `structure_map` was 45% of the two stages' time.
    shares: dict[int, np.ndarray] = {}

    def share(size: int) -> np.ndarray:
        if size not in shares:
            shares[size] = np.maximum(
                ndimage.uniform_filter(weight, size=size, mode="constant"), 1e-6
            )
        return shares[size]

    def mean(weighted: np.ndarray, size: int) -> np.ndarray:
        """``weighted`` has already been multiplied by the domain weights.

        Both windows want the mean of the same two arrays, so weighting them
        per call multiplied each one twice over.
        """
        return ndimage.uniform_filter(
            weighted, size=size, mode="constant"
        ) / share(size)

    grey_weighted = grey * weight
    squared_weighted = squared * weight

    def deviation(size: int) -> np.ndarray:
        m = mean(grey_weighted, size)
        return np.sqrt(np.maximum(mean(squared_weighted, size) - m * m, 0.0))

    wide = deviation(max(int(outer), 1))
    fine = mean(
        deviation(max(int(inner), 1)) * weight, max(int(outer), 1)
    )
    return wide / np.maximum(fine, 1e-3)


def structure_seed(
    colour: np.ndarray, lit: np.ndarray, domain: np.ndarray,
    config: ContrastConfig,
) -> np.ndarray:
    """Whole pieces of ``lit`` that hold a step somewhere, not just the steps.

    `structure_map` reads a seven-texel window, so it answers for a texel and
    its immediate surroundings.  Asking it of every texel separately punches
    holes: a stroke's reading rises and falls along its own length, and where
    it dips under the bar the mark is cut rather than declined.  Grouping first
    and asking once per piece is the same question put to the object instead of
    the texel -- a mark is a mark along its whole edge once any part of it
    reads as one.

    2026-08-28: one passing texel is enough, and that is not a knob.  At three
    the corpus seed falls below what the per-texel gate kept (226,506 against
    226,962) while the landmarks get worse, and at eight it falls to 209,338 --
    a piece that reads as a step in several places is a piece the bar was never
    the question for.  What the reclaim cannot do is rescue a piece that reads
    under the bar everywhere: two limbs of that same play triangle peak at 1.36
    and 1.23 against a bar of 1.5, so they stay declined.  That is
    `min_structure`'s to answer, not this.

    The worry this deserved was a glyph joined to a grain field by one lucky
    pass.  Measured over the corpus, it does not happen: the largest gain on
    etk800 is 100 texels onto a 173-texel piece that already had 73 passing,
    and every large gain anywhere is a piece where most texels passed already.
    Grain arrives as pieces of one to three texels, which have nothing to grow
    into -- ardente's confetti panel goes from 19 texels to 518 out of 262,500.
    Growing across a *gap* is what breaks that: at one texel the same panel
    reaches 4,927 and at two the field is a single piece and the corpus more
    than doubles.
    """
    seeds = lit & (
        structure_map(
            colour, domain, config.structure_px, config.structure_inner_px
        ) >= config.min_structure
    )
    labels, count = ndimage.label(lit, structure=NEIGHBOURHOOD)
    if count == 0:
        return seeds
    holds = np.bincount(labels[seeds], minlength=count + 1) > 0
    holds[0] = False
    return holds[labels]


def seed_density(
    seed: np.ndarray, domain: np.ndarray, window_px: int
) -> np.ndarray:
    """What share of each texel's neighbourhood is seeded, in-domain only.

    A box filter over the seed divided by a box filter over the domain, the
    same masked-mean shape as everything else in this module and for the same
    reason: a window hanging off the edge of an island would otherwise read as
    empty, and every seed along a seam would be called lonely.

    Separable, so the window costs the same at 15 as at 3.
    """
    width = max(int(window_px), 1)
    width += 1 - width % 2  # centred, so the count has no directional bias
    inside = ndimage.uniform_filter(
        domain.astype(np.float32), size=width, mode="constant"
    )
    seeded = ndimage.uniform_filter(
        (seed & domain).astype(np.float32), size=width, mode="constant"
    )
    return seeded / np.maximum(inside, 1e-6)


def local_response_floor(
    response: np.ndarray, domain: np.ndarray, window_px: int
) -> np.ndarray:
    """The mean response over a wide neighbourhood, in-domain texels only.

    What a texel is being asked to stand out *from*.  Masked by the domain for
    the same reason the colour mean is: a window that borrowed the zeros
    outside would read every texel near a seam as remarkable.

    A box filter is separable, so the window costs the same at 31 as at 3.
    """
    width = max(int(window_px), 1)
    weight = domain.astype(np.float32)
    total = ndimage.uniform_filter(response * weight, size=width, mode="constant")
    share = ndimage.uniform_filter(weight, size=width, mode="constant")
    return total / np.maximum(share, 1e-6)


def cluster_body(detected: np.ndarray, gap: int) -> np.ndarray:
    """The silhouette a group of marks makes together.

    Closed across the grouping distance and then filled, so the path is drawn
    around the group -- a word rather than each of its strokes -- and follows
    its shape rather than boxing it.

    Padded first: the erosion half treats everything outside the array as
    background, so without the pad a mark against the edge of its crop would be
    chewed back by exactly what the dilation grew it.
    """
    return closed_body(detected, gap)


def mark_outline(
    shape: np.ndarray, filled: np.ndarray, min_void: int, thickness: int = 1
) -> np.ndarray:
    """One mark's path: its silhouette, plus the rim of any real void in it.

    A shape minus its own erosion is its outline, and that is the whole of the
    kernel work here.  Taken from the filled mark for the outside, so a glyph
    gets one clean silhouette rather than both sides of every stroke.

    An annulus -- a dial bezel, a button ring -- then also gets its inner
    circle, because a void that is itself big enough to have been a mark is big
    enough to be worth drawing.  The speckle of unfired texels inside a thick
    digit is not, and gets nothing, which is the whole reason for the size test
    rather than outlining every hole.
    """
    depth = max(int(thickness), 1)
    outline = filled & ~ndimage.binary_erosion(
        filled, structure=NEIGHBOURHOOD, iterations=depth, border_value=0
    )
    voids = filled & ~shape
    if voids.any():
        void_labels, count = ndimage.label(voids, structure=NEIGHBOURHOOD)
        sizes = np.bincount(void_labels.ravel(), minlength=count + 1)
        real = sizes >= min_void
        real[0] = False
        if real.any():
            outline |= shape & ndimage.binary_dilation(
                real[void_labels], structure=NEIGHBOURHOOD, iterations=depth
            )
    return outline


def group_outline(mask: np.ndarray, config: ContrastConfig) -> np.ndarray:
    """The paths the marks stage would draw around a stage's texels.

    2026-08-27: a stage viewer that only shows texels answers half the
    question.  What is being argued about is usually a *group* -- whether a run
    of stitching arrives as one thing or five, where a word stops being a word
    -- and that is invisible in a mask.

    2026-08-28: it is `regions_from_marks` now, so a stage's outlines are drawn
    by exactly the recipe the marks stage uses rather than by a second copy of
    it.  The copy read two knobs the old region stage owned, and went with
    them.

    There is no size floor any more, so an early stage draws a path round every
    speck.  That is the honest picture of what the stage holds, and it is only
    paid for when the viewer asks for a stage at all.
    """
    _regions, painted = regions_from_marks(
        mask, config.mark_gap_px, 0, (0, 0), config
    )
    return painted


def regions_in_domain(
    colour: np.ndarray,
    domain: np.ndarray,
    label: int,
    offset: tuple[int, int],
    config: LabConfig,
    sparse: np.ndarray,
    repeat_texture: RepeatTextureIndex | None = None,
    stages: dict[str, np.ndarray] | None = None,
    outlines: dict[str, np.ndarray] | None = None,
    counts: dict[str, int] | None = None,
    timings: dict[str, float] | None = None,
    probe: RegionProbe | None = None,
    texture_name: str = "",
) -> tuple[list[Region], np.ndarray, np.ndarray]:
    """One domain, start to finish: the marks in it, their paths, their texels.

    Every step is a filter over the whole domain and every one is optional --
    a knob at 0 makes its step a no-op rather than branching around it, so the
    stages below always happen and always mean the same thing.

    2026-08-28: this used to run on for another nine stages -- growing the seed
    back through the threshold, hysteresis, a denoise pass, the texel filters,
    a second grouping, and a region stage that judged each mark on its shape.
    All of it was the algorithm this rebuild replaces, and it had been switched
    off behind `stop_after_stage` for long enough to be sure.  It is in the
    history if any of it is wanted back.
    """
    reached = time.perf_counter()

    def note(
        name: str, mask: np.ndarray, painted: np.ndarray | None = None,
        found: list[Region] | None = None,
    ) -> None:
        """Record one stage, when anyone is watching.  Costs nothing when not.

        ``painted`` is the stage's own region paths where it has them, so the
        viewer draws what *that* stage decided rather than the finished
        article.  Without it the paths are derived from the mask, which is all
        a texel stage can offer.

        The clock is read before the capture and restarted after it, so what a
        stage is charged is the work that produced it and never the work of
        recording it -- drawing the paths alone is dearer than most of the
        stages it draws.
        """
        nonlocal reached
        now = time.perf_counter()
        if timings is not None:
            timings[name] = timings.get(name, 0.0) + (now - reached)
        if stages is not None:
            stages[name] = mask
            if painted is None and STAGES.index(name) >= STAGE_FIRST["outlines"]:
                painted = group_outline(mask, config.contrast)
            if painted is not None:
                outlines[name] = painted
            if found is not None:
                counts[name] = counts.get(name, 0) + len(found)
        reached = time.perf_counter()

    stop = int(config.contrast.stop_after_stage)
    if not 1 <= stop <= LAST_STOP:
        stop = 0

    def halted(name: str) -> bool:
        """Is ``name`` the stage the pipeline was told to end on?"""
        return stop > 0 and STAGES.index(name) + 1 >= stop

    def stopped(
        mask: np.ndarray, found: list[Region] | None = None,
        painted: np.ndarray | None = None,
    ) -> tuple[list[Region], np.ndarray, np.ndarray]:
        """End here: these texels are the answer.

        Regions only where the stage stopped on has grouped them -- the marks
        stage, and the candidates before it.  Stopping midway through the front
        end leaves a mask and says so, rather than inventing an answer out of a
        stage that has not decided anything yet.
        """
        blank = np.zeros(mask.shape, dtype=bool)
        return found or [], blank if painted is None else painted, mask

    if outlines is None:
        outlines = {}
    if counts is None:
        counts = {}
    contrast = config.contrast
    sample = sampling_mask(domain, contrast.boundary_skip_px)
    response = contrast_response(colour, domain, contrast, sample)
    # 2026-08-27: the same mask from here down, not `domain`.  The rim texels
    # now carry a response of 0, and a floor or a percentile that averaged them
    # in would read that as flat surface and drop its own bar -- letting noise
    # through in a band around every island, which is the opposite of what
    # taking the rim out was for.
    if contrast.local_floor_multiple > 0:
        level = np.maximum(
            contrast.local_floor_multiple * local_response_floor(
                response, sample, contrast.local_floor_px
            ),
            contrast.min_response,
        )
    else:
        level = response_threshold(response, sample, contrast)

    lit = (response >= level) & sample
    # 2026-08-28: kept for the bridging stage, which reads the voids in it.  It
    # has to be the threshold itself and not a later mask: what the gate needs
    # to know is whether the surface has any clear ground on it, and every
    # stage after this one has already cleared some.
    threshold = lit
    # What the bridging stage reads, kept from here because it is the only
    # place the threshold and the response are both still in hand.  A texel's
    # response as a fraction of the bar it was held to: 1 is the bar, and the
    # bridges are what lies under it.
    ratio = np.divide(
        response, level, out=np.zeros_like(response), where=np.asarray(level) > 0
    )
    # An empty list, not nothing: it records that this stage has a region
    # count and that the count is zero, which is what the viewer needs to
    # say "no regions yet" rather than falling back on the final tally.
    note(AT_THRESHOLD, lit, found=[])
    if halted(AT_THRESHOLD):
        return stopped(lit)

    # A marker, not a filter: `lit` goes on untouched.  What is recorded is the
    # blank ground the marks stand on, which is the one thing a mask of what
    # fired cannot show.
    blank = sparse_field(
        lit, sample, contrast.sparse_kernel_px, contrast.max_sparse_texels
    )
    sparse[:] = blank
    # The stage answers with the *inverse* of what it computes: the blank
    # ground is the overlay, kept on `Detection.sparse` and drawn under
    # everything, and the holes punched in it are the marks.
    # Every hole the field left, kept for the stages below: the crowd stage
    # counts company among *all* of them, not only the ones that clear the
    # size floor, because the floor has already thinned the very field it
    # would otherwise be judging.
    holes = regions_from_holes(blank, sample, label, offset)
    # Painting every hole is dear and only the viewer ever wants it, so it
    # happens when a stage is being captured or when the run stops here.
    unsorted = (
        paint_holes(holes, sample.shape, offset, contrast)
        if stages is not None or halted(SPARSE) else None
    )
    note(SPARSE, sample & ~blank,
         None if unsorted is None else unsorted[0], holes)
    if halted(SPARSE):
        # Every hole, unsorted, which is what stopping here is for.
        painted, kept = unsorted
        return holes, painted, kept

    found = clear_of_a_crowd(
        holes, holes, contrast.crowd_radius_px, contrast.max_neighbours,
        contrast.crowd_large_region_area_px,
    )
    painted, kept = paint_holes(found, sample.shape, offset, contrast)
    note(UNCROWDED, kept, painted, found)
    if halted(UNCROWDED):
        return found, painted, kept

    found = past_the_size_floor(
        found, contrast.min_width_px, contrast.min_height_px,
        contrast.min_area_px,
    )
    painted, kept = paint_holes(found, sample.shape, offset, contrast)
    note(SIZED, kept, painted, found)
    if halted(SIZED):
        return found, painted, kept

    found = clear_of_a_repeat(found, repeat_texture, contrast)
    painted, kept = paint_holes(found, sample.shape, offset, contrast)
    note(UNREPEATED, kept, painted, found)
    if halted(UNREPEATED):
        return found, painted, kept

    rim_measurements = rim_artifact_candidates(
        found, sample, offset, contrast.min_rim_margin_px,
        contrast.max_rim_margin_px, contrast.rim_margin_share,
        contrast.min_rim_contact, contrast.rim_artifact_max_span_px,
        contrast.rim_artifact_min_perimeter_contact,
        contrast.rim_artifact_large_max_fill,
    )
    classify_rim_candidates(
        rim_measurements, contrast.min_rim_contact,
        contrast.rim_artifact_max_span_px,
        contrast.rim_artifact_min_perimeter_contact,
        contrast.rim_artifact_large_max_fill,
    )
    if probe is not None:
        for candidate in rim_measurements:
            probe.observe(RegionObservation(
                stage=UNRIMMED, texture_name=texture_name, domain=label,
                region=candidate.region,
                features=rim_candidate_features(candidate),
                discarded=candidate.discarded,
            ))
    found = clear_of_the_rim(
        found, sample, offset, contrast.min_rim_margin_px,
        contrast.max_rim_margin_px, contrast.rim_margin_share,
        contrast.min_rim_contact, contrast.rim_artifact_max_span_px,
        contrast.rim_artifact_min_perimeter_contact,
        contrast.rim_artifact_large_max_fill,
        measurements=rim_measurements,
    )
    painted, kept = paint_holes(found, sample.shape, offset, contrast)
    note(UNRIMMED, kept, painted, found)
    if halted(UNRIMMED):
        return found, painted, kept

    transition_measurements = material_transition_candidates(
        found, colour, contrast.transition_min_span_px,
        contrast.transition_max_fill, contrast.transition_close_gap_px, offset,
        include_all=True, measure_all=probe is not None,
    )
    classify_material_transition_candidates(
        transition_measurements, contrast.transition_min_enclosed_texels,
        contrast.transition_min_body_void_share,
        contrast.transition_max_interior_detail,
        contrast.transition_min_long_aspect,
        contrast.transition_max_thin_trace_span_px,
        contrast.transition_min_thin_trace_aspect,
        contrast.transition_max_flat_loop_texels,
        contrast.transition_context_gap_px,
    )
    if probe is not None:
        for candidate in transition_measurements:
            probe.observe(RegionObservation(
                stage=UNTRANSITIONED, texture_name=texture_name, domain=label,
                region=candidate.region,
                features=material_transition_features(candidate),
                discarded=candidate.discarded,
            ))
    found = clear_of_material_transitions(
        found, colour, contrast.transition_min_span_px,
        contrast.transition_max_fill, contrast.transition_close_gap_px,
        contrast.transition_min_enclosed_texels,
        contrast.transition_min_body_void_share,
        contrast.transition_max_interior_detail,
        contrast.transition_min_long_aspect,
        contrast.transition_max_thin_trace_span_px,
        contrast.transition_min_thin_trace_aspect,
        contrast.transition_max_flat_loop_texels,
        contrast.transition_context_gap_px,
        measurements=transition_measurements, offset=offset,
    )
    painted, kept = paint_holes(found, sample.shape, offset, contrast)
    note(UNTRANSITIONED, kept, painted, found)
    if halted(UNTRANSITIONED):
        return found, painted, kept

    lit = clear_of_boundary(lit, sample, contrast.min_boundary_reach_px)
    note(OFF_BOUNDARY, lit)
    if halted(OFF_BOUNDARY):
        return stopped(lit)

    if contrast.min_structure > 0:
        lit = structure_seed(colour, lit, sample, contrast)
    note(SEEDED, lit)
    if halted(SEEDED):
        return stopped(lit)

    if contrast.min_seed_density > 0:
        lit = lit & (
            seed_density(lit, sample, contrast.seed_density_px)
            >= contrast.min_seed_density
        )
    note(DENSE_SEED, lit)
    if halted(DENSE_SEED):
        return stopped(lit)

    # A marker, not a filter: `lit` goes on untouched.  What is recorded is
    # which groups the transform will be run over, so where it will look can be
    # seen apart from what it decided.  Stopping here answers with those
    # groups, which is the point of being able to stop here.
    candidates = pattern_candidates(
        lit, sample, contrast.pattern_gap_px, contrast
    )
    note(PATTERNED, candidates)
    if halted(PATTERNED):
        found, painted = regions_from_marks(
            candidates, contrast.pattern_gap_px, label, offset, contrast
        )
        return stopped(candidates, found, painted)

    lit = lit & ~confirmed_repeats(
        candidates, contrast.pattern_gap_px, contrast
    )
    note(TRANSFORMED, lit)
    if halted(TRANSFORMED):
        return stopped(lit)

    # The one stage that adds rather than removes.  `drawn` is kept apart from
    # here down: everything downstream groups the bridged mask, and only the
    # stroke test still asks what was actually detected.
    drawn = lit
    lit = lit | gap_bridges(
        lit, ratio, sample, contrast.bridge_span_px,
        contrast.min_bridge_response, contrast.mark_gap_px,
        contrast.max_bridge_share, contrast.min_bridge_void,
        contrast.bridge_clear_px, contrast.bridge_void_px, threshold,
    )
    note(BRIDGED, lit)
    if halted(BRIDGED):
        return stopped(lit)

    lit = marks_from_seed(
        lit, contrast.mark_gap_px, contrast.min_stroke_texels, drawn
    )
    found, _painted = regions_from_marks(
        lit, contrast.mark_gap_px, label, offset, contrast
    )
    # 2026-08-29: the earlier rim pass filters sparse-field holes for its
    # stage view, but this is the list that becomes the response.  Apply the
    # same decision here, then redraw both the texels and paths from exactly
    # the kept regions so a rim setting affects the displayed result.
    found = clear_of_the_rim(
        found, sample, offset, contrast.min_rim_margin_px,
        contrast.max_rim_margin_px, contrast.rim_margin_share,
        contrast.min_rim_contact, contrast.rim_artifact_max_span_px,
        contrast.rim_artifact_min_perimeter_contact,
        contrast.rim_artifact_large_max_fill,
    )
    found = clear_of_material_transitions(
        found, colour, contrast.transition_min_span_px,
        contrast.transition_max_fill, contrast.transition_close_gap_px,
        contrast.transition_min_enclosed_texels,
        contrast.transition_min_body_void_share,
        contrast.transition_max_interior_detail,
        contrast.transition_min_long_aspect,
        contrast.transition_max_thin_trace_span_px,
        contrast.transition_min_thin_trace_aspect,
        contrast.transition_max_flat_loop_texels,
        contrast.transition_context_gap_px,
        offset=offset,
    )
    painted, lit = paint_mark_regions(
        found, lit.shape, offset, contrast.mark_gap_px, contrast
    )
    # Noted last so the regions are charged to the stage that makes them and
    # so stopping at it answers with the same filtered marks the app returns.
    note(MARKED, lit, painted, found)
    return found, painted, lit


def detect(
    texture: CorpusTexture,
    config: LabConfig | None = None,
    capture: bool = False,
    probe: RegionProbe | None = None,
) -> Detection:
    """Detect over every UV domain of one texture.

    ``capture`` records every stage in ``STAGES`` as it happens, for the
    viewer.  It costs a dict of references per domain and a pack at the end;
    nothing at all when it is off.
    """
    config = config or LabConfig()
    stages: dict[str, np.ndarray] = {}
    stage_outlines: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    timings: dict[str, float] = {}
    fired = np.zeros(texture.shape, dtype=bool)
    outlines = np.zeros(texture.shape, dtype=bool)
    sparse = np.zeros(texture.shape, dtype=bool)
    regions: list[Region] = []
    stop = int(config.contrast.stop_after_stage)
    if not 1 <= stop <= LAST_STOP:
        stop = 0
    runs_repeat_stage = stop == 0 or stop >= STAGES.index(UNREPEATED) + 1
    repeat_started = time.perf_counter()
    repeat_texture = (
        build_repeat_texture_index(texture.colour, config.contrast)
        if runs_repeat_stage and config.contrast.enable_repeat_texture_filter
        else None
    )
    if repeat_texture is not None:
        timings[UNREPEATED] = time.perf_counter() - repeat_started
    for label, window in enumerate(texture.domain_windows(), start=1):
        if window is None:
            continue
        x, y = int(window[1].start), int(window[0].start)
        domain = texture.domains[window] == label
        captured: dict[str, np.ndarray] | None = {} if capture else None
        drawn: dict[str, np.ndarray] | None = {} if capture else None
        blank = np.zeros(domain.shape, dtype=bool)
        found, outline, kept = regions_in_domain(
            texture.colour[window], domain, label, (x, y), config, blank,
            repeat_texture, captured, drawn, counts, timings=timings,
            probe=probe, texture_name=texture.name,
        )
        sparse[window] |= blank
        for source, whole_atlas in ((captured, stages), (drawn, stage_outlines)):
            if source is None:
                continue
            for name, mask in source.items():
                whole = whole_atlas.get(name)
                if whole is None:
                    whole = np.zeros(texture.shape, dtype=bool)
                    whole_atlas[name] = whole
                whole[window] |= mask
        regions.extend(found)
        fired[window] |= kept
        outlines[window] |= outline
    return Detection(
        regions=regions, fired=fired, outlines=outlines, sparse=sparse,
        stages={name: np.packbits(mask) for name, mask in stages.items()},
        stage_outlines={
            name: np.packbits(mask) for name, mask in stage_outlines.items()
        },
        stage_counts=counts,
        stage_times=timings,
    )
