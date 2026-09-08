"""A normal-map detector that looks for regions bounded by slopes.

This deliberately does not share the colour local-contrast pipeline.  A
normal edge-steepness image already says where the surface changes direction:
the useful question is where those slopes are absent, because a relief glyph
forms a bounded hole in otherwise slope-free ground.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from glyph_lab.corpus import CorpusTexture
from glyph_lab.detect import (
    NEIGHBOURHOOD,
    Detection,
    Region,
    paint_holes,
    regions_from_holes,
    sampling_mask,
    sparse_field,
)
from glyph_lab.mask_compute import (
    compute_backend,
    pipeline_candidate,
    prepared_items,
    sparse_result,
    submit_sparse_field,
)
from glyph_lab.region_grouping import (
    GroupingConfig,
    group_regions_across_flat_ground,
    mesh_region_ids,
    mesh_region_labels,
)
from glyph_lab.region_hulls import convex_region_hulls, convex_region_mask
from glyph_lab.region_padding import pad_regions
from glyph_lab.region_rectangles import fit_region_rectangles

NORMAL_ABSENT_SLOPES = "1 absence of slopes"
NORMAL_CLEAR_OF_CROWD = "2 clear of a crowd"
NORMAL_PAST_SIZE_FLOOR = "3 past the size floor"
NORMAL_CLEAR_OF_RIM = "4 clear of the rim"
NORMAL_CLEAR_OF_FRAME = "5 clear of a frame"
NORMAL_CLEAR_OF_STRIPE = "6 clear of a stripe"
NORMAL_GROUPED_FLAT_GROUND = "7 grouped across flat ground"
NORMAL_CLEAR_OF_PANEL = "8 clear of a panel face"
NORMAL_CONVEX_HULLED = "9 convex region hull"
NORMAL_FITTED_RECTANGLE = "10 best fit rectangle"
NORMAL_PADDED = "11 final padding"
NORMAL_STAGES = (
    NORMAL_ABSENT_SLOPES,
    NORMAL_CLEAR_OF_CROWD,
    NORMAL_PAST_SIZE_FLOOR,
    NORMAL_CLEAR_OF_RIM,
    NORMAL_CLEAR_OF_FRAME,
    NORMAL_CLEAR_OF_STRIPE,
    NORMAL_GROUPED_FLAT_GROUND,
    NORMAL_CLEAR_OF_PANEL,
    NORMAL_CONVEX_HULLED,
    NORMAL_FITTED_RECTANGLE,
    NORMAL_PADDED,
)
NORMAL_STAGE_OF = {
    "stop_after_stage": 0,
    "boundary_skip_px": 1,
    "slope_threshold": 1,
    "absence_kernel_px": 1,
    "max_slope_texels": 1,
    "min_void_texels": 1,
    "outline_px": 1,
    "crowd_radius_px": 2,
    "max_neighbours": 2,
    "crowd_large_region_area_px": 2,
    "min_width_px": 3,
    "min_height_px": 3,
    "min_area_px": 3,
    "min_rim_margin_px": 4,
    "max_rim_margin_px": 4,
    "rim_margin_share": 4,
    "min_rim_contact": 4,
    "rim_artifact_max_span_px": 4,
    "rim_artifact_min_perimeter_contact": 4,
    "rim_artifact_large_max_fill": 4,
    "frame_min_span_px": 5,
    "frame_min_enclosure": 5,
    "frame_max_holes": 5,
    "stripe_min_span_px": 6,
    "stripe_min_aspect": 6,
    "stripe_min_oblique_span_px": 6,
    "stripe_min_oblique_aspect": 6,
    "stripe_max_oblique_stroke_px": 6,
    "panel_sparse_min_span_px": 8,
    "panel_sparse_max_fill": 8,
    "panel_sparse_max_hull_fill": 8,
    "panel_solid_min_minor_span_px": 8,
    "panel_solid_min_fill": 8,
    "panel_solid_min_hull_fill": 8,
    "rectangle_chain_tolerance_degrees": 10,
    "rectangle_min_axis_lead_ratio": 10,
    "final_padding_px": 11,
}


@dataclass(frozen=True, slots=True)
class NormalMapConfig:
    """Only the decisions the normal-map pipeline currently makes."""

    # 0 means every normal-map stage runs, matching the colour convention.
    stop_after_stage: int = 0
    # Keep the one texel the UV raster can blend out of every measurement.
    boundary_skip_px: int = 1
    # Edge-steepness is 0..255; values at or above this are slopes.
    slope_threshold: int = 24
    # A disc with at most this many slope texels is slope-free ground.
    absence_kernel_px: int = 5
    max_slope_texels: int = 1
    # `paint_holes` uses these only to draw a region's visible path.
    min_void_texels: int = 4
    outline_px: int = 1
    # Normal-map-owned copy of the colour detector's crowd stage.
    crowd_radius_px: int = 120
    max_neighbours: int = 40
    crowd_large_region_area_px: int = 1024
    # Normal-map-owned copy of the colour detector's size floor.
    min_width_px: int = 16
    min_height_px: int = 16
    min_area_px: int = 400
    # This is deliberately a normal-map copy of the colour detector's rim
    # stage.  The two pipelines will tune it independently from here on.
    min_rim_margin_px: int = 0
    max_rim_margin_px: int = 1000
    rim_margin_share: float = 0.50
    min_rim_contact: float = 0.10
    rim_artifact_max_span_px: int = 32
    rim_artifact_min_perimeter_contact: float = 0.02
    rim_artifact_large_max_fill: float = 1.0
    # A large low-complexity closed bezel is not a glyph.  This runs only
    # after crowd, size, and rim filters, so its region-local measurements are
    # cheap even on a 4k atlas.
    frame_min_span_px: int = 64
    frame_min_enclosure: float = 0.30
    frame_max_holes: int = 3
    # Parallel trim lines form large, highly elongated slope-free regions.
    # Their bounds are already known, so this is effectively free.
    stripe_min_span_px: int = 96
    stripe_min_aspect: float = 7.0
    # 2026-09-01: the direct bounds test above cannot see a diagonal trim
    # line.  Its oriented span/aspect and narrow stroke separate it from the
    # retained Ardente word (aspect 9.27, stroke 8.9) with room to spare.
    stripe_min_oblique_span_px: int = 64
    stripe_min_oblique_aspect: float = 8.0
    stripe_max_oblique_stroke_px: float = 5.0
    # 2026-09-01: after grouping, false normal hits are either broad sparse
    # panel outlines or sizeable solid control faces.  The two signatures have
    # clear gaps from the corpus's multi-stroke glyphs, so both remain cheap
    # region-local tests rather than another atlas-wide image pass.
    panel_sparse_min_span_px: int = 128
    panel_sparse_max_fill: float = 0.20
    panel_sparse_max_hull_fill: float = 0.25
    panel_solid_min_minor_span_px: int = 40
    panel_solid_min_fill: float = 0.60
    panel_solid_min_hull_fill: float = 0.65
    # 2026-09-01: text baselines often have a few rasterised hull steps, but
    # a permissive angle follows a nearby glyph side rather than the baseline.
    rectangle_chain_tolerance_degrees: float = 5.0
    rectangle_min_axis_lead_ratio: float = 1.25
    final_padding_px: int = 2


@dataclass(slots=True)
class NormalRimCandidate:
    """Normal-pipeline-owned measurements for one possible rim artefact."""

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


def normal_boundary_margin(size: float, floor: int, ceiling: int, share: float) -> int:
    """The normal detector's private copy of the UV-rim margin rule."""
    low = max(int(floor), 0)
    high = max(int(ceiling), low)
    return int(min(max(float(share) * float(size), low), high))


def normal_rim_contact(mask: np.ndarray, rim: np.ndarray, length: int) -> float:
    """How much of a normal candidate runs along the sampled UV rim."""
    if not mask.any():
        return 0.0
    grown = ndimage.binary_dilation(mask, structure=NEIGHBOURHOOD)
    return float((rim & grown).sum()) / float(max(int(length), 1))


def normal_rim_perimeter_contact(mask: np.ndarray, rim: np.ndarray) -> float:
    """The fraction of one normal candidate's perimeter that touches the rim."""
    if not mask.any():
        return 0.0
    perimeter = mask & ~ndimage.binary_erosion(
        mask, structure=NEIGHBOURHOOD, border_value=0,
    )
    touching = rim & ndimage.binary_dilation(mask, structure=NEIGHBOURHOOD)
    return float(touching.sum()) / float(max(int(perimeter.sum()), 1))


def normal_rim_artifact_candidates(
    regions: list[Region], sample: np.ndarray, offset: tuple[int, int],
    floor: int, ceiling: int, share: float, contact: float = 0.0,
    max_band_span: int = 0, min_perimeter_contact: float = 0.30,
    large_max_fill: float = 0.15,
) -> list[NormalRimCandidate]:
    """Measure normal-map regions that live wholly in a UV-rim margin."""
    if (floor <= 0 and share <= 0) or not regions:
        return []
    margin = normal_boundary_margin(
        (float(sample.shape[0]) + float(sample.shape[1])) / 2.0,
        floor, ceiling, share,
    )
    if margin <= 0:
        return []
    interior = ndimage.binary_erosion(
        sample, structure=NEIGHBOURHOOD, iterations=margin, border_value=0,
    )
    rim = sample & ~ndimage.binary_erosion(
        sample, structure=NEIGHBOURHOOD, border_value=0,
    )
    max_span = max(int(max_band_span), 0)
    long_run = max(max_span * 3, 48)
    x_offset, y_offset = offset
    candidates: list[NormalRimCandidate] = []
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
            contact_mask, structure=NEIGHBOURHOOD,
        ) & rim_piece
        perimeter = contact_mask & ~ndimage.binary_erosion(
            contact_mask, structure=NEIGHBOURHOOD, border_value=0,
        )
        rows_on_rim, columns_on_rim = np.nonzero(touches)
        minor, major = min(width, height), max(width, height)
        length = normal_rim_contact(contact_mask, rim_piece, major)
        perimeter_share = normal_rim_perimeter_contact(contact_mask, rim_piece)
        score = length if major >= long_run else perimeter_share
        width_eligible = max_span == 0 or minor <= max_span
        perimeter_eligible = (
            float(min_perimeter_contact) > 0.0
            and perimeter_share >= float(min_perimeter_contact)
        )
        candidates.append(NormalRimCandidate(
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


def classify_normal_rim_candidates(
    candidates: list[NormalRimCandidate], contact: float, max_band_span: int,
    min_perimeter_contact: float = 0.30, large_max_fill: float = 0.15,
) -> None:
    """Make the normal pipeline's rim-artifact decisions independently."""
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
        if candidate.major_span_px >= long_run or candidate.perimeter_eligible:
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


def clear_of_normal_rim(
    regions: list[Region], sample: np.ndarray, offset: tuple[int, int],
    config: NormalMapConfig,
) -> list[Region]:
    """The normal-map-owned copy of the colour detector's rim filter."""
    measured = normal_rim_artifact_candidates(
        regions, sample, offset, config.min_rim_margin_px,
        config.max_rim_margin_px, config.rim_margin_share,
        config.min_rim_contact, config.rim_artifact_max_span_px,
        config.rim_artifact_min_perimeter_contact,
        config.rim_artifact_large_max_fill,
    )
    classify_normal_rim_candidates(
        measured, config.min_rim_contact, config.rim_artifact_max_span_px,
        config.rim_artifact_min_perimeter_contact,
        config.rim_artifact_large_max_fill,
    )
    discarded = {id(candidate.region) for candidate in measured if candidate.discarded}
    return [region for region in regions if id(region) not in discarded]


def normal_region_centres(regions: list[Region]) -> np.ndarray:
    """Return normal-pipeline region centres as an (x, y) array."""
    if not regions:
        return np.zeros((0, 2), dtype=np.float64)
    return np.array(
        [(x + width / 2.0, y + height / 2.0)
         for x, y, width, height in (region.bounds for region in regions)],
        dtype=np.float64,
    )


def clear_of_normal_crowd(
    regions: list[Region], company: list[Region], config: NormalMapConfig,
    region_ids: list[int | None] | None = None,
    company_ids: list[int | None] | None = None,
) -> list[Region]:
    """The normal-map-owned copy of the colour detector's crowd filter.

    2026-09-04: company is counted only within the mark's own seam-bounded mesh
    piece, matching the colour detector and `_links`.  Relief runs into this
    harder than colour does: a stitched or perforated panel fires densely and
    sits hard against quieter charts in the atlas, so its company was crossing
    the seam and suppressing them.
    """
    radius, limit = config.crowd_radius_px, config.max_neighbours
    if radius <= 0 or limit < 0 or not regions or not company:
        return regions
    floor = max(int(config.crowd_large_region_area_px), 0)
    judged = [
        index for index, region in enumerate(regions)
        if floor == 0 or region.texels < floor
    ]
    if not judged:
        return regions
    same = regions is company
    if region_ids is None or company_ids is None:
        trees = {None: cKDTree(normal_region_centres(company))}
        buckets = {index: None for index in judged}
    else:
        by_piece: dict[int, list[Region]] = {}
        for piece, member in zip(company_ids, company):
            if piece is not None:
                by_piece.setdefault(piece, []).append(member)
        trees = {
            piece: cKDTree(normal_region_centres(members))
            for piece, members in by_piece.items()
        }
        buckets = {index: region_ids[index] for index in judged}
    counted: dict[int, int] = {}
    for piece in set(buckets.values()):
        asking = [index for index in judged if buckets[index] == piece]
        tree = trees.get(piece)
        if tree is None:
            # No chart owns this mark, so nothing is on its chart to crowd it.
            counted.update({index: 0 for index in asking})
            continue
        near = tree.query_ball_point(
            normal_region_centres([regions[index] for index in asking]),
            float(radius), return_length=True,
        )
        counted.update({index: int(count) for index, count in zip(asking, near)})
    refused = {
        index for index in judged
        if counted[index] - (1 if same else 0) > int(limit)
    }
    return [
        region for index, region in enumerate(regions) if index not in refused
    ]


def past_normal_size_floor(
    regions: list[Region], config: NormalMapConfig,
) -> list[Region]:
    """The normal-map-owned copy of the colour detector's size floor."""
    return [
        region for region in regions
        if region.bounds[2] >= int(config.min_width_px)
        and region.bounds[3] >= int(config.min_height_px)
        and region.texels >= int(config.min_area_px)
    ]


def clear_of_normal_frame(
    regions: list[Region], config: NormalMapConfig,
) -> list[Region]:
    """Remove large, simple enclosed bezels without reading the atlas again."""
    min_span = max(int(config.frame_min_span_px), 0)
    min_enclosure = float(config.frame_min_enclosure)
    max_holes = int(config.frame_max_holes)
    if min_span <= 0 or min_enclosure <= 0.0 or max_holes < 0:
        return regions
    return [
        region for region in regions
        if (
            max(region.bounds[2], region.bounds[3]) < min_span
            or region.enclosure < min_enclosure
            or region.holes > max_holes
        )
    ]


def clear_of_normal_stripe(
    regions: list[Region], config: NormalMapConfig,
) -> list[Region]:
    """Remove large, highly elongated regions left by parallel trim lines."""
    min_span = max(int(config.stripe_min_span_px), 0)
    min_aspect = float(config.stripe_min_aspect)
    oblique_span = max(int(config.stripe_min_oblique_span_px), 0)
    oblique_aspect = float(config.stripe_min_oblique_aspect)
    max_oblique_stroke = float(config.stripe_max_oblique_stroke_px)
    if (
        (min_span <= 0 or min_aspect <= 1.0)
        and (oblique_span <= 0 or oblique_aspect <= 1.0
             or max_oblique_stroke <= 0.0)
    ):
        return regions
    kept: list[Region] = []
    for region in regions:
        width, height = region.width, region.height
        axis_aligned = (
            min_span > 0 and min_aspect > 1.0
            and max(width, height) >= min_span
            and max(width, height) / max(min(width, height), 1) >= min_aspect
        )
        oblique = False
        if oblique_span > 0 and oblique_aspect > 1.0 and max_oblique_stroke > 0.0:
            rows, columns = np.nonzero(region.mask)
            if rows.size >= 2:
                points = np.column_stack((columns, rows)).astype(np.float64)
                centred = points - points.mean(axis=0)
                _left, values, directions = np.linalg.svd(
                    centred, full_matrices=False,
                )
                minor = max(float(values[1]) if values.size > 1 else 0.0, 1e-6)
                oriented_aspect = float(values[0]) / minor
                oriented_span = float(np.ptp(centred @ directions[0]))
                oblique = (
                    oriented_span >= oblique_span
                    and oriented_aspect >= oblique_aspect
                    and region.stroke <= max_oblique_stroke
                )
        if not axis_aligned and not oblique:
            kept.append(region)
    return kept


def clear_of_normal_panel_face(
    regions: list[Region], config: NormalMapConfig,
) -> list[Region]:
    """Reject normal-map hardware faces, retaining deliberately ambiguous marks.

    This runs after joining so a word or badge is measured as one mark, but
    before hulling so its genuine concavities remain evidence.  A convex hull
    is rasterised only for the few regions that reach this late stage.
    """
    sparse_span = max(int(config.panel_sparse_min_span_px), 0)
    sparse_fill = float(config.panel_sparse_max_fill)
    sparse_hull_fill = float(config.panel_sparse_max_hull_fill)
    solid_minor = max(int(config.panel_solid_min_minor_span_px), 0)
    solid_fill = float(config.panel_solid_min_fill)
    solid_hull_fill = float(config.panel_solid_min_hull_fill)
    if not regions or (sparse_span <= 0 and solid_minor <= 0):
        return regions
    kept: list[Region] = []
    for region in regions:
        hull_texels = max(int(convex_region_mask(region).sum()), 1)
        hull_fill = float(region.texels) / hull_texels
        major = max(region.width, region.height)
        minor = min(region.width, region.height)
        broad_sparse_perimeter = (
            sparse_span > 0
            and major >= sparse_span
            and region.fill <= sparse_fill
            and hull_fill <= sparse_hull_fill
        )
        solid_control_face = (
            solid_minor > 0
            and minor >= solid_minor
            and region.fill >= solid_fill
            and hull_fill >= solid_hull_fill
        )
        if not broad_sparse_perimeter and not solid_control_face:
            kept.append(region)
    return kept


def detect_normal(
    texture: CorpusTexture,
    config: NormalMapConfig | None = None,
    capture: bool = False,
    grouping: GroupingConfig | None = None,
    reference_colour: np.ndarray | None = None,
    backend: str | None = None,
) -> Detection:
    """Paint holes left in a slope-free field over every UV domain."""
    config = config or NormalMapConfig()
    backend = compute_backend(backend)
    grouping = grouping or GroupingConfig()
    colour_evidence = texture.colour if reference_colour is None else reference_colour
    mesh_regions = mesh_region_labels(texture.domains, texture.seams)
    started = time.perf_counter()
    fired = np.zeros(texture.shape, dtype=bool)
    outlines = np.zeros(texture.shape, dtype=bool)
    sparse = np.zeros(texture.shape, dtype=bool)
    regions: list[Region] = []
    stage_counts = dict.fromkeys(NORMAL_STAGES, 0)
    stage_times = dict.fromkeys(NORMAL_STAGES, 0.0)
    stage_masks = (
        {stage: np.zeros(texture.shape, dtype=bool) for stage in NORMAL_STAGES}
        if capture else {}
    )
    stage_outlines = (
        {stage: np.zeros(texture.shape, dtype=bool) for stage in NORMAL_STAGES}
        if capture else {}
    )
    stop = int(config.stop_after_stage)
    if not 1 <= stop <= len(NORMAL_STAGES):
        stop = 0

    windows = [(label, window) for label, window in enumerate(texture.domain_windows(), start=1)
               if window is not None]
    parallel = pipeline_candidate(windows, config.absence_kernel_px, backend)

    def prepare(item):
        label, window = item
        domain = texture.domains[window] == label
        sample = sampling_mask(domain, config.boundary_skip_px)
        slopes = (texture.colour[window][:, :, 0] >= config.slope_threshold) & sample
        future = submit_sparse_field(
            slopes, sample, None, config.absence_kernel_px, config.max_slope_texels, backend if parallel else "cpu",
        )
        return label, window, sample, slopes, future

    with prepared_items(windows, prepare, parallel) as prepared:
        for label, window, sample, slopes, future in prepared:
            x, y = int(window[1].start), int(window[0].start)
            slope_free, _sample = sparse_result(
                future,
                lambda slopes=slopes, sample=sample: (sparse_field(slopes, sample, config.absence_kernel_px,
                                      config.max_slope_texels), sample),
            )
            holes = regions_from_holes(slope_free, sample, label, (x, y))
            outline, kept = paint_holes(holes, sample.shape, (x, y), config)  # type: ignore[arg-type]
            stage_counts[NORMAL_ABSENT_SLOPES] += len(holes)
            if capture:
                stage_masks[NORMAL_ABSENT_SLOPES][window] |= kept
                stage_outlines[NORMAL_ABSENT_SLOPES][window] |= outline
            if stop == 1:
                accepted = holes
                accepted_outline, accepted_kept = outline, kept
            else:
                before_crowd = time.perf_counter()
                hole_ids = mesh_region_ids(holes, mesh_regions[window], (x, y))
                uncrowded = clear_of_normal_crowd(
                    holes, holes, config,
                    region_ids=hole_ids, company_ids=hole_ids,
                )
                crowd_outline, crowd_kept = paint_holes(
                    uncrowded, sample.shape, (x, y), config,  # type: ignore[arg-type]
                )
                stage_times[NORMAL_CLEAR_OF_CROWD] += time.perf_counter() - before_crowd
                stage_counts[NORMAL_CLEAR_OF_CROWD] += len(uncrowded)
                if capture:
                    stage_masks[NORMAL_CLEAR_OF_CROWD][window] |= crowd_kept
                    stage_outlines[NORMAL_CLEAR_OF_CROWD][window] |= crowd_outline
                if stop == 2:
                    accepted = uncrowded
                    accepted_outline, accepted_kept = crowd_outline, crowd_kept
                else:
                    before_size = time.perf_counter()
                    sized = past_normal_size_floor(uncrowded, config)
                    size_outline, size_kept = paint_holes(
                        sized, sample.shape, (x, y), config,  # type: ignore[arg-type]
                    )
                    stage_times[NORMAL_PAST_SIZE_FLOOR] += (
                        time.perf_counter() - before_size
                    )
                    stage_counts[NORMAL_PAST_SIZE_FLOOR] += len(sized)
                    if capture:
                        stage_masks[NORMAL_PAST_SIZE_FLOOR][window] |= size_kept
                        stage_outlines[NORMAL_PAST_SIZE_FLOOR][window] |= size_outline
                    if stop == 3:
                        accepted = sized
                        accepted_outline, accepted_kept = size_outline, size_kept
                    else:
                        before_rim = time.perf_counter()
                        rimmed = clear_of_normal_rim(sized, sample, (x, y), config)
                        rim_outline, rim_kept = paint_holes(
                            rimmed, sample.shape, (x, y), config,  # type: ignore[arg-type]
                        )
                        stage_times[NORMAL_CLEAR_OF_RIM] += (
                            time.perf_counter() - before_rim
                        )
                        stage_counts[NORMAL_CLEAR_OF_RIM] += len(rimmed)
                        if capture:
                            stage_masks[NORMAL_CLEAR_OF_RIM][window] |= rim_kept
                            stage_outlines[NORMAL_CLEAR_OF_RIM][window] |= rim_outline
                        if stop == 4:
                            accepted = rimmed
                            accepted_outline, accepted_kept = rim_outline, rim_kept
                        else:
                            before_frame = time.perf_counter()
                            framed = clear_of_normal_frame(rimmed, config)
                            frame_outline, frame_kept = paint_holes(
                                framed, sample.shape, (x, y), config,  # type: ignore[arg-type]
                            )
                            stage_times[NORMAL_CLEAR_OF_FRAME] += (
                                time.perf_counter() - before_frame
                            )
                            stage_counts[NORMAL_CLEAR_OF_FRAME] += len(framed)
                            if capture:
                                stage_masks[NORMAL_CLEAR_OF_FRAME][window] |= frame_kept
                                stage_outlines[NORMAL_CLEAR_OF_FRAME][window] |= frame_outline
                            if stop == 5:
                                accepted = framed
                                accepted_outline, accepted_kept = frame_outline, frame_kept
                            else:
                                before_stripe = time.perf_counter()
                                accepted = clear_of_normal_stripe(framed, config)
                                accepted_outline, accepted_kept = paint_holes(
                                    accepted, sample.shape, (x, y), config,  # type: ignore[arg-type]
                                )
                                stage_times[NORMAL_CLEAR_OF_STRIPE] += (
                                    time.perf_counter() - before_stripe
                                )
                                stage_counts[NORMAL_CLEAR_OF_STRIPE] += len(accepted)
                                if capture:
                                    stage_masks[NORMAL_CLEAR_OF_STRIPE][window] |= accepted_kept
                                    stage_outlines[NORMAL_CLEAR_OF_STRIPE][window] |= accepted_outline
                                if stop != 6:
                                    before_group = time.perf_counter()
                                    accepted = group_regions_across_flat_ground(
                                        accepted, colour_evidence[window], slope_free,
                                        grouping, (x, y), mesh_regions[window],
                                    )
                                    accepted_outline, accepted_kept = paint_holes(
                                        accepted, sample.shape, (x, y), config,  # type: ignore[arg-type]
                                    )
                                    stage_times[NORMAL_GROUPED_FLAT_GROUND] += (
                                        time.perf_counter() - before_group
                                    )
                                    stage_counts[NORMAL_GROUPED_FLAT_GROUND] += len(accepted)
                                if capture:
                                    stage_masks[NORMAL_GROUPED_FLAT_GROUND][window] |= accepted_kept
                                    stage_outlines[NORMAL_GROUPED_FLAT_GROUND][window] |= accepted_outline
                                if stop not in (6, 7):
                                    before_panel = time.perf_counter()
                                    accepted = clear_of_normal_panel_face(accepted, config)
                                    accepted_outline, accepted_kept = paint_holes(
                                        accepted, sample.shape, (x, y), config,  # type: ignore[arg-type]
                                    )
                                    stage_times[NORMAL_CLEAR_OF_PANEL] += (
                                        time.perf_counter() - before_panel
                                    )
                                    stage_counts[NORMAL_CLEAR_OF_PANEL] += len(accepted)
                                    if capture:
                                        stage_masks[NORMAL_CLEAR_OF_PANEL][window] |= accepted_kept
                                        stage_outlines[NORMAL_CLEAR_OF_PANEL][window] |= accepted_outline
                                if stop not in (6, 7, 8):
                                    before_hull = time.perf_counter()
                                    accepted = convex_region_hulls(accepted)
                                    accepted_outline, accepted_kept = paint_holes(
                                        accepted, sample.shape, (x, y), config,  # type: ignore[arg-type]
                                    )
                                    stage_times[NORMAL_CONVEX_HULLED] += (
                                        time.perf_counter() - before_hull
                                    )
                                    stage_counts[NORMAL_CONVEX_HULLED] += len(accepted)
                                    if capture:
                                        stage_masks[NORMAL_CONVEX_HULLED][window] |= accepted_kept
                                        stage_outlines[NORMAL_CONVEX_HULLED][window] |= accepted_outline
                                if stop not in (6, 7, 8, 9):
                                    before_rectangle = time.perf_counter()
                                    fitted = fit_region_rectangles(
                                        accepted, config.rectangle_chain_tolerance_degrees,
                                        config.rectangle_min_axis_lead_ratio,
                                    )
                                    accepted_outline, accepted_kept = paint_holes(
                                        fitted, sample.shape, (x, y), config,  # type: ignore[arg-type]
                                    )
                                    stage_times[NORMAL_FITTED_RECTANGLE] += (
                                        time.perf_counter() - before_rectangle
                                    )
                                    stage_counts[NORMAL_FITTED_RECTANGLE] += len(fitted)
                                    if capture:
                                        stage_masks[NORMAL_FITTED_RECTANGLE][window] |= accepted_kept
                                        stage_outlines[NORMAL_FITTED_RECTANGLE][window] |= accepted_outline
                                    # 2026-09-03: the fitted rectangle describes
                                    # a baseline for rotation, but must not replace
                                    # the convex region that correction flips. The
                                    # latter is passed through the exporter as a
                                    # mask, closed over its reflected partner.
                                if stop not in (6, 7, 8, 9, 10):
                                    before_padding = time.perf_counter()
                                    accepted = pad_regions(
                                        accepted, config.final_padding_px, sample, (x, y),
                                    )
                                    accepted_outline, accepted_kept = paint_holes(
                                        accepted, sample.shape, (x, y), config,  # type: ignore[arg-type]
                                    )
                                    stage_times[NORMAL_PADDED] += (
                                        time.perf_counter() - before_padding
                                    )
                                    stage_counts[NORMAL_PADDED] += len(accepted)
                                    if capture:
                                        stage_masks[NORMAL_PADDED][window] |= accepted_kept
                                        stage_outlines[NORMAL_PADDED][window] |= accepted_outline
            sparse[window] |= slope_free
            fired[window] |= accepted_kept
            outlines[window] |= accepted_outline
            regions.extend(accepted)

    elapsed = time.perf_counter() - started
    stage_times[NORMAL_ABSENT_SLOPES] = max(
        elapsed - stage_times[NORMAL_CLEAR_OF_RIM]
        - stage_times[NORMAL_PAST_SIZE_FLOOR]
        - stage_times[NORMAL_CLEAR_OF_CROWD]
        - stage_times[NORMAL_CLEAR_OF_FRAME]
        - stage_times[NORMAL_CLEAR_OF_STRIPE]
        - stage_times[NORMAL_GROUPED_FLAT_GROUND]
        - stage_times[NORMAL_CLEAR_OF_PANEL]
        - stage_times[NORMAL_CONVEX_HULLED]
        - stage_times[NORMAL_FITTED_RECTANGLE]
        - stage_times[NORMAL_PADDED], 0.0,
    )
    stages = {name: np.packbits(mask) for name, mask in stage_masks.items()}
    packed_outlines = {
        name: np.packbits(mask) for name, mask in stage_outlines.items()
    }
    return Detection(
        regions=regions, fired=fired, outlines=outlines, sparse=sparse,
        stages=stages, stage_outlines=packed_outlines,
        stage_counts=stage_counts, stage_times=stage_times,
    )
