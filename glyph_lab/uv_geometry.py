"""Simple UV-chart frames recovered from the authored mesh-seam raster."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

# 2026-09-02: the hull helpers moved to ``glyph_lab.hulls`` and are re-exported
# here so callers that always found them in this module still do.  The move is
# about imports, not about geometry: this module reads production defaults, and
# the detector -- which the export path now imports -- must not reach them.
from glyph_lab.hulls import (
    SEAM_CORNER_TOLERANCE_PX,
    convex_hull,
    simplified_hull,
)
from mesh_segmentation_transform.mirror_texture_for_rhd import (
    DEFAULT_RHD_CONFIG,
)

DEFAULT_SEAM_PARALLELOGRAM_OPPOSITE_EDGE_DOT = 0.90
PARALLELOGRAM_MIN_EDGE_PX = 2.0
PARALLELOGRAM_MIN_BOUNDARY_COVERAGE = 0.90
PARALLELOGRAM_MAX_BOUNDARY_GAP_PX = 1.0


@dataclass(frozen=True, slots=True)
class SimpleUvGeometryConfig:
    """The seam-frame decisions Glyph Lab can tune without loading a DAE."""

    seam_parallelogram_opposite_edge_dot: float = (
        DEFAULT_SEAM_PARALLELOGRAM_OPPOSITE_EDGE_DOT
    )
    seam_symmetry_min_score: float = 0.90
    seam_symmetry_min_area_px: int = DEFAULT_RHD_CONFIG.simple_uv_min_area_px
    seam_symmetry_axis_search_ratio: float = DEFAULT_RHD_CONFIG.island_axis_search_ratio
    # 2026-08-30: zero keeps the pixel-error gate disabled for existing tuning.
    seam_symmetry_max_unmirrored_area_px: int = 256
    seam_corner_tolerance_px: float = SEAM_CORNER_TOLERANCE_PX
    parallelogram_min_edge_px: float = PARALLELOGRAM_MIN_EDGE_PX
    parallelogram_min_boundary_coverage: float = PARALLELOGRAM_MIN_BOUNDARY_COVERAGE
    parallelogram_max_boundary_gap_px: float = PARALLELOGRAM_MAX_BOUNDARY_GAP_PX


DEFAULT_SIMPLE_UV_GEOMETRY_CONFIG = SimpleUvGeometryConfig()


def _has_axis_aligned_edge(edges: np.ndarray, tolerance_px: float) -> bool:
    """Whether a frame side is indistinguishable from horizontal or vertical."""
    tolerance = max(float(tolerance_px), 1e-9)
    return bool(
        (np.abs(edges[:, 0]) <= tolerance).any()
        or (np.abs(edges[:, 1]) <= tolerance).any()
    )


def fit_seam_parallelogram(
    seams: np.ndarray,
    opposite_edge_dot: float = DEFAULT_SEAM_PARALLELOGRAM_OPPOSITE_EDGE_DOT,
    origin: tuple[int, int] = (0, 0),
    corner_tolerance_px: float = SEAM_CORNER_TOLERANCE_PX,
    min_edge_px: float = PARALLELOGRAM_MIN_EDGE_PX,
) -> tuple[tuple[float, float], ...] | None:
    """Recover a chart frame from its four authored mesh-seam corners.

    2026-08-30: source boundaries do not always expose four distinct corners,
    while their discontinuity lines do.  A seam region must expose exactly four
    hull corners; arbitrary many-corner regions are not reduced to a convenient
    quadrilateral.  Require both opposing edge pairs to point opposite ways,
    then average opposing sides about the original centre for the drawn frame.
    """
    rows, columns = np.nonzero(seams)
    if len(rows) < 4:
        return None
    x_offset, y_offset = origin
    hull = simplified_hull(
        convex_hull(np.column_stack((columns + x_offset, rows + y_offset))),
        corner_tolerance_px,
    )
    if len(hull) != 4:
        return None
    corners = np.asarray(hull, dtype=np.float64)
    edges = np.roll(corners, -1, axis=0) - corners
    lengths = np.linalg.norm(edges, axis=1)
    if bool((lengths <= max(float(min_edge_px), 1e-6)).any()):
        return None
    if _has_axis_aligned_edge(edges, corner_tolerance_px):
        return None
    unit = edges / lengths[:, None]
    if (float(unit[0] @ unit[2]) > -opposite_edge_dot
            or float(unit[1] @ unit[3]) > -opposite_edge_dot):
        return None
    first = (edges[0] - edges[2]) / 2.0
    second = (edges[1] - edges[3]) / 2.0
    centre = corners.mean(axis=0)
    averaged = np.asarray((
        centre - (first + second) / 2.0,
        centre + (first - second) / 2.0,
        centre + (first + second) / 2.0,
        centre + (-first + second) / 2.0,
    ))
    return tuple(tuple(float(value) for value in point) for point in averaged)


def fit_source_seam_parallelogram(
    segments: tuple[tuple[tuple[float, float], tuple[float, float]], ...],
    size: tuple[int, int],
    config: SimpleUvGeometryConfig = DEFAULT_SIMPLE_UV_GEOMETRY_CONFIG,
) -> tuple[tuple[float, float], ...] | None:
    """Fit four raw authored seam corners, before a wrapped chart is clipped."""
    if len(segments) < 4:
        return None
    width, height = size
    points = np.asarray(segments, dtype=np.float64).reshape(-1, 2)
    pixels = points.copy()
    pixels[:, 0] *= max(width - 1, 1)
    pixels[:, 1] = (1.0 - pixels[:, 1]) * max(height - 1, 1)
    corners = simplified_hull(
        convex_hull(pixels), config.seam_corner_tolerance_px,
    )
    if len(corners) != 4:
        return None
    edges = np.roll(corners, -1, axis=0) - corners
    lengths = np.linalg.norm(edges, axis=1)
    if bool((lengths <= max(float(config.parallelogram_min_edge_px), 1e-9)).any()):
        return None
    if _has_axis_aligned_edge(edges, config.seam_corner_tolerance_px):
        return None
    unit = edges / lengths[:, None]
    threshold = config.seam_parallelogram_opposite_edge_dot
    if (float(unit[0] @ unit[2]) > -threshold
            or float(unit[1] @ unit[3]) > -threshold):
        return None
    first = (edges[0] - edges[2]) / 2.0
    second = (edges[1] - edges[3]) / 2.0
    centre = corners.mean(axis=0)
    averaged = np.asarray((
        centre - (first + second) / 2.0,
        centre + (first - second) / 2.0,
        centre + (first + second) / 2.0,
        centre + (-first + second) / 2.0,
    ))
    source = averaged.copy()
    source[:, 0] /= max(width - 1, 1)
    source[:, 1] = 1.0 - source[:, 1] / max(height - 1, 1)
    return tuple(tuple(float(value) for value in point) for point in source)


def _clip_to_tile(
    first: np.ndarray, second: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the part of a source-UV line inside the canonical tile."""
    delta = second - first
    lower, upper = 0.0, 1.0
    for change, room in (
        (-delta[0], first[0]), (delta[0], 1.0 - first[0]),
        (-delta[1], first[1]), (delta[1], 1.0 - first[1]),
    ):
        if abs(change) < 1e-12:
            if room < 0.0:
                return None
            continue
        distance = room / change
        if change < 0.0:
            if distance > upper:
                return None
            lower = max(lower, distance)
        else:
            if distance < lower:
                return None
            upper = min(upper, distance)
    if upper < lower:
        return None
    return first + delta * lower, first + delta * upper


def wrapped_frame_segments(
    corners: tuple[tuple[float, float], ...], size: tuple[int, int],
) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    """Clip each fitted source-UV frame edge into the displayed atlas tile."""
    width, height = size
    pieces: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for first, second in zip(corners, (*corners[1:], corners[0]), strict=True):
        first_array, second_array = np.asarray(first), np.asarray(second)
        for shift_u in range(
            int(np.floor(min(first[0], second[0]))),
            int(np.floor(max(first[0], second[0]))) + 1,
        ):
            for shift_v in range(
                int(np.floor(min(first[1], second[1]))),
                int(np.floor(max(first[1], second[1]))) + 1,
            ):
                clipped = _clip_to_tile(
                    first_array - (shift_u, shift_v),
                    second_array - (shift_u, shift_v),
                )
                if clipped is None:
                    continue
                pieces.append(tuple(
                    # 2026-08-30: source UVs have V up; corpus pixels have Y down.
                    (float(point[0] * (width - 1)),
                     float((1.0 - point[1]) * (height - 1)))
                    for point in clipped
                ))
    return tuple(pieces)


def fit_domain_seam_parallelogram(
    domains: np.ndarray,
    seams: np.ndarray,
    label: int,
    opposite_edge_dot: float = DEFAULT_SEAM_PARALLELOGRAM_OPPOSITE_EDGE_DOT,
) -> tuple[tuple[float, float], ...] | None:
    """Fit one domain, treating non-zero seam extents as seam geometry."""
    return fit_seam_parallelogram(
        np.asarray(seams, dtype=bool) & (domains == label), opposite_edge_dot,
    )


def seam_region_parallelograms(
    domains: np.ndarray,
    seams: np.ndarray,
    config: SimpleUvGeometryConfig = DEFAULT_SIMPLE_UV_GEOMETRY_CONFIG,
) -> tuple[tuple[int, tuple[tuple[float, float], ...]], ...]:
    """Fit each connected mesh seam region independently.

    2026-08-30: labels in the packed corpus may coalesce overlapping charts.
    They identify where a seam lies for display only; they never participate in
    constructing a candidate frame.
    """
    seam_labels, _count = ndimage.label(
        np.asarray(seams, dtype=bool), structure=np.ones((3, 3), dtype=np.uint8),
    )
    candidates: list[tuple[int, tuple[tuple[float, float], ...]]] = []
    for seam_label, window in enumerate(ndimage.find_objects(seam_labels), start=1):
        if window is None:
            continue
        rows, columns = window
        region = seam_labels[rows, columns] == seam_label
        corners = fit_seam_parallelogram(
            region,
            config.seam_parallelogram_opposite_edge_dot,
            (columns.start, rows.start),
            config.seam_corner_tolerance_px,
            config.parallelogram_min_edge_px,
        )
        if corners is None:
            continue
        nearby = ndimage.binary_dilation(region, structure=np.ones((3, 3), dtype=bool))
        labels = domains[rows, columns][nearby]
        labels = labels[labels > 0]
        if not len(labels):
            continue
        values, counts = np.unique(labels, return_counts=True)
        candidates.append((int(values[np.argmax(counts)]), corners))
    return tuple(candidates)


def seam_region_axis_candidates(
    region: np.ndarray,
    origin: tuple[int, int],
    minimum_area_px: int = 4,
) -> tuple[tuple[float, tuple[tuple[float, float], tuple[float, float]]], ...]:
    """Every reflected-area score whose coordinates come from one seam region.

    2026-08-30: a mesh seam is an outline, not the chart it encloses.  Its
    one-pixel strokes make intersection-over-union overly sensitive to a
    harmless raster step or a missing seam texel.  Fill its enclosed area and
    score the percentage of that individual area which survives reflection.
    """
    if int(region.sum()) < minimum_area_px:
        return ()
    # A tight region touches every side of its crop.  Padding gives
    # ``binary_fill_holes`` a real outside, then the original crop coordinates
    # remain the geometry coordinates used for the guide.
    reflected_area = ndimage.binary_fill_holes(np.pad(region, 1))[1:-1, 1:-1]
    return _reflected_area_axis_candidates(reflected_area, origin, minimum_area_px)


def _reflected_area_axis_candidates(
    area_mask: np.ndarray,
    origin: tuple[float, float],
    minimum_area_px: int,
    axis_search_ratio: float = DEFAULT_RHD_CONFIG.island_axis_search_ratio,
) -> tuple[tuple[float, tuple[tuple[float, float], tuple[float, float]]], ...]:
    """Measure both axes as reflected overlap divided by this area's size."""
    height, width = area_mask.shape
    area = int(area_mask.sum())
    if area < minimum_area_px:
        return ()
    def area_similarity(axis: str, max_shift: int) -> tuple[float, int]:
        """Best reflected overlap as a fraction of this chart's own area."""
        index = 1 if axis == "horizontal" else 0
        limit = max(int(max_shift), 0)
        padding = [(0, 0), (0, 0)]
        padding[index] = (limit, limit)
        padded = np.pad(area_mask, padding)
        flipped = np.flip(padded, axis=index)
        best_score, best_shift = -1.0, 0
        for shift in sorted(range(-limit, limit + 1),
                            key=lambda value: (abs(value), value)):
            overlap = int((padded & np.roll(flipped, shift, axis=index)).sum())
            score = overlap / area
            if score > best_score:
                best_score, best_shift = score, shift
        return best_score, best_shift

    x_offset, y_offset = origin
    axes: list[tuple[float, tuple[tuple[float, float], tuple[float, float]]]] = []
    for axis, extent in (("horizontal", width), ("vertical", height)):
        score, shift = area_similarity(
            axis, round(extent * max(float(axis_search_ratio), 0.0)),
        )
        if axis == "horizontal":
            centre = x_offset + (extent - 1 + shift) / 2.0
            axes.append((score, ((centre, y_offset), (centre, y_offset + height - 1))))
        else:
            centre = y_offset + (height - 1 + shift) / 2.0
            axes.append((score, ((x_offset, centre), (x_offset + width - 1, centre))))
    return tuple(axes)


def _source_seam_area_mask(
    segments: tuple[tuple[tuple[float, float], tuple[float, float]], ...],
    size: tuple[int, int],
)-> tuple[np.ndarray, tuple[float, float]] | None:
    """Fill one raw source seam island locally, before UV tile clipping."""
    if not segments:
        return None
    width, height = size
    points = np.asarray(segments, dtype=np.float64).reshape(-1, 2)
    if len(points) < 4:
        return None
    pixel_points = points.copy()
    pixel_points[:, 0] *= max(width - 1, 1)
    pixel_points[:, 1] = (1.0 - pixel_points[:, 1]) * max(height - 1, 1)
    lower = np.floor(pixel_points.min(axis=0)).astype(int) - 1
    upper = np.ceil(pixel_points.max(axis=0)).astype(int) + 1
    local_width, local_height = (upper - lower + 1).tolist()
    if local_width <= 0 or local_height <= 0:
        return None
    outline = Image.new("1", (local_width, local_height), 0)
    draw = ImageDraw.Draw(outline)
    for first, second in np.asarray(segments, dtype=np.float64):
        mapped = np.asarray((first, second), dtype=np.float64)
        mapped[:, 0] *= max(width - 1, 1)
        mapped[:, 1] = (1.0 - mapped[:, 1]) * max(height - 1, 1)
        draw.line(
            [tuple(round(value) for value in point - lower) for point in mapped],
            fill=1,
            width=1,
        )
    area_mask = ndimage.binary_fill_holes(
        np.pad(np.asarray(outline, dtype=bool), 1),
    )[1:-1, 1:-1]
    return area_mask, (float(lower[0]), float(lower[1]))


def source_seam_axis_measurements(
    segments: tuple[tuple[tuple[float, float], tuple[float, float]], ...],
    size: tuple[int, int],
    minimum_area_px: int = 4,
    axis_search_ratio: float = DEFAULT_RHD_CONFIG.island_axis_search_ratio,
) -> tuple[tuple[int, float, tuple[tuple[float, float], tuple[float, float]]], ...]:
    """Measure reflected overlap for one source seam island before clipping.

    2026-08-30: a chart straddling U=0 or V=0 is one authored seam region.
    Rasterising it into the displayed tile first splits that region in two and
    understates its reflected area.  Rasterise the raw source segments locally
    instead, then return the resulting axis in atlas-pixel coordinates.
    """
    result = _source_seam_area_mask(segments, size)
    if result is None:
        return ()
    area_mask, origin = result
    area = int(area_mask.sum())
    return tuple(
        (area, score, axis)
        for score, axis in _reflected_area_axis_candidates(
            area_mask, origin, minimum_area_px, axis_search_ratio,
        )
    )


def source_seam_axis_candidates(
    segments: tuple[tuple[tuple[float, float], tuple[float, float]], ...],
    size: tuple[int, int],
    minimum_area_px: int = 4,
    axis_search_ratio: float = DEFAULT_RHD_CONFIG.island_axis_search_ratio,
) -> tuple[tuple[float, tuple[tuple[float, float], tuple[float, float]]], ...]:
    """The score-and-axis view of :func:`source_seam_axis_measurements`."""
    return tuple(
        (score, axis) for _area, score, axis in source_seam_axis_measurements(
            segments, size, minimum_area_px, axis_search_ratio,
        )
    )


def source_frame_boundary_coverage(
    segments: tuple[tuple[tuple[float, float], tuple[float, float]], ...],
    frame: tuple[tuple[float, float], ...],
    size: tuple[int, int],
    max_gap_px: float,
) -> float:
    """How much of an authored seam actually follows its fitted frame."""
    width, height = size
    source = np.asarray(segments, dtype=np.float64).reshape(-1, 2)
    fitted = np.asarray(frame, dtype=np.float64).reshape(-1, 2)
    points = np.concatenate((source, fitted), axis=0)
    points[:, 0] *= max(width - 1, 1)
    points[:, 1] = (1.0 - points[:, 1]) * max(height - 1, 1)
    lower = np.floor(points.min(axis=0)).astype(int) - int(np.ceil(max_gap_px)) - 2
    upper = np.ceil(points.max(axis=0)).astype(int) + int(np.ceil(max_gap_px)) + 2
    local_width, local_height = (upper - lower + 1).tolist()
    if local_width <= 0 or local_height <= 0:
        return 0.0
    outline = Image.new("1", (local_width, local_height), 0)
    frame_outline = Image.new("1", (local_width, local_height), 0)
    draw = ImageDraw.Draw(outline)
    frame_draw = ImageDraw.Draw(frame_outline)

    def pixels(raw: np.ndarray) -> list[tuple[int, int]]:
        mapped = raw.copy()
        mapped[:, 0] *= max(width - 1, 1)
        mapped[:, 1] = (1.0 - mapped[:, 1]) * max(height - 1, 1)
        return [tuple(round(value) for value in point - lower) for point in mapped]

    for first, second in np.asarray(segments, dtype=np.float64):
        draw.line(pixels(np.asarray((first, second))), fill=1, width=1)
    frame_draw.line(pixels(np.asarray((*frame, frame[0]))), fill=1, width=1)
    observed = np.asarray(outline, dtype=bool)
    expected = np.asarray(frame_outline, dtype=bool)
    if not observed.any() or not expected.any():
        return 0.0
    distance = ndimage.distance_transform_edt(~expected)
    return float((distance[observed] <= max(float(max_gap_px), 0.0)).mean())


def source_seam_guides(
    segments: tuple[tuple[tuple[float, float], tuple[float, float]], ...],
    size: tuple[int, int],
    config: SimpleUvGeometryConfig = DEFAULT_SIMPLE_UV_GEOMETRY_CONFIG,
) -> tuple[tuple[str, object, float], ...]:
    """Classify one authored seam boundary using only tunable source evidence."""
    max_unmirrored = config.seam_symmetry_max_unmirrored_area_px
    axes = tuple(
        (score, axis) for area, score, axis in source_seam_axis_measurements(
            segments, size, config.seam_symmetry_min_area_px,
            config.seam_symmetry_axis_search_ratio,
        )
        if (score >= config.seam_symmetry_min_score
            and (max_unmirrored <= 0
                 or round(area * (1.0 - score)) <= max_unmirrored))
    )
    if axes:
        return tuple(("symmetric", axis, score) for score, axis in axes)
    frame = fit_source_seam_parallelogram(segments, size, config)
    if frame is None:
        return ()
    coverage = source_frame_boundary_coverage(
        segments, frame, size, config.parallelogram_max_boundary_gap_px,
    )
    if coverage < config.parallelogram_min_boundary_coverage:
        return ()
    wraps = any(
        coordinate < 0.0 or coordinate > 1.0
        for point in frame for coordinate in point
    )
    if wraps:
        return (("parallelogram_segments", wrapped_frame_segments(frame, size), coverage),)
    width, height = size
    corners = tuple(
        (point[0] * (width - 1), (1.0 - point[1]) * (height - 1))
        for point in frame
    )
    return (("parallelogram", corners, coverage),)


def wrapped_axis_copies(
    axis: tuple[tuple[float, float], tuple[float, float]],
    size: tuple[int, int],
) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    """Translate an unwrapped source-UV axis into every visible atlas tile."""
    width, height = size
    points = np.asarray(axis, dtype=np.float64)
    minimum, maximum = points.min(axis=0), points.max(axis=0)
    x_shifts = range(
        math.ceil(-maximum[0] / width),
        math.floor((width - 1 - minimum[0]) / width) + 1,
    )
    y_shifts = range(
        math.ceil(-maximum[1] / height),
        math.floor((height - 1 - minimum[1]) / height) + 1,
    )
    copies = []
    for shift_x in x_shifts:
        for shift_y in y_shifts:
            shifted = points + (shift_x * width, shift_y * height)
            copies.append(tuple(tuple(float(value) for value in point) for point in shifted))
    return tuple(copies)


def seam_region_axes(
    region: np.ndarray,
    origin: tuple[int, int],
    minimum_score: float = DEFAULT_RHD_CONFIG.min_island_symmetry,
    minimum_area_px: int = DEFAULT_RHD_CONFIG.simple_uv_min_area_px,
) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    """Reflection axes above the requested score, in one seam region."""
    return tuple(
        axis for score, axis in seam_region_axis_candidates(
            region, origin, minimum_area_px,
        )
        if score >= minimum_score
    )


def seam_region_symmetry_candidates(
    domains: np.ndarray,
    seams: np.ndarray,
    minimum_area_px: int = 4,
) -> tuple[tuple[int, int, float, tuple[tuple[float, float], tuple[float, float]]], ...]:
    """All individually-scored seam axes, including candidates below default."""
    seam_labels, _count = ndimage.label(
        np.asarray(seams, dtype=bool), structure=np.ones((3, 3), dtype=np.uint8),
    )
    candidates: list[tuple[int, int, float, tuple[tuple[float, float], tuple[float, float]]]] = []
    for seam_label, window in enumerate(ndimage.find_objects(seam_labels), start=1):
        if window is None:
            continue
        rows, columns = window
        region = seam_labels[rows, columns] == seam_label
        nearby = ndimage.binary_dilation(region, structure=np.ones((3, 3), dtype=bool))
        labels = domains[rows, columns][nearby]
        labels = labels[labels > 0]
        if len(labels):
            values, counts = np.unique(labels, return_counts=True)
            label = int(values[np.argmax(counts)])
        else:
            # 2026-08-30: source seam geometry can be useful without a packed
            # domain label, particularly where a chart wraps a tile edge.
            label = 0
        area = int(region.sum())
        candidates.extend(
            (label, area, score, axis)
            for score, axis in seam_region_axis_candidates(
                region, (columns.start, rows.start), minimum_area_px,
            )
        )
    return tuple(candidates)


def source_seam_symmetry_candidates(
    domains: np.ndarray,
    region: np.ndarray,
    segments: tuple[tuple[tuple[float, float], tuple[float, float]], ...],
    size: tuple[int, int],
    minimum_area_px: int = 4,
) -> tuple[tuple[int, int, float, tuple[tuple[float, float], tuple[float, float]]], ...]:
    """Every scored axis from one authored source seam island, ungrouped."""
    nearby = ndimage.binary_dilation(region, structure=np.ones((3, 3), dtype=bool))
    labels = domains[nearby]
    labels = labels[labels > 0]
    if len(labels):
        values, counts = np.unique(labels, return_counts=True)
        label = int(values[np.argmax(counts)])
    else:
        # 2026-08-30: wrapped source charts can have no in-tile domain label.
        label = 0
    candidates = source_seam_axis_candidates(segments, size, minimum_area_px)
    if not candidates:
        return ()
    raw_area = int(region.sum())
    return tuple(
        (label, raw_area, score, axis) for score, axis in candidates
    )


def seam_region_geometry(
    domains: np.ndarray,
    seams: np.ndarray,
    config: SimpleUvGeometryConfig = DEFAULT_SIMPLE_UV_GEOMETRY_CONFIG,
) -> tuple[tuple[int, str, tuple[tuple[float, float], ...], float | None], ...]:
    """Classify each seam component as a cyan axis or orange frame."""
    seam_labels, _count = ndimage.label(
        np.asarray(seams, dtype=bool), structure=np.ones((3, 3), dtype=np.uint8),
    )
    geometry: list[tuple[int, str, tuple[tuple[float, float], ...], float | None]] = []
    for seam_label, window in enumerate(ndimage.find_objects(seam_labels), start=1):
        if window is None:
            continue
        rows, columns = window
        region = seam_labels[rows, columns] == seam_label
        nearby = ndimage.binary_dilation(region, structure=np.ones((3, 3), dtype=bool))
        labels = domains[rows, columns][nearby]
        labels = labels[labels > 0]
        if not len(labels):
            continue
        values, counts = np.unique(labels, return_counts=True)
        label = int(values[np.argmax(counts)])
        axes = seam_region_axis_candidates(
            region, (columns.start, rows.start), config.seam_symmetry_min_area_px,
        )
        axes = tuple(
            (score, axis) for score, axis in axes
            if score >= config.seam_symmetry_min_score
        )
        if axes:
            geometry.extend(
                (label, "symmetric", axis, score) for score, axis in axes
            )
            continue
        corners = fit_seam_parallelogram(
            region,
            config.seam_parallelogram_opposite_edge_dot,
            (columns.start, rows.start),
            config.seam_corner_tolerance_px,
            config.parallelogram_min_edge_px,
        )
        if corners is not None:
            geometry.append((label, "parallelogram", corners, None))
    return tuple(geometry)
