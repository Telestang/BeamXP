"""Baseline-aligned rectangular envelopes for grouped normal regions."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from glyph_lab.hulls import convex_hull
from glyph_lab.regions import Region


def longest_straight_hull_chain_axis(
    points: np.ndarray, tolerance_degrees: float, min_axis_lead_ratio: float,
) -> np.ndarray | None:
    """Find the longest roughly straight *run* around a grouped mark's hull."""
    hull = convex_hull(points)
    if len(hull) < 3:
        return None
    edges = np.roll(hull, -1, axis=0) - hull
    lengths = np.linalg.norm(edges, axis=1)
    # A text baseline can be split into several nearly collinear steps by its
    # terminals, while a single side edge may be longer than any one step.
    tolerance = max(min(float(tolerance_degrees), 89.0), 0.0)
    minimum_dot = float(np.cos(np.deg2rad(tolerance)))
    runs: list[tuple[float, np.ndarray]] = []
    edge_count = len(edges)
    for start in range(edge_count):
        first_length = float(lengths[start])
        if first_length <= 1e-9:
            continue
        axis = edges[start] / first_length
        run_length = 0.0
        for offset in range(edge_count):
            index = (start + offset) % edge_count
            length = float(lengths[index])
            if length <= 1e-9:
                continue
            direction = edges[index] / length
            if float(direction @ axis) < minimum_dot:
                break
            run_length += length
            span = hull[(index + 1) % edge_count] - hull[start]
            span_length = float(np.linalg.norm(span))
            if span_length <= 1e-9:
                continue
            runs.append((run_length, span / span_length))
    if not runs:
        return None
    best_length, best_axis = max(runs, key=lambda item: item[0])
    # 2026-09-01: opposite sides of a text envelope are useful corroboration,
    # not competing choices, so orientation is compared without direction.
    # A circle, diamond, or other round control has similarly strong runs at
    # different angles; it has no defensible text baseline and stays as-is.
    distinct_orientation_dot = float(np.cos(np.deg2rad(max(15.0, tolerance * 2.0))))
    competing_length = max((
        length for length, axis in runs
        if abs(float(axis @ best_axis)) < distinct_orientation_dot
    ), default=0.0)
    if (competing_length > 0.0
            and best_length < competing_length * max(float(min_axis_lead_ratio), 1.0)):
        return None
    return best_axis


def region_rectangle_corners(
    region: Region, tolerance_degrees: float, min_axis_lead_ratio: float,
) -> np.ndarray | None:
    """The four corners of a region's baseline-aligned rectangle, or None.

    In coordinates local to ``region.bounds``.  Split out of
    ``fitted_rectangle`` so the shape can be asked for without rasterising it:
    the export path mirrors about these corners directly, and drawing them into
    a mask only to fit a rectangle back around it would lose the angle that is
    the whole point of having them.  2026-09-02.
    """
    rows, columns = np.nonzero(region.mask)
    if rows.size == 0:
        return None
    points = np.column_stack((columns, rows)).astype(float)
    axis_u = longest_straight_hull_chain_axis(
        points, tolerance_degrees, min_axis_lead_ratio,
    )
    if axis_u is None:
        return None
    axis_v = np.array((-axis_u[1], axis_u[0]))
    # Fit pixel squares, rather than their centres: the result is an envelope
    # even at the one-pixel terminals of a relief glyph.
    pixel_corners = points[:, None, :] + np.asarray((
        (-0.5, -0.5), (-0.5, 0.5), (0.5, -0.5), (0.5, 0.5),
    ))
    pixel_corners = pixel_corners.reshape(-1, 2)
    along = pixel_corners @ axis_u
    across = pixel_corners @ axis_v
    u0, u1 = float(along.min()), float(along.max())
    v0, v1 = float(across.min()), float(across.max())
    return np.asarray((
        axis_u * u0 + axis_v * v0,
        axis_u * u1 + axis_v * v0,
        axis_u * u1 + axis_v * v1,
        axis_u * u0 + axis_v * v1,
    ))


def atlas_rectangle_corners(
    region: Region, tolerance_degrees: float, min_axis_lead_ratio: float,
) -> tuple[tuple[float, float], ...] | None:
    """``region_rectangle_corners`` moved into atlas coordinates."""
    corners = region_rectangle_corners(
        region, tolerance_degrees, min_axis_lead_ratio,
    )
    if corners is None:
        return None
    return tuple(
        (float(x) + region.x, float(y) + region.y) for x, y in corners
    )


def fitted_rectangle(
    region: Region, tolerance_degrees: float, min_axis_lead_ratio: float,
) -> Region:
    """Fit one grouped region to the rectangle aligned with its longest edge.

    Text and engraved legends usually contribute one long near-straight hull
    edge: their baseline.  The least-area orientation instead compromises
    between every serif and terminal.  Use that strongest edge as the local
    horizontal axis, and take the full normal extent perpendicular to it.
    """
    corners = region_rectangle_corners(
        region, tolerance_degrees, min_axis_lead_ratio,
    )
    if corners is None:
        return region
    local_left, local_top = np.floor(corners.min(axis=0)).astype(int)
    local_right, local_bottom = np.ceil(corners.max(axis=0)).astype(int)
    width, height = int(local_right - local_left + 1), int(local_bottom - local_top + 1)
    image = Image.new("1", (width, height))
    ImageDraw.Draw(image).polygon(
        [tuple(point - (local_left, local_top)) for point in corners], fill=1,
    )
    fitted = np.asarray(image, dtype=bool).copy()
    source_rows = slice(-local_top, -local_top + region.height)
    source_columns = slice(-local_left, -local_left + region.width)
    fitted[source_rows, source_columns] |= region.mask
    return Region(
        (region.x + int(local_left), region.y + int(local_top), width, height),
        fitted, region.domain,
    )


def fit_region_rectangles(
    regions: list[Region], tolerance_degrees: float, min_axis_lead_ratio: float,
) -> list[Region]:
    """Replace every grouped region with its baseline-aligned rectangle."""
    return [
        fitted_rectangle(region, tolerance_degrees, min_axis_lead_ratio)
        for region in regions
    ]
