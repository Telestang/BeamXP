"""Fit a parallelogram to a feature's convex hull, and say how well it fits.

The shape a region is mirrored about has to be a shape the mirror preserves.
An axis-aligned box, a rotated rectangle and a parallelogram are the same
family seen at three levels of freedom, so one fit decides all three: fit the
most general shape, measure how much of it the hull actually supports, then
degrade to the cheapest member the evidence justifies.  Trying the cheap
shapes first cannot work -- a rotated rectangle fits a sheared glyph badly and
still passes a fill test, so the shear is never discovered.

Which shape comes out is read from how much of each side the *feature*
supports.  Two inputs, and they cannot be the same one: the hull fixes the
shape, because it is what the parallelogram encloses, but the feature is what
the sides are judged against.  Every side of an enclosing parallelogram is a
supporting line, so a side laid along a hull edge is flush with the hull by
construction and would score perfectly however meaningless that edge is -- and
they are meaningless often, because a region is frequently a group of separate
marks whose hull is bounded by chords leaping from one mark to another across
empty space.  Measured against the feature instead, a text baseline runs flush
for most of its length, the cap line above it falls away either side of the
one capital, and a bridging chord is not believed at all.

Pure numpy.  This module is on the path that has to outlive the OpenCV
dependency, so the hull arrives as points and everything after that is
projection and a 2x2 solve.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "ParallelogramFit",
    "fit_parallelogram",
    "measure_parallelogram",
    "pad_outline",
    "squared_outline",
    "enclosing_parallelogram",
    "hull_edge_directions",
]


@dataclass(frozen=True, slots=True)
class ParallelogramFit:
    """One parallelogram around a hull, with the evidence for each of its sides.

    ``corners`` cycle so that side *i* runs from ``corners[i]`` to
    ``corners[(i + 1) % 4]``.  Sides 0 and 2 are the pair parallel to
    ``directions[0]``; sides 1 and 3 the pair parallel to ``directions[1]``.
    """

    corners: tuple[tuple[float, float], ...]
    directions: tuple[tuple[float, float], tuple[float, float]]
    area: float
    hull_area: float
    # Root-mean-square perpendicular gap between each side and the feature
    # facing it, in pixels, in side order.
    side_gap_px: tuple[float, float, float, float]
    # ``(1 - gap / height) * coverage`` per side, clamped to [0, 1]: how far
    # the feature actually supports that side.  1 is flush along its whole
    # length.  The two factors catch different failures -- a side the feature
    # hugs over only a third of its length, and one it reaches everywhere but
    # never closely.
    side_evidence: tuple[float, float, float, float]

    @property
    def fill(self) -> float:
        """Hull area over parallelogram area.

        Deliberately not the feature-texels-over-area that ``FeatureShape``
        measures: that mixes in how solid the mark is, and a hollow outlined
        icon would score badly on it while being a perfect parallelogram.
        """
        return self.hull_area / max(self.area, 1e-9)

    @property
    def direction_evidence(self) -> tuple[float, float]:
        """Evidence per direction, taking the stronger of that direction's sides.

        A text baseline determines the angle whether or not the cap line above
        it agrees -- that is the whole 'one long straight edge, ragged
        elsewhere' case, and averaging the pair would throw it away.
        """
        return (
            max(self.side_evidence[0], self.side_evidence[2]),
            max(self.side_evidence[1], self.side_evidence[3]),
        )

    @property
    def interior_angle_degrees(self) -> float:
        """Angle between the two side directions, in [0, 90]."""
        first, second = self.directions
        cross = abs(first[0] * second[1] - first[1] * second[0])
        dot = abs(first[0] * second[0] + first[1] * second[1])
        return math.degrees(math.atan2(cross, dot))


def _unit(vector) -> tuple[float, float] | None:
    norm = math.hypot(float(vector[0]), float(vector[1]))
    if norm < 1e-9:
        return None
    return (float(vector[0]) / norm, float(vector[1]) / norm)


def hull_edge_directions(
    hull: np.ndarray,
    min_edge_px: float = 1.0,
    angle_tolerance_degrees: float = 1.0,
) -> list[tuple[tuple[float, float], float]]:
    """Distinct hull edge directions, longest edge first.

    Directions are taken modulo 180 degrees, because a side and the side
    opposite it describe the same direction, and near-duplicates are dropped so
    the pair search does not evaluate the same parallelogram repeatedly.

    Seeding the search from hull edges rather than sweeping angles is not only
    cheaper: the minimum-area enclosing parallelogram of a convex polygon has
    at least two sides flush with hull edges, so nothing is lost, and a free
    angular sweep can shave area by leaning a degree off a real straight edge
    -- which then reads as weak evidence on the very side that mattered.
    """
    points = np.asarray(hull, dtype=float).reshape(-1, 2)
    if len(points) < 3:
        return []
    edges = np.roll(points, -1, axis=0) - points
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    tolerance = math.radians(angle_tolerance_degrees)
    kept: list[tuple[float, float]] = []
    for index in np.argsort(lengths)[::-1]:
        length = float(lengths[index])
        if length < min_edge_px:
            break  # sorted, so nothing after this is long enough either
        direction = _unit(edges[index])
        if direction is None:
            continue
        angle = math.atan2(direction[1], direction[0]) % math.pi
        if any(
            min(abs(angle - other), math.pi - abs(angle - other)) < tolerance
            for other, _ in kept
        ):
            continue
        kept.append((angle, length))
    return [((math.cos(angle), math.sin(angle)), length) for angle, length in kept]


def enclosing_parallelogram(
    points: np.ndarray,
    first: tuple[float, float],
    second: tuple[float, float],
) -> tuple[tuple[float, float], ...] | None:
    """Smallest parallelogram with sides along the two directions given.

    Once the directions are chosen the shape is forced: all four sides are
    supporting lines, so they are the extremes of the hull projected onto the
    two side normals.
    """
    cross = first[0] * second[1] - first[1] * second[0]
    if abs(cross) < 1e-6:
        return None
    normal_first = np.array((-first[1], first[0]), dtype=float)
    normal_second = np.array((-second[1], second[0]), dtype=float)
    along_first = points @ normal_first
    along_second = points @ normal_second
    low_first, high_first = float(along_first.min()), float(along_first.max())
    low_second, high_second = float(along_second.min()), float(along_second.max())
    inverse = np.linalg.inv(np.array([normal_first, normal_second], dtype=float))
    # Adjacent corners differ in exactly one projection, so this order cycles.
    return tuple(
        tuple(float(value) for value in inverse @ np.array(pair, dtype=float))
        for pair in (
            (low_first, low_second),
            (low_first, high_second),
            (high_first, high_second),
            (high_first, low_second),
        )
    )


def _gap_profile(
    along: np.ndarray, height: np.ndarray, length: float
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest feature height per unit step along a side.

    Deliberately not the lower *convex* chain.  Asking the convex hull whether
    the hull's own edges are real is circular: every side of an enclosing
    parallelogram lies on a supporting line, so a side laid along a hull edge
    has zero gap by construction and scores perfectly however meaningless that
    edge is.

    It is meaningless often.  A region is frequently a *group* of separate
    marks -- a column of switch legends, a stack of warning icons -- and the
    hull of a group is bounded by chords that leap from one mark's corner to
    another's, across space containing nothing.  Those chords are what turn a
    stack of centred rows into a fitted diamond.  Measured against the feature
    the same chord reads as what it is: the marks fall away from it in the
    middle, so the gap opens up and the side is not believed.
    """
    steps = max(int(round(length)), 1)
    index = np.clip((along / max(length, 1e-9) * steps).astype(np.int64), 0, steps - 1)
    inside = (along >= -0.5) & (along <= length + 0.5)
    profile = np.full(steps, np.inf, dtype=float)
    if bool(inside.any()):
        np.minimum.at(profile, index[inside], height[inside])
    occupied = np.isfinite(profile)
    return profile, occupied


def _side_support(
    points: np.ndarray,
    origin,
    direction: tuple[float, float],
    normal: tuple[float, float],
    length: float,
) -> tuple[float, float]:
    """How well the feature supports one side: its RMS gap, and its coverage.

    The gap is measured perpendicular to the side rather than to the nearest
    feature point: nearest-point distance rounds off at the corners, which
    would stop a flush baseline reading as flush.

    Coverage is reported separately because the two failures are different.  A
    side the feature runs along but only over a third of its length is a real
    edge of a small part of the region; a side the feature reaches everywhere
    but never closely is not an edge at all.
    """
    if length <= 1e-9:
        return 0.0, 0.0
    local = points - np.asarray(origin, dtype=float)
    along = local @ np.asarray(direction, dtype=float)
    height = local @ np.asarray(normal, dtype=float)
    profile, occupied = _gap_profile(along, height, length)
    if not bool(occupied.any()):
        return float("inf"), 0.0
    gaps = profile[occupied]
    return float(np.sqrt(float(np.mean(gaps * gaps)))), float(occupied.mean())


def _polygon_area(corners) -> float:
    points = np.asarray(corners, dtype=float).reshape(-1, 2)
    rolled = np.roll(points, -1, axis=0)
    return float(
        abs(float(np.sum(points[:, 0] * rolled[:, 1] - rolled[:, 0] * points[:, 1])))
        / 2.0
    )


def _densified(polygon: np.ndarray, step_px: float = 1.0) -> np.ndarray:
    """Walk a polygon's outline at roughly one point per pixel.

    Support is measured in per-pixel bins along each side, so a bare vertex
    list leaves almost every bin empty and reads as no support at all.  This is
    only for the case where a hull has to stand in for its own feature.
    """
    walked: list[np.ndarray] = []
    for index in range(len(polygon)):
        start = polygon[index]
        end = polygon[(index + 1) % len(polygon)]
        steps = max(int(np.hypot(*(end - start)) / max(step_px, 1e-6)), 1)
        walked.append(
            start + (end - start) * np.linspace(0.0, 1.0, steps, endpoint=False)[:, None]
        )
    return np.concatenate(walked, axis=0) if walked else polygon


def measure_parallelogram(
    hull: np.ndarray,
    corners: tuple[tuple[float, float], ...],
    directions: tuple[tuple[float, float], tuple[float, float]],
    feature: np.ndarray | None = None,
) -> ParallelogramFit:
    """Measure how well the feature supports each side of its parallelogram.

    ``hull`` fixes the shape -- it is what the parallelogram encloses -- while
    ``feature`` is what the sides are judged against.  They must be different
    inputs; see ``_gap_profile`` for why the hull cannot judge itself.  With no
    feature given the hull stands in for it, which is only honest for a region
    holding one solid mark.
    """
    points = np.asarray(hull, dtype=float).reshape(-1, 2)
    support = (
        _densified(points)
        if feature is None
        else np.asarray(feature, dtype=float).reshape(-1, 2)
    )
    coverages: list[float] = []
    gaps: list[float] = []
    heights: list[float] = []
    for index in range(4):
        start = np.asarray(corners[index], dtype=float)
        end = np.asarray(corners[(index + 1) % 4], dtype=float)
        edge = end - start
        length = float(np.hypot(edge[0], edge[1]))
        direction = _unit(edge)
        if direction is None:
            gaps.append(0.0)
            heights.append(1.0)
            continue
        # Inward normal: the opposite corner has to lie on the positive side.
        normal = (-direction[1], direction[0])
        opposite = np.asarray(corners[(index + 2) % 4], dtype=float) - start
        reach = float(opposite[0] * normal[0] + opposite[1] * normal[1])
        if reach < 0.0:
            normal = (direction[1], -direction[0])
            reach = -reach
        # Scale the gap against the smaller of the side's own length and its
        # reach to the opposite side.  Judging it purely by the reach flatters
        # the short pair of an elongated mark: the left and right sides of a
        # word are a fifth of its length, so a sag that consumes half of one
        # still reads as a few per cent of the width and passes as flush.
        heights.append(min(abs(reach), length))
        gap, coverage = _side_support(support, start, direction, normal, length)
        gaps.append(gap)
        coverages.append(coverage)
    evidence = tuple(
        float(min(max(1.0 - gap / max(height, 1e-9), 0.0), 1.0) * coverage)
        for gap, height, coverage in zip(gaps, heights, coverages)
    )
    return ParallelogramFit(
        corners=tuple(tuple(float(v) for v in corner) for corner in corners),
        directions=directions,
        area=_polygon_area(corners),
        hull_area=_polygon_area(points),
        side_gap_px=tuple(min(gap, 1e9) for gap in gaps),  # type: ignore[arg-type]
        side_evidence=evidence,  # type: ignore[arg-type]
    )


def squared_outline(
    hull: np.ndarray,
    direction: tuple[float, float],
) -> tuple[tuple[float, float], ...] | None:
    """The enclosing rectangle whose sides run along ``direction``.

    The direction comes from the fit that measured it, so a rectangle is the
    parallelogram with its second direction squared off rather than a separate
    shape found a separate way.  One fit, one axis, whichever member of the
    family the evidence ends up justifying.
    """
    perpendicular = (-direction[1], direction[0])
    return enclosing_parallelogram(
        np.asarray(hull, dtype=float).reshape(-1, 2), direction, perpendicular
    )


def pad_outline(
    corners,
    padding_px: float,
    bounds: tuple[int, int] | None = None,
) -> tuple[tuple[float, float], ...]:
    """Grow a fitted outline outwards by a perpendicular margin on every side.

    Works for a rectangle and a parallelogram alike, because it is expressed in
    the outline's own edge coordinates: the corners move along the two edge
    directions, so the shape keeps its angle and its centre and simply gets
    bigger.  Clipping the corners to the image instead would change the angle,
    which for a sheared outline is the one thing that must not happen.

    Where the margin would leave the image it is reduced rather than clipped,
    for the same reason the circle case reduces it: a region that grew by
    different amounts on different sides is no longer the shape that was
    measured.
    """
    points = np.asarray(corners, dtype=float).reshape(-1, 2)
    if len(points) != 4 or padding_px <= 0.0:
        return tuple(tuple(float(v) for v in point) for point in points)
    origin = points[0]
    first = points[1] - points[0]
    second = points[2] - points[1]
    cross = abs(first[0] * second[1] - first[1] * second[0])
    if cross <= 1e-9:
        return tuple(tuple(float(v) for v in point) for point in points)

    def grown(margin: float) -> np.ndarray:
        # A margin of ``margin`` texels perpendicular to the sides costs this
        # much of each edge coordinate, which is what keeps the growth square
        # to the outline rather than to the image.
        along = margin * float(np.hypot(*second)) / cross
        across = margin * float(np.hypot(*first)) / cross
        return np.asarray(
            [
                origin - along * first - across * second,
                origin + (1.0 + along) * first - across * second,
                origin + (1.0 + along) * first + (1.0 + across) * second,
                origin - along * first + (1.0 + across) * second,
            ]
        )

    margin = float(padding_px)
    if bounds is not None:
        height, width = bounds
        base = grown(0.0)
        step = grown(1.0) - base
        limits = [margin]
        for axis, extent in ((0, width - 1), (1, height - 1)):
            for corner in range(4):
                direction = step[corner, axis]
                if abs(direction) <= 1e-9:
                    continue
                start = base[corner, axis]
                room = (0.0 - start) if direction < 0 else (extent - start)
                limits.append(max(room / direction, 0.0))
        margin = max(min(limits), 0.0)
    return tuple(tuple(float(v) for v in point) for point in grown(margin))


def fit_parallelogram(
    hull: np.ndarray,
    feature: np.ndarray | None = None,
    min_edge_px: float = 2.0,
    area_tie_ratio: float = 0.02,
) -> ParallelogramFit | None:
    """Fit the parallelogram the hull's own edges support.

    Minimum area picks the candidate, but only to within ``area_tie_ratio``:
    among parallelograms that enclose the hull equally tightly, the one whose
    sides the feature actually lies along is the one that describes the mark.

    The tie-break must not prefer the squarer candidate, tempting as that is.
    It decides the fitted angle, not just the shape's name, and a squarer fit
    of equal area is not a more likely one -- measured on a 400x160 bar sheared
    by 0.30, preferring rectangularity returned 75.96 degrees for a true 73.30,
    and that angle is what the mirror is applied about.  Whether the lean is
    believed at all is decided later, by the snap threshold, against how well
    evidenced the directions are.
        """
    points = np.asarray(hull, dtype=float).reshape(-1, 2)
    if len(points) < 3:
        return None
    directions = hull_edge_directions(points, min_edge_px)
    if len(directions) < 2:
        return None

    candidates: list[tuple[float, tuple, tuple]] = []
    for index, (first, _length) in enumerate(directions):
        for second, _other in directions[index + 1 :]:
            corners = enclosing_parallelogram(points, first, second)
            if corners is None:
                continue
            candidates.append((_polygon_area(corners), corners, (first, second)))
    if not candidates:
        return None

    smallest = min(area for area, _corners, _pair in candidates)
    limit = smallest * (1.0 + max(area_tie_ratio, 0.0))
    best: ParallelogramFit | None = None
    best_score: float | None = None
    for area, corners, pair in candidates:
        if area > limit:
            continue
        fit = measure_parallelogram(points, corners, pair, feature)
        # Every side, not ``direction_evidence``.  That takes the stronger of
        # each pair, which is what the snap decision wants but the wrong thing
        # to fit by: it rewards a parallelogram that leans to hug one
        # incidental stroke -- a capital A, a 4, a 7 -- and ignores the
        # opposite side falling away from nothing.
        score = sum(fit.side_evidence)
        if best_score is None or score > best_score:
            best, best_score = fit, score
    return best
