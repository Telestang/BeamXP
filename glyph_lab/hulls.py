"""Convex hulls, with no dependency beyond numpy.

Split out of ``uv_geometry`` on 2026-09-02.  Both live under ``glyph_lab``, but
``uv_geometry`` reads three defaults off ``mirror_texture_for_rhd`` -- which is
fine for a viewer overlay and fatal for the detector, because the export path
now imports the detector and the two would close a cycle.  Hulls have nothing
to do with either module's configuration, so they go where anything can reach
them.
"""

from __future__ import annotations

import numpy as np

# Raster staircase vertices sit within half a texel of the edge they belong to;
# a real chart corner does not.
SEAM_CORNER_TOLERANCE_PX = 0.5


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Return the monotonic-chain hull without adding an OpenCV dependency."""
    unique = sorted({(float(x), float(y)) for x, y in np.asarray(points).reshape(-1, 2)})
    if len(unique) < 3:
        return np.asarray(unique, dtype=np.float64)

    def cross(origin, first, second) -> float:
        return ((first[0] - origin[0]) * (second[1] - origin[1])
                - (first[1] - origin[1]) * (second[0] - origin[0]))

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def simplified_hull(
    hull: np.ndarray, tolerance_px: float = SEAM_CORNER_TOLERANCE_PX,
) -> np.ndarray:
    """Remove raster staircase vertices without erasing a real chart corner."""
    points = np.asarray(hull, dtype=np.float64).copy()
    while len(points) > 3:
        distances: list[float] = []
        for index, point in enumerate(points):
            first, second = points[index - 1], points[(index + 1) % len(points)]
            edge = second - first
            length = float(np.linalg.norm(edge))
            distance = (
                abs(edge[0] * (point[1] - first[1])
                    - edge[1] * (point[0] - first[0])) / length
                if length > 1e-6 else 0.0
            )
            distances.append(distance)
        candidate = int(np.argmin(distances))
        if distances[candidate] > tolerance_px:
            break
        points = np.delete(points, candidate, axis=0)
    return points
