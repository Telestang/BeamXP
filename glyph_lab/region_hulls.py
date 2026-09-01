"""Convex envelopes for final mark regions."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from glyph_lab.regions import Region, enclosed_voids
from glyph_lab.uv_geometry import convex_hull


def convex_region_mask(region: Region) -> np.ndarray:
    """Rasterise the convex hull of one region in its own tight bounds."""
    rows, columns = np.nonzero(region.mask)
    if len(rows) < 3:
        return region.mask.copy()
    hull = convex_hull(np.column_stack((columns, rows)))
    if len(hull) < 3:
        return region.mask.copy()
    image = Image.new("1", (region.width, region.height))
    ImageDraw.Draw(image).polygon(
        [tuple(point) for point in hull], fill=1,
    )
    return np.asarray(image, dtype=bool).copy()


def _swallowed_region_indexes(regions: list[Region]) -> set[int]:
    """Return marks wholly enclosed by another original mark or its hull."""
    swallowed: set[int] = set()
    for owner_index, owner in enumerate(regions):
        holes = enclosed_voids(owner.mask)
        hull = convex_region_mask(owner)
        for candidate_index, candidate in enumerate(regions):
            if candidate_index == owner_index or candidate.domain != owner.domain:
                continue
            left = candidate.x - owner.x
            top = candidate.y - owner.y
            right = left + candidate.width
            bottom = top + candidate.height
            if left < 0 or top < 0 or right > owner.width or bottom > owner.height:
                continue
            candidate_hole = holes[top:bottom, left:right]
            candidate_hull = hull[top:bottom, left:right]
            # 2026-09-01: The strict size check prevents identical coincident
            # masks from mutually deleting each other.  A contained mark is
            # ownership of the enclosing hull, not a foreign mark to carve
            # back out.
            hull_contains = (
                int(hull.sum()) > candidate.texels
                and candidate.mask.any()
                and candidate_hull[candidate.mask].all()
            )
            if candidate.mask.any() and (candidate_hole[candidate.mask].all()
                                         or hull_contains):
                swallowed.add(candidate_index)
    return swallowed


def convex_region_hulls(regions: list[Region]) -> list[Region]:
    """Wrap every mark in its convex hull without consuming separate marks.

    Regions have already been grouped when this runs, so their original masks
    state ownership.  A mark fully enclosed by another's original boundary or
    convex hull is absorbed.  Otherwise, a texel another surviving region owns
    is removed from a hull defensively.
    """
    swallowed = _swallowed_region_indexes(regions)
    regions = [region for index, region in enumerate(regions) if index not in swallowed]
    if not regions:
        return []
    x0 = min(region.x for region in regions)
    y0 = min(region.y for region in regions)
    x1 = max(region.x + region.width for region in regions)
    y1 = max(region.y + region.height for region in regions)
    occupied = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    for region in regions:
        rows = slice(region.y - y0, region.y - y0 + region.height)
        columns = slice(region.x - x0, region.x - x0 + region.width)
        occupied[rows, columns] |= region.mask

    output: list[Region] = []
    for region in regions:
        hull = convex_region_mask(region)
        rows = slice(region.y - y0, region.y - y0 + region.height)
        columns = slice(region.x - x0, region.x - x0 + region.width)
        foreign = occupied[rows, columns] & ~region.mask
        hull &= ~foreign
        output.append(Region(region.bounds, hull, region.domain))
    return output
