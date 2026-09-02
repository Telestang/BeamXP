"""Carrying a detected mark's own texels alongside its bounding rectangle.

A region has always been a rectangle here, and every merge, dedupe and rescale
in the export path moves rectangles.  A detector that knows which texels inside
that rectangle are actually the mark can say two more things worth having:

* which texels a flip should *write* -- the mark and the ground it reflects
  onto, rather than everything the rectangle happens to enclose; and
* which texels a flip could *tear* -- so a rectangle overhanging its UV island
  stops counting the overhang against the mark it contains.

A mask is always stored local to its region's bounds, never atlas-wide: a 4096
atlas is 16.8 MB as bool, and a plan carries dozens of steps.  ``None`` means
"the whole rectangle", which is what a detector that reports only rectangles
produces, and every operation here preserves it -- so a plan built without
masks behaves exactly as it did before they existed.

2026-09-02: written for the glyph_lab detector changeover.
"""

from __future__ import annotations

import numpy as np

Bounds = tuple[int, int, int, int]


def paint_region_mask(
    target: np.ndarray,
    bounds: Bounds,
    mask: np.ndarray | None,
) -> bool:
    """Paint one region into an atlas-frame scratch buffer.

    ``target`` is cleared first, so one buffer serves a whole plan.  Returns
    whether anything was painted; a region clipped entirely off the atlas
    paints nothing and must not then be flipped through an empty stencil.
    """
    target[:] = False
    height, width = target.shape[:2]
    x, y, w, h = (int(value) for value in bounds)
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, width), min(y + h, height)
    if x1 <= x0 or y1 <= y0:
        return False
    if mask is None:
        target[y0:y1, x0:x1] = True
        return True
    local = np.asarray(mask, dtype=bool)
    if local.shape != (h, w):
        raise ValueError(
            f"region mask {local.shape} does not match its bounds {(h, w)}"
        )
    target[y0:y1, x0:x1] = local[y0 - y:y1 - y, x0 - x:x1 - x]
    return bool(target[y0:y1, x0:x1].any())


def region_mask_in_bounds(mask: np.ndarray | None, bounds: Bounds) -> np.ndarray:
    """The mask as a dense array, materialising ``None`` as the full rectangle."""
    _x, _y, w, h = (int(value) for value in bounds)
    if mask is None:
        return np.ones((h, w), dtype=bool)
    local = np.asarray(mask, dtype=bool)
    if local.shape != (h, w):
        raise ValueError(
            f"region mask {local.shape} does not match its bounds {(h, w)}"
        )
    return local


def union_region_masks(
    union: Bounds,
    parts: list[tuple[Bounds, np.ndarray | None]],
) -> np.ndarray | None:
    """Combine the masks of regions being merged into one union rectangle.

    ``None`` where every part is a plain rectangle, because the union of
    rectangles is a rectangle and the merged region should stay one -- that is
    what keeps a maskless plan bit-identical to the plan it was before.  Where
    any part carries texels, a part that does not contributes its whole
    rectangle, which is the honest reading of "this whole box is the mark".
    """
    if all(mask is None for _bounds, mask in parts):
        return None
    ux, uy, uw, uh = (int(value) for value in union)
    combined = np.zeros((uh, uw), dtype=bool)
    for bounds, mask in parts:
        x, y, w, h = (int(value) for value in bounds)
        local = region_mask_in_bounds(mask, bounds)
        # Clip to the union, which contains every part by construction -- but
        # a caller passing a union that does not is a bug worth surviving
        # rather than an IndexError halfway through an export.
        x0, y0 = max(x, ux), max(y, uy)
        x1, y1 = min(x + w, ux + uw), min(y + h, uy + uh)
        if x1 <= x0 or y1 <= y0:
            continue
        combined[y0 - uy:y1 - uy, x0 - ux:x1 - ux] |= local[
            y0 - y:y1 - y, x0 - x:x1 - x
        ]
    return combined


def scale_region_mask(
    mask: np.ndarray | None,
    bounds: Bounds,
    scaled_bounds: Bounds,
    divisor: int,
    offset: int,
) -> np.ndarray | None:
    """Restate a mask at a companion map's resolution.

    Subsampled from the middle of each block, matching how ``rescale_plan``
    shrinks the mirror and label images, so a mask and the stencils it is
    intersected with keep describing the same texels.  The result is then
    resized to the rescaled bounds: those come from rounding both edges
    independently and can differ by a texel from the subsample's own extent.

    A mask that subsamples to nothing -- a hairline stroke on a half-resolution
    map -- falls back to its rectangle rather than to an empty stencil, because
    an empty stencil silently declines the flip while the rectangle is what the
    export would have used before masks existed.
    """
    if mask is None:
        return None
    x, y, _w, _h = (int(value) for value in bounds)
    _sx, _sy, scaled_w, scaled_h = (int(value) for value in scaled_bounds)
    local = region_mask_in_bounds(mask, bounds)
    # The subsample is taken in atlas coordinates so the retained texels are
    # the same ones the shrunken mirror mask retains, then trimmed to the mask.
    first_row = (offset - y) % divisor
    first_column = (offset - x) % divisor
    shrunk = local[first_row::divisor, first_column::divisor]
    if shrunk.size == 0 or not bool(shrunk.any()):
        return None
    resized = np.zeros((scaled_h, scaled_w), dtype=bool)
    rows = min(shrunk.shape[0], scaled_h)
    columns = min(shrunk.shape[1], scaled_w)
    resized[:rows, :columns] = shrunk[:rows, :columns]
    if not bool(resized.any()):
        return None
    return resized
