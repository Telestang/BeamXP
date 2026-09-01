"""Shared final-pixel padding for colour and normal-map regions."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from glyph_lab.regions import NEIGHBOURHOOD, Region


def pad_regions(
    regions: list[Region], padding_px: int, allowed: np.ndarray,
    offset: tuple[int, int],
) -> list[Region]:
    """Grow regions into allowed texels, retaining distinct-region ownership."""
    padding = max(int(padding_px), 0)
    if not regions or padding == 0:
        return regions
    height, width = allowed.shape
    x_offset, y_offset = offset
    prepared: list[tuple[Region, np.ndarray, int, int, np.ndarray]] = []
    coverage = np.zeros((height, width), dtype=np.uint16)
    originals = np.zeros((height, width), dtype=bool)
    for region in regions:
        left = max(region.x - x_offset - padding, 0)
        top = max(region.y - y_offset - padding, 0)
        right = min(region.x - x_offset + region.width + padding, width)
        bottom = min(region.y - y_offset + region.height + padding, height)
        if right <= left or bottom <= top:
            continue
        source = np.zeros((bottom - top, right - left), dtype=bool)
        source_top = region.y - y_offset - top
        source_left = region.x - x_offset - left
        source[
            source_top:source_top + region.height,
            source_left:source_left + region.width,
        ] = region.mask
        grown = ndimage.binary_dilation(
            source, structure=NEIGHBOURHOOD, iterations=padding,
        ) & allowed[top:bottom, left:right]
        coverage[top:bottom, left:right] += grown
        originals[top:bottom, left:right] |= source
        prepared.append((region, grown, left, top, source))

    output: list[Region] = []
    for region, grown, left, top, source in prepared:
        rows = slice(top, top + grown.shape[0])
        columns = slice(left, left + grown.shape[1])
        # 2026-09-01: padding is presentation margin, not a late grouping
        # rule.  Keep each original owner and drop contested added texels, so
        # two close marks cannot silently become one region here.
        foreign_original = originals[rows, columns] & ~source
        padded = grown & ~foreign_original
        padded &= (coverage[rows, columns] <= 1) | source
        output.append(Region(
            (x_offset + left, y_offset + top, padded.shape[1], padded.shape[0]),
            padded, region.domain,
        ))
    return output
