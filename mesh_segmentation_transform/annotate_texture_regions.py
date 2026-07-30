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

Optional, for DDS input only:
    Pillow          (PIL)

Usage:
    python annotate_texture_regions.py scintilla_gauges.png
    python annotate_texture_regions.py scintilla_interior_b.color.DDS -o annotated.png

Tune detection behaviour by editing DEFAULT_CONFIG near the top of this file.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Optional DDS support via Pillow
# ---------------------------------------------------------------------------

try:
    from PIL import Image as _PILImage

    _HAS_PILLOW = True
except ImportError:
    _HAS_PILLOW = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MserConfig:
    """Tuneable MSER and grouping parameters.

    Defaults target BeamNG dashboard and interior labels on flat or
    near-flat colour backgrounds.
    """

    # MSER detector
    delta:         int   = 8  # default: 5
    min_area:      int   = 30  # default: 30
    max_area:      int   = 1_024  # default: 1_024
    max_variation: float = 0.25  # default: 0.25
    min_diversity: float = 0.2  # default: 0.2

    # Post-detection filtering
    enable_min_component_area_filter: bool  = True  # default: True
    min_component_area_px:            int   = 16  # default: 16
    enable_mser_area_fraction_filter: bool  = True  # default: True
    max_area_fraction:                float = 0.25  # default: 0.25
    enable_aspect_ratio_filter:       bool  = True  # default: True
    min_aspect:                       float = 0.05  # default: 0.05
    max_aspect:                       float = 40.0  # default: 40.0

    # Grouping
    merge_distance_px:              int  = 15  # default: 10
    group_dilate_px:                int  = 3  # default: 3
    require_union_region_group:     bool = True  # default: False
    min_group_union_region_px:      int  = 100  # default: 64
    enable_group_area_filter:       bool = True  # default: True
    enable_group_degenerate_filter: bool = True  # default: True

    # Final bounding-box padding around each group
    bbox_padding_px: int = 0  # default: 0

    # Final cleanup
    require_distinct_background:       bool  = True  # default: True

    require_continuous_border:         bool  = False  # default: True
    enable_border_repair:              bool  = False  # default: True
    enable_circular_border_repair:     bool  = False  # default: True
    border_var_thesh:                  int   = 28  # default: 8
    border_width_px:                   int   = 3  # default: 3
    min_border_colour_fraction:        float = 0.70  # default: 0.70
    max_border_gap_px:                 int   = 3  # default: 3
    border_adjust_inward_fraction:     float = 0.25  # default: 0.25
    border_adjust_outward_fraction:    float = 0.5  # default: 0.25
    border_adjust_coarse_step_px:      int   = 3  # default: 3
    border_adjust_refine_candidates:   int   = 12  # default: 12

    enable_background_ring_filter:     bool  = False  # default: True
    background_quant_step:             int   = 16  # default: 16
    background_ring_fraction:          float = 0.18  # default: 0.18
    min_background_ring_fraction:      float = 0.45  # default: 0.45

    enable_background_region_filter:   bool  = False  # default: True
    min_background_region_fraction:    float = 0.30  # default: 0.30

    enable_foreground_fraction_filter: bool  = False  # default: True
    foreground_distance_px:            int   = 38  # default: 38
    min_foreground_fraction:           float = 0.01  # default: 0.01
    max_foreground_fraction:           float = 0.70  # default: 0.70
    
    enable_background_edge_filter:     bool  = False  # default: True
    background_edge_threshold:         int   = 24  # default: 24
    max_background_edge_fraction:      float = 0.12  # default: 0.12

    # Annotation drawing
    green_colour:    tuple[int, int, int] = (0, 255, 0)  # default: (0, 255, 0)
    red_colour:      tuple[int, int, int] = (0, 0, 255)  # default: (0, 0, 255)
    green_thickness: int                  = 1  # default: 1
    red_thickness:   int                  = 1  # default: 1


# ---------------------------------------------------------------------------
# Active tuning preset
# ---------------------------------------------------------------------------
#
# Edit the MserConfig field values above while tuning.  DEFAULT_CONFIG is kept
# deliberately empty so those field values are the single source of truth.

DEFAULT_CONFIG = MserConfig()


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------


def load_image(path: Path) -> np.ndarray:
    """Load a texture as a BGR uint8 ndarray.

    PNG, JPG, BMP, TIFF and other formats supported by the local OpenCV
    build are read directly.  DDS requires Pillow; a clear error is raised when Pillow
    is unavailable.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Texture not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".dds":
        if not _HAS_PILLOW:
            raise RuntimeError(
                f"Reading DDS textures requires Pillow.  "
                f"Install it with:  python -m pip install Pillow\n"
                f"Alternatively, convert {path.name} to PNG first."
            )
        with _PILImage.open(path) as pil_image:
            rgba = pil_image.convert("RGBA")
        bgr = cv2.cvtColor(np.asarray(rgba, dtype=np.uint8), cv2.COLOR_RGBA2BGR)
        return bgr

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(
            f"OpenCV could not decode {path.name}.  "
            f"Use PNG, JPG, BMP, TIFF, DDS with Pillow, or another format "
            f"supported by your OpenCV build."
        )
    return image


# ---------------------------------------------------------------------------
# MSER detection and filtering
# ---------------------------------------------------------------------------


def detect_mser_boxes(
    grey: np.ndarray,
    config: MserConfig,
) -> np.ndarray:
    """Return filtered axis-aligned bounding boxes from MSER detection.

    Each row is ``[x, y, w, h]`` in pixel coordinates.  Boxes that fail
    the area, aspect-ratio, or image-fraction filters are discarded.
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

    image_area = grey.shape[0] * grey.shape[1]
    max_component_area = int(image_area * config.max_area_fraction)

    boxes: list[tuple[int, int, int, int]] = []
    for region in regions:
        x, y, w, h = cv2.boundingRect(region)
        area = w * h
        if config.enable_min_component_area_filter and area < config.min_component_area_px:
            continue
        if config.enable_mser_area_fraction_filter and area > max_component_area:
            continue
        aspect = max(w, h) / max(min(w, h), 1)
        if config.enable_aspect_ratio_filter and (
            aspect < config.min_aspect or aspect > config.max_aspect
        ):
            continue
        boxes.append((x, y, w, h))

    if not boxes:
        return np.empty((0, 4), dtype=np.int32)

    return np.asarray(boxes, dtype=np.int32)


# ---------------------------------------------------------------------------
# Spatial grouping via mask flood
# ---------------------------------------------------------------------------


def legacy_mask_group_boxes(
    boxes: np.ndarray,
    image_shape: tuple[int, int],
    config: MserConfig,
) -> list[tuple[int, int, int, int]]:
    """Merge nearby boxes with the previous expanded-mask connected components."""
    if len(boxes) == 0:
        return []

    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    margin = config.merge_distance_px
    for x, y, w, h in boxes:
        x0 = max(x - margin, 0)
        y0 = max(y - margin, 0)
        x1 = min(x + w + margin, width)
        y1 = min(y + h + margin, height)
        mask[y0:y1, x0:x1] = 255

    if config.group_dilate_px > 0:
        kernel_size = config.group_dilate_px * 2 + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_size, kernel_size)
        )
        mask = cv2.dilate(mask, kernel, iterations=1)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    pad = config.bbox_padding_px
    groups: list[tuple[int, int, int, int]] = []
    for label_id in range(1, num_labels):  # skip background (0)
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])

        # Reject groups that are essentially the whole image (background leak)
        if config.enable_group_area_filter and (
            w * h > image_shape[0] * image_shape[1] * config.max_area_fraction
        ):
            continue
        # Reject degenerate single-pixel noise groups
        if config.enable_group_degenerate_filter and (w < 4 or h < 4):
            continue

        x0 = max(x - pad, 0)
        y0 = max(y - pad, 0)
        x1 = min(x + w + pad, width)
        y1 = min(y + h + pad, height)
        groups.append((x0, y0, x1 - x0, y1 - y0))

    groups.sort(key=lambda g: (g[1], g[0]))
    return groups


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


def union_region_group_boxes(
    boxes: np.ndarray,
    image_shape: tuple[int, int],
    config: MserConfig,
) -> list[tuple[int, int, int, int]]:
    """Group boxes when their expanded union/contact region is large enough."""
    if len(boxes) == 0:
        return []

    height, width = image_shape[:2]
    distance = max(0, config.merge_distance_px + config.group_dilate_px)
    min_union_area = max(1, config.min_group_union_region_px)
    box_tuples = [tuple(int(value) for value in box) for box in boxes]
    expanded_boxes = [expanded_box(box, distance) for box in box_tuples]
    parent = list(range(len(box_tuples)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for i, source in enumerate(box_tuples):
        for j in range(i + 1, len(box_tuples)):
            if intersection_area(expanded_boxes[i], expanded_boxes[j]) >= min_union_area:
                union(i, j)

    components: dict[int, list[tuple[int, int, int, int]]] = {}
    for index, box in enumerate(box_tuples):
        components.setdefault(find(index), []).append(box)

    pad = config.bbox_padding_px
    group_expand = distance
    groups: list[tuple[int, int, int, int]] = []
    for component_boxes in components.values():
        x0 = max(min(x for x, _y, _w, _h in component_boxes) - group_expand - pad, 0)
        y0 = max(min(y for _x, y, _w, _h in component_boxes) - group_expand - pad, 0)
        x1 = min(
            max(x + w for x, _y, w, _h in component_boxes) + group_expand + pad,
            width,
        )
        y1 = min(
            max(y + h for _x, y, _w, h in component_boxes) + group_expand + pad,
            height,
        )
        w = x1 - x0
        h = y1 - y0

        if config.enable_group_area_filter and (
            w * h > image_shape[0] * image_shape[1] * config.max_area_fraction
        ):
            continue
        if config.enable_group_degenerate_filter and (w < 4 or h < 4):
            continue

        groups.append((x0, y0, w, h))

    groups.sort(key=lambda g: (g[1], g[0]))
    return groups


def group_boxes(
    boxes: np.ndarray,
    image_shape: tuple[int, int],
    config: MserConfig,
) -> list[tuple[int, int, int, int]]:
    """Merge bounding boxes into coherent text/symbol groups."""
    if config.require_union_region_group:
        return union_region_group_boxes(boxes, image_shape, config)
    return legacy_mask_group_boxes(boxes, image_shape, config)


# ---------------------------------------------------------------------------
# Final cleanup
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


def ordered_border_pixels(crop: np.ndarray, border_width: int) -> np.ndarray:
    """Return border pixels in perimeter order, outer ring first."""
    height, width = crop.shape[:2]
    rings: list[np.ndarray] = []

    for inset in range(border_width):
        top = inset
        left = inset
        bottom = height - 1 - inset
        right = width - 1 - inset
        if top > bottom or left > right:
            break

        if top == bottom:
            rings.append(crop[top, left : right + 1])
            continue
        if left == right:
            rings.append(crop[top : bottom + 1, left])
            continue

        rings.extend(
            (
                crop[top, left : right + 1],
                crop[top + 1 : bottom + 1, right],
                crop[bottom, left:right][::-1],
                crop[top + 1 : bottom, left][::-1],
            )
        )

    if not rings:
        return np.empty((0, 3), dtype=crop.dtype)
    return np.concatenate(rings, axis=0)


def fill_short_false_runs(matches: np.ndarray, max_gap: int) -> np.ndarray:
    """Bridge short mismatching gaps in a circular boolean sequence."""
    if len(matches) == 0 or max_gap <= 0:
        return matches

    filled = matches.copy()
    doubled = np.concatenate([matches, matches])
    length = len(matches)
    index = 0
    while index < len(doubled):
        if doubled[index]:
            index += 1
            continue

        start = index
        while index < len(doubled) and not doubled[index]:
            index += 1
        end = index

        if end - start <= max_gap:
            before = start > 0 and doubled[start - 1]
            after = end < len(doubled) and doubled[end]
            if before and after:
                for doubled_index in range(start, end):
                    filled[doubled_index % length] = True

    return filled


def longest_circular_true_run_fraction(matches: np.ndarray) -> float:
    """Return longest contiguous true run as a fraction of sequence length."""
    if len(matches) == 0:
        return 0.0
    if bool(matches.all()):
        return 1.0

    doubled = np.concatenate([matches, matches])
    best = 0
    current = 0
    for value in doubled:
        if value:
            current += 1
            best = min(max(best, current), len(matches))
        else:
            current = 0
    return best / len(matches)


def dominant_colour_matches(pixels: np.ndarray, quant_step: int) -> np.ndarray:
    """Return mask for pixels matching the dominant quantized BGR colour."""
    quant_step = max(quant_step, 1)
    quantized = (pixels // quant_step).astype(np.int32)
    dominant_bin = dominant_quantized_colour(pixels, quant_step)
    return np.all(quantized == dominant_bin, axis=1)


def dominant_quantized_colour(pixels: np.ndarray, quant_step: int) -> np.ndarray:
    """Return the dominant quantized BGR colour bin for a pixel collection."""
    quant_step = max(quant_step, 1)
    quantized = (pixels // quant_step).astype(np.int32)
    levels = (255 // quant_step) + 1
    if levels**3 > 1_000_000:
        bins, counts = np.unique(quantized.reshape(-1, 3), axis=0, return_counts=True)
        return bins[int(np.argmax(counts))]

    codes = (quantized[:, 0] * levels + quantized[:, 1]) * levels + quantized[:, 2]
    dominant_code = int(np.bincount(codes).argmax())
    b = dominant_code // (levels * levels)
    remainder = dominant_code % (levels * levels)
    g = remainder // levels
    r = remainder % levels
    return np.asarray((b, g, r), dtype=np.int32)


def quantized_colour_matches(
    pixels: np.ndarray,
    quantized_colour: np.ndarray,
    quant_step: int,
) -> np.ndarray:
    """Return mask for pixels matching one shared quantized BGR colour bin."""
    quant_step = max(quant_step, 1)
    quantized = (pixels // quant_step).astype(np.int32)
    return np.all(quantized == quantized_colour, axis=1)


def continuous_border_colour_fraction(
    image: np.ndarray,
    group: tuple[int, int, int, int],
    config: MserConfig,
) -> float:
    """Return the longest continuous dominant-colour run around the border."""
    clamped = clamp_group(group, image.shape)
    if clamped is None:
        return 0.0

    x, y, w, h = clamped
    crop = image[y : y + h, x : x + w]
    if crop.size == 0:
        return 0.0

    border_width = min(
        max(config.border_width_px, 1),
        max(1, min(w, h) // 2),
    )
    border_pixels = ordered_border_pixels(crop, border_width)
    if len(border_pixels) == 0:
        return 0.0

    matches = dominant_colour_matches(border_pixels, config.border_var_thesh)
    matches = fill_short_false_runs(matches, config.max_border_gap_px)
    return longest_circular_true_run_fraction(matches)


def has_continuous_border(
    image: np.ndarray,
    group: tuple[int, int, int, int],
    config: MserConfig,
) -> bool:
    """Return whether a rectangle border has one continuous dominant-colour run."""
    if not config.require_continuous_border:
        return True
    return (
        continuous_border_colour_fraction(image, group, config)
        >= config.min_border_colour_fraction
    )


def border_side_colour_fraction(
    image: np.ndarray,
    group: tuple[int, int, int, int],
    side: str,
    config: MserConfig,
    shared_colour: np.ndarray | None = None,
) -> float:
    """Return continuous shared-colour fraction for one rectangle side."""
    clamped = clamp_group(group, image.shape)
    if clamped is None:
        return 0.0

    x, y, w, h = clamped
    crop = image[y : y + h, x : x + w]
    if crop.size == 0:
        return 0.0

    border_width = min(
        max(config.border_width_px, 1),
        max(1, min(w, h) // 2),
    )
    if side == "left":
        pixels = crop[:, :border_width].reshape(-1, 3)
    elif side == "right":
        pixels = crop[:, -border_width:].reshape(-1, 3)
    elif side == "top":
        pixels = crop[:border_width, :].reshape(-1, 3)
    elif side == "bottom":
        pixels = crop[-border_width:, :].reshape(-1, 3)
    else:
        raise ValueError(f"Unknown border side: {side}")

    if len(pixels) == 0:
        return 0.0

    colour = (
        shared_colour
        if shared_colour is not None
        else dominant_quantized_colour(pixels, config.border_var_thesh)
    )
    matches = quantized_colour_matches(pixels, colour, config.border_var_thesh)
    matches = fill_short_false_runs(matches, config.max_border_gap_px)
    return longest_circular_true_run_fraction(matches)


def border_side_pixels(
    image: np.ndarray,
    group: tuple[int, int, int, int],
    side: str,
    config: MserConfig,
) -> np.ndarray:
    """Return pixels sampled from one candidate rectangle border side."""
    clamped = clamp_group(group, image.shape)
    if clamped is None:
        return np.empty((0, 3), dtype=image.dtype)

    x, y, w, h = clamped
    crop = image[y : y + h, x : x + w]
    if crop.size == 0:
        return np.empty((0, 3), dtype=image.dtype)

    border_width = min(
        max(config.border_width_px, 1),
        max(1, min(w, h) // 2),
    )
    if side == "left":
        return crop[:, :border_width].reshape(-1, 3)
    if side == "right":
        return crop[:, -border_width:].reshape(-1, 3)
    if side == "top":
        return crop[:border_width, :].reshape(-1, 3)
    if side == "bottom":
        return crop[-border_width:, :].reshape(-1, 3)
    raise ValueError(f"Unknown border side: {side}")


def border_sides_pass(
    image: np.ndarray,
    group: tuple[int, int, int, int],
    sides: tuple[str, ...],
    config: MserConfig,
) -> bool:
    """Return whether all requested sides share one continuous border colour."""
    side_pixels = [
        border_side_pixels(image, group, side, config)
        for side in sides
    ]
    if not side_pixels or any(len(pixels) == 0 for pixels in side_pixels):
        return False

    shared_colour = dominant_quantized_colour(
        np.concatenate(side_pixels, axis=0),
        config.border_var_thesh,
    )
    return all(
        border_side_colour_fraction(image, group, side, config, shared_colour)
        >= config.min_border_colour_fraction
        for side in sides
    )


def circular_border_pixels(
    image: np.ndarray,
    centre_x: float,
    centre_y: float,
    radius: int,
    border_width: int,
) -> np.ndarray:
    """Return ordered pixels sampled from a circular border annulus."""
    if radius < 1:
        return np.empty((0, 3), dtype=image.dtype)

    height, width = image.shape[:2]
    border_width = max(border_width, 1)
    sample_count = max(24, int(round(2 * np.pi * radius)))
    angles = np.linspace(0.0, 2 * np.pi, sample_count, endpoint=False)
    pixels: list[np.ndarray] = []

    for angle in angles:
        cos_a = float(np.cos(angle))
        sin_a = float(np.sin(angle))
        for radial_offset in range(border_width):
            sample_radius = max(0, radius - radial_offset)
            x = int(round(centre_x + cos_a * sample_radius))
            y = int(round(centre_y + sin_a * sample_radius))
            if 0 <= x < width and 0 <= y < height:
                pixels.append(image[y, x])

    if not pixels:
        return np.empty((0, 3), dtype=image.dtype)
    return np.asarray(pixels, dtype=image.dtype)


def circular_border_colour_fraction(
    image: np.ndarray,
    centre_x: float,
    centre_y: float,
    radius: int,
    config: MserConfig,
) -> float:
    """Return longest shared-colour run around a circular border."""
    pixels = circular_border_pixels(
        image,
        centre_x,
        centre_y,
        radius,
        config.border_width_px,
    )
    if len(pixels) == 0:
        return 0.0

    matches = dominant_colour_matches(pixels, config.border_var_thesh)
    matches = fill_short_false_runs(matches, config.max_border_gap_px)
    return longest_circular_true_run_fraction(matches)


def circle_group(
    centre_x: float,
    centre_y: float,
    radius: int,
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    """Return a clamped square group enclosing a circle."""
    x0 = int(round(centre_x - radius))
    y0 = int(round(centre_y - radius))
    diameter = int(round(radius * 2))
    return clamp_group((x0, y0, diameter, diameter), image_shape)


def circle_radius_values(
    base_radius: int,
    config: MserConfig,
) -> list[int]:
    """Return candidate radii spanning inward and outward circular search."""
    inward = max(0, int(round(base_radius * config.border_adjust_inward_fraction)))
    outward = max(0, int(round(base_radius * config.border_adjust_outward_fraction)))
    min_radius = max(2, base_radius - inward)
    max_radius = max(min_radius, base_radius + outward)
    return adjustment_values(min_radius, max_radius, config.border_adjust_coarse_step_px)


def circular_border_candidates(
    image: np.ndarray,
    group: tuple[int, int, int, int],
    config: MserConfig,
) -> list[tuple[int, int, int, int]]:
    """Return circular border candidates as bounding-square groups."""
    x, y, w, h = group
    centre_x = x + w / 2.0
    centre_y = y + h / 2.0
    base_radius = max(2, int(round(max(w, h) / 2.0)))

    coarse_radii = [
        radius
        for radius in circle_radius_values(base_radius, config)
        if circular_border_colour_fraction(image, centre_x, centre_y, radius, config)
        >= config.min_border_colour_fraction
    ]
    if not coarse_radii:
        return []

    refined_radii: set[int] = set()
    min_radius = min(circle_radius_values(base_radius, config))
    max_radius = max(circle_radius_values(base_radius, config))
    for radius in coarse_radii:
        refined_radii.update(
            refined_adjustment_values(
                radius,
                min_radius,
                max_radius,
                config.border_adjust_coarse_step_px,
            )
        )

    candidates: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for radius in sorted(refined_radii):
        if (
            circular_border_colour_fraction(image, centre_x, centre_y, radius, config)
            < config.min_border_colour_fraction
        ):
            continue
        candidate = circle_group(centre_x, centre_y, radius, image.shape)
        if candidate is None or candidate[2] < 4 or candidate[3] < 4:
            continue
        candidates.append((abs(radius - base_radius), candidate[2] * candidate[3], candidate))

    candidates.sort(key=lambda item: (item[0], item[1]))
    return [candidate for _distance, _area, candidate in candidates]


def adjustment_values(min_offset: int, max_offset: int, coarse_step: int) -> list[int]:
    """Return coarse offsets spanning the allowed adjustment range."""
    coarse_step = max(coarse_step, 1)
    if min_offset > max_offset:
        min_offset, max_offset = max_offset, min_offset
    values = set(range(min_offset, max_offset + 1, coarse_step))
    values.update({min_offset, 0, max_offset})
    return sorted(values)


def refined_adjustment_values(
    center: int,
    min_offset: int,
    max_offset: int,
    coarse_step: int,
) -> list[int]:
    """Return 1 px offsets around one coarse adjustment."""
    radius = max(coarse_step, 1) - 1
    if min_offset > max_offset:
        min_offset, max_offset = max_offset, min_offset
    start = max(min_offset, center - radius)
    end = min(max_offset, center + radius)
    values = set(range(start, end + 1))
    values.add(center)
    return sorted(values)


def edge_adjustment_bounds(
    side: str,
    base_size: int,
    config: MserConfig,
) -> tuple[int, int]:
    """Return signed offset bounds for one edge.

    Positive offsets move left/top inward but right/bottom outward.
    """
    inward = max(0, int(round(base_size * config.border_adjust_inward_fraction)))
    outward = max(0, int(round(base_size * config.border_adjust_outward_fraction)))
    if side in {"left", "top"}:
        return -outward, inward
    if side in {"right", "bottom"}:
        return -inward, outward
    raise ValueError(f"Unknown border side: {side}")


def adjust_edge(
    group: tuple[int, int, int, int],
    side: str,
    offset: int,
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    """Return a group with one edge moved by offset pixels."""
    x, y, w, h = group
    x0 = x
    y0 = y
    x1 = x + w
    y1 = y + h

    if side == "left":
        x0 = x + offset
    elif side == "right":
        x1 = x + w + offset
    elif side == "top":
        y0 = y + offset
    elif side == "bottom":
        y1 = y + h + offset
    else:
        raise ValueError(f"Unknown border side: {side}")

    return clamp_group((x0, y0, x1 - x0, y1 - y0), image_shape)


def passing_edge_offsets(
    image: np.ndarray,
    group: tuple[int, int, int, int],
    side: str,
    offsets: list[int],
    config: MserConfig,
) -> list[int]:
    """Return offsets where one adjusted edge has a continuous colour."""
    passing: list[int] = []
    for offset in offsets:
        candidate = adjust_edge(group, side, offset, image.shape)
        if candidate is None or candidate[2] < 4 or candidate[3] < 4:
            continue
        if border_sides_pass(image, candidate, (side,), config):
            passing.append(offset)
    return passing


def ranked_offset_pairs(
    start_offsets: list[int],
    end_offsets: list[int],
    base_size: int,
    max_pairs: int,
) -> list[tuple[int, int]]:
    """Return edge-offset pairs ranked by smallest resulting size."""
    pairs: list[tuple[int, int, int]] = []
    for start_offset in start_offsets:
        for end_offset in end_offsets:
            size = base_size - start_offset + end_offset
            if size >= 4:
                pairs.append((size, start_offset, end_offset))

    pairs.sort(key=lambda item: item[0])
    return [(start, end) for _, start, end in pairs[: max(max_pairs, 1)]]


def combine_horizontal_offsets(
    group: tuple[int, int, int, int],
    left_offset: int,
    right_offset: int,
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    """Return a group with left and right edges adjusted together."""
    x, y, w, h = group
    return clamp_group(
        (x + left_offset, y, w - left_offset + right_offset, h),
        image_shape,
    )


def combine_vertical_offsets(
    group: tuple[int, int, int, int],
    top_offset: int,
    bottom_offset: int,
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    """Return a group with top and bottom edges adjusted together."""
    x, y, w, h = group
    return clamp_group(
        (x, y + top_offset, w, h - top_offset + bottom_offset),
        image_shape,
    )


def width_candidates_to_vertical_border(
    image: np.ndarray,
    group: tuple[int, int, int, int],
    config: MserConfig,
) -> list[tuple[int, int, int, int]]:
    """Return width-adjusted candidates whose vertical borders are continuous."""
    _, _, w, _ = group
    left_min, left_max = edge_adjustment_bounds("left", w, config)
    right_min, right_max = edge_adjustment_bounds("right", w, config)

    left_coarse = passing_edge_offsets(
        image,
        group,
        "left",
        adjustment_values(left_min, left_max, config.border_adjust_coarse_step_px),
        config,
    )
    right_coarse = passing_edge_offsets(
        image,
        group,
        "right",
        adjustment_values(right_min, right_max, config.border_adjust_coarse_step_px),
        config,
    )
    coarse_pairs = ranked_offset_pairs(
        left_coarse,
        right_coarse,
        w,
        config.border_adjust_refine_candidates,
    )
    if not coarse_pairs:
        return []

    left_offsets: set[int] = set()
    right_offsets: set[int] = set()
    for left_center, right_center in coarse_pairs:
        left_offsets.update(
            refined_adjustment_values(
                left_center,
                left_min,
                left_max,
                config.border_adjust_coarse_step_px,
            )
        )
        right_offsets.update(
            refined_adjustment_values(
                right_center,
                right_min,
                right_max,
                config.border_adjust_coarse_step_px,
            )
        )

    left_refined = passing_edge_offsets(
        image,
        group,
        "left",
        sorted(left_offsets),
        config,
    )
    right_refined = passing_edge_offsets(
        image,
        group,
        "right",
        sorted(right_offsets),
        config,
    )

    refined_candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for left_offset, right_offset in ranked_offset_pairs(
        left_refined,
        right_refined,
        w,
        config.border_adjust_refine_candidates,
    ):
        candidate = combine_horizontal_offsets(
            group,
            left_offset,
            right_offset,
            image.shape,
        )
        if candidate is None or candidate[2] < 4:
            continue
        if border_sides_pass(image, candidate, ("left", "right"), config):
            refined_candidates.append((candidate[2] * candidate[3], candidate))

    refined_candidates.sort(key=lambda item: item[0])
    return [candidate for _, candidate in refined_candidates]


def adjust_width_to_vertical_border(
    image: np.ndarray,
    group: tuple[int, int, int, int],
    config: MserConfig,
) -> tuple[int, int, int, int] | None:
    """Adjust left/right edges until both vertical borders are continuous."""
    candidates = width_candidates_to_vertical_border(image, group, config)
    if not candidates:
        return None
    return candidates[0]


def height_candidates_to_horizontal_border(
    image: np.ndarray,
    group: tuple[int, int, int, int],
    config: MserConfig,
) -> list[tuple[int, int, int, int]]:
    """Return height-adjusted candidates whose horizontal borders are continuous."""
    _, _, _, h = group
    top_min, top_max = edge_adjustment_bounds("top", h, config)
    bottom_min, bottom_max = edge_adjustment_bounds("bottom", h, config)

    top_coarse = passing_edge_offsets(
        image,
        group,
        "top",
        adjustment_values(top_min, top_max, config.border_adjust_coarse_step_px),
        config,
    )
    bottom_coarse = passing_edge_offsets(
        image,
        group,
        "bottom",
        adjustment_values(bottom_min, bottom_max, config.border_adjust_coarse_step_px),
        config,
    )
    coarse_pairs = ranked_offset_pairs(
        top_coarse,
        bottom_coarse,
        h,
        config.border_adjust_refine_candidates,
    )
    if not coarse_pairs:
        return []

    top_offsets: set[int] = set()
    bottom_offsets: set[int] = set()
    for top_center, bottom_center in coarse_pairs:
        top_offsets.update(
            refined_adjustment_values(
                top_center,
                top_min,
                top_max,
                config.border_adjust_coarse_step_px,
            )
        )
        bottom_offsets.update(
            refined_adjustment_values(
                bottom_center,
                bottom_min,
                bottom_max,
                config.border_adjust_coarse_step_px,
            )
        )

    top_refined = passing_edge_offsets(
        image,
        group,
        "top",
        sorted(top_offsets),
        config,
    )
    bottom_refined = passing_edge_offsets(
        image,
        group,
        "bottom",
        sorted(bottom_offsets),
        config,
    )

    refined_candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for top_offset, bottom_offset in ranked_offset_pairs(
        top_refined,
        bottom_refined,
        h,
        config.border_adjust_refine_candidates,
    ):
        candidate = combine_vertical_offsets(
            group,
            top_offset,
            bottom_offset,
            image.shape,
        )
        if candidate is None or candidate[3] < 4:
            continue
        if border_sides_pass(image, candidate, ("top", "bottom"), config):
            refined_candidates.append((candidate[2] * candidate[3], candidate))

    refined_candidates.sort(key=lambda item: item[0])
    return [candidate for _, candidate in refined_candidates]


def adjust_height_to_horizontal_border(
    image: np.ndarray,
    group: tuple[int, int, int, int],
    config: MserConfig,
) -> tuple[int, int, int, int] | None:
    """Adjust top/bottom edges until both horizontal borders are continuous."""
    candidates = height_candidates_to_horizontal_border(image, group, config)
    if not candidates:
        return None
    return candidates[0]


def adjust_group_to_continuous_border(
    image: np.ndarray,
    group: tuple[int, int, int, int],
    config: MserConfig,
    require_distinct_background: bool = False,
) -> tuple[int, int, int, int] | None:
    """Repair width then height to find a continuous-border candidate."""
    if has_continuous_border(image, group, config) and (
        not require_distinct_background
        or has_distinct_background(image, group, config)
    ):
        return group

    if not config.enable_border_repair:
        return None

    for width_adjusted in width_candidates_to_vertical_border(image, group, config):
        for height_adjusted in height_candidates_to_horizontal_border(
            image,
            width_adjusted,
            config,
        ):
            if not has_continuous_border(image, height_adjusted, config):
                continue
            if require_distinct_background and not has_distinct_background(
                image,
                height_adjusted,
                config,
            ):
                continue
            return height_adjusted

    if config.enable_circular_border_repair:
        for circle_adjusted in circular_border_candidates(image, group, config):
            if require_distinct_background and not has_distinct_background(
                image,
                circle_adjusted,
                config,
            ):
                continue
            return circle_adjusted

    return None


def has_distinct_background(
    image: np.ndarray,
    group: tuple[int, int, int, int],
    config: MserConfig,
) -> bool:
    """Return whether a group looks like markings on a stable background."""
    if not config.require_distinct_background:
        return True

    x, y, w, h = group
    if w <= 0 or h <= 0:
        return False

    crop = image[y : y + h, x : x + w]
    if crop.size == 0:
        return False

    ring_width = max(2, int(round(min(w, h) * config.background_ring_fraction)))
    ring_width = min(ring_width, max(1, min(w, h) // 2))

    ring_mask = np.zeros((h, w), dtype=bool)
    ring_mask[:ring_width, :] = True
    ring_mask[-ring_width:, :] = True
    ring_mask[:, :ring_width] = True
    ring_mask[:, -ring_width:] = True

    ring_pixels = crop[ring_mask]
    if len(ring_pixels) == 0:
        return False

    quant_step = max(config.background_quant_step, 1)
    quantized_ring = (ring_pixels // quant_step).astype(np.int16)
    bins, counts = np.unique(
        quantized_ring.reshape(-1, 3), axis=0, return_counts=True
    )
    dominant_index = int(np.argmax(counts))
    dominant_bin = bins[dominant_index]
    ring_fraction = float(counts[dominant_index]) / len(ring_pixels)
    if (
        config.enable_background_ring_filter
        and ring_fraction < config.min_background_ring_fraction
    ):
        return False

    quantized_crop = (crop // quant_step).astype(np.int16)
    background_mask = np.all(quantized_crop == dominant_bin, axis=2)
    background_fraction = float(background_mask.mean())
    if (
        config.enable_background_region_filter
        and background_fraction < config.min_background_region_fraction
    ):
        return False

    crop_int = crop.astype(np.int16)
    horizontal_delta = np.linalg.norm(
        crop_int[:, 1:, :] - crop_int[:, :-1, :], axis=2
    )
    vertical_delta = np.linalg.norm(
        crop_int[1:, :, :] - crop_int[:-1, :, :], axis=2
    )
    horizontal_ring = ring_mask[:, 1:] & ring_mask[:, :-1]
    vertical_ring = ring_mask[1:, :] & ring_mask[:-1, :]
    edge_threshold = max(config.background_edge_threshold, 1)
    edge_count = int((horizontal_delta[horizontal_ring] >= edge_threshold).sum())
    edge_count += int((vertical_delta[vertical_ring] >= edge_threshold).sum())
    edge_total = int(horizontal_ring.sum() + vertical_ring.sum())
    if edge_total > 0:
        edge_fraction = edge_count / edge_total
        if (
            config.enable_background_edge_filter
            and edge_fraction > config.max_background_edge_fraction
        ):
            return False

    if not config.enable_foreground_fraction_filter:
        return True

    background_pixels = crop[background_mask]
    if len(background_pixels) == 0:
        return False

    background_colour = np.median(background_pixels, axis=0).astype(np.float32)
    colour_distance = np.linalg.norm(
        crop.astype(np.float32) - background_colour, axis=2
    )
    foreground_fraction = float(
        (colour_distance >= config.foreground_distance_px).mean()
    )

    return (
        config.min_foreground_fraction
        <= foreground_fraction
        <= config.max_foreground_fraction
    )


def cleanup_groups_by_background(
    image: np.ndarray,
    groups: list[tuple[int, int, int, int]],
    config: MserConfig,
) -> tuple[list[tuple[int, int, int, int]], int]:
    """Apply border repair and background validation to completed groups."""
    if not config.require_distinct_background:
        return groups, 0

    cleaned_groups: list[tuple[int, int, int, int]] = []
    adjusted_count = 0
    for group in groups:
        candidate = adjust_group_to_continuous_border(
            image,
            group,
            config,
            require_distinct_background=True,
        )
        if candidate is None:
            continue
        cleaned_groups.append(candidate)
        if candidate != group:
            adjusted_count += 1

    return cleaned_groups, adjusted_count


def cleanup_groups(
    image: np.ndarray,
    groups: list[tuple[int, int, int, int]],
    config: MserConfig,
) -> tuple[list[tuple[int, int, int, int]], int]:
    """Apply final post-group cleanup."""
    return cleanup_groups_by_background(image, groups, config)


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


def annotate_texture(
    input_path: Path,
    output_path: Path,
    config: MserConfig | None = None,
) -> dict[str, object]:
    """Run the full detect → group → annotate pipeline and save the result.

    Returns a small summary dict suitable for printing or logging.
    """
    config = config or DEFAULT_CONFIG

    bgr = load_image(input_path)
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    boxes = detect_mser_boxes(grey, config)
    candidate_groups = group_boxes(boxes, grey.shape, config)
    groups, adjusted_groups = cleanup_groups(bgr, candidate_groups, config)
    annotated = draw_annotations(bgr, groups, config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        wrote_output = cv2.imwrite(str(output_path), annotated)
    except cv2.error as exc:
        raise ValueError(f"OpenCV could not write output image: {output_path}") from exc
    if not wrote_output:
        raise ValueError(f"OpenCV could not write output image: {output_path}")

    return {
        "input": str(input_path),
        "output": str(output_path),
        "image_size": f"{bgr.shape[1]}x{bgr.shape[0]}",
        "mser_boxes": len(boxes),
        "candidate_groups": len(candidate_groups),
        "background_filter_enabled": config.require_distinct_background,
        "background_rejected": len(candidate_groups) - len(groups),
        "cleanup_adjusted": adjusted_groups,
        "grouped_regions": len(groups),
        "groups": [
            {"x": x, "y": y, "w": w, "h": h} for x, y, w, h in groups
        ],
    }


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

    config = DEFAULT_CONFIG

    started = time.perf_counter()

    try:
        summary = annotate_texture(input_path, output_path, config)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    elapsed_ms = (time.perf_counter() - started) * 1000.0

    print(
        f"{summary['image_size']}  "
        f"{summary['mser_boxes']} MSER boxes → "
        f"{summary['grouped_regions']} annotated groups"
    )
    if summary["background_filter_enabled"]:
        print(
            f"{summary['candidate_groups']} candidate groups, "
            f"{summary['background_rejected']} rejected by background filter, "
            f"{summary['cleanup_adjusted']} adjusted"
        )
    else:
        print(f"{summary['candidate_groups']} candidate groups, background filter off")
    print(f"Wrote {summary['output']}")
    print(f"elapsed: {elapsed_ms:.1f} ms")

    for index, group in enumerate(summary["groups"], start=1):
        print(
            f"  [{index:3d}]  x={group['x']:5d}  y={group['y']:5d}  "
            f"w={group['w']:4d}  h={group['h']:4d}"
        )


if __name__ == "__main__":
    main()
