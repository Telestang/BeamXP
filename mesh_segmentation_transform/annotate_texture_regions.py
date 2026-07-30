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
    merge_distance_px:              int  = 10  # default: 10
    group_dilate_px:                int  = 8  # default: 3
    min_group_union_region_px:      int  = 169  # default: 64
    enable_group_area_filter:       bool = True  # default: True
    enable_group_degenerate_filter: bool = True  # default: True

    # Final bounding-box padding around each group
    bbox_padding_px: int = 0  # default: 0

    # UV/magic-wand post cleanup
    enable_uv_magic_wand_refine:       bool  = True  # default: False
    require_uv_magic_wand_refine:      bool  = True  # default: False
    uv_island_mask_path:               str   = "mesh_segmentation_transform/segmentation_outputs/scintilla_interior_b.color.full_uv_filled_mask.png"  # black UV islands on white background
    magic_wand_colour_thresh:          int   = 100  # default: 28
    magic_wand_seed_attempts:          int   = 32  # default: 64
    magic_wand_output_padding_px:      int   = 0  # default: 2
    magic_wand_min_island_area_px:     int   = 12  # default: 4
    magic_wand_noise_bg_std_weight:    float = 1.00  # default: 1.00
    magic_wand_noise_bg_std_scale:     float = 24.0  # default: 24.0
    magic_wand_noise_min_signal:       float = 8.0  # default: 8.0
    enable_magic_wand_noise_filter:    bool  = True  # default: False
    max_magic_wand_noise_score:        float = 1.0  # default: 0.50
    enable_final_size_filter:          bool  = True  # default: False
    final_min_width_px:                int   = 4  # default: 4
    final_min_height_px:               int   = 4  # default: 4
    final_min_area_px:                 int   = 36  # default: 36
    enable_final_single_colour_filter: bool  = True  # default: False
    final_single_colour_quant_step:    int   = 12  # default: 12
    final_single_colour_fraction:      float = 0.98  # default: 0.98
    enable_final_offset_background_filter: bool = True  # default: False
    final_offset_background_width_px:      int  = 2  # default: 2
    final_offset_background_colour_tol:    int  = 32  # default: 32
    final_offset_background_min_fraction:  float = 0.70  # default: 0.70

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


def load_uv_island_mask(path: Path, image_shape: tuple[int, int]) -> np.ndarray:
    """Load black-island-on-white UV mask as a boolean island mask."""
    mask_image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask_image is None:
        raise FileNotFoundError(f"UV island mask not found or unreadable: {path}")

    height, width = image_shape[:2]
    if mask_image.shape[:2] != (height, width):
        mask_image = cv2.resize(
            mask_image,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
    return mask_image < 128


def resolve_config_path(path_text: str) -> Path:
    """Resolve config paths relative to cwd first, then repo root."""
    path = Path(path_text)
    if path.is_absolute() or path.is_file():
        return path

    script_root = Path(__file__).resolve().parent
    repo_root = script_root.parent
    repo_relative = repo_root / path
    if repo_relative.is_file():
        return repo_relative

    script_relative = script_root / path
    if script_relative.is_file():
        return script_relative

    return repo_relative


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
    return union_region_group_boxes(boxes, image_shape, config)


# ---------------------------------------------------------------------------
# Geometry and colour helpers
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


def dominant_colour_matches(pixels: np.ndarray, quant_step: int) -> np.ndarray:
    """Return mask for pixels matching the dominant quantized BGR colour."""
    quant_step = max(quant_step, 1)
    quantized = (pixels // quant_step).astype(np.int32)
    dominant_bin = dominant_quantized_colour(pixels, quant_step)
    return np.all(quantized == dominant_bin, axis=1)


def estimate_background_colour(
    crop: np.ndarray,
    domain_mask: np.ndarray,
    colour_thresh: int,
) -> np.ndarray | None:
    """Estimate the dominant background BGR colour inside the current domain."""
    domain_pixels = crop[domain_mask]
    if len(domain_pixels) == 0:
        return None

    dominant_matches = dominant_colour_matches(
        domain_pixels,
        max(colour_thresh, 1),
    )
    if not bool(dominant_matches.any()):
        return None

    return np.median(domain_pixels[dominant_matches], axis=0).astype(np.int16)


def background_colour_mask(
    crop: np.ndarray,
    domain_mask: np.ndarray,
    background_colour: np.ndarray,
    colour_thresh: int,
) -> np.ndarray:
    """Return domain pixels close enough to the estimated background colour."""
    colour_delta = np.linalg.norm(crop.astype(np.int16) - background_colour, axis=2)
    return (colour_delta <= max(colour_thresh, 0)) & domain_mask


def magic_wand_noise_metrics(
    crop: np.ndarray,
    background_mask: np.ndarray,
    signal_mask: np.ndarray,
    config: MserConfig,
) -> dict[str, float]:
    """Return OpenCV-backed background-noise/glyph-contrast diagnostics."""
    if not bool(background_mask.any()) or not bool(signal_mask.any()):
        return {
            "score": float("inf"),
            "signal": 0.0,
            "background_std": 0.0,
        }

    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    background_u8 = background_mask.astype(np.uint8)
    bg_mean, bg_stddev = cv2.meanStdDev(lab, mask=background_u8)
    bg_mean_vec = bg_mean.reshape(3).astype(np.float32)
    bg_std = float(np.linalg.norm(bg_stddev.reshape(3)))

    signal_pixels = lab[signal_mask].astype(np.float32)
    signal_delta = np.linalg.norm(signal_pixels - bg_mean_vec, axis=1)
    signal = float(np.median(signal_delta)) if len(signal_delta) > 0 else 0.0

    if signal < config.magic_wand_noise_min_signal:
        score = float("inf")
    else:
        weighted_noise = bg_std * config.magic_wand_noise_bg_std_weight
        ratio_score = weighted_noise / max(signal, 0.001)
        background_penalty = bg_std / max(config.magic_wand_noise_bg_std_scale, 0.001)
        score = ratio_score + background_penalty

    return {
        "score": float(score),
        "signal": signal,
        "background_std": bg_std,
    }


def group_domain_mask(
    group: tuple[int, int, int, int],
    uv_island_mask: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """Return UV-island domain within a group crop."""
    clamped = clamp_group(group, uv_island_mask.shape)
    if clamped is None:
        return None
    x, y, w, h = clamped
    domain = uv_island_mask[y : y + h, x : x + w].copy()
    if not bool(domain.any()):
        return None
    return domain, clamped


def flood_background_component(
    crop: np.ndarray,
    domain_mask: np.ndarray,
    seed: tuple[int, int],
    background_colour: np.ndarray,
    colour_thresh: int,
) -> np.ndarray:
    """Flood fill estimated-background-colour pixels from a valid seed."""
    seed_x, seed_y = seed
    if not domain_mask[seed_y, seed_x]:
        return np.zeros(domain_mask.shape, dtype=bool)

    allowed = background_colour_mask(
        crop,
        domain_mask,
        background_colour,
        colour_thresh,
    )
    if not allowed[seed_y, seed_x]:
        return np.zeros(domain_mask.shape, dtype=bool)

    flood_input = allowed.astype(np.uint8) * 255
    mask = np.zeros((flood_input.shape[0] + 2, flood_input.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(flood_input, mask, seed, 128)
    return flood_input == 128


def feature_bbox_from_background(
    crop: np.ndarray,
    background_component: np.ndarray,
    domain_mask: np.ndarray,
    config: MserConfig,
) -> tuple[tuple[int, int, int, int], dict[str, float]] | None:
    """Return padded bbox around all non-background islands in the domain."""
    feature_mask = domain_mask & ~background_component
    if not bool(feature_mask.any()):
        return None

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        feature_mask.astype(np.uint8),
        connectivity=8,
    )
    if num_labels <= 1:
        return None

    x0: int | None = None
    y0: int | None = None
    x1: int | None = None
    y1: int | None = None
    height, width = domain_mask.shape[:2]
    min_area = max(config.magic_wand_min_island_area_px, 0)
    signal_mask = np.zeros(feature_mask.shape, dtype=bool)
    for label_id in range(1, num_labels):
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if x <= 0 or y <= 0 or x + w >= width or y + h >= height:
            continue
        if area < min_area:
            continue
        signal_mask[labels == label_id] = True
        x0 = x if x0 is None else min(x0, x)
        y0 = y if y0 is None else min(y0, y)
        x1 = x + w if x1 is None else max(x1, x + w)
        y1 = y + h if y1 is None else max(y1, y + h)

    if x0 is None or y0 is None or x1 is None or y1 is None:
        return None

    noise_metrics = magic_wand_noise_metrics(
        crop,
        background_component,
        signal_mask,
        config,
    )

    pad = max(config.magic_wand_output_padding_px, 0)
    out_x0 = max(x0 - pad, 0)
    out_y0 = max(y0 - pad, 0)
    out_x1 = min(x1 + pad, width)
    out_y1 = min(y1 + pad, height)
    if out_x1 <= out_x0 or out_y1 <= out_y0:
        return None
    return (out_x0, out_y0, out_x1 - out_x0, out_y1 - out_y0), noise_metrics


def refine_group_with_uv_magic_wand(
    image: np.ndarray,
    group: tuple[int, int, int, int],
    uv_island_mask: np.ndarray,
    config: MserConfig,
    rng: np.random.Generator,
) -> tuple[tuple[int, int, int, int], dict[str, float]] | None:
    """Refine a group using UV-domain constrained random-seed magic wand."""
    domain_result = group_domain_mask(group, uv_island_mask)
    if domain_result is None:
        return None

    domain_mask, clamped = domain_result
    x, y, w, h = clamped
    crop = image[y : y + h, x : x + w]
    background_colour = estimate_background_colour(
        crop,
        domain_mask,
        config.magic_wand_colour_thresh,
    )
    if background_colour is None:
        return None

    seed_mask = background_colour_mask(
        crop,
        domain_mask,
        background_colour,
        config.magic_wand_colour_thresh,
    )
    seed_points = np.argwhere(seed_mask)
    if len(seed_points) == 0:
        return None

    attempts = min(max(config.magic_wand_seed_attempts, 1), len(seed_points))
    seed_indices = rng.choice(len(seed_points), size=attempts, replace=False)
    best: tuple[
        tuple[int, int, int],
        tuple[int, int, int, int],
        dict[str, float],
    ] | None = None
    for seed_index in seed_indices:
        seed_y, seed_x = (int(value) for value in seed_points[int(seed_index)])
        background = flood_background_component(
            crop,
            domain_mask,
            (seed_x, seed_y),
            background_colour,
            config.magic_wand_colour_thresh,
        )
        if not bool(background.any()):
            continue
        bg_ys, bg_xs = np.where(background)
        bg_w = int(bg_xs.max() - bg_xs.min() + 1)
        bg_h = int(bg_ys.max() - bg_ys.min() + 1)
        bg_area = int(background.sum())
        bbox_result = feature_bbox_from_background(
            crop,
            background,
            domain_mask,
            config,
        )
        if bbox_result is None:
            continue
        bbox, noise_metrics = bbox_result
        noise_metrics["background_b"] = float(background_colour[0])
        noise_metrics["background_g"] = float(background_colour[1])
        noise_metrics["background_r"] = float(background_colour[2])
        bx, by, bw, bh = bbox
        score = (max(bg_w, bg_h), min(bg_w, bg_h), bg_area)
        if best is None or score > best[0]:
            best = (score, (x + bx, y + by, bw, bh), noise_metrics)

    if best is None:
        return None
    return best[1], best[2]


def refine_groups_with_uv_magic_wand(
    image: np.ndarray,
    groups: list[tuple[int, int, int, int]],
    config: MserConfig,
) -> tuple[list[tuple[int, int, int, int]], int, int, list[dict[str, float] | None]]:
    """Apply UV-domain magic-wand refinement to groups when configured."""
    if not config.enable_uv_magic_wand_refine or not config.uv_island_mask_path:
        return groups, 0, 0, [None] * len(groups)

    uv_mask = load_uv_island_mask(resolve_config_path(config.uv_island_mask_path), image.shape)
    rng = np.random.default_rng(0)
    refined: list[tuple[tuple[int, int, int, int], dict[str, float] | None]] = []
    changed = 0
    discarded = 0
    for group in groups:
        candidate = refine_group_with_uv_magic_wand(
            image,
            group,
            uv_mask,
            config,
            rng,
        )
        if candidate is None:
            if config.require_uv_magic_wand_refine:
                discarded += 1
                continue
            refined.append((group, None))
            continue
        candidate_group, noise_metrics = candidate
        refined.append((candidate_group, noise_metrics))
        if candidate_group != group:
            changed += 1

    refined.sort(key=lambda item: (item[0][1], item[0][0]))
    return [item[0] for item in refined], changed, discarded, [item[1] for item in refined]


def filter_groups_by_magic_wand_noise(
    groups: list[tuple[int, int, int, int]],
    noise_metrics: list[dict[str, float] | None],
    config: MserConfig,
) -> tuple[list[tuple[int, int, int, int]], list[dict[str, float] | None], int]:
    """Drop refined groups whose diagnostic noise score is above threshold."""
    if not config.enable_magic_wand_noise_filter:
        return groups, noise_metrics, 0

    filtered_groups: list[tuple[int, int, int, int]] = []
    filtered_metrics: list[dict[str, float] | None] = []
    rejected = 0
    for group, metrics in zip(groups, noise_metrics):
        if metrics is not None and metrics["score"] > config.max_magic_wand_noise_score:
            rejected += 1
            continue
        filtered_groups.append(group)
        filtered_metrics.append(metrics)

    return filtered_groups, filtered_metrics, rejected


def filter_groups_by_final_size(
    groups: list[tuple[int, int, int, int]],
    noise_metrics: list[dict[str, float] | None],
    config: MserConfig,
) -> tuple[list[tuple[int, int, int, int]], list[dict[str, float] | None], int]:
    """Drop remaining groups that are too small to be useful detections."""
    if not config.enable_final_size_filter:
        return groups, noise_metrics, 0

    filtered_groups: list[tuple[int, int, int, int]] = []
    filtered_metrics: list[dict[str, float] | None] = []
    rejected = 0
    for group, metrics in zip(groups, noise_metrics):
        _x, _y, w, h = group
        if (
            w < config.final_min_width_px
            or h < config.final_min_height_px
            or w * h < config.final_min_area_px
        ):
            rejected += 1
            continue
        filtered_groups.append(group)
        filtered_metrics.append(metrics)

    return filtered_groups, filtered_metrics, rejected


def dominant_colour_fraction(crop: np.ndarray, quant_step: int) -> float:
    """Return fraction of pixels in the dominant quantized colour bin."""
    pixels = crop.reshape(-1, 3)
    if len(pixels) == 0:
        return 1.0

    quant_step = max(quant_step, 1)
    quantized = (pixels // quant_step).astype(np.int32)
    levels = (255 // quant_step) + 1
    if levels**3 > 1_000_000:
        _bins, counts = np.unique(quantized.reshape(-1, 3), axis=0, return_counts=True)
        return float(counts.max()) / len(pixels)

    codes = (quantized[:, 0] * levels + quantized[:, 1]) * levels + quantized[:, 2]
    counts = np.bincount(codes)
    return float(counts.max()) / len(pixels)


def filter_groups_by_single_colour(
    image: np.ndarray,
    groups: list[tuple[int, int, int, int]],
    noise_metrics: list[dict[str, float] | None],
    config: MserConfig,
) -> tuple[list[tuple[int, int, int, int]], list[dict[str, float] | None], int]:
    """Drop final groups that are effectively one flat colour."""
    if not config.enable_final_single_colour_filter:
        return groups, noise_metrics, 0

    filtered_groups: list[tuple[int, int, int, int]] = []
    filtered_metrics: list[dict[str, float] | None] = []
    rejected = 0
    for group, metrics in zip(groups, noise_metrics):
        clamped = clamp_group(group, image.shape)
        if clamped is None:
            rejected += 1
            continue
        x, y, w, h = clamped
        crop = image[y : y + h, x : x + w]
        flat_fraction = dominant_colour_fraction(
            crop,
            config.final_single_colour_quant_step,
        )
        if flat_fraction >= config.final_single_colour_fraction:
            rejected += 1
            continue
        filtered_groups.append(group)
        filtered_metrics.append(metrics)

    return filtered_groups, filtered_metrics, rejected


def offset_background_fraction(
    image: np.ndarray,
    group: tuple[int, int, int, int],
    background_colour: np.ndarray,
    offset_width: int,
    colour_tolerance: int,
) -> float:
    """Return fraction of outward offset-ring pixels matching background colour."""
    ring_width = max(offset_width, 1)
    x, y, w, h = group
    expanded = clamp_group(
        (x - ring_width, y - ring_width, w + ring_width * 2, h + ring_width * 2),
        image.shape,
    )
    inner = clamp_group(group, image.shape)
    if expanded is None or inner is None:
        return 0.0

    ex, ey, ew, eh = expanded
    ix, iy, iw, ih = inner
    crop = image[ey : ey + eh, ex : ex + ew]
    ring_mask = np.ones((eh, ew), dtype=bool)

    inner_x0 = max(ix - ex, 0)
    inner_y0 = max(iy - ey, 0)
    inner_x1 = min(inner_x0 + iw, ew)
    inner_y1 = min(inner_y0 + ih, eh)
    ring_mask[inner_y0:inner_y1, inner_x0:inner_x1] = False

    ring_pixels = crop[ring_mask]
    if len(ring_pixels) == 0:
        return 0.0

    colour_delta = np.linalg.norm(
        ring_pixels.astype(np.float32) - background_colour.astype(np.float32),
        axis=1,
    )
    return float((colour_delta <= max(colour_tolerance, 0)).mean())


def filter_groups_by_offset_background(
    image: np.ndarray,
    groups: list[tuple[int, int, int, int]],
    noise_metrics: list[dict[str, float] | None],
    config: MserConfig,
) -> tuple[list[tuple[int, int, int, int]], list[dict[str, float] | None], int]:
    """Drop groups whose outward ring does not match detected background colour."""
    if not config.enable_final_offset_background_filter:
        return groups, noise_metrics, 0

    filtered_groups: list[tuple[int, int, int, int]] = []
    filtered_metrics: list[dict[str, float] | None] = []
    rejected = 0
    for group, metrics in zip(groups, noise_metrics):
        if metrics is None or not all(
            key in metrics for key in ("background_b", "background_g", "background_r")
        ):
            rejected += 1
            continue

        background_colour = np.asarray(
            (
                metrics["background_b"],
                metrics["background_g"],
                metrics["background_r"],
            ),
            dtype=np.float32,
        )
        coverage = offset_background_fraction(
            image,
            group,
            background_colour,
            config.final_offset_background_width_px,
            config.final_offset_background_colour_tol,
        )
        metrics["offset_background_fraction"] = coverage
        if coverage < config.final_offset_background_min_fraction:
            rejected += 1
            continue

        filtered_groups.append(group)
        filtered_metrics.append(metrics)

    return filtered_groups, filtered_metrics, rejected


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
    groups = group_boxes(boxes, grey.shape, config)
    candidate_group_count = len(groups)
    (
        groups,
        uv_magic_wand_adjusted,
        uv_magic_wand_discarded,
        noise_metrics,
    ) = refine_groups_with_uv_magic_wand(
        bgr,
        groups,
        config,
    )
    groups, noise_metrics, magic_wand_noise_rejected = filter_groups_by_magic_wand_noise(
        groups,
        noise_metrics,
        config,
    )
    groups, noise_metrics, final_size_rejected = filter_groups_by_final_size(
        groups,
        noise_metrics,
        config,
    )
    groups, noise_metrics, final_single_colour_rejected = filter_groups_by_single_colour(
        bgr,
        groups,
        noise_metrics,
        config,
    )
    groups, noise_metrics, final_offset_background_rejected = (
        filter_groups_by_offset_background(
            bgr,
            groups,
            noise_metrics,
            config,
        )
    )
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
        "candidate_groups": candidate_group_count,
        "uv_magic_wand_adjusted": uv_magic_wand_adjusted,
        "uv_magic_wand_discarded": uv_magic_wand_discarded,
        "magic_wand_noise_rejected": magic_wand_noise_rejected,
        "final_size_rejected": final_size_rejected,
        "final_single_colour_rejected": final_single_colour_rejected,
        "final_offset_background_rejected": final_offset_background_rejected,
        "grouped_regions": len(groups),
        "groups": [
            {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "noise_metrics": group_noise_metrics,
            }
            for (x, y, w, h), group_noise_metrics in zip(groups, noise_metrics)
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
    print(
        f"{summary['candidate_groups']} candidate groups, "
        f"{summary['uv_magic_wand_adjusted']} UV/magic-wand refined, "
        f"{summary['uv_magic_wand_discarded']} discarded without UV/magic-wand, "
        f"{summary['magic_wand_noise_rejected']} rejected by magic-wand noise, "
        f"{summary['final_size_rejected']} rejected by final size, "
        f"{summary['final_single_colour_rejected']} rejected as single colour, "
        f"{summary['final_offset_background_rejected']} rejected by offset background"
    )
    print(f"Wrote {summary['output']}")
    print(f"elapsed: {elapsed_ms:.1f} ms")

    for index, group in enumerate(summary["groups"], start=1):
        noise_metrics = group["noise_metrics"]
        if noise_metrics is None:
            noise_text = "n/a"
        else:
            noise_text = (
                f"{noise_metrics['score']:.3f} "
                f"signal={noise_metrics['signal']:.1f} "
                f"bgstd={noise_metrics['background_std']:.1f} "
                f"offsetbg={noise_metrics.get('offset_background_fraction', 0.0):.3f}"
            )
        print(
            f"  [{index:3d}]  x={group['x']:5d}  y={group['y']:5d}  "
            f"w={group['w']:4d}  h={group['h']:4d}  noise={noise_text}"
        )


if __name__ == "__main__":
    main()
