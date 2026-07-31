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
    Pillow          (PIL)      DDS decoding

Usage:
    python annotate_texture_regions.py scintilla_gauges.png
    python annotate_texture_regions.py scintilla_interior_b.color.DDS -o annotated.png

Tune detection behaviour by editing DEFAULT_CONFIG near the top of this file.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, fields, replace
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as _PILImage

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
    delta:         int   = 10  # default: 5
    min_area:      int   = 30  # default: 30
    max_area:      int   = 10000  # default: 1_024
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

    # Early box filters, applied to raw MSER boxes before grouping
    enable_flat_box_filter:      bool  = True  # default: True
    flat_box_colour_tolerance:   int   = 16  # default: 16 (magic-wand sensitivity)
    flat_box_min_coverage:       float = 0.97  # default: 0.97 (reject at or above)
    flat_box_min_domain_px:      int   = 8  # default: 8
    flat_box_context_px:         int   = 3  # default: 3 (an MSER box alone is always flat)
    min_box_uv_coverage:         float = 0.75  # default: 0.75 (share of the box inside the UV domain)
    flat_box_min_feature_px:     int   = 24  # default: 24 (largest non-background blob)
    # Off by default: on the scintilla/ardente/etk800 atlases this statistic did
    # not separate woven or perforated material from glyphs.  Measured on
    # hand-checked patches, weave scored 0.03-0.22 and glyphs 0.00-0.20 -- fully
    # overlapping.  Kept so the score stays inspectable in the tuning harness.
    enable_pattern_box_filter:   bool  = False  # default: False
    pattern_box_window_scale:    float = 3.0  # default: 3.0 (context around the box)
    pattern_box_min_window_px:   int   = 48  # default: 48
    pattern_box_min_period_px:   int   = 4  # default: 4
    pattern_box_max_period_px:   int   = 48  # default: 48
    max_pattern_autocorrelation: float = 0.45  # default: 0.45 (reject above)

    # The same test on assembled groups, where it does work: a group covers a
    # whole panel, so the statistic sees several periods.  Measured on the dash,
    # perforated grilles score 0.93-0.95 and the chevron weave 0.61-0.63, while
    # every glyph stays at or below 0.29.
    enable_pattern_group_filter:       bool  = True  # default: True
    pattern_group_window_scale:        float = 1.0  # default: 1.0 (the group itself)
    pattern_group_max_period_px:       int   = 160  # default: 160
    max_pattern_group_autocorrelation: float = 0.45  # default: 0.45 (reject above)

    # Grouping
    merge_distance_px:              int  = 10  # default: 10
    group_dilate_px:                int  = 8  # default: 8
    min_group_union_region_px:      int  = 169  # default: 169
    enable_group_area_filter:       bool = True  # default: True
    enable_group_degenerate_filter: bool = True  # default: True
    enable_island_bounded_grouping: bool = True  # default: True
    # How much UV-domain coverage a merge may cost before the merged box has to
    # justify itself by being round.  Absolute coverage is the wrong test: a
    # dial's box always has dead corners, and boxes already low stay low when
    # merged with their own duplicates.
    max_group_uv_coverage_drop:     float = 0.00  # default: 0.02

    # Round regions: inscribe a circle in a squarish group and keep it when the
    # corners it drops hold nothing -- background colour or no UV domain at all.
    enable_circular_groups:            bool  = True  # default: True
    circular_group_min_squareness:     float = 0.80  # default: 0.80 (min side / max side)
    circular_group_padding_px:         int   = 3  # default: 3 (grown past a strict inscribe)
    circular_group_colour_tolerance:   int   = 24  # default: 24
    circular_group_max_corner_content: float = 0.05  # default: 0.05 of the corner area

    # UV/magic-wand post cleanup
    enable_uv_mask_before_mser:            bool  = False  # default: False
    enable_uv_magic_wand_refine:           bool  = False  # default: False
    require_uv_magic_wand_refine:          bool  = False  # default: False
    uv_island_mask_path:                   str   = "mesh_segmentation_transform/segmentation_outputs/scintilla_interior_b.color.full_uv_filled_mask.png"  # black UV islands on white background
    magic_wand_colour_thresh:              int   = 100  # default: 100
    magic_wand_output_padding_px:          int   = 2  # default: 2
    magic_wand_min_island_area_px:         int   = 12  # default: 12
    # Contrast, measured with two standard quantities in CIELAB rather than a
    # bespoke score: Delta-E is the perceptual distance between the mark and its
    # background, and contrast-to-noise is that distance in units of the
    # background's own standard deviation.
    enable_contrast_filter:                bool  = False  # default: True
    contrast_background_tolerance:         int   = 100  # default: 100 (background colour band)
    contrast_surround_px:                  int   = 12  # default: 12
    contrast_surround_scale:               float = 0.5  # default: 0.5 of the longest side
    contrast_surround_gap_px:              int   = 3  # default: 3 (guard band)
    min_contrast_background_px:            int   = 64  # default: 64
    max_contrast_region_area_px:           int   = 6000  # default: 6000 (larger regions skip the test)
    min_contrast_delta_e:                  float = 2.0  # default: 2.0 (CIE76)
    min_contrast_to_noise:                 float = 3.5  # default: 3.5 (background sigmas)
    enable_final_size_filter:              bool  = True  # default: True
    final_min_width_px:                    int   = 4  # default: 4
    final_min_height_px:                   int   = 4  # default: 4
    final_min_area_px:                     int   = 50  # default: 100
    enable_final_aspect_filter:            bool  = True  # default: True
    final_max_aspect:                      float = 12.0  # default: 12.0 (long side / short)

    # Last word on the region, measured on its shaped form: a dial's box has
    # dead corners but its circle sits wholly inside the domain, while a
    # rectangle straddling an island edge still does not.
    enable_region_domain_filter:           bool  = True  # default: True
    min_region_uv_coverage:                float = 0.98  # default: 0.98
    enable_final_single_colour_filter:     bool  = True  # default: True
    final_single_colour_quant_step:        int   = 12  # default: 12
    final_single_colour_fraction:          float = 0.98  # default: 0.98
    enable_final_offset_background_filter: bool  = False  # default: False
    final_offset_background_width_px:      int   = 2  # default: 2
    final_offset_background_colour_tol:    int   = 32  # default: 32
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


@dataclass(frozen=True, slots=True)
class TextureRegionAnnotationOutputs:
    """External files written by the annotation pipeline."""

    annotated_image: Path
    summary_json: Path | None = None


@dataclass(frozen=True, slots=True)
class DetectionStage:
    """One step of the pipeline, with what it kept and what it removed.

    ``rejected`` holds the boxes this step alone discarded, so a tuning UI can
    show why a region disappeared without re-running earlier steps.
    """

    key: str
    title: str
    kept: tuple[tuple[int, int, int, int], ...]
    rejected: tuple[tuple[int, int, int, int], ...] = ()
    detail: str = ""
    adjusted: int = 0  # boxes this stage moved rather than removed
    # Radius per kept region, or None where the region stays rectangular.
    circles: tuple[int | None, ...] = ()


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------


def load_image(path: Path) -> np.ndarray:
    """Load a texture as a BGR uint8 ndarray.

    PNG, JPG, BMP, TIFF and other formats supported by the local OpenCV build
    are read directly; DDS goes through Pillow.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Texture not found: {path}")

    if path.suffix.lower() == ".dds":
        with _PILImage.open(path) as pil_image:
            rgba = pil_image.convert("RGBA")
        return cv2.cvtColor(np.asarray(rgba, dtype=np.uint8), cv2.COLOR_RGBA2BGR)

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(
            f"OpenCV could not decode {path.name}.  "
            f"Use PNG, JPG, BMP, TIFF, DDS, or another format supported by "
            f"your OpenCV build."
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


def apply_uv_mask_for_mser(image: np.ndarray, uv_mask: np.ndarray) -> np.ndarray:
    """Return a copy where pixels outside the UV domain are neutral white."""
    masked = image.copy()
    masked[~uv_mask] = (255, 255, 255)
    return masked


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


def config_with_uv_mask_path(
    config: MserConfig,
    uv_island_mask_path: Path | str | None,
) -> MserConfig:
    """Return config with an explicit external UV mask path when provided."""
    if uv_island_mask_path is None:
        return config
    return replace(config, uv_island_mask_path=str(uv_island_mask_path))


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


def context_box(
    box: tuple[int, int, int, int],
    ring: int,
) -> tuple[int, int, int, int]:
    """Grow a box by a fixed ring of pixels on every side.

    Deliberately additive rather than a multiple of the box: scaling a long or
    tall box reaches far enough to swallow whole neighbouring regions, so a box
    sitting in flat black beside a light panel measured as half-and-half and
    never registered as flat.  A small ring stays within the region the box is
    actually in, while still bringing in the background around a glyph stroke.
    """
    x, y, w, h = box
    if ring <= 0:
        return box
    return (x - ring, y - ring, w + ring * 2, h + ring * 2)


def dominant_island_component(
    domain: np.ndarray,
    inner: tuple[int, int, int, int],
) -> np.ndarray:
    """Keep only the UV island the box itself sits on.

    Islands can be separated by a gap of two or three pixels, so a context ring
    hops straight into the neighbouring island and mixes in its colour: a box on
    a flat grey panel next to a flat black one measured as two-toned and escaped
    the flat filter.  ``inner`` is the box within the window, used to pick which
    connected component of the domain to keep.
    """
    count, labels = cv2.connectedComponents(domain.astype(np.uint8), connectivity=8)
    if count <= 2:  # background plus at most one island
        return domain
    x, y, w, h = inner
    within = labels[y : y + h, x : x + w]
    within = within[within > 0]
    if within.size == 0:
        return domain
    return labels == int(np.bincount(within).argmax())


def channel_colour_mask(
    crop: np.ndarray,
    domain_mask: np.ndarray,
    colour: np.ndarray,
    tolerance: int,
) -> np.ndarray:
    """Return domain pixels within ``tolerance`` of a colour on every channel.

    Per-channel rather than Euclidean, to stay consistent with the quantised
    bins ``estimate_background_colour`` uses to pick that colour: a bin of width
    t spans up to t*sqrt(3) in Euclidean distance, so an Euclidean test with the
    same number splits colours the binning already called identical.  Uniform
    dark trim carrying +/-20 of compression dither reads as 75% flat under the
    Euclidean test and 100% under this one.
    """
    delta = np.abs(crop.astype(np.int16) - colour.astype(np.int16))
    return (delta.max(axis=2) <= max(tolerance, 0)) & domain_mask


def box_uv_coverage(
    uv_mask: np.ndarray | None,
    box: tuple[int, int, int, int],
    image_shape: tuple[int, ...],
) -> float:
    """Return the share of a box's area that lies inside the UV domain."""
    if uv_mask is None:
        return 1.0
    clamped = clamp_group(box, image_shape)
    if clamped is None:
        return 0.0
    x, y, w, h = clamped
    _bx, _by, bw, bh = box
    area = max(bw * bh, 1)
    return float(uv_mask[y : y + h, x : x + w].sum()) / float(area)


def box_flat_colour_coverage(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    box: tuple[int, int, int, int],
    config: MserConfig,
) -> tuple[float, int] | None:
    """Return a box's single-colour coverage and its largest non-background blob.

    Coverage alone cannot tell blank trim from a thin glyph stroke on a wide
    plate -- both are ~85% one colour.  The second value is the area of the
    largest connected run of non-dominant pixels: speckle noise on blank trim
    breaks into tiny fragments, while a stroke stays whole.

    Uses the magic wand's colour model: the dominant colour is estimated inside
    the UV domain, then coverage is the share of domain pixels within
    ``flat_box_colour_tolerance`` of it.  ``None`` means the box has too little
    UV domain to judge.

    The test covers the box plus ``flat_box_context_px``, because an MSER region
    is homogeneous by definition -- measured on the bare box, a glyph stroke
    looks as single-coloured as blank trim does.  With a ring of context, a
    stroke brings its background in and stops looking flat.
    """
    inner = box
    box = context_box(box, config.flat_box_context_px)
    clamped = clamp_group(box, image.shape)
    if clamped is None:
        return None
    x, y, w, h = clamped
    crop = image[y : y + h, x : x + w]
    domain = (
        uv_mask[y : y + h, x : x + w]
        if uv_mask is not None
        else np.ones((h, w), dtype=bool)
    )
    if uv_mask is not None:
        domain = dominant_island_component(
            domain,
            (
                max(inner[0] - x, 0),
                max(inner[1] - y, 0),
                max(min(inner[2], w), 1),
                max(min(inner[3], h), 1),
            ),
        )
    if int(domain.sum()) < max(config.flat_box_min_domain_px, 1):
        return None

    tolerance = max(config.flat_box_colour_tolerance, 0)
    dominant = estimate_background_colour(crop, domain, max(tolerance, 1))
    if dominant is None:
        return None
    matching = channel_colour_mask(crop, domain, dominant, tolerance)
    coverage = float(matching.sum()) / float(domain.sum())

    feature = domain & ~matching
    if not bool(feature.any()):
        return coverage, 0
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        feature.astype(np.uint8), connectivity=8
    )
    largest = max(
        (int(stats[label, cv2.CC_STAT_AREA]) for label in range(1, count)),
        default=0,
    )
    return coverage, largest


def filter_boxes_by_flat_colour(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    boxes: np.ndarray,
    config: MserConfig,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Drop MSER boxes that are single-coloured or mostly outside the UV domain.

    The pre-MSER fill makes every island silhouette a hard edge, so MSER finds a
    few regions straddling the boundary.  Those carry no geometry over most of
    their area and would otherwise drag groups out across the gaps.
    """
    if not config.enable_flat_box_filter or len(boxes) == 0:
        return boxes, []

    kept: list[tuple[int, int, int, int]] = []
    rejected: list[tuple[int, int, int, int]] = []
    for raw in boxes:
        box = (int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3]))
        if box_uv_coverage(uv_mask, box, image.shape) < config.min_box_uv_coverage:
            rejected.append(box)
            continue
        metrics = box_flat_colour_coverage(image, uv_mask, box, config)
        if metrics is None:
            rejected.append(box)
            continue
        coverage, feature_px = metrics
        if (
            coverage >= config.flat_box_min_coverage
            or feature_px < config.flat_box_min_feature_px
        ):
            rejected.append(box)
            continue
        kept.append(box)
    return (
        np.asarray(kept, dtype=np.int32) if kept else np.empty((0, 4), dtype=np.int32),
        rejected,
    )


def repeating_pattern_score(window: np.ndarray, config: MserConfig) -> float:
    """Return how strongly a patch revives its own correlation at a shift.

    Normalised autocorrelation through the FFT (Wiener-Khinchin), reduced to a
    profile over lag radius.  Non-repeating content decays away from zero lag
    and never recovers, so the statistic is the largest rise back above the
    running minimum: a genuine second peak, not just a high correlation.  The
    raw peak is useless here because it mostly measures smoothness -- blank trim
    scores higher than a woven panel.

    Note the score did not separate weave from glyphs on the atlases tested; it
    is reported so the tuning harness can show it, and its filter is off by
    default.
    """
    patch = window.astype(np.float32)
    patch -= float(patch.mean())
    if float(np.abs(patch).max()) < 1e-3:
        return 0.0  # featureless: the flat-colour filter owns this case

    height, width = patch.shape
    max_lag = min(
        max(config.pattern_box_max_period_px, 1),
        (height - 1) // 2,
        (width - 1) // 2,
    )
    min_lag = max(config.pattern_box_min_period_px, 2)
    if max_lag < min_lag:
        return 0.0

    # Taper the edges so the patch border does not fake a periodic signal.
    patch = patch * np.hanning(height)[:, None] * np.hanning(width)[None, :]
    spectrum = np.fft.rfft2(patch, s=(height * 2, width * 2))
    correlation = np.fft.irfft2(np.abs(spectrum) ** 2, s=(height * 2, width * 2))
    peak = float(correlation[0, 0])
    if peak <= 0.0:
        return 0.0
    correlation /= peak

    # Non-negative row lags cover the plane by symmetry; keep both column signs
    # so diagonal weaves are caught whichever way they lean.
    rows = correlation[: max_lag + 1]
    region = np.concatenate([rows[:, : max_lag + 1], rows[:, -max_lag:]], axis=1)
    lag_y = np.arange(max_lag + 1, dtype=np.float32)[:, None]
    lag_x = np.concatenate(
        [
            np.arange(max_lag + 1, dtype=np.float32),
            np.arange(-max_lag, 0, dtype=np.float32),
        ]
    )[None, :]
    radius = np.sqrt(lag_y**2 + lag_x**2)

    best = 0.0
    running_minimum = 1.0
    for lag in range(1, max_lag + 1):
        ring = (radius >= lag - 0.5) & (radius < lag + 0.5)
        if not bool(ring.any()):
            continue
        value = float(region[ring].max())
        if lag >= min_lag:
            best = max(best, value - running_minimum)
        running_minimum = min(running_minimum, value)
    return best


def pattern_window_for_box(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    config: MserConfig,
) -> np.ndarray | None:
    """Return a grey neighbourhood around a box, wide enough to show repeats."""
    x, y, w, h = box
    target = max(
        int(round(max(w, h) * max(config.pattern_box_window_scale, 1.0))),
        max(config.pattern_box_min_window_px, 4),
    )
    centre_x = x + w // 2
    centre_y = y + h // 2
    half = target // 2
    clamped = clamp_group(
        (centre_x - half, centre_y - half, half * 2, half * 2), image.shape
    )
    if clamped is None:
        return None
    cx, cy, cw, ch = clamped
    if cw < 8 or ch < 8:
        return None
    return cv2.cvtColor(image[cy : cy + ch, cx : cx + cw], cv2.COLOR_BGR2GRAY)


def filter_boxes_by_pattern(
    image: np.ndarray,
    boxes: np.ndarray,
    config: MserConfig,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Drop MSER boxes sitting on a repeating material such as a carbon weave."""
    if not config.enable_pattern_box_filter or len(boxes) == 0:
        return boxes, []

    kept: list[tuple[int, int, int, int]] = []
    rejected: list[tuple[int, int, int, int]] = []
    for raw in boxes:
        box = (int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3]))
        window = pattern_window_for_box(image, box, config)
        if window is None:
            kept.append(box)
            continue
        if repeating_pattern_score(window, config) > config.max_pattern_autocorrelation:
            rejected.append(box)
            continue
        kept.append(box)
    return (
        np.asarray(kept, dtype=np.int32) if kept else np.empty((0, 4), dtype=np.int32),
        rejected,
    )


def circle_uv_coverage(
    uv_mask: np.ndarray | None,
    centre: tuple[float, float],
    radius: float,
) -> float:
    """Return the share of a disc that lies inside the UV domain."""
    if uv_mask is None:
        return 1.0
    height, width = uv_mask.shape[:2]
    centre_x, centre_y = centre
    x0 = max(int(np.floor(centre_x - radius)), 0)
    y0 = max(int(np.floor(centre_y - radius)), 0)
    x1 = min(int(np.ceil(centre_x + radius)), width)
    y1 = min(int(np.ceil(centre_y + radius)), height)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    columns = np.arange(x0, x1, dtype=np.float32)[None, :] + 0.5
    rows = np.arange(y0, y1, dtype=np.float32)[:, None] + 0.5
    disc = ((columns - centre_x) ** 2 + (rows - centre_y) ** 2) <= radius**2
    area = int(disc.sum())
    if area == 0:
        return 0.0
    return float((uv_mask[y0:y1, x0:x1] & disc).sum()) / float(area)


def inscribed_circle_radius(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    group: tuple[int, int, int, int],
    config: MserConfig,
) -> int | None:
    """Return an inscribed-circle radius when the corners it drops are empty.

    A round gauge or button fills a square box, but the four corners outside its
    inscribed circle carry nothing: either background colour, or no UV domain at
    all.  When that holds the circle is the honest region and the corners are
    dead area; when anything else lives there the square is kept.

    The circle is grown by ``circular_group_padding_px`` past a strict inscribe,
    so a round mark that touches its box edge is not clipped by a circle drawn
    exactly through it.
    """
    if not config.enable_circular_groups:
        return None

    x, y, w, h = group
    if w <= 0 or h <= 0:
        return None
    if min(w, h) / max(w, h) < config.circular_group_min_squareness:
        return None

    clamped = clamp_group(group, image.shape)
    if clamped is None:
        return None
    cx, cy, cw, ch = clamped
    crop = image[cy : cy + ch, cx : cx + cw]
    domain = (
        uv_mask[cy : cy + ch, cx : cx + cw]
        if uv_mask is not None
        else np.ones((ch, cw), dtype=bool)
    )

    radius = min(w, h) / 2.0 + max(config.circular_group_padding_px, 0)
    # A circle is only worth having if it is the tighter region.  Padding is a
    # fixed number of pixels, so on a small box it more than undoes the corners
    # the inscribe saves and the "circle" ends up larger than the square.
    if math.pi * radius**2 >= w * h:
        return None
    if not _circle_keeps_uv_coverage(uv_mask, group, radius, config):
        return None
    centre_x = (x + w / 2.0) - cx
    centre_y = (y + h / 2.0) - cy
    columns = np.arange(cw, dtype=np.float32)[None, :] + 0.5
    rows = np.arange(ch, dtype=np.float32)[:, None] + 0.5
    outside_circle = ((columns - centre_x) ** 2 + (rows - centre_y) ** 2) > radius**2

    # Corners with no UV domain are already dead; only judge domain pixels.
    corners = outside_circle & domain
    corner_area = int(corners.sum())
    if corner_area == 0:
        return int(round(radius))

    background = estimate_background_colour(
        crop, domain, config.circular_group_colour_tolerance
    )
    if background is None:
        return None
    matching = channel_colour_mask(
        crop, corners, background, config.circular_group_colour_tolerance
    )
    stray = float((corners & ~matching).sum()) / float(corner_area)
    if stray > config.circular_group_max_corner_content:
        return None
    return int(round(radius))


def _circle_keeps_uv_coverage(
    uv_mask: np.ndarray | None,
    group: tuple[int, int, int, int],
    radius: float,
    config: MserConfig,
) -> bool:
    """Hold the circle to the same UV-domain rule that gates group merges.

    Padding grows the circle past the box edges, so the region can reach outside
    the domain even though the rectangle it replaces did not.
    """
    x, y, w, h = group
    return (
        circle_uv_coverage(uv_mask, (x + w / 2.0, y + h / 2.0), radius)
        >= config.min_box_uv_coverage
    )


def filter_groups_by_pattern(
    image: np.ndarray,
    groups: list[tuple[int, int, int, int]],
    config: MserConfig,
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    """Re-test assembled groups for repeating material.

    A group spans a whole panel rather than one glyph stroke, so the statistic
    finally has several periods to work with -- the same test on a single MSER
    box could not separate anything.  The window is the group itself: padding it
    out dilutes the pattern with surrounding trim.
    """
    if not config.enable_pattern_group_filter or not groups:
        return groups, []

    scoped = replace(
        config,
        pattern_box_window_scale=config.pattern_group_window_scale,
        pattern_box_max_period_px=config.pattern_group_max_period_px,
    )
    kept: list[tuple[int, int, int, int]] = []
    rejected: list[tuple[int, int, int, int]] = []
    for group in groups:
        window = pattern_window_for_box(image, group, scoped)
        if window is None:
            kept.append(group)
            continue
        score = repeating_pattern_score(window, scoped)
        if score > config.max_pattern_group_autocorrelation:
            rejected.append(group)
            continue
        kept.append(group)
    return kept, rejected


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


def grid_cells_for_box(
    box: tuple[int, int, int, int],
    cell_size: int,
) -> list[tuple[int, int]]:
    """Return flat grid cell ids spanned by a box."""
    x, y, w, h = box
    cell_size = max(cell_size, 1)
    x0 = x // cell_size
    y0 = y // cell_size
    x1 = (x + w - 1) // cell_size
    y1 = (y + h - 1) // cell_size
    return [(cx, cy) for cy in range(y0, y1 + 1) for cx in range(x0, x1 + 1)]


def union_region_group_boxes(
    boxes: np.ndarray,
    image: np.ndarray,
    config: MserConfig,
    uv_mask: np.ndarray | None = None,
    strict_merges: bool = False,
) -> list[tuple[int, int, int, int]]:
    """Group boxes when their expanded union/contact region is large enough.

    Merging is allowed to run first and judged afterwards.  A finished group
    that falls below ``min_box_uv_coverage`` gets one chance to justify itself
    by being round: shaping it into a circle drops the dead corners a dial's
    bounding box necessarily has.  Failing that the merge is undone and the
    members stand alone, with no attempt to re-group a subset of them.

    With ``enable_island_bounded_grouping`` a merge is refused up front when the
    two boxes sit on different islands: separate islands are separate surfaces,
    so text on one never continues onto the other however close they are.
    """
    if len(boxes) == 0:
        return []

    image_shape = image.shape
    height, width = image_shape[:2]
    distance = max(0, config.merge_distance_px + config.group_dilate_px)
    min_union_area = max(1, config.min_group_union_region_px)
    box_tuples = [tuple(int(value) for value in box) for box in boxes]
    expanded_boxes = [expanded_box(box, distance) for box in box_tuples]
    cell_size = max(distance + int(np.sqrt(min_union_area)), 16)
    parent = list(range(len(box_tuples)))
    # Tight bounds per component, maintained as merges happen.
    bounds = [(x, y, x + w, y + h) for x, y, w, h in box_tuples]

    # Summed-area table makes each candidate merge an O(1) coverage test.
    coverage_limit = config.min_box_uv_coverage if uv_mask is not None else 0.0
    integral = (
        cv2.integral(uv_mask.astype(np.uint8))
        if uv_mask is not None and coverage_limit > 0.0
        else None
    )

    # Which island each box sits on, by majority label over the box.
    island = [0] * len(box_tuples)
    if uv_mask is not None and config.enable_island_bounded_grouping:
        _count, labels = cv2.connectedComponents(
            uv_mask.astype(np.uint8), connectivity=8
        )
        for index, (bx, by, bw, bh) in enumerate(box_tuples):
            clamped = clamp_group((bx, by, bw, bh), uv_mask.shape)
            if clamped is None:
                continue
            cx, cy, cw, ch = clamped
            within = labels[cy : cy + ch, cx : cx + cw]
            within = within[within > 0]
            if within.size:
                island[index] = int(np.bincount(within).argmax())

    def covered(rect: tuple[int, int, int, int]) -> float:
        if integral is None:
            return 1.0
        x0 = min(max(rect[0], 0), width)
        y0 = min(max(rect[1], 0), height)
        x1 = min(max(rect[2], 0), width)
        y1 = min(max(rect[3], 0), height)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        inside = (
            int(integral[y1, x1])
            - int(integral[y0, x1])
            - int(integral[y1, x0])
            + int(integral[y0, x0])
        )
        return inside / float((rect[2] - rect[0]) * (rect[3] - rect[1]))

    # Per original box, never mutated; ``coverage`` tracks the growing component.
    box_coverage = [covered(rect) for rect in bounds]
    coverage = list(box_coverage)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if island[left_root] != island[right_root]:
            return
        first = bounds[left_root]
        second = bounds[right_root]
        merged = (
            min(first[0], second[0]),
            min(first[1], second[1]),
            max(first[2], second[2]),
            max(first[3], second[3]),
        )
        merged_coverage = covered(merged)
        # A ring of marks is only round once every member is in, so merges are
        # not judged here unless this is the conservative second pass.
        if strict_merges and merged_coverage < (
            min(coverage[left_root], coverage[right_root])
            - max(config.max_group_uv_coverage_drop, 0.0)
        ):
            return
        parent[right_root] = left_root
        bounds[left_root] = merged
        coverage[left_root] = merged_coverage

    grid: dict[tuple[int, int], list[int]] = {}
    for i, expanded in enumerate(expanded_boxes):
        checked: set[int] = set()
        cells = list(grid_cells_for_box(expanded, cell_size))
        for cell in cells:
            for j in grid.get(cell, []):
                if j in checked:
                    continue
                checked.add(j)
                if intersection_area(expanded, expanded_boxes[j]) >= min_union_area:
                    union(i, j)
        for cell in cells:
            grid.setdefault(cell, []).append(i)

    members: dict[int, list[int]] = {}
    for index in range(len(box_tuples)):
        members.setdefault(find(index), []).append(index)

    groups: list[tuple[int, int, int, int]] = []

    def emit(rect: tuple[int, int, int, int]) -> None:
        x0 = max(rect[0], 0)
        y0 = max(rect[1], 0)
        x1 = min(rect[2], width)
        y1 = min(rect[3], height)
        w = x1 - x0
        h = y1 - y0
        if config.enable_group_area_filter and (
            w * h > image_shape[0] * image_shape[1] * config.max_area_fraction
        ):
            return
        if config.enable_group_degenerate_filter and (w < 4 or h < 4):
            return
        groups.append((x0, y0, w, h))

    drop = max(config.max_group_uv_coverage_drop, 0.0)
    for root, indices in members.items():
        rect = bounds[root]
        # The merge distance decides what joins; the group itself is just the
        # smallest rectangle enclosing its members.
        if strict_merges or len(indices) == 1:
            emit(rect)
            continue

        cheapest = min(box_coverage[index] for index in indices)
        if covered(rect) >= cheapest - drop:
            emit(rect)
            continue

        # The finished group cost UV-domain coverage.  It earns that back by
        # being round, which drops the dead corners a dial's box always has.
        candidate = (
            max(rect[0], 0),
            max(rect[1], 0),
            min(rect[2], width) - max(rect[0], 0),
            min(rect[3], height) - max(rect[1], 0),
        )
        if inscribed_circle_radius(image, uv_mask, candidate, config):
            emit(rect)
            continue

        # Otherwise undo it and regroup the members conservatively, refusing any
        # merge that costs coverage.  Subsets are never circle-fitted: either the
        # whole group justified itself as round or none of it does.
        subset = np.asarray([box_tuples[index] for index in indices], dtype=np.int32)
        for regrouped in union_region_group_boxes(
            subset, image, config, uv_mask, strict_merges=True
        ):
            gx, gy, gw, gh = regrouped
            emit((gx, gy, gx + gw, gy + gh))

    groups.sort(key=lambda g: (g[1], g[0]))
    return groups


def group_boxes(
    boxes: np.ndarray,
    image: np.ndarray,
    config: MserConfig,
    uv_mask: np.ndarray | None = None,
) -> list[tuple[int, int, int, int]]:
    """Merge bounding boxes into coherent text/symbol groups."""
    return union_region_group_boxes(boxes, image, config, uv_mask)


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
    """Return contrast diagnostics for a mark against its own background.

    Two standard quantities, both in CIELAB:

    ``delta_e``
        CIE76 colour difference between the mark and the background -- how far
        apart they are perceptually, in the units perceptual thresholds are
        quoted in.
    ``contrast_to_noise``
        That difference divided by the background's standard deviation, the
        usual contrast-to-noise ratio.  It answers the question a flat-colour or
        variance test cannot: a mark on clean trim stands many sigmas clear,
        while a shadow on woven or ribbed material does not, however dark it is.
    """
    if not bool(background_mask.any()) or not bool(signal_mask.any()):
        return {"delta_e": 0.0, "contrast_to_noise": 0.0, "background_std": 0.0}

    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    background_mean, background_stddev = cv2.meanStdDev(
        lab, mask=background_mask.astype(np.uint8)
    )
    background_std = float(np.linalg.norm(background_stddev.reshape(3)))
    signal_mean = lab[signal_mask].astype(np.float32).mean(axis=0)
    delta_e = float(np.linalg.norm(signal_mean - background_mean.reshape(3)))

    return {
        "delta_e": delta_e,
        "contrast_to_noise": delta_e / max(background_std, 1e-6),
        "background_std": background_std,
    }


def occupied_region_mask(
    shape: tuple[int, ...],
    groups: list[tuple[int, int, int, int]],
    gap: int,
) -> np.ndarray:
    """Return every region's footprint, grown by a guard band.

    Used to keep the contrast surround off other detections and off the
    antialiased halo around its own mark; both would land in the background
    sample and inflate the noise term for exactly the crisp marks worth keeping.
    """
    height, width = shape[:2]
    occupied = np.zeros((height, width), dtype=np.uint8)
    for x, y, w, h in groups:
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, width), min(y + h, height)
        if x1 > x0 and y1 > y0:
            occupied[y0:y1, x0:x1] = 1
    if gap > 0:
        occupied = cv2.dilate(
            occupied, cv2.getStructuringElement(cv2.MORPH_RECT, (gap * 2 + 1,) * 2)
        )
    return occupied > 0


def region_contrast_metrics(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    group: tuple[int, int, int, int],
    config: MserConfig,
    blocked: np.ndarray | None = None,
) -> dict[str, float] | None:
    """Measure the mark against the material surrounding it.

    Signal and noise are measured over different areas, which is the whole
    point: a 9x11 px region has no meaningful noise estimate inside itself, so
    the surround supplies one.  The region is dilated, intersected with the UV
    domain and with the island the region sits on -- pixels off the island or
    off the atlas are not this material and must not colour the estimate -- and
    the ring outside the region becomes the background sample.

    ``delta_e``
        CIE76 distance between the mark's mean colour and the surround's.
    ``contrast_to_noise``
        That distance over the surround's standard deviation.  Because the noise
        term comes from the material rather than from the mark, a faint but
        well-formed mark on clean trim still scores well: it is measuring how
        far the mark sits from the surface's own variation, not how loud it is.
    """
    clamped = clamp_group(group, image.shape)
    if clamped is None:
        return None
    x, y, w, h = clamped

    margin = max(
        config.contrast_surround_px,
        int(round(max(w, h) * max(config.contrast_surround_scale, 0.0))),
    )
    outer = clamp_group((x - margin, y - margin, w + margin * 2, h + margin * 2), image.shape)
    if outer is None:
        return None
    ox, oy, ow, oh = outer
    crop = image[oy : oy + oh, ox : ox + ow]
    domain = (
        uv_mask[oy : oy + oh, ox : ox + ow]
        if uv_mask is not None
        else np.ones((oh, ow), dtype=bool)
    )

    inner = np.zeros((oh, ow), dtype=bool)
    inner[y - oy : y - oy + h, x - ox : x - ox + w] = True
    if uv_mask is not None:
        domain = dominant_island_component(domain, (x - ox, y - oy, w, h))

    # Keep the surround off every detection and off its own guard band.
    if blocked is not None:
        excluded = blocked[oy : oy + oh, ox : ox + ow]
    else:
        excluded = occupied_region_mask(
            (oh, ow),
            [(x - ox, y - oy, w, h)],
            max(config.contrast_surround_gap_px, 0),
        )
    surround = domain & ~inner & ~excluded
    if int(surround.sum()) < max(config.min_contrast_background_px, 1):
        return None
    mark = domain & inner
    if not bool(mark.any()):
        return None

    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    background = lab[surround]
    tolerance = max(config.contrast_background_tolerance, 1)
    background_colour = estimate_background_colour(crop, surround, tolerance)
    if background_colour is not None:
        # Judge the mark by the pixels that differ from the surrounding material,
        # not by the region's average, which a large flat margin would dominate.
        matching = channel_colour_mask(crop, mark, background_colour, tolerance)
        distinct = mark & ~matching
        if bool(distinct.any()):
            mark = distinct

    delta_e = float(np.linalg.norm(lab[mark].mean(axis=0) - background.mean(axis=0)))
    # Floor the spread at one code value: below that the ratio measures
    # quantisation, and an unbounded number is no use as a threshold.
    background_std = max(float(np.linalg.norm(background.std(axis=0))), 1.0)
    return {
        "delta_e": delta_e,
        "contrast_to_noise": delta_e / background_std,
        "background_std": background_std,
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
) -> tuple[tuple[int, int, int, int], dict[str, float]] | None:
    """Refine a group using UV-domain constrained background components."""
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
    if not bool(seed_mask.any()):
        return None

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        seed_mask.astype(np.uint8),
        connectivity=8,
    )
    if num_labels <= 1:
        return None

    candidates: list[tuple[tuple[int, int, int], int]] = []
    for label_id in range(1, num_labels):
        bg_w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        bg_h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        bg_area = int(stats[label_id, cv2.CC_STAT_AREA])
        candidates.append(((max(bg_w, bg_h), min(bg_w, bg_h), bg_area), label_id))

    candidates.sort(key=lambda item: item[0], reverse=True)
    for _score, label_id in candidates:
        background = labels == label_id
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
        return (x + bx, y + by, bw, bh), noise_metrics

    return None


def refine_groups_with_uv_magic_wand(
    image: np.ndarray,
    groups: list[tuple[int, int, int, int]],
    uv_mask: np.ndarray | None,
    config: MserConfig,
) -> tuple[
    list[tuple[int, int, int, int]],
    int,
    list[tuple[int, int, int, int]],
    list[dict[str, float] | None],
]:
    """Apply UV-domain magic-wand refinement to groups when configured."""
    if not config.enable_uv_magic_wand_refine or uv_mask is None:
        return groups, 0, [], [None] * len(groups)

    refined: list[tuple[tuple[int, int, int, int], dict[str, float] | None]] = []
    changed = 0
    discarded: list[tuple[int, int, int, int]] = []
    for group in groups:
        candidate = refine_group_with_uv_magic_wand(
            image,
            group,
            uv_mask,
            config,
        )
        if candidate is None:
            if config.require_uv_magic_wand_refine:
                discarded.append(group)
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
) -> tuple[
    list[tuple[int, int, int, int]],
    list[dict[str, float] | None],
    list[tuple[int, int, int, int]],
]:
    """Drop groups whose mark does not stand clear of its own background."""
    if not config.enable_contrast_filter:
        return groups, noise_metrics, []

    filtered_groups: list[tuple[int, int, int, int]] = []
    filtered_metrics: list[dict[str, float] | None] = []
    rejected: list[tuple[int, int, int, int]] = []
    for group, metrics in zip(groups, noise_metrics):
        if metrics is not None and (
            metrics["delta_e"] < config.min_contrast_delta_e
            or metrics["contrast_to_noise"] < config.min_contrast_to_noise
        ):
            rejected.append(group)
            continue
        filtered_groups.append(group)
        filtered_metrics.append(metrics)

    return filtered_groups, filtered_metrics, rejected


def filter_groups_by_final_size(
    groups: list[tuple[int, int, int, int]],
    noise_metrics: list[dict[str, float] | None],
    config: MserConfig,
) -> tuple[
    list[tuple[int, int, int, int]],
    list[dict[str, float] | None],
    list[tuple[int, int, int, int]],
]:
    """Drop remaining groups too small, or too elongated, to be a detection."""
    if not config.enable_final_size_filter and not config.enable_final_aspect_filter:
        return groups, noise_metrics, []

    filtered_groups: list[tuple[int, int, int, int]] = []
    filtered_metrics: list[dict[str, float] | None] = []
    rejected: list[tuple[int, int, int, int]] = []
    for group, metrics in zip(groups, noise_metrics):
        _x, _y, w, h = group
        if config.enable_final_size_filter and (
            w < config.final_min_width_px
            or h < config.final_min_height_px
            or w * h < config.final_min_area_px
        ):
            rejected.append(group)
            continue
        aspect = max(w, h) / max(min(w, h), 1)
        if config.enable_final_aspect_filter and aspect > config.final_max_aspect:
            rejected.append(group)
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
) -> tuple[
    list[tuple[int, int, int, int]],
    list[dict[str, float] | None],
    list[tuple[int, int, int, int]],
]:
    """Drop final groups that are effectively one flat colour."""
    if not config.enable_final_single_colour_filter:
        return groups, noise_metrics, []

    filtered_groups: list[tuple[int, int, int, int]] = []
    filtered_metrics: list[dict[str, float] | None] = []
    rejected: list[tuple[int, int, int, int]] = []
    for group, metrics in zip(groups, noise_metrics):
        clamped = clamp_group(group, image.shape)
        if clamped is None:
            rejected.append(group)
            continue
        x, y, w, h = clamped
        crop = image[y : y + h, x : x + w]
        flat_fraction = dominant_colour_fraction(
            crop,
            config.final_single_colour_quant_step,
        )
        if flat_fraction >= config.final_single_colour_fraction:
            rejected.append(group)
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
) -> tuple[
    list[tuple[int, int, int, int]],
    list[dict[str, float] | None],
    list[tuple[int, int, int, int]],
]:
    """Drop groups whose outward ring does not match detected background colour."""
    if not config.enable_final_offset_background_filter:
        return groups, noise_metrics, []

    filtered_groups: list[tuple[int, int, int, int]] = []
    filtered_metrics: list[dict[str, float] | None] = []
    rejected: list[tuple[int, int, int, int]] = []
    for group, metrics in zip(groups, noise_metrics):
        if metrics is None or not all(
            key in metrics for key in ("background_b", "background_g", "background_r")
        ):
            rejected.append(group)
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
            rejected.append(group)
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


@dataclass(slots=True)
class DetectionState:
    """Everything one pipeline step hands to the next."""

    boxes: np.ndarray
    groups: list[tuple[int, int, int, int]]
    noise_metrics: list[dict[str, float] | None]

    def copy(self) -> "DetectionState":
        return DetectionState(
            boxes=self.boxes.copy(),
            groups=list(self.groups),
            noise_metrics=[
                dict(metrics) if metrics is not None else None
                for metrics in self.noise_metrics
            ],
        )


def _step_mser(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig,
    state: DetectionState,
) -> tuple[DetectionState, DetectionStage]:
    masked = config.enable_uv_mask_before_mser and uv_mask is not None
    source = apply_uv_mask_for_mser(image, uv_mask) if masked else image
    grey = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    boxes = detect_mser_boxes(grey, config)
    return DetectionState(boxes, [], []), DetectionStage(
        key="mser",
        title="MSER boxes",
        kept=tuple(tuple(int(v) for v in box) for box in boxes),  # type: ignore[misc]
        detail=(
            f"delta={config.delta}, area {config.min_area}-{config.max_area} px"
            + (", UV-masked" if masked else "")
        ),
    )


def _step_flat_box(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig,
    state: DetectionState,
) -> tuple[DetectionState, DetectionStage]:
    boxes, rejected = filter_boxes_by_flat_colour(image, uv_mask, state.boxes, config)
    return DetectionState(boxes, [], []), DetectionStage(
        key="flat_box",
        title="Flat-colour boxes",
        kept=tuple(tuple(int(v) for v in box) for box in boxes),  # type: ignore[misc]
        rejected=tuple(rejected),
        detail=(
            f"need >= {config.min_box_uv_coverage:.0%} of the box inside the UV domain; "
            f"reject at >= {config.flat_box_min_coverage:.0%} of the domain area "
            f"within {config.flat_box_colour_tolerance} of one colour"
        ),
    )


def _step_pattern_box(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig,
    state: DetectionState,
) -> tuple[DetectionState, DetectionStage]:
    boxes, rejected = filter_boxes_by_pattern(image, state.boxes, config)
    return DetectionState(boxes, [], []), DetectionStage(
        key="pattern_box",
        title="Repeating pattern (boxes)",
        kept=tuple(tuple(int(v) for v in box) for box in boxes),  # type: ignore[misc]
        rejected=tuple(rejected),
        detail=(
            f"reject above {config.max_pattern_autocorrelation} autocorrelation "
            f"at lags {config.pattern_box_min_period_px}-{config.pattern_box_max_period_px} px"
        ),
    )


def _step_grouped(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig,
    state: DetectionState,
) -> tuple[DetectionState, DetectionStage]:
    groups = group_boxes(state.boxes, image, config, uv_mask)
    return DetectionState(state.boxes, groups, []), DetectionStage(
        key="grouped",
        title="Grouped",
        kept=tuple(groups),
        detail=(
            f"merge {config.merge_distance_px} px + dilate {config.group_dilate_px} px, "
            f"union >= {config.min_group_union_region_px} px^2, "
            f"merged box must stay >= {config.min_box_uv_coverage:.0%} in the UV domain"
        ),
    )


def _step_pattern_group(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig,
    state: DetectionState,
) -> tuple[DetectionState, DetectionStage]:
    groups, rejected = filter_groups_by_pattern(image, state.groups, config)
    return DetectionState(state.boxes, groups, []), DetectionStage(
        key="pattern_group",
        title="Repeating pattern (groups)",
        kept=tuple(groups),
        rejected=tuple(rejected),
        detail=(
            f"reject above {config.max_pattern_group_autocorrelation} autocorrelation "
            f"at lags {config.pattern_box_min_period_px}-{config.pattern_box_max_period_px} px"
        ),
    )


def _step_uv_refine(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig,
    state: DetectionState,
) -> tuple[DetectionState, DetectionStage]:
    groups, adjusted, discarded, metrics = refine_groups_with_uv_magic_wand(
        image, state.groups, uv_mask, config
    )
    return DetectionState(state.boxes, groups, metrics), DetectionStage(
        key="uv_refine",
        title="UV magic wand",
        kept=tuple(groups),
        rejected=tuple(discarded),
        adjusted=adjusted,
        detail=(
            f"colour threshold {config.magic_wand_colour_thresh}, "
            f"min island {config.magic_wand_min_island_area_px} px^2"
        ),
    )


def _step_noise(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig,
    state: DetectionState,
) -> tuple[DetectionState, DetectionStage]:
    # Measure here rather than relying on the magic wand having run, so the
    # stage works the same with refinement switched off.
    blocked = occupied_region_mask(
        image.shape, list(state.groups), max(config.contrast_surround_gap_px, 0)
    )
    measured: list[dict[str, float] | None] = []
    for group, existing in zip(state.groups, state.noise_metrics):
        # Large regions skip the test: a noisy detection is an MSER failure at
        # small scale, and there is no such thing as a large noise blob here.
        if group[2] * group[3] > config.max_contrast_region_area_px:
            measured.append(existing)
            continue
        contrast = region_contrast_metrics(image, uv_mask, group, config, blocked)
        if contrast is None:
            measured.append(existing)
            continue
        measured.append({**(existing or {}), **contrast})

    groups, metrics, rejected = filter_groups_by_magic_wand_noise(
        state.groups, measured, config
    )
    return DetectionState(state.boxes, groups, metrics), DetectionStage(
        key="noise",
        title="Contrast",
        kept=tuple(groups),
        rejected=tuple(rejected),
        detail=(
            f"Delta-E >= {config.min_contrast_delta_e}, "
            f"contrast-to-noise >= {config.min_contrast_to_noise}"
        ),
    )


def _step_size(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig,
    state: DetectionState,
) -> tuple[DetectionState, DetectionStage]:
    groups, metrics, rejected = filter_groups_by_final_size(
        state.groups, state.noise_metrics, config
    )
    return DetectionState(state.boxes, groups, metrics), DetectionStage(
        key="size",
        title="Final size",
        kept=tuple(groups),
        rejected=tuple(rejected),
        detail=(
            f"min {config.final_min_width_px}x{config.final_min_height_px} px, "
            f"min area {config.final_min_area_px} px^2, "
            f"aspect <= {config.final_max_aspect}"
        ),
    )


def _step_flat(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig,
    state: DetectionState,
) -> tuple[DetectionState, DetectionStage]:
    groups, metrics, rejected = filter_groups_by_single_colour(
        image, state.groups, state.noise_metrics, config
    )
    return DetectionState(state.boxes, groups, metrics), DetectionStage(
        key="flat",
        title="Single colour",
        kept=tuple(groups),
        rejected=tuple(rejected),
        detail=f"dominant colour fraction < {config.final_single_colour_fraction}",
    )


def _step_offset_background(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig,
    state: DetectionState,
) -> tuple[DetectionState, DetectionStage]:
    groups, metrics, rejected = filter_groups_by_offset_background(
        image, state.groups, state.noise_metrics, config
    )
    return DetectionState(state.boxes, groups, metrics), DetectionStage(
        key="offset_background",
        title="Offset background",
        kept=tuple(groups),
        rejected=tuple(rejected),
        detail=(
            f"{config.final_offset_background_width_px} px ring, "
            f">= {config.final_offset_background_min_fraction} matching"
        ),
    )


# Stages holding assembled regions rather than raw MSER boxes; only these are
# worth shaping, and only these are few enough for it to be cheap.
GROUP_STAGE_KEYS = frozenset(
    {
        "grouped",
        "pattern_group",
        "uv_refine",
        "noise",
        "size",
        "flat",
        "offset_background",
        "region_domain",
    }
)

def _step_region_domain(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig,
    state: DetectionState,
) -> tuple[DetectionState, DetectionStage]:
    """Drop regions that do not fit inside the UV domain once shaped."""
    if not config.enable_region_domain_filter or uv_mask is None:
        return state, DetectionStage(
            key="region_domain",
            title="Fits the domain",
            kept=tuple(state.groups),
            detail="disabled" if uv_mask is not None else "no UV mask",
        )

    kept: list[tuple[int, int, int, int]] = []
    metrics: list[dict[str, float] | None] = []
    rejected: list[tuple[int, int, int, int]] = []
    for group, group_metrics in zip(state.groups, state.noise_metrics):
        radius = inscribed_circle_radius(image, uv_mask, group, config)
        if radius:
            x, y, w, h = group
            coverage = circle_uv_coverage(
                uv_mask, (x + w / 2.0, y + h / 2.0), float(radius)
            )
        else:
            coverage = box_uv_coverage(uv_mask, group, image.shape)
        if coverage < config.min_region_uv_coverage:
            rejected.append(group)
            continue
        kept.append(group)
        metrics.append(group_metrics)

    return DetectionState(state.boxes, kept, metrics), DetectionStage(
        key="region_domain",
        title="Fits the domain",
        kept=tuple(kept),
        rejected=tuple(rejected),
        detail=(
            f"shaped region must be >= {config.min_region_uv_coverage:.0%} "
            "inside the UV domain"
        ),
    )


PIPELINE_STEPS = (
    _step_mser,
    _step_flat_box,
    _step_pattern_box,
    _step_grouped,
    _step_pattern_group,
    _step_uv_refine,
    _step_noise,
    _step_size,
    _step_flat,
    _step_offset_background,
    _step_region_domain,
)

# Which step first reads each parameter.  A parameter read by more than one step
# maps to the earliest.  Anything missing here forces a full re-run, so a new
# MserConfig field is safe by default -- it just will not resume.
PARAMETER_STEP = {
    "delta": 0,
    "min_area": 0,
    "max_area": 0,
    "max_variation": 0,
    "min_diversity": 0,
    "enable_min_component_area_filter": 0,
    "min_component_area_px": 0,
    "enable_mser_area_fraction_filter": 0,
    "max_area_fraction": 0,  # also the group area filter
    "enable_aspect_ratio_filter": 0,
    "min_aspect": 0,
    "max_aspect": 0,
    "enable_uv_mask_before_mser": 0,
    "enable_flat_box_filter": 1,
    "flat_box_colour_tolerance": 1,
    "flat_box_min_coverage": 1,
    "flat_box_min_domain_px": 1,
    "flat_box_context_px": 1,
    "min_box_uv_coverage": 1,
    "enable_pattern_box_filter": 2,
    "pattern_box_window_scale": 2,  # also the group pattern filter
    "pattern_box_min_window_px": 2,
    "pattern_box_min_period_px": 2,
    "pattern_box_max_period_px": 2,
    "max_pattern_autocorrelation": 2,
    "merge_distance_px": 3,
    "group_dilate_px": 3,
    "min_group_union_region_px": 3,
    "enable_group_area_filter": 3,
    "enable_group_degenerate_filter": 3,
    "enable_island_bounded_grouping": 3,
    "max_group_uv_coverage_drop": 3,
    "enable_circular_groups": 3,
    "circular_group_min_squareness": 3,
    "circular_group_padding_px": 3,
    "circular_group_colour_tolerance": 3,
    "circular_group_max_corner_content": 3,
    "enable_pattern_group_filter": 4,
    "pattern_group_window_scale": 4,
    "pattern_group_max_period_px": 4,
    "max_pattern_group_autocorrelation": 4,
    "enable_uv_magic_wand_refine": 5,
    "require_uv_magic_wand_refine": 5,
    "magic_wand_colour_thresh": 5,
    "magic_wand_output_padding_px": 5,
    "magic_wand_min_island_area_px": 5,
    "enable_contrast_filter": 6,
    "contrast_background_tolerance": 6,
    "contrast_surround_px": 6,
    "contrast_surround_scale": 6,
    "contrast_surround_gap_px": 6,
    "max_contrast_region_area_px": 6,
    "min_contrast_background_px": 6,
    "min_contrast_delta_e": 6,
    "min_contrast_to_noise": 6,
    "enable_final_size_filter": 7,
    "final_min_width_px": 7,
    "final_min_height_px": 7,
    "final_min_area_px": 7,
    "enable_final_aspect_filter": 7,
    "final_max_aspect": 7,
    "enable_final_single_colour_filter": 8,
    "final_single_colour_quant_step": 8,
    "final_single_colour_fraction": 8,
    "enable_final_offset_background_filter": 9,
    "final_offset_background_width_px": 9,
    "final_offset_background_colour_tol": 9,
    "final_offset_background_min_fraction": 9,
    "enable_region_domain_filter": 10,
    "min_region_uv_coverage": 10,
    "uv_island_mask_path": len(PIPELINE_STEPS),  # resolved before detection runs
    "green_colour": len(PIPELINE_STEPS),
    "red_colour": len(PIPELINE_STEPS),
    "green_thickness": len(PIPELINE_STEPS),
    "red_thickness": len(PIPELINE_STEPS),
}


@dataclass(slots=True)
class DetectionRun:
    """A completed pipeline run, resumable by a later run."""

    config: MserConfig
    stages: list[DetectionStage]
    noise_metrics: list[dict[str, float] | None]
    entry_states: list[DetectionState]  # state each step started from
    resumed_from: int = 0


def first_changed_step(previous: MserConfig, current: MserConfig) -> int:
    """Return the earliest pipeline step affected by a config change."""
    earliest = len(PIPELINE_STEPS)
    for field in fields(MserConfig):
        if getattr(previous, field.name) == getattr(current, field.name):
            continue
        earliest = min(earliest, PARAMETER_STEP.get(field.name, 0))
    return earliest


def run_detection(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig | None = None,
    previous: DetectionRun | None = None,
) -> DetectionRun:
    """Run the pipeline, resuming from the first step a config change affects.

    Steps before that are reused verbatim from ``previous``, so tuning a late
    filter does not re-run MSER over the whole atlas.
    """
    config = config or DEFAULT_CONFIG
    start = 0
    if previous is not None and len(previous.entry_states) == len(PIPELINE_STEPS):
        start = first_changed_step(previous.config, config)
        if start >= len(PIPELINE_STEPS):
            # Nothing the pipeline reads changed, so the previous run still
            # stands; re-running would only reproduce it.
            return DetectionRun(
                config=config,
                stages=list(previous.stages),
                noise_metrics=previous.noise_metrics,
                entry_states=previous.entry_states,
                resumed_from=start,
            )

    stages = list(previous.stages[:start]) if previous is not None and start else []
    entry_states = list(previous.entry_states[:start]) if previous is not None and start else []
    state = (
        previous.entry_states[start].copy()
        if previous is not None and start
        else DetectionState(np.empty((0, 4), dtype=np.int32), [], [])
    )

    for index in range(start, len(PIPELINE_STEPS)):
        entry_states.append(state.copy())
        state, stage = PIPELINE_STEPS[index](image, uv_mask, config, state)
        if stage.key in GROUP_STAGE_KEYS and config.enable_circular_groups:
            # Shape is a property of the region's pixels, so it is derived per
            # stage rather than carried: a stage that moves a box reshapes it.
            stage = replace(
                stage,
                circles=tuple(
                    inscribed_circle_radius(image, uv_mask, group, config)
                    for group in stage.kept
                ),
            )
        stages.append(stage)

    return DetectionRun(
        config=config,
        stages=stages,
        noise_metrics=state.noise_metrics,
        entry_states=entry_states,
        resumed_from=start,
    )


def detect_texture_region_stages(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig | None = None,
) -> tuple[list[DetectionStage], list[dict[str, float] | None]]:
    """Run detection and return every intermediate stage."""
    run = run_detection(image, uv_mask, config)
    return run.stages, run.noise_metrics


def detect_texture_regions(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig | None = None,
) -> dict[str, object]:
    """Run texture-region detection on an already loaded BGR image."""
    stages, noise_metrics = detect_texture_region_stages(image, uv_mask, config)
    by_key = {stage.key: stage for stage in stages}
    groups = by_key["region_domain"].kept

    return {
        "image_size": f"{image.shape[1]}x{image.shape[0]}",
        "mser_boxes": len(by_key["mser"].kept),
        "flat_boxes_rejected": len(by_key["flat_box"].rejected),
        "pattern_boxes_rejected": len(by_key["pattern_box"].rejected),
        "candidate_groups": len(by_key["grouped"].kept),
        "pattern_groups_rejected": len(by_key["pattern_group"].rejected),
        "uv_magic_wand_adjusted": by_key["uv_refine"].adjusted,
        "uv_magic_wand_discarded": len(by_key["uv_refine"].rejected),
        "magic_wand_noise_rejected": len(by_key["noise"].rejected),
        "final_size_rejected": len(by_key["size"].rejected),
        "final_single_colour_rejected": len(by_key["flat"].rejected),
        "final_offset_background_rejected": len(by_key["offset_background"].rejected),
        "region_domain_rejected": len(by_key["region_domain"].rejected),
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


def write_annotation_outputs(
    image: np.ndarray,
    summary: dict[str, object],
    outputs: TextureRegionAnnotationOutputs,
    config: MserConfig,
) -> None:
    """Write requested external annotation artifacts."""
    annotated = draw_annotations(
        image,
        [
            (
                int(group["x"]),
                int(group["y"]),
                int(group["w"]),
                int(group["h"]),
            )
            for group in summary["groups"]
        ],
        config,
    )

    outputs.annotated_image.parent.mkdir(parents=True, exist_ok=True)
    try:
        wrote_output = cv2.imwrite(str(outputs.annotated_image), annotated)
    except cv2.error as exc:
        raise ValueError(
            f"OpenCV could not write output image: {outputs.annotated_image}"
        ) from exc
    if not wrote_output:
        raise ValueError(f"OpenCV could not write output image: {outputs.annotated_image}")

    if outputs.summary_json is not None:
        outputs.summary_json.parent.mkdir(parents=True, exist_ok=True)
        outputs.summary_json.write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )


def annotate_texture_regions(
    texture_path: Path,
    outputs: TextureRegionAnnotationOutputs,
    uv_island_mask_path: Path | str | None = None,
    config: MserConfig | None = None,
) -> dict[str, object]:
    """Detect texture regions and write caller-specified output files."""
    config = config_with_uv_mask_path(config or DEFAULT_CONFIG, uv_island_mask_path)
    image = load_image(texture_path)

    uv_mask = None
    if (
        config.uv_island_mask_path
        and (
            config.enable_uv_mask_before_mser
            or config.enable_uv_magic_wand_refine
        )
    ):
        uv_mask = load_uv_island_mask(
            resolve_config_path(config.uv_island_mask_path),
            image.shape,
        )

    summary = detect_texture_regions(image, uv_mask, config)
    summary.update(
        {
            "input": str(texture_path),
            "output": str(outputs.annotated_image),
            "uv_island_mask": str(uv_island_mask_path or config.uv_island_mask_path),
        }
    )
    if outputs.summary_json is not None:
        summary["summary_json"] = str(outputs.summary_json)

    write_annotation_outputs(image, summary, outputs, config)
    return summary


def annotate_texture(
    input_path: Path,
    output_path: Path,
    config: MserConfig | None = None,
) -> dict[str, object]:
    """Backward-compatible wrapper for the original CLI-style call."""
    config = config or DEFAULT_CONFIG
    return annotate_texture_regions(
        input_path,
        TextureRegionAnnotationOutputs(annotated_image=output_path),
        config.uv_island_mask_path,
        config,
    )


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
    parser.add_argument(
        "--uv-mask",
        type=Path,
        default=None,
        help="UV island mask path. Defaults to DEFAULT_CONFIG.uv_island_mask_path.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional JSON summary output path.",
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
        summary = annotate_texture_regions(
            input_path,
            TextureRegionAnnotationOutputs(
                annotated_image=output_path,
                summary_json=args.summary_json,
            ),
            args.uv_mask,
            config,
        )
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
            contrast_text = "n/a"
        else:
            contrast_text = (
                f"dE={noise_metrics['delta_e']:.1f} "
                f"cnr={noise_metrics['contrast_to_noise']:.1f} "
                f"bgstd={noise_metrics['background_std']:.1f} "
                f"offsetbg={noise_metrics.get('offset_background_fraction', 0.0):.3f}"
            )
        print(
            f"  [{index:3d}]  x={group['x']:5d}  y={group['y']:5d}  "
            f"w={group['w']:4d}  h={group['h']:4d}  {contrast_text}"
        )


if __name__ == "__main__":
    main()
