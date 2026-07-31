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
from dataclasses import dataclass, field, fields, replace
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
    delta:         int   = 12  # default: 5
    min_area:      int   = 10  # default: 30
    max_area:      int   = 10000  # default: 1_024
    max_variation: float = 1.0  # default: 0.25
    min_diversity: float = 0.1  # default: 0.2

    # Post-detection filtering.  MSER's own max_area already caps a region far
    # below any sane fraction of the atlas, so no image-fraction test is applied
    # to boxes; measured over eight 4k dashboards it rejected nothing.
    enable_aspect_ratio_filter:       bool  = True  # default: True
    min_aspect:                       float = 0.1  # default: 0.1
    max_aspect:                       float = 10.0  # default: 10.0

    # Early box filters, applied to raw MSER boxes before grouping.  The test
    # that survives is the feature-blob size: an MSER box is homogeneous by
    # definition, so it is judged together with a ring of context, and what
    # separates a glyph stroke from blank trim is whether the non-background
    # pixels form one whole run or break into speckle.
    enable_box_feature_filter:   bool  = True  # default: True
    min_box_width_px:            int   = 3  # default: 3
    min_box_height_px:           int   = 3  # default: 3
    box_feature_colour_tolerance: int  = 16  # default: 16 (magic-wand sensitivity)
    box_feature_min_domain_px:   int   = 8  # default: 8
    box_feature_context_px:      int   = 3  # default: 3 (an MSER box alone is always flat)
    min_box_uv_coverage:         float = 0.75  # default: 0.75 (share of the box inside the UV domain)
    box_min_feature_px:          int   = 24  # default: 24 (largest non-background blob)
    # Repeating-pattern rejection, applied to assembled groups.  A group covers
    # a whole panel, so the statistic sees several periods; the same test on a
    # single MSER box separated nothing.  Measured on the dash, perforated
    # grilles score 0.93-0.95 and the chevron weave 0.61-0.63, while every glyph
    # stays at or below 0.29.
    enable_pattern_group_filter: bool  = True  # default: True
    max_pattern_autocorrelation: float = 0.45  # default: 0.45 (reject above)
    pattern_window_scale:        float = 1.0  # default: 1.0 (the group itself)
    pattern_min_window_px:       int   = 48  # default: 48
    pattern_min_period_px:       int   = 4  # default: 4
    pattern_max_period_px:       int   = 160  # default: 160

    # Grouping
    # Boxes are expanded by this many pixels on every side before the
    # minimum union-area test.  This replaces the old additive
    # merge_distance_px + group_dilate_px pair.
    #
    # No size or area cap is applied while candidates are assembled.  A group
    # that comes out too broad still owns its exact member boxes, and domain
    # recovery below can split it back down; dropping it here would throw those
    # members away, which is the one case recovery exists to handle.
    merge_distance_px:              int  = 18  # default: 18
    min_group_union_region_px:      int  = 1  # default: 1 (any overlap merges)
    enable_island_bounded_grouping: bool = True  # default: True
    enable_overlap_group_merge:     bool = True  # default: True

    # Round regions: inscribe a circle in a squarish group and keep it when the
    # corners it drops are one solid colour or contain no UV domain at all.
    enable_circular_groups:            bool  = True  # default: True
    circular_group_min_squareness:     float = 0.80  # default: 0.80 (min side / max side)
    circular_group_padding_px:         int   = 3  # default: 3 (grown past a strict inscribe)
    circular_group_colour_tolerance:   int   = 24  # default: 24
    circular_group_max_corner_content: float = 0.05  # default: 0.05 of the corner area

    # UV island mask used when the caller does not pass one in
    uv_island_mask_path:                   str   = "mesh_segmentation_transform/segmentation_outputs/scintilla_interior_b.color.full_uv_filled_mask.png"  # black UV islands on white background
    # Final size and shape
    enable_final_size_filter:              bool  = True  # default: True
    final_min_width_px:                    int   = 4  # default: 4
    final_min_height_px:                   int   = 4  # default: 4
    final_min_area_px:                     int   = 50  # default: 50
    enable_final_aspect_filter:            bool  = True  # default: True
    final_max_aspect:                      float = 12.0  # default: 12.0 (long side / short)

    # Two-pass shaped-domain recovery.  An initial group is tested after circle
    # inference.  A failing group is split back into its exact source boxes,
    # invalid boxes are removed and survivors are regrouped.  A broad rebuilt
    # group that still fails is retried once at half the merge distance.
    enable_region_domain_filter:           bool  = True  # default: True
    min_region_uv_coverage:                float = 0.98  # default: 0.98

    # Terminal output shaping.  Applied only after every detection/filter stage,
    # so padding cannot cause a region to fail an earlier size, colour, or domain
    # test.  Zero leaves the detected bounds unchanged.
    final_region_padding_px:               int   = 2  # default: 2

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
class UvIslandSymmetryConfig:
    """Independent UV-island reflection-symmetry annotation settings.

    These settings are not read by the MSER, grouping, or clean-up pipeline.
    They only control a blue contour overlay derived directly from the UV mask.
    """

    enable_uv_island_symmetry: bool = True
    min_uv_island_symmetry: float = 0.98
    blue_colour: tuple[int, int, int] = (255, 0, 0)  # BGR
    blue_thickness: int = 1


DEFAULT_UV_ISLAND_SYMMETRY_CONFIG = UvIslandSymmetryConfig()


@dataclass(frozen=True, slots=True)
class UvIslandSymmetryMatch:
    """One connected UV island accepted by either reflection-axis test."""

    label: int
    bounds: tuple[int, int, int, int]
    area_px: int
    x_similarity: float
    y_similarity: float
    matched_axes: tuple[str, ...]
    contours: tuple[tuple[tuple[int, int], ...], ...]


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
    adjusted: int = 0  # regions this stage moved, expanded, or absorbed
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


def _reflection_similarity(
    component: np.ndarray,
    reflected: np.ndarray,
) -> float:
    """Return intersection-over-union agreement with a reflected island mask."""
    union = component | reflected
    union_area = int(union.sum())
    if union_area == 0:
        return 1.0
    intersection = component & reflected
    return float(intersection.sum()) / float(union_area)


def analyse_uv_island_symmetry(
    uv_mask: np.ndarray | None,
    config: UvIslandSymmetryConfig | None = None,
) -> tuple[UvIslandSymmetryMatch, ...]:
    """Find connected UV islands symmetric about either local bounding-box axis.

    X-axis symmetry compares the island with a top/bottom reflection.  Y-axis
    symmetry compares it with a left/right reflection.  Similarity is measured
    as mask intersection-over-union, so 1.0 is exact and lower values tolerate
    an increasing fraction of asymmetric rasterised area.
    """
    config = config or DEFAULT_UV_ISLAND_SYMMETRY_CONFIG
    if (
        not config.enable_uv_island_symmetry
        or uv_mask is None
        or not bool(uv_mask.any())
    ):
        return ()

    threshold = min(max(float(config.min_uv_island_symmetry), 0.0), 1.0)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        uv_mask.astype(np.uint8), connectivity=8
    )
    matches: list[UvIslandSymmetryMatch] = []

    for label in range(1, count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        component = labels[y : y + h, x : x + w] == label

        x_similarity = _reflection_similarity(component, np.flipud(component))
        y_similarity = _reflection_similarity(component, np.fliplr(component))
        axes = tuple(
            axis
            for axis, score in (("x", x_similarity), ("y", y_similarity))
            if score >= threshold
        )
        if not axes:
            continue

        local_contours, _hierarchy = cv2.findContours(
            component.astype(np.uint8),
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        contours: list[tuple[tuple[int, int], ...]] = []
        for contour in local_contours:
            points = contour.reshape(-1, 2)
            if points.size == 0:
                continue
            contours.append(
                tuple((int(px) + x, int(py) + y) for px, py in points)
            )

        matches.append(
            UvIslandSymmetryMatch(
                label=label,
                bounds=(x, y, w, h),
                area_px=area,
                x_similarity=x_similarity,
                y_similarity=y_similarity,
                matched_axes=axes,
                contours=tuple(contours),
            )
        )

    return tuple(matches)


def draw_uv_island_symmetry_annotations(
    image: np.ndarray,
    matches: tuple[UvIslandSymmetryMatch, ...],
    config: UvIslandSymmetryConfig | None = None,
) -> np.ndarray:
    """Draw accepted UV-island borders blue without changing detection state."""
    config = config or DEFAULT_UV_ISLAND_SYMMETRY_CONFIG
    if not config.enable_uv_island_symmetry or not matches:
        return image

    annotated = image.copy()
    thickness = max(int(config.blue_thickness), 1)
    for match in matches:
        for contour in match.contours:
            if not contour:
                continue
            points = np.asarray(contour, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(
                annotated,
                [points],
                True,
                config.blue_colour,
                thickness,
                cv2.LINE_8,
            )
    return annotated


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

    Each row is ``[x, y, w, h]`` in pixel coordinates.  Only the aspect-ratio
    test is applied here.  Size is left to ``config.min_area``/``max_area``,
    which MSER enforces on the region itself rather than on its bounding box,
    and to the minimum box dimensions in the next stage.
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

    boxes: list[tuple[int, int, int, int]] = []
    for region in regions:
        x, y, w, h = cv2.boundingRect(region)
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


def box_feature_blob_px(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    box: tuple[int, int, int, int],
    config: MserConfig,
) -> int | None:
    """Return the area of a box's largest connected non-background run.

    Speckle noise on blank trim breaks into tiny fragments, while a glyph
    stroke stays whole, so this one number separates the two.  The share of the
    box that is a single colour was measured alongside it for a long time and
    dropped: across eight 4k dashboards it rejected between zero and six boxes,
    none of which survived to affect a final region.

    Uses the magic wand's colour model: the dominant colour is estimated inside
    the UV domain, and anything further than ``box_feature_colour_tolerance``
    from it on any channel counts as feature.  ``None`` means the box has too
    little UV domain to judge.

    The test covers the box plus ``box_feature_context_px``, because an MSER
    region is homogeneous by definition -- measured on the bare box, a glyph
    stroke has no background to stand out against.  With a ring of context, the
    stroke's surroundings come in and the stroke becomes the feature.
    """
    inner = box
    box = context_box(box, config.box_feature_context_px)
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
    if int(domain.sum()) < max(config.box_feature_min_domain_px, 1):
        return None

    tolerance = max(config.box_feature_colour_tolerance, 0)
    dominant = estimate_background_colour(crop, domain, max(tolerance, 1))
    if dominant is None:
        return None

    feature = domain & ~channel_colour_mask(crop, domain, dominant, tolerance)
    if not bool(feature.any()):
        return 0
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        feature.astype(np.uint8), connectivity=8
    )
    return max(
        (int(stats[label, cv2.CC_STAT_AREA]) for label in range(1, count)),
        default=0,
    )


def filter_boxes_by_feature(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    boxes: np.ndarray,
    config: MserConfig,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Drop undersized, featureless, or out-of-domain MSER boxes.

    The pre-MSER fill makes every island silhouette a hard edge, so MSER finds a
    few regions straddling the boundary.  Those carry no geometry over most of
    their area and would otherwise drag groups out across the gaps.
    """
    if len(boxes) == 0:
        return boxes, []

    kept: list[tuple[int, int, int, int]] = []
    rejected: list[tuple[int, int, int, int]] = []
    for raw in boxes:
        box = (int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3]))
        if (
            box[2] < max(config.min_box_width_px, 1)
            or box[3] < max(config.min_box_height_px, 1)
        ):
            rejected.append(box)
            continue
        if not config.enable_box_feature_filter:
            kept.append(box)
            continue
        if box_uv_coverage(uv_mask, box, image.shape) < config.min_box_uv_coverage:
            rejected.append(box)
            continue
        feature_px = box_feature_blob_px(image, uv_mask, box, config)
        if feature_px is None or feature_px < config.box_min_feature_px:
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

    The measure only separates anything once it sees several periods, so it is
    applied to assembled groups and not to single MSER boxes.  Even then it is
    narrow: across eight 4k dashboards it changed the outcome on one.
    """
    patch = window.astype(np.float32)
    patch -= float(patch.mean())
    if float(np.abs(patch).max()) < 1e-3:
        return 0.0  # featureless: the flat-colour filter owns this case

    height, width = patch.shape
    max_lag = min(
        max(config.pattern_max_period_px, 1),
        (height - 1) // 2,
        (width - 1) // 2,
    )
    min_lag = max(config.pattern_min_period_px, 2)
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
    """Return a square grey neighbourhood centred on a box.

    The side is the box's longer edge, so a wide, short group is scored over a
    tall square that reaches well past it.  That is deliberate for a weave,
    which continues beyond whatever slice of it MSER happened to group, but it
    does dilute the statistic for a long strip of text.
    """
    x, y, w, h = box
    target = max(
        int(round(max(w, h) * max(config.pattern_window_scale, 1.0))),
        max(config.pattern_min_window_px, 4),
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
    inscribed circle carry nothing: either one locally solid colour, or no UV
    domain at all.  The corner colour does not need to match the dominant colour
    elsewhere in the group.  When that holds the circle is the honest region and
    the corners are dead area; when anything else lives there the square is kept.

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

    # Judge the discarded ring against its own dominant colour, not against the
    # primary background colour of the complete group.  This permits a uniformly
    # coloured bezel/ring even when the gauge face uses a different background.
    ring_colour = estimate_background_colour(
        crop, corners, config.circular_group_colour_tolerance
    )
    if ring_colour is None:
        return None
    matching = channel_colour_mask(
        crop, corners, ring_colour, config.circular_group_colour_tolerance
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
    box could not separate anything.  See ``pattern_window_for_box`` for the
    window the score is actually measured over.
    """
    if not config.enable_pattern_group_filter or not groups:
        return groups, []

    kept: list[tuple[int, int, int, int]] = []
    rejected: list[tuple[int, int, int, int]] = []
    for group in groups:
        window = pattern_window_for_box(image, group, config)
        if window is None:
            kept.append(group)
            continue
        score = repeating_pattern_score(window, config)
        if score > config.max_pattern_autocorrelation:
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


@dataclass(frozen=True, slots=True)
class GroupCandidate:
    """One proximity group together with the exact boxes that formed it."""

    bounds: tuple[int, int, int, int]
    members: tuple[tuple[int, int, int, int], ...]


@dataclass(frozen=True, slots=True)
class UvDomainIndex:
    """Whole-mask derivatives shared by every grouping call in one run.

    Domain recovery regroups the members of each failed candidate separately,
    so without this the connected-component pass over the entire atlas ran once
    per failure -- work that does not depend on the boxes being regrouped.  The
    labels are read-only and safe to share between calls and between resumed
    runs of the same texture.
    """

    island_labels: np.ndarray


def build_uv_domain_index(uv_mask: np.ndarray | None) -> UvDomainIndex | None:
    """Label the UV islands once, for reuse by every grouping call."""
    if uv_mask is None:
        return None
    _count, labels = cv2.connectedComponents(
        uv_mask.astype(np.uint8), connectivity=8
    )
    return UvDomainIndex(island_labels=labels)


def union_region_group_candidates(
    boxes: np.ndarray,
    image: np.ndarray,
    config: MserConfig,
    uv_mask: np.ndarray | None = None,
    domain: UvDomainIndex | None = None,
) -> list[GroupCandidate]:
    """Group boxes by proximity and retain exact membership for later recovery.

    Deliberately permissive, and deliberately without any cap on the result: a
    group that comes out too broad is handed to domain recovery, which can
    split it back into the boxes it was built from.  Discarding it here would
    discard those boxes with it.

    ``domain`` is the shared island labelling; it is built on demand when the
    caller has none.
    """
    if len(boxes) == 0:
        return []

    height, width = image.shape[:2]
    distance = max(0, config.merge_distance_px)
    min_union_area = max(1, config.min_group_union_region_px)
    box_tuples = [tuple(int(value) for value in box) for box in boxes]
    expanded_boxes = [expanded_box(box, distance) for box in box_tuples]
    cell_size = max(distance + int(np.sqrt(min_union_area)), 16)
    parent = list(range(len(box_tuples)))
    bounds = [(x, y, x + w, y + h) for x, y, w, h in box_tuples]

    island = [0] * len(box_tuples)
    if uv_mask is not None and config.enable_island_bounded_grouping:
        if domain is None:
            domain = build_uv_domain_index(uv_mask)
        labels = domain.island_labels  # type: ignore[union-attr]
        for index, (bx, by, bw, bh) in enumerate(box_tuples):
            clamped = clamp_group((bx, by, bw, bh), uv_mask.shape)
            if clamped is None:
                continue
            cx, cy, cw, ch = clamped
            within = labels[cy : cy + ch, cx : cx + cw]
            within = within[within > 0]
            if within.size:
                island[index] = int(np.bincount(within).argmax())

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def merge(root: int, other_root: int) -> None:
        """Attach one resolved root to another, growing the surviving bounds."""
        first = bounds[root]
        second = bounds[other_root]
        parent[other_root] = root
        bounds[root] = (
            min(first[0], second[0]),
            min(first[1], second[1]),
            max(first[2], second[2]),
            max(first[3], second[3]),
        )

    # Neighbour search over a uniform grid whose cell is the merge distance, so
    # two expanded boxes that overlap always share at least one cell.  Each box
    # is inserted only after it has been compared with the boxes already there,
    # which is what keeps every pair from being considered twice.
    grid: dict[tuple[int, int], list[int]] = {}
    for index, expanded in enumerate(expanded_boxes):
        checked: set[int] = set()
        cells = grid_cells_for_box(expanded, cell_size)
        for cell in cells:
            for other in grid.get(cell, ()):
                if other in checked:
                    continue
                checked.add(other)
                # Resolve membership before measuring any overlap.  Dense glyph
                # clusters put a box in a cell with hundreds of others, nearly
                # all of which it has already been merged with transitively --
                # 98% of pairs on the busiest atlas -- and a pair that cannot
                # merge does not need its overlap computed at all.
                root = find(index)
                other_root = find(other)
                if root == other_root or island[root] != island[other_root]:
                    continue
                if intersection_area(expanded, expanded_boxes[other]) >= min_union_area:
                    merge(root, other_root)
        for cell in cells:
            grid.setdefault(cell, []).append(index)

    members: dict[int, list[int]] = {}
    for index in range(len(box_tuples)):
        members.setdefault(find(index), []).append(index)

    candidates: list[GroupCandidate] = []
    for root, indices in members.items():
        rect = bounds[root]
        x0 = max(rect[0], 0)
        y0 = max(rect[1], 0)
        x1 = min(rect[2], width)
        y1 = min(rect[3], height)
        if x1 <= x0 or y1 <= y0:
            continue
        candidates.append(
            GroupCandidate(
                bounds=(x0, y0, x1 - x0, y1 - y0),
                members=tuple(box_tuples[index] for index in indices),
            )
        )

    candidates.sort(key=lambda candidate: (candidate.bounds[1], candidate.bounds[0]))
    return candidates


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


def dominant_quantized_bin(quantized: np.ndarray, levels: int) -> np.ndarray:
    """Return the most populated quantized BGR bin, given quantized pixels."""
    if levels**3 > 1_000_000:
        bins, counts = np.unique(quantized.reshape(-1, 3), axis=0, return_counts=True)
        return bins[int(np.argmax(counts))]

    codes = (quantized[:, 0] * levels + quantized[:, 1]) * levels + quantized[:, 2]
    dominant_code = int(np.bincount(codes).argmax())
    b = dominant_code // (levels * levels)
    remainder = dominant_code % (levels * levels)
    return np.asarray((b, remainder // levels, remainder % levels), dtype=np.int32)


def estimate_background_colour(
    crop: np.ndarray,
    domain_mask: np.ndarray,
    colour_thresh: int,
) -> np.ndarray | None:
    """Estimate the dominant background BGR colour inside the current domain.

    Quantises once and reuses that array for both the bin histogram and the
    membership test; this runs per MSER box, so it is the pipeline's hottest
    piece of arithmetic.
    """
    domain_pixels = crop[domain_mask]
    if len(domain_pixels) == 0:
        return None

    quant_step = max(colour_thresh, 1)
    quantized = (domain_pixels // quant_step).astype(np.int32)
    levels = (255 // quant_step) + 1
    dominant_matches = np.all(
        quantized == dominant_quantized_bin(quantized, levels), axis=1
    )
    if not bool(dominant_matches.any()):
        return None

    return np.median(domain_pixels[dominant_matches], axis=0).astype(np.int16)


def filter_groups_by_final_size(
    groups: list[tuple[int, int, int, int]],
    config: MserConfig,
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    """Drop remaining groups too small, or too elongated, to be a detection."""
    if not config.enable_final_size_filter and not config.enable_final_aspect_filter:
        return groups, []

    kept: list[tuple[int, int, int, int]] = []
    rejected: list[tuple[int, int, int, int]] = []
    for group in groups:
        _x, _y, w, h = group
        if config.enable_final_size_filter and (
            w < config.final_min_width_px
            or h < config.final_min_height_px
            or w * h < config.final_min_area_px
        ):
            rejected.append(group)
            continue
        if (
            config.enable_final_aspect_filter
            and max(w, h) / max(min(w, h), 1) > config.final_max_aspect
        ):
            rejected.append(group)
            continue
        kept.append(group)

    return kept, rejected


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
    candidates: list[GroupCandidate] = field(default_factory=list)
    # Built once by the grouping step and read by domain recovery.  Shared by
    # reference: the labelling is read-only and depends only on the UV mask,
    # which never changes within a run or across a resumed one.
    domain: UvDomainIndex | None = None

    def copy(self) -> "DetectionState":
        return DetectionState(
            boxes=self.boxes.copy(),
            groups=list(self.groups),
            candidates=list(self.candidates),
            domain=self.domain,
        )


def _step_mser(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig,
    state: DetectionState,
) -> tuple[DetectionState, DetectionStage]:
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    boxes = detect_mser_boxes(grey, config)
    return DetectionState(boxes, []), DetectionStage(
        key="mser",
        title="MSER boxes",
        kept=tuple(tuple(int(v) for v in box) for box in boxes),  # type: ignore[misc]
        detail=f"delta={config.delta}, area {config.min_area}-{config.max_area} px",
    )


def _step_box_filter(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig,
    state: DetectionState,
) -> tuple[DetectionState, DetectionStage]:
    boxes, rejected = filter_boxes_by_feature(image, uv_mask, state.boxes, config)
    return DetectionState(boxes, []), DetectionStage(
        key="box_filter",
        title="Box filtering",
        kept=tuple(tuple(int(v) for v in box) for box in boxes),  # type: ignore[misc]
        rejected=tuple(rejected),
        detail=(
            f"minimum {max(config.min_box_width_px, 1)}x"
            f"{max(config.min_box_height_px, 1)} px; "
            + (
                f"need >= {config.min_box_uv_coverage:.0%} of the box inside the UV "
                f"domain and a non-background blob of >= {config.box_min_feature_px} px "
                f"within {config.box_feature_colour_tolerance} of one colour"
                if config.enable_box_feature_filter
                else "feature and UV-domain checks disabled"
            )
        ),
    )


def _step_grouped(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig,
    state: DetectionState,
) -> tuple[DetectionState, DetectionStage]:
    """Create permissive initial groups and retain their exact members."""
    domain = build_uv_domain_index(uv_mask)
    candidates = union_region_group_candidates(
        state.boxes, image, config, uv_mask, domain
    )
    groups = [candidate.bounds for candidate in candidates]
    return DetectionState(state.boxes, groups, candidates, domain), DetectionStage(
        key="grouped",
        title="Initial grouping",
        kept=tuple(groups),
        detail=(
            f"boxes expanded by {config.merge_distance_px} px per side; "
            "no size cap here -- membership is retained so an over-broad group "
            "can be split by domain recovery"
        ),
    )


def _step_pattern_group(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig,
    state: DetectionState,
) -> tuple[DetectionState, DetectionStage]:
    groups, rejected = filter_groups_by_pattern(image, state.groups, config)
    return DetectionState(state.boxes, groups), DetectionStage(
        key="pattern_group",
        title="Repeating pattern (groups)",
        kept=tuple(groups),
        rejected=tuple(rejected),
        detail=(
            f"reject above {config.max_pattern_autocorrelation} autocorrelation "
            f"at lags {config.pattern_min_period_px}-{config.pattern_max_period_px} px"
        ),
    )


def _step_size(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig,
    state: DetectionState,
) -> tuple[DetectionState, DetectionStage]:
    groups, rejected = filter_groups_by_final_size(state.groups, config)
    return DetectionState(state.boxes, groups), DetectionStage(
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


def _region_shape_and_coverage(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    group: tuple[int, int, int, int],
    config: MserConfig,
) -> tuple[int | None, float]:
    """Infer a region's circle, then measure that circle or its rectangle."""
    radius = inscribed_circle_radius(image, uv_mask, group, config)
    if radius is not None:
        x, y, w, h = group
        return radius, circle_uv_coverage(
            uv_mask,
            (x + w / 2.0, y + h / 2.0),
            float(radius),
        )
    return None, box_uv_coverage(uv_mask, group, image.shape)


def _step_region_domain(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig,
    state: DetectionState,
) -> tuple[DetectionState, DetectionStage]:
    """Recover failed groups with a full-distance pass and a half-distance retry.

    Every initial group first gets its circle decision and shaped-domain test.
    A passing group remains intact.  A failing group is split back into the
    exact MSER boxes that formed it; boxes failing the same shaped-domain test
    are removed and the survivors are regrouped at the configured merge
    distance.  If a rebuilt group still fails, its already-valid members are
    regrouped once more at half the merge distance before any final rejection.
    """
    candidates = state.candidates or [
        GroupCandidate(bounds=group, members=(group,)) for group in state.groups
    ]
    domain = state.domain or build_uv_domain_index(uv_mask)
    if not config.enable_region_domain_filter or uv_mask is None:
        return state, DetectionStage(
            key="region_domain",
            title="Domain recovery",
            kept=tuple(state.groups),
            detail="disabled" if uv_mask is not None else "no UV mask",
        )

    kept_candidates: list[GroupCandidate] = []
    rejected: list[tuple[int, int, int, int]] = []
    failed_initial = 0
    removed_members = 0
    half_distance_attempts = 0
    half_distance_rescued = 0
    final_rejected = 0
    half_merge_distance = max(config.merge_distance_px // 2, 0)
    half_distance_config = replace(
        config,
        merge_distance_px=half_merge_distance,
    )

    for candidate in candidates:
        _radius, initial_coverage = _region_shape_and_coverage(
            image, uv_mask, candidate.bounds, config
        )
        if initial_coverage >= config.min_region_uv_coverage:
            kept_candidates.append(candidate)
            continue

        failed_initial += 1
        rejected.append(candidate.bounds)

        valid_members: list[tuple[int, int, int, int]] = []
        for member in candidate.members:
            _member_radius, member_coverage = _region_shape_and_coverage(
                image, uv_mask, member, config
            )
            if member_coverage < config.min_region_uv_coverage:
                rejected.append(member)
                removed_members += 1
            else:
                valid_members.append(member)

        if not valid_members:
            continue

        rebuilt = union_region_group_candidates(
            np.asarray(valid_members, dtype=np.int32),
            image,
            config,
            uv_mask,
            domain,
        )
        for rebuilt_candidate in rebuilt:
            _rebuilt_radius, rebuilt_coverage = _region_shape_and_coverage(
                image, uv_mask, rebuilt_candidate.bounds, config
            )
            if rebuilt_coverage >= config.min_region_uv_coverage:
                kept_candidates.append(rebuilt_candidate)
                continue

            # The full-distance rebuild is still too broad.  Do not discard all
            # of its individually valid members: reduce only this recovery pass
            # to half the normal merge distance and try once more.
            rejected.append(rebuilt_candidate.bounds)
            half_distance_attempts += 1
            retry_candidates = union_region_group_candidates(
                np.asarray(rebuilt_candidate.members, dtype=np.int32),
                image,
                half_distance_config,
                uv_mask,
                domain,
            )
            rescued_here = 0
            for retry_candidate in retry_candidates:
                _retry_radius, retry_coverage = _region_shape_and_coverage(
                    image, uv_mask, retry_candidate.bounds, config
                )
                if retry_coverage < config.min_region_uv_coverage:
                    rejected.append(retry_candidate.bounds)
                    final_rejected += 1
                    continue
                kept_candidates.append(retry_candidate)
                rescued_here += 1
            half_distance_rescued += rescued_here

    kept_candidates.sort(key=lambda candidate: (candidate.bounds[1], candidate.bounds[0]))
    kept = [candidate.bounds for candidate in kept_candidates]
    return DetectionState(state.boxes, kept, kept_candidates, domain), DetectionStage(
        key="region_domain",
        title="Domain recovery",
        kept=tuple(kept),
        rejected=tuple(rejected),
        detail=(
            f"first shaped test >= {config.min_region_uv_coverage:.0%}; "
            f"{failed_initial} initial group{'s' if failed_initial != 1 else ''} split, "
            f"{removed_members} invalid member{'s' if removed_members != 1 else ''} removed; "
            f"{half_distance_attempts} broad rebuild"
            f"{'s' if half_distance_attempts != 1 else ''} retried at "
            f"{half_merge_distance} px, {half_distance_rescued} retry group"
            f"{'s' if half_distance_rescued != 1 else ''} rescued, "
            f"{final_rejected} final group{'s' if final_rejected != 1 else ''} rejected"
        ),
    )



def _shaped_regions_overlap(
    first: tuple[int, int, int, int],
    first_radius: int | None,
    second: tuple[int, int, int, int],
    second_radius: int | None,
) -> bool:
    """Return whether two final shaped regions overlap with positive area.

    Circular groups are tested as discs rather than by their square source
    bounds.  Tangential contact alone is not overlap: the shapes must share
    actual interior area.
    """
    ax, ay, aw, ah = first
    bx, by, bw, bh = second

    if first_radius is None and second_radius is None:
        return (
            max(ax, bx) < min(ax + aw, bx + bw)
            and max(ay, by) < min(ay + ah, by + bh)
        )

    if first_radius is not None and second_radius is not None:
        first_centre_x = ax + aw / 2.0
        first_centre_y = ay + ah / 2.0
        second_centre_x = bx + bw / 2.0
        second_centre_y = by + bh / 2.0
        distance_sq = (
            (first_centre_x - second_centre_x) ** 2
            + (first_centre_y - second_centre_y) ** 2
        )
        return distance_sq < float(first_radius + second_radius) ** 2

    # Keep the circle in the first slot so the circle/rectangle calculation is
    # written once.
    if first_radius is None:
        return _shaped_regions_overlap(second, second_radius, first, first_radius)

    centre_x = ax + aw / 2.0
    centre_y = ay + ah / 2.0
    nearest_x = min(max(centre_x, bx), bx + bw)
    nearest_y = min(max(centre_y, by), by + bh)
    distance_sq = (centre_x - nearest_x) ** 2 + (centre_y - nearest_y) ** 2
    return distance_sq < float(first_radius) ** 2


def _shaped_region_bounds(
    group: tuple[int, int, int, int],
    radius: int | None,
    image_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    """Return image-clamped bounds enclosing the rectangle or inferred disc."""
    x, y, w, h = group
    if radius is None:
        return clamp_group(group, image_shape) or group

    centre_x = x + w / 2.0
    centre_y = y + h / 2.0
    x0 = int(math.floor(centre_x - radius))
    y0 = int(math.floor(centre_y - radius))
    x1 = int(math.ceil(centre_x + radius))
    y1 = int(math.ceil(centre_y + radius))
    shape = (x0, y0, x1 - x0, y1 - y0)
    return clamp_group(shape, image_shape) or group


def _shaped_region_uv_coverage(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    group: tuple[int, int, int, int],
    radius: int | None,
) -> float:
    """Return UV-domain coverage for one already-shaped region."""
    if uv_mask is None:
        return 1.0
    if radius is not None:
        x, y, w, h = group
        return circle_uv_coverage(
            uv_mask,
            (x + w / 2.0, y + h / 2.0),
            float(radius),
        )
    return box_uv_coverage(uv_mask, group, image.shape)


def _step_overlap_group(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig,
    state: DetectionState,
) -> tuple[DetectionState, DetectionStage]:
    """Force positive-area shaped overlaps, then validate each merged result."""
    groups = list(state.groups)

    if not config.enable_overlap_group_merge or len(groups) < 2:
        return DetectionState(state.boxes, groups), DetectionStage(
            key="overlap_group",
            title="Post-circle forced merge",
            kept=tuple(groups),
            detail=(
                "disabled"
                if not config.enable_overlap_group_merge
                else "fewer than two recovered groups; nothing to merge"
            ),
        )

    radii = [
        inscribed_circle_radius(image, uv_mask, group, config)
        if config.enable_circular_groups
        else None
        for group in groups
    ]

    parent = list(range(len(groups)))

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

    for left in range(len(groups)):
        for right in range(left + 1, len(groups)):
            if _shaped_regions_overlap(
                groups[left], radii[left], groups[right], radii[right]
            ):
                union(left, right)

    components: dict[int, list[int]] = {}
    for index in range(len(groups)):
        components.setdefault(find(index), []).append(index)

    merged: list[tuple[int, int, int, int]] = []
    rejected: list[tuple[int, int, int, int]] = []
    absorbed = 0
    for indices in components.values():
        if len(indices) == 1:
            result = groups[indices[0]]
        else:
            absorbed += len(indices) - 1
            bounds = [
                _shaped_region_bounds(groups[index], radii[index], image.shape)
                for index in indices
            ]
            x0 = min(x for x, _y, _w, _h in bounds)
            y0 = min(y for _x, y, _w, _h in bounds)
            x1 = max(x + w for x, _y, w, _h in bounds)
            y1 = max(y + h for _x, y, _w, h in bounds)
            result = (x0, y0, x1 - x0, y1 - y0)

        if config.enable_region_domain_filter and uv_mask is not None:
            _radius, coverage = _region_shape_and_coverage(
                image, uv_mask, result, config
            )
            if coverage < config.min_region_uv_coverage:
                rejected.append(result)
                continue
        merged.append(result)

    merged.sort(key=lambda group: (group[1], group[0]))
    detail = (
        f"absorbed {absorbed} overlapping region{'s' if absorbed != 1 else ''}; "
        f"{len(rejected)} merged group{'s' if len(rejected) != 1 else ''} "
        "failed the final shaped-domain test"
    )
    return DetectionState(state.boxes, merged), DetectionStage(
        key="overlap_group",
        title="Post-circle forced merge",
        kept=tuple(merged),
        rejected=tuple(rejected),
        detail=detail,
        adjusted=absorbed,
    )


def _step_final_padding(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig,
    state: DetectionState,
) -> tuple[DetectionState, DetectionStage]:
    """Pad final regions without feeding the larger bounds back into detection.

    Rectangles are expanded independently to the image edges.  A circular
    region is expanded symmetrically so its centre remains unchanged; when an
    image edge is too close, its usable padding is reduced accordingly.
    """
    requested = max(int(config.final_region_padding_px), 0)
    padded: list[tuple[int, int, int, int]] = []
    circles: list[int | None] = []
    adjusted = 0
    image_height, image_width = image.shape[:2]

    for group in state.groups:
        radius = inscribed_circle_radius(image, uv_mask, group, config)
        x, y, w, h = group

        if radius is not None:
            centre_x = x + w / 2.0
            centre_y = y + h / 2.0
            # Keep the inferred circle centre fixed.  Existing circle padding can
            # already extend beyond the rectangular group, so bound the extra
            # padding against the circle itself rather than just the group box.
            edge_room = min(
                centre_x - radius,
                centre_y - radius,
                image_width - (centre_x + radius),
                image_height - (centre_y + radius),
            )
            usable = min(requested, max(int(math.floor(edge_room)), 0))
            result = (
                max(x - usable, 0),
                max(y - usable, 0),
                min(w + usable * 2, image_width),
                min(h + usable * 2, image_height),
            )
            circles.append(radius + usable)
        else:
            result = clamp_group(
                (x - requested, y - requested, w + requested * 2, h + requested * 2),
                image.shape,
            )
            if result is None:
                # A valid incoming region cannot normally reach this branch, but
                # retaining it is safer than silently deleting a final detection.
                result = group
            circles.append(None)

        if result != group:
            adjusted += 1
        padded.append(result)

    detail = (
        f"expanded by {requested} px on each side, clamped to the image"
        if requested
        else "0 px; final bounds unchanged"
    )
    return DetectionState(state.boxes, padded), DetectionStage(
        key="final_padding",
        title="Final padding",
        kept=tuple(padded),
        detail=detail,
        adjusted=adjusted,
        circles=tuple(circles),
    )


# Stages holding assembled regions rather than raw MSER boxes; only these are
# worth shaping, and only these are few enough for it to be cheap.
GROUP_STAGE_KEYS = frozenset(
    {"grouped", "region_domain", "overlap_group", "pattern_group", "size"}
)

PIPELINE_STEPS = (
    _step_mser,
    _step_box_filter,
    _step_grouped,
    _step_region_domain,
    _step_overlap_group,
    _step_pattern_group,
    _step_size,
    _step_final_padding,
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
    "enable_aspect_ratio_filter": 0,
    "min_aspect": 0,
    "max_aspect": 0,
    "enable_box_feature_filter": 1,
    "min_box_width_px": 1,
    "min_box_height_px": 1,
    "box_feature_colour_tolerance": 1,
    "box_feature_min_domain_px": 1,
    "box_feature_context_px": 1,
    "min_box_uv_coverage": 1,
    "box_min_feature_px": 1,
    "merge_distance_px": 2,
    "min_group_union_region_px": 2,
    "enable_island_bounded_grouping": 2,
    "enable_circular_groups": 2,
    "circular_group_min_squareness": 2,
    "circular_group_padding_px": 2,
    "circular_group_colour_tolerance": 2,
    "circular_group_max_corner_content": 2,
    "enable_region_domain_filter": 3,
    "min_region_uv_coverage": 3,
    "enable_overlap_group_merge": 4,
    "enable_pattern_group_filter": 5,
    "max_pattern_autocorrelation": 5,
    "pattern_window_scale": 5,
    "pattern_min_window_px": 5,
    "pattern_min_period_px": 5,
    "pattern_max_period_px": 5,
    "enable_final_size_filter": 6,
    "final_min_width_px": 6,
    "final_min_height_px": 6,
    "final_min_area_px": 6,
    "enable_final_aspect_filter": 6,
    "final_max_aspect": 6,
    "final_region_padding_px": 7,
    "uv_island_mask_path": len(PIPELINE_STEPS),
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
                entry_states=previous.entry_states,
                resumed_from=start,
            )

    stages = list(previous.stages[:start]) if previous is not None and start else []
    entry_states = list(previous.entry_states[:start]) if previous is not None and start else []
    state = (
        previous.entry_states[start].copy()
        if previous is not None and start
        else DetectionState(np.empty((0, 4), dtype=np.int32), [])
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
        entry_states=entry_states,
        resumed_from=start,
    )


def detect_texture_region_stages(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig | None = None,
) -> list[DetectionStage]:
    """Run detection and return every intermediate stage."""
    return run_detection(image, uv_mask, config).stages


def detect_texture_regions(
    image: np.ndarray,
    uv_mask: np.ndarray | None,
    config: MserConfig | None = None,
) -> dict[str, object]:
    """Run texture-region detection on an already loaded BGR image."""
    stages = detect_texture_region_stages(image, uv_mask, config)
    by_key = {stage.key: stage for stage in stages}
    final = by_key["final_padding"]

    return {
        "image_size": f"{image.shape[1]}x{image.shape[0]}",
        "mser_boxes": len(by_key["mser"].kept),
        "boxes_rejected": len(by_key["box_filter"].rejected),
        "candidate_groups": len(by_key["grouped"].kept),
        "pattern_groups_rejected": len(by_key["pattern_group"].rejected),
        "final_size_rejected": len(by_key["size"].rejected),
        "domain_recovery_rejected": len(by_key["region_domain"].rejected),
        "region_domain_rejected": len(by_key["region_domain"].rejected),
        "overlap_regions_absorbed": by_key["overlap_group"].adjusted,
        "final_padding_px": max(int((config or DEFAULT_CONFIG).final_region_padding_px), 0),
        "grouped_regions": len(final.kept),
        "groups": [
            {"x": x, "y": y, "w": w, "h": h, "radius": radius}
            for (x, y, w, h), radius in zip(
                final.kept, final.circles or (None,) * len(final.kept)
            )
        ],
    }


def write_annotation_outputs(
    image: np.ndarray,
    summary: dict[str, object],
    outputs: TextureRegionAnnotationOutputs,
    config: MserConfig,
    symmetry_matches: tuple[UvIslandSymmetryMatch, ...] = (),
    symmetry_config: UvIslandSymmetryConfig | None = None,
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
    # This overlay is intentionally drawn after the red/green annotations and
    # is not represented in DetectionState or PIPELINE_STEPS.
    annotated = draw_uv_island_symmetry_annotations(
        annotated, symmetry_matches, symmetry_config
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
    symmetry_config: UvIslandSymmetryConfig | None = None,
) -> dict[str, object]:
    """Detect texture regions and write caller-specified output files."""
    config = config_with_uv_mask_path(config or DEFAULT_CONFIG, uv_island_mask_path)
    image = load_image(texture_path)

    uv_mask = None
    if config.uv_island_mask_path:
        uv_mask = load_uv_island_mask(
            resolve_config_path(config.uv_island_mask_path),
            image.shape,
        )

    summary = detect_texture_regions(image, uv_mask, config)
    symmetry_config = symmetry_config or DEFAULT_UV_ISLAND_SYMMETRY_CONFIG
    symmetry_matches = analyse_uv_island_symmetry(uv_mask, symmetry_config)
    summary.update(
        {
            "input": str(texture_path),
            "output": str(outputs.annotated_image),
            "uv_island_mask": str(uv_island_mask_path or config.uv_island_mask_path),
            "symmetric_uv_islands": len(symmetry_matches),
            "uv_island_symmetry_threshold": min(
                max(float(symmetry_config.min_uv_island_symmetry), 0.0), 1.0
            ),
            "uv_island_symmetry": [
                {
                    "x": match.bounds[0],
                    "y": match.bounds[1],
                    "w": match.bounds[2],
                    "h": match.bounds[3],
                    "area_px": match.area_px,
                    "x_similarity": match.x_similarity,
                    "y_similarity": match.y_similarity,
                    "matched_axes": list(match.matched_axes),
                }
                for match in symmetry_matches
            ],
        }
    )
    if outputs.summary_json is not None:
        summary["summary_json"] = str(outputs.summary_json)

    write_annotation_outputs(
        image,
        summary,
        outputs,
        config,
        symmetry_matches,
        symmetry_config,
    )
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
    parser.add_argument(
        "--min-box-width",
        type=int,
        default=None,
        metavar="PX",
        help="Reject raw MSER boxes narrower than this before grouping.",
    )
    parser.add_argument(
        "--min-box-height",
        type=int,
        default=None,
        metavar="PX",
        help="Reject raw MSER boxes shorter than this before grouping.",
    )
    parser.add_argument(
        "--final-padding",
        type=int,
        default=None,
        metavar="PX",
        help="Padding added to every final detected region, in pixels.",
    )
    parser.add_argument(
        "--uv-symmetry-threshold",
        type=float,
        default=None,
        metavar="FRACTION",
        help=(
            "Minimum reflected-mask similarity for a UV island to receive a "
            "blue outline; either X- or Y-axis symmetry may qualify."
        ),
    )
    parser.add_argument(
        "--no-uv-symmetry",
        action="store_true",
        help="Disable the independent blue UV-island symmetry overlay.",
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

    overrides: dict[str, int] = {}
    if args.min_box_width is not None:
        overrides["min_box_width_px"] = max(args.min_box_width, 1)
    if args.min_box_height is not None:
        overrides["min_box_height_px"] = max(args.min_box_height, 1)
    if args.final_padding is not None:
        overrides["final_region_padding_px"] = max(args.final_padding, 0)
    config = replace(DEFAULT_CONFIG, **overrides) if overrides else DEFAULT_CONFIG

    symmetry_overrides: dict[str, object] = {}
    if args.uv_symmetry_threshold is not None:
        symmetry_overrides["min_uv_island_symmetry"] = min(
            max(float(args.uv_symmetry_threshold), 0.0), 1.0
        )
    if args.no_uv_symmetry:
        symmetry_overrides["enable_uv_island_symmetry"] = False
    symmetry_config = (
        replace(DEFAULT_UV_ISLAND_SYMMETRY_CONFIG, **symmetry_overrides)
        if symmetry_overrides
        else DEFAULT_UV_ISLAND_SYMMETRY_CONFIG
    )

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
            symmetry_config,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    elapsed_ms = (time.perf_counter() - started) * 1000.0

    print(
        f"{summary['image_size']}  "
        f"{summary['mser_boxes']} MSER boxes → "
        f"{summary['grouped_regions']} annotated groups; "
        f"{summary['symmetric_uv_islands']} symmetric UV islands"
    )
    print(
        f"{summary['boxes_rejected']} boxes rejected, "
        f"{summary['candidate_groups']} candidate groups, "
        f"{summary['pattern_groups_rejected']} rejected as repeating pattern, "
        f"{summary['final_size_rejected']} rejected by final size, "
        f"{summary['region_domain_rejected']} rejected as outside the UV domain, "
        f"{summary['overlap_regions_absorbed']} overlapping regions absorbed"
    )
    print(f"Wrote {summary['output']}")
    print(f"elapsed: {elapsed_ms:.1f} ms")

    for index, group in enumerate(summary["groups"], start=1):
        shape = f"circle r={group['radius']}" if group.get("radius") else "rect"
        print(
            f"  [{index:3d}]  x={group['x']:5d}  y={group['y']:5d}  "
            f"w={group['w']:4d}  h={group['h']:4d}  {shape}"
        )


if __name__ == "__main__":
    main()