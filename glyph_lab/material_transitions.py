"""Measurements for regions caused by an image-material transition.

This is deliberately a candidate measure, not a filter yet.  A dashboard's
chrome/leather, carbon/vent and decal boundaries can all make a large sparse
region, but the same geometry also describes real badges and controls.  The
probe gathers every plausible case before a late-stage decision is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from glyph_lab.regions import Region, closed_body

MATERIAL_TRANSITION_PROBE = "material transition candidates"


def is_material_transition_candidate(
    region: Region, min_span: int, max_fill: float,
) -> bool:
    """A cheap, deliberately broad prefilter for the transition gallery."""
    return min(region.width, region.height) >= max(int(min_span), 0) and (
        max_fill <= 0.0 or region.fill <= float(max_fill)
    )


@dataclass(slots=True)
class MaterialTransitionCandidate:
    """One broad sparse region and the evidence for its transition decision."""

    region: Region
    body_void_texels: int | None
    body_void_share: float | None
    interior_detail: float | None
    enclosed_texels: int | None = None
    rule_eligible: bool = False
    loop_eligible: bool = False
    discarded: bool = False


def material_transition_candidates(
    regions: list[Region], colour: np.ndarray, min_span: int, max_fill: float,
    close_gap: int, offset: tuple[int, int] = (0, 0),
    include_all: bool = False, measure_all: bool = False,
) -> list[MaterialTransitionCandidate]:
    """Measure all transition-shaped regions, without deciding them yet.

    2026-08-29: `closed_body` permits a near-loop to close over a small gap.
    This gives a broken seam an interior measure without pretending it was
    already a real loop; actual enclosure remains a separate fact on Region.
    """
    x_offset, y_offset = offset
    candidates: list[MaterialTransitionCandidate] = []
    for region in regions:
        eligible = is_material_transition_candidate(region, min_span, max_fill)
        if not include_all and not eligible:
            continue
        minor = min(region.width, region.height)
        loop_eligible = minor >= max(int(min_span), 0)
        needs_body = eligible or measure_all
        needs_loop = loop_eligible or measure_all
        body_void_texels: int | None = None
        body_void_share: float | None = None
        if needs_body:
            body_void = closed_body(region.mask, max(int(close_gap), 0)) & ~region.mask
            body_void_texels = int(body_void.sum())
            body_void_share = body_void_texels / max(region.texels, 1)
        enclosed_texels: int | None = None
        detail: float | None = None
        if needs_loop:
            actual_void = region.filled & ~region.mask
            enclosed_texels = int(actual_void.sum())
        else:
            actual_void = None
        if actual_void is not None and actual_void.any():
            crop = colour[
                region.y - y_offset:region.y - y_offset + region.height,
                region.x - x_offset:region.x - x_offset + region.width,
                :3,
            ].astype(np.float32)
            luminance = crop @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
            detail = float(np.abs(
                luminance - ndimage.gaussian_filter(luminance, 3.0)
            )[actual_void].mean())
        candidates.append(MaterialTransitionCandidate(
            region=region,
            body_void_texels=body_void_texels,
            body_void_share=body_void_share,
            interior_detail=detail,
            enclosed_texels=enclosed_texels,
            rule_eligible=eligible,
            loop_eligible=loop_eligible,
        ))
    return candidates


def classify_material_transition_candidates(
    candidates: list[MaterialTransitionCandidate], min_enclosed_texels: int,
    min_body_void_share: float, max_interior_detail: float,
    min_long_aspect: float, max_thin_trace_span: int,
    min_thin_trace_aspect: float, max_flat_loop_texels: int, context_gap: int,
) -> None:
    """Discard empty loops, long traces, and their attached fragments."""
    for candidate in candidates:
        region = candidate.region
        aspect = max(region.width, region.height) / max(min(region.width, region.height), 1)
        actual_loop = (
            candidate.loop_eligible
            and (candidate.enclosed_texels or 0) >= max(int(min_enclosed_texels), 0)
        )
        empty_loop = actual_loop and (candidate.interior_detail or 0.0) <= max_interior_detail
        unclosed_trace = (
            candidate.rule_eligible and not actual_loop
            and (candidate.body_void_share or 0.0) <= min_body_void_share
        )
        thin_trace = (
            min(region.width, region.height) <= max(int(max_thin_trace_span), 0)
            and aspect >= min_thin_trace_aspect
        )
        # 2026-08-30: the original broad/sparse gate stays conservative.  The
        # two extensions come from the full-region annotation: an extremely
        # long thin trace is itself sufficient evidence, while an empty loop
        # can be thicker than the initial fill ceiling but is capped so a large
        # simple badge is not caught merely for having a quiet interior.
        candidate.discarded = (
            candidate.rule_eligible and (empty_loop or unclosed_trace)
        ) or aspect >= min_long_aspect or thin_trace or (
            empty_loop and region.texels <= max_flat_loop_texels
        )
    gap = max(int(context_gap), 0)
    if gap <= 0:
        return
    confirmed = [candidate for candidate in candidates if candidate.discarded]
    for candidate in candidates:
        if candidate.discarded or (
                candidate.loop_eligible
                and (candidate.enclosed_texels or 0) >= min_enclosed_texels):
            continue
        candidate.discarded = any(
            candidate.region.domain == other.region.domain
            and texels_within_gap(candidate.region, other.region, gap)
            for other in confirmed
        )


def box_gap(first: Region, second: Region) -> int:
    """The free bounding-box lower bound used by short-fragment context."""
    ax, ay, aw, ah = first.bounds
    bx, by, bw, bh = second.bounds
    return max(bx - (ax + aw), ax - (bx + bw), by - (ay + ah), ay - (by + bh), 0)


def texels_within_gap(first: Region, second: Region, gap: int) -> bool:
    """Check a small exact Chebyshev gap without building a texel-pair matrix."""
    if box_gap(first, second) > gap:
        return False
    left = min(first.x, second.x)
    top = min(first.y, second.y)
    right = max(first.x + first.width, second.x + second.width)
    bottom = max(first.y + first.height, second.y + second.height)
    first_mask = np.zeros((bottom - top, right - left), dtype=bool)
    first_mask[
        first.y - top:first.y - top + first.height,
        first.x - left:first.x - left + first.width,
    ] = first.mask
    second_y, second_x = np.nonzero(second.mask)
    distance = ndimage.distance_transform_cdt(~first_mask, metric="chessboard")
    return bool(np.any(distance[
        second_y + second.y - top, second_x + second.x - left,
    ] <= gap))


def clear_of_material_transitions(
    regions: list[Region], colour: np.ndarray, min_span: int, max_fill: float,
    close_gap: int, min_enclosed_texels: int, min_body_void_share: float,
    max_interior_detail: float, min_long_aspect: float,
    max_thin_trace_span: int, min_thin_trace_aspect: float,
    max_flat_loop_texels: int, context_gap: int,
    measurements: list[MaterialTransitionCandidate] | None = None,
    offset: tuple[int, int] = (0, 0),
) -> list[Region]:
    """Remove the measured material-transition artefacts from a region list."""
    candidates = measurements if measurements is not None else material_transition_candidates(
        regions, colour, min_span, max_fill, close_gap, offset, include_all=True,
    )
    classify_material_transition_candidates(
        candidates, min_enclosed_texels, min_body_void_share, max_interior_detail,
        min_long_aspect, max_thin_trace_span, min_thin_trace_aspect,
        max_flat_loop_texels, context_gap,
    )
    discarded = {id(candidate.region) for candidate in candidates if candidate.discarded}
    return [region for region in regions if id(region) not in discarded]


def material_transition_features(
    candidate: MaterialTransitionCandidate,
) -> dict[str, int | float]:
    """Geometry shared by the decision and generic candidate probe."""
    region = candidate.region
    minor = min(region.width, region.height)
    major = max(region.width, region.height)
    return {
        "x": region.x,
        "y": region.y,
        "width_px": region.width,
        "height_px": region.height,
        "texels": region.texels,
        "minor_span_px": minor,
        "major_span_px": major,
        "aspect": major / max(minor, 1),
        "fill": region.fill,
        "enclosed_texels": candidate.enclosed_texels,
        "enclosure": (
            candidate.enclosed_texels / max(region.texels, 1)
            if candidate.enclosed_texels is not None else None
        ),
        "holes": region.holes,
        "stroke_px": region.stroke,
        "stroke_variation": region.stroke_variation,
        "body_void_texels": candidate.body_void_texels,
        "body_void_share": candidate.body_void_share,
        "interior_detail": candidate.interior_detail,
        "rule_eligible": candidate.rule_eligible,
    }
