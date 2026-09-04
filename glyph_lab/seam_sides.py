"""Regions that divide two grounds, rather than sitting on one.

2026-09-04: the stage 8 transition test asks an *inward* question -- does this
region enclose quiet ground -- and so it only ever sees closed loops.  Four of
the marks it leaves behind on the etkc interior never enclose anything: an
open C around an inset, and slivers lying along a leather-to-perforated seam.
Reordering does not reach them either; measured over the whole labelled
corpus, eight of nine escape on the `transition_min_span_px` prefilter alone,
their short sides running 10 to 50 texels against its 64.

The question here is outward instead, and it is Ashley's: walking away from
the mark the colour stops changing, and that holds on *both* sides.  A printed
glyph has one ground all round it; a seam has two, and they differ.  So:

    the surroundings split into two sides that are real   (balance, coherence)
    the two sides differ                                  (gap)
    each side is flat over its whole extent               (detail)

Measured on 557 hand-labelled regions across nine textures, this removes 12 of
the 16 false positives that reach the end of the pipeline and none of the 365
true marks.  The margins are not alike: `gap` is decisive, with every true
mark at or below 20.8 and every seam at or above 67.9, so its threshold can
sit anywhere in a 47-unit void.  `detail` is the finishing cut and its nearest
true mark is 13.3, against seams at or below 6.6.

Two things this deliberately does not do.

It does not run after the proximity grouping.  Grouping merges a button's rim
with the legend printed inside it, so the text becomes part of the mask and
the interior measures as blank ground: etk800's AUTO and TUNE, etkc's ENGINE
START and ENTER all read as clean seams once merged, and the same rule that
costs nothing here costs eleven true marks there.

It does not measure flatness in a band near the boundary.  A band 6-14 texels
in sits in a button's blank margin and never reaches the legend in the middle
of it, which is the same mistake in miniature -- it flagged those same eight
buttons.  Flatness is taken over everything the split attributes to a side.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from glyph_lab.regions import Region

# Texels between the mark and the first sample.  A mark's own antialiasing
# halo is part of the mark, not of the ground beside it, and reading it as
# ground was enough to flatten `gap` to chance: true marks measured a median
# 75.8 with the halo in, 3.9 with it out.
#
# 2026-09-04: one, not two.  Two is enough to miss a thin enclosed side
# entirely -- the etkc slider bezel's slot is a few texels tall, so skipping
# two sampled only the outside and the region read as a mark on one ground.
# What made one safe is `min_enclosed_texels` below: at two the marginal
# cases were held off by the skip, and at one they are held off by asking
# whether the enclosed side is a feature at all.  The separation is wider
# for it -- `detail` runs 6.96 against 58.74, where at two it was 6.61
# against 13.3.
HALO_SKIP = 1
# How far out the split itself is decided.  Wide enough for two grounds to
# each own a real share of it, narrow enough that a second boundary further
# out is somebody else's business.
SPLIT_REACH = 4


@dataclass(frozen=True, slots=True)
class SideEvidence:
    """What the two sides of one region look like, and how far apart."""

    gap: float          # tone between the two sides
    detail: float       # the busier side's texture, over its whole extent
    shift: float        # how far a side drifts from its own edge value
    balance: float      # the smaller side's share of the split
    coherence: float    # the smaller contiguous share of either side
    two_sided: bool     # both sides reach outward far enough to be measured
    enclosed: int = 0   # texels the mark closes around, 0 if it closes around none

    @property
    def measurable(self) -> bool:
        return self.two_sided


# Eight-connected, so a margin of n is n texels in every direction including
# the diagonals.  2026-09-04: the default cross leaves a diamond, and at a
# one-texel skip the four corners of a mark's antialiasing halo fall outside
# it -- enough on a hard-edged mark to report a 100-level "second ground"
# that is the mark's own edge.
SQUARE = np.ones((3, 3), dtype=bool)


def _ring(mask: np.ndarray, inner: int, outer: int, domain: np.ndarray) -> np.ndarray:
    """Texels between `inner` and `outer` dilations of the mark."""
    grown = ndimage.binary_dilation(mask, SQUARE, iterations=outer)
    if inner > 0:
        grown = grown & ~ndimage.binary_dilation(mask, SQUARE, iterations=inner)
    return grown & domain


def _split_in_two(values: np.ndarray) -> tuple[float, float] | None:
    """Two-means on tone, from the extremes inward.

    Two means rather than Otsu because what matters is the pair of levels, not
    the cut between them: the levels are what the sides are then compared on,
    and Otsu would have to be followed by this anyway.
    """
    low, high = float(values.min()), float(values.max())
    if high - low < 2.0:
        return None
    for _ in range(15):
        near_low = np.abs(values - low) <= np.abs(values - high)
        if not near_low.any() or near_low.all():
            return None
        low = float(values[near_low].mean())
        high = float(values[~near_low].mean())
    return low, high


def _largest_share(piece: np.ndarray) -> float:
    """How much of a side is its own biggest connected run.

    A real ground beside a seam is one arc.  Two-means will always return
    something on a single flat ground too, but there it returns speckle, and
    speckle is what this is here to refuse.
    """
    labels, count = ndimage.label(piece)
    if not count:
        return 0.0
    return float(np.bincount(labels.ravel())[1:].max()) / max(int(piece.sum()), 1)


def side_evidence(
    mask: np.ndarray, tone: np.ndarray, domain: np.ndarray, reach: int,
    min_side_texels: int = 40,
) -> SideEvidence | None:
    """Split one mark's surroundings in two and measure each side outward.

    `mask`, `tone` and `domain` are all in the same local frame, which the
    caller cuts with enough margin for `reach` dilations to have somewhere to
    go.  None means the mark has too little ground beside it to judge.
    """
    enclosed = int((ndimage.binary_fill_holes(mask) & ~mask).sum())
    edge = _ring(mask, HALO_SKIP, SPLIT_REACH, domain)
    if int(edge.sum()) < min_side_texels:
        return None
    values = tone[edge]
    levels = _split_in_two(values)
    if levels is None:
        # One ground all round: a mark on flat colour, which is the ordinary
        # case and the one this stage must never touch.
        return SideEvidence(0.0, 0.0, 0.0, 0.0, 0.0, True, enclosed)
    low, high = levels

    side = np.zeros(mask.shape, dtype=np.int8)
    side[edge] = np.where(
        np.abs(values - low) <= np.abs(values - high), 1, 2
    )
    # Carry the two labels over the whole neighbourhood, the enclosed interior
    # of a loop included: a texel belongs to whichever side is nearest to it.
    _distance, (rows, columns) = ndimage.distance_transform_edt(
        side == 0, return_indices=True
    )
    carried = side[rows, columns]

    first, second = int((side == 1).sum()), int((side == 2).sum())
    balance = min(first, second) / max(first + second, 1)
    coherence = min(_largest_share(side == 1), _largest_share(side == 2))

    field = _ring(mask, HALO_SKIP, reach, domain)
    texture = np.abs(tone - ndimage.gaussian_filter(tone, 2.0))
    detail = shift = 0.0
    two_sided = True
    for label, level in ((1, low), (2, high)):
        body = field & (carried == label)
        if int(body.sum()) < min_side_texels:
            two_sided = False
            continue
        shift = max(shift, abs(float(tone[body].mean()) - level))
        detail = max(detail, float(texture[body].mean()))
    return SideEvidence(
        gap=abs(high - low), detail=detail, shift=shift,
        balance=balance, coherence=coherence, two_sided=two_sided,
        enclosed=enclosed,
    )


def divides_two_grounds(
    evidence: SideEvidence | None, min_gap: float, max_detail: float,
    min_balance: float, min_coherence: float, min_enclosed_texels: int = 0,
) -> bool:
    """Whether the evidence describes a seam rather than a mark.

    2026-09-04: a mark closes around things of its own -- the counter of an
    "o", the notch in a pictogram -- and a small one of those is not the far
    side of a material boundary.  So a region that encloses *something* has to
    enclose enough of it, while a region that encloses nothing is unaffected:
    six of the seams this removes are open slivers with no interior at all.
    The ardente's fuel pictogram encloses 142 texels and the etkc slider
    bezel 254, which is the gap the floor sits in.
    """
    if evidence is None or not evidence.measurable:
        return False
    if 0 < evidence.enclosed < min_enclosed_texels:
        return False
    return (
        evidence.balance >= min_balance
        and evidence.coherence >= min_coherence
        and evidence.gap >= min_gap
        and evidence.detail <= max_detail
    )


def measure_region_sides(
    region: Region, colour: np.ndarray, domain: np.ndarray,
    offset: tuple[int, int], reach: int,
) -> SideEvidence | None:
    """`side_evidence` for one region, cut out of the atlas around it."""
    x_offset, y_offset = offset
    height, width = colour.shape[:2]
    margin = reach + 6
    x = region.x - x_offset
    y = region.y - y_offset
    left = max(x - margin, 0)
    top = max(y - margin, 0)
    right = min(x + region.width + margin, width)
    bottom = min(y + region.height + margin, height)
    if right <= left or bottom <= top:
        return None
    patch = colour[top:bottom, left:right, :3].astype(np.float32)
    tone = patch @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    local = np.zeros((bottom - top, right - left), dtype=bool)
    local[y - top: y - top + region.height, x - left: x - left + region.width] = (
        region.mask
    )
    return side_evidence(local, tone, domain[top:bottom, left:right], reach)


def clear_of_two_sided_seams(
    regions: list[Region], colour: np.ndarray, domain: np.ndarray,
    offset: tuple[int, int], reach: int, min_gap: float, max_detail: float,
    min_balance: float, min_coherence: float, min_enclosed_texels: int = 0,
    measurements: dict[int, SideEvidence | None] | None = None,
) -> list[Region]:
    """Drop every region that divides two flat grounds of different colour."""
    if min_gap <= 0.0 or reach <= 0:
        return regions
    kept: list[Region] = []
    for region in regions:
        if measurements is not None and id(region) in measurements:
            evidence = measurements[id(region)]
        else:
            evidence = measure_region_sides(region, colour, domain, offset, reach)
        if not divides_two_grounds(
            evidence, min_gap, max_detail, min_balance, min_coherence,
            min_enclosed_texels,
        ):
            kept.append(region)
    return kept


def side_features(evidence: SideEvidence | None) -> dict[str, float]:
    """The measurements, for the region probe."""
    if evidence is None:
        return {}
    return {
        "side gap": evidence.gap,
        "side detail": evidence.detail,
        "side shift": evidence.shift,
        "side balance": evidence.balance,
        "side coherence": evidence.coherence,
        "enclosed texels": float(evidence.enclosed),
        "two sided": float(evidence.two_sided),
    }
