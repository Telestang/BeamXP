"""A detected mark and the texels it was found in.

The first pipeline carried ``(x, y, w, h)`` between its stages, because it began
on MSER and a rectangle is what MSER hands you.  Thirteen of its twenty-one
stages then went back to the atlas to re-crop the pixels, because a rectangle
cannot answer the questions they ask -- and the questions it cannot answer are
the interesting ones.  A contrast response draws a glyph as a closed loop and a
grained material as a scatter of dots; both are rectangles, and the two are
indistinguishable once you have thrown the texels away.

So a ``Region`` owns its mask.  Every measurement below reads that mask, and
none of them re-open the texture.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

# 8-connectivity, matching how the components were labelled.
NEIGHBOURHOOD = np.ones((3, 3), dtype=bool)


# The eight neighbours, in the bit order `_hairline_table` is indexed by.
_RING = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def _hairline_table() -> np.ndarray:
    """Which of the 256 neighbourhood patterns read as one texel wide.

    A texel is one wide when at most two of its eight neighbours are set, and
    -- where there are two -- they do not touch each other.  Two neighbours
    that touch mean the mark turns a corner here and is two texels across the
    turn, which is why the exception matters:

        1 . .        1 . .        1 . .
        1 1 .        . 1 1        . 1 .
        . . .        . 1 .        1 . .
        not thin     not thin     thin

    The first has two neighbours side by side, the second has three, and only
    the third is a line passing through.  Written as a table because the test
    is per-pattern rather than morphological: no structuring element expresses
    "unless they are adjacent".
    """
    table = np.zeros(256, dtype=bool)
    for pattern in range(256):
        on = [_RING[bit] for bit in range(8) if pattern >> bit & 1]
        if len(on) > 2:
            continue
        if len(on) == 2:
            (ay, ax), (by, bx) = on
            if max(abs(ay - by), abs(ax - bx)) == 1:
                continue
        table[pattern] = True
    return table


_HAIRLINE = _hairline_table()


def _shifted(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """``mask`` moved so that the result at (y, x) holds mask[y + dy, x + dx].

    Sliced rather than rolled: rolling wraps, which would let the top of an
    atlas count as a neighbour of the bottom.
    """
    height, width = mask.shape
    out = np.zeros_like(mask)
    out[
        max(0, -dy):height - max(0, dy), max(0, -dx):width - max(0, dx)
    ] = mask[
        max(0, dy):height - max(0, -dy), max(0, dx):width - max(0, -dx)
    ]
    return out


def enclosed_voids(mask: np.ndarray) -> np.ndarray:
    """What a mark closes around: the holes in it, and nothing else.

    2026-08-27: the fill is left at scipy's default 4-connected structure, and
    that is the whole point of this function existing rather than the call
    being written out at each site.  ``binary_fill_holes`` floods the
    *background*, so its connectivity is the background's -- and an 8-connected
    background is the dual of a 4-connected foreground, not of the 8-connected
    one everything here labels with.  Passing ``NEIGHBOURHOOD`` therefore lets
    the flood step diagonally between two texels that touch at a corner, so any
    curve with a diagonal in it leaks and is judged to enclose nothing.

    Measured: a diamond drawn from 8-connected diagonals encloses 0 that way
    and 162 this way, and a 148x110 rounded panel outline off etk800 -- an
    ordinary drawn box -- went from 0 to 7,947.

    Padded, for the second half of the same problem: a region's crop is tight
    to its mark, so a ring lies *on* all four borders of its own array and the
    flood starts inside it.
    """
    padded = np.pad(mask, 1)
    return (ndimage.binary_fill_holes(padded) & ~padded)[1:-1, 1:-1]


def closed_body(mask: np.ndarray, gap: int) -> np.ndarray:
    """The mark closed across ``gap`` and filled: a fit that jumps only so far.

    A dilation followed by an erosion of the same radius, which is a closing --
    it joins anything within ``gap`` of anything else and leaves everything
    further apart alone, then the fill closes whatever that made a ring of.

    Padded first: the erosion treats everything outside the array as
    background, so without the pad a mark against the edge of its crop would be
    chewed back by exactly what the dilation grew it.
    """
    if gap <= 0:
        return mask | enclosed_voids(mask)
    padded = np.pad(mask, gap)
    padded = ndimage.binary_dilation(
        padded, structure=NEIGHBOURHOOD, iterations=gap
    )
    padded = ndimage.binary_erosion(
        padded, structure=NEIGHBOURHOOD, iterations=gap, border_value=1
    )
    body = padded[gap:-gap, gap:-gap]
    return body | enclosed_voids(body)


def hairline_texels(mask: np.ndarray) -> np.ndarray:
    """The texels of a mark that are one texel wide.  See `_hairline_table`.

    Eight shifts pack the neighbourhood into a byte and one lookup answers it,
    so the whole atlas is done at once.
    """
    code = np.zeros(mask.shape, dtype=np.uint8)
    for bit, (dy, dx) in enumerate(_RING):
        code |= _shifted(mask, dy, dx).astype(np.uint8) << bit
    return mask & _HAIRLINE[code]


def prune_hairlines(body: np.ndarray) -> np.ndarray:
    """A filled mark with everything one texel wide taken out of it.

    A hairline encloses nothing, and it shows: walk the boundary of a mark and
    a one-texel spur is traced on the way out and again on the way back, the
    same texels twice.  Whatever it joins, it joins weakly -- an edge run along
    a panel border passing within a texel of a glyph is enough to make the two
    one region and send the path all the way round the panel.

    Measured on what the mark *encloses* rather than on the mark, which is the
    whole trick.  A glyph's own outline narrows to a texel here and there, and
    testing the outline chops the glyph up -- it broke the Ginetta badge into
    164 pieces -- but the area that outline draws around is wide and stays
    whole.  An edge run draws around nothing and is thin either way.

    2026-08-27: the first version kept whatever sat in a solid 2x2 block, which
    called a staircase diagonal one texel wide and threw it away.  Rasterised
    edges are made of staircases.

    2026-08-27: and a texel that closes a loop is exempt, however thin it is.
    A sharp enough contrast is genuinely one texel wide, and the width was
    never the point -- a spur is traced twice by the same walk and encloses
    nothing, while a loop is traced once and encloses what it goes round.  So
    the test is not thinness but thinness *without* enclosing: anything lying
    against a void the mark closes around stays, and a spur hanging off that
    same loop still goes.

    Callers that fill first see no change; a filled body has no voids left to
    be exempt by.
    """
    thin = hairline_texels(body)
    voids = enclosed_voids(body)
    if voids.any():
        thin &= ~ndimage.binary_dilation(voids, structure=NEIGHBOURHOOD)
    return body & ~thin


@dataclass(frozen=True, slots=True)
class Region:
    """One connected mark: where it is, which texels it owns, whose domain."""

    bounds: tuple[int, int, int, int]   # x, y, w, h in atlas coordinates
    mask: np.ndarray                    # (h, w) bool, local to `bounds`
    domain: int                         # the UV domain label it sits on
    # Populated lazily; a frozen slots dataclass cannot use cached_property.
    _cache: dict = field(default_factory=dict, compare=False, repr=False)

    # -- geometry ---------------------------------------------------------
    @property
    def x(self) -> int: return self.bounds[0]

    @property
    def y(self) -> int: return self.bounds[1]

    @property
    def width(self) -> int: return self.bounds[2]

    @property
    def height(self) -> int: return self.bounds[3]

    @property
    def box_area(self) -> int:
        return max(self.width * self.height, 1)

    @property
    def texels(self) -> int:
        """How many texels the detector actually fired on."""
        return self._memo("texels", lambda: int(self.mask.sum()))

    @property
    def fill(self) -> float:
        """Share of the bounding rectangle the mark occupies.

        A one-texel outline is a few percent of its own box; a grained patch
        is most of it.  This is the cheapest of the loop/scatter measures and
        the least decisive on its own -- a long diagonal stroke is also sparse.
        """
        return self.texels / self.box_area

    # -- the loop --------------------------------------------------------
    @property
    def filled(self) -> np.ndarray:
        """The mark with whatever it closes around filled in.

        The response fires at edges, so a mark of any shape comes back as its
        own outline: a solid "1" arrives as a closed ring around the stroke,
        not as the stroke.  Filling that ring recovers the shape the mark is
        an outline *of*, which is what the enclosure measures below are
        differences against.

        2026-08-27: detection no longer calls this -- it fills every mark in a
        domain with one `binary_fill_holes` and counts the results in bulk, so
        nothing is filled a component at a time on the hot path.  Kept for
        asking about one region on its own.
        """
        return self._memo(
            "filled", lambda: self.mask | enclosed_voids(self.mask)
        )

    @property
    def enclosed(self) -> int:
        """Texels the mark encloses without covering.

        Enclosing something is what a loop does, and grain -- scattered dots
        closing around nothing -- does not.
        """
        return self._memo(
            "enclosed", lambda: int(np.count_nonzero(enclosed_voids(self.mask)))
        )

    @property
    def enclosure(self) -> float:
        """Enclosed area over the mark's own texels.

        Scale free, and the direction that matters: a loop encloses far more
        than it is made of, a scatter encloses nothing.
        """
        return self.enclosed / max(self.texels, 1)

    @property
    def holes(self) -> int:
        """How many separate voids the mark closes around."""
        def measure() -> int:
            _labels, count = ndimage.label(
                enclosed_voids(self.mask), structure=NEIGHBOURHOOD
            )
            return int(count)
        return self._memo("holes", measure)

    @property
    def stroke(self) -> float:
        """Median stroke thickness, in texels.

        Twice the median distance from a mark texel to the nearest texel that
        is not one: the standard thickness estimate, and it separates a drawn
        outline from a filled blob without a shape model.
        """
        def measure() -> float:
            if not self.mask.any():
                return 0.0
            distance = ndimage.distance_transform_edt(
                np.pad(self.mask, 1), return_distances=True
            )[1:-1, 1:-1]
            return float(np.median(distance[self.mask]) * 2.0)
        return self._memo("stroke", measure)

    @property
    def stroke_variation(self) -> float:
        """Spread of that thickness, over its median.

        A drawn mark holds one pen width; grain does not hold any.
        """
        def measure() -> float:
            if not self.mask.any():
                return 0.0
            distance = ndimage.distance_transform_edt(
                np.pad(self.mask, 1), return_distances=True
            )[1:-1, 1:-1]
            widths = distance[self.mask] * 2.0
            median = float(np.median(widths))
            return float(np.std(widths) / median) if median > 0 else 0.0
        return self._memo("stroke_variation", measure)

    # -- relations --------------------------------------------------------
    def overlaps(self, other: Region) -> bool:
        """Whether the two rectangles share interior area."""
        ax, ay, aw, ah = self.bounds
        bx, by, bw, bh = other.bounds
        return (
            max(ax, bx) < min(ax + aw, bx + bw)
            and max(ay, by) < min(ay + ah, by + bh)
        )

    def texel_gap(self, other: Region) -> float:
        """Chebyshev distance between the two marks' texels.

        The question the first pipeline asked of bounding boxes, asked of the
        marks instead.  ``0`` means they touch; a frame whose rectangle
        encloses a glyph is nowhere near it, and says so here.
        """
        if not self.overlaps(other) and _box_gap(self.bounds, other.bounds) > 0:
            # Rectangles apart by n cannot have texels closer than n.
            floor = _box_gap(self.bounds, other.bounds)
        else:
            floor = 0
        ys, xs = np.nonzero(self.mask)
        oys, oxs = np.nonzero(other.mask)
        if not ys.size or not oys.size:
            return float("inf")
        dx = np.abs((xs + self.x)[:, None] - (oxs + other.x)[None, :])
        dy = np.abs((ys + self.y)[:, None] - (oys + other.y)[None, :])
        return float(max(np.minimum.reduce(np.maximum(dx, dy).ravel()), floor))

    def __repr__(self) -> str:
        return (
            f"Region({self.bounds}, texels={self.texels}, fill={self.fill:.3f}, "
            f"enclosure={self.enclosure:.2f}, stroke={self.stroke:.1f})"
        )

    def _memo(self, key: str, produce):
        if key not in self._cache:
            self._cache[key] = produce()
        return self._cache[key]


def _box_gap(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> int:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    return max(bx - (ax + aw), ax - (bx + bw), by - (ay + ah), ay - (by + bh), 0)
