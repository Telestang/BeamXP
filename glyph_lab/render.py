"""Turning a texture and its detection into a picture, with nothing tkinter in it.

Kept pure and separate on purpose.  The equivalent code in the first app lived
inside the viewer's ``render`` method, sliced a numpy array with the viewer's
float rectangle, and blanked the whole canvas the first time anyone ticked the
box -- because an exception in a paint method has nowhere to go.  Everything
here takes plain arrays and returns an image, so it can be checked without a
display.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from glyph_lab.corpus import UvDomainGeometry

TEXEL_COLOUR = (255, 176, 32)
REGION_COLOUR = (60, 255, 60)
UV_EDGE_COLOUR = (70, 160, 255)
# Where the mesh stops being one continuous surface, drawn whether or not
# anything fired there -- the same job the blue does for the UV domains.  Violet
# so it reads against both of them.
MESH_SEAM_COLOUR = (176, 96, 255)
UV_SYMMETRY_COLOUR = (32, 232, 232)
PARALLELOGRAM_COLOUR = (255, 138, 42)
# The blank ground the marks stand on.  Drawn under everything else, because it
# is the field the rest sits on and it covers most of a clean panel -- over the
# top it would bury the very texels it is there to be read against.
SPARSE_COLOUR = (150, 70, 230)
OUTSIDE_UV_DIM = 0.30
TEXEL_ALPHA_LIFT = 4.0
# What shows where the view runs past the edge of the texture.
BACKGROUND = (18, 18, 18)


@dataclass(frozen=True, slots=True)
class ViewOptions:
    show_texels: bool = True
    show_regions: bool = True
    dim_outside_uv: bool = True
    show_uv_edges: bool = False
    show_mesh_seams: bool = False
    show_uv_geometry: bool = False
    show_sparse: bool = False
    # How far the texture is knocked back outside any UV island.
    outside_uv_dim: float = OUTSIDE_UV_DIM
    # A 4k atlas in a contact-sheet cell is around 1/16 scale, where a
    # one-texel loop covers a sixteenth of a screen pixel.  Area-averaging
    # keeps it from vanishing; this lifts what is left until it reads.
    texel_alpha_lift: float = TEXEL_ALPHA_LIFT


DEFAULT_VIEW = ViewOptions()


def snap_rect(
    rect: tuple[float, float, float, float], shape: tuple[int, int]
) -> tuple[int, int, int, int] | None:
    """Whole-texel bounds for a float view rectangle, clamped to the texture.

    Numpy cannot slice between texels and PIL's resize box can, which is
    exactly the mismatch that broke the first version of this.  Snapping
    outwards keeps the view a superset of what was asked for.
    """
    height, width = shape
    x0 = max(int(np.floor(rect[0])), 0)
    y0 = max(int(np.floor(rect[1])), 0)
    x1 = min(int(np.ceil(rect[2])), width)
    y1 = min(int(np.ceil(rect[3])), height)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def compose_view(
    colour: np.ndarray,
    domains: np.ndarray,
    fired: np.ndarray | None,
    outlines: np.ndarray | None,
    rect: tuple[float, float, float, float],
    out_size: tuple[int, int],
    options: ViewOptions = DEFAULT_VIEW,
    boundaries: np.ndarray | None = None,
    seams: np.ndarray | None = None,
    sparse: np.ndarray | None = None,
    uv_geometry: tuple[UvDomainGeometry, ...] = (),
    geometry_seams: np.ndarray | None = None,
) -> Image.Image:
    """Render one rectangle of a texture at ``out_size``, with its overlays.

    ``rect`` is what the caller wants to see and may run past the texture on
    any side -- a square atlas fitted into a wide cell does exactly that, and
    so does zooming out.  Where it does, the texture is placed at its true
    scale and the rest is left as background.

    2026-08-26: the first version clamped the rectangle to the texture and
    then resized *that* to the full output, which silently stretched every
    atlas to its cell's aspect and put every region box at the wrong width.
    It also made zooming out look broken, because a clamped view refills the
    canvas however far out you go.
    """
    out_width, out_height = (max(int(value), 1) for value in out_size)
    canvas = Image.new("RGB", (out_width, out_height), BACKGROUND)
    left, top, right, bottom = rect
    span_x, span_y = right - left, bottom - top
    if span_x <= 0 or span_y <= 0:
        return canvas
    # One scale per axis, taken from the rectangle that was *asked* for.  For a
    # fitted cell or a zoom step the two are equal; deriving them separately
    # costs nothing and keeps the function honest about any other rectangle.
    scale_x, scale_y = out_width / span_x, out_height / span_y

    snapped = snap_rect(rect, colour.shape[:2])
    if snapped is not None:
        x0, y0, x1, y1 = snapped
        target_w = max(round((x1 - x0) * scale_x), 1)
        target_h = max(round((y1 - y0) * scale_y), 1)
        at = (round((x0 - left) * scale_x), round((y0 - top) * scale_y))

        patch = colour[y0:y1, x0:x1]
        if options.dim_outside_uv:
            outside = domains[y0:y1, x0:x1] == 0
            patch = patch.copy()
            patch[outside] = (
                patch[outside] * options.outside_uv_dim
            ).astype(np.uint8)

        shrinking = target_w < (x1 - x0)
        patch_image = Image.fromarray(np.ascontiguousarray(patch)).resize(
            (target_w, target_h), Image.BOX if shrinking else Image.NEAREST
        )

        # Bottom to top: where the geometry ends, what fired, and last the
        # paths drawn round it.  Where a mark sits on a seam what matters is
        # seeing the mark cross the line.
        #
        # The seams go above the UV edges rather than below.  The two coincide
        # wherever an island was not coalesced with anything, and it is the
        # seams the eye needs to find -- the interesting ones are precisely the
        # lines with no blue under them.
        washes = []
        if options.show_sparse and sparse is not None:
            washes.append((sparse[y0:y1, x0:x1], SPARSE_COLOUR))
        if options.show_uv_edges and boundaries is not None:
            washes.append((boundaries[y0:y1, x0:x1], UV_EDGE_COLOUR))
        if options.show_mesh_seams and seams is not None:
            washes.append((seams[y0:y1, x0:x1], MESH_SEAM_COLOUR))
        if options.show_uv_geometry and geometry_seams is not None:
            washes.append((geometry_seams[y0:y1, x0:x1], UV_SYMMETRY_COLOUR))
        if options.show_texels and fired is not None:
            washes.append((fired[y0:y1, x0:x1], TEXEL_COLOUR))
        # The region paths go on top of everything: they are the answer, and
        # the rest is how it was arrived at.
        if options.show_regions and outlines is not None:
            washes.append((outlines[y0:y1, x0:x1], REGION_COLOUR))
        for window, wash_colour in washes:
            if not window.any():
                continue
            alpha = Image.fromarray(
                np.ascontiguousarray(window).astype(np.uint8) * 255
            ).resize((target_w, target_h), Image.BOX)
            if shrinking:
                lift = options.texel_alpha_lift
                alpha = alpha.point(
                    lambda value, lift=lift: min(int(value * lift), 255)
                )
            patch_image.paste(
                Image.new("RGB", (target_w, target_h), wash_colour), (0, 0), alpha
            )
        canvas.paste(patch_image, at)

    if options.show_uv_geometry and uv_geometry:
        draw = ImageDraw.Draw(canvas)

        def project(point: tuple[float, float]) -> tuple[float, float]:
            return ((point[0] - left) * scale_x, (point[1] - top) * scale_y)

        # 2026-08-30: output coordinates follow the same floating crop used
        # for pan and zoom, instead of snapping geometry to source texels.
        for item in uv_geometry:
            if item.kind == "symmetric" and item.axis is not None:
                draw.line([project(point) for point in item.axis], fill=UV_SYMMETRY_COLOUR,
                          width=2)
            elif item.kind == "parallelogram" and len(item.corners) == 4:
                points = [project(point) for point in item.corners]
                draw.line([*points, points[0]], fill=PARALLELOGRAM_COLOUR, width=2)
            elif item.kind == "parallelogram" and item.segments:
                for segment in item.segments:
                    draw.line([project(point) for point in segment],
                              fill=PARALLELOGRAM_COLOUR, width=2)

    return canvas


def fit_rect(
    shape: tuple[int, int], out_size: tuple[int, int]
) -> tuple[float, float, float, float]:
    """The whole texture, letterboxed into a cell without distorting it."""
    height, width = shape
    out_width, out_height = out_size
    scale = min(out_width / width, out_height / height)
    span_x = out_width / scale
    span_y = out_height / scale
    return (
        (width - span_x) / 2.0, (height - span_y) / 2.0,
        (width + span_x) / 2.0, (height + span_y) / 2.0,
    )
