"""Compositing for the Glyph Lab viewer.

Separate from the app and tested without a display, because the equivalent
code in the first harness lived inside a paint method, sliced a numpy array
with the viewer's float rectangle, and blanked the canvas the first time the
overlay was ticked.  Every rectangle here is float, as the viewer's are.
"""

from __future__ import annotations

import unittest

import numpy as np

from glyph_lab.corpus import UvDomainGeometry, domain_boundaries
from glyph_lab.render import (
    OUTSIDE_UV_DIM,
    PARALLELOGRAM_COLOUR,
    REGION_COLOUR,
    TEXEL_COLOUR,
    UV_EDGE_COLOUR,
    UV_SYMMETRY_COLOUR,
    ViewOptions,
    compose_view,
    fit_rect,
    snap_rect,
)


def texture(size: int = 64):
    colour = np.full((size, size, 3), 90, dtype=np.uint8)
    domains = np.zeros((size, size), dtype=np.uint16)
    domains[8:size - 8, 8:size - 8] = 1
    fired = np.zeros((size, size), dtype=bool)
    fired[16, 16:48] = True          # a one-texel horizontal stroke
    return colour, domains, fired


class SnapRectTests(unittest.TestCase):
    def test_it_snaps_outwards(self) -> None:
        self.assertEqual(snap_rect((10.9, 20.1, 30.2, 40.8), (64, 64)), (10, 20, 31, 41))

    def test_it_clamps_to_the_texture(self) -> None:
        self.assertEqual(snap_rect((-9.0, -9.0, 999.0, 999.0), (64, 64)), (0, 0, 64, 64))

    def test_it_declines_an_empty_rectangle(self) -> None:
        self.assertIsNone(snap_rect((70.0, 70.0, 90.0, 90.0), (64, 64)))
        self.assertIsNone(snap_rect((10.0, 10.0, 10.0, 20.0), (64, 64)))


class ComposeViewTests(unittest.TestCase):
    def test_a_float_rectangle_renders_at_the_size_asked_for(self) -> None:
        colour, domains, fired = texture()
        image = compose_view(colour, domains, fired, None,
                             (3.5, 7.25, 60.75, 61.5), (200, 200))
        self.assertEqual(image.size, (200, 200))

    def test_an_offscreen_rectangle_gives_a_blank_of_the_right_size(self) -> None:
        colour, domains, fired = texture()
        image = compose_view(colour, domains, fired, None, (500.0, 500.0, 600.0, 600.0),
                             (64, 48))
        self.assertEqual(image.size, (64, 48))

    def test_the_texel_overlay_paints_amber_where_it_fired(self) -> None:
        colour, domains, fired = texture()
        image = compose_view(colour, domains, fired, None, (0.0, 0.0, 64.0, 64.0),
                             (64, 64), ViewOptions(show_texels=True, show_regions=False))
        self.assertEqual(image.getpixel((20, 16)), TEXEL_COLOUR)
        self.assertNotEqual(image.getpixel((20, 30)), TEXEL_COLOUR)

    def test_the_texel_overlay_can_be_switched_off(self) -> None:
        colour, domains, fired = texture()
        image = compose_view(
            colour, domains, fired, None, (0.0, 0.0, 64.0, 64.0), (64, 64),
            ViewOptions(show_texels=False, show_regions=False),
        )
        self.assertNotEqual(image.getpixel((20, 16)), TEXEL_COLOUR)

    def test_a_one_texel_stroke_survives_being_shrunk(self) -> None:
        """Sampling would drop it; area-averaging plus the lift keeps it."""
        colour, domains, fired = texture(512)
        fired[:] = False  # replaced by a single long stroke below
        fired[256, 64:448] = True
        image = compose_view(colour, domains, fired, None, (0.0, 0.0, 512.0, 512.0),
                             (64, 64), ViewOptions(show_texels=True, show_regions=False))
        pixels = np.asarray(image)
        self.assertTrue((pixels[:, :, 0] > pixels[:, :, 2]).any(), "no amber survived")

    def test_regions_outside_the_view_are_not_drawn(self) -> None:
        colour, domains, _fired = texture()
        outlines = np.zeros((64, 64), dtype=bool)
        outlines[20:24, 20:24] = True
        outlines[60:64, 60:64] = True
        image = compose_view(
            colour, domains, None, outlines, (0.0, 0.0, 40.0, 40.0),
            (40, 40), ViewOptions(show_texels=False),
        )
        pixels = np.asarray(image)
        drawn = (pixels == np.array(REGION_COLOUR, dtype=np.uint8)).all(axis=2)
        self.assertTrue(drawn.any(), "the region inside the view was not drawn")
        self.assertFalse(drawn[35:, 35:].any(), "a region outside the view was drawn")

    def test_outside_the_uv_domain_is_dimmed_only_when_asked(self) -> None:
        colour, domains, _fired = texture()
        rect, size = (0.0, 0.0, 64.0, 64.0), (64, 64)
        # 2026-09-02: the brightness is passed explicitly.  ``outside_uv_dim``
        # defaults to 1.0 -- no wash at all -- so leaving it out would compare
        # an undimmed view against an undimmed view and pass on anything.
        dim = compose_view(colour, domains, None, None, rect, size,
                           ViewOptions(show_texels=False, show_regions=False,
                                       outside_uv_dim=OUTSIDE_UV_DIM))
        plain = compose_view(colour, domains, None, None, rect, size,
                             ViewOptions(show_texels=False, show_regions=False,
                                         outside_uv_dim=OUTSIDE_UV_DIM,
                                         dim_outside_uv=False))
        self.assertLess(dim.getpixel((2, 2))[0], plain.getpixel((2, 2))[0])
        self.assertEqual(dim.getpixel((32, 32)), plain.getpixel((32, 32)))


class UvEdgeTests(unittest.TestCase):
    def test_boundaries_mark_the_inside_of_an_island(self) -> None:
        domains = np.zeros((16, 16), dtype=np.uint16)
        domains[4:12, 4:12] = 1
        edges = domain_boundaries(domains)
        self.assertTrue(edges[4, 6], "the island's own edge is not marked")
        self.assertFalse(edges[3, 6], "the bare atlas outside it is marked")
        self.assertFalse(edges[8, 8], "the island's interior is marked")

    def test_two_domains_meeting_mark_both_sides(self) -> None:
        domains = np.zeros((16, 16), dtype=np.uint16)
        domains[:, :8] = 1
        domains[:, 8:] = 2
        edges = domain_boundaries(domains)
        self.assertTrue(edges[8, 7])
        self.assertTrue(edges[8, 8])
        self.assertFalse(edges[8, 4])

    def test_the_overlay_draws_only_when_asked(self) -> None:
        colour, domains, _fired = texture()
        edges = domain_boundaries(domains)
        rect, size = (0.0, 0.0, 64.0, 64.0), (64, 64)
        off = compose_view(colour, domains, None, None, rect, size,
                           ViewOptions(show_texels=False, show_regions=False),
                           edges)
        on = compose_view(colour, domains, None, None, rect, size,
                          ViewOptions(show_texels=False, show_regions=False,
                                      show_uv_edges=True),
                          edges)
        self.assertEqual(on.getpixel((20, 8)), UV_EDGE_COLOUR)
        self.assertNotEqual(off.getpixel((20, 8)), UV_EDGE_COLOUR)

    def test_the_detector_texels_sit_on_top_of_the_edges(self) -> None:
        """Where a mark crosses a seam, the mark is what you need to see."""
        colour, domains, fired = texture()
        edges = domain_boundaries(domains)
        fired[:] = False
        fired[8, 16:48] = True          # laid along the island's top edge
        image = compose_view(
            colour, domains, fired, None, (0.0, 0.0, 64.0, 64.0), (64, 64),
            ViewOptions(show_texels=True, show_regions=False, show_uv_edges=True),
            edges,
        )
        self.assertEqual(image.getpixel((20, 8)), TEXEL_COLOUR)


class UvGeometryTests(unittest.TestCase):
    def test_the_axis_and_source_fitted_frame_draw_in_their_own_colours(self) -> None:
        colour, domains, _fired = texture()
        geometry = (
            UvDomainGeometry(1, "symmetric", axis=((20.0, 8.0), (20.0, 56.0))),
            UvDomainGeometry(1, "parallelogram", corners=(
                (32.0, 20.0), (48.0, 20.0), (52.0, 36.0), (36.0, 36.0),
            )),
        )
        image = compose_view(
            colour, domains, None, None, (0.0, 0.0, 64.0, 64.0), (64, 64),
            ViewOptions(show_texels=False, show_regions=False, show_uv_geometry=True),
            uv_geometry=geometry,
        )
        self.assertEqual(image.getpixel((20, 32)), UV_SYMMETRY_COLOUR)
        self.assertEqual(image.getpixel((40, 20)), PARALLELOGRAM_COLOUR)

    def test_geometry_follows_a_cropped_zoom(self) -> None:
        colour, domains, _fired = texture()
        geometry = (UvDomainGeometry(
            1, "parallelogram", corners=((20.0, 20.0), (28.0, 20.0),
                                            (28.0, 28.0), (20.0, 28.0)),
        ),)
        image = compose_view(
            colour, domains, None, None, (16.0, 16.0, 32.0, 32.0), (64, 64),
            ViewOptions(show_texels=False, show_regions=False, show_uv_geometry=True),
            uv_geometry=geometry,
        )
        # Source (24, 20) lands at output (32, 16) in this 4x zoom.
        self.assertEqual(image.getpixel((32, 16)), PARALLELOGRAM_COLOUR)

    def test_a_wrapped_frame_draws_only_its_in_tile_edge_pieces(self) -> None:
        colour, domains, _fired = texture()
        geometry = (UvDomainGeometry(
            1, "parallelogram", segments=(
                ((0.0, 20.0), (12.0, 20.0)), ((52.0, 20.0), (63.0, 20.0)),
            ),
        ),)
        image = compose_view(
            colour, domains, None, None, (0.0, 0.0, 64.0, 64.0), (64, 64),
            ViewOptions(show_texels=False, show_regions=False, show_uv_geometry=True),
            uv_geometry=geometry,
        )
        self.assertEqual(image.getpixel((6, 20)), PARALLELOGRAM_COLOUR)
        self.assertEqual(image.getpixel((58, 20)), PARALLELOGRAM_COLOUR)
        self.assertNotEqual(image.getpixel((32, 20)), PARALLELOGRAM_COLOUR)

    def test_geometry_uses_the_mesh_seam_context(self) -> None:
        colour, domains, _fired = texture()
        seams = np.zeros((64, 64), dtype=bool)
        seams[20, 8:56] = True
        image = compose_view(
            colour, domains, None, None, (0.0, 0.0, 64.0, 64.0), (64, 64),
            ViewOptions(show_texels=False, show_regions=False, show_uv_geometry=True),
            geometry_seams=seams,
        )
        self.assertEqual(image.getpixel((32, 20)), UV_SYMMETRY_COLOUR)


class LetterboxTests(unittest.TestCase):
    """A view wider than the texture must not stretch it.

    The first version clamped the rectangle to the texture and resized that to
    the full output, so every square atlas filled its wide cell, every region
    box was drawn at the wrong width, and zooming out refilled the canvas
    instead of shrinking the texture.  The tests that existed all used a
    rectangle that matched the texture exactly, and saw none of it.
    """

    def test_a_square_texture_in_a_wide_cell_keeps_its_shape(self) -> None:
        colour, domains, _fired = texture(64)
        colour[:] = 200                                  # bright, so bars show
        cell = (200, 100)
        image = compose_view(
            colour, domains, None, None, fit_rect((64, 64), cell), cell,
            ViewOptions(show_texels=False, show_regions=False,
                        dim_outside_uv=False),
        )
        pixels = np.asarray(image)
        lit_columns = np.flatnonzero((pixels[:, :, 0] > 150).any(axis=0))
        # 64x64 into 200x100 fits at 100/64, so the texture is 100 wide and
        # centred: bars either side, not a stretch across all 200.
        self.assertEqual(len(lit_columns), 100)
        self.assertAlmostEqual(lit_columns.mean(), 99.5, delta=1.0)
        self.assertTrue((pixels[:, 0] < 150).all(), "no bar on the left")
        self.assertTrue((pixels[:, -1] < 150).all(), "no bar on the right")

    def test_a_region_lands_where_the_texture_does(self) -> None:
        colour, domains, _fired = texture(64)
        outlines = np.zeros((64, 64), dtype=bool)
        outlines[0:8, 0:8] = True                        # the texture's corner
        cell = (200, 100)
        image = compose_view(
            colour, domains, None, outlines, fit_rect((64, 64), cell), cell,
            ViewOptions(show_texels=False, dim_outside_uv=False),
        )
        drawn = (np.asarray(image) == np.array(REGION_COLOUR, np.uint8)).all(axis=2)
        columns = np.flatnonzero(drawn.any(axis=0))
        # The texture starts 50 px in, and 8 texels at 100/64 is 12.5 px.
        self.assertAlmostEqual(int(columns.min()), 50, delta=1)
        self.assertAlmostEqual(int(columns.max()), 62, delta=2)

    def test_zooming_out_shrinks_the_texture(self) -> None:
        colour, domains, _fired = texture(64)
        colour[:] = 200
        options = ViewOptions(show_texels=False, show_regions=False,
                              dim_outside_uv=False)
        span = []
        for zoom in (64.0, 128.0, 256.0):               # ever wider view
            image = compose_view(
                colour, domains, None, None,
                (0.0, 0.0, zoom, zoom), (128, 128), options,
            )
            pixels = np.asarray(image)
            span.append(int((pixels[:, :, 0] > 150).any(axis=0).sum()))
        self.assertEqual(span, sorted(span, reverse=True), f"did not shrink: {span}")
        self.assertLess(span[-1], span[0] // 2)


class FitRectTests(unittest.TestCase):
    def test_a_square_texture_in_a_wide_cell_is_letterboxed_not_stretched(self) -> None:
        x0, y0, x1, y1 = fit_rect((100, 100), (200, 100))
        self.assertAlmostEqual(x1 - x0, 200.0)   # the wide axis spans further
        self.assertAlmostEqual(y1 - y0, 100.0)
        self.assertAlmostEqual((x0 + x1) / 2, 50.0)   # still centred
        self.assertAlmostEqual((y0 + y1) / 2, 50.0)


if __name__ == "__main__":
    unittest.main()
