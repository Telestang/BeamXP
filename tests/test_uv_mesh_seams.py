"""Where the mesh stops being one continuous surface, read off the topology.

``discontinuous_edge_mask`` answers a question the rasterised UV islands cannot,
because coalescing throws the answer away: two islands whose masks overlap
become one domain, and the seam between them leaves no trace in the label image
at all.  So what is pinned here is the topological rule itself -- an edge is
continuous only when exactly two triangles carry the same corner keys -- and
that it keeps answering the same way when the mesh is authored in the awkward
ways real DAEs are: duplicated vertices, mirrored charts, UVs outside the tile.
"""

from __future__ import annotations

import unittest
from copy import deepcopy
from xml.etree import ElementTree as ET

import numpy as np

from mesh_segmentation_transform.extract_uv_island_paths import (
    _clip_segment_to_unit_tile,
    discontinuous_edge_mask,
    uv_island_masks,
)

Corner = tuple[tuple[float, float], tuple[float, float, float]]


def dae(triangles: list[tuple[int, int, int]], corners: list[Corner]) -> ET.Element:
    """The smallest COLLADA document the extractor will read.

    One shared index for POSITION and TEXCOORD, so a corner is a (uv, xyz)
    pair and splitting one is a matter of listing it twice -- which is exactly
    how a seam is authored.
    """
    uvs: list[float] = []
    positions: list[float] = []
    for (u, v), (x, y, z) in corners:
        uvs += [u, v]
        positions += [x, y, z]
    indices = " ".join(str(i) for triangle in triangles for i in triangle)
    return ET.fromstring(
        f"""<COLLADA><library_geometries><geometry><mesh>
          <source id="P"><float_array>{" ".join(map(str, positions))}</float_array>
            <technique_common><accessor stride="3"/></technique_common></source>
          <source id="T"><float_array>{" ".join(map(str, uvs))}</float_array>
            <technique_common><accessor stride="2"/></technique_common></source>
          <vertices id="V"><input semantic="POSITION" source="#P"/></vertices>
          <triangles material="m" count="{len(triangles)}">
            <input semantic="VERTEX" source="#V" offset="0"/>
            <input semantic="TEXCOORD" source="#T" offset="0"/>
            <p>{indices}</p></triangles></mesh></geometry></library_geometries></COLLADA>"""
    )


# A square chart with a little margin, split into two triangles across the
# anti-diagonal.  Every test below is some way of cutting or not cutting it.
QUAD: list[Corner] = [
    ((0.1, 0.1), (0.0, 0.0, 0.0)),
    ((0.9, 0.1), (1.0, 0.0, 0.0)),
    ((0.9, 0.9), (1.0, 1.0, 0.0)),
    ((0.1, 0.9), (0.0, 1.0, 0.0)),
]
SPLIT = [(0, 1, 2), (0, 2, 3)]
SIZE = (64, 64)


def seams(triangles, corners, size=SIZE) -> tuple[np.ndarray, dict]:
    return discontinuous_edge_mask(dae(triangles, corners), "m", size)


# The chart's corners sit *on* its anti-diagonal, so counting the whole band
# picks up six texels of outline whether or not the diagonal itself was drawn.
# Staying well inside the border is what makes this measure the join alone.
INTERIOR = slice(12, 52)


def diagonal(mask: np.ndarray) -> int:
    """Texels on the chart's anti-diagonal, where the two triangles meet."""
    inside = np.zeros(mask.shape, dtype=bool)
    inside[INTERIOR, INTERIOR] = True
    rows, columns = np.nonzero(mask & inside)
    return int(np.count_nonzero(np.abs((rows + columns) - (SIZE[0] - 1)) <= 1))


class ContinuityTests(unittest.TestCase):
    def test_a_shared_edge_is_not_a_seam(self) -> None:
        mask, stats = seams(SPLIT, QUAD)
        self.assertEqual(stats["edges"], 5, "six edge slots, one of them shared")
        self.assertEqual(stats["discontinuous_edges"], 4, "the outline, and no more")
        self.assertEqual(diagonal(mask), 0)

    def test_the_same_edge_becomes_a_seam_when_the_surface_moves(self) -> None:
        """Same UVs, but the far triangle is somewhere else in space.

        This is the case the surface half of the key exists for, and the one an
        atlas-only test gets wrong: in UV the two triangles still meet exactly.
        """
        moved = QUAD + [
            ((0.1, 0.1), (0.0, 0.0, 9.0)),
            ((0.9, 0.9), (1.0, 1.0, 9.0)),
        ]
        mask, stats = seams([(0, 1, 2), (4, 5, 3)], moved)
        self.assertEqual(stats["edges"], 6, "nothing is shared any more")
        self.assertEqual(stats["discontinuous_edges"], 6)
        self.assertGreater(diagonal(mask), 30, "the diagonal is drawn now")

    def test_a_split_uv_over_one_surface_is_a_seam_on_both_lips(self) -> None:
        """The other half of the pair: continuous in space, cut in the atlas.

        A UV seam has two lips, in different places in the atlas, and each is a
        border where the local mean goes one-sided.  Both have to be drawn --
        marking only the near one would leave the far chart's edge unfiltered.
        """
        # The right-hand triangle keeps its 3D positions but is packed apart.
        elsewhere = QUAD + [
            ((0.1, 0.1), (0.0, 0.0, 0.0)),
            ((0.9, 0.9), (1.0, 1.0, 0.0)),
        ]
        # Move that copy's UVs to a different corner of the atlas.
        elsewhere[4] = ((0.3, 0.4), elsewhere[4][1])
        elsewhere[5] = ((0.6, 0.7), elsewhere[5][1])
        _mask, stats = seams([(0, 1, 2), (4, 5, 3)], elsewhere)
        self.assertEqual(stats["discontinuous_edges"], 6, "both lips, plus the rest")

    def test_duplicated_vertices_still_weld(self) -> None:
        """A hard edge splits a vertex in the DAE; it is not a discontinuity.

        Split normals, split tangents and per-face materials all produce this,
        so a rule keyed on vertex index rather than on position would report a
        seam down every crease in the vehicle.
        """
        duplicated = QUAD + [QUAD[0], QUAD[2]]
        mask, stats = seams([(0, 1, 2), (4, 5, 3)], duplicated)
        self.assertEqual(stats["discontinuous_edges"], 4, "the outline, as before")
        self.assertEqual(diagonal(mask), 0)

    def test_a_third_face_on_an_edge_does_not_cut_it(self) -> None:
        """Non-manifold: a fin butts into a surface that carries on regardless.

        The rule is "at least two carriers", not "exactly two", and this is the
        case that decides it.  Two of the three faces do carry the surface
        through the edge, so the texture across it is continuous and there is
        nothing to filter; requiring exactly two drew a line down every one of
        them -- 3.6% of scintilla's open edges, cutting across plainly
        continuous panels.
        """
        fanned = QUAD + [((0.5, 0.5), (0.5, 0.5, 5.0))]
        _mask, stats = seams([(0, 1, 2), (0, 2, 3), (0, 2, 4)], fanned)
        self.assertEqual(
            [stats["edges"], stats["discontinuous_edges"]], [7, 6],
            "the three-carrier edge should be the one edge left uncut",
        )
        # Not measured with `diagonal`: the fin's own two free edges run from
        # the chart's corners to its apex, so they lie along that same
        # anti-diagonal -- and they are borders, correctly drawn.


class PlacementTests(unittest.TestCase):
    def test_geometry_ids_exclude_another_mesh_using_the_same_material(self) -> None:
        document = dae(SPLIT, QUAD)
        library = document.find("library_geometries")
        assert library is not None
        dashboard = library.find("geometry")
        assert dashboard is not None
        dashboard.set("id", "dashboard")
        pedal = deepcopy(dashboard)
        pedal.set("id", "pedal")
        library.append(pedal)

        _all, all_stats = uv_island_masks(document, "m", SIZE)
        _dashboard, selected_stats = uv_island_masks(
            document, "m", SIZE, frozenset({"dashboard"}),
        )

        self.assertEqual(all_stats["geometries"], 2)
        self.assertEqual(selected_stats["geometries"], 1)

    def test_the_seams_are_the_islands_own_outlines(self) -> None:
        """The two functions weld on the same key, so they cannot disagree.

        Not an implementation detail: it is the reason a seam mask can be
        trusted to cover what the domain erosion covers *and* the interior
        lines the erosion never sees.
        """
        document = dae(SPLIT, QUAD)
        mask, _stats = discontinuous_edge_mask(document, "m", SIZE)
        crops, _island_stats = uv_island_masks(document, "m", SIZE)
        self.assertEqual(len(crops), 1, "one welded chart")
        island = np.zeros(SIZE, dtype=bool)
        island[
            crops[0].y:crops[0].y + crops[0].mask.shape[0],
            crops[0].x:crops[0].x + crops[0].mask.shape[1],
        ] = crops[0].mask
        # Every seam texel lands on the island, within the texel of disagreement
        # between a Bresenham line and a filled polygon.
        from scipy import ndimage
        self.assertTrue(
            (mask & ~ndimage.binary_dilation(island, iterations=1)).sum() == 0,
            "a seam was drawn off the chart it belongs to",
        )

    def test_uvs_outside_the_tile_land_where_the_geometry_did(self) -> None:
        """Tiled UVs are wrapped, not clamped onto the atlas border.

        A chart authored at u in 1..2 rasterises into the same atlas as one at
        0..1, and its edges have to follow it there rather than piling up in
        column 0.
        """
        shifted = [((u + 1.0, v), point) for (u, v), point in QUAD]
        mask, _stats = seams(SPLIT, shifted)
        plain, _ = seams(SPLIT, QUAD)
        self.assertTrue(np.array_equal(mask, plain))

    def test_a_chart_with_no_triangles_is_an_error_not_an_empty_mask(self) -> None:
        """Same contract as `uv_island_masks`, which the caller already handles."""
        with self.assertRaises(ValueError):
            discontinuous_edge_mask(dae(SPLIT, QUAD), "other-material", SIZE)


class SegmentClipTests(unittest.TestCase):
    """Liang-Barsky, because the polygon clipper cannot do segments."""

    def test_a_segment_inside_the_tile_is_returned_whole(self) -> None:
        self.assertEqual(
            _clip_segment_to_unit_tile((0.2, 0.2), (0.8, 0.8)),
            ((0.2, 0.2), (0.8, 0.8)),
        )

    def test_a_segment_crossing_the_border_is_cut_at_it(self) -> None:
        self.assertEqual(
            _clip_segment_to_unit_tile((-0.5, 0.5), (0.5, 0.5)),
            ((0.0, 0.5), (0.5, 0.5)),
        )

    def test_a_segment_outside_the_tile_is_dropped(self) -> None:
        self.assertIsNone(_clip_segment_to_unit_tile((1.5, 0.5), (2.0, 0.5)))

    def test_a_segment_lying_along_the_border_survives(self) -> None:
        """Parallel to an edge and exactly on it: the degenerate case that
        divides by zero if the parallel branch is not handled."""
        self.assertEqual(
            _clip_segment_to_unit_tile((0.0, 0.0), (0.0, 1.0)),
            ((0.0, 0.0), (0.0, 1.0)),
        )


if __name__ == "__main__":
    unittest.main()
