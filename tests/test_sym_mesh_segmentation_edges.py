from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from mesh_segmentation_transform.beamxp_transform_sym_mesh_POC import (
    IslandCandidate,
    SourceFaceRef,
    _candidate_union_boundary,
    _classify_region,
    _find_eager_child_adoptions,
    _IslandGeometry,
    _resolve_touching_symmetric_unions,
    _surface_points_and_samples,
    analyse_symmetry_sweep,
    build_topology,
    measure_perimeter_symmetry,
    pseudo_aspect_ratio_from_area_perimeter,
)


def _open_square_topology():
    vertices = np.array(
        [
            [-1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
            [1.0, 0.0, 1.0],
            [-1.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    triangles = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
        ],
        dtype=np.int64,
    )
    source_faces = [
        SourceFaceRef(0, "test", 0, 0),
        SourceFaceRef(0, "test", 0, 1),
    ]
    return build_topology(vertices, triangles, source_faces)


def _curved_host_with_child_topology():
    host_bulge = 0.023
    child_offset = host_bulge + 0.010
    vertices = np.array(
        [
            [-1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
            [1.0, 0.0, 1.0],
            [-1.0, 0.0, 1.0],
            [0.0, host_bulge, 0.0],
            [-0.010, child_offset, -0.010],
            [0.010, child_offset, -0.010],
            [0.010, child_offset, 0.010],
            [-0.010, child_offset, 0.010],
        ],
        dtype=float,
    )
    triangles = np.array(
        [
            [0, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4],
            [5, 6, 7],
            [5, 7, 8],
        ],
        dtype=np.int64,
    )
    source_faces = [
        SourceFaceRef(0, "test", 0, triangle_index)
        for triangle_index in range(len(triangles))
    ]
    return build_topology(vertices, triangles, source_faces)


def _flat_panel_with_proud_child_topology():
    vertices = np.array(
        [
            [-1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
            [1.0, 0.0, 1.0],
            [-1.0, 0.0, 1.0],
            [-0.050, 0.030, -0.050],
            [0.050, 0.030, -0.050],
            [0.050, 0.030, 0.050],
            [-0.050, 0.030, 0.050],
        ],
        dtype=float,
    )
    triangles = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
        ],
        dtype=np.int64,
    )
    source_faces = [
        SourceFaceRef(0, "test", 0, triangle_index)
        for triangle_index in range(len(triangles))
    ]
    return build_topology(vertices, triangles, source_faces)


def _touching_symmetric_regions_topology():
    vertices = np.array(
        [
            [-2.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [2.0, 0.0, -1.0],
            [-2.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    triangles = np.array(
        [
            [0, 1, 4],
            [0, 4, 3],
            [1, 2, 5],
            [1, 5, 4],
        ],
        dtype=np.int64,
    )
    source_faces = [
        SourceFaceRef(0, "test", 0, triangle_index)
        for triangle_index in range(len(triangles))
    ]
    return build_topology(vertices, triangles, source_faces)


def _candidate_for_faces(topology, key: tuple[int, int], faces: tuple[int, ...]) -> IslandCandidate:
    boundary = _candidate_union_boundary(topology, faces)
    measurement = measure_perimeter_symmetry(
        topology.vertices,
        boundary.loops,
        sample_spacing=0.25,
        rms_tolerance=1e-9,
        direct_rms_tolerance=1e-9,
    )
    area = float(topology.face_areas[list(faces)].sum())
    return IslandCandidate(
        key=key,
        island_index=1,
        accepted_level=1,
        accepted_angle=90.0,
        faces=faces,
        area=area,
        pseudo_aspect_ratio=pseudo_aspect_ratio_from_area_perimeter(
            area,
            boundary.perimeter,
        ),
        boundary_edges=boundary.edges,
        boundary_loops=boundary.loops,
        perimeter=boundary.perimeter,
        measurement=measurement,
        host_faces=faces,
    )


class SymMeshSegmentationEdgeTests(unittest.TestCase):
    def test_pseudo_aspect_ratio_matches_rectangle_aspect(self) -> None:
        scale = 0.37
        aspect = 5.0
        area = aspect * scale * scale
        perimeter = 2.0 * (aspect + 1.0) * scale

        self.assertAlmostEqual(
            pseudo_aspect_ratio_from_area_perimeter(area, perimeter),
            aspect,
        )

    def test_pseudo_aspect_ratio_clamps_compact_shapes_to_one(self) -> None:
        self.assertEqual(pseudo_aspect_ratio_from_area_perimeter(math.pi, 2.0 * math.pi), 1.0)

    def test_symmetry_that_cannot_reach_direct_fit_is_rejected(self) -> None:
        vertices = np.array(
            [
                [-1.0, 0.0, -1.0],
                [1.0015, 0.0, -1.0],
                [1.0, 0.0, 1.0],
                [-1.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        measurement = measure_perimeter_symmetry(
            vertices,
            ((0, 1, 2, 3),),
            sample_spacing=0.05,
            rms_tolerance=0.001,
            direct_rms_tolerance=0.0005,
        )

        self.assertFalse(measurement.passed)
        self.assertTrue(measurement.tilt_search_applied)
        self.assertLessEqual(measurement.initial_rms_error, 0.001)
        self.assertGreater(measurement.rms_error, 0.0005)

    def test_failed_corrected_fit_remains_on_mirrored_carrier(self) -> None:
        topology = _open_square_topology()
        topology.vertices[1, 0] += 0.0015

        with patch(
            "mesh_segmentation_transform.beamxp_transform_sym_mesh_POC.topology_for_part",
            return_value=topology,
        ):
            result = analyse_symmetry_sweep(
                SimpleNamespace(),
                SimpleNamespace(key="test"),
                crease_max=90.0,
                crease_min=15.0,
                threshold_steps=2,
                min_region_faces=1,
                max_pseudo_aspect_ratio=10.0,
                symmetry_tolerance_metres=0.001,
                direct_symmetry_tolerance_metres=0.0005,
                sample_spacing_metres=0.05,
            )

        self.assertEqual(result.candidates, [])
        self.assertEqual(result.remaining_faces, (0, 1))

    def test_symmetry_above_outer_threshold_is_rejected_without_tilt_search(self) -> None:
        vertices = np.array(
            [
                [-1.0, 0.0, -1.0],
                [1.0025, 0.0, -1.0],
                [1.0, 0.0, 1.0],
                [-1.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        measurement = measure_perimeter_symmetry(
            vertices,
            ((0, 1, 2, 3),),
            sample_spacing=0.05,
            rms_tolerance=0.001,
            direct_rms_tolerance=0.0005,
        )

        self.assertFalse(measurement.passed)
        self.assertFalse(measurement.tilt_search_applied)
        self.assertGreater(measurement.initial_rms_error, 0.001)

    def test_main_region_with_closed_mesh_edges_is_symmetry_candidate(self) -> None:
        topology = _open_square_topology()
        edge_lookup = {
            edge: (faces, topology.edge_angles.get(edge))
            for edge, faces in topology.edge_faces.items()
        }

        region = _classify_region(
            topology,
            island_index=1,
            is_main=True,
            faces=(0, 1),
            active={0, 1},
            threshold=30.0,
            fallback_carrier_faces=set(),
            max_pseudo_aspect_ratio=10.0,
            edge_lookup=edge_lookup,
        )

        self.assertTrue(region.eligible)
        self.assertEqual(region.role, "candidate")
        self.assertEqual(region.boundary.mesh_edges, 4)
        self.assertTrue(region.boundary.closed)
        self.assertAlmostEqual(region.pseudo_aspect_ratio, 1.0)

        measurement = measure_perimeter_symmetry(
            topology.vertices,
            region.boundary.loops,
            sample_spacing=0.25,
            rms_tolerance=1e-9,
            direct_rms_tolerance=1e-9,
        )

        self.assertTrue(measurement.passed)
        self.assertAlmostEqual(measurement.rms_error, 0.0)

    def test_explicit_main_carrier_fallback_still_rejects_region(self) -> None:
        topology = _open_square_topology()
        edge_lookup = {
            edge: (faces, topology.edge_angles.get(edge))
            for edge, faces in topology.edge_faces.items()
        }

        region = _classify_region(
            topology,
            island_index=1,
            is_main=True,
            faces=(0, 1),
            active={0, 1},
            threshold=30.0,
            fallback_carrier_faces={0, 1},
            max_pseudo_aspect_ratio=10.0,
            edge_lookup=edge_lookup,
        )

        self.assertFalse(region.eligible)
        self.assertEqual(region.role, "main carrier")

    def test_curved_host_can_adopt_close_projected_child(self) -> None:
        topology = _curved_host_with_child_topology()
        islands = ((0, 1, 2, 3), (4, 5))
        island_geometry = []
        for faces in islands:
            points, samples = _surface_points_and_samples(topology, faces)
            island_geometry.append(
                _IslandGeometry(
                    area=float(topology.face_areas[list(faces)].sum()),
                    points=points,
                    samples=samples,
                )
            )

        adoptions = _find_eager_child_adoptions(
            topology,
            host_faces=islands[0],
            host_boundary_loops=_candidate_union_boundary(topology, islands[0]).loops,
            host_area=island_geometry[0].area,
            host_island_index=1,
            islands=islands,
            island_geometry=tuple(island_geometry),
            active_by_island=[set(islands[0]), set(islands[1])],
            island_candidate_counts=[0, 0],
        )

        self.assertEqual(len(adoptions), 1)
        self.assertEqual(adoptions[0].island_index, 2)
        self.assertGreaterEqual(adoptions[0].extruded_perimeter_sample_ratio, 0.05)
        self.assertLessEqual(adoptions[0].median_surface_gap, 0.015)
        self.assertLessEqual(adoptions[0].p90_surface_gap, 0.025)

    def test_proud_child_inside_extruded_parent_perimeter_is_adopted(self) -> None:
        topology = _flat_panel_with_proud_child_topology()
        islands = ((0, 1), (2, 3))
        island_geometry = []
        for faces in islands:
            points, samples = _surface_points_and_samples(topology, faces)
            island_geometry.append(
                _IslandGeometry(
                    area=float(topology.face_areas[list(faces)].sum()),
                    points=points,
                    samples=samples,
                )
            )

        adoptions = _find_eager_child_adoptions(
            topology,
            host_faces=islands[0],
            host_boundary_loops=_candidate_union_boundary(topology, islands[0]).loops,
            host_area=island_geometry[0].area,
            host_island_index=1,
            islands=islands,
            island_geometry=tuple(island_geometry),
            active_by_island=[set(islands[0]), set(islands[1])],
            island_candidate_counts=[0, 0],
        )

        self.assertEqual(len(adoptions), 1)
        self.assertEqual(adoptions[0].island_index, 2)
        self.assertGreaterEqual(adoptions[0].extruded_perimeter_sample_ratio, 0.05)
        self.assertGreater(adoptions[0].median_surface_gap, 0.015)

    def test_touching_symmetric_candidates_merge_on_union_perimeter(self) -> None:
        topology = _touching_symmetric_regions_topology()
        left = _candidate_for_faces(topology, (1, 1), (0, 1))
        right = _candidate_for_faces(topology, (1, 2), (2, 3))

        merged, absorbed = _resolve_touching_symmetric_unions(
            topology,
            [left, right],
            {(1, 1): 1, (1, 2): 2},
            sample_spacing=0.25,
            rms_tolerance=1e-9,
            direct_rms_tolerance=1e-9,
        )

        self.assertEqual(len(merged), 1)
        parent = merged[0]
        self.assertEqual(parent.host_faces, (0, 1, 2, 3))
        self.assertEqual(parent.faces, (0, 1, 2, 3))
        self.assertTrue(parent.measurement.passed)
        self.assertNotIn((1, 4), parent.boundary_edges)
        self.assertEqual(absorbed, {(1, 2): (1, 1)})
        self.assertEqual(len(parent.adoptions), 1)
        self.assertEqual(parent.adoptions[0].adoption_mode, "touching_union")

if __name__ == "__main__":
    unittest.main()
