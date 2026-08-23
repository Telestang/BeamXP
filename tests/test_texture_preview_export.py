from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image

from mesh_segmentation_transform import mirror_texture_for_rhd as rhd
from mesh_segmentation_transform import beamxp_transform_sym_mesh_POC as sym_mesh
from mesh_segmentation_transform.beamxp_transform_sym_mesh_POC import (
    ArchiveMaterialRecord,
    ArchiveMaterialSwitchState,
    ArchiveTextureBinding,
    DaePart,
    GeometryInstance,
    SourceFaceRef,
)


def part(key: str) -> DaePart:
    return DaePart(
        key=key,
        label=key,
        node_id=key,
        node_name=key,
        matrix=np.eye(4),
        instances=(GeometryInstance(f"{key}-mesh"),),
    )


class RuntimeDisplayUvFlipTests(unittest.TestCase):
    def test_selected_runtime_material_resolves_to_dae_bound_aliases(self) -> None:
        archive = SimpleNamespace(
            runtime_material_aliases=("lc500_gps", "lc500_gauges_screen"),
            material_switch_states={
                "lc500_centralscreen": (
                    ArchiveMaterialSwitchState("on_intense", "lc500_gps"),
                )
            },
        )
        with patch.object(
            rhd,
            "material_aliases_for_parts",
            return_value={"lc500_centralscreen", "lc500_gauges_screen"},
        ):
            aliases = rhd.runtime_display_dae_aliases_for_parts(
                archive, SimpleNamespace(), []
            )

        self.assertEqual(aliases, {"lc500_centralscreen", "lc500_gauges_screen"})

    def test_production_opacity_layers_use_direct_foreground_detection(self) -> None:
        colour = rhd._production_layer_detection_config(
            rhd.DEFAULT_CONFIG, True, ("baseColorMap", "emissiveMap")
        )
        opacity = rhd._production_layer_detection_config(
            rhd.DEFAULT_CONFIG, True, ("opacityMap",)
        )
        authoritative = rhd._production_layer_detection_config(
            rhd.DEFAULT_CONFIG,
            True,
            ("baseColorMap",),
            authoritative_opacity_mask=True,
        )

        self.assertEqual(colour.box_source, "contrast_gpu")
        self.assertTrue(colour.enable_feature_extension_filter)
        self.assertEqual(colour.feature_extension_context_px, 12)
        self.assertEqual(colour.feature_extension_reference_extent_px, 25)
        self.assertEqual(colour.feature_extension_grace_px, 4)
        self.assertEqual(colour.feature_extension_grace_max_fraction, 0.20)
        self.assertEqual(colour.feature_extension_soft_fringe_ratio, 0.75)
        self.assertEqual(colour.feature_extension_min_ratio, 0.03)
        self.assertEqual(opacity.box_source, "opacity_mask")
        self.assertEqual(authoritative.box_source, "opacity_mask")
        self.assertEqual(opacity.merge_distance_px, 0)
        self.assertEqual(authoritative.merge_distance_px, 0)
        self.assertGreater(colour.merge_distance_px, 0)
        self.assertFalse(opacity.enable_region_domain_filter)
        self.assertFalse(authoritative.enable_region_domain_filter)
        self.assertFalse(opacity.enable_feature_extension_filter)
        self.assertFalse(authoritative.enable_feature_extension_filter)

    def test_opacity_grouping_keeps_distant_controls_separate(self) -> None:
        image = np.zeros((120, 32, 3), dtype=np.uint8)
        # Each pair is close enough to be one authored label, while the two
        # controls merely share a UV chart and must retain separate centres.
        image[8:16, 4:8] = 255
        image[8:16, 12:16] = 255
        image[88:96, 4:8] = 255
        image[88:96, 12:16] = 255
        detector = rhd._production_layer_detection_config(
            rhd.DEFAULT_CONFIG,
            True,
            ("baseColorMap",),
            authoritative_opacity_mask=True,
        )

        detection = rhd.run_detection(
            image, np.ones(image.shape[:2], dtype=bool), detector
        )
        grouped = next(stage for stage in detection.stages if stage.key == "grouped")

        self.assertEqual(len(grouped.kept), 2)

    def test_exact_dae_switch_symbol_does_not_absorb_its_backing_material(self) -> None:
        loaded, _sweep = _screens_fixture()
        binding = SimpleNamespace(
            dae_material="ardente_gps_screen",
            material_key="ardente_gauges_screen",
        )

        self.assertEqual(
            rhd.material_symbols_for_binding(loaded, binding),
            ("ardente_gps_screen-material",),
        )

    def test_material_record_name_remains_a_symbol_fallback(self) -> None:
        loaded, _sweep = _screens_fixture()
        binding = SimpleNamespace(
            dae_material="switch_alias_not_exported",
            material_key="ardente_gauges_screen",
        )

        self.assertEqual(
            rhd.material_symbols_for_binding(loaded, binding),
            ("ardente_gauges_screen-material",),
        )

    def test_runtime_display_uv_flip_is_material_scoped(self) -> None:
        namespace = "http://www.collada.org/2005/11/COLLADASchema"
        geometry = ET.fromstring(
            f"""<geometry xmlns="{namespace}" id="screen">
              <mesh>
                <source id="screen-uv">
                  <float_array id="screen-uv-array" count="8">
                    0.1 0.0 0.4 0.0 0.7 0.0 0.9 0.0
                  </float_array>
                  <technique_common>
                    <accessor source="#screen-uv-array" count="4" stride="2">
                      <param name="S" type="float"/>
                      <param name="T" type="float"/>
                    </accessor>
                  </technique_common>
                </source>
                <triangles material="lc500_centralscreen-material" count="1">
                  <input semantic="TEXCOORD" source="#screen-uv" offset="0"/>
                  <p>0 1 0</p>
                </triangles>
                <triangles material="lc500_trim-material" count="1">
                  <input semantic="TEXCOORD" source="#screen-uv" offset="0"/>
                  <p>2 3 2</p>
                </triangles>
              </mesh>
            </geometry>"""
        )

        result = sym_mesh._flip_material_texcoord_islands(
            geometry,
            namespace,
            {"lc500_centralscreen"},
        )

        values = [
            float(value)
            for value in geometry.find(
                f"{{{namespace}}}mesh/{{{namespace}}}source/"
                f"{{{namespace}}}float_array"
            ).text.split()
        ]
        self.assertEqual(values[0::2], [0.4, 0.1, 0.7, 0.9])
        self.assertEqual(result["matched_materials"], ["lc500_centralscreen"])
        self.assertEqual(result["modified_texcoords"], 2)

    def test_rigid_symmetric_display_keeps_its_texcoords(self) -> None:
        """A screen reaching the far side rigidly was never mirrored to undo.

        The Ardente carries its gauge cluster and its satnav on one mesh.  The
        cluster is residual and rides the reflected carrier, so its UVs need
        turning back; the satnav is self-symmetric about the car's centreline
        and its rigid transform is the identity, so flipping it shipped the
        only screen in the cabin that was already the right way round
        backwards.
        """
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "screens_rhd.dae"
            loaded, sweep = _screens_fixture()
            info = sym_mesh.export_transformed_part_dae(
                loaded,
                sweep,
                output,
                runtime_display_uv_flip_materials={
                    "ardente_gauges_screen",
                    "ardente_gps_screen",
                },
            )

            uvs = _texcoords_by_material(output)

        # Subsetting reindexes each geometry's own source, so compare the U
        # values a material carries rather than the order they land in.
        # The carrier's cluster is reflected, so its U is turned back about the
        # island's own extent: 0.0/1.0 exchange and 0.25 becomes 0.75.
        self.assertEqual(
            sorted(uvs["ardente_gauges_screen-material"]), [0.0, 0.75, 1.0]
        )
        # The rigid satnav keeps the U it was authored with -- 0.25, not 0.75.
        self.assertEqual(
            sorted(uvs["ardente_gps_screen-material"]), [0.0, 0.25, 1.0]
        )

        flips = info["runtime_display_uv_flip"]
        self.assertEqual(
            [entry["role"] for entry in flips["geometries"]],
            ["mirrored_carrier"],
        )
        self.assertEqual(
            flips["geometries"][0]["matched_materials"],
            ["ardente_gauges_screen"],
        )


def _screens_fixture() -> tuple[object, object]:
    """One mesh, two display materials: one residual, one rigid candidate."""
    namespace = "http://www.collada.org/2005/11/COLLADASchema"
    root = ET.fromstring(
        f"""<COLLADA xmlns="{namespace}" version="1.4.1">
          <library_geometries>
            <geometry id="screens-mesh" name="screens">
              <mesh>
                <source id="screens-pos">
                  <float_array id="screens-pos-array" count="18">
                    -1 0 0  -1 1 0  -0.5 0 0
                     0.5 0 1   -0.5 0 1   0 1 1
                  </float_array>
                  <technique_common>
                    <accessor source="#screens-pos-array" count="6" stride="3">
                      <param name="X" type="float"/>
                      <param name="Y" type="float"/>
                      <param name="Z" type="float"/>
                    </accessor>
                  </technique_common>
                </source>
                <source id="screens-uv">
                  <float_array id="screens-uv-array" count="12">
                    0 0  1 0  0.25 1
                    0 0  1 0  0.25 1
                  </float_array>
                  <technique_common>
                    <accessor source="#screens-uv-array" count="6" stride="2">
                      <param name="S" type="float"/>
                      <param name="T" type="float"/>
                    </accessor>
                  </technique_common>
                </source>
                <vertices id="screens-verts">
                  <input semantic="POSITION" source="#screens-pos"/>
                </vertices>
                <triangles material="ardente_gauges_screen-material" count="1">
                  <input semantic="VERTEX" source="#screens-verts" offset="0"/>
                  <input semantic="TEXCOORD" source="#screens-uv" offset="1"/>
                  <p>0 0 1 1 2 2</p>
                </triangles>
                <triangles material="ardente_gps_screen-material" count="1">
                  <input semantic="VERTEX" source="#screens-verts" offset="0"/>
                  <input semantic="TEXCOORD" source="#screens-uv" offset="1"/>
                  <p>3 3 4 4 5 5</p>
                </triangles>
              </mesh>
            </geometry>
          </library_geometries>
          <library_visual_scenes>
            <visual_scene id="scene">
              <node id="ardente_screens" name="ardente_screens" type="NODE">
                <instance_geometry url="#screens-mesh"/>
              </node>
            </visual_scene>
          </library_visual_scenes>
          <scene><instance_visual_scene url="#scene"/></scene>
        </COLLADA>"""
    )
    loaded = sym_mesh.LoadedDae(
        path=Path("screens.dae"),
        tree=ET.ElementTree(root),
        namespace=namespace,
        unit_scale=1.0,
        parts=[],
        geometries={"screens-mesh": root.find(f"{{{namespace}}}library_geometries/"
                                              f"{{{namespace}}}geometry")},
    )
    selected = DaePart(
        key="ardente_screens",
        label="ardente_screens",
        node_id="ardente_screens",
        node_name="ardente_screens",
        matrix=np.eye(4),
        instances=(GeometryInstance("screens-mesh"),),
    )
    # The satnav triangle (face 1) is the accepted candidate; it straddles the
    # centreline symmetrically, so G L is the identity and nothing moves.
    measurement = SimpleNamespace(
        centroid=(0.0, 0.0, 1.0),
        plane_normal=(1.0, 0.0, 0.0),
        initial_plane_normal=(1.0, 0.0, 0.0),
        rms_error=0.0,
        initial_rms_error=0.0,
        mirror_plane_y_tilt_degrees=0.0,
        rigid_y_rotation_correction_degrees=0.0,
        tilt_search_applied=False,
    )
    sweep = SimpleNamespace(
        part=selected,
        topology=SimpleNamespace(
            triangles=np.asarray(((0, 1, 2), (3, 4, 5))),
            source_faces=(
                SourceFaceRef(0, "screens-mesh", 0, 0),
                SourceFaceRef(0, "screens-mesh", 1, 0),
            ),
        ),
        candidates=[
            SimpleNamespace(
                candidate_id=1,
                faces=(1,),
                measurement=measurement,
                accepted_angle=120.0,
            )
        ],
    )
    return loaded, sweep


def _texcoords_by_material(dae_path: Path) -> dict[str, list[float]]:
    """The U of every texcoord an exported geometry's material actually uses."""
    root = ET.parse(dae_path).getroot()
    namespace = root.tag[1:].split("}")[0]
    found: dict[str, list[float]] = {}
    for geometry in root.iter(f"{{{namespace}}}geometry"):
        mesh = geometry.find(f"{{{namespace}}}mesh")
        if mesh is None:
            continue
        sources = {
            source.get("id"): [
                float(value)
                for value in source.find(f"{{{namespace}}}float_array").text.split()
            ]
            for source in mesh.findall(f"{{{namespace}}}source")
        }
        for primitive in mesh.findall(f"{{{namespace}}}triangles"):
            inputs = primitive.findall(f"{{{namespace}}}input")
            stride = max(int(item.get("offset", "0")) for item in inputs) + 1
            texcoord = next(
                item
                for item in inputs
                if (item.get("semantic") or "").upper() == "TEXCOORD"
            )
            offset = int(texcoord.get("offset", "0"))
            values = sources[(texcoord.get("source") or "")[1:]]
            indices = [
                int(value)
                for value in primitive.find(f"{{{namespace}}}p").text.split()
            ][offset::stride]
            found[primitive.get("material")] = [values[index * 2] for index in indices]
    return found


class SurfaceFlipAxisTests(unittest.TestCase):
    def test_forced_mirror_domain_ignores_symmetric_candidate_labels(self) -> None:
        selected = part("door_source")
        uv = np.asarray(((0.1, 0.1), (0.2, 0.1), (0.1, 0.2)), dtype=float)
        topology = SimpleNamespace(
            vertices=np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
            triangles=np.asarray(((0, 1, 2),)),
            source_faces=(SourceFaceRef(0, "door-mesh", 0, 0),),
        )
        sweep = SimpleNamespace(
            topology=topology,
            candidates=(SimpleNamespace(candidate_id=7, faces=(0,)),),
        )
        with (
            patch.object(
                rhd,
                "uv_triangles_by_source",
                return_value={(0, 0, 0): uv},
            ),
            patch.object(rhd, "sweep_part", return_value=sweep),
        ):
            mirrored, rigid, _surfaces = rhd.split_mirrored_and_rigid(
                SimpleNamespace(),
                selected,
                {"door-material"},
                rhd.RhdTextureConfig(),
            )
            forced, forced_rigid, forced_surfaces = rhd.split_mirrored_and_rigid(
                SimpleNamespace(),
                selected,
                {"door-material"},
                rhd.RhdTextureConfig(),
                force_mirrored=True,
            )

        self.assertEqual(mirrored, [])
        self.assertEqual(len(rigid), 1)
        self.assertEqual(len(forced), 1)
        self.assertEqual(forced_rigid, [])
        self.assertEqual(len(forced_surfaces), 1)

    def test_side_surface_horizontal_flip_when_image_vertical_plane_matches_z(self) -> None:
        uv = np.asarray([[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]])
        xyz = np.asarray(
            [
                [
                    (0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0),
                ]
            ]
        )

        axes = rhd.surface_flip_axes(uv, xyz, width=100, height=100)

        self.assertEqual(axes.tolist(), [rhd.AXIS_HORIZONTAL])

    def test_side_surface_vertical_flip_when_image_horizontal_plane_matches_z(self) -> None:
        uv = np.asarray([[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]])
        xyz = np.asarray(
            [
                [
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0),
                    (1.0, 0.0, 0.0),
                ]
            ]
        )

        axes = rhd.surface_flip_axes(uv, xyz, width=100, height=100)

        self.assertEqual(axes.tolist(), [rhd.AXIS_VERTICAL])

    def test_horizontal_surface_horizontal_flip_when_image_vertical_plane_is_yz(self) -> None:
        uv = np.asarray([[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]])
        xyz = np.asarray(
            [
                [
                    (0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                ]
            ]
        )

        axes = rhd.surface_flip_axes(uv, xyz, width=100, height=100)

        self.assertEqual(axes.tolist(), [rhd.AXIS_HORIZONTAL])

    def test_horizontal_surface_vertical_flip_when_image_horizontal_plane_is_yz(self) -> None:
        uv = np.asarray([[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]])
        xyz = np.asarray(
            [
                [
                    (0.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (1.0, 0.0, 0.0),
                ]
            ]
        )

        axes = rhd.surface_flip_axes(uv, xyz, width=100, height=100)

        self.assertEqual(axes.tolist(), [rhd.AXIS_VERTICAL])

    def test_skew_delta_flags_texture_reflection_that_is_not_a_flat_flip(self) -> None:
        uv = (np.asarray([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]),)
        xyz = (
            np.asarray(
                [
                    (0.0, 0.0, 0.0),
                    (100.0, 0.0, 0.0),
                    (-50.0, -100.0, 0.0),
                ]
            ),
        )
        config = replace(rhd.DEFAULT_RHD_CONFIG, skewed_region_min_delta=0.01)

        delta = rhd.skew_delta_for_region(
            uv, xyz, (20, 20, 30, 30), "horizontal", 100, 100, config
        )

        self.assertIsNotNone(delta)
        self.assertGreater(delta, config.skewed_region_min_delta)

    def test_skew_delta_ignores_axis_aligned_texture_reflection(self) -> None:
        uv = (np.asarray([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]),)
        xyz = (
            np.asarray(
                [
                    (0.0, 0.0, 0.0),
                    (100.0, 0.0, 0.0),
                    (0.0, -100.0, 0.0),
                ]
            ),
        )

        delta = rhd.skew_delta_for_region(
            uv, xyz, (20, 20, 30, 30), "horizontal", 100, 100,
            rhd.DEFAULT_RHD_CONFIG,
        )

        self.assertIsNone(delta)

    def test_skew_delta_allows_the_ardente_shear_at_each_resolution(self) -> None:
        """A modest coefficient causes only bounded relative deformation.

        This is the shape of the Ardente door-card ON/OFF mapping: the exact
        vertical reflection adds a small horizontal shear, but omitting it
        moves the edge of the detected legend by about 7.5% of its short side,
        which is much safer than leaving the writing backwards.
        """
        uv = (np.asarray([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]),)
        xyz = (
            np.asarray(
                [
                    (0.0, 0.0, 0.0),
                    (100.0, 0.0, 0.0),
                    (-7.5, -100.0, 0.0),
                ]
            ),
        )
        config = replace(
            rhd.DEFAULT_RHD_CONFIG,
            skewed_region_min_delta=0.08,
            skewed_region_tolerable_error_px=3.0,
            skewed_region_tolerable_error_ratio=0.10,
        )

        for scale in (1, 2, 4):
            with self.subTest(scale=scale):
                delta = rhd.skew_delta_for_region(
                    uv,
                    xyz,
                    (20 * scale, 20 * scale, 34 * scale, 30 * scale),
                    "vertical",
                    100 * scale,
                    100 * scale,
                    config,
                )
                self.assertIsNone(delta)

    def test_skew_delta_allows_a_near_rigid_lean_across_a_long_label(self) -> None:
        """The Andronisk door card's "L MIRROR R", 73x18 on a 4096 atlas.

        The exact reflection adds a 0.052 shear: 1.9 px of lean over the
        word's length, which the short side reads as 10.5% and refuses.  Split
        into its parts the residual is 2.6% strain and a 1.5 degree turn, so
        the word comes out legible and correctly handed rather than backwards.
        """
        uv = (np.asarray([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]),)
        xyz = (
            np.asarray(
                [
                    (0.0, 0.0, 0.0),
                    (100.0, 0.0, 0.0),
                    (-2.58, -100.0, 0.0),
                ]
            ),
        )

        delta = rhd.skew_delta_for_region(
            uv, xyz, (20, 20, 18, 73), "vertical", 4096, 4096,
            rhd.DEFAULT_RHD_CONFIG,
        )

        self.assertIsNone(delta)

    def test_skew_delta_refuses_the_same_lean_across_a_seam_sliver(self) -> None:
        """Scintilla's interior seam: the residual is smaller still, but an
        8 px slice of a 300 px run is not a mark whose rotation is harmless --
        reversing the slice's lean notches it against the run it belongs to.
        The floor is read against the atlas so a half-resolution companion map
        reaches the same verdict as the colour layer it composites with."""
        uv = (np.asarray([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]),)
        xyz = (
            np.asarray(
                [
                    (0.0, 0.0, 0.0),
                    (100.0, 0.0, 0.0),
                    (-2.58, -100.0, 0.0),
                ]
            ),
        )

        for atlas, width, height in ((4096, 8, 40), (2048, 4, 20)):
            with self.subTest(atlas=atlas):
                delta = rhd.skew_delta_for_region(
                    uv, xyz, (10, 10, width, height), "vertical", atlas, atlas,
                    rhd.DEFAULT_RHD_CONFIG,
                )
                self.assertIsNotNone(delta)

    def test_skew_delta_still_refuses_a_deforming_lean_on_the_same_label(self) -> None:
        """Twice the shear is 5.9% strain: the stalk legends stay refused."""
        uv = (np.asarray([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]),)
        xyz = (
            np.asarray(
                [
                    (0.0, 0.0, 0.0),
                    (100.0, 0.0, 0.0),
                    (-5.9, -100.0, 0.0),
                ]
            ),
        )

        delta = rhd.skew_delta_for_region(
            uv, xyz, (20, 20, 18, 73), "vertical", 4096, 4096,
            rhd.DEFAULT_RHD_CONFIG,
        )

        self.assertIsNotNone(delta)

    def test_skew_delta_rejects_high_relative_error_on_a_small_region(self) -> None:
        uv = (np.asarray([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]),)
        config = replace(
            rhd.DEFAULT_RHD_CONFIG,
            skewed_region_min_delta=0.01,
            skewed_region_tolerable_error_px=3.0,
            skewed_region_tolerable_error_ratio=0.10,
        )

        for shear, expected_delta in ((0.4, 0.8), (1.0, 2.0)):
            xyz = (
                np.asarray(
                    [
                        (0.0, 0.0, 0.0),
                        (100.0, 0.0, 0.0),
                        (-100.0 * shear, -100.0, 0.0),
                    ]
                ),
            )
            for size in (4, 8):
                with self.subTest(shear=shear, size=size):
                    delta = rhd.skew_delta_for_region(
                        uv,
                        xyz,
                        (20, 20, size, size),
                        "vertical",
                        100,
                        100,
                        config,
                    )
                    self.assertIsNotNone(delta)
                    self.assertAlmostEqual(delta, expected_delta)

    def test_skew_delta_decision_is_exactly_scale_invariant_near_the_limit(self) -> None:
        uv = (np.asarray([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]),)
        xyz = (
            np.asarray(
                [
                    (0.0, 0.0, 0.0),
                    (100.0, 0.0, 0.0),
                    (-4.5, -100.0, 0.0),
                ]
            ),
        )
        config = replace(
            rhd.DEFAULT_RHD_CONFIG,
            skewed_region_min_delta=0.08,
            skewed_region_tolerable_error_px=3.0,
            skewed_region_tolerable_error_ratio=0.10,
        )

        decisions = [
            rhd.skew_delta_for_region(
                uv,
                xyz,
                (20 * scale, 20 * scale, 4 * scale, 8 * scale),
                "vertical",
                100 * scale,
                100 * scale,
                config,
            )
            for scale in (1, 2, 4)
        ]

        self.assertEqual(decisions, [None, None, None])

    def test_skew_delta_floor_does_not_hide_long_region_displacement(self) -> None:
        """A small coefficient can still move a slender label many pixels."""
        uv = (np.asarray([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]),)
        xyz = (
            np.asarray(
                [
                    (0.0, 0.0, 0.0),
                    (100.0, 0.0, 0.0),
                    (-3.0, -100.0, 0.0),
                ]
            ),
        )
        config = replace(
            rhd.DEFAULT_RHD_CONFIG,
            skewed_region_min_delta=0.08,
            skewed_region_tolerable_error_px=3.0,
            skewed_region_tolerable_error_ratio=0.10,
        )

        delta = rhd.skew_delta_for_region(
            uv,
            xyz,
            (20, 20, 4, 1000),
            "vertical",
            2000,
            2000,
            config,
        )

        self.assertIsNotNone(delta)
        self.assertAlmostEqual(delta, 0.06)


class RepeatingRegionTests(unittest.TestCase):
    """A flip that only slides a repeating moulding must be declined.

    The ETK door card is the case: its switch pads repeat every 40 px down the
    armrest and the relief detector boxed a 73 px slice of them, so the
    vertical flip carried the pads -- and the window icons printed on them --
    19 px off their buttons instead of reversing anything.
    """

    @staticmethod
    def lattice(height: int, width: int, period: int) -> np.ndarray:
        image = np.zeros((height, width, 3), dtype=np.uint8)
        for row in range(0, height, period):
            image[row : row + period // 2, :, :] = 220
        return image

    @staticmethod
    def mark(height: int, width: int) -> np.ndarray:
        """A wedge: handed on both axes and self-similar under no shift."""
        image = np.zeros((height, width, 3), dtype=np.uint8)
        for row in range(height):
            image[row, : 4 + (row * (width - 8)) // height, :] = 230
        return image

    def test_repeating_pads_are_declined_on_the_axis_that_slides_them(self) -> None:
        image = self.lattice(240, 172, 40)

        verdict = rhd.region_repeat_shift(
            image, (0, 60, 172, 73), "vertical", rhd.DEFAULT_RHD_CONFIG
        )

        self.assertIsNotNone(verdict)
        match, shift, in_place = verdict
        self.assertGreaterEqual(match, rhd.DEFAULT_RHD_CONFIG.max_region_repeat_match)
        self.assertNotEqual(shift, 0)
        self.assertGreater(match - in_place, rhd.DEFAULT_RHD_CONFIG.min_region_repeat_margin)

    def test_a_handed_mark_is_kept(self) -> None:
        image = np.zeros((240, 172, 3), dtype=np.uint8)
        image[60:133, 0:80] = self.mark(73, 80)

        self.assertIsNone(
            rhd.region_repeat_shift(
                image, (0, 60, 80, 73), "vertical", rhd.DEFAULT_RHD_CONFIG
            )
        )
        self.assertIsNone(
            rhd.region_repeat_shift(
                image, (0, 60, 80, 73), "horizontal", rhd.DEFAULT_RHD_CONFIG
            )
        )

    def test_a_mark_that_is_its_own_reflection_is_kept(self) -> None:
        image = np.zeros((240, 172, 3), dtype=np.uint8)
        image[80:120, 40:120] = 200
        image[95:105, 60:100] = 40

        self.assertIsNone(
            rhd.region_repeat_shift(
                image, (30, 70, 100, 60), "vertical", rhd.DEFAULT_RHD_CONFIG
            )
        )

    def test_a_flat_region_concludes_nothing(self) -> None:
        image = np.full((240, 172, 3), 128, dtype=np.uint8)

        self.assertIsNone(
            rhd.region_repeat_shift(
                image, (0, 60, 172, 73), "vertical", rhd.DEFAULT_RHD_CONFIG
            )
        )

    def test_the_filter_can_be_switched_off(self) -> None:
        image = self.lattice(240, 172, 40)
        config = replace(rhd.DEFAULT_RHD_CONFIG, max_region_repeat_match=1.01)

        self.assertIsNone(
            rhd.region_repeat_shift(image, (0, 60, 172, 73), "vertical", config)
        )


class CroppedDetectionTests(unittest.TestCase):
    def test_cropped_detection_reports_full_atlas_coordinates(self) -> None:
        bgr = np.zeros((100, 120, 3), dtype=np.uint8)
        mirror = np.zeros((100, 120), dtype=bool)
        mirror[40:60, 50:80] = True
        domain = mirror.copy()
        fake_stage = SimpleNamespace(
            kept=[(10, 8, 4, 5)],
            rotations=[((10.0, 8.0), (14.0, 8.0), (14.0, 13.0), (10.0, 13.0))],
        )
        fake_run = SimpleNamespace(stages=[fake_stage])
        config = rhd.RhdTextureConfig(
            enable_containment_filter=False,
            enable_blob_filter=False,
            detection_crop_padding_px=5,
        )

        with patch.object(rhd, "run_detection", return_value=fake_run) as run:
            result = rhd.detect_flip_regions(
                bgr,
                mirror,
                domain,
                config,
                replace(rhd.DEFAULT_CONFIG, enable_symmetry_rotation=True),
                log=lambda *_a: None,
            )

        cropped_image = run.call_args.args[0]
        self.assertEqual(cropped_image.shape[:2], (30, 40))
        self.assertEqual(result.regions, [(55, 43, 4, 5)])
        self.assertEqual(
            result.rotations,
            [((55.0, 43.0), (59.0, 43.0), (59.0, 48.0), (55.0, 48.0))],
        )

    def test_collaged_detection_reports_full_atlas_coordinates(self) -> None:
        bgr = np.zeros((8, 40, 3), dtype=np.uint8)
        mirror = np.zeros((8, 40), dtype=bool)
        mirror[0:4, 0:4] = True
        mirror[0:4, 20:24] = True
        domain = mirror.copy()
        fake_stage = SimpleNamespace(
            kept=[(6, 0, 4, 4)],
            rotations=[((6.0, 0.0), (10.0, 0.0), (10.0, 4.0), (6.0, 4.0))],
        )
        fake_run = SimpleNamespace(stages=[fake_stage])
        config = rhd.RhdTextureConfig(
            enable_containment_filter=False,
            enable_blob_filter=False,
            detection_crop_padding_px=0,
            detect_island_tiles_individually=False,
            detection_collage_gutter_px=2,
        )

        with patch.object(rhd, "run_detection", return_value=fake_run) as run:
            result = rhd.detect_flip_regions(
                bgr,
                mirror,
                domain,
                config,
                replace(rhd.DEFAULT_CONFIG, enable_symmetry_rotation=True),
                log=lambda *_a: None,
            )

        collage = run.call_args.args[0]
        self.assertEqual(collage.shape[:2], (4, 10))
        self.assertEqual(result.regions, [(20, 0, 4, 4)])
        self.assertEqual(
            result.rotations,
            [((20.0, 0.0), (24.0, 0.0), (24.0, 4.0), (20.0, 4.0))],
        )

    def test_individual_tile_detection_reports_full_atlas_coordinates(self) -> None:
        bgr = np.zeros((8, 40, 3), dtype=np.uint8)
        mirror = np.zeros((8, 40), dtype=bool)
        mirror[0:4, 0:4] = True
        mirror[0:4, 20:24] = True
        domain = mirror.copy()
        runs = [
            SimpleNamespace(stages=[SimpleNamespace(kept=[], rotations=[])]),
            SimpleNamespace(
                stages=[
                    SimpleNamespace(
                        kept=[(0, 0, 4, 4)],
                        rotations=[
                            (
                                (0.0, 0.0),
                                (4.0, 0.0),
                                (4.0, 4.0),
                                (0.0, 4.0),
                            )
                        ],
                    )
                ]
            ),
        ]
        config = rhd.RhdTextureConfig(
            enable_containment_filter=False,
            enable_blob_filter=False,
            detection_crop_padding_px=0,
            detection_collage_gutter_px=2,
        )

        with patch.object(rhd, "run_detection", side_effect=runs) as run:
            result = rhd.detect_flip_regions(
                bgr,
                mirror,
                domain,
                config,
                replace(rhd.DEFAULT_CONFIG, enable_symmetry_rotation=True),
                log=lambda *_a: None,
            )

        self.assertEqual(run.call_count, 2)
        self.assertEqual(result.regions, [(20, 0, 4, 4)])
        self.assertEqual(
            result.rotations,
            [((20.0, 0.0), (24.0, 0.0), (24.0, 4.0), (20.0, 4.0))],
        )

    def test_neighbouring_islands_share_one_detection_tile(self) -> None:
        bgr = np.zeros((8, 40, 3), dtype=np.uint8)
        mirror = np.zeros((8, 40), dtype=bool)
        mirror[0:4, 0:4] = True
        mirror[0:4, 6:10] = True
        mirror[0:4, 30:34] = True
        domain = mirror.copy()
        runs = [
            SimpleNamespace(
                stages=[
                    SimpleNamespace(
                        kept=[(6, 0, 4, 4)],
                        rotations=[
                            (
                                (6.0, 0.0),
                                (10.0, 0.0),
                                (10.0, 4.0),
                                (6.0, 4.0),
                            )
                        ],
                    )
                ]
            ),
            SimpleNamespace(stages=[SimpleNamespace(kept=[], rotations=[])]),
        ]
        config = rhd.RhdTextureConfig(
            enable_containment_filter=False,
            enable_blob_filter=False,
            detection_crop_padding_px=0,
            detection_tile_group_gap_px=4,
            detection_tile_group_max_area_growth=1.5,
        )

        with patch.object(rhd, "run_detection", side_effect=runs) as run:
            result = rhd.detect_flip_regions(
                bgr,
                mirror,
                domain,
                config,
                replace(rhd.DEFAULT_CONFIG, enable_symmetry_rotation=True),
                log=lambda *_a: None,
            )

        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args_list[0].args[0].shape[:2], (4, 10))
        self.assertEqual(run.call_args_list[1].args[0].shape[:2], (4, 4))
        self.assertEqual(result.regions, [(6, 0, 4, 4)])

    def test_foreground_detection_never_groups_touching_uv_islands(self) -> None:
        bgr = np.zeros((8, 12, 3), dtype=np.uint8)
        first = np.zeros((8, 12), dtype=bool)
        second = np.zeros((8, 12), dtype=bool)
        # The raster masks touch, but they are separate topological islands.
        first[0:4, 0:4] = True
        second[0:4, 4:8] = True
        runs = [
            rhd.RegionDetection(source="first", detected=1, regions=[(0, 0, 4, 4)], rotations=[None]),
            rhd.RegionDetection(source="second", detected=1, regions=[(4, 0, 4, 4)], rotations=[None]),
        ]
        config = rhd.RhdTextureConfig(
            enable_containment_filter=False,
            enable_blob_filter=False,
            detection_crop_padding_px=0,
        )

        with patch.object(rhd, "detect_flip_regions", side_effect=runs) as detect:
            result = rhd.detect_flip_regions_by_uv_island(
                bgr,
                (first, second),
                config,
                rhd.DEFAULT_CONFIG,
                log=lambda *_a: None,
            )

        self.assertEqual(detect.call_count, 2)
        self.assertTrue(np.array_equal(detect.call_args_list[0].args[1], first))
        self.assertTrue(np.array_equal(detect.call_args_list[1].args[1], second))
        self.assertEqual(result.regions, [(0, 0, 4, 4), (4, 0, 4, 4)])

    def test_near_duplicate_uv_consumers_are_coalesced_before_detection(self) -> None:
        first = np.zeros((24, 32), dtype=bool)
        second = np.zeros((24, 32), dtype=bool)
        first[2:22, 4:24] = True
        second[2:22, 5:25] = True

        merged = rhd._coalesce_overlapping_uv_consumers((first, second))

        self.assertEqual(len(merged), 1)
        consumer, island_bits = merged[0]
        self.assertIsInstance(consumer, rhd.UvIslandCrop)
        assert isinstance(consumer, rhd.UvIslandCrop)
        self.assertEqual(consumer.bounds, (4, 2, 25, 22))
        self.assertTrue(consumer.mask.all())
        # Which chart owns each pixel survives the union, so grouping can still
        # refuse to reach from one into the other.
        self.assertIsNotNone(island_bits)
        assert island_bits is not None
        self.assertEqual(island_bits.shape, consumer.mask.shape)
        self.assertEqual(int(island_bits[0, 0]), 0b01)
        self.assertEqual(int(island_bits[0, -1]), 0b10)
        self.assertEqual(int(island_bits[0, 10]), 0b11)

    def test_a_lone_uv_consumer_records_no_chart_membership(self) -> None:
        only = np.zeros((16, 16), dtype=bool)
        only[2:10, 2:10] = True

        merged = rhd._coalesce_overlapping_uv_consumers((only,))

        self.assertEqual(len(merged), 1)
        consumer, island_bits = merged[0]
        self.assertIs(consumer, only)
        # Nothing was coalesced, so no join could cross anything.
        self.assertIsNone(island_bits)

    def test_topological_islands_do_not_use_raster_connectivity(self) -> None:
        # The almost-coincident boundary rasterises as a shared edge, but the
        # UV vertices are not shared, so these remain independent islands.
        triangles = (
            np.asarray([(0.0, 0.0), (0.4999999, 0.0), (0.0, 1.0)]),
            np.asarray([(0.5000001, 0.0), (1.0, 0.0), (1.0, 1.0)]),
        )
        mirror = rhd.rasterise_uv_triangles(list(triangles), 8, 8)
        islands = rhd.mirrored_uv_island_masks(triangles, mirror)

        self.assertEqual(len(islands), 2)

    def test_a_shared_uv_corner_needs_the_meshes_to_meet_there_too(self) -> None:
        # Two charts packed so they share their boundary UV vertices exactly.
        # Nothing in the atlas separates them; only the surface does.
        triangles = (
            np.asarray([(0.1, 0.1), (0.5, 0.1), (0.1, 0.5)]),
            np.asarray([(0.5, 0.1), (0.9, 0.5), (0.1, 0.5)]),
        )
        mirror = rhd.rasterise_uv_triangles(list(triangles), 32, 32)

        joined = (
            np.asarray([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]),
            np.asarray([(1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]),
        )
        apart = (
            joined[0],
            joined[1] + np.asarray([0.0, 0.0, 0.25]),
        )

        self.assertEqual(len(rhd.mirrored_uv_island_masks(triangles, mirror)), 1)
        self.assertEqual(
            len(rhd.mirrored_uv_island_masks(triangles, mirror, joined)), 1
        )
        self.assertEqual(
            len(rhd.mirrored_uv_island_masks(triangles, mirror, apart)), 2
        )

    def test_surface_corners_only_split_where_the_mesh_is_really_apart(self) -> None:
        # A 10 um offset is transform noise on a welded vertex, not a boundary.
        triangles = (
            np.asarray([(0.1, 0.1), (0.5, 0.1), (0.1, 0.5)]),
            np.asarray([(0.5, 0.1), (0.9, 0.5), (0.1, 0.5)]),
        )
        mirror = rhd.rasterise_uv_triangles(list(triangles), 32, 32)
        surfaces = (
            np.asarray([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]),
            np.asarray([(1.0, 0.0, 1e-6), (1.0, 1.0, 0.0), (0.0, 1.0, 1e-6)]),
        )

        self.assertEqual(
            len(rhd.mirrored_uv_island_masks(triangles, mirror, surfaces)), 1
        )

    def test_mismatched_surface_corners_fall_back_to_the_uv_grouping(self) -> None:
        triangles = (
            np.asarray([(0.1, 0.1), (0.5, 0.1), (0.1, 0.5)]),
            np.asarray([(0.5, 0.1), (0.9, 0.5), (0.1, 0.5)]),
        )
        mirror = rhd.rasterise_uv_triangles(list(triangles), 32, 32)

        self.assertEqual(
            len(
                rhd.mirrored_uv_island_masks(
                    triangles,
                    mirror,
                    (np.asarray([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]),),
                )
            ),
            1,
        )

    def test_compact_uv_island_crops_reconstruct_full_masks_exactly(self) -> None:
        triangles = (
            np.asarray([(0.1, 0.1), (0.4, 0.1), (0.1, 0.4)]),
            np.asarray([(0.6, 0.6), (0.9, 0.6), (0.9, 0.9)]),
        )
        mirror = rhd.rasterise_uv_triangles(list(triangles), 32, 24)
        full = rhd.mirrored_uv_island_masks(triangles, mirror)
        crops = rhd.mirrored_uv_island_crops(triangles, mirror, padding_px=3)

        rebuilt = []
        for crop in crops:
            mask = np.zeros_like(mirror)
            x0, y0, x1, y1 = crop.bounds
            mask[y0:y1, x0:x1] = crop.mask
            rebuilt.append(mask)

        self.assertEqual(len(rebuilt), len(full))
        for actual, expected in zip(rebuilt, full):
            self.assertTrue(np.array_equal(actual, expected))

    def test_cropped_uv_raster_matches_full_atlas_for_wrapped_triangles(self) -> None:
        triangles = [
            np.asarray([(0.10, 0.15), (0.55, 0.20), (0.20, 0.70)]),
            np.asarray([(0.85, 0.25), (1.15, 0.30), (1.05, 0.80)]),
            np.asarray([(-0.10, 0.65), (0.15, 0.55), (0.05, 1.10)]),
        ]
        mirror = np.ones((47, 61), dtype=bool)
        mirror[8:14, 20:34] = False
        expected = rhd.rasterise_uv_triangles(triangles, 61, 47) & mirror

        compact = rhd.rasterise_uv_triangles_crop(triangles, mirror, padding_px=4)

        self.assertIsNotNone(compact)
        assert compact is not None
        actual = np.zeros_like(mirror)
        x0, y0, x1, y1 = compact.bounds
        actual[y0:y1, x0:x1] = compact.mask
        self.assertTrue(np.array_equal(actual, expected))

    def test_gpu_contrast_islands_are_dispatched_as_one_batch(self) -> None:
        bgr = np.zeros((12, 32, 3), dtype=np.uint8)
        first = rhd.UvIslandCrop((2, 2, 8, 8), np.ones((6, 6), dtype=bool))
        second = rhd.UvIslandCrop((20, 2, 26, 8), np.ones((6, 6), dtype=bool))
        detections = [
            rhd.LocalContrastDetection(
                np.empty((0, 4), dtype=np.int32),
                np.zeros((6, 6), dtype=np.float32),
                4.0,
            ),
            rhd.LocalContrastDetection(
                np.asarray([(1, 1, 2, 2)], dtype=np.int32),
                np.zeros((6, 6), dtype=np.float32),
                4.0,
            ),
        ]
        config = rhd.RhdTextureConfig(
            enable_containment_filter=False,
            enable_blob_filter=False,
            detection_crop_padding_px=0,
        )
        detector = replace(rhd.DEFAULT_CONFIG, box_source="contrast_gpu")

        def fake_run(_image, _mask, _config, **kwargs):
            kept = [tuple(int(value) for value in box) for box in kwargs["initial_boxes"]]
            return SimpleNamespace(
                stages=[SimpleNamespace(kept=kept, rotations=[None] * len(kept))]
            )

        with (
            patch.object(
                rhd, "detect_local_contrast_gpu_batch", return_value=detections
            ) as batch,
            patch.object(rhd, "run_detection", side_effect=fake_run) as run,
        ):
            result = rhd.detect_flip_regions_by_uv_island(
                bgr,
                (first, second),
                config,
                detector,
                log=lambda *_a: None,
            )

        self.assertEqual(batch.call_count, 1)
        self.assertEqual(len(batch.call_args.args[0]), 2)
        self.assertEqual(run.call_count, 2)
        responses = {
            id(call.kwargs["initial_contrast_response"])
            for call in run.call_args_list
        }
        self.assertEqual(responses, {id(item.response) for item in detections})
        self.assertEqual(result.regions, [(21, 3, 2, 2)])


class MultiPartPreviewTests(unittest.TestCase):
    def test_sjson_material_keeps_normal_and_scalar_companions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_zip = root / "vehicle.zip"
            materials_member = "vehicles/car/screens.materials.json"
            base_member = "vehicles/car/screen_b.color.dds"
            normal_member = "vehicles/car/screen_n.normal.dds"
            roughness_member = "vehicles/car/screen_r.data.dds"
            with zipfile.ZipFile(source_zip, "w") as archive:
                archive.writestr("vehicles/car/car.dae", "<COLLADA/>")
                archive.writestr(
                    materials_member,
                    """{
                      // BeamNG accepts comments and commas as whitespace.
                      "car_screen": {
                        "name": "car_screen",
                        "mapTo": "car_screen",
                        "Stages": [{
                          "baseColorMap":
                            "/vehicles/car/screen_b.color.png",
                          /* These maps share the same authored UV layout. */
                          "normalMap":
                            "/vehicles/car/screen_n.normal.png",
                          "roughnessMap":
                            "/vehicles/car/screen_r.data.png",
                        },],
                      },
                    }""",
                )
                archive.writestr(base_member, b"base")
                archive.writestr(normal_member, b"normal")
                archive.writestr(roughness_member, b"roughness")

            vehicle = rhd.scan_vehicle_archive(source_zip, root / "workspace")
            binding = ArchiveTextureBinding(
                dae_material="car_screen",
                material_key="car_screen",
                materials_member=materials_member,
                texture_reference="/vehicles/car/screen_b.color.png",
                texture_member=base_member,
            )

            companions = rhd.companion_maps_for_binding(vehicle, binding)

        self.assertEqual(
            companions,
            (
                rhd.CompanionMap(normal_member, "normalMap", "normal"),
                rhd.CompanionMap(roughness_member, "roughnessMap", "scalar"),
            ),
        )

    def test_manifest_reassembles_independently_corrected_material_layers(self) -> None:
        source = {
            "key": "lc500_screen_on",
            "aliases": ["lc500_screen_on"],
            "materialsMember": "vehicles/lc500/main.materials.json",
            "material": {"Stages": [{}]},
        }

        def result(member: str, stage_key: str, kind: str) -> rhd.RhdTextureResult:
            stem = Path(member).stem
            png = Path(f"{stem}_rhd.png")
            return rhd.RhdTextureResult(
                texture_member=member,
                size=(4, 4),
                parts_analysed=1,
                mirrored_triangles=1,
                rigid_triangles=0,
                mirror_coverage=1.0,
                rigid_coverage=0.0,
                conflict_coverage=0.0,
                glyph_regions=1,
                mirrored_glyph_regions=1,
                material_aliases=("lc500_screen", "lc500_screen_on"),
                png_path=png,
                dds_path=Path(f"{stem}_rhd.dds"),
                report={
                    "layer_bindings": [
                        {
                            "dae_material": "lc500_screen",
                            "material_key": "lc500_screen_on",
                            "materials_member": source["materialsMember"],
                            "stage_key": stage_key,
                            "kind": kind,
                        }
                    ],
                    "source_materials": [source],
                    "switch_base_aliases": ["lc500_screen"],
                    "outputs": {"preview": None},
                },
            )

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            rhd.write_blender_preview(
                output,
                [
                    result("vehicles/lc500/screen.png", "baseColorMap", "colour"),
                    result("vehicles/lc500/screen_glow.png", "emissiveMap", "scalar"),
                ],
                log=lambda *_a: None,
            )
            manifest = json.loads((output / "rhd_materials.json").read_text())

        self.assertEqual(len(manifest["materials"]), 1)
        entry = manifest["materials"][0]
        self.assertEqual(set(entry["maps"]), {"baseColorMap", "emissiveMap"})
        self.assertEqual(len(entry["outputMaps"]), 2)

    def test_same_texture_material_states_are_kept_as_distinct_bindings(self) -> None:
        dash = part("dash")
        off = ArchiveTextureBinding(
            dae_material="ardente_interior",
            material_key="ardente_interior",
            materials_member="vehicles/vivace/ardente/main.materials.json",
            texture_reference="/vehicles/vivace/ardente/ardente_interior_b.color.png",
            texture_member="vehicles/vivace/ardente/ardente_interior_b.color.png",
        )
        on = ArchiveTextureBinding(
            dae_material="ardente_interior",
            material_key="ardente_interior_on",
            materials_member="vehicles/vivace/ardente/main.materials.json",
            texture_reference="/vehicles/vivace/ardente/ardente_interior_b.color.png",
            texture_member="vehicles/vivace/ardente/ardente_interior_b.color.png",
        )

        with patch.object(
            rhd,
            "archive_texture_choices_for_part",
            return_value=(off, on),
        ):
            grouped = rhd.texture_bindings_for_parts(
                SimpleNamespace(),
                SimpleNamespace(),
                [dash],
            )

        bindings = grouped["vehicles/vivace/ardente/ardente_interior_b.color.png"]
        self.assertEqual([binding.material_key for _part, binding in bindings], [
            "ardente_interior",
            "ardente_interior_on",
        ])

    def test_same_texture_target_keeps_distinct_dae_source_materials(self) -> None:
        dash = part("dash")
        member = "vehicles/lc500/textures/dashlights.png"
        left = ArchiveTextureBinding(
            dae_material="lc500_intsignal_L",
            material_key="lc500_dashlights",
            materials_member="vehicles/lc500/main.materials.json",
            texture_reference=f"/{member}",
            texture_member=member,
        )
        right = ArchiveTextureBinding(
            dae_material="lc500_intsignal_R",
            material_key="lc500_dashlights",
            materials_member="vehicles/lc500/main.materials.json",
            texture_reference=f"/{member}",
            texture_member=member,
        )

        with patch.object(
            rhd,
            "archive_texture_choices_for_part",
            return_value=(left, right),
        ):
            grouped = rhd.texture_bindings_for_parts(
                SimpleNamespace(),
                SimpleNamespace(),
                [dash],
            )

        bindings = grouped[member]
        self.assertEqual(
            [binding.dae_material for _part, binding in bindings],
            ["lc500_intsignal_L", "lc500_intsignal_R"],
        )

    def test_every_archive_backed_material_layer_becomes_its_own_binding(self) -> None:
        dash = part("dash")
        materials_member = "vehicles/lc500/main.materials.json"
        base_member = "vehicles/lc500/textures/screen.png"
        glow_member = "vehicles/lc500/textures/screen_glow.png"
        normal_member = "vehicles/lc500/textures/screen_n.png"
        record = ArchiveMaterialRecord(
            key="lc500_screen_on",
            name="lc500_screen_on",
            map_to="",
            materials_member=materials_member,
            base_colour_reference=f"/{base_member}",
            source_material={
                "Stages": [
                    {
                        "baseColorMap": f"/{base_member}",
                        "emissiveMap": f"/{glow_member}",
                        "normalMap": f"/{normal_member}",
                    },
                    {
                        # A duplicate reference is one physical job but retains
                        # both material slots for final wiring.
                        "baseColorMap": f"/{glow_member}",
                    },
                ]
            },
        )
        binding = ArchiveTextureBinding(
            dae_material="lc500_screen",
            material_key=record.key,
            materials_member=materials_member,
            texture_reference=f"/{base_member}",
            texture_member=base_member,
        )
        members = (base_member, glow_member, normal_member)
        archive = SimpleNamespace(
            materials=(record,),
            members=members,
            member_by_lower={member.lower(): member for member in members},
            member_archive_indices={},
        )

        with patch.object(
            rhd,
            "archive_texture_choices_for_part",
            return_value=(binding,),
        ):
            grouped = rhd.texture_bindings_for_parts(archive, SimpleNamespace(), [dash])

        self.assertEqual(set(grouped), set(members))
        self.assertEqual(
            [layer.stage_key for _part, layer in grouped[glow_member]],
            ["emissiveMap", "baseColorMap"],
        )
        self.assertEqual(grouped[normal_member][0][1].kind, "normal")

    def test_layer_resolves_the_normal_from_its_own_material_stage(self) -> None:
        dash = part("dash")
        materials_member = "vehicles/lc500/main.materials.json"
        base_member = "vehicles/lc500/textures/screen.png"
        normal_member = "vehicles/lc500/textures/screen_n.png"
        other_normal_member = "vehicles/lc500/textures/trim_n.png"
        record = ArchiveMaterialRecord(
            key="lc500_screen",
            name="lc500_screen",
            map_to="",
            materials_member=materials_member,
            base_colour_reference=f"/{base_member}",
            source_material={
                "Stages": [
                    {
                        "baseColorMap": f"/{base_member}",
                        "normalMap": f"/{normal_member}",
                    },
                    {
                        "baseColorMap": "/vehicles/lc500/textures/trim.png",
                        "normalMap": f"/{other_normal_member}",
                    },
                ]
            },
        )
        binding = rhd.MaterialTextureLayerBinding(
            dae_material="lc500_screen",
            material_key=record.key,
            materials_member=materials_member,
            texture_reference=f"/{base_member}",
            texture_member=base_member,
            stage_key="baseColorMap",
            kind="colour",
        )
        members = (
            base_member,
            normal_member,
            "vehicles/lc500/textures/trim.png",
            other_normal_member,
        )
        archive = SimpleNamespace(
            materials=(record,),
            members=members,
            member_by_lower={member.lower(): member for member in members},
            member_archive_indices={},
        )

        normals = rhd.normal_maps_for_layer_bindings(
            archive, [(dash, binding)], base_member
        )

        self.assertEqual(
            normals,
            (rhd.CompanionMap(normal_member, "normalMap", "normal"),),
        )

    def test_opacity_map_is_authoritative_only_when_every_binding_is_masked(self) -> None:
        dash = part("dash")
        materials_member = "vehicles/lc500/main.materials.json"
        base_member = "vehicles/lc500/textures/intemis.dds"
        mask_member = "vehicles/lc500/textures/intemis_opmap.png"
        masked = ArchiveMaterialRecord(
            key="lc500_intemis_on",
            name="lc500_intemis_on",
            map_to="",
            materials_member=materials_member,
            base_colour_reference=f"/{base_member}",
            source_material={
                "Stages": [{
                    "baseColorMap": f"/{base_member}",
                    "opacityMap": f"/{mask_member}",
                }]
            },
        )
        unmasked = ArchiveMaterialRecord(
            key="lc500_trim",
            name="lc500_trim",
            map_to="",
            materials_member=materials_member,
            base_colour_reference=f"/{base_member}",
            source_material={"Stages": [{"baseColorMap": f"/{base_member}"}]},
        )
        members = (base_member, mask_member)
        archive = SimpleNamespace(
            materials=(masked,),
            members=members,
            member_by_lower={member.lower(): member for member in members},
            member_archive_indices={},
        )

        def binding(material_key: str) -> rhd.MaterialTextureLayerBinding:
            return rhd.MaterialTextureLayerBinding(
                dae_material=material_key,
                material_key=material_key,
                materials_member=materials_member,
                texture_reference=f"/{base_member}",
                texture_member=base_member,
                stage_key="baseColorMap",
                kind="colour",
            )

        self.assertEqual(
            rhd.authoritative_visibility_masks_for_layer_bindings(
                archive, [(dash, binding(masked.key))], base_member
            ),
            (rhd.CompanionMap(mask_member, "opacityMap", "scalar"),),
        )

        archive.materials = (masked, unmasked)
        self.assertEqual(
            rhd.authoritative_visibility_masks_for_layer_bindings(
                archive,
                [(dash, binding(masked.key)), (dash, binding(unmasked.key))],
                base_member,
            ),
            (),
        )

    def test_masked_layer_uses_only_opacity_detection_regions(self) -> None:
        dash = part("dash")
        materials_member = "vehicles/lc500/main.materials.json"
        base_member = "vehicles/lc500/textures/intemis.dds"
        mask_member = "vehicles/lc500/textures/intemis_opmap.png"
        record = ArchiveMaterialRecord(
            key="lc500_intemis_on",
            name="lc500_intemis_on",
            map_to="",
            materials_member=materials_member,
            base_colour_reference=f"/{base_member}",
            source_material={
                "Stages": [{
                    "baseColorMap": f"/{base_member}",
                    "opacityMap": f"/{mask_member}",
                }]
            },
        )
        binding = rhd.MaterialTextureLayerBinding(
            dae_material="lc500_intemissive",
            material_key=record.key,
            materials_member=materials_member,
            texture_reference=f"/{base_member}",
            texture_member=base_member,
            stage_key="baseColorMap",
            kind="colour",
        )
        archive = SimpleNamespace(
            materials=(record,),
            members=(base_member, mask_member),
            member_by_lower={
                base_member.lower(): base_member,
                mask_member.lower(): mask_member,
            },
            member_archive_indices={},
            material_switch_targets={},
        )

        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            base_path = workspace / "intemis.dds"
            mask_path = workspace / "intemis_opmap.png"
            Image.new("RGBA", (8, 8), (255, 0, 255, 255)).save(
                base_path, format="PNG"
            )
            Image.new("RGBA", (8, 8), (0, 0, 0, 255)).save(mask_path)
            mirror = np.zeros((8, 8), dtype=bool)
            mirror[:, :4] = True
            masks = rhd.DomainMasks(
                mirror=mirror,
                rigid=~mirror,
                conflict_coverage=0.0,
                mirrored_triangles=1,
                rigid_triangles=1,
                parts_analysed=1,
            )
            planned_regions: list[tuple[int, int, int, int]] = []

            def extracted(_archive, member):
                return mask_path if member == mask_member else base_path

            def detect(
                _image,
                _islands,
                _config,
                detector_config,
                source,
                _log,
                relief_bridge_response=None,
            ):
                self.assertTrue(str(source).startswith("opacityMap:"))
                self.assertEqual(detector_config.box_source, "opacity_mask")
                self.assertIsNone(relief_bridge_response)
                return rhd.RegionDetection(
                    source=str(source),
                    detected=1,
                    regions=[(0, 1, 4, 3)],
                    rotations=[None],
                )

            def plan(_mirror, regions, _config, _axis_map, _components=None):
                planned_regions.extend(regions)
                return [], np.zeros((8, 8), dtype=bool)

            with (
                patch.object(
                    rhd,
                    "scoped_parts_using_material",
                    return_value=[(dash, binding)],
                ),
                patch.object(
                    rhd,
                    "material_symbols_for_binding",
                    return_value=("lc500_intemissive-material",),
                ),
                patch.object(rhd, "extract_archive_member", side_effect=extracted),
                patch.object(
                    rhd,
                    "detect_flip_regions_by_uv_island",
                    side_effect=detect,
                ),
                patch.object(rhd, "plan_island_flips", side_effect=plan),
                patch.object(
                    rhd,
                    "skew_delta_for_region",
                    side_effect=AssertionError(
                        "authoritative opacity regions must bypass inferred UV skew"
                    ),
                ),
                patch.object(
                    rhd,
                    "normal_maps_for_layer_bindings",
                    side_effect=AssertionError("masked layers must not inspect normals"),
                ),
            ):
                result = rhd.build_rhd_texture(
                    archive,
                    SimpleNamespace(),
                    base_member,
                    workspace,
                    config=rhd.RhdTextureConfig(
                        detect_on_normal_map=True,
                        write_debug_overlays=False,
                    ),
                    part_scope=[dash],
                    masks=masks,
                    log=lambda *_a: None,
                )

        self.assertEqual(planned_regions, [(0, 1, 4, 3)])
        self.assertEqual(result.report["detection_authority"], "opacity_mask")
        self.assertFalse(result.report["skewed_region_filter_applied"])
        self.assertEqual(
            [(step["x"], step["y"], step["w"], step["h"])
             for step in result.report["in_place_flips"]],
            [(0, 1, 4, 3)],
        )
        self.assertEqual(
            result.report["in_place_flips"][0]["stencil"],
            rhd.STENCIL_REGION,
        )
        self.assertEqual(result.report["authoritative_mask_sources"], [mask_member])
        self.assertEqual(
            result.report["detection"][0]["pipeline"],
            "authoritative_opacity_mask",
        )

    def _capture_production_detector(
        self, *, with_normal: bool
    ) -> tuple[dict[str, object], rhd.RhdTextureResult]:
        dash = part("dash")
        base_member = "vehicles/lc500/textures/screen.png"
        normal_member = "vehicles/lc500/textures/screen_n.png"
        binding = rhd.MaterialTextureLayerBinding(
            dae_material="lc500_screen",
            material_key="lc500_screen",
            materials_member="vehicles/lc500/main.materials.json",
            texture_reference=f"/{base_member}",
            texture_member=base_member,
            stage_key="baseColorMap",
            kind="colour",
        )
        captured: dict[str, object] = {}
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            base = workspace / "screen.png"
            normal = workspace / "screen_n.png"
            Image.new("RGBA", (4, 4), (0, 0, 0, 255)).save(base)
            Image.new("RGB", (4, 4), (128, 128, 255)).save(normal)
            mirror = np.zeros((4, 4), dtype=bool)
            mirror[:, :2] = True
            rigid = ~mirror
            masks = rhd.DomainMasks(
                mirror=mirror,
                rigid=rigid,
                conflict_coverage=0.0,
                mirrored_triangles=1,
                rigid_triangles=1,
                parts_analysed=1,
            )
            response = np.full((4, 4), 23.0, dtype=np.float32)
            edge_data = SimpleNamespace(
                edge_response=response,
                render_seconds=0.1,
                edge_seconds=0.2,
            )
            session = rhd.ProductionDetectionSession()

            def detect(
                _image,
                _islands,
                _config,
                detector,
                source,
                _log,
                relief_bridge_response=None,
            ):
                captured["box_source"] = detector.box_source
                captured["source"] = source
                captured["relief_bridge_response"] = relief_bridge_response
                return rhd.RegionDetection(source=str(source), detected=0)

            def extracted(_archive, member):
                return normal if member == normal_member else base

            normal_maps = (
                (rhd.CompanionMap(normal_member, "normalMap", "normal"),)
                if with_normal
                else ()
            )
            with (
                patch.object(rhd, "scoped_parts_using_material", return_value=[(dash, binding)]),
                patch.object(rhd, "material_symbols_for_binding", return_value=("lc500_screen-material",)),
                patch.object(rhd, "extract_archive_member", side_effect=extracted),
                patch.object(rhd, "build_domain_masks", return_value=masks),
                patch.object(rhd, "normal_maps_for_layer_bindings", return_value=normal_maps),
                patch.object(session, "normal_edge_data", return_value=(edge_data, False)),
                patch.object(rhd, "detect_flip_regions_by_uv_island", side_effect=detect),
                patch.object(rhd, "plan_island_flips", return_value=([], np.zeros((4, 4), dtype=bool))),
            ):
                result = rhd.build_rhd_texture(
                    SimpleNamespace(materials=[]),
                    SimpleNamespace(),
                    base_member,
                    workspace,
                    config=rhd.RhdTextureConfig(
                        detect_on_normal_map=True,
                        write_debug_overlays=False,
                    ),
                    part_scope=[dash],
                    masks=masks,
                    detection_session=session,
                    log=lambda *_a: None,
                )
        return captured, result

    def test_production_uses_colour_glyphs_with_cached_relief_edges(self) -> None:
        captured, result = self._capture_production_detector(with_normal=True)

        self.assertEqual(captured["box_source"], "contrast_gpu")
        self.assertIsNotNone(captured["relief_bridge_response"])
        self.assertEqual(
            result.report["detection"][0]["pipeline"],
            "colour_glyphs_relief_edge_grouping",
        )

    def test_production_falls_back_to_gpu_colour_without_a_normal(self) -> None:
        captured, result = self._capture_production_detector(with_normal=False)

        self.assertEqual(captured["box_source"], "contrast_gpu")
        self.assertIsNone(captured["relief_bridge_response"])
        self.assertEqual(
            result.report["detection"][0]["pipeline"],
            "colour_local_contrast_gpu",
        )

    def test_normal_layer_uses_island_scoped_gpu_relief_detection(self) -> None:
        dash = part("dash")
        normal_member = "vehicles/vivace/ardente/ardente_interior_nm.normal.png"
        binding = rhd.MaterialTextureLayerBinding(
            dae_material="ardente_interior",
            material_key="ardente_interior",
            materials_member="vehicles/vivace/ardente/main.materials.json",
            texture_reference=f"/{normal_member}",
            texture_member=normal_member,
            stage_key="normalMap",
            kind="normal",
        )
        captured: dict[str, object] = {}
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            normal = workspace / "ardente_interior_nm.normal.png"
            Image.new("RGB", (4, 4), (128, 128, 255)).save(normal)
            mirror = np.zeros((4, 4), dtype=bool)
            mirror[:, :2] = True
            masks = rhd.DomainMasks(
                mirror=mirror,
                rigid=~mirror,
                conflict_coverage=0.0,
                mirrored_triangles=1,
                rigid_triangles=1,
                parts_analysed=1,
            )
            relief = np.full((4, 4, 3), 71, dtype=np.uint8)
            response = np.full((4, 4), 29.0, dtype=np.float32)
            edge_data = SimpleNamespace(
                relief_bgr=relief,
                edge_response=response,
                render_seconds=0.1,
                edge_seconds=0.2,
            )
            session = rhd.ProductionDetectionSession()

            def detect(
                image,
                islands,
                _config,
                detector,
                source,
                _log,
                relief_bridge_response=None,
            ):
                captured["image"] = image
                captured["box_source"] = detector.box_source
                captured["detector"] = detector
                captured["source"] = source
                captured["relief_bridge_response"] = relief_bridge_response
                captured["islands"] = islands
                return rhd.RegionDetection(source=str(source), detected=0)

            with (
                patch.object(
                    rhd,
                    "scoped_parts_using_material",
                    return_value=[(dash, binding)],
                ),
                patch.object(
                    rhd,
                    "material_symbols_for_binding",
                    return_value=("ardente_interior-material",),
                ),
                patch.object(rhd, "extract_archive_member", return_value=normal),
                patch.object(
                    rhd,
                    "normal_maps_for_layer_bindings",
                    side_effect=AssertionError(
                        "a physical normal layer must use its own pixels"
                    ),
                ),
                patch.object(session, "normal_edge_data", return_value=(edge_data, False)),
                patch.object(
                    rhd,
                    "detect_flip_regions_by_uv_island",
                    side_effect=detect,
                ),
                patch.object(
                    rhd,
                    "detect_flip_regions",
                    side_effect=AssertionError(
                        "normal relief must match the tuning harness's UV scope"
                    ),
                ),
                patch.object(
                    rhd,
                    "plan_island_flips",
                    return_value=([], np.zeros((4, 4), dtype=bool)),
                ),
            ):
                result = rhd.build_rhd_texture(
                    SimpleNamespace(materials=[]),
                    SimpleNamespace(),
                    normal_member,
                    workspace,
                    config=rhd.RhdTextureConfig(write_debug_overlays=False),
                    part_scope=[dash],
                    masks=masks,
                    detection_session=session,
                    log=lambda *_a: None,
                )

        self.assertIs(captured["image"], relief)
        self.assertEqual(captured["box_source"], "edge_gpu")
        self.assertEqual(captured["source"], "relief")
        self.assertIs(captured["relief_bridge_response"], response)
        self.assertGreater(len(captured["islands"]), 0)
        self.assertEqual(
            captured["detector"],
            replace(rhd.DEFAULT_RELIEF_DETECTION_CONFIG, box_source="edge_gpu"),
        )
        self.assertEqual(result.report["detection"][0]["source"], "relief")
        self.assertEqual(
            result.report["detection"][0]["pipeline"],
            "relief_edge_gpu",
        )
        self.assertFalse(result.report["normal_region_detail_gate"])

    def test_companion_maps_do_not_replay_the_colour_plan(self) -> None:
        dash = part("dash")
        binding = ArchiveTextureBinding(
            dae_material="ardente_interior",
            material_key="ardente_interior_on",
            materials_member="vehicles/vivace/ardente/main.materials.json",
            texture_reference="/vehicles/vivace/ardente/ardente_interior_b.color.png",
            texture_member="vehicles/vivace/ardente/ardente_interior_b.color.png",
        )

        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            base = workspace / "ardente_interior_b.color.png"
            Image.new("RGBA", (4, 4), (0, 0, 0, 255)).save(base)
            mask = np.ones((4, 4), dtype=bool)
            masks = rhd.DomainMasks(
                mirror=mask,
                rigid=np.zeros((4, 4), dtype=bool),
                conflict_coverage=0.0,
                mirrored_triangles=1,
                rigid_triangles=0,
                parts_analysed=1,
            )
            companion_result = rhd.CompanionResult(
                member="vehicles/vivace/ardente/ardente_interior_g.color.png",
                stage_key="emissiveMap",
                kind="scalar",
                codec="rgba",
                texels_moved=16,
                png_path=workspace / "ardente_interior_g.color_rhd.png",
                dds_path=workspace / "ardente_interior_g.color_rhd.dds",
            )

            with (
                patch.object(rhd, "scoped_parts_using_material", return_value=[(dash, binding)]),
                patch.object(rhd, "material_symbols_for_binding", return_value=("ardente_interior-material",)),
                patch.object(rhd, "extract_archive_member", return_value=base),
                patch.object(rhd, "build_domain_masks", return_value=masks),
                patch.object(
                    rhd,
                    "companion_maps_for_binding",
                    return_value=(
                        rhd.CompanionMap(
                            member="vehicles/vivace/ardente/ardente_interior_g.color.png",
                            stage_key="emissiveMap",
                            kind="scalar",
                        ),
                    ),
                ),
                patch.object(
                    rhd,
                    "detect_flip_regions",
                    return_value=rhd.RegionDetection(
                        source="colour",
                        detected=1,
                        regions=[(0, 0, 4, 4)],
                        rotations=[None],
                    ),
                ),
                patch.object(
                    rhd,
                    "plan_island_flips",
                    return_value=(
                        [
                            rhd.IslandFlip(
                                label=1,
                                bounds=(0, 0, 4, 4),
                                area_px=16,
                                axis="horizontal",
                                horizontal_similarity=1.0,
                                vertical_similarity=1.0,
                                glyph_count=1,
                            )
                        ],
                        mask,
                    ),
                ),
                patch.object(rhd, "rebuild_companion_map", return_value=companion_result) as rebuild,
            ):
                result = rhd.build_rhd_texture(
                    SimpleNamespace(materials=[]),
                    SimpleNamespace(),
                    binding.texture_member,
                    workspace,
                    config=rhd.RhdTextureConfig(
                        rebuild_companion_maps=True,
                        detect_on_normal_map=False,
                    ),
                    part_scope=[dash],
                    log=lambda *_a: None,
                )

        rebuild.assert_not_called()
        self.assertEqual(result.companions, [])

    def test_emissive_companion_regions_do_not_join_base_flip_plan(self) -> None:
        dash = part("dash")
        binding = ArchiveTextureBinding(
            dae_material="lc500_door_buttons",
            material_key="lc500_door_buttons_on",
            materials_member="vehicles/lc500/main.materials.json",
            texture_reference="/vehicles/lc500/textures/buttons.dds",
            texture_member="vehicles/lc500/textures/buttons.dds",
        )

        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            base = workspace / "buttons.dds"
            glow = workspace / "buttons_glow.png"
            Image.new("RGBA", (4, 4), (0, 0, 0, 255)).save(base, format="PNG")
            Image.new("RGBA", (4, 4), (255, 0, 0, 255)).save(glow)
            mask = np.ones((4, 4), dtype=bool)
            masks = rhd.DomainMasks(
                mirror=mask,
                rigid=np.zeros((4, 4), dtype=bool),
                conflict_coverage=0.0,
                mirrored_triangles=1,
                rigid_triangles=0,
                parts_analysed=1,
            )
            planned_regions: list[tuple[int, int, int, int]] = []

            def detection(
                _image,
                _mirror,
                _domain,
                _config,
                _mser_config,
                source,
                _log,
            ) -> rhd.RegionDetection:
                if str(source).startswith("emissiveMap:"):
                    return rhd.RegionDetection(
                        source=str(source),
                        detected=1,
                        regions=[(0, 0, 4, 4)],
                        rotations=[None],
                    )
                return rhd.RegionDetection(source=str(source), detected=0)

            def plan(
                _mirror,
                regions,
                _config,
                _axis_map,
                _components=None,
            ):
                planned_regions.extend(regions)
                return (
                    [
                        rhd.IslandFlip(
                            label=1,
                            bounds=(0, 0, 4, 4),
                            area_px=16,
                            axis="horizontal",
                            horizontal_similarity=1.0,
                            vertical_similarity=1.0,
                            glyph_count=1,
                        )
                    ],
                    mask,
                )

            with (
                patch.object(rhd, "scoped_parts_using_material", return_value=[(dash, binding)]),
                patch.object(rhd, "material_symbols_for_binding", return_value=("lc500_door_buttons-material",)),
                patch.object(rhd, "extract_archive_member", side_effect=[base, glow]),
                patch.object(rhd, "build_domain_masks", return_value=masks),
                patch.object(
                    rhd,
                    "companion_maps_for_binding",
                    return_value=(
                        rhd.CompanionMap(
                            member="vehicles/lc500/textures/buttons_glow.png",
                            stage_key="emissiveMap",
                            kind="scalar",
                        ),
                    ),
                ),
                patch.object(rhd, "detect_flip_regions", side_effect=detection),
                patch.object(rhd, "plan_island_flips", side_effect=plan),
                patch.object(rhd, "rebuild_companion_map", return_value=None),
            ):
                result = rhd.build_rhd_texture(
                    SimpleNamespace(materials=[]),
                    SimpleNamespace(),
                    binding.texture_member,
                    workspace,
                    config=rhd.RhdTextureConfig(
                        rebuild_companion_maps=True,
                        detect_on_normal_map=False,
                    ),
                    part_scope=[dash],
                    log=lambda *_a: None,
                )

        self.assertEqual(planned_regions, [(0, 0, 4, 4)])
        self.assertEqual(result.report["companion_regions_added"], 0)

    def test_switch_group_state_regions_do_not_join_base_flip_plan(self) -> None:
        dash = part("dash")
        binding = ArchiveTextureBinding(
            dae_material="lc500_screen_off",
            material_key="lc500_screen_off_off",
            materials_member="vehicles/lc500/main.materials.json",
            texture_reference="/vehicles/lc500/LEX_LC5.dds",
            texture_member="vehicles/lc500/LEX_LC5.dds",
        )
        on_record = ArchiveMaterialRecord(
            key="lc500_screen_off_on",
            name="lc500_screen_off_on",
            map_to="",
            materials_member="vehicles/lc500/main.materials.json",
            base_colour_reference="/vehicles/lc500/textures/screen_on.png",
            source_material={
                "Stages": [
                    {
                        "baseColorMap": "/vehicles/lc500/textures/screen_on.png",
                        "emissiveMap": "/vehicles/lc500/textures/screen_on.png",
                    }
                ]
            },
        )

        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            base = workspace / "LEX_LC5.dds"
            on = workspace / "screen_on.png"
            Image.new("RGBA", (8, 8), (0, 0, 0, 255)).save(base, format="PNG")
            Image.new("RGBA", (4, 4), (255, 0, 0, 255)).save(on)
            mask = np.ones((8, 8), dtype=bool)
            masks = rhd.DomainMasks(
                mirror=mask,
                rigid=np.zeros((8, 8), dtype=bool),
                conflict_coverage=0.0,
                mirrored_triangles=1,
                rigid_triangles=0,
                parts_analysed=1,
            )
            archive = SimpleNamespace(
                materials=(on_record,),
                material_switch_targets={
                    "lc500_screen_off": ("lc500_screen_off_off", "lc500_screen_off_on")
                },
                material_switch_states={
                    "lc500_screen_off": (
                        ArchiveMaterialSwitchState("off", "lc500_screen_off_off"),
                        ArchiveMaterialSwitchState("on", "lc500_screen_off_on"),
                    )
                },
                member_by_lower={
                    "vehicles/lc500/lex_lc5.dds": "vehicles/lc500/LEX_LC5.dds",
                    "vehicles/lc500/textures/screen_on.png": (
                        "vehicles/lc500/textures/screen_on.png"
                    ),
                },
                members=(
                    "vehicles/lc500/LEX_LC5.dds",
                    "vehicles/lc500/textures/screen_on.png",
                ),
                member_archive_indices={},
            )
            planned_regions: list[tuple[int, int, int, int]] = []

            def detection(
                _image,
                _mirror,
                _domain,
                _config,
                _mser_config,
                source,
                _log,
            ) -> rhd.RegionDetection:
                if str(source).startswith("state:"):
                    return rhd.RegionDetection(
                        source=str(source),
                        detected=1,
                        regions=[(1, 1, 2, 2)],
                        rotations=[None],
                    )
                return rhd.RegionDetection(source=str(source), detected=0)

            def plan(
                _mirror,
                regions,
                _config,
                _axis_map,
                _components=None,
            ):
                planned_regions.extend(regions)
                return (
                    [
                        rhd.IslandFlip(
                            label=1,
                            bounds=(2, 2, 4, 4),
                            area_px=16,
                            axis="horizontal",
                            horizontal_similarity=1.0,
                            vertical_similarity=1.0,
                            glyph_count=1,
                        )
                    ],
                    mask,
                )

            def member_path(_archive, member):
                return {
                    "vehicles/lc500/LEX_LC5.dds": base,
                    "vehicles/lc500/textures/screen_on.png": on,
                }[member]

            with (
                patch.object(rhd, "scoped_parts_using_material", return_value=[(dash, binding)]),
                patch.object(rhd, "material_symbols_for_binding", return_value=("lc500_screen_off-material",)),
                patch.object(rhd, "extract_archive_member", side_effect=member_path),
                patch.object(rhd, "build_domain_masks", return_value=masks),
                patch.object(rhd, "companion_maps_for_binding", return_value=()),
                patch.object(rhd, "detect_flip_regions", side_effect=detection),
                patch.object(rhd, "plan_island_flips", side_effect=plan),
            ):
                result = rhd.build_rhd_texture(
                    archive,
                    SimpleNamespace(),
                    binding.texture_member,
                    workspace,
                    config=rhd.RhdTextureConfig(
                        rebuild_companion_maps=True,
                        detect_on_normal_map=False,
                    ),
                    part_scope=[dash],
                    log=lambda *_a: None,
                )

        self.assertEqual(planned_regions, [(0, 0, 8, 8)])
        self.assertEqual(result.report["state_group_regions_added"], 0)
        self.assertFalse(
            any(
                str(entry["source"]).startswith("state:")
                for entry in result.report["detection"]
            )
        )

    def test_switch_group_detection_does_not_merge_shared_off_states(self) -> None:
        dash = part("dash")
        binding = ArchiveTextureBinding(
            dae_material="lc500_centralscreen",
            material_key="lc500_screens_off",
            materials_member="vehicles/lc500/main.materials.json",
            texture_reference="/vehicles/lc500/textures/lc500_bootscreen.png",
            texture_member="vehicles/lc500/textures/lc500_bootscreen.png",
        )
        own_on = ArchiveMaterialRecord(
            key="lc500_centralscreen_on",
            name="lc500_centralscreen_on",
            map_to="",
            materials_member="vehicles/lc500/main.materials.json",
            base_colour_reference="/vehicles/lc500/textures/lc500_bootscreen_on.png",
            source_material={
                "Stages": [
                    {"baseColorMap": "/vehicles/lc500/textures/lc500_bootscreen_on.png"}
                ]
            },
        )
        other_on = ArchiveMaterialRecord(
            key="lc500_screens_on",
            name="lc500_screens_on",
            map_to="",
            materials_member="vehicles/lc500/main.materials.json",
            base_colour_reference="/vehicles/lc500/textures/screen.dds",
            source_material={
                "Stages": [
                    {"baseColorMap": "/vehicles/lc500/textures/screen.dds"}
                ]
            },
        )
        archive = SimpleNamespace(
            materials=(own_on, other_on),
            material_switch_targets={
                "lc500_centralscreen": (
                    "lc500_screens_off",
                    "lc500_centralscreen_on",
                    "lc500_gps",
                ),
                "lc500_screens": ("lc500_screens_off", "lc500_screens_on"),
            },
            material_switch_states={
                "lc500_centralscreen": (
                    ArchiveMaterialSwitchState("off", "lc500_screens_off"),
                    ArchiveMaterialSwitchState("on", "lc500_centralscreen_on"),
                    ArchiveMaterialSwitchState("on_intense", "lc500_gps"),
                ),
                "lc500_screens": (
                    ArchiveMaterialSwitchState("off", "lc500_screens_off"),
                    ArchiveMaterialSwitchState("on", "lc500_screens_on"),
                ),
            },
            material_switch_triggers={
                "lc500_centralscreen": ("ignitionLevel",),
                "lc500_screens": ("running",),
            },
            member_by_lower={
                "vehicles/lc500/textures/lc500_bootscreen.png": (
                    "vehicles/lc500/textures/lc500_bootscreen.png"
                ),
                "vehicles/lc500/textures/lc500_bootscreen_on.png": (
                    "vehicles/lc500/textures/lc500_bootscreen_on.png"
                ),
                "vehicles/lc500/textures/screen.dds": (
                    "vehicles/lc500/textures/screen.dds"
                ),
            },
            members=(
                "vehicles/lc500/textures/lc500_bootscreen.png",
                "vehicles/lc500/textures/lc500_bootscreen_on.png",
                "vehicles/lc500/textures/screen.dds",
            ),
            member_archive_indices={},
        )

        maps = rhd.switch_group_detection_maps_for_candidates(
            archive,
            [(dash, binding)],
            binding.texture_member,
        )

        self.assertEqual(
            [(item.material_key, item.switch_state, item.member) for item in maps],
            [
                (
                    "lc500_centralscreen_on",
                    "on",
                    "vehicles/lc500/textures/lc500_bootscreen_on.png",
                )
            ],
        )

    def test_switch_group_detection_borrows_ignition_on_sibling_states(self) -> None:
        dash = part("dash")
        binding = ArchiveTextureBinding(
            dae_material="lc500_centralscreen",
            material_key="lc500_screens_off",
            materials_member="vehicles/lc500/main.materials.json",
            texture_reference="/vehicles/lc500/textures/lc500_bootscreen.png",
            texture_member="vehicles/lc500/textures/lc500_bootscreen.png",
        )
        own_on = ArchiveMaterialRecord(
            key="lc500_centralscreen_on",
            name="lc500_centralscreen_on",
            map_to="",
            materials_member="vehicles/lc500/main.materials.json",
            base_colour_reference="/vehicles/lc500/textures/lc500_bootscreen_on.png",
            source_material={
                "Stages": [
                    {"baseColorMap": "/vehicles/lc500/textures/lc500_bootscreen_on.png"}
                ]
            },
        )
        sibling_on = ArchiveMaterialRecord(
            key="lc500_screens_on",
            name="lc500_screens_on",
            map_to="",
            materials_member="vehicles/lc500/main.materials.json",
            base_colour_reference="/vehicles/lc500/textures/screen.dds",
            source_material={
                "Stages": [
                    {"baseColorMap": "/vehicles/lc500/textures/screen.dds"}
                ]
            },
        )
        running_on = ArchiveMaterialRecord(
            key="lc500_gauges_needle_on",
            name="lc500_gauges_needle_on",
            map_to="",
            materials_member="vehicles/lc500/main.materials.json",
            base_colour_reference="/vehicles/lc500/textures/needle.png",
            source_material={
                "Stages": [
                    {"emissiveMap": "/vehicles/lc500/textures/needle.png"}
                ]
            },
        )
        archive = SimpleNamespace(
            materials=(own_on, sibling_on, running_on),
            material_switch_targets={
                "lc500_centralscreen": (
                    "lc500_screens_off",
                    "lc500_centralscreen_on",
                    "lc500_gps",
                ),
                "lc500_screens": ("lc500_screens_off", "lc500_screens_on"),
                "lc500_gauges_needle": ("invis", "lc500_gauges_needle_on"),
            },
            material_switch_states={
                "lc500_centralscreen": (
                    ArchiveMaterialSwitchState("off", "lc500_screens_off"),
                    ArchiveMaterialSwitchState("on", "lc500_centralscreen_on"),
                    ArchiveMaterialSwitchState("on_intense", "lc500_gps"),
                ),
                "lc500_screens": (
                    ArchiveMaterialSwitchState("off", "lc500_screens_off"),
                    ArchiveMaterialSwitchState("on", "lc500_screens_on"),
                    ArchiveMaterialSwitchState("on_intense", "lc500_screens_on"),
                ),
                "lc500_gauges_needle": (
                    ArchiveMaterialSwitchState("off", "invis"),
                    ArchiveMaterialSwitchState("on", "lc500_gauges_needle_on"),
                ),
            },
            material_switch_triggers={
                "lc500_centralscreen": ("ignitionLevel",),
                "lc500_screens": ("ignitionLevel",),
                "lc500_gauges_needle": ("running",),
            },
            member_by_lower={
                "vehicles/lc500/textures/lc500_bootscreen.png": (
                    "vehicles/lc500/textures/lc500_bootscreen.png"
                ),
                "vehicles/lc500/textures/lc500_bootscreen_on.png": (
                    "vehicles/lc500/textures/lc500_bootscreen_on.png"
                ),
                "vehicles/lc500/textures/screen.dds": (
                    "vehicles/lc500/textures/screen.dds"
                ),
                "vehicles/lc500/textures/needle.png": (
                    "vehicles/lc500/textures/needle.png"
                ),
            },
            members=(
                "vehicles/lc500/textures/lc500_bootscreen.png",
                "vehicles/lc500/textures/lc500_bootscreen_on.png",
                "vehicles/lc500/textures/screen.dds",
                "vehicles/lc500/textures/needle.png",
            ),
            member_archive_indices={},
        )

        maps = rhd.switch_group_detection_maps_for_candidates(
            archive,
            [(dash, binding)],
            binding.texture_member,
        )

        self.assertEqual(
            [(item.material_key, item.switch_state, item.member) for item in maps],
            [
                (
                    "lc500_screens_on",
                    "on_intense",
                    "vehicles/lc500/textures/screen.dds",
                ),
                (
                    "lc500_centralscreen_on",
                    "on",
                    "vehicles/lc500/textures/lc500_bootscreen_on.png",
                ),
            ],
        )

    def test_shared_atlas_is_rebuilt_once_for_selected_parts(self) -> None:
        dash = part("dash")
        console = part("console")
        binding = ArchiveTextureBinding(
            dae_material="interior",
            material_key="interior",
            materials_member="vehicles/car/main.materials.json",
            texture_reference="/vehicles/car/interior.dds",
            texture_member="vehicles/car/interior.dds",
        )

        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            source = workspace / "interior.dds"
            Image.new("RGBA", (4, 4), (0, 0, 0, 255)).save(source, format="PNG")
            mask = np.ones((4, 4), dtype=bool)
            masks = rhd.DomainMasks(
                mirror=mask,
                rigid=np.zeros((4, 4), dtype=bool),
                conflict_coverage=0.0,
                mirrored_triangles=2,
                rigid_triangles=0,
                parts_analysed=2,
            )
            texture_result = rhd.RhdTextureResult(
                texture_member=binding.texture_member,
                size=(4, 4),
                parts_analysed=2,
                mirrored_triangles=2,
                rigid_triangles=0,
                mirror_coverage=1.0,
                rigid_coverage=0.0,
                conflict_coverage=0.0,
                glyph_regions=0,
                mirrored_glyph_regions=0,
                material_aliases=("interior",),
                png_path=workspace / "interior_rhd.png",
                report={
                    "texture": binding.texture_member,
                    "selected_parts": [{"key": "dash"}, {"key": "console"}],
                },
            )
            export_info = {
                "carrier": {
                    "node_id": "generated_carrier",
                    "geometry_ids": ["generated_carrier_mesh"],
                    "triangle_count": 2,
                },
                "rigid_symmetric_nodes": [],
            }
            events: list[dict[str, object]] = []

            with (
                patch.object(
                    rhd,
                    "texture_bindings_for_parts",
                    return_value={binding.texture_member: [(dash, binding), (console, binding)]},
                ),
                patch.object(rhd, "material_symbols_for_binding", return_value=("interior-material",)),
                patch.object(rhd, "extract_archive_member", return_value=source),
                patch.object(
                    rhd, "build_domain_masks", return_value=masks,
                ) as build_masks,
                patch.object(rhd, "build_rhd_texture", return_value=texture_result) as build_texture,
                patch.object(rhd, "write_blender_preview", return_value=None),
                patch.object(rhd, "sweep_part", return_value=object()),
                patch.object(
                    rhd, "export_transformed_part_dae", return_value=export_info
                ) as export_dae,
            ):
                preview = rhd.export_parts_preview(
                    SimpleNamespace(),
                    SimpleNamespace(),
                    [dash, console],
                    workspace,
                    bake=False,
                    log=lambda *_a: None,
                    progress=events.append,
                )
            self.assertIsNotNone(preview.report_path)
            report = json.loads(preview.report_path.read_text(encoding="utf-8"))

        # Each mesh is measured for its own domain, but the two come out equal,
        # so one correction serves both rather than competing per-part copies.
        self.assertEqual(build_masks.call_count, 2)
        build_texture.assert_called_once()
        self.assertEqual(build_texture.call_args.kwargs["part_scope"], [dash, console])
        self.assertEqual(build_texture.call_args.kwargs["part_group_index"], 0)
        self.assertIs(build_texture.call_args.kwargs["masks"], masks)
        self.assertEqual(export_dae.call_count, 2)
        self.assertEqual(len(preview.dae_paths), 2)
        self.assertEqual(len(report["selected_parts"]), 2)
        self.assertEqual(len(report["texture_jobs"]), 1)
        self.assertEqual(len(report["dae_exports"]), 2)
        self.assertEqual(
            report["texture_detection_inventory"]["candidate_files"],
            [binding.texture_member],
        )
        self.assertEqual(
            report["texture_detection_inventory"]["detection_attempted_files"],
            [binding.texture_member],
        )
        self.assertEqual(
            report["dae_exports"][0]["generated_flexbody_rows"][0]["node_id"],
            "generated_carrier",
        )
        phases = [event["phase"] for event in events]
        self.assertIn("preview_export", phases)
        self.assertIn("resolve_materials", phases)
        self.assertIn("build_domain_masks", phases)
        self.assertIn("export_part_dae", phases)
        self.assertIn("write_preview_report", phases)
        reported_phases = [entry["phase"] for entry in report["phase_timings"]]
        self.assertIn("resolve_materials", reported_phases)
        self.assertIn("build_domain_masks", reported_phases)
        self.assertIn("export_part_dae", reported_phases)
        self.assertIn("write_preview_report", reported_phases)

    def _preview_with_export_results(self, results: list[object]):
        """Run a two-part export whose per-part DAE writes give `results`."""
        dash = part("dash")
        console = part("console")
        binding = ArchiveTextureBinding(
            dae_material="interior",
            material_key="interior",
            materials_member="vehicles/car/main.materials.json",
            texture_reference="/vehicles/car/interior.dds",
            texture_member="vehicles/car/interior.dds",
        )
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            source = workspace / "interior.dds"
            Image.new("RGBA", (4, 4), (0, 0, 0, 255)).save(source, format="PNG")
            mask = np.ones((4, 4), dtype=bool)
            masks = rhd.DomainMasks(
                mirror=mask,
                rigid=np.zeros((4, 4), dtype=bool),
                conflict_coverage=0.0,
                mirrored_triangles=2,
                rigid_triangles=0,
                parts_analysed=2,
            )
            texture_result = rhd.RhdTextureResult(
                texture_member=binding.texture_member,
                size=(4, 4),
                parts_analysed=2,
                mirrored_triangles=2,
                rigid_triangles=0,
                mirror_coverage=1.0,
                rigid_coverage=0.0,
                conflict_coverage=0.0,
                glyph_regions=0,
                mirrored_glyph_regions=0,
                material_aliases=("interior",),
                png_path=workspace / "interior_rhd.png",
                report={"texture": binding.texture_member, "selected_parts": []},
            )
            with (
                patch.object(
                    rhd,
                    "texture_bindings_for_parts",
                    return_value={binding.texture_member: [(dash, binding), (console, binding)]},
                ),
                patch.object(rhd, "material_symbols_for_binding", return_value=("interior-material",)),
                patch.object(rhd, "extract_archive_member", return_value=source),
                patch.object(rhd, "build_domain_masks", return_value=masks),
                patch.object(rhd, "build_rhd_texture", return_value=texture_result),
                patch.object(rhd, "write_blender_preview", return_value=None),
                patch.object(rhd, "sweep_part", return_value=object()),
                patch.object(rhd, "export_transformed_part_dae", side_effect=results),
            ):
                return rhd.export_parts_preview(
                    SimpleNamespace(),
                    SimpleNamespace(),
                    [dash, console],
                    workspace,
                    bake=False,
                    log=lambda *_a: None,
                )

    def test_a_part_with_no_bindable_atlas_does_not_sink_the_others(self) -> None:
        # Scintilla's race console carries only scintilla_main_carbon, a detail
        # material with no base-colour atlas, so it cannot be wired. It used to
        # abort the whole per-DAE export, costing the nine other marked meshes
        # in scintilla.dae their correction with no error raised anywhere.
        export_info = {
            "carrier": {
                "node_id": "generated_carrier",
                "geometry_ids": ["generated_carrier_mesh"],
                "triangle_count": 2,
            },
            "rigid_symmetric_nodes": [],
        }
        preview = self._preview_with_export_results(
            [ValueError("archive texture aliases did not match carbon"), export_info]
        )

        self.assertEqual(len(preview.dae_paths), 1)
        self.assertEqual(len(preview.failed_parts), 1)
        self.assertEqual(preview.failed_parts[0]["source_part"]["key"], "dash")
        self.assertIn("carbon", preview.failed_parts[0]["error"])
        self.assertEqual(
            [entry["source_part"]["key"] for entry in preview.report["dae_exports"]],
            ["console"],
        )

    def test_an_export_where_every_part_fails_is_still_an_error(self) -> None:
        with self.assertRaises(ValueError) as caught:
            self._preview_with_export_results(
                [ValueError("no atlas"), ValueError("no atlas")]
            )
        self.assertIn("No part could be exported", str(caught.exception))


class UnchangedStageIsNotEncodedTests(unittest.TestCase):
    """A stage whose plan is empty must not be re-encoded to reproduce itself."""

    def _build(self, detected_regions):
        dash = part("dash")
        member = "vehicles/lc500/textures/screen.png"
        binding = rhd.MaterialTextureLayerBinding(
            dae_material="lc500_screen",
            material_key="lc500_screen",
            materials_member="vehicles/lc500/main.materials.json",
            texture_reference=f"/{member}",
            texture_member=member,
            stage_key="baseColorMap",
            kind="colour",
        )
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            source = workspace / "screen.png"
            Image.new("RGBA", (4, 4), (0, 0, 0, 255)).save(source)
            mirror = np.zeros((4, 4), dtype=bool)
            mirror[:, :2] = True
            masks = rhd.DomainMasks(
                mirror=mirror,
                rigid=~mirror,
                conflict_coverage=0.0,
                mirrored_triangles=2,
                rigid_triangles=2,
                parts_analysed=1,
            )

            def detect(*_args, **_kwargs):
                return rhd.RegionDetection(
                    source="colour",
                    detected=len(detected_regions),
                    regions=list(detected_regions),
                    rotations=[None] * len(detected_regions),
                )

            def plan(_mirror, regions, _config, _axis_map, _components=None):
                # Whatever was detected, nothing survives into a flip step.
                return [], np.zeros((4, 4), dtype=bool)

            with (
                patch.object(
                    rhd, "scoped_parts_using_material", return_value=[(dash, binding)]
                ),
                patch.object(
                    rhd, "material_symbols_for_binding",
                    return_value=("lc500_screen-material",),
                ),
                patch.object(rhd, "extract_archive_member", return_value=source),
                patch.object(
                    rhd, "detect_flip_regions_by_uv_island", side_effect=detect
                ),
                patch.object(rhd, "plan_island_flips", side_effect=plan),
                patch.object(
                    rhd, "normal_maps_for_layer_bindings", return_value=(),
                ),
            ):
                result = rhd.build_rhd_texture(
                    SimpleNamespace(materials=[]),
                    SimpleNamespace(),
                    member,
                    workspace,
                    config=rhd.RhdTextureConfig(write_debug_overlays=False),
                    part_scope=[dash],
                    masks=masks,
                    log=lambda *_a: None,
                )
            written = sorted(
                p.name for p in workspace.iterdir() if p.name.startswith("screen_rhd")
            )
        return result, written

    def test_an_empty_plan_writes_no_image(self) -> None:
        result, written = self._build([])

        self.assertEqual(result.island_flips, [])
        self.assertEqual(result.in_place_flips, [])
        self.assertIsNone(result.png_path)
        self.assertIsNone(result.dds_path)
        # The plan is still recorded -- it is the evidence nothing was needed.
        self.assertEqual(written, ["screen_rhd.plan.json"])
        self.assertIsNone(result.report["outputs"]["png"])
        self.assertIsNone(result.report["outputs"]["dds"])

    def test_a_skipped_stage_reaches_no_manifest_entry(self) -> None:
        result, _written = self._build([])

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            rhd.write_blender_preview(output, [result], log=lambda *_a: None)
            # No corrected file means no corrected material: the mesh keeps the
            # texture it shipped with.
            self.assertFalse((output / "rhd_materials.json").exists())

    def test_a_planned_flip_still_encodes(self) -> None:
        dash = part("dash")
        member = "vehicles/lc500/textures/screen.png"
        binding = rhd.MaterialTextureLayerBinding(
            dae_material="lc500_screen",
            material_key="lc500_screen",
            materials_member="vehicles/lc500/main.materials.json",
            texture_reference=f"/{member}",
            texture_member=member,
            stage_key="baseColorMap",
            kind="colour",
        )
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            source = workspace / "screen.png"
            Image.new("RGBA", (8, 8), (0, 0, 0, 255)).save(source)
            mirror = np.ones((8, 8), dtype=bool)
            masks = rhd.DomainMasks(
                mirror=mirror,
                rigid=np.zeros((8, 8), dtype=bool),
                conflict_coverage=0.0,
                mirrored_triangles=2,
                rigid_triangles=0,
                parts_analysed=1,
            )

            def detect(*_args, **_kwargs):
                return rhd.RegionDetection(
                    source="colour", detected=1,
                    regions=[(0, 0, 4, 4)], rotations=[None],
                )

            with (
                patch.object(
                    rhd, "scoped_parts_using_material", return_value=[(dash, binding)]
                ),
                patch.object(
                    rhd, "material_symbols_for_binding",
                    return_value=("lc500_screen-material",),
                ),
                patch.object(rhd, "extract_archive_member", return_value=source),
                patch.object(
                    rhd, "detect_flip_regions_by_uv_island", side_effect=detect
                ),
                patch.object(rhd, "normal_maps_for_layer_bindings", return_value=()),
            ):
                result = rhd.build_rhd_texture(
                    SimpleNamespace(materials=[]),
                    SimpleNamespace(),
                    member,
                    workspace,
                    config=rhd.RhdTextureConfig(write_debug_overlays=False),
                    part_scope=[dash],
                    masks=masks,
                    log=lambda *_a: None,
                )
            names = {p.name for p in workspace.iterdir()}

        self.assertTrue(result.island_flips or result.in_place_flips)
        self.assertIsNotNone(result.png_path)
        self.assertIn("screen_rhd.png", names)


class PreviewPngTests(unittest.TestCase):
    """The inspection PNG is scratch; the non-power-of-two PNG is the asset."""

    def _build(self, size, *, write_preview_png):
        dash = part("dash")
        member = "vehicles/lc500/textures/screen.png"
        binding = rhd.MaterialTextureLayerBinding(
            dae_material="lc500_screen",
            material_key="lc500_screen",
            materials_member="vehicles/lc500/main.materials.json",
            texture_reference=f"/{member}",
            texture_member=member,
            stage_key="baseColorMap",
            kind="colour",
        )
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            source = workspace / "screen.png"
            Image.new("RGBA", size, (0, 0, 0, 255)).save(source)
            mirror = np.ones((size[1], size[0]), dtype=bool)
            masks = rhd.DomainMasks(
                mirror=mirror,
                rigid=np.zeros((size[1], size[0]), dtype=bool),
                conflict_coverage=0.0,
                mirrored_triangles=2,
                rigid_triangles=0,
                parts_analysed=1,
            )

            def detect(*_args, **_kwargs):
                return rhd.RegionDetection(
                    source="colour", detected=1,
                    regions=[(0, 0, 4, 4)], rotations=[None],
                )

            with (
                patch.object(
                    rhd, "scoped_parts_using_material", return_value=[(dash, binding)]
                ),
                patch.object(
                    rhd, "material_symbols_for_binding",
                    return_value=("lc500_screen-material",),
                ),
                patch.object(rhd, "extract_archive_member", return_value=source),
                patch.object(
                    rhd, "detect_flip_regions_by_uv_island", side_effect=detect
                ),
                patch.object(rhd, "normal_maps_for_layer_bindings", return_value=()),
            ):
                result = rhd.build_rhd_texture(
                    SimpleNamespace(materials=[]),
                    SimpleNamespace(),
                    member,
                    workspace,
                    config=rhd.RhdTextureConfig(
                        write_debug_overlays=False,
                        write_preview_png=write_preview_png,
                    ),
                    part_scope=[dash],
                    masks=masks,
                    log=lambda *_a: None,
                )
            names = {p.name for p in workspace.iterdir()}
            sizes = {
                p.name: p.stat().st_size
                for p in workspace.iterdir()
                if p.suffix == ".png"
            }
        return result, names, sizes

    def test_a_shipped_dds_needs_no_inspection_png(self) -> None:
        result, names, _sizes = self._build((64, 64), write_preview_png=False)

        self.assertIsNotNone(result.dds_path)
        self.assertIsNone(result.png_path)
        self.assertIn("screen_rhd.dds", names)
        self.assertNotIn("screen_rhd.png", names)

    def test_the_inspection_png_is_kept_when_asked_for(self) -> None:
        result, names, _sizes = self._build((64, 64), write_preview_png=True)

        self.assertIsNotNone(result.dds_path)
        self.assertIsNotNone(result.png_path)
        self.assertIn("screen_rhd.png", names)

    def test_a_non_power_of_two_ships_its_png_even_unasked(self) -> None:
        # Block compression has no such size, so here the PNG is the asset.
        result, names, _sizes = self._build((48, 64), write_preview_png=False)

        self.assertIsNone(result.dds_path)
        self.assertIsNotNone(result.png_path)
        self.assertIn("screen_rhd.png", names)
        self.assertNotIn("screen_rhd.dds", names)

    def test_a_shipped_png_is_deflated_and_a_scratch_one_is_not(self) -> None:
        # Same pixels either way, so any size difference is the compression.
        _shipped, _n1, shipped_sizes = self._build((48, 64), write_preview_png=False)
        self.assertEqual(rhd.SHIPPED_PNG_COMPRESS_LEVEL, 3)
        with tempfile.TemporaryDirectory() as temporary:
            reference = Path(temporary) / "raw.png"
            Image.new("RGBA", (48, 64), (0, 0, 0, 255)).save(
                reference, compress_level=0
            )
            raw_bytes = reference.stat().st_size
        self.assertLess(shipped_sizes["screen_rhd.png"], raw_bytes)


class PerMeshCorrectionJobTests(unittest.TestCase):
    """A correction is scoped to one mesh's use of one material."""

    def _part(self, key: str):
        return SimpleNamespace(key=key, node_id=key, node_name=key, label=key)

    def _binding(self, material: str, stage_key: str = "baseColorMap"):
        return SimpleNamespace(
            dae_material=material,
            material_key=f"{material}_on",
            materials_member="vehicles/lc500/main.materials.json",
            stage_key=stage_key,
            kind="colour",
        )

    def _jobs(self, entries):
        with patch.object(
            rhd,
            "material_symbols_for_binding",
            side_effect=lambda _l, b: (f"{b.dae_material}-material",),
        ):
            return rhd.correction_jobs_for_texture(SimpleNamespace(), entries)

    def test_one_mesh_on_one_material_is_one_job(self) -> None:
        dash = self._part("dash")
        jobs = self._jobs([(dash, self._binding("interior"))])

        self.assertEqual(len(jobs), 1)
        self.assertIs(jobs[0].part, dash)
        self.assertEqual(jobs[0].material, "interior")
        self.assertEqual(jobs[0].symbols, frozenset({"interior-material"}))

    def test_two_meshes_on_one_material_stay_separate(self) -> None:
        # Pooling these is what let the LC500 facelift's rigid domain erase the
        # base interior's mirrored one.
        binding = self._binding("lc500_screens")
        jobs = self._jobs(
            [
                (self._part("lc500_interior"), binding),
                (self._part("lc500_interior_facelift"), binding),
            ]
        )

        self.assertEqual(
            [(job.part.key, job.material) for job in jobs],
            [("lc500_interior", "lc500_screens"),
             ("lc500_interior_facelift", "lc500_screens")],
        )

    def test_two_materials_on_one_mesh_stay_separate(self) -> None:
        # screen.dds carries the 8.9% HVAC strip under lc500_screens and a
        # full-atlas quad under lc500_centralscreen. Unioning their domains
        # made the strip flip about the atlas centre instead of itself.
        interior = self._part("lc500_interior")
        jobs = self._jobs(
            [
                (interior, self._binding("lc500_screens")),
                (interior, self._binding("lc500_centralscreen")),
            ]
        )

        self.assertEqual(
            [job.material for job in jobs],
            ["lc500_screens", "lc500_centralscreen"],
        )
        self.assertEqual(jobs[0].symbols, frozenset({"lc500_screens-material"}))
        self.assertEqual(
            jobs[1].symbols, frozenset({"lc500_centralscreen-material"})
        )

    def test_stage_keys_of_one_material_share_a_job(self) -> None:
        # A base colour and its emissive are the same UV layout.
        interior = self._part("lc500_interior")
        jobs = self._jobs(
            [
                (interior, self._binding("lc500_screens", "baseColorMap")),
                (interior, self._binding("lc500_screens", "emissiveMap")),
            ]
        )

        self.assertEqual(len(jobs), 1)


class PerMeshCorrectionOutputTests(unittest.TestCase):
    """Each job builds its own texture, scoped to its own mesh and material."""

    def _record_export(self, _loaded, _sweep, path, **_kwargs):
        """Remember which mesh was converted, so a test can check the set.

        Read off the output path: the sweep is a stub here, and the exporter
        names each file for the part it is converting.
        """
        self.exported.append(Path(path).name[: -len("_rhd.dae")])
        return {"rigid_symmetric_nodes": []}

    def setUp(self):
        self.exported: list[str] = []

    def _preview(self, entries, mirrored_by_part, layouts=None):
        parts: list[SimpleNamespace] = []
        for part, _binding in entries:
            if not any(part is seen for seen in parts):
                parts.append(part)
        calls: list[dict] = []
        layouts = layouts or {}
        # Meshes get distinct domains unless the test says they share a layout,
        # so an identical mask means a real shared unwrap, not a stub artefact.
        seen_layouts: list[object] = []

        def masks(_loaded, scope, symbols, *_a, **_k):
            mirror = np.zeros((4, 4), dtype=bool)
            if mirrored_by_part.get(scope[0].key, True):
                layout = layouts.get(
                    scope[0].key, (scope[0].key, frozenset(symbols))
                )
                if layout not in seen_layouts:
                    seen_layouts.append(layout)
                mirror[seen_layouts.index(layout):] = True
            return rhd.DomainMasks(
                mirror=mirror, rigid=np.zeros((4, 4), dtype=bool),
                conflict_coverage=0.0, mirrored_triangles=1,
                rigid_triangles=0, parts_analysed=1,
            )

        def build(*_args, **kwargs):
            calls.append(kwargs)
            return rhd.RhdTextureResult(
                texture_member="vehicles/lc500/textures/screen.dds",
                size=(4, 4), parts_analysed=1, mirrored_triangles=1,
                rigid_triangles=0, mirror_coverage=1.0, rigid_coverage=0.0,
                conflict_coverage=0.0, glyph_regions=0, mirrored_glyph_regions=0,
            )

        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            source = workspace / "screen.dds"
            Image.new("RGBA", (4, 4)).save(source, format="PNG")
            with (
                patch.object(
                    rhd, "texture_bindings_for_parts",
                    return_value={"vehicles/lc500/textures/screen.dds": entries},
                ),
                patch.object(
                    rhd, "material_symbols_for_binding",
                    side_effect=lambda _l, b: (f"{b.dae_material}-material",),
                ),
                patch.object(rhd, "extract_archive_member", return_value=source),
                patch.object(rhd, "build_domain_masks", side_effect=masks),
                patch.object(rhd, "build_rhd_texture", side_effect=build),
                patch.object(rhd, "write_blender_preview", return_value=None),
                patch.object(rhd, "sweep_part", return_value=object()),
                patch.object(
                    rhd, "export_transformed_part_dae",
                    side_effect=self._record_export,
                ),
            ):
                rhd.export_parts_preview(
                    SimpleNamespace(), SimpleNamespace(), parts, workspace,
                    bake=False, log=lambda *_a: None,
                )
        return calls

    def _part(self, key: str):
        return SimpleNamespace(
            key=key, node_id=key, node_name=key, label=key, instances=[],
        )

    def _binding(self, material: str):
        return SimpleNamespace(
            dae_material=material, material_key=f"{material}_on",
            materials_member="vehicles/lc500/main.materials.json",
            stage_key="baseColorMap", kind="colour",
        )

    def test_a_single_job_keeps_the_names_it_has_always_had(self) -> None:
        dash = self._part("dash")
        calls = self._preview([(dash, self._binding("interior"))], {})

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["part_group_index"], 0)
        self.assertEqual(calls[0]["part_scope"], [dash])
        self.assertEqual(calls[0]["material_scope"], ("interior",))

    def test_every_selected_mesh_is_still_converted_whatever_the_grouping(self) -> None:
        """The hand conversion is the point; grouping only decides file count.

        Grouping the texture jobs must not touch which meshes get converted.
        A local named ``parts`` inside the grouping shadowed this function's
        own parameter, so the DAE export below iterated the last group instead
        of the selection: scintilla converted one mesh of fifteen, and the
        thirteen whose corrected material was scoped to them were minted,
        bound by nothing and pruned. Nothing about the corrected textures
        themselves was wrong, which is why every plan-level check passed.
        """
        interior = self._part("lc500_interior")
        facelift = self._part("lc500_interior_facelift")
        badge = self._part("lc500_badge")
        self._preview(
            [
                (interior, self._binding("lc500_screens")),
                (facelift, self._binding("lc500_centralscreen")),
                (badge, self._binding("lc500_badges")),
            ],
            {},
        )

        self.assertEqual(
            self.exported,
            ["lc500_interior", "lc500_interior_facelift", "lc500_badge"],
            "a selected mesh was left unconverted",
        )

    def test_each_mesh_and_material_builds_its_own_texture(self) -> None:
        interior = self._part("lc500_interior")
        facelift = self._part("lc500_interior_facelift")
        calls = self._preview(
            [
                (interior, self._binding("lc500_screens")),
                (interior, self._binding("lc500_centralscreen")),
                (facelift, self._binding("lc500_screens")),
            ],
            {},
        )

        self.assertEqual(
            [(c["part_scope"][0].key, c["material_scope"][0]) for c in calls],
            [
                ("lc500_interior", "lc500_screens"),
                ("lc500_interior", "lc500_centralscreen"),
                ("lc500_interior_facelift", "lc500_screens"),
            ],
        )
        self.assertEqual([c["part_group_index"] for c in calls], [1, 2, 3])

    def test_a_job_with_nothing_mirrored_is_skipped_not_the_texture(self) -> None:
        interior = self._part("lc500_interior")
        facelift = self._part("lc500_interior_facelift")
        calls = self._preview(
            [
                (interior, self._binding("lc500_screens")),
                (facelift, self._binding("lc500_screens")),
            ],
            {"lc500_interior_facelift": False},
        )

        self.assertEqual([c["part_scope"][0].key for c in calls], ["lc500_interior"])

    def test_meshes_sharing_a_uv_layout_share_one_correction(self) -> None:
        # A left and right panel on one unwrap would correct to the same image
        # twice, so the equal domains merge into a single build.
        left = self._part("door_panel_L")
        right = self._part("door_panel_R")
        calls = self._preview(
            [
                (left, self._binding("interior")),
                (right, self._binding("interior")),
            ],
            {},
            layouts={"door_panel_L": "shared", "door_panel_R": "shared"},
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["part_scope"], [left, right])
        self.assertEqual(calls[0]["part_group_index"], 0)


class SplitTextureManifestTests(unittest.TestCase):
    """The manifest has to say which meshes each corrected copy is for."""

    def _result(self, group_index: int, scope: list[str]) -> rhd.RhdTextureResult:
        stem = "screen" if group_index < 2 else f"screen_{group_index}"
        report = {
            "layer_bindings": [
                {
                    "dae_material": "lc500_screens",
                    "material_key": "lc500_screens_on",
                    "materials_member": "vehicles/lc500/main.materials.json",
                    "stage_key": "baseColorMap",
                    "kind": "colour",
                }
            ],
            "source_materials": [
                {
                    "key": "lc500_screens_on",
                    "aliases": ["lc500_screens_on"],
                    "materialsMember": "vehicles/lc500/main.materials.json",
                    "material": {"Stages": [{}]},
                }
            ],
            "switch_base_aliases": [],
            "outputs": {"preview": None},
        }
        if group_index:
            report["part_group"] = group_index
            report["part_scope"] = scope
            report["material_scope"] = ["lc500_screens"]
        return rhd.RhdTextureResult(
            texture_member="vehicles/lc500/textures/screen.dds",
            size=(4, 4), parts_analysed=1, mirrored_triangles=1,
            rigid_triangles=0, mirror_coverage=1.0, rigid_coverage=0.0,
            conflict_coverage=0.0, glyph_regions=1, mirrored_glyph_regions=1,
            material_aliases=("lc500_screens", "lc500_screens_on"),
            png_path=Path(f"{stem}_rhd.png"),
            dds_path=Path(f"{stem}_rhd.dds"),
            report=report,
        )

    def _manifest(self, results: list[rhd.RhdTextureResult]) -> dict:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            rhd.write_blender_preview(output, results, log=lambda *_a: None)
            return json.loads((output / "rhd_materials.json").read_text())

    def test_an_unscoped_correction_names_no_parts(self) -> None:
        materials = self._manifest([self._result(0, [])])["materials"]

        self.assertEqual(len(materials), 1)
        self.assertNotIn("partKeys", materials[0])

    def _tell_tale(self, index: int, dae_material: str) -> rhd.RhdTextureResult:
        """One of several DAE materials sharing a single materials-JSON entry."""
        stem = "buttons" if index < 2 else f"buttons_{index}"
        return rhd.RhdTextureResult(
            texture_member="vehicles/lc500/faceliftbuttons.png",
            size=(4, 4), parts_analysed=1, mirrored_triangles=1,
            rigid_triangles=0, mirror_coverage=1.0, rigid_coverage=0.0,
            conflict_coverage=0.0, glyph_regions=1, mirrored_glyph_regions=1,
            material_aliases=(dae_material, "lc500_dashlights"),
            png_path=Path(f"{stem}_rhd.png"),
            dds_path=Path(f"{stem}_rhd.dds"),
            report={
                "layer_bindings": [
                    {
                        "dae_material": dae_material,
                        "material_key": "lc500_dashlights",
                        "materials_member": "vehicles/lc500/main.materials.json",
                        "stage_key": "opacityMap",
                        "kind": "scalar",
                    }
                ],
                "source_materials": [
                    {
                        "key": "lc500_dashlights",
                        "aliases": [
                            "lc500_intsignal_L",
                            "lc500_intsignal_R",
                            "lc500_inthighbeam",
                        ],
                        "materialsMember": "vehicles/lc500/main.materials.json",
                        "material": {"Stages": [{}]},
                    }
                ],
                "switch_base_aliases": [],
                "outputs": {"preview": None},
                "part_group": index,
                "part_scope": ["lc500_interior"],
                "material_scope": [dae_material],
            },
        )

    def test_materials_sharing_one_json_entry_keep_their_own_textures(self) -> None:
        # The LC500's turn-signal and high-beam tell-tales are three DAE
        # materials over one lc500_dashlights entry, each on its own island.
        # Keyed only on that entry they folded together and two of the three
        # corrected files were orphaned.
        materials = self._manifest(
            [
                self._tell_tale(1, "lc500_intsignal_L"),
                self._tell_tale(2, "lc500_intsignal_R"),
                self._tell_tale(3, "lc500_inthighbeam"),
            ]
        )["materials"]

        self.assertEqual(len(materials), 3)
        self.assertEqual(
            [entry["maps"]["opacityMap"] for entry in materials],
            ["buttons_rhd.png", "buttons_2_rhd.png", "buttons_3_rhd.png"],
        )
        # A scoped correction must not claim its siblings' aliases, or they
        # would all retarget onto whichever was minted first.
        self.assertEqual(
            [entry["aliases"][0] for entry in materials],
            ["lc500_intsignal_L", "lc500_intsignal_R", "lc500_inthighbeam"],
        )
        for entry in materials:
            self.assertNotIn(
                "lc500_inthighbeam",
                entry["aliases"][1:],
            )

    def test_two_corrections_of_one_texture_stay_separate_entries(self) -> None:
        materials = self._manifest(
            [self._result(1, ["interior"]), self._result(2, ["facelift"])]
        )["materials"]

        # Same alias, same source material, different meshes and different
        # files: folding these together is what left the facelift mirrored.
        self.assertEqual(len(materials), 2)
        self.assertEqual(
            [entry["partKeys"] for entry in materials],
            [["interior"], ["facelift"]],
        )
        self.assertEqual(
            [entry["maps"]["baseColorMap"] for entry in materials],
            ["screen_rhd.png", "screen_2_rhd.png"],
        )


class SharedLayerRegionTests(unittest.TestCase):
    """One material/part must use one accepted region set across its maps."""

    def test_task_groups_do_not_cross_a_material_or_part_boundary(self) -> None:
        dash = part("dashboard")
        door = part("door")
        mask = np.ones((2, 2), dtype=bool)

        def task(member: str, material: str, scope: DaePart):
            return rhd.TextureCorrectionTask(
                member=member,
                material=material,
                part_scope=(scope,),
                masks=rhd.DomainMasks(
                    mirror=mask,
                    rigid=np.zeros_like(mask),
                    conflict_coverage=0.0,
                    mirrored_triangles=1,
                    rigid_triangles=0,
                    parts_analysed=1,
                ),
                part_group_index=0,
            )

        groups = rhd.shared_layer_task_groups(
            [
                task("interior_b.dds", "interior", dash),
                task("interior_nm.dds", "interior", dash),
                task("trim_r.dds", "trim", dash),
                task("interior_ao.dds", "interior", door),
            ]
        )

        self.assertEqual(
            [[entry.member for _index, entry in group] for group in groups],
            [
                ["interior_b.dds", "interior_nm.dds"],
                ["trim_r.dds"],
                ["interior_ao.dds"],
            ],
        )

    def test_a_layer_with_no_detection_inherits_its_siblings_region(self) -> None:
        ledger = rhd.SharedLayerRegionLedger()
        ledger.record(
            "vehicles/ardente/interior_nm.normal.dds",
            (8, 8),
            [(1, 2, 5, 3)],
            [None],
        )
        ledger.record(
            "vehicles/ardente/interior_b.color.dds",
            (8, 8),
            [],
            [],
        )

        regions, rotations, sources, added = ledger.combined(
            "vehicles/ardente/interior_b.color.dds",
            (8, 8),
            [],
            [],
        )

        self.assertEqual(regions, [(1, 2, 5, 3)])
        self.assertEqual(rotations, [None])
        self.assertEqual(
            sources, ("vehicles/ardente/interior_nm.normal.dds",)
        )
        self.assertEqual(added, 1)

    def test_sibling_boxes_and_rotations_scale_to_this_layers_resolution(self) -> None:
        ledger = rhd.SharedLayerRegionLedger()
        ledger.record(
            "vehicles/ardente/interior_nm.normal.dds",
            (4, 2),
            [(1, 0, 2, 1)],
            [((1.0, 0.0), (3.0, 0.0), (3.0, 1.0), (1.0, 1.0))],
        )

        regions, rotations, _sources, _added = ledger.combined(
            "vehicles/ardente/interior_r.data.dds",
            (8, 4),
            [],
            [],
        )

        self.assertEqual(regions, [(2, 0, 4, 2)])
        self.assertEqual(
            rotations,
            [((2.0, 0.0), (6.0, 0.0), (6.0, 2.0), (2.0, 2.0))],
        )

    def test_authoritative_evidence_keeps_mask_native_precision(self) -> None:
        native_region = (23, 192, 65, 128)
        detection = rhd.RegionDetection(
            source="opacityMap:labels.png",
            detected=1,
            regions=[native_region],
            rotations=[None],
        )
        evidence = rhd.shared_layer_evidence_at_finest_resolution(
            [((1024, 1024), detection)], rhd.DEFAULT_RHD_CONFIG
        )
        self.assertIsNotNone(evidence)
        self.assertEqual(evidence.size, (1024, 1024))

        # The old 1024 -> 32 -> 1024 route visibly broadens this interval.
        coarse = rhd._scale_detection_box(native_region, (1024, 1024), (32, 32))
        broadened = rhd._scale_detection_box(coarse, (32, 32), (1024, 1024))
        self.assertNotEqual(broadened, native_region)

        ledger = rhd.SharedLayerRegionLedger()
        ledger.record_evidence("vehicles/car/null_n.dds", evidence)
        regions, _rotations, _sources, _added = ledger.combined(
            "vehicles/car/labels.color.dds", (1024, 1024), [], []
        )
        self.assertEqual(regions, [native_region])

    def test_build_records_authoritative_regions_before_target_downscaling(self) -> None:
        dash = part("dashboard")
        target_member = "vehicles/car/null_n.dds"
        mask_member = "vehicles/car/labels_o.data.dds"
        binding = rhd.MaterialTextureLayerBinding(
            dae_material="labels",
            material_key="labels",
            materials_member="vehicles/car/main.materials.json",
            texture_reference=f"/{target_member}",
            texture_member=target_member,
            stage_key="normalMap",
            kind="normal",
        )
        mirror = np.zeros((32, 32), dtype=bool)
        mirror[:, :16] = True
        masks = rhd.DomainMasks(
            mirror=mirror,
            rigid=np.zeros_like(mirror),
            conflict_coverage=0.0,
            mirrored_triangles=1,
            rigid_triangles=0,
            parts_analysed=1,
        )
        native_region = (23, 192, 65, 128)
        ledger = rhd.SharedLayerRegionLedger()

        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            target = workspace / "null_n.dds"
            opacity = workspace / "labels_o.data.dds"
            Image.new("RGBA", (32, 32), (128, 128, 255, 255)).save(
                target, format="PNG"
            )
            Image.new("RGBA", (1024, 1024), (0, 0, 0, 255)).save(
                opacity, format="PNG"
            )

            def extracted(_archive, member):
                return opacity if member == mask_member else target

            with (
                patch.object(
                    rhd,
                    "scoped_parts_using_material",
                    return_value=[(dash, binding)],
                ),
                patch.object(
                    rhd,
                    "material_symbols_for_binding",
                    return_value=("labels-material",),
                ),
                patch.object(rhd, "extract_archive_member", side_effect=extracted),
                patch.object(
                    rhd,
                    "authoritative_visibility_masks_for_layer_bindings",
                    return_value=(
                        rhd.CompanionMap(mask_member, "opacityMap", "scalar"),
                    ),
                ),
                patch.object(
                    rhd,
                    "detect_flip_regions_by_uv_island",
                    return_value=rhd.RegionDetection(
                        source="opacityMap:labels_o.data.dds",
                        detected=1,
                        regions=[native_region],
                        rotations=[None],
                    ),
                ),
            ):
                result = rhd.build_rhd_texture(
                    SimpleNamespace(materials=[]),
                    SimpleNamespace(),
                    target_member,
                    workspace,
                    config=replace(
                        rhd.DEFAULT_RHD_CONFIG,
                        detect_on_normal_map=False,
                    ),
                    part_scope=[dash],
                    material_scope=("labels",),
                    masks=masks,
                    shared_layer_regions=ledger,
                    detect_only=True,
                    log=lambda *_a: None,
                )

        self.assertIsNone(result)
        evidence = ledger.layers[target_member.lower()]
        self.assertEqual(evidence.size, (1024, 1024))
        self.assertEqual(evidence.regions, (native_region,))

    def test_build_plans_from_scaled_sibling_evidence_when_own_detector_is_empty(
        self,
    ) -> None:
        dash = part("ardente_dashboard")
        target_member = "vehicles/ardente/interior_b.color.dds"
        binding = ArchiveTextureBinding(
            dae_material="ardente_interior",
            material_key="ardente_interior",
            materials_member="vehicles/ardente/main.materials.json",
            texture_reference=f"/{target_member}",
            texture_member=target_member,
        )
        ledger = rhd.SharedLayerRegionLedger()
        ledger.record(
            "vehicles/ardente/interior_nm.normal.dds",
            (4, 4),
            [(1, 1, 2, 2)],
            [None],
        )
        mirror = np.zeros((8, 8), dtype=bool)
        mirror[:, :4] = True
        masks = rhd.DomainMasks(
            mirror=mirror,
            rigid=np.zeros_like(mirror),
            conflict_coverage=0.0,
            mirrored_triangles=1,
            rigid_triangles=0,
            parts_analysed=1,
        )
        planned_regions: list[tuple[int, int, int, int]] = []

        def plan(_mirror, regions, _config, _axis_map, _components=None):
            planned_regions.extend(regions)
            return [], np.zeros_like(mirror)

        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            source = workspace / "interior_b.color.dds"
            Image.new("RGBA", (8, 8), (0, 0, 0, 255)).save(
                source, format="PNG"
            )
            with (
                patch.object(
                    rhd,
                    "scoped_parts_using_material",
                    return_value=[(dash, binding)],
                ),
                patch.object(
                    rhd,
                    "material_symbols_for_binding",
                    return_value=("ardente_interior-material",),
                ),
                patch.object(rhd, "extract_archive_member", return_value=source),
                patch.object(
                    rhd,
                    "detect_flip_regions",
                    return_value=rhd.RegionDetection(source="colour", detected=0),
                ),
                patch.object(rhd, "plan_island_flips", side_effect=plan),
            ):
                result = rhd.build_rhd_texture(
                    SimpleNamespace(materials=[]),
                    SimpleNamespace(),
                    target_member,
                    workspace,
                    config=replace(
                        rhd.DEFAULT_RHD_CONFIG,
                        detect_on_normal_map=False,
                        enable_skewed_region_filter=False,
                    ),
                    part_scope=[dash],
                    material_scope=("ardente_interior",),
                    masks=masks,
                    shared_layer_regions=ledger,
                    log=lambda *_a: None,
                )

        self.assertIsNotNone(result)
        self.assertEqual(planned_regions, [(2, 2, 4, 4)])
        self.assertEqual(result.report["shared_layer_regions_added"], 1)


class ParallelTextureCorrectionTests(unittest.TestCase):
    """Planning every correction before running any must not reorder the log.

    Corrections used to follow their texture's heading immediately. They are
    now planned for every texture first so the pass can be spread over
    processes, which without care lists every heading and then every
    correction. Each texture holds its lines and replays them with its own.
    """

    def _part(self, key: str):
        return SimpleNamespace(
            key=key, node_id=key, node_name=key, label=key, instances=[],
        )

    def _binding(self, material: str):
        return SimpleNamespace(
            dae_material=material, material_key=f"{material}_on",
            materials_member="vehicles/lc500/main.materials.json",
            stage_key="baseColorMap", kind="colour",
        )

    def test_each_texture_heading_still_precedes_its_own_corrections(self) -> None:
        dash = self._part("dash")
        bindings = {
            "vehicles/lc500/textures/a.dds": [(dash, self._binding("a"))],
            "vehicles/lc500/textures/b.dds": [(dash, self._binding("b"))],
        }

        def masks(*_args, **_kwargs):
            return rhd.DomainMasks(
                mirror=np.ones((4, 4), dtype=bool),
                rigid=np.zeros((4, 4), dtype=bool),
                conflict_coverage=0.0, mirrored_triangles=1,
                rigid_triangles=0, parts_analysed=1,
            )

        def build(*_args, **kwargs):
            kwargs["log"](f"    corrected {kwargs['material_scope'][0]}")
            return rhd.RhdTextureResult(
                texture_member="vehicles/lc500/textures/a.dds",
                size=(4, 4), parts_analysed=1, mirrored_triangles=1,
                rigid_triangles=0, mirror_coverage=1.0, rigid_coverage=0.0,
                conflict_coverage=0.0, glyph_regions=0, mirrored_glyph_regions=0,
            )

        lines: list[str] = []
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            source = workspace / "a.dds"
            Image.new("RGBA", (4, 4)).save(source, format="PNG")
            with (
                patch.object(
                    rhd, "texture_bindings_for_parts", return_value=bindings,
                ),
                patch.object(
                    rhd, "material_symbols_for_binding",
                    side_effect=lambda _l, b: (f"{b.dae_material}-material",),
                ),
                patch.object(rhd, "extract_archive_member", return_value=source),
                patch.object(rhd, "build_domain_masks", side_effect=masks),
                patch.object(rhd, "build_rhd_texture", side_effect=build),
                patch.object(rhd, "write_blender_preview", return_value=None),
                patch.object(rhd, "sweep_part", return_value=object()),
                patch.object(
                    rhd, "export_transformed_part_dae",
                    return_value={"rigid_symmetric_nodes": []},
                ),
            ):
                rhd.export_parts_preview(
                    SimpleNamespace(), SimpleNamespace(), [dash], workspace,
                    bake=False, log=lines.append,
                )

        interesting = [
            line for line in lines
            if line.strip().startswith(("a.dds", "b.dds", "corrected"))
        ]
        self.assertEqual(
            interesting,
            ["\na.dds", "    corrected a", "\nb.dds", "    corrected b"],
        )


class TextureWorkerCountTests(unittest.TestCase):
    """One means this process; zero means choose; anything else is taken as is."""

    def _config(self, workers: int):
        return replace(rhd.DEFAULT_RHD_CONFIG, texture_job_workers=workers)

    def test_the_default_keeps_the_pass_in_this_process(self) -> None:
        # A pool puts the work beyond reach of anything that patched
        # build_rhd_texture, so the exporter stays serial until asked.
        self.assertEqual(rhd.DEFAULT_RHD_CONFIG.texture_job_workers, 1)
        self.assertEqual(
            rhd.texture_correction_worker_count(self._config(1), 40), 1
        )

    def test_a_count_is_never_more_than_there_is_work(self) -> None:
        self.assertEqual(rhd.texture_correction_worker_count(self._config(8), 3), 3)
        self.assertEqual(rhd.texture_correction_worker_count(self._config(0), 1), 1)
        self.assertEqual(rhd.texture_correction_worker_count(self._config(8), 0), 1)

    def test_zero_chooses_and_a_negative_count_is_refused(self) -> None:
        chosen = rhd.texture_correction_worker_count(self._config(0), 40)
        self.assertGreaterEqual(chosen, 1)
        self.assertLessEqual(chosen, 4)
        self.assertEqual(rhd.texture_correction_worker_count(self._config(-3), 40), 1)



class MaterialAliasOrderTests(unittest.TestCase):
    """Aliases must come out in the same order whatever the hash seed.

    They are assembled from a set of COLLADA symbols, and Python randomises
    string hashing per process. A pooled correction pass and a serial one over
    the V60 agreed on all 59 plans except the order of this one field, because
    the pool's workers hashed differently from the parent.
    """

    def _binding(self, material: str, key: str):
        return SimpleNamespace(dae_material=material, material_key=key)

    def test_the_binding_leads_and_the_symbols_follow_in_order(self) -> None:
        candidates = [(SimpleNamespace(key="dash"), self._binding("buttons", "buttons_off"))]
        symbols = {"buttons-material", "buttons_off-material", "aaa-material"}

        self.assertEqual(
            rhd.material_aliases_for_candidates(candidates, symbols),
            (
                "buttons",
                "buttons_off",
                "aaa-material",
                "buttons-material",
                "buttons_off-material",
            ),
        )

    def test_the_order_does_not_follow_the_sets_iteration(self) -> None:
        # Sets of these strings iterate differently per process; building the
        # same aliases from equal sets constructed differently must not.
        candidates = [(SimpleNamespace(key="dash"), self._binding("m", "m_off"))]
        forward = {f"sym{index}-material" for index in range(12)}
        backward = {f"sym{index}-material" for index in reversed(range(12))}
        self.assertEqual(
            rhd.material_aliases_for_candidates(candidates, forward),
            rhd.material_aliases_for_candidates(candidates, backward),
        )

    def test_a_name_is_kept_at_its_first_appearance(self) -> None:
        candidates = [
            (SimpleNamespace(key="a"), self._binding("m", "m_off")),
            (SimpleNamespace(key="b"), self._binding("m", "m_on")),
        ]
        aliases = rhd.material_aliases_for_candidates(candidates, {"m-material"})
        self.assertEqual(aliases, ("m", "m_off", "m-material", "m_on"))
        self.assertEqual(len(aliases), len(set(aliases)))

    def test_an_empty_name_is_dropped(self) -> None:
        candidates = [(SimpleNamespace(key="a"), self._binding("m", ""))]
        self.assertEqual(
            rhd.material_aliases_for_candidates(candidates, {"m-material"}),
            ("m", "m-material"),
        )

class PoolFailureFallbackTests(unittest.TestCase):
    """A correction a worker loses is done again here, and said out loud.

    Four workers hold four GL contexts on one card, and a correction that
    cannot get one raises rather than degrading. Dropping it silently would
    ship the mesh on an uncorrected atlas looking like nothing had gone wrong.
    """

    def _part(self, key: str):
        return SimpleNamespace(
            key=key, node_id=key, node_name=key, label=key, instances=[],
        )

    def _binding(self, material: str):
        return SimpleNamespace(
            dae_material=material, material_key=f"{material}_on",
            materials_member="vehicles/lc500/main.materials.json",
            stage_key="baseColorMap", kind="colour",
        )

    class _FakeFuture:
        def __init__(self, outcome):
            self._outcome = outcome

        def result(self):
            return self._outcome

    def _fake_pool(self, outcome_for_worker):
        test = self

        class FakePool:
            def __init__(self, *_args, **_kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

            def submit(self, _fn, task, _directory):
                return test._FakeFuture(outcome_for_worker(task))

        return FakePool

    def _run(self, outcome_for_worker, build_side_effect, workers=2):
        dash = self._part("dash")
        bindings = {
            "vehicles/lc500/textures/a.dds": [(dash, self._binding("a"))],
            "vehicles/lc500/textures/b.dds": [(dash, self._binding("b"))],
        }

        def masks(*_args, **_kwargs):
            return rhd.DomainMasks(
                mirror=np.ones((4, 4), dtype=bool),
                rigid=np.zeros((4, 4), dtype=bool),
                conflict_coverage=0.0, mirrored_triangles=1,
                rigid_triangles=0, parts_analysed=1,
            )

        lines: list[str] = []
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            source = workspace / "a.dds"
            Image.new("RGBA", (4, 4)).save(source, format="PNG")
            with (
                patch.object(rhd, "texture_bindings_for_parts", return_value=bindings),
                patch.object(
                    rhd, "material_symbols_for_binding",
                    side_effect=lambda _l, b: (f"{b.dae_material}-material",),
                ),
                patch.object(rhd, "extract_archive_member", return_value=source),
                patch.object(rhd, "build_domain_masks", side_effect=masks),
                patch.object(rhd, "build_rhd_texture", side_effect=build_side_effect),
                patch.object(rhd, "write_blender_preview", return_value=None),
                patch.object(rhd, "sweep_part", return_value=object()),
                patch.object(
                    rhd, "export_transformed_part_dae",
                    return_value={"rigid_symmetric_nodes": []},
                ),
                patch.object(
                    rhd, "ProcessPoolExecutor", self._fake_pool(outcome_for_worker),
                ),
                patch.object(rhd, "_texture_worker_init", lambda *_a, **_k: None),
            ):
                preview = rhd.export_parts_preview(
                    SimpleNamespace(),
                    # The pool branch reads loaded.path to tell a worker which
                    # DAE to rebuild; the serial branch never does.
                    SimpleNamespace(path=workspace / "vehicle.dae", parts=[dash]),
                    [dash], workspace,
                    replace(
                        rhd.DEFAULT_RHD_CONFIG, texture_job_workers=workers,
                    ),
                    bake=False, log=lines.append,
                )
        return preview, lines

    def _result(self):
        return rhd.RhdTextureResult(
            texture_member="vehicles/lc500/textures/a.dds",
            size=(4, 4), parts_analysed=1, mirrored_triangles=1,
            rigid_triangles=0, mirror_coverage=1.0, rigid_coverage=0.0,
            conflict_coverage=0.0, glyph_regions=0, mirrored_glyph_regions=0,
        )

    def test_a_correction_a_worker_lost_is_recovered_in_process(self) -> None:
        def worker_fails(_task):
            return rhd.TextureCorrectionTaskOutcome(
                None, [], "LocalContrastGpuUnavailable: no GL context"
            )

        preview, lines = self._run(worker_fails, lambda *_a, **k: self._result())
        text = chr(10).join(lines)

        self.assertIn("failed on a worker", text)
        self.assertIn("running them again on the serial path", text)
        self.assertIn("recovered on the serial path", text)
        self.assertNotIn("FAILED, uncorrected", text)
        self.assertEqual(len(preview.textures), 2)

    def test_a_failure_that_survives_the_retry_is_reported_not_hidden(self) -> None:
        def worker_fails(_task):
            return rhd.TextureCorrectionTaskOutcome(None, [], "ValueError: bad atlas")

        def build_also_fails(*_args, **_kwargs):
            raise ValueError("bad atlas")

        preview, lines = self._run(worker_fails, build_also_fails)
        text = chr(10).join(lines)

        self.assertIn("running them again on the serial path", text)
        self.assertIn("FAILED, uncorrected", text)
        self.assertNotIn("recovered on the serial path", text)
        self.assertEqual(preview.textures, [])


if __name__ == "__main__":
    unittest.main()
