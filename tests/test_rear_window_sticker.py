"""2026-09-08: Surface fit, texture orientation and export isolation checks."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image

from beamxp import hand_drive_core as core
from beamxp import rear_window_sticker as sticker
from beamxp.core.sjson import decode


def pane():
    corners = np.array([[.7, 2., 1.], [-.7, 2., 1.], [-.6, 1.7, 1.5], [.6, 1.7, 1.5]])
    return corners[[[0, 1, 2], [0, 2, 3]]]


class StickerTests(unittest.TestCase):
    def test_lower_right_fit_and_readable_uv(self):
        triangles, uv = sticker.place_sticker(pane(), 1500 / 430)
        triangles, uv = sticker._clip(triangles, uv, (0, 1, 0, 1))
        self.assertTrue((triangles[:, :, 0] < 0).all())
        self.assertLess(triangles[:, :, 2].max(), 1.2)
        np.testing.assert_allclose(np.ptp(triangles[:, :, 0]), sticker.WIDTH)
        self.assertAlmostEqual(uv.min(), 0)
        self.assertAlmostEqual(uv.max(), 1)
        # 2026-09-08: U increases to vehicle -X when viewed from behind.
        self.assertLess(np.corrcoef(triangles[:, :, 0].ravel(), uv[:, :, 0].ravel())[0, 1], -.999)

    def test_sticker_keeps_glass_topology(self):
        source = pane()
        triangles, uv = sticker.place_sticker(source, 3.5)
        normal = np.cross(source[0, 1] - source[0, 0], source[0, 2] - source[0, 0])
        normal /= np.linalg.norm(normal)
        for face in triangles - normal * sticker.LIFT:
            self.assertTrue(any(np.allclose(face, original, atol=1e-12) for original in source))
        _, visible = sticker._clip(triangles, uv, (0, 1, 0, 1))
        self.assertAlmostEqual(sticker._area(visible), 1.)

    def test_rejects_side_panes_and_duplicate_surfaces(self):
        side = pane()[:, :, [1, 0, 2]]
        self.assertIsNone(sticker.place_sticker(side, 3.5))
        self.assertIsNone(sticker.place_sticker(np.concatenate([pane(), pane()]), 3.5))
        self.assertIsNone(sticker.place_sticker(pane() * .1, 3.5))

    def test_follows_sloped_lower_edge_and_stays_outside(self):
        for slope in (-.12, .12):
            with self.subTest(slope=slope):
                corners = np.array([[.7, 2., 1. - .7 * slope], [-.7, 2., 1. + .7 * slope],
                                    [-.7, 2., 1.6], [.7, 2., 1.6]])
                mesh, uv = sticker.place_sticker(corners[[[0, 1, 2], [0, 2, 3]]], 3.5)
                mapping = np.linalg.lstsq(np.column_stack((uv.reshape(-1, 2), np.ones(uv.size // 2))),
                                         mesh.reshape(-1, 3), rcond=None)[0]
                expected = np.array([-1., 0., slope])
                expected /= np.linalg.norm(expected)
                np.testing.assert_allclose(mapping[0] / sticker.WIDTH, expected, atol=1e-10)
                self.assertAlmostEqual(np.linalg.norm(mapping[1]), sticker.WIDTH / 3.5)
                np.testing.assert_allclose(mesh[:, :, 1], 2. + sticker.LIFT)

    def test_lower_edge_fit_ignores_internal_triangulation(self):
        uv = np.array([[[0., 0.], [1., .2], [1., 1.]], [[0., 0.], [1., 1.], [0., 1.]]])
        direction = sticker._lower_edge_direction(uv, .4, .8)
        np.testing.assert_allclose(direction, np.array([1., .2]) / np.hypot(1, .2))

    def test_dae_round_trip(self):
        mesh, uv = sticker.place_sticker(pane(), 3.5)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sticker.dae"
            sticker.write_sticker_dae(path, "test_sticker", "test_material", mesh, uv)
            actual = core.surface_triangles_from_tree(ET.parse(path))["test_sticker"]
            np.testing.assert_allclose(actual, mesh, atol=1e-8)

    def test_export_clones_only_active_pane_and_keeps_damage_binding(self):
        text = json.dumps({
            "car": {"slotType": "main", "slots": [["type", "default", "description"], ["glass", "pane", "Glass"]],
                    "nodes": [["id", "posX", "posY", "posZ"], {"group": "body"}, ["n1", 0, 0, 0]]},
            "pane": {"slotType": "glass", "information": {"name": "Rear Glass"},
                     "flexbodies": [["mesh", "[group]:", "nonFlexMaterials"],
                                    {"deformGroup": "glass_break", "deformSound": "smash"},
                                    ["rear", ["body"], []], {"deformGroup": ""}]}})
        index = core.build_part_body_index({"source.jbeam": text})
        context = SimpleNamespace(part_body_index=index, vehicle_id="car", source_vehicle_id="car",
                                  source_zip=Path("car.zip"), vehicle_path="vehicles/car", jbeam_texts={"source.jbeam": text},
                                  objects={"rear": SimpleNamespace(dae_source_zip=None, dae_path="car.dae")})
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            (out / "jbeam").mkdir()
            generated = {"pane_generated": json.loads(text)["pane"]}
            (out / "jbeam/generated.jbeam").write_text(json.dumps(generated))
            for name, pc in (("converted", {"parts": {"glass": "pane_generated"}}),
                             ("plates", {}), ("removed", {"parts": {"glass": ""}})):
                (out / f"{name}.pc").write_text(json.dumps(pc))
            plans = [{"output": name, "kind": kind} for name, kind in
                     (("converted", core.BUILD_CONVERTED), ("plates", core.BUILD_ORIGINAL), ("removed", core.BUILD_CONVERTED))]
            with (patch.object(core, "mesh_material_symbols", return_value={"rear": ("glass",)}),
                  patch.object(core, "material_flags_for_context", return_value={"glass": {"glass": True}}),
                  patch.object(core, "parse_dae"),
                  patch.object(core, "surface_triangles_from_tree", return_value={"rear": pane()})):
                report = sticker.apply_to_build(context, {"rearWindowSticker": True}, out, plans)
            self.assertEqual(len(report["placed"]), 1)
            self.assertTrue(report["placed"][0]["part"].startswith("pane_generated_"))
            self.assertEqual(len(report["skipped"]), 1)
            self.assertEqual((out / "plates.pc").read_text(), "{}")
            materials = json.loads((out / "beamxp_stickers.materials.json").read_text())
            visible = next(m for m in materials.values() if m["Stages"][0].get("colorMap"))
            self.assertFalse(visible["translucent"])
            self.assertTrue(visible["alphaTest"])
            self.assertEqual(visible["alphaRef"], 128)
            with (Image.open(out / "beamxp_imported_by.color.png") as colour,
                  Image.open(out / "beamxp_imported_by_opacity.data.png") as opacity):
                self.assertTrue(all(16 <= axis <= 4096 and axis & (axis - 1) == 0 for axis in colour.size))
                self.assertEqual(opacity.size, colour.size)
                self.assertEqual(colour.mode, "RGB")
                self.assertEqual(opacity.mode, "L")
                alpha = np.asarray(opacity)
                self.assertTrue((alpha[[0, -1]] == 0).all())
                self.assertTrue((alpha[:, [0, -1]] == 0).all())
                self.assertGreater(alpha.max(), 0)
            clone_text = (out / "beamxp_stickers.jbeam").read_text()
            clone_index = core.build_part_body_index({"sticker.jbeam": clone_text})
            body = next(iter(clone_index.values()))[0]
            self.assertEqual(core.part_information_name(body), "Rear Glass (BeamXP sticker)")
            # 2026-09-08: Check raw column positions, independently of our
            # permissive row reader: the engine only merges trailing options.
            raw = decode(core.transform_helpers.extract_named_array(body, "flexbodies"))
            self.assertEqual(raw[1][:3], [report["placed"][0]["mesh"], ["body"], []])
            self.assertEqual(len(raw[1]), len(raw[0]) + 1)
            self.assertEqual(raw[1][3]["deformGroup"], "glass_break")
            self.assertEqual(raw[1][3]["deformMaterialBase"], visible["name"])
            rows = list(core.iter_jbeam_table_rows(core.transform_helpers.extract_named_array(body, "flexbodies")))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["deformGroup"], "glass_break")
            self.assertEqual(rows[0]["deformSound"], "")
            self.assertEqual(rows[1]["deformSound"], "smash")
            self.assertEqual(rows[0]["[group]:"], ["body"])
            selected = core.resolve_selected_parts(json.loads((out / "converted.pc").read_text()), {}, vehicle_id="car",
                                                   part_body_index={**index, **clone_index})
            self.assertIn(report["placed"][0]["part"], selected["parts"])

    def test_disabled_does_not_read_assets(self):
        with patch.object(sticker, "sticker_asset", side_effect=AssertionError):
            self.assertFalse(sticker.apply_to_build(None, {}, None, None)["enabled"])

    def test_texture_cooker_minimum_size_and_source_preservation(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            source = out / "source.png"
            Image.new("RGBA", (7, 17), (20, 40, 60, 128)).save(source)
            original = source.read_bytes()
            sticker.write_sticker_textures(source, out)
            self.assertEqual(source.read_bytes(), original)
            with Image.open(out / "beamxp_imported_by.color.png") as colour:
                self.assertEqual(colour.size, (16, 32))
            with Image.open(out / "beamxp_imported_by_opacity.data.png") as opacity:
                self.assertEqual(opacity.size, (16, 32))
                self.assertEqual(opacity.getextrema(), (128, 128))

    def test_padded_atlas_keeps_artwork_and_uv_orientation(self):
        chart = np.array([[[-.2, -.7], [1.2, -.7], [1.2, 1.3]]])
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            source = out / "source.png"
            pixels = np.zeros((32, 64, 4), dtype=np.uint8)
            pixels[:16] = [255, 0, 0, 255]
            pixels[16:] = [0, 0, 255, 128]
            Image.fromarray(pixels).save(source)
            scale, offset = sticker.write_sticker_textures(source, out, [chart])
            with (Image.open(out / "beamxp_imported_by.color.png") as colour,
                  Image.open(out / "beamxp_imported_by_opacity.data.png") as alpha):
                size = np.array(colour.size)
                xy = np.rint(offset * size).astype(int)
                width, height = np.rint(scale * size).astype(int)
                region = (xy[0], size[1] - xy[1] - height, xy[0] + width, size[1] - xy[1])
                np.testing.assert_array_equal(colour.crop(region), pixels[:, :, :3])
                np.testing.assert_array_equal(alpha.crop(region), pixels[:, :, 3])
                remapped = chart * scale + offset
                self.assertTrue(((remapped > 0) & (remapped < 1)).all())
                self.assertEqual(np.count_nonzero(alpha), 32 * 64)


if __name__ == "__main__":
    unittest.main()
