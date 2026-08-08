from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from beamxp import hand_drive_core as core
from beamxp import mesh_preview


class MeshPreviewSceneTests(unittest.TestCase):
    def test_stable_instance_ref_aliases_duplicate_current_instances(self) -> None:
        geometry = mesh_preview.DaeGeometry(
            geoms={
                "geom": (
                    np.array(
                        [
                            [0.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                        ],
                        dtype=np.float32,
                    ),
                    np.array([[0, 1, 2]], dtype=np.int32),
                )
            },
            nodes={"headliner_node": [(np.eye(4), "geom")]},
            node_matrices={"headliner_node": np.eye(4)},
        )
        payload = {
            "dae_files": [{"path": "dummy.dae"}],
            "instances": [
                {
                    "dae": 0,
                    "node": "headliner_node",
                    "mesh": "racingseat_base",
                    "instance_ref": "racingseat_base@@/seat_FL/",
                    "kind": "flex",
                    "mode": "skip",
                    "matrix": np.eye(4).reshape(-1).tolist(),
                },
                {
                    "dae": 0,
                    "node": "headliner_node",
                    "mesh": "racingseat_base",
                    "instance_ref": "racingseat_base@@/seat_FR/",
                    "kind": "flex",
                    "mode": "skip",
                    "matrix": np.eye(4).reshape(-1).tolist(),
                }
            ],
        }
        with patch.object(mesh_preview, "load_dae_geometry", return_value=geometry):
            scene = mesh_preview.build_scene(payload, Path("cache"))

        self.assertIn("racingseat_base", scene.groups)
        self.assertIn("racingseat_base@@/seat_FL/", scene.groups)
        self.assertIn("racingseat_base@@/seat_FR/", scene.groups)
        self.assertEqual(scene.alias_to_mesh["racingseat_base@@/seat_FL/"], "racingseat_base")
        self.assertEqual(
            scene.pick_names,
            ["racingseat_base@@/seat_FL/", "racingseat_base@@/seat_FR/"],
        )


class TriggerBoxSceneTests(unittest.TestCase):
    """Trigger shapes are drawn geometry, not loaded geometry.

    A trigger has no mesh -- three node references, an offset and a shape --
    so the preview builds its surface and appends it to the same buffers as
    everything else. That is what makes them selectable without a second
    render path.
    """

    BOX_FACES = [
        [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
        [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
        [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
    ]

    @staticmethod
    def _cube(centre, half=0.05):
        return [
            [centre[0] + sx * half, centre[1] + sy * half, centre[2] + sz * half]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ]

    def _box(self, trigger_id, centre, mode="skip", at=None, stock=None):
        """One box-shaped trigger, in the shape the payload builder emits."""
        return {
            "id": trigger_id,
            "at": list(at if at is not None else centre),
            "mode": mode,
            "shape": "box",
            "faces": [list(face) for face in self.BOX_FACES],
            "vertices": self._cube(centre),
            "vertices_stock": self._cube(stock if stock is not None else centre),
        }

    def _scene(self, boxes):
        payload = {"dae_files": [], "instances": [], "trigger_boxes": boxes}
        return mesh_preview.build_scene(payload, Path("cache"))

    def test_a_box_becomes_a_pickable_twelve_triangle_group(self) -> None:
        scene = self._scene([self._box("hood_int", (0.69, -0.76, 0.51), "mirror")])
        name = mesh_preview.trigger_scene_name("hood_int", [0.69, -0.76, 0.51])
        self.assertIn(name, scene.groups)
        self.assertIn(name, scene.pick_names)
        self.assertEqual(scene.trigger_names, [name])
        tri_start, tri_end, vert_start, vert_end = scene.groups[name][0]
        self.assertEqual(tri_end - tri_start, 12)
        self.assertEqual(vert_end - vert_start, 8)

    def test_a_sphere_is_drawn_as_a_sphere_not_its_bounding_cube(self) -> None:
        """The shape a sphere trigger actually is.

        etk800 alone ships nine of them and the Ardente twenty-one, all at
        15-25 mm, where the difference between a ball and the cube around it
        is most of what you see.
        """
        vertices, faces = core.trigger_sphere_mesh((0.4, -0.5, 0.9), 0.02)
        scene = self._scene([{
            "id": "sw_ignition",
            "at": [0.4, -0.5, 0.9],
            "mode": "translate",
            "shape": "sphere",
            "faces": [list(face) for face in faces],
            "vertices": [list(v) for v in vertices],
            "vertices_stock": [list(v) for v in vertices],
        }])
        name = scene.trigger_names[0]
        tri_start, tri_end, vert_start, vert_end = scene.groups[name][0]
        self.assertGreater(tri_end - tri_start, 12)
        self.assertEqual(vert_end - vert_start, len(vertices))

        # Every drawn point sits on the sphere, so it never exceeds the radius
        # the way the old bounding cube did at its corners.
        block = scene.verts_converted[vert_start:vert_end]
        radii = np.linalg.norm(block - np.asarray([0.4, -0.5, 0.9], dtype=np.float32), axis=1)
        self.assertTrue(np.allclose(radii, 0.02, atol=1e-5))

    def _colours(self, scene):
        out = []
        for name in scene.trigger_names:
            _ts, _te, vert_start, _ve = scene.groups[name][0]
            out.append(int(scene.color_ids[vert_start]))
        return out

    def test_the_box_is_coloured_by_the_transform_it_will_receive(self) -> None:
        scene = self._scene([
            self._box("a", (0.0, 0.0, 0.0), "mirror"),
            self._box("b", (1.0, 0.0, 0.0), "translate"),
        ])
        expected = mesh_preview.MODE_PALETTE_INDEX
        self.assertEqual(self._colours(scene), [expected["mirror"], expected["translate"]])

    def test_an_untranslated_box_is_red_not_the_meshes_grey(self) -> None:
        """A box nothing moves is the failure the table exists to catch.

        Skipped MESHES are grey, so a box left behind has to be something
        else or it reads as ordinary.
        """
        for action in ("skip", "", None):
            with self.subTest(action=action):
                box = self._box("left_behind", (0.0, 0.0, 0.0))
                box["action"] = action
                scene = self._scene([box])
                self.assertEqual(
                    self._colours(scene), [mesh_preview.TRIGGER_UNMOVED_INDEX]
                )
        self.assertNotEqual(
            mesh_preview.TRIGGER_UNMOVED_INDEX, mesh_preview.MODE_PALETTE_INDEX["skip"]
        )
        red = mesh_preview.TRIGGER_UNMOVED_COLOR
        self.assertGreater(red[0], red[1] + 0.3)
        self.assertGreater(red[0], red[2] + 0.3)

    def test_the_action_decides_the_colour_over_a_stale_mode(self) -> None:
        # mode is the user's answer; action is what will actually happen, and
        # an unanswered box still moves if the attribution resolved it.
        box = self._box("resolved", (0.0, 0.0, 0.0), mode="")
        box["action"] = "mirror"
        scene = self._scene([box])
        self.assertEqual(
            self._colours(scene), [mesh_preview.MODE_PALETTE_INDEX["mirror"]]
        )

    def test_a_scene_of_only_triggers_still_renders(self) -> None:
        # No parts loaded yet, or every mesh filtered out: the triggers must
        # not take the "nothing to draw" early return with them.
        scene = self._scene([self._box("solo", (0.0, 0.0, 0.0))])
        self.assertEqual(scene.triangle_count, 12)

    def test_a_malformed_trigger_is_skipped_rather_than_breaking_the_scene(self) -> None:
        bad = self._box("bad", (0.0, 0.0, 0.0))
        bad["vertices"] = [[0.0, 0.0, 0.0]]
        scene = self._scene([bad, self._box("good", (1.0, 0.0, 0.0))])
        self.assertEqual(len(scene.trigger_names), 1)
        self.assertIn("good", scene.trigger_names[0])

    def test_a_trigger_with_no_faces_is_skipped(self) -> None:
        faceless = self._box("faceless", (0.0, 0.0, 0.0))
        faceless["faces"] = []
        self.assertEqual(self._scene([faceless]).trigger_names, [])

    def test_converted_and_stock_placements_go_to_their_own_buffers(self) -> None:
        """Original layout must move the triggers with the geometry."""
        scene = self._scene([
            self._box("hood_int", (-0.7, 0.0, 0.0), "mirror",
                      at=(0.7, 0.0, 0.0), stock=(0.7, 0.0, 0.0)),
        ])
        _ts, _te, vert_start, vert_end = scene.groups[scene.trigger_names[0]][0]
        self.assertAlmostEqual(
            float(scene.verts_converted[vert_start:vert_end].mean(axis=0)[0]), -0.7, places=5
        )
        self.assertAlmostEqual(
            float(scene.verts_stock[vert_start:vert_end].mean(axis=0)[0]), 0.7, places=5
        )

    def test_a_trigger_with_no_stock_placement_uses_its_converted_one(self) -> None:
        box = self._box("hood_int", (0.4, 0.0, 0.0))
        box.pop("vertices_stock")
        scene = self._scene([box])
        _ts, _te, vert_start, vert_end = scene.groups[scene.trigger_names[0]][0]
        self.assertTrue(
            np.allclose(
                scene.verts_converted[vert_start:vert_end],
                scene.verts_stock[vert_start:vert_end],
            )
        )

    def test_triggers_survive_the_parts_visibility_filter(self) -> None:
        """The filter comes from the parts table, which has no trigger rows.

        Passing trigger groups through it left every one out of the index
        buffer, so none of them were drawn at all.
        """
        scene = self._scene([self._box("hood_int", (0.0, 0.0, 0.0), "mirror")])
        box = scene.trigger_names[0]

        # what the app actually passes: the visible mesh rows, no triggers
        names = mesh_preview.visible_group_names(scene, {"some_mesh"})
        self.assertIn(box, names)
        self.assertIn("some_mesh", names)

        # and an empty filter must not take them down with it
        self.assertIn(box, mesh_preview.visible_group_names(scene, set()))

    def test_triggers_can_still_be_switched_off_deliberately(self) -> None:
        scene = self._scene([self._box("hood_int", (0.0, 0.0, 0.0), "mirror")])
        names = mesh_preview.visible_group_names(scene, {"some_mesh"}, False)
        self.assertEqual(names, {"some_mesh"})

    def test_selecting_outlines_it_without_repainting_it(self) -> None:
        """Renders for real when a GL context is available.

        The outline is the only thing that marks a selection now, so this
        checks both halves: outline-coloured pixels appear, and the object's
        own pixels keep the colour that says what it is set to.
        """
        renderer = mesh_preview.create_renderer()
        if renderer is None:
            self.skipTest("no OpenGL context available")
        scene = self._scene([self._box("hood_int", (0.0, 0.0, 0.0), "mirror")])
        renderer.upload_scene(scene)
        name = scene.trigger_names[0]

        def shot(selected: bool) -> np.ndarray:
            renderer.set_selection({name} if selected else set())
            image = renderer.render(
                240, 180, target=(0.0, 0.0, 0.0), yaw=0.8, pitch=0.4, distance=0.5
            )
            return np.asarray(image).astype(int)

        plain, picked = shot(False), shot(True)
        outline = np.asarray(mesh_preview.OUTLINE_COLOR) * 255
        on_outline = np.abs(picked - outline).sum(axis=2) < 90
        self.assertGreater(int(on_outline.sum()), 100, "no outline appeared")

        elsewhere = ~on_outline
        kept = np.abs(picked - plain).sum(axis=2)[elsewhere] < 30
        self.assertGreater(kept.mean(), 0.95, "selection repainted the object")

    def test_the_outline_draws_in_front_of_occluding_geometry(self) -> None:
        """A selected part behind something still shows its outline.

        The outline pass runs after the geometry with depth testing off, which
        is the whole point: a trigger set into a dashboard would otherwise be
        marked selected somewhere you cannot see.
        """
        renderer = mesh_preview.create_renderer()
        if renderer is None:
            self.skipTest("no OpenGL context available")
        scene = self._scene([
            self._box("behind", (0.0, 0.0, 0.0), "mirror"),
            {
                **self._box("infront", (0.3, -0.3, 0.2), "skip"),
                "vertices": self._cube((0.3, -0.3, 0.2), 0.30),
                "vertices_stock": self._cube((0.3, -0.3, 0.2), 0.30),
            },
        ])
        renderer.upload_scene(scene)
        hidden = next(n for n in scene.trigger_names if "behind" in n)

        def shot(names) -> np.ndarray:
            renderer.set_selection(names)
            return np.asarray(renderer.render(
                240, 180, target=(0.0, 0.0, 0.0), yaw=0.8, pitch=0.4, distance=0.9
            )).astype(int)

        plain, picked = shot(set()), shot({hidden})
        outline = np.asarray(mesh_preview.OUTLINE_COLOR) * 255
        drawn = (np.abs(picked - outline).sum(axis=2) < 90) & (
            np.abs(picked - plain).sum(axis=2) > 30
        )
        self.assertGreater(
            int(drawn.sum()), 50,
            "the occluded selection drew no outline through the geometry in front",
        )

    def test_boxes_are_drawn_half_see_through_in_their_own_pass(self) -> None:
        """Renders for real when a GL context is available.

        The boxes surround geometry, so an opaque one hides the very switch it
        marks. They are drawn last with depth writes off, which is also what
        stops two overlapping boxes rejecting each other.
        """
        renderer = mesh_preview.create_renderer()
        if renderer is None:
            self.skipTest("no OpenGL context available")
        box = self._box("left_behind", (0.0, -0.4, 0.0))
        box["action"] = "skip"
        scene = self._scene([box])
        renderer.upload_scene(scene)
        renderer.set_visible({"some_mesh"})     # meshes only, as the app passes

        # the boxes go in their own index buffer, not the mesh one
        self.assertEqual(renderer._index_count, 0)
        self.assertEqual(renderer._trigger_index_count, 12 * 3)

        image = np.asarray(renderer.render(
            240, 180, target=(0.0, 0.0, 0.0), yaw=0.0, pitch=0.0, distance=1.2
        )).astype(int)
        background = np.asarray(mesh_preview.PREVIEW_BACKGROUND) * 255
        painted = np.abs(image - background).sum(axis=2) > 20
        self.assertGreater(int(painted.sum()), 100, "the box was not drawn")

        pixels = image[painted]
        # red channel leads, and the result is blended rather than flat red
        self.assertGreater(pixels[:, 0].mean(), pixels[:, 1].mean() + 20)
        self.assertGreater(pixels[:, 0].mean(), pixels[:, 2].mean() + 20)
        flat = np.asarray(mesh_preview.TRIGGER_UNMOVED_COLOR) * 255
        self.assertGreater(
            float(np.abs(pixels.mean(axis=0) - flat).sum()), 20,
            "the box looks opaque rather than half see-through",
        )

    def test_a_box_can_be_clicked_even_with_no_mesh_on_screen(self) -> None:
        """Regression: the translucent pass took the boxes out of picking.

        Moving the trigger triangles into their own index buffer left the pick
        VAOs bound to the mesh buffer, so boxes were drawn but not clickable --
        and pick() bailed outright when no mesh was visible.
        """
        renderer = mesh_preview.create_renderer()
        if renderer is None:
            self.skipTest("no OpenGL context available")
        scene = self._scene([self._box("hood_int", (0.0, 0.0, 0.0), "mirror")])
        renderer.upload_scene(scene)
        renderer.set_visible({"some_mesh"})     # meshes only, as the app passes
        self.assertEqual(renderer._index_count, 0)

        view = dict(
            width=240, height=180, target=(0.0, 0.0, 0.0),
            yaw=0.0, pitch=0.0, distance=0.4,
        )
        self.assertEqual(renderer.pick(120, 90, **view), scene.trigger_names[0])
        self.assertIsNone(renderer.pick(2, 2, **view), "empty space picked something")

    def test_a_hidden_box_stays_unclickable(self) -> None:
        # Picking follows what is on screen: the boxes are exempt from the mesh
        # filter but not from the triggers toggle itself.
        renderer = mesh_preview.create_renderer()
        if renderer is None:
            self.skipTest("no OpenGL context available")
        renderer.upload_scene(self._scene([self._box("hood_int", (0.0, 0.0, 0.0))]))
        renderer.set_triggers_visible(False)
        renderer.set_visible({"some_mesh"})
        self.assertEqual(renderer._trigger_index_count, 0)
        self.assertIsNone(renderer.pick(
            120, 90, width=240, height=180,
            target=(0.0, 0.0, 0.0), yaw=0.0, pitch=0.0, distance=0.4,
        ))

    def test_the_scene_name_is_the_table_row_id(self) -> None:
        # The preview and the Triggers table have to agree on one string, or a
        # picked trigger cannot find its row.
        from beamxp.hand_drive_ui.triggers_workflow import TRIGGER_ROW_PREFIX

        at = (0.691, -0.756, 0.513)
        row_id = f"{TRIGGER_ROW_PREFIX}|hood_int|{at[0]}|{at[1]}|{at[2]}"
        self.assertEqual(mesh_preview.trigger_scene_name("hood_int", list(at)), row_id)


if __name__ == "__main__":
    unittest.main()
