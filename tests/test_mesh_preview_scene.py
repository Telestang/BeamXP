from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

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


if __name__ == "__main__":
    unittest.main()
