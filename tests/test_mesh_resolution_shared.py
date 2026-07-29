from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from beamxp.core import mesh_resolution


def load_classifier_standalone_module():
    path = Path(__file__).resolve().parents[1] / "spatial classifier" / "classifier_standalone.py"
    spec = importlib.util.spec_from_file_location("classifier_standalone", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SharedMeshResolutionTests(unittest.TestCase):
    def test_selected_part_instances_use_part_order_and_options(self) -> None:
        selected = {
            "parts": {"part_b", "part_a"},
            "parts_order": ["part_a", "part_b"],
            "part_slot_options": {"part_b": ("opt1", "opt2")},
        }

        instances = mesh_resolution.selected_part_instances(selected)

        self.assertEqual([instance["part_id"] for instance in instances], ["part_a", "part_b"])
        self.assertEqual(mesh_resolution.part_instance_options(instances[1]), ("opt1", "opt2"))

    def test_part_instance_variable_scope_prefers_instance_scope(self) -> None:
        selected = {
            "part_variables": {"part_a": {"$x": 1.0}},
            "part_instance_variables": {
                "instance:part_b": {"$x": 2.0},
            },
        }
        instance = {"instance_id": "instance:part_b", "part_id": "part_b"}

        scope = mesh_resolution.part_instance_variable_scope(selected, instance)

        self.assertEqual(scope["$x"], 2.0)

    def test_classifier_standalone_exports_shared_mesh_resolution_wrappers(self) -> None:
        module = load_classifier_standalone_module()

        selected = {
            "parts": {"part_b", "part_a"},
            "parts_order": ["part_a", "part_b"],
            "part_slot_options": {"part_b": ("opt1",)}
        }
        instances = module.selected_part_instances(selected)

        self.assertEqual([instance["part_id"] for instance in instances], ["part_a", "part_b"])
        self.assertEqual(module.part_instance_options(instances[1]), ("opt1",))
