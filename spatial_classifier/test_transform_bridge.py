from __future__ import annotations

import unittest

import assembly_scoring as scoring


class TransformBridgeRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = scoring.load_engine()

    def test_node_transform_accepts_variable_scope_and_merges_components(self) -> None:
        ops = self.engine.node_transform_ops(
            (
                '{"nodeMove":{"x":1.25,"y":-0.5}}',
                '{"nodeMove":{"z":"$=$lift+0.25"}}',
            ),
            {"$lift": 2.0},
        )
        move = ops[("nodeMove", 0)]
        self.assertAlmostEqual(move["x"], 1.25)
        self.assertAlmostEqual(move["y"], -0.5)
        self.assertAlmostEqual(move["z"], 2.25)

    def test_object_number_property_keeps_old_two_argument_call(self) -> None:
        self.assertEqual(
            self.engine.object_number_property('{"x":1.5}', "x"),
            1.5,
        )


if __name__ == "__main__":
    unittest.main()
