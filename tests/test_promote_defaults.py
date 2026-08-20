"""Promoting a tuned value into the shipped defaults.

Editing source from a GUI earns more suspicion than most changes, so the
behaviour worth pinning is what it refuses, what it leaves alone, and what it
does when the write goes wrong.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mesh_segmentation_transform.promote_detection_defaults import (
    EXCLUDED_FIELDS,
    PROMOTION_TARGETS,
    config_source_path,
    plan_promotion,
    promote_defaults,
    promotion_is_possible,
)

MODULE = '''"""A stand-in for the real defaults module."""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class MserConfig:
    alpha: float = 1.0
    beta: int = 2
    gamma: bool = False
    box_source: str = "contrast"


DEFAULT_COLOUR_CONFIG = replace(
    MserConfig(),
    # Measured on the ardente, and the reason is the point of this comment.
    alpha=1.5,
)
DEFAULT_RELIEF_DETECTION_CONFIG = replace(
    DEFAULT_COLOUR_CONFIG,
    beta=7,
)
'''


class PlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = TemporaryDirectory()
        self.path = Path(self._directory.name) / "annotate_texture_regions.py"
        self.path.write_text(MODULE, encoding="utf-8")
        self.addCleanup(self._directory.cleanup)

    def test_a_value_already_shipped_is_not_a_change(self) -> None:
        self.assertEqual(plan_promotion({"alpha": 1.5}, "colour", self.path), [])

    def test_an_existing_value_is_reported_with_what_it_replaces(self) -> None:
        (change,) = plan_promotion({"alpha": 2.5}, "colour", self.path)
        self.assertEqual((change.field, change.before, change.after), ("alpha", 1.5, 2.5))
        self.assertFalse(change.is_new)

    def test_a_value_not_yet_in_the_literal_is_reported_as_new(self) -> None:
        (change,) = plan_promotion({"beta": 9}, "colour", self.path)
        self.assertTrue(change.is_new)
        self.assertIsNone(change.before)

    def test_the_front_end_is_never_promoted(self) -> None:
        """Production picks it per layer, so promoting it would pin a build."""
        self.assertIn("box_source", EXCLUDED_FIELDS)
        self.assertEqual(
            plan_promotion({"box_source": "mser"}, "colour", self.path), []
        )

    def test_each_family_reads_its_own_literal(self) -> None:
        self.assertEqual(plan_promotion({"beta": 7}, "relief", self.path), [])
        self.assertEqual(len(plan_promotion({"beta": 7}, "colour", self.path)), 1)


class WriteTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = TemporaryDirectory()
        self.path = Path(self._directory.name) / "annotate_texture_regions.py"
        self.path.write_text(MODULE, encoding="utf-8")
        self.addCleanup(self._directory.cleanup)

    def _literal(self, name: str) -> dict[str, object]:
        tree = ast.parse(self.path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets
            ):
                return {
                    keyword.arg: ast.literal_eval(keyword.value)
                    for keyword in node.value.keywords  # type: ignore[union-attr]
                    if keyword.arg is not None
                }
        raise AssertionError(f"{name} missing")

    def test_an_existing_value_is_replaced_in_place(self) -> None:
        promote_defaults({"alpha": 2.5}, "colour", self.path)
        self.assertEqual(self._literal("DEFAULT_COLOUR_CONFIG")["alpha"], 2.5)

    def test_the_reason_written_beside_a_value_survives(self) -> None:
        """The blocks carry why each number is where it is.

        Regenerating them would be simpler and would throw all of that away,
        which is the whole reason this rewrites rather than regenerates.
        """
        promote_defaults({"alpha": 2.5}, "colour", self.path)
        self.assertIn(
            "# Measured on the ardente, and the reason is the point of this comment.",
            self.path.read_text(encoding="utf-8"),
        )

    def test_a_new_value_is_appended_with_its_provenance(self) -> None:
        promote_defaults({"beta": 9}, "colour", self.path)
        text = self.path.read_text(encoding="utf-8")
        self.assertEqual(self._literal("DEFAULT_COLOUR_CONFIG")["beta"], 9)
        self.assertIn("Promoted", text)
        self.assertIn("tuning harness", text)

    def test_promoting_relief_leaves_colour_alone(self) -> None:
        promote_defaults({"beta": 9}, "relief", self.path)
        self.assertEqual(self._literal("DEFAULT_RELIEF_DETECTION_CONFIG")["beta"], 9)
        self.assertNotIn("beta", self._literal("DEFAULT_COLOUR_CONFIG"))

    def test_the_result_still_parses(self) -> None:
        promote_defaults({"alpha": 2.5, "beta": 9, "gamma": True}, "colour", self.path)
        ast.parse(self.path.read_text(encoding="utf-8"))
        literal = self._literal("DEFAULT_COLOUR_CONFIG")
        self.assertEqual(literal["alpha"], 2.5)
        self.assertEqual(literal["beta"], 9)
        self.assertIs(literal["gamma"], True)

    def test_promoting_nothing_leaves_the_file_untouched(self) -> None:
        before = self.path.read_text(encoding="utf-8")
        self.assertEqual(promote_defaults({"alpha": 1.5}, "colour", self.path), [])
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)

    def test_types_survive_the_round_trip(self) -> None:
        """A bool written as an int would silently change a switch to a number."""
        promote_defaults({"gamma": True}, "colour", self.path)
        self.assertIs(self._literal("DEFAULT_COLOUR_CONFIG")["gamma"], True)


class RefusalTests(unittest.TestCase):
    def test_a_missing_module_is_refused_rather_than_created(self) -> None:
        with TemporaryDirectory() as directory:
            possible, reason = promotion_is_possible(Path(directory) / "absent.py")
            self.assertFalse(possible)
            self.assertIn("Cannot find", reason)

    def test_the_real_module_is_the_promotion_target(self) -> None:
        self.assertEqual(config_source_path().name, "annotate_texture_regions.py")
        self.assertTrue(config_source_path().is_file())

    def test_both_families_have_a_target(self) -> None:
        self.assertEqual(set(PROMOTION_TARGETS), {"colour", "relief"})


if __name__ == "__main__":
    unittest.main()
