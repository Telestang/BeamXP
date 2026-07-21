from __future__ import annotations

import unittest

import beamng_hand_drive_core as core
from scripts.validate_spatial_classifier import (
    is_expected_structural_fallback,
    recommendation_modes_for_trim,
)


class StructuralFallbackTests(unittest.TestCase):
    def test_twin_absent_fallback_is_semantic_agreement_when_pair_exists(self) -> None:
        self.assertTrue(is_expected_structural_fallback(
            core.MODE_MIRROR_STRUCTURAL,
            core.MODE_MIRROR,
            True,
        ))

    def test_other_structural_to_mirror_changes_are_not_ignored(self) -> None:
        self.assertFalse(is_expected_structural_fallback(
            core.MODE_MIRROR_STRUCTURAL,
            core.MODE_MIRROR,
            False,
        ))
        self.assertFalse(is_expected_structural_fallback(
            core.MODE_SKIP,
            core.MODE_MIRROR,
            True,
        ))

    def test_global_pair_is_structural_or_falls_back_by_trim(self) -> None:
        recommendations = [{
            "kind": "pair",
            "object_id": "left",
            "source_id": "right",
            "mode": core.MODE_MIRROR_STRUCTURAL,
            "reason": "geometric twin",
        }]
        paired = recommendation_modes_for_trim(recommendations, {"left", "right"})
        self.assertEqual(paired["left"][0], core.MODE_MIRROR_STRUCTURAL)
        self.assertEqual(paired["right"][0], core.MODE_MIRROR_STRUCTURAL)
        self.assertFalse(paired["left"][2])

        lone = recommendation_modes_for_trim(recommendations, {"left"})
        self.assertEqual(lone["left"][0], core.MODE_MIRROR)
        self.assertIn("twin absent", lone["left"][1])
        self.assertTrue(lone["left"][2])


if __name__ == "__main__":
    unittest.main()
