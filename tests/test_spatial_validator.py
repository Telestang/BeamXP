from __future__ import annotations

import unittest
from types import SimpleNamespace

from beamxp import hand_drive_core as core
from scripts.validate_spatial_classifier import (
    classifier_detection_methods,
    functionally_sided_skip_reasons,
    is_expected_functionally_sided_skip,
    is_expected_structural_fallback,
    markdown_report,
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


class FunctionallySidedSkipTests(unittest.TestCase):
    def test_material_safe_structural_skip_is_semantic_agreement(self) -> None:
        self.assertTrue(is_expected_functionally_sided_skip(
            core.MODE_MIRROR_STRUCTURAL,
            core.MODE_SKIP,
            (
                "functionally sided: materials differ, needs build-side "
                "material rebind"
            ),
        ))

    def test_unrelated_structural_skips_are_not_ignored(self) -> None:
        self.assertFalse(is_expected_functionally_sided_skip(
            core.MODE_MIRROR_STRUCTURAL,
            core.MODE_SKIP,
            "no confident geometric twin",
        ))
        self.assertFalse(is_expected_functionally_sided_skip(
            core.MODE_MIRROR,
            core.MODE_SKIP,
            "functionally sided: materials differ, needs build-side material rebind",
        ))
        self.assertFalse(is_expected_functionally_sided_skip(
            core.MODE_MIRROR_STRUCTURAL,
            core.MODE_MIRROR,
            "functionally sided: materials differ, needs build-side material rebind",
        ))

    def test_reasons_come_from_classifier_memo_because_skip_rows_are_not_emitted(
        self,
    ) -> None:
        context = SimpleNamespace(_spatial_recommendation_state={
            "memo": {
                "safe_skip": (
                    "functional_skip",
                    (
                        "functionally sided: materials differ, needs build-side "
                        "material rebind"
                    ),
                    "high",
                    {},
                ),
                "ordinary_skip": ("none", "not visible", "med", {}),
                "unrelated_internal_skip": (
                    "functional_skip", "another reason", "high", {}
                ),
            },
        })
        self.assertEqual(
            functionally_sided_skip_reasons(context),
            {
                "safe_skip": (
                    "functionally sided: materials differ, needs build-side "
                    "material rebind"
                ),
            },
        )


class DetectionMethodReportTests(unittest.TestCase):
    def test_detection_methods_come_from_classifier_memo(self) -> None:
        context = SimpleNamespace(_spatial_recommendation_state={
            "memo": {
                "rear_part": (
                    "mirror",
                    "one-sided interior part",
                    "med",
                    {"detection": "cabin enclosure shell"},
                ),
                "unclassified": ("none", "", "med", {}),
            },
        })
        self.assertEqual(
            classifier_detection_methods(context),
            {
                "rear_part": "cabin enclosure shell",
                "unclassified": "not admitted by spatial scope or vetoed",
            },
        )

    def test_markdown_includes_detection_method_column(self) -> None:
        report = markdown_report([{
            "vehicle": "pickup",
            "differences": 1,
            "checks": 1,
            "agreement_percent": 0.0,
            "unique_mismatches": 1,
            "ignored_structural_fallbacks": 0,
            "ignored_functionally_sided_skips": 0,
            "transitions": [{
                "baseline": core.MODE_SKIP,
                "recommended": core.MODE_MIRROR,
                "count": 1,
            }],
            "rows": [{
                "object_id": "rear_part",
                "baseline": core.MODE_SKIP,
                "recommended": core.MODE_MIRROR,
                "trim_count": 1,
                "trims": ["deserttruck_prerunner_A"],
                "detection_method": "cabin enclosure shell",
            }],
        }], show_trims=False)
        self.assertIn("| Part | Detection method | Affected trims | Trims |", report)
        self.assertIn(
            "| `rear_part` | cabin enclosure shell | 1 | "
            "`deserttruck_prerunner_A` |",
            report,
        )


if __name__ == "__main__":
    unittest.main()
