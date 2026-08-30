"""The broad candidate measure behind the material-transition probe."""

from __future__ import annotations

import unittest

import numpy as np

from glyph_lab.material_transitions import (
    MaterialTransitionCandidate,
    box_gap,
    classify_material_transition_candidates,
    is_material_transition_candidate,
    material_transition_features,
    texels_within_gap,
)
from glyph_lab.regions import Region


class MaterialTransitionProbeTests(unittest.TestCase):
    @staticmethod
    def candidate(
        bounds: tuple[int, int, int, int], mask: np.ndarray, *, detail: float = 0.0,
        eligible: bool = False,
    ) -> MaterialTransitionCandidate:
        region = Region(bounds=bounds, mask=mask, domain=1)
        return MaterialTransitionCandidate(
            region=region, body_void_texels=0, body_void_share=0.0,
            interior_detail=detail, enclosed_texels=region.enclosed,
            rule_eligible=eligible, loop_eligible=True,
        )

    def test_the_prefilter_is_large_and_sparse_but_not_a_decision(self) -> None:
        region = Region(
            bounds=(10, 20, 120, 90),
            mask=np.pad(np.ones((12, 90), dtype=bool), ((0, 78), (0, 30))),
            domain=1,
        )
        self.assertTrue(is_material_transition_candidate(region, 64, 0.30))
        self.assertFalse(is_material_transition_candidate(region, 128, 0.30))
        self.assertFalse(is_material_transition_candidate(region, 64, 0.05))

    def test_features_include_open_and_closed_shape_evidence(self) -> None:
        mask = np.ones((10, 10), dtype=bool)
        mask[2:8, 2:8] = False
        region = Region(bounds=(0, 0, 10, 10), mask=mask, domain=1)
        facts = material_transition_features(MaterialTransitionCandidate(
            region=region, body_void_texels=36, body_void_share=36 / 64,
            interior_detail=0.0, enclosed_texels=36, loop_eligible=True,
        ))
        self.assertGreater(facts["enclosed_texels"], 0)
        self.assertEqual(facts["holes"], 1)

    def test_long_trace_and_quiet_compact_loop_do_not_need_the_broad_gate(self) -> None:
        trace = self.candidate((0, 0, 60, 10), np.ones((10, 60), dtype=bool))
        loop_mask = np.ones((30, 30), dtype=bool)
        loop_mask[5:25, 5:25] = False
        loop = self.candidate((100, 0, 30, 30), loop_mask)
        classify_material_transition_candidates(
            [trace, loop], 100, 0.03, 6.0, 4.5, 10, 3.5, 6_000, 3,
        )
        self.assertTrue(trace.discarded)
        self.assertTrue(loop.discarded)

    def test_short_dense_trace_is_discarded_below_the_broad_gate(self) -> None:
        trace = self.candidate((0, 0, 34, 9), np.ones((9, 34), dtype=bool))
        classify_material_transition_candidates(
            [trace], 512, 0.03, 6.0, 4.5, 10, 3.5, 6_000, 3,
        )
        self.assertTrue(trace.discarded)

    def test_context_uses_mark_gap_not_the_overlapping_bounds(self) -> None:
        frame_mask = np.zeros((20, 20), dtype=bool)
        frame_mask[[0, -1], :] = True
        frame_mask[:, [0, -1]] = True
        frame = Region(bounds=(0, 0, 20, 20), mask=frame_mask, domain=1)
        mark = Region(bounds=(7, 7, 4, 4), mask=np.ones((4, 4), dtype=bool), domain=1)
        self.assertEqual(box_gap(frame, mark), 0)
        self.assertFalse(texels_within_gap(frame, mark, 3))


if __name__ == "__main__":
    unittest.main()
