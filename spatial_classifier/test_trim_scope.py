from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import assembly_scoring as scoring


class _FakeEngine:
    def __init__(self, projects_dir: Path):
        self.PROJECTS_DIR = projects_dir

    @staticmethod
    def prepare_vehicle_context(context) -> None:
        pass

    @staticmethod
    def driver_frame_for_context(context, config_name=None):
        return SimpleNamespace(
            eye=(0.0, 0.0, 1.0),
            forward=(0.0, -1.0, 0.0),
            center_x=0.0,
            side=1,
        )

    @staticmethod
    def authoritative_used_meshes_for_config(context, trim):
        # The chosen tree contains only the real driver seat.  The dormant
        # passenger racing seat is merely a leftover .pc entry.
        return {"miramar_seat_FL"}

    @staticmethod
    def used_meshes_for_config(context, trim):
        # Deliberately stale implementation, matching the bug this guard fixes.
        return {"miramar_seat_FL", "racing_seat_FR"}

    @staticmethod
    def _spatial_entries_for_trim(context, trim, available):
        points = {
            "miramar_seat_FL": np.array([[0.4, 0.0, 0.0]]),
            "racing_seat_FR": np.array([[0.0, 0.0, 0.0]]),
        }
        return sorted(points), points


class TrimScopeTests(unittest.TestCase):
    def test_iter_trials_drops_orphan_pc_mesh_before_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            projects = Path(temp)
            vehicle_dir = projects / "miramar"
            vehicle_dir.mkdir()
            context = SimpleNamespace(
                objects={
                    "miramar_seat_FL": object(),
                    "racing_seat_FR": object(),
                },
                preview_by_id={
                    "miramar_seat_FL": {},
                    "racing_seat_FR": {},
                },
                variants={"problem_trim": object()},
            )
            with (vehicle_dir / "context.cache").open("wb") as handle:
                pickle.dump({"context": context}, handle)

            trials = list(scoring.iter_trials(_FakeEngine(projects), "miramar"))

        self.assertEqual(len(trials), 1)
        self.assertEqual(trials[0].present, ["miramar_seat_FL"])
        self.assertEqual(set(trials[0].entries_np), {"miramar_seat_FL"})

    def test_absent_labelled_passenger_seat_is_not_applicable(self) -> None:
        scorer = scoring.RoleScorer(
            {
                "miramar": {
                    "seats": ["miramar_seat_FL", "racing_seat_FR"],
                    "dashboard": [],
                    "door_cards": [],
                }
            },
            roles={"seats"},
        )
        trial = scoring.Trial(
            vehicle="miramar",
            trim="problem_trim",
            context=None,
            frame=SimpleNamespace(center_x=0.0, side=1),
            present=["miramar_seat_FL"],
            entries_np={
                "miramar_seat_FL": np.array([[0.4, 0.0, 0.0]]),
            },
        )
        scorer.score_trim(
            trial,
            {
                "driver_seat": "miramar_seat_FL",
                "passenger_seat": None,
            },
        )

        self.assertEqual(
            scorer._counts[("miramar", "driver_seat")],
            [1, 1],
        )
        self.assertNotIn(("miramar", "passenger_seat"), scorer._counts)


if __name__ == "__main__":
    unittest.main()
