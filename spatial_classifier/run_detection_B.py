#!/usr/bin/env python3
"""Score candidate B (classifier_standalone_B.py) per anchor role.

B returns one best anchor per slot (or None); we report, per role, how many of
the three vehicles it got right.

    python run_detection_B.py                 # all labelled vehicles, all roles
    python run_detection_B.py pickup          # a single vehicle
    python run_detection_B.py --roles seats   # only run/score the seat detector
    python run_detection_B.py --roles seats,doors   # a couple of roles
"""

from __future__ import annotations

import argparse
from pathlib import Path

import assembly_scoring as scoring


def _role_set(value: str) -> set[str]:
    """Parse/validate a --roles value like "seats,doors" into a group set."""
    roles = {r.strip() for r in value.split(",") if r.strip()}
    unknown = roles - set(scoring.ALL_ROLE_GROUPS)
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown role(s) {sorted(unknown)}; "
            f"choose from {', '.join(scoring.ALL_ROLE_GROUPS)}"
        )
    return roles


def _mesh_id(anchor) -> str | None:
    return anchor.mesh_id if anchor is not None else None


def detections_for(anchors) -> dict:
    doors = [_mesh_id(anchors.left_door_card), _mesh_id(anchors.right_door_card)]
    return {
        "driver_seat": _mesh_id(anchors.driver_seat),
        "passenger_seat": _mesh_id(anchors.passenger_seat),
        "dashboard": _mesh_id(anchors.dashboard),
        "door_cards": [mesh for mesh in doors if mesh],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("vehicles", nargs="*", default=list(scoring.DEFAULT_VEHICLES))
    ap.add_argument("--no-detail", dest="detail", action="store_false",
                    help="skip the per-trim detected-vs-expected dump")
    ap.add_argument("--out", type=Path, default=scoring.HERE / "detection_detail_B.txt",
                    help="file to write the per-trim detail to")
    ap.add_argument("--roles", type=_role_set, default=None,
                    help="comma-separated detector groups to run/score: "
                         f"{', '.join(scoring.ALL_ROLE_GROUPS)} (default: all). "
                         "Skips the others' search and columns entirely.")
    args = ap.parse_args()

    engine = scoring.load_engine()
    candidate = scoring.load_candidate("B")
    scorer = scoring.RoleScorer(scoring.load_labels(), roles=args.roles)

    def run_trial(trial):
        anchors = candidate.detect_assembly_anchors(
            points_by_id=trial.entries_np,
            eye=trial.frame.eye,
            forward=trial.frame.forward,
            center_x=trial.frame.center_x,
            side=trial.frame.side,
            roles=args.roles,
        )
        scorer.score_trim(trial, detections_for(anchors))

    return scoring.execute(
        "Candidate B — assembly role detection",
        args.vehicles, engine, scorer, run_trial,
        detail=args.detail, detail_path=args.out,
    )


if __name__ == "__main__":
    raise SystemExit(main())
