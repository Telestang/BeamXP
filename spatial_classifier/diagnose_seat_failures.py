#!/usr/bin/env python3
"""For every seat that isn't detected correctly, report WHERE it drops out.

Runs candidate B's seat detector with tracing on and, for each trim where the
driver or passenger seat role is wrong, walks each expected (present, correct-
side) seat mesh through the pipeline and names the first gate it failed:

    few_points | diagonal_out | shape_no_bend | shape_low_bend | shape_thin_band
    position_out(<axis ...>) | backing_low | size_dissimilar | out_competed:<winner>

Results are deduped across trims: one line per (vehicle, side, mesh, failure)
with the trim count and an example. Seats-only, so it is quick.

    python diagnose_seat_failures.py                # all labelled vehicles
    python diagnose_seat_failures.py us_semi bx     # a subset
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

import assembly_scoring as scoring

SIDE_ROLE = {"driver": "driver_seat", "passenger": "passenger_seat"}


def _driver_side(entries, mesh, cx, side, tol=0.05):
    pts = entries.get(mesh)
    if pts is None or len(pts) == 0:
        return 0
    rel = (float(np.asarray(pts, dtype=float)[:, 0].mean()) - cx) * side
    return 1 if rel > tol else -1 if rel < -tol else 0


def _classify(trace: dict, mesh: str, side: str, sp) -> str:
    """First gate that rejected *mesh* on *side* ('driver'|'passenger')."""
    rec = trace.get("meshes", {}).get(mesh)
    if rec is None:
        return "not_reached"  # filtered before the seat loop even saw it
    stage = rec["stage"]
    if stage != "windows":
        return stage
    info = rec[side]
    if not info["in_window"]:
        rel = info["rel"]
        axes = []
        if abs(rel["lateral"]) > sp.lateral_tolerance:
            axes.append(f"lateral {rel['lateral']:+.2f} (max {sp.lateral_tolerance})")
        if not (sp.longitudinal_range[0] <= rel["ahead"] <= sp.longitudinal_range[1]):
            axes.append(f"ahead {rel['ahead']:+.2f} (range {sp.longitudinal_range})")
        if not (sp.vertical_range[0] <= rel["below"] <= sp.vertical_range[1]):
            axes.append(f"below {rel['below']:+.2f} (range {sp.vertical_range})")
        return "position_out: " + "; ".join(axes)
    winner = trace.get(f"{side}_winner")
    if winner == mesh:
        return "selected"  # correct on this mesh (should not appear for a failure)
    backing = info["backing"]
    if backing is not None and backing < trace["min_backing"]:
        return f"backing_low: {backing:.2f} < {trace['min_backing']}"
    if side == "passenger":
        ratio = rec["diag"] / max(trace.get("driver_diag", 0.9), 0.01)
        lo, hi = trace["size_similarity"]
        if not (lo <= ratio <= hi):
            return f"size_dissimilar: ratio {ratio:.2f} (range {trace['size_similarity']})"
    return f"out_competed_by: {winner}"


def _category(failure: str) -> str:
    """Group key: the failure kind without the trim-specific numbers."""
    return failure.split(":")[0].split(" (")[0].strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("vehicles", nargs="*", default=list(scoring.DEFAULT_VEHICLES))
    args = ap.parse_args()

    engine = scoring.load_engine()
    candidate = scoring.load_candidate("B")
    labels = scoring.load_labels()
    sp = candidate.AssemblyParams().seat

    # (vehicle, side, mesh, category) -> {"trims": [...], "example": failure str}
    agg: dict[tuple, dict] = defaultdict(lambda: {"trims": [], "example": ""})
    present_vehicles, missing = scoring.partition_by_cache(engine, dict.fromkeys(args.vehicles))

    for veh in present_vehicles:
        seats = set(labels.get(veh, {}).get("seats", ()))
        if not seats:
            continue
        for trial in scoring.iter_trials(engine, veh):
            entries = trial.entries_np
            cx = float(trial.frame.center_x)
            fside = int(trial.frame.side)
            present = set(trial.present)
            seat_present = [m for m in seats if m in present]
            if not seat_present:
                continue
            gt = {
                "driver": [m for m in seat_present if _driver_side(entries, m, cx, fside) != -1],
                "passenger": [m for m in seat_present if _driver_side(entries, m, cx, fside) != 1],
            }

            trace: dict = {}
            anchors = candidate.detect_assembly_anchors(
                points_by_id=entries, eye=trial.frame.eye, forward=trial.frame.forward,
                center_x=cx, side=fside, roles={"seats"}, seat_trace=trace,
            )
            detected = {"driver": anchors.driver_seat, "passenger": anchors.passenger_seat}

            for side, role in SIDE_ROLE.items():
                if not gt[side]:
                    continue  # role not applicable this trim
                det = detected[side].mesh_id if detected[side] else None
                bad_side = -1 if side == "driver" else 1
                correct = det in seats and _driver_side(entries, det, cx, fside) != bad_side
                if correct:
                    continue
                # role failed this trim -> where did each expected mesh drop out?
                for mesh in gt[side]:
                    failure = _classify(trace, mesh, side, sp)
                    if failure == "selected":
                        continue
                    cell = agg[(veh, side, mesh, _category(failure))]
                    cell["trims"].append(trial.trim or "(default)")
                    cell["example"] = failure

    # ---- report ----
    if not agg:
        print("no seat failures.")
        return 0
    by_vs = defaultdict(list)
    for (veh, side, mesh, cat), cell in agg.items():
        by_vs[(veh, side, mesh)].append((cat, cell))
    for (veh, side, mesh) in sorted(by_vs):
        print(f"# {veh}  {SIDE_ROLE[side]}: {mesh}")
        for cat, cell in sorted(by_vs[(veh, side, mesh)], key=lambda t: -len(t[1]["trims"])):
            trims = cell["trims"]
            ex = ", ".join(sorted(trims)[:4]) + (" ..." if len(trims) > 4 else "")
            print(f"    {cell['example']}")
            print(f"        in {len(trims)} trim(s): {ex}")
        print()
    if missing:
        print(f"(skipped, no cache: {', '.join(missing)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
