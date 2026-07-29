#!/usr/bin/env python3
"""Explain dashboard detector failures across vehicles and trims.

For every trim where candidate B fails the dashboard role, walk each labelled
 dashboard mesh actually present in that trim through the detector and report
 the first gate it failed. The traced gates are:

    excluded | few_points | diagonal_out | lateral_extent_low
    width_fraction_low | depth_out | vertical_extent_low
    position_ahead_out | position_below_out | top_above_eye
    nearest_too_far | complex_no_bands | complex_front_width_low
    backing_low | out_competed_by:<winner>

Results are deduplicated across trims by vehicle, expected mesh and failure
category. Candidate rows include backing, fronting and exposed angular-footprint
measurements so hidden panels can be separated from sparse exposed structures.

Usage:
    python diagnose_dashboard_failures.py
    python diagnose_dashboard_failures.py miramar bx us_semi
    python diagnose_dashboard_failures.py --all
    python diagnose_dashboard_failures.py --summary-only
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict

import assembly_scoring as scoring


def _fmt(value: object, digits: int = 3) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _winner_text(trace: dict) -> str:
    winner = trace.get("winner")
    if not winner:
        return "(none)"
    kind = trace.get("winner_kind") or "?"
    score = trace.get("winner_score")
    backing = trace.get("winner_backing")
    fronting = trace.get("winner_fronting")
    return (
        f"{winner} [{kind}, score {_fmt(score)}, backing {_fmt(backing)}, "
        f"fronting {_fmt(fronting)}]"
    )


def _classify(trace: dict, mesh: str, dp) -> str:
    """First gate that rejected one expected dashboard mesh."""
    rec = trace.get("meshes", {}).get(mesh)
    if rec is None:
        return "not_reached"

    stage = rec.get("stage", "not_reached")
    if stage == "selected":
        return "selected"
    if stage == "excluded":
        return "excluded"
    if stage == "few_points":
        return f"few_points: {rec.get('point_count', 0)} < 30"
    if stage == "diagonal_out":
        return (
            f"diagonal_out: {_fmt(rec.get('diag'))} "
            f"not in {dp.diagonal_range}"
        )
    if stage == "lateral_extent_low":
        return (
            f"lateral_extent_low: {_fmt(rec['extents'][0])} "
            f"< {dp.min_lateral_extent}"
        )
    if stage == "width_fraction_low":
        return (
            f"width_fraction_low: {_fmt(rec.get('width_fraction'))} "
            f"< {dp.min_width_fraction_of_cabin} "
            f"(cabin half-width {_fmt(trace.get('cabin_halfwidth'))})"
        )
    if stage == "depth_out":
        return (
            f"depth_out: {_fmt(rec.get('depth'))} > {dp.max_complex_depth} "
            f"(simple max {dp.max_depth})"
        )
    if stage == "vertical_extent_low":
        return (
            f"vertical_extent_low: {_fmt(rec['extents'][2])} "
            f"< {dp.min_vertical_extent}"
        )
    if stage == "position_ahead_out":
        return (
            f"position_ahead_out: {_fmt(rec['rel']['ahead'])} "
            f"not in {dp.forward_range}"
        )
    if stage == "position_below_out":
        return (
            f"position_below_out: {_fmt(rec['rel']['below'])} "
            f"not in {dp.centroid_z_below_eye_range}"
        )
    if stage == "top_above_eye":
        return (
            f"top_above_eye: {_fmt(rec.get('top_above_eye'))} "
            f"> {dp.top_z_max_above_eye}"
        )
    if stage == "nearest_too_far":
        return (
            f"nearest_too_far: {_fmt(rec.get('nearest'))} "
            f"> {dp.max_nearest_distance}"
        )
    if stage == "complex_no_bands":
        return "complex_no_bands: front/tail split produced an empty band"
    if stage == "complex_front_width_low":
        return (
            f"complex_front_width_low: width {_fmt(rec.get('front_width'))} "
            f"(min {dp.min_lateral_extent}), fraction "
            f"{_fmt(rec.get('front_width_fraction'))} "
            f"(min {dp.min_width_fraction_of_cabin})"
        )
    if stage == "complex_tail_too_wide":
        return (
            f"complex_tail_too_wide: {_fmt(rec.get('tail_width_fraction'))} "
            f"> {dp.max_console_tail_width_fraction_of_cabin}"
        )
    if stage == "complex_tail_offcentre":
        return (
            f"complex_tail_offcentre: |{_fmt(rec.get('tail_mid'))}| "
            f"> {_fmt(rec.get('max_tail_offset'))}"
        )
    if stage == "complex_tail_not_rearward":
        return (
            f"complex_tail_not_rearward: rear ahead {_fmt(rec.get('rear_ahead'))} "
            f"> {dp.console_rear_ahead_max}"
        )
    if stage == "backing_low":
        return (
            f"backing_low: {_fmt(rec.get('backing'))} "
            f"< {dp.min_backing}; score {_fmt(rec.get('score'))}; "
            f"fronting {_fmt(rec.get('fronting'))}"
        )
    if stage in {"out_competed", "candidate_simple", "candidate_complex"}:
        return (
            f"out_competed_by: {_winner_text(trace)}; expected "
            f"[{rec.get('candidate_kind')}, score {_fmt(rec.get('score'))}, "
            f"backing {'not evaluated' if rec.get('backing') is None else _fmt(rec.get('backing'))}, "
            f"fronting {'not evaluated' if rec.get('fronting') is None else _fmt(rec.get('fronting'))}]"
        )
    return str(stage)


def _category(failure: str) -> str:
    return failure.split(":", 1)[0].strip()


def _candidate_snapshot(trace: dict, limit: int = 5) -> str:
    rows = []
    ranked = trace.get("ranked_order")
    if not ranked:
        ranked = [
            {"mesh": item["mesh"], "score": item["score"], "kind": kind}
            for kind in ("simple", "complex")
            for item in trace.get(f"{kind}_order", ())
        ]

    for item in ranked:
        rec = trace.get("meshes", {}).get(item["mesh"], {})
        backing = rec.get("backing")
        fronting = rec.get("fronting")
        exposed = rec.get("exposed_bin_count")
        occupied = rec.get("occupied_bin_count")
        span = rec.get("horizontal_span_degrees")
        fill = rec.get("angular_fill_ratio")
        visible = (
            "?/?" if exposed is None or occupied is None
            else f"{exposed}/{occupied}"
        )
        rows.append(
            f"{item['mesh']}[{item.get('kind', '?')} score={_fmt(item['score'])} "
            f"backing={'?' if backing is None else _fmt(backing)} "
            f"fronting={'?' if fronting is None else _fmt(fronting)} "
            f"visible={visible} "
            f"span={'?' if span is None else _fmt(span, 1)}deg "
            f"fill={'?' if fill is None else _fmt(fill)}]"
        )
    return " | ".join(rows[:limit]) or "(no candidates reached ranking)"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("vehicles", nargs="*", default=list(scoring.DEFAULT_VEHICLES))
    ap.add_argument(
        "--all", action="store_true",
        help="also list successful expected dashboards, not only failed trims",
    )
    ap.add_argument(
        "--summary-only", action="store_true",
        help="print only failure-category totals",
    )
    ap.add_argument(
        "--candidate-limit", type=int, default=5,
        help="maximum ranked candidates shown in each failure example (default: 5)",
    )
    args = ap.parse_args()

    engine = scoring.load_engine()
    candidate = scoring.load_candidate("B")
    labels = scoring.load_labels()
    dp = candidate.AssemblyParams().dashboard

    # (vehicle, expected mesh, category, detected mesh) -> aggregate cell
    agg: dict[tuple, dict] = defaultdict(
        lambda: {"trims": [], "example": "", "candidates": ""}
    )
    categories: Counter[str] = Counter()
    successful = 0
    applicable = 0

    present_vehicles, missing = scoring.partition_by_cache(
        engine, dict.fromkeys(args.vehicles)
    )

    for vehicle in present_vehicles:
        dashboards = set(labels.get(vehicle, {}).get("dashboard", ()))
        if not dashboards:
            continue
        for trial in scoring.iter_trials(engine, vehicle):
            expected = sorted(dashboards.intersection(trial.present))
            if not expected:
                continue
            applicable += 1

            trace: dict = {}
            anchors = candidate.detect_assembly_anchors(
                points_by_id=trial.entries_np,
                eye=trial.frame.eye,
                forward=trial.frame.forward,
                center_x=float(trial.frame.center_x),
                side=int(trial.frame.side),
                roles={"dashboard"},
                dashboard_trace=trace,
            )
            detected = anchors.dashboard.mesh_id if anchors.dashboard else None
            correct = detected in dashboards
            if correct:
                successful += 1
                if not args.all:
                    continue

            for mesh in expected:
                failure = "selected" if correct and mesh == detected else _classify(trace, mesh, dp)
                if correct and failure != "selected":
                    # Multiple labelled dashboard meshes can coexist. The scorer
                    # accepts any one of them; do not call the unselected labelled
                    # companion a detector failure.
                    failure = f"trim_correct_via: {detected}"
                category = _category(failure)
                if not correct:
                    categories[category] += 1
                key = (vehicle, mesh, category, detected or "(none)")
                cell = agg[key]
                cell["trims"].append(trial.trim or "(default)")
                cell["example"] = failure
                cell["candidates"] = _candidate_snapshot(trace, args.candidate_limit)

    if not args.summary_only:
        if not agg:
            print("no dashboard failures.")
        else:
            by_vm = defaultdict(list)
            for (vehicle, mesh, category, detected), cell in agg.items():
                if not args.all and category in {"selected", "trim_correct_via"}:
                    continue
                by_vm[(vehicle, mesh)].append((category, detected, cell))

            if not by_vm:
                print("no dashboard failures.")
            for (vehicle, mesh) in sorted(by_vm):
                print(f"# {vehicle}  dashboard: {mesh}")
                rows = sorted(
                    by_vm[(vehicle, mesh)],
                    key=lambda item: (-len(item[2]["trims"]), item[0], item[1]),
                )
                for category, detected, cell in rows:
                    trims = cell["trims"]
                    examples = ", ".join(sorted(trims)[:4])
                    if len(trims) > 4:
                        examples += " ..."
                    print(f"    {cell['example']}")
                    print(f"        detected: {detected}")
                    print(f"        candidates: {cell['candidates']}")
                    print(f"        in {len(trims)} trim(s): {examples}")
                print()

    print("=== dashboard diagnostic summary ===")
    print(f"applicable trims: {applicable}")
    print(f"correct trims:    {successful}")
    print(f"failed trims:     {applicable - successful}")
    if categories:
        print("failure records by first gate:")
        for category, count in categories.most_common():
            print(f"  {category:<30} {count}")
    else:
        print("no failed dashboard records.")
    if missing:
        print(f"(skipped, no cache: {', '.join(missing)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
