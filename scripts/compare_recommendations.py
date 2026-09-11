"""2026-09-10: Read-only comparison with hand-authored project conversions.

Pass project directory names explicitly; writes only the requested JSON report.
Cached contexts are local trusted pickles, never downloaded inputs.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from dataclasses import MISSING, fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beamxp import hand_drive_core as core
from beamxp.hand_drive_ui.recommendation_engine import (
    build_mode_recommendations,
)
from beamxp.hand_drive_ui.triggers_workflow import TriggersWorkflowMixin


def load_project(path):
    data = json.loads((path / "conversion.json").read_text(encoding="utf-8"))
    with (path / "context.cache").open("rb") as stream:
        context = pickle.load(stream)["context"]
    for field in fields(context):
        if not hasattr(context, field.name) and field.default_factory is not MISSING:
            setattr(context, field.name, field.default_factory())
    configs = tuple(sorted(k for k, v in data["variants"].items() if v.get("selected")))
    # 2026-09-10: Resolve current selected trims; inventory/UI visibility is not a label.
    used = set().union(*(core.used_meshes_for_config(context, name) for name in configs))
    boxes = TriggersWorkflowMixin._collect_trigger_boxes(context, configs)
    return context, data, configs, used, boxes


def compare(path):
    context, data, configs, used, boxes = load_project(path)
    start = time.perf_counter()
    rows = build_mode_recommendations(context, sorted(used), configs, boxes)
    seconds = time.perf_counter() - start
    warm_start = time.perf_counter()
    assert rows == build_mode_recommendations(context, sorted(used), configs, boxes)
    warm_seconds = time.perf_counter() - warm_start
    modes, textures, offsets, sources = {}, set(), {}, {}
    for row in rows:
        mesh = row["object_id"]
        if row["kind"] == "texture":
            textures.add(mesh)
        elif row["kind"] in {"single", "pair"}:
            modes[mesh] = row["mode"]
            offsets[mesh] = row.get("offset")
            if row.get("source_id"):
                twin = row["source_id"]
                modes[twin] = row["mode"]
                sources[mesh], sources[twin] = twin, mesh
    mesh_rows = []
    for mesh in sorted(used):
        expected = data["parts"].get(mesh, {})
        mesh_rows.append({
            "mesh": mesh, "expected": expected.get("mode", core.MODE_SKIP),
            "actual": modes.get(mesh, core.MODE_SKIP),
            "texture_expected": bool(expected.get("textureCorrection")),
            "texture_actual": mesh in textures,
            "offset_expected": expected.get("translateOffset"), "offset_actual": offsets.get(mesh),
            "effective_offset_expected": core.part_translate_magnitude(context, data, mesh, expected.get("mode", core.MODE_SKIP)),
            "source_expected": expected.get("mirrorSource"), "source_actual": sources.get(mesh),
        })
    predicted = {(r["trigger_key"][0], tuple(r["trigger_key"][1])): r
                 for r in rows if r["kind"] == "trigger"}
    authored = core.trigger_mode_map(data)
    trigger_rows = []
    for box in boxes:
        key = box["key"]
        if key not in authored:
            continue
        row = predicted.get(key, {})
        trigger_rows.append({"key": key, "expected": authored[key],
                             "actual": row.get("mode", core.MODE_SKIP),
                             "host": row.get("host_id"), "reason": row.get("reason")})
    return {"project": path.name, "seconds": seconds, "warm_seconds": warm_seconds, "meshes": mesh_rows,
            "triggers": trigger_rows, "recommendations": rows,
            "unmatched_authored_triggers": [key for key in authored if key not in {b["key"] for b in boxes}]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("projects", nargs="+")
    parser.add_argument("--root", type=Path, default=Path(os.environ["LOCALAPPDATA"]) / "BeamXP/handedness_conversion_projects")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    reports = []
    for project in args.projects:
        report = compare(args.root / project)
        reports.append(report)
        mismatch = sum(r["expected"] != r["actual"] for r in report["meshes"])
        print(f"{project}: {mismatch} mesh differences; {report['seconds']:.3f}s", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(reports, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
