"""Diff spatial Recommend Modes against saved per-project baselines.

Run from anywhere after the three validation projects have built context.cache::

    python scripts/validate_spatial_classifier.py
    python scripts/validate_spatial_classifier.py pickup --show-trims
    python scripts/validate_spatial_classifier.py --format json --output report.json

Pairs count both members, matching the conversion UI and build behaviour.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import pickle
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import beamng_hand_drive_core as core  # noqa: E402
from beamng_hand_drive_tool import build_mode_recommendations  # noqa: E402


DEFAULT_VEHICLES = ("etk800", "pickup", "sunburst2")
CLASSIFIER_CACHE_ATTRS = (
    "_spatial_recommendation_state",
    "_mesh_symmetry_cache",
    "_full_clouds",
    "_full_cloud_parsed_files",
)


def reset_classifier_caches(context: core.VehicleContext) -> None:
    """Discard recommendation state that must not leak between validations."""
    for attr in CLASSIFIER_CACHE_ATTRS:
        if hasattr(context, attr):
            delattr(context, attr)


def recommendation_modes_for_trim(
    recommendations: list[dict[str, str]],
    used_meshes: set[str],
) -> dict[str, tuple[str, str, bool]]:
    """Apply global recommendations to one trim.

    The recommender is intentionally a global batch classifier. Structural
    pairs remain structural when both members are fitted, while a lone member
    uses the build's desired aesthetic-Mirror fallback. The boolean marks that
    semantic fallback so baseline comparison can treat it as agreement.
    """
    modes = {
        object_id: (core.MODE_SKIP, "", False)
        for object_id in used_meshes
    }
    for row in recommendations:
        object_id = row["object_id"]
        source_id = row.get("source_id", "")
        reason = row.get("reason", "")
        if row.get("kind") == "pair" and source_id:
            present = [member for member in (object_id, source_id) if member in used_meshes]
            if len(present) == 2:
                verdict = (core.MODE_MIRROR_STRUCTURAL, reason, False)
                modes[object_id] = verdict
                modes[source_id] = verdict
            elif len(present) == 1:
                modes[present[0]] = (
                    core.MODE_MIRROR,
                    f"{reason}; twin absent in this trim",
                    True,
                )
            continue
        if object_id in used_meshes:
            modes[object_id] = (row["mode"], reason, False)
    return modes


def is_expected_structural_fallback(
    baseline: str,
    recommended: str,
    structural_fallback: bool,
) -> bool:
    """Whether a structural baseline correctly fell back for a missing twin."""
    return (
        baseline == core.MODE_MIRROR_STRUCTURAL
        and recommended == core.MODE_MIRROR
        and structural_fallback
    )


def validate_vehicle(
    projects_root: str,
    vehicle: str,
) -> dict[str, Any]:
    project_dir = Path(projects_root) / vehicle
    conversion_path = project_dir / "conversion.json"
    context_path = project_dir / "context.cache"
    if not conversion_path.is_file():
        raise FileNotFoundError(f"missing baseline: {conversion_path}")
    if not context_path.is_file():
        raise FileNotFoundError(f"missing context cache: {context_path}")

    with conversion_path.open("r", encoding="utf-8") as handle:
        conversion = json.load(handle)
    with context_path.open("rb") as handle:
        payload = pickle.load(handle)
    context = payload.get("context") if isinstance(payload, dict) else None
    if not isinstance(context, core.VehicleContext):
        raise ValueError(f"invalid context payload: {context_path}")
    reset_classifier_caches(context)

    baseline_parts = conversion.get("parts") or {}
    mismatch_rows: Counter[tuple[str, str, str]] = Counter()
    mismatch_trims: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    transition_counts: Counter[tuple[str, str]] = Counter()
    observations: list[tuple[str, str, str, str, bool]] = []
    checks = 0

    used_by_trim = {
        trim: (
            core.used_meshes_for_config(context, trim)
            & set(context.objects)
            & set(context.preview_by_id)
        )
        for trim in sorted(context.variants)
    }
    all_used = set().union(*used_by_trim.values()) if used_by_trim else set()
    recommendations = build_mode_recommendations(context, sorted(all_used))

    for trim, used in used_by_trim.items():
        actual_verdicts = recommendation_modes_for_trim(recommendations, used)
        for object_id in sorted(used):
            expected = baseline_parts.get(object_id, {}).get("mode", core.MODE_SKIP)
            actual, _reason, structural_fallback = actual_verdicts[object_id]
            checks += 1
            observations.append(
                (trim, object_id, expected, actual, structural_fallback)
            )

    ignored_structural_fallbacks = 0
    for trim, object_id, expected, actual, structural_fallback in observations:
        if expected == actual:
            continue
        if is_expected_structural_fallback(
            expected,
            actual,
            structural_fallback,
        ):
            ignored_structural_fallbacks += 1
            continue
        key = (object_id, expected, actual)
        mismatch_rows[key] += 1
        mismatch_trims[key].append(trim)
        transition_counts[(expected, actual)] += 1

    rows = [
        {
            "object_id": object_id,
            "baseline": baseline,
            "recommended": recommended,
            "trim_count": mismatch_rows[(object_id, baseline, recommended)],
            "trims": mismatch_trims[(object_id, baseline, recommended)],
        }
        for object_id, baseline, recommended in sorted(
            mismatch_rows,
            key=lambda key: (key[1], key[2], key[0].lower()),
        )
    ]
    differences = sum(mismatch_rows.values())
    return {
        "vehicle": vehicle,
        "trim_count": len(context.variants),
        "checks": checks,
        "differences": differences,
        "agreement_percent": 100.0 * (checks - differences) / checks if checks else 100.0,
        "unique_mismatches": len(rows),
        "ignored_structural_fallbacks": ignored_structural_fallbacks,
        "transitions": [
            {
                "baseline": baseline,
                "recommended": recommended,
                "count": count,
            }
            for (baseline, recommended), count in sorted(transition_counts.items())
        ],
        "rows": rows,
    }


def markdown_report(results: list[dict[str, Any]], show_trims: bool) -> str:
    lines = ["# Spatial classifier mismatch report", ""]
    for result in results:
        lines.extend([
            f"## {result['vehicle']}",
            "",
            (
                f"{result['differences']:,} mismatches across {result['checks']:,} "
                f"per-trim checks; **{result['agreement_percent']:.2f}% agreement**; "
                f"{result['unique_mismatches']} unique part/mode rows; "
                f"{result['ignored_structural_fallbacks']:,} expected twin-absent "
                "fallbacks excluded."
            ),
            "",
        ])
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in result["rows"]:
            grouped[(row["baseline"], row["recommended"])].append(row)
        for transition in result["transitions"]:
            key = (transition["baseline"], transition["recommended"])
            lines.extend([
                f"### `{key[0]} → {key[1]}` — {transition['count']} checks",
                "",
                "| Part | Affected trims | Trims |",
                "|---|---:|---|",
            ])
            for row in grouped[key]:
                trims = row["trims"]
                if show_trims or len(trims) <= 3:
                    trim_text = ", ".join(f"`{trim}`" for trim in trims)
                else:
                    trim_text = ", ".join(f"`{trim}`" for trim in trims[:3])
                    trim_text += f", +{len(trims) - 3} more"
                lines.append(
                    f"| `{row['object_id']}` | {row['trim_count']} | {trim_text} |"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare spatial recommendations with cached conversion baselines."
    )
    parser.add_argument(
        "vehicles",
        nargs="*",
        default=list(DEFAULT_VEHICLES),
        help="project directory names (default: etk800 pickup sunburst2)",
    )
    parser.add_argument(
        "--projects-root",
        type=Path,
        default=core.PROJECTS_DIR,
        help=f"project root (default: {core.PROJECTS_DIR})",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="report format (default: markdown)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write the report to this path instead of stdout",
    )
    parser.add_argument(
        "--show-trims",
        action="store_true",
        help="show every affected trim rather than the first three",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(len(DEFAULT_VEHICLES), os.cpu_count() or 1),
        help="parallel vehicle processes (default: up to 3)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    vehicles = list(dict.fromkeys(args.vehicles))
    jobs = max(1, min(args.jobs, len(vehicles)))
    projects_root = str(args.projects_root.resolve())

    if jobs == 1:
        results = [validate_vehicle(projects_root, vehicle) for vehicle in vehicles]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {
                vehicle: pool.submit(validate_vehicle, projects_root, vehicle)
                for vehicle in vehicles
            }
            results = [futures[vehicle].result() for vehicle in vehicles]

    if args.format == "json":
        report = json.dumps(results, indent=2) + "\n"
    else:
        report = markdown_report(results, args.show_trims)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(args.output.resolve())
    else:
        print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
