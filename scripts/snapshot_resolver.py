"""Regression net for the jbeam-resolver rewrite.

Captures, per vehicle+config, the resolver's geometry-affecting outputs --
which parts are selected (by slot) and where every resolved node ends up -- so
each refactor step can diff against a pre-refactor baseline. Intended geometry
fixes show up as reviewable diffs; anything else changing is a regression.

Resolver-only: it builds a minimal VehicleContext with no DAE parsing (via the
shared core.load_resolver_inputs seam), so a full 34-vehicle sweep runs in
seconds and stays honest to the live resolver code it guards.

Usage (point BEAMXP_DATA_DIR at a scratch dir so nothing touches AppData):

    python scripts/snapshot_resolver.py capture baseline.json
    python scripts/snapshot_resolver.py diff   baseline.json
    python scripts/snapshot_resolver.py diff   baseline.json --tol 1e-6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import beamng_hand_drive_core as core  # noqa: E402

STOCK_DIR = Path("G:/SteamLibrary/steamapps/common/BeamNG.drive/content/vehicles")
VEHICLE_LIST = REPO_ROOT / "spatial classifier" / "vanilla_cars_trucks.txt"
ROUND = 6  # positions rounded to micrometres before hashing/diffing


def vehicle_ids() -> list[str]:
    ids: list[str] = []
    for line in VEHICLE_LIST.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ids.append(line[:-4] if line.endswith(".zip") else line)
    return ids


_common_cache: dict[str, tuple[dict[str, str], dict[str, tuple[str, str]]]] = {}


def _common_for(source_zip: Path) -> tuple[dict[str, str], dict[str, tuple[str, str]]]:
    """vehicles/common is identical for every stock vehicle in one folder, so
    parse+index it once (it dominates load_resolver_inputs' cost otherwise)."""
    key = str(source_zip.parent)
    if key not in _common_cache:
        texts = core.load_common_jbeam_texts(source_zip)
        _common_cache[key] = (texts, core.build_part_body_index(texts) if texts else {})
    return _common_cache[key]


def resolver_context(source_zip: Path, vehicle_id: str) -> core.VehicleContext:
    """Minimal context carrying only what the resolver reads (no DAE work)."""
    common_texts, common_index = _common_for(source_zip)
    jbeam_texts, part_body_index, node_positions = core.load_resolver_inputs(
        source_zip, vehicle_id, common_texts=common_texts, common_index=common_index
    )
    variants: dict[str, core.VariantInfo] = {}
    for pc_path in core.direct_vehicle_files(source_zip, vehicle_id, ".pc"):
        name = Path(pc_path).stem
        variants[name] = core.VariantInfo(
            name=name, pc_path=pc_path, info_path=None, display_name=name
        )
    return core.VehicleContext(
        source_zip=source_zip,
        vehicle_id=vehicle_id,
        vehicle_path=core.vehicle_prefix(vehicle_id),
        dae_paths=[],
        variants=variants,
        objects={},
        preview_by_id={},
        jbeam_texts=jbeam_texts,
        node_positions=node_positions,
        project_dir=Path("."),
        part_body_index=part_body_index,
    )


def snapshot_config(context: core.VehicleContext, config_name: str) -> dict[str, object]:
    selected = core.selected_parts_for_config(context, config_name)
    nodes = core.selected_node_positions_for_config(context, config_name)
    return {
        "main_part": selected.get("main_part"),
        "selected_by_slot": {
            str(k): str(v) for k, v in sorted(dict(selected.get("selected_by_slot", {})).items())
        },
        "parts": sorted(str(p) for p in selected.get("parts", set())),
        "missing_parts": sorted(str(p) for p in selected.get("missing_parts", set())),
        "nodes": {
            nid: [round(c, ROUND) for c in pos] for nid, pos in sorted(nodes.items())
        },
    }


def capture() -> dict[str, object]:
    out: dict[str, object] = {}
    for vid in vehicle_ids():
        zip_path = STOCK_DIR / f"{vid}.zip"
        if not zip_path.exists():
            out[vid] = {"error": "zip not found"}
            continue
        try:
            context = resolver_context(zip_path, vid)
            out[vid] = {
                cfg: snapshot_config(context, cfg) for cfg in sorted(context.variants)
            }
        except Exception as exc:  # noqa: BLE001 - a broken vehicle is data, not a crash
            out[vid] = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"  {vid}: {'error' if 'error' in out[vid] else str(len(out[vid])) + ' configs'}")
    return out


def _diff_config(old: dict, new: dict, tol: float) -> list[str]:
    msgs: list[str] = []
    for key in ("main_part",):
        if old.get(key) != new.get(key):
            msgs.append(f"    {key}: {old.get(key)!r} -> {new.get(key)!r}")
    for key in ("selected_by_slot",):
        o, n = old.get(key, {}), new.get(key, {})
        for slot in sorted(set(o) | set(n)):
            if o.get(slot) != n.get(slot):
                msgs.append(f"    slot {slot}: {o.get(slot)!r} -> {n.get(slot)!r}")
    o_parts, n_parts = set(old.get("parts", [])), set(new.get("parts", []))
    if o_parts != n_parts:
        added, removed = sorted(n_parts - o_parts), sorted(o_parts - n_parts)
        if added:
            msgs.append(f"    parts +{added}")
        if removed:
            msgs.append(f"    parts -{removed}")
    o_nodes, n_nodes = old.get("nodes", {}), new.get("nodes", {})
    moved = 0
    for nid in sorted(set(o_nodes) | set(n_nodes)):
        if nid not in o_nodes:
            msgs.append(f"    node +{nid} {n_nodes[nid]}")
        elif nid not in n_nodes:
            msgs.append(f"    node -{nid} {o_nodes[nid]}")
        else:
            op, np_ = o_nodes[nid], n_nodes[nid]
            if any(abs(a - b) > tol for a, b in zip(op, np_)):
                moved += 1
                if moved <= 8:
                    msgs.append(f"    node ~{nid} {op} -> {np_}")
    if moved > 8:
        msgs.append(f"    ... {moved - 8} more moved nodes")
    return msgs


def diff(baseline_path: Path, tol: float) -> int:
    old = json.loads(baseline_path.read_text(encoding="utf-8"))
    new = capture()
    total = 0
    for vid in sorted(set(old) | set(new)):
        o_veh, n_veh = old.get(vid, {}), new.get(vid, {})
        if "error" in o_veh or "error" in n_veh:
            if o_veh.get("error") != n_veh.get("error"):
                print(f"{vid}: error changed {o_veh.get('error')!r} -> {n_veh.get('error')!r}")
                total += 1
            continue
        for cfg in sorted(set(o_veh) | set(n_veh)):
            if cfg not in o_veh:
                print(f"{vid}/{cfg}: NEW config")
                total += 1
                continue
            if cfg not in n_veh:
                print(f"{vid}/{cfg}: REMOVED config")
                total += 1
                continue
            msgs = _diff_config(o_veh[cfg], n_veh[cfg], tol)
            if msgs:
                print(f"{vid}/{cfg}:")
                print("\n".join(msgs))
                total += 1
    print(f"\n{total} changed config(s)" if total else "\nno differences")
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("capture", "diff"))
    parser.add_argument("path", type=Path, help="output JSON (capture) or baseline JSON (diff)")
    parser.add_argument("--tol", type=float, default=1e-6, help="node position tolerance for diff")
    args = parser.parse_args()

    if args.mode == "capture":
        data = capture()
        args.path.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")
        n = sum(len(v) for v in data.values() if "error" not in v)
        print(f"\nwrote {args.path} ({n} configs across {len(data)} vehicles)")
        return 0
    return 1 if diff(args.path, args.tol) else 0


if __name__ == "__main__":
    raise SystemExit(main())
