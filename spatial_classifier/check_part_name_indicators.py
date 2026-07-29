#!/usr/bin/env python3
"""Check mesh-name indicators across vanilla BeamNG cars & trucks.

For each vehicle we collect every flexbody mesh name (the first token of each
row in the "flexbodies" blocks inside vehicles/<name>/**/*.jbeam) -- the level
the spatial classifier actually sees -- and test three candidate naming
patterns that look like reliable indicators for our parts of interest.

Meshes, not top-level part names: interior parts of interest are frequently
attached as flexbodies without a part of their own (e.g. the Bluebuck's dash
is the mesh "bluebuck_dash", carried by the body part). Scanning meshes finds
those; scanning part keys misses them.

    door card  ->  *_doorpanel_*   (substring "_doorpanel_")
    dashboard  ->  *_dash / *_dashboard  (tail "_dash" or "_dashboard")
    seat       ->  *_seat_* / *_seats_*  (substring "_seat_" or "_seats_")

It reports, per vehicle, which patterns hit and the matching part names, then a
coverage summary: on how many of the vehicles each pattern is present. If a
pattern is present on (nearly) all vehicles it is a reliable indicator.

Usage:
    python check_part_name_indicators.py
    python check_part_name_indicators.py --vehicles-dir "G:/.../content/vehicles"
    python check_part_name_indicators.py --list vanilla_cars_trucks.txt
    python check_part_name_indicators.py --show-near-misses   # _dash_*, _seats_, ...
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

from jbeam_parser import strip_json_comments

HERE = Path(__file__).resolve().parent
DEFAULT_VEHICLES_DIR = Path(
    r"G:/SteamLibrary/steamapps/common/BeamNG.drive/content/vehicles"
)
DEFAULT_LIST = HERE / "vanilla_cars_trucks.txt"

# name -> compiled regex applied to a part name (exactly as the user specified).
PATTERNS = {
    "doorpanel": re.compile(r"_doorpanel_"),
    # *_dash or *_dashboard, allowing a handedness suffix (bx/miramar use
    # *_dashboard_lhd / *_dashboard_rhd). No vehicle uses a bare "dash"/
    # "dashboard" part name, so no un-prefixed alternative is needed.
    "dash": re.compile(r"_dash(board)?(_lhd|_rhd)?$"),
    "seat": re.compile(r"_seats?_"),  # singular part name or plural mesh name
}
# Looser variants that sit just outside the strict patterns, reported with
# --show-near-misses so we can judge whether the strict pattern is too narrow.
NEAR_MISS = {
    "doorpanel": re.compile(r"doorpanel"),
    "dash": re.compile(r"_dash"),          # _dash, _dash_wagon, _dash_stripped
    "seat": re.compile(r"seat"),           # _seats_R, race_seat_, ...
}


def read_vehicle_list(path: Path) -> list[str]:
    """Base names (no .zip) from the list file, ignoring comments/blanks."""
    names: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        names.append(line[:-4] if line.lower().endswith(".zip") else line)
    return names


_FLEX_SECTION = re.compile(r'"flexbodies"\s*:\s*\[')


def flexbody_mesh_names(text: str) -> set[str]:
    """Every flexbody mesh name in the file -- the level the classifier sees.

    A mesh is the first quoted token of each row inside a "flexbodies": [ ... ]
    block (the "mesh" header row excluded), matching classifier_standalone's
    flexbody_row_mesh(). We scan for these rather than top-level part keys
    because the interior parts of interest are often *only* flexbodies: e.g.
    the Bruckell Bluebuck's dashboard is the mesh "bluebuck_dash" attached in
    the body part, with no part of that name. A depth-aware, string-aware scan
    is used because BeamNG's lax jbeam dialect defeats strict JSON parsing on
    most files.
    """
    text = strip_json_comments(text.lstrip("﻿"))
    names: set[str] = set()
    n = len(text)
    for section in _FLEX_SECTION.finditer(text):
        i = section.end() - 1  # the '[' opening the flexbodies array
        depth = 0
        in_string = False
        escape = False
        str_start = -1
        awaiting_first = False  # inside a fresh row, next string is the mesh
        while i < n:
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                    if awaiting_first:
                        awaiting_first = False
                        mesh = text[str_start + 1:i]
                        if mesh != "mesh":
                            names.add(mesh)
                i += 1
                continue
            if ch == '"':
                in_string = True
                str_start = i
                i += 1
                continue
            if ch == "[":
                depth += 1
                if depth == 2:  # opened a row within the flexbodies array
                    awaiting_first = True
                i += 1
                continue
            if ch == "{":
                depth += 1
                i += 1
                continue
            if ch in "}]":
                depth -= 1
                i += 1
                if ch == "]" and depth == 0:  # end of this flexbodies array
                    break
                continue
            i += 1
    return names


def mesh_names_in_zip(zip_path: Path) -> tuple[set[str], int]:
    """Every flexbody mesh name across the vehicle's jbeam files (the trailing
    0 is a vestigial failure count; the scanner never rejects a file)."""
    meshes: set[str] = set()
    with zipfile.ZipFile(zip_path) as archive:
        jbeams = [
            n for n in archive.namelist()
            if n.lower().endswith(".jbeam") and "/vehicles/" in f"/{n}"
        ]
        for name in jbeams:
            text = archive.read(name).decode("utf-8", errors="replace")
            meshes |= flexbody_mesh_names(text)
    return meshes, 0


def match(parts: set[str], patterns: dict[str, re.Pattern]) -> dict[str, list[str]]:
    return {
        label: sorted(p for p in parts if rx.search(p))
        for label, rx in patterns.items()
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vehicles-dir", type=Path, default=DEFAULT_VEHICLES_DIR)
    ap.add_argument("--list", type=Path, default=DEFAULT_LIST)
    ap.add_argument("--show-near-misses", action="store_true")
    args = ap.parse_args()

    if not args.vehicles_dir.is_dir():
        print(f"vehicles dir not found: {args.vehicles_dir}", file=sys.stderr)
        return 2
    if not args.list.is_file():
        print(f"list not found: {args.list}", file=sys.stderr)
        return 2

    vehicles = read_vehicle_list(args.list)
    labels = list(PATTERNS)
    coverage = {label: 0 for label in labels}
    present_vehicles = {label: [] for label in labels}
    missing_vehicles = {label: [] for label in labels}
    checked = 0

    header = f"{'vehicle':<14} " + " ".join(f"{l:>10}" for l in labels)
    print(header)
    print("-" * len(header))

    for veh in vehicles:
        zip_path = args.vehicles_dir / f"{veh}.zip"
        if not zip_path.is_file():
            print(f"{veh:<14} (zip not found)")
            continue
        meshes, failures = mesh_names_in_zip(zip_path)
        checked += 1
        hits = match(meshes, PATTERNS)

        cells = []
        for label in labels:
            got = bool(hits[label])
            cells.append("  yes" if got else "   -- ")
            if got:
                coverage[label] += 1
                present_vehicles[label].append(veh)
            else:
                missing_vehicles[label].append(veh)
        note = f"   [{failures} jbeam unparsed]" if failures else ""
        print(f"{veh:<14} " + " ".join(f"{c:>10}" for c in cells) + note)

        # Matching mesh names underneath, for inspection.
        for label in labels:
            if hits[label]:
                print(f"    {label:<10}: {', '.join(hits[label])}")
        if args.show_near_misses:
            near = match(meshes, NEAR_MISS)
            for label in labels:
                extra = sorted(set(near[label]) - set(hits[label]))
                if extra:
                    print(f"    ~{label:<9}: {', '.join(extra)}")

    print()
    print(f"=== Coverage over {checked} vehicles ===")
    for label in labels:
        n = coverage[label]
        print(f"  {label:<10} {n}/{checked}"
              + (f"   missing: {', '.join(missing_vehicles[label])}"
                 if missing_vehicles[label] else "   (all vehicles)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
