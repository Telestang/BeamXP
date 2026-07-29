#!/usr/bin/env python3
"""Build context.cache for labelled vehicles from their stock BeamNG zips.

The scoring runners (run_detection_A/B.py) read a per-vehicle context.cache from
PROJECTS_DIR; only a handful ship pre-built. This populates the rest so the full
26-vehicle torture test can run. Each vehicle is loaded straight from
content/vehicles/<name>.zip through the real tool loader, which writes the cache
to the same PROJECTS_DIR the scorer reads.

    python build_caches.py                 # build every labelled vehicle lacking a cache
    python build_caches.py etkc etki       # build specific vehicles
    python build_caches.py --force         # rebuild even if a cache exists
    python build_caches.py --vehicles-dir "G:/.../content/vehicles"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(HERE), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import assembly_scoring as scoring  # noqa: E402
from beamxp import hand_drive_core as core  # noqa: E402

DEFAULT_VEHICLES_DIR = Path(
    r"G:/SteamLibrary/steamapps/common/BeamNG.drive/content/vehicles"
)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("vehicles", nargs="*", help="defaults to all labelled vehicles")
    ap.add_argument("--vehicles-dir", type=Path, default=DEFAULT_VEHICLES_DIR)
    ap.add_argument("--force", action="store_true",
                    help="rebuild even when a cache already exists")
    args = ap.parse_args()

    if not args.vehicles_dir.is_dir():
        print(f"vehicles dir not found: {args.vehicles_dir}", file=sys.stderr)
        return 2

    # NOTE: we deliberately do NOT import/load the classifier_standalone engine
    # here. Loading that amalgamation initialises the lazy GPU (ModernGL) spatial
    # backend, which attaches an unpicklable thread-local to the freshly built
    # VehicleContext -- save_vehicle_context_cache() then swallows the pickle
    # error and no cache lands. core.PROJECTS_DIR is the same path the scorer
    # reads, so we confirm against it directly and keep this process GPU-free.
    wanted = list(dict.fromkeys(args.vehicles)) or list(scoring.labelled_vehicles())

    built, present, failed = [], [], []
    for veh in wanted:
        if not args.force and scoring.vehicle_has_cache(core, veh):
            present.append(veh)
            continue
        zip_path = args.vehicles_dir / f"{veh}.zip"
        if not zip_path.is_file():
            print(f"  ! {veh}: zip not found at {zip_path}")
            failed.append(veh)
            continue
        start = time.time()
        try:
            core.load_vehicle_context(zip_path, veh, use_cache=False)
        except Exception as exc:
            print(f"  ! {veh}: {type(exc).__name__}: {exc}")
            failed.append(veh)
            continue
        if scoring.vehicle_has_cache(core, veh):
            print(f"  + {veh}  ({time.time() - start:.1f}s)")
            built.append(veh)
        else:
            print(f"  ? {veh}: built but no cache at {scoring.cache_path_for(core, veh)}")
            failed.append(veh)

    print(f"\nbuilt {len(built)}, already-present {len(present)}, failed {len(failed)}")
    if failed:
        print("failed:", ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
