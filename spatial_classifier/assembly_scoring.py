"""Shared plumbing for the assembly-detector validation runners.

run_detection_A.py and run_detection_B.py drive their (differently shaped)
detector across the same vehicles/trims and score the result per anchor role.
Everything identical between them lives here: loading the engine + a candidate,
unpickling a project context, producing (frame, present, entries_np) per trim,
and rolling per-trim role hits up into a per-vehicle 0/3..3/3 tally.

Scoring is deliberately outcome-only: for each role we ask "did the detector
put a correct mesh in that slot for this vehicle?" and count how many of the
three vehicles it got right. False positives / negatives are not tracked.
A vehicle counts for a role when the detector is correct in the majority of
that vehicle's trims where the role is actually present.
"""

from __future__ import annotations

import importlib.util
import json
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))  # so the engine can import jbeam_parser

LABELS_PATH = HERE / "assembly_labels.json"


def _read_labels_file() -> dict:
    """Parse assembly_labels.json, tolerating // comments and trailing commas.

    The labels file is hand-curated, so authors annotate it (e.g. // to shelve a
    mesh) and leave trailing commas. We strip both before json.loads rather than
    forcing strict JSON on an editable ground-truth file.
    """
    from jbeam_parser import strip_json_comments

    text = LABELS_PATH.read_text(encoding="utf-8")
    text = strip_json_comments(text)
    text = re.sub(r",(\s*[\]}])", r"\1", text)  # drop trailing commas
    return json.loads(text)


def labelled_vehicles() -> tuple[str, ...]:
    """Every vehicle carrying ground-truth labels (the torture-test set)."""
    return tuple(k for k in _read_labels_file() if not k.startswith("_"))


# The full confident label set. iter_trials() silently skips any of these whose
# context.cache has not been built yet, so running against this list "torture
# tests" across whatever is available and reports the rest as skipped.
DEFAULT_VEHICLES = labelled_vehicles()

# Display order requested for the report.
ROLE_ORDER = (
    "driver_door_card",
    "passenger_door_card",
    "driver_seat",
    "passenger_seat",
    "dashboard",
)
ROLE_DISPLAY = {
    "driver_door_card": "driver door card",
    "passenger_door_card": "passenger door card",
    "driver_seat": "driver seat",
    "passenger_seat": "passenger seat",
    "dashboard": "dashboard",
}
# Short grid-column labels, aligned with ROLE_ORDER.
ROLE_ABBR = {
    "driver_door_card": "driver",
    "passenger_door_card": "pass.",
    "driver_seat": "seatD",
    "passenger_seat": "seatP",
    "dashboard": "dash",
}
# Detector group -> the scoring sub-roles it produces. The group names match
# classifier_standalone_B.ANCHOR_ROLES so a --roles toggle drives both the
# detector (skip the search) and the scorer (skip the columns) in step.
ROLE_GROUPS = {
    "seats": ("driver_seat", "passenger_seat"),
    "dashboard": ("dashboard",),
    "doors": ("driver_door_card", "passenger_door_card"),
}
ALL_ROLE_GROUPS = tuple(ROLE_GROUPS)


def subroles_for_groups(groups: "set[str] | frozenset[str] | None") -> tuple[str, ...]:
    """The ROLE_ORDER-ordered sub-roles for a set of detector groups (all if
    None). Raises on an unknown group name so a typo'd --roles fails loudly."""
    if groups is None:
        return ROLE_ORDER
    unknown = set(groups) - set(ROLE_GROUPS)
    if unknown:
        raise ValueError(f"unknown role group(s): {sorted(unknown)}; "
                         f"choose from {ALL_ROLE_GROUPS}")
    active = {sr for g in groups for sr in ROLE_GROUPS[g]}
    return tuple(r for r in ROLE_ORDER if r in active)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # register before exec so self-imports resolve
    spec.loader.exec_module(module)
    return module


def load_engine():
    """The old amalgamation, reused only for its context/frame/entries plumbing.

    Loaded first so its module-name registrations exist before we unpickle a
    context written against the original module names.
    """
    return _load_module(HERE / "classifier_standalone.py", "engine")


def load_candidate(letter: str):
    letter = letter.upper()
    return _load_module(
        HERE / f"classifier_standalone_{letter}.py", f"candidate_{letter}"
    )


def load_labels() -> dict:
    """The hand-curated ground-truth labels (// comments and trailing commas ok)."""
    return _read_labels_file()


@dataclass
class Trial:
    """One trim of one vehicle, ready to feed a detector."""

    vehicle: str
    trim: str | None
    context: object
    frame: object
    present: list[str]
    entries_np: dict[str, object]


def cache_path_for(engine, vehicle: str) -> Path:
    return engine.PROJECTS_DIR / vehicle / "context.cache"


def vehicle_has_cache(engine, vehicle: str) -> bool:
    return cache_path_for(engine, vehicle).is_file()


def partition_by_cache(engine, vehicles):
    """Split vehicles into (runnable, missing_cache), preserving order."""
    present = [v for v in vehicles if vehicle_has_cache(engine, v)]
    missing = [v for v in vehicles if not vehicle_has_cache(engine, v)]
    return present, missing


def iter_trials(engine, vehicle: str):
    """Yield a Trial per trim; skips vehicles with no trustworthy driver frame."""
    cache_path = cache_path_for(engine, vehicle)
    if not cache_path.is_file():
        raise FileNotFoundError(f"no context cache: {cache_path}")
    with cache_path.open("rb") as handle:
        payload = pickle.load(handle)
    context = payload.get("context") if isinstance(payload, dict) else payload
    engine.prepare_vehicle_context(context)

    # No camera + no wheel anywhere: nothing trustworthy to detect at all.
    if engine.driver_frame_for_context(context) is None:
        return

    available = {o for o in context.objects if o in context.preview_by_id}
    trims = sorted(context.variants) if context.variants else [None]
    for trim in trims:
        # Per-trim frame: the driver camera carries the trim's cab nodeMove
        # (a us_semi cabover vs conventional cab move their eye differently),
        # so a single context-level eye is wrong for multi-cab vehicles.
        frame = engine.driver_frame_for_context(context, config_name=trim)
        if frame is None:
            continue
        present, entries_np = engine._spatial_entries_for_trim(
            context, trim, available
        )

        # The scorer's candidate/ground-truth admission boundary is the same
        # resolved slot tree used by BeamXP itself.  Never let a stale standalone
        # helper broaden it with orphaned .pc leftovers: Miramar, for example,
        # mentions racing_seat_FR in one config even though the chosen part tree
        # never exposes that slot.
        if trim is None:
            resolved_scope = set(available)
        else:
            scope_fn = getattr(
                engine,
                "authoritative_used_meshes_for_config",
                engine.used_meshes_for_config,
            )
            resolved_scope = set(scope_fn(context, trim)) & set(available)
        present = sorted(set(present) & resolved_scope)
        entries_np = {
            mesh: entries_np[mesh]
            for mesh in present
            if mesh in entries_np
        }
        present = [mesh for mesh in present if mesh in entries_np]
        if present:
            yield Trial(vehicle, trim, context, frame, present, entries_np)


class RoleScorer:
    """Accumulates per-vehicle, per-role correctness across trims.

    detections passed to score_trim is a dict:
        {"driver_seat": mesh|None, "passenger_seat": mesh|None,
         "dashboard": mesh|None, "door_cards": [mesh, ...]}
    Left/right door cards are resolved to driver/passenger here from geometry,
    so the runners don't have to know each detector's side convention.
    """

    def __init__(self, labels: dict, roles: "set[str] | frozenset[str] | None" = None):
        self.labels = labels
        # Active sub-roles (ROLE_ORDER-ordered) for the requested detector groups;
        # every report iterates these, so toggled-off roles never appear.
        self.roles: tuple[str, ...] = subroles_for_groups(roles)
        self._active: frozenset[str] = frozenset(self.roles)
        # (vehicle, role) -> [correct_trims, applicable_trims]
        self._counts: dict[tuple[str, str], list[int]] = {}
        # per-(vehicle, trim, role) detected-vs-expected records, in scoring order
        self._detail: list[dict] = []

    def _bump(self, vehicle: str, role: str, correct: bool, applicable: bool) -> None:
        if not applicable:
            return
        cell = self._counts.setdefault((vehicle, role), [0, 0])
        cell[1] += 1
        if correct:
            cell[0] += 1

    def _record(self, trial: "Trial", role: str, detected, expected,
                correct: bool, applicable: bool) -> None:
        """Tally the role AND keep the explicit detected/expected meshes.

        A no-op for roles outside the active set, so a --roles run neither
        scores nor lists the detectors it skipped."""
        if role not in self._active:
            return
        self._bump(trial.vehicle, role, correct, applicable)
        self._detail.append({
            "vehicle": trial.vehicle,
            "trim": trial.trim or "(default)",
            "role": role,
            "detected": [m for m in detected if m],
            "expected": sorted(expected),
            "correct": bool(correct),
            "applicable": bool(applicable),
        })

    def score_trim(self, trial: Trial, detections: dict) -> None:
        veh = self.labels.get(trial.vehicle, {})
        seats = set(veh.get("seats", ()))
        dash = set(veh.get("dashboard", ()))
        doors = set(veh.get("door_cards", ()))
        present = set(trial.present)
        entries = trial.entries_np
        cx = float(trial.frame.center_x)
        side = int(trial.frame.side)

        def driver_side(mesh: str | None, tol: float = 0.05) -> int:
            # +1 driver side of centreline, -1 passenger side, 0 central/unknown.
            pts = entries.get(mesh) if mesh else None
            if pts is None or len(pts) == 0:
                return 0
            rel = (float(np.asarray(pts, dtype=float)[:, 0].mean()) - cx) * side
            return 1 if rel > tol else -1 if rel < -tol else 0

        # --- seats: expected = the front seats actually present this trim, split
        # by side (driver side excludes passenger-only meshes, and vice versa).
        # Applicability is per-side, like door cards: a single-seat trim with no
        # passenger seat does not penalise the detector for finding none there.
        seat_present = [m for m in seats if m in present]
        gt_driver_seat = [m for m in seat_present if driver_side(m) != -1]
        gt_passenger_seat = [m for m in seat_present if driver_side(m) != 1]
        d_seat = detections.get("driver_seat")
        p_seat = detections.get("passenger_seat")
        self._record(
            trial, "driver_seat", [d_seat], gt_driver_seat,
            d_seat in seats and driver_side(d_seat) != -1, bool(gt_driver_seat),
        )
        self._record(
            trial, "passenger_seat", [p_seat], gt_passenger_seat,
            p_seat in seats and driver_side(p_seat) != 1, bool(gt_passenger_seat),
        )

        # --- dashboard
        dash_present = [m for m in dash if m in present]
        dash_det = detections.get("dashboard")
        self._record(
            trial, "dashboard", [dash_det], dash_present,
            dash_det in dash, bool(dash_present),
        )

        # --- door cards, resolved to driver/passenger side from geometry
        door_present = [m for m in doors if m in present]
        gt_driver = {m for m in door_present if driver_side(m) == 1}
        gt_passenger = {m for m in door_present if driver_side(m) == -1}
        detected = [m for m in detections.get("door_cards", []) if m]
        det_driver = {m for m in detected if driver_side(m) == 1}
        det_passenger = {m for m in detected if driver_side(m) == -1}
        self._record(
            trial, "driver_door_card", sorted(det_driver), gt_driver,
            bool(det_driver & gt_driver), bool(gt_driver),
        )
        self._record(
            trial, "passenger_door_card", sorted(det_passenger), gt_passenger,
            bool(det_passenger & gt_passenger), bool(gt_passenger),
        )

    def report(self, title: str) -> str:
        vehicles = sorted({vehicle for (vehicle, _role) in self._counts})
        width = max(len(ROLE_DISPLAY[r]) for r in self.roles)
        lines = [f"=== {title} ===", "", "(vehicles correct in EVERY applicable trim)", ""]
        for role in self.roles:
            applicable = correct = 0
            for vehicle in vehicles:
                cell = self._counts.get((vehicle, role))
                if not cell or cell[1] == 0:
                    continue
                applicable += 1
                if cell[0] == cell[1]:  # correct in EVERY applicable trim (strict)
                    correct += 1
            total = applicable or len(vehicles)
            lines.append(f"  {ROLE_DISPLAY[role]:<{width}}  {correct}/{total}")
        return "\n".join(lines) + "\n"

    def report_by_vehicle(self, title: str) -> str:
        """One row per vehicle, one column per role: ok / X / -- (n/a).

        A cell is 'ok' only when the detector was correct in EVERY trim where
        that role is present; a single wrong trim makes it 'X'. '--' means the
        role does not apply to that vehicle. This is the torture-test view: it
        shows exactly which model/role combinations the algorithm fails.
        """
        vehicles = sorted({vehicle for (vehicle, _role) in self._counts})
        head = tuple(ROLE_ABBR[r] for r in self.roles)
        lines = [f"=== {title} ===", ""]
        header = f"{'vehicle':<14} " + " ".join(f"{h:>6}" for h in head)
        lines.append(header)
        lines.append("-" * len(header))
        totals = {role: [0, 0] for role in self.roles}
        for vehicle in vehicles:
            cells = []
            for role in self.roles:
                cell = self._counts.get((vehicle, role))
                if not cell or cell[1] == 0:
                    cells.append("--")
                    continue
                ok = cell[0] == cell[1]  # strict: every applicable trim correct
                cells.append("ok" if ok else "X")
                totals[role][1] += 1
                totals[role][0] += 1 if ok else 0
            lines.append(f"{vehicle:<14} " + " ".join(f"{c:>6}" for c in cells))
        lines.append("-" * len(header))
        summ = [f"{totals[r][0]}/{totals[r][1]}" for r in self.roles]
        lines.append(f"{'TOTAL':<14} " + " ".join(f"{s:>6}" for s in summ))
        return "\n".join(lines) + "\n"

    def _grid_verdict(self, vehicle: str, role: str) -> str:
        """The console grid's ok/X/-- for one (vehicle, role): ok only when
        correct in EVERY applicable trim (one wrong trim => X). Ties the detail
        rows back to the grid so the two views reconcile."""
        cell = self._counts.get((vehicle, role))
        if not cell or cell[1] == 0:
            return "--"
        return "ok" if cell[0] == cell[1] else "X"

    def report_detail(self, title: str) -> str:
        """Only the MISMATCHES, deduped across trims (unique, not per-trim).

        Every applicable role the detector got wrong is reduced to its unique
        (role, got, expected) signature per vehicle, reported once with the
        count of that role's trims it covers. Grading is strict: any wrong trim
        makes the role 'X' in the grid, so every role listed here is a grid-'X'
        role. The vehicle header repeats the full grid row for context.
        Vehicles with no mismatch are omitted.
        """
        # (vehicle, role, got, expected) -> [trims...]
        uniq: dict[tuple, list[str]] = {}
        role_trims: dict[tuple[str, str], set[str]] = {}
        for r in self._detail:
            if not r["applicable"]:
                continue
            role_trims.setdefault((r["vehicle"], r["role"]), set()).add(r["trim"])
            if r["correct"]:
                continue
            key = (r["vehicle"], r["role"], tuple(r["detected"]), tuple(r["expected"]))
            uniq.setdefault(key, []).append(r["trim"])

        by_v: dict[str, list[tuple]] = {}
        for key, trims in uniq.items():
            by_v.setdefault(key[0], []).append((key[1], key[2], key[3], trims))

        lines = [
            f"=== {title} — unique mismatches (deduped across trims) ===",
            "",
            "grid[...] repeats the console grid for the vehicle. Grading is "
            "strict: a role is ok only if EVERY trim is correct, so every role "
            "listed below is X in the grid.",
            "",
        ]
        if not by_v:
            lines.append("no mismatches.")
            return "\n".join(lines) + "\n"

        for vehicle in sorted(by_v):
            rows = sorted(by_v[vehicle],
                          key=lambda t: (self.roles.index(t[0]), -len(t[3])))
            grid = " ".join(
                f"{ROLE_ABBR[role]}:{self._grid_verdict(vehicle, role)}"
                for role in self.roles
            )
            lines.append(f"# {vehicle}   grid[ {grid} ]   "
                         f"({len(rows)} unique mismatch"
                         f"{'' if len(rows) == 1 else 'es'})")
            for role, got, exp, trims in rows:
                total = len(role_trims.get((vehicle, role), ()))
                got_s = " | ".join(got) or "(none)"
                exp_s = " | ".join(exp) or "(none)"
                lines.append(f"    {ROLE_DISPLAY[role]:<20} got: {got_s}")
                lines.append(f"    {'':<20} exp: {exp_s}")
                lines.append(f"    {'':<20} in {len(trims)}/{total} trims")
            lines.append("")
        return "\n".join(lines) + "\n"


def execute(title, vehicles, engine, scorer, run_trial, *,
            detail: bool = True, detail_path: Path | None = None) -> int:
    """Drive run_trial(trial) over every buildable vehicle, then print reports.

    Shared spine for the A/B runners. Vehicles without a built context.cache are
    skipped and listed (with a build hint) instead of raising, and a failure on
    one vehicle is caught and reported so the rest of the torture test still
    runs. Returns a process exit code (0 always; the report is the signal).
    """
    vehicles = list(dict.fromkeys(vehicles))
    present, missing = partition_by_cache(engine, vehicles)
    ran, no_frame, errored = [], [], []
    for vehicle in present:
        try:
            trials = list(iter_trials(engine, vehicle))
        except Exception as exc:  # torture test: one bad vehicle must not abort
            errored.append((vehicle, exc))
            continue
        if not trials:
            no_frame.append(vehicle)
            continue
        for trial in trials:
            try:
                run_trial(trial)
            except Exception as exc:
                errored.append((f"{vehicle}:{trial.trim}", exc))
        ran.append(vehicle)

    # The mismatch detail can be large; write it to a file and keep the console
    # to the compact grid + totals. Fall back to stdout only if no path is given.
    detail_written = None
    if detail:
        text = scorer.report_detail(title)
        if detail_path is None:
            detail_path = HERE / "detection_detail.txt"
        try:
            detail_path.write_text(text, encoding="utf-8")
            detail_written = detail_path
        except OSError as exc:
            print(f"(could not write detail to {detail_path}: {exc}; printing instead)")
            print(text)

    print(scorer.report_by_vehicle(title))
    print(scorer.report(title))
    if detail_written is not None:
        print(f"unique mismatches (deduped across trims) written to: {detail_written}")
    elif not detail and detail_path is not None and detail_path.is_file():
        # Guard the footgun: --no-detail leaves the on-disk file untouched, so it
        # reflects an earlier (possibly different-code) run. Say so explicitly.
        print(f"note: {detail_path.name} NOT regenerated (--no-detail); it is "
              f"from an earlier run and may not match the grid above.")
    print(f"ran {len(ran)}/{len(vehicles)} vehicles: {', '.join(ran) or '(none)'}")
    if no_frame:
        print(f"no driver frame ({len(no_frame)}): {', '.join(no_frame)}")
    if errored:
        print(f"errors ({len(errored)}):")
        for name, exc in errored:
            print(f"  {name}: {type(exc).__name__}: {exc}")
    if missing:
        print(f"skipped, no context.cache ({len(missing)}): {', '.join(missing)}")
        print("  build them with:  python build_caches.py " + " ".join(missing))
    return 0
