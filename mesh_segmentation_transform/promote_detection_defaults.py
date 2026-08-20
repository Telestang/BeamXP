"""Promote tuned detector values into the shipped defaults.

The harness remembers what you were experimenting with, but experimenting is
not shipping: a session is per-user and never leaves the machine it was tuned
on.  What a build uses is the literal in ``annotate_texture_regions``, so
promoting means editing that literal, and the file itself is the record of what
the product does.  This module does that edit.

It rewrites rather than regenerates.  Those blocks carry the reasoning for
almost every value in them -- what was measured, on which vehicle, and why the
number is where it is -- and a generated list would throw all of it away.  So
an existing entry has only its value replaced, in place, keeping the comment
sitting beside it, and anything new is appended with a note saying where it
came from.

Stdlib only, and deliberately no import of the module it edits: this has to be
able to run against a file that a bad edit has just made unimportable.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

__all__ = [
    "PromotionChange",
    "PROMOTION_TARGETS",
    "config_source_path",
    "plan_promotion",
    "promote_defaults",
    "promotion_is_possible",
]

# Which literal each detection family is promoted into.  These are the two
# objects the exporter imports; every colour front end shares the first and
# every relief front end the second.
PROMOTION_TARGETS = {
    "colour": "DEFAULT_COLOUR_CONFIG",
    "relief": "DEFAULT_RELIEF_DETECTION_CONFIG",
}

# Never promoted.  The front end is a property of the path being run rather
# than a tuned value, and production selects it per layer regardless, so
# writing it would pin a build to whichever pipeline the harness had open.
EXCLUDED_FIELDS = frozenset({"box_source"})


@dataclass(frozen=True, slots=True)
class PromotionChange:
    """One value the promotion would write."""

    field: str
    before: object
    after: object
    is_new: bool


def config_source_path() -> Path:
    """The module whose literals a build reads."""
    return Path(__file__).resolve().with_name("annotate_texture_regions.py")


def promotion_is_possible(path: Path | None = None) -> tuple[bool, str]:
    """Whether the shipped defaults can be edited from here, and why not.

    A packaged build has no source to promote into -- its literals were frozen
    when it was built -- so this refuses rather than writing a file that will
    never be read.
    """
    if getattr(sys, "frozen", False):
        return False, (
            "This is a packaged build, so the shipped defaults are already "
            "fixed in it.  Promote from a source checkout and rebuild."
        )
    target = path or config_source_path()
    if not target.is_file():
        return False, f"Cannot find {target}."
    try:
        with target.open("a", encoding="utf-8"):
            pass
    except OSError as exc:
        return False, f"{target} is not writable: {exc}"
    return True, ""


def _target_call(tree: ast.AST, name: str) -> ast.Call:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            continue
        if isinstance(node.value, ast.Call):
            return node.value
    raise LookupError(f"{name} is not assigned from a call in this module")


def plan_promotion(
    values: dict[str, object],
    key: str,
    path: Path | None = None,
) -> list[PromotionChange]:
    """What promoting ``values`` into ``key``'s literal would change.

    Values already equal to what the literal says are left out, so a promotion
    that would change nothing reports nothing rather than rewriting the file.
    """
    name = PROMOTION_TARGETS[key]
    source = (path or config_source_path()).read_text(encoding="utf-8")
    call = _target_call(ast.parse(source), name)
    present = {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in call.keywords
        if keyword.arg is not None
    }
    changes: list[PromotionChange] = []
    for field, value in sorted(values.items()):
        if field in EXCLUDED_FIELDS:
            continue
        if field in present:
            if present[field] != value:
                changes.append(
                    PromotionChange(field, present[field], value, is_new=False)
                )
        else:
            changes.append(PromotionChange(field, None, value, is_new=True))
    return changes


def _rewrite(source: str, name: str, changes: list[PromotionChange]) -> str:
    call = _target_call(ast.parse(source), name)
    lines = source.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))

    def offset(lineno: int, col: int) -> int:
        return starts[lineno - 1] + col

    by_field = {change.field: change for change in changes}
    edits: list[tuple[int, int, str]] = []
    for keyword in call.keywords:
        change = by_field.get(keyword.arg)
        if change is None or change.is_new:
            continue
        # Only the value is touched, so any comment on the line survives it.
        edits.append(
            (
                offset(keyword.value.lineno, keyword.value.col_offset),
                offset(keyword.value.end_lineno, keyword.value.end_col_offset),
                repr(change.after),
            )
        )

    added = [change for change in changes if change.is_new]
    if added:
        # In front of the closing bracket, which owns the last line of the call.
        insert_at = starts[call.end_lineno - 1]
        note = f"    # Promoted {date.today().isoformat()} from the tuning harness.\n"
        block = note + "".join(
            f"    {change.field}={change.after!r},\n" for change in added
        )
        edits.append((insert_at, insert_at, block))

    updated = source
    for start, end, text in sorted(edits, reverse=True):
        updated = updated[:start] + text + updated[end:]
    return updated


def promote_defaults(
    values: dict[str, object],
    key: str,
    path: Path | None = None,
) -> list[PromotionChange]:
    """Write ``values`` into the shipped literal for ``key``.

    The file is verified before it is kept: it has to parse, and a fresh
    interpreter has to import it and agree that the values took.  Anything less
    and the original is restored, because a half-promoted defaults file breaks
    every conversion rather than one tuning session.
    """
    target = path or config_source_path()
    changes = plan_promotion(values, key, target)
    if not changes:
        return []
    original = target.read_text(encoding="utf-8")
    updated = _rewrite(original, PROMOTION_TARGETS[key], changes)
    try:
        ast.parse(updated)
    except SyntaxError as exc:  # pragma: no cover - guards a bug in _rewrite
        raise RuntimeError(f"Promotion would not parse, so it was abandoned: {exc}")
    target.write_text(updated, encoding="utf-8")
    try:
        _verify(values, key, target)
    except Exception:
        target.write_text(original, encoding="utf-8")
        raise
    return changes


def _verify(values: dict[str, object], key: str, target: Path) -> None:
    """Check the rewrite actually took, in the strongest way available.

    The literal is always re-read: if promoting left anything still to promote,
    the edit landed somewhere it should not have.  The real module is also
    imported in a fresh interpreter, which is the only way to catch a file that
    parses but no longer loads -- a check a temporary fixture cannot support
    and does not need.
    """
    remaining = plan_promotion(values, key, target)
    if remaining:
        raise RuntimeError(
            "the rewritten file still reports "
            + ", ".join(change.field for change in remaining)
            + " as unpromoted"
        )
    if target.resolve() != config_source_path():
        return
    name = PROMOTION_TARGETS[key]
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from mesh_segmentation_transform.annotate_texture_regions import "
            + name,
        ],
        capture_output=True,
        text=True,
        cwd=str(config_source_path().parent.parent),
    )
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip().splitlines()
        raise RuntimeError(
            "the promoted file no longer imports: "
            + (detail[-1] if detail else "unknown error")
        )
