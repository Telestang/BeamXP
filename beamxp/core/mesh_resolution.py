from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping

from beamxp import transform_helpers

NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
TOP_LEVEL_STRING_RE = re.compile(r'"(?:[^"\\]|\\.)*"')
_NODE_ROW_RE = re.compile(
    rf'^\s*\[\s*"(?P<id>(?:[^"\\]|\\.)*)"\s*,\s*'
    rf'(?P<x>{NUMBER_RE})\s*,\s*(?P<y>{NUMBER_RE})\s*,\s*(?P<z>{NUMBER_RE})'
)


def selected_parts_in_merge_order(selected: Mapping[str, object]) -> list[str]:
    """Selected part ids in tree (parent-before-child) order."""
    order = selected.get("parts_order")
    parts = {str(item) for item in selected.get("parts", set())}
    if isinstance(order, list) and order:
        seen: set[str] = set()
        result = [
            str(part_id)
            for part_id in order
            if str(part_id) in parts and not (str(part_id) in seen or seen.add(str(part_id)))
        ]
        result.extend(sorted(parts - set(result)))
        return result
    return sorted(parts)


def selected_part_instances(selected: Mapping[str, object]) -> list[dict[str, object]]:
    raw = selected.get("part_instances")
    if isinstance(raw, list):
        valid = [dict(item) for item in raw if isinstance(item, dict)]
        if valid:
            return valid

    part_slot_options = selected.get("part_slot_options", {})
    result: list[dict[str, object]] = []
    for index, part_id in enumerate(selected_parts_in_merge_order(selected)):
        options: tuple[str, ...] = ()
        if isinstance(part_slot_options, dict):
            raw_options = part_slot_options.get(part_id, ())
            if isinstance(raw_options, (list, tuple)):
                options = tuple(str(item) for item in raw_options if item)
        result.append(
            {
                "instance_id": f"legacy:{index}:{part_id}",
                "part_id": part_id,
                "slot_id": "legacy",
                "slot_path": "/",
                "parent_instance_id": None,
                "inherited_options": options,
                "source_file": None,
            }
        )
    return result


def part_instance_options(instance: Mapping[str, object]) -> tuple[str, ...]:
    raw = instance.get("inherited_options", ())
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(str(item) for item in raw if item)


def part_variable_scope(selected: Mapping[str, object], part_id: str) -> dict[str, object]:
    scopes = selected.get("part_variables")
    if isinstance(scopes, dict):
        scope = scopes.get(part_id)
        if isinstance(scope, dict):
            return dict(scope)
    return {}


def part_instance_variable_scope(
    selected: Mapping[str, object],
    instance: Mapping[str, object],
) -> dict[str, object]:
    instance_scopes = selected.get("part_instance_variables")
    instance_id = str(instance.get("instance_id") or "")
    if isinstance(instance_scopes, dict):
        scope = instance_scopes.get(instance_id)
        if isinstance(scope, dict):
            return dict(scope)
    return part_variable_scope(selected, str(instance.get("part_id") or ""))


def iter_active_top_level_rows(array_text: str) -> list[str]:
    rows: list[str] = []
    idx = 1 if array_text.startswith("[") else 0
    length = len(array_text)
    while idx < length:
        ch = array_text[idx]
        if ch == "/" and array_text.startswith("//", idx):
            newline = array_text.find("\n", idx)
            idx = length if newline < 0 else newline + 1
            continue
        if ch == "/" and array_text.startswith("/*", idx):
            close = array_text.find("*/", idx + 2)
            idx = length if close < 0 else close + 2
            continue
        if ch == '"':
            match = TOP_LEVEL_STRING_RE.match(array_text, idx)
            idx = match.end() if match else idx + 1
            continue
        if ch == "[":
            try:
                end = transform_helpers.find_matching(array_text, idx, "[", "]")
            except ValueError:
                idx += 1
                continue
            rows.append(array_text[idx:end])
            idx = end
            continue
        idx += 1
    return rows


def iter_node_rows(
    node_array: str,
    variables: Mapping[str, object] | None = None,
    resolve_row_strings: Callable[[str, Mapping[str, object] | None], str] | None = None,
) -> Iterable[tuple[str, tuple[float, float, float], str]]:
    """Yield active node rows after resolving instance-specific expressions."""
    resolver = resolve_row_strings or (lambda row, _variables: row)
    for raw_row in iter_active_top_level_rows(node_array):
        row = resolver(raw_row, variables)
        match = _NODE_ROW_RE.match(row)
        if match is None:
            continue
        node_id = match.group("id")
        if node_id in {"id", "type", "mesh", "func"}:
            continue
        position = (
            float(match.group("x")),
            float(match.group("y")),
            float(match.group("z")),
        )
        yield node_id, position, row
