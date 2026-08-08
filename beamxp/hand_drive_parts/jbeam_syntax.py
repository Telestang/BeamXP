"""Implementation slice extracted from ``beamng_hand_drive_core``.

Original source lines 585-2070. Import the public orchestration module
``beamng_hand_drive_core`` rather than this implementation module directly.
"""

from __future__ import annotations

import ast
import json
import math
import re
import zipfile
from pathlib import Path
from collections.abc import Iterable
from beamxp import transform_helpers
from beamxp.core.beam_json import (
    add_missing_json_commas,
    display_name_for,
    info_path_for_config,
    json_line_needs_comma,
    load_info,
    load_pc,
    parse_beamng_json,
    strip_json_comments,
    zip_json_by_name as _zip_json_by_name,
)
from beamxp.core.constants import (
    ACTION_OPPOSITE,
    ACTION_SKIP,
    ACTION_TO_LHD,
    ACTION_TO_RHD,
    APP_DIR,
    APP_SETTINGS_PATH,
    BUILD_BOTH,
    BUILD_CHOICES,
    BUILD_CONVERTED,
    BUILD_OFF,
    BUILD_ORIGINAL,
    HAND_AUTO,
    HAND_CHOICES,
    HAND_LHD,
    HAND_RHD,
    HAND_UNKNOWN,
    MODE_CHOICES,
    MODE_MIRROR,
    MODE_MIRROR_STRUCTURAL,
    MODE_SKIP,
    MODE_TRANSLATE,
    NS,
    NUMBER_RE,
    PREVIEW_FAR_LIMIT,
    PROJECTS_DIR,
    SOURCE_ROOT_DIR,
    STEERING_NAME_EXCLUDES,
    THIS_DIR,
    TOOL_VERSION,
    USER_DATA_DIR,
    WORKSPACE_DIR,
    default_beamng_mods_dir,
    default_user_data_dir,
)
from beamxp.core.files import (
    beamng_game_common_zips,
    clean_dir,
    common_zip_candidates,
    direct_vehicle_files,
    fs_path,
    list_vehicle_files,
    load_jbeam_texts,
    make_zip,
    project_dir_for,
    read_json_file,
    safe_id,
    safe_project_segment,
    vehicle_ids_in_zip,
    vehicle_prefix,
    write_bytes_file,
    write_text_file,
    write_xml_tree,
    zip_member_path,
)
from beamxp.core.geometry import (
    PROP_VECTOR_RE,
    base_rotation_global_matrix3,
    brg_rotation_matrix3,
    clamp_value,
    cross_product,
    euler_from_matrix3,
    euler_matrix3,
    euler_yzx_from_matrix3,
    identity_matrix,
    matrix3_from_axes,
    matrix3_from_matrix4,
    matrix4_flat,
    mirror_rotation_matrix_x,
    mirror_x_matrix4,
    multiply_matrix,
    multiply_matrix3,
    normalize_vector,
    prop_base_rotation_matrix3,
    prop_row_vector_objects,
    rotation_transpose_matrix3,
    rotation_transpose_matrix4,
    rotation_x_matrix,
    rotation_y_matrix,
    rotation_z_matrix,
    scale_matrix,
    sign_number,
    translation_matrix,
    vector_subtract,
)
from beamxp.core.models import (
    BakedMeshSpec,
    BuildResult,
    MeshPlacement,
    ResolvedMeshPosition,
    SharedBakeContext,
    SlotDef,
    VariantInfo,
    VehicleContext,
)

def load_resolver_inputs(
    source_zip: Path,
    vehicle_id: str,
    *,
    common_texts: dict[str, str] | None = None,
    common_index: dict[str, tuple[str, str]] | None = None,
) -> tuple[dict[str, str], dict[str, tuple[str, str]], dict[str, tuple[float, float, float]]]:
    """Assemble the jbeam inputs the slot/part resolver needs, independent of
    any DAE/visibility work.

    Returns (jbeam_texts, part_body_index, node_positions). The vehicle's own
    jbeam is indexed first; parts under vehicles/common are pulled in only when
    the vehicle's slot graph can reach them (reachable_common_part_index). This
    is the single seam load_vehicle_context and the resolver regression harness
    share, so changes to namespace scoping land in one place.

    common_texts/common_index let a caller supply an already-parsed vehicles/common
    index (it is vehicle-independent for a given source folder), so a batch tool
    parses common once instead of per vehicle. The app path passes neither and
    behaves exactly as before.
    """
    jbeam_texts = load_jbeam_texts(source_zip, vehicle_id)
    part_body_index = build_part_body_index(jbeam_texts)
    if common_texts is None:
        common_texts = load_common_jbeam_texts(source_zip)
    if common_index is None:
        common_index = build_part_body_index(common_texts) if common_texts else {}
    if common_index:
        reachable_common = reachable_common_part_index(part_body_index, common_index)
        if reachable_common:
            part_body_index.update(reachable_common)
            for _body, filename in reachable_common.values():
                jbeam_texts.setdefault(filename, common_texts[filename])
    node_positions = build_node_position_index(jbeam_texts)
    return jbeam_texts, part_body_index, node_positions


def load_common_jbeam_texts(source_zip: Path) -> dict[str, str]:
    texts: dict[str, str] = {}
    for candidate_zip in common_zip_candidates(source_zip):
        try:
            with zipfile.ZipFile(candidate_zip) as zf:
                for name in zf.namelist():
                    norm = name.replace("\\", "/")
                    if (
                        norm.lower().startswith("vehicles/common/")
                        and norm.lower().endswith(".jbeam")
                        and norm not in texts
                    ):
                        texts[norm] = zf.read(name).decode("utf-8", errors="replace")
        except Exception:
            continue
    return texts


def slot_demand_types(part_body: str) -> set[str]:
    demanded: set[str] = set()
    slots = transform_helpers.extract_named_array(part_body, "slots")
    if slots:
        for row in iter_active_top_level_rows(slots):
            values = split_top_level_values(row)
            if len(values) < 2:
                continue
            slot_type = quoted_string_value(values[0])
            if slot_type and slot_type not in {"type", "name"}:
                demanded.add(slot_type)
    slots2 = transform_helpers.extract_named_array(part_body, "slots2")
    if slots2:
        for row in iter_active_top_level_rows(slots2):
            values = split_top_level_values(row)
            if len(values) < 4:
                continue
            name = quoted_string_value(values[0])
            if not name or name in {"name", "type"}:
                continue
            demanded.add(name)
            demanded.update(re.findall(r'"((?:[^"\\]|\\.)*)"', values[1]))
    return demanded


def reachable_common_part_index(
    vehicle_part_index: dict[str, tuple[str, str]],
    common_part_index: dict[str, tuple[str, str]],
) -> dict[str, tuple[str, str]]:
    """Parts defined under vehicles/common are only indexed when the vehicle's
    slot graph can actually pull them in, keeping the part inventory focused."""
    parts_by_slot_type: dict[str, list[str]] = {}
    for part_id, (body, _filename) in common_part_index.items():
        for slot_type in transform_helpers.extract_part_slot_types(body):
            parts_by_slot_type.setdefault(slot_type, []).append(part_id)

    demanded: set[str] = set()
    reachable: dict[str, tuple[str, str]] = {}
    pending = [body for body, _filename in vehicle_part_index.values()]
    while pending:
        body = pending.pop()
        # sorted(): slot_demand_types returns a set, and Python randomises str
        # hashing per process, so unsorted iteration made this dict's insertion
        # order vary run to run. That order reaches collect_flexbody_mesh_placements
        # and therefore which points sample_points keeps, making previews (and the
        # context cache) irreproducible between runs over identical input.
        for slot_type in sorted(slot_demand_types(body)):
            if slot_type in demanded:
                continue
            demanded.add(slot_type)
            for part_id in parts_by_slot_type.get(slot_type, []):
                if part_id in reachable or part_id in vehicle_part_index:
                    continue
                entry = common_part_index[part_id]
                reachable[part_id] = entry
                pending.append(entry[0])
    return reachable


def extract_node_positions_from_array(array_text: str) -> dict[str, tuple[float, float, float]]:
    node_re = re.compile(
        rf'^\s*\[\s*"(?P<id>(?:[^"\\]|\\.)*)"\s*,\s*'
        rf'(?P<x>{NUMBER_RE})\s*,\s*(?P<y>{NUMBER_RE})\s*,\s*(?P<z>{NUMBER_RE})',
        re.MULTILINE,
    )
    nodes: dict[str, tuple[float, float, float]] = {}
    for match in node_re.finditer(array_text):
        node_id = match.group("id")
        if node_id in {"id", "type", "mesh", "func"}:
            continue
        nodes[node_id] = (
            float(match.group("x")),
            float(match.group("y")),
            float(match.group("z")),
        )
    return nodes


def build_node_position_index(jbeam_texts: dict[str, str]) -> dict[str, tuple[float, float, float]]:
    nodes: dict[str, tuple[float, float, float]] = {}
    pattern = re.compile(r'"nodes"\s*:[\s,]*\[')
    for text in jbeam_texts.values():
        for match in pattern.finditer(text):
            bracket = text.rfind("[", match.start(), match.end())
            if bracket < 0:
                continue
            try:
                end = transform_helpers.find_matching(text, bracket, "[", "]")
            except Exception:
                continue
            for node_id, position in extract_node_positions_from_array(text[bracket:end]).items():
                nodes.setdefault(node_id, position)
    return nodes


def build_part_body_index(jbeam_texts: dict[str, str]) -> dict[str, tuple[str, str]]:
    index: dict[str, tuple[str, str]] = {}
    # [\s,]* tolerates the stray comma stock jbeam ships between the colon
    # and the brace ("bluebuck_bumper_F":, {...}); the game accepts it.
    key_pattern = re.compile(r'"((?:[^"\\]|\\.)*)"\s*:[\s,]*\{')
    for filename, text in jbeam_texts.items():
        for match in key_pattern.finditer(text):
            part_id = match.group(1)
            if part_id in index:
                continue
            brace = text.find("{", match.start(), match.end())
            if brace < 0:
                continue
            try:
                end = transform_helpers.find_matching(text, brace, "{", "}")
            except Exception:
                continue
            body = text[match.start() : end]
            if '"slotType"' not in body:
                continue
            index[part_id] = (body, filename)
    return index


TOP_LEVEL_STRING_RE = re.compile(r'"(?:[^"\\]|\\.)*"')


def iter_top_level_rows(array_text: str) -> list[str]:
    rows: list[str] = []
    idx = 1 if array_text.startswith("[") else 0
    length = len(array_text)
    while idx < length:
        ch = array_text[idx]
        if ch == '"':
            match = TOP_LEVEL_STRING_RE.match(array_text, idx)
            idx = match.end() if match else idx + 1
            continue
        if ch == "[":
            try:
                end = transform_helpers.find_matching(array_text, idx, "[", "]")
            except ValueError:
                # Stock jbeam ships the odd row whose quotes/brackets never
                # balance (usually inside a commented-out line); skip the
                # bracket instead of failing the whole array.
                idx += 1
                continue
            rows.append(array_text[idx:end])
            idx = end
            continue
        idx += 1
    return rows


def iter_active_top_level_rows(array_text: str) -> list[str]:
    """Like iter_top_level_rows, but skips rows that are commented out
    (``//`` line comments and ``/* */`` block comments), matching what the
    game's jbeam parser actually loads. Used by the preview payloads and by
    slot resolution (commented-out slot rows must not select parts); the
    build path keeps iter_top_level_rows so commented text is preserved
    verbatim in rewritten jbeam."""
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


def split_top_level_values(row: str) -> list[str]:
    text = row.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    values: list[str] = []
    start = 0
    depth = 0
    in_string = False
    escape = False
    for idx, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch in "[{":
            depth += 1
            continue
        if ch in "]}":
            depth -= 1
            continue
        if ch == "," and depth == 0:
            values.append(text[start:idx].strip())
            start = idx + 1
    tail = text[start:].strip()
    if tail:
        values.append(tail)
    return values


def quoted_string_value(value: str) -> str | None:
    match = re.match(r'\s*"((?:[^"\\]|\\.)*)"', value)
    if match is None:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return match.group(1)


def trailing_options_object(values: list[str]) -> str | None:
    if not values:
        return None
    value = values[-1].strip()
    if value.startswith("{") and value.endswith("}"):
        return value
    # Missing comma before the options dict glues it onto the description, so
    # split_top_level_values yields one value like  "Rear Wheels" {"nodeOffset":..}
    # instead of two. Recover the trailing {...} object (the Bolide's centre-lug
    # wheel slot loses its nodeOffset otherwise, parking the wheel at the origin).
    if value.endswith("}"):
        brace = value.find("{")
        if brace > 0 and value[:brace].rstrip().endswith('"'):
            try:
                end = transform_helpers.find_matching(value, brace, "{", "}")
            except ValueError:
                return None
            if value[end:].strip() == "":
                return value[brace:end].strip()
    return None


# Resolving one trim asks the same few hundred part bodies for their slots
# over and over -- 11,894 calls over 435 distinct bodies for the etk800's 29
# trims -- and each one re-masks the comments and re-scans the text. The
# answer depends on nothing but the body, so it is remembered.
_SLOT_DEF_CACHE: dict[str, tuple[SlotDef, ...]] = {}
# A vehicle's whole part set is a few hundred bodies; the limit is only there
# so a long session across many vehicles cannot grow without bound. Clearing
# outright beats evicting one at a time: the next resolve refills what it
# needs, and part bodies come in whole-vehicle sets rather than singly.
_SLOT_DEF_CACHE_LIMIT = 4096


def extract_slot_defs(part_body: str) -> list[SlotDef]:
    cached = _SLOT_DEF_CACHE.get(part_body)
    if cached is None:
        cached = tuple(_parse_slot_defs(part_body))
        if len(_SLOT_DEF_CACHE) >= _SLOT_DEF_CACHE_LIMIT:
            _SLOT_DEF_CACHE.clear()
        _SLOT_DEF_CACHE[part_body] = cached
    # A fresh list each time: SlotDef is frozen, but callers have always been
    # handed a list of their own and one of them may yet append to it.
    return list(cached)


def _parse_slot_defs(part_body: str) -> list[SlotDef]:
    out: list[SlotDef] = []
    seen: set[str] = set()

    slots = transform_helpers.extract_named_array(part_body, "slots")
    if slots:
        for row in iter_active_top_level_rows(slots):
            values = split_top_level_values(row)
            if len(values) < 2:
                continue
            slot_type = quoted_string_value(values[0])
            default_part = quoted_string_value(values[1])
            if not slot_type or slot_type in {"type", "name"} or default_part is None:
                continue
            # v1 slot: a part fits iff its slotType equals this slot's type.
            out.append(
                SlotDef(
                    slot_type,
                    default_part,
                    trailing_options_object(values),
                    allow_types=(slot_type,),
                )
            )
            seen.add(slot_type)

    slots2 = transform_helpers.extract_named_array(part_body, "slots2")
    if slots2:
        for row in iter_active_top_level_rows(slots2):
            values = split_top_level_values(row)
            if len(values) < 4:
                continue
            slot_type = quoted_string_value(values[0])
            default_part = quoted_string_value(values[3])
            if not slot_type or slot_type in {"type", "name"} or default_part is None or slot_type in seen:
                continue
            # slots2 row: ["name", "allowTypes", "denyTypes", "default", ...].
            allow_types = tuple(re.findall(r'"((?:[^"\\]|\\.)*)"', values[1]))
            deny_types = tuple(re.findall(r'"((?:[^"\\]|\\.)*)"', values[2]))
            out.append(
                SlotDef(
                    slot_type,
                    default_part,
                    trailing_options_object(values),
                    allow_types=allow_types,
                    deny_types=deny_types,
                )
            )
            seen.add(slot_type)

    return out


def part_fits_slot(part_slot_types: Iterable[str], slot: SlotDef) -> bool:
    """Whether a part with these slotTypes may fill this slot, mirroring jbeam
    slotSystem.partFitsSlot: it must match one of the slot's allow_types and
    none of its deny_types. A part declares one or more slotTypes; the vehicle
    author-side .pc is normally valid, so this only bites on invalid/mod configs
    (a mismatched pick is reset to the slot default, as the engine does)."""
    types = set(part_slot_types)
    if not types:
        return False
    allow = set(slot.allow_types) or {slot.slot_type}
    if types.isdisjoint(allow):
        return False
    return not (slot.deny_types and not types.isdisjoint(slot.deny_types))


def vector_from_row(
    row: str, key: str, variables: dict[str, float] | None = None
) -> tuple[float, float, float] | None:
    """A row's "pos"/"rot"/"scale" vector, one component at a time.

    Some stock content makes a component a jbeam variable expression instead
    of a literal -- e.g. the D-Series heavy hub's spacer:
    "pos":{"x":"$=-$trackoffset_F-0.885", "y":-1.463, "z":0.46}. Reading x/y/z
    independently via object_number_property (which resolves the expression when
    a variable scope is supplied, else falls back to approximate_expression_number)
    means a literal y and z are recovered even when x is an expression. The
    previous all-three-or-nothing regex discarded the whole vector in that case
    -- not just the unparseable x, but the entirely literal y and z along with
    it -- and any node_translation_offset that reads this vector's sign lost the
    L/R distinction it exists to make.

    variables is passed by the config-specific preview/position readers so
    expressions resolve to real values; the build/rewrite path leaves it None so
    it keeps working on the authored expression text (it must not bake a variable
    value into rewritten jbeam)."""
    match = re.search(rf'"{re.escape(key)}"\s*:\s*\{{', row)
    if match is None:
        return None
    brace = row.rfind("{", match.start(), match.end())
    try:
        end = transform_helpers.find_matching(row, brace, "{", "}")
    except ValueError:
        return None
    object_text = row[brace:end]
    x = object_number_property(object_text, "x", variables)
    y = object_number_property(object_text, "y", variables)
    z = object_number_property(object_text, "z", variables)
    if x is None or y is None or z is None:
        return None
    return (x, y, z)


def prop_row_position(
    row: str,
    node_positions: dict[str, tuple[float, float, float]],
    inherited_options: Iterable[str] = (),
) -> tuple[float, float, float] | None:
    global_translation = vector_from_row(row, "baseTranslationGlobal")
    if global_translation is not None:
        return pos_after_node_transforms(row, global_translation, inherited_options)

    strings = re.findall(r'"((?:[^"\\]|\\.)*)"', row)
    if len(strings) < 5:
        return None
    func, mesh, id_ref = strings[:3]
    if func == "func" or mesh == "mesh":
        return None
    ref_pos = node_positions.get(id_ref)
    if ref_pos is None:
        return None
    local_translation = vector_from_row(row, "baseTranslation") or (0.0, 0.0, 0.0)
    if len(strings) < 5:
        return ref_pos
    id_x, id_y = strings[3], strings[4]
    x_pos = node_positions.get(id_x)
    y_pos = node_positions.get(id_y)
    if x_pos is None or y_pos is None:
        position = (
            ref_pos[0] + local_translation[0],
            ref_pos[1] + local_translation[1],
            ref_pos[2] + local_translation[2],
        )
        return pos_after_node_transforms(row, position, inherited_options)

    axis_x = vector_subtract(x_pos, ref_pos)
    axis_y = vector_subtract(y_pos, ref_pos)
    axis_z = normalize_vector(cross_product(axis_y, axis_x))
    position = (
        ref_pos[0] + axis_x[0] * local_translation[0] + axis_y[0] * local_translation[1] + axis_z[0] * local_translation[2],
        ref_pos[1] + axis_x[1] * local_translation[0] + axis_y[1] * local_translation[1] + axis_z[1] * local_translation[2],
        ref_pos[2] + axis_x[2] * local_translation[0] + axis_y[2] * local_translation[1] + axis_z[2] * local_translation[2],
    )
    return pos_after_node_transforms(row, position, inherited_options)


def part_information_name(part_body: str) -> str | None:
    info = transform_helpers.extract_keyed_object(part_body, "information")
    if not info:
        return None
    match = re.search(r'"name"\s*:\s*"((?:[^"\\]|\\.)*)"', info)
    return match.group(1) if match else None


def collect_prop_only_objects(
    jbeam_texts: dict[str, str],
    node_positions: dict[str, tuple[float, float, float]],
    existing_objects: dict[str, DaeObject],
    part_body_index: dict[str, tuple[str, str]],
) -> tuple[dict[str, DaeObject], dict[str, dict[str, object]]]:
    positions: dict[str, list[tuple[float, float, float]]] = {}
    labels: dict[str, str] = {}
    for part_body, _filename in part_body_index.values():
        props = transform_helpers.extract_named_array(part_body, "props")
        if not props:
            continue
        part_name = part_information_name(part_body)
        for row in iter_top_level_rows(props):
            strings = re.findall(r'"((?:[^"\\]|\\.)*)"', row)
            if len(strings) < 2:
                continue
            func, mesh = strings[:2]
            if func == "func" or mesh == "mesh" or mesh in existing_objects:
                continue
            position = prop_row_position(row, node_positions)
            if position is not None:
                positions.setdefault(mesh, []).append(position)
                if part_name:
                    labels.setdefault(mesh, part_name)

    objects: dict[str, DaeObject] = {}
    previews: dict[str, dict[str, object]] = {}
    for mesh, mesh_positions in positions.items():
        if not mesh_positions:
            continue
        x = sum(pos[0] for pos in mesh_positions) / len(mesh_positions)
        y = sum(pos[1] for pos in mesh_positions) / len(mesh_positions)
        z = sum(pos[2] for pos in mesh_positions) / len(mesh_positions)
        objects[mesh] = DaeObject(
            id=mesh,
            name=labels.get(mesh, mesh),
            dae_path="",
            x=x,
            y=y,
            z=z,
            geometry_ids=(),
        )
        pad = 0.035
        previews[mesh] = {
            "bounds": ((x - pad, y - pad, z - pad), (x + pad, y + pad, z + pad)),
            "center": (x, y, z),
            "sample_points": [(x, y, z)],
            "geometry_ids": [],
        }
    return objects, previews


def flexbody_row_mesh(row: str) -> str | None:
    match = re.match(r'\s*\[\s*"((?:[^"\\]|\\.)*)"', row)
    if match is None:
        return None
    mesh = match.group(1)
    return None if mesh == "mesh" else mesh


NODE_TRANSFORM_KEY_RE = re.compile(r'"(?P<key>node(?:Rotate|Offset|Move)(?P<index>\d*)?)"\s*:')


def approximate_expression_number(value: str) -> float | None:
    text = value.strip()
    text = text.removeprefix("$=")
    try:
        return float(text)
    except ValueError:
        pass
    constants = [float(match.group(0)) for match in re.finditer(NUMBER_RE, text)]
    if not constants:
        return None
    return sum(constants)


# Functions BeamNG's expressionParser exposes to $= expressions (math.* is
# flattened into the context, plus a few helpers). Enough to cover stock jbeam;
# anything referencing something outside this set falls back to the constant-sum
# approximation.
_EXPR_NAMESPACE: dict[str, object] = {
    "abs": abs, "min": min, "max": max, "round": round, "pow": pow,
    "sqrt": math.sqrt, "exp": math.exp, "log": math.log,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan, "atan2": math.atan2,
    "floor": math.floor, "ceil": math.ceil, "fmod": math.fmod,
    "rad": math.radians, "deg": math.degrees, "pi": math.pi, "huge": math.inf,
    "square": lambda v: v * v,
    "sign": lambda v: (v > 0) - (v < 0),
    "clamp": lambda v, lo, hi: max(lo, min(hi, v)),
    "smoothstep": lambda v: v * v * (3 - 2 * v),
    "nil": None,
}

# Matches $variable and dotted $components.path.field references. Variable names
# may start with a digit (us_semi's $5wheelPos fifth-wheel slide), so the first
# char after $ allows digits too. The dotted form only occurs for $components in
# stock jbeam; ordinary variables have no dots, so including '.' is safe and lets
# a whole component path resolve as one token from the (flattened) scope.
_EXPR_VAR_RE = re.compile(r"\$[A-Za-z0-9_][A-Za-z0-9_.]*")


# BEAMXP_PART_INSTANCE_FIX_V1: typed, side-effect-free evaluation for the
# geometry-affecting JBeam expression subset. This adds string concatenation
# and namespace variables without exposing Python eval() to mod archives.
def _lua_truthy(value: object) -> bool:
    return value is not None and value is not False


def _split_top_level_lua_concat(expression: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    quote = ""
    escaped = False
    index = 0
    while index < len(expression):
        char = expression[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            index += 1
            continue
        if char in "([{":
            depth += 1
            index += 1
            continue
        if char in ")]}":
            depth = max(0, depth - 1)
            index += 1
            continue
        if depth == 0 and expression.startswith("..", index):
            parts.append(expression[start:index].strip())
            index += 2
            start = index
            continue
        index += 1
    if parts:
        parts.append(expression[start:].strip())
        return parts
    return [expression.strip()]


def _safe_jbeam_ast_eval(node: ast.AST, values: dict[str, object]) -> object:
    if isinstance(node, ast.Expression):
        return _safe_jbeam_ast_eval(node.body, values)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in values:
            return values[node.id]
        if node.id in _EXPR_NAMESPACE and not callable(_EXPR_NAMESPACE[node.id]):
            return _EXPR_NAMESPACE[node.id]
        raise ValueError(f"unknown name {node.id}")
    if isinstance(node, ast.UnaryOp):
        value = _safe_jbeam_ast_eval(node.operand, values)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return +value
        if isinstance(node.op, ast.Not):
            return not _lua_truthy(value)
        raise ValueError("unsupported unary operator")
    if isinstance(node, ast.BinOp):
        left = _safe_jbeam_ast_eval(node.left, values)
        right = _safe_jbeam_ast_eval(node.right, values)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.Pow):
            return left ** right
        raise ValueError("unsupported binary operator")
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            result: object = True
            for child in node.values:
                result = _safe_jbeam_ast_eval(child, values)
                if not _lua_truthy(result):
                    return result
            return result
        if isinstance(node.op, ast.Or):
            result: object = None
            for child in node.values:
                result = _safe_jbeam_ast_eval(child, values)
                if _lua_truthy(result):
                    return result
            return result
        raise ValueError("unsupported boolean operator")
    if isinstance(node, ast.Compare):
        left = _safe_jbeam_ast_eval(node.left, values)
        for operator, comparator in zip(node.ops, node.comparators):
            right = _safe_jbeam_ast_eval(comparator, values)
            if isinstance(operator, ast.Eq):
                ok = left == right
            elif isinstance(operator, ast.NotEq):
                ok = left != right
            elif isinstance(operator, ast.Lt):
                ok = left < right
            elif isinstance(operator, ast.LtE):
                ok = left <= right
            elif isinstance(operator, ast.Gt):
                ok = left > right
            elif isinstance(operator, ast.GtE):
                ok = left >= right
            else:
                raise TypeError("unsupported comparison")
            if not ok:
                return False
            left = right
        return True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        args = [_safe_jbeam_ast_eval(arg, values) for arg in node.args]
        if node.func.id == "case":
            if len(args) != 3:
                raise ValueError("case expects three arguments")
            return args[1] if _lua_truthy(args[0]) else args[2]
        function = _EXPR_NAMESPACE.get(node.func.id)
        if not callable(function):
            raise TypeError(f"unsupported function {node.func.id}")
        return function(*args)
    raise TypeError(f"unsupported expression node {type(node).__name__}")


def _evaluate_lua_expression(expression: str, variables: dict[str, object]) -> object:
    concat = _split_top_level_lua_concat(expression)
    if len(concat) > 1:
        values: list[object] = []
        for part in concat:
            value = _evaluate_lua_expression(part, variables)
            if value is None:
                raise ValueError("nil cannot be concatenated")
            values.append(value)
        return "".join(str(value) for value in values)

    replacements: dict[str, object] = {}

    def substitute(match: re.Match[str]) -> str:
        key = match.group(0)
        placeholder = f"_jv_{len(replacements)}"
        replacements[placeholder] = variables.get(key)
        return placeholder

    python_expression = _EXPR_VAR_RE.sub(substitute, expression)
    python_expression = python_expression.replace("~=", "!=").replace("^", "**")
    python_expression = re.sub(r"\bnil\b", "None", python_expression)
    python_expression = re.sub(r"\btrue\b", "True", python_expression, flags=re.IGNORECASE)
    python_expression = re.sub(r"\bfalse\b", "False", python_expression, flags=re.IGNORECASE)
    tree = ast.parse(python_expression, mode="eval")
    return _safe_jbeam_ast_eval(tree, replacements)


def resolve_jbeam_value(value: str, variables: dict[str, object] | None = None) -> object:
    text = value.strip()
    variables = variables or {}
    if text.startswith("$."):
        prefix = variables.get("$prefix")
        suffix = variables.get("$suffix")
        prefix_text = "" if prefix is None else str(prefix)
        suffix_text = "" if suffix is None else str(suffix)
        return f"{prefix_text}{text[2:]}{suffix_text}"
    if not text.startswith("$"):
        return value
    return evaluate_jbeam_expression(text, variables)


_JBEAM_STRING_TOKEN_RE = re.compile(r'"(?:[^"\\]|\\.)*"')


def resolve_jbeam_row_strings(
    row: str,
    variables: dict[str, object] | None = None,
) -> str:
    """Resolve dynamic strings/numbers inside one JBeam table row."""
    variables = variables or {}

    def replace(match: re.Match[str]) -> str:
        raw = match.group(0)
        # A quoted token followed by a colon is an object key, not a value.
        # Slot-variable names such as "$prefix" must remain keys.
        if re.match(r"\s*:", row[match.end():]):
            return raw
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        if not isinstance(decoded, str) or not decoded.startswith("$"):
            return raw
        try:
            resolved = resolve_jbeam_value(decoded, variables)
        except Exception:
            return raw
        if resolved is None:
            return raw
        return json.dumps(resolved, ensure_ascii=False)

    return _JBEAM_STRING_TOKEN_RE.sub(replace, row)

def evaluate_jbeam_expression(
    value: str, variables: dict[str, object] | None
) -> object:
    """Resolve a direct variable or a side-effect-free JBeam expression."""
    text = value.strip()
    if not text.startswith("$"):
        return None
    variables = variables or {}
    if not text.startswith("$="):
        return variables.get(text)
    try:
        return _evaluate_lua_expression(text[2:].strip(), variables)
    except Exception:
        return None



def expression_number(
    value: str, variables: dict[str, object] | None = None
) -> float | None:
    """Resolve a numeric JBeam value, retaining the existing approximation only
    when exact typed evaluation is unavailable."""
    if variables is not None and value.strip().startswith("$"):
        resolved = evaluate_jbeam_expression(value, variables)
        if isinstance(resolved, (int, float)) and not isinstance(resolved, bool):
            return float(resolved)
    return approximate_expression_number(value)



def parse_part_variable_defs(part_body: str) -> dict[str, tuple[float, float, float]]:
    """{'$name': (default, min, max)} for each range variable a part declares.

    Columns are read by the section header (``["name","type",...,"default",
    "min","max",...]``) rather than fixed positions, since parts vary the extra
    trailing columns. Non-numeric defaults/min/max are skipped."""
    array = transform_helpers.extract_named_array(part_body, "variables")
    if not array:
        return {}
    rows = [split_top_level_values(row) for row in iter_active_top_level_rows(array)]
    if not rows:
        return {}
    header = [quoted_string_value(v) for v in rows[0]]
    try:
        i_name = header.index("name")
        i_def = header.index("default")
        i_min = header.index("min")
        i_max = header.index("max")
    except ValueError:
        return {}
    out: dict[str, tuple[float, float, float]] = {}
    for row in rows[1:]:
        if len(row) <= max(i_name, i_def, i_min, i_max):
            continue
        name = quoted_string_value(row[i_name])
        if not name or not name.startswith("$"):
            continue
        try:
            out[name] = (
                float(approximate_expression_number(row[i_def]) if "$" in row[i_def] else row[i_def]),
                float(row[i_min]),
                float(row[i_max]),
            )
        except (TypeError, ValueError):
            continue
    return out


def parse_slot_variable_overrides(options_json: str | None) -> dict[str, str]:
    """A slot option object's ``"variables"`` map ({'$name': value_or_expr}),
    which sets variable values for the slot's subtree."""
    if not options_json:
        return {}
    try:
        parsed = parse_beamng_json(options_json, label="slot options")
    except Exception:
        return {}
    variables = parsed.get("variables") if isinstance(parsed, dict) else None
    if not isinstance(variables, dict):
        return {}
    return {str(k): v for k, v in variables.items() if str(k).startswith("$")}


def _deep_merge_into(target: dict, source: dict) -> None:
    """Recursively merge source into target (later source wins), like jbeam's
    unifyComponents accumulation of the components tree across parts."""
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge_into(target[key], value)
        else:
            target[key] = value


def collect_components(
    parts: Iterable[str],
    jbeam_texts: dict[str, str],
    part_body_index: dict[str, tuple[str, str]] | None,
) -> dict[str, object]:
    """Merged ``components`` tree across the selected parts.

    jbeam parts define a nested ``components`` dict (e.g. a wheel's
    ``dualyOffsets_R`` carrying offsetInner/offsetOuter); references like
    ``$components.dualyOffsets_R.offsetInner`` read from the accumulation of all
    selected parts' components. Merged in tree order so a deeper part overrides.
    """
    order = list(parts)
    merged: dict[str, object] = {}
    for part_id in order:
        found = find_part_body(part_id, jbeam_texts, part_body_index)
        if found is None:
            continue
        raw = transform_helpers.extract_keyed_object(found[0], "components")
        if not raw:
            continue
        # extract_keyed_object returns the `"components": {...}` pair; wrap it so
        # the tolerant parser sees a document, then take the value.
        try:
            parsed = parse_beamng_json("{" + raw + "}", label=f"{part_id} components")
        except Exception:
            continue
        section = parsed.get("components")
        if isinstance(section, dict):
            _deep_merge_into(merged, section)
    return merged


def flatten_component_values(
    components: dict[str, object], variables: dict[str, float]
) -> dict[str, float]:
    """Flatten the components tree to ``$components.a.b.c -> number`` entries,
    evaluating expression-valued leaves against ``variables``. Non-numeric,
    unresolvable leaves are dropped (callers fall back to the approximation)."""
    out: dict[str, float] = {}

    def walk(prefix: str, value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                walk(f"{prefix}.{key}", child)
        elif isinstance(value, bool):
            return
        elif isinstance(value, (int, float)):
            out[prefix] = float(value)
        elif isinstance(value, str):
            resolved = expression_number(value, variables)
            if resolved is not None:
                out[prefix] = resolved

    walk("$components", components)
    return out


def build_part_variable_scopes(
    parts: Iterable[str],
    part_slot_options: dict[str, tuple[str, ...]],
    jbeam_texts: dict[str, str],
    part_body_index: dict[str, tuple[str, str]] | None,
    user_vars: dict[str, object],
) -> dict[str, dict[str, float]]:
    """The resolved variable value map in force inside each selected part.

    Mirrors jbeam's variable pipeline for the geometry that needs it: every
    selected part contributes its variable DEFINITIONS (default/min/max); the
    effective value is the .pc's user value if given, else the default, clamped
    to range. A slot may then OVERRIDE a variable for its subtree (slot
    "variables"), so each part's scope re-applies the overrides collected along
    its slot path (root -> part order, deepest wins), evaluating override
    expressions against the scope built so far.
    """
    parts = list(parts)
    defaults: dict[str, tuple[float, float, float]] = {}
    for part_id in parts:
        found = find_part_body(part_id, jbeam_texts, part_body_index)
        if found is not None:
            # first definition wins; duplicate variable names across parts are
            # expected to agree on range, as in stock content
            for name, spec in parse_part_variable_defs(found[0]).items():
                defaults.setdefault(name, spec)

    def resolved_default(name: str, spec: tuple[float, float, float]) -> float:
        default, lo, hi = spec
        raw = user_vars.get(name)
        value = float(raw) if isinstance(raw, (int, float)) and not isinstance(raw, bool) else default
        return clamp_value(value, lo, hi)

    base = {name: resolved_default(name, spec) for name, spec in defaults.items()}

    # Fold the merged components tree in as $components.a.b.c entries, so
    # expressions like $=$components.dualyOffsets_R.offsetInner+0.814 (the
    # us_semi dual-wheel spacing) resolve instead of collapsing to a constant.
    components = collect_components(parts, jbeam_texts, part_body_index)
    base.update(flatten_component_values(components, base))

    scopes: dict[str, dict[str, float]] = {}
    for part_id in parts:
        scope = dict(base)
        for options_json in part_slot_options.get(part_id, ()):  # root -> part order
            for name, raw in parse_slot_variable_overrides(options_json).items():
                if name in user_vars:  # user value always wins over slot override
                    continue
                if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                    value: float | None = float(raw)
                else:
                    value = expression_number(str(raw), scope)
                if value is None:
                    continue
                if name in defaults:
                    _d, lo, hi = defaults[name]
                    value = clamp_value(value, lo, hi)
                scope[name] = value
        scopes[part_id] = scope
    return scopes


# BEAMXP_PART_INSTANCE_FIX_V1: selected state belongs to the slot-tree occurrence,
# not merely to the source part ID.
def build_part_instance_variable_scopes(
    part_instances: Iterable[dict[str, object]],
    jbeam_texts: dict[str, str],
    part_body_index: dict[str, tuple[str, str]] | None,
    user_vars: dict[str, object],
) -> dict[str, dict[str, object]]:
    instances = list(part_instances)
    unique_parts: list[str] = []
    for instance in instances:
        part_id = str(instance.get("part_id") or "")
        if part_id and part_id not in unique_parts:
            unique_parts.append(part_id)

    base_scopes = build_part_variable_scopes(
        unique_parts, {}, jbeam_texts, part_body_index, user_vars
    )
    defaults: dict[str, tuple[float, float, float]] = {}
    for part_id in unique_parts:
        found = find_part_body(part_id, jbeam_texts, part_body_index)
        if found is not None:
            for name, spec in parse_part_variable_defs(found[0]).items():
                defaults.setdefault(name, spec)

    scopes: dict[str, dict[str, object]] = {}
    for instance in instances:
        instance_id = str(instance.get("instance_id") or "")
        part_id = str(instance.get("part_id") or "")
        scope: dict[str, object] = dict(base_scopes.get(part_id, {}))
        for name, raw in user_vars.items():
            name = str(name)
            if name.startswith("$") and isinstance(raw, (str, bool)):
                scope[name] = raw

        raw_options = instance.get("inherited_options", ())
        options = raw_options if isinstance(raw_options, (list, tuple)) else ()
        for options_json in options:
            for name, raw in parse_slot_variable_overrides(str(options_json)).items():
                if name in user_vars:
                    continue
                if isinstance(raw, str) and raw.startswith("$"):
                    resolved = evaluate_jbeam_expression(raw, scope)
                    if resolved is None:
                        continue
                    value: object = resolved
                else:
                    value = raw
                if (
                    name in defaults
                    and isinstance(value, (int, float))
                    and not isinstance(value, bool)
                ):
                    _default, low, high = defaults[name]
                    value = clamp_value(float(value), low, high)
                scope[name] = value
        scopes[instance_id] = scope
    return scopes

def object_number_property(
    object_text: str, key: str, variables: dict[str, float] | None = None
) -> float | None:
    match = re.search(
        rf'"{re.escape(key)}"\s*:\s*(?P<value>{NUMBER_RE}|"(?:[^"\\]|\\.)*")',
        object_text,
    )
    if not match:
        return None
    raw = match.group("value")
    if raw.startswith('"'):
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            decoded = raw.strip('"')
        return expression_number(str(decoded), variables)
    return float(raw)


def node_transform_kind(key: str) -> tuple[str, int] | None:
    for prefix in ("nodeRotate", "nodeOffset", "nodeMove"):
        if key.startswith(prefix):
            suffix = key[len(prefix) :]
            if suffix and not suffix.isdigit():
                return None
            return prefix, int(suffix or 0)
    return None


def node_transform_ops(
    texts: Iterable[str], variables: dict[str, float] | None = None
) -> dict[tuple[str, int], dict[str, float]]:
    ops: dict[tuple[str, int], dict[str, float]] = {}
    for text in texts:
        for match in NODE_TRANSFORM_KEY_RE.finditer(text):
            parsed_key = node_transform_kind(match.group("key"))
            if parsed_key is None:
                continue
            idx = match.end()
            while idx < len(text) and text[idx].isspace():
                idx += 1
            if idx >= len(text) or text[idx] != "{":
                ops.pop(parsed_key, None)
                continue
            try:
                end = transform_helpers.find_matching(text, idx, "{", "}")
            except ValueError:
                ops.pop(parsed_key, None)
                continue
            object_text = text[idx:end]
            x = object_number_property(object_text, "x", variables)
            y = object_number_property(object_text, "y", variables)
            z = object_number_property(object_text, "z", variables)
            if x is None and y is None and z is None:
                ops.pop(parsed_key, None)
                continue
            # Merge component-wise onto any inherited value for this same
            # transform, mirroring jbeam slotSystem's tableMerge: a child slot's
            # option overrides the parent's for the components it specifies and
            # leaves the rest intact, rather than replacing the whole vector.
            # (texts arrive parent-first, then the node row, so later specified
            # components win.) Unspecified components stay absent and default to
            # 0 at read time.
            op = dict(ops.get(parsed_key, {}))
            if x is not None:
                op["x"] = x
            if y is not None:
                op["y"] = y
            if z is not None:
                op["z"] = z
            for pivot_key in ("px", "py", "pz"):
                pivot_value = object_number_property(object_text, pivot_key, variables)
                if pivot_value is not None:
                    op[pivot_key] = pivot_value
            ops[parsed_key] = op
    return ops


def node_op_indices(ops: dict[tuple[str, int], dict[str, float]]) -> range:
    if not ops:
        return range(0)
    indices = [idx for _kind, idx in ops]
    return range(min(indices), max(indices) + 1)


def has_node_rotations(ops: dict[tuple[str, int], dict[str, float]]) -> bool:
    return any(kind == "nodeRotate" for kind, _idx in ops)


def node_translation_offset(
    ops: dict[tuple[str, int], dict[str, float]],
    pos_x_sign: int,
) -> tuple[float, float, float]:
    x = y = z = 0.0
    for idx in node_op_indices(ops):
        offset = ops.get(("nodeOffset", idx))
        if offset is not None:
            x += pos_x_sign * offset.get("x", 0.0)
            y += offset.get("y", 0.0)
            z += offset.get("z", 0.0)
        move = ops.get(("nodeMove", idx))
        if move is not None:
            x += move.get("x", 0.0)
            y += move.get("y", 0.0)
            z += move.get("z", 0.0)
    return x, y, z


def matrix4_from_matrix3(rotation: list[list[float]]) -> list[list[float]]:
    matrix = identity_matrix()
    for row in range(3):
        for col in range(3):
            matrix[row][col] = rotation[row][col]
    return matrix


def inverse_affine_matrix(matrix: list[list[float]]) -> list[list[float]]:
    inverse3 = transform_helpers.inverse_3x3(matrix)
    tx, ty, tz = matrix[0][3], matrix[1][3], matrix[2][3]
    out = identity_matrix()
    for row in range(3):
        for col in range(3):
            out[row][col] = inverse3[row][col]
        out[row][3] = -(inverse3[row][0] * tx + inverse3[row][1] * ty + inverse3[row][2] * tz)
    return out


def node_transform_matrix(
    ops: dict[tuple[str, int], dict[str, float]],
    pos_x: float,
) -> list[list[float]]:
    matrix = identity_matrix()
    pos_x_sign = sign_number(pos_x)
    for idx in node_op_indices(ops):
        rotation = ops.get(("nodeRotate", idx))
        if rotation is not None:
            rotation_matrix = matrix4_from_matrix3(
                euler_matrix3(
                    (-rotation.get("x", 0.0), -rotation.get("y", 0.0), -rotation.get("z", 0.0))
                )
            )
            if any(key in rotation for key in ("px", "py", "pz")):
                pivot = (
                    rotation.get("px", 0.0),
                    rotation.get("py", 0.0),
                    rotation.get("pz", 0.0),
                )
                rotation_matrix = multiply_matrix(
                    multiply_matrix(translation_matrix(pivot), rotation_matrix),
                    translation_matrix((-pivot[0], -pivot[1], -pivot[2])),
                )
            matrix = multiply_matrix(matrix, rotation_matrix)

        offset = ops.get(("nodeOffset", idx))
        if offset is not None:
            matrix = multiply_matrix(
                matrix,
                translation_matrix(
                    (pos_x_sign * offset.get("x", 0.0), offset.get("y", 0.0), offset.get("z", 0.0))
                ),
            )

        move = ops.get(("nodeMove", idx))
        if move is not None:
            matrix = multiply_matrix(
                matrix,
                translation_matrix(
                    (move.get("x", 0.0), move.get("y", 0.0), move.get("z", 0.0))
                ),
            )
    return matrix


def node_transform_source_texts(
    row: str,
    inherited_options: Iterable[str] = (),
) -> list[str]:
    return [text for text in [*inherited_options, row] if text]


def pos_after_node_transforms(
    row: str,
    position: tuple[float, float, float],
    inherited_options: Iterable[str] = (),
    variables: dict[str, float] | None = None,
) -> tuple[float, float, float]:
    ops = node_transform_ops(node_transform_source_texts(row, inherited_options), variables)
    if not ops:
        return position
    if not has_node_rotations(ops):
        dx, dy, dz = node_translation_offset(ops, sign_number(position[0]))
        return position[0] + dx, position[1] + dy, position[2] + dz
    return transform_helpers.transform_point(node_transform_matrix(ops, position[0]), position)


def pos_before_node_transforms(
    row: str,
    position: tuple[float, float, float],
    inherited_options: Iterable[str] = (),
    variables: dict[str, float] | None = None,
) -> tuple[float, float, float]:
    ops = node_transform_ops(node_transform_source_texts(row, inherited_options), variables)
    if not ops:
        return position

    if not has_node_rotations(ops):
        fallback = position
        for pos_x_sign in (1, 0, -1):
            dx, dy, dz = node_translation_offset(ops, pos_x_sign)
            candidate = (position[0] - dx, position[1] - dy, position[2] - dz)
            fallback = candidate
            if sign_number(candidate[0]) == pos_x_sign:
                return candidate
        return fallback

    fallback = position
    for pos_x_sign in (1, 0, -1):
        matrix = node_transform_matrix(ops, float(pos_x_sign))
        candidate = transform_helpers.transform_point(inverse_affine_matrix(matrix), position)
        fallback = candidate
        if sign_number(candidate[0]) == pos_x_sign:
            return candidate
    return fallback


def pos_rot_before_node_transforms(
    row: str,
    position: tuple[float, float, float],
    rotation: tuple[float, float, float],
    inherited_options: Iterable[str] = (),
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    ops = node_transform_ops(node_transform_source_texts(row, inherited_options))
    if not ops:
        return position, rotation
    if not has_node_rotations(ops):
        return pos_before_node_transforms(row, position, inherited_options), rotation

    fallback_pos = position
    fallback_rot = rotation
    for pos_x_sign in (1, 0, -1):
        inverse = inverse_affine_matrix(node_transform_matrix(ops, float(pos_x_sign)))
        candidate_pos = transform_helpers.transform_point(inverse, position)
        inverse_rotation = matrix3_from_matrix4(inverse)
        neg_rotation = euler_matrix3((-rotation[0], -rotation[1], -rotation[2]))
        candidate_neg_rotation = multiply_matrix3(neg_rotation, inverse_rotation)
        euler = euler_from_matrix3(candidate_neg_rotation)
        candidate_rot = (-euler[0], -euler[1], -euler[2])
        fallback_pos, fallback_rot = candidate_pos, candidate_rot
        if sign_number(candidate_pos[0]) == pos_x_sign:
            return candidate_pos, candidate_rot
    return fallback_pos, fallback_rot


def prop_row_global_rotation_matrix(
    row: str,
    node_positions: dict[str, tuple[float, float, float]],
) -> list[list[float]] | None:
    global_rotation = vector_from_row(row, "baseRotationGlobal")
    if global_rotation is not None:
        return base_rotation_global_matrix3(global_rotation)

    strings = re.findall(r'"((?:[^"\\]|\\.)*)"', row)
    if len(strings) < 5:
        return None
    vectors = prop_row_vector_objects(row)
    if not vectors:
        return None

    ref_pos = node_positions.get(strings[2])
    x_pos = node_positions.get(strings[3])
    y_pos = node_positions.get(strings[4])
    if ref_pos is None or x_pos is None or y_pos is None:
        return None

    axis_x = normalize_vector(vector_subtract(x_pos, ref_pos))
    axis_y_seed = normalize_vector(vector_subtract(y_pos, ref_pos))
    axis_z = normalize_vector(cross_product(axis_y_seed, axis_x))
    if axis_x == (0.0, 0.0, 0.0) or axis_z == (0.0, 0.0, 0.0):
        return None
    axis_y = normalize_vector(cross_product(axis_x, axis_z))
    if axis_y == (0.0, 0.0, 0.0):
        return None

    frame = matrix3_from_axes(axis_x, axis_y, axis_z)
    return multiply_matrix3(frame, prop_base_rotation_matrix3(vectors[0]))


def mirrored_prop_global_rotation(
    row: str,
    node_positions: dict[str, tuple[float, float, float]],
) -> tuple[float, float, float] | None:
    rotation = prop_row_global_rotation_matrix(row, node_positions)
    if rotation is None:
        return None
    return euler_yzx_from_matrix3(mirror_rotation_matrix_x(rotation))


def prop_frame_axes(
    row: str,
    node_positions: dict[str, tuple[float, float, float]],
) -> tuple[tuple[float, float, float], ...] | None:
    strings = re.findall(r'"((?:[^"\\]|\\.)*)"', row)
    if len(strings) < 5:
        return None
    ref_pos = node_positions.get(strings[2])
    x_pos = node_positions.get(strings[3])
    y_pos = node_positions.get(strings[4])
    if ref_pos is None or x_pos is None or y_pos is None:
        return None
    axis_x = normalize_vector(vector_subtract(x_pos, ref_pos))
    seed = normalize_vector(vector_subtract(y_pos, ref_pos))
    axis_z = normalize_vector(cross_product(seed, axis_x))
    if axis_x == (0.0, 0.0, 0.0) or axis_z == (0.0, 0.0, 0.0):
        return None
    axis_y = normalize_vector(cross_product(axis_x, axis_z))
    if axis_y == (0.0, 0.0, 0.0):
        return None
    return ref_pos, axis_x, axis_y, axis_z


def prop_row_pivot_position(
    row: str,
    node_positions: dict[str, tuple[float, float, float]],
    pivot: tuple[float, float, float] | None,
    inherited_options: Iterable[str] = (),
    variables: dict[str, float] | None = None,
) -> tuple[float, float, float] | None:
    """World rest position of the prop mesh's DAE pivot.

    Engine rule, verified against an in-game dump of getBaseTranslationGlobal()
    for every prop of the stock sunburst2 rally config:
      1. row has baseTranslationGlobal -> that value verbatim;
      2. row has baseTranslation      -> refNode + normalizedFrame * baseTranslation
                                         (the mesh pivot contributes nothing);
      3. neither                      -> the mesh's authored DAE pivot (identity rest).
    Hand conversion must mirror/translate this position.
    """
    global_translation = vector_from_row(row, "baseTranslationGlobal", variables)
    if global_translation is not None:
        return pos_after_node_transforms(row, global_translation, inherited_options, variables)

    base_translation = vector_from_row(row, "baseTranslation", variables)
    if base_translation is None:
        if pivot is None:
            return None
        return pos_after_node_transforms(row, pivot, inherited_options, variables)

    frame = prop_frame_axes(row, node_positions)
    if frame is None:
        return None
    ref_pos, axis_x, axis_y, axis_z = frame
    return (
        ref_pos[0] + axis_x[0] * base_translation[0] + axis_y[0] * base_translation[1] + axis_z[0] * base_translation[2],
        ref_pos[1] + axis_x[1] * base_translation[0] + axis_y[1] * base_translation[1] + axis_z[1] * base_translation[2],
        ref_pos[2] + axis_x[2] * base_translation[0] + axis_y[2] * base_translation[1] + axis_z[2] * base_translation[2],
    )


def matrix4_with_rotation_translation(
    rotation: list[list[float]] | None,
    position: tuple[float, float, float],
) -> list[list[float]]:
    matrix = identity_matrix()
    if rotation is not None:
        for row in range(3):
            for col in range(3):
                matrix[row][col] = rotation[row][col]
    matrix[0][3], matrix[1][3], matrix[2][3] = position
    return matrix


def prop_row_source_matrix(
    row: str,
    node_positions: dict[str, tuple[float, float, float]],
    inherited_options: Iterable[str] = (),
    rotation_override: list[list[float]] | None = None,
) -> list[list[float]] | None:
    position = prop_row_position(row, node_positions, inherited_options)
    if position is None:
        return None
    rotation = rotation_override
    if rotation is None:
        rotation = prop_row_global_rotation_matrix(row, node_positions)
    return matrix4_with_rotation_translation(rotation, position)


def flexbody_row_matrix(
    row: str, variables: dict[str, float] | None = None
) -> list[list[float]]:
    pos = vector_from_row(row, "pos", variables) or (0.0, 0.0, 0.0)
    rot = vector_from_row(row, "rot", variables) or (0.0, 0.0, 0.0)
    scale = vector_from_row(row, "scale", variables) or (1.0, 1.0, 1.0)
    matrix = translation_matrix(pos)
    # Game flexbody rot euler is "+Z +X +Y intrinsic" (meshs.lua) with the
    # sequence listed innermost-first: Z is applied to the mesh first, then X,
    # then Y, i.e. v = pos + Ry*Rx*Rz*(scale*v). Ground truth: the sunburst2
    # boot spare (rot x:75 z:90) must lie flat under its authored-in-place
    # strap (axis (0,.26,.97)), not stand vertically (axis y); the offroad
    # swing-mount spare (z:90 only) pins the signs as positive. Single-axis
    # rows are order-insensitive, which is why this stayed hidden.
    for next_matrix in (
        rotation_y_matrix(rot[1]),
        rotation_x_matrix(rot[0]),
        rotation_z_matrix(rot[2]),
        scale_matrix(scale),
    ):
        matrix = multiply_matrix(matrix, next_matrix)
    return matrix


def flexbody_row_source_matrix(
    row: str,
    inherited_options: Iterable[str] = (),
    variables: dict[str, float] | None = None,
) -> list[list[float]]:
    matrix = flexbody_row_matrix(row, variables)
    ops = node_transform_ops(node_transform_source_texts(row, inherited_options), variables)
    if not ops:
        return matrix
    if not has_node_rotations(ops):
        pos = vector_from_row(row, "pos", variables) or (0.0, 0.0, 0.0)
        dx, dy, dz = node_translation_offset(ops, sign_number(pos[0]))
        return multiply_matrix(translation_matrix((dx, dy, dz)), matrix)
    pos = vector_from_row(row, "pos", variables) or (0.0, 0.0, 0.0)
    return multiply_matrix(node_transform_matrix(ops, pos[0]), matrix)

__all__ = ['load_resolver_inputs', 'load_common_jbeam_texts', 'slot_demand_types', 'reachable_common_part_index', 'extract_node_positions_from_array', 'build_node_position_index', 'build_part_body_index', 'TOP_LEVEL_STRING_RE', 'iter_top_level_rows', 'iter_active_top_level_rows', 'split_top_level_values', 'quoted_string_value', 'trailing_options_object', 'extract_slot_defs', 'part_fits_slot', 'vector_from_row', 'prop_row_position', 'part_information_name', 'collect_prop_only_objects', 'flexbody_row_mesh', 'NODE_TRANSFORM_KEY_RE', 'approximate_expression_number', '_EXPR_NAMESPACE', '_EXPR_VAR_RE', '_lua_truthy', '_split_top_level_lua_concat', '_safe_jbeam_ast_eval', '_evaluate_lua_expression', 'resolve_jbeam_value', '_JBEAM_STRING_TOKEN_RE', 'resolve_jbeam_row_strings', 'evaluate_jbeam_expression', 'expression_number', 'parse_part_variable_defs', 'parse_slot_variable_overrides', '_deep_merge_into', 'collect_components', 'flatten_component_values', 'build_part_variable_scopes', 'build_part_instance_variable_scopes', 'object_number_property', 'node_transform_kind', 'node_transform_ops', 'node_op_indices', 'has_node_rotations', 'node_translation_offset', 'matrix4_from_matrix3', 'inverse_affine_matrix', 'node_transform_matrix', 'node_transform_source_texts', 'pos_after_node_transforms', 'pos_before_node_transforms', 'pos_rot_before_node_transforms', 'prop_row_global_rotation_matrix', 'mirrored_prop_global_rotation', 'prop_frame_axes', 'prop_row_pivot_position', 'matrix4_with_rotation_translation', 'prop_row_source_matrix', 'flexbody_row_matrix', 'flexbody_row_source_matrix']
