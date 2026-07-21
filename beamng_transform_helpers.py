from __future__ import annotations

import copy
import re
from functools import lru_cache
from xml.etree import ElementTree as ET


NS = {"c": "http://www.collada.org/2005/11/COLLADASchema"}

PROP_FUNC_MESH_RE = re.compile(
    r'(\[\s*"((?:[^"\\]|\\.)*)"\s*(?:,\s*|\s+))"((?:[^"\\]|\\.)*)"(?=\s*(?:,|"))'
)

ET.register_namespace("", NS["c"])


def parse_matrix(text: str) -> list[list[float]]:
    vals = [float(v) for v in text.split()]
    if len(vals) != 16:
        raise ValueError(f"Expected 16 matrix values, got {len(vals)}")
    return [vals[i : i + 4] for i in range(0, 16, 4)]


def format_num(value: float) -> str:
    if abs(value) < 1e-10:
        value = 0.0
    return f"{value:.8g}"


def format_matrix(matrix: list[list[float]]) -> str:
    return " ".join(format_num(v) for row in matrix for v in row)


def mirror_matrix_x(matrix: list[list[float]]) -> list[list[float]]:
    out = [row[:] for row in matrix]
    for col in range(4):
        out[0][col] *= -1
    for row in range(4):
        out[row][0] *= -1
    return out


def translate_matrix_x(matrix: list[list[float]], delta_x: float) -> list[list[float]]:
    out = [row[:] for row in matrix]
    out[0][3] += delta_x
    return out


def source_has_xyz(source: ET.Element) -> bool:
    accessor = source.find(".//c:accessor", NS)
    if accessor is None or accessor.get("stride") != "3":
        return False
    names = [p.get("name", "").upper() for p in accessor.findall("c:param", NS)]
    return names[:3] == ["X", "Y", "Z"]


def mirror_xyz_float_array(source: ET.Element) -> None:
    float_array = source.find("c:float_array", NS)
    if float_array is None or not float_array.text:
        return
    values = [float(v) for v in float_array.text.split()]
    for idx in range(0, len(values), 3):
        values[idx] *= -1
    float_array.text = " ".join(format_num(v) for v in values)


def position_source_ids(mesh: ET.Element) -> set[str]:
    vertices = mesh.find("c:vertices", NS)
    if vertices is None:
        return set()
    out: set[str] = set()
    for input_elem in vertices.findall("c:input", NS):
        if input_elem.get("semantic") != "POSITION":
            continue
        source_url = input_elem.get("source", "")
        if source_url.startswith("#"):
            out.add(source_url[1:])
    return out


def translate_xyz_float_array(source: ET.Element, delta: tuple[float, float, float]) -> None:
    float_array = source.find("c:float_array", NS)
    if float_array is None or not float_array.text:
        return
    values = [float(v) for v in float_array.text.split()]
    dx, dy, dz = delta
    for idx in range(0, len(values), 3):
        values[idx] += dx
        values[idx + 1] += dy
        values[idx + 2] += dz
    float_array.text = " ".join(format_num(v) for v in values)


def inverse_3x3(matrix: list[list[float]]) -> list[list[float]]:
    a = [[matrix[row][col] for col in range(3)] for row in range(3)]
    det = (
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    )
    if abs(det) < 1e-12:
        raise ValueError("Cannot invert transform matrix with near-zero determinant")
    return [
        [
            (
                a[(col + 1) % 3][(row + 1) % 3] * a[(col + 2) % 3][(row + 2) % 3]
                - a[(col + 1) % 3][(row + 2) % 3] * a[(col + 2) % 3][(row + 1) % 3]
            )
            / det
            for col in range(3)
        ]
        for row in range(3)
    ]


def local_delta_for_world_translation(
    matrix: list[list[float]],
    delta: tuple[float, float, float],
) -> tuple[float, float, float]:
    inverse = inverse_3x3(matrix)
    return (
        inverse[0][0] * delta[0] + inverse[0][1] * delta[1] + inverse[0][2] * delta[2],
        inverse[1][0] * delta[0] + inverse[1][1] * delta[1] + inverse[1][2] * delta[2],
        inverse[2][0] * delta[0] + inverse[2][1] * delta[1] + inverse[2][2] * delta[2],
    )


def reverse_vertex_order(primitive: ET.Element, vertex_count: int | None = None) -> None:
    inputs = primitive.findall("c:input", NS)
    if not inputs:
        return
    stride = max(int(inp.get("offset", "0")) for inp in inputs) + 1
    for p in primitive.findall("c:p", NS):
        if not p.text:
            continue
        values = p.text.split()
        out: list[str] = []
        if vertex_count is not None:
            group_width = vertex_count * stride
            for start in range(0, len(values), group_width):
                verts = [
                    values[start + i * stride : start + (i + 1) * stride]
                    for i in range(vertex_count)
                ]
                for vert in reversed(verts):
                    out.extend(vert)
        else:
            verts = [values[i : i + stride] for i in range(0, len(values), stride)]
            for vert in reversed(verts):
                out.extend(vert)
        p.text = " ".join(out)


def reverse_polylist(polylist: ET.Element) -> None:
    inputs = polylist.findall("c:input", NS)
    vcount = polylist.find("c:vcount", NS)
    p = polylist.find("c:p", NS)
    if not inputs or vcount is None or p is None or not vcount.text or not p.text:
        return
    stride = max(int(inp.get("offset", "0")) for inp in inputs) + 1
    values = p.text.split()
    counts = [int(v) for v in vcount.text.split()]
    cursor = 0
    out: list[str] = []
    for count in counts:
        poly_values = values[cursor : cursor + count * stride]
        cursor += count * stride
        verts = [poly_values[i : i + stride] for i in range(0, len(poly_values), stride)]
        for vert in reversed(verts):
            out.extend(vert)
    p.text = " ".join(out)


def texcoord_source_ids(mesh: ET.Element) -> set[str]:
    out: set[str] = set()
    for input_elem in mesh.findall(".//c:input", NS):
        if input_elem.get("semantic") != "TEXCOORD":
            continue
        source_url = input_elem.get("source", "")
        if source_url.startswith("#"):
            out.add(source_url[1:])
    return out


def flip_texcoord_s_float_array(source: ET.Element) -> None:
    """Reflect the S (horizontal) texture coordinate across the source's own
    S range. Reflecting within the existing bounds instead of around 0.5 keeps
    the mesh sampling the exact same texel region of the atlas, so display
    meshes (nav screens, badges) read un-mirrored after a geometric mirror."""
    float_array = source.find("c:float_array", NS)
    accessor = source.find(".//c:accessor", NS)
    if float_array is None or accessor is None or not float_array.text:
        return
    values = [float(v) for v in float_array.text.split()]
    stride = int(accessor.get("stride", "2"))
    offset = int(accessor.get("offset", "0"))
    if stride < 1:
        return
    params = [p.get("name", "").upper() for p in accessor.findall("c:param", NS)]
    s_slot = params.index("S") if "S" in params else 0
    s_indices = list(range(offset + s_slot, len(values), stride))
    if not s_indices:
        return
    pivot = min(values[idx] for idx in s_indices) + max(values[idx] for idx in s_indices)
    for idx in s_indices:
        values[idx] = pivot - values[idx]
    float_array.text = " ".join(format_num(v) for v in values)


def mirrored_geometry(geometry: ET.Element, new_id: str, *, flip_texture: bool = False) -> ET.Element:
    out = copy.deepcopy(geometry)
    old_id = out.get("id")
    out.set("id", new_id)
    if out.get("name"):
        out.set("name", f"{out.get('name')}_rhd")

    mesh = out.find("c:mesh", NS)
    if mesh is None:
        return out

    for source in mesh.findall("c:source", NS):
        if source_has_xyz(source):
            mirror_xyz_float_array(source)
    for triangles in mesh.findall("c:triangles", NS):
        reverse_vertex_order(triangles, vertex_count=3)
    for polylist in mesh.findall("c:polylist", NS):
        reverse_polylist(polylist)
    for polygons in mesh.findall("c:polygons", NS):
        for _p in polygons.findall("c:p", NS):
            reverse_vertex_order(polygons)
            break
    if flip_texture:
        texcoords = texcoord_source_ids(mesh)
        for source in mesh.findall("c:source", NS):
            if source.get("id") in texcoords:
                flip_texcoord_s_float_array(source)

    if old_id:
        for elem in out.iter():
            for attr, value in list(elem.attrib.items()):
                if value == old_id:
                    elem.set(attr, new_id)
                elif value == f"#{old_id}":
                    elem.set(attr, f"#{new_id}")
    return out


def copied_geometry(geometry: ET.Element, new_id: str) -> ET.Element:
    out = copy.deepcopy(geometry)
    old_id = out.get("id")
    out.set("id", new_id)
    if out.get("name"):
        out.set("name", f"{out.get('name')}_rhd")
    if old_id:
        for elem in out.iter():
            for attr, value in list(elem.attrib.items()):
                if value == old_id:
                    elem.set(attr, new_id)
                elif value == f"#{old_id}":
                    elem.set(attr, f"#{new_id}")
    return out


def translated_geometry(
    geometry: ET.Element,
    new_id: str,
    delta: tuple[float, float, float],
) -> ET.Element:
    out = copied_geometry(geometry, new_id)
    mesh = out.find("c:mesh", NS)
    if mesh is None:
        return out
    positions = position_source_ids(mesh)
    for source in mesh.findall("c:source", NS):
        if source.get("id") in positions and source_has_xyz(source):
            translate_xyz_float_array(source, delta)
    return out


def transform_point(
    matrix: list[list[float]],
    point: tuple[float, float, float],
) -> tuple[float, float, float]:
    x, y, z = point
    return (
        matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z + matrix[0][3],
        matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z + matrix[1][3],
        matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z + matrix[2][3],
    )


def source_xyz_points(source: ET.Element) -> list[tuple[float, float, float]]:
    float_array = source.find("c:float_array", NS)
    accessor = source.find(".//c:accessor", NS)
    if float_array is None or accessor is None or not float_array.text:
        return []
    values = [float(v) for v in float_array.text.split()]
    stride = int(accessor.get("stride", "3"))
    offset = int(accessor.get("offset", "0"))
    count = int(accessor.get("count", str(len(values) // stride)))
    params = [p.get("name", "").upper() for p in accessor.findall("c:param", NS)]
    try:
        x_idx = params.index("X")
        y_idx = params.index("Y")
        z_idx = params.index("Z")
    except ValueError:
        return []
    points: list[tuple[float, float, float]] = []
    for idx in range(count):
        base_idx = offset + idx * stride
        if base_idx + max(x_idx, y_idx, z_idx) >= len(values):
            break
        points.append(
            (
                values[base_idx + x_idx],
                values[base_idx + y_idx],
                values[base_idx + z_idx],
            )
        )
    return points


def geometry_position_points(geometry: ET.Element) -> list[tuple[float, float, float]]:
    mesh = geometry.find("c:mesh", NS)
    if mesh is None:
        return []
    sources_by_id = {
        source.get("id"): source
        for source in mesh.findall("c:source", NS)
        if source.get("id")
    }
    source_ids: list[str] = []
    vertices = mesh.find("c:vertices", NS)
    if vertices is not None:
        for input_elem in vertices.findall("c:input", NS):
            if input_elem.get("semantic") != "POSITION":
                continue
            source_url = input_elem.get("source", "")
            if source_url.startswith("#"):
                source_ids.append(source_url[1:])
    if not source_ids:
        for source_id, source in sources_by_id.items():
            if source_id and source_has_xyz(source):
                source_ids.append(source_id)
                break
    points: list[tuple[float, float, float]] = []
    for source_id in source_ids:
        source = sources_by_id.get(source_id)
        if source is not None:
            points.extend(source_xyz_points(source))
    return points


def geometry_surface_triangles(
    geometry: ET.Element,
) -> list[tuple[tuple[float, float, float], ...]]:
    """Return the indexed triangle surface of one COLLADA geometry.

    Position clouds alone do not describe the filled surface between vertices,
    which makes them unsuitable for exact visibility/occlusion tests.  Resolve
    each primitive's POSITION (usually routed through VERTEX), then triangulate
    polygonal primitives with a deterministic fan.  Malformed primitives are
    skipped locally rather than discarding the rest of the geometry.
    """
    mesh = geometry.find("c:mesh", NS)
    if mesh is None:
        return []

    sources_by_id = {
        source.get("id"): source
        for source in mesh.findall("c:source", NS)
        if source.get("id")
    }
    points_by_source = {
        source_id: source_xyz_points(source)
        for source_id, source in sources_by_id.items()
    }
    vertices_positions: dict[str, str] = {}
    for vertices in mesh.findall("c:vertices", NS):
        vertices_id = vertices.get("id")
        if not vertices_id:
            continue
        for input_elem in vertices.findall("c:input", NS):
            if input_elem.get("semantic") != "POSITION":
                continue
            source_url = input_elem.get("source", "")
            if source_url.startswith("#"):
                vertices_positions[vertices_id] = source_url[1:]
                break

    triangles: list[tuple[tuple[float, float, float], ...]] = []
    primitive_tags = ("triangles", "polylist", "polygons")
    for tag in primitive_tags:
        for primitive in mesh.findall(f"c:{tag}", NS):
            inputs = primitive.findall("c:input", NS)
            if not inputs:
                continue
            stride = max(int(item.get("offset", "0")) for item in inputs) + 1
            position_offset: int | None = None
            position_source: str | None = None
            for item in inputs:
                semantic = item.get("semantic")
                source_url = item.get("source", "")
                if not source_url.startswith("#"):
                    continue
                source_id = source_url[1:]
                if semantic == "VERTEX":
                    source_id = vertices_positions.get(source_id, "")
                elif semantic != "POSITION":
                    continue
                if source_id in points_by_source:
                    position_offset = int(item.get("offset", "0"))
                    position_source = source_id
                    break
            if position_offset is None or position_source is None:
                continue
            source_points = points_by_source[position_source]

            polygons: list[list[int]] = []
            if tag == "triangles":
                for p_elem in primitive.findall("c:p", NS):
                    values = [int(value) for value in (p_elem.text or "").split()]
                    indices = values[position_offset::stride]
                    polygons.extend(
                        indices[index : index + 3]
                        for index in range(0, len(indices) - 2, 3)
                    )
            elif tag == "polylist":
                p_elem = primitive.find("c:p", NS)
                vcount_elem = primitive.find("c:vcount", NS)
                if p_elem is None or vcount_elem is None:
                    continue
                values = [int(value) for value in (p_elem.text or "").split()]
                indices = values[position_offset::stride]
                cursor = 0
                for count_text in (vcount_elem.text or "").split():
                    count = int(count_text)
                    polygons.append(indices[cursor : cursor + count])
                    cursor += count
            else:
                for p_elem in primitive.findall("c:p", NS):
                    values = [int(value) for value in (p_elem.text or "").split()]
                    polygons.append(values[position_offset::stride])

            for polygon in polygons:
                if len(polygon) < 3:
                    continue
                for index in range(1, len(polygon) - 1):
                    corner_indices = (polygon[0], polygon[index], polygon[index + 1])
                    if any(i < 0 or i >= len(source_points) for i in corner_indices):
                        continue
                    triangles.append(tuple(source_points[i] for i in corner_indices))
    return triangles


def bounds_from_points(
    points: list[tuple[float, float, float]],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    zs = [point[2] for point in points]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def sample_points(
    points: list[tuple[float, float, float]],
    max_points: int,
) -> list[tuple[float, float, float]]:
    if len(points) <= max_points:
        return points
    stride = max(1, len(points) // max_points)
    return points[::stride][:max_points]


def find_matching(text: str, open_idx: int, open_char: str, close_char: str) -> int:
    depth = 0
    in_string = False
    escape = False
    line_comment = False
    block_comment = False
    idx = open_idx
    # length is hoisted: this loop runs per character over hundreds of MB of
    # jbeam across a cold load, and len() in the condition is ~40% overhead.
    length = len(text)
    while idx < length:
        ch = text[idx]
        nxt = text[idx + 1] if idx + 1 < length else ""
        if line_comment:
            if ch in "\r\n":
                line_comment = False
            idx += 1
            continue
        if block_comment:
            if ch == "*" and nxt == "/":
                block_comment = False
                idx += 2
                continue
            idx += 1
            continue
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            idx += 1
            continue
        if ch == "/" and nxt == "/":
            line_comment = True
            idx += 2
            continue
        if ch == "/" and nxt == "*":
            block_comment = True
            idx += 2
            continue
        if ch == '"':
            in_string = True
            idx += 1
            continue
        if ch == open_char:
            depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                return idx + 1
        idx += 1
    raise ValueError(f"Unclosed {open_char}{close_char} block")


@lru_cache(maxsize=512)
def mask_comments_preserve_offsets(text: str) -> str:
    """Blank out comments while keeping every offset intact.

    Cached: callers re-mask the same part bodies repeatedly (extract_named_array
    and extract_keyed_object each mask the whole text to locate one key), and
    roughly a third of calls during a cold load are exact repeats.
    """
    out = list(text)
    in_string = False
    escape = False
    line_comment = False
    block_comment = False
    idx = 0
    length = len(text)
    while idx < length:
        ch = text[idx]
        nxt = text[idx + 1] if idx + 1 < length else ""
        if line_comment:
            if ch in "\r\n":
                line_comment = False
            else:
                out[idx] = " "
            idx += 1
            continue
        if block_comment:
            if ch == "*" and nxt == "/":
                out[idx] = " "
                out[idx + 1] = " "
                block_comment = False
                idx += 2
            else:
                out[idx] = ch if ch in "\r\n" else " "
                idx += 1
            continue
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            idx += 1
            continue
        if ch == "/" and nxt == "/":
            out[idx] = " "
            out[idx + 1] = " "
            line_comment = True
            idx += 2
            continue
        if ch == "/" and nxt == "*":
            out[idx] = " "
            out[idx + 1] = " "
            block_comment = True
            idx += 2
            continue
        if ch == '"':
            in_string = True
        idx += 1
    return "".join(out)


def extract_keyed_object(text: str, key: str) -> str | None:
    if f'"{key}"' not in text:
        return None
    # [\s,]* tolerates the stray comma stock jbeam ships between the colon
    # and the brace ("key":, {...}); the game's lenient parser accepts it.
    pattern = re.compile(rf'"{re.escape(key)}"\s*:[\s,]*\{{')
    masked = mask_comments_preserve_offsets(text)
    match = pattern.search(masked)
    if match is None:
        return None
    brace = masked.rfind("{", match.start(), match.end())
    end = find_matching(text, brace, "{", "}")
    return text[match.start() : end]


def extract_named_array(text: str, key: str) -> str | None:
    if f'"{key}"' not in text:
        return None
    pattern = re.compile(rf'"{re.escape(key)}"\s*:[\s,]*\[')
    masked = mask_comments_preserve_offsets(text)
    match = pattern.search(masked)
    if match is None:
        return None
    bracket = masked.rfind("[", match.start(), match.end())
    end = find_matching(text, bracket, "[", "]")
    return text[bracket:end]


def extract_part_mesh_names(part_body: str) -> set[str]:
    meshes: set[str] = set()
    flexbodies = extract_named_array(part_body, "flexbodies")
    if flexbodies:
        for mesh in re.findall(r'\[\s*"((?:[^"\\]|\\.)*)"\s*(?=,|\[|\{)', flexbodies):
            if mesh and mesh != "mesh":
                meshes.add(mesh)
    props = extract_named_array(part_body, "props")
    if props:
        for _full, _func, prop_mesh in PROP_FUNC_MESH_RE.findall(props):
            if prop_mesh and prop_mesh != "mesh":
                meshes.add(prop_mesh)
    return meshes


def extract_part_slot_types(part_body: str) -> list[str]:
    match = re.search(r'"slotType"\s*:', part_body)
    if match is None:
        return []
    idx = match.end()
    while idx < len(part_body) and part_body[idx].isspace():
        idx += 1
    if idx >= len(part_body):
        return []
    if part_body[idx] == '"':
        string_match = re.match(r'"((?:[^"\\]|\\.)*)"', part_body[idx:])
        return [string_match.group(1)] if string_match else []
    if part_body[idx] == "[":
        try:
            end = find_matching(part_body, idx, "[", "]")
        except Exception:
            return []
        return re.findall(r'"((?:[^"\\]|\\.)*)"', part_body[idx:end])
    return []


def replace_array_region(text: str, key: str, transform) -> str:
    pattern = re.compile(rf'"{re.escape(key)}"\s*:[\s,]*\[')
    match = pattern.search(text)
    if match is None:
        return text
    bracket = text.rfind("[", match.start(), match.end())
    end = find_matching(text, bracket, "[", "]")
    old = text[bracket:end]
    new = transform(old)
    return text[:bracket] + new + text[end:]


def rewrite_flexbody_meshes(array_text: str, mesh_map: dict[str, str]) -> str:
    out = array_text
    for old_mesh, new_mesh in sorted(mesh_map.items(), key=lambda item: len(item[0]), reverse=True):
        out = re.sub(
            rf'(\[\s*)"{re.escape(old_mesh)}"(?=\s*(?:,|\[|\{{))',
            rf'\1"{new_mesh}"',
            out,
        )
    return out


def replace_first(text: str, old: str, new: str) -> str:
    idx = text.find(old)
    if idx < 0:
        raise ValueError(f"Could not find {old!r}")
    return text[:idx] + new + text[idx + len(old) :]
